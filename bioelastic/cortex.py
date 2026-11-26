"""Cortex: jerarquía de áreas sensoriales + áreas de asociación que votan la decisión.

    entrada -> área 1 -> área 2 -> ... -> área L          (rasgos locales, no supervisado)
                  \\          \\               \\
                   +----------+---------------+--> áreas de asociación -> decisión

Profundidad y anchura son elásticas y configurables:
  * `areas`: lista de niveles sensoriales (`FeatureArea`); cada uno ve un campo
    receptivo mayor que el anterior.
  * `heads`: áreas de asociación (`BioElasticNet`). Cada una elige su fuente
    ("top", "all", índice de área o "input") y su memoria de entrada (retardos o traza
    lenta con su propia constante de tiempo). Todas votan sumando potenciales sobre las
    mismas neuronas de salida, y todas reciben el mismo error/recompensa (el
    neuromodulador es global). Es el análogo local de las conexiones residuales.

No hay backprop: las áreas sensoriales aprenden sin supervisión y las de asociación con
la regla de tres factores. Agregar un área nueva no borra lo que saben las demás.

Paralelismo (`workers` > 1): las áreas de asociación que leen la entrada cruda no
dependen de las áreas sensoriales, así que su pase y su aprendizaje corren en otro hilo
mientras la jerarquía codifica o aprende. Los resultados no cambian: cada hilo toca sólo
su propio estado y las sumas se hacen en el mismo orden. La ganancia está limitada por
el GIL (sólo las operaciones NumPy grandes lo liberan).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .area import FeatureArea
from .network import BioElasticNet
from .readout import Readout


class Cortex:
    def __init__(self, input_shape, n_outputs: int, *, areas=(), heads=({},), projection=None,
                 replay: float = 0.0, rehearse: float = 0.0, replay_focus: str = "uniform",
                 rehearse_focus: str = "uniform", confusion_p: float = 0.05,
                 joint: bool = True, temperature: float = 0.25, seed: int = 0, workers: int = 1):
        self._config = dict(input_shape=input_shape, n_outputs=int(n_outputs),
                            areas=[dict(a) for a in areas], heads=[],
                            projection=projection, replay=float(replay),
                            rehearse=float(rehearse), replay_focus=str(replay_focus),
                            rehearse_focus=str(rehearse_focus), confusion_p=float(confusion_p),
                            joint=bool(joint),
                            temperature=float(temperature), seed=int(seed))
        self.is_sheet = not np.isscalar(input_shape)
        self.input_shape = tuple(int(v) for v in input_shape) if self.is_sheet else int(input_shape)
        self.C = int(n_outputs)
        self.temperature = float(temperature)
        self.replay = float(replay); self._replay_debt = 0.0; self.replays = 0
        self.rehearse = float(rehearse); self._rehearse_debt = 0.0; self.rehearsals = 0
        if replay_focus not in ("uniform", "absent", "confusion"):
            raise ValueError(f"replay_focus desconocido: {replay_focus}")
        if rehearse_focus not in ("uniform", "confusion"):
            raise ValueError(f"rehearse_focus desconocido: {rehearse_focus}")
        self.replay_focus = replay_focus
        self.rehearse_focus = rehearse_focus
        self.confusion_p = float(confusion_p)
        self.directed = 0                                     # repasos y ensayos dirigidos a la confusión
        self.joint = bool(joint)
        self.rng = np.random.default_rng(seed)
        self.areas: list[FeatureArea] = []
        shape = self.input_shape
        if areas and not self.is_sheet:
            raise ValueError("las áreas sensoriales necesitan una lámina (alto, ancho, canales)")
        for i, spec in enumerate(areas):
            a = FeatureArea(shape, **spec)
            a.on_reuse = lambda f, i=i: self._on_reuse(i, f)
            self.areas.append(a); shape = a.out_shape
        self._sizes = [int(np.prod(a.out_shape)) for a in self.areas]
        self.heads: list[BioElasticNet] = []
        self._src: list = []
        for i, spec in enumerate(heads):
            self.add_head(**spec, seed=seed + i)
        # Proyección directa (lámina 5) de las áreas sensoriales a la decisión.
        self.proj = None; self._proj_src = None
        if projection is not None and self.areas:
            spec = dict(projection); self._proj_src = spec.pop("source", "top")
            n_in = sum(self._sizes[i] for i, _ in self._offsets(self._proj_src))
            self.proj = Readout(self.C, capacity=n_in, **{"consolidation": 0.0, "lr": 0.5, **spec})
            self.proj.n = n_in
        self._V = None; self.pred = 0; self._pv = None
        self.workers = int(workers)    # hilos (no es parte del modelo: no se guarda)
        self._pool = None

    # ------------------------------------------------------------------ estructura
    def _input_size(self) -> int:
        return int(np.prod(self.input_shape)) if self.is_sheet else self.input_shape

    def _offsets(self, src):
        """[(área, desplazamiento)] que forman la volea de una fuente."""
        if src == "input" or not self.areas:
            return []
        if src == "top":
            return [(len(self.areas) - 1, 0)]
        if src == "all":
            off = np.concatenate([[0], np.cumsum(self._sizes)[:-1]])
            return [(i, int(o)) for i, o in enumerate(off)]
        return [(int(src), 0)]

    def add_head(self, source="top", seed: int = 0, **spec) -> BioElasticNet:
        """Agrega un área de asociación (se puede hacer en caliente, con la red ya entrenada)."""
        parts = self._offsets(source)
        n_in = sum(self._sizes[i] for i, _ in parts) if parts else self._input_size()
        head = BioElasticNet(n_in, self.C, seed=seed, **spec)
        self.heads.append(head); self._src.append(source)
        self._config["heads"].append(dict(spec, source=source, seed=seed))
        return head

    def _on_reuse(self, area: int, f: int):
        a = self.areas[area]
        F = a.F
        if area + 1 < len(self.areas):
            self.areas[area + 1].forget_channel(f)
        ids = np.arange(a.ph * a.pw) * F + f
        for head, src in zip(self.heads, self._src):
            for i, off in self._offsets(src):
                if i == area:
                    head.layer.forget_afferents(ids + off)
        if self.proj is not None:                             # la proyección también olvida ese canal
            for i, off in self._offsets(self._proj_src):
                if i == area:
                    self.proj.W[:, ids + off] = 0.0

    # ------------------------------------------------------------------ inferencia
    def _volley(self, src, x, codes):
        parts = self._offsets(src)
        if not parts:
            return x
        idxs, vals = [], []
        for i, off in parts:
            c = codes[i].ravel()
            idx = np.flatnonzero(c != 0)                      # (sobre bool es ~10x más rápido)
            idxs.append(idx + off); vals.append(c[idx])
        if len(idxs) == 1:
            return idxs[0], vals[0]
        return np.concatenate(idxs), np.concatenate(vals)

    def _parallel(self):
        """Ejecutor y áreas de asociación que leen la entrada cruda, si hay con qué
        superponer trabajo; si no, (None, [])."""
        if self.workers < 2 or not self.areas:
            return None, []
        indep = [i for i, src in enumerate(self._src) if not self._offsets(src)]
        if not indep:
            return None, []
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.workers - 1)
        return self._pool, indep

    def forward(self, x, explore: bool = False) -> int:
        pool, indep = self._parallel()
        job = None
        if pool is not None:                                  # en paralelo con la jerarquía
            job = pool.submit(lambda: [self.heads[i].forward(x) for i in indep])
        codes = []
        if self.areas:
            X = np.asarray(x, dtype=np.float32).reshape(self.input_shape)
            for a in self.areas:
                X = a.encode(X); codes.append(X)
        for i, (head, src) in enumerate(zip(self.heads, self._src)):
            if job is None or i not in indep:
                head.forward(self._volley(src, x, codes))
        if job is not None:
            job.result()
        V = np.zeros(self.C, dtype=np.float32)
        for head in self.heads:                               # mismo orden de suma
            V = V + head._last.V
        if self.proj is not None:
            idx, val = self._volley(self._proj_src, x, codes)
            h = val / np.float32(max(float(np.sqrt(val @ val)), 1e-6))
            self._pv = (idx, h)
            V = V + self.proj.potentials(idx, h)
        self._V = V
        if explore:
            self.pred = int(self.rng.choice(self.C, p=self._p()))
        else:
            self.pred = int(np.argmax(V))
        return self.pred

    predict = forward

    # ------------------------------------------------------------------ aprendizaje
    def _p(self):
        e = np.exp((self._V - self._V.max()) / self.temperature)
        return e / e.sum()

    def _learn_areas_and(self, fn):
        """Aprendizaje de las áreas sensoriales (en otro hilo) mientras en este corre
        `fn(i)` para las áreas de asociación que leen la entrada cruda. Devuelve cuáles ya
        se hicieron; las demás deben esperar, porque el reuso de un rasgo puede cortar sus
        sinapsis (`_on_reuse`)."""
        pool, par = self._parallel()
        if pool is None:
            for a in self.areas:
                a.learn()
            return []
        job = pool.submit(lambda: [a.learn() for a in self.areas])
        try:
            for i in par:
                fn(i)
        finally:
            job.result()
        return par

    def teach(self, target: int):
        p = self._p()                                         # no depende del aprendizaje de las áreas
        conf = None
        if (self.replay_focus == "confusion" and self.replay > 0.0 and self.proj is not None) or \
                (self.rehearse_focus == "confusion" and self.rehearse > 0.0):
            conf = self._confusion(target, p)                 # antes de que la memoria cambie

        def one(i):
            if self.joint:
                self.heads[i].teach(target, pred=self.pred, p=p)
            else:                                             # cada área aprende de su propio error
                self.heads[i].teach(target)
        done = self._learn_areas_and(one)
        for i in range(len(self.heads)):
            if i not in done:
                one(i)
        if self.proj is not None:
            self.proj.learn(*self._pv, target, p=p)
            if self.replay > 0.0:
                self._replay_debt += self.replay
                aimed = conf is not None and self.replay_focus == "confusion"
                while self._replay_debt >= 1.0:
                    self._replay_debt -= 1.0
                    if aimed:                                 # un repaso por paso va a la confusión
                        self._replay_pattern(conf[0], conf[1], conf[3]); aimed = False
                        self.directed += 1
                    else:
                        self._replay_one()
        if self.rehearse > 0.0:
            self._rehearse_debt += self.rehearse
            aimed = conf is not None and self.rehearse_focus == "confusion"
            while self._rehearse_debt >= 1.0:
                self._rehearse_debt -= 1.0
                if aimed:
                    self._rehearse_one(conf); aimed = False
                    self.directed += 1
                else:
                    self._rehearse_one()
        if conf is not None:
            conf[2].watch = -1

    def _confusion(self, target: int, p):
        """Repaso dirigido a la confusión (cf. MIR, Aljundi et al. 2019; ASER, Shim et al.
        2021): la clase que más compite con el objetivo en esta entrada (la de mayor
        probabilidad después de él, si pasa de `confusion_p`) y, de esa clase, el recuerdo
        de la memoria más parecido a lo que entra: la ganadora de esa clase si disparó, si
        no la de mayor coincidencia entre sus neuronas. Es el ejemplo viejo que el
        aprendizaje de ahora más va a perturbar, justo en la frontera.

        Devuelve (recuerdo denso, clase, memoria, espacio) o None. El recuerdo se copia
        antes de que la memoria aprenda (puede desalojar o mover neuronas); la neurona
        queda marcada en `memoria.watch` para excluirla del ensayo."""
        mem, space = self._replay_source()
        if mem is None or mem._last is None:
            return None
        q = np.asarray(p, dtype=np.float64).copy()
        q[target] = -1.0
        c2 = int(np.argmax(q))
        if q[c2] < self.confusion_p:
            return None
        L = mem.layer
        v = mem._last.v
        if len(v.winners) and int(v.winners.max()) >= L.n:
            return None
        pw = mem.readout.preferred(v.winners)
        hit = np.flatnonzero(pw == c2)
        if len(hit):
            j = int(v.winners[hit[0]])
        else:
            if v.norm == 0.0:
                return None
            rows = np.flatnonzero(mem.readout.preferred(np.arange(L.n)) == c2)
            if len(rows) == 0:
                return None
            cos = L._cos_at(rows, v.idx, v.val, v.norm, L._fan_max())
            j = int(rows[int(np.argmax(cos))])
        mem.watch = j
        return mem.episode(j).copy(), c2, mem, space

    def _replay_source(self):
        """Área de asociación de la que se puede repasar y en qué espacio guarda sus recuerdos.

        "input": sus neuronas son patrones de entrada; el recuerdo se reactiva y recorre
                 las áreas sensoriales (como en el sueño, sin entrenar los rasgos).
        "code" : sus neuronas ya están en el espacio de la proyección (lee de la misma
                 fuente); el recuerdo se usa directamente.
        Sin ninguna de las dos, no hay de dónde repasar."""
        usable = [(h, src) for h, src in zip(self.heads, self._src)
                  if h.layer.n and h.context == 1 and h.tau <= 0]
        for h, src in usable:
            if not self._offsets(src):
                return h, "input"
        for h, src in usable:
            if self._offsets(src) == self._offsets(self._proj_src):
                return h, "code"
        return None, None

    def _replay_one(self):
        """Consolidación por repetición: un recuerdo del área de asociación (el patrón que
        guarda una de sus neuronas) se reactiva y entrena la proyección directa con la clase
        de esa neurona. Las clases se eligen de forma pareja, así lo viejo se repasa aunque
        el flujo ya no lo muestre. Con replay_focus="absent" (y una proyección con
        `active_window`) se repasan sólo las clases ausentes del flujo: las presentes ya
        reciben ejemplos reales."""
        mem, space = self._replay_source()
        if mem is None:
            return
        L = mem.layer
        pref = mem.readout.preferred(np.arange(L.n))
        classes = np.unique(pref[pref >= 0])
        if self.replay_focus == "absent" and self.proj.active_window > 0 and len(classes):
            absent = classes[~self.proj.active()[classes]]
            if len(absent):
                classes = absent
        if len(classes) == 0:
            return
        c = int(self.rng.choice(classes))
        j = int(self.rng.choice(np.flatnonzero(pref == c)))
        self._replay_pattern(mem.episode(j), c, space)

    def _replay_pattern(self, x, c: int, space: str):
        """Reactiva el recuerdo denso `x` (espacio de la memoria) y entrena la proyección
        directa con la clase c, con el error completo."""
        if space == "input":
            X = (x / max(float(x.max()), 1e-8)).reshape(self.input_shape)
            codes = []
            for a in self.areas:
                X = a.encode(X); codes.append(X)
                a._cache = None                               # el repaso no entrena los rasgos
            idx, val = self._volley(self._proj_src, None, codes)
        else:
            idx = np.flatnonzero(x > 0); val = x[idx]
        if len(idx) == 0:
            return
        h = val / np.float32(max(float(np.sqrt(val @ val)), 1e-6))
        V = self.proj.potentials(idx, h)
        e = np.exp((V - V.max()) / self.temperature)
        self.proj.learn(idx, h, c, p=e / e.sum(), replay=True)
        self.replays += 1

    def _rehearse_one(self, conf=None):
        """Ensayo en la propia memoria: el recuerdo de una neurona (clase elegida de forma
        pareja) vuelve a entrar como si fuera un ejemplo, sin la neurona que lo guarda, y
        las demás ganadoras corrigen su lectura con el error completo. Mantiene la frontera
        con clases ausentes: las neuronas nuevas que invaden el territorio de una clase
        vieja aprenden que ahí también vive esa clase. No crea ni mueve neuronas.
        Con `conf` (ver `_confusion`) el recuerdo es el de la clase que compite con lo que
        acaba de entrar, el más cercano a la entrada."""
        if conf is not None:
            x, c, mem, _ = conf
            j = mem.watch
        else:
            mem = next((h for h, src in zip(self.heads, self._src)
                        if not self._offsets(src) and h.context == 1 and h.tau <= 0 and h.layer.n), None)
            if mem is None:
                return
            pref = mem.readout.preferred(np.arange(mem.layer.n))
            classes = np.unique(pref[pref >= 0])
            if len(classes) == 0:
                return
            c = int(self.rng.choice(classes))
            j = int(self.rng.choice(np.flatnonzero(pref == c)))
            x = mem.episode(j)
        L, R = mem.layer, mem.readout
        idx = np.flatnonzero(x > 0); val = x[idx] / max(float(x.max()), 1e-8)
        v = L.forward(idx, val)
        keep = v.winners != j
        if v.novel or not keep.any():
            return
        wn = v.winners[keep]
        h = mem._votes(v)[keep]
        h = h / h.sum()
        R.learn(wn, h, c, replay=True)
        self.rehearsals += 1

    def reinforce(self, reward: float):
        p = self._p()
        one = lambda i: self.heads[i].reinforce(reward, pred=self.pred, p=p)
        done = self._learn_areas_and(one)
        for i in range(len(self.heads)):
            if i not in done:
                one(i)
        if self.proj is not None:
            self.proj.learn_reward(*self._pv, self.pred, reward, p=p)

    def observe(self, x):
        self.forward(x)
        one = lambda i: self.heads[i].observe()
        done = self._learn_areas_and(one)
        for i in range(len(self.heads)):
            if i not in done:
                one(i)

    def step(self, x, target: int) -> int:
        pred = self.forward(x)
        self.teach(target)
        return pred

    def reset_context(self):
        for head in self.heads:
            head.reset_context()

    # ------------------------------------------------------------------ estado
    def stats(self) -> dict:
        areas = [a.stats() for a in self.areas]
        heads = [dict(h.stats(), synapses=h.layer.synapses()) for h in self.heads]
        return dict(
            depth=len(self.areas) + 1,
            areas=areas, heads=heads,
            features=sum(a["features"] for a in areas),
            neurons=sum(h["neurons"] for h in heads),
            units=sum(a["units"] for a in areas) + sum(h["neurons"] for h in heads) + self.C,
            synapses=sum(a["synapses"] for a in areas) + sum(h["synapses"] for h in heads)
                     + sum(h["neurons"] for h in heads) * self.C
                     + (self.proj.n * self.C if self.proj is not None else 0),
            replays=self.replays,
        )

    def cost(self) -> dict:
        """Operaciones sinápticas (multiplicación-suma) por inferencia y memoria, por parte.

        Cuenta lo que hace el código, no lo que guarda: un área sensorial evalúa su
        diccionario en cada posición, y un área de asociación con el pase denso evalúa
        todas sus neuronas (n x fan-in máximo). Es una cota superior para las áreas que
        usan el índice invertido. El repaso agrega, por paso de entrenamiento, `replay`
        codificaciones más de la jerarquía."""
        parts = {}
        for i, a in enumerate(self.areas):
            c = a.cost()
            parts[f"area{i}"] = dict(macs=c["macs"], memory_bytes=c["memory_bytes"])
        for i, h in enumerate(self.heads):
            L = h.layer
            kf = int(L.fill[:L.n].max()) if L.n else 0
            parts[f"head{i}"] = dict(macs=L.n * kf,
                                     memory_bytes=L.memory()["total"] + h.readout.W.nbytes + h.readout.count.nbytes)
        if self.proj is not None:
            parts["proj"] = dict(macs=self.C * self.proj.n,          # cota: todas las entradas activas
                                 memory_bytes=self.proj.W.nbytes + self.proj.count.nbytes)
        infer = sum(p["macs"] for p in parts.values())
        area_macs = sum(parts[f"area{i}"]["macs"] for i in range(len(self.areas)))
        return dict(parts=parts, macs_per_inference=infer,
                    macs_per_training_step=infer + self.replay * (area_macs + parts.get("proj", {}).get("macs", 0)),
                    memory_bytes=sum(p["memory_bytes"] for p in parts.values()))

    def save(self, path):
        """Guarda estructura y pesos. El contexto en curso (retardos, trazas) no se guarda."""
        out = {"config": np.array(json.dumps(self._config)),
               "meta": np.array([self.replays], dtype=np.int64)}
        for i, a in enumerate(self.areas):
            out.update({f"a{i}_{k}": v for k, v in a.state().items()})
        for i, h in enumerate(self.heads):
            out.update({f"h{i}_{k}": v for k, v in h.state().items()})
        if self.proj is not None:
            out.update(proj_W=self.proj.W, proj_count=self.proj.count,
                       proj_baseline=np.float64(self.proj.baseline))
        np.savez_compressed(path, **out)

    @classmethod
    def load(cls, path) -> "Cortex":
        with np.load(path, allow_pickle=False) as z:
            cfg = json.loads(str(z["config"]))
            shape = cfg.pop("input_shape")
            head_specs = cfg.pop("heads")
            net = cls(tuple(shape) if isinstance(shape, list) else shape, cfg.pop("n_outputs"),
                      heads=[], **cfg)
            for i, a in enumerate(net.areas):
                a.restore(z, f"a{i}_")
            for i, spec in enumerate(head_specs):
                sub = {k[len(f"h{i}_"):]: z[k] for k in z.files if k.startswith(f"h{i}_")}
                net.heads.append(BioElasticNet.from_state(sub))
                net._src.append(spec.get("source", "top"))
            net._config["heads"] = head_specs
            if net.proj is not None:
                net.proj.W[:] = z["proj_W"]; net.proj.count[:] = z["proj_count"]
                net.proj.baseline = float(z["proj_baseline"])
            net.replays = int(z["meta"][0])
        return net
