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

No hay backprop: las áreas sensoriales aprenden sin supervisión (o con un error local
propio, ver `FeatureArea`) y las de asociación con la regla de tres factores. Agregar un
área nueva no borra lo que saben las demás.

Decisión: por defecto (v0.2) se suman los potenciales de todos los votantes (áreas de
asociación, proyección directa y, si las hay, lectores locales de las áreas sensoriales).
Con `arbiter > 0` cada votante aprende con su propio error y un árbitro pondera sus
log-probabilidades, V = sum_i beta_i log p_i, con pesos que aprenden online del error
global: d beta_i = arbiter * (log p_i[objetivo] - E_p[log p_i]). Es un control de ganancia
por vía: un votante que no aporta pierde voz en vez de arrastrar la decisión.

Homeostasis de las salidas (`homeostasis` > 0, opcional; por defecto apagada): cada vez
que la red aprende de una decisión (`teach` o `reinforce`), el umbral de la salida elegida
sube `homeostasis` y el de las demás baja homeostasis/C (suma cero), como el `theta_plus`
de las capas ocultas. Una salida que acapara decisiones se vuelve más difícil de disparar
y una que nunca gana termina por ganar alguna. Contra el colapso de símbolos cuando la red
aprende sólo con recompensa (en la integración con el Thinker, desde cero, la pureza de la
percepción subió de 0.21 a 0.58 y de 0.62 a 0.85 en la identidad, 3 semillas). No cambia
cómo se reparte el crédito. Con etiquetas no hace falta y no se usa.

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
                 joint: bool = True, temperature: float = 0.25, arbiter: float = 0.0,
                 recall: str = "off", recall_k: int = 8, recall_margin: float = 0.2,
                 recall_gain: float = 1.0, recall_tau: float = 0.05, recall_source: str = "episodes",
                 stored_labels: bool = False, slow_error: str = "joint",
                 schema: str = "off", schema_fit: float = 0.5, schema_boost: float = 4.0,
                 schema_misfit_lr: float = 1.0, schema_beta: float = 0.02, schema_known: int = 20,
                 seed: int = 0, workers: int = 1, homeostasis: float = 0.0):
        self._config = dict(input_shape=input_shape, n_outputs=int(n_outputs),
                            areas=[dict(a) for a in areas], heads=[],
                            projection=projection, replay=float(replay),
                            rehearse=float(rehearse), replay_focus=str(replay_focus),
                            rehearse_focus=str(rehearse_focus), confusion_p=float(confusion_p),
                            joint=bool(joint), temperature=float(temperature),
                            arbiter=float(arbiter),
                            recall=str(recall), recall_k=int(recall_k), recall_margin=float(recall_margin),
                            recall_gain=float(recall_gain), recall_tau=float(recall_tau),
                            recall_source=str(recall_source), stored_labels=bool(stored_labels),
                            slow_error=str(slow_error),
                            schema=str(schema), schema_fit=float(schema_fit),
                            schema_boost=float(schema_boost), schema_misfit_lr=float(schema_misfit_lr),
                            schema_beta=float(schema_beta), schema_known=int(schema_known),
                            seed=int(seed))
        self.arbiter = float(arbiter)
        self.beta = None                                      # pesos del árbitro (uno por votante)
        self._L = None
        self._zs = []
        self.homeostasis = float(homeostasis)
        if self.homeostasis > 0.0:
            self._config["homeostasis"] = self.homeostasis
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
        # Dos sistemas explícitos (ver `forward` y `_recall_vote`).
        if recall not in ("off", "gate", "always"):
            raise ValueError(f"recall desconocido: {recall}")
        if recall_source not in ("episodes", "readout"):
            raise ValueError(f"recall_source desconocido: {recall_source}")
        self.recall = recall; self.recall_k = int(recall_k); self.recall_margin = float(recall_margin)
        self.recall_gain = float(recall_gain); self.recall_tau = float(recall_tau)
        self.recall_source = recall_source
        self.stored_labels = bool(stored_labels)
        if slow_error not in ("joint", "own"):
            raise ValueError(f"slow_error desconocido: {slow_error}")
        # "joint": la proyección aprende con el error de la decisión conjunta (lo que la
        # memoria ya acierta no le enseña nada al lento); "own": con su propio error, como
        # el neocórtex de CLS, que no queda «explicado» por el hipocampo.
        self.slow_error = slow_error
        self.recalls = 0                                      # inferencias que consultaron el almacén
        self._slow = None; self._fast = None; self._recalled = False
        # Esquemas (ver `_schema_fit`).
        if schema not in ("off", "measure", "gate", "always"):
            raise ValueError(f"schema desconocido: {schema}")
        self.schema = schema; self.schema_fit = float(schema_fit); self.schema_boost = float(schema_boost)
        self.schema_misfit_lr = float(schema_misfit_lr); self.schema_beta = float(schema_beta)
        self.schema_known = int(schema_known)
        self.reliability = 0.0                                # acierto reciente del lento en clases conocidas
        self.fit = float("nan"); self.fit_parts = (float("nan"),) * 3
        self.assimilated = 0
        self._taught = np.zeros(self.C, dtype=np.int64)       # enseñanzas por clase
        self.rng = np.random.default_rng(seed)
        self.areas: list[FeatureArea] = []
        shape = self.input_shape
        if areas and not self.is_sheet:
            raise ValueError("las áreas sensoriales necesitan una lámina (alto, ancho, canales)")
        for i, spec in enumerate(areas):
            extra = dict(n_classes=self.C, seed=seed + 100 + i) if spec.get("local") else {}
            a = FeatureArea(shape, **spec, **extra)
            a.on_reuse = lambda f, i=i: self._on_reuse(i, f)
            self.areas.append(a); shape = a.out_shape
        self._sizes = [int(np.prod(a.out_shape)) for a in self.areas]
        self.heads: list[BioElasticNet] = []
        self._src: list = []
        for i, spec in enumerate(heads):
            self.add_head(**spec, seed=seed + i)
        # Proyección directa (lámina 5) de las áreas sensoriales a la decisión.
        self.proj = None; self._proj_src = None; self._balance = False
        if projection is not None and self.areas:
            spec = dict(projection); self._proj_src = spec.pop("source", "top")
            self._balance = bool(spec.pop("balance", False))
            n_in = sum(self._sizes[i] for i, _ in self._offsets(self._proj_src))
            self.proj = Readout(self.C, capacity=n_in, **{"consolidation": 0.0, "lr": 0.5, **spec})
            self.proj.n = n_in
        self._V = None; self.pred = 0; self._pv = None
        self.umbral = np.zeros(self.C, dtype=np.float32)          # homeostasis de las salidas
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

    def _proj_input(self, codes):
        """Volea normalizada que recibe la proyección directa. Con `balance`, cada área se
        normaliza por separado (misma voz para todas, aunque una tenga muchas más unidades
        activas que otra); si no, se normaliza la concatenación."""
        if not self._balance:
            idx, val = self._volley(self._proj_src, None, codes)
            return idx, val / np.float32(max(float(np.sqrt(val @ val)), 1e-6))
        parts = self._offsets(self._proj_src)
        idxs, vals = [], []
        for i, off in parts:
            c = codes[i].ravel(); j = np.flatnonzero(c); v = c[j]
            nv = float(np.sqrt(v @ v))
            if nv > 0:
                idxs.append(j + off); vals.append(v / np.float32(nv * np.sqrt(len(parts))))
        if not idxs:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
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
        votes = [(head._last.V, head.readout.temperature) for head in self.heads]
        Vp = None
        if self.proj is not None:
            idx, h = self._proj_input(codes)
            self._pv = (idx, h)
            Vp = self.proj.potentials(idx, h)
            votes.append((Vp, self.proj.temperature))
        self._zs = []
        for a in self.areas:                                  # lectores locales (si hay error local)
            z = a.potentials()
            self._zs.append(z)
            if z is not None:
                votes.append((z, a.clf.temperature))
        if self.arbiter > 0.0 and votes:
            if self.beta is None or len(self.beta) != len(votes):
                self.beta = np.ones(len(votes), dtype=np.float32)
            L = np.stack([z / np.float32(T) for z, T in votes])
            L = L - L.max(axis=1, keepdims=True)
            L = L - np.log(np.exp(L).sum(axis=1, keepdims=True))   # log-probabilidades
            self._L = L.astype(np.float32)
            V = self.beta @ self._L
        else:
            V = np.zeros(self.C, dtype=np.float32)
            for z, _ in votes:                                # mismo orden de suma
                V = V + z
        self._slow = None; self._fast = None; self._recalled = False
        if self.recall != "off" and self.arbiter <= 0.0:      # (con árbitro, V son log-probabilidades)
            V = self._two_systems(V, Vp)
        if self.homeostasis > 0.0:
            V = V - self.umbral
        self._V = V
        if explore:
            self.pred = int(self.rng.choice(self.C, p=self._p()))
        else:
            self.pred = int(np.argmax(V))
        return self.pred

    predict = forward

    # ------------------------------------------------------------------ dos sistemas
    def _store(self):
        """Almacén rápido: el área de asociación de la que se repasa, si guarda huellas."""
        mem, _ = self._replay_source()
        return mem if mem is not None and mem.episodes is not None else None

    def _slow_part(self, store, Vp):
        """Sistema lento: la proyección directa y las áreas de asociación que no son el almacén."""
        S = np.zeros(self.C, dtype=np.float32)
        for head in self.heads:
            if head is not store:
                S = S + head._last.V
        if Vp is not None:
            S = S + Vp
        for z in self._zs:                                    # lectores locales de las áreas
            if z is not None:
                S = S + z
        return S

    def _two_systems(self, V, Vp):
        """Complementary Learning Systems (McClelland et al. 1995; Kumaran, Hassabis y
        McClelland 2016): el sistema lento (proyección + áreas) responde solo cuando está
        seguro; si no, se recuperan los `recall_k` episodios más parecidos del almacén
        rápido y se combinan (razonamiento basado en memoria).

          recall="gate"    se consulta el almacén sólo si el margen p1 - p2 del sistema
                           lento (softmax a la temperatura de la red) es < `recall_margin`
          recall="always"  siempre lento + almacén (sin compuerta)
          recall_source="episodes"  voto de los episodios recuperados con su etiqueta
                           guardada, pesos softmax(coseno / `recall_tau`)
          recall_source="readout"   voto de la lectura aprendida del almacén (sólo compuerta)
        """
        store = self._store()
        if store is None or store._last is None:
            return V
        S = self._slow_part(store, Vp)
        self._slow = S
        use = True
        if self.recall == "gate":
            e = np.exp((S - S.max()) / self.temperature)
            q = np.sort(e / e.sum())
            use = (q[-1] - (q[-2] if len(q) > 1 else 0.0)) < self.recall_margin
        if not use:
            return S
        self._fast = self._recall_vote(store)
        self._recalled = True
        self.recalls += 1
        return S + np.float32(self.recall_gain) * self._fast

    def _recall_vote(self, store) -> np.ndarray:
        if self.recall_source == "readout":
            return store._last.V.astype(np.float32)
        v = store._last.v
        ids, sim = store.recall(v.idx, v.val, self.recall_k)
        out = np.zeros(self.C, dtype=np.float32)
        if len(ids) == 0:
            return out
        y = store.ep["y"][ids].astype(np.int64)
        if (y < 0).any():                                     # sin etiqueta: la de su lectura
            y = np.where(y >= 0, y, store.readout.preferred(ids))
        w = np.exp((sim - sim.max()) / max(self.recall_tau, 1e-6))
        ok = y >= 0
        if ok.any():
            np.add.at(out, y[ok], (w[ok] / w[ok].sum()).astype(np.float32))
        return out

    def parts(self) -> dict:
        """Aporte de cada sistema en la última inferencia: «slow» (proyección + áreas que no
        son el almacén) y «fast» (el almacén: su lectura aprendida o, con `recall`, el voto
        de los episodios recuperados, calculado aunque la compuerta no lo haya pedido).
        «recalled»: si la respuesta consultó el almacén."""
        store = self._store() or (self.heads[0] if self.heads else None)
        Vp = self.proj.potentials(*self._pv) if self.proj is not None and self._pv is not None else None
        slow = self._slow if self._slow is not None else self._slow_part(store, Vp)
        if self.recall == "off" or store is None or store.episodes is None:
            fast = store._last.V if store is not None else np.zeros(self.C, dtype=np.float32)
        else:
            fast = self._fast if self._fast is not None else self._recall_vote(store)
        return dict(slow=slow, fast=fast, recalled=self._recalled)

    def _labels(self, mem) -> np.ndarray:
        """Clase de cada recuerdo del almacén para el repaso: la etiqueta guardada en el
        episodio (`stored_labels`) o la preferida de la lectura aprendida (puede derivar)."""
        return mem.labels(stored=self.stored_labels)

    # ------------------------------------------------------------------ aprendizaje
    def _p(self):
        T = 1.0 if self._arbitrated() else self.temperature     # con árbitro V ya es log-prob
        e = np.exp((self._V - self._V.max()) / T)
        return e / e.sum()

    def _arbitrated(self) -> bool:
        return self.arbiter > 0.0 and self._L is not None

    def _learn_areas(self, target=None):
        """Aprendizaje de las áreas sensoriales. Con `target`, las que tienen error local lo
        usan (con el error residual de las anteriores, si así se configuró)."""
        prior = None
        zs = self._zs if len(self._zs) == len(self.areas) else [None] * len(self.areas)
        for a, z in zip(self.areas, zs):
            a.learn(target, prior)
            if z is not None:
                prior = z if prior is None else prior + z

    def _learn_areas_and(self, fn, target=None):
        """Aprendizaje de las áreas sensoriales (en otro hilo) mientras en este corre
        `fn(i)` para las áreas de asociación que leen la entrada cruda. Devuelve cuáles ya
        se hicieron; las demás deben esperar, porque el reuso de un rasgo puede cortar sus
        sinapsis (`_on_reuse`)."""
        pool, par = self._parallel()
        if pool is None:
            self._learn_areas(target)
            return []
        job = pool.submit(lambda: self._learn_areas(target))
        try:
            for i in par:
                fn(i)
        finally:
            job.result()
        return par

    def _adapt_thresholds(self):
        """Homeostasis de las salidas: sube el umbral de la elegida, baja el de las demás."""
        if self.homeostasis > 0.0:
            self.umbral -= np.float32(self.homeostasis / self.C)
            self.umbral[self.pred] += np.float32(self.homeostasis)

    def teach(self, target: int):
        self._adapt_thresholds()
        p = self._p()                                         # no depende del aprendizaje de las áreas
        arb = self._arbitrated()
        if arb:                                               # el árbitro aprende del error global
            g = self._L[:, target] - self._L @ p
            self.beta = np.maximum(self.beta + np.float32(self.arbiter) * g.astype(np.float32), 0.0)
        conf = None
        if (self.replay_focus == "confusion" and self.replay > 0.0 and self.proj is not None) or \
                (self.rehearse_focus == "confusion" and self.rehearse > 0.0):
            conf = self._confusion(target, p)                 # antes de que la memoria cambie
        grow, slow_lr = None, 1.0
        if self.schema != "off":                              # antes de que las áreas aprendan
            grow, slow_lr = self._schema_route(target)
        store = self._store() if grow is not None else None

        def one(i):
            g = {} if grow is None or self.heads[i] is not store else dict(grow=grow)
            if self.joint and not arb:
                self.heads[i].teach(target, pred=self.pred, p=p, **g)
            else:                                             # cada área aprende de su propio error
                self.heads[i].teach(target, **g)
        done = self._learn_areas_and(one, target)
        for i in range(len(self.heads)):
            if i not in done:
                one(i)
        self._taught[target] += 1
        if self.proj is not None:
            pp = None if arb else p
            if self.slow_error == "own" and not arb:
                S = self._slow if self._slow is not None else self._slow_part(
                    self._store(), self.proj.potentials(*self._pv))
                e = np.exp((S - S.max()) / self.temperature); pp = e / e.sum()
            if slow_lr == 1.0:
                self.proj.learn(*self._pv, target, p=pp)
            else:
                self.proj.learn(*self._pv, target, p=pp, lr_scale=slow_lr)
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

    # ------------------------------------------------------------------ esquemas
    def _coverage(self) -> float:
        """Qué parte de la entrada explican las áreas sensoriales: fracción de la energía de
        los parches válidos cuyo mejor rasgo supera el umbral del área (promedio entre
        áreas). Usa la última codificación (antes de que las áreas aprendan)."""
        covs = []
        for a in self.areas:
            if a._cache is None:
                continue
            Pm, norms, vr, act = a._cache[:4]
            if act is None or len(vr) == 0:
                covs.append(0.0); continue
            w = norms[vr]
            covs.append(float((w * (act.max(axis=1) >= a.theta)).sum() / max(float(w.sum()), 1e-12)))
        return float(np.mean(covs)) if covs else 1.0

    def _schema_fit(self, target: int) -> float:
        """Encaje de (entrada, etiqueta) con el conocimiento del sistema lento (esquema, Tse
        et al. 2007; su lectura computacional, McClelland 2013):

            encaje = cobertura x fiabilidad x (1 - conflicto)

          cobertura   cuánto de la entrada explican los diccionarios de las áreas
          fiabilidad  acierto reciente (media móvil, tasa `schema_beta`) del sistema lento
                      en clases ya enseñadas >= `schema_known` veces: si lo conocido no se
                      predice bien, no hay esquema
          conflicto   probabilidad de la clase equivocada más fuerte según el lento: un
                      ítem que contradice lo que el lento cree con seguridad exigiría
                      reestructurarlo (interferencia)
        """
        store = self._store()
        S = self._slow if self._slow is not None else self._slow_part(
            store, self.proj.potentials(*self._pv) if self.proj is not None else None)
        e = np.exp((S - S.max()) / self.temperature); q = e / e.sum()
        conflict = float(np.max(np.delete(q, target))) if self.C > 1 else 0.0
        cov = self._coverage()
        rel = self.reliability
        fit = cov * rel * (1.0 - conflict)
        if self._taught[target] >= self.schema_known:         # la fiabilidad se mide en lo conocido
            ok = float(int(np.argmax(S)) == target)
            self.reliability += self.schema_beta * (ok - self.reliability)
        self.fit = fit; self.fit_parts = (cov, rel, conflict)
        return fit

    def _schema_route(self, target: int):
        """(neurogénesis permitida en el almacén, factor de tasa del sistema lento).

          schema="measure"  sólo mide el encaje (diagnóstico, sin efecto)
          schema="gate"     encaja (>= `schema_fit`): asimilación rápida en el lento (tasa
                            x `schema_boost`) y sin neurona nueva en el almacén; no encaja:
                            queda en el almacén (neurogénesis habilitada) y el lento aprende
                            con tasa x `schema_misfit_lr` (0 = sólo desde el repaso)
          schema="always"   asimilación rápida de todo (ablación: sin medir el encaje)
        """
        fit = self._schema_fit(target)
        if self.schema == "gate":
            if fit >= self.schema_fit:
                self.assimilated += 1
                return False, self.schema_boost
            return None, self.schema_misfit_lr
        if self.schema == "always":
            self.assimilated += 1
            return None, self.schema_boost
        return None, 1.0

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
        lab = self._labels(mem)
        pw = lab[v.winners]
        hit = np.flatnonzero(pw == c2)
        if len(hit):
            j = int(v.winners[hit[0]])
        else:
            if v.norm == 0.0:
                return None
            rows = np.flatnonzero(lab == c2)
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
        pref = self._labels(mem)
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
                a.learn_readout(c, X)                         # pero sí los lectores locales
            idx, h = self._proj_input(codes)
        else:
            idx = np.flatnonzero(x > 0); val = x[idx]
            h = val / np.float32(max(float(np.sqrt(val @ val)), 1e-6))
        if len(idx) == 0:
            return
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
            pref = self._labels(mem)
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
        self._adapt_thresholds()
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
            replays=self.replays, recalls=self.recalls, assimilated=self.assimilated,
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
        if self.beta is not None:
            out["beta"] = self.beta
        if self.homeostasis > 0.0:
            out["umbral"] = self.umbral
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
            if "beta" in z.files:
                net.beta = z["beta"].copy()
            if "umbral" in z.files:
                net.umbral[:] = z["umbral"]
        return net
