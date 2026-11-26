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
"""
from __future__ import annotations

import json

import numpy as np

from .area import FeatureArea
from .network import BioElasticNet
from .readout import Readout


class Cortex:
    def __init__(self, input_shape, n_outputs: int, *, areas=(), heads=({},), projection=None,
                 replay: float = 0.0, joint: bool = True, temperature: float = 0.25,
                 arbiter: float = 0.0, seed: int = 0):
        self._config = dict(input_shape=input_shape, n_outputs=int(n_outputs),
                            areas=[dict(a) for a in areas], heads=[],
                            projection=projection, replay=float(replay), joint=bool(joint),
                            temperature=float(temperature), arbiter=float(arbiter), seed=int(seed))
        self.arbiter = float(arbiter)
        self.beta = None                                      # pesos del árbitro (uno por votante)
        self._L = None
        self.is_sheet = not np.isscalar(input_shape)
        self.input_shape = tuple(int(v) for v in input_shape) if self.is_sheet else int(input_shape)
        self.C = int(n_outputs)
        self.temperature = float(temperature)
        self.replay = float(replay); self._replay_debt = 0.0; self.replays = 0
        self.joint = bool(joint)
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

    # ------------------------------------------------------------------ inferencia
    def _volley(self, src, x, codes):
        parts = self._offsets(src)
        if not parts:
            return x
        idxs, vals = [], []
        for i, off in parts:
            c = codes[i].ravel()
            idx = np.flatnonzero(c)
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

    def _voters(self, x, codes):
        """[(potenciales, temperatura)] de cada votante, en orden fijo."""
        out = []
        for head, src in zip(self.heads, self._src):
            head.forward(self._volley(src, x, codes))
            out.append((head._last.V, head.readout.temperature))
        if self.proj is not None:
            idx, h = self._proj_input(codes)
            self._pv = (idx, h)
            out.append((self.proj.potentials(idx, h), self.proj.temperature))
        self._zs = []
        for a in self.areas:
            z = a.potentials()
            self._zs.append(z)
            if z is not None:
                out.append((z, a.clf.temperature))
        return out

    def forward(self, x, explore: bool = False) -> int:
        codes = []
        if self.areas:
            X = np.asarray(x, dtype=np.float32).reshape(self.input_shape)
            for a in self.areas:
                X = a.encode(X); codes.append(X)
        votes = self._voters(x, codes)
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
            for z, _ in votes:
                V = V + z
        self._V = V
        if explore:
            self.pred = int(self.rng.choice(self.C, p=self._p()))
        else:
            self.pred = int(np.argmax(V))
        return self.pred

    predict = forward

    # ------------------------------------------------------------------ aprendizaje
    def _p(self):
        T = 1.0 if self._arbitrated() else self.temperature     # con árbitro V ya es log-prob
        e = np.exp((self._V - self._V.max()) / T)
        return e / e.sum()

    def _arbitrated(self) -> bool:
        return self.arbiter > 0.0 and self._L is not None

    def teach(self, target: int):
        prior = None                                          # error residual entre áreas (opcional)
        for a, z in zip(self.areas, getattr(self, "_zs", [None] * len(self.areas))):
            a.learn(target, prior)                            # sólo usa el objetivo si tiene error local
            if z is not None:
                prior = z if prior is None else prior + z
        p = self._p()
        arb = self._arbitrated()
        if arb:                                               # el árbitro aprende del error global
            g = self._L[:, target] - self._L @ p
            self.beta = np.maximum(self.beta + np.float32(self.arbiter) * g.astype(np.float32), 0.0)
        for head in self.heads:
            if self.joint and not arb:
                head.teach(target, pred=self.pred, p=p)
            else:                                             # cada área aprende de su propio error
                head.teach(target)
        if self.proj is not None:
            self.proj.learn(*self._pv, target, p=None if arb else p)
            if self.replay > 0.0:
                self._replay_debt += self.replay
                while self._replay_debt >= 1.0:
                    self._replay_debt -= 1.0
                    self._replay_one()

    def _replay_one(self):
        """Consolidación por repetición: un recuerdo del área de asociación (el patrón que
        guarda una de sus neuronas) se reactiva, recorre las áreas sensoriales y entrena la
        proyección directa con la clase de esa neurona. Las clases se eligen de forma pareja,
        así lo viejo se repasa aunque el flujo ya no lo muestre."""
        mem = next((h for h, src in zip(self.heads, self._src) if not self._offsets(src)), None)
        if mem is None or mem.layer.n == 0 or mem.context != 1 or mem.tau > 0:
            return
        L = mem.layer
        pref = mem.readout.preferred(np.arange(L.n))
        classes = np.unique(pref[pref >= 0])
        if len(classes) == 0:
            return
        c = int(self.rng.choice(classes))
        j = int(self.rng.choice(np.flatnonzero(pref == c)))
        x = L.dense(j)
        X = (x / max(float(x.max()), 1e-8)).reshape(self.input_shape)
        codes = []
        for a in self.areas:
            X = a.encode(X); codes.append(X)
            a._cache = None                                   # el repaso no entrena los rasgos
            a.learn_readout(c, X)                             # pero sí los lectores locales
        idx, h = self._proj_input(codes)
        if len(idx) == 0:
            return
        V = self.proj.potentials(idx, h)
        e = np.exp((V - V.max()) / self.temperature)
        self.proj.learn(idx, h, c, p=e / e.sum())
        self.replays += 1

    def reinforce(self, reward: float):
        for a in self.areas:
            a.learn()
        p = self._p()
        for head in self.heads:
            head.reinforce(reward, pred=self.pred, p=p)
        if self.proj is not None:
            self.proj.learn_reward(*self._pv, self.pred, reward, p=p)

    def observe(self, x):
        self.forward(x)
        for a in self.areas:
            a.learn()
        for head in self.heads:
            head.observe()

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

    def save(self, path):
        """Guarda estructura y pesos. El contexto en curso (retardos, trazas) no se guarda."""
        out = {"config": np.array(json.dumps(self._config)),
               "meta": np.array([self.replays], dtype=np.int64)}
        if self.beta is not None:
            out["beta"] = self.beta
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
            if "beta" in z.files:
                net.beta = z["beta"].copy()
        return net
