"""BioElasticNet: un área de asociación elástica con su capa de decisión.

    volea -> memoria de entrada (retardos o traza lenta) -> columna elástica -> decisión

Es la red completa de una sola área y también el bloque con el que `Cortex` arma
redes profundas. No conoce la modalidad: recibe voleas de latencia.

Memoria de entrada (dos constantes de tiempo dendríticas):
  context/delay  línea de retardos: el frame de hace `lag` pasos llega con sus spikes
                 atrasados `lag * delay`. En código de latencia eso es restarle fuerza,
                 así que el *orden* reciente queda escrito en una sola volea.
  tau            traza lenta: cada aferente conserva su último spike con decaimiento
                 exp(-1/tau). Sin orden, pero con alcance largo (memoria de trabajo).

Tres regímenes de aprendizaje, todos online y locales:
  teach(y)      supervisado: error por salida como tercer factor.
  reinforce(R)  recompensa escalar: R-STDP con baseline (sin etiquetas).
  observe()     no supervisado: sólo STDP + neurogénesis por novedad.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass

import numpy as np

from .layer import ElasticLayer, Volley
from .readout import Readout

_EPS = 1e-3
_TRACE_FLOOR = 0.05


@dataclass
class _State:
    v: Volley
    h: np.ndarray       # fuerza normalizada de cada ganadora (voto)
    V: np.ndarray       # potenciales de salida
    pred: int


class BioElasticNet:
    def __init__(self, n_inputs: int, n_outputs: int, *, context: int = 1, delay: float = 0.15,
                 tau: float = 0.0, grow_on_error: bool = True, rho_sup: float = 0.98,
                 anti: float = 0.3, grow_per_error: int = 4, maintain_every: int = 500,
                 layer: dict | None = None, readout: dict | None = None, seed: int = 0):
        self._config = dict(n_inputs=int(n_inputs), n_outputs=int(n_outputs), context=int(context),
                            delay=float(delay), tau=float(tau), grow_on_error=bool(grow_on_error),
                            rho_sup=float(rho_sup), anti=float(anti),
                            grow_per_error=int(grow_per_error), maintain_every=int(maintain_every),
                            layer=dict(layer or {}), readout=dict(readout or {}), seed=int(seed))
        self.N = int(n_inputs)
        self.tau = float(tau)
        self.context = 1 if self.tau > 0 else int(context)
        self.delay = float(delay)
        self.grow_on_error = bool(grow_on_error)
        self.rho_sup = float(rho_sup)
        self.grow_per_error = int(grow_per_error)
        self.anti = float(anti)
        self.maintain_every = int(maintain_every)
        self.rng = np.random.default_rng(seed)
        self.layer = ElasticLayer(self.N * self.context, **(layer or {}))
        self.readout = Readout(n_outputs, **(readout or {}))
        self.layer.on_add = self._on_add
        self.layer.on_remove = self._on_remove
        self._pending_cls: int | None = None
        self._acc: dict | None = None
        self._buf: deque = deque(maxlen=self.context)
        self._trace = np.zeros(self.N, dtype=np.float32) if self.tau > 0 else None
        self._decay = np.float32(np.exp(-1.0 / self.tau)) if self.tau > 0 else None
        self._last: _State | None = None

    # ------------------------------------------------------------------ hooks estructurales
    def _on_add(self, j: int):
        self.readout.add_input(self._pending_cls)

    def _on_remove(self, j: int, last: int):
        self.readout.remove_input(j, last)
        if self._acc is not None:
            self._acc.pop(j, None)
            if last != j and last in self._acc:
                self._acc[j] = self._acc.pop(last)

    def _recruit(self, v: Volley, cls: int | None) -> int:
        self._pending_cls = cls
        j = self.layer.recruit(v)
        self._pending_cls = None
        return j

    # ------------------------------------------------------------------ entrada
    @staticmethod
    def _as_sparse(frame):
        if isinstance(frame, (int, np.integer)):             # one-hot (símbolo)
            return np.array([int(frame)], dtype=np.int64), np.ones(1, dtype=np.float32)
        if isinstance(frame, tuple):                          # ya viene (idx, val)
            return frame
        f = np.asarray(frame, dtype=np.float32).ravel()
        idx = np.flatnonzero(f > 0)
        return idx, f[idx]

    def reset_context(self):
        self._buf.clear()
        if self._trace is not None:
            self._trace[:] = 0.0

    def _push(self, frame):
        idx, val = self._as_sparse(frame)
        if self._trace is None:
            self._buf.appendleft((idx, val))
        else:
            self._trace *= self._decay
            self._trace[idx] = np.maximum(self._trace[idx], val)

    def _composite(self):
        if self._trace is not None:
            idx = np.flatnonzero(self._trace > _TRACE_FLOOR)
            return idx, self._trace[idx]
        if self.context == 1:
            return self._buf[0]
        idxs, vals = [], []
        for lag, (i, v) in enumerate(self._buf):
            if lag:
                v = v - np.float32(lag * self.delay)
                keep = v > 0
                if not keep.all():
                    i = i[keep]; v = v[keep]
                i = i + lag * self.N
            idxs.append(i); vals.append(v)
        return np.concatenate(idxs), np.concatenate(vals)

    # ------------------------------------------------------------------ inferencia
    def _votes(self, v: Volley) -> np.ndarray:
        h = (1.0 - v.times + _EPS).astype(np.float32)
        return h / h.sum() if len(h) else h

    def forward(self, frame, explore: bool = False) -> int:
        """Procesa un frame y devuelve la salida. Guarda el estado para aprender después."""
        self._push(frame)
        idx, val = self._composite()
        v = self.layer.forward(idx, val)
        h = self._votes(v)
        V = self.readout.potentials(v.winners, h)
        if explore:
            pred = int(self.rng.choice(self.readout.C, p=self.readout.probs(V)))
        else:
            pred = int(np.argmax(V))
        self._last = _State(v, h, V, pred)
        return pred

    predict = forward

    # ------------------------------------------------------------------ aprendizaje
    def _after_learning(self):
        if self.maintain_every and self.layer.t % self.maintain_every == 0:
            self.layer.maintain()

    def teach(self, target: int, pred: int | None = None, p: np.ndarray | None = None):
        """Supervisado: corrige la decisión y adapta/crea neuronas ocultas.

        `pred` y `p` permiten que varias áreas compartan una misma decisión y un mismo
        error (el neuromodulador es global): los pasa `Cortex`.
        """
        st = self._last; v = st.v; L = self.layer; R = self.readout
        pred = st.pred if pred is None else pred
        L.tick()
        wn = v.winners
        pref = R.preferred(wn)
        if not v.novel:                                       # sin spike presináptico no hay plasticidad
            R.learn(wn, st.h, target, st.V, p)
        agree = np.flatnonzero(pref == target)
        best_agree = float(v.cos[agree[0]]) if len(agree) and not v.novel else 0.0
        grow = v.novel or (self.grow_on_error and pred != target and best_agree < self.rho_sup)
        if grow and self._recruit(v, target) >= 0:
            pass
        elif len(wn):
            if len(agree):                                    # la más temprana que acierta: LTP
                j = int(wn[agree[0]])
                L.stdp(j, v, +1.0); L.credit(j)
            if pref[0] >= 0 and pref[0] != target:            # la primera, si se equivoca: anti-STDP
                L.stdp(int(wn[0]), v, -self.anti)
        self._after_learning()

    def reinforce(self, reward: float, pred: int | None = None, p: np.ndarray | None = None):
        """Recompensa escalar (sin etiqueta): R-STDP con baseline."""
        st = self._last; v = st.v; L = self.layer; R = self.readout
        pred = st.pred if pred is None else pred
        L.tick()
        wn = v.winners
        if v.novel:                                           # columna en silencio: neurona nueva
            self._recruit(v, None)
        else:
            p = R.probs(st.V) if p is None else p
            p_act = float(p[pred])
            R.learn_reward(wn, st.h, pred, reward, st.V, p)
            if reward > 0:
                j = int(wn[0]); L.stdp(j, v, +1.0); L.credit(j)
            elif self.grow_on_error and p_act > 0.5 and v.cos[0] < self.rho_sup:
                self._recruit(v, None)                        # seguro y equivocado: dividir
        self._after_learning()

    def observe(self, frame=None):
        """No supervisado: sólo aprende representación. Sin `frame` usa el último forward."""
        if frame is not None:
            self.forward(frame)
        v = self._last.v; L = self.layer
        L.tick()
        if v.novel:
            self._recruit(v, None)
        else:
            j = int(v.winners[0]); L.stdp(j, v, +1.0); L.credit(j)
        self._after_learning()

    def step(self, frame, target: int) -> int:
        """Test-then-train: predice, y recién después aprende del objetivo."""
        pred = self.forward(frame)
        self.teach(target)
        return pred

    # ------------------------------------------------------------------ secuencias
    def run_sequence(self, frames, target: int | None = None, learn: bool = True,
                     decide: str = "end") -> int:
        """Clasifica un segmento de frames (p. ej. una palabra hablada).

        decide="end"        : una sola decisión sobre la volea que contiene todo el
                              segmento vía la línea de retardos (context >= len(frames)).
        decide="accumulate" : decisión por evidencia acumulada frame a frame.
        """
        self.reset_context()
        if decide == "end":
            for f in frames[:-1]:
                self._push(f)
            pred = self.forward(frames[-1])
            if learn and target is not None:
                self.teach(target)
            return pred

        L = self.layer; R = self.readout
        self._acc = {}
        orphans = []                                          # frames sin ninguna ganadora del objetivo
        for f in frames:
            self._push(f)
            idx, val = self._composite()
            v = L.forward(idx, val)
            if learn and target is not None:
                L.tick()
                if v.novel:
                    j = self._recruit(v, target)
                    if j >= 0:
                        self._acc[j] = self._acc.get(j, 0.0) + 1.0
                    continue
                agree = np.flatnonzero(R.preferred(v.winners) == target)
                if len(agree):
                    j = int(v.winners[agree[0]]); L.stdp(j, v, +1.0); L.credit(j)
                else:
                    orphans.append(v)
            for j, hj in zip(v.winners.tolist(), self._votes(v).tolist()):
                self._acc[j] = self._acc.get(j, 0.0) + hj
        idx = np.fromiter(self._acc.keys(), dtype=np.int64, count=len(self._acc))
        h = np.fromiter(self._acc.values(), dtype=np.float32, count=len(self._acc))
        self._acc = None
        if len(h):
            h /= h.sum()
        V = R.potentials(idx, h)
        pred = int(np.argmax(V))
        if learn and target is not None:
            R.learn(idx, h, target, V)
            if self.grow_on_error and pred != target:         # error: neuronas para los peores frames
                orphans.sort(key=lambda u: float(u.cos[0]))
                for u in orphans[:self.grow_per_error]:
                    self._recruit(u, target)
            self._after_learning()
        return pred

    # ------------------------------------------------------------------ estado
    def stats(self) -> dict:
        L = self.layer
        return dict(neurons=L.n, created=L.created, pruned=L.pruned, evicted=L.evicted)

    def state(self) -> dict:
        """Estructura, pesos y contadores como diccionario de arrays."""
        L, R = self.layer, self.readout
        n = L.n
        out = {f"L_{name}": getattr(L, name)[:n] for name in L._STATS}
        out.update(config=np.array(json.dumps(self._config)), L_pre=L.pre[:n], L_w=L.w[:n],
                   L_meta=np.array([L.t, L.created, L.pruned, L.evicted], dtype=np.int64),
                   R_W=R.W[:, :n], R_count=R.count[:n], R_baseline=np.float64(R.baseline))
        return out

    @classmethod
    def from_state(cls, z) -> "BioElasticNet":
        cfg = json.loads(str(z["config"]))
        n = z["L_pre"].shape[0]
        cap = max(256, 1 << int(np.ceil(np.log2(max(n, 1)))))
        cfg["layer"]["capacity"] = cap; cfg["readout"]["capacity"] = cap
        net = cls(cfg.pop("n_inputs"), cfg.pop("n_outputs"), **cfg)
        L, R = net.layer, net.readout
        L.pre[:n] = z["L_pre"]; L.w[:n] = z["L_w"]
        for name in L._STATS:
            getattr(L, name)[:n] = z[f"L_{name}"]
        L.t, L.created, L.pruned, L.evicted = (int(v) for v in z["L_meta"])
        L.n = n
        R.W[:, :n] = z["R_W"]; R.count[:n] = z["R_count"]; R.n = n
        R.baseline = float(z["R_baseline"])
        return net

    def save(self, path):
        np.savez_compressed(path, **self.state())

    @classmethod
    def load(cls, path) -> "BioElasticNet":
        with np.load(path, allow_pickle=False) as z:
            return cls.from_state(z)
