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
    _EP_FIELDS = (("y", np.int16), ("t", np.int64), ("ctx", np.int32), ("norm", np.float32))

    def __init__(self, n_inputs: int, n_outputs: int, *, context: int = 1, delay: float = 0.15,
                 tau: float = 0.0, grow_on_error: bool = True, rho_sup: float = 0.98,
                 anti: float = 0.3, grow_per_error: int = 4, maintain_every: int = 500,
                 protect: bool = False, vote_tau: float = 0.0, episodic: bool = False,
                 separation: float = 0.0, separation_relax: float = 0.0, recon_lock: int = 0,
                 layer: dict | None = None, readout: dict | None = None, seed: int = 0):
        self._config = dict(n_inputs=int(n_inputs), n_outputs=int(n_outputs), context=int(context),
                            delay=float(delay), tau=float(tau), grow_on_error=bool(grow_on_error),
                            rho_sup=float(rho_sup), anti=float(anti),
                            grow_per_error=int(grow_per_error), maintain_every=int(maintain_every),
                            protect=bool(protect), vote_tau=float(vote_tau), episodic=bool(episodic),
                            separation=float(separation), separation_relax=float(separation_relax),
                            recon_lock=int(recon_lock),
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
        self.protect = bool(protect)        # no tocar neuronas de clases ausentes (ver `teach`)
        self.vote_tau = float(vote_tau)     # > 0: voto por orden de llegada (ver `_votes`)
        self.separation = float(separation) # > 0: separación de patrones al reclutar (ver `_separate`)
        # > 0: separación adaptativa (propuesta de Continuo 2): el margen al reclutar para la
        # clase c es separation x relax / (relax + usos de c); una clase que recién llega
        # se separa con todo el margen y una ya establecida, con menos.
        self.separation_relax = float(separation_relax)
        self._uses = np.zeros(int(n_outputs), dtype=np.float64)
        # > 0: reconsolidación de los prototipos: una neurona que ganó (LTP) `recon_lock`
        # veces queda bloqueada; sólo vuelve a moverse si el área se equivocó en ese paso
        # (error de predicción al recuperarla). Ver Readout `recon_lock` para la lectura.
        self.recon_lock = int(recon_lock)
        self.locked_skips = 0
        self.rng = np.random.default_rng(seed)
        self.layer = ElasticLayer(self.N * self.context, **(layer or {}))
        self.readout = Readout(n_outputs, **(readout or {}))
        self.layer.on_add = self._on_add
        self.layer.on_remove = self._on_remove
        self.layer.group_of = lambda: self.readout.preferred(np.arange(self.layer.n))
        self.layer.on_merge = self.readout.merge_inputs
        self._pending_cls: int | None = None
        self._pending_v: Volley | None = None
        # Huella episódica: cada neurona guarda la volea completa que la creó (uint8, sin el
        # recorte del fan-in). Es lo que se reactiva en el repaso (ver Cortex._replay_one).
        # Con la huella, el área es el almacén rápido de un sistema de aprendizaje
        # complementario (McClelland, McNaughton y O'Reilly 1995; Kumaran, Hassabis y
        # McClelland 2016): cada episodio es (código, etiqueta, contexto) escrito en un solo
        # ensayo, con capacidad finita (`max_neurons`) y olvido (desalojo y poda). Los
        # campos `ep` acompañan a cada neurona (se mueven con el swap-remove):
        #   y      etiqueta con la que se escribió (-1 si llegó sin etiqueta); no deriva
        #          como la lectura aprendida
        #   t      paso en que se escribió;  ctx  etiqueta de contexto (`context_tag`)
        #   norm   norma del episodio (para recuperar por coseno, ver `recall`)
        self.episodic = bool(episodic)
        self.episodes = np.zeros((0, self.layer.D), dtype=np.uint8) if self.episodic else None
        self.ep = {name: np.zeros(0, dtype=dt) for name, dt in self._EP_FIELDS} if self.episodic else None
        self.context_tag = 0
        self.watch = -1                     # índice de una neurona que alguien sigue (swap-remove)
        self._acc: dict | None = None
        self._buf: deque = deque(maxlen=self.context)
        self._trace = np.zeros(self.N, dtype=np.float32) if self.tau > 0 else None
        self._decay = np.float32(np.exp(-1.0 / self.tau)) if self.tau > 0 else None
        self._last: _State | None = None

    # ------------------------------------------------------------------ hooks estructurales
    def _on_add(self, j: int):
        self.readout.add_input(self._pending_cls)
        if self.episodes is not None:
            if j >= len(self.episodes):
                m = max(256, 2 * len(self.episodes))
                ep = np.zeros((m, self.layer.D), dtype=np.uint8)
                ep[:len(self.episodes)] = self.episodes
                self.episodes = ep
            if j >= len(self.ep["y"]):
                m = len(self.episodes)
                for name, arr in self.ep.items():
                    new = np.zeros(m, dtype=arr.dtype); new[:len(arr)] = arr
                    self.ep[name] = new
            self.episodes[j] = 0
            v = self._pending_v
            if v is not None and len(v.idx):
                self.episodes[j, v.idx] = np.round(np.clip(v.val, 0.0, 1.0) * 255.0).astype(np.uint8)
            e = self.episodes[j].astype(np.float32)
            ep = self.ep
            ep["y"][j] = -1 if self._pending_cls is None else int(self._pending_cls)
            ep["t"][j] = self.layer.t; ep["ctx"][j] = int(self.context_tag)
            ep["norm"][j] = np.float32(np.sqrt(e @ e) / 255.0)

    def _on_remove(self, j: int, last: int):
        self.readout.remove_input(j, last)
        if self.episodes is not None and j != last:
            self.episodes[j] = self.episodes[last]
            for arr in self.ep.values():
                arr[j] = arr[last]
        if self.watch == j:
            self.watch = -1
        elif self.watch == last:
            self.watch = j
        if self._acc is not None:
            self._acc.pop(j, None)
            if last != j and last in self._acc:
                self._acc[j] = self._acc.pop(last)

    def _recruit(self, v: Volley, cls: int | None) -> int:
        self._pending_cls = cls; self._pending_v = v
        j = self.layer.recruit(v)
        self._pending_cls = None; self._pending_v = None
        return j

    def episode(self, j: int) -> np.ndarray:
        """Recuerdo de la neurona j como volea densa sobre las D aferentes: la huella
        episódica si existe, si no sus pesos (recortados al fan-in)."""
        if self.episodes is not None:
            return self.episodes[j].astype(np.float32) / np.float32(255.0)
        return self.layer.dense(j)

    def labels(self, stored: bool = True) -> np.ndarray:
        """Clase de cada neurona: la etiqueta con la que se escribió su episodio (si hay
        huella y `stored`) o, si no, la clase preferida de su lectura aprendida."""
        n = self.layer.n
        pref = self.readout.preferred(np.arange(n))
        if stored and self.ep is not None:
            y = self.ep["y"][:n].astype(np.int64)
            return np.where(y >= 0, y, pref)
        return pref

    def recall(self, idx: np.ndarray, val: np.ndarray, k: int = 8):
        """Recuperación desde el almacén rápido: los `k` episodios más parecidos (coseno
        entre la volea y la huella completa) y su parecido. Sin huella: (vacío, vacío)."""
        n = self.layer.n
        if self.episodes is None or n == 0 or len(idx) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        nv = float(np.sqrt(val @ val))
        if nv == 0.0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        dots = self.episodes[:n, idx].astype(np.float32) @ val.astype(np.float32)
        sim = dots / (np.float32(255.0 * nv) * np.maximum(self.ep["norm"][:n], np.float32(1e-6)))
        k = min(int(k), n)
        top = np.argpartition(-sim, k - 1)[:k] if k < n else np.arange(n)
        top = top[np.argsort(-sim[top], kind="stable")]
        return top, sim[top]

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
        """Peso del voto de cada ganadora.

        Por defecto, la fuerza del spike (1 - t). Como las ganadoras llegan casi juntas
        (t ~ 0.8-0.95), eso es casi un voto por mayoría entre las k: decide la *densidad*
        de neuronas de cada clase cerca de la entrada, no la más parecida.
        Con `vote_tau` > 0, el voto decae con el retraso respecto de la primera:
        exp(-(t - t_1) / vote_tau), un código por orden de llegada (Thorpe 1998)."""
        if self.vote_tau > 0.0 and len(v.times):
            h = np.exp(-(v.times - v.times.min()) / self.vote_tau).astype(np.float32)
        else:
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

    def peek(self, frame, confidence: bool = False):
        """Qué respondería ante `frame`, sin tocar el contexto ni el estado para aprender.
        Con `confidence`, devuelve también si la respuesta sale de conocimiento establecido
        (una neurona que disparó y que ya fue útil antes)."""
        idx, val = self._as_sparse(frame)
        v = self.layer.forward(idx, val)
        out = int(np.argmax(self.readout.potentials(v.winners, self._votes(v))))
        if not confidence:
            return out
        sure = (not v.novel) and len(v.winners) > 0 and self.layer.hits[int(v.winners[0])] >= 1
        return out, bool(sure)

    # ------------------------------------------------------------------ aprendizaje
    def _after_learning(self):
        if self.maintain_every and self.layer.t % self.maintain_every == 0:
            self.layer.maintain()

    def teach(self, target: int, pred: int | None = None, p: np.ndarray | None = None,
              grow: bool | None = None):
        """Supervisado: corrige la decisión y adapta/crea neuronas ocultas.

        `pred` y `p` permiten que varias áreas compartan una misma decisión y un mismo
        error (el neuromodulador es global): los pasa `Cortex`.
        `grow=False` impide la neurogénesis en este paso (lo que el sistema lento ya
        asimiló no necesita una neurona nueva, ver Cortex `schema`).

        Con `protect` (y una lectura con `active_window`), las ganadoras comprometidas con
        una clase ausente no cambian ni su lectura ni sus pesos: el error de lo que se
        enseña ahora no reescribe recuerdos que no se pueden contrastar. Si estorban, la
        respuesta es una neurona nueva, no recablear la vieja.
        """
        st = self._last; v = st.v; L = self.layer; R = self.readout
        pred = st.pred if pred is None else pred
        L.tick()
        self._uses[target] += 1.0
        wn = v.winners
        pref = R.preferred(wn)
        guard = np.zeros(len(wn), dtype=bool)
        if self.protect and R.active_window > 0:
            act = R.active()
            guard = (pref >= 0) & (pref != target) & ~act[np.maximum(pref, 0)]
        if not v.novel:                                       # sin spike presináptico no hay plasticidad
            if guard.any():
                R.learn(wn[~guard], st.h[~guard], target, st.V, p)
            else:
                R.learn(wn, st.h, target, st.V, p)
        agree = np.flatnonzero(pref == target)
        best_agree = float(v.cos[agree[0]]) if len(agree) and not v.novel else 0.0
        allow = grow is None or grow
        grow = allow and (v.novel or (self.grow_on_error and pred != target and best_agree < self.rho_sup))
        if grow and self._recruit(v, target) >= 0:
            if self.separation > 0.0 and not v.novel:
                self._separate(L.n - 1, v, pref, target)
        elif len(wn):
            if len(agree):                                    # la más temprana que acierta: LTP
                j = int(wn[agree[0]])
                if self.recon_lock and pred == target and L.wins[j] >= self.recon_lock:
                    self.locked_skips += 1                    # consolidada y sin error: no cambia
                else:
                    L.stdp(j, v, +1.0)
                L.credit(j)
            if pref[0] >= 0 and pref[0] != target and not guard[0]:   # la primera, si se equivoca: anti-STDP
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

    def _separate(self, j: int, v: Volley, pref: np.ndarray, target: int):
        """Separación de patrones al reclutar (radio de influencia de las redes RCE,
        Reilly, Cooper y Elbaum 1982; «match tracking» de ARTMAP, Carpenter, Grossberg y
        Reynolds 1991). La neurona nueva j nace en la entrada que otra clase reclamaba: su
        umbral propio queda justo por encima de la coincidencia de esa entrada con la
        ganadora de otra clase más parecida, así la recién llegada sólo responde a lo que
        está más cerca de ella que de los recuerdos ajenos y no invade su territorio.
        Las neuronas viejas no cambian. El umbral nunca supera la coincidencia de la
        neurona con la entrada que la creó (si no, no respondería ni a su propio ejemplo)."""
        other = (pref >= 0) & (pref != target)
        if not other.any():
            return
        L = self.layer
        c_other = float(np.max(v.cos[other]))
        own = float(L._cos_at(np.array([j]), v.idx, v.val, v.norm, L._fan_max())[0])
        sep = self.separation
        if self.separation_relax > 0.0:
            sep *= self.separation_relax / (self.separation_relax + float(self._uses[target]))
        th = min(c_other + sep, own - sep)
        if th > L.theta:
            L.set_threshold(j, th)

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
        if self.episodes is not None:
            out["episodes"] = self.episodes[:n]
            out.update({f"ep_{name}": arr[:n] for name, arr in self.ep.items()})
        if self.separation > 0.0:
            out["L_vig"] = L.vig[:n] if L.vig is not None else np.zeros(n, dtype=np.float32)
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
        L.reindex()
        R.W[:, :n] = z["R_W"]; R.count[:n] = z["R_count"]; R.n = n
        R.baseline = float(z["R_baseline"])
        if "L_vig" in z and np.any(z["L_vig"] > 0):
            L.vig = np.zeros(L.cap, dtype=np.float32)
            L.vig[:n] = z["L_vig"]; L._has_vig = True
        if net.episodes is not None and "episodes" in z:
            net.episodes = np.zeros((cap, L.D), dtype=np.uint8)
            net.episodes[:n] = z["episodes"]
            for name, dt in cls._EP_FIELDS:
                arr = np.zeros(cap, dtype=dt)
                if f"ep_{name}" in z:
                    arr[:n] = z[f"ep_{name}"]
                elif name == "y":                             # guardado sin campos: etiqueta = lectura
                    arr[:n] = R.preferred(np.arange(n))
                elif name == "norm":
                    e = net.episodes[:n].astype(np.float32)
                    arr[:n] = np.sqrt(np.einsum("ij,ij->i", e, e)) / 255.0
                net.ep[name] = arr
        return net

    def save(self, path):
        np.savez_compressed(path, **self.state())

    @classmethod
    def load(cls, path) -> "BioElasticNet":
        with np.load(path, allow_pickle=False) as z:
            return cls.from_state(z)
