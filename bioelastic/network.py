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
    _EP_FIELDS = (("y", np.int16), ("t", np.int64), ("ctx", np.int32), ("norm", np.float32),
                  ("due", np.int64), ("gap", np.float32), ("fail", np.float32), ("ok", np.int8),
                  ("tests", np.int32), ("m", np.float32), ("m_t", np.int64), ("cc", np.int16),
                  ("nm", np.float32), ("v", np.float32))
    KEY_DIMS = 32                    # dimensiones de la expectativa de clave de cada recuerdo
    RATE_EVENTS = 16                 # conflictos recientes con los que se mide el ritmo de un par

    def __init__(self, n_inputs: int, n_outputs: int, *, context: int = 1, delay: float = 0.15,
                 tau: float = 0.0, grow_on_error: bool = True, rho_sup: float = 0.98,
                 anti: float = 0.3, grow_per_error: int = 4, maintain_every: int = 500,
                 protect: bool = False, vote_tau: float = 0.0, episodic: bool = False,
                 separation: float = 0.0, separation_relax: float = 0.0, recon_lock: int = 0,
                 causes: bool = False, cause_n: float = 10.0, cause_shift: float = 0.04,
                 cause_tau: float = 2000.0, cause_rate: float = 1.0, cause_streak: int = 4,
                 cause_rho: float = 0.7, cause_est: int = 2000, cause_mode: str = "relabel",
                 layer: dict | None = None, readout: dict | None = None, seed: int = 0):
        self._config = dict(n_inputs=int(n_inputs), n_outputs=int(n_outputs), context=int(context),
                            delay=float(delay), tau=float(tau), grow_on_error=bool(grow_on_error),
                            rho_sup=float(rho_sup), anti=float(anti),
                            grow_per_error=int(grow_per_error), maintain_every=int(maintain_every),
                            protect=bool(protect), vote_tau=float(vote_tau), episodic=bool(episodic),
                            separation=float(separation), separation_relax=float(separation_relax),
                            recon_lock=int(recon_lock), causes=bool(causes), cause_n=float(cause_n),
                            cause_shift=float(cause_shift), cause_tau=float(cause_tau),
                            cause_rate=float(cause_rate),
                            cause_streak=int(cause_streak), cause_rho=float(cause_rho),
                            cause_est=int(cause_est), cause_mode=str(cause_mode),
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
        # Inferencia de causas latentes (ver `_causes`).
        self.causes = bool(causes); self.cause_n = float(cause_n); self.cause_shift = float(cause_shift)
        self.cause_tau = float(cause_tau); self.cause_rate = float(cause_rate)
        self.cause_streak = int(cause_streak); self.cause_rho = float(cause_rho)
        self.cause_est = int(cause_est)
        if cause_mode not in ("relabel", "parallel"):
            raise ValueError(f"cause_mode desconocido: {cause_mode}")
        # "relabel": ante «misma causa» el recuerdo se reescribe (reconsolidación).
        # "parallel": las dos hipótesis quedan en paralelo: el recuerdo viejo se conserva
        # intacto pero pierde voz (validez `ep["v"]`, se recupera con una confirmación) y
        # nace uno nuevo; la evidencia posterior decide. Nada se destruye por una
        # inferencia equivocada (cf. los conjuntos de tareas del Thinker).
        self.cause_mode = cause_mode
        self.superseded = 0
        self._cfirst = np.full(int(n_outputs), -1, dtype=np.int64)   # primera confirmación de cada clase
        self._ci = np.zeros(int(n_outputs), dtype=np.float64)   # intervalo medio entre confirmaciones de cada clase
        self._ct = np.full(int(n_outputs), -1, dtype=np.int64)  # última confirmación
        if self.causes and not episodic:
            raise ValueError("causes necesita la huella episódica (episodic=True)")
        self._pairs: dict = {}                           # (a, b) -> (media de d, media de |d|^2, n, t, decisión, tiempos)
        # Expectativa de clave: cada recuerdo guarda la media de las entradas que lo
        # confirmaron, comprimida por una proyección aleatoria fija a KEY_DIMS dimensiones.
        self._P = None; self._kq = None
        if self.causes:
            D = self.N * self.context
            self._P = (np.random.default_rng([seed, 97]).standard_normal((self.KEY_DIMS, D))
                       / np.sqrt(self.KEY_DIMS)).astype(np.float32)
            self._kq = np.zeros((0, self.KEY_DIMS), dtype=np.float32)
        self.relabeled = 0; self.protected = 0; self.new_causes = 0
        self.protect_classes = np.empty(0, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.layer = ElasticLayer(self.N * self.context, **(layer or {}))
        self.readout = Readout(n_outputs, **(readout or {}))
        self.layer.on_add = self._on_add
        self.layer.on_remove = self._on_remove
        self.layer.group_of = lambda: self.readout.preferred(np.arange(self.layer.n))
        self.layer.on_merge = self.readout.merge_inputs
        self._pending_cls: int | None = None
        self._pending_soft = None
        self.soft_grow = 0.5                # objetivo blando: crecer ante un error sólo si es claro
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
        #   due, gap, fail, ok, tests   calendario y resultado del autoexamen (Cortex)
        #   m, m_t  último margen del sistema lento al reactivarlo, y cuándo (sueño)
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
        if self._pending_soft is not None:                    # nace con la lectura del objetivo blando
            self.readout.add_input(None)
            self.readout.W[:, self.readout.n - 1] = self.readout.w_init * np.asarray(self._pending_soft, dtype=np.float32)
            return
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
            ep["due"][j] = self.layer.t; ep["gap"][j] = 0.0; ep["fail"][j] = 0.0
            ep["ok"][j] = -1; ep["tests"][j] = 0; ep["m"][j] = -1.0; ep["m_t"][j] = -1
            ep["cc"][j] = 0; ep["nm"][j] = 1.0; ep["v"][j] = 1.0
            if self._kq is not None:
                if j >= len(self._kq):
                    kq = np.zeros((len(self.episodes), self.KEY_DIMS), dtype=np.float32)
                    kq[:len(self._kq)] = self._kq
                    self._kq = kq
                self._kq[j] = self._P @ (e / np.float32(max(float(np.sqrt(e @ e)), 1e-12)))

    def _on_remove(self, j: int, last: int):
        self.readout.remove_input(j, last)
        if self.episodes is not None and j != last:
            self.episodes[j] = self.episodes[last]
            for arr in self.ep.values():
                arr[j] = arr[last]
            if self._kq is not None:
                self._kq[j] = self._kq[last]
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
            pref = np.where(y >= 0, y, pref)
        if self.causes and self.cause_mode == "parallel":     # lo superado no se repasa ni se recuerda
            pref = np.where(self.ep["v"][:n] >= 0.5, pref, -1)
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
        if self.causes and self.cause_mode == "parallel" and len(v.winners) and not v.novel:
            h = h * self.ep["v"][v.winners]                   # un recuerdo superado no vota
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
              soft: np.ndarray | None = None, grow: bool | None = None):
        """Supervisado: corrige la decisión y adapta/crea neuronas ocultas.

        `pred` y `p` permiten que varias áreas compartan una misma decisión y un mismo
        error (el neuromodulador es global): los pasa `Cortex`.
        `grow=False` impide la neurogénesis en este paso (lo que el sistema lento ya
        asimiló no necesita una neurona nueva, ver Cortex `schema`).

        Con `protect` (y una lectura con `active_window`), las ganadoras comprometidas con
        una clase ausente no cambian ni su lectura ni sus pesos: el error de lo que se
        enseña ahora no reescribe recuerdos que no se pueden contrastar. Si estorban, la
        respuesta es una neurona nueva, no recablear la vieja.

        `soft` (opcional): el objetivo es una distribución sobre las salidas (p. ej. un
        posterior). La lectura aprende con el tercer factor soft - p; `target` es su clase
        más probable. Una neurona nueva nace con la lectura soft * w_init, y sólo se crece
        ante un error si el objetivo es claro (soft[target] >= `soft_grow`).
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
        relabel = np.zeros(len(wn), dtype=bool)
        self.protect_classes = np.empty(0, dtype=np.int64)
        if self.causes and not v.novel and len(wn):
            relabel, prot = self._causes(v, st, target, pref)
            guard |= prot | relabel                           # lo protegido y lo que se reconsolida aparte
        if not v.novel:                                       # sin spike presináptico no hay plasticidad
            if guard.any():
                if (~guard).any():
                    R.learn(wn[~guard], st.h[~guard], target, st.V, p, soft=soft)
                else:
                    R.mark(target)
            else:
                R.learn(wn, st.h, target, st.V, p, soft=soft)
            if relabel.any():
                self._reconsolidate(wn[relabel], st, target, p)
        agree = np.flatnonzero(pref == target)
        best_agree = float(v.cos[agree[0]]) if len(agree) and not v.novel else 0.0
        clear = soft is None or float(soft[target]) >= self.soft_grow
        allow = (grow is None or grow) and not relabel[:1].any()   # la primera se actualizó: no hace falta otra
        grow = allow and (v.novel or (self.grow_on_error and clear and pred != target
                                      and best_agree < self.rho_sup))
        if grow and self.causes and not v.novel:
            self.new_causes += 1
        self._pending_soft = soft
        recruited = grow and self._recruit(v, target) >= 0
        self._pending_soft = None
        if recruited:
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

    # ------------------------------------------------------------------ causas latentes
    def _causes(self, v: Volley, st, target: int, pref: np.ndarray):
        """Inferencia de causas latentes (Gershman, Monfils, Norman y Niv 2017): ante un
        error de etiqueta, ¿la entrada viene de la misma causa que el recuerdo recuperado
        (su etiqueta cambió: se modifica) o de una causa nueva parecida (memoria nueva, la
        vieja se protege)?

        Medido (validación): el parecido de UNA entrada con el recuerdo (el coseno con la
        primera ganadora, la misma señal que `Cortex.familiarity()` de la ronda de
        integración 3) no separa los dos casos: un pulóver reetiquetado da 0,747 con las
        neuronas de pulóver y un abrigo, 0,751. Tampoco la diferencia con la huella
        episódica (las neuronas nacen de entradas atípicas: novedad o error, y eso sesga
        la comparación) ni una media por etiqueta (se contamina cuando la etiqueta cambia
        de significado). Lo que los separa es el error de predicción de la CLAVE acumulado
        a lo largo de las repeticiones: cada recuerdo j guarda su expectativa de clave k_j,
        la media de las entradas que lo confirmaron (proyectada a KEY_DIMS dimensiones por
        una matriz aleatoria fija). Si es la misma causa, las entradas en conflicto son
        como las que j esperaba: d = P x - k_j promedia cero; si es una causa vecina, d
        apunta siempre hacia el mismo lado.

        Para cada par «a -> b» (una ganadora de la clase a, dormida: lleva `cause_streak`
        conflictos seguidos sin que una entrada de su clase la confirme; el objetivo es b)
        se acumulan, con olvido de
        constante `cause_tau` (contexto temporal), la media m de d y la media s2 de |d|^2.
        Con n observaciones efectivas (a lo sumo 50), T = n |m|^2 / (s2 - |m|^2) es el
        estadístico de Hotelling isotrópico (~1 si es la misma causa) y

            c = (T - 1) / n  ~  |media de d|^2 / varianza de d

        el corrimiento relativo (un tamaño de efecto: no crece con n como T).

        La clave sola no alcanza cuando una clase vecina se superpone con la vieja (medido:
        las camisas que caen en neuronas de remera dan c ~ 0,03, como una remera
        reetiquetada). Lo que cambia es el ritmo (contexto temporal): si la causa es la
        misma, sus entradas siguen llegando tan seguido como antes, ahora en conflicto; de
        una clase vecina sólo llega la parte que se superpone. rho = (intervalo medio entre
        confirmaciones de a cuando estaba activa) / (intervalo medio entre los últimos
        RATE_EVENTS conflictos a -> b).

        Con al menos `cause_n` conflictos: c + 2 e.e. < `cause_shift` (e.e. = sqrt(2 /
        KEY_DIMS) / n, el error típico de c si es la misma causa: reescribir un recuerdo por
        error cuesta mucho más que demorarse), rho >= `cause_rho` y la etiqueta b ya
        establecida (se confirma desde hace al menos `cause_est` pasos: una etiqueta que
        acaba de aparecer nombra algo nuevo; sin esto, clases parecidas que llegan de a una
        y al mismo ritmo, como en chainF, son indistinguibles de un cambio de etiqueta)
        -> la misma causa con otra etiqueta (ritmo y antigüedad sólo se exigen para entrar
        en ese estado: al reconsolidar, los conflictos se van acabando solos):
        reconsolidar las ganadoras de a (`_reconsolidate`); si no, causa nueva:
        neurogénesis, y lo de a queda protegido (sin cambio de lectura ni anti-STDP;
        `protect_classes` lo protege también en el sistema lento). Mientras la evidencia
        no alcanza, se trata como causa nueva (provisoria). Las ganadoras de causas activas
        (su etiqueta se sigue confirmando entre un conflicto y otro: dos clases que
        conviven en esa zona, aunque sus claves se superpongan) aprenden como siempre.
        Con cause_mode="parallel" ninguna inferencia reescribe nada: ante «misma causa»
        el recuerdo viejo queda protegido pero deja de votar y de repasarse (validez 0, que
        una confirmación posterior restituye) y la entrada crea un recuerdo nuevo. Si las
        dos hipótesis no se pueden separar con lo que se sabe en el momento (una clase
        nueva con las mismas claves y el mismo ritmo, contra un cambio de etiqueta), se
        mantienen las dos y decide la evidencia posterior.
        Devuelve (máscara de reconsolidación, máscara de protección) sobre las ganadoras."""
        ep = self.ep; t = self.layer.t; wn = v.winners
        n = len(wn)
        relabel = np.zeros(n, dtype=bool); prot = np.zeros(n, dtype=bool)
        px = self._P[:, v.idx] @ (v.val / np.float32(max(v.norm, 1e-12)))
        agree = np.flatnonzero(pref == target)
        if len(agree):                                        # las que aciertan siguen activas
            ep["cc"][wn[agree]] = 0
            ep["v"][wn[agree]] = 1.0                          # y, si estaban superadas, vuelven a valer
            if self._ct[target] >= 0:                         # ritmo de confirmación de la clase
                dt = float(t - self._ct[target])
                if self._ci[target] <= 0.0:
                    self._ci[target] = dt
                elif dt < 20.0 * self._ci[target]:            # (una vuelta tras una ausencia no cuenta)
                    self._ci[target] += 0.05 * (dt - self._ci[target])
            self._ct[target] = t
            if self._cfirst[target] < 0:
                self._cfirst[target] = t
            j = int(wn[agree[0]])                             # la más temprana ajusta su expectativa
            ep["nm"][j] += 1.0
            self._kq[j] += np.float32(max(1.0 / float(ep["nm"][j]), 0.02)) * (px - self._kq[j])
        conf = np.flatnonzero((pref >= 0) & (pref != target))
        if len(conf) == 0:
            return relabel, prot
        ep["cc"][wn[conf]] = np.minimum(ep["cc"][wn[conf]] + 1, 30000)
        dormant = ep["cc"][wn[conf]] >= self.cause_streak
        if not dormant.any():
            return relabel, prot
        verdicts = {}
        for i, k in enumerate(conf):
            a = int(pref[k])
            if not dormant[i] or a in verdicts:
                continue                                      # una vez por causa: la más temprana
            d = px - self._kq[int(wn[k])]
            key = (a, int(target))
            md, s2, nab, tl, was, times = self._pairs.get(key, (np.zeros_like(px), 0.0, 0.0, t, None, []))
            times = (times + [t])[-self.RATE_EVENTS:]         # ritmo de los conflictos de este par
            nab = nab * float(np.exp(-(t - tl) / self.cause_tau)) + 1.0
            r = 1.0 / min(nab, 50.0)
            md = md + np.float32(r) * (d - md)
            s2 = s2 + r * (float(d @ d) - s2)
            same = None                                       # None: aún sin evidencia
            if nab >= self.cause_n:
                mm = float(md @ md); ne = min(nab, 50.0)
                c = (ne * mm / max(s2 - mm, 1e-9) - 1.0) / ne
                rho = 0.0
                if len(times) == self.RATE_EVENTS and self._ci[a] > 0.0:
                    rho = self._ci[a] * (len(times) - 1) / max(float(times[-1] - times[0]), 1.0)
                known = self._cfirst[target] >= 0 and t - self._cfirst[target] >= self.cause_est
                same = bool(c + 2.0 * np.sqrt(2.0 / self.KEY_DIMS) / ne < self.cause_shift
                            and ((rho >= self.cause_rho and known) or was is True))
            self._pairs[key] = (md, s2, nab, t, same, times)
            verdicts[a] = same
        for i, k in enumerate(conf):
            if not dormant[i]:
                continue                                      # causa activa: discriminación común
            if verdicts.get(int(pref[k])) and self.cause_mode == "relabel":
                relabel[k] = True
                ep["cc"][int(wn[k])] = 0                      # recién actualizada: vuelve a estar activa
            else:
                prot[k] = True
                if verdicts.get(int(pref[k])):                # paralelo: se conserva, pero pierde voz
                    j = int(wn[k])
                    self.superseded += int(ep["v"][j] >= 0.5)
                    ep["v"][j] *= np.float32(1.0 - min(max(self.cause_rate, 0.0), 1.0))
        self.relabeled += int(relabel.sum()); self.protected += int(prot.sum())
        # en el sistema lento se protege sólo lo que se infirió como causa nueva (o sin decidir)
        self.protect_classes = np.unique([int(pref[k]) for i, k in enumerate(conf)
                                          if dormant[i] and not verdicts.get(int(pref[k]))]).astype(np.int64)
        return relabel, prot

    def _reconsolidate(self, cols, st, target: int, p):
        """Reconsolidación de las causas inferidas como la misma: el recuerdo recuperado se
        vuelve lábil y su asociación con la etiqueta se reescribe en un ensayo: el peso que
        tenía hacia su etiqueta vieja pasa a la nueva (fracción `cause_rate`; 1 = se
        intercambian). El contenido del recuerdo (su prototipo y su huella) no cambia; la
        etiqueta guardada del episodio sí."""
        R = self.readout
        rho = np.float32(min(max(self.cause_rate, 0.0), 1.0))
        old = R.preferred(cols)
        for j, a in zip(cols.tolist(), old.tolist()):
            if a < 0 or a == target:
                continue
            d = R.W[a, j] - R.W[target, j]
            R.W[target, j] += rho * d
            R.W[a, j] -= rho * d
        rows = np.zeros(R.C, dtype=bool); rows[target] = True; rows[np.unique(old[old >= 0])] = True
        R._scale_rows(rows)
        new = R.preferred(cols)
        flip = cols[new == target]
        if len(flip):
            self.ep["y"][flip] = target

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
            if self._kq is not None:
                out["key_expect"] = self._kq[:n]
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
                elif name in ("m", "m_t"):
                    arr[:n] = -1
                elif name == "v":
                    arr[:n] = 1.0
                elif name == "norm":
                    e = net.episodes[:n].astype(np.float32)
                    arr[:n] = np.sqrt(np.einsum("ij,ij->i", e, e)) / 255.0
                net.ep[name] = arr
            if net._kq is not None:
                net._kq = np.zeros((cap, cls.KEY_DIMS), dtype=np.float32)
                if "key_expect" in z:
                    net._kq[:n] = z["key_expect"]
        return net

    def save(self, path):
        np.savez_compressed(path, **self.state())

    @classmethod
    def load(cls, path) -> "BioElasticNet":
        with np.load(path, allow_pickle=False) as z:
            return cls.from_state(z)
