"""Capa elástica: competencia TTFS + STDP + homeostasis + neurogénesis + poda.

La capa no conoce la modalidad: recibe voleas (idx, val) de aferentes activas y
devuelve qué neuronas dispararon primero y cuándo.

Elasticidad (plasticidad estructural):
  * Neurogénesis: `recruit` imprime la volea actual en una neurona nueva. Quién
    decide cuándo llamarla es la red (novedad: la capa quedó en silencio; o error
    con mala coincidencia).
  * Supervivencia en período crítico: una neurona nueva que no vuelve a ser útil
    `survival_hits` veces antes de `probation` pasos, muere (`maintain`).
  * Consolidación: la tasa de STDP de cada neurona decae con su uso; las neuronas
    viejas son estables y no se pisan con datos nuevos.
  * Presupuesto: con `max_neurons`, reclutar desaloja a la de menor utilidad reciente.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neuron import first_spike_times


@dataclass
class Volley:
    """Resultado de un pase por la capa."""
    idx: np.ndarray        # aferentes activas
    val: np.ndarray        # fuerza (1 - t_spike) de cada aferente activa
    norm: float            # ||val||
    winners: np.ndarray    # neuronas que dispararon, en orden de spike (k-WTA)
    times: np.ndarray      # tiempo de spike de cada ganadora, en (0, 1]
    cos: np.ndarray        # coincidencia (V final normalizado) de cada ganadora
    novel: bool            # True si ninguna neurona alcanzó el umbral


class ElasticLayer:
    def __init__(self, n_inputs: int, *, k: int = 4, theta: float = 0.7,
                 theta_plus: float = 0.0, theta_max: float = 0.15, theta_tau: float = 2000.0,
                 lr: float = 0.3, lr_min: float = 0.01, lr_half: float = 10.0,
                 max_neurons: int | None = None, probation: int = 20000,
                 survival_hits: int = 1, util_tau: float = 20000.0,
                 max_candidates: int = 64, capacity: int = 256):
        self.D = int(n_inputs)
        self.k = int(k)
        self.theta = float(theta)
        self.theta_plus = float(theta_plus)
        self.theta_max = float(theta_max)
        self.theta_tau = float(theta_tau)
        self.lr = float(lr); self.lr_min = float(lr_min); self.lr_half = float(lr_half)
        self.max_neurons = max_neurons
        self.probation = int(probation)
        self.survival_hits = int(survival_hits)
        self.util_tau = float(util_tau)
        self.max_candidates = int(max_candidates)

        self.cap = int(capacity)
        self.n = 0                      # neuronas vivas
        self.t = 0                      # pasos de aprendizaje
        self.created = 0; self.pruned = 0; self.evicted = 0
        self.Wt = np.zeros((self.D, self.cap), dtype=np.float32)   # pesos, una columna por neurona
        self._alloc_stats(self.cap)
        self.on_add = None              # hook(j)
        self.on_remove = None           # hook(j, last): la neurona `last` pasó a la posición j

    # ------------------------------------------------------------------ memoria
    def _alloc_stats(self, cap):
        self.wins = np.zeros(cap, dtype=np.int32)      # veces que aprendió (STDP+)
        self.hits = np.zeros(cap, dtype=np.int32)      # veces que fue útil
        self.born = np.zeros(cap, dtype=np.int64)
        self.util = np.zeros(cap, dtype=np.float32)    # EMA perezosa de uso
        self.util_t = np.zeros(cap, dtype=np.int64)
        self.hoff = np.zeros(cap, dtype=np.float32)    # offset homeostático de umbral
        self.hoff_t = np.zeros(cap, dtype=np.int64)

    _STATS = ("wins", "hits", "born", "util", "util_t", "hoff", "hoff_t")

    def _grow_capacity(self):
        new_cap = self.cap * 2
        Wt = np.zeros((self.D, new_cap), dtype=np.float32)
        Wt[:, :self.n] = self.Wt[:, :self.n]
        self.Wt = Wt
        old = {name: getattr(self, name) for name in self._STATS}
        self._alloc_stats(new_cap)
        for name, arr in old.items():
            getattr(self, name)[:self.n] = arr[:self.n]
        self.cap = new_cap

    # ------------------------------------------------------------------ dinámica
    def thresholds(self) -> np.ndarray:
        n = self.n
        if self.theta_plus <= 0.0:
            return np.full(n, self.theta, dtype=np.float32)
        decay = np.exp(-(self.t - self.hoff_t[:n]) / self.theta_tau)
        return (self.theta + self.hoff[:n] * decay).astype(np.float32)

    def forward(self, idx: np.ndarray, val: np.ndarray) -> Volley:
        """Pase TTFS: devuelve las k primeras neuronas en disparar."""
        norm = float(np.sqrt(val @ val)) if len(val) else 0.0
        empty = np.empty(0, dtype=np.int64)
        if self.n == 0 or norm == 0.0:
            return Volley(idx, val, norm, empty, np.empty(0), np.empty(0), True)
        n = self.n
        cos = (val @ self.Wt[idx, :n]) / norm          # V(1) normalizado de cada neurona
        theta = self.thresholds()
        cand = np.flatnonzero(cos >= theta)
        if len(cand) == 0:
            # Capa en silencio: decisión forzada por la de mayor potencial (sin spike).
            j = int(np.argmax(cos))
            return Volley(idx, val, norm, np.array([j]), np.array([1.0]), cos[j:j + 1], True)
        if len(cand) > self.max_candidates:            # inhibición: sólo las de mayor margen
            margin = cos[cand] - theta[cand]
            cand = cand[np.argpartition(-margin, self.max_candidates - 1)[:self.max_candidates]]
        order = np.argsort(-val, kind="stable")        # más fuerte = más temprano
        t_sorted = 1.0 - val[order].astype(np.float64)
        Wc = self.Wt[idx[order][:, None], cand[None, :]]
        t_star = first_spike_times(Wc, t_sorted, theta[cand].astype(np.float64), 1.0 / norm)
        t_star = np.minimum(t_star, 1.0)               # cos >= theta garantiza cruce en t <= 1
        k = min(self.k, len(cand))
        sel = np.argsort(t_star, kind="stable")[:k]
        w = cand[sel]
        return Volley(idx, val, norm, w, t_star[sel], cos[w], False)

    # ------------------------------------------------------------------ plasticidad sináptica
    def stdp(self, j: int, v: Volley, mod: float = 1.0):
        """STDP modulado por tercer factor sobre la neurona j.

        mod > 0: Hebbiano. Las aferentes de la volea se potencian en proporción a su
                 adelanto (más temprana = más LTP); las silenciosas se deprimen.
        mod < 0: anti-Hebbiano (la neurona se aleja de este patrón).
        Luego escalado sináptico a norma 1.
        """
        eta = max(self.lr_min, self.lr / (1.0 + self.wins[j] / self.lr_half)) * abs(mod)
        w = self.Wt[:, j]
        x = (v.val / v.norm).astype(np.float32)
        if mod > 0:
            w *= np.float32(1.0 - eta)
            w[v.idx] += np.float32(eta) * x
            self.wins[j] += 1
            if self.theta_plus > 0.0:                  # homeostasis: ganar sube el umbral propio
                cur = self.hoff[j] * np.exp(-(self.t - self.hoff_t[j]) / self.theta_tau)
                self.hoff[j] = min(self.theta_max, cur + self.theta_plus)
                self.hoff_t[j] = self.t
        else:
            w *= np.float32(1.0 + eta)
            w[v.idx] -= np.float32(eta) * x
            np.maximum(w, 0.0, out=w)
        nrm = float(np.sqrt(w @ w))
        if nrm > 1e-12:
            w /= np.float32(nrm)

    def credit(self, j: int):
        """Marca que la neurona j fue útil (alimenta supervivencia y utilidad)."""
        self.hits[j] += 1
        self.util[j] = self.util[j] * np.exp(-(self.t - self.util_t[j]) / self.util_tau) + 1.0
        self.util_t[j] = self.t

    # ------------------------------------------------------------------ plasticidad estructural
    def recruit(self, v: Volley) -> int:
        """Neurogénesis: imprime la volea en una neurona nueva. Devuelve su índice o -1."""
        if v.norm == 0.0:
            return -1
        if self.max_neurons is not None and self.n >= self.max_neurons:
            if not self._evict():
                return -1
        if self.n == self.cap:
            self._grow_capacity()
        j = self.n
        self.Wt[:, j] = 0.0
        self.Wt[v.idx, j] = (v.val / v.norm).astype(np.float32)
        self.wins[j] = 0; self.hits[j] = 0; self.born[j] = self.t
        self.util[j] = 1.0; self.util_t[j] = self.t
        self.hoff[j] = 0.0; self.hoff_t[j] = self.t
        self.n += 1; self.created += 1
        if self.on_add is not None:
            self.on_add(j)
        return j

    def utility(self) -> np.ndarray:
        n = self.n
        return self.util[:n] * np.exp(-(self.t - self.util_t[:n]) / self.util_tau)

    def _evict(self) -> bool:
        if self.n == 0:
            return False
        self.remove(int(np.argmin(self.utility())))
        self.evicted += 1
        return True

    def remove(self, j: int):
        """Elimina la neurona j (swap-remove con la última)."""
        last = self.n - 1
        if j != last:
            self.Wt[:, j] = self.Wt[:, last]
            for name in self._STATS:
                arr = getattr(self, name)
                arr[j] = arr[last]
        self.n -= 1; self.pruned += 1
        if self.on_remove is not None:
            self.on_remove(j, last)

    def maintain(self) -> int:
        """Poda de período crítico: mueren las neuronas nuevas que no volvieron a usarse."""
        n = self.n
        age = self.t - self.born[:n]
        dead = np.flatnonzero((age > self.probation) & (self.hits[:n] < self.survival_hits))
        for j in dead[::-1]:                           # descendente: seguro con swap-remove
            self.remove(int(j))
        return len(dead)

    def tick(self):
        self.t += 1
