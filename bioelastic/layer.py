"""Columna elástica: competencia TTFS + STDP + homeostasis + neurogénesis + poda.

La columna no conoce la modalidad: recibe voleas (idx, val) de aferentes activas y
devuelve qué neuronas dispararon primero y cuándo.

Almacenamiento disperso con *fan-in acotado*: cada neurona guarda a lo sumo `fan_in`
sinapsis (aferente + peso), como una neurona real. La memoria es O(neuronas x fan_in)
y no depende del número de aferentes posibles; esto es lo que permite escalar.

Plasticidad estructural a dos niveles:
  * Sinapsis: el STDP crea sinapsis nuevas hacia aferentes activas (sinaptogénesis) y,
    con el fan-in lleno, reemplaza a las más débiles.
  * Neuronas: `recruit` imprime la volea actual en una neurona nueva (quién decide
    cuándo es la red: novedad o error). Una neurona nueva que no vuelve a ser útil
    antes de `probation` pasos muere (`maintain`). La tasa de STDP de cada neurona
    decae con su uso (consolidación). Con `max_neurons`, reclutar exige liberar lugar:
      evict="utility"  desaloja a la de menor utilidad reciente (la que menos se usó
                       últimamente: las clases ausentes pierden primero).
      evict="merge"    funde las dos neuronas más parecidas del grupo (clase preferida)
                       con más neuronas. El recuerdo no se borra: se comprime. Es el
                       mantenimiento de presupuesto por fusión de los SVM con presupuesto
                       (Wang, Crammer y Vucetic 2012), con cupo parejo entre clases como
                       el muestreador de GDumb (Prabhu et al. 2020).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neuron import first_spike_times


@dataclass
class Volley:
    """Resultado de un pase por la columna."""
    idx: np.ndarray        # aferentes activas
    val: np.ndarray        # fuerza (1 - t_spike) de cada aferente activa
    norm: float            # ||val||
    winners: np.ndarray    # neuronas que dispararon, en orden de spike (k-WTA)
    times: np.ndarray      # tiempo de spike de cada ganadora, en (0, 1]
    cos: np.ndarray        # coincidencia (V final normalizado) de cada ganadora
    novel: bool            # True si ninguna neurona alcanzó el umbral


class ElasticLayer:
    _STATS = ("fill", "wins", "hits", "born", "util", "util_t", "hoff", "hoff_t")

    def __init__(self, n_inputs: int, *, fan_in: int | None = None, k: int = 4, theta: float = 0.7,
                 theta_plus: float = 0.0, theta_max: float = 0.15, theta_tau: float = 2000.0,
                 lr: float = 0.3, lr_min: float = 0.01, lr_half: float = 10.0,
                 max_neurons: int | None = None, probation: int = 20000,
                 survival_hits: int = 1, util_tau: float = 20000.0,
                 max_candidates: int = 64, capacity: int = 256, neurogenesis: bool = True,
                 evict: str = "utility", merge_candidates: int = 48):
        self.D = int(n_inputs)
        self.K = int(min(fan_in or 256, self.D))       # sinapsis por neurona
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
        self.neurogenesis = bool(neurogenesis)     # False: no nacen neuronas (sólo plasticidad sináptica)
        if evict not in ("utility", "merge"):
            raise ValueError(f"evict desconocido: {evict}")
        self.evict = evict
        self.merge_candidates = int(merge_candidates)
        self.merged = 0
        self._rng = np.random.default_rng(0)

        self.cap = int(capacity)
        self.n = 0                      # neuronas vivas
        self.t = 0                      # pasos de aprendizaje
        self.created = 0; self.pruned = 0; self.evicted = 0
        self.pre = np.zeros((self.cap, self.K), dtype=np.int32)     # aferente de cada sinapsis
        self.w = np.zeros((self.cap, self.K), dtype=np.float32)     # peso de cada sinapsis
        self._alloc_stats(self.cap)
        self._s = np.zeros(self.D, dtype=np.float32)                # volea densa (scratch)
        self._slot = np.full(self.D, -1, dtype=np.int32)            # scratch para búsquedas
        self.on_add = None              # hook(j)
        self.on_remove = None           # hook(j, last): la neurona `last` pasó a la posición j
        self.group_of = None            # () -> grupo (clase preferida) de cada neurona, -1 sin grupo
        self.on_merge = None            # hook(a, b, alpha): b se funde en a con peso alpha para a

    # ------------------------------------------------------------------ memoria
    def _alloc_stats(self, cap):
        self.fill = np.zeros(cap, dtype=np.int32)      # sinapsis usadas
        self.wins = np.zeros(cap, dtype=np.int32)      # veces que aprendió (STDP+)
        self.hits = np.zeros(cap, dtype=np.int32)      # veces que fue útil
        self.born = np.zeros(cap, dtype=np.int64)
        self.util = np.zeros(cap, dtype=np.float32)    # EMA perezosa de uso
        self.util_t = np.zeros(cap, dtype=np.int64)
        self.hoff = np.zeros(cap, dtype=np.float32)    # offset homeostático de umbral
        self.hoff_t = np.zeros(cap, dtype=np.int64)

    def _grow_capacity(self):
        new_cap = self.cap * 2
        for name in ("pre", "w"):
            old = getattr(self, name)
            new = np.zeros((new_cap, self.K), dtype=old.dtype)
            new[:self.n] = old[:self.n]
            setattr(self, name, new)
        old = {name: getattr(self, name) for name in self._STATS}
        self._alloc_stats(new_cap)
        for name, arr in old.items():
            getattr(self, name)[:self.n] = arr[:self.n]
        self.cap = new_cap

    def nbytes(self) -> int:
        """Memoria de las sinapsis de las neuronas vivas (aferente int32 + peso float32)."""
        return int(self.n) * self.K * 8

    def synapses(self) -> int:
        return int(self.fill[:self.n].sum())

    def dense(self, j: int) -> np.ndarray:
        """Pesos de la neurona j como vector denso sobre las D aferentes (inspección)."""
        out = np.zeros(self.D, dtype=np.float32)
        f = int(self.fill[j])
        out[self.pre[j, :f]] = self.w[j, :f]
        return out

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
        kf = int(self.fill[:n].max())
        s = self._s
        s[idx] = val
        cos = np.einsum("ij,ij->i", self.w[:n, :kf], s[self.pre[:n, :kf]]) / norm   # V(1) normalizado
        s[idx] = 0.0
        theta = self.thresholds()
        cand = np.flatnonzero(cos >= theta)
        if len(cand) == 0:
            # Columna en silencio: decisión forzada por la de mayor potencial (sin spike).
            j = int(np.argmax(cos))
            return Volley(idx, val, norm, np.array([j]), np.array([1.0]), cos[j:j + 1], True)
        if len(cand) > self.max_candidates:            # inhibición: sólo las de mayor margen
            margin = cos[cand] - theta[cand]
            cand = cand[np.argpartition(-margin, self.max_candidates - 1)[:self.max_candidates]]
        # Tiempo exacto del primer spike de cada candidata.
        order = np.argsort(-val, kind="stable")        # más fuerte = más temprano
        m = len(idx); M = len(cand)
        rank = self._slot
        rank[idx[order]] = np.arange(m, dtype=np.int32)
        R = rank[self.pre[cand, :kf]]                  # posición temporal de cada sinapsis
        rank[idx] = -1
        R = np.where(R < 0, m, R)                      # aferentes silenciosas: columna descartada
        flat = (np.arange(M)[:, None] * (m + 1) + R).ravel()
        Wc = np.bincount(flat, weights=self.w[cand, :kf].ravel(), minlength=M * (m + 1))
        Wc = Wc.reshape(M, m + 1)[:, :m].T             # (m, M) pesos sobre la volea ordenada
        t_sorted = 1.0 - val[order].astype(np.float64)
        t_star = first_spike_times(Wc, t_sorted, theta[cand].astype(np.float64), 1.0 / norm)
        t_star = np.minimum(t_star, 1.0)               # cos >= theta garantiza cruce en t <= 1
        k = min(self.k, M)
        sel = np.argsort(t_star, kind="stable")[:k]
        w = cand[sel]
        return Volley(idx, val, norm, w, t_star[sel], cos[w], False)

    # ------------------------------------------------------------------ plasticidad sináptica
    def stdp(self, j: int, v: Volley, mod: float = 1.0):
        """STDP modulado por tercer factor sobre la neurona j.

        mod > 0: Hebbiano. Las aferentes de la volea se potencian en proporción a su
                 adelanto (más temprana = más LTP) y las silenciosas se deprimen. Si la
                 aferente no tenía sinapsis se crea una; con el fan-in lleno sobreviven
                 las `fan_in` más fuertes.
        mod < 0: anti-Hebbiano (la neurona se aleja de este patrón).
        Luego escalado sináptico a norma 1.
        """
        eta = np.float32(max(self.lr_min, self.lr / (1.0 + self.wins[j] / self.lr_half)) * abs(mod))
        f = int(self.fill[j])
        pre = self.pre[j]; w = self.w[j]
        x = (v.val / v.norm).astype(np.float32)
        slot = self._slot
        slot[pre[:f]] = np.arange(f, dtype=np.int32)
        sl = slot[v.idx]
        slot[pre[:f]] = -1
        hit = sl >= 0
        if mod > 0:
            w[:f] *= np.float32(1.0) - eta
            w[sl[hit]] += eta * x[hit]
            if not hit.all():                          # sinaptogénesis
                new_pre = v.idx[~hit]; new_w = eta * x[~hit]
                m = len(new_pre)
                if f + m <= self.K:
                    pre[f:f + m] = new_pre; w[f:f + m] = new_w; f += m
                else:                                  # recambio: quedan las más fuertes
                    all_pre = np.concatenate([pre[:f], new_pre])
                    all_w = np.concatenate([w[:f], new_w])
                    keep = np.argpartition(-all_w, self.K - 1)[:self.K]
                    pre[:] = all_pre[keep]; w[:] = all_w[keep]; f = self.K
                self.fill[j] = f
            self.wins[j] += 1
            if self.theta_plus > 0.0:                  # homeostasis: ganar sube el umbral propio
                cur = self.hoff[j] * np.exp(-(self.t - self.hoff_t[j]) / self.theta_tau)
                self.hoff[j] = min(self.theta_max, cur + self.theta_plus)
                self.hoff_t[j] = self.t
        else:
            w[:f] *= np.float32(1.0) + eta
            w[sl[hit]] -= eta * x[hit]
            np.maximum(w[:f], 0.0, out=w[:f])
        nrm = float(np.sqrt(w[:f] @ w[:f]))
        if nrm > 1e-12:
            w[:f] /= np.float32(nrm)

    def credit(self, j: int):
        """Marca que la neurona j fue útil (alimenta supervivencia y utilidad)."""
        self.hits[j] += 1
        self.util[j] = self.util[j] * np.exp(-(self.t - self.util_t[j]) / self.util_tau) + 1.0
        self.util_t[j] = self.t

    def forget_afferents(self, ids: np.ndarray):
        """Corta las sinapsis que vienen de las aferentes `ids` (la fuente murió o cambió)."""
        n = self.n
        if n == 0 or len(ids) == 0:
            return
        mark = self._slot
        mark[ids] = 0
        dead = (mark[self.pre[:n]] == 0) & (np.arange(self.K)[None, :] < self.fill[:n, None])
        mark[ids] = -1
        self.w[:n][dead] = 0.0

    # ------------------------------------------------------------------ plasticidad estructural
    def recruit(self, v: Volley) -> int:
        """Neurogénesis: imprime la volea en una neurona nueva. Devuelve su índice o -1."""
        if v.norm == 0.0 or not self.neurogenesis:
            return -1
        if self.max_neurons is not None and self.n >= self.max_neurons:
            if not self._evict():
                return -1
        if self.n == self.cap:
            self._grow_capacity()
        j = self.n
        idx, val = v.idx, v.val
        if len(idx) > self.K:                          # fan-in acotado: las más tempranas
            top = np.argpartition(-val, self.K - 1)[:self.K]
            idx, val = idx[top], val[top]
        m = len(idx)
        self.pre[j] = 0; self.w[j] = 0.0
        self.pre[j, :m] = idx
        self.w[j, :m] = val / np.float32(np.sqrt(val @ val))
        self.fill[j] = m
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
        if self.evict == "merge" and self.n >= 2 and self._merge_most_redundant():
            self.evicted += 1
            return True
        self.remove(int(np.argmin(self.utility())))
        self.evicted += 1
        return True

    def _merge_most_redundant(self) -> bool:
        """Funde el par más parecido (coseno de pesos) dentro del grupo más numeroso."""
        n = self.n
        g = self.group_of() if self.group_of is not None else np.zeros(n, dtype=np.int64)
        vals, cnt = np.unique(g, return_counts=True)
        cnt = np.where(vals < 0, cnt * 2, cnt)                # las no comprometidas primero
        if cnt.max() < 2:
            return False
        cand = np.flatnonzero(g == vals[int(np.argmax(cnt))])
        if len(cand) > self.merge_candidates:
            cand = self._rng.choice(cand, self.merge_candidates, replace=False)
        kf = int(self.fill[cand].max())
        P = self.pre[cand, :kf]; Wt = self.w[cand, :kf]
        U, inv = np.unique(P, return_inverse=True)            # aferentes de los candidatos
        dense = np.zeros((len(cand), len(U)), dtype=np.float32)
        np.add.at(dense, (np.repeat(np.arange(len(cand)), kf), inv.ravel()), Wt.ravel())
        S = dense @ dense.T
        np.fill_diagonal(S, -np.inf)
        a, b = np.unravel_index(int(np.argmax(S)), S.shape)
        ja, jb = int(cand[a]), int(cand[b])
        ma = float(self.wins[ja]) + 1.0; mb = float(self.wins[jb]) + 1.0
        alpha = ma / (ma + mb)
        v = alpha * dense[a] + (1.0 - alpha) * dense[b]
        keep = np.flatnonzero(v > 0)
        if len(keep) > self.K:
            keep = keep[np.argpartition(-v[keep], self.K - 1)[:self.K]]
        m = len(keep)
        self.pre[ja] = 0; self.w[ja] = 0.0
        self.pre[ja, :m] = U[keep]
        self.w[ja, :m] = v[keep] / np.float32(max(float(np.sqrt(v[keep] @ v[keep])), 1e-12))
        self.fill[ja] = m
        u = self.utility()
        self.util[ja] = u[ja] + u[jb]; self.util_t[ja] = self.t
        self.wins[ja] += self.wins[jb]; self.hits[ja] += self.hits[jb]
        self.born[ja] = min(self.born[ja], self.born[jb])
        if self.on_merge is not None:
            self.on_merge(ja, jb, alpha)
        self.remove(jb)
        self.merged += 1
        return True

    def remove(self, j: int):
        """Elimina la neurona j (swap-remove con la última)."""
        last = self.n - 1
        if j != last:
            self.pre[j] = self.pre[last]; self.w[j] = self.w[last]
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
