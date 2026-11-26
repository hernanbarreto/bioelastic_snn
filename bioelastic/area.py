"""Área sensorial con campos receptivos locales (un nivel de la jerarquía cortical).

Entrada y salida son *láminas* (alto, ancho, canales) de fuerzas de spike en [0, 1].
Para una imagen los ejes son espacio; para audio, tiempo x frecuencia.

Microcircuito (simplificación del canónico):
  lámina 4    células simples: cada posición compara su parche local con un
              diccionario elástico de rasgos (coseno = potencial final de una IF).
              El diccionario se comparte entre posiciones (todas las columnas del
              área aprenden la misma estadística), lo que da equivarianza a traslación.
  láminas 2/3 inhibición lateral (k-WTA entre rasgos en cada posición) y células
              complejas (máximo sobre un vecindario): tolerancia a desplazamientos.
              Su salida es la proyección feedforward al área siguiente.

Aprendizaje no supervisado y online: cada rasgo se mueve hacia los parches que gana
(STDP promediado sobre la lámina, con tasa que decae con el uso) y un parche que
ningún rasgo reconoce crea un rasgo nuevo. Un rasgo que no se reutiliza en su
período crítico muere y su lugar queda libre.

A diferencia de `ElasticLayer`, aquí la latencia de salida se calcula desde el
potencial final (no desde el cruce exacto del umbral): es lo que permite resolver
toda la lámina con un producto de matrices.
"""
from __future__ import annotations

import numpy as np


def _pair(x):
    return (int(x), int(x)) if np.isscalar(x) else (int(x[0]), int(x[1]))


class FeatureArea:
    def __init__(self, in_shape, *, rf=5, stride=1, pool=2, max_features: int = 64,
                 theta: float = 0.8, floor: float = 0.5, k: int = 2, lr: float = 0.2,
                 lr_min: float = 0.002, lr_half: float = 20.0, min_energy: float = 0.2,
                 recruit_per_step: int = 2, probation: int = 2000, survival_hits: int = 5,
                 maintain_every: int = 500):
        H, W, C = (int(v) for v in in_shape)
        self.in_shape = (H, W, C)
        self.rf = _pair(rf); self.stride = _pair(stride); self.pool = _pair(pool)
        self.oh = (H - self.rf[0]) // self.stride[0] + 1
        self.ow = (W - self.rf[1]) // self.stride[1] + 1
        self.ph = self.oh // self.pool[0]; self.pw = self.ow // self.pool[1]
        if min(self.ph, self.pw) < 1:
            raise ValueError(f"lámina {in_shape} demasiado chica para rf={rf}, pool={pool}")
        self.F = int(max_features)
        self.out_shape = (self.ph, self.pw, self.F)
        self.D = self.rf[0] * self.rf[1] * C
        self.theta = float(theta); self.floor = float(floor); self.k = int(k)
        self.lr = float(lr); self.lr_min = float(lr_min); self.lr_half = float(lr_half)
        self.min_energy = float(min_energy)
        self.recruit_per_step = int(recruit_per_step)
        self.probation = int(probation); self.survival_hits = int(survival_hits)
        self.maintain_every = int(maintain_every)

        self.W = np.zeros((self.F, self.D), dtype=np.float32)     # diccionario (lámina 4)
        self.alive = np.zeros(self.F, dtype=bool)
        self.wins = np.zeros(self.F, dtype=np.int64)
        self.born = np.zeros(self.F, dtype=np.int64)
        self.n = 0                       # marca de agua: rasgos alguna vez usados
        self.free: list[int] = []        # lugares de rasgos muertos
        self.t = 0
        self.created = 0; self.pruned = 0
        self.plastic = True
        self.on_reuse = None             # hook(f): el lugar f pasa a significar otra cosa

        ys = np.arange(self.oh) * self.stride[0]; xs = np.arange(self.ow) * self.stride[1]
        base = (ys[:, None] * W + xs[None, :]).ravel()
        off = (np.arange(self.rf[0])[:, None] * W + np.arange(self.rf[1])[None, :]).ravel()
        self._gather = base[:, None] + off[None, :]               # (P, rf_h*rf_w)
        self._cache = None

    # ------------------------------------------------------------------ pase
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Lámina de entrada (H, W, C) -> código disperso (ph, pw, F).

        Las posiciones sin energía suficiente (`valid` falso) dan código nulo: la
        competencia, el umbral y el pooling se calculan sólo sobre las válidas. El
        producto con el diccionario se hace completo, como siempre: un producto sobre un
        subconjunto de filas puede cambiar los bits del resultado según la ruta de BLAS.
        """
        H, W, C = self.in_shape
        P = self.oh * self.ow
        Pm = np.asarray(X, dtype=np.float32).reshape(H * W, C)[self._gather].reshape(P, self.D)
        norms = np.sqrt(np.einsum("pd,pd->p", Pm, Pm))
        mx = float(norms.max())
        valid = norms >= max(self.min_energy * mx, 1e-6)
        vr = np.flatnonzero(valid)
        n = self.n
        act = None
        code = np.zeros(self.out_shape, dtype=np.float32)
        if n and mx > 0:
            full = Pm @ self.W[:n].T
            rows = vr
            if self.floor < 0.0:                                  # el código no es nulo fuera de `valid`
                full /= np.maximum(norms, 1e-6)[:, None]
                full[~valid] = 0.0
                rows = np.arange(P); a_rows = full; act = full[vr]
            else:
                act = a_rows = full[vr] / np.maximum(norms[vr], 1e-6)[:, None]
            gain = np.float32(1.0 / (1.0 - self.floor))
            if self.floor >= 0.0:                                 # mismas operaciones, sin temporales
                out = a_rows - np.float32(self.floor)
                np.maximum(out, 0.0, out=out)
                out *= gain
                if n > self.k:                                    # inhibición lateral: k rasgos por posición
                    kth = np.partition(a_rows, n - self.k, axis=1)[:, n - self.k]
                    out[~(a_rows >= kth[:, None])] = 0.0          # (perdedoras: max(0 - floor, 0) = 0)
            else:
                out = a_rows
                if n > self.k:
                    kth = np.partition(a_rows, n - self.k, axis=1)[:, n - self.k]
                    out = np.where(a_rows >= kth[:, None], a_rows, 0.0)
                out = np.maximum(out - self.floor, 0.0) * gain
            if len(rows) < P:
                sheet = np.zeros((P, n), dtype=np.float32)
                sheet[rows] = out
            else:
                sheet = out
            ph, pw = self.ph, self.pw; a, b = self.pool
            o = sheet.reshape(self.oh, self.ow, n)[:ph * a, :pw * b].reshape(ph, a, pw * b * n)
            code[:, :, :n] = o.max(axis=1).reshape(ph, pw, b, n).max(axis=2)   # células complejas
        self._cache = (Pm, norms, vr, act)
        return code

    # ------------------------------------------------------------------ plasticidad
    def learn(self):
        """Actualiza el diccionario con la última lámina vista (no supervisado)."""
        if not self.plastic or self._cache is None:
            return
        Pm, norms, vr, act = self._cache
        self._cache = None
        self.t += 1
        n = self.n
        if act is not None:
            best = act.argmax(axis=1)
            bc = act[np.arange(len(best)), best]
        else:
            best = np.zeros(len(vr), dtype=np.int64); bc = np.zeros(len(vr), dtype=np.float32)
        ok = bc >= self.theta
        sel = vr[ok]                                              # posiciones válidas reconocidas
        if len(sel):                                              # STDP promediado sobre la lámina
            b = best[ok]
            u, inv = np.unique(b, return_inverse=True)            # sólo los rasgos que ganaron
            M = np.zeros((len(u), self.D), dtype=np.float32)
            np.add.at(M, inv, Pm[sel] / norms[sel, None])         # mismo orden de suma que antes
            cnt = np.bincount(inv, minlength=len(u))
            eta = np.maximum(self.lr_min, self.lr / (1.0 + self.wins[u] / self.lr_half)).astype(np.float32)
            Wn = (1.0 - eta)[:, None] * self.W[u] + eta[:, None] * (M / cnt[:, None])
            self.W[u] = Wn / np.maximum(np.linalg.norm(Wn, axis=1, keepdims=True), 1e-8)
            self.wins[u] += 1
        nov = vr[bc < self.theta]
        if len(nov):                                              # neurogénesis por novedad
            nov = nov[np.argsort(-norms[nov])][:16]               # primero los parches con más energía
            fresh = []
            for c in nov:
                if len(fresh) >= self.recruit_per_step:
                    break
                p = Pm[c] / norms[c]
                if fresh and float((self.W[fresh] @ p).max()) >= self.theta:
                    continue                                      # ya cubierto por uno recién creado
                f = self._new_slot()
                if f < 0:
                    break
                self.W[f] = p; self.alive[f] = True
                self.wins[f] = 0; self.born[f] = self.t
                self.created += 1
                fresh.append(f)
        if self.maintain_every and self.t % self.maintain_every == 0:
            self.maintain()

    def _new_slot(self) -> int:
        if self.free:
            f = self.free.pop()
            if self.on_reuse is not None:
                self.on_reuse(f)
            return f
        if self.n < self.F:
            self.n += 1
            return self.n - 1
        return -1

    def maintain(self) -> int:
        """Poda de período crítico: mueren los rasgos nuevos que casi no se reutilizaron."""
        age = self.t - self.born
        dead = np.flatnonzero(self.alive & (age > self.probation) & (self.wins < self.survival_hits))
        for f in dead:
            self.alive[f] = False
            self.W[f] = 0.0
            self.free.append(int(f))
        self.pruned += len(dead)
        return len(dead)

    def forget_channel(self, f: int):
        """El canal de entrada f cambió de significado: se cortan sus sinapsis."""
        C = self.in_shape[2]
        self.W[:, f::C] = 0.0

    # ------------------------------------------------------------------ estado
    def stats(self) -> dict:
        feats = int(self.alive.sum())
        return dict(features=feats, created=self.created, pruned=self.pruned,
                    units=feats * (self.oh * self.ow + self.ph * self.pw),   # simples + complejas
                    synapses=feats * self.D)

    def cost(self) -> dict:
        """Operaciones por lámina codificada. El diccionario se comparte entre posiciones:
        se guardan F x D sinapsis pero se evalúan P x F x D (una por posición)."""
        n = self.n
        P = self.oh * self.ow
        return dict(stored_synapses=int(self.alive.sum()) * self.D, macs=P * n * self.D,
                    memory_bytes=self.W.nbytes + self.alive.nbytes + self.wins.nbytes + self.born.nbytes)

    def state(self) -> dict:
        return dict(W=self.W, alive=self.alive, wins=self.wins, born=self.born,
                    free=np.array(self.free, dtype=np.int64),
                    meta=np.array([self.n, self.t, self.created, self.pruned], dtype=np.int64))

    def restore(self, z, prefix: str = ""):
        self.W[:] = z[prefix + "W"]; self.alive[:] = z[prefix + "alive"]
        self.wins[:] = z[prefix + "wins"]; self.born[:] = z[prefix + "born"]
        self.free = [int(v) for v in z[prefix + "free"]]
        self.n, self.t, self.created, self.pruned = (int(v) for v in z[prefix + "meta"])
