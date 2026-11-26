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

Error local (opcional, `local=dict(...)` y `n_classes`): el área tiene su propio lector
de clase (lámina 5/6 local) que aprende con la regla de 3 factores sobre su propio
código, y le devuelve a cada célula compleja activa un error dendrítico
g = U^T (objetivo - p). Cada rasgo se mueve hacia (g > 0) o se aleja de (g < 0) el parche
que lo hizo ganar en su vecindario: tres factores (presináptico = parche, postsináptico =
ganadora del pooling, modulador = error local). El error nunca cruza de un área a otra
(no hay backprop entre áreas; es el esquema de clasificadores locales de Nøkland y
Eidnes 2019 y Mostafa et al. 2018). Con `feedback="random"` el lector local es fijo y
aleatorio y no aprende.

Predicción descendente (opcional, `predictive=dict(...)`; apagada por defecto): el área
aprende pesos generativos G (uno por rasgo, con la forma de su campo receptivo) que predicen
su lámina de entrada a partir de su propio código, como en la codificación predictiva de Rao y
Ballard (1999). Cada célula compleja activa proyecta c * G_f sobre el parche de la célula
simple que ganó su pooling (los "switches" del pooling); las predicciones se suman. El residuo
R = entrada - predicción lo calcula la lámina inferior, y G aprende con la regla delta sobre
ese residuo (presináptico = célula compleja de arriba, postsináptico = residuo de la unidad de
abajo): dG_f = lr * sum_q c_q R_q / max(sum_q c_q^2, 1). `pe` es la fracción de la energía
de la entrada que la predicción no explica.

Plasticidad dirigida por el error de predicción (opcional, dentro de `predictive`):
  modulate  exponente b: la tasa del diccionario (STDP y error local) se multiplica por
            g = clip((s / s_media)^b, gain_min, gain_max), donde s es la sorpresa (por defecto
            `pe` del área) y s_media su media móvil (constante `tau`). Mucho error: más
            aprendizaje; poco error: estabilidad.
  neuro     exponente de la misma ganancia sobre la neurogénesis: el número de rasgos que
            pueden nacer en el paso es recruit_per_step * g (con acarreo), y los candidatos se
            ordenan por la energía del residuo en su parche en vez de por la del parche.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .readout import Readout


def _pair(x):
    return (int(x), int(x)) if np.isscalar(x) else (int(x[0]), int(x[1]))


def _segment_sum(keys: np.ndarray, rows: np.ndarray, weights: np.ndarray | None = None):
    """Suma (ponderada) de filas por clave: (claves únicas, sumas). Equivale a `np.add.at`
    sobre un arreglo de ceros, pero como producto disperso (mucho más rápido)."""
    u, inv = np.unique(keys, return_inverse=True)
    w = np.ones(len(keys), dtype=np.float32) if weights is None else weights.astype(np.float32)
    S = sparse.csr_matrix((w, (inv, np.arange(len(keys)))), shape=(len(u), len(keys)))
    return u, np.asarray(S @ rows, dtype=np.float32)


class FeatureArea:
    def __init__(self, in_shape, *, rf=5, stride=1, pool=2, max_features: int = 64,
                 theta: float = 0.8, floor: float = 0.5, k: int = 2, lr: float = 0.2,
                 lr_min: float = 0.002, lr_half: float = 20.0, min_energy: float = 0.2,
                 recruit_per_step: int = 2, probation: int = 2000, survival_hits: int = 5,
                 maintain_every: int = 500, neurogenesis: bool = True,
                 local: dict | None = None, n_classes: int = 0, seed: int = 0,
                 predictive: dict | None = None):
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
        self.neurogenesis = bool(neurogenesis)   # False: no nacen rasgos nuevos

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

        # posición (en la lámina de células simples) de cada desplazamiento dentro del pooling
        a, b = self.pool
        qy, qx = np.meshgrid(np.arange(self.ph) * a, np.arange(self.pw) * b, indexing="ij")
        my, mx = np.divmod(np.arange(a * b), b)
        self._pool_pos = ((qy[:, :, None] + my) * self.ow + (qx[:, :, None] + mx)).reshape(-1, a * b)

        # error local: lector de clase propio del área
        self.local = None
        if local and n_classes:
            spec = dict(lr=0.02, lr_readout=0.5, temperature=0.25, feedback="sym", residual=False)
            spec.update(local)
            self.local = spec
            n_out = self.ph * self.pw * self.F
            self.clf = Readout(int(n_classes), capacity=n_out, consolidation=0.0,
                               lr=float(spec["lr_readout"]), temperature=float(spec["temperature"]))
            self.clf.n = n_out
            if spec["feedback"] == "random":
                rng = np.random.default_rng(seed)
                self.clf.W[:] = rng.standard_normal(self.clf.W.shape).astype(np.float32)
            elif spec["feedback"] != "sym":
                raise ValueError(f"feedback desconocido: {spec['feedback']}")
        self.local_hits = 0; self.local_seen = 0

        # predicción descendente: pesos generativos del área sobre su entrada
        self.pc = None
        self._x = None; self._pred = None; self._pinfo = None
        self.pe = float("nan"); self.gain = 1.0; self.pe_parts = (float("nan"), float("nan"))
        if predictive:
            spec = dict(lr=0.1, init=0.0, modulate=0.0, neuro=0.0, tau=500.0, warmup=100,
                        gain_min=0.25, gain_max=4.0)
            unknown = set(predictive) - set(spec)
            if unknown:
                raise ValueError(f"opciones de `predictive` desconocidas: {sorted(unknown)}")
            spec.update(predictive)
            self.pc = spec
            self.G = np.zeros((self.F, self.D), dtype=np.float32)
            self.sur_avg = 0.0; self.sur_n = 0               # media móvil de la sorpresa
            self.gain_avg = 1.0; self._rdebt = 0.0
            self.pred_macs = 0

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
        act = None; arg = None
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
            if self.local is None and self.pc is None:
                o = sheet.reshape(self.oh, self.ow, n)[:ph * a, :pw * b].reshape(ph, a, pw * b * n)
                code[:, :, :n] = o.max(axis=1).reshape(ph, pw, b, n).max(axis=2)   # células complejas
            else:                                                 # se recuerda qué célula simple ganó
                o = sheet.reshape(self.oh, self.ow, n)[:ph * a, :pw * b].reshape(ph, a, pw, b, n)
                o = o.transpose(0, 2, 1, 3, 4).reshape(ph * pw, a * b, n)
                arg = o.argmax(axis=1)                            # (ph*pw, n)
                code[:, :, :n] = np.take_along_axis(o, arg[:, None, :], axis=1).reshape(ph, pw, n)
        self._cache = (Pm, norms, vr, act, arg, code)
        if self.pc is not None:
            self._x = np.asarray(X, dtype=np.float32).reshape(H * W, C)
            self._pred = None
        return code

    # ------------------------------------------------------------------ plasticidad
    def potentials(self, code: np.ndarray | None = None) -> np.ndarray | None:
        """Potenciales del lector local sobre el último código (None si no hay error local)."""
        if self.local is None:
            return None
        if code is None:
            if self._cache is None:
                return None
            code = self._cache[5]
        c = code.ravel(); idx = np.flatnonzero(c)
        if len(idx) == 0:
            return np.zeros(self.clf.C, dtype=np.float32)
        h = c[idx]
        return self.clf.potentials(idx, h / np.float32(np.sqrt(h @ h)))

    def learn_readout(self, target: int, code: np.ndarray):
        """Sólo el lector local aprende (el diccionario no cambia): lo usa el repaso."""
        if self.local is None or self.local["feedback"] != "sym":
            return
        c = code.ravel(); idx = np.flatnonzero(c)
        if len(idx):
            h = c[idx]
            self.clf.learn(idx, h / np.float32(np.sqrt(h @ h)), int(target))

    def _learn_local(self, target: int, Pm, norms, act, arg, code, prior=None, scale: float = 1.0):
        """Error local de 3 factores: lector propio + rasgos ganadores (ver docstring del módulo).
        `prior`: potenciales de los lectores de las áreas anteriores (error residual)."""
        n = self.n
        c = code[:, :, :n].reshape(-1)                            # (ph*pw*n) sobre rasgos usados
        act_i = np.flatnonzero(c)
        if len(act_i) == 0:
            return
        q, f = np.divmod(act_i, n)
        cols = q * self.F + f                                     # índice en el código completo
        h = c[act_i]; nrm = np.float32(np.sqrt(h @ h)); hn = h / nrm
        R = self.clf
        V = R.potentials(cols, hn)
        if prior is not None:                                     # residual: lo que abajo aún falla
            V = V + prior
        p = R.probs(V)
        self.local_seen += 1; self.local_hits += int(np.argmax(V) == target)
        err = -p; err[target] += 1.0
        err = err.astype(np.float32)
        g = R.W[:, cols].T @ err                                  # error dendrítico de cada célula compleja
        g = (g - hn * float(hn @ g)) / nrm                        # derivada de la normalización del código
        if self.local["feedback"] == "sym":
            R.W[:, cols] += (R.lr * err)[:, None] * hn[None, :]
        # célula simple que ganó el pooling: posición y potencial
        pos = self._pool_pos[q, arg[q, f]]
        s = act[pos, f]
        gi = (g / np.float32(1.0 - self.floor)).astype(np.float32)
        # dW_f = sum_q g (x_q - s_q W_f) = sum_q g x_q - (sum_q g s_q) W_f,  x_q = parche normalizado
        u, gx = _segment_sum(f, Pm[pos], gi / np.maximum(norms[pos], 1e-6))
        gs = np.bincount(np.searchsorted(u, f), weights=gi * s, minlength=len(u)).astype(np.float32)
        lr = np.float32(self.local["lr"]) if scale == 1.0 else np.float32(self.local["lr"] * scale)
        Wn = self.W[u] + lr * (gx - gs[:, None] * self.W[u])
        self.W[u] = Wn / np.maximum(np.linalg.norm(Wn, axis=1, keepdims=True), 1e-8)

    def learn(self, target: int | None = None, prior: np.ndarray | None = None,
              surprise: float | None = None):
        """Actualiza el diccionario con la última lámina vista. Sin `target` (o sin error
        local configurado) el aprendizaje es puramente no supervisado. Con `prior` (sólo si
        `local["residual"]`), el error local es el de la suma de los lectores hasta esta área.
        Si el área es predictiva, también aprenden los pesos generativos y, si se configuró,
        la sorpresa (`surprise`, por defecto el `pe` propio) modula la plasticidad."""
        if not self.plastic or self._cache is None:
            return
        g = gn = 1.0
        if self.pc is not None:
            if self._pred is None:
                self.prediction_error()
            self._learn_generative()
            g, gn = self._surprise_gain(self.pe if surprise is None else surprise)
        Pm, norms, vr, act, arg, code = self._cache
        self._cache = None
        if target is not None and self.local is not None and act is not None:
            act_full = np.zeros((len(norms), act.shape[1]), dtype=np.float32)
            act_full[vr] = act                                    # potencial por posición (inválidas: 0)
            self._learn_local(int(target), Pm, norms, act_full, arg, code,
                              prior if self.local.get("residual") else None, scale=g)
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
            if g != 1.0:                                          # plasticidad dirigida por la sorpresa
                eta = np.minimum(eta * np.float32(g), np.float32(1.0))
            Wn = (1.0 - eta)[:, None] * self.W[u] + eta[:, None] * (M / cnt[:, None])
            self.W[u] = Wn / np.maximum(np.linalg.norm(Wn, axis=1, keepdims=True), 1e-8)
            self.wins[u] += 1
        nov = vr[bc < self.theta]
        if len(nov):                                              # neurogénesis por novedad
            limit = self.recruit_per_step
            if self.pc is not None and self.pc["neuro"] > 0.0 and self._pred is not None:
                rps = float(self.recruit_per_step)                # cupo modulado por la sorpresa
                self._rdebt = min(self._rdebt + rps * gn, rps * float(self.pc["gain_max"]))
                limit = int(self._rdebt); self._rdebt -= limit
                Rn = self._pred[2][self._gather[nov]].reshape(len(nov), self.D)
                energy = np.einsum("pd,pd->p", Rn, Rn)
                nov = nov[np.argsort(-energy)][:16]               # primero lo que la predicción no explica
            else:
                nov = nov[np.argsort(-norms[nov])][:16]           # primero los parches con más energía
            fresh = []
            for c in nov:
                if len(fresh) >= limit:
                    break
                p = Pm[c] / norms[c]
                if fresh and float((self.W[fresh] @ p).max()) >= self.theta:
                    continue                                      # ya cubierto por uno recién creado
                f = self._new_slot()
                if f < 0:
                    break
                self.W[f] = p; self.alive[f] = True
                if self.pc is not None:                           # predice el parche que lo creó
                    self.G[f] = np.float32(self.pc["init"]) * Pm[c]
                self.wins[f] = 0; self.born[f] = self.t
                self.created += 1
                fresh.append(f)
        if self.maintain_every and self.t % self.maintain_every == 0:
            self.maintain()

    # ------------------------------------------------------------------ predicción descendente
    def predict(self) -> np.ndarray | None:
        """Predicción de la lámina de entrada (H, W, C) a partir del último código: cada
        célula compleja activa proyecta c * G_f sobre el parche de la célula simple que ganó
        su pooling. None si el área no es predictiva o no hay código."""
        if self.pc is None or self._cache is None:
            return None
        H, W, C = self.in_shape
        Q = np.zeros((H * W, C), dtype=np.float32)
        arg, code = self._cache[4], self._cache[5]
        n = self.n
        self._pinfo = None; self.pred_macs = 0
        if n == 0 or arg is None:
            return Q.reshape(H, W, C)
        c = code[:, :, :n].reshape(-1)
        act = np.flatnonzero(c)
        if len(act) == 0:
            return Q.reshape(H, W, C)
        q, f = np.divmod(act, n)
        pos = self._pool_pos[q, arg[q, f]]                        # célula simple ganadora
        cv = c[act]
        vals = (cv[:, None] * self.G[f]).reshape(-1, C)           # (activas * rf_h * rf_w, C)
        u, s = _segment_sum(self._gather[pos].ravel(), vals)
        Q[u] = s
        self._pinfo = (f, pos, cv)
        self.pred_macs = len(act) * self.D
        return Q.reshape(H, W, C)

    def prediction_error(self, target: np.ndarray | None = None) -> float:
        """Residuo de la entrada contra la predicción del área. `target`: la lámina que se
        quiere predecir (por defecto, la última que codificó el área). Devuelve `pe`, la
        fracción de la energía de la entrada que la predicción no explica."""
        if self.pc is None or self._cache is None:
            return float("nan")
        C = self.in_shape[2]
        Q = self.predict().reshape(-1, C)
        T = self._x if target is None else np.asarray(target, dtype=np.float32).reshape(-1, C)
        R = T - Q
        e = float(np.einsum("ij,ij->", T, T))
        R2 = R * R
        self.pe = float(R2.sum()) / e if e > 1e-12 else 0.0
        miss = float(R2[T > 0].sum()) / e if e > 1e-12 else 0.0
        self.pe_parts = (miss, self.pe - miss)                    # en unidades activas / silenciosas
        self._pred = (T, Q, R)
        return self.pe

    def _learn_generative(self):
        """Regla delta normalizada sobre el residuo: cada rasgo activo mueve su predicción
        hacia lo que faltó (o sobró) en el parche donde se activó,
            dG_f = lr / n_f * sum_q c_q R_q / cov,
        con cov(u) = sum de c^2 de las células que predicen la unidad u (las predicciones se
        solapan) y n_f las veces que el rasgo f está activo (el diccionario se comparte entre
        posiciones). Así un paso corrige la fracción `lr` del residuo sin importar cuántas
        predicciones se sumen en cada unidad (estable para lr < 1)."""
        lr = float(self.pc["lr"])
        if lr <= 0.0 or self._pinfo is None or self._pred is None:
            return
        f, pos, cv = self._pinfo
        R = self._pred[2]
        H, W, C = self.in_shape
        rows = self._gather[pos]                                  # (activas, rf_h * rf_w)
        cov = np.bincount(rows.ravel(), weights=np.repeat(cv * cv, rows.shape[1]),
                          minlength=H * W).astype(np.float32)
        Rn = (R[rows] / np.maximum(cov[rows], np.float32(0.05))[:, :, None]).reshape(len(f), self.D)
        u, s = _segment_sum(f, Rn, cv)
        nf = np.bincount(np.searchsorted(u, f), minlength=len(u)).astype(np.float32)
        self.G[u] += (np.float32(lr) / nf)[:, None] * s

    def _surprise_gain(self, s) -> tuple[float, float]:
        """Ganancias de plasticidad y de neurogénesis: (s / media)^b, acotadas. Durante el
        calentamiento (sin media confiable) valen 1."""
        pc = self.pc
        if s is None or not np.isfinite(s):
            return 1.0, 1.0
        g = gn = 1.0
        if self.sur_n >= pc["warmup"] and self.sur_avg > 0.0:
            r = max(float(s), 0.0) / self.sur_avg
            lo, hi = float(pc["gain_min"]), float(pc["gain_max"])
            if pc["modulate"] > 0.0:
                g = float(np.clip(r ** float(pc["modulate"]), lo, hi))
            if pc["neuro"] > 0.0:
                gn = float(np.clip(r ** float(pc["neuro"]), lo, hi))
        self.sur_n += 1
        self.sur_avg += (float(s) - self.sur_avg) / min(self.sur_n, float(pc["tau"]))
        self.gain = g
        self.gain_avg += 0.01 * (g - self.gain_avg)
        return g, gn

    def _new_slot(self) -> int:
        if not self.neurogenesis:
            return -1
        if self.free:
            f = self.free.pop()
            if self.local is not None and self.local["feedback"] == "sym":
                self.clf.W[:, f::self.F] = 0.0                    # el lector local olvida ese rasgo
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
            if self.pc is not None:
                self.G[f] = 0.0
            self.free.append(int(f))
        self.pruned += len(dead)
        return len(dead)

    def forget_channel(self, f: int):
        """El canal de entrada f cambió de significado: se cortan sus sinapsis."""
        C = self.in_shape[2]
        self.W[:, f::C] = 0.0
        if self.pc is not None:                                   # y su predicción de ese canal
            self.G[:, f::C] = 0.0

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
        out = dict(stored_synapses=int(self.alive.sum()) * self.D, macs=P * n * self.D,
                   memory_bytes=self.W.nbytes + self.alive.nbytes + self.wins.nbytes + self.born.nbytes)
        if self.pc is not None:                                   # pesos generativos y última predicción
            out["stored_synapses"] *= 2
            out["memory_bytes"] += self.G.nbytes
            out["pred_macs"] = int(self.pred_macs)
        return out

    def state(self) -> dict:
        out = dict(W=self.W, alive=self.alive, wins=self.wins, born=self.born,
                   free=np.array(self.free, dtype=np.int64),
                   meta=np.array([self.n, self.t, self.created, self.pruned], dtype=np.int64))
        if self.local is not None:
            out["clf"] = self.clf.W
        if self.pc is not None:
            out["G"] = self.G
            out["pcstat"] = np.array([self.sur_avg, self.sur_n, self.gain_avg, self._rdebt], dtype=np.float64)
        return out

    def restore(self, z, prefix: str = ""):
        self.W[:] = z[prefix + "W"]; self.alive[:] = z[prefix + "alive"]
        self.wins[:] = z[prefix + "wins"]; self.born[:] = z[prefix + "born"]
        self.free = [int(v) for v in z[prefix + "free"]]
        self.n, self.t, self.created, self.pruned = (int(v) for v in z[prefix + "meta"])
        if self.local is not None and prefix + "clf" in z:
            self.clf.W[:] = z[prefix + "clf"]
        if self.pc is not None and prefix + "G" in z:
            self.G[:] = z[prefix + "G"]
            a, n, g, d = (float(v) for v in z[prefix + "pcstat"])
            self.sur_avg, self.sur_n, self.gain_avg, self._rdebt = a, int(n), g, d
