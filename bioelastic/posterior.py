"""Crédito posterior entre percepción y razonamiento, y evidencia trans-situacional.

Cuando la única señal es la respuesta final, un error no dice qué imagen se vio mal: una
suma equivocada se explica de muchas maneras. En vez de comprometer el símbolo percibido
(argmax) o buscar una sola explicación, se combina lo que cada parte sabe:

    q(s_1, ..., s_k) ∝ prod_i p_i(s_i | imagen_i)^gamma * L(objetivo | s_1, ..., s_k)

`p_i` es la distribución de la percepción sobre los símbolos de la ranura i y `L` la
verosimilitud de la respuesta observada bajo cada asignación, según el conocimiento del
razonador (lo que da por establecido) y según la estadística acumulada entre episodios.
Las marginales de q son el objetivo de la percepción y q de lo percibido dice cuánto
creerle al episodio para aprender el conocimiento.

Aprendizaje trans-situacional (Smith y Yu, 2008: los bebés aprenden pares palabra-referente
acumulando co-ocurrencias a través de escenas que una por una son ambiguas):
`CoOccurrence` acumula, con decaimiento exponencial, cuántas veces cada contenido (la
asignación de símbolos y el estado interno) fue seguido de cada respuesta, ponderado por el
posterior. La ambigüedad se resuelve a lo largo de muchos ensayos, no en uno.
"""
from __future__ import annotations

from itertools import product

import numpy as np


class CoOccurrence:
    """Contadores con decaimiento: clave -> conteos sobre `n_out` respuestas.

    `decay` es la vida media en actualizaciones (0 = sin decaimiento). El decaimiento es
    perezoso: cada clave guarda cuándo se tocó por última vez y se lo aplica al leerla, así
    que el costo es por clave tocada, no por paso."""

    def __init__(self, n_out: int, decay: float = 0.0, alpha: float = 1.0):
        self.n_out = int(n_out)
        self.half = float(decay)
        self.alpha = float(alpha)
        self.t = 0
        self.T: dict = {}                                   # clave -> [conteos, último t]

    def _factor(self, dt: int) -> float:
        return 1.0 if self.half <= 0 else float(0.5 ** (dt / self.half))

    def counts(self, key) -> np.ndarray:
        e = self.T.get(key)
        if e is None:
            return np.zeros(self.n_out)
        if e[1] != self.t:
            e[0] *= self._factor(self.t - e[1]); e[1] = self.t
        return e[0]

    def tick(self):
        """Una actualización más (el reloj del decaimiento)."""
        self.t += 1

    def add(self, key, out: int, w: float = 1.0):
        c = self.counts(key)
        if key not in self.T:
            self.T[key] = [c, self.t]
        self.T[key][0][int(out)] += float(w)

    def prob(self, key) -> np.ndarray:
        """P(respuesta | clave) con suavizado de Dirichlet `alpha` (uniforme si no se vio)."""
        c = self.counts(key)
        return (c + self.alpha) / (c.sum() + self.alpha * self.n_out)

    def likelihood(self, key, out: int) -> float:
        return float(self.prob(key)[int(out)])

    def evidence(self, key) -> float:
        return float(self.counts(key).sum())

    def confident(self, key, n_min: float, p_min: float):
        """(respuesta, True) si la clave acumuló al menos `n_min` de evidencia y una
        respuesta se lleva al menos `p_min` de ella; si no (-1, False)."""
        c = self.counts(key)
        tot = c.sum()
        if tot < n_min:
            return -1, False
        o = int(np.argmax(c))
        return (o, True) if c[o] >= p_min * tot else (-1, False)


def top_candidates(p: np.ndarray, k: int, keep: int | None = None) -> np.ndarray:
    """Los `k` símbolos más probables según `p` (con `keep` siempre incluido)."""
    order = np.argsort(-np.asarray(p), kind="stable")[:int(k)]
    if keep is not None and int(keep) not in order:
        order = np.concatenate([order[:-1], [int(keep)]])
    return order.astype(np.int64)


def joint_posterior(priors, lik: np.ndarray, gamma: float = 1.0):
    """Posterior conjunto sobre el producto de candidatos de cada ranura.

    `priors`: por ranura, las probabilidades de sus candidatos (en el mismo orden que los
    ejes de `lik`). `lik`: verosimilitud de la respuesta bajo cada asignación conjunta.
    `gamma` < 1 descuenta la evidencia perceptiva (la percepción es poco fiable).
    Devuelve (q conjunto normalizado, [marginal de cada ranura]). Si todo tiene peso cero
    devuelve el prior."""
    prior = np.ones(lik.shape)
    for i, p in enumerate(priors):
        sh = [1] * lik.ndim; sh[i] = len(p)
        pp = np.maximum(np.asarray(p, dtype=np.float64), 1e-12) ** float(gamma)
        prior = prior * (pp / pp.sum()).reshape(sh)
    q = prior * lik
    z = q.sum()
    q = prior / prior.sum() if z <= 0 else q / z
    margs = [q.sum(axis=tuple(j for j in range(q.ndim) if j != i)) for i in range(q.ndim)]
    return q, margs


def status_likelihood(status: np.ndarray, hit: float = 0.9, miss: float = 0.01,
                      unknown: float = 0.1) -> np.ndarray:
    """Verosimilitud de la respuesta según el conocimiento establecido del razonador: +1
    (con esa asignación respondería el objetivo), -1 (respondería otra cosa con
    conocimiento establecido) o 0 (no lo sabe: la respuesta es tan probable como cualquier
    otra de las 10)."""
    s = np.asarray(status)
    return np.where(s > 0, hit, np.where(s < 0, miss, unknown)).astype(np.float64)


def predictive_conflict(outs: np.ndarray, sure: np.ndarray, priors, answer: int,
                        gamma: float = 1.0) -> float:
    """Conflicto entre percepción y razonamiento antes del feedback: probabilidad (bajo la
    percepción) de que lo visto lleve, con conocimiento establecido, a una respuesta
    distinta de la dada. `outs`/`sure`: `Thinker.content_outputs` sobre el producto de
    candidatos; `priors`: probabilidad de cada candidato por ranura."""
    prior = np.ones(outs.shape)
    for i, p in enumerate(priors):
        sh = [1] * outs.ndim; sh[i] = len(p)
        pp = np.maximum(np.asarray(p, dtype=np.float64), 1e-12) ** float(gamma)
        prior = prior * (pp / pp.sum()).reshape(sh)
    return float((prior * (sure & (outs != int(answer)))).sum())


def state_key(reasoner) -> tuple:
    """El estado interno que acompaña a una pregunta: `reasoner.state_key()` si existe;
    si no, los registros que el Thinker leyó al responder."""
    f = getattr(reasoner, "state_key", None)
    if f is not None:
        return tuple(f())
    return tuple(int(x) for x in reasoner._q[2])


class PosteriorCredit:
    """Crédito posterior entre una percepción y un razonador (`Thinker` o cualquier objeto
    con `consistency`, `feedback(trust=...)` y, para consolidar, `teach_content`).

    Uso, en cada pregunta con k ranuras percibidas, después de `answer` y en lugar de
    `feedback`:

        res = pc.step(thinker, perceived, priors, target)
        for slot, (sym, w) in enumerate(res["teach"]):     # None: no enseñar esa ranura
            ...enseñarle `sym` a la percepción para la entrada de esa ranura...

      1. candidatos por ranura: los `k` símbolos más probables para la percepción;
      2. verosimilitud de cada asignación conjunta: lo que el razonador da por establecido
         (+1 / -1 / no sabe) y, con `half_life`, la co-ocurrencia contenido -> respuesta
         acumulada entre episodios (`CoOccurrence`, ponderada por el posterior);
      3. posterior q ∝ prod p_i^gamma * L;
      4. el razonador aprende del episodio salvo que el posterior crea, con más de
         `tau_thinker`, en otra asignación (`feedback(trust=1 - max q de las otras)`; se
         le fija `doubt = 1 - tau_thinker`);
      5. objetivo para la percepción por ranura: el máximo de la marginal si supera `tau`
         (`mode="map"`) o una muestra de la marginal (`mode="sample"`). Con `min_gain` > 0
         sólo si además la evidencia lo movió: q >= min_gain * prior (se aprende de la
         respuesta, no del propio prior: contra la auto-confirmación);
      6. con `consolidate=(n, p)`: si la asignación más probable acumuló al menos n de
         evidencia con una respuesta que se lleva una fracción p, y el conocimiento del
         razonador para ese contenido no la da por establecida, se escribe
         (`teach_content`): de la estadística al conocimiento.

    Incertidumbre esperada (ACh, `reliability`): la percepción declara una probabilidad
    para lo que eligió. Cuando el conocimiento establecido del razonador alcanza para
    juzgar lo percibido (+1 o -1 en la asignación percibida), eso es evidencia de su
    fiabilidad que no pasa por el prior de la percepción (usar el posterior la confirmaría
    a sí misma). Con r_par = frecuencia de asignaciones percibidas consistentes, la
    fiabilidad por ranura es r = r_par^(1/k); `gamma` se elige para que una distribución
    con la certeza declarada media, elevada a gamma, le dé r a lo elegido: si la percepción
    es menos fiable de lo que declara, su evidencia se descuenta (gamma < 1); si es más, se
    le cree más. dict(rate, lo, hi)."""

    def __init__(self, n_out: int = 21, k: int = 10, gamma: float = 1.0, half_life=None,
                 alpha: float = 1.0, use_thinker: bool = True, mode: str = "map", tau: float = 0.5,
                 tau_thinker: float = 0.5, consolidate=None, hit: float = 0.9, miss: float = 0.01,
                 unknown: float = 0.1, offsets=(0, 10), reliability: dict | None = None,
                 min_gain: float = 0.0, seed: int = 0):
        self.k = int(k); self.gamma = float(gamma)
        self.table = None if half_life is None else CoOccurrence(n_out, decay=half_life, alpha=alpha)
        self.use_thinker = bool(use_thinker)
        if mode not in ("map", "sample"):
            raise ValueError(f"modo desconocido: {mode}")
        self.mode = mode; self.tau = float(tau); self.tau_thinker = float(tau_thinker)
        self.consolidate = consolidate
        self.hit, self.miss, self.unknown = float(hit), float(miss), float(unknown)
        self.offsets = tuple(int(o) for o in offsets)
        self.min_gain = float(min_gain)
        self.reliability = None if not reliability else dict(dict(rate=0.01, lo=0.2, hi=2.0), **reliability)
        self.rel_obs = None; self.rel_decl = None         # fiabilidad observada (por par) / declarada (EMA)
        self.rel_events = 0
        self.rng = np.random.default_rng(seed)
        self.n = dict(steps=0, distrust=0, taught=0, self_confirm=0, consolidated=0)
        self.last: dict | None = None

    def likelihood(self, reasoner, target: int, cands) -> np.ndarray:
        L = np.ones(tuple(len(c) for c in cands))
        self._status = None
        if self.use_thinker:
            st = reasoner.consistency(target, [np.asarray(c) + o for c, o in zip(cands, self.offsets)])[0]
            self._status = st
            L = L * status_likelihood(st, self.hit, self.miss, self.unknown)
        if self.table is not None:
            rk = state_key(reasoner)
            for pos in np.ndindex(*L.shape):
                L[pos] *= self.table.likelihood(tuple(int(c[j]) for c, j in zip(cands, pos)) + rk, target)
        return L

    def _update_reliability(self, consistent: bool, declared: float, k: int):
        r = self.reliability
        if self.rel_obs is None:
            self.rel_obs, self.rel_decl = float(consistent), declared
        self.rel_obs += r["rate"] * (float(consistent) - self.rel_obs)
        self.rel_decl += r["rate"] * (declared - self.rel_decl)
        self.rel_events += 1
        if self.rel_events >= r.get("warmup", 50):
            actual = max(self.rel_obs, 1e-3) ** (1.0 / k)
            self.gamma = float(np.clip(gamma_for(self.rel_decl, actual), r["lo"], r["hi"]))

    def step(self, reasoner, perceived, priors, target: int) -> dict:
        perceived = [int(s) for s in perceived]
        cands = [top_candidates(p, self.k, keep=s) for p, s in zip(priors, perceived)]
        L = self.likelihood(reasoner, int(target), cands)
        q, margs = joint_posterior([np.asarray(p)[c] for p, c in zip(priors, cands)], L, self.gamma)
        at = tuple(int(np.flatnonzero(c == s)[0]) for c, s in zip(cands, perceived))
        other = q.copy(); other[at] = 0.0
        trust = 1.0 - float(other.max()) if other.size > 1 else 1.0
        rk = state_key(reasoner)
        if self.reliability is not None and self._status is not None and self._status[at] != 0:
            self._update_reliability(bool(self._status[at] > 0),
                                     float(np.mean([np.asarray(p)[s] for p, s in zip(priors, perceived)])),
                                     len(perceived))
        self.n["steps"] += 1
        self.n["distrust"] += int(trust < 1.0 - self.tau_thinker)
        if hasattr(reasoner, "doubt"):                    # no aprende si cree más en otra asignación
            reasoner.doubt = 1.0 - self.tau_thinker
        reasoner.feedback(target=int(target), trust=trust)
        if self.table is not None:
            self.table.tick()
            for pos in zip(*np.nonzero(q > 1e-3)):
                self.table.add(tuple(int(c[j]) for c, j in zip(cands, pos)) + rk, int(target), float(q[pos]))
            if self.consolidate is not None and hasattr(reasoner, "teach_content"):
                best = np.unravel_index(int(np.argmax(q)), q.shape)
                syms = tuple(int(c[j]) for c, j in zip(cands, best))
                o, ok = self.table.confident(syms + rk, *self.consolidate)
                if ok:
                    toks = tuple(s + off for s, off in zip(syms, self.offsets))
                    o2, sure = reasoner._know(reasoner._content(toks, reg=np.array(rk)))
                    if o2 != o or not sure:
                        reasoner.teach_content(toks, o, reg=np.array(rk))
                        self.n["consolidated"] += 1
        teach = []
        for c, qm, s, p in zip(cands, margs, perceived, priors):
            pr = np.maximum(np.asarray(p, dtype=np.float64)[c], 1e-12) ** self.gamma
            pr = pr / pr.sum()
            if self.mode == "map":
                j = int(np.argmax(qm))
                if qm[j] < self.tau or (self.min_gain > 0 and qm[j] < self.min_gain * pr[j]):
                    teach.append(None); continue
            else:
                j = int(self.rng.choice(len(qm), p=qm / qm.sum()))
            sym = int(c[j])
            teach.append((sym, float(qm[j])))
            self.n["taught"] += 1; self.n["self_confirm"] += int(sym == s)
        self.last = dict(q=q, marginals=margs, candidates=cands, trust=trust, teach=teach,
                         q_perceived=float(q[at]))
        return self.last


def _tempered_top(p: float, gamma: float, n: int = 10) -> float:
    """Probabilidad de lo elegido si la distribución (p en lo elegido, el resto parejo
    entre los otros n-1) se eleva a gamma y se renormaliza."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    a = p ** gamma; b = (n - 1) * ((1 - p) / (n - 1)) ** gamma
    return a / (a + b)


def gamma_for(declared: float, actual: float, lo: float = 0.05, hi: float = 4.0) -> float:
    """El exponente que lleva la certeza declarada a la fiabilidad real (bisección: la
    probabilidad templada de lo elegido crece con gamma cuando declared > 1/n)."""
    if declared <= 0.1 + 1e-6:
        return 1.0
    a, b = lo, hi
    for _ in range(40):
        m = 0.5 * (a + b)
        if _tempered_top(declared, m) < actual:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


__all__ = ["CoOccurrence", "PosteriorCredit", "top_candidates", "joint_posterior",
           "status_likelihood", "predictive_conflict", "state_key", "gamma_for", "entropy", "product"]
