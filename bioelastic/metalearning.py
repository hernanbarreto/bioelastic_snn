"""Aprender a aprender: meta-parámetros del Thinker aprendidos por recompensa.

En el `Thinker` los parámetros de cómo aprende rápido (cuánto pisa una ligadura nueva a
las anteriores, cuánto se olvida, cuánto explora, cuándo una corrección se consolida...)
son constantes elegidas a mano. Acá pasan a ser salidas de un controlador lento,
condicionado por el estado del propio aprendizaje y entrenado con la recompensa a través
de muchos episodios y tareas, sin retropropagación:

  * `GradedMemory`: la memoria episódica con dos perillas. `rate` (tasa de ligadura): al
    ligar clave -> símbolo, las ligaduras anteriores de la misma clave se multiplican por
    (1 - rate) y la nueva suma `rate` (un promedio exponencial por clave: con rate=1 sólo
    queda la última; con rate chico, gana la mayoría reciente). `forget` (decaimiento):
    todas las ligaduras pierden esa fracción en cada paso. La compuerta de escritura es la
    misma perilla: rate -> 0 es no escribir.
  * `MetaController`: política gaussiana sobre los meta-parámetros (en un espacio latente
    sin cotas, llevado a su rango con una sigmoide), lineal en señales del estado del
    aprendizaje, entrenada por perturbación modulada por recompensa (REINFORCE / «node
    perturbation», Williams 1992; Fiete y Seung 2006) con un crítico lineal como línea de
    base. Un episodio = una muestra: al empezar se sortean los meta-parámetros; al
    terminar, la recompensa media del episodio menos la predicha por el crítico, por el
    ruido que se sorteó, por la presináptica (las señales), mueve los pesos.
  * `MetaThinker`: un `Thinker` con controlador. Mide las señales (éxito reciente, sorpresa
    de la memoria episódica, |error de predicción de recompensa|, novedad de lo que se
    pregunta y persistencia de la sorpresa) y aplica los meta-parámetros en cada episodio.

Por qué perturbación y no un crítico sobre los meta-parámetros: el efecto de un
meta-parámetro sobre la recompensa pasa por operaciones discretas (búsqueda, recuerdo con
empates, reclutamiento de neuronas) que no tienen gradiente; la única señal disponible es
escalar y llega al final del episodio. La perturbación con línea de base es insesgada con
esa sola señal, es local (ruido x recompensa x presináptica, la forma de una sinapsis
modulada por dopamina) y su varianza crece con el número de meta-parámetros, que acá es
chico (1 a 7). Un crítico Q(estado, meta-parámetros) necesitaría aprender una función sobre
un espacio de acciones continuo y después maximizarla: más parámetros, y su sesgo no se
corrige solo. El crítico que sí se usa es V(estado), sólo para restar la línea de base.
"""
from __future__ import annotations

import numpy as np

from .thinker import FastMemory, Thinker

# nombre: (mínimo, máximo, entero). Los valores por defecto salen del Thinker.
PARAMS = {
    "rate": (0.05, 1.0, False),       # tasa de ligadura de la memoria episódica
    "forget": (0.0, 0.5, False),      # decaimiento de la memoria episódica por paso
    "sure": (0.5, 0.95, False),       # umbral de certeza (say_sure y modelo interno)
    "epsilon": (0.0, 0.3, False),     # exploración del control
    "decay": (0.0, 0.05, False),      # contracción L1 del control
    "capture": (1, 3, True),          # repeticiones para consolidar una corrección
    "lr": (0.05, 1.0, False),         # tasa del actor-crítico del control
}
SIGNALS = ("exito", "sorpresa", "td", "novedad", "persistencia")


class GradedMemory(FastMemory):
    """Memoria episódica con tasa de ligadura y decaimiento (ver el docstring del módulo).
    El tercer elemento de cada evento es su fuerza (en `FastMemory`, el momento en que se
    ligó): `_recall_events` toma, por símbolo, la mayor fuerza entre los eventos que ganan
    la conjunción, así que el recuerdo y la búsqueda interna no cambian."""

    def __init__(self, n: int, rate: float = 1.0, forget: float = 0.0, floor: float = 0.05):
        super().__init__(n)
        self.rate = float(rate)
        self.forget = float(forget)
        self.floor = float(floor)

    def bind(self, tokens: tuple, certainty: float = 1.0):
        self.t += 1
        tokens = tuple(tokens)
        ev, cert = self.events, self.cert
        if self.forget > 0.0 and ev:
            k = 1.0 - self.forget
            keep = [(e[0], e[1], e[2] * k, c) for e, c in zip(ev, cert) if e[2] * k >= self.floor]
            ev = [x[:3] for x in keep]; cert = [x[3] for x in keep]
        if tokens and self.prev:
            key = frozenset(self.prev)
            a = self.rate
            c = min(self.prev_cert, float(certainty))
            out_e, out_c, found = [], [], False
            for (k2, toks, s), c2 in zip(ev, cert):
                if k2 == key:
                    if toks == tokens:
                        s = s * (1.0 - a) + a; found = True; c2 = min(c2, c)
                    else:
                        s = s * (1.0 - a)
                    if s < self.floor:
                        continue
                out_e.append((k2, toks, s)); out_c.append(c2)
            if not found:
                out_e.append((key, tokens, a)); out_c.append(c)
            ev, cert = out_e, out_c
        self.events, self.cert = ev, cert
        self.prev = tokens
        self.prev_cert = float(certainty)


def _logit(p):
    # el valor inicial se aleja de las cotas: en una cota la sigmoide satura y la
    # perturbación no mueve el meta-parámetro (con rate=1 de partida no aprendía nada)
    p = float(np.clip(p, 0.2, 0.8))
    return float(np.log(p / (1.0 - p)))


class MetaController:
    """Política gaussiana lineal sobre meta-parámetros, entrenada por perturbación con un
    crítico lineal como línea de base. `signals`: qué señales del estado ve (vacío = sólo
    el sesgo: un único valor aprendido para todo)."""

    def __init__(self, params, init: dict, signals=("exito", "sorpresa", "td", "novedad"),
                 sigma: float = 1.0, lr: float = 1.0, lr_critic: float = 0.1, seed: int = 0):
        self.params = tuple(params)
        for p in self.params:
            if p not in PARAMS:
                raise ValueError(f"meta-parámetro desconocido: {p}")
        self.signals = tuple(signals)
        for s in self.signals:
            if s not in SIGNALS:
                raise ValueError(f"señal desconocida: {s}")
        self.F = 1 + len(self.signals)
        self.W = np.zeros((len(self.params), self.F))
        for i, p in enumerate(self.params):
            lo, hi, _ = PARAMS[p]
            self.W[i, 0] = _logit((init[p] - lo) / (hi - lo))
        self.v = np.zeros(self.F)
        self.sigma = float(sigma)
        self.lr = float(lr)
        self.lr_critic = float(lr_critic)
        self.rng = np.random.default_rng(seed)
        self._m = None; self._xi = None
        self.updates = 0
        self.norm_rate = 0.01
        self._mu = None; self._var = None

    def features(self, stats: dict, update: bool = False) -> np.ndarray:
        """Sesgo y cada señal estandarizada con su media y desvío recientes (promedios
        exponenciales lentos: normalización homeostática de la entrada). Sin esto, las
        señales variaban en rangos de ~0.2 y el peso necesario para separar dos regímenes
        era ~15: la perturbación no llegaba a aprenderlo (la tasa quedaba igual en los dos)."""
        x = np.array([float(stats[s]) for s in self.signals])
        if update and len(x):
            if self._mu is None:
                self._mu = x.copy(); self._var = np.full(len(x), 0.01)
            else:
                d = x - self._mu
                self._mu += self.norm_rate * d
                self._var += self.norm_rate * (d * d - self._var)
        if self._mu is None or not len(x):
            z = np.zeros(len(x))
        else:
            z = np.clip((x - self._mu) / np.sqrt(self._var + 1e-4), -3.0, 3.0)
        return np.concatenate([[1.0], z])

    @staticmethod
    def value(p, u):
        lo, hi, integer = PARAMS[p]
        x = lo + (hi - lo) / (1.0 + np.exp(-np.clip(u, -30.0, 30.0)))
        return int(np.clip(np.rint(x), lo, hi)) if integer else float(x)

    def begin(self, stats: dict, explore: bool, learn: bool | None = None) -> dict:
        """Sortea los meta-parámetros del episodio. `explore`: con ruido. `learn` (por
        defecto = explore): si la muestra se usará para aprender (y se actualiza la
        normalización de las señales)."""
        learn = explore if learn is None else learn
        m = self.features(stats, update=learn)
        mu = self.W @ m
        xi = self.rng.standard_normal(len(self.params)) if explore else np.zeros(len(self.params))
        u = mu + self.sigma * xi
        self._m, self._xi = (m, xi) if learn else (None, None)
        return {p: self.value(p, ui) for p, ui in zip(self.params, u)}

    def mean_params(self, stats: dict) -> dict:
        """Los meta-parámetros que daría sin ruido ante estas señales."""
        u = self.W @ self.features(stats)
        return {p: self.value(p, ui) for p, ui in zip(self.params, u)}

    def end(self, R: float):
        """Cierra la muestra: R es la recompensa media del episodio."""
        if self._m is None:
            return
        m, xi = self._m, self._xi
        delta = float(R - self.v @ m)
        n2 = float(m @ m)                       # paso normalizado (NLMS): estable con señales estandarizadas
        self.W += (self.lr / n2) * delta * np.outer(xi, m)
        self.v += (self.lr_critic / n2) * delta * m
        self._m = None; self._xi = None
        self.updates += 1


class MetaThinker(Thinker):
    """Un `Thinker` cuyos meta-parámetros salen de un `MetaController` (o quedan fijos en
    `fixed`). `meta`: lista de meta-parámetros aprendidos (vacía: ninguno). `signals`: qué
    señales ve el controlador. `graded`: usar `GradedMemory` (obligatorio si se aprende
    rate o forget). `fixed`: valores fijos de meta-parámetros (p. ej. {"rate": 0.3}).
    `beta`: tasa de los promedios exponenciales de las señales (por respuesta).

    Las señales son estado, no plasticidad: se siguen midiendo con `plastic=False`
    (entonces el controlador da su media, sin ruido, y no aprende; con
    `meta_noise_frozen=True` sigue sorteando con ruido, sin aprender)."""

    def __init__(self, n_symbols: int, *, meta=(), signals=("exito", "sorpresa", "td", "novedad"),
                 graded: bool = False, fixed: dict | None = None, beta: float = 0.05,
                 meta_sigma: float = 1.0, meta_lr: float = 1.0, **kw):
        # los meta-parámetros los fija este controlador: sin los neuromoduladores del Thinker
        # (que por defecto regulan los mismos) salvo que se pidan
        kw.setdefault("modulators", False)
        super().__init__(n_symbols, **kw)
        fixed = dict(fixed or {})
        graded = graded or bool({"rate", "forget"} & (set(meta) | set(fixed)))
        if graded and self.fast is not None:
            self.fast = GradedMemory(self.N, rate=fixed.get("rate", 1.0), forget=fixed.get("forget", 0.0))
        self._apply(fixed)
        self.beta = float(beta)
        self.sig = dict(exito=0.5, sorpresa=0.5, td=0.5, novedad=0.5, persistencia=0.5)
        self._seen = np.zeros(self.N, dtype=bool)
        self._ep_r: list = []
        self._last_surprise = None                  # objetivo del último ensayo sorpresivo
        self.meta = None
        if meta:
            init = dict(rate=self.fast.rate if isinstance(self.fast, GradedMemory) else 1.0,
                        forget=self.fast.forget if isinstance(self.fast, GradedMemory) else 0.0,
                        sure=self.say_sure, epsilon=self.epsilon, decay=self.decay,
                        capture=self.capture, lr=self.lr)
            init.update(fixed)
            self.meta = MetaController(meta, init, signals, sigma=meta_sigma, lr=meta_lr,
                                       seed=int(kw.get("seed", 0)) + 7)
        self.meta_noise_frozen = False
        self.trace: list = []                       # (episodio, meta-parámetros) por episodio
        self.episodes = 0

    def _apply(self, params: dict):
        for p, v in params.items():
            if p in ("rate", "forget"):
                if isinstance(self.fast, GradedMemory):
                    setattr(self.fast, p, float(v))
            elif p == "sure":
                self.say_sure = float(v); self.M.sure = float(v)
            elif p == "capture":
                self.capture = int(v)
            else:
                setattr(self, p, float(v))

    def reset(self):
        if getattr(self, "meta", None) is not None and hasattr(self, "_ep_r"):
            if self._ep_r and self.plastic:
                self.meta.end(float(np.mean(self._ep_r)))
            params = self.meta.begin(self.sig, explore=self.plastic or self.meta_noise_frozen,
                                     learn=self.plastic)
            self._apply(params)
            self.trace.append(params)
        if hasattr(self, "_ep_r"):
            self._ep_r = []
            self._last_surprise = None
            self.episodes += 1
        super().reset()

    def answer(self, tokens, cue: int = 0, explore: bool = False, probe=None, certainty: float = 1.0) -> int:
        toks = self._tok(tokens) + (self._tok(probe) if probe is not None else ())
        if toks:
            idx = np.array(toks, dtype=np.int64)
            nov = float(np.mean(~self._seen[idx]))
            self._seen[idx] = True
            self.sig["novedad"] += self.beta * (nov - self.sig["novedad"])
        return super().answer(tokens, cue=cue, explore=explore, probe=probe, certainty=certainty)

    def feedback(self, target: int | None = None, reward: float | None = None, alternatives=None):
        if target is not None:
            t = int(target)
            r = (1.0 if self._out == t else -1.0) if reward is None else float(reward)
            b = self.beta
            self.sig["exito"] += b * (float(r > 0) - self.sig["exito"])
            v = self._V if self._V is not None else 0.0
            self.sig["td"] += b * (min(abs(r - v), 2.0) / 2.0 - self.sig["td"])
            if self.fast is not None:
                pred = self._best(self.fast.recall(self._q[0][0]))
                if pred >= 0:
                    surprise = pred != t
                    self.sig["sorpresa"] += b * (float(surprise) - self.sig["sorpresa"])
                if self._last_surprise is not None:      # ¿el cambio sorpresivo se sostuvo?
                    self.sig["persistencia"] += 2 * b * (float(t == self._last_surprise) - self.sig["persistencia"])
                self._last_surprise = t if (pred >= 0 and pred != t) else None
            self._ep_r.append(float(r > 0))
        super().feedback(target=target, reward=reward, alternatives=alternatives)
