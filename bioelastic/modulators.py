"""Meta-parámetros regulados por neuromoduladores (Doya, 2002), cada uno con su señal local.

La ronda anterior aprendía los meta-parámetros con una sola perturbación de alta dimensión
modulada por la recompensa: con 7 a la vez no escalaba (0.50 ± 0.36). Acá cada canal tiene
su propia señal, medida localmente, y una función fija (con 1 o 2 constantes elegidas en
validación) de esa señal al meta-parámetro:

  DA    error de predicción de recompensa: el δ de TD del control (ya estaba). Es la señal
        de enseñanza; los demás canales regulan cuánto y cómo se aprende de ella.
  ACh   incertidumbre *esperada* (Yu y Dayan, 2005): lo que ya se sabe que es incierto.
          - Memoria episódica (`BayesMemory`): el ruido de los desenlaces (con qué
            frecuencia el objetivo difiere de la causa vigente sin que la causa cambie).
          - Plasticidad del control: el ruido conocido del desenlace en el estado de la
            decisión, p(1 - p) con p la tasa de acierto reciente de ese (estado, acción),
            pesado por cuánta experiencia hay. La ganancia es 1 - 4·p(1 - p)·n/(n + n0): un
            desenlace que se sabe al azar (el ensayo 1 de Harlow) no mueve los pesos; un error
            sistemático (p cerca de 0: la estrategia está mal) sí, con toda la tasa. (Una
            ganancia 1/(n+1) que baja con la experiencia sin mirar el ruido congelaba el
            control antes de aprender estrategias largas: el cierre del anillo no se aprendía.)
          - Consolidación (`capture`) y certeza exigida (`sure`): con desenlaces ruidosos se
            pide más repetición y más certeza antes de dar algo por sabido.
  NE    incertidumbre *inesperada* / volatilidad.
          - Memoria: la probabilidad de cambio de la causa por ensayo.
          - Control: lo poco fiable que es la regla que rige (sus predicciones confiadas que
            fallan). Sube la exploración (`epsilon`); con la regla fiable, la exploración baja
            y deja de perturbar lo aprendido (la oscilación de Harlow congelado).
  5-HT  horizonte de recompensa (Doya: escala temporal de la predicción): la tasa media de
        recompensa a largo plazo. Con recompensa sostenida el horizonte es largo y se olvida
        poco (`forget` de la memoria, `decay` del control); con poca, se acorta.

La estimación de ruido y volatilidad de la memoria es un aprendiz bayesiano sobre una
grilla de hipótesis (Behrens et al., 2007): cada hipótesis (volatilidad H, ruido q) es un
filtro de un modelo oculto (la causa vigente de cada clave salta con probabilidad H; el
desenlace la refleja con probabilidad 1 - q) y su peso es la verosimilitud predictiva, con
olvido para seguir cambios de régimen. Es una población de unidades sintonizadas a
distintas escalas temporales; los niveles tónicos de NE y ACh son las medias posteriores de
H y q. La fuerza de cada ligadura de la memoria es la creencia promediada sobre las
hipótesis: con volatilidad alta y ruido bajo gana lo último (tasa 1, como `FastMemory`); con
ruido alto y volatilidad baja se integra (cuenta, tasa ~1/n).

Las señales son estado, no plasticidad: se siguen midiendo con los pesos congelados.
"""
from __future__ import annotations

import numpy as np

from .thinker import FastMemory

# Volatilidades hasta 0.2 por ensayo: con más (la causa cambia casi un ensayo de por medio) una
# memoria no sirve, y esa hipótesis explica por verosimilitud casi tan bien como «ruido 0.3 sin
# cambios» una secuencia ruidosa y estable (P(igual al anterior) 0.60 contra 0.58): con ella en
# la grilla el régimen ruidoso tardaba ~8 problemas en reconocerse.
H_GRID = np.array([0.0, 0.05, 0.1, 0.2])
Q_GRID = np.array([0.0, 0.05, 0.12, 0.2, 0.3, 0.4])


class BayesMemory(FastMemory):
    """Memoria episódica cuya tasa de ligadura la fijan la volatilidad (NE) y el ruido (ACh)
    estimados (ver el docstring del módulo). Los eventos tienen la misma forma que en
    `FastMemory` (clave, símbolos, fuerza): `_recall_events` y la búsqueda no cambian.

    `forget`: decaimiento por paso de todas las fuerzas (lo fija 5-HT). `prior_h`/`prior_q`:
    prior sobre la grilla (por defecto favorece poco ruido: sin evidencia, lo último manda,
    como en `FastMemory`). `phi`: olvido del log-peso de las hipótesis."""

    def __init__(self, n: int, phi: float = 0.02, floor: float = 0.05, prior_h: float = 0.0,
                 prior_q: float = 1.0):
        super().__init__(n)
        self.phi = float(phi)
        self.floor = float(floor)
        self.forget = 0.0
        hh, qq = np.meshgrid(H_GRID, Q_GRID, indexing="ij")
        # prior: más peso a ruido bajo (y, con prior_h > 0, a volatilidad alta)
        self.logp0 = prior_h * hh - prior_q * qq
        self.logw = self.logp0.copy()
        self._w = None
        self._H = hh[..., None]; self._Q = qq[..., None]
        self.keys: dict = {}                       # clave -> [símbolos vistos, creencias (H, Q, M)]
        self.rate = 1.0                            # tasa efectiva media del episodio (claves ya ligadas)
        self.rates: list = []
        self.binds = 0

    def reset(self):
        super().reset()
        self.keys = {}
        self.rates = []
        self.rate = 1.0

    def posterior(self) -> np.ndarray:
        if self._w is None:                      # (se recalcula sólo cuando cambió la evidencia)
            w = np.exp(self.logw - self.logw.max())
            self._w = w / w.sum()
        return self._w

    def volatility(self) -> float:                 # NE tónica (memoria)
        return float((self.posterior() * self._H[..., 0]).sum())

    def noise(self) -> float:                      # ACh tónica (memoria)
        return float((self.posterior() * self._Q[..., 0]).sum())

    def _observe_key(self, key, y) -> np.ndarray:
        """Actualiza los filtros de la clave con el desenlace y y devuelve la creencia
        promediada sobre las hipótesis (por símbolo visto). La causa vigente es uno de los
        símbolos vistos o, mientras sólo se vio uno, «otro» (uno todavía no visto): el segundo
        símbolo hereda la creencia de «otro», así un primer desenlace con ruido no se toma como
        certeza. El ruido y los saltos se reparten entre los símbolos vistos."""
        st = self.keys.get(key)
        if st is None:                                          # primera vez: [y, otro] con 1 - q, q
            B = np.empty(self._H.shape[:2] + (2,))
            B[..., 0] = 1.0 - self._Q[..., 0]; B[..., 1] = self._Q[..., 0]
            self.keys[key] = [[y], B]
            return np.array([1.0 - self.noise()])
        else:
            targets, B = st
            if y not in targets:
                targets.append(y)                               # «otro» pasa a ser y
                if len(targets) > 2:                            # (desde el tercero, sin «otro»: entra en 0)
                    B = np.concatenate([B, np.zeros(B.shape[:2] + (1,))], axis=2)
        K = B.shape[2]                                          # vistos (+ «otro» si sólo se vio uno)
        j = targets.index(y)
        # paso de predicción: la causa salta con probabilidad H a otra (al azar entre las demás)
        Bp = (1.0 - self._H) * B + self._H * (1.0 - B) / (K - 1)
        # predictiva del desenlace y bajo cada hipótesis (el ruido reparte q entre las demás)
        p = Bp[..., j] * (1.0 - self._Q[..., 0]) + (1.0 - Bp[..., j]) * self._Q[..., 0] / (K - 1)
        self.logw = (1.0 - self.phi) * self.logw + self.phi * self.logp0 + np.log(np.clip(p, 1e-3, 1.0))
        self._w = None
        lik = np.broadcast_to(self._Q / (K - 1), Bp.shape).copy()
        lik[..., j] = 1.0 - self._Q[..., 0]
        Bn = Bp * lik
        Bn /= np.clip(Bn.sum(axis=2, keepdims=True), 1e-12, None)
        st[1] = Bn
        w = self.posterior()
        return (w[..., None] * Bn).sum(axis=(0, 1))[:len(targets)]

    def bind(self, tokens: tuple, certainty: float = 1.0):
        self.t += 1
        tokens = tuple(tokens)
        ev, cert = self.events, self.cert
        if self.forget > 0.0 and ev:                # 5-HT: olvido de todas las ligaduras
            k = 1.0 - self.forget
            keep = [(e[0], e[1], e[2] * k, c) for e, c in zip(ev, cert) if e[2] * k >= self.floor]
            ev = [x[:3] for x in keep]; cert = [x[3] for x in keep]
        if tokens and self.prev:
            key = frozenset(self.prev)
            c = min(self.prev_cert, float(certainty))
            old = {toks: (s, c2) for (k2, toks, s), c2 in zip(ev, cert) if k2 == key}
            known = key in self.keys
            belief = self._observe_key(key, tokens)
            targets = self.keys[key][0]
            before = old.get(tokens, (0.0, c))[0]
            new_s = float(belief[targets.index(tokens)])
            r = (new_s - before) / max(1.0 - before, 1e-9) if before < 1.0 else 1.0
            if known:                               # tasa efectiva al volver a ligar una clave conocida
                self.rates.append(r)
                self.rate = float(np.mean(self.rates))
            self.binds += 1
            out_e = [(k2, toks, s) for (k2, toks, s) in ev if k2 != key]
            out_c = [c2 for (k2, _, _), c2 in zip(ev, cert) if k2 != key]
            for toks, b in zip(targets, belief):
                if b >= self.floor:
                    out_e.append((key, toks, float(b)))
                    out_c.append(min(old.get(toks, (0.0, c))[1], c) if toks in old else c)
            ev, cert = out_e, out_c
        self.events, self.cert = ev, cert
        self.prev = tokens
        self.prev_cert = float(certainty)


class Neuromodulators:
    """Los canales que fijan los 7 meta-parámetros del `Thinker` en cada ensayo.

    `eps`, `temp`: (mínimo, máximo) de exploración y de temperatura del softmax (el máximo,
    por defecto, el del Thinker); `n0`: experiencia con la que el ruido estimado de un estado
    ya cuenta la mitad; `keep`: retención de las cuentas por estado (olvido para seguir
    cambios); `forget_max`: olvido máximo de la memoria; `beta`: tasa de NE; `beta5`: tasa de
    5-HT (lenta); `confident`: confianza mínima de una predicción para medir la fiabilidad."""

    def __init__(self, net, eps=(0.0, None), temp=(0.1, None), n0: float = 5.0, keep: float = 0.98,
                 forget_max: float = 0.02, beta: float = 0.05, beta5: float = 0.01,
                 confident: float = 0.6, capture_noise: float = 0.15, sure_gain: float = 0.5,
                 control: bool = True, horizon: str = "ensayo", explained: float = 0.8,
                 unexplained_gain: float = 0.0, ne: str = "inesperada", ne_gain: float = 4.0,
                 ne_floor: float = 0.0):
        self.net = net
        self.horizon = str(horizon)        # 5-HT: escala temporal de la predicción de recompensa
        self.explained = float(explained)  # fiabilidad mínima de una explicación para aprender a pleno
        self.unexplained_gain = float(unexplained_gain)
        self.ne_mode = str(ne)             # "inesperada": fallos por encima de lo esperado; "fiabilidad": versión anterior
        self.ne_gain = float(ne_gain)
        self.ne_floor = float(ne_floor)    # zona muerta: lo que el promedio rápido fluctúa sin que nada cambie
        self._s0 = self.ne_floor + 1.0 / max(self.ne_gain, 1e-9)   # un contexto nuevo empieza explorando (NE = 1)
        self.surprise_c: dict = {}         # NE: por contexto, fallos por encima de los esperados (rápido - lento)
        self._ctx = 0
        self.eps_min = float(eps[0])
        self.eps_max = float(net.epsilon if eps[1] is None else eps[1])
        self.temp_min = float(temp[0])
        self.temp_max = float(net.temperature if temp[1] is None else temp[1])
        self.n0 = float(n0)
        self.keep = float(keep)
        self.forget_max = float(forget_max)
        self.decay0 = float(net.decay)
        self.sure0 = float(net.say_sure)
        self.capture0 = int(net.capture)
        self.beta = float(beta)
        self.beta5 = float(beta5)
        self.confident = float(confident)
        self.capture_noise = float(capture_noise)
        self.sure_gain = float(sure_gain)
        self.control = bool(control)       # False: sólo la memoria (ablación)
        if self.control and self.horizon == "ensayo":
            net._trial_td = True
        self.rho_c: dict = {}              # NE: por contexto (consigna), fiabilidad de las predicciones
        self.unexp_c: dict = {}            # confiadas de la regla y frecuencia de errores sin explicación
        self.serotonin = 0.5               # 5-HT: la recompensa se sostuvo (horizonte), promedio lento
        self.counts: dict = {}             # (conjunto, estado, acción) -> [aciertos, intentos] con olvido
        self.trace: list = []              # meta-parámetros por ensayo (para medir)
        self.keep_trace = False
        self.params = {}

    # ------------------------------------------------------------------ señales
    @property
    def rho(self) -> float:
        return self.rho_c.get(self._ctx, 0.5)

    def ne_control(self, ctx=None) -> float:
        """NE del control en el contexto (consigna) actual: lo poco fiable que es la regla que
        rige (0 fiable, 1 al azar) o, si es mayor, la frecuencia de errores que ni la búsqueda
        retrospectiva sabe explicar (el mundo hace algo que el modelo no contempla)."""
        ctx = self._ctx if ctx is None else ctx
        if self.ne_mode == "inesperada":
            base = self.ne_gain * (self.surprise_c.get(ctx, self._s0) - self.ne_floor)
        else:
            base = 2.0 * (1.0 - self.rho_c.get(ctx, 0.5))
        return float(np.clip(max(base, self.unexplained_gain * self.unexp_c.get(ctx, 0.0)), 0.0, 1.0))

    def memory_signals(self):
        m = self.net.fast
        if isinstance(m, BayesMemory):
            return m.volatility(), m.noise()
        return 0.0, 0.0

    # ------------------------------------------------------------------ un ensayo
    def step(self, target, reward, pred, explained: float = 0.0):
        """Tras el desenlace (antes de aprender): actualiza las señales y fija los
        meta-parámetros. `pred`: (respuesta, confianza) de la política más probable.
        `explained`: fiabilidad del camino que habría acertado (0 si no hubo error o no se
        encontró ninguno)."""
        net = self.net
        ok = (reward > 0) if reward is not None else (net._out == target)
        out, conf = pred
        self._ctx = ctx = net._q[1]
        # ACh del control: ruido conocido del desenlace en el estado de la decisión
        T = net.task_sets
        # el estado de la decisión sin las señales de confianza (que cambian con el modelo
        # interno y dispersaban las cuentas: el ruido del ensayo 1 de Harlow no se estimaba)
        key = (T.active if T is not None else 0, net._base(net._ls)[0].tobytes(), net._key[1])
        c = self.counts.get(key)
        if c is None:
            c = self.counts[key] = [0.0, 0.0]
        n = c[1]
        pr = (c[0] + 1.0) / (n + 2.0)
        ach_c = 4.0 * pr * (1.0 - pr) * n / (n + self.n0)
        if net.plastic:
            c[0] = self.keep * c[0] + float(ok); c[1] = self.keep * c[1] + 1.0
        w = 1.0 - ach_c                      # lo que se sabe que es azar no es evidencia
        # NE (incertidumbre inesperada): se falló donde se esperaba acertar. `pr` es lo esperado
        # (lento, por estado y acción: ACh); el promedio rápido de pr - acierto vale 0 mientras el
        # mundo no cambie, por ruidoso que sea, y sube cuando la estrategia deja de rendir lo
        # que rendía (utilidad reciente contra la de largo plazo: Aston-Jones y Cohen, 2005)
        sp = self.surprise_c.get(ctx, self._s0)
        self.surprise_c[ctx] = sp + self.beta * ((pr - float(ok)) - sp)
        # NE: ¿la regla que rige predijo con confianza y acertó?
        if out >= 0 and target is not None and conf >= self.confident:
            r = self.rho_c.get(ctx, 0.5)
            self.rho_c[ctx] = r + self.beta * w * (float(out == target) - r)
        if target is not None and net.plastic:       # error sin explicación (sólo se sabe aprendiendo)
            u = self.unexp_c.get(ctx, 0.0)
            unexplained = (not ok) and explained == 0.0 and net._cached is not None and net._cached[0] is None
            self.unexp_c[ctx] = u + self.beta * (float(unexplained) - u)
        # 5-HT: ¿la recompensa se sostuvo? (lento: horizonte)
        self.serotonin += self.beta5 * w * (float(ok) - self.serotonin)
        ne_c = self.ne_control()
        ne_m, ach_m = self.memory_signals()
        p = {}
        # ACh -> lr. Un error de estrategia con una explicación fiable (el camino que habría
        # acertado suele acertar) no es azar: se aprende con toda la tasa aunque el estado en
        # que se respondió sea ruidoso (si no, un atajo que acierta el 90 % no se corregía)
        gain = max(w, 1.0 if explained >= self.explained else 0.0)
        p["lr"] = net.lr * gain
        p["ach_control"] = ach_c
        # NE -> aleatoriedad de la selección: piso de exploración y temperatura de la conducta
        # (no la del aprendizaje: con márgenes chicos el congelado cambiaba de regla por nada)
        p["epsilon"] = self.eps_min + (self.eps_max - self.eps_min) * ne_c
        p["temperature"] = self.temp_min + (self.temp_max - self.temp_min) * ne_c
        # 5-HT -> forget (memoria) y decay (control): con horizonte largo no se olvida
        short = float(np.clip(2.0 * (1.0 - self.serotonin), 0.0, 1.5))
        p["forget"] = self.forget_max * min(short, 1.0)
        p["decay"] = self.decay0 * short
        # ACh (ruido de los desenlaces en la memoria) -> capture y sure
        p["capture"] = self.capture0 + int(ach_m > self.capture_noise)
        p["sure"] = float(np.clip(self.sure0 + self.sure_gain * max(ach_m - 0.1, 0.0), 0.5, 0.95))
        # NE y ACh -> rate (la fija la memoria bayesiana en cada ligadura)
        p["rate"] = float(getattr(net.fast, "rate", 1.0))
        p["volatility"] = ne_m; p["noise"] = ach_m; p["ne"] = ne_c; p["serotonin"] = self.serotonin
        self.params = p
        if self.keep_trace:
            self.trace.append(p)
        if isinstance(net.fast, BayesMemory):
            net.fast.forget = p["forget"] if self.control else 0.0
        if not self.control:
            return p
        net._ctl_gain *= gain
        net._beh_eps = p["epsilon"]
        net._beh_temp = p["temperature"]
        net.decay = p["decay"]
        net.capture = p["capture"]
        net.say_sure = p["sure"]; net.M.sure = p["sure"]
        return p
