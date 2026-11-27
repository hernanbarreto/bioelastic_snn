"""Conjuntos de tareas y causas latentes para las reglas del control del `Thinker`.

Un *conjunto de tareas* (task set, TS) es un control completo: los pesos del actor y del
crítico, la fiabilidad de cada (estado, acción) que usa la búsqueda y los errores pendientes
de repetir. El conocimiento de largo plazo (qué decir, qué juzgar, el modelo interno) es
compartido: lo que cambia entre familias de reglas es *cómo* pensar, no *qué* se sabe.

Como en el modelo C-TS (Collins y Frank, 2013) y la inferencia de causas latentes
(Gershman, Monfils, Norman y Niv, 2017), el que aprende no recibe qué regla rige: la infiere.

  * Creencia. Por consigna (contexto observable; sin consigna hay uno solo) se mantiene una
    distribución sobre los conjuntos existentes más la opción «uno nuevo».
  * Prior tipo restaurante chino con autocorrelación temporal. En cada ensayo la creencia
    se mezcla con el prior: (1 - h) · creencia + h · CRP. El CRP da a cada conjunto un peso
    proporcional a su popularidad (en cuántas consignas rigió) y a uno nuevo un peso `alpha`. Con `h` chico la causa latente tiende a seguir siendo la
    misma (autocorrelación); con una consigna nunca vista la creencia arranca en el CRP, así
    que se prueba primero lo más popular (reutilizar) y se crea uno nuevo si nada explica.
  * Verosimilitud por simulación interna. Tras cada respuesta, cada conjunto candidato
    simula sin efectos (con el modelo interno, como la búsqueda retrospectiva) qué habría
    respondido su política más probable en el mismo ensayo. Si lo habría acertado, el
    desenlace es evidencia a su favor en la medida de su confianza: la del paso más dudoso
    de su camino y su fiabilidad histórica (`rho`, la frecuencia con que acertaron sus
    predicciones confiadas). Un conjunto sin preferencia (o uno nuevo) no predice nada
    (0.5). Por eso un error por exploración no cuenta contra el conjunto que rige (su
    política más probable acertaba) y el ensayo 1 de un problema, que nadie puede predecir,
    no mueve la creencia.
  * Sólo se simulan los demás conjuntos cuando el que rige no explicó con confianza el
    desenlace (vigilancia de alternativas como en PROBE: Collins y Koechlin, 2012); si lo
    explicó, las alternativas quedan en 0.5. Costo acotado: una simulación por ensayo en
    régimen, K cuando algo cambia.
  * El control aprende sólo en el conjunto que rige después de ver el desenlace: si la
    evidencia cambió de conjunto, el ensayo enseña al nuevo y el viejo queda intacto. Al
    cambiar se corta la traza TD (el crédito de un conjunto no pasa a otro).
  * Crear sólo con plasticidad: con los pesos congelados la creencia se sigue actualizando
    (es actividad, no sinapsis) entre los conjuntos que ya existen.

`shrink` > 0: un conjunto nuevo arranca con `shrink` por el promedio (pesado por
popularidad) de los existentes, en vez de en cero: lo que comparten las reglas aprendidas
sobrevive al promedio y lo que las distingue se cancela (prior jerárquico sobre reglas).
"""
from __future__ import annotations

import numpy as np


class TaskSet:
    """Un control: actor, crítico, fiabilidad por (estado, acción) y repetición pendiente."""

    def __init__(self, Wc, wv, rel=None, pending=None):
        self.Wc = Wc
        self.wv = wv
        self.rel = {} if rel is None else rel
        self.pending = {} if pending is None else pending
        self.rho = 0.5            # fiabilidad de sus predicciones confiadas (promedio exponencial)
        self.segments = 1         # veces que pasó a regir (popularidad)
        self.trials = 0           # ensayos en que rigió
        self.cues: set = set()    # consignas en que rigió


class TaskSets:
    def __init__(self, alpha: float = 0.5, hazard: float = 0.03, kmax: int = 16,
                 shrink: float = 0.0, explained: float = 0.8, rho_rate: float = 0.05,
                 rho_max: float = 0.98, confident: float = 0.6, gated: bool = True,
                 novel: float = 0.6, lik: str = "beta", stick: bool = True):
        self.alpha = float(alpha)          # peso de «uno nuevo» en el CRP
        self.hazard = float(hazard)        # prior de la probabilidad por ensayo de que cambie la causa latente
        self.kmax = int(kmax)              # conjuntos como máximo (después se recicla el menos usado)
        self.shrink = float(shrink)        # inicialización de uno nuevo (ver docstring)
        self.explained = float(explained)  # verosimilitud del que rige por encima de la cual no se vigila al resto
        self.rho_rate = float(rho_rate)
        self.rho_max = float(rho_max)
        self.confident = float(confident)  # confianza mínima de una predicción para medir la fiabilidad
        self.gated = bool(gated)           # el control aprende en proporción a la creencia en el que rige
        self.novel = float(novel)          # creencia mínima en «uno nuevo» para una consigna nunca vista
        self.sets: list[TaskSet] = []
        self.belief: dict = {}             # consigna -> creencia sobre [conjuntos..., nuevo]
        self.active = -1
        self.switches = 0
        self.created = 0
        self.cue_trials: dict = {}         # por consigna: ensayos con plasticidad, cambios de conjunto
        self.cue_switches: dict = {}       # y último conjunto que rigió (para medir)
        self.cue_k: dict = {}
        # Verosimilitud (`lik`). "beta": la de C-TS (Collins y Frank, 2013): la probabilidad del
        # desenlace bajo el propio modelo del conjunto, la media posterior Beta-Bernoulli de que
        # acierte en el (estado, acción) con que respondería (su tabla de fiabilidad). Un conjunto
        # que viene fallando ahí (aprendiendo) no queda mal parado por otro error; uno que acertaba
        # siempre, sí. "confianza" (versión anterior): fiabilidad global del conjunto x confianza
        # de la política; un conjunto que aprende con errores confiados perdía contra uno que no
        # predecía nada (0.5) y una consigna terminaba en el conjunto de otra.
        self.lik = str(lik)
        # Pegajosidad por consigna (`stick`): el cambio de causa (hazard) lleva hacia lo que rigió
        # esa consigna en su historia (cuántos ensayos con cada conjunto), más la popularidad entre
        # consignas y «uno nuevo»; no hacia el prior global, que con dos consignas daba lo mismo
        # quedarse que pasar al conjunto de la otra.
        self.stick = bool(stick)
        self.cue_hist: dict = {}
        self.recycled = 0
        self.simulated = 0                 # simulaciones (costo)

    # ------------------------------------------------------------------ prior y creencia
    def popularity(self) -> np.ndarray:
        """Peso de cada conjunto en el CRP: en cuántas consignas rigió (como en C-TS, la
        popularidad es entre contextos). Contar las veces que pasó a regir infla la de un
        conjunto que alterna por error con otro."""
        return np.array([max(len(s.cues), 1) for s in self.sets], dtype=np.float64)

    def prior(self, cue=None) -> np.ndarray:
        pop = self.popularity()
        h = self.cue_hist.get(cue) if (self.stick and cue is not None) else None
        if h is not None and h.sum() > 0:
            hh = np.zeros(len(pop)); hh[:min(len(h), len(pop))] = h[:len(pop)]
            pop = pop + hh
        m = np.concatenate([pop, [self.alpha]])
        return m / m.sum()

    def _vec(self, cue) -> np.ndarray:
        """La creencia de la consigna, extendida a los conjuntos creados desde la última vez."""
        K = len(self.sets)
        b = self.belief.get(cue)
        if b is None:
            # consigna nunca vista: «uno nuevo» parte con al menos `novel` de la creencia. Se
            # reutiliza un conjunto sólo si sus predicciones confiadas aciertan (transferencia);
            # uno que no predice nada ahí no la explica y no se mezcla con la tarea nueva (sin
            # esto, en la batería C0-C6 cinco tareas distintas quedaban en un mismo conjunto)
            b = self.prior()
            if K and b[-1] < self.novel:
                b[:-1] *= (1.0 - self.novel) / b[:-1].sum()
                b[-1] = self.novel
        elif len(b) < K + 1:
            b = np.concatenate([b[:-1], np.zeros(K + 1 - len(b)), b[-1:]])
        self.belief[cue] = b
        return b

    def count(self, cue, k: int, plastic: bool):
        """Un ensayo de la consigna con el conjunto k rigiendo (sólo con plasticidad)."""
        if not plastic:
            return
        self.cue_trials[cue] = self.cue_trials.get(cue, 0) + 1
        if self.cue_k.get(cue, k) != k:
            self.cue_switches[cue] = self.cue_switches.get(cue, 0) + 1
        self.cue_k[cue] = k
        h = self.cue_hist.get(cue, np.zeros(0))
        if len(h) <= k:
            h = np.concatenate([h, np.zeros(k + 1 - len(h))])
        h[k] += 1.0
        self.cue_hist[cue] = h

    def predict(self, cue) -> np.ndarray:
        """Transición de un ensayo: la causa latente sigue con probabilidad 1 - hazard."""
        b = self._vec(cue)
        h = self.hazard
        b = (1.0 - h) * b + h * self.prior(cue)
        self.belief[cue] = b / b.sum()
        return self.belief[cue]

    def update(self, cue, lik: np.ndarray):
        b = self._vec(cue) * lik
        s = b.sum()
        self.belief[cue] = b / s if s > 0 else self.prior(cue)

    def choose(self, cue, allow_new: bool) -> int:
        """El conjunto más probable; `len(sets)` = crear uno nuevo."""
        b = self._vec(cue)
        if not allow_new or not self.sets:
            if not self.sets:
                return 0
            return int(np.argmax(b[:-1]))
        return int(np.argmax(b))

    # ------------------------------------------------------------------ verosimilitud
    def likelihood(self, k: int, out: int, conf: float, target, q=None) -> float:
        """P(desenlace | conjunto k): su predicción `out` contra el objetivo. Con "beta", `q` es
        la media posterior de que acierte en el (estado, acción) con que respondería; con
        "confianza", la fiabilidad global x `conf` (la probabilidad de su paso más dudoso). Sin
        predicción, 0.5 (como un (estado, acción) nunca visto bajo Beta(1, 1))."""
        if out < 0 or target is None:
            return 0.5
        if self.lik == "beta" and q is not None:
            return float(q) if out == target else 1.0 - float(q)
        g = float(np.clip(2.0 * conf - 1.0, 0.0, 1.0))
        p = 0.5 + (min(self.sets[k].rho, self.rho_max) - 0.5) * g
        return p if out == target else 1.0 - p

    def gain(self, cue) -> float:
        """Ganancia de la plasticidad del control: la creencia en que rige el conjunto activo
        (aprendizaje pesado por la responsabilidad, como en C-TS). Mientras la evidencia
        dice que la regla cambió, el conjunto viejo casi no aprende: no se sobrescribe antes
        de que la inferencia cambie de conjunto."""
        if not self.gated or self.active < 0:
            return 1.0
        return float(self._vec(cue)[self.active])

    def learn_rho(self, k: int, out: int, conf: float, target):
        if out < 0 or target is None or conf < self.confident:
            return
        ts = self.sets[k]
        ts.rho += self.rho_rate * (float(out == target) - ts.rho)

    # ------------------------------------------------------------------ estructura
    def init_weights(self, Wc, wv):
        """Pesos de un conjunto nuevo: cero, o `shrink` por el promedio de los existentes."""
        if self.shrink <= 0.0 or not self.sets:
            return np.zeros_like(Wc), np.zeros_like(wv)
        m = self.popularity()
        m /= m.sum()
        W = sum(mi * s.Wc for mi, s in zip(m, self.sets))
        v = sum(mi * s.wv for mi, s in zip(m, self.sets))
        return self.shrink * W, self.shrink * v

    def add(self, ts: TaskSet, cue) -> int:
        """Agrega un conjunto: hereda, en la consigna `cue`, la masa de «uno nuevo»."""
        if len(self.sets) >= self.kmax:                  # reciclar el menos popular que no rige
            pop = [(p, i) for i, p in enumerate(self.popularity()) if i != self.active]
            _, k = min(pop)
            self.sets[k] = ts
            for c, b in self.belief.items():
                b = self._vec(c)
                if c == cue:
                    b[k] += b[-1]; b[-1] = 0.0
                else:
                    b[k] = 0.0
                s = b.sum()
                self.belief[c] = b / s if s > 0 else self.prior()
            self.recycled += 1
            for hh in self.cue_hist.values():                # su historia no es la del reciclado
                if len(hh) > k:
                    hh[k] = 0.0
            ts.cues.add(cue)
            return k
        b = self._vec(cue)
        self.sets.append(ts)
        for c in list(self.belief):
            self._vec(c)                                  # extiende con un cero
        b = self.belief[cue]
        b[-2] = b[-1] if self.sets[:-1] else 1.0          # la masa de «nuevo» pasa al conjunto creado
        b[-1] = 0.0
        s = b.sum()
        self.belief[cue] = b / s if s > 0 else self.prior()
        self.created += 1
        ts.cues.add(cue)
        return len(self.sets) - 1

    def activate(self, k: int, cue, plastic: bool):
        if k != self.active:
            if plastic and self.active >= 0:
                self.sets[k].segments += 1
            self.switches += int(self.active >= 0)
            self.active = k
        if plastic:
            self.sets[k].cues.add(cue)

    def stats(self) -> dict:
        return dict(task_sets=len(self.sets), ts_switches=int(self.switches), ts_created=int(self.created),
                    ts_cue_switches=int(sum(self.cue_switches.values())),
                    ts_recycled=int(self.recycled), ts_simulated=int(self.simulated))
