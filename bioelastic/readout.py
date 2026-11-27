"""Capa de decisión: una neurona por salida, plasticidad de tres factores.

Regla local:  dW[o, j] = lr * M_o * h_j
  h_j : actividad presináptica (fuerza del spike de la neurona oculta j)
  M_o : tercer factor (neuromodulador) de la salida o
        - supervisado : M_o = objetivo_o - p_o            (señal de error por salida)
        - recompensa  : M_o = (R - baseline) * (1[o = acción] - p_o)   (R-STDP / actor)
p es la competencia softmax entre salidas. El número de entradas crece y se
reduce junto con la capa oculta.

Aprendizaje continuo (opcional, apagado por defecto):
  active_window  error asimétrico. Una salida cuya clase no fue objetivo en los últimos
                 `active_window` aprendizajes está *inactiva*: su error negativo (-p_o)
                 no se aplica. Así lo que se enseña ahora no deprime a las clases
                 ausentes (análogo local de ER-ACE, Caccia et al. 2022, y del «labels
                 trick», Zeno et al. 2018). El repaso usa el error completo.
  ace            con `active_window`: la competencia que fija el tercer factor de lo que
                 llega se resuelve sólo entre las salidas activas (p renormalizada sobre
                 ellas), como la entropía cruzada enmascarada de ER-ACE. Sin esto, una
                 clase ausente que se parece a la entrada baja p(objetivo) y agranda el
                 error positivo del objetivo: la fila nueva crece sobre los rasgos que
                 comparte con la vieja y después le gana en sus propios ejemplos (sesgo
                 hacia lo reciente). El repaso sigue usando el error completo.
  row_norm       escalado sináptico de cada salida: tras aprender, el vector de pesos
                 que llega a cada salida se reescala a norma `row_norm`. Una clase no
                 gana por acumular peso sino por parecerse (clasificador por coseno, como
                 LUCIR, Hou et al. 2019; escalado homeostático, Turrigiano 2008).
  rule="mean"    regla hebbiana sólo sobre la salida objetivo: cada fila sigue la media
                 de las entradas de su clase (tasa 1/n con piso `mean_floor`). Sin término
                 negativo: lo que se enseña ahora no mueve las filas de otras clases
                 (clasificador por media de clase, Mensink et al. 2013; es la parte de
                 medias del LDA en flujo de Hayes & Kanan 2020).
  floor          piso de plasticidad: la consolidación nunca baja la tasa de una entrada
                 por debajo de `floor` (fracción de la tasa de una entrada nueva). Sin piso,
                 una neurona usada mil veces conserva 0,5 % de plasticidad y una red que ya
                 no puede crear neuronas deja de aprender clases nuevas.
  relax          metaplasticidad que se relaja: el contador de consolidación de cada
                 entrada decae con constante `relax` (en aprendizajes), así una sinapsis
                 que dejó de usarse recupera plasticidad (cf. Benna & Fusi 2016).
  recon_lock     reconsolidación gobernada por error de predicción (Nader et al. 2000;
                 Sevenster, Beckers y Kindt 2013, 2014). Una entrada modificada
                 `recon_lock` veces queda consolidada: plasticidad 0. Sólo se vuelve lábil
                 cuando, al recuperarla (estar activa en un aprendizaje o repaso), hay
                 error de predicción: la predicción es otra clase, o la sorpresa
                 -log p(objetivo) supera `recon_kappa` veces su media móvil (tasa
                 `recon_beta`). La ventana lábil dura `recon_window` aprendizajes de esta
                 lectura, con tasa x `recon_rate`, y toca sólo las filas del error (el
                 objetivo y la clase predicha); después la entrada vuelve a quedar
                 bloqueada (re-consolidación). Sin error no cambia nada. (La replicación
                 de 2022 de Sevenster et al. no encontró el efecto en humanos: es una
                 hipótesis de mecanismo, no un hecho establecido.)
"""
from __future__ import annotations

import numpy as np


class Readout:
    def __init__(self, n_outputs: int, *, lr: float = 0.5, temperature: float = 0.25,
                 w_init: float = 1.0, consolidation: float = 5.0, baseline_beta: float = 0.02,
                 active_window: int = 0, relax: float = 0.0, row_norm: float = 0.0,
                 rule: str = "delta", mean_floor: float = 1e-3, floor: float = 0.0,
                 recon_lock: float = 0.0, recon_window: int = 1, recon_kappa: float = 2.0,
                 recon_beta: float = 0.01, recon_rate: float = 1.0,
                 capacity: int = 256, ace: bool = False):
        self.C = int(n_outputs)
        self.lr = float(lr)
        self.temperature = float(temperature)
        self.w_init = float(w_init)
        self.consolidation = float(consolidation)    # nº de cambios a mitad de plasticidad (0 = off)
        self.bl_beta = float(baseline_beta)
        self.baseline = 0.0
        self.cap = int(capacity)
        self.n = 0
        self.W = np.zeros((self.C, self.cap), dtype=np.float32)
        self.count = np.zeros(self.cap, dtype=np.float32)   # veces que se modificó cada entrada
        self.active_window = int(active_window)
        self.ace = bool(ace)
        self.relax = float(relax)
        self.row_norm = float(row_norm)
        self.floor = float(floor)
        if rule not in ("delta", "mean"):
            raise ValueError(f"regla desconocida: {rule}")
        self.rule = rule
        self.mean_floor = float(mean_floor)
        self.n_class = np.zeros(self.C, dtype=np.float64)  # ejemplos por clase (regla "mean")
        self.t = 0                                          # aprendizajes con objetivo propio
        self.last_target = np.full(self.C, -(10 ** 12), dtype=np.int64)
        self._count_t = np.zeros(self.cap, dtype=np.int64)  # último ajuste del contador (relajación)
        self.recon_lock = float(recon_lock); self.recon_window = int(recon_window)
        self.recon_kappa = float(recon_kappa); self.recon_beta = float(recon_beta)
        self.recon_rate = float(recon_rate)
        self.labile_until = np.zeros(self.cap, dtype=np.int64)   # evento hasta el que la entrada es lábil
        self._ev = 0                                        # aprendizajes de esta lectura (incluye repasos)
        self._smean = -1.0                                  # media móvil de la sorpresa (-1: sin datos)
        self.opened = 0; self.relearned = 0; self.blocked = 0

    # --- sincronía con la capa elástica
    def add_input(self, cls: int | None = None):
        if self.n == self.cap:
            W = np.zeros((self.C, self.cap * 2), dtype=np.float32)
            W[:, :self.n] = self.W[:, :self.n]
            count = np.zeros(self.cap * 2, dtype=np.float32)
            count[:self.n] = self.count[:self.n]
            ct = np.zeros(self.cap * 2, dtype=np.int64)
            ct[:self.n] = self._count_t[:self.n]
            lu = np.zeros(self.cap * 2, dtype=np.int64)
            lu[:self.n] = self.labile_until[:self.n]
            self.W = W; self.count = count; self._count_t = ct; self.labile_until = lu; self.cap *= 2
        self.W[:, self.n] = 0.0
        self.count[self.n] = 0.0
        self._count_t[self.n] = self.t
        self.labile_until[self.n] = 0
        if cls is not None:
            self.W[cls, self.n] = self.w_init      # asociación de un solo disparo
        self.n += 1

    def remove_input(self, j: int, last: int):
        if j != last:
            self.W[:, j] = self.W[:, last]
            self.count[j] = self.count[last]
            self._count_t[j] = self._count_t[last]
            self.labile_until[j] = self.labile_until[last]
        self.n -= 1

    def merge_inputs(self, a: int, b: int, alpha: float):
        """La entrada b se funde en a (la capa fusionó dos neuronas): promedio ponderado."""
        self.W[:, a] = alpha * self.W[:, a] + (1.0 - alpha) * self.W[:, b]
        self.count[a] = max(self.count[a], self.count[b])
        self.labile_until[a] = max(self.labile_until[a], self.labile_until[b])

    # --- actividad de las clases (para el error asimétrico)
    def active(self) -> np.ndarray:
        """Máscara de salidas activas (objetivo reciente). Todas activas si está apagado."""
        if self.active_window <= 0:
            return np.ones(self.C, dtype=bool)
        return (self.t - self.last_target) <= self.active_window

    def mark(self, target: int):
        self.t += 1
        self.last_target[target] = self.t

    # --- inferencia
    def potentials(self, idx: np.ndarray, h: np.ndarray) -> np.ndarray:
        if len(idx) == 0:
            return np.zeros(self.C, dtype=np.float32)
        return self.W[:, idx] @ h

    def probs(self, V: np.ndarray) -> np.ndarray:
        z = (V - V.max()) / self.temperature
        e = np.exp(z)
        return e / e.sum()

    def preferred(self, idx: np.ndarray) -> np.ndarray:
        """Clase preferida de cada neurona oculta (-1 si aún no está comprometida)."""
        if len(idx) == 0:
            return np.empty(0, dtype=np.int64)
        cols = self.W[:, idx]
        pref = cols.argmax(axis=0)
        pref[cols.max(axis=0) <= 0.0] = -1
        return pref

    # --- aprendizaje
    def _plasticity(self, idx: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Consolidación: las sinapsis de una entrada cambian menos cuanto más se usaron.
        Evita que datos nuevos recableen lo que ya estaba aprendido."""
        if self.consolidation <= 0.0:
            return h
        if self.relax > 0.0:                                # el contador decae mientras no se usa
            dt = (self.t - self._count_t[idx]).astype(np.float32)
            self.count[idx] *= np.exp(-dt / np.float32(self.relax))
            self._count_t[idx] = self.t
        pre = h * np.maximum(self.floor, 1.0 / (1.0 + self.count[idx] / self.consolidation))
        self.count[idx] += 1.0
        return pre.astype(np.float32)

    def learn(self, idx: np.ndarray, h: np.ndarray, target: int, V: np.ndarray | None = None,
              p: np.ndarray | None = None, replay: bool = False, soft: np.ndarray | None = None,
              lr_scale: float = 1.0):
        """`p` externo: la competencia entre salidas la resolvió la red completa.
        `replay`: el ejemplo es un recuerdo (error completo, no cuenta como actividad).
        `soft`: objetivo como distribución (p. ej. un posterior sobre las clases): el tercer
        factor de cada salida es soft_o - p_o en vez de 1[o = objetivo] - p_o (`target` es
        entonces la clase más probable, para la actividad y la regla de medias).
        `lr_scale`: factor sobre la tasa (asimilación rápida de un esquema, ver Cortex)."""
        if not replay:
            self.mark(target)
        if len(idx) == 0 or lr_scale == 0.0:
            return
        if self.rule == "mean":
            self._learn_mean(idx, h, target)
            return
        if p is None:
            p = self.probs(self.potentials(idx, h) if V is None else V)
        err = -p
        if self.active_window > 0 and not replay:
            act = self.active()
            err = np.where(act, err, 0.0)
            if self.ace:                                    # competencia sólo entre las presentes
                tot = float(-err.sum())
                if tot > 0.0:
                    err = err / tot
        if soft is None:
            err[target] += 1.0
        else:
            err = err + np.asarray(soft, dtype=err.dtype)
        lr = self.lr if lr_scale == 1.0 else self.lr * float(lr_scale)
        if self.recon_lock > 0.0:
            self._learn_recon(idx, h, target, p, err, lr)
        else:
            self.W[:, idx] += (lr * err)[:, None].astype(np.float32) * self._plasticity(idx, h)[None, :]
        self._scale_rows(err != 0)

    def surprise_error(self, p: np.ndarray, target: int) -> bool:
        """¿Hay error de predicción? Predicción equivocada, o sorpresa -log p(objetivo) por
        encima de `recon_kappa` veces su media móvil (que se actualiza acá)."""
        s = -float(np.log(max(float(p[target]), 1e-12)))
        wrong = int(np.argmax(p)) != target
        surprised = self._smean > 0.0 and s > self.recon_kappa * self._smean
        self._smean = s if self._smean < 0.0 else self._smean + self.recon_beta * (s - self._smean)
        return wrong or surprised

    def _learn_recon(self, idx, h, target: int, p, err, lr: float):
        """Delta con reconsolidación: las entradas libres aprenden como siempre (y cuentan
        hacia el bloqueo); las consolidadas sólo si están lábiles, y sólo en las filas del
        error (objetivo y predicha)."""
        ev = self._ev; self._ev += 1
        opened = self.surprise_error(p, target)
        cons = self.count[idx] >= self.recon_lock
        free = ~cons
        if free.any():
            fi = idx[free]
            hf = h[free]
            if self.consolidation > 0.0:
                hf = hf * np.maximum(self.floor, 1.0 / (1.0 + self.count[fi] / self.consolidation))
            self.count[fi] += 1.0
            self.W[:, fi] += (lr * err)[:, None].astype(np.float32) * hf.astype(np.float32)[None, :]
        if not cons.any():
            return
        ci = idx[cons]
        if opened:                                          # la recuperación con error abre la ventana
            self.labile_until[ci] = ev + self.recon_window
            self.opened += len(ci)
        lab = self.labile_until[ci] > ev
        self.blocked += int((~lab).sum())
        if not lab.any():
            return
        li = ci[lab]
        pred = int(np.argmax(p))
        rows = np.array([target] if pred == target else [target, pred])
        dW = (lr * self.recon_rate * err[rows])[:, None] * h[cons][lab][None, :]
        self.W[np.ix_(rows, li)] += dW.astype(np.float32)
        self.relearned += len(li)

    def _learn_mean(self, idx, h, target: int):
        eta = np.float32(max(self.mean_floor, 1.0 / (self.n_class[target] + 1.0)))
        self.n_class[target] += 1.0
        self.W[target, :self.n] *= np.float32(1.0) - eta
        self.W[target, idx] += eta * h.astype(np.float32)
        rows = np.zeros(self.C, dtype=bool); rows[target] = True
        self._scale_rows(rows)

    def _scale_rows(self, rows: np.ndarray):
        """Escalado sináptico: las salidas que cambiaron vuelven a norma `row_norm`."""
        if self.row_norm <= 0.0 or not rows.any():
            return
        Wr = self.W[rows, :self.n]
        nrm = np.sqrt(np.einsum("ij,ij->i", Wr, Wr))
        ok = nrm > 1e-12
        if ok.any():
            sub = np.flatnonzero(rows)[ok]
            self.W[sub, :self.n] *= (self.row_norm / nrm[ok])[:, None].astype(np.float32)

    def learn_reward(self, idx: np.ndarray, h: np.ndarray, action: int, R: float,
                     V: np.ndarray | None = None, p: np.ndarray | None = None):
        if len(idx) == 0:
            return
        if p is None:
            p = self.probs(self.potentials(idx, h) if V is None else V)
        if R > 0:
            self.mark(action)                               # la acción premiada es la clase
        if self.rule == "mean":
            if R > 0:
                self._learn_mean(idx, h, action)
            return
        elig = -p
        elig[action] += 1.0
        adv = R - self.baseline
        self.W[:, idx] += (self.lr * adv * elig)[:, None].astype(np.float32) * h[None, :]
        self._scale_rows(elig != 0)
        self.baseline += self.bl_beta * (R - self.baseline)
