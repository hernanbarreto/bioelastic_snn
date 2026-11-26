"""Capa de decisión: una neurona por salida, plasticidad de tres factores.

Regla local:  dW[o, j] = lr * M_o * h_j
  h_j : actividad presináptica (fuerza del spike de la neurona oculta j)
  M_o : tercer factor (neuromodulador) de la salida o
        - supervisado : M_o = objetivo_o - p_o            (señal de error por salida)
        - recompensa  : M_o = (R - baseline) * (1[o = acción] - p_o)   (R-STDP / actor)
p es la competencia softmax entre salidas. El número de entradas crece y se
reduce junto con la capa oculta.
"""
from __future__ import annotations

import numpy as np


class Readout:
    def __init__(self, n_outputs: int, *, lr: float = 0.5, temperature: float = 0.25,
                 w_init: float = 1.0, consolidation: float = 5.0, baseline_beta: float = 0.02,
                 capacity: int = 256):
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

    # --- sincronía con la capa elástica
    def add_input(self, cls: int | None = None):
        if self.n == self.cap:
            W = np.zeros((self.C, self.cap * 2), dtype=np.float32)
            W[:, :self.n] = self.W[:, :self.n]
            count = np.zeros(self.cap * 2, dtype=np.float32)
            count[:self.n] = self.count[:self.n]
            self.W = W; self.count = count; self.cap *= 2
        self.W[:, self.n] = 0.0
        self.count[self.n] = 0.0
        if cls is not None:
            self.W[cls, self.n] = self.w_init      # asociación de un solo disparo
        self.n += 1

    def remove_input(self, j: int, last: int):
        if j != last:
            self.W[:, j] = self.W[:, last]
            self.count[j] = self.count[last]
        self.n -= 1

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
        pre = h / (1.0 + self.count[idx] / self.consolidation)
        self.count[idx] += 1.0
        return pre.astype(np.float32)

    def learn(self, idx: np.ndarray, h: np.ndarray, target: int, V: np.ndarray | None = None):
        if len(idx) == 0:
            return
        p = self.probs(self.potentials(idx, h) if V is None else V)
        err = -p
        err[target] += 1.0
        self.W[:, idx] += (self.lr * err)[:, None].astype(np.float32) * self._plasticity(idx, h)[None, :]

    def learn_reward(self, idx: np.ndarray, h: np.ndarray, action: int, R: float,
                     V: np.ndarray | None = None):
        if len(idx) == 0:
            return
        p = self.probs(self.potentials(idx, h) if V is None else V)
        elig = -p
        elig[action] += 1.0
        adv = R - self.baseline
        self.W[:, idx] += (self.lr * adv * elig)[:, None].astype(np.float32) * h[None, :]
        self.baseline += self.bl_beta * (R - self.baseline)
