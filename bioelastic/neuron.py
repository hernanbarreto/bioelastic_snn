"""Neurona integrate-and-fire TTFS, resuelta de forma exacta y dirigida por eventos.

Modelo: IF sin fuga con sinapsis de corriente (cada spike presináptico enciende una
corriente constante w_i). El potencial es lineal a tramos:

    V_j(t) = g * sum_{i : t_i < t} w_ji * (t - t_i)

Consecuencias que usa todo el paquete:
  * Al final de la ventana (t = 1):  V_j(1) = g * w_j . s   con  s_i = 1 - t_i.
    Con ||w_j|| = 1 y g = 1/||s||, V_j(1) es el coseno entre el peso y la entrada.
  * El umbral se cruza dentro de un tramo, y el tiempo de cruce tiene forma cerrada.
    No hace falta simular paso a paso.

Convención de código de latencia: una "volea" es un vector de fuerzas s en [0, 1];
s_i = 0 significa "sin spike" y s_i > 0 significa un spike en t_i = 1 - s_i
(más fuerte = más temprano).
"""
from __future__ import annotations

import numpy as np


def first_spike_times(Wc: np.ndarray, t_sorted: np.ndarray, theta: np.ndarray,
                      gain: float) -> np.ndarray:
    """Tiempo exacto del primer spike de cada neurona candidata.

    Wc       : (n_act, M) pesos de M neuronas sobre las aferentes activas,
               ordenadas por tiempo de spike ascendente.
    t_sorted : (n_act,) tiempos de spike ascendentes en [0, 1).
    theta    : (M,) umbrales.
    gain     : ganancia divisiva de entrada (1/||s||).

    Devuelve (M,) tiempos en (0, 1]; np.inf si la neurona no dispara en la ventana.
    """
    A = np.cumsum(Wc, axis=0, dtype=np.float64)                       # pendiente tras cada evento
    B = np.cumsum(Wc * t_sorted[:, None], axis=0, dtype=np.float64)
    t_next = np.append(t_sorted[1:], 1.0)
    V_next = gain * (A * t_next[:, None] - B)                         # V justo antes del evento siguiente
    crossed = V_next >= theta[None, :]
    m = crossed.argmax(axis=0)                                        # primer tramo donde cruza
    cols = np.arange(Wc.shape[1])
    fired = crossed[m, cols]
    a = A[m, cols]
    t_star = np.full(Wc.shape[1], np.inf)
    ok = fired & (a > 0)
    t_star[ok] = (theta[ok] / gain + B[m, cols][ok]) / a[ok]
    return t_star
