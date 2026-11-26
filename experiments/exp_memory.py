"""Memoria de trabajo: recordar una clave después de una espera más larga que la ventana.

Cada episodio: una clave (1 de 4 símbolos), luego entre 10 y 30 distractores al azar
(8 símbolos) y al final una pregunta. La respuesta correcta es la clave. La red sólo
recibe el objetivo en la pregunta; el resto del episodio pasa sin enseñanza.

  window        un área con línea de retardos de 8 pasos: la clave ya salió de su ventana
  window+trace  se agrega un área con traza lenta (tau = 32 pasos)

  python experiments/exp_memory.py
"""
from __future__ import annotations

import numpy as np

from common import CsvLog
from bioelastic import Cortex

KEYS, DISTRACTORS = 4, 8
QUERY = KEYS + DISTRACTORS                 # símbolo de la pregunta
N_SYM = QUERY + 1
EPISODES, WIN = 6000, 500


def episode(rng):
    key = int(rng.integers(KEYS))
    gap = int(rng.integers(10, 31))
    return key, [key] + list(KEYS + rng.integers(DISTRACTORS, size=gap)) + [QUERY]


def main():
    setups = {
        "window": [dict(context=8, delay=0.1)],
        "window+trace": [dict(context=8, delay=0.1), dict(tau=32.0)],
    }
    log = CsvLog("memory", ["episode", "model", "acc_win", "assoc_neurons"])
    for model, heads in setups.items():
        rng = np.random.default_rng(0)
        net = Cortex(N_SYM, KEYS, heads=heads)
        win = 0
        for ep in range(1, EPISODES + 1):
            key, seq = episode(rng)
            net.reset_context()
            for s in seq[:-1]:
                net.forward(int(s))                    # la espera: sin enseñanza
            win += net.step(int(seq[-1]), key) == key
            if ep % WIN == 0:
                log.row(ep, model, win / WIN, net.stats()["neurons"]); win = 0
    log.close()


if __name__ == "__main__":
    main()
