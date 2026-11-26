"""Mismo problema que bioNet_online_B (generador, semilla y ruido idénticos).

Protocolo del original: 800 muestras de evaluación, 3000 de warm-up, 16000 pasos
online, evaluación cada 1000. Acá las 3000 de warm-up se descartan (no se usan).

  python experiments/exp_synthetic.py [n_classes]
"""
from __future__ import annotations

import sys
from collections import deque

import numpy as np

from common import CsvLog
from bioelastic import BioElasticNet, LatencyEncoder
from bioelastic.datasets import bionet_stream

N_STEPS, EVAL_EVERY, WIN = 16000, 1000, 600


def run(mode: str, C: int):
    gen = bionet_stream(n_classes=C, seed=11)
    enc = LatencyEncoder(256, jitter=3.5 / 95.0, drop_p=0.08, seed=7)   # ruido del original
    X_eval, y_eval = zip(*(next(gen) for _ in range(800)))
    for _ in range(3000):
        next(gen)
    net = BioElasticNet(256, C, seed=7)
    cent = np.zeros((C, 256)); cnt = np.zeros(C)                         # referencia: centroide por clase
    log = CsvLog(f"synthetic_C{C}_{mode}",
                 ["step", "acc_win", "acc_eval", "acc_eval_centroid", "neurons", "created", "pruned"])
    hits = deque(maxlen=WIN)
    for step in range(1, N_STEPS + 1):
        x, y = next(gen)
        s = enc(x)
        if mode == "supervised":
            pred = net.step(s, y)
        else:                                                           # sólo recompensa +1/-1
            pred = net.forward(s, explore=True)
            net.reinforce(+1.0 if pred == y else -1.0)
        hits.append(pred == y)
        cnt[y] += 1; cent[y] += (s - cent[y]) / cnt[y]
        if step % EVAL_EVERY == 0:
            S = [enc(x) for x in X_eval]
            acc = float(np.mean([net.predict(s) == t for s, t in zip(S, y_eval)]))
            cn = cent / np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-8)
            acc_c = float(np.mean([int(np.argmax(cn @ s)) == t for s, t in zip(S, y_eval)]))
            st = net.stats()
            log.row(step, float(np.mean(hits)), acc, acc_c, st["neurons"], st["created"], st["pruned"])
    log.close()


if __name__ == "__main__":
    C = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for mode in ("supervised", "reward"):
        print(f"--- {mode}, C={C}")
        run(mode, C)
