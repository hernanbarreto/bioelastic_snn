"""Voz: dígitos hablados (Free Spoken Digit Dataset), una pasada online.
Se repite con 4 órdenes de presentación distintos.

Split oficial del dataset: grabaciones 0-4 de cada locutor/dígito a test (300),
el resto a entrenamiento (2700).

  stream : la red recibe el audio frame a frame (20 ms), ve los últimos 8 frames por
           la línea de retardos y decide por evidencia acumulada al final de la palabra.
  whole  : toda la palabra (24 frames normalizados) entra como una sola volea.

  python experiments/exp_fsdd.py
"""
from __future__ import annotations

import time

import numpy as np

from common import CsvLog
from bioelastic import AudioEncoder, BioElasticNet
from bioelastic.datasets import load_fsdd

SEEDS = (0, 1, 2, 3)


def main():
    recs, sr = load_fsdd()
    train = [r for r in recs if r[3] >= 5]
    test = [r for r in recs if r[3] < 5]
    summ = CsvLog("fsdd_summary", ["mode", "order_seed", "preq_acc", "test_acc", "neurons",
                                   "created", "pruned", "train_seconds"])
    setups = {
        "stream": (AudioEncoder(sr=sr, n_bands=32, hop_ms=20, dyn_range_db=35),
                   dict(context=8, delay=0.05), "accumulate"),
        "whole": (AudioEncoder(sr=sr, n_bands=32, n_frames=24, dyn_range_db=35),
                  dict(context=24, delay=0.0), "end"),
    }
    for mode, (enc, kw, decide) in setups.items():
        Tr = [enc(r[0]) for r in train]; Te = [enc(r[0]) for r in test]
        accs = []
        for seed in SEEDS:                                   # el test son sólo 300 palabras
            order = np.random.default_rng(seed).permutation(len(train))
            net = BioElasticNet(enc.n_inputs, 10, **kw)
            log = CsvLog(f"fsdd_{mode}_seed{seed}",
                         ["utterance", "acc_win", "neurons", "created", "pruned"])
            t0 = time.time(); ok = 0; win = 0
            for n, i in enumerate(order, 1):
                hit = net.run_sequence(list(Tr[i]), train[i][1], decide=decide) == train[i][1]
                ok += hit; win += hit
                if n % 300 == 0:
                    st = net.stats()
                    log.row(n, win / 300, st["neurons"], st["created"], st["pruned"]); win = 0
            log.close()
            secs = time.time() - t0
            acc = float(np.mean([net.run_sequence(list(f), learn=False, decide=decide) == r[1]
                                 for f, r in zip(Te, test)]))
            accs.append(acc)
            st = net.stats()
            summ.row(mode, seed, ok / len(train), acc, st["neurons"], st["created"], st["pruned"],
                     round(secs, 1))
        summ.row(mode, "mean", "", float(np.mean(accs)), "", "", "", "")
    summ.close()


if __name__ == "__main__":
    main()
