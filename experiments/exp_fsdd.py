"""Voz: dígitos hablados (Free Spoken Digit Dataset), una pasada online.
Se repite con 4 órdenes de presentación distintos (el test son sólo 300 palabras).

Split oficial del dataset: grabaciones 0-4 de cada locutor/dígito a test (300),
el resto a entrenamiento (2700).

  stream  el audio entra frame a frame (20 ms); un área de asociación ve los últimos 8
          frames por la línea de retardos y decide por evidencia acumulada.
  depth1  la palabra completa (espectrograma 24 x 32) como una lámina; un área de asociación.
  depth2  la misma lámina por la jerarquía: A1 (rasgos tiempo-frecuencia) + memoria +
          proyección directa + repaso. Misma configuración que en imagen.
  depth3  con una segunda área sensorial.

  python experiments/exp_fsdd.py
"""
from __future__ import annotations

import time

import numpy as np

from common import CsvLog
from bioelastic import AudioEncoder, BioElasticNet, Cortex
from bioelastic.datasets import load_fsdd
from exp_vision import CONFIGS

SEEDS = (0, 1, 2, 3)
SHEET = (24, 32, 1)


def main():
    recs, sr = load_fsdd()
    train = [r for r in recs if r[3] >= 5]
    test = [r for r in recs if r[3] < 5]
    summ = CsvLog("fsdd_summary", ["model", "order_seed", "preq_acc", "test_acc", "features",
                                   "assoc_neurons", "train_seconds"])
    stream = AudioEncoder(sr=sr, n_bands=32, hop_ms=20, dyn_range_db=35)
    whole = AudioEncoder(sr=sr, n_bands=32, n_frames=24, dyn_range_db=35)
    data = {
        "stream": ([list(stream(r[0])) for r in train], [list(stream(r[0])) for r in test]),
        "whole": ([whole(r[0]).reshape(SHEET) for r in train], [whole(r[0]).reshape(SHEET) for r in test]),
    }
    for model in ("stream", "depth1", "depth2", "depth3"):
        Tr, Te = data["stream" if model == "stream" else "whole"]
        accs = []
        for seed in SEEDS:
            order = np.random.default_rng(seed).permutation(len(train))
            if model == "stream":
                net = BioElasticNet(32, 10, context=8, delay=0.05)
                fit = lambda x, y: net.run_sequence(x, y, decide="accumulate")
                predict = lambda x: net.run_sequence(x, learn=False, decide="accumulate")
            else:
                net = Cortex(SHEET, 10, **CONFIGS[model])
                fit, predict = net.step, net.predict
            log = CsvLog(f"fsdd_{model}_seed{seed}", ["utterance", "acc_win", "assoc_neurons"])
            t0 = time.time(); ok = 0; win = 0
            for n, i in enumerate(order, 1):
                hit = fit(Tr[i], train[i][1]) == train[i][1]
                ok += hit; win += hit
                if n % 300 == 0:
                    log.row(n, win / 300, net.stats()["neurons"]); win = 0
            log.close()
            secs = time.time() - t0
            acc = float(np.mean([predict(x) == r[1] for x, r in zip(Te, test)]))
            accs.append(acc)
            st = net.stats()
            summ.row(model, seed, ok / len(train), acc, st.get("features", 0), st["neurons"],
                     round(secs, 1))
        summ.row(model, "mean", "", float(np.mean(accs)), "", "", "")
    summ.close()


if __name__ == "__main__":
    main()
