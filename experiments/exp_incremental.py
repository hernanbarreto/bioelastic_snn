"""Elasticidad: MNIST incremental por clases (5 tareas de 2 dígitos, sin repaso).

Los dígitos llegan de a pares (0-1, 2-3, ...). Cada muestra se ve una vez y nunca
se vuelve a mostrar una tarea anterior. Después de cada tarea se evalúa sobre los
10 dígitos. Referencia: MLP (256 ocultas) entrenado online por SGD con el mismo flujo.

  python experiments/exp_incremental.py
"""
from __future__ import annotations

import warnings

import numpy as np

from common import CsvLog
from bioelastic import BioElasticNet, ImageEncoder
from bioelastic.datasets import load_mnist

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
LOG_EVERY = 1000


def main():
    from sklearn.neural_network import MLPClassifier
    warnings.filterwarnings("ignore")
    Xtr, ytr, Xte, yte = load_mnist()
    enc = ImageEncoder((28, 28), blur=0.7)
    Ste = np.stack([enc(x) for x in Xte])
    net = BioElasticNet(enc.n_inputs, 10)
    mlp = MLPClassifier(hidden_layer_sizes=(256,), learning_rate_init=1e-3, random_state=0)
    classes = np.arange(10)

    growth = CsvLog("incremental_growth", ["step", "task", "neurons", "created", "pruned"])
    evals = CsvLog("incremental_eval",
                   ["after_task", "model", "acc_all"] + [f"acc_task{k + 1}" for k in range(len(TASKS))])

    def evaluate(tag, pred):
        accs = [float(np.mean(pred[np.isin(yte, t)] == yte[np.isin(yte, t)])) for t in TASKS]
        evals.row(tag[0], tag[1], float(np.mean(pred == yte)), *accs)

    step = 0
    for k, task in enumerate(TASKS, 1):
        idx = np.flatnonzero(np.isin(ytr, task))
        buf_x, buf_y = [], []
        for i in idx:
            s = enc(Xtr[i]); y = int(ytr[i])
            net.step(s, y)
            buf_x.append(s); buf_y.append(y)
            if len(buf_x) == 10:                                  # SGD online en mini-lotes de 10
                mlp.partial_fit(np.stack(buf_x), buf_y, classes=classes)
                buf_x, buf_y = [], []
            step += 1
            if step % LOG_EVERY == 0:
                st = net.stats()
                growth.row(step, k, st["neurons"], st["created"], st["pruned"])
        evaluate((k, "bioelastic"), np.array([net.predict(s) for s in Ste]))
        evaluate((k, "mlp_sgd"), mlp.predict(Ste))
    growth.close(); evals.close()


if __name__ == "__main__":
    main()
