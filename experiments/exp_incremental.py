"""Elasticidad: MNIST incremental por clases (5 tareas de 2 dígitos, sin repaso de datos).

Los dígitos llegan de a pares (0-1, 2-3, ...). Cada muestra se ve una vez y nunca se
vuelve a mostrar una tarea anterior. Después de cada tarea se evalúa sobre los 10 dígitos.

  depth1            un área de asociación sobre los píxeles
  depth2_no_replay  V1 + memoria + proyección directa, sin consolidación
  depth2            lo mismo con repaso: los recuerdos de la memoria reentrenan la proyección
  mlp_sgd           MLP (256 ocultas) con backprop sobre el mismo flujo

  python experiments/exp_incremental.py
"""
from __future__ import annotations

import warnings

import numpy as np

from common import CsvLog
from bioelastic import Cortex
from bioelastic.datasets import load_mnist
from exp_vision import CONFIGS, encoder

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
LOG_EVERY = 1000


def main():
    from sklearn.neural_network import MLPClassifier
    warnings.filterwarnings("ignore")
    Xtr, ytr, Xte, yte = load_mnist()
    E = encoder()
    Ste = [E(x) for x in Xte]
    nets = {
        "depth1": Cortex((28, 28, 1), 10, **CONFIGS["depth1"]),
        "depth2_no_replay": Cortex((28, 28, 1), 10, **dict(CONFIGS["depth2"], replay=0.0)),
        "depth2": Cortex((28, 28, 1), 10, **CONFIGS["depth2"]),
    }
    mlp = MLPClassifier(hidden_layer_sizes=(256,), learning_rate_init=1e-3, random_state=0)
    classes = np.arange(10)
    growth = CsvLog("incremental_growth", ["step", "task", "model", "features", "assoc_neurons"])
    evals = CsvLog("incremental_eval",
                   ["after_task", "model", "acc_all"] + [f"acc_task{k + 1}" for k in range(len(TASKS))])

    def evaluate(k, model, pred):
        accs = [float(np.mean(pred[np.isin(yte, t)] == yte[np.isin(yte, t)])) for t in TASKS]
        evals.row(k, model, float(np.mean(pred == yte)), *accs)

    step = 0
    for k, task in enumerate(TASKS, 1):
        buf_x, buf_y = [], []
        for i in np.flatnonzero(np.isin(ytr, task)):
            s = E(Xtr[i]); y = int(ytr[i])
            for net in nets.values():
                net.step(s, y)
            buf_x.append(s.ravel()); buf_y.append(y)
            if len(buf_x) == 10:                                  # SGD online en mini-lotes de 10
                mlp.partial_fit(np.stack(buf_x), buf_y, classes=classes)
                buf_x, buf_y = [], []
            step += 1
            if step % LOG_EVERY == 0:
                for model in ("depth1", "depth2"):
                    st = nets[model].stats()
                    growth.row(step, k, model, st["features"], st["neurons"])
        for model, net in nets.items():
            evaluate(k, model, np.array([net.predict(s) for s in Ste]))
        evaluate(k, "mlp_sgd", mlp.predict(np.stack([s.ravel() for s in Ste])))
    growth.close(); evals.close()


if __name__ == "__main__":
    main()
