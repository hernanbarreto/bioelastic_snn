"""Elasticidad: MNIST incremental por clases (5 tareas de 2 dígitos, sin repaso de datos).

Los dígitos llegan de a pares (0-1, 2-3, ...). Cada muestra se ve una vez y nunca se
vuelve a mostrar una tarea anterior. Después de cada tarea se evalúa sobre los 10 dígitos.

  depth1            un área de asociación sobre los píxeles
  depth2_no_replay  V1 + memoria + proyección directa, sin consolidación
  depth2            lo mismo con repaso: los recuerdos de la memoria reentrenan la proyección
  mlp_sgd           MLP (256 ocultas) con backprop sobre el mismo flujo

Con `--eval val` se entrena con las primeras 55 000 de entrenamiento y se evalúa sobre las
últimas 5 000 (no toca el test); `--models` elige las redes (nombres de `exp_vision.CONFIGS`,
con sufijo `_no_replay` para apagar el repaso) y `--per-task` acota las muestras por tarea.
Nota: desde v0.3 `depth2`/`depth3` son las redes con error local y árbitro; las de v0.2
son `depth2_v02`/`depth3_v02`.

  python experiments/exp_incremental.py [--eval test|val] [--models ...] [--per-task N]
"""
from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="test", choices=("test", "val"))
    ap.add_argument("--models", default="depth1,depth2_v02_no_replay,depth2_v02")
    ap.add_argument("--per-task", type=int, default=0, help="0 = todas las muestras de cada tarea")
    args = ap.parse_args()
    Xtr, ytr, Xte, yte = load_mnist()
    suffix = ""
    if args.eval == "val":                                   # el test no se toca
        Xtr, ytr, Xte, yte = Xtr[:55_000], ytr[:55_000], Xtr[55_000:], ytr[55_000:]
        suffix = "_val"
    if args.per_task:
        keep = np.concatenate([np.flatnonzero(np.isin(ytr, t))[:args.per_task] for t in TASKS])
        keep.sort(); Xtr, ytr = Xtr[keep], ytr[keep]
        suffix += f"_n{args.per_task}"
    E = encoder()
    Ste = [E(x) for x in Xte]
    nets = {}
    for m in args.models.split(","):
        base = m[:-len("_no_replay")] if m.endswith("_no_replay") else m
        cfg = dict(CONFIGS[base], replay=0.0) if base != m else CONFIGS[base]
        nets[m] = Cortex((28, 28, 1), 10, **cfg)
    mlp = MLPClassifier(hidden_layer_sizes=(256,), learning_rate_init=1e-3, random_state=0)
    classes = np.arange(10)
    growth = CsvLog(f"incremental_growth{suffix}", ["step", "task", "model", "features", "assoc_neurons"])
    evals = CsvLog(f"incremental_eval{suffix}",
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
                for model in nets:
                    st = nets[model].stats()
                    growth.row(step, k, model, st["features"], st["neurons"])
        for model, net in nets.items():
            evaluate(k, model, np.array([net.predict(s) for s in Ste]))
        evaluate(k, "mlp_sgd", mlp.predict(np.stack([s.ravel() for s in Ste])))
    growth.close(); evals.close()


if __name__ == "__main__":
    main()
