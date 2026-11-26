"""Imagen: MNIST o Fashion-MNIST en una sola pasada online, por profundidad de la red.

  depth1  un área de asociación sobre los píxeles (la red de v0.1)
  depth2  V1 (rasgos locales) + memoria + proyección directa + repaso
  depth3  V1 + V2 + memoria + proyección directa + repaso

Mismo encoder y mismos hiperparámetros en los dos datasets. Referencia con backprop:
MLP (256 ocultas) por SGD/Adam sobre el mismo flujo, una pasada, lotes de 10; se prueba
con tres tasas de aprendizaje y se informa la mejor.

  python experiments/exp_vision.py mnist|fashion
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np

from common import CsvLog
from bioelastic import Cortex, ImageEncoder
from bioelastic.datasets import load_fashion, load_mnist

LOG_EVERY = 2000
V1 = dict(rf=5, pool=2, max_features=256, theta=0.95, floor=0.3, k=8)
V2 = dict(rf=3, pool=2, max_features=256, theta=0.5, floor=0.2, k=4)
DEEP = dict(heads=[dict(source="input")], projection=dict(source="all"), replay=0.5, joint=False)
CONFIGS = {
    "depth1": dict(heads=[{}]),
    "depth2": dict(areas=[V1], **DEEP),
    "depth3": dict(areas=[V1, V2], **DEEP),
}


def encoder():
    enc = ImageEncoder((28, 28), blur=0.4)
    return lambda x: enc(x).reshape(28, 28, 1)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    Xtr, ytr, Xte, yte = load_mnist() if name == "mnist" else load_fashion()
    E = encoder()
    summ = CsvLog(f"{name}_summary", ["model", "test_acc", "preq_acc", "areas", "features",
                                      "assoc_neurons", "units", "synapses", "train_seconds"])
    for model, kw in CONFIGS.items():
        net = Cortex((28, 28, 1), 10, **kw)
        log = CsvLog(f"{name}_{model}", ["step", "acc_win", "features", "assoc_neurons"])
        t0 = time.time(); win = 0; ok = 0
        for i in range(len(Xtr)):
            hit = net.step(E(Xtr[i]), int(ytr[i])) == ytr[i]
            win += hit; ok += hit
            if (i + 1) % LOG_EVERY == 0:
                st = net.stats()
                log.row(i + 1, win / LOG_EVERY, st["features"], st["neurons"]); win = 0
        secs = time.time() - t0
        log.close()
        acc = float(np.mean([net.predict(E(x)) == y for x, y in zip(Xte, yte)]))
        st = net.stats()
        summ.row(model, acc, ok / len(Xtr), st["depth"], st["features"], st["neurons"], st["units"],
                 st["synapses"], round(secs, 1))

    from sklearn.neural_network import MLPClassifier
    warnings.filterwarnings("ignore")
    Str = np.stack([E(x).ravel() for x in Xtr]); Ste = np.stack([E(x).ravel() for x in Xte])
    best = (0.0, None)
    for lr in (1e-3, 3e-3, 1e-2):
        mlp = MLPClassifier(hidden_layer_sizes=(256,), learning_rate_init=lr, random_state=0)
        for i in range(0, len(Str), 10):
            mlp.partial_fit(Str[i:i + 10], ytr[i:i + 10], classes=np.arange(10))
        best = max(best, (float(np.mean(mlp.predict(Ste) == yte)), lr))
    summ.row(f"mlp_backprop_lr{best[1]:g}", best[0], "", 2, "", 256, 256 + 10, 784 * 256 + 2560, "")
    summ.close()


if __name__ == "__main__":
    main()
