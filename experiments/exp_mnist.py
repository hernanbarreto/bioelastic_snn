"""Imagen: MNIST en una sola pasada online (cada muestra se ve una vez, test-then-train).

  python experiments/exp_mnist.py
"""
from __future__ import annotations

import time

import numpy as np

from common import CsvLog
from bioelastic import BioElasticNet, ImageEncoder
from bioelastic.datasets import load_mnist

LOG_EVERY = 2000


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    enc = ImageEncoder((28, 28), blur=0.7)
    net = BioElasticNet(enc.n_inputs, 10)
    log = CsvLog("mnist_online", ["step", "acc_win", "neurons", "created", "pruned"])
    t0 = time.time(); win = 0
    for i in range(len(Xtr)):
        win += net.step(enc(Xtr[i]), int(ytr[i])) == ytr[i]
        if (i + 1) % LOG_EVERY == 0:
            st = net.stats()
            log.row(i + 1, win / LOG_EVERY, st["neurons"], st["created"], st["pruned"]); win = 0
    train_s = time.time() - t0
    acc = float(np.mean([net.predict(enc(x)) == y for x, y in zip(Xte, yte)]))
    log.close()

    # Referencia con backprop: MLP (256 ocultas) por SGD, mismo flujo, una pasada, lotes de 10.
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(256,), learning_rate_init=1e-3, random_state=0)
    Str = np.stack([enc(x) for x in Xtr])
    for i in range(0, len(Str), 10):
        mlp.partial_fit(Str[i:i + 10], ytr[i:i + 10], classes=np.arange(10))
    acc_mlp = float(np.mean(mlp.predict(np.stack([enc(x) for x in Xte])) == yte))

    summ = CsvLog("mnist_summary", ["test_acc", "mlp_sgd_test_acc", "neurons", "created", "pruned",
                                    "train_seconds"])
    st = net.stats()
    summ.row(acc, acc_mlp, st["neurons"], st["created"], st["pruned"], round(train_s, 1)); summ.close()


if __name__ == "__main__":
    main()
