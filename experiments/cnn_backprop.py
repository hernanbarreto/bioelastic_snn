"""Referencia: CNN entrenada con backprop (NumPy), mismo flujo, misma pasada online.

Geometría igual a la jerarquía de `Cortex` para comparar profundidad con profundidad:
  cnn1  conv 5x5x32 + ReLU + max 2x2 (12x12)                      -> lineal -> softmax
  cnn2  conv 5x5x32 + max 2x2 -> conv 3x3x64 + max 2x2 (5x5)        -> lineal -> softmax
Adam, lotes de 10, una pasada sobre las primeras 55 000 (orden de la semilla). La tasa de
aprendizaje se elige en validación (últimas 5 000 de entrenamiento); el test sólo con --test.

  python experiments/cnn_backprop.py mnist|fashion [--seeds 0,1,2] [--train 55000] [--test]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from common import CsvLog
from exp_vision import encoder, sheet_shape, split
from perturb import PERTURBATIONS, perturb


class CNN:
    def __init__(self, convs=((5, 32), (3, 64)), n_out=10, side=28, channels=1, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.convs = []
        C, H = channels, side
        for k, F in convs:
            W = (rng.standard_normal((k * k * C, F)) * np.sqrt(2.0 / (k * k * C))).astype(np.float32)
            self.convs.append([k, W, np.zeros(F, dtype=np.float32)])
            H = (H - k + 1) // 2; C = F
        n = H * H * C
        self.Wo = (rng.standard_normal((n, n_out)) * np.sqrt(1.0 / n)).astype(np.float32)
        self.bo = np.zeros(n_out, dtype=np.float32)
        self.lr = lr; self.t = 0
        self._m = [np.zeros_like(p) for p in self.params()]
        self._v = [np.zeros_like(p) for p in self.params()]

    def params(self):
        out = []
        for _, W, b in self.convs:
            out += [W, b]
        return out + [self.Wo, self.bo]

    def forward(self, X, keep=False):
        cache = []
        for k, W, b in self.convs:
            B, H, Wd, C = X.shape
            cols = sliding_window_view(X, (k, k), axis=(1, 2))           # (B, oh, ow, C, k, k)
            oh, ow = cols.shape[1], cols.shape[2]
            cols = cols.transpose(0, 1, 2, 4, 5, 3).reshape(B * oh * ow, k * k * C)
            Z = (cols @ W + b).reshape(B, oh, ow, -1)
            A = np.maximum(Z, 0.0)
            ph, pw = oh // 2, ow // 2
            Ap = A[:, :ph * 2, :pw * 2].reshape(B, ph, 2, pw, 2, -1)
            P = Ap.max(axis=(2, 4))
            if keep:
                mask = Ap == P[:, :, None, :, None]
                cache.append((X.shape, cols, Z, mask, (oh, ow)))
            X = P
        h = X.reshape(len(X), -1)
        logits = h @ self.Wo + self.bo
        return logits, (cache, h)

    def train_batch(self, X, y):
        logits, (cache, h) = self.forward(X, keep=True)
        p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        d = p; d[np.arange(len(y)), y] -= 1.0; d /= len(y)
        grads = [None] * (2 * len(self.convs)) + [h.T @ d, d.sum(0)]
        dX = (d @ self.Wo.T).reshape((len(y),) + self._top_shape(cache))
        for li in range(len(self.convs) - 1, -1, -1):
            k, W, b = self.convs[li]
            xshape, cols, Z, mask, (oh, ow) = cache[li]
            B, H, Wd, C = xshape
            dA = np.zeros_like(Z)
            ph, pw = oh // 2, ow // 2
            dA[:, :ph * 2, :pw * 2] = (mask * dX[:, :, None, :, None, :]).reshape(B, ph * 2, pw * 2, -1)
            dZ = (dA * (Z > 0)).reshape(B * oh * ow, -1)
            grads[2 * li] = cols.T @ dZ; grads[2 * li + 1] = dZ.sum(0)
            if li:
                dcols = (dZ @ W.T).reshape(B, oh, ow, k, k, C)
                dX = np.zeros(xshape, dtype=np.float32)
                for i in range(k):
                    for j in range(k):
                        dX[:, i:i + oh, j:j + ow] += dcols[:, :, :, i, j]
        self._adam(grads)

    @staticmethod
    def _top_shape(cache):
        xshape, _, Z, _, (oh, ow) = cache[-1]
        return (oh // 2, ow // 2, Z.shape[-1])

    def _adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for p, g, m, v in zip(self.params(), grads, self._m, self._v):
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            mh = m / (1 - b1 ** self.t); vh = v / (1 - b2 ** self.t)
            p -= (self.lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)

    def predict(self, X, batch=500):
        return np.concatenate([self.forward(X[i:i + batch])[0].argmax(1)
                               for i in range(0, len(X), batch)])


ARCH = {"cnn1": ((5, 32),), "cnn2": ((5, 32), (3, 64))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="mnist")
    ap.add_argument("--models", default="cnn1,cnn2")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--train", type=int, default=0, help="0 = todo el entrenamiento")
    ap.add_argument("--val", type=int, default=5000)
    ap.add_argument("--lrs", default="3e-4,1e-3,3e-3")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--perturb", type=int, default=10_000,
                    help="imágenes de test para las pruebas de perturbación (0 = ninguna)")
    args = ap.parse_args()
    E = encoder(args.name)
    side, _, channels = sheet_shape(args.name)
    S = lambda X: np.stack([E(x) for x in X]).astype(np.float32)
    cols = ["model", "seed", "n_train", "lr", "val_acc", "test_acc"] + \
           [f"test_{p}" for p in PERTURBATIONS] + ["train_seconds"]
    log = CsvLog(f"{args.name}{args.tag}_cnn_summary", cols)
    if not args.train:
        args.train = 45_000 if args.name == "cifar10" else 55_000
    for seed in (int(s) for s in args.seeds.split(",")):
        (Xtr, ytr), (Xva, yva), (Xte, yte) = split(args.name, seed)
        Str, Sva = S(Xtr[:args.train]), S(Xva[:args.val])
        for model in args.models.split(","):
            best = None
            for lr in (float(v) for v in args.lrs.split(",")):
                net = CNN(ARCH[model], side=side, channels=channels, lr=lr, seed=seed)
                t0 = time.time()
                for i in range(0, len(Str), 10):
                    net.train_batch(Str[i:i + 10], ytr[i:i + 10].astype(np.int64))
                secs = time.time() - t0
                acc = float(np.mean(net.predict(Sva) == yva[:args.val]))
                print(f"{model} seed={seed} lr={lr:g} val={acc:.4f} ({secs:.0f}s)", flush=True)
                if best is None or acc > best[0]:
                    best = (acc, lr, net, secs)
            acc, lr, net, secs = best
            row = [model, seed, args.train, lr, acc]
            if args.test:
                row.append(float(np.mean(net.predict(S(Xte)) == yte)))
                n = args.perturb
                for p in PERTURBATIONS:
                    row.append(float(np.mean(net.predict(S(perturb(Xte[:n], p))) == yte[:n]))
                               if n else "")
            else:
                row += [""] * (1 + len(PERTURBATIONS))
            log.row(*row, round(secs, 1))
    log.close()


if __name__ == "__main__":
    main()
