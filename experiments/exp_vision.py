"""Imagen: MNIST o Fashion-MNIST en una sola pasada online, por profundidad de la red.

  depth1  un área de asociación sobre los píxeles (la red de v0.1)
  depth2  V1 (rasgos locales) + memoria + proyección directa + repaso
  depth3  V1 + V2 + memoria + proyección directa + repaso

Las variantes `*_v02` son las de v0.2 (áreas sin supervisión, proyección sobre la
concatenación, suma de potenciales); las `depthN` actuales agregan error local por área,
proyección balanceada y árbitro de la decisión (ver `bioelastic/area.py` y `Cortex`).
Elegidas en validación con `explore_depth.py` y `explore_area.py`.

Protocolo estricto: se entrena en una pasada con las primeras 55 000 imágenes de
entrenamiento (en el orden que fija la semilla), se ajusta mirando sólo las últimas 5 000
de entrenamiento (validación) y el test de 10 000 se evalúa únicamente con `--test`.

  python experiments/exp_vision.py mnist|fashion [--models depth1,depth2,depth3]
         [--seeds 0,1,2] [--train 55000] [--val 5000] [--test] [--mlp] [--tag nombre]
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

from common import CsvLog
from perturb import PERTURBATIONS, perturb
from bioelastic import Cortex, ImageEncoder
from bioelastic.datasets import load_cifar10, load_fashion, load_mnist

LOG_EVERY = 5000
N_FIT = 55_000
V1 = dict(rf=5, pool=2, max_features=256, theta=0.95, floor=0.3, k=8)
V2 = dict(rf=3, pool=2, max_features=256, theta=0.5, floor=0.2, k=4)
DEEP = dict(heads=[dict(source="input")], projection=dict(source="all"), replay=0.5, joint=False)

LOCAL = dict(lr=0.5)                                     # error local por área
V1L = dict(V1, local=LOCAL)
V2L = dict(V2, local=LOCAL)
DEEP_B = dict(DEEP, projection=dict(source="all", balance=True), arbiter=0.01)

CONFIGS = {
    "depth1": dict(heads=[{}]),
    "depth2_v02": dict(areas=[V1], **DEEP),
    "depth3_v02": dict(areas=[V1, V2], **DEEP),
    "depth2": dict(areas=[V1L], **DEEP_B),
    "depth3": dict(areas=[V1L, V2L], **DEEP_B),
}


def encoder(name: str = "mnist"):
    if name == "cifar10":
        return cifar_retina
    enc = ImageEncoder((28, 28), blur=0.4)
    return lambda x: enc(x).reshape(28, 28, 1)


def cifar_retina(img) -> np.ndarray:
    """Retina para color: contraste normalizado por imagen y separado en canales ON y OFF
    (32, 32, 6), fuerzas en [0, 1]."""
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - x.mean()) / max(float(x.std()), 1e-3)
    s = np.concatenate([np.maximum(x, 0.0), np.maximum(-x, 0.0)], axis=2)
    s = s / max(float(s.max()), 1e-6)
    s[s < 0.05] = 0.0
    return s


def sheet_shape(name: str):
    return (32, 32, 6) if name == "cifar10" else (28, 28, 1)


def split(name: str, seed: int):
    """Entrenamiento = todas menos las últimas 5 000 de entrenamiento (55 000 en MNIST y
    Fashion, 45 000 en CIFAR-10), en el orden de la semilla; validación = últimas 5 000."""
    loaders = dict(mnist=load_mnist, fashion=load_fashion, cifar10=load_cifar10)
    Xtr, ytr, Xte, yte = loaders[name]()
    n_fit = len(Xtr) - 5000
    order = np.random.default_rng(seed).permutation(n_fit)
    return (Xtr[order], ytr[order]), (Xtr[n_fit:], ytr[n_fit:]), (Xte, yte)


def parts_accuracy(net: Cortex, E, X, y, combos: bool = False):
    """Exactitud de la red completa y de cada componente por separado (sin aprender).
    Con `combos`, también de sumas de log-probabilidades de componentes (exploración)."""
    hits = {}
    for x, t in zip(X, y):
        pots = {"net": None}
        pred = net.predict(E(x))
        hits["net"] = hits.get("net", 0) + (pred == t)
        if net.heads:
            pots["memory"] = net.heads[0]._last.V
        if net.proj is not None:
            pots["proj"] = net.proj.potentials(*net._pv)
        for i, a in enumerate(net.areas):
            V = a.potentials()
            if V is not None:
                pots[f"local{i + 1}"] = V
        for k, V in pots.items():
            if V is not None:
                hits[k] = hits.get(k, 0) + (int(np.argmax(V)) == t)
        if combos:
            lp = {k: np.log(net.areas[int(k[5:]) - 1].clf.probs(V) + 1e-9) if k.startswith("local")
                  else np.log(_softmax(V, net.temperature) + 1e-9)
                  for k, V in pots.items() if V is not None}
            names = [k for k in lp if k.startswith("local")]
            combos_ = {tuple(c) for c in (["memory"] + names[:1], ["memory"] + names, names,
                                          ["memory", "proj"] + names)}
            for combo in combos_:
                if all(c in lp for c in combo) and len(combo) > 1:
                    key = "+".join(combo)
                    hits[key] = hits.get(key, 0) + (int(np.argmax(sum(lp[c] for c in combo))) == t)
    return {k: v / len(y) for k, v in hits.items()}


def _softmax(V, T):
    e = np.exp((V - V.max()) / T)
    return e / e.sum()


def run(name, model, seed, n_train, n_val, test, tag="", cfg=None, combos=False):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(name, seed)
    E = encoder(name)
    net = Cortex(sheet_shape(name), 10, seed=seed, **(CONFIGS[model] if cfg is None else cfg))
    log = CsvLog(f"{name}{tag}_{model}_s{seed}", ["step", "acc_win", "features", "assoc_neurons"])
    t0 = time.time(); win = 0; ok = 0
    for i in range(n_train):
        hit = net.step(E(Xtr[i]), int(ytr[i])) == ytr[i]
        win += hit; ok += hit
        if (i + 1) % LOG_EVERY == 0:
            st = net.stats()
            log.row(i + 1, win / LOG_EVERY, st["features"], st["neurons"]); win = 0
    secs = time.time() - t0
    log.close()
    for a in net.areas:                                   # evaluación: nada aprende
        a._cache = None
    val = parts_accuracy(net, E, Xva[:n_val], yva[:n_val], combos)
    tst = {}
    if test:                                              # el test se toca una sola vez, al final
        tst = parts_accuracy(net, E, Xte, yte)
        for p in PERTURBATIONS:                           # perturbaciones nunca vistas
            tst[p] = float(np.mean([net.predict(E(x)) == t for x, t in zip(perturb(Xte, p), yte)]))
    return net, ok / n_train, val, tst, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="mnist")
    ap.add_argument("--models", default="depth1,depth2,depth3")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--train", type=int, default=0, help="0 = todo el entrenamiento")
    ap.add_argument("--val", type=int, default=5000)
    ap.add_argument("--test", action="store_true", help="evalúa el test (una sola vez, al final)")
    ap.add_argument("--mlp", action="store_true", help="referencia MLP con backprop, misma pasada")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    summ = CsvLog(f"{args.name}{args.tag}_summary",
                  ["model", "seed", "n_train", "preq_acc", "val_acc", "val_parts", "test_acc",
                   "test_parts"] + [f"test_{p}" for p in PERTURBATIONS] +
                  ["features", "assoc_neurons", "synapses", "train_seconds"])
    fmt = lambda d: " ".join(f"{k}:{v:.4f}" for k, v in d.items()
                             if k != "net" and k not in PERTURBATIONS)
    if not args.train:
        args.train = 45_000 if args.name == "cifar10" else N_FIT
    for model in args.models.split(","):
        for seed in (int(s) for s in args.seeds.split(",")):
            net, preq, val, tst, secs = run(args.name, model, seed, args.train, args.val,
                                            args.test, args.tag)
            st = net.stats()
            summ.row(model, seed, args.train, preq, val["net"], fmt(val), tst.get("net", ""),
                     fmt(tst), *[tst.get(p, "") for p in PERTURBATIONS],
                     st["features"], st["neurons"], st["synapses"], round(secs, 1))
    if args.mlp:
        from sklearn.neural_network import MLPClassifier
        warnings.filterwarnings("ignore")
        E = encoder(args.name)
        for seed in (int(s) for s in args.seeds.split(",")):
            (Xtr, ytr), (Xva, yva), (Xte, yte) = split(args.name, seed)
            S = lambda X: np.stack([E(x).ravel() for x in X])
            Str, Sva = S(Xtr[:args.train]), S(Xva[:args.val])
            best = None
            for lr in (1e-3, 3e-3, 1e-2):                 # la tasa se elige en validación
                mlp = MLPClassifier(hidden_layer_sizes=(256,), learning_rate_init=lr, random_state=seed)
                for i in range(0, len(Str), 10):
                    mlp.partial_fit(Str[i:i + 10], ytr[i:i + 10], classes=np.arange(10))
                acc = float(np.mean(mlp.predict(Sva) == yva[:args.val]))
                if best is None or acc > best[0]:
                    best = (acc, lr, mlp)
            te = float(np.mean(best[2].predict(S(Xte)) == yte)) if args.test else ""
            summ.row(f"mlp_backprop_lr{best[1]:g}", seed, args.train, "", best[0], "", te, "",
                     *[""] * len(PERTURBATIONS), "", 256, 784 * 256 + 2560, "")
    summ.close()


if __name__ == "__main__":
    main()
