"""MNIST-addition (Manhaeve et al., 2018, DeepProbLog): pares de imágenes, sólo la suma
(0-18) como etiqueta y la regla de la suma conocida. La percepción nunca ve una etiqueta
de dígito.

Equivalente local de lo que hace DeepProbLog (marginalizar sobre las asignaciones
conjuntas de dígitos y entrenar la percepción con esa probabilidad): el razonamiento
calcula, como inferencia interna acotada, el posterior conjunto

    q(a, b) ∝ p(a | x1)^gamma p(b | x2)^gamma 1[a + b = y]

y la percepción (Cortex) aprende de cada imagen con su marginal como objetivo blando:
cada regla local usa q - p como tercer factor (`Cortex.teach(soft=q)`). Online, una sola
pasada, test-then-train (la suma predicha se mide antes de aprender del par).

Referencias publicadas: DeepProbLog 97.2 ± 0.45 % de exactitud en la suma con 30 000
pares, 92.2 ± 1.6 % con 3 000 (varias épocas, backprop). Criterio del proyecto: >= 95 %
por dígito sin etiquetas de dígito.

Protocolo: entrenamiento = pares de MNIST train[:55000] (cada imagen una vez, en el
orden de la semilla); validación = pares de train[55000:] (elige hiperparámetros); test =
pares de t10k, tocado sólo en las corridas finales (--test).

  python experiments/exp_mnist_addition.py --seed S [--variante V] [--pares N] [--test]
Escribe results/integracion3_suma_mnist[val]_<variante>_seed<S>.csv (formato largo).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import RESULTS, CsvLog
from bioelastic import Cortex
from bioelastic.datasets import load_mnist
from exp_vision import CONFIGS, encoder

HEADER = ["prueba", "variante", "semilla", "metrica", "valor", "referencia", "nota"]
A = np.arange(10)
REGLA = (A[:, None] + A[None, :])                 # la suma conocida: a + b


VARIANTES = {
    # gamma: cuánto pesa la percepción en el posterior; h: homeostasis de las salidas
    "base": dict(config="depth2", gamma=1.0, h=0.0),
    "g0.5": dict(config="depth2", gamma=0.5, h=0.0),
    "h": dict(config="depth2", gamma=1.0, h=0.02),
    "d1": dict(config="depth1", gamma=1.0, h=0.0),
    # control: objetivo duro (el máximo del posterior) en vez del blando
    "duro": dict(config="depth2", gamma=1.0, h=0.0, duro=True),
    # ablación del ensayo de conceptos (repaso de la memoria a la proyección: Marconato et al. 2023)
    "sin_repaso": dict(config="depth2", gamma=1.0, h=0.0, kw=dict(replay=0.0)),
}


def posterior_suma(p1, p2, y, gamma=1.0):
    """Posterior conjunto sobre (a, b) dada la suma y; marginales de cada imagen."""
    q = (p1[:, None] ** gamma) * (p2[None, :] ** gamma) * (REGLA == int(y))
    z = q.sum()
    if z <= 0:
        q = (REGLA == int(y)).astype(float); z = q.sum()
    q = q / z
    return q, q.sum(1), q.sum(0)


def suma_predicha(p1, p2) -> int:
    """La suma más probable bajo la percepción: argmax_y sum_{a+b=y} p(a) p(b)."""
    J = p1[:, None] * p2[None, :]
    return int(np.argmax(np.bincount(REGLA.ravel(), weights=J.ravel(), minlength=19)))


class Red:
    def __init__(self, cfg, seed):
        self.net = Cortex((28, 28, 1), 10, seed=seed, homeostasis=cfg["h"],
                          **dict(CONFIGS[cfg["config"]], **cfg.get("kw", {})))
        self.E = encoder()

    def p(self, img):
        self.net.forward(self.E(img))
        return self.net._p().astype(np.float64)

    def aprender(self, img, q, duro=False):
        self.net.forward(self.E(img))
        if duro:
            self.net.teach(int(np.argmax(q)))
        else:
            self.net.teach(soft=q)


def evaluar(red, X, y, pares):
    """Suma (par) y dígito (imagen) en `pares` pares de (X, y), sin aprender."""
    ok_s = ok_d = 0; conf = np.zeros((10, 10), dtype=np.int64)
    for i in range(pares):
        a, b = 2 * i, 2 * i + 1
        p1, p2 = red.p(X[a]), red.p(X[b])
        ok_s += suma_predicha(p1, p2) == int(y[a]) + int(y[b])
        for p, t in ((p1, y[a]), (p2, y[b])):
            conf[int(t), int(np.argmax(p))] += 1
    return ok_s / pares, float(np.trace(conf) / conf.sum()), conf


def correr(var, seed, n_pares, test, out, extra, cada=2500):
    cfg = VARIANTES[var]
    Xtr, ytr, Xte, yte = load_mnist()
    orden = np.random.default_rng(seed).permutation(55000)
    Xf, yf = Xtr[orden], ytr[orden]
    if test:
        Xe, ye = Xte, yte; n_eval = 5000
    else:
        Xe, ye = Xtr[55000:], ytr[55000:]; n_eval = 2500
    red = Red(cfg, seed)
    hits = []; t0 = time.time()
    for i in range(n_pares):
        x1, x2 = Xf[2 * i], Xf[2 * i + 1]
        s = int(yf[2 * i]) + int(yf[2 * i + 1])
        p1, p2 = red.p(x1), red.p(x2)
        hits.append(suma_predicha(p1, p2) == s)              # test-then-train
        _, q1, q2 = posterior_suma(p1, p2, s, cfg["gamma"])
        red.aprender(x1, q1, cfg.get("duro", False))
        red.aprender(x2, q2, cfg.get("duro", False))
        if (i + 1) % cada == 0:
            out("suma_mnist", var, f"curva suma prequential par={i + 1}", float(np.mean(hits[-cada:])), "", "")
    out("suma_mnist", var, "segundos entrenamiento", round(time.time() - t0, 1), "", f"{n_pares} pares")
    acc_s, acc_d, conf = evaluar(red, Xe, ye, n_eval)
    nombre = "test" if test else "validación"
    out("suma_mnist", var, f"exactitud suma ({nombre})", acc_s, 0.972 if test else "", "DeepProbLog 30k pares: 0.972")
    out("suma_mnist", var, f"exactitud dígito ({nombre})", acc_d, 0.95, "criterio del proyecto")
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-conf)
    out("suma_mnist", var, f"pureza dígito ({nombre})", float(conf[r, c].sum() / conf.sum()), "", "mejor biyección")
    out("suma_mnist", var, "neuronas", red.net.stats()["neurons"], "", "")
    extra[var] = dict(confusion=conf.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variante", default="base")
    ap.add_argument("--pares", type=int, default=27500)
    ap.add_argument("--test", action="store_true", help="evaluar en test (sólo corridas finales)")
    a = ap.parse_args()
    tag = "" if a.test else "val"
    nombre = f"integracion3_suma_mnist{tag}_{a.variante}_p{a.pares}_seed{a.seed}"
    log = CsvLog(nombre, HEADER)
    extra: dict = {}
    correr(a.variante, a.seed, a.pares, a.test,
           lambda *r: log.row(r[0] + tag, r[1], a.seed, *r[2:]), extra)
    log.close()
    (RESULTS / f"{nombre}.json").write_text(json.dumps(extra))


if __name__ == "__main__":
    main()
