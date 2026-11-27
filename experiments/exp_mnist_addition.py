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
    # ACh: gamma sigue a la fiabilidad medida de la percepción (cuán a menudo el par más
    # probable es consistente con la suma) frente a la que declara (contra la
    # auto-confirmación que sostiene los atajos de desplazamientos compensados)
    "ach": dict(config="depth2", gamma=1.0, h=0.0, ach=dict(rate=0.01, lo=0.05, hi=1.0)),
    # EM estocástico: el objetivo es una muestra de la marginal (sale de óptimos locales)
    "muestreo": dict(config="depth2", gamma=1.0, h=0.0, muestreo=True),
    "ach+muestreo": dict(config="depth2", gamma=1.0, h=0.0, muestreo=True, ach=dict(rate=0.01, lo=0.05, hi=1.0)),
    # fiabilidad: gamma = r^2, con r = cuán a menudo el par más probable acertó la suma (por
    # encima del azar). Mientras la percepción no demuestra que acierta, su evidencia casi no
    # pesa en el posterior (no se compromete temprano); un error sostenido en la suma (el
    # código está mal) le baja el peso y reabre el código en vez de consolidarlo. Exponente
    # y piso elegidos en el modelo tabular de prototipos (sale de la trampa d+-1 en 6/6).
    "fiab": dict(config="depth2", gamma=1.0, h=0.0, ach=dict(modo="potencia", k=2.0, rate=0.01, lo=0.02, hi=1.0)),
}
# reapertura: los primeros 2500 pares sin la compuerta (gamma = 1: con la semilla 2 el código
# cae en el atajo) y después con ella: ¿el error sostenido en la suma reabre un código ya
# consolidado?
VARIANTES["fiab_tardia"] = dict(VARIANTES["fiab"], desde=2500)
# + reconsolidación: mientras la fiabilidad medida es baja, lo consolidado se libera
# (Cortex.labilize) y la tasa sube: tasa x (1 + 3 (1 - gamma)), liberación 0.01 (1 - gamma) por par
VARIANTES["fiab_labil"] = dict(VARIANTES["fiab"], labil=dict(tasa=3.0, libera=0.01))
VARIANTES["fiab_tardia_labil"] = dict(VARIANTES["fiab_labil"], desde=2500)
AZAR_PAR = 0.067        # P(dos dígitos al azar sumen lo observado) = sum_y P(y)^2


class Fiabilidad:
    """ACh: incertidumbre esperada del canal perceptivo. Lleva, con promedios móviles, la
    frecuencia con que el par de dígitos más probable es consistente con la suma
    observada (r_par: evidencia que no pasa por el prior de la percepción) y la certeza
    que la percepción declara; gamma es el exponente que lleva una a la otra
    (`bioelastic.posterior.gamma_for`) y descuenta la evidencia perceptiva en el
    posterior cuando la percepción es menos fiable de lo que cree."""

    def __init__(self, rate=0.01, lo=0.05, hi=1.0, modo="calibracion", k=2.0):
        self.rate, self.lo, self.hi = float(rate), float(lo), float(hi)
        self.modo, self.k = modo, float(k)
        self.r = None; self.d = None; self.gamma = 1.0 if modo == "calibracion" else self.lo
        from bioelastic.posterior import ReliabilityGate
        self.compuerta = ReliabilityGate(AZAR_PAR, self.k, self.rate, self.lo, self.hi)   # modo "potencia"

    def actualizar(self, p1, p2, y):
        from bioelastic.posterior import gamma_for
        ok = float(int(np.argmax(p1)) + int(np.argmax(p2)) == int(y))
        dec = 0.5 * (float(p1.max()) + float(p2.max()))
        if self.r is None:
            self.r, self.d = (ok, dec) if self.modo == "calibracion" else (AZAR_PAR, dec)
        if self.modo != "calibracion":                   # `bioelastic.posterior.ReliabilityGate`
            g = self.compuerta.update(bool(ok))
            self.r, self.gamma = self.compuerta.r, self.compuerta.gamma
            return g
        self.r += self.rate * (ok - self.r); self.d += self.rate * (dec - self.d)
        self.gamma = float(np.clip(gamma_for(self.d, max(self.r, 1e-3) ** 0.5), self.lo, self.hi))
        return self.gamma


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

    def aprender(self, img, q, duro=False, gain=1.0):
        self.net.forward(self.E(img))
        if duro:
            self.net.teach(int(np.argmax(q)), gain=gain)
        else:
            self.net.teach(soft=q, gain=gain)


def evaluar(red, X, y, pares, meta=None):
    """Suma (par) y dígito (imagen) en `pares` pares de (X, y), sin aprender. Con `meta`
    (dict) guarda confianza y acierto de cada dígito y de cada suma (metacognición)."""
    ok_s = ok_d = 0; conf = np.zeros((10, 10), dtype=np.int64)
    for i in range(pares):
        a, b = 2 * i, 2 * i + 1
        p1, p2 = red.p(X[a]), red.p(X[b])
        s_hat = suma_predicha(p1, p2)
        ok_s += s_hat == int(y[a]) + int(y[b])
        for p, t in ((p1, y[a]), (p2, y[b])):
            conf[int(t), int(np.argmax(p))] += 1
            if meta is not None:
                meta.setdefault("pd", []).append(float(p.max())); meta.setdefault("okd", []).append(int(np.argmax(p)) == int(t))
        if meta is not None:
            J = p1[:, None] * p2[None, :]
            ps = np.bincount(REGLA.ravel(), weights=J.ravel(), minlength=19)
            meta.setdefault("ps", []).append(float(ps[s_hat])); meta.setdefault("oks", []).append(s_hat == int(y[a]) + int(y[b]))
    return ok_s / pares, float(np.trace(conf) / conf.sum()), conf


def correr(var, seed, n_pares, test, out, extra, cada=2500, diag=0):
    """`diag` > 0: cada `diag` pares mide, sobre 300 imágenes fijas de validación, la
    fracción en la identidad y en d+-1, la certeza declarada y cuántas sumas ancla (0 y 18:
    una sola explicación) y casi ancla (1 y 17) hubo: cuándo se compromete el código."""
    cfg = VARIANTES[var]
    Xtr, ytr, Xte, yte = load_mnist()
    orden = np.random.default_rng(seed).permutation(55000)
    Xf, yf = Xtr[orden], ytr[orden]
    if test:
        Xe, ye = Xte, yte; n_eval = 5000
    else:
        Xe, ye = Xtr[55000:], ytr[55000:]; n_eval = 2500
    red = Red(cfg, seed)
    fia = Fiabilidad(**cfg["ach"]) if cfg.get("ach") else None
    rng = np.random.default_rng(30_000 + seed)
    hits = []; gam = []; t0 = time.time()
    anclas = [0, 0]
    Xd, yd = Xtr[55000:55300], ytr[55000:55300].astype(int)
    for i in range(n_pares):
        x1, x2 = Xf[2 * i], Xf[2 * i + 1]
        s = int(yf[2 * i]) + int(yf[2 * i + 1])
        p1, p2 = red.p(x1), red.p(x2)
        hits.append(suma_predicha(p1, p2) == s)              # test-then-train
        g = fia.actualizar(p1, p2, s) if fia is not None else cfg["gamma"]
        if i < cfg.get("desde", 0):
            g = 1.0
        gam.append(g)
        _, q1, q2 = posterior_suma(p1, p2, s, g)
        if cfg.get("muestreo"):
            q1 = np.eye(10)[int(rng.choice(10, p=q1 / q1.sum()))]
            q2 = np.eye(10)[int(rng.choice(10, p=q2 / q2.sum()))]
        gan = 1.0
        if cfg.get("labil") and i >= cfg.get("desde", 0):
            lb = cfg["labil"]
            gan = 1.0 + lb["tasa"] * (1.0 - g)
            red.net.labilize(lb["libera"] * (1.0 - g))
        red.aprender(x1, q1, cfg.get("duro", False), gan)
        red.aprender(x2, q2, cfg.get("duro", False), gan)
        if diag:
            anclas[0] += s in (0, 18); anclas[1] += s in (1, 17)
            if (i + 1) % diag == 0:
                ps = [red.p(x) for x in Xd]
                arg = np.array([int(np.argmax(x)) for x in ps])
                out("suma_mnist", var, f"diag identidad par={i + 1}", float(np.mean(arg == yd)), "", "")
                out("suma_mnist", var, f"diag vecinos d+-1 par={i + 1}", float(np.mean(np.abs(arg - yd) == 1)), "", "")
                out("suma_mnist", var, f"diag certeza declarada par={i + 1}", float(np.mean([x.max() for x in ps])), "", "")
                out("suma_mnist", var, f"diag suma prequential par={i + 1}", float(np.mean(hits[-diag:])), "", "")
                out("suma_mnist", var, f"diag gamma par={i + 1}", float(np.mean(gam[-diag:])), "", "")
                out("suma_mnist", var, f"diag anclas (0, 18) acumuladas par={i + 1}", anclas[0], anclas[1], "casi anclas (1, 17) en referencia")
        if (i + 1) % cada == 0:
            out("suma_mnist", var, f"curva suma prequential par={i + 1}", float(np.mean(hits[-cada:])), "", "")
            if fia is not None:
                out("suma_mnist", var, f"curva gamma par={i + 1}", float(np.mean(gam[-cada:])), "", "ACh")
    out("suma_mnist", var, "segundos entrenamiento", round(time.time() - t0, 1), "", f"{n_pares} pares")
    meta: dict = {}
    acc_s, acc_d, conf = evaluar(red, Xe, ye, n_eval, meta)
    nombre = "test" if test else "validación"
    from bioelastic.metacog import auroc, ece
    for k, (c_, o_) in (("dígito", ("pd", "okd")), ("suma", ("ps", "oks"))):
        cc, oo = np.array(meta[c_]), np.array(meta[o_], dtype=bool)
        out("suma_mnist", var, f"AUROC error {k} ({nombre}): 1-confianza", auroc(1 - cc, ~oo), 0.85, "criterio >= 0.85")
        out("suma_mnist", var, f"ECE {k} ({nombre})", ece(cc, oo)[0], 0.05, "criterio <= 0.05")
    out("suma_mnist", var, f"exactitud suma ({nombre})", acc_s, 0.972 if test else "", "DeepProbLog 30k pares: 0.972")
    out("suma_mnist", var, f"exactitud dígito ({nombre})", acc_d, 0.95, "criterio del proyecto")
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-conf)
    out("suma_mnist", var, f"pureza dígito ({nombre})", float(conf[r, c].sum() / conf.sum()), "", "mejor biyección")
    out("suma_mnist", var, "neuronas", red.net.stats()["neurons"], "", "")
    extra[var] = dict(confusion=conf.tolist())


def correr_n(var, seed, N, test, out, extra, cada=500, max_ej=None):
    """MNIST-addition de N cifras (protocolo de DeepProbLog y sucesores): dos números de N
    cifras, cada cifra una imagen; la etiqueta es la suma entera. Una sola pasada: cada
    imagen de train[:55000] se usa una vez (55000 // (2N) ejemplos). El razonamiento
    calcula el posterior conjunto sobre todas las cifras y los acarreos dada la suma
    (`chain_posterior` con la suma conocida: las cifras de la suma y el acarreo final) y la
    percepción aprende cada imagen con su marginal (objetivo blando). Predicción: la suma de
    los dígitos más probables. Test: t10k en grupos de 2N imágenes."""
    from bioelastic.posterior import chain_posterior, known_sum_factor
    cfg = VARIANTES[var]
    Xtr, ytr, Xte, yte = load_mnist()
    orden = np.random.default_rng(seed).permutation(55000)
    Xf, yf = Xtr[orden], ytr[orden]
    Xe, ye = (Xte, yte) if test else (Xtr[55000:], ytr[55000:])
    red = Red(cfg, seed)
    n_ej = 55000 // (2 * N) if max_ej is None else int(max_ej)
    pot = 10 ** np.arange(N)
    hits = []; t0 = time.time()
    ach = cfg.get("ach") if (cfg.get("ach") or {}).get("modo") == "potencia" else None
    r_ej = 0.0; gamma = ach["lo"] if ach else cfg["gamma"]
    for k in range(n_ej):
        idx = np.arange(2 * N * k, 2 * N * (k + 1))
        da, db = yf[idx[:N]].astype(int), yf[idx[N:]].astype(int)       # cifra menos significativa primero
        s = int(da @ pot + db @ pot)
        pa = [red.p(Xf[i]) for i in idx[:N]]; pb = [red.p(Xf[i]) for i in idx[N:]]
        pred = int(np.array([int(np.argmax(x)) for x in pa]) @ pot + np.array([int(np.argmax(x)) for x in pb]) @ pot)
        hits.append(pred == s)                                            # test-then-train
        digs = [(s // 10 ** i) % 10 for i in range(N)]
        qa, qb, _, _, _ = chain_posterior(pa, pb, [known_sum_factor(d) for d in digs], np.eye(2)[s // 10 ** N],
                                          gamma)
        if ach:                                 # fiabilidad por par de cifras: r_ej^(1/N); gamma = eso^k
            r_ej += ach["rate"] * (float(pred == s) - r_ej)
            gamma = float(np.clip(max(r_ej, 0.0) ** (ach["k"] / N), ach["lo"], ach["hi"]))
        for i, q in zip(idx, list(qa) + list(qb)):
            red.aprender(Xf[i], q, cfg.get("duro", False))
        if (k + 1) % cada == 0:
            out("suma_mnist", var, f"N={N} curva suma prequential ej={k + 1}", float(np.mean(hits[-cada:])), "", "")
    out("suma_mnist", var, f"N={N} segundos entrenamiento", round(time.time() - t0, 1), "", f"{n_ej} ejemplos")
    n_t = len(ye) // (2 * N) if test else 2500 // N
    if max_ej is not None:
        n_t = min(n_t, int(max_ej))
    ok = 0; okd = 0; nd = 0
    for k in range(n_t):
        idx = np.arange(2 * N * k, 2 * N * (k + 1))
        da, db = ye[idx[:N]].astype(int), ye[idx[N:]].astype(int)
        ps = [int(np.argmax(red.p(Xe[i]))) for i in idx]
        ok += int(np.array(ps[:N]) @ pot + np.array(ps[N:]) @ pot) == int(da @ pot + db @ pot)
        okd += sum(int(a == b) for a, b in zip(ps, ye[idx])); nd += 2 * N
    nombre = "test" if test else "validación"
    out("suma_mnist", var, f"N={N} exactitud suma ({nombre})", ok / n_t, "", f"{n_t} ejemplos")
    out("suma_mnist", var, f"N={N} exactitud dígito ({nombre})", okd / nd, "", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variante", default="base")
    ap.add_argument("--pares", type=int, default=27500)
    ap.add_argument("--test", action="store_true", help="evaluar en test (sólo corridas finales)")
    ap.add_argument("--skip-existing", action="store_true", help="no correr si la salida ya está completa")
    ap.add_argument("--cifras", type=int, default=1, help="cifras de cada sumando (1 = el par original)")
    ap.add_argument("--diag", type=int, default=0, help="diagnóstico del compromiso cada tantos pares")
    a = ap.parse_args()
    tag = "" if a.test else "val"
    nombre = f"integracion3_suma_mnist{tag}_{a.variante}_p{a.pares}_seed{a.seed}" if a.cifras == 1 else \
        f"integracion3_suma_mnist{tag}_{a.variante}_N{a.cifras}_seed{a.seed}"
    if a.diag:
        nombre = nombre.replace("suma_mnist", "suma_mnistdiag", 1)
    from exp_integration3 import completo
    if a.skip_existing and completo(RESULTS / f"{nombre}.csv"):
        print("ya está:", nombre)
        return
    log = CsvLog(nombre, HEADER)
    extra: dict = {}
    t0 = time.time()
    fn = lambda *r: log.row(r[0] + tag, r[1], a.seed, *r[2:])  # noqa: E731
    if a.cifras == 1:
        correr(a.variante, a.seed, a.pares, a.test, fn, extra, diag=a.diag)
    else:
        correr_n(a.variante, a.seed, a.cifras, a.test, fn, extra)
    log.row("suma_mnist" + tag, "total", a.seed, "segundos", round(time.time() - t0, 1), "", "")
    log.close()
    (RESULTS / f"{nombre}.json").write_text(json.dumps(extra))


if __name__ == "__main__":
    main()
