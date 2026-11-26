"""Comparación A/B en un mismo proceso: código de referencia contra código actual.

En una máquina compartida la carga cambia minuto a minuto; correr las dos versiones
alternadas (A, B, A, B, ...) en el mismo proceso, sobre las mismas redes ya entrenadas,
y quedarse con el mínimo de cada una es la comparación más justa. Se mide tiempo de CPU
del proceso (no de reloj): con otros procesos compitiendo por los núcleos, el tiempo de
reloj incluye las esperas. Además verifica que las dos versiones den las mismas respuestas.

  python experiments/bench_ab.py /ruta/al/checkout_de_referencia [rondas]

Escribe results/bench_ab.csv (latencia en us por llamada; x = antes / después).
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from common import ROOT, CsvLog


def load_package(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path / "bioelastic" / "__init__.py",
                                                  submodule_search_locations=[str(path / "bioelastic")])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    base = Path(sys.argv[1]).resolve()
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    A = load_package(base, "bioelastic_ref")
    B = load_package(ROOT, "bioelastic_now")
    from bioelastic.datasets import load_mnist, load_text
    sys.path.insert(0, str(ROOT / "experiments"))
    from exp_vision import CONFIGS, encoder

    rng = np.random.default_rng(0)
    Xtr, ytr, Xte, yte = load_mnist()
    E = encoder()
    X = np.stack([E(Xtr[i]) for i in range(1800)]); Y = ytr[:1800].astype(int)
    Xt = np.stack([E(Xte[i]) for i in range(200)])
    tmp = Path(tempfile.mkdtemp())
    cases = []                                               # (componente, caso, fábrica(mod) -> fn)

    # redes entrenadas una vez (con la versión actual: el formato de archivo es el mismo)
    for model in ("depth1", "depth2", "depth3"):
        net = B.Cortex((28, 28, 1), 10, **CONFIGS[model])
        for i in range(1500):
            net.step(X[i], int(Y[i]))
        net.save(tmp / f"{model}.npz")

        def predict(mod, model=model):
            twin = mod.Cortex.load(tmp / f"{model}.npz")
            return lambda: [twin.predict(x) for x in Xt]

        def train(mod, model=model):
            twin = mod.Cortex.load(tmp / f"{model}.npz")
            return lambda: [twin.step(X[i], int(Y[i])) for i in range(1500, 1800)]
        cases.append((f"Cortex.predict {model}", "MNIST, entrenada con 1500", predict, len(Xt), True))
        cases.append((f"Cortex.step {model}", "MNIST, 300 pasos sobre la entrenada", train, 300, False))

    vols = []
    for x in X[:2300]:
        idx = np.flatnonzero(x.ravel() > 0); vols.append((idx, x.ravel()[idx]))

    def dense_layer(mod):
        L = mod.ElasticLayer(784, fan_in=256)
        for v in vols[:1500]:
            L.recruit(L.forward(*v))
        return lambda: [L.forward(*v) for v in vols[1500:1800]]
    cases.append(("ElasticLayer.forward", "densa (imagen, fan-in 256)", dense_layer, 300, False))

    text = np.frombuffer(load_text()[:30000], dtype=np.uint8)
    ctx = [(np.array([int(text[t - lag]) + 256 * lag for lag in range(8)]),
            (1.0 - 0.1 * np.arange(8)).astype(np.float32)) for t in range(8, len(text))]

    def text_layer(mod):
        L = mod.ElasticLayer(2048, fan_in=16)
        for idx, val in ctx[:20000]:
            L.recruit(mod.Volley(idx, val, float(np.sqrt(val @ val)), None, None, None, True))
        return lambda: [L.forward(*v) for v in ctx[20000:20400]]
    cases.append(("ElasticLayer.forward", "dispersa (texto 8x256, 20000 neuronas)", text_layer, 400, False))

    def column(mod):
        r = np.random.default_rng(1)
        L = mod.ElasticLayer(1024, fan_in=32, capacity=1000)
        start = r.integers(0, 1024, size=(1000, 1))
        L.pre[:] = (start + r.permutation(1024)[None, :32]) % 1024
        w = r.random((1000, 32), dtype=np.float32); L.w[:] = w / np.linalg.norm(w, axis=1, keepdims=True)
        L.fill[:] = 32; L.n = 1000
        vv = [(np.sort(r.choice(1024, 32, replace=False)), r.random(32, dtype=np.float32) * 0.9 + 0.1)
              for _ in range(256)]
        return lambda: [L.forward(*v) for v in vv]
    cases.append(("ElasticLayer.forward", "columna 1000x32 (exp_scaling)", column, 256, False))

    def thinker(mod, kind):
        thk = importlib.import_module(mod.__name__ + ".thinker")
        def run():
            rr = np.random.default_rng(1)
            if kind == "cadenas":
                net = thk.Thinker(64, memory=True, seed=1)
                for _ in range(400):
                    hops = int(rr.integers(1, 3)); syms = rr.permutation(64); chain = syms[:hops + 1]
                    pairs = [(int(chain[i]), int(chain[i + 1])) for i in range(hops)]
                    rest = syms[hops + 1:]
                    pairs += [(int(rest[2 * i]), int(rest[2 * i + 1])) for i in range(4)]
                    net.reset()
                    for i in rr.permutation(len(pairs)):
                        a, b = pairs[i]; net.observe(a); net.observe(b); net.observe(None)
                    net.answer(int(chain[0]), explore=True); net.feedback(target=int(chain[-1]))
            else:
                f = np.roll(np.arange(16), 5)
                net = thk.Thinker(16, n_cues=2, speak=True, seed=3)
                for i in range(400):
                    cue = 0 if i < 100 else int(rr.integers(2))
                    x = int(rr.integers(16)) if cue == 0 else int(rr.integers(8))
                    y = int(f[x]) if cue == 0 else int(f[f[x]])
                    net.reset(); net.answer(x, cue=cue, explore=True); net.feedback(target=y)
            return net.decisions.copy()
        return run
    cases.append(("Thinker episodio", "T2 cadenas, 400 episodios desde cero",
                  lambda mod: thinker(mod, "cadenas"), 400, True))
    cases.append(("Thinker episodio", "T4 composición, 400 episodios desde cero",
                  lambda mod: thinker(mod, "composicion"), 400, True))

    # Control de ruido: la misma versión contra sí misma (debería dar x = 1).
    cases.append(("control A/A", "columna 1000x32, referencia contra referencia", column, 256, False))
    log = CsvLog("bench_ab", ["component", "case", "calls", "us_before", "us_after", "x", "same_output"])
    for comp, case, make, calls, check in cases:
        mods = {"A": A, "B": A if comp.startswith("control") else B}
        fns = {key: make(mod) for key, mod in mods.items()}
        best = {"A": np.inf, "B": np.inf}
        outs = {}
        for r in range(rounds):
            for key in (("A", "B") if r % 2 == 0 else ("B", "A")):   # se alterna quién va primero
                if comp.startswith("Cortex.step") or comp.startswith("Thinker"):
                    fns[key] = make(mods[key])                          # estado fresco en cada ronda
                t0 = time.process_time()
                outs[key] = fns[key]()
                best[key] = min(best[key], time.process_time() - t0)
        same = ""
        if check:
            same = bool(np.array_equal(np.asarray(outs["A"]), np.asarray(outs["B"])))
        log.row(comp, case, calls, best["A"] / calls * 1e6, best["B"] / calls * 1e6,
                best["A"] / best["B"], same)
    log.close()


if __name__ == "__main__":
    main()
