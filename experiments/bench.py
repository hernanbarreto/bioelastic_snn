"""Banco de latencia y throughput por componente (para comparar antes/después).

Cada caso se repite y se informa el *mínimo* del tiempo de reloj. Latencia = tiempo por
respuesta (un forward, un paso de entrenamiento, una respuesta del Thinker); throughput
= su inversa, en respuestas por segundo y por proceso. Con la máquina compartida son
cotas superiores y varían entre corridas; para comparar versiones usar bench_ab.py
(tiempo de CPU, versiones alternadas en un mismo proceso, con control A/A).
Incluye una referencia convencional (MLP 784-256-10 con backprop) en el mismo núcleo.

  python experiments/bench.py [etiqueta] [--workers N]

Escribe results/bench_<etiqueta>.csv. Usa MNIST, FSDD y tiny-shakespeare de data/.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from common import CsvLog
from bioelastic import AudioEncoder, BioElasticNet, Cortex, ElasticLayer
from bioelastic.datasets import bionet_stream, load_fsdd, load_mnist, load_text
from bioelastic.thinker import Thinker
from exp_vision import CONFIGS, encoder

REPEATS = 3


def best(run, repeats=REPEATS):
    """Mínimo de `repeats` corridas; `run()` devuelve (segundos, llamadas)."""
    out = [run() for _ in range(repeats)]
    secs = min(s for s, _ in out)
    return secs, out[0][1]


def timed(fn, items):
    t0 = time.perf_counter()
    for it in items:
        fn(it)
    return time.perf_counter() - t0, len(items)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "now"
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    log = CsvLog(f"bench_{label}", ["component", "case", "calls", "us_per_call", "calls_per_s",
                                    "macs_per_call"])

    def row(component, case, res, macs=""):
        secs, calls = res
        log.row(component, case, calls, secs / calls * 1e6, calls / secs, macs)

    rng = np.random.default_rng(0)
    Xtr, ytr, Xte, yte = load_mnist()
    E = encoder()
    X = np.stack([E(Xtr[i]) for i in range(3000)])
    Y = ytr[:3000].astype(int)
    Xt = np.stack([E(Xte[i]) for i in range(300)])

    # ------------------------------------------------------------ columna elástica
    # densa: prototipos de imagen (fan-in 256 sobre 784 píxeles), 2000 neuronas
    L = ElasticLayer(784, fan_in=256)
    vols = []
    for x in X[:2300]:
        idx = np.flatnonzero(x.ravel() > 0); vols.append((idx, x.ravel()[idx]))
    for v in vols[:2000]:
        L.recruit(L.forward(*v)) if L.n < 2000 else None
    row("ElasticLayer.forward", f"densa n={L.n} fan_in=256 (imagen)",
        best(lambda: timed(lambda v: L.forward(*v), vols[2000:2300])))

    # dispersa: contexto de texto (8 bytes x 256, fan-in 16), 20000 neuronas
    text = np.frombuffer(load_text()[:40000], dtype=np.uint8)
    T = ElasticLayer(256 * 8, fan_in=16)
    ctx = []
    for t in range(8, len(text)):
        idx = np.array([int(text[t - lag]) + 256 * lag for lag in range(8)])
        ctx.append((idx, (1.0 - 0.1 * np.arange(8)).astype(np.float32)))
    for v in ctx[:20000]:
        T.recruit(_volley(v))
    row("ElasticLayer.forward", f"dispersa n={T.n} fan_in=16 (texto, 8 activas)",
        best(lambda: timed(lambda v: T.forward(*v), ctx[20000:20500])))

    # columna de exp_scaling: 1000 neuronas, fan-in 32, 32 activas de 1024
    C = ElasticLayer(1024, fan_in=32, capacity=1000)
    start = rng.integers(0, 1024, size=(1000, 1))
    C.pre[:] = (start + rng.permutation(1024)[None, :32]) % 1024
    w = rng.random((1000, 32), dtype=np.float32); C.w[:] = w / np.linalg.norm(w, axis=1, keepdims=True)
    C.fill[:] = 32; C.n = 1000
    cv = [(np.sort(rng.choice(1024, 32, replace=False)), rng.random(32, dtype=np.float32) * 0.9 + 0.1)
          for _ in range(256)]
    row("ElasticLayer.forward", "columna 1000x32 (exp_scaling)",
        best(lambda: timed(lambda v: C.forward(*v), cv)))

    # ------------------------------------------------------------ área de asociación
    gen = bionet_stream(n_classes=10, seed=11)
    stream = [next(gen) for _ in range(2000)]

    def net_steps():
        net = BioElasticNet(256, 10)
        return timed(lambda xy: net.step(*xy), stream)
    row("BioElasticNet.step", "bionet 256->10, 2000 pasos desde cero", best(net_steps))

    def text_steps():
        net = BioElasticNet(256, 256, context=8, delay=0.1, layer=dict(max_neurons=30000, fan_in=16))
        def one(t):
            net.forward(int(text[t])); net.teach(int(text[t + 1]))
        return timed(one, range(6000))
    row("BioElasticNet.step", "texto 8x256 (exp_text), 6000 bytes desde cero", best(text_steps, 2))

    recs, sr = load_fsdd()
    enc = AudioEncoder(sr=sr, n_bands=32, hop_ms=20, dyn_range_db=35)
    utt = [(list(enc(r[0])), r[1]) for r in recs[:400]]

    def seq_steps():
        net = BioElasticNet(32, 10, context=8, delay=0.05)
        return timed(lambda u: net.run_sequence(u[0], u[1], decide="accumulate"), utt)
    row("BioElasticNet.run_sequence", "FSDD stream, 400 palabras desde cero", best(seq_steps, 2))
    row("AudioEncoder", "FSDD, 400 grabaciones", best(lambda: timed(lambda r: enc(r[0]), recs[:400])))

    # ------------------------------------------------------------ jerarquía (visión)
    tmp = Path(tempfile.mkdtemp())
    for model in ("depth1", "depth2", "depth3"):
        kw = dict(CONFIGS[model])
        if workers > 1:
            kw["workers"] = workers
        n_train = 1500

        def train():
            net = Cortex((28, 28, 1), 10, **kw)
            return timed(lambda i: net.step(X[i], int(Y[i])), range(n_train))
        row(f"Cortex.step {model}", f"MNIST, {n_train} pasos desde cero", best(train, 2))
        net = Cortex((28, 28, 1), 10, **kw)
        for i in range(n_train):
            net.step(X[i], int(Y[i]))
        net.save(tmp / f"{model}.npz")
        cost = net.cost() if hasattr(net, "cost") else {}
        row(f"Cortex.predict {model}", f"MNIST, red entrenada con {n_train}",
            best(lambda: timed(net.predict, Xt)), cost.get("macs_per_inference", ""))

        def more():
            twin = Cortex.load(tmp / f"{model}.npz")
            if workers > 1:
                twin.workers = workers
            return timed(lambda i: twin.step(X[i], int(Y[i])), range(n_train, n_train + 300))
        row(f"Cortex.step {model}", f"MNIST, 300 pasos sobre red entrenada", best(more),
            cost.get("macs_per_training_step", ""))
        if model == "depth3":
            for k, a in enumerate(net.areas):
                xin = X[2900]
                codes = [xin.reshape(28, 28, 1)]
                for b in net.areas[:k]:
                    codes.append(b.encode(codes[-1])); b._cache = None
                row(f"FeatureArea.encode V{k + 1}", f"rasgos={int(a.alive.sum())} entrada {a.in_shape}",
                    best(lambda: timed(lambda _: a.encode(codes[-1]), range(200))))
                a._cache = None

    # ------------------------------------------------------------ referencia convencional
    # MLP 784-256-10: la referencia con backprop de exp_vision (203 k sinapsis), en NumPy
    # float32 y en el mismo núcleo. Mismo costo con pesos entrenados o al azar.
    W1 = (rng.standard_normal((784, 256)) * 0.05).astype(np.float32); b1 = np.zeros(256, np.float32)
    W2 = (rng.standard_normal((256, 10)) * 0.05).astype(np.float32); b2 = np.zeros(10, np.float32)
    mlp_macs = 784 * 256 + 256 * 10

    def mlp(x):
        h = np.maximum(x.reshape(-1) @ W1 + b1, 0.0)
        return int(np.argmax(h @ W2 + b2))

    def mlp_step(i):                                          # SGD de un ejemplo
        x = X[i].reshape(-1)
        a = x @ W1 + b1; h = np.maximum(a, 0.0); z = h @ W2 + b2
        p = np.exp(z - z.max()); p /= p.sum(); p[Y[i]] -= 1.0
        gh = (W2 @ p) * (a > 0)
        W2[:] -= 0.01 * np.outer(h, p); b2[:] -= 0.01 * p
        W1[:] -= 0.01 * np.outer(x, gh); b1[:] -= 0.01 * gh
    Xb = Xt.reshape(len(Xt), -1)
    row("MLP 784-256-10 (backprop)", "predict, 1 imagen por llamada", best(lambda: timed(mlp, Xt)), mlp_macs)
    row("MLP 784-256-10 (backprop)", "predict en lote de 300 (por imagen)",
        best(lambda: (timed(lambda _: np.argmax(np.maximum(Xb @ W1 + b1, 0) @ W2 + b2, axis=1), [0])[0],
                      len(Xt))), mlp_macs)
    row("MLP 784-256-10 (backprop)", "paso SGD de 1 ejemplo", best(lambda: timed(mlp_step, range(1500))),
        3 * mlp_macs)

    # ------------------------------------------------------------ razonamiento
    def r1_like():
        rr = np.random.default_rng(0)
        net = Cortex(17, 8, heads=[dict(context=12, delay=0.05), dict(tau=16.0)])
        def ep(_):
            keys = rr.permutation(8)[:4]; vals = rr.integers(8, size=4); q = int(rr.integers(4))
            seq = [t for k, v in zip(keys, vals) for t in (int(k), 8 + int(v))] + [16, int(keys[q])]
            net.reset_context()
            for s in seq[:-1]:
                net.forward(s)
            net.step(seq[-1], int(vals[q]))
        return timed(ep, range(800))
    row("Cortex simbólico", "R1 ligadura, 800 episodios (10 forward c/u)", best(r1_like, 2))

    def chains():
        rr = np.random.default_rng(1)
        net = Thinker(64, memory=True, seed=1)
        def ep(_):
            hops = int(rr.integers(1, 3)); syms = rr.permutation(64); chain = syms[:hops + 1]
            pairs = [(int(chain[i]), int(chain[i + 1])) for i in range(hops)]
            rest = syms[hops + 1:]
            pairs += [(int(rest[2 * i]), int(rest[2 * i + 1])) for i in range(4)]
            net.reset()
            for i in rr.permutation(len(pairs)):
                a, b = pairs[i]; net.observe(a); net.observe(b); net.observe(None)
            net.answer(int(chain[0]), explore=True); net.feedback(target=int(chain[-1]))
        return timed(ep, range(800))
    row("Thinker episodio", "T2 cadenas (memoria), 800 episodios", best(chains, 2))

    def compose():
        rr = np.random.default_rng(3)
        f = np.roll(np.arange(16), 5)
        net = Thinker(16, n_cues=2, speak=True, seed=3)
        def ep(i):
            cue = 0 if i < 200 else int(rr.integers(2))
            x = int(rr.integers(16)) if cue == 0 else int(rr.integers(8))
            y = int(f[x]) if cue == 0 else int(f[f[x]])
            net.reset(); net.answer(x, cue=cue, explore=True); net.feedback(target=y)
        return timed(ep, range(800))
    row("Thinker episodio", "T4 composición (habla interna), 800 episodios", best(compose, 2))
    log.close()


def _volley(v):
    """Volea sin forward (para poblar rápido una columna)."""
    from bioelastic.layer import Volley
    idx, val = v
    return Volley(idx, val, float(np.sqrt(val @ val)), np.empty(0, np.int64), np.empty(0),
                  np.empty(0), True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
