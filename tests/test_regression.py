"""Regresión bit a bit contra salidas de referencia del código previo a las optimizaciones.

Cada escenario corre una sesión corta con semillas fijas y devuelve un diccionario de
arrays: predicciones, voleas (ganadoras, tiempos, coincidencias), estructura (neuronas,
sinapsis, rasgos) y pesos. Se compara con `tests/golden/<escenario>.npz`, que se generó
con el código de referencia (commit base de la rama de rendimiento):

    python tests/test_regression.py --regen        # SÓLO desde el commit de referencia

La comparación es exacta, bit a bit, también en punto flotante. Los bits de los pesos
dependen de la ruta SIMD/BLAS de la máquina: en otra plataforma se puede relajar sólo la
parte flotante con BIOELASTIC_GOLDEN_RTOL=1e-5 (predicciones, enteros y estructura
siguen exigiéndose idénticos).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bioelastic import AudioEncoder, BioElasticNet, Cortex, ElasticLayer  # noqa: E402
from bioelastic.thinker import Thinker  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden"


# ---------------------------------------------------------------- registro de salidas
class _Rec:
    """Acumula listas de arrays de largo variable y las aplana con sus largos."""

    def __init__(self):
        self.lists: dict[str, list] = {}
        self.out: dict[str, np.ndarray] = {}

    def add(self, key, value):
        self.lists.setdefault(key, []).append(np.atleast_1d(np.asarray(value)))

    def volley(self, prefix, v):
        self.add(prefix + "winners", v.winners.astype(np.int64))
        self.add(prefix + "times", np.asarray(v.times, dtype=np.float64))
        self.add(prefix + "cos", np.asarray(v.cos))
        self.add(prefix + "novel", np.array([v.novel]))

    def put(self, key, value):
        self.out[key] = np.asarray(value)

    def layer(self, prefix, L: ElasticLayer):
        n = L.n
        self.put(prefix + "pre", L.pre[:n]); self.put(prefix + "w", L.w[:n])
        for name in L._STATS:
            self.put(prefix + name, getattr(L, name)[:n])
        self.put(prefix + "meta", np.array([L.n, L.t, L.created, L.pruned, L.evicted]))

    def done(self) -> dict:
        res = dict(self.out)
        for key, parts in self.lists.items():
            res[key] = np.concatenate(parts)
            res[key + "_len"] = np.array([len(p) for p in parts], dtype=np.int64)
        return res


def _sparse(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    idx = np.flatnonzero(x > 0)
    return idx, x[idx]


# ---------------------------------------------------------------- escenarios: columna
def scenario_layer_dense() -> dict:
    """Columna chica con homeostasis, presupuesto (desalojo), poda, recambio sináptico,
    inhibición de candidatas (max_candidates) y decisión forzada en silencio."""
    rng = np.random.default_rng(10)
    P = (rng.random((8, 60)) < 0.25).astype(np.float32)
    L = ElasticLayer(60, fan_in=12, k=3, theta=0.55, theta_plus=0.02, theta_max=0.1,
                     theta_tau=50.0, max_neurons=40, probation=60, max_candidates=5, capacity=4)
    r = _Rec()
    for t in range(600):
        x = P[rng.integers(8)] * rng.uniform(0.5, 1.0, 60) + 0.3 * rng.random(60) * (rng.random(60) < 0.1)
        idx, val = _sparse(np.clip(x, 0, 1))
        v = L.forward(idx, val)
        r.volley("", v)
        if (v.novel and t % 3) or (not v.novel and v.cos[0] < 0.97 and t % 2):
            r.add("recruit", L.recruit(v))
        elif len(v.winners):
            j = int(v.winners[0]); L.stdp(j, v, +1.0); L.credit(j)
            if len(v.winners) > 1:
                L.stdp(int(v.winners[1]), v, -0.3)
        L.tick()
        if t % 50 == 49:
            r.add("maintain", L.maintain())
        if t % 97 == 96:
            L.forget_afferents(rng.choice(60, 4, replace=False))
    r.put("thresholds", L.thresholds()); r.put("utility", L.utility())
    r.layer("L_", L)
    return r.done()


def scenario_layer_sparse() -> dict:
    """Columna grande con entrada dispersa (régimen del índice invertido): muchas neuronas,
    pocas aferentes activas, sinaptogénesis y recambio, poda y voleas vacías."""
    rng = np.random.default_rng(11)
    D, M = 4000, 60
    protos = [np.sort(rng.choice(D, 30, replace=False)) for _ in range(M)]
    L = ElasticLayer(D, fan_in=16, k=4, theta=0.6, probation=300, capacity=16)
    r = _Rec()
    for t in range(1500):
        p = protos[int(rng.integers(M))].copy()
        swap = rng.random(30) < 0.15
        p[swap] = rng.integers(D, size=int(swap.sum()))
        p = np.unique(p)
        val = rng.uniform(0.2, 1.0, len(p)).astype(np.float32)
        if t % 211 == 0:
            p, val = p[:0], val[:0]                          # volea vacía
        elif t % 157 == 0:
            p, val = p[:1], val[:1]                          # una sola aferente
        v = L.forward(p, val)
        r.volley("", v)
        L.tick()
        if v.novel:
            r.add("recruit", L.recruit(v))
        elif len(v.winners):
            j = int(v.winners[0]); L.stdp(j, v, +1.0); L.credit(j)
            if len(v.winners) > 2 and t % 4 == 0:
                L.stdp(int(v.winners[2]), v, -0.5)
        if t % 100 == 99:
            r.add("maintain", L.maintain())
    r.layer("L_", L)
    return r.done()


def scenario_layer_columns() -> dict:
    """Columnas pobladas al azar como en exp_scaling (sólo inferencia)."""
    rng = np.random.default_rng(12)
    COL, FAN, D, ACT = 300, 32, 1024, 32
    r = _Rec()
    for c in range(4):
        L = ElasticLayer(D, fan_in=FAN, capacity=COL, theta=(0.03, 0.06, 0.1, 0.2)[c])
        start = rng.integers(0, D, size=(COL, 1))
        L.pre[:] = (start + rng.permutation(D)[None, :FAN]) % D
        w = rng.random((COL, FAN), dtype=np.float32)
        L.w[:] = w / np.linalg.norm(w, axis=1, keepdims=True)
        L.fill[:] = FAN
        L.n = COL
        for _ in range(60):
            idx = np.sort(rng.choice(D, ACT, replace=False))
            val = rng.random(ACT, dtype=np.float32) * 0.9 + 0.1
            r.volley(f"c{c}_", L.forward(idx, val))
    return r.done()


# ---------------------------------------------------------------- escenarios: área de asociación
def _protos(C, D=60, on=12, seed=0):
    rng = np.random.default_rng(seed)
    P = np.zeros((C, D), dtype=np.float32)
    for c in range(C):
        P[c, rng.choice(D, on, replace=False)] = 1.0
    return P


def _sample(P, c, rng, noise=0.15):
    x = P[c] * rng.uniform(0.7, 1.0, P.shape[1]) + noise * rng.random(P.shape[1])
    return np.clip(x, 0, 1).astype(np.float32)


def scenario_net() -> dict:
    """BioElasticNet: supervisado con línea de retardos, recompensa con exploración,
    no supervisado, peek con confianza, traza lenta y secuencias."""
    rng = np.random.default_rng(13)
    P = _protos(6, seed=13)
    r = _Rec()
    net = BioElasticNet(60, 6, context=2, delay=0.2, layer=dict(probation=200, k=3),
                        maintain_every=50, seed=3)
    for _ in range(500):
        c = int(rng.integers(6))
        r.add("sup_pred", net.step(_sample(P, c, rng), c))
        r.add("sup_V", net._last.V)
    for _ in range(300):
        c = int(rng.integers(6))
        a = net.forward(_sample(P, c, rng), explore=True)
        r.add("rl_act", a)
        net.reinforce(1.0 if a == c else -1.0)
    for _ in range(100):
        net.observe(_sample(P, int(rng.integers(6)), rng, noise=0.05))
    for c in range(6):
        out, sure = net.peek(_sample(P, c, rng), confidence=True)
        r.add("peek", np.array([out, int(sure)]))
    st = net.state()
    for k, v in st.items():
        if k != "config":
            r.put("net_" + k, v)

    tr = BioElasticNet(8, 3, tau=4.0, seed=5)
    for _ in range(300):
        key = int(rng.integers(3))
        tr.reset_context(); tr.forward(key)
        for s in rng.integers(3, 7, size=int(rng.integers(2, 6))):
            tr.forward(int(s))
        r.add("trace_pred", tr.step(7, key))
    r.layer("trace_L_", tr.layer)

    for decide in ("end", "accumulate"):
        sq = BioElasticNet(8, 2, context=3, delay=0.2, seed=6)
        f = np.zeros((2, 8), dtype=np.float32); f[0, :4] = 1.0; f[1, 4:] = 1.0
        for _ in range(150):
            c = int(rng.integers(2))
            order = (0, 1, 0) if c == 0 else (1, 0, 0)
            frames = [f[i] * rng.uniform(0.8, 1.0, 8).astype(np.float32) for i in order]
            r.add(f"seq_{decide}", sq.run_sequence(frames, c, decide=decide))
        r.layer(f"seq_{decide}_L_", sq.layer)
        r.put(f"seq_{decide}_RW", sq.readout.W[:, :sq.readout.n])
    return r.done()


# ---------------------------------------------------------------- escenarios: jerarquía
def _img(rng, c, size=12):
    img = 0.03 * rng.random((size, size, 1)).astype(np.float32)
    y, x = (int(v) for v in rng.integers(0, size - 5, size=2))
    if c == 0:
        img[y, x:x + 5, 0] = 1.0
    elif c == 1:
        img[y:y + 5, x, 0] = 1.0
    elif c == 2:
        img[np.arange(5) + y, np.arange(5) + x, 0] = 1.0
    else:
        img[y:y + 3, x:x + 3, 0] = rng.uniform(0.6, 1.0, (3, 3))
    if rng.random() < 0.1:
        img[:] = 0.0                                          # lámina vacía
    return img


def _cortex_record(r, net, prefix):
    st = net.stats()
    r.put(prefix + "units", np.array([st["units"], st["synapses"], st["neurons"], st["features"],
                                      st["replays"]]))
    for i, a in enumerate(net.areas):
        for k, v in a.state().items():
            r.put(f"{prefix}a{i}_{k}", v)
    for i, h in enumerate(net.heads):
        for k, v in h.state().items():
            if k != "config":
                r.put(f"{prefix}h{i}_{k}", v)
    if net.proj is not None:
        r.put(prefix + "proj_W", net.proj.W); r.put(prefix + "proj_count", net.proj.count)


def scenario_cortex_deep() -> dict:
    """Dos áreas con muerte y reuso de rasgos, tres áreas de asociación (input, top, all
    con traza), proyección directa, repaso; luego recompensa con exploración."""
    rng = np.random.default_rng(14)
    net = Cortex((12, 12, 1), 4,
                 areas=[dict(rf=3, pool=2, max_features=12, theta=0.8, probation=30,
                             survival_hits=25, maintain_every=20),
                        dict(rf=2, pool=1, max_features=10, theta=0.5, floor=0.2, probation=60,
                             survival_hits=3, maintain_every=30)],
                 heads=[dict(source="input"), dict(source="top", layer=dict(theta=0.4)),
                        dict(source="all", tau=3.0)],
                 projection=dict(source="all"), replay=0.7, joint=False, seed=4)
    r = _Rec()
    for _ in range(250):
        c = int(rng.integers(4))
        r.add("pred", net.step(_img(rng, c), c))
        r.add("V", net._V)
    for _ in range(80):
        c = int(rng.integers(4))
        a = net.forward(_img(rng, c), explore=True)
        r.add("rl_act", a)
        net.reinforce(1.0 if a == c else -1.0)
    for _ in range(30):
        r.add("test", net.predict(_img(rng, int(rng.integers(4)))))
    _cortex_record(r, net, "")
    return r.done()


def scenario_cortex_wide() -> dict:
    """Lámina de 2 canales, campo receptivo y paso rectangulares, pooling con recorte,
    floor 0, fuente por índice, decisión conjunta y aprendizaje no supervisado."""
    rng = np.random.default_rng(15)
    net = Cortex((10, 10, 2), 3,
                 areas=[dict(rf=(3, 2), stride=(1, 2), pool=(3, 1), max_features=8, theta=0.85,
                             floor=0.0, k=3)],
                 heads=[dict(source=0), dict(source="input", context=2)],
                 projection=dict(source="top"), joint=True, seed=7)
    r = _Rec()
    for t in range(200):
        c = int(rng.integers(3))
        x = np.concatenate([_img(rng, c, 10), _img(rng, (c + 1) % 3, 10)], axis=2)
        if t % 7 == 0:
            net.observe(x)
        else:
            r.add("pred", net.step(x, c))
            r.add("V", net._V)
    _cortex_record(r, net, "")
    return r.done()


def scenario_area_floor() -> dict:
    """Áreas sueltas con floor negativo (código no nulo fuera de las posiciones válidas),
    floor 0 y k mayor que el diccionario."""
    from bioelastic import FeatureArea
    rng = np.random.default_rng(18)
    r = _Rec()
    for i, (floor, k) in enumerate(((-0.2, 2), (0.0, 1), (0.4, 50))):
        a = FeatureArea((9, 9, 2), rf=3, pool=2, max_features=8, theta=0.8, floor=floor, k=k,
                        probation=10, survival_hits=3, maintain_every=5)
        for t in range(120):
            x = (rng.random((9, 9, 2)) * (rng.random((9, 9, 2)) < 0.3)).astype(np.float32)
            if t % 13 == 0:
                x[:] = 0.0
            r.add(f"a{i}_code", a.encode(x).ravel())
            a.learn()
        for key, v in a.state().items():
            r.put(f"a{i}_{key}", v)
    return r.done()


# ---------------------------------------------------------------- escenarios: pensamiento
def scenario_thinker() -> dict:
    r = _Rec()
    rng = np.random.default_rng(16)
    net = Thinker(16, memory=True, seed=0)
    for _ in range(150):
        keys = rng.permutation(8)[:5]; vals = 8 + rng.integers(8, size=4)
        net.reset()
        for k, v in zip(keys[:4], vals):
            net.observe(int(k)); net.observe(int(v)); net.observe(None)
        bound = rng.random() < 0.5
        q = int(keys[rng.integers(4)]) if bound else int(keys[4])
        y = int(vals[list(keys[:4]).index(q)]) if bound else 8 + (q * 3) % 8
        r.add("mem_out", net.answer(q, explore=True)); net.feedback(target=y)

    f = np.roll(np.arange(16), 5)
    sp = Thinker(16, n_cues=2, speak=True, seed=3)
    for i in range(300):
        cue = 0 if i < 100 else int(rng.integers(2))
        x = int(rng.integers(16)) if cue == 0 else int(rng.integers(8))
        y = int(f[x]) if cue == 0 else int(f[f[x]])
        sp.reset(); r.add("speak_out", sp.answer(x, cue=cue, explore=True)); sp.feedback(target=y)

    ad = Thinker(5, registers=1, seed=2)
    for _ in range(150):
        n = int(rng.integers(1, 5))
        a = rng.integers(2, size=n); b = rng.integers(2, size=n)
        ad.reset(); carry = 0
        for i in range(n + 1):
            if i < n:
                s = int(a[i]) + int(b[i]) + carry
                digit, carry = s % 2, s // 2
                out = ad.answer((int(a[i]), 2 + int(b[i])), explore=True)
            else:
                digit = carry; out = ad.answer(4, explore=True)
            ad.feedback(target=digit)
            r.add("add_out", out)
    for name, t in (("mem", net), ("speak", sp), ("add", ad)):
        r.put(f"{name}_Wc", t.Wc); r.put(f"{name}_Wr", t.Wr); r.put(f"{name}_wv", t.wv)
        r.put(f"{name}_dec", t.decisions); r.put(f"{name}_blame", t.blame)
        r.layer(f"{name}_say_", t.say.layer)
        r.put(f"{name}_say_RW", t.say.readout.W[:, :t.say.readout.n])
    return r.done()


# ---------------------------------------------------------------- escenarios: codificadores
def scenario_audio() -> dict:
    sr = 8000; t = np.arange(sr // 2) / sr
    rng = np.random.default_rng(17)
    wave = (np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 1300 * t)
            + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
    wave = np.concatenate([np.zeros(800, np.float32), wave, 1e-4 * np.ones(600, np.float32)])
    return dict(plain=AudioEncoder(sr=sr, n_bands=24)(wave),
                frames=AudioEncoder(sr=sr, n_bands=32, n_frames=16, dyn_range_db=35)(wave),
                short=AudioEncoder(sr=sr)(wave[1000:1100]))


SCENARIOS = {name[len("scenario_"):]: fn for name, fn in globals().items()
             if name.startswith("scenario_")}


# ---------------------------------------------------------------- comparación
def _compare(name: str, got: dict):
    path = GOLDEN / f"{name}.npz"
    assert path.exists(), f"falta {path}: generarlo desde el commit de referencia"
    rtol = float(os.environ.get("BIOELASTIC_GOLDEN_RTOL", "0"))
    with np.load(path) as ref:
        assert sorted(ref.files) == sorted(got), name
        for key in ref.files:
            a, b = ref[key], np.asarray(got[key])
            assert a.dtype == b.dtype and a.shape == b.shape, (name, key, a.dtype, b.dtype, a.shape, b.shape)
            if rtol and a.dtype.kind == "f":
                np.testing.assert_allclose(b, a, rtol=rtol, atol=rtol, err_msg=f"{name}:{key}")
            else:
                np.testing.assert_array_equal(b, a, err_msg=f"{name}:{key}")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_identical_to_reference(name):
    _compare(name, SCENARIOS[name]())


@pytest.mark.parametrize("name", ["layer_dense", "layer_sparse", "layer_columns", "net", "cortex_deep",
                                  "cortex_wide", "thinker"])
def test_identical_with_inverted_index_forced(name, monkeypatch):
    """Con el índice invertido forzado en toda columna, las salidas siguen idénticas."""
    for attr, v in (("SPARSE_MIN_NEURONS", 1), ("SPARSE_MIN_SYNAPSES", 1),
                    ("SPARSE_MAX_ACTIVE", 1.0), ("SPARSE_MAX_TOUCHED", np.inf)):
        monkeypatch.setattr(ElasticLayer, attr, v)
    calls = []
    orig = ElasticLayer._sparse_pass
    monkeypatch.setattr(ElasticLayer, "_sparse_pass",
                        lambda self, *a: calls.append(1) or orig(self, *a))
    _compare(name, SCENARIOS[name]())
    assert calls                                          # la ruta dispersa se usó de verdad




@pytest.mark.parametrize("name", ["cortex_deep", "cortex_wide"])
def test_identical_with_parallel_workers(name, monkeypatch):
    """Con dos hilos (áreas de asociación sobre la entrada en paralelo con la jerarquía),
    las salidas siguen idénticas."""
    init = Cortex.__init__
    used = []

    def init2(self, *a, **kw):
        init(self, *a, **kw)
        self.workers = 2
        used.append(self)
    monkeypatch.setattr(Cortex, "__init__", init2)
    _compare(name, SCENARIOS[name]())
    assert used and all(c._pool is not None for c in used)


if __name__ == "__main__":
    if "--regen" in sys.argv:
        GOLDEN.mkdir(exist_ok=True)
        for name, fn in SCENARIOS.items():
            if "--only" in sys.argv and name not in sys.argv:
                continue
            np.savez_compressed(GOLDEN / f"{name}.npz", **fn())
            print("golden", name)
