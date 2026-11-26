import numpy as np

from bioelastic.metalearning import GradedMemory, MetaController, MetaThinker
from bioelastic.thinker import Thinker


def _bind(m, pairs):
    for k, v in pairs:
        m.bind((k,)); m.bind((v,)); m.bind(())


def test_graded_memory_rate_one_keeps_only_the_last_binding():
    m = GradedMemory(10, rate=1.0)
    _bind(m, [(1, 5), (1, 6), (2, 7)])
    idx, val = m.recall((1,))
    assert list(idx) == [6] and val[0] == 1.0
    assert list(m.recall((2,))[0]) == [7]


def test_graded_memory_low_rate_integrates_the_recent_majority():
    for rate, want in ((0.2, 5), (1.0, 6)):
        m = GradedMemory(10, rate=rate)
        _bind(m, [(1, 5), (1, 5), (1, 5), (1, 6)])
        idx, val = m.recall((1,))
        assert int(idx[np.argmax(val)]) == want
    m = GradedMemory(10, rate=0.2)
    _bind(m, [(1, 5), (1, 6)])
    assert len(m.recall((1,))[0]) == 2                  # con tasa baja quedan las dos (empate roto por fuerza)


def test_graded_memory_forgets():
    m = GradedMemory(10, rate=1.0, forget=0.5)
    _bind(m, [(1, 5)])
    assert list(m.recall((1,))[0]) == [5]
    for _ in range(4):
        m.bind(())                                       # 1 * 0.5^5 < piso: se olvida
    assert len(m.recall((1,))[0]) == 0
    assert len(m.events) == len(m.cert) == 0


def _bind_episode(net, rng, learn):
    keys = rng.permutation(8)[:5]
    vals = 8 + rng.integers(8, size=4)
    net.reset()
    for k, v in zip(keys[:4], vals):
        net.observe(int(k)); net.observe(int(v)); net.observe(None)
    bound = rng.random() < 0.5
    q = int(keys[rng.integers(4)]) if bound else int(keys[4])
    y = int(vals[list(keys[:4]).index(q)]) if bound else 8 + (q * 3) % 8
    out = net.answer(q, explore=learn)
    net.feedback(target=y)
    return out


def test_meta_thinker_without_meta_parameters_is_the_thinker():
    """Sin meta-parámetros ni memoria graduada, MetaThinker es el Thinker bit a bit (las
    señales se miden pero no se usan)."""
    outs = []
    for cls in (Thinker, MetaThinker):
        rng = np.random.default_rng(0)
        net = cls(16, memory=True, seed=0)
        outs.append(([_bind_episode(net, rng, True) for _ in range(300)], net.Wc.copy(), net.wv.copy()))
    assert outs[0][0] == outs[1][0]
    assert np.array_equal(outs[0][1], outs[1][1]) and np.array_equal(outs[0][2], outs[1][2])


def test_meta_controller_learns_a_context_dependent_parameter():
    """Bandido de contexto sintético: con la señal «éxito» baja conviene rate chico y con la
    señal alta, rate grande. La política gaussiana entrenada por perturbación lo separa."""
    rng = np.random.default_rng(0)
    mc = MetaController(("rate",), dict(rate=0.5), signals=("exito",), seed=0)
    for _ in range(3000):
        ctx = int(rng.integers(2))
        p = mc.begin({"exito": 0.2 + 0.6 * ctx + 0.05 * rng.standard_normal()}, explore=True)
        want = 0.15 if ctx == 0 else 0.9
        mc.end(1.0 - abs(p["rate"] - want) + 0.1 * rng.standard_normal())
    lo = mc.mean_params({"exito": 0.2})["rate"]
    hi = mc.mean_params({"exito": 0.8})["rate"]
    assert lo < 0.35 and hi > 0.7


def test_meta_thinker_samples_while_learning_and_uses_the_mean_when_frozen():
    net = MetaThinker(16, memory=True, meta=("rate",), signals=("exito",), seed=0)
    assert isinstance(net.fast, GradedMemory)
    rng = np.random.default_rng(0)
    for _ in range(50):
        _bind_episode(net, rng, True)
    assert len(set(round(p["rate"], 6) for p in net.trace)) > 10      # muestras con ruido
    assert net.meta.updates > 0
    net.plastic = False
    net.reset(); r1 = net.fast.rate
    net.reset(); r2 = net.fast.rate
    W = net.meta.W.copy()
    _bind_episode(net, rng, False)
    assert r1 == r2 and np.array_equal(W, net.meta.W)                # media, sin aprender


def test_frozen_control_needs_zero_decay():
    """La contracción L1 del control no depende de lr: con lr=0 el control igual se borra.
    (Por eso «congelar el control» en los experimentos es lr=0 y decay=0.)"""
    net = Thinker(16, memory=True, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(300):
        _bind_episode(net, rng, True)
    W = net.Wc.copy()
    net.lr = 0.0
    for _ in range(300):
        _bind_episode(net, rng, True)
    assert np.abs(net.Wc).sum() < np.abs(W).sum()
    net.Wc[:] = W; net.decay = 0.0
    for _ in range(100):
        _bind_episode(net, rng, True)
    assert np.array_equal(net.Wc, W)

