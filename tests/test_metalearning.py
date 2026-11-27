import numpy as np

from bioelastic.metalearning import GradedMemory, MetaController, MetaThinker
from bioelastic.thinker import FastMemory, Thinker


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
    señales se miden pero no se usan). Los meta-parámetros del MetaThinker los fija su
    controlador: por defecto va sin los neuromoduladores del Thinker, así que se compara con
    el Thinker sin ellos."""
    outs = []
    for cls in (Thinker, MetaThinker):
        rng = np.random.default_rng(0)
        net = cls(16, memory=True, seed=0, modulators=False)
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
    (Por eso «congelar el control» en los experimentos es lr=0 y decay=0.) Sin conjuntos de
    tareas: con ellos, aprendiendo, puede pasar a regir otro conjunto (otra matriz)."""
    net = Thinker(16, memory=True, seed=0, task_sets=False)
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


# ---------------------------------------------------------------- F2: lo rápido en el estado
def _harlow(net, rng, objs, T=6, learn=True, reverse_at=None):
    good = int(rng.integers(2)); ok = []
    net.reset()
    for t in range(T):
        if reverse_at is not None and t == reverse_at:
            good = 1 - good
        q = [objs[i] for i in rng.permutation(2)]
        out = net.answer(q[0], probe=q[1], explore=learn)
        ok.append(out == objs[good])
        net.feedback(target=objs[good])
    return ok


def _train_harlow(prev_io, P=300, seed=0):
    rng = np.random.default_rng(seed)
    net = Thinker(2 * (P + 200) + 2, memory=True, speak=True, judge=True, max_steps=10,
                  gate=True, prev_io=prev_io, seed=seed)
    objs = rng.permutation(2 * (P + 200))
    for i in range(P):
        _harlow(net, rng, [int(objs[2 * i]), int(objs[2 * i + 1])])
    net.reset(); net.plastic = False
    return net, rng, objs[2 * P:]


def test_previous_answer_and_reward_solve_new_problems_with_frozen_weights():
    """Harlow sin memoria episódica del resultado: el control y la compuerta ven si cada
    objeto es la última respuesta y si fue premiada. Con los pesos congelados, el ensayo 2
    de problemas nuevos sale bien (quedarse si ganó, cambiar si perdió), y una reversión
    a mitad del problema se corrige en un ensayo, sin cambiar pesos."""
    net, rng, objs = _train_harlow(True)
    W = net.Wc.copy()
    accs = np.array([_harlow(net, rng, [int(objs[2 * i]), int(objs[2 * i + 1])], learn=False) for i in range(60)])
    assert accs[:, 1:].mean() > 0.95 and 0.25 < accs[:, 0].mean() < 0.75
    rev = np.array([_harlow(net, rng, [int(objs[2 * i]), int(objs[2 * i + 1])], T=8, learn=False, reverse_at=3)
                    for i in range(60, 120)])
    assert rev[:, 3].mean() < 0.1 and rev[:, 4:].mean() > 0.95
    assert np.array_equal(W, net.Wc)


def test_without_previous_answer_and_reward_there_is_no_learning_set():
    net, rng, objs = _train_harlow(False)
    accs = np.array([_harlow(net, rng, [int(objs[2 * i]), int(objs[2 * i + 1])], learn=False) for i in range(60)])
    assert accs[:, 1].mean() < 0.75


def test_previous_answer_and_reward_are_state_even_when_frozen():
    net = Thinker(10, memory=True, gate=True, prev_io=True, seed=0)
    net.plastic = False
    net.reset(); out = net.answer(3, probe=4)
    net.feedback(target=out)
    assert net._prev == (out, True)
    net.answer(3, probe=4)
    assert net._q[0].prev == (out, True)                 # el episodio guardado lo lleva
    net.feedback(target=7)
    assert net._prev[1] is False
    net.reset()
    assert net._prev is None


# ---------------------------------------------------------------- F3: mapa recorrible al revés
def test_recall_back_returns_what_preceded():
    from bioelastic.thinker import _recall_back
    m = FastMemory(20)
    for a, b in ((1, 2), (2, 3), (7, 3)):
        m.bind((a,)); m.bind((b,)); m.bind(())
    idx, val = _recall_back(m.events, (2,))
    assert list(idx) == [1]
    idx, val = _recall_back(m.events, (3,))
    assert set(idx) == {2, 7} and int(idx[np.argmax(val)]) == 7      # el más reciente, más fuerte
    assert len(_recall_back(m.events, (1,))[0]) == 0


def test_back_option_off_keeps_the_state_layout():
    old = dict(prev_io=False, back=False, task_sets=False, cogmap=False, modulators=False)   # la v0.3
    a = Thinker(20, memory=True, judge=True, speak=True, seed=0, **old)
    b = Thinker(20, memory=True, judge=True, speak=True, seed=0, **dict(old, back=False))
    assert a.Wc.shape == b.Wc.shape == (8, a.D_state) and a.D_state == b.D_state
    c = Thinker(20, memory=True, judge=True, speak=True, seed=0, **dict(old, back="automatico"))
    assert c.Wc.shape[0] == 9 and c.D_state == a.D_state + 1 + 3
    d = Thinker(20, memory=True, judge=True, speak=True, seed=0, **dict(old, back=True))   # + ¿sostenido?, ¿falló?
    assert d.Wc.shape[0] == 9 and d.D_state == a.D_state + 1 + 6


def _pred_block(net, rng, learn, L=5):
    syms = [int(x) for x in rng.permutation(40)[:L]]
    net.reset()
    for i in rng.permutation(L - 1):
        net.observe(syms[i]); net.observe(syms[i + 1]); net.observe(None)
    j = int(rng.integers(1, L))
    out = net.answer(syms[j], explore=learn)
    net.feedback(target=syms[j - 1])
    return out == syms[j - 1]


def test_inverse_relation_is_inferred_only_with_the_bidirectional_map():
    """Se observan sólo relaciones «X -> sucesor»; se pregunta el anterior de X. Sin
    recorrido inverso no hay forma (0); con RETROCEDER el control lo aprende con unos
    símbolos y, congelado, lo aplica a símbolos nuevos (y a cadenas más largas)."""
    res = {}
    for back in (False, True):
        rng = np.random.default_rng(0)
        net = Thinker(40, memory=True, speak=True, judge=True, max_steps=10, seed=0, back=back)
        for _ in range(300):
            _pred_block(net, rng, True)
        net.reset(); net.plastic = False
        res[back] = (np.mean([_pred_block(net, rng, False) for _ in range(100)]),
                     np.mean([_pred_block(net, rng, False, L=9) for _ in range(100)]))
    assert res[False][0] < 0.1
    assert res[True][0] > 0.95 and res[True][1] > 0.95
