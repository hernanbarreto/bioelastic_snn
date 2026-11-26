"""Mecanismos de aprendizaje continuo: error asimétrico, protección de clases ausentes,
desalojo por fusión y metaplasticidad que se relaja."""
import numpy as np

from bioelastic import BioElasticNet, ElasticLayer, Readout


def _protos(C, D=60, on=12, seed=0):
    rng = np.random.default_rng(seed)
    P = np.zeros((C, D), dtype=np.float32)
    for c in range(C):
        P[c, rng.choice(D, on, replace=False)] = 1.0
    return P


def _sample(P, c, rng, noise=0.15):
    x = P[c] * rng.uniform(0.7, 1.0, P.shape[1]) + noise * rng.random(P.shape[1])
    return np.clip(x, 0, 1).astype(np.float32)


def test_asymmetric_error_does_not_depress_absent_classes():
    idx = np.array([0]); h = np.ones(1, dtype=np.float32)
    plain, gated = Readout(3, consolidation=0.0), Readout(3, consolidation=0.0, active_window=10)
    for r in (plain, gated):
        r.add_input(cls=0)
        for _ in range(20):
            r.learn(idx, h, 0)                         # la clase 0 se aprende...
        for _ in range(30):
            r.learn(idx, h, 1)                         # ...y deja de aparecer
    assert plain.W[0, 0] < gated.W[0, 0]
    assert not gated.active()[0] and gated.active()[1] and plain.active().all()
    w0 = gated.W[0, 0]
    gated.learn(idx, h, 1, replay=False)
    assert gated.W[0, 0] == w0                         # inactiva: su error negativo no se aplica
    t = gated.t
    gated.learn(idx, h, 2, replay=True)                # el repaso usa el error completo
    assert gated.W[1, 0] < gated.W[1, 0] + 1 and gated.t == t   # y no cuenta como actividad


def test_reward_marks_the_rewarded_action_as_active():
    r = Readout(3, active_window=5)
    r.add_input(cls=None)
    r.learn_reward(np.array([0]), np.ones(1, dtype=np.float32), 2, 1.0)
    assert r.active()[2] and not r.active()[0]
    r.learn_reward(np.array([0]), np.ones(1, dtype=np.float32), 0, -1.0)
    assert not r.active()[0]


def test_relaxing_metaplasticity_restores_plasticity_of_unused_inputs():
    idx = np.array([0]); h = np.ones(1, dtype=np.float32)
    fixed, relax = Readout(2, consolidation=5.0), Readout(2, consolidation=5.0, relax=50.0)
    for r in (fixed, relax):
        r.add_input(cls=0); r.add_input(cls=1)
        for _ in range(200):
            r.learn(idx, h, 0)
        for _ in range(500):                           # la entrada 0 queda sin uso
            r.learn(np.array([1]), h, 1)
    assert fixed.count[0] == 200
    relax._plasticity(idx, h)
    assert relax.count[0] < 1.5                        # el contador decayó: vuelve a ser plástica


def test_protect_keeps_absent_class_neurons_untouched():
    def run(protect):
        rng = np.random.default_rng(0)
        P = _protos(4)
        P[2] = np.clip(P[0] + P[2] * 0.5, 0, 1)        # 2 se parece a 0, 3 a 1
        P[3] = np.clip(P[1] + P[3] * 0.5, 0, 1)
        net = BioElasticNet(60, 4, protect=protect, readout=dict(active_window=20))
        for _ in range(600):
            c = int(rng.integers(2)); net.step(_sample(P, c, rng), c)
        n0 = net.layer.n
        for t in range(600):                           # sólo clases nuevas, parecidas a las viejas
            if t == 30:                                # 0 y 1 ya están inactivas
                W0 = net.readout.W[:, :n0].copy()
            c = 2 + int(rng.integers(2)); net.step(_sample(P, c, rng), c)
        old = np.mean([net.predict(_sample(P, c, rng)) == c for c in rng.integers(0, 2, size=300)])
        new = np.mean([net.predict(_sample(P, c, rng)) == c for c in rng.integers(2, 4, size=300)])
        drift = np.abs(net.readout.W[:, :n0] - W0).sum()
        return old, new, drift
    old0, new0, drift0 = run(False)
    old1, new1, drift1 = run(True)
    assert drift1 < 0.1 * drift0                       # las neuronas de 0 y 1 no se recablean
    assert old1 >= old0 and new1 > 0.9


def test_merge_eviction_compresses_redundant_neurons_and_keeps_rare_classes():
    rng = np.random.default_rng(1)
    P = _protos(4, seed=1)
    net = BioElasticNet(60, 4, layer=dict(max_neurons=8, evict="merge", theta=0.98))
    for _ in range(400):                               # la clase 0 domina el flujo
        c = 0 if rng.random() < 0.85 else int(rng.integers(1, 4))
        net.step(_sample(P, c, rng, noise=0.3), c)
        assert net.layer.n == net.readout.n <= 8
    L = net.layer
    assert L.merged > 0 and L.evicted == L.merged
    pref = net.readout.preferred(np.arange(L.n))
    assert set(range(4)) <= set(pref.tolist())         # ninguna clase se quedó sin neuronas
    assert np.allclose(np.linalg.norm(L.w[:L.n], axis=1), 1.0, atol=1e-4)
    acc = np.mean([net.predict(_sample(P, c, rng, noise=0.3)) == c for c in range(4) for _ in range(50)])
    assert acc > 0.95


def test_merge_picks_the_most_similar_pair_of_the_largest_group():
    L = ElasticLayer(10, theta=0.1, max_neurons=4, evict="merge")
    base = np.zeros(10, dtype=np.float32)
    pats = [(0, 1, 2), (0, 1, 3), (5, 6, 7), (8, 9)]
    for p in pats:
        x = base.copy(); x[list(p)] = 1.0
        L.recruit(L.forward(np.flatnonzero(x), x[x > 0]))
    L.group_of = lambda: np.array([0, 0, 0, 1])[:L.n]
    x = base.copy(); x[[4]] = 1.0
    L.recruit(L.forward(np.flatnonzero(x), x[x > 0]))
    assert L.n == 4 and L.merged == 1
    d = np.stack([L.dense(j) for j in range(L.n)])
    assert (d[:, 0] > 0).sum() == 1 and (d[:, 2] > 0).sum() == 1 and (d[:, 3] > 0).sum() == 1


def test_class_mean_rule_only_moves_the_target_row_and_rows_keep_their_norm():
    r = Readout(3, rule="mean", row_norm=2.0, capacity=8)
    r.n = 8
    rng = np.random.default_rng(0)
    for _ in range(50):
        h = rng.random(8).astype(np.float32); h[:4] += 1.0
        r.learn(np.arange(8), h / np.linalg.norm(h), 0)
    W0 = r.W.copy()
    for _ in range(50):                                # otra clase: la fila 0 no cambia
        h = rng.random(8).astype(np.float32); h[4:] += 1.0
        r.learn(np.arange(8), h / np.linalg.norm(h), 1)
    assert np.array_equal(r.W[0], W0[0])
    assert np.allclose(np.linalg.norm(r.W[:2, :8], axis=1), 2.0, atol=1e-4)
    assert r.W[0, :4].mean() > r.W[0, 4:].mean() and r.W[1, 4:].mean() > r.W[1, :4].mean()
    r.learn_reward(np.arange(8), np.ones(8, dtype=np.float32) / np.sqrt(8), 2, -1.0)
    assert not r.W[2].any()                            # castigo: la regla de medias no aprende
    r.learn_reward(np.arange(8), np.ones(8, dtype=np.float32) / np.sqrt(8), 2, 1.0)
    assert r.W[2].any()


def test_row_norm_scales_delta_rule_outputs():
    r = Readout(2, consolidation=0.0, row_norm=1.0, capacity=4)
    r.n = 4
    for _ in range(30):
        r.learn(np.array([0, 1]), np.ones(2, dtype=np.float32), 0)
    assert np.allclose(np.linalg.norm(r.W[:, :4], axis=1), 1.0, atol=1e-4)


def _blob(rng, c):
    img = 0.05 * rng.random((8, 8, 1)).astype(np.float32)
    y, x = divmod(c, 2)
    img[y * 4:y * 4 + 4, x * 4:x * 4 + 4, 0] += rng.uniform(0.6, 1.0, (4, 4)).astype(np.float32)
    return np.clip(img, 0, 1)


def test_replay_works_from_a_memory_that_reads_the_same_code_as_the_projection():
    from bioelastic import Cortex
    rng = np.random.default_rng(3)
    net = Cortex((8, 8, 1), 4, areas=[dict(rf=3, pool=2, max_features=16, theta=0.8)],
                 heads=[dict(source="top", layer=dict(theta=0.5))], projection=dict(source="all"),
                 replay=1.0, joint=False)
    mem, space = net._replay_source()
    assert mem is None                                 # todavía no hay recuerdos
    for _ in range(300):
        c = int(rng.integers(4)); net.step(_blob(rng, c), c)
    mem, space = net._replay_source()
    assert mem is net.heads[0] and space == "code"
    assert net.replays > 100
    no_src = Cortex((8, 8, 1), 4, areas=[dict(rf=3, pool=2, max_features=16, theta=0.8)],
                    heads=[dict(source="top", tau=4.0)], projection=dict(source="all"), replay=1.0)
    for _ in range(50):
        c = int(rng.integers(4)); no_src.step(_blob(rng, c), c)
    assert no_src.replays == 0                         # traza lenta: no hay recuerdo que reactivar


def test_reused_feature_slot_also_clears_the_projection():
    from bioelastic import Cortex
    rng = np.random.default_rng(0)
    net = Cortex((8, 8, 1), 4, areas=[dict(rf=3, pool=2, max_features=16, theta=0.8)],
                 heads=[dict(source="input")], projection=dict(source="all"))
    for _ in range(100):
        c = int(rng.integers(4)); net.step(_blob(rng, c), c)
    a = net.areas[0]
    ids = np.arange(a.ph * a.pw) * a.F + 0
    net.proj.W[:, ids] = 1.0
    net._on_reuse(0, 0)
    assert not net.proj.W[:, ids].any()


def test_replay_focus_on_absent_classes():
    from bioelastic import Cortex
    rng = np.random.default_rng(5)
    net = Cortex((8, 8, 1), 4, areas=[dict(rf=3, pool=2, max_features=16, theta=0.8)],
                 heads=[dict(source="input")], projection=dict(source="all", active_window=20),
                 replay=1.0, replay_focus="absent", joint=False)
    for _ in range(300):
        c = int(rng.integers(4)); net.step(_blob(rng, c), c)
    seen = []
    learn = net.proj.learn
    net.proj.learn = lambda idx, h, c, p=None, replay=False: (seen.append((c, replay)), learn(idx, h, c, p=p, replay=replay))
    for _ in range(200):                               # sólo 0 y 1: el repaso va a 2 y 3
        c = int(rng.integers(2)); net.step(_blob(rng, c), c)
    rep = [c for c, r in seen[60:] if r]
    assert rep and set(rep) <= {2, 3}


def test_plasticity_floor_keeps_mature_inputs_learnable():
    idx = np.array([0]); h = np.ones(1, dtype=np.float32)
    plain, floored = Readout(3), Readout(3, floor=0.2)
    for r in (plain, floored):
        r.add_input(cls=0)
        for _ in range(1000):
            r.learn(idx, h, 0)
    for r in (plain, floored):
        w = r.W[1, 0]
        r.learn(idx, h, 1)
        r.dw = r.W[1, 0] - w
    assert floored.dw > 20 * plain.dw
