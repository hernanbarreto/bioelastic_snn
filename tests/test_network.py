import numpy as np

from bioelastic import BioElasticNet
from bioelastic.datasets import bionet_stream


def _protos(C, D=60, on=12, seed=0):
    rng = np.random.default_rng(seed)
    P = np.zeros((C, D), dtype=np.float32)
    for c in range(C):
        P[c, rng.choice(D, on, replace=False)] = 1.0
    return P


def _sample(P, c, rng, noise=0.15):
    x = P[c] * rng.uniform(0.7, 1.0, P.shape[1]) + noise * rng.random(P.shape[1])
    return np.clip(x, 0, 1).astype(np.float32)


def test_supervised_learns_and_structures_stay_in_sync():
    rng = np.random.default_rng(0); P = _protos(6)
    net = BioElasticNet(60, 6, layer=dict(probation=200), maintain_every=50)
    hits = []
    for _ in range(1500):
        c = int(rng.integers(6))
        hits.append(net.step(_sample(P, c, rng), c) == c)
        assert net.layer.n == net.readout.n
    assert np.mean(hits[-300:]) > 0.97


def test_reward_only_learning_without_labels():
    rng = np.random.default_rng(1); P = _protos(4)
    net = BioElasticNet(60, 4, seed=1)
    for _ in range(3000):
        c = int(rng.integers(4))
        a = net.forward(_sample(P, c, rng), explore=True)
        net.reinforce(1.0 if a == c else -1.0)
    acc = np.mean([net.predict(_sample(P, c, rng)) == c for c in rng.integers(4, size=400)])
    assert acc > 0.9


def test_legacy_bionet_task_beats_legacy_result():
    gen = bionet_stream(n_classes=10, seed=11)
    net = BioElasticNet(256, 10)
    hits = [net.step(x, y) == y for x, y in (next(gen) for _ in range(2000))]
    assert np.mean(hits[-500:]) > 0.95                 # el bioNet original llegó a 0.68 en 16k pasos


def test_growth_follows_information_not_time():
    rng = np.random.default_rng(2); P = _protos(8)
    net = BioElasticNet(60, 8)
    for _ in range(800):                               # sólo 3 clases
        c = int(rng.integers(3)); net.step(_sample(P, c, rng), c)
    n_a = net.layer.n
    for _ in range(800):                               # mismo flujo: nada nuevo que aprender
        c = int(rng.integers(3)); net.step(_sample(P, c, rng), c)
    n_b = net.layer.n
    for _ in range(800):                               # aparecen 5 clases nuevas
        c = int(rng.integers(8)); net.step(_sample(P, c, rng), c)
    n_c = net.layer.n
    assert n_b - n_a <= 2
    assert n_c - n_b >= 5


def test_no_catastrophic_forgetting_class_incremental():
    rng = np.random.default_rng(3); P = _protos(6)
    net = BioElasticNet(60, 6)
    for task in ((0, 1), (2, 3), (4, 5)):
        for _ in range(600):
            c = int(rng.choice(task)); net.step(_sample(P, c, rng), c)
    old = np.mean([net.predict(_sample(P, c, rng)) == c for c in rng.integers(0, 2, size=300)])
    assert old > 0.95


def test_delay_line_encodes_order_in_spike_strength():
    net = BioElasticNet(10, 2, context=3, delay=0.2)
    for sym in (4, 7, 1):
        net._push(sym)
    idx, val = net._composite()
    assert list(idx) == [1, 10 + 7, 20 + 4]            # más reciente primero, desplazado por lag
    assert np.allclose(val, [1.0, 0.8, 0.6])


def test_symbol_stream_needs_and_uses_context():
    # tras 'a' viene 'b' o 'c' según el símbolo anterior: sin contexto es impredecible
    seq = [0, 1, 3, 0, 2, 4] * 300                     # a b d a c e ...
    net = BioElasticNet(5, 5, context=3, delay=0.15)
    hits = []
    for t in range(len(seq) - 1):
        pred = net.forward(seq[t]); net.teach(seq[t + 1])
        hits.append(pred == seq[t + 1])
    assert np.mean(hits[-300:]) == 1.0


def _seq(order, rng):
    f = np.zeros((2, 8), dtype=np.float32)
    f[0, :4] = 1.0; f[1, 4:] = 1.0
    out = [f[i] * rng.uniform(0.8, 1.0, 8).astype(np.float32) for i in order]
    return out


def test_sequence_classes_that_differ_only_in_order():
    for decide in ("end", "accumulate"):
        rng = np.random.default_rng(4)
        net = BioElasticNet(8, 2, context=3, delay=0.2)
        hits = []
        for _ in range(200):
            c = int(rng.integers(2))
            order = (0, 1, 0) if c == 0 else (1, 0, 0)
            hits.append(net.run_sequence(_seq(order, rng), c, decide=decide) == c)
            assert net.layer.n == net.readout.n
        assert np.mean(hits[-80:]) == 1.0, decide


def test_unsupervised_observe_clusters_without_labels():
    rng = np.random.default_rng(5); P = _protos(4)
    net = BioElasticNet(60, 4)
    for _ in range(600):
        net.observe(_sample(P, int(rng.integers(4)), rng, noise=0.05))
    assert 4 <= net.layer.n <= 12
    winners = [int(net.layer.forward(*net._as_sparse(_sample(P, c, rng, 0.05))).winners[0])
               for c in range(4) for _ in range(20)]
    by_class = [set(winners[c * 20:(c + 1) * 20]) for c in range(4)]
    assert all(by_class[a].isdisjoint(by_class[b]) for a in range(4) for b in range(a + 1, 4))


def test_sklearn_digits_single_online_pass():
    from sklearn.datasets import load_digits
    d = load_digits()
    X = (d.data / 16.0).astype(np.float32); y = d.target
    perm = np.random.default_rng(0).permutation(len(X))
    tr, te = perm[:1400], perm[1400:]
    net = BioElasticNet(64, 10)
    for i in tr:
        net.step(X[i], int(y[i]))
    acc = np.mean([net.predict(X[i]) == y[i] for i in te])
    assert acc > 0.93


def test_readout_consolidation_protects_mature_synapses():
    from bioelastic import Readout
    idx = np.array([0]); h = np.ones(1, dtype=np.float32)
    plastic, stable = Readout(3, consolidation=0.0), Readout(3, consolidation=5.0)
    for r in (plastic, stable):
        r.add_input(cls=0)
        for _ in range(200):
            r.learn(idx, h, 0)                         # uso prolongado con la clase 0
        for _ in range(3):
            r.learn(idx, h, 2)                         # unos pocos ejemplos contradictorios
    assert plastic.preferred(idx)[0] == 2              # sin consolidar: recableada
    assert stable.preferred(idx)[0] == 0               # consolidada: conserva lo aprendido


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(6); P = _protos(5)
    net = BioElasticNet(60, 5, layer=dict(k=3))
    for _ in range(500):
        c = int(rng.integers(5)); net.step(_sample(P, c, rng), c)
    path = tmp_path / "net.npz"
    net.save(path)
    twin = BioElasticNet.load(path)
    assert twin.stats() == net.stats() and twin.layer.k == 3 and twin.layer.t == net.layer.t
    X = [_sample(P, int(c), rng) for c in rng.integers(5, size=100)]
    assert [twin.predict(x) for x in X] == [net.predict(x) for x in X]
    for x in X:                                        # sigue aprendiendo igual que la original
        assert twin.step(x, 0) == net.step(x, 0)
    assert twin.stats() == net.stats()
