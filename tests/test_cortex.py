import numpy as np
import pytest

from bioelastic import BioElasticNet, Cortex, ElasticLayer, FeatureArea, Readout
from bioelastic.blueprint import BRAIN_SIM_REFERENCE, HUMAN, plan


def _bar(rng, vertical: bool, lo: int, hi: int, size: int = 12, length: int = 5):
    """Barra horizontal o vertical cuya esquina cae en [lo, hi) x [lo, hi)."""
    img = np.zeros((size, size, 1), dtype=np.float32)
    y, x = (int(v) for v in rng.integers(lo, hi, size=2))
    if vertical:
        img[y:y + length, x, 0] = 1.0
    else:
        img[y, x:x + length, 0] = 1.0
    return img


# ---------------------------------------------------------------- almacenamiento disperso
def test_fan_in_is_bounded_and_synapses_turn_over():
    layer = ElasticLayer(100, fan_in=8, theta=0.2)
    a = np.zeros(100, dtype=np.float32); a[:20] = np.linspace(1.0, 0.1, 20)
    layer.recruit(layer.forward(np.flatnonzero(a), a[a > 0]))
    assert layer.fill[0] == 8 and set(layer.pre[0]) == set(range(8))      # las 8 más tempranas
    assert np.linalg.norm(layer.w[0]) == pytest.approx(1.0, abs=1e-5)
    b = np.zeros(100, dtype=np.float32); b[:4] = 1.0; b[50:54] = 1.0
    for _ in range(30):                                                    # sinaptogénesis + recambio
        layer.stdp(0, layer.forward(np.flatnonzero(b), b[b > 0]))
    assert layer.fill[0] == 8
    assert set(layer.pre[0]) == {0, 1, 2, 3, 50, 51, 52, 53}
    assert layer.nbytes() == 8 * 8 and layer.synapses() == 8


def test_memory_does_not_depend_on_input_dimension():
    small, huge = ElasticLayer(1_000, fan_in=16), ElasticLayer(1_000_000, fan_in=16)
    for layer in (small, huge):
        idx = np.arange(16) * 7
        for _ in range(50):
            layer.recruit(layer.forward(idx, np.ones(16, dtype=np.float32)))
    assert small.nbytes() == huge.nbytes() == 50 * 16 * 8


def test_forget_afferents_cuts_only_those_synapses():
    layer = ElasticLayer(20, theta=0.2)
    v = layer.forward(np.arange(6), np.ones(6, dtype=np.float32))
    layer.recruit(v)
    layer.forget_afferents(np.array([1, 4]))
    w = layer.dense(0)
    assert w[1] == 0 and w[4] == 0 and (w[[0, 2, 3, 5]] > 0).all()


# ---------------------------------------------------------------- área sensorial
def test_feature_area_learns_a_bounded_dictionary_and_tolerates_shifts():
    rng = np.random.default_rng(0)
    area = FeatureArea((12, 12, 1), rf=5, pool=4, max_features=16, theta=0.8)
    for _ in range(300):
        area.encode(_bar(rng, bool(rng.integers(2)), 0, 7)); area.learn()
    assert 2 <= int(area.alive.sum()) <= 16
    assert area.out_shape == (2, 2, 16)
    img = np.zeros((12, 12, 1), dtype=np.float32); img[2:7, 3, 0] = 1.0
    shifted = np.roll(img, 1, axis=1)
    other = np.zeros((12, 12, 1), dtype=np.float32); other[3, 2:7, 0] = 1.0
    a, b, c = (area.encode(x).ravel() for x in (img, shifted, other))
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    assert cos(img.ravel(), shifted.ravel()) == 0.0    # en píxeles, corrida un lugar es otra cosa
    assert cos(a, b) > 0.5                             # en el código sigue siendo la misma barra
    assert cos(a, c) < 0.1                             # y la otra orientación, no


def test_dead_feature_slot_is_reused_and_downstream_is_notified():
    area = FeatureArea((6, 6, 1), rf=3, pool=1, max_features=2, theta=0.9, probation=5,
                       survival_hits=3, maintain_every=1)
    reused = []
    area.on_reuse = reused.append
    rng = np.random.default_rng(0)
    x = np.zeros((6, 6, 1), dtype=np.float32); x[1, :, 0] = 1.0
    area.encode(x); area.learn()                       # crea rasgos de este patrón
    y = np.zeros((6, 6, 1), dtype=np.float32); y[:, 2, 0] = 1.0
    for _ in range(12):                                # el patrón x no vuelve: sus rasgos mueren
        area.encode(y + 0.01 * rng.random((6, 6, 1)).astype(np.float32)); area.learn()
    assert area.pruned >= 1 and len(reused) >= 1 and int(area.alive.sum()) <= 2


# ---------------------------------------------------------------- jerarquía
def test_hierarchy_generalizes_to_unseen_positions_where_a_single_area_cannot():
    def run(net):
        rng = np.random.default_rng(1)
        for _ in range(400):                           # entrenamiento: sólo la esquina superior izquierda
            c = int(rng.integers(2)); net.step(_bar(rng, bool(c), 0, 3), c)
        test = [(int(c), _bar(rng, bool(c), 5, 8)) for c in rng.integers(2, size=200)]
        return np.mean([net.predict(x) == c for c, x in test])   # test: esquina inferior derecha

    flat = Cortex((12, 12, 1), 2, heads=[{}])
    deep = Cortex((12, 12, 1), 2, areas=[dict(rf=5, pool=8, max_features=16, theta=0.8)],
                  heads=[dict(source="input")], projection=dict(source="all"), replay=0.5, joint=False)
    assert run(flat) < 0.75
    assert run(deep) > 0.95


def test_three_level_stack_runs_and_reports_structure():
    rng = np.random.default_rng(2)
    net = Cortex((20, 20, 1), 2,
                 areas=[dict(rf=3, pool=2, max_features=16, theta=0.8),
                        dict(rf=3, pool=1, max_features=24, theta=0.5, floor=0.2),
                        dict(rf=3, pool=2, max_features=24, theta=0.5, floor=0.2)],
                 heads=[dict(source="input")], projection=dict(source="all"))
    hits = []
    for _ in range(300):
        c = int(rng.integers(2)); hits.append(net.step(_bar(rng, bool(c), 0, 15, size=20), c) == c)
    st = net.stats()
    assert st["depth"] == 4 and len(st["areas"]) == 3 and all(a["features"] > 0 for a in st["areas"])
    assert np.mean(hits[-100:]) > 0.9


def _blob(rng, c, n=4):
    img = 0.05 * rng.random((8, 8, 1)).astype(np.float32)
    y, x = divmod(c, 2) if n == 4 else (c, 0)
    img[y * 4:y * 4 + 4, x * 4:x * 4 + 4, 0] += rng.uniform(0.6, 1.0, (4, 4)).astype(np.float32)
    img[3:5, 3:5, 0] = 1.0                               # zona compartida entre clases
    return np.clip(img, 0, 1)


def test_replay_protects_the_direct_projection_from_forgetting():
    def run(replay):
        rng = np.random.default_rng(3)
        net = Cortex((8, 8, 1), 4, areas=[dict(rf=3, pool=2, max_features=16, theta=0.8)],
                     heads=[dict(source="input")], projection=dict(source="all"),
                     replay=replay, joint=False)
        for task in ((0, 1), (2, 3)):
            for _ in range(600):
                c = int(rng.choice(task)); net.step(_blob(rng, c), c)
        net.heads = []; net._src = []                    # se evalúa sólo la proyección directa
        old = [int(c) for c in rng.integers(0, 2, size=200)]
        return np.mean([net.predict(_blob(rng, c)) == c for c in old]), net.replays
    forgot, n0 = run(0.0)
    kept, n1 = run(1.0)
    assert n0 == 0 and n1 > 1000
    assert kept > 0.95 and kept > forgot + 0.2


def test_area_added_to_a_trained_network_does_not_erase_it():
    rng = np.random.default_rng(4)
    net = Cortex((8, 8, 1), 4, heads=[dict(source="input")])
    for _ in range(400):
        c = int(rng.integers(4)); net.step(_blob(rng, c), c)
    test = [(int(c), _blob(rng, c)) for c in rng.integers(4, size=200)]
    before = np.mean([net.predict(x) == c for c, x in test])
    net.add_head(source="input", layer=dict(theta=0.9), seed=9)       # área nueva, vacía
    assert np.mean([net.predict(x) == c for c, x in test]) == before
    for _ in range(200):
        c = int(rng.integers(4)); net.step(_blob(rng, c), c)
    assert net.heads[1].layer.n > 0
    assert np.mean([net.predict(x) == c for c, x in test]) >= before - 0.02


def test_slow_trace_area_remembers_beyond_the_delay_line():
    def run(heads):
        rng = np.random.default_rng(5)
        net = Cortex(8, 2, heads=heads)
        hits = []
        for _ in range(1500):
            key = int(rng.integers(2))
            net.reset_context()
            net.forward(key)
            for s in rng.integers(2, 7, size=int(rng.integers(8, 16))):
                net.forward(int(s))                    # espera más larga que la ventana
            hits.append(net.step(7, key) == key)
        return np.mean(hits[-300:])
    assert run([dict(context=4, delay=0.2)]) < 0.65
    assert run([dict(context=4, delay=0.2), dict(tau=16.0)]) > 0.9


def test_cortex_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(6)
    net = Cortex((8, 8, 1), 4, areas=[dict(rf=3, pool=2, max_features=16, theta=0.8)],
                 heads=[dict(source="input"), dict(source="top", layer=dict(theta=0.4))],
                 projection=dict(source="all"), replay=0.5, joint=False)
    for _ in range(300):
        c = int(rng.integers(4)); net.step(_blob(rng, c), c)
    net.add_head(source="input", tau=4.0)              # un área agregada en caliente también se guarda
    for _ in range(50):
        c = int(rng.integers(4)); net.step(_blob(rng, c), c)
    path = tmp_path / "cortex.npz"
    net.save(path)
    twin = Cortex.load(path)
    assert len(twin.heads) == 3 and twin.heads[2].tau == 4.0
    assert twin.stats() == net.stats() and twin.replay == 0.5 and not twin.joint
    assert np.array_equal(twin.proj.W, net.proj.W) and np.abs(net.proj.W).sum() > 0
    X = [_blob(rng, int(c)) for c in rng.integers(4, size=100)]
    net.reset_context(); twin.reset_context()          # el contexto en curso no se guarda
    for x in X:
        assert twin.predict(x) == net.predict(x)
        assert np.allclose(twin._V, net._V)


def test_deep_network_learns_from_reward_only():
    rng = np.random.default_rng(8)
    net = Cortex((8, 8, 1), 4, areas=[dict(rf=3, pool=2, max_features=16, theta=0.8)],
                 heads=[dict(source="input")], projection=dict(source="all"), seed=8)
    for _ in range(2500):
        c = int(rng.integers(4))
        a = net.forward(_blob(rng, c), explore=True)
        net.reinforce(1.0 if a == c else -1.0)
    acc = np.mean([net.predict(_blob(rng, int(c))) == c for c in rng.integers(4, size=300)])
    assert acc > 0.9


def test_single_head_cortex_equals_bioelasticnet():
    rng = np.random.default_rng(7)
    a, b = Cortex(40, 3, heads=[{}]), BioElasticNet(40, 3)
    for _ in range(300):
        x = rng.random(40).astype(np.float32) * (rng.random(40) < 0.3); y = int(rng.integers(3))
        assert a.step(x, y) == b.step(x, y)
    assert a.stats()["neurons"] == b.stats()["neurons"]


# ---------------------------------------------------------------- plano de escala
def test_blueprint_numbers_are_consistent():
    p = plan(1.0, syn_per_core_second=1e8)
    assert p["neurons"] == HUMAN["neurons_cortex"] == 16e9
    assert p["synapses"] == pytest.approx(16e9 * 7000)
    assert p["memory_bytes"] == pytest.approx(p["synapses"] * 8)
    assert p["areas"] == 360 and p["laminae"] == 6
    assert p["core_seconds_per_step"] == pytest.approx(p["synapses"] / 1e8)
    assert plan(1e-6)["neurons"] == pytest.approx(16e3)
    assert HUMAN["neurons_cortex"] + HUMAN["neurons_cerebellum"] <= HUMAN["neurons_total"]
    assert BRAIN_SIM_REFERENCE["neurons"] / HUMAN["neurons_total"] == pytest.approx(0.02, abs=0.001)


# ---------------------------------------------------------------- error local por área
def _cue(rng, c):
    """Tres barras fuertes al azar (distractores) y una pista tenue de clase: una L o su espejo."""
    img = np.zeros((10, 10, 1), dtype=np.float32)
    for _ in range(3):
        y, x = (int(v) for v in rng.integers(0, 8, 2))
        if rng.random() < 0.5:
            img[y, x:x + 3, 0] = 1.0
        else:
            img[y:y + 3, x, 0] = 1.0
    y, x = (int(v) for v in rng.integers(0, 8, 2))
    cells = ((0, 0), (1, 0), (1, 1)) if c == 0 else ((0, 1), (1, 1), (1, 0))
    for dy, dx in cells:
        img[y + dy, x + dx, 0] = 0.6
    return img


def _decodability(area, rng, n_fit=2000, n_test=1000):
    """Exactitud de una lectura de 3 factores entrenada sobre el código congelado del área."""
    area.plastic = False
    R = Readout(2, capacity=int(np.prod(area.out_shape)), consolidation=0.0); R.n = R.cap

    def code(c):
        z = area.encode(_cue(rng, c)).ravel(); i = np.flatnonzero(z)
        return i, z[i] / max(float(np.linalg.norm(z[i])), 1e-9)
    for c in rng.integers(2, size=n_fit):
        R.learn(*code(int(c)), int(c))
    hits = [int(np.argmax(R.potentials(*code(int(c))))) == c for c in rng.integers(2, size=n_test)]
    return float(np.mean(hits))


def test_local_error_turns_area_features_toward_the_task():
    def run(local):
        rng = np.random.default_rng(0)
        a = FeatureArea((10, 10, 1), rf=3, pool=4, max_features=8, theta=0.9, k=2,
                        local=local, n_classes=2)
        for _ in range(3000):
            c = int(rng.integers(2)); a.encode(_cue(rng, c)); a.learn(c)
        return _decodability(a, rng), a
    unsup, _ = run(None)
    sup, a = run(dict(lr=0.05))
    assert unsup < 0.65                                  # sin error: el diccionario mira los distractores
    assert sup > unsup + 0.1
    assert a.local_seen >= 2990 and a.local_hits / a.local_seen > 0.6   # salvo antes del primer rasgo


def test_area_without_target_learns_exactly_as_unsupervised():
    rng = np.random.default_rng(1)
    imgs = [_cue(rng, int(c)) for c in rng.integers(2, size=200)]
    a = FeatureArea((10, 10, 1), rf=3, pool=2, max_features=8, theta=0.9)
    b = FeatureArea((10, 10, 1), rf=3, pool=2, max_features=8, theta=0.9, local=dict(lr=0.1),
                    n_classes=2)
    for x in imgs:
        assert np.allclose(a.encode(x), b.encode(x))
        a.learn(); b.learn()                             # sin objetivo: el error local no actúa
    assert np.array_equal(a.W, b.W)


def test_segment_sum_matches_add_at():
    from bioelastic.area import _segment_sum
    rng = np.random.default_rng(2)
    keys = rng.integers(0, 7, size=50); rows = rng.random((50, 4)).astype(np.float32)
    w = rng.random(50).astype(np.float32)
    ref = np.zeros((7, 4), dtype=np.float32); np.add.at(ref, keys, rows * w[:, None])
    u, s = _segment_sum(keys, rows, w)
    assert np.allclose(ref[u], s, atol=1e-5) and np.all(ref[np.setdiff1d(np.arange(7), u)] == 0)


def test_balanced_projection_gives_each_area_the_same_voice():
    net = Cortex((12, 12, 1), 2, areas=[dict(rf=3, pool=2, max_features=16, theta=0.8),
                                        dict(rf=3, pool=1, max_features=16, theta=0.5, floor=0.2)],
                 heads=[], projection=dict(source="all", balance=True))
    rng = np.random.default_rng(3)
    for _ in range(50):
        c = int(rng.integers(2)); net.step(_bar(rng, bool(c), 0, 7), c)
    net.forward(_bar(rng, True, 2, 5))
    idx, h = net._pv
    n1 = int(np.prod(net.areas[0].out_shape))
    for part in (idx < n1, idx >= n1):
        assert float(np.sqrt(h[part] @ h[part])) == pytest.approx(np.sqrt(0.5), rel=1e-5)


def test_deep_cortex_with_local_errors_learns_and_roundtrips(tmp_path):
    rng = np.random.default_rng(6)
    net = Cortex((8, 8, 1), 4, areas=[dict(rf=3, pool=2, max_features=16, theta=0.8,
                                           local=dict(lr=0.05)),
                                      dict(rf=2, pool=1, max_features=16, theta=0.5, floor=0.2,
                                           local=dict(lr=0.05))],
                 heads=[dict(source="input")], projection=dict(source="all", balance=True),
                 replay=0.5, joint=False)
    hits = []
    for _ in range(600):
        c = int(rng.integers(4)); hits.append(net.step(_blob(rng, c), c) == c)
    assert np.mean(hits[-200:]) > 0.9
    assert all(a.local_seen >= 590 for a in net.areas) and net.replays > 0
    path = tmp_path / "local.npz"
    net.save(path)
    twin = Cortex.load(path)
    assert np.array_equal(twin.areas[1].clf.W, net.areas[1].clf.W)
    X = [_blob(rng, int(c)) for c in rng.integers(4, size=50)]
    for x in X:
        assert twin.predict(x) == net.predict(x)
        assert np.allclose(twin.areas[1].potentials(), net.areas[1].potentials())
