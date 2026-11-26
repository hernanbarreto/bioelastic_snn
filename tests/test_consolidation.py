"""Consolidación, corrección y refuerzo: dos sistemas (CLS), esquemas, reconsolidación,
autoexamen espaciado y sueño."""
import numpy as np

from bioelastic import BioElasticNet, Cortex, Readout


def _blob(rng, c):
    img = 0.05 * rng.random((8, 8, 1)).astype(np.float32)
    y, x = divmod(c, 2)
    img[y * 4:y * 4 + 4, x * 4:x * 4 + 4, 0] += rng.uniform(0.6, 1.0, (4, 4)).astype(np.float32)
    return np.clip(img, 0, 1)


def _cortex(**kw):
    return Cortex((8, 8, 1), 4, areas=[dict(rf=3, pool=2, max_features=16, theta=0.8)],
                  heads=[dict(source="input", episodic=True)], projection=dict(source="all"),
                  **{"replay": 1.0, "joint": False, **kw})


# ---------------------------------------------------------------- F1: dos sistemas
def test_episode_records_label_time_context_and_follow_swap_remove():
    rng = np.random.default_rng(0)
    D = 30
    net = BioElasticNet(D, 3, episodic=True, layer=dict(theta=0.99, fan_in=8))
    for i in range(280):                               # casi todas novedosas: crece la capacidad
        net.context_tag = i % 5
        x = np.zeros(D, dtype=np.float32); x[rng.choice(D, 10, replace=False)] = rng.uniform(0.2, 1.0, 10)
        net.step(x, i % 3)
    L, ep = net.layer, net.ep
    n = L.n
    assert n > 256 and len(ep["y"]) >= n
    assert set(np.unique(ep["y"][:n])) <= {0, 1, 2} and (ep["ctx"][:n] >= 0).all()
    e = net.episodes[:n].astype(np.float32)
    assert np.allclose(ep["norm"][:n], np.linalg.norm(e, axis=1) / 255.0, rtol=1e-5)
    last = {k: v[n - 1] for k, v in ep.items()}
    L.remove(2)                                        # la última pasa a la 2, con sus campos
    assert all(ep[k][2] == last[k] for k in ep)
    back = BioElasticNet.from_state(net.state())
    for k in ep:
        assert np.array_equal(back.ep[k][:L.n], ep[k][:L.n])


def test_stored_label_does_not_drift_with_the_readout():
    net = BioElasticNet(10, 3, episodic=True)
    x = np.zeros(10, dtype=np.float32); x[[1, 4, 7]] = [1.0, 0.8, 0.6]
    net.step(x, 0)
    net.readout.W[:, 0] = [0.0, 0.0, 2.0]               # la lectura derivó hacia la clase 2
    assert net.labels(stored=False)[0] == 2 and net.labels(stored=True)[0] == 0


def test_recall_returns_the_most_similar_episodes():
    rng = np.random.default_rng(1)
    D = 40
    net = BioElasticNet(D, 4, episodic=True, layer=dict(theta=0.95))
    X = []
    for i in range(40):
        x = np.zeros(D, dtype=np.float32); x[rng.choice(D, 8, replace=False)] = rng.uniform(0.3, 1.0, 8)
        X.append(x); net.step(x, i % 4)
    q = X[7]; idx = np.flatnonzero(q); j7 = int(np.flatnonzero(net.ep["t"][:net.layer.n] == 8)[0])
    ids, sim = net.recall(idx, q[idx], k=5)
    assert ids[0] == j7 and sim[0] > 0.99 and np.all(np.diff(sim) <= 1e-7)


def test_gate_consults_the_store_only_when_the_slow_system_is_unsure():
    rng = np.random.default_rng(3)
    net = _cortex(recall="gate", recall_margin=0.5)
    for _ in range(300):
        c = int(rng.integers(4)); net.step(_blob(rng, c), c)
    sure, unsure = 0, 0
    for _ in range(60):
        c = int(rng.integers(4))
        r0 = net.recalls
        net.forward(_blob(rng, c))
        parts = net.parts()
        S = parts["slow"]
        e = np.exp((S - S.max()) / net.temperature); q = np.sort(e / e.sum())
        if q[-1] - q[-2] >= 0.5:                       # seguro: responde sólo el lento
            assert net.recalls == r0 and np.array_equal(net._V, S); sure += 1
        else:                                          # inseguro: lento + episodios recuperados
            assert net.recalls == r0 + 1
            assert np.allclose(net._V, S + parts["fast"]) and np.isclose(parts["fast"].sum(), 1.0)
            unsure += 1
    assert sure > 0
    always = _cortex(recall="always")
    for _ in range(50):
        c = int(rng.integers(4)); always.step(_blob(rng, c), c)
    assert always.recalls >= 49                        # (en el primer paso el almacén está vacío)


def test_recall_off_is_the_default_and_leaves_inference_unchanged():
    rng = np.random.default_rng(4)
    a, b = _cortex(), _cortex(recall="off", stored_labels=False)
    for _ in range(150):
        c = int(rng.integers(4)); x = _blob(rng, c)
        assert a.step(x, c) == b.step(x, c)
    assert np.array_equal(a._V, b._V) and a.recalls == 0


def test_replay_with_stored_labels_uses_the_label_of_the_episode():
    rng = np.random.default_rng(5)
    net = _cortex(stored_labels=True)
    for _ in range(200):
        c = int(rng.integers(4)); net.step(_blob(rng, c), c)
    mem = net.heads[0]
    n = mem.layer.n
    mem.readout.W[:, :n] = mem.readout.W[::-1, :n]      # todas las lecturas derivaron
    seen = []
    learn = net.proj.learn
    net.proj.learn = lambda idx, h, c, p=None, replay=False, **kw: (seen.append(c) if replay else None,
                                                                   learn(idx, h, c, p=p, replay=replay, **kw))
    for _ in range(40):
        net._replay_one()
    assert seen and set(seen) <= set(np.unique(mem.ep["y"][:mem.layer.n]))
    assert set(np.unique(mem.labels(True))) == set(np.unique(mem.ep["y"][:mem.layer.n]))


def test_slow_system_can_learn_from_its_own_error():
    rng = np.random.default_rng(17)
    nets = {m: _cortex(replay=0.0, slow_error=m) for m in ("joint", "own")}
    xs = [(_blob(rng, c), int(c)) for c in rng.integers(4, size=120)]
    for net in nets.values():
        for x, c in xs[:100]:
            net.step(x, c)
    own = nets["own"]
    x, c = xs[100]
    own.forward(x)
    S = own.proj.potentials(*own._pv)
    e = np.exp((S - S.max()) / own.temperature)
    ref = own.proj.W.copy()
    ref[:, own._pv[0]] += (own.proj.lr * (np.eye(4)[c] - e / e.sum()))[:, None].astype(np.float32) * own._pv[1][None, :]
    own.teach(c)
    assert np.allclose(own.proj.W, ref, atol=1e-6)      # error del lento solo, sin el voto de la memoria
    j = nets["joint"]
    assert not np.array_equal(j.proj.W, own.proj.W)


def test_cortex_with_two_systems_saves_and_loads(tmp_path):
    rng = np.random.default_rng(6)
    net = _cortex(recall="gate", stored_labels=True)
    for _ in range(120):
        c = int(rng.integers(4)); net.step(_blob(rng, c), c)
    net.save(tmp_path / "n.npz")
    back = Cortex.load(tmp_path / "n.npz")
    assert back.recall == "gate" and back.stored_labels
    x = _blob(rng, 2)
    assert back.forward(x) == net.forward(x) and np.allclose(back._V, net._V)

