"""Optimizaciones: equivalencia exacta de las rutas rápidas y contabilidad de memoria/costo."""
import numpy as np
import pytest

from bioelastic import Cortex, ElasticLayer
from bioelastic import blueprint as bp


def _lockstep(make, stream, learn=True):
    """Corre la misma secuencia en una columna con pase denso y otra con índice forzado y
    exige voleas y estado idénticos bit a bit. Devuelve cuántas veces se usó el índice."""
    dense, sparse = make(), make()
    dense._wants_sparse = lambda *a: False
    sparse.SPARSE_MIN_NEURONS = 1; sparse.SPARSE_MIN_SYNAPSES = 1
    sparse.SPARSE_MAX_ACTIVE = 1.0; sparse.SPARSE_MAX_TOUCHED = np.inf
    used = [0]
    orig = sparse._sparse_pass

    def counted(*a):
        used[0] += 1
        return orig(*a)
    sparse._sparse_pass = counted
    for t, (idx, val) in enumerate(stream):
        a, b = dense.forward(idx, val), sparse.forward(idx, val)
        for f in ("winners", "times", "cos"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), (t, f)
            assert getattr(a, f).dtype == getattr(b, f).dtype
        assert a.novel == b.novel
        if not learn:
            continue
        for L, v in ((dense, a), (sparse, b)):
            L.tick()
            if v.novel or t % 7 == 0:
                L.recruit(v)
            else:
                j = int(v.winners[0]); L.stdp(j, v, +1.0); L.credit(j)
                if len(v.winners) > 1:
                    L.stdp(int(v.winners[-1]), v, -0.4)
            if t % 50 == 49:
                L.maintain()
            if t % 83 == 82:
                L.forget_afferents(np.arange(t % 40, t % 40 + 5))
    for name in ("pre", "w") + ElasticLayer._STATS:
        assert np.array_equal(getattr(dense, name)[:dense.n], getattr(sparse, name)[:sparse.n]), name
    return used[0]


@pytest.mark.parametrize("theta,theta_plus,max_neurons", [(0.9, 0.0, None), (0.6, 0.03, 120), (0.3, 0.0, 80)])
def test_inverted_index_is_bit_identical_near_threshold(theta, theta_plus, max_neurons):
    """Voleas que repiten patrones guardados con poco ruido: muchas neuronas quedan con el
    potencial pegado al umbral, que es donde la cota del índice podría equivocarse."""
    rng = np.random.default_rng(int(theta * 100))
    D = 3000
    protos = [np.sort(rng.choice(D, 24, replace=False)) for _ in range(40)]
    stream = []
    for t in range(700):
        p = protos[int(rng.integers(40))]
        keep = rng.random(len(p)) < 0.9
        idx = np.unique(np.concatenate([p[keep], rng.integers(D, size=2)]))
        stream.append((idx, rng.uniform(0.5, 1.0, len(idx)).astype(np.float32)))
    make = lambda: ElasticLayer(D, fan_in=20, k=3, theta=theta, theta_plus=theta_plus, theta_tau=40.0,
                                max_neurons=max_neurons, probation=150, max_candidates=6, capacity=8)
    assert _lockstep(make, stream) > 0


def test_inverted_index_on_random_columns_without_learning():
    rng = np.random.default_rng(3)
    L0 = ElasticLayer(1024, fan_in=32, capacity=600, theta=0.08)
    start = rng.integers(0, 1024, size=(600, 1))
    pre = (start + rng.permutation(1024)[None, :32]) % 1024
    w = rng.random((600, 32), dtype=np.float32); w /= np.linalg.norm(w, axis=1, keepdims=True)

    def make():
        L = ElasticLayer(1024, fan_in=32, capacity=600, theta=0.08)
        L.pre[:] = pre; L.w[:] = w; L.fill[:] = 32; L.n = 600
        return L
    stream = [(np.sort(rng.choice(1024, 32, replace=False)), rng.random(32, dtype=np.float32) * 0.9 + 0.1)
              for _ in range(200)]
    assert _lockstep(make, stream, learn=False) > 0
    assert L0.n == 0


def test_writing_synapses_by_hand_needs_reindex():
    """El índice es una foto: tras escribir `pre` a mano hay que llamar a reindex()."""
    rng = np.random.default_rng(4)
    L = ElasticLayer(4000, fan_in=16, capacity=400, theta=0.5)
    L.pre[:] = rng.integers(0, 4000, size=(400, 16)); L.w[:] = 0.25; L.fill[:] = 16; L.n = 400
    L.SPARSE_MIN_NEURONS = 1; L.SPARSE_MIN_SYNAPSES = 1
    idx = np.unique(L.pre[7]); val = np.ones(len(idx), dtype=np.float32)
    assert 7 in L.forward(idx, val).winners
    L.pre[7] = L.pre[8]; L.reindex()
    assert 7 in L.forward(np.unique(L.pre[8]), np.ones(len(np.unique(L.pre[8])), np.float32)).winners


def test_memory_accounting_matches_the_arrays():
    rng = np.random.default_rng(5)
    L = ElasticLayer(5000, fan_in=24, capacity=16)
    L.SPARSE_MIN_NEURONS = 1; L.SPARSE_MIN_SYNAPSES = 1
    for _ in range(300):
        idx = np.sort(rng.choice(5000, 20, replace=False))
        v = L.forward(idx, rng.random(20, dtype=np.float32) + 0.1)
        L.recruit(v)
    L.forward(idx, np.ones(20, np.float32))                 # construye el índice
    m = L.memory()
    arrays = [v for v in vars(L).values() if isinstance(v, np.ndarray)]
    arrays += list(L._ix)
    assert m["total"] == sum(a.nbytes for a in arrays)       # no queda memoria sin contar
    assert m["synapses"] == L.nbytes() and m["slack"] == (L.cap - L.n) * 24 * 8
    S = len(L._ix[1])                                        # sinapsis de la última foto del índice
    assert 0 < S <= int(L.fill[:L.n].sum())
    assert m["stats"] == L.cap * bp.BYTES_PER_NEURON
    assert m["index"] == S * bp.INDEX_BYTES_PER_SYNAPSE + (L.D + 1) * bp.INDEX_BYTES_PER_AFFERENT
    assert m["scratch"] == L.D * bp.SCRATCH_BYTES_PER_AFFERENT


def test_blueprint_separates_synapse_memory_from_total_and_per_synapse_from_total_cost():
    small = bp.plan(1e-4, syn_per_core_second=4e8, index=True, slack=1.5)
    big = bp.plan(1e-2, syn_per_core_second=4e8, index=True, slack=1.5)
    assert small["ns_per_synapse"] == big["ns_per_synapse"] == pytest.approx(2.5)    # constante
    assert big["core_seconds_per_step"] == pytest.approx(100 * small["core_seconds_per_step"])  # crece
    for p in (small, big):
        assert p["memory_total_bytes"] > 2 * p["memory_bytes"]           # 8 B/sinapsis subestima
        per_neuron = (p["memory_total_bytes"] - p["column_buffer_bytes"]) / p["neurons"]
        assert per_neuron == pytest.approx(bp.bytes_per_neuron(7000, index=True, slack=1.5))
    assert len(bp.NOT_DEMONSTRATED) >= 4


def test_cortex_cost_counts_evaluated_not_stored_synapses():
    rng = np.random.default_rng(6)
    net = Cortex((12, 12, 1), 3, areas=[dict(rf=3, pool=2, max_features=8, theta=0.8)],
                 heads=[dict(source="input")], projection=dict(source="all"), replay=0.5)
    for _ in range(60):
        x = (rng.random((12, 12, 1)) < 0.2).astype(np.float32)
        net.step(x, int(rng.integers(3)))
    a = net.areas[0]
    c = net.cost()
    assert c["parts"]["area0"]["macs"] == a.oh * a.ow * a.n * a.D       # cada posición evalúa el diccionario
    assert c["parts"]["area0"]["macs"] > 50 * net.stats()["areas"][0]["synapses"]
    L = net.heads[0].layer
    assert c["parts"]["head0"]["macs"] == L.n * int(L.fill[:L.n].max())
    assert c["macs_per_training_step"] > c["macs_per_inference"]          # el repaso cuesta
