import numpy as np
import pytest

from bioelastic import ElasticLayer, first_spike_times


def _volley(layer, s):
    s = np.asarray(s, dtype=np.float32)
    idx = np.flatnonzero(s > 0)
    return layer.forward(idx, s[idx])


def test_closed_form_matches_time_stepped_simulation():
    rng = np.random.default_rng(0)
    n, M = 30, 10
    W = rng.random((n, M))
    t = np.sort(rng.random(n) * 0.9)
    theta = rng.uniform(0.2, 0.95, M)
    gain = 1.0 / np.linalg.norm(1.0 - t)
    exact = first_spike_times(W, t, theta, gain)
    grid = np.linspace(0.0, 1.0, 100001)
    V = gain * (np.maximum(grid[:, None] - t[None, :], 0.0) @ W)        # (G, M)
    for m in range(M):
        crossed = V[:, m] >= theta[m]
        if crossed.any():
            assert exact[m] == pytest.approx(grid[crossed.argmax()], abs=2e-5)
        else:
            assert np.isinf(exact[m])


def test_final_potential_is_cosine_so_firing_iff_cos_reaches_threshold():
    rng = np.random.default_rng(1)
    W = rng.random((50, 20)); W /= np.linalg.norm(W, axis=0, keepdims=True)
    s = rng.random(50)
    order = np.argsort(-s)
    cos = (s @ W) / np.linalg.norm(s)
    theta = np.full(20, float(np.median(cos)))
    ts = first_spike_times(W[order], 1.0 - s[order], theta, 1.0 / np.linalg.norm(s))
    assert np.array_equal(np.isfinite(ts), cos >= theta)


def test_stronger_match_fires_earlier():
    layer = ElasticLayer(8, k=2, theta=0.3)
    a = np.array([1, .9, .8, .7, 0, 0, 0, 0], dtype=np.float32)
    b = np.array([1, .9, 0, 0, .8, .7, 0, 0], dtype=np.float32)
    layer.recruit(_volley(layer, a)); layer.recruit(_volley(layer, b))
    v = _volley(layer, a)
    assert list(v.winners) == [0, 1]
    assert v.times[0] < v.times[1]
    assert v.cos[0] == pytest.approx(1.0, abs=1e-5)


def test_novelty_and_recruit_imprint():
    layer = ElasticLayer(6, theta=0.7)
    a = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
    b = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    v = _volley(layer, a)
    assert v.novel and layer.n == 0
    j = layer.recruit(v)
    assert j == 0 and not _volley(layer, a).novel
    assert _volley(layer, b).novel                     # ortogonal: la capa queda en silencio


def test_stdp_hebbian_attracts_antihebbian_repels_and_norm_is_kept():
    layer = ElasticLayer(6, theta=0.2)
    a = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
    c = np.array([1, 1, 0, 0, 1, 0], dtype=np.float32)
    layer.recruit(_volley(layer, a))
    cos0 = _volley(layer, c).cos[0]
    layer.stdp(0, _volley(layer, c), +1.0)
    cos1 = _volley(layer, c).cos[0]
    assert cos1 > cos0
    layer.stdp(0, _volley(layer, c), -0.5)
    assert _volley(layer, c).cos[0] < cos1
    assert np.linalg.norm(layer.dense(0)) == pytest.approx(1.0, abs=1e-5)
    assert (layer.dense(0) >= 0).all()


def test_consolidation_lowers_plasticity_with_use():
    layer = ElasticLayer(6, theta=0.2, lr=0.3, lr_half=5)
    a = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
    c = np.array([0, 0, 1, 1, 1, 0], dtype=np.float32)
    layer.recruit(_volley(layer, a))
    w0 = layer.dense(0); layer.stdp(0, _volley(layer, c)); young = np.abs(layer.dense(0) - w0).sum()
    layer.wins[0] = 500
    w1 = layer.dense(0); layer.stdp(0, _volley(layer, c)); old = np.abs(layer.dense(0) - w1).sum()
    assert old < 0.2 * young


def test_critical_period_pruning_keeps_only_reused_neurons():
    layer = ElasticLayer(4, theta=0.7, probation=10, survival_hits=1)
    eye = np.eye(4, dtype=np.float32)
    for k in range(4):
        layer.recruit(_volley(layer, eye[k]))
    layer.credit(1); layer.credit(3)
    kept = {tuple(np.round(layer.dense(j), 3)) for j in (1, 3)}
    for _ in range(11):
        layer.tick()
    assert layer.maintain() == 2 and layer.n == 2
    assert {tuple(np.round(layer.dense(j), 3)) for j in range(2)} == kept


def test_budget_evicts_least_useful_and_capacity_growth_preserves_weights():
    layer = ElasticLayer(64, theta=0.7, max_neurons=40, capacity=4)
    eye = np.eye(64, dtype=np.float32)
    for k in range(40):
        layer.recruit(_volley(layer, eye[k])); layer.tick()
    assert layer.n == 40 and layer.cap >= 40
    assert all(layer.dense(k)[k] == pytest.approx(1.0) for k in range(40))
    for k in range(1, 40):
        layer.credit(k)                                # la 0 queda como la menos útil
    layer.recruit(_volley(layer, eye[50]))
    assert layer.n == 40 and layer.evicted == 1
    assert not any(layer.dense(j)[0] > 0.5 for j in range(layer.n))   # la del patrón 0 fue desalojada


def test_homeostasis_raises_threshold_of_frequent_winner():
    layer = ElasticLayer(4, theta=0.5, theta_plus=0.05, theta_max=0.2, theta_tau=100)
    a = np.array([1, 1, 0, 0], dtype=np.float32)
    layer.recruit(_volley(layer, a))
    for _ in range(10):
        layer.stdp(0, _volley(layer, a)); layer.tick()
    th = layer.thresholds()[0]
    assert 0.5 < th <= 0.7 + 1e-6
    for _ in range(1000):
        layer.tick()
    assert layer.thresholds()[0] == pytest.approx(0.5, abs=1e-3)   # decae al valor base
