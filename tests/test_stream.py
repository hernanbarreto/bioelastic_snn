import numpy as np
import pytest

from bioelastic.stream import (GROUPS, SCHEDULE, ContinualStream, first_appearance, forgetting,
                               reappearances, steps_to, transform, window_acc)


def _pools(n=12):
    rng = np.random.default_rng(0)
    return {c: rng.integers(0, 256, size=(n, 28, 28), dtype=np.uint8) for c in range(20)}


def test_default_schedule_is_long_and_non_stationary():
    s = ContinualStream(_pools())
    assert len(s) >= 200_000
    b = s.bounds()
    assert b[0][0] == 0 and b[-1][1] == len(s)
    assert all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))
    modes = {p.mode for p in SCHEDULE}; ctxs = {p.ctx for p in SCHEDULE}
    assert modes == {"sup", "reward", "unlab"}
    assert {"clean", "noise", "occlude", "rot90"} <= ctxs
    back = reappearances()
    assert len(back) >= 3 and all(k1 - k0 >= 2 for k0, k1, _ in back)
    assert any(ctx == "rot90" for _, _, ctx in back)             # un contexto que vuelve
    # las clases del sondeo no aparecen antes de la última fase
    first = first_appearance()
    assert first["P"] == len(SCHEDULE) - 1
    assert all(not (set(p.classes) | set(p.mix_classes)) & set(GROUPS["P"]) for p in SCHEDULE[:-1])


def test_stream_is_deterministic_and_follows_the_schedule():
    a = list(ContinualStream(_pools(), scale=0.002, seed=3))
    b = list(ContinualStream(_pools(), scale=0.002, seed=3))
    c = list(ContinualStream(_pools(), scale=0.002, seed=4))
    assert [e.y for e in a] == [e.y for e in b]
    assert all(np.array_equal(x.x, y.x) for x, y in zip(a, b))
    assert [e.y for e in a] != [e.y for e in c]
    for e in a:
        ph = SCHEDULE[e.phase]
        assert e.mode == ph.mode and e.ctx == ph.ctx
        assert e.y in (ph.classes if e.new else ph.mix_classes)
        assert e.x.shape == (28, 28) and 0.0 <= e.x.min() and e.x.max() <= 1.0


def test_probe_phase_mixes_new_and_known_classes():
    s = ContinualStream(_pools(), scale=0.05, seed=1)
    k = len(SCHEDULE) - 1
    evs = list(s.phase_events(k))
    new = np.array([e.new for e in evs])
    assert 0.4 < new.mean() < 0.6
    assert {e.y for e in evs if e.new} == set(GROUPS["P"])


def test_contexts():
    rng = np.random.default_rng(0)
    img = np.zeros((28, 28), dtype=np.float32); img[5:20, 10:14] = 1.0
    r = transform(img, "rot90", rng)
    assert np.array_equal(r, np.rot90(img))
    o = transform(np.ones((28, 28), dtype=np.float32), "occlude", rng)
    assert 0.3 < (o == 0).mean() < 0.5                      # bloque 12x12 + 20 % perdido
    n = transform(img, "noise", rng)
    assert n.min() >= 0 and n.max() <= 1 and np.abs(n - img).mean() > 0.05
    assert np.array_equal(transform(img, "clean", rng), img)
    with pytest.raises(ValueError):
        transform(img, "nope", rng)


def test_metrics():
    hist = [{"A": 0.9}, {"A": 0.8, "B": 0.0}, {"A": 0.5, "B": 0.7}, {"A": 0.6, "B": 0.6}]
    f = forgetting(hist)
    assert f["A"] == pytest.approx(0.3) and f["B"] == pytest.approx(0.1)
    assert "C" not in f
    hits = np.r_[np.zeros(50), np.ones(50)]
    assert window_acc(hits, 0, 50) == 0.0 and window_acc(hits, 25, 50) == 0.5
    assert steps_to(hits, 0, 100, 0.9, 10) == 59              # la ventana [49, 59) ya tiene 9 de 10
    assert steps_to(np.zeros(100), 0, 100, 0.9, 10) == 100
