"""Predicción descendente (codificación predictiva) en FeatureArea y Cortex."""
import numpy as np
import pytest

from bioelastic import Cortex, FeatureArea


def _blob(rng, c, size=8):
    img = np.zeros((size, size, 1), dtype=np.float32)
    y, x = (1 + 3 * (c // 2), 1 + 3 * (c % 2))
    y += int(rng.integers(-1, 2)); x += int(rng.integers(-1, 2))
    img[max(y, 0):y + 3, max(x, 0):x + 3, 0] = rng.uniform(0.6, 1.0, (3, 3))[:min(3, size - y), :min(3, size - x)]
    return img


def _bar(rng, vertical, size=12, length=5):
    img = np.zeros((size, size, 1), dtype=np.float32)
    y, x = (int(v) for v in rng.integers(1, size - length, size=2))
    if vertical:
        img[y:y + length, x, 0] = 1.0
    else:
        img[y, x:x + length, 0] = 1.0
    return img


AREAS = [dict(rf=3, pool=2, max_features=16, theta=0.8, local=dict(lr=0.05)),
         dict(rf=2, pool=1, max_features=16, theta=0.5, floor=0.2, local=dict(lr=0.05))]


def _with_pred(areas, **pred):
    return [dict(a, predictive=dict(pred)) for a in areas]


# ---------------------------------------------------------------- área
def test_prediction_places_each_template_on_the_winning_simple_cell():
    a = FeatureArea((6, 6, 1), rf=3, pool=2, max_features=1, theta=0.5, floor=0.9, k=1,
                    predictive=dict(lr=0.0))
    patch = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    a.W[0] = patch.ravel() / np.linalg.norm(patch); a.alive[0] = True; a.n = 1
    a.G[0] = 2.0 * patch.ravel()
    x = np.zeros((6, 6, 1), dtype=np.float32); x[1:4, 2:5, 0] = patch   # diagonal en (1, 2)
    code = a.encode(x)
    Q = a.predict()[:, :, 0]
    q = int(np.argmax(code[:, :, 0].ravel()))
    c = float(code[:, :, 0].ravel()[q])
    assert c == pytest.approx(1.0, abs=1e-5)
    expect = np.zeros((6, 6), dtype=np.float32); expect[1:4, 2:5] = 2.0 * patch
    assert np.allclose(Q, expect, atol=1e-5)                 # sólo en el parche ganador
    pe = a.prediction_error()
    assert pe == pytest.approx(1.0, abs=1e-5)                # (x - 2x)^2 / x^2


def test_generative_weights_learn_to_explain_the_input():
    rng = np.random.default_rng(0)
    a = FeatureArea((12, 12, 1), rf=3, pool=2, max_features=16, theta=0.8,
                    predictive=dict(lr=0.3))
    pes = []
    for _ in range(400):
        a.encode(_bar(rng, bool(rng.integers(2)))); pes.append(a.prediction_error()); a.learn()
    assert pes[0] == pytest.approx(1.0)                      # al principio no predice nada
    assert np.mean(pes[-50:]) < 0.15                         # después explica casi toda la entrada
    assert np.isfinite(a.G).all()


def test_predictive_area_without_modulation_learns_its_dictionary_exactly_as_before():
    rng = np.random.default_rng(1)
    imgs = [_bar(rng, bool(c)) for c in rng.integers(2, size=150)]
    plain = FeatureArea((12, 12, 1), rf=3, pool=2, max_features=8, theta=0.85, local=dict(lr=0.1),
                        n_classes=2)
    pred = FeatureArea((12, 12, 1), rf=3, pool=2, max_features=8, theta=0.85, local=dict(lr=0.1),
                       n_classes=2, predictive=dict(lr=0.2))
    for x, c in zip(imgs, rng.integers(2, size=150)):
        assert np.array_equal(plain.encode(x), pred.encode(x))
        plain.learn(int(c)); pred.learn(int(c))
    assert np.array_equal(plain.W, pred.W) and np.abs(pred.G).sum() > 0


def test_surprise_raises_plasticity_and_is_bounded():
    rng = np.random.default_rng(2)
    a = FeatureArea((12, 12, 1), rf=3, pool=2, max_features=16, theta=0.8,
                    predictive=dict(lr=0.3, modulate=1.0, warmup=20, tau=50, gain_max=3.0))
    for _ in range(300):                                     # sólo barras horizontales
        a.encode(_bar(rng, False)); a.learn()
    calm = a.gain
    x = np.zeros((12, 12, 1), dtype=np.float32); x[2:9, 2:9, 0] = rng.random((7, 7)) > 0.5
    a.encode(x); a.learn()                                   # algo que nunca vio
    assert a.gain > 1.5 > calm and a.gain <= 3.0


def test_neurogenesis_follows_the_surprise():
    def births(neuro):
        rng = np.random.default_rng(3)
        a = FeatureArea((12, 12, 1), rf=3, pool=2, max_features=64, theta=0.95,
                        predictive=dict(lr=0.3, neuro=neuro, warmup=20, tau=50, gain_min=0.0))
        for _ in range(200):
            a.encode(_bar(rng, False)); a.learn()
        before = a.created
        for _ in range(30):
            a.encode(_bar(rng, False)); a.learn()
        calm = a.created - before
        before = a.created
        for _ in range(5):
            a.encode(_bar(rng, True)); a.learn()
        return calm, a.created - before
    calm0, novel0 = births(0.0)
    calm1, novel1 = births(2.0)
    assert calm1 <= calm0                                    # poco error: menos nacimientos
    assert novel1 >= novel0 and novel1 > 0                   # lo inesperado sí recluta


def test_forget_channel_also_cuts_the_prediction_of_that_channel():
    a = FeatureArea((6, 6, 3), rf=2, pool=1, max_features=4, theta=0.5, predictive=dict(lr=0.1))
    a.G[:] = 1.0
    a.forget_channel(1)
    G = a.G.reshape(4, 4, 3)
    assert (G[:, :, 1] == 0).all() and (G[:, :, [0, 2]] == 1).all()


def test_unknown_predictive_option_is_rejected():
    with pytest.raises(ValueError):
        FeatureArea((6, 6, 1), rf=3, pool=2, predictive=dict(lrr=0.1))
    with pytest.raises(ValueError):
        Cortex((8, 8, 1), 2, areas=AREAS[:1], feedback=dict(iter=1))


# ---------------------------------------------------------------- cortex
def _stream(net, n, seed=6, C=4):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = int(rng.integers(C)); out.append(net.step(_blob(rng, c), c))
    return out


KW = dict(heads=[dict(source="input")], projection=dict(source="all", balance=True), replay=0.5,
          joint=False, arbiter=0.01, seed=1)


def test_single_pass_prediction_does_not_change_the_feedforward_network():
    a = Cortex((8, 8, 1), 4, areas=AREAS, **KW)
    b = Cortex((8, 8, 1), 4, areas=_with_pred(AREAS, lr=0.2), **KW)
    assert _stream(a, 300) == _stream(b, 300)
    for x, y in zip(a.areas, b.areas):
        assert np.array_equal(x.W, y.W)
    assert all(np.isfinite(y.pe) and y.pe < 1.0 for y in b.areas)
    assert all(np.abs(y.G).sum() > 0 for y in b.areas)


def test_inference_loop_without_feedback_terms_equals_the_single_pass():
    a = Cortex((8, 8, 1), 4, areas=_with_pred(AREAS, lr=0.2), **KW)
    b = Cortex((8, 8, 1), 4, areas=_with_pred(AREAS, lr=0.2),
               feedback=dict(iters=2, up=0.0, context=0.0), **KW)
    assert _stream(a, 200) == _stream(b, 200)                # re-codificar lo mismo no cambia nada
    assert b.cost()["parts"]["area0"]["macs"] == 3 * a.cost()["parts"]["area0"]["macs"]
    assert "pred1" in b.cost()["parts"]


def test_context_fills_the_lower_code_with_the_top_down_prediction():
    areas = [AREAS[0], dict(AREAS[1], predictive=dict(lr=0.2))]   # sólo el área 2 predice
    net = Cortex((8, 8, 1), 4, areas=areas, feedback=dict(iters=1, context=0.5), **KW)
    _stream(net, 300)
    rng = np.random.default_rng(9)
    x = _blob(rng, 1)
    net.forward(x)
    lower = net._codes[0].copy()                             # lo que leen los lectores
    ff = net.areas[0].encode(x)                              # código ascendente puro
    net.areas[1].encode(ff)
    Q = net.areas[1].predict()                               # lo que el área 2 predice del área 1
    assert Q.max() > 0
    assert np.allclose(lower, np.maximum(ff + 0.5 * Q, 0.0), atol=1e-5)
    assert not np.allclose(lower, ff)


def test_residual_goes_up_when_asked():
    """Con `up` el área de arriba recibe código + up * (código - su predicción)."""
    areas = [AREAS[0], dict(AREAS[1], predictive=dict(lr=0.2))]   # sólo el área 2 predice
    net = Cortex((8, 8, 1), 4, areas=areas, feedback=dict(iters=1, up=4.0), **KW)
    _stream(net, 300)
    rng = np.random.default_rng(10)
    x = _blob(rng, 2)
    net.forward(x)
    lower = net._codes[0]
    a2 = net.areas[1]
    got = a2._x.reshape(lower.shape).copy()                  # lo que recibió el área 2
    a2.encode(lower)                                         # pase ascendente (iteración 0)
    Q = a2.predict()
    expect = np.maximum(lower + 4.0 * (lower - Q), 0.0)
    assert np.allclose(got, expect, atol=1e-5)
    assert not np.allclose(got, lower)


def test_deep_predictive_cortex_learns_and_roundtrips(tmp_path):
    net = Cortex((8, 8, 1), 4, areas=_with_pred(AREAS, lr=0.2, modulate=1.0, neuro=1.0),
                 feedback=dict(iters=1, up=1.0, context=0.2, source="code"), **KW)
    hits = _stream(net, 600)
    rng = np.random.default_rng(6)
    assert all(np.isfinite(a.G).all() for a in net.areas)
    path = tmp_path / "pred.npz"
    net.save(path)
    twin = Cortex.load(path)
    assert twin.fb == net.fb
    assert np.array_equal(twin.areas[1].G, net.areas[1].G)
    assert twin.areas[0].sur_avg == net.areas[0].sur_avg
    X = [_blob(rng, int(c)) for c in rng.integers(4, size=40)]
    for x in X:
        assert twin.predict(x) == net.predict(x)
    assert len(hits) == 600


def test_shuffled_surprise_is_a_control_with_the_same_distribution():
    net = Cortex((8, 8, 1), 4, areas=_with_pred(AREAS, lr=0.2, modulate=1.0),
                 feedback=dict(source="shuffle"), **KW)
    _stream(net, 200)
    assert all(len(h) == 200 for h in net._pe_hist)
    assert all(0.1 < a.gain_avg < 4.0 for a in net.areas)


def test_gain_only_amplifies_what_is_already_active():
    areas = [AREAS[0], dict(AREAS[1], predictive=dict(lr=0.2))]
    net = Cortex((8, 8, 1), 4, areas=areas, feedback=dict(iters=1, gain=1.0), **KW)
    _stream(net, 300)
    x = _blob(np.random.default_rng(11), 3)
    net.forward(x)
    lower = net._codes[0].copy()
    ff = net.areas[0].encode(x)
    net.areas[1].encode(ff)
    Q = net.areas[1].predict()
    assert np.allclose(lower, np.maximum(ff * (1.0 + Q), 0.0), atol=1e-5)
    assert ((lower > 0) <= (ff > 0)).all()                   # no aparece actividad nueva


# ---------------------------------------------------------------- F5: redundancia
def test_lateral_inhibition_learns_to_separate_features_that_fire_together():
    a = FeatureArea((5, 5, 1), rf=3, pool=1, max_features=3, theta=0.5, floor=0.1, k=2,
                    lateral=dict(lr=0.2, cand=2), neurogenesis=False, lr=0.0, lr_min=0.0)
    bar = np.zeros((3, 3), dtype=np.float32); bar[1, :] = 1.0
    dot = np.zeros((3, 3), dtype=np.float32); dot[0, 0] = 1.0
    for f, w in enumerate((bar, bar + 0.05, bar + 2 * dot)):  # dos casi iguales y uno distinto
        a.W[f] = w.ravel() / np.linalg.norm(w); a.alive[f] = True
    a.n = 3
    x = np.zeros((5, 5, 1), dtype=np.float32); x[2, 1:4, 0] = 1.0
    first = a.encode(x)[1, 1]
    assert (first[:2] > 0).all() and first[2] == 0           # al principio ganan los dos duplicados
    for _ in range(50):
        a.encode(x); a.lat_m[:] = 0.0; a.learn()
    assert a.V[0, 1] > 0 and a.V[1, 0] > 0 and np.all(np.diag(a.V) == 0)
    last = a.encode(x)[1, 1]
    assert (last[:2] > 0).sum() == 1 and last[2] > 0         # ahora uno de ellos cede su lugar


def test_lateral_and_residual_dictionary_are_off_by_default_and_roundtrip(tmp_path):
    a = FeatureArea((8, 8, 1), rf=3, pool=2, max_features=8, theta=0.8)
    assert a.lat is None and not hasattr(a, "V")
    net = Cortex((8, 8, 1), 4, areas=[dict(AREAS[0], lateral=dict(lr=0.05),
                                           predictive=dict(lr=0.2, dict_res=1.0)),
                                      dict(AREAS[1], predictive=dict(lr=0.2, dict_res=0.5))], **KW)
    hits = _stream(net, 300)
    assert np.mean(hits[-100:]) > 0.5 and net.areas[0].V.max() > 0
    net.save(tmp_path / "f5.npz")
    twin = Cortex.load(tmp_path / "f5.npz")
    assert np.array_equal(twin.areas[0].V, net.areas[0].V)
    rng = np.random.default_rng(4)
    for x in [_blob(rng, int(c)) for c in rng.integers(4, size=30)]:
        assert twin.predict(x) == net.predict(x)


def test_residual_dictionary_does_not_move_what_is_already_explained():
    rng = np.random.default_rng(5)
    a = FeatureArea((12, 12, 1), rf=3, pool=2, max_features=16, theta=0.8,
                    predictive=dict(lr=0.5, dict_res=1.0), neurogenesis=True)
    b = FeatureArea((12, 12, 1), rf=3, pool=2, max_features=16, theta=0.8)
    x = _bar(rng, False)
    for _ in range(200):                                     # un solo estímulo, aprendido de memoria
        a.encode(x); a.learn(); b.encode(x); b.learn()
    Wa, Wb = a.W.copy(), b.W.copy()
    a.encode(x); pe = a.prediction_error(); a.learn(); b.encode(x); b.learn()
    assert pe < 0.05
    assert np.abs(a.W - Wa).max() < 1e-4                     # lo ya explicado casi no se mueve


def test_conjunction_features_stay_on_their_support_and_learn():
    rng = np.random.default_rng(7)
    net = Cortex((8, 8, 1), 4, areas=[AREAS[0], dict(AREAS[1], support=6, theta=0.7)], **KW)
    hits = _stream(net, 400)
    a = net.areas[1]
    assert np.mean(hits[-100:]) > 0.6 and a.alive.sum() > 0
    assert (a.M.sum(axis=1)[a.alive] <= 6).all()
    assert not a.W[~a.M].any()                               # nada fuera del soporte
    assert a.cost()["macs"] == 2 * a.oh * a.ow * int(a.M[:a.n].sum())
    code = a.encode(net.areas[0].encode(_blob(rng, 1)))
    assert code.max() <= 1.0 + 1e-5


def test_passthrough_adds_the_pooled_input_to_the_code():
    net = Cortex((8, 8, 1), 4, areas=[AREAS[0], dict(AREAS[1], passthrough=True)], **KW)
    a1, a2 = net.areas
    assert a2.out_shape == (a2.ph, a2.pw, a2.F + a1.out_shape[2])
    hits = _stream(net, 400)
    assert np.mean(hits[-100:]) > 0.6
    x = _blob(np.random.default_rng(8), 2)
    c1 = a1.encode(x); c2 = a2.encode(c1)
    # cada célula compleja (pool 1, rf 2) ve un bloque 2x2 de la entrada: su máximo por canal
    assert np.allclose(c2[0, 0, a2.F:], c1[:2, :2].max(axis=(0, 1)))
    assert a2.potentials() is not None and np.isfinite(a2.potentials()).all()


def test_trace_rule_pulls_a_feature_toward_the_next_view():
    def run(trace):
        a = FeatureArea((10, 10, 1), rf=3, pool=2, max_features=4, theta=0.99, floor=0.85, k=1,
                        neurogenesis=False, lr=0.0, lr_min=0.0, trace=trace)
        h = np.zeros((3, 3), dtype=np.float32); h[1, :] = 1.0          # barra horizontal
        d = np.eye(3, dtype=np.float32)                                # diagonal
        a.W[0] = h.ravel() / np.linalg.norm(h); a.W[1] = d.ravel() / np.linalg.norm(d)
        a.alive[:2] = True; a.n = 2
        x1 = np.zeros((10, 10, 1), dtype=np.float32); x1[4, 3:6, 0] = 1.0   # vista 1: barra
        c = np.zeros((3, 3), dtype=np.float32); c[1, :2] = 1.0; c[2, 1] = 1.0   # la barra doblada
        x2 = np.zeros((10, 10, 1), dtype=np.float32); x2[4:7, 4:7, 0] = c   # vista 2: se movió y dobló
        a.reset_trace() if trace else None
        a.encode(x1); a.learn(); a.encode(x2); a.learn()
        return a, float(a.W[0].reshape(3, 3)[2].sum())       # peso fuera de la fila de la barra
    plain, c0 = run(None)
    tr, c1 = run(dict(lr=0.5, decay=0.5))
    assert c0 == 0.0 and c1 > 0.2                            # la barra aprendió de la vista siguiente
    assert tr._trace.max() > 0
    tr.reset_trace()
    assert not tr._trace.any()


def test_cortex_new_sequence_resets_area_traces():
    net = Cortex((8, 8, 1), 4, areas=[AREAS[0], dict(AREAS[1], trace=dict(lr=0.1))], **KW)
    _stream(net, 50)
    assert net.areas[1]._trace.any()
    net.new_sequence()
    assert not net.areas[1]._trace.any()
    assert net.areas[0].tr is None
