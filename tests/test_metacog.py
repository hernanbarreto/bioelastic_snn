"""Pruebas de las piezas de la ronda 3 de integración: metacognición (`bioelastic.metacog`,
opciones de `Cortex` y `Thinker`), crédito posterior y evidencia trans-situacional
(`bioelastic.posterior`) y currículo (`bioelastic.curriculum`). Todas apagadas por
defecto: también se prueba que sin ellas nada cambia."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bioelastic import Cortex  # noqa: E402
from bioelastic.curriculum import Curriculum, per_slot_information, sample_sum, sum_stages  # noqa: E402
from bioelastic.metacog import Calibrator, SurpriseTracker, auroc, ece  # noqa: E402
from bioelastic.posterior import (CoOccurrence, PosteriorCredit, gamma_for, joint_posterior,  # noqa: E402
                                  predictive_conflict, status_likelihood, top_candidates)
from bioelastic.thinker import Thinker  # noqa: E402

D2 = dict(areas=[dict(rf=5, pool=2, max_features=32, theta=0.95, floor=0.3, k=8, local=dict(lr=0.5))],
          heads=[dict(source="input")], projection=dict(source="all", balance=True), replay=0.5,
          joint=False, arbiter=0.01)


def _barras(n, seed=0, C=3):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        c = int(rng.integers(C)); img = rng.random((28, 28)) * 0.15
        img[4 + 7 * c:7 + 7 * c, 4:24] = 1.0
        X.append(img.reshape(28, 28, 1).astype(np.float32)); y.append(c)
    return X, y


# ---------------------------------------------------------------- metacog
def test_auroc_y_ece_en_casos_conocidos():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert np.isnan(auroc([0.1, 0.2], [0, 0]))
    e, curva = ece([0.9] * 10, [1] * 9 + [0])
    assert abs(e) < 1e-9 and curva == [(0.9, 0.9, 10)]
    e, _ = ece([0.9] * 10, [0] * 10)
    assert abs(e - 0.9) < 1e-9


def test_el_calibrador_aprende_que_una_confianza_alta_acierta_poco():
    cal = Calibrator(n_extra=0, lr=0.05)
    rng = np.random.default_rng(0)
    f = cal.features(0.95)
    assert abs(cal.predict(f) - 0.95) < 1e-6                 # empieza en la identidad
    for _ in range(3000):
        cal.update(f, rng.random() < 0.5)
    assert abs(cal.predict(f) - 0.5) < 0.08


def test_la_sorpresa_inesperada_sube_con_un_cambio_y_no_con_ruido_estable():
    rng = np.random.default_rng(0)
    tr = SurpriseTracker(slow=0.01, fast=0.1, warmup=100)
    ne_estable = [tr.update(tr.surprise(0.8, rng.random() < 0.8)) for _ in range(1500)]
    ne_cambio = [tr.update(tr.surprise(0.8, rng.random() < 0.2)) for _ in range(100)]
    assert max(ne_cambio) > 3.0 > np.mean(ne_estable)
    assert np.mean(np.array(ne_estable[200:]) > 3.0) < 0.02     # falsas alarmas raras
    assert tr.expected > 0.0


# ---------------------------------------------------------------- Cortex
def test_cortex_sin_opciones_no_cambia_y_expone_senales():
    X, y = _barras(60)
    a = Cortex((28, 28, 1), 3, seed=0, **D2)
    b = Cortex((28, 28, 1), 3, seed=0, **D2, metacog=None, hypercorrection=0.0, neuromod=None)
    for x, t in zip(X, y):
        assert a.step(x, t) == b.step(x, t)
    np.testing.assert_array_equal(a.heads[0].readout.W, b.heads[0].readout.W)
    s = a.signals()
    assert 0.0 <= s["conflict"] <= 1.0 and s["confidence"] == s["raw_confidence"]
    assert a.voter_probs().shape == (3, 3)                  # memoria, proyección y lector local
    assert "metacog" not in a._config


def test_cortex_calibrado_hipercorreccion_y_neuromodulacion():
    X, y = _barras(300, seed=1)
    net = Cortex((28, 28, 1), 3, seed=0, **D2, metacog=dict(lr=0.05), hypercorrection=1.0,
                 neuromod=dict(warmup=50))
    gains = []
    for x, t in zip(X, y):
        p = net.forward(x)
        c = net.raw_confidence()
        net.teach(t)
        if p != t:
            gains.append((c, net.lr_gain))
    assert net._cal.n == 300 and net.expected_uncertainty > 0.0
    assert all(g >= 1.0 for _, g in gains)
    if len(gains) > 2:                                        # más confianza en el error, más plasticidad
        cs, gs = np.array(gains).T
        assert np.corrcoef(cs, gs)[0, 1] > 0.0
    # al guardar y cargar se conserva el calibrador
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        net.save(Path(d) / "c.npz")
        m = Cortex.load(Path(d) / "c.npz")
    np.testing.assert_allclose(m._cal.w, net._cal.w)
    assert m.neuromod is not None and m.hypercorrection == 1.0


def test_hipercorreccion_plana_usa_la_ganancia_media_de_los_errores():
    net = Cortex((28, 28, 1), 3, seed=0, **D2, hypercorrection=1.0, hypercorrection_mode="plano")
    X, y = _barras(80, seed=2)
    vals = []
    for x, t in zip(X, y):
        p = net.forward(x); net.teach(t)
        if p != t:
            vals.append(net.lr_gain)
    m = net._err_conf
    assert m[1] == len(vals) and abs(vals[-1] - (1 + m[0] / m[1])) < 1e-9


# ---------------------------------------------------------------- Thinker
def _thinker_suma_mod10(seed=0, **kw):
    th = Thinker(20, seed=seed, **kw)
    rng = np.random.default_rng(seed)
    for _ in range(1500):
        a, b = rng.integers(10, size=2)
        th.reset(); th.answer((int(a), 10 + int(b)), explore=True); th.feedback(target=int((a + b) % 10))
    return th


def test_consistencia_conjunta_y_conflicto_predictivo():
    th = _thinker_suma_mod10()
    th.reset(); out = th.answer((3, 15))
    assert out == 8
    st = th.consistency(8, [np.arange(10), np.arange(10, 20)])[0]
    ok = [(x, z) for x in range(10) for z in range(10) if st[x, z] == 1]
    assert (3, 5) in ok and all((x + z) % 10 == 8 for x, z in ok) and len(ok) >= 8
    outs, sure = th.content_outputs([np.array([3, 4]), np.array([15])])
    assert outs[0, 0] == 8 and outs[1, 0] == 9 and sure.all()
    c = predictive_conflict(outs, sure, [np.array([0.6, 0.4]), np.array([1.0])], 8)
    assert abs(c - 0.4) < 1e-9
    conf, est = th.answer_confidence()
    assert 0.0 < conf <= 1.0 and est


def test_trust_bajo_hace_que_el_thinker_no_aprenda():
    th = Thinker(20, seed=0, doubt=0.5)
    th.reset(); th.answer((1, 12)); n0 = th.say.layer.n
    th.feedback(target=3, trust=0.1)
    assert th.abstained == 1 and th.say.layer.n == n0
    th.reset(); th.answer((1, 12)); th.feedback(target=3, trust=0.9)
    assert th.abstained == 1


def test_teach_content_escribe_y_deja_establecido():
    th = Thinker(20, seed=0)
    th.teach_content((2, 13), 5, reg=np.zeros(0, dtype=np.int64))
    th.reset(); assert th.answer((2, 13)) == 5
    assert th._sim("say", th._content((2, 13)))[1]


def test_thinker_hipercorreccion_y_reinicio_por_sorpresa():
    th = Thinker(10, seed=0, hypercorrect=1.0,
                 neuromod=dict(slow=0.01, fast=0.2, ne_reset=3.0, window=50, warmup=100))
    rng = np.random.default_rng(0)
    eps0 = th.epsilon
    for i in range(800):
        x = int(rng.integers(10))
        th.reset(); th.answer((x,), explore=True)
        th.feedback(target=(x + (5 if i >= 500 else 0)) % 10)
    assert th.resets >= 1                                     # el cambio de regla disparó NE
    assert th.epsilon == eps0 or th._reset_left > 0
    assert th.signals()["expected_uncertainty"] > 0.0
    th0 = Thinker(10, seed=0)                                 # sin opciones: sin señales
    assert th0.signals()["unexpected_uncertainty"] == 0.0 and th0.resets == 0


# ---------------------------------------------------------------- posterior
def test_coocurrencia_con_decaimiento_y_confianza():
    t = CoOccurrence(5, decay=10.0, alpha=1.0)
    for _ in range(6):
        t.add("k", 2); t.tick()
    assert t.confident("k", 3, 0.8) == (2, True)
    np.testing.assert_allclose(t.prob("nada"), np.full(5, 0.2))
    c0 = t.evidence("k")
    for _ in range(10):
        t.tick()
    assert abs(t.evidence("k") - c0 / 2) < 1e-9


def test_posterior_conjunto_y_marginales():
    L = np.array([[1.0, 0.0], [0.0, 1.0]])
    q, (qa, qb) = joint_posterior([np.array([0.9, 0.1]), np.array([0.5, 0.5])], L)
    np.testing.assert_allclose(q, [[0.9, 0.0], [0.0, 0.1]])
    np.testing.assert_allclose(qa, [0.9, 0.1]); np.testing.assert_allclose(qb, [0.9, 0.1])
    q2, _ = joint_posterior([np.array([0.9, 0.1]), np.array([0.5, 0.5])], L, gamma=0.0)
    np.testing.assert_allclose(q2, [[0.5, 0.0], [0.0, 0.5]])
    assert list(top_candidates(np.array([0.1, 0.5, 0.4]), 2, keep=0)) == [1, 0]
    np.testing.assert_allclose(status_likelihood(np.array([1, -1, 0])), [0.9, 0.01, 0.1])
    g = gamma_for(0.95, 0.8)
    assert 0.0 < g < 1.0


def test_credito_posterior_corrige_la_ranura_que_el_conocimiento_desmiente():
    """Thinker que sabe (a+b) mod 10; la percepción vio 8 donde había un 3 (con 3 como
    segunda opción): el posterior le enseña 3 a esa ranura y el Thinker no aprende de ese
    episodio."""
    th = _thinker_suma_mod10()
    pc = PosteriorCredit(n_out=20, k=10, tau=0.5)
    pa = np.full(10, 0.01); pa[8] = 0.6; pa[3] = 0.32
    pb = np.full(10, 0.01); pb[5] = 0.91
    th.reset(); th.answer((8, 15))                            # 8 + 5 = 3: dice 3; el objetivo es 8
    res = pc.step(th, (8, 5), (pa / pa.sum(), pb / pb.sum()), 8)
    assert res["teach"][0][0] == 3 and res["trust"] < 0.5 and th.abstained == 1
    assert res["teach"][1] is None or res["teach"][1][0] == 5


def test_credito_posterior_con_tabla_consolida_en_el_thinker():
    """Cada episodio es ambiguo (cuatro asignaciones igual de probables, así que el Thinker
    no aprende de ninguno: tau_thinker bajo), pero la co-ocurrencia acumulada sí alcanza
    para escribir el conocimiento."""
    th = Thinker(20, seed=0)
    pc = PosteriorCredit(n_out=20, k=2, half_life=1000, consolidate=(3, 0.8), tau=0.99, use_thinker=False,
                         tau_thinker=0.2)
    pa = np.full(10, 0.0); pa[[4, 6]] = 0.5
    pb = np.full(10, 0.0); pb[[1, 2]] = 0.5
    for i in range(16):                                       # el 4 + 1 siempre responde 5 (ambiguo cada vez)
        th.reset(); th.answer((4, 11))
        pc.step(th, (4, 1), (pa, pb), 5)
    assert pc.n["consolidated"] >= 1 and th.abstained == 16
    th.reset(); assert th.answer((4, 11)) == 5
    for key in ((4, 1), (6, 2), (4, 2), (6, 1)):              # la evidencia se repartió entre las 4
        assert abs(pc.table.evidence(key) - 4.0) < 0.1


# ---------------------------------------------------------------- currículo
def test_curriculo_por_etapas_y_presupuesto():
    cu = Curriculum(sum_stages("mas0"), budget=1000)
    assert cu.stage(0) == 0 and cu.stage(149) == 0 and cu.stage(151) == 1 and cu.stage(10 ** 6) == 3
    rng = np.random.default_rng(0)
    a, b = sample_sum(cu.params(0), rng)
    assert len(a) == 1 and b[0] == 0
    for _ in range(50):
        a, b = sample_sum(Curriculum(sum_stages("conteo"), 100).params(0), rng)
        assert a.max() <= 2 and b.max() <= 2
    assert per_slot_information(range(10), range(10)) < 1e-9
    assert abs(per_slot_information(range(10), [0]) - np.log2(10)) < 1e-9
    assert per_slot_information(range(3), range(3)) > 0.5
    with pytest.raises(ValueError):
        sum_stages("nada")


def test_consistencia_con_prev_io_conserva_el_estado_previo():
    th = Thinker(20, seed=0, prev_io=True)
    rng = np.random.default_rng(0)
    for _ in range(300):
        a, b = rng.integers(10, size=2)
        th.reset(); th.answer((int(a), 10 + int(b)), explore=True); th.feedback(target=int((a + b) % 10))
    th.reset(); th.answer((3, 15))
    st = th.consistency(8, [np.arange(10), np.arange(10, 20)])
    assert st.shape == (1, 10, 10) and set(np.unique(st)) <= {-1, 0, 1}


# ---------------------------------------------------------------- objetivo blando y posterior conjunto
def test_objetivo_blando_en_la_lectura_y_en_el_cortex():
    from bioelastic.readout import Readout
    r = Readout(3, consolidation=0.0)
    r.add_input(None); r.add_input(None)
    idx = np.array([0, 1]); h = np.array([0.5, 0.5], dtype=np.float32)
    r.learn(idx, h, 0, soft=np.array([0.5, 0.5, 0.0]))
    assert r.W[0, 0] > 0 and r.W[1, 0] > 0 and r.W[2, 0] < 0      # sube lo que el objetivo pide, baja el resto
    r2 = Readout(3, consolidation=0.0); r2.add_input(None); r2.add_input(None)
    r2.learn(idx, h, 1, soft=np.eye(3)[1])
    r3 = Readout(3, consolidation=0.0); r3.add_input(None); r3.add_input(None)
    r3.learn(idx, h, 1)
    np.testing.assert_allclose(r2.W, r3.W)                        # one-hot blando == objetivo duro
    X, y = _barras(150, seed=3)
    net = Cortex((28, 28, 1), 3, seed=0, **D2)
    for x, t in zip(X, y):                                        # sólo objetivos blandos (ruidosos)
        net.forward(x)
        q = np.full(3, 0.15); q[t] = 0.7
        net.teach(soft=q)
    assert np.mean([net.predict(x) == t for x, t in zip(*_barras(60, seed=4))]) > 0.8


def test_hard_teach_es_igual_con_y_sin_la_ruta_blanda():
    X, y = _barras(40, seed=5)
    a = Cortex((28, 28, 1), 3, seed=0, **D2); b = Cortex((28, 28, 1), 3, seed=0, **D2)
    for x, t in zip(X, y):
        a.forward(x); a.teach(t)
        b.forward(x); b.teach(target=t, soft=None)
    np.testing.assert_array_equal(a.heads[0].readout.W, b.heads[0].readout.W)


def test_familiaridad_baja_ante_lo_nunca_visto():
    X, y = _barras(100, seed=6)
    net = Cortex((28, 28, 1), 3, seed=0, **D2)
    for x, t in zip(X, y):
        net.step(x, t)
    net.forward(X[0]); f_conocido = net.familiarity()
    rara = np.zeros((28, 28, 1), dtype=np.float32); rara[::3, ::3] = 1.0
    net.forward(rara)
    assert net.familiarity() < f_conocido and "familiarity" in net.signals()


def test_posterior_de_cadena_exacto_con_acarreo_final():
    from bioelastic.posterior import chain_posterior, known_sum_factor
    u = [np.ones(10) / 10] * 2
    # 95 + 17 = 112 -> dígitos [2, 1], acarreo final 1; con percepción uniforme
    qa, qb, Q, cin, lik = chain_posterior(u, u, [known_sum_factor(2), known_sum_factor(1)], np.array([0.0, 1.0]))
    assert abs(sum(x.sum() for x in qa) - 2) < 1e-9 and lik > 0
    # el acarreo final 1 obliga a a1 + b1 + c1 >= 10: con dígito 1, a1 + b1 + c1 = 11
    assert all(Q[1][a, b] == 0 for a in range(10) for b in range(10) if a + b not in (10, 11))
    # con percepción casi segura, la marginal recupera los dígitos
    pa = [np.eye(10)[5] * 0.9 + 0.01, np.eye(10)[9] * 0.9 + 0.01]
    pb = [np.eye(10)[7] * 0.9 + 0.01, np.eye(10)[1] * 0.9 + 0.01]
    qa, qb, _, cin, _ = chain_posterior(pa, pb, [known_sum_factor(2), known_sum_factor(1)], np.array([0.0, 1.0]))
    assert [int(np.argmax(x)) for x in qa] == [5, 9] and [int(np.argmax(x)) for x in qb] == [7, 1]
    assert cin[1][1] > 0.99


def test_regla_de_cadena_aprendida_con_percepcion_perfecta():
    """Con símbolos perfectos, la regla con acarreo latente aprende la suma de varias cifras
    sólo de los dígitos de cada columna y el acarreo final."""
    from bioelastic.posterior import ChainRule
    rng = np.random.default_rng(0)
    r = ChainRule(seed=0)
    oh = lambda d: np.eye(10)[d] * 0.999 + 0.0001  # noqa: E731
    for _ in range(3000):
        n = int(rng.integers(1, 4)); A = rng.integers(10, size=n); B = rng.integers(10, size=n)
        digs = []; c = 0
        for a, b in zip(A, B):
            s = int(a + b + c); digs.append(s % 10); c = s // 10
        r.learn([oh(a) for a in A], [oh(b) for b in B], digs, c)
    ok = 0
    for _ in range(200):
        A = rng.integers(10, size=3); B = rng.integers(10, size=3)
        r.reset(); c = 0; bien = True
        for a, b in zip(A, B):
            s = int(a + b + c); bien &= r.answer((int(a), 10 + int(b))) == s % 10; c = s // 10
        bien &= r.answer(20) == c
        ok += bien
    assert ok / 200 > 0.9


def test_regla_de_cadena_ne_detecta_el_cambio_de_regla_y_olvida():
    from bioelastic.posterior import ChainRule
    rng = np.random.default_rng(1)
    r = ChainRule(seed=0, neuromod=dict(warmup=50))
    oh = lambda d: np.eye(10)[d] * 0.999 + 0.0001  # noqa: E731
    ne = []
    for e in range(1600):
        k = 3 if e >= 1200 else 0
        n = int(rng.integers(1, 3)); A = rng.integers(10, size=n); B = rng.integers(10, size=n)
        digs = []; c = 0
        for a, b in zip(A, B):
            s = int(a + b + c); digs.append((s + k) % 10); c = s // 10
        ev0 = r.T.sum()
        r.learn([oh(a) for a in A], [oh(b) for b in B], digs, c)
        ne.append(r.unexpected_uncertainty)
    assert max(ne[1200:1300]) > 3.0 > np.mean(ne[800:1200])
    assert r.resets >= 1
    pd = r.answer_distribution(oh(2), oh(3))
    assert pd.shape == (10,) and abs(pd.sum() - 1) < 1e-9


def test_distribucion_de_respuesta_del_oraculo_y_compuerta_ne():
    th = e3_oraculo()
    th.reset(); th.answer((4, 15))
    pd = th.answer_distribution(np.eye(10)[2], np.eye(10)[3])   # con acarreo 0: 2 + 3 = 5
    assert pd[5] == 1.0


def e3_oraculo():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    import exp_integration3 as e3
    return e3.ThinkerOraculo()


def test_equiparticion_favorece_los_simbolos_subusados():
    from bioelastic.posterior import Equipartition
    eq = Equipartition(3, beta=1.0, rate=0.5)
    for _ in range(20):
        eq(np.array([0.9, 0.05, 0.05]))                       # el símbolo 0 acapara
    q = eq(np.array([1 / 3, 1 / 3, 1 / 3]))
    assert q[0] < 1 / 3 < q[1]


def test_compuerta_de_fiabilidad_sube_con_los_aciertos_y_baja_con_el_error_sostenido():
    from bioelastic.posterior import ReliabilityGate
    g = ReliabilityGate(chance=0.067, k=2.0, rate=0.05, lo=0.02)
    assert g.update(False) == 0.02                              # sin evidencia: casi no pesa
    for _ in range(200):
        g.update(True)
    alto = g.gamma
    assert alto > 0.9
    rng = np.random.default_rng(0)
    for _ in range(300):                                        # el código «acierta la mitad»
        g.update(rng.random() < 0.5)
    assert 0.1 < g.gamma < 0.45 < alto


def test_labilizar_libera_la_consolidacion_y_la_ganancia_escala_la_tasa():
    X, y = _barras(80, seed=7)
    net = Cortex((28, 28, 1), 3, seed=0, **D2)
    for x, t in zip(X, y):
        net.step(x, t)
    c0 = float(net.heads[0].readout.count[:net.heads[0].readout.n].sum())
    assert c0 > 0
    net.labilize(0.5)
    assert abs(float(net.heads[0].readout.count[:net.heads[0].readout.n].sum()) - 0.5 * c0) < 1e-3 * c0
    a = Cortex((28, 28, 1), 3, seed=0, **D2); b = Cortex((28, 28, 1), 3, seed=0, **D2)
    a.forward(X[0]); a.teach(int(y[0])); b.forward(X[0]); b.teach(int(y[0]), gain=1.0)
    np.testing.assert_array_equal(a.proj.W, b.proj.W)             # ganancia 1: igual que antes
    lr = b.proj.lr
    b.forward(X[1]); b.teach(int(y[1]), gain=3.0)
    assert b.proj.lr == lr                                         # la tasa vuelve a su valor
