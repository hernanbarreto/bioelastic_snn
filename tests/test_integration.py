"""Pruebas rápidas del arnés de integración (experiments/exp_integration.py).

El arnés es lo que decide si una falla es de la red o de la prueba, así que se verifica
con piezas controladas: un razonador que suma de verdad, percepciones que se equivocan a
propósito y datos sintéticos (no hace falta descargar nada).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import exp_integration as ei  # noqa: E402
from bioelastic import BioElasticNet  # noqa: E402
from bioelastic.thinker import Thinker  # noqa: E402


def _sinteticos(n_por_digito=20, seed=0):
    """Dígitos sintéticos 28x28: cada dígito es una barra (horizontal del 0 al 4, vertical
    del 5 al 9) en una posición propia, con un desplazamiento de +-1 píxel y ruido."""
    rng = np.random.default_rng(seed)

    def lote(n):
        X, y = [], []
        for d in range(10):
            for _ in range(n):
                img = rng.random((28, 28)) * 40
                r = 3 + 5 * (d % 5) + int(rng.integers(-1, 2))
                if d < 5:
                    img[r:r + 2, 4:24] = 255
                else:
                    img[4:24, r:r + 2] = 255
                X.append(img.astype(np.uint8)); y.append(d)
        return np.array(X), np.array(y, dtype=np.uint8)
    Xtr, ytr = lote(n_por_digito)
    Xte, yte = lote(n_por_digito // 2)
    p = rng.permutation(len(ytr))
    return ei.Digitos(n_pre=len(ytr) // 2, arrays=(Xtr[p], ytr[p], Xte, yte))


class _Sumador:
    """Razonador de referencia que suma de verdad (para probar el arnés, no la red)."""

    def reset(self):
        self.c = 0

    def answer(self, tokens, cue=0, explore=False):
        if tokens == ei.END:
            return self.c
        a, b = tokens
        s = a + (b - 10) + self.c
        self.c = s // 10
        return s % 10

    def feedback(self, target=None, reward=None):
        pass


def test_pureza_es_invariante_a_permutar_simbolos_y_la_cota_es_el_producto():
    conf = np.diag(np.arange(1, 11))
    perm = np.random.default_rng(0).permutation(10)
    assert ei.pureza(conf[:, perm]) == 1.0
    mitad = np.full((10, 10), 1); assert abs(ei.pureza(mitad) - 0.1) < 1e-12
    assert abs(ei.cota_independiente(0.9, 6, 0.5) - 0.9 ** 6 * 0.5) < 1e-12


def test_suma_con_percepcion_perfecta_y_razonador_exacto_es_exacta():
    datos = _sinteticos(4)
    rng = np.random.default_rng(1)
    m = ei.evaluar_suma(_Sumador(), ei.oraculo, datos, rng, 4, 50)
    assert m["exacta"] == 1.0 and m["imagen"] == 1.0 and m["todas_bien"] == 1.0


def test_un_error_de_percepcion_rompe_la_suma_y_queda_atribuido():
    datos = _sinteticos(4)
    rng = np.random.default_rng(2)
    ve_mal_el_3 = lambda img, d, explore: 8 if d == 3 else int(d)
    m = ei.evaluar_suma(_Sumador(), ve_mal_el_3, datos, rng, 3, 300)
    assert m["exacta_si_todas_bien"] == 1.0
    assert m["exacta_si_alguna_mal"] == 0.0          # 3 -> 8 siempre cambia el dígito de la columna
    assert abs(m["imagen"] - 0.9) < 0.05


def test_el_refuerzo_llega_con_el_acierto_de_cada_columna():
    datos = _sinteticos(4)
    rng = np.random.default_rng(3)
    recibido = []
    th = Thinker(21, registers=2, seed=0)
    r = ei.episodio_suma(th, ei.oraculo, datos, "train", rng, 3, True,
                         lambda pares, rew: recibido.append((len(pares), rew)))
    assert len(recibido) == 3 and all(n == 2 for n, _ in recibido)
    assert [rew > 0 for _, rew in recibido] == r["columnas"][:3]


def test_thinker_con_oraculo_aprende_la_suma_decimal_con_dos_registros():
    datos = _sinteticos(4)
    rng = np.random.default_rng(0)
    th = ei.thinker_suma(0, registers=2)
    ei.entrenar_suma(th, ei.oraculo, datos, rng, 1500)       # 0.70-0.87 con 4 semillas
    assert ei.evaluar_suma(th, ei.oraculo, datos, rng, 2, 200)["exacta"] > 0.5


def test_ligadura_con_oraculo_y_preguntas_con_otra_imagen():
    datos = _sinteticos(4)
    rng = np.random.default_rng(4)
    F = np.random.default_rng(7).permutation(10)
    th = Thinker(20, memory=True, seed=0)
    for _ in range(1500):
        ei.episodio_ligadura(th, ei.oraculo, datos, "train", rng, True, None, F)
    res = ei.evaluar_ligadura(th, ei.oraculo, datos, rng, F, 200)
    assert res[True]["acierto"] > 0.95 and res[False]["acierto"] > 0.95
    assert res[True]["todas_bien"] == 1.0


def test_la_vista_aprende_con_etiquetas_y_el_refuerzo_mueve_su_decision():
    datos = _sinteticos(20)
    v = ei.Vista(0, "depth1")
    v.entrenar(datos.X["pre"], datos.y["pre"])
    conf = v.confusion(datos.X["test"], datos.y["test"])
    assert np.trace(conf) / conf.sum() > 0.9
    img = datos.X["test"][0]
    s0 = v.simbolo(img)
    otro = (s0 + 1) % 10
    for _ in range(30):                               # premiar otro símbolo ante esa imagen
        v.reforzar(img, otro, +1.0); v.reforzar(img, s0, -1.0)
    assert v.simbolo(img) == otro


def test_memoria_asociativa_completa_la_otra_modalidad_desde_un_ejemplar_nuevo():
    rng = np.random.default_rng(0)
    A = rng.random((5, 40)) ** 4; B = rng.random((5, 30)) ** 4       # 5 conceptos, 2 modalidades
    mem = BioElasticNet(70, 1, seed=0, layer=dict(fan_in=70, theta=0.7))
    for _ in range(100):
        c = int(rng.integers(5))
        mem.observe(ei.conjunto(A[c] + 0.05 * rng.random(40), B[c] + 0.05 * rng.random(30)))
    hits = 0
    for c in range(5):
        rec = ei.completar(mem, A[c] + 0.05 * rng.random(40), "img", 40)
        hits += int(np.argmax([ei._unidad(rec) @ ei._unidad(b) for b in B]) == c)
    assert hits == 5
    assert ei.p_at_1([B[2]], np.array([2]), list(B), np.arange(5)) == 1.0


def test_homeostasis_de_salida_evita_que_un_simbolo_acapare_las_decisiones():
    """Una percepción sesgada (enseñada a decir siempre 0) sigue diciendo 0 si nada la
    corrige; con homeostasis de salida el umbral del símbolo que acapara sube (con suma
    cero) hasta que otros símbolos ganan decisiones."""
    datos = _sinteticos(10)

    def usados(h):
        v = ei.Vista(0, "depth1", homeostasis=h)
        v.entrenar(datos.X["pre"][:50], np.zeros(50, dtype=int))
        rng = np.random.default_rng(0)
        elegidos = set()
        for _ in range(300):
            img = datos.una("train", int(rng.integers(10)), rng)
            s = v.simbolo(img)
            elegidos.add(s)
            v.reforzar(img, s, 0.0)
        if h:
            assert abs(float(v.net.umbral.sum())) < 1e-3 and int(np.argmax(v.net.umbral)) == 0
        return len(elegidos)
    assert usados(0.0) == 1
    assert usados(0.05) >= 5


# ---------------------------------------------------------------- ronda 2 (exp_integration2.py)
import exp_integration2 as e2  # noqa: E402


class _VeMalEl3:
    """Percepción que confunde el 3 con el 8, con el 3 como segunda opción; anota lo que se
    le enseña."""
    k = 2

    def __init__(self):
        self.ensenado = []

    def ver(self, img, d, explore):
        return (8, 0.4, [3]) if int(d) == 3 else (int(d), 0.99, [(int(d) + 1) % 10])

    def ensenar(self, img, sym):
        self.ensenado.append(int(sym))


def test_percepcion_pasa_certeza_y_alternativas_y_aprende_lo_que_se_le_ensena():
    datos = _sinteticos(20)
    v = ei.Vista(0, "depth1")
    v.entrenar(datos.X["pre"], datos.y["pre"])
    for certeza in ("p", "pcos"):
        per = e2.Percepcion(v, k=3, certeza=certeza)
        img = datos.X["test"][0]
        s, c, alts = per.ver(img, -1, False)
        assert s == v.simbolo(img) and 0.0 <= c <= 1.0 and len(alts) == 2 and s not in alts
    otro = (s + 1) % 10
    for _ in range(10):
        per.ensenar(img, otro)
    assert v.simbolo(img) == otro and per.ensenadas == 10
    tau = per.umbral(datos.X["test"][:40], 0.2)
    cs = [per.ver(x, -1, False)[1] for x in datos.X["test"][:40]]
    assert abs(np.mean(np.array(cs) < tau) - 0.2) <= 0.05


def test_el_credito_simbolico_le_ensena_a_la_percepcion_el_digito_que_confundio():
    """Thinker que ya sabe (a + b) mod 10 y una percepción que confunde el 3 con el 8: cada
    error se explica por una sola ranura y la percepción recibe el 3 (nunca otra cosa)."""
    datos = _sinteticos(4)
    rng = np.random.default_rng(0)
    th = Thinker(20, seed=0)
    for _ in range(600):
        e2.episodio_escalera("L1", th, e2.Oraculo(), datos, "train", rng, True, dict(cred=False), {})
    per = _VeMalEl3(); st = {}
    for _ in range(200):
        e2.episodio_escalera("L1", th, per, datos, "train", rng, True, dict(cred=True), st)
    assert st["creditos"] >= 10 and st["creditos_bien"] == st["creditos"]
    assert set(per.ensenado) == {3}


def test_digitos_tipograficos_tienen_el_formato_de_mnist():
    tipo = e2.digitos_tipograficos(3, 0)
    if tipo is None:
        pytest.skip("no hay tipografías instaladas")
    X, y = tipo
    assert X.shape == (30, 28, 28) and X.dtype == np.uint8 and np.bincount(y).tolist() == [3] * 10
    from scipy import ndimage
    cy, cx = ndimage.center_of_mass(X.astype(np.float64).sum(0))
    assert abs(cy - 14) < 2 and abs(cx - 14) < 2
    assert (X[:, :2].max() == 0) and (X[:, :, :2].max() == 0)      # margen de MNIST


# ---------------------------------------------------------------- ronda 3 (exp_integration3.py)
import exp_integration3 as e3  # noqa: E402


def test_analisis_la_suma_mod_10_no_da_senal_por_ranura():
    an = e3.analisis()
    assert an["bits_I_a_d"] < 1e-9                          # L2/L3: una ranura sola no dice nada
    assert abs(an["bits_I_ab_d"] - np.log2(10)) < 1e-9      # las dos juntas lo dicen todo
    assert abs(an["bits_I_a_d_dado_b"] - np.log2(10)) < 1e-9  # L1: con b perfecto, todo
    assert 0.2 < an["bits_I_a_acarreo"] < 0.4               # el acarreo sí depende de cada sumando
    assert abs(e3.p_acarreo(1) - 0.45) < 1e-12 and abs(e3.p_acarreo(3) - 0.4995) < 1e-12
    assert e3.ventaja_por_ranura(0.1) == pytest.approx(0.0)
    assert e3.ventaja_por_ranura(1.0) == pytest.approx(1.0)


def test_thinker_oraculo_suma_explica_y_da_consistencia():
    datos = _sinteticos(4)
    m = e3.evaluar(e3.ThinkerOraculo(), e3.OraculoP(), datos, np.random.default_rng(0), 3, 50)
    assert m["exacta"] == 1.0
    th = e3.ThinkerOraculo(); th.reset()
    th.answer((8, 15))                                      # vio 8 donde había 3: dice 3, el objetivo es 8
    th.feedback(target=8, alternatives=[[3], [16]])
    assert th.input_credit == [(0, 3)]
    st = th.consistency(8, [np.arange(10), np.arange(10, 20)])[0]
    assert all(st[a, b] == (1 if (a + b) % 10 == 8 else -1) for a in range(10) for b in range(10))
    assert th.state_key() == (0,)


def test_episodio_con_digitos_dados_regla_nueva_y_objetivo_barajado():
    datos = _sinteticos(4)
    rng = np.random.default_rng(0)
    r = e3.episodio(e3.ThinkerOraculo(), e3.OraculoP(), datos, "test", rng, 0, False, {}, {},
                    digitos=([7, 2], [5, 1]))
    assert r["columnas"] == [True, True, True]              # 27 + 15 = 42 (y acarreo final 0)
    r = e3.episodio(e3.ThinkerOraculo(), e3.OraculoP(), datos, "test", rng, 0, False, dict(regla=3), {},
                    digitos=([7], [5]))
    assert r["columnas"][0] is False                         # con la regla nueva el dígito es 5, no 2
    f = e3.objetivos_barajados(0)
    ys = [f(0, 1, 4) for _ in range(500)]
    assert len(set(ys)) == 10 and abs(np.mean(np.array(ys) == 4) - 0.1) < 0.05


def test_codigo_y_fusiones():
    conf = np.eye(10, dtype=int)[:, [3, 1, 2, 0, 4, 5, 6, 7, 8, 9]] * 10
    assert list(e3.codigo(conf)) == [3, 1, 2, 0, 4, 5, 6, 7, 8, 9]
    conf2 = np.eye(10, dtype=int) * 10; conf2[1] = conf2[0]
    assert e3.fusiones(conf2) == 1


def test_credito_posterior_en_el_episodio_ensena_a_la_percepcion():
    """Thinker oráculo y percepción que ve 8 en vez de 3 (3 segunda opción): el crédito
    posterior le enseña 3 en esa ranura (y nada falso)."""
    datos = _sinteticos(4)
    rng = np.random.default_rng(0)

    class VeMal:
        k = 2

        def __init__(self):
            self.ensenado = []

        def ver(self, img, d, explore):
            d = int(d)
            s = 8 if d == 3 else d
            self.p = np.full(10, 0.01); self.p[s] = 0.6 if d == 3 else 0.9
            if d == 3:
                self.p[3] = 0.3
            self.p /= self.p.sum()
            return s, float(self.p[s]), []

        def ensenar(self, img, sym):
            self.ensenado.append(int(sym))

    per = VeMal(); st = {}
    cp = e3.CreditoPosterior(k=10, tau=0.5)
    for _ in range(100):
        e3.episodio(e3.ThinkerOraculo(), per, datos, "train", rng, 1, True, dict(posterior=cp), st)
    assert 3 in per.ensenado and st["creditos"] > 0
    malos = [(d, s) for d, s in st["pares"] if s != d]
    assert not malos


def test_entrenar_con_curriculo_respeta_el_presupuesto_de_columnas():
    from bioelastic.curriculum import Curriculum, sum_stages
    datos = _sinteticos(4)
    cu = Curriculum(sum_stages("mas0"), budget=60)
    curva = []
    n = e3.entrenar_curriculo(e3.ThinkerOraculo(), e3.OraculoP(), datos, np.random.default_rng(0), cu,
                              {}, {}, curva, 20, None, None)
    etapas = [f["etapa"] for f in curva]
    assert n > 0 and curva[-1]["ep"] >= 60 and etapas == sorted(etapas) and etapas[-1] == 3


# ---------------------------------------------------------------- MNIST-addition (exp_mnist_addition.py)
def test_posterior_de_la_suma_y_anclas():
    import exp_mnist_addition as ma
    u = np.ones(10) / 10
    q, q1, q2 = ma.posterior_suma(u, u, 0)                    # suma 0: sólo 0 + 0
    assert q1[0] == 1.0 and q2[0] == 1.0
    q, q1, q2 = ma.posterior_suma(u, u, 17)                   # suma 17: 8 + 9 o 9 + 8
    assert abs(q1[8] - 0.5) < 1e-12 and abs(q1[9] - 0.5) < 1e-12
    q, q1, q2 = ma.posterior_suma(np.eye(10)[3] * 0.9 + 0.01, u, 7, gamma=0.0)
    np.testing.assert_allclose(q1[:8], 1 / 8)                 # gamma 0: sólo cuenta la suma
    assert ma.suma_predicha(np.eye(10)[3], np.eye(10)[9]) == 12
    assert abs(sum(((10 - abs(y - 9)) / 100) ** 2 for y in range(19)) - ma.AZAR_PAR) < 1e-3


def test_fiabilidad_arranca_sin_creerle_a_la_percepcion_y_sube_si_acierta():
    import exp_mnist_addition as ma
    f = ma.Fiabilidad(modo="potencia", k=2.0, rate=0.05, lo=0.02, hi=1.0)
    p3, p4 = np.eye(10)[3] * 0.9 + 0.01, np.eye(10)[4] * 0.9 + 0.01
    assert f.actualizar(p3, p4, 7) == 0.02
    for _ in range(150):
        f.actualizar(p3, p4, 7)
    assert f.gamma > 0.9
    for _ in range(150):
        f.actualizar(p3, p4, 9)                               # ahora el par más probable no da la suma
    assert f.gamma < 0.1
