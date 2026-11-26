"""Pruebas rápidas del arnés de integración (experiments/exp_integration.py).

El arnés es lo que decide si una falla es de la red o de la prueba, así que se verifica
con piezas controladas: un razonador que suma de verdad, percepciones que se equivocan a
propósito y datos sintéticos (no hace falta descargar nada).
"""
import sys
from pathlib import Path

import numpy as np

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
