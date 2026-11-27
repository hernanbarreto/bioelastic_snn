"""Tests del arnés de la batería de medición (experiments/battery)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from battery import almacen, esquema, registro  # noqa: E402
from battery.base import NoAplicable, Prueba  # noqa: E402


# ---------------------------------------------------------------- índice normalizado
def test_indice_sin_recortar():
    assert esquema.indice(0.75, 0.5, 1.0) == pytest.approx(0.5)
    assert esquema.indice(1.2, 0.5, 1.0) == pytest.approx(1.4)        # sobre la referencia
    assert esquema.indice(0.4, 0.5, 1.0) == pytest.approx(-0.2)       # bajo el azar
    assert esquema.indice(0.7, 0.5, 0.5) is None                      # referencia = azar
    assert esquema.indice(None, 0.5, 1.0) is None


def test_indice_intervalo_puntual_rango_y_cota():
    pt = esquema.indice_intervalo([0.7, 0.9], 0.5, dict(id="r", valor=0.9, tipo_valor="media"))
    assert pt["min"] == pt["max"] == pytest.approx(0.75)
    assert pt["desvio"] == pytest.approx(0.25)
    rg = esquema.indice_intervalo([0.65], 0.5, dict(id="r", valor_min=0.55, valor_max=0.75, tipo_valor="rango"))
    assert rg["min"] == pytest.approx(0.6) and rg["max"] == pytest.approx(3.0)
    # cota inferior de la referencia: el índice contra ella es la punta alta; la baja, contra el techo
    ct = esquema.indice_intervalo([0.5], 0.05, dict(id="r", valor=0.955, tipo_valor="cota_inferior"), techo=1.0)
    assert ct["max"] == pytest.approx(0.45 / 0.905)
    assert ct["min"] == pytest.approx(0.45 / 0.95)


def test_comparar_especies():
    r = dict(id="x", valor=0.8, tipo_valor="media")
    assert esquema.comparar(0.9, 0.01, r) == "especie por debajo"
    assert esquema.comparar(0.7, 0.01, r) == "especie por encima"
    assert esquema.comparar(0.79, 0.02, r) == "a la par"
    assert esquema.comparar(5.0, 0.0, r, mayor_es_mejor=False) == "especie por encima"
    assert esquema.comparar(0.9, 0.0, dict(id="q", valor=None, tipo_valor="cualitativo")) == "cualitativa"
    assert esquema.comparar(0.95, 0.0, dict(id="c", valor=0.9, tipo_valor="cota_inferior")).startswith("no concluyente")


# ---------------------------------------------------------------- catálogo
def test_catalogo_consistente():
    cat = esquema.cargar_catalogo()
    dims = set(cat["dimensiones"])
    niveles = {"primaria", "secundaria", "sin_verificar", "primaria (resumen)"}
    for pid, ent in cat["pruebas"].items():
        assert ent["dimension"] in dims, pid
        assert "valor" in ent["azar"] and ent["azar"].get("derivacion"), pid
        ids = [r["id"] for r in esquema.referencias(ent)]
        assert len(ids) == len(set(ids)), f"ids repetidos en {pid}"
        for r in esquema.referencias(ent):
            assert r["verificacion"] in niveles, (pid, r["id"])
            assert r.get("url", "").startswith("http"), (pid, r["id"])
            if r["verificacion"] == "sin_verificar":
                assert r.get("valor") is None, (pid, r["id"])
        ri = ent.get("referencia_indice")
        if ri:
            r = esquema.buscar(ent, ri)
            assert r.get("releida"), f"{pid}: la referencia del índice no fue releída"
            assert r.get("valor") is not None or r.get("valor_min") is not None


# ---------------------------------------------------------------- azar
def test_azar_analitico_reversion():
    from battery.reversion import azar_errores
    assert azar_errores(10) == pytest.approx(5.0)                      # 10 ensayos siempre
    assert azar_errores(20) < 10.0                                     # a veces llega al criterio
    rng = np.random.default_rng(0)                                     # contra una simulación
    tot = 0
    for _ in range(4000):
        h = []
        for _ in range(40):
            h.append(rng.random() < 0.5)
            if len(h) >= 10 and sum(h[-10:]) >= 9:
                break
        tot += len(h) - sum(h)
    assert tot / 4000 == pytest.approx(azar_errores(40), rel=0.03)


def test_ajuste_regla_delta_recupera_tasas():
    from battery.reversion import ajustar_delta, calendario
    rng = np.random.default_rng(3)
    prob, fase = calendario(rng, 0)
    o = rng.random(len(prob)) < prob
    v, x, c = np.empty(len(o)), 0.5, np.empty(len(o), dtype=bool)
    al = np.where(fase == 0, 0.1, 0.5)
    for t in range(len(o)):
        v[t] = x
        c[t] = rng.random() < 1 / (1 + np.exp(-12 * (x - 0.5)))
        x += al[t] * (o[t] - x)
    fit = ajustar_delta(c, o, fase)
    assert fit["alfa_volatil"] > fit["alfa_estable"]


# ---------------------------------------------------------------- filas, almacén y perfil
def _cat_falso():
    return {"dimensiones": {"d1": "D1", "d2": "D2"},
            "convenciones": {"indice_global": "promedio simple"},
            "pruebas": {
                "p1": {"nombre": "P1", "dimension": "d1", "azar": {"valor": 0.5, "derivacion": "dos"},
                       "referencia_indice": "h", "humano": [{"id": "h", "especie": "humano", "valor": 0.9,
                                                              "tipo_valor": "media", "releida": "hoy",
                                                              "verificacion": "primaria", "url": "http://x"}],
                       "animales": [{"id": "a", "especie": "paloma", "valor": 0.45, "tipo_valor": "media",
                                     "verificacion": "primaria", "url": "http://x"}], "maquinas": []},
                "p2": {"nombre": "P2", "dimension": "d2", "azar": {"valor": 0.0, "derivacion": "cero"},
                       "referencia_indice": "r", "humano": [{"id": "r", "especie": "humano", "valor_min": 1.0,
                                                              "valor_max": 2.0, "tipo_valor": "rango",
                                                              "releida": "hoy", "verificacion": "primaria",
                                                              "url": "http://x"}], "animales": [], "maquinas": []}}}


def _prueba(pid="p1", fn=None, version=1, equivalente=True, **kw):
    llamadas = []

    def correr(semilla, escala=1.0):
        llamadas.append(semilla)
        if fn is not None:
            return fn(semilla)
        return dict(puntaje=0.5 + 0.1 * semilla, medidas=dict(x=float(semilla)))
    p = Prueba(id=pid, variante="v", condicion="simbolos", metrica="m", protocolo="pr", desviaciones=[],
               correr=correr, version=version, semillas=(0, 1, 2), equivalente=equivalente,
               comparar={"a": "puntaje"}, **kw)
    return p, llamadas


def test_fila_esquema_e_indice(tmp_path):
    cat = _cat_falso()
    p, _ = _prueba()
    alm = almacen.Almacen("c0", tmp_path)
    crudos, sel, completa = registro.correr(p, alm, 1.0, log=lambda *a: None)
    assert completa and sel is None
    f = registro.fila(p, crudos, sel, cat, "c0")
    assert esquema.validar_fila(f) == []
    assert f["red"]["media"] == pytest.approx(0.6)
    assert f["indice_normalizado"]["media"] == pytest.approx((0.6 - 0.5) / 0.4)
    assert f["referencias_animales"][0]["comparacion"] == "especie por debajo"
    fila_csv = esquema.fila_csv(f)
    assert fila_csv["especies_por_debajo"].startswith("paloma")


def test_almacen_reanudable(tmp_path):
    p, llamadas = _prueba()
    alm = almacen.Almacen("c0", tmp_path)
    registro.correr(p, alm, 1.0, log=lambda *a: None)
    assert llamadas == [0, 1, 2]
    registro.correr(p, alm, 1.0, log=lambda *a: None)                 # todo guardado: no corre nada
    assert llamadas == [0, 1, 2]
    p2, ll2 = _prueba(version=2)                                       # otra versión del adaptador
    registro.correr(p2, alm, 1.0, log=lambda *a: None)
    assert ll2 == [0, 1, 2]
    p3, ll3 = _prueba(version=2)                                       # otra escala
    registro.correr(p3, alm, 0.5, log=lambda *a: None)
    assert ll3 == [0, 1, 2]
    alm.ruta(p.clave, 1).write_text("{cortado")                        # archivo cortado: se rehace
    p4, ll4 = _prueba(version=2)
    registro.correr(p4, alm, 0.5, log=lambda *a: None)
    assert ll4 == [1]
    # límite de tiempo vencido: no empieza semillas nuevas y lo informa
    p5, ll5 = _prueba(pid="p2", version=7)
    _, _, completa = registro.correr(p5, alm, 1.0, limite=0.0, log=lambda *a: None)
    assert ll5 == [] and not completa


def test_no_aplicable(tmp_path):
    def falla(semilla):
        raise NoAplicable("falta la API")
    p, _ = _prueba(fn=falla)
    crudos, _, _ = registro.correr(p, almacen.Almacen("c0", tmp_path), 1.0, log=lambda *a: None)
    f = registro.fila(p, crudos, None, _cat_falso(), "c0")
    assert f["estado"] == "no aplicable hoy" and "falta la API" in f["motivo"]
    assert esquema.validar_fila(f) == []


def test_seleccion_en_validacion(tmp_path):
    vistos = []

    def validar(cfg, semilla, escala):
        vistos.append(semilla)
        return cfg["calidad"]
    p, _ = _prueba(configs={"mala": {"calidad": 0.2}, "buena": {"calidad": 0.8}}, validar=validar)

    def correr(semilla, escala=1.0, config=None):
        return dict(puntaje=config["calidad"], medidas={})
    p.correr = correr
    crudos, sel, _ = registro.correr(p, almacen.Almacen("c0", tmp_path), 1.0, log=lambda *a: None)
    assert sel["elegida"] == "buena"
    assert set(vistos) == set(p.semillas_validacion) and not set(vistos) & set(p.semillas)
    assert all(c["config"] == "buena" and c["puntaje"] == 0.8 for c in crudos)


def test_perfil_promedio_simple_y_rangos(tmp_path):
    cat = _cat_falso()
    p1, _ = _prueba("p1")
    p2, _ = _prueba("p2", fn=lambda s: dict(puntaje=2.0, medidas={}), equivalente=False)
    alm = almacen.Almacen("c0", tmp_path)
    filas = [registro.fila(p, registro.correr(p, alm, 1.0, log=lambda *a: None)[0], None, cat, "c0")
             for p in (p1, p2)]
    pf = esquema.perfil(filas, cat)
    d1 = pf["todas"]["por_dimension"]["d1"]
    d2 = pf["todas"]["por_dimension"]["d2"]
    assert d1["min"] == d1["max"] == pytest.approx(0.25)
    assert (d2["min"], d2["max"]) == (pytest.approx(1.0), pytest.approx(2.0))
    g = pf["todas"]["global_"]
    assert g["min"] == pytest.approx((0.25 + 1.0) / 2) and g["max"] == pytest.approx((0.25 + 2.0) / 2)
    assert list(pf["equivalentes"]["por_dimension"]) == ["d1"]       # p2 no es equivalente


# ---------------------------------------------------------------- reproducibilidad
@pytest.mark.parametrize("modulo,fn", [("harlow", "correr"), ("reversion", "correr_serial"),
                                       ("reversion", "correr_volatil"), ("inferencia", "correr_7"),
                                       ("memoria_trabajo", "correr")])
def test_reproducible_con_semilla(modulo, fn):
    m = __import__(f"battery.{modulo}", fromlist=[fn])
    f = getattr(m, fn)
    a = f(3, 0.05); b = f(3, 0.05)
    a.pop("segundos"); b.pop("segundos")
    assert json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)


def test_pruebas_registradas_tienen_catalogo():
    cat = esquema.cargar_catalogo()
    claves = set()
    for p in registro.pruebas():
        assert p.id in cat["pruebas"], p.id
        assert p.condicion in esquema.CONDICIONES
        assert p.clave not in claves
        claves.add(p.clave)
        for rid in p.comparar:
            esquema.buscar(cat["pruebas"][p.id], rid)                 # existe en el catálogo
