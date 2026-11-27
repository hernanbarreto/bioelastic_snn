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


def test_perfil_no_mezcla_perfiles_de_opciones(tmp_path):
    """Con filas del Thinker en dos perfiles, el perfil por dimensión usa uno por vez: el de
    defecto sólo filas de defecto; el de R3, las de R3 donde las hay y las de defecto en el
    resto (una prueba que no usa el Thinker)."""
    cat = _cat_falso()
    alm = almacen.Almacen("c0", tmp_path)
    p_def, _ = _prueba("p1")                                          # índice (0.6 - 0.5) / 0.4 = 0.25
    import dataclasses
    p_r3 = dataclasses.replace(_prueba("p1", fn=lambda s: dict(puntaje=0.9, medidas={}))[0], variante="v+R3",
                               perfil="R3")
    p_otra, _ = _prueba("p2", fn=lambda s: dict(puntaje=2.0, medidas={}))
    filas = [registro.fila(p, registro.correr(p, alm, 1.0, log=lambda *a: None)[0], None, cat, "c0")
             for p in (p_def, p_r3, p_otra)]
    assert esquema.perfiles_presentes(filas) == ["defecto", "R3"]
    d = esquema.perfil(filas, cat)
    r3 = esquema.perfil(filas, cat, "R3")
    assert d["todas"]["por_dimension"]["d1"]["min"] == pytest.approx(0.25)
    assert [x["variantes"] for x in d["todas"]["por_dimension"]["d1"]["pruebas"]] == [["v"]]
    assert r3["todas"]["por_dimension"]["d1"]["min"] == pytest.approx(1.0)
    assert [x["variantes"] for x in r3["todas"]["por_dimension"]["d1"]["pruebas"]] == [["v+R3"]]
    assert r3["todas"]["por_dimension"]["d2"] == d["todas"]["por_dimension"]["d2"]
    assert r3["pruebas_con_el_perfil"] == ["p1"]
    esquema.escribir_perfil_csv([d, r3], tmp_path / "perfil.csv")
    lineas = (tmp_path / "perfil.csv").read_text().splitlines()
    assert lineas[1].startswith("equivalentes,defecto") and any(x.startswith("equivalentes,R3") for x in lineas)


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


# ---------------------------------------------------------------- F2: percepción e ingesta
def test_auroc_tipo2():
    from battery.metacognicion import auroc
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    assert auroc(conf, np.array([1, 1, 1, 0, 0, 0], dtype=bool)) == pytest.approx(1.0)
    assert auroc(conf, np.array([0, 0, 0, 1, 1, 1], dtype=bool)) == pytest.approx(0.0)
    assert auroc(np.ones(6), np.array([1, 0, 1, 0, 1, 0], dtype=bool)) == pytest.approx(0.5)   # empates
    assert auroc(conf, np.ones(6, dtype=bool)) is None
    rng = np.random.default_rng(0)                                    # contra el recuento de pares
    c = rng.random(200); o = rng.random(200) < c
    pares = np.mean([(a > b) + 0.5 * (a == b) for a in c[o] for b in c[~o]])
    assert auroc(c, o) == pytest.approx(pares)


def test_omniglot_a28():
    from battery.omniglot import a28
    t = np.zeros((105, 105), dtype=bool); t[40:60, 50:52] = True
    x = a28(t)
    assert x.shape == (28, 28) and x.dtype == np.uint8
    assert x.max() > 0 and x.sum() == pytest.approx(255 * t.sum() / 16, rel=0.01)


def test_ingesta_agregada_y_perfil_csv(tmp_path):
    from battery import ingesta
    cat = _cat_falso()
    src = tmp_path / "f0.json"
    src.write_text(json.dumps([{"name": "x", "depth": 2, "cond": "ff", "seeds": [0, 1, 2], "n_train": 10,
                                "n_test": 10, "test_acc": [0.7, 0.02, 3], "val_acc": [0.69, 0.01, 3]}]))
    f = ingesta.Fuente(prueba="p1", variante="v", condicion="imagenes", metrica="m", protocolo="pr",
                       desviaciones=[], archivo=src, leer=ingesta.leer_predictivo_f0("x", 2), agregado=True)
    fila = ingesta.fila_de(f, cat)
    assert fila["estado"] == "ingerido" and esquema.validar_fila(fila) == []
    assert fila["red"]["media"] == 0.7 and fila["red"]["desvio"] == 0.02 and fila["red"]["por_semilla"] is None
    assert fila["indice_normalizado"]["media"] == pytest.approx(0.5)
    assert fila["indice_normalizado"]["desvio"] == pytest.approx(0.05)
    falta = ingesta.Fuente(prueba="p1", variante="w", condicion="imagenes", metrica="m", protocolo="pr",
                           desviaciones=[], archivo=tmp_path / "no.json", leer=None, pendiente="todavía no")
    assert ingesta.fila_de(falta, cat)["estado"] == "pendiente"
    pf = esquema.perfil([fila], cat)
    esquema.escribir_perfil_csv(pf, tmp_path / "perfil.csv")
    filas = (tmp_path / "perfil.csv").read_text().splitlines()
    assert filas[1].startswith("equivalentes")                       # lo equivalente va primero
    assert any("todas" in x and "no equivalentes" in x for x in filas)


def test_reproducible_igual_distinto_simbolos():
    from battery import harlow, igual_distinto
    a = igual_distinto.correr_simbolos(5, 0.05, harlow.CONFIG); b = igual_distinto.correr_simbolos(5, 0.05, harlow.CONFIG)
    a.pop("segundos"); b.pop("segundos")
    assert a == b


def test_ingesta_por_semilla_y_lenguaje(tmp_path):
    from battery import ingesta
    from battery.lenguaje import NgramaWB, normalizar
    for s, v in ((0, 0.9), (1, 0.8)):
        (tmp_path / f"r_seed{s}.csv").write_text("prueba,variante,semilla,metrica,valor,referencia,nota\n"
                                                 f"x,y,{s},exactitud suma (test),{v},,\nx,y,{s},otra,{v / 2},,\n")
    leer = ingesta.leer_por_semilla("r_seed{s}.csv", "exactitud suma (test)", {"otra": "otra"})
    crudos = leer(tmp_path / "r_seed0.csv")
    assert [c["semilla"] for c in crudos] == [0, 1] and crudos[1]["medidas"]["otra"] == pytest.approx(0.4)
    assert normalizar("Who should Derek hug, after shocking Richard?") == b"who should derek hug , after shocking richard ?"
    ng = NgramaWB(3); ng.entrenar(b"\nthe dog runs .\nthe dog runs .\nthe cat runs .\n")
    assert ng.oracion(b"the dog runs .") > ng.oracion(b"the runs dog .")
    tot = sum(np.exp(ng.logp(b"th", y)) for y in range(256))           # distribución que suma 1
    assert tot == pytest.approx(1.0, abs=1e-9)


def test_ingesta_scan_experimento_1(tmp_path):
    from battery import ingesta
    th = "Thinker (valores por defecto + contexto temporal), semilla 0 (exploratorio, 1 semilla)"
    filas = ["prueba,sistema,tramo,medida,media,desvio,semillas"]
    for pc, n, v in ((1, 209, 0.0), (2, 418, 0.25)):
        filas += [f'"S","{th}","{pc} % ({n} comandos), 1 pasada(s), comandos ya vistos",secuencia exacta,0.5,0,1',
                  f'"S","{th}","{pc} % ({n} comandos), 1 pasada(s), test (muestra de 1000)",secuencia exacta,{v},0,1',
                  f'"S","{th}","{pc} % ({n} comandos), 1 pasada(s), test (muestra de 1000)",primer símbolo,0.4,0,1']
    filas.append(f'"S","seq2seq, publicado","1 %",secuencia exacta,0.05,0,0')
    (tmp_path / "scan.csv").write_text("\n".join(filas) + "\n")
    c = ingesta.leer_scan_exp1(2, 1)(tmp_path / "scan.csv")
    assert c == [dict(semilla=0, puntaje=0.25, medidas=dict(test_primer_símbolo=0.4, entrenamiento_secuencia_exacta=0.5))]
    assert ingesta.leer_scan_exp1(1, 3)(tmp_path / "scan.csv") == []             # sin esa corrida, sin filas


def test_kneser_ney_compacto_igual_al_de_diccionarios():
    from battery.lenguaje import KneserNey
    from battery.lenguaje_escala import KneserNeyCompacto, bits_por_posicion_kn
    rng = np.random.default_rng(0)
    pal = [b"the", b"dog", b"cat", b"runs", b"sits", b"a", b"big", b"red", b"ball", b"?", b"."]
    texto = b"\n".join(b" ".join(pal[j] for j in rng.integers(len(pal), size=rng.integers(2, 7)))
                       for _ in range(400))
    prueba = b"\nthe big dog runs .\nzebra sits ?\n" + texto[-300:]       # «z» y «e»+«b» no vistos juntos
    a = KneserNey(6, 256).entrenar([texto]); b = KneserNeyCompacto(6, 256).entrenar([texto])
    for K in (1, 2, 4, 6):
        a.orden(K); b.orden(K)
        assert b.secuencia(prueba) == pytest.approx(a.secuencia(prueba), rel=1e-12, abs=1e-9)
        assert b.logp(b"the bi", ord("g")) == pytest.approx(a.logp(b"the bi", ord("g")), rel=1e-12)
        assert b.logp(b"", ord("z")) == pytest.approx(a.logp(b"", ord("z")), rel=1e-12)
        np.testing.assert_allclose(bits_por_posicion_kn(b, prueba), bits_por_posicion_kn(a, prueba), rtol=1e-12)
    assert sum(np.exp(b.logp(b"th", y)) for y in range(256)) == pytest.approx(1.0, abs=1e-9)


def test_no_aplicables_registrados():
    ids = {(p.id, p.variante) for p in registro.pruebas()}
    assert ("deteccion_de_cambio", "dos_items") in ids and ("auroc_tipo2", "apuesta_espera") in ids
    p = next(p for p in registro.pruebas() if p.clave == "deteccion_de_cambio__dos_items")
    with pytest.raises(NoAplicable):
        p.correr(0, 1.0)


def test_kneser_ney_distribucion_y_orden():
    from battery.lenguaje import KneserNey, _ids, _palabras
    kb = KneserNey(4, 256).entrenar([b"\nthe dog runs .\nthe cat runs .\n"])
    for K in (1, 2, 4):
        kb.orden(K)
        assert sum(np.exp(kb.logp(b"th", y)) for y in range(256)) == pytest.approx(1.0, abs=1e-9)
    lin, vocab = _palabras(b"the dog runs .\nthe cat runs .\nthe dog sits .\n")
    km = KneserNey(2, len(vocab)).entrenar([_ids(l, vocab, 2) for l in lin])
    assert sum(np.exp(km.logp((vocab["the"],), w)) for w in range(len(vocab))) == pytest.approx(1.0, abs=1e-9)
    s_bien = km.secuencia(_ids("the dog runs .".split(), vocab, 2), desde=1)
    s_mal = km.secuencia(_ids("runs the . dog".split(), vocab, 2), desde=1)
    assert s_bien > s_mal


def test_equivalencias_de_percepcion(tmp_path, monkeypatch):
    from battery import version
    f = tmp_path / "eq.json"
    monkeypatch.setattr(version, "EQUIVALENCIAS", f)
    assert version.huella_equivalente("pB") == "pB"                    # sin archivo: la propia
    f.write_text(json.dumps({"percepcion": {"pC": {"igual_a": "pB"}, "pB": {"igual_a": "pA"}}}))
    assert version.huella_equivalente("pC") == "pA"                    # se sigue la cadena
    f.write_text(json.dumps({"percepcion": {"pX": {"igual_a": "pY"}, "pY": {"igual_a": "pX"}}}))
    assert version.huella_equivalente("pX") in ("pX", "pY")            # un ciclo no cuelga
    assert version.huella_percepcion("HEAD") == version.huella_percepcion("HEAD")


def test_perfiles_del_thinker():
    from battery.harlow import PERFILES
    ps = {p.clave: p for p in registro.pruebas()}
    for base in ("harlow__simbolos", "reversion_volatil__simbolos", "inferencia_transitiva_5__simbolos",
                 "deteccion_de_cambio__simbolos"):
        r3 = ps[base + "+R3"]
        assert r3.perfil == "R3" and "task_sets" not in next(iter(r3.configs.values()))
        assert all(all(v[k] == PERFILES["R3"][k] for k in PERFILES["R3"]) for v in r3.configs.values())
        assert next(iter(ps[base + "+R3+ts"].configs.values()))["task_sets"] is True
    assert ps["auroc_tipo2__imagenes+metacog"].configs["cal"]["metacog"]
