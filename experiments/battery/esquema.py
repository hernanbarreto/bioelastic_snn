"""Esquema común de resultados, índice normalizado, comparación con especies y perfil.

Una fila de resultado (un dict) tiene siempre los campos de `CAMPOS`. El índice
normalizado se calcula por semilla, (red - azar) / (referencia - azar), sin recortar a
[0, 1], y se guarda como intervalo [min, max]:
  * referencia puntual: min = max;
  * referencia de rango entre sujetos (p. ej. 0,55 a 0,75): un extremo contra cada borde;
  * referencia que es una cota inferior (p. ej. «error de 4,5 % como máximo»): el índice
    contra la cota es una cota superior; la otra punta se calcula contra el máximo posible
    de la métrica (si la métrica tiene techo, p. ej. una exactitud). No se interpola nada.

Desvío entre semillas: poblacional (ddof = 0), como el resto del repositorio.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

CATALOGO = Path(__file__).with_name("catalogo.json")

ESTADOS = ("medido", "ingerido", "pendiente", "no aplicable hoy", "sin referencia verificada", "error")
CONDICIONES = ("simbolos", "imagenes", "texto")
CAMPOS = ("prueba", "variante", "nombre", "dimension", "condicion_entrada", "equivalente",
          "metrica", "protocolo", "desviaciones", "estado", "motivo", "semillas", "red", "azar",
          "referencia_humana", "referencias_animales", "referencias_maquinas",
          "indice_normalizado", "detalle", "firmas", "commit", "fuente_resultado", "segundos")


# ---------------------------------------------------------------- catálogo
def cargar_catalogo(path: Path | str = CATALOGO) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def referencias(entrada: dict) -> list[dict]:
    """Todas las referencias de una prueba del catálogo, con su grupo."""
    out = []
    for grupo in ("humano", "animales", "maquinas"):
        for r in entrada.get(grupo, []):
            out.append(dict(r, grupo=grupo))
    return out


def buscar(entrada: dict, rid: str | None) -> dict | None:
    if rid is None:
        return None
    for r in referencias(entrada):
        if r["id"] == rid:
            return r
    raise KeyError(f"referencia {rid!r} no está en el catálogo")


# ---------------------------------------------------------------- números
def media_desvio(xs) -> tuple[float | None, float | None]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return None, None
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), float(a.std())


def indice(red: float | None, azar: float | None, ref: float | None) -> float | None:
    """(red - azar) / (referencia - azar), sin recortar. None si falta algo o si la
    referencia es igual al azar (el índice no está definido)."""
    if red is None or azar is None or ref is None:
        return None
    den = float(ref) - float(azar)
    if abs(den) < 1e-12:
        return None
    return (float(red) - float(azar)) / den


def extremos_referencia(r: dict, techo: float | None = None) -> tuple[float, float] | None:
    """Los valores de referencia que delimitan el índice: (para el índice máximo, para el
    mínimo). Con una referencia puntual son el mismo. Devuelve None si no hay número."""
    tipo = r.get("tipo_valor")
    if tipo == "rango" and r.get("valor_min") is not None and r.get("valor_max") is not None:
        return float(r["valor_min"]), float(r["valor_max"])
    v = r.get("valor")
    if v is None:
        return None
    if tipo == "cota_inferior":
        return float(v), (float(techo) if techo is not None else float(v))
    return float(v), float(v)


def indice_intervalo(puntajes: list, azar: float, r: dict, techo: float | None = None) -> dict | None:
    """Índice por semilla contra los extremos de la referencia; media y desvío."""
    ext = extremos_referencia(r, techo)
    if ext is None or azar is None:
        return None
    ref_hi, ref_lo = ext            # ref_hi da el índice mayor (referencia más baja)
    if ref_hi > ref_lo:
        ref_hi, ref_lo = ref_lo, ref_hi
    por_hi = [indice(p, azar, ref_hi) for p in puntajes]
    por_lo = [indice(p, azar, ref_lo) for p in puntajes]
    if all(x is None for x in por_hi):
        return None
    m_hi, s_hi = media_desvio(por_hi)
    m_lo, s_lo = media_desvio(por_lo)
    if m_lo is None:
        m_lo, s_lo, por_lo = m_hi, s_hi, por_hi
    lo, hi = sorted((m_lo, m_hi))
    return dict(contra=r["id"], min=lo, max=hi, media=m_hi if ref_hi == ref_lo else None,
                desvio=s_hi if ref_hi == ref_lo else None,
                por_semilla=[x for x in por_hi] if ref_hi == ref_lo else None,
                por_semilla_min=por_lo if ref_hi != ref_lo else None,
                por_semilla_max=por_hi if ref_hi != ref_lo else None,
                referencia=(ref_hi if ref_hi == ref_lo else [ref_hi, ref_lo]),
                tipo_valor=r.get("tipo_valor"))


def comparar(media: float | None, desvio: float | None, r: dict, mayor_es_mejor: bool = True) -> str:
    """Dónde queda la especie respecto de la red, con la misma métrica."""
    if media is None:
        return "sin dato de la red"
    ext = extremos_referencia(r)
    if ext is None:
        return "cualitativa"
    lo, hi = sorted(ext)
    tol = desvio or 0.0
    signo = 1.0 if mayor_es_mejor else -1.0
    if r.get("tipo_valor") == "cota_inferior":
        if signo * (media - lo) < -tol:
            return "especie por encima"
        return "no concluyente (la referencia es una cota)"
    if lo - tol <= media <= hi + tol:
        return "a la par"
    return "especie por debajo" if signo * (media - hi) > 0 else "especie por encima"


# ---------------------------------------------------------------- filas
def _ref_publica(r: dict) -> dict:
    keep = ("id", "especie", "sistema", "valor", "valor_min", "valor_max", "tipo_valor", "n",
            "cita", "url", "verificacion", "releida", "protocolo", "nota", "cualitativo",
            "tras_problemas", "tramo", "par", "conjunto")
    return {k: r[k] for k in keep if k in r}


def fila_resultado(prueba, crudos: list[dict], catalogo: dict, commit: str) -> dict:
    """Arma la fila común a partir de los resultados crudos por semilla de `prueba`
    (`base.Prueba`)."""
    ent = catalogo["pruebas"][prueba.id]
    azar_v = prueba.azar if prueba.azar is not None else ent.get("azar", {}).get("valor")
    azar_d = prueba.azar_derivacion or ent.get("azar", {}).get("derivacion")
    base = dict(prueba=prueba.id, variante=prueba.variante, nombre=ent["nombre"],
                dimension=ent["dimension"], condicion_entrada=prueba.condicion,
                equivalente=dict(valor=bool(prueba.equivalente), motivo=prueba.motivo_equivalencia),
                metrica=prueba.metrica, protocolo=prueba.protocolo,
                desviaciones=list(prueba.desviaciones), semillas=[c["semilla"] for c in crudos],
                azar=dict(valor=azar_v, derivacion=azar_d), commit=commit,
                fuente_resultado=f"medicion@{commit}",
                segundos=float(sum(c.get("segundos", 0.0) for c in crudos)))
    no_aplica = [c for c in crudos if c.get("no_aplicable")]
    if no_aplica or not crudos:
        motivo = no_aplica[0]["no_aplicable"] if no_aplica else "sin corridas"
        return _completar(dict(base, estado="no aplicable hoy", motivo=motivo, red=None,
                               referencia_humana=None, referencias_animales=[],
                               referencias_maquinas=[], indice_normalizado=None, detalle={}, firmas={}))
    puntajes = [c.get("puntaje") for c in crudos]
    m, s = media_desvio(puntajes)
    red = dict(media=m, desvio=s, por_semilla=puntajes)
    vals = [float(x) for x in puntajes if x is not None]
    if vals:                                      # la distribución entre semillas, no sólo media y desvío
        red.update(mediana=float(np.median(vals)), minimo=min(vals), maximo=max(vals))
        if azar_v is not None:
            red["semillas_en_azar"] = int(sum(v <= azar_v + 1e-9 for v in vals)) if prueba.mayor_es_mejor else None
        if prueba.techo is not None:
            red["semillas_en_techo"] = int(sum(v >= prueba.techo - 1e-9 for v in vals))
    # detalle: cada medida, media y desvío entre semillas
    claves = sorted({k for c in crudos for k, v in c.get("medidas", {}).items()
                     if isinstance(v, (int, float, bool)) or v is None})
    detalle = {}
    for k in claves:
        vals = [c.get("medidas", {}).get(k) for c in crudos]
        vals = [float(v) if v is not None else None for v in vals]
        mk, sk = media_desvio(vals)
        detalle[k] = dict(media=mk, desvio=sk, por_semilla=vals)
    series = {k: [c.get("medidas", {}).get(k) for c in crudos]
              for k in sorted({k for c in crudos for k, v in c.get("medidas", {}).items()
                               if isinstance(v, list)})}
    if series:
        detalle["_series"] = series
    firmas = {}
    for k, desc in prueba.firmas.items():
        vals = [bool(c.get("medidas", {}).get(k)) for c in crudos if c.get("medidas", {}).get(k) is not None]
        firmas[k] = dict(descripcion=desc, semillas_con_firma=int(sum(vals)), semillas=len(vals))
    # referencias: la del índice y las comparables
    ref_id = ent.get("referencia_indice") if prueba.referencia == "catalogo" else prueba.referencia
    r_idx = buscar(ent, ref_id) if ref_id else None
    if r_idx is not None and not r_idx.get("releida"):
        r_idx = None                                       # sólo cifras releídas sirven de índice
    ix = indice_intervalo(puntajes, azar_v, r_idx, prueba.techo) if r_idx is not None else None
    humanos, animales, maquinas = [], [], []
    for r in referencias(ent):
        pub = _ref_publica(r)
        medida = prueba.comparar.get(r["id"])
        if r_idx is not None and r["id"] == r_idx["id"] and medida is None:
            medida = "puntaje"
        if medida is not None:
            if medida == "puntaje":
                mm, ss = m, s
            else:
                d = detalle.get(medida) or {}
                mm, ss = d.get("media"), d.get("desvio")
            pub["medida_red"] = medida
            pub["valor_red"] = mm
            pub["comparacion"] = comparar(mm, ss, r, prueba.mayor_es_mejor)
            exp_red = prueba.experiencia.get(medida)
            if exp_red is not None and r.get("tras_problemas") and r["tras_problemas"] > exp_red:
                if pub["comparacion"] in ("especie por encima", "a la par"):
                    pub["comparacion"] = "no concluyente (la especie tuvo más problemas)"
                else:
                    pub["comparacion"] += " (la red con menos problemas)"
        else:
            pub["comparacion"] = "cualitativa" if extremos_referencia(r) is None else "no comparable (otra métrica o protocolo)"
        if r_idx is not None and r["id"] == r_idx["id"]:
            pub["referencia_del_indice"] = True
        if r["grupo"] == "maquinas":
            pub["comparacion"] = pub["comparacion"].replace("especie", "sistema")
        {"humano": humanos, "animales": animales, "maquinas": maquinas}[r["grupo"]].append(pub)
    ref_h = next((h for h in humanos if h.get("referencia_del_indice")), humanos[0] if humanos else None)
    curva = []
    for tramo, medida, rid in prueba.curva:
        d = detalle.get(medida) or {}
        r = buscar(ent, rid) if rid else None
        curva.append(dict(tramo=tramo, medida=medida, red_media=d.get("media"), red_desvio=d.get("desvio"),
                          referencia=rid, especie=(r or {}).get("especie"),
                          valor=(r or {}).get("valor"), tipo_valor=(r or {}).get("tipo_valor"),
                          comparacion=comparar(d.get("media"), d.get("desvio"), r, prueba.mayor_es_mejor) if r else None))
    estado = "medido"
    motivo = ""
    if ix is None:
        estado = "medido"
        motivo = ("sin referencia humana verificada con la misma métrica: no hay índice normalizado"
                  if ref_id is None else f"la referencia {ref_id} no permite calcular el índice")
    return _completar(dict(base, estado=estado, motivo=motivo, red=red, referencia_humana=ref_h,
                           referencias_animales=animales, referencias_maquinas=maquinas,
                           indice_normalizado=ix, detalle=detalle, firmas=firmas, curva=curva,
                           otras_referencias_humanas=[h for h in humanos if h is not ref_h]))


def _completar(f: dict) -> dict:
    for k in CAMPOS:
        f.setdefault(k, None)
    return f


def validar_fila(f: dict) -> list[str]:
    """Errores de esquema de una fila (lista vacía si está bien)."""
    err = [f"falta {k}" for k in CAMPOS if k not in f]
    if f.get("estado") not in ESTADOS:
        err.append(f"estado desconocido: {f.get('estado')}")
    if f.get("condicion_entrada") not in CONDICIONES:
        err.append(f"condición de entrada desconocida: {f.get('condicion_entrada')}")
    if f.get("estado") in ("medido", "ingerido"):
        red = f.get("red") or {}
        if red.get("media") is None and not f.get("firmas"):
            err.append("medido sin puntaje ni firmas")
        if not f.get("semillas"):
            err.append("sin semillas")
        ix = f.get("indice_normalizado")
        if ix is not None and (ix.get("min") is None or ix.get("max") is None or ix["min"] > ix["max"] + 1e-12):
            err.append("índice mal formado")
    if not isinstance(f.get("desviaciones"), list):
        err.append("desviaciones no es una lista")
    return err


# ---------------------------------------------------------------- perfil
def perfil(filas: list[dict], catalogo: dict) -> dict:
    """Índice por prueba, promedio por dimensión y (como convención) promedio simple
    entre dimensiones. Dos versiones: con todas las condiciones de entrada y sólo con las
    equivalentes al protocolo publicado. Dentro de una prueba, las variantes que entran se
    promedian primero (una prueba pesa una vez en su dimensión)."""
    def armar(filtro):
        por_prueba: dict = {}
        for f in filas:
            ix = f.get("indice_normalizado")
            if not ix or not filtro(f):
                continue
            por_prueba.setdefault((f["dimension"], f["prueba"]), []).append((ix["min"], ix["max"], f["variante"]))
        dims: dict = {}
        for (d, p), v in por_prueba.items():
            lo = float(np.mean([a for a, _, _ in v])); hi = float(np.mean([b for _, b, _ in v]))
            dims.setdefault(d, []).append(dict(prueba=p, variantes=[x for _, _, x in v], min=lo, max=hi))
        out = {}
        for d, ps in dims.items():
            out[d] = dict(nombre=catalogo["dimensiones"].get(d, d), min=float(np.mean([x["min"] for x in ps])),
                          max=float(np.mean([x["max"] for x in ps])), pruebas=ps)
        glob = None
        if out:
            glob = dict(min=float(np.mean([v["min"] for v in out.values()])),
                        max=float(np.mean([v["max"] for v in out.values()])),
                        dimensiones=len(out), de=len(catalogo["dimensiones"]),
                        convencion=catalogo["convenciones"].get("indice_global"))
        return dict(por_dimension=out, global_=glob)
    return dict(equivalentes=armar(lambda f: (f.get("equivalente") or {}).get("valor")),
                todas=dict(armar(lambda f: True), advertencia=ADVERTENCIA_TODAS),
                sin_indice=sorted({f"{f['prueba']}/{f['variante']}" for f in filas
                                   if not f.get("indice_normalizado")}))


# ---------------------------------------------------------------- salida
def _num(x, nd=3):
    return "" if x is None else f"{x:.{nd}f}"


def _ms(d, nd=3):
    if not d or d.get("media") is None:
        return ""
    return f"{d['media']:.{nd}f} ± {d['desvio']:.{nd}f}" if d.get("desvio") is not None else _num(d["media"], nd)


def _ix(ix):
    if not ix:
        return ""
    if abs(ix["max"] - ix["min"]) < 0.005:
        return f"{ix['min']:.2f}" + (f" ± {ix['desvio']:.2f}" if ix.get("desvio") is not None else "")
    return f"{ix['min']:.2f} a {ix['max']:.2f}"


COLUMNAS_CSV = ("prueba", "variante", "dimension", "condicion_entrada", "equivalente", "motivo_equivalencia", "estado",
                "red_media", "red_desvio", "red_mediana", "red_min", "red_max", "semillas_en_azar", "semillas_en_techo",
                "n_semillas", "azar", "referencia_humana",
                "referencia_valor", "referencia_fuente", "indice_min", "indice_max",
                "especies_por_encima", "especies_por_debajo", "a_la_par", "sistemas_por_encima",
                "sistemas_por_debajo", "fuente_resultado", "motivo")


def fila_csv(f: dict) -> dict:
    red = f.get("red") or {}
    rh = f.get("referencia_humana") or {}
    ix = f.get("indice_normalizado") or {}
    refs = (f.get("referencias_animales") or []) + ([rh] if rh else []) + (f.get("otras_referencias_humanas") or [])
    nombre = lambda r: f"{r.get('especie') or r.get('sistema')} ({r['id']})"
    arriba = [nombre(r) for r in refs if r.get("comparacion") == "especie por encima"]
    abajo = [nombre(r) for r in refs if str(r.get("comparacion", "")).startswith("especie por debajo")]
    par = [nombre(r) for r in refs + (f.get("referencias_maquinas") or []) if r.get("comparacion") == "a la par"]
    maq = f.get("referencias_maquinas") or []
    s_arriba = [nombre(r) for r in maq if r.get("comparacion") == "sistema por encima"]
    s_abajo = [nombre(r) for r in maq if str(r.get("comparacion", "")).startswith("sistema por debajo")]
    rv = rh.get("valor") if rh.get("valor") is not None else (
        f"{rh.get('valor_min')}-{rh.get('valor_max')}" if rh.get("valor_min") is not None else "")
    return dict(prueba=f["prueba"], variante=f["variante"], dimension=f["dimension"],
                condicion_entrada=f["condicion_entrada"],
                equivalente="sí" if (f.get("equivalente") or {}).get("valor") else "no",
                motivo_equivalencia=(f.get("equivalente") or {}).get("motivo", ""),
                estado=f["estado"], red_media=_num(red.get("media"), 4), red_desvio=_num(red.get("desvio"), 4),
                red_mediana=_num(red.get("mediana"), 4), red_min=_num(red.get("minimo"), 4),
                red_max=_num(red.get("maximo"), 4),
                semillas_en_azar="" if red.get("semillas_en_azar") is None else red["semillas_en_azar"],
                semillas_en_techo="" if red.get("semillas_en_techo") is None else red["semillas_en_techo"],
                n_semillas=len(f.get("semillas") or []), azar=_num((f.get("azar") or {}).get("valor"), 4),
                referencia_humana=rh.get("id", ""), referencia_valor=rv,
                referencia_fuente=rh.get("cita", ""), indice_min=_num(ix.get("min"), 3),
                indice_max=_num(ix.get("max"), 3), especies_por_encima="; ".join(arriba),
                especies_por_debajo="; ".join(abajo), a_la_par="; ".join(par),
                sistemas_por_encima="; ".join(s_arriba), sistemas_por_debajo="; ".join(s_abajo),
                fuente_resultado=f.get("fuente_resultado") or "", motivo=f.get("motivo") or "")


def escribir_csv(filas: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS_CSV)
        w.writeheader()
        for f in filas:
            w.writerow(fila_csv(f))


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


ADVERTENCIA_TODAS = ("mezcla comparaciones no equivalentes (símbolos frente a objetos vistos, otro procedimiento "
                     "de prueba o memoria exacta por diseño): no leer como «porcentaje humano»")
COLUMNAS_PERFIL = ("conjunto", "dimension", "nombre", "indice_min", "indice_max", "pruebas", "dimensiones_con_indice",
                   "dimensiones_total", "advertencia")


def escribir_perfil_csv(pf: dict, path: Path):
    """El perfil por dimensión y el promedio entre dimensiones: primero el de las comparaciones
    equivalentes y después el de todas, con su advertencia."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS_PERFIL)
        w.writeheader()
        for conj in ("equivalentes", "todas"):
            adv = "" if conj == "equivalentes" else ADVERTENCIA_TODAS
            g = pf[conj]["global_"]
            if g:
                w.writerow(dict(conjunto=conj, dimension="(promedio entre dimensiones, convención)", nombre="",
                                indice_min=_num(g["min"]), indice_max=_num(g["max"]), pruebas="",
                                dimensiones_con_indice=g["dimensiones"], dimensiones_total=g["de"], advertencia=adv))
            for d, v in pf[conj]["por_dimension"].items():
                w.writerow(dict(conjunto=conj, dimension=d, nombre=v["nombre"], indice_min=_num(v["min"]),
                                indice_max=_num(v["max"]),
                                pruebas="; ".join(f"{x['prueba']} ({', '.join(x['variantes'])})" for x in v["pruebas"]),
                                dimensiones_con_indice="", dimensiones_total="", advertencia=adv))


def escribir_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=_jsonable)
        fh.write("\n")


def tabla(filas: list[dict]) -> str:
    """Tabla compacta para la consola."""
    cab = ("prueba/variante", "entrada", "equiv.", "red (media ± desvío)", "semillas [min, mediana, max]", "azar",
           "ref. humana", "índice", "estado")
    rows = []
    for f in filas:
        rh = f.get("referencia_humana") or {}
        rv = rh.get("valor") if rh.get("valor") is not None else (
            f"{rh.get('valor_min')}-{rh.get('valor_max')}" if rh.get("valor_min") is not None else "-")
        red = f.get("red") or {}
        dist = ""
        if red.get("mediana") is not None:
            n = len(f.get("semillas") or [])
            dist = f"[{red['minimo']:.2f}, {red['mediana']:.2f}, {red['maximo']:.2f}]"
            if red.get("semillas_en_azar"):
                dist += f" ≤azar {red['semillas_en_azar']}/{n}"
            if red.get("semillas_en_techo"):
                dist += f" techo {red['semillas_en_techo']}/{n}"
        rows.append((f"{f['prueba']}/{f['variante']}", f["condicion_entrada"],
                     "sí" if (f.get("equivalente") or {}).get("valor") else "NO", _ms(red), dist,
                     _num((f.get("azar") or {}).get("valor"), 2), str(rv), _ix(f.get("indice_normalizado")) or "-",
                     f["estado"]))
    w = [max(len(str(x)) for x in col) for col in zip(cab, *rows)] if rows else [len(c) for c in cab]
    line = lambda r: "  ".join(str(x).ljust(n) for x, n in zip(r, w))
    return "\n".join([line(cab), line(["-" * n for n in w])] + [line(r) for r in rows])
