"""Batería de medición: la red contra experimentos publicados con humanos y animales.

Corre las pruebas de `experiments/battery/` contra el paquete `bioelastic/` de un commit,
guarda cada prueba y semilla en results/bateria/<commit>/ (reanudable: lo hecho se salta),
ingiere los resultados publicados por otras ramas y escribe el perfil en
results/bateria_<commit>.json y .csv, con una tabla compacta por consola.

  python experiments/exp_battery.py correr  [--pruebas id,id] [--semillas 0,1,2] [--commit REF]
                                            [--escala 1.0] [--minutos 9]
  python experiments/exp_battery.py informe [--commit REF]
  python experiments/exp_battery.py listar

`--commit`: cualquier referencia de git (HEAD por defecto). `--minutos`: después de ese
tiempo no se empiezan semillas nuevas (para encadenar corridas cortas). Correr con
OMP_NUM_THREADS=1.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from battery import version  # noqa: E402  (no importa bioelastic)


def _args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=("correr", "informe", "listar"))
    ap.add_argument("--pruebas", default="", help="ids separados por coma (por defecto, todas)")
    ap.add_argument("--semillas", default="", help="semillas separadas por coma (por defecto, las de cada prueba)")
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--escala", type=float, default=1.0)
    ap.add_argument("--minutos", type=float, default=None)
    ap.add_argument("--resultados", default=None, help="carpeta de resultados (por defecto results/)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="no rehace las semillas ya guardadas; es lo que hace siempre el almacén (se acepta por "
                         "uniformidad con los otros experimentos)")
    return ap.parse_args(argv)


def _elegidas(todas, spec: str):
    if not spec:
        return todas
    ids = [x.strip() for x in spec.split(",") if x.strip()]
    out = [p for p in todas if p.id in ids or p.clave in ids]
    faltan = [i for i in ids if not any(p.id == i or p.clave == i for p in todas)]
    if faltan:
        raise SystemExit(f"pruebas desconocidas: {faltan}")
    return out


def _herencia(p, info, almacen, escala):
    """Para una prueba que no usa el Thinker y no se corrió en este commit: los resultados de
    otro commit con la misma huella de percepción (mismo código, mismo resultado). Sólo con las
    opciones por defecto: la comprobación de identidad no cubre perfiles como `metacog`."""
    from battery import almacen as alm
    if p.perfiles or p.usa_thinker or p.perfil != "defecto":
        return None, None
    propia = version.huella_equivalente(version.huella_percepcion(info["sha"]))
    for d in sorted(almacen.dir.parent.iterdir()) if almacen.dir.parent.exists() else []:
        if d.name == almacen.dir.name or not (d / p.clave).exists():
            continue
        try:
            if version.huella_equivalente(version.huella_percepcion(d.name.split("+")[0])) != propia:
                continue
        except Exception:
            continue
        otro = alm.Almacen(d.name, almacen.dir.parent.parent)
        crudos = [c for s in p.semillas if (c := otro.cargar(p.clave, s, p.version, escala)) is not None]
        if crudos:
            return crudos, d.name
    return None, None


def informe(info: dict, pruebas, almacen, catalogo, resultados: Path, escala: float, log=print) -> dict:
    from battery import esquema, ingesta, registro
    filas = []
    for p in pruebas:
        crudos = [c for s in p.semillas
                  if (c := almacen.cargar(p.clave, s, p.version, escala)) is not None]
        heredado = None
        if not crudos:
            crudos, heredado = _herencia(p, info, almacen, escala)
        if not crudos:
            continue
        sel = registro.seleccionar(p, almacen, escala, log=lambda *a: None, calcular=False) if p.configs else None
        ctrl = None
        if sel and sel.get("elegida"):
            crudos = [c for c in crudos if c.get("config") == sel["elegida"]]
            ctrl = registro.controles(p, almacen, escala, sel, limite=0.0, log=lambda *a: None)
        f = registro.fila(p, crudos, sel, catalogo, info["commit"], ctrl)
        if heredado:
            f["fuente_resultado"] = (f"medicion@{heredado} (heredado: la percepción de {info['commit']} es la misma, "
                                     f"por archivos idénticos o por la comprobación bit a bit de "
                                     f"results/bateria/equivalencias.json, y la prueba no usa el Thinker)")
            f["commit"] = heredado
        filas.append(f)
    filas += ingesta.filas(catalogo)
    for f in filas:
        err = esquema.validar_fila(f)
        if err:
            raise RuntimeError(f"fila inválida {f['prueba']}/{f['variante']}: {err}")
    perfil = esquema.perfil(filas, catalogo)
    otros = {po: esquema.perfil(filas, catalogo, po) for po in esquema.perfiles_presentes(filas)[1:]}
    import numpy
    out = dict(meta=dict(commit=info["commit"], sha=info["sha"], sucio=info["sucio"], escala=escala,
                         fecha=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
                         numpy=numpy.__version__, catalogo=dict(estado=catalogo.get("estado"),
                                                                actualizado=catalogo.get("actualizado")),
                         convenciones=catalogo["convenciones"]),
               perfil=perfil, perfiles_de_opciones=otros, filas=filas)
    nombre = f"bateria_{info['commit']}"
    esquema.escribir_json(out, resultados / f"{nombre}.json")
    esquema.escribir_csv(filas, resultados / f"{nombre}.csv")
    esquema.escribir_perfil_csv([perfil, *otros.values()], resultados / f"{nombre}_perfil.csv")
    esquema.escribir_perfiles_csv(filas, resultados / f"{nombre}_perfiles_thinker.csv")
    log(esquema.tabla(filas))
    log("")
    for tipo in ("equivalentes", "todas"):
        log(f"perfil ({'sólo comparaciones equivalentes' if tipo == 'equivalentes' else 'todas las condiciones'}):")
        if tipo == "todas":
            log(f"  ADVERTENCIA: {esquema.ADVERTENCIA_TODAS}")
        for pf in (perfil, *otros.values()):
            po = pf["perfil_opciones"]
            g = pf[tipo]["global_"]
            if po != "defecto":
                log(f"  con el perfil {po} (en {', '.join(pf['pruebas_con_el_perfil'])}; el resto con las opciones "
                    f"por defecto):")
            for d, v in pf[tipo]["por_dimension"].items():
                log(f"  {d:28s} {v['min']:.2f}" + ("" if f"{v['max']:.2f}" == f"{v['min']:.2f}" else f" a {v['max']:.2f}")
                    + f"  ({', '.join(x['prueba'] for x in v['pruebas'])})")
            if g:
                log(f"  promedio entre dimensiones (convención): {g['min']:.2f}"
                    + ("" if f"{g['max']:.2f}" == f"{g['min']:.2f}" else f" a {g['max']:.2f}")
                    + f", {g['dimensiones']} de {g['de']} dimensiones")
    log(f"\n-> {resultados / (nombre + '.json')}")
    return out


def main(argv=None):
    a = _args(argv)
    info = version.preparar(a.commit)
    from battery import almacen as alm, esquema, registro
    catalogo = esquema.cargar_catalogo()
    todas = registro.pruebas()
    if a.accion == "listar":
        for p in todas:
            print(f"{p.clave:40s} {p.condicion:9s} semillas={list(p.semillas)}  {p.metrica}")
        return
    pruebas = _elegidas(todas, a.pruebas)
    resultados = Path(a.resultados) if a.resultados else alm.RESULTS
    almacen = alm.Almacen(info["commit"], resultados)
    print(f"commit {info['commit']} (bioelastic desde {info['raiz']})", flush=True)
    if a.accion == "correr":
        limite = time.time() + 60 * a.minutos if a.minutos else None
        semillas = [int(x) for x in a.semillas.split(",")] if a.semillas else None
        pendientes = []
        for p in pruebas:
            print(f"{p.clave}", flush=True)
            _, sel, completa = registro.correr(p, almacen, a.escala, semillas, limite,
                                               log=lambda m: print(m, flush=True))
            if sel and sel.get("elegida"):
                registro.controles(p, almacen, a.escala, sel, semillas, limite, log=lambda m: print(m, flush=True))
            if not completa:
                pendientes.append(p.clave)
        if pendientes:
            print(f"quedan semillas sin correr (límite de tiempo): {pendientes}")
    informe(info, todas, almacen, catalogo, resultados, a.escala)


if __name__ == "__main__":
    main()
