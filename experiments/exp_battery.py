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


def informe(info: dict, pruebas, almacen, catalogo, resultados: Path, escala: float, log=print) -> dict:
    from battery import esquema, ingesta, registro
    filas = []
    for p in pruebas:
        crudos = [c for s in p.semillas
                  if (c := almacen.cargar(p.clave, s, p.version, escala)) is not None]
        if not crudos:
            continue
        sel = registro.seleccionar(p, almacen, escala, log=lambda *a: None) if p.configs else None
        if sel and sel.get("elegida"):
            crudos = [c for c in crudos if c.get("config") == sel["elegida"]]
        filas.append(registro.fila(p, crudos, sel, catalogo, info["commit"]))
    filas += ingesta.filas(catalogo)
    for f in filas:
        err = esquema.validar_fila(f)
        if err:
            raise RuntimeError(f"fila inválida {f['prueba']}/{f['variante']}: {err}")
    perfil = esquema.perfil(filas, catalogo)
    import numpy
    out = dict(meta=dict(commit=info["commit"], sha=info["sha"], sucio=info["sucio"], escala=escala,
                         fecha=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
                         numpy=numpy.__version__, catalogo=dict(estado=catalogo.get("estado"),
                                                                actualizado=catalogo.get("actualizado")),
                         convenciones=catalogo["convenciones"]),
               perfil=perfil, filas=filas)
    nombre = f"bateria_{info['commit']}"
    esquema.escribir_json(out, resultados / f"{nombre}.json")
    esquema.escribir_csv(filas, resultados / f"{nombre}.csv")
    log(esquema.tabla(filas))
    log("")
    for tipo in ("todas", "equivalentes"):
        g = perfil[tipo]["global_"]
        log(f"perfil ({tipo}):")
        for d, v in perfil[tipo]["por_dimension"].items():
            log(f"  {d:28s} {v['min']:.2f}" + ("" if abs(v['max'] - v['min']) < 1e-9 else f" a {v['max']:.2f}")
                + f"  ({', '.join(x['prueba'] for x in v['pruebas'])})")
        if g:
            log(f"  promedio entre dimensiones (convención): {g['min']:.2f}"
                + ("" if abs(g['max'] - g['min']) < 1e-9 else f" a {g['max']:.2f}")
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
            _, _, completa = registro.correr(p, almacen, a.escala, semillas, limite,
                                             log=lambda m: print(m, flush=True))
            if not completa:
                pendientes.append(p.clave)
        if pendientes:
            print(f"quedan semillas sin correr (límite de tiempo): {pendientes}")
    informe(info, todas, almacen, catalogo, resultados, a.escala)


if __name__ == "__main__":
    main()
