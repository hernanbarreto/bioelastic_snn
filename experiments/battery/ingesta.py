"""Resultados que corren otros agentes (o la propia rama), leídos en modo lectura.

Cada fuente dice de qué archivo sale la cifra, en qué commit se produjo (el último commit
que tocó el archivo, si está versionado) y con qué protocolo; la fila resultante usa el
mismo esquema, el mismo azar y las mismas referencias del catálogo que las pruebas que
corre la batería. Si el archivo no existe todavía, la fila queda «pendiente».
"""
from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import esquema
from .base import Prueba

ROOT = Path(__file__).resolve().parents[2]
WT = ROOT.parent                        # /home/claude/wt: los árboles de los otros agentes


@dataclass
class Fuente:
    prueba: str
    variante: str
    condicion: str
    metrica: str
    protocolo: str
    desviaciones: list
    archivo: Path
    leer: Callable                      # Path -> list[dict(semilla, puntaje, medidas)]
    agente: str = "v0.3-dev"
    equivalente: bool = True
    motivo_equivalencia: str = ""
    comparar: dict = field(default_factory=dict)
    referencia: str | None = "catalogo"
    azar: float | None = None
    azar_derivacion: str | None = None
    firmas: dict = field(default_factory=dict)
    pendiente: str = ""                 # qué falta si el archivo no está
    agregado: bool = False              # la fuente sólo publica media ± desvío (sin valores por semilla)


def commit_de(archivo: Path) -> str:
    """Último commit que tocó el archivo en su propio árbol de trabajo ('sin commit' si no
    está versionado o tiene cambios sin commitear)."""
    try:
        cwd = archivo.parent
        h = subprocess.run(["git", "log", "-1", "--format=%h", "--", archivo.name], cwd=cwd,
                           capture_output=True, text=True, check=True).stdout.strip()
        sucio = subprocess.run(["git", "status", "--porcelain", "--", archivo.name], cwd=cwd,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return "sin commit"
    if not h:
        return "sin commit"
    return h + ("+cambios sin commitear" if sucio else "")


def _csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- lectores
def leer_vision(modelo: str):
    """Filas de results/<dataset>_final_summary.csv de exp_vision (test 10k, tocado una vez)."""
    def leer(path: Path):
        out = []
        for r in _csv(path):
            if r["model"] != modelo:
                continue
            out.append(dict(semilla=int(r["seed"]), puntaje=_f(r["test_acc"]),
                            medidas=dict(validacion=_f(r["val_acc"]), test_ruido=_f(r["test_noise"]),
                                         test_oclusion=_f(r["test_occlude"]), test_desplazamiento3=_f(r["test_shift3"]),
                                         imagenes_entrenamiento=_f(r["n_train"]))))
        return out
    return leer


def leer_predictivo_f0(nombre: str, depth: int, cond: str = "ff"):
    """results/predictivo_f0.json de la rama predictivo: cada medida como [media, desvío, n]."""
    def leer(path: Path):
        for r in leer_json(path):
            if r["name"] == nombre and int(r["depth"]) == depth and r["cond"] == cond:
                m, sd, n = r["test_acc"]
                return dict(media=float(m), desvio=float(sd), n=int(n), semillas=list(r["seeds"]),
                            medidas=dict(validacion=float(r["val_acc"][0]), imagenes_entrenamiento=float(r["n_train"]),
                                         imagenes_test=float(r["n_test"])))
        return None
    return leer


def fuentes() -> list[Fuente]:
    res = ROOT / "results"
    out = []
    for ds, pid in (("mnist", "mnist"), ("fashion", "fashion_mnist")):
        for modelo in ("depth2", "depth3"):
            out.append(Fuente(
                prueba=pid, variante=modelo, condicion="imagenes",
                metrica="exactitud en el test de 10 000 (tocado una vez)",
                protocolo=f"exp_vision {modelo}: una pasada en línea por 55 000 imágenes de entrenamiento en el orden de la "
                          "semilla; hiperparámetros elegidos en validación (últimas 5 000)",
                desviaciones=["la red ve cada imagen de entrenamiento una sola vez, en línea; las personas no entrenan con el "
                              "conjunto: reconocen con su experiencia previa",
                              "MNIST: la referencia humana es una estimación de 0,2 % de error que no se midió sobre todo "
                              "el test (Keysers 2007, citando a Simard et al. 1993)" if pid == "mnist" else
                              "Fashion: 1 000 imágenes al azar del test, 3 etiquetas por imagen y voto mayoritario; los "
                              "mantenedores no verificaron esa fila"],
                archivo=res / f"{ds}_final_summary.csv", leer=leer_vision(modelo),
                equivalente=True, motivo_equivalencia="la red y las personas ven las mismas imágenes de 28 x 28",
            ))
    for depth in (2, 3):
        out.append(Fuente(
            prueba="cifar10", variante=f"depth{depth}_5000", condicion="imagenes",
            metrica="exactitud en las primeras 5 000 imágenes del test",
            protocolo=(f"rama predictivo, fase F0, condición «ff» (la jerarquía de v0.3-dev sin los mecanismos nuevos), "
                       f"profundidad {depth}: una pasada en línea por 5 000 imágenes de entrenamiento"),
            desviaciones=["sólo 5 000 imágenes de entrenamiento (de 50 000) y las primeras 5 000 del test (de 10 000)",
                          "las personas de CIFAR-10H clasifican sin entrenar con el conjunto; la red aprende de él",
                          "la fuente publica media ± desvío de 3 semillas, sin los valores por semilla"],
            archivo=WT / "predictivo" / "results" / "predictivo_f0.json",
            leer=leer_predictivo_f0("cifar10", depth), agente="predictivo", agregado=True,
            equivalente=True, motivo_equivalencia="la red y las personas ven las mismas imágenes de 32 x 32",
            pendiente="todavía no hay resultados de CIFAR-10 en la rama predictivo",
        ))
    return out


def fila_de(f: Fuente, catalogo: dict) -> dict:
    p = Prueba(id=f.prueba, variante=f.variante, condicion=f.condicion, metrica=f.metrica, protocolo=f.protocolo,
               desviaciones=list(f.desviaciones), correr=None, equivalente=f.equivalente,
               motivo_equivalencia=f.motivo_equivalencia, comparar=dict(f.comparar), referencia=f.referencia,
               azar=f.azar, azar_derivacion=f.azar_derivacion, firmas=dict(f.firmas))
    if not f.archivo.exists():
        fila = esquema.fila_resultado(p, [], catalogo, "")
        fila.update(estado="pendiente", motivo=f.pendiente or f"todavía no existe {f.archivo}",
                    fuente_resultado=f"{f.agente}: {f.archivo}", commit="")
        return fila
    commit = commit_de(f.archivo)
    if f.agregado:
        a = f.leer(f.archivo)
        if a is None:
            fila = esquema.fila_resultado(p, [], catalogo, commit)
            fila.update(estado="pendiente", motivo=f.pendiente, fuente_resultado=f"{f.agente}: {f.archivo}")
            return fila
        fila = esquema.fila_resultado(p, [dict(semilla=-1, puntaje=a["media"], medidas=a["medidas"])], catalogo, commit)
        fila["semillas"] = a["semillas"]
        fila["red"] = dict(media=a["media"], desvio=a["desvio"], por_semilla=None, n=a["n"],
                           nota="la fuente publica media ± desvío; sin valores por semilla")
        ix = fila.get("indice_normalizado")
        if ix and ix.get("media") is not None:
            den = abs(float(ix["referencia"]) - float(fila["azar"]["valor"]))
            ix.update(desvio=a["desvio"] / den if den else None, por_semilla=None)
    else:
        crudos = f.leer(f.archivo)
        fila = esquema.fila_resultado(p, crudos, catalogo, commit)
    if fila.get("red"):
        fila["estado"] = "ingerido"
    fila["fuente_resultado"] = f"{f.agente}@{commit}: {f.archivo}"
    return fila


def filas(catalogo: dict) -> list[dict]:
    return [fila_de(f, catalogo) for f in fuentes()]


def leer_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
