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
    patron: str = ""                    # archivos por semilla ({s}): todos cuentan para el commit


def commit_de(archivos) -> str:
    """Último commit que tocó alguno de los archivos en su propio árbol de trabajo ('sin
    commit' si no están versionados; «+cambios sin commitear» si alguno tiene cambios)."""
    archivos = [archivos] if isinstance(archivos, Path) else list(archivos)
    if not archivos:
        return "sin commit"
    cwd = archivos[0].parent
    nombres = [a.name for a in archivos]
    try:
        h = subprocess.run(["git", "log", "-1", "--format=%h", "--", *nombres], cwd=cwd,
                           capture_output=True, text=True, check=True).stdout.strip()
        sucio = subprocess.run(["git", "status", "--porcelain", "--", *nombres], cwd=cwd,
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


def leer_por_semilla(patron: str, principal: str, otras: dict | None = None, semillas=range(10)):
    """CSV por semilla con columnas (semilla, metrica, valor), como los de integracion3. `patron`
    lleva {s}; `principal` es la métrica del puntaje y `otras` {métrica: nombre de la medida}."""
    def leer(path: Path):
        out = []
        for s in semillas:
            f = path.parent / patron.format(s=s)
            if not f.exists():
                continue
            vals = {r["metrica"]: _f(r["valor"]) for r in _csv(f)}
            if principal not in vals:
                continue
            med = {nombre: vals.get(m) for m, nombre in (otras or {}).items()}
            out.append(dict(semilla=s, puntaje=vals[principal], medidas=med))
        return out
    return leer


def leer_split(escenario: str, modelo: str):
    def leer(path: Path):
        return [dict(semilla=int(r["seed"]), puntaje=_f(r["acc"]),
                     medidas=dict(olvido=_f(r["forgetting"]), bytes=_f(r["bytes"]), neuronas=_f(r["neurons"])))
                for r in _csv(path) if r["bench"] == "split" and r["scenario"] == escenario and r["model"] == modelo]
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
    i3 = WT / "integracion3" / "results"
    for pares, sem in ((27500, range(3)), (3000, range(8))):
        out.append(Fuente(
            prueba="mnist_addition", variante=f"fiab_{pares}", condicion="imagenes",
            metrica="exactitud de la suma en 5 000 pares del test (t10k)",
            protocolo=(f"rama integracion3, variante «fiab» (compuerta de fiabilidad): {pares} pares de entrenamiento, una "
                       "pasada en línea, reglas locales, la suma como única etiqueta (posterior conjunto como objetivo "
                       "blando)"),
            desviaciones=["una pasada en línea; las cifras publicadas usan varias épocas con backprop",
                          f"{pares} pares de entrenamiento (las publicadas, 30 000)"],
            archivo=i3 / f"integracion3_suma_mnist_fiab_p{pares}_seed0.csv",
            patron=f"integracion3_suma_mnist_fiab_p{pares}_seed{{s}}.csv",
            leer=leer_por_semilla(f"integracion3_suma_mnist_fiab_p{pares}_seed{{s}}.csv", "exactitud suma (test)",
                                  {"exactitud dígito (test)": "exactitud_digito",
                                   "AUROC error suma (test): 1-confianza": "auroc_error_suma",
                                   "ECE suma (test)": "ece_suma"}, sem),
            agente="integracion3", equivalente=True,
            motivo_equivalencia="la misma tarea de pares de MNIST que las cifras publicadas (sin referencia humana)",
            comparar={"deepproblog": "puntaje", "cnn_base": "puntaje", "deepstochlog": "puntaje", "anesi": "puntaje",
                      "referencia_099": "puntaje"},
        ))
    out.append(Fuente(
        prueba="auroc_tipo2", variante="flujo_continuo_integracion3", condicion="imagenes",
        metrica="AUROC del error propio con la confianza calibrada (flujo continuo, escala 0,1)",
        protocolo=("rama integracion3, metacognición en Cortex solo sobre el flujo continuo de exp_continual a escala 0,1 "
                   "(20 000 pasos), confianza calibrada de `Cortex.signals()`"),
        desviaciones=["sin escalera: la dificultad no se iguala (acierto de ~84 %, no 71 %) y aparecen clases nuevas en el "
                      "flujo; el AUROC de tipo 2 depende del nivel de acierto",
                      "la confianza es una señal de la red calibrada en línea, no un juicio de 1 a 6"],
        archivo=i3 / "integracion3_metacogx0.1_cal_seed0.csv", patron="integracion3_metacogx0.1_cal_seed{s}.csv",
        leer=leer_por_semilla("integracion3_metacogx0.1_cal_seed{s}.csv", "AUROC error sup: 1-confianza calibrada",
                              {"ECE calibrada (sup)": "ece_calibrada",
                               "Kepecs: AUROC de la apuesta (acierto vs error)": "auroc_apuesta_kepecs",
                               "Kepecs: pendiente apuesta vs facilidad en aciertos": "pendiente_apuesta_aciertos",
                               "Kepecs: pendiente apuesta vs facilidad en errores": "pendiente_apuesta_errores"},
                              range(3)),
        agente="integracion3", equivalente=False,
        motivo_equivalencia="dificultad no igualada y flujo con clases nuevas: otro procedimiento que Fleming et al.",
    ))
    cons = WT / "consolidacion" / "results" / "consolidacion_f6_split.csv"
    out.append(Fuente(
        prueba="split_mnist", variante="c2", condicion="imagenes",
        metrica="exactitud final sobre las 10 clases (test oficial), incremental por clases",
        protocolo=("rama consolidacion, F6: 5 tareas de 2 clases, un solo pase en línea por las imágenes de entrenamiento "
                   "de cada tarea (sin 500 por clase de validación), test oficial, argmax entre las 10 clases"),
        desviaciones=["un solo pase; las cifras publicadas: MLP 2x400 con 2 000 iteraciones de lotes de 128 por contexto",
                      "memoria: el almacén de c2 ocupa ~8,8 MB (unas 10 000 imágenes en uint8), diez veces el repaso de "
                      "100 por clase de la tabla publicada",
                      "semillas 3 a 5 (las de la rama)"],
        archivo=cons, leer=leer_split("mnist", "c2"), agente="consolidacion", equivalente=True,
        motivo_equivalencia="el mismo escenario incremental por clases sobre las mismas imágenes (sin referencia humana)",
        comparar={k: "puntaje" for k in ("conjunto", "ninguno", "ewc", "si", "lwf", "agem", "fromp", "er", "dgr",
                                         "icarl", "clasificador_generativo", "bir")},
    ))
    out.append(Fuente(
        prueba="split_fashion", variante="c2", condicion="imagenes",
        metrica="exactitud final sobre las 10 clases (test oficial), incremental por clases",
        protocolo="rama consolidacion, F6: el mismo procedimiento que Split-MNIST sobre Fashion-MNIST",
        desviaciones=["sin referencias publicadas verificadas por la batería"],
        archivo=cons, leer=leer_split("fashion", "c2"), agente="consolidacion", equivalente=True,
        motivo_equivalencia="el mismo escenario incremental por clases (sin referencia humana ni de máquina verificada)",
    ))
    r3 = WT / "razonamiento3" / "results"
    for pid, nombre in (("scan", "SCAN"), ("babi", "bAbI"), ("instrucciones_pocos_ejemplos", "Lake y Baroni 2023")):
        out.append(Fuente(
            prueba=pid, variante="thinker", condicion="simbolos", metrica="(la define la rama razonamiento3)",
            protocolo=f"{nombre}: lo corre la rama razonamiento3 (guion de F6 en af8d14f)", desviaciones=[],
            archivo=r3 / f"razonamiento3_f6_{pid}.csv", leer=None, agente="razonamiento3", equivalente=False,
            pendiente=f"{nombre}: la rama razonamiento3 tiene el guion (af8d14f) pero todavía no hay resultados",
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
    archivos = [f.archivo]
    if f.patron:
        archivos = sorted(f.archivo.parent.glob(f.patron.format(s="*")))
    commit = commit_de(archivos)
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
