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
    variabilidad: str = ""              # de dónde sale el ± de la fuente


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
                variabilidad="la semilla de exp_vision cambia el orden de las 55 000 imágenes y la semilla de la red; el "
                             "test es siempre el mismo",
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
            variabilidad="semillas 0 a 2 de la rama predictivo: orden de las imágenes y semilla de la red",
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
            variabilidad="semillas de la rama integracion3: armado de los pares, orden y semilla de la red; el test "
                         "(5 000 pares) es el mismo",
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
        variabilidad="semillas 0 a 2 de la rama integracion3: el flujo continuo (orden de las muestras) y la red",
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
        variabilidad="semillas 3 a 5 de la rama consolidacion: asignación de dígitos a tareas, orden y red",
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
        variabilidad="semillas 3 a 5 de la rama consolidacion",
        motivo_equivalencia="el mismo escenario incremental por clases (sin referencia humana ni de máquina verificada)",
    ))
    r3 = WT / "razonamiento3" / "results"
    out.append(Fuente(
        prueba="babi", variante="thinker_defecto_contexto", condicion="simbolos",
        metrica="exactitud media de las 20 tareas (1 000 ejemplos de entrenamiento por tarea, test congelado)",
        protocolo=("rama razonamiento3, F6c (en curso, exploratorio): Thinker con valores por defecto + contexto temporal, "
                   "entrenado por tarea con los 1 000 ejemplos y medido congelado en el test"),
        desviaciones=["una sola semilla (exploratorio, según la rama)",
                      "las palabras llegan como símbolos ya separados; las cifras publicadas son de redes que leen texto",
                      "azar: respuesta mayoritaria por tarea, tomada de la misma tabla de la rama"],
        archivo=r3 / "razonamiento3_f6_babi.csv", leer=leer_babi, agente="razonamiento3", equivalente=False,
        variabilidad="una sola semilla: no hay ± entre semillas",
        motivo_equivalencia="sin referencia humana medida; las palabras llegan como símbolos",
        comparar={"memnn": "puntaje", "lstm": "puntaje", "ngramas": "puntaje"}, referencia=None,
        azar=None, azar_derivacion="respuesta mayoritaria por tarea (de la tabla de la rama)",
        pendiente="bAbI: todavía no hay resultados en la rama razonamiento3",
    ))
    out.append(Fuente(
        prueba="instrucciones_pocos_ejemplos", variante="thinker_r3_contexto", condicion="simbolos",
        metrica="proporción de salidas idénticas a la algebraica en el episodio de las personas (300 episodios de "
                "meta-entrenamiento)",
        protocolo=("rama razonamiento3, F6b (en curso): Thinker con todo encendido + contexto temporal, meta-entrenado "
                   "con 0, 30, 100 y 300 episodios de la familia y probado en el episodio de las 25 personas"),
        desviaciones=["las personas ven pseudopalabras y círculos de colores y traen su lenguaje; el Thinker recibe símbolos",
                      "MLC se meta-entrenó con 100 000 episodios; acá, 300 como máximo",
                      "la fuente publica media ± desvío de 3 semillas, sin los valores por semilla"],
        archivo=r3 / "razonamiento3_f6_mlc.csv", leer=leer_mlc, agente="razonamiento3", agregado=True,
        variabilidad="3 semillas del Thinker en la rama razonamiento3; el episodio de prueba es el mismo (el de las "
                     "personas)",
        equivalente=False, motivo_equivalencia="símbolos en lugar de palabras y colores vistos; meta-entrenamiento mínimo",
        comparar={"mlc": "puntaje"},
        pendiente="Lake y Baroni: todavía no hay resultados en la rama razonamiento3",
    ))
    for pc, pasadas in ((1, 1), (2, 1), (4, 1), (8, 1), (1, 3)):
        ref = f"lake2018_exp1_{pc}pc" if pc in (1, 2, 4) else None
        out.append(Fuente(
            prueba="scan", variante=f"exp1_{pc}pc" + (f"_{pasadas}pasadas" if pasadas > 1 else ""),
            condicion="simbolos",
            metrica=f"exactitud de secuencia exacta en una muestra de 1 000 comandos del test ({pc} % para entrenar)",
            protocolo=(f"rama razonamiento3, F6a (exploratorio): Thinker con valores por defecto + contexto temporal; "
                       f"{pasadas} pasada(s) en línea por el {pc} % de los comandos (división al azar del experimento 1 "
                       f"de Lake & Baroni 2018) y prueba congelada en 1 000 de los comandos restantes"),
            desviaciones=["una sola semilla (exploratorio, según la rama)",
                          f"{pasadas} pasada(s) en línea por los comandos de entrenamiento; el seq2seq publicado ve "
                          "100 000 presentaciones con reemplazo",
                          "muestra de 1 000 comandos del test en lugar de todos los restantes",
                          "palabras y acciones como símbolos, igual que en el seq2seq publicado"]
                         + ([] if ref else ["con 8 % no hay cifra publicada verificada (sólo en la figura 3)"]),
            archivo=r3 / "razonamiento3_f6_scan_exp1.csv", leer=leer_scan_exp1(pc, pasadas), agente="razonamiento3",
            equivalente=False, variabilidad="una sola semilla: no hay ± entre semillas",
            motivo_equivalencia=f"{pasadas} pasada(s) en línea contra 100 000 presentaciones; se compara con un sistema",
            comparar={ref: "puntaje"} if ref else {},
            pendiente="SCAN experimento 1: todavía no hay resultados en la rama razonamiento3",
        ))
    out.append(Fuente(
        prueba="scan", variante="jump", condicion="simbolos", metrica="exactitud de secuencia exacta (división jump)",
        protocolo="SCAN, división jump: la rama razonamiento3 no la corrió todavía", desviaciones=[],
        archivo=r3 / "razonamiento3_f6_scan.parts.json", leer=None, agente="razonamiento3", equivalente=False,
        pendiente="SCAN división jump: todavía no hay resultados en la rama razonamiento3",
    ))
    return out


def leer_scan_exp1(pc: int, pasadas: int):
    """Una fila por semilla (la rama publica una, la 0) con la secuencia exacta en la muestra
    de 1 000 comandos del test y los diagnósticos de la misma tabla."""
    pre = f"{pc} % ("
    suf = f"{pasadas} pasada(s)"

    def leer(path: Path):
        med = {}
        puntaje = None
        n = None
        for r in _csv(path):
            if not r["sistema"].startswith("Thinker") or not r["tramo"].startswith(pre) or suf not in r["tramo"]:
                continue
            tramo = r["tramo"].split(suf, 1)[1].strip(", ")
            n = int(float(r["semillas"]))
            if tramo == "test (muestra de 1000)":
                if r["medida"] == "secuencia exacta":
                    puntaje = _f(r["media"])
                elif r["medida"] in ("símbolos en su posición", "primer símbolo", "largo correcto", "nunca dijo FIN"):
                    med[f"test_{r['medida'].replace(' ', '_')}"] = _f(r["media"])
            elif tramo == "comandos ya vistos" and r["medida"] == "secuencia exacta":
                med["entrenamiento_secuencia_exacta"] = _f(r["media"])
            elif tramo == "" and r["medida"].startswith("comandos del test con el mismo multiconjunto"):
                med["test_ambiguos_como_bolsa_de_palabras"] = _f(r["media"])
        if puntaje is None:
            return []
        assert n == 1, "la rama publicó más de una semilla: leer los valores por semilla"
        return [dict(semilla=0, puntaje=puntaje, medidas=med)]
    return leer


def leer_babi(path: Path):
    filas = _csv(path)
    tareas = {}
    azar = {}
    for r in filas:
        if r["sistema"].startswith("Thinker (valores por defecto") and r["tramo"].startswith("test"):
            tareas[r["medida"]] = _f(r["media"])
        if r["sistema"] == "azar (mayoritaria)":
            azar[r["medida"]] = _f(r["media"])
    if not tareas:
        return []
    med = {f"tarea_{k.split(':')[0]}": v for k, v in tareas.items()}
    med.update(tareas_resueltas_95=float(sum(v >= 0.95 for v in tareas.values())),
               azar_mayoritaria_media=float(sum(azar.values()) / len(azar)) if azar else None)
    return [dict(semilla=0, puntaje=float(sum(tareas.values()) / len(tareas)), medidas=med)]


def leer_mlc(path: Path):
    for r in _csv(path):
        if (r["sistema"].startswith("Thinker") and r["tramo"].startswith("300 episodios")
                and r["medida"] == "secuencia exacta, episodio de las personas"):
            return dict(media=_f(r["media"]), desvio=_f(r["desvio"]), n=int(float(r["semillas"])), semillas=[0, 1, 2],
                        medidas=dict(episodios_meta=300.0))
    return None


def fila_de(f: Fuente, catalogo: dict) -> dict:
    p = Prueba(id=f.prueba, variante=f.variante, condicion=f.condicion, metrica=f.metrica, protocolo=f.protocolo,
               desviaciones=list(f.desviaciones), correr=None, equivalente=f.equivalente,
               motivo_equivalencia=f.motivo_equivalencia, comparar=dict(f.comparar), referencia=f.referencia,
               azar=f.azar, azar_derivacion=f.azar_derivacion, firmas=dict(f.firmas), variabilidad=f.variabilidad)
    if not f.archivo.exists():
        fila = esquema.fila_resultado(p, [], catalogo, "")
        fila.update(estado="pendiente", motivo=f.pendiente or f"todavía no existe {f.archivo}",
                    fuente_resultado=f"{f.agente}: {f.archivo}", commit="")
        return fila
    archivos = [f.archivo]
    if f.patron:
        archivos = sorted(f.archivo.parent.glob(f.patron.format(s="*")))
    commit = commit_de(archivos)
    if f.leer is None:
        fila = esquema.fila_resultado(p, [], catalogo, commit)
        fila.update(estado="pendiente", motivo=f.pendiente, fuente_resultado=f"{f.agente}: {f.archivo}")
        return fila
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
