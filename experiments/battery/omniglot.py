"""Omniglot: clasificar con un solo ejemplo, 20 opciones dentro del mismo alfabeto.

Protocolo publicado (Lake et al. 2015; repositorio brendenlake/omniglot): 20 corridas;
en cada una, 20 caracteres nuevos de un alfabeto de evaluación con un ejemplo cada uno y
20 imágenes de prueba; para cada imagen de prueba hay que elegir cuál de los 20 es. Antes
se puede aprender del conjunto de fondo (30 alfabetos, 964 caracteres), que no comparte
alfabetos ni caracteres con la evaluación. Personas: error de 4,5 % como máximo.

Lo que hace la batería:
  * carga las 20 corridas oficiales de all_runs.zip y lo comprueba reproduciendo la línea
    base del repositorio (distancia de Hausdorff modificada: 38,8 % de error; la batería
    da 38,75 %);
  * «imagenes»: el área sensorial publicada (V1 de exp_vision: rf 5, pool 2, 256 rasgos)
    aprende sin etiquetas, en una sola pasada en línea, el conjunto de fondo (19 280
    imágenes); después, en cada corrida, un área de asociación nueva (`BioElasticNet`,
    opciones por defecto) aprende los 20 ejemplos sobre el código de V1 (que queda quieta)
    y clasifica las 20 de prueba. Es el aprendizaje de un ejemplo propio de la red: una
    neurona nueva por patrón nuevo;
  * «pixeles»: lo mismo sin conjunto de fondo, con el área de asociación sobre los píxeles
    (es el mecanismo de la memoria de `depth2`, que lee la entrada).

Por qué V1 sin etiquetas y no la red `depth2` entera: con 964 clases, una pasada de
`depth2` por el fondo tarda unos 100 minutos por semilla en esta máquina compartida (300
imágenes en 92 s). Queda pendiente; V1 es la misma área, sin el error local.

Las imágenes de 105 x 105 se invierten (trazo = 1) y se reducen a 28 x 28 (relleno a
112 y promedio en bloques de 4 x 4), el tamaño con el que está publicada la red.
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np

from .base import NoAplicable, Prueba

DIR = Path(__file__).resolve().parents[2] / "ext_data" / "omniglot"
CACHE = DIR / "cache"

# Copia literal de exp_vision.V1 (la batería no importa los scripts de experimentos:
# `common` reordena sys.path y mezclaría versiones de la red).
V1 = dict(rf=5, pool=2, max_features=256, theta=0.95, floor=0.3, k=8)


# ---------------------------------------------------------------- datos
def _tinta(zf: zipfile.ZipFile, nombre: str) -> np.ndarray:
    from PIL import Image
    with Image.open(io.BytesIO(zf.read(nombre))) as im:
        return ~np.asarray(im.convert("1"), dtype=bool)          # True = trazo


def a28(tinta: np.ndarray) -> np.ndarray:
    """105 x 105 (bool, trazo) -> 28 x 28 uint8 (promedio en bloques de 4 x 4 sobre 112 x 112)."""
    x = np.zeros((112, 112), dtype=np.float32)
    x[3:108, 3:108] = tinta
    return np.round(255.0 * x.reshape(28, 4, 28, 4).mean(axis=(1, 3))).astype(np.uint8)


def _zip(nombre: str) -> zipfile.ZipFile:
    p = DIR / nombre
    if not p.exists():
        raise NoAplicable(f"faltan los datos de Omniglot ({p}); ver catalogo.json, prueba omniglot")
    return zipfile.ZipFile(p)


def cargar_fondo():
    """Conjunto de fondo a 28 x 28: (X uint8 (N, 28, 28), y clase 0..963, alfabeto 0..29)."""
    c = CACHE / "fondo28.npz"
    if c.exists():
        with np.load(c, allow_pickle=False) as z:
            return z["X"], z["y"], z["alfabeto"]
    zf = _zip("images_background.zip")
    nombres = sorted(n for n in zf.namelist() if n.endswith(".png"))
    clases = sorted({n.rsplit("/", 1)[0] for n in nombres})
    alf = sorted({c.split("/")[1] for c in clases})
    ci = {c: i for i, c in enumerate(clases)}; ai = {a: i for i, a in enumerate(alf)}
    X = np.stack([a28(_tinta(zf, n)) for n in nombres])
    y = np.array([ci[n.rsplit("/", 1)[0]] for n in nombres], dtype=np.int64)
    a = np.array([ai[n.split("/")[1]] for n in nombres], dtype=np.int64)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(c, X=X, y=y, alfabeto=a)
    return X, y, a


def cargar_evaluacion_primeras() -> np.ndarray:
    """La primera imagen (en orden de archivo) de cada uno de los 659 caracteres del
    conjunto de evaluación, a 28 x 28."""
    c = CACHE / "evaluacion_primeras28.npz"
    if c.exists():
        with np.load(c, allow_pickle=False) as z:
            return z["X"]
    zf = _zip("images_evaluation.zip")
    nombres = sorted(n for n in zf.namelist() if n.endswith(".png"))
    vistos, X = set(), []
    for n in nombres:
        car = n.rsplit("/", 1)[0]
        if car not in vistos:
            vistos.add(car); X.append(a28(_tinta(zf, n)))
    X = np.stack(X)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(c, X=X)
    return X


def cargar_corridas(tinta: bool = False) -> list[dict]:
    """Las 20 corridas oficiales. Cada una: entrenamiento (20 imágenes, en el orden de los
    archivos), prueba (20) y, para cada imagen de prueba, el índice de su par correcto."""
    zf = _zip("all_runs.zip")
    out = []
    for r in range(1, 21):
        run = f"run{r:02d}"
        pares = [l.split() for l in zf.read(f"{run}/class_labels.txt").decode().splitlines() if l.strip()]
        test = sorted(p[0] for p in pares)
        train = sorted(p[1] for p in pares)
        resp = dict(pares)
        conv = (lambda n: _tinta(zf, n)) if tinta else (lambda n: a28(_tinta(zf, n)))
        out.append(dict(entrenamiento=[conv(n) for n in train], prueba=[conv(n) for n in test],
                        correcta=[train.index(resp[t]) for t in test]))
    return out


# ---------------------------------------------------------------- línea base del repositorio
def _puntos(t: np.ndarray) -> np.ndarray:
    r, c = np.nonzero(t)
    D = np.stack([r, c], axis=1).astype(float)
    return D - D.mean(axis=0)


def hausdorff_modificada(A: np.ndarray, B: np.ndarray) -> float:
    from scipy.spatial.distance import cdist
    D = cdist(A, B)
    return max(D.min(axis=1).mean(), D.min(axis=0).mean())


def error_hausdorff() -> float:
    """Error de la línea base del repositorio con estas corridas (debe dar 38,8 %)."""
    errs = []
    for run in cargar_corridas(tinta=True):
        tr = [_puntos(t) for t in run["entrenamiento"]]
        ok = 0
        for t, c in zip(run["prueba"], run["correcta"]):
            p = _puntos(t)
            ok += int(np.argmin([hausdorff_modificada(p, q) for q in tr]) == c)
        errs.append(100.0 * (1 - ok / len(run["prueba"])))
    return float(np.mean(errs))


# ---------------------------------------------------------------- red
def _encoder():
    from bioelastic import ImageEncoder
    enc = ImageEncoder((28, 28), blur=0.4)
    return lambda x: enc(x).reshape(28, 28, 1)


def entrenar_v1(semilla: int, escala: float = 1.0):
    """V1 aprende sin etiquetas el conjunto de fondo en una pasada (orden de la semilla;
    semilla del área = semilla + 100, como el área 0 de un `Cortex`)."""
    from bioelastic import FeatureArea
    X, _, _ = cargar_fondo()
    orden = np.random.default_rng(semilla).permutation(len(X))
    if escala < 1.0:
        orden = orden[:max(500, int(round(len(X) * escala)))]
    E = _encoder()
    area = FeatureArea((28, 28, 1), seed=semilla + 100, **V1)
    t0 = time.time()
    for i in orden:
        area.encode(E(X[i])); area.learn(None)
    return area, dict(imagenes=len(orden), segundos=time.time() - t0, **area.stats())


def un_ejemplo(codigo, runs, semilla: int) -> dict:
    """Exactitud de un área de asociación nueva por corrida sobre `codigo(img)`."""
    from bioelastic import BioElasticNet
    accs = []
    for k, run in enumerate(runs):
        tr = [codigo(x) for x in run["entrenamiento"]]
        te = [codigo(x) for x in run["prueba"]]
        net = BioElasticNet(int(tr[0].size), len(tr), seed=1000 * semilla + k)
        for c, x in enumerate(tr):
            net.forward(x.ravel()); net.teach(c)
        accs.append(float(np.mean([net.forward(x.ravel()) == c for x, c in zip(te, run["correcta"])])))
    return dict(exactitud=float(np.mean(accs)), por_corrida=accs)


def correr_imagenes(semilla: int, escala: float = 1.0) -> dict:
    t0 = time.time()
    runs = cargar_corridas()
    area, info = entrenar_v1(semilla, escala)
    E = _encoder()
    r = un_ejemplo(lambda x: area.encode(E(x)), runs, semilla)
    med = dict(exactitud=r["exactitud"], error=1.0 - r["exactitud"], por_corrida=r["por_corrida"],
               fondo_imagenes=float(info["imagenes"]), fondo_segundos=info["segundos"],
               rasgos_v1=float(info["features"]))
    return dict(puntaje=r["exactitud"], medidas=med, segundos=time.time() - t0)


def correr_pixeles(semilla: int, escala: float = 1.0) -> dict:
    t0 = time.time()
    runs = cargar_corridas()
    E = _encoder()
    r = un_ejemplo(lambda x: E(x), runs, semilla)
    return dict(puntaje=r["exactitud"], medidas=dict(exactitud=r["exactitud"], por_corrida=r["por_corrida"]),
                segundos=time.time() - t0)


_DESV = [
    "las imágenes se reducen de 105 x 105 a 28 x 28 (el tamaño con que está publicada la red)",
    "las personas traen una vida de experiencia visual y con escritura; la red sólo el conjunto de fondo",
    "las personas respondían rápido y sin incentivo: su error de 4,5 % es una cota superior",
]
PRUEBAS = [
    Prueba(
        id="omniglot", variante="imagenes", condicion="imagenes",
        metrica="exactitud media en las 20 corridas oficiales (400 ensayos)",
        protocolo=("V1 publicada aprende sin etiquetas una pasada en línea por el conjunto de fondo; por corrida, un "
                   "área de asociación nueva aprende un ejemplo de cada uno de los 20 caracteres sobre el código de V1 "
                   "congelada y clasifica las 20 imágenes de prueba"),
        desviaciones=_DESV + ["la elección de la lectura de un ejemplo (área de asociación nueva sobre el código de V1) es "
                              "de la batería: la red no tiene una vía publicada para clases nuevas con un ejemplo",
                              "V1 sin el error local de depth2 (con 964 clases, depth2 tarda ~100 min por semilla aquí)"],
        correr=correr_imagenes, version=1, semillas=(0, 1, 2),
        variabilidad="la semilla cambia V1 (orden del conjunto de fondo y semilla del área); las 20 corridas de prueba "
                     "son siempre las oficiales y el lector de un ejemplo es determinista",
        equivalente=True, motivo_equivalencia="mismas imágenes y mismo protocolo de 20 opciones, a menor resolución",
        techo=1.0,
        comparar={"bpl": "puntaje", "convnet_simple": "puntaje", "prototipos": "puntaje", "hausdorff": "puntaje"},
    ),
    Prueba(
        id="omniglot", variante="pixeles", condicion="imagenes",
        metrica="exactitud media en las 20 corridas oficiales, sin conjunto de fondo",
        protocolo="por corrida, un área de asociación nueva aprende un ejemplo de cada carácter sobre los píxeles",
        desviaciones=_DESV + ["sin conjunto de fondo: control de lo que aporta lo aprendido antes"],
        correr=correr_pixeles, version=1, semillas=(0, 1, 2),
        variabilidad="determinista: el lector de un ejemplo no usa la semilla y las corridas son las oficiales",
        equivalente=True, motivo_equivalencia="mismas imágenes y mismo protocolo, a menor resolución",
        techo=1.0,
        comparar={"hausdorff": "puntaje", "convnet_simple": "puntaje"},
    ),
]
