"""Concepto igual/distinto con transferencia a estímulos nuevos, en función del tamaño del
conjunto de entrenamiento.

Protocolo publicado (Katz, Wright y colaboradores, resumido en Katz, Sturz y Wright 2010):
dos figuras; si son la misma se responde «igual» y si no, «distinto». Sesiones de 100
ensayos, mitad iguales. Se entrena con un conjunto de 8 figuras, se prueba con figuras
nuevas y se duplica el conjunto (16, 32, ...). Con 8 figuras ni monos ni palomas
transfieren; la transferencia iguala a la línea de base entre 64 y 128 figuras en monos y
cascanueces y con 256 en palomas (Wright et al. 2016).

Lo que hace la batería, para cada tamaño k en 8, 16, ..., 256 (el conjunto se duplica
agregando figuras): sesiones de 100 ensayos con premio (50 iguales y 50 distintos) hasta 80
% en una sesión o 30 sesiones; después, sobre una copia congelada, 100 ensayos de
transferencia con figuras nunca vistas (cada una se usa una sola vez en toda la corrida) y
100 de línea de base con las del conjunto.

  «imagenes»: las figuras son caracteres de Omniglot (una imagen por carácter, 28 x 28; las
              dos figuras una arriba de la otra, 56 x 28) y la red es `depth2` de exp_vision
              con dos salidas (igual, distinto).
  «simbolos»: las figuras son símbolos y la red es el Thinker (configuración elegida en
              validación); los dos llegan como ítems separados.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from .base import Prueba
from .harlow import CONFIGS_ELECCION, elegir, thinker

TAMANOS = (8, 16, 32, 64, 128, 256)
SESION = 100
CRITERIO = 0.8
MAX_SESIONES = 30
N_TRANSFER = 100


def _pares(rng, conj: list, n: int):
    """n ensayos, mitad iguales, con ítems de `conj` (con reposición entre ensayos)."""
    out = []
    for i in rng.permutation(n):
        if i < n // 2:
            x = conj[int(rng.integers(len(conj)))]; out.append((x, x, True))
        else:
            a, b = rng.choice(len(conj), size=2, replace=False); out.append((conj[a], conj[b], False))
    return out


def _nuevos(rng, pool: list, n: int):
    """n ensayos de transferencia con ítems nunca vistos, cada uno usado una vez."""
    out = []
    for i in rng.permutation(n):
        if i < n // 2:
            x = pool.pop(); out.append((x, x, True))
        else:
            a = pool.pop(); b = pool.pop(); out.append((a, b, False))
    return out


def curva(paso, congelar, rng, items: list, pool: list, tamanos=TAMANOS, max_ses=MAX_SESIONES) -> dict:
    """`paso(a, b, igual, aprender) -> acierto` presenta un ensayo; `congelar()` devuelve una
    función igual a `paso` sobre una copia que no aprende."""
    med = {}
    for k in tamanos:
        conj = items[:k]
        ses = 0
        for ses in range(1, max_ses + 1):
            acc = np.mean([paso(a, b, ig, True) for a, b, ig in _pares(rng, conj, SESION)])
            if acc >= CRITERIO:
                break
        fz = congelar()
        tr = np.mean([fz(a, b, ig, False) for a, b, ig in _nuevos(rng, pool, N_TRANSFER)])
        base = np.mean([fz(a, b, ig, False) for a, b, ig in _pares(rng, conj, N_TRANSFER)])
        med[f"transferencia_{k}"] = float(tr); med[f"base_{k}"] = float(base)
        med[f"sesiones_{k}"] = float(ses); med[f"llego_{k}"] = bool(acc >= CRITERIO)
    return med


# ---------------------------------------------------------------- símbolos
def correr_simbolos(semilla: int, escala: float = 1.0, config: dict | None = None, solo_entrenar: bool = False) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    tam = TAMANOS if escala >= 1.0 else TAMANOS[:2]
    n_items = max(tam); n_pool = len(tam) * (N_TRANSFER // 2) * 3
    N = n_items + n_pool + 2
    IGUAL, DISTINTO = N - 2, N - 1
    perm = [int(x) for x in rng.permutation(N - 2)]
    items, pool = perm[:n_items], perm[n_items:]
    th = thinker(N, semilla, config)
    estado = dict(net=th)

    def paso_de(net):
        def paso(a, b, igual, aprender):
            net.reset()
            out = elegir(net, a, b, aprender)
            y = IGUAL if igual else DISTINTO
            if aprender:
                net.feedback(target=y)
            return bool(out == y)
        return paso

    def congelar():
        fz = copy.deepcopy(estado["net"]); fz.plastic = False
        return paso_de(fz)
    if solo_entrenar:
        paso = paso_de(th)
        accs = [paso(a, b, ig, True) for _ in range(10) for a, b, ig in _pares(rng, items[:8], SESION)]
        return dict(entrenamiento=float(np.mean(accs[-SESION:])))
    med = curva(paso_de(th), congelar, rng, items, pool, tam)
    k = max(tam)
    return dict(puntaje=med[f"transferencia_{k}"], medidas=dict(med, transferencia_8=med["transferencia_8"]),
                segundos=time.time() - t0)


def validar_simbolos(config: dict, semilla: int, escala: float = 1.0) -> float:
    """Criterio de validación: exactitud en la décima sesión de entrenamiento con 8 figuras."""
    return correr_simbolos(semilla, escala, config, solo_entrenar=True)["entrenamiento"]


# ---------------------------------------------------------------- imágenes
def _figuras(rng, n: int):
    """Una imagen (el primer dibujante) por carácter de Omniglot, de fondo y de evaluación,
    en un orden al azar: (n, 28, 28) uint8. Devuelve además los caracteres de fondo que no
    se usaron (para los controles) con todos sus dibujos."""
    from . import omniglot
    X, y, _ = omniglot.cargar_fondo()
    primero = np.r_[0, np.flatnonzero(np.diff(y)) + 1]                 # primera imagen de cada carácter
    ev = omniglot.cargar_evaluacion_primeras()
    todas = np.concatenate([X[primero], ev])
    perm = rng.permutation(len(todas))
    libres = [int(c) for c in perm[n:] if c < len(primero)]            # caracteres de fondo sin usar
    dibujos = {c: X[y == c] for c in libres}
    return todas[perm[:n]], dibujos


def correr_imagenes(semilla: int, escala: float = 1.0) -> dict:
    from bioelastic import Cortex, ImageEncoder
    from .omniglot import V1
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    tam = TAMANOS if escala >= 1.0 else TAMANOS[:2]
    max_ses = MAX_SESIONES if escala >= 1.0 else 3
    n_items = max(tam); n_pool = len(tam) * (N_TRANSFER // 2) * 3
    figs, dibujos = _figuras(rng, n_items + n_pool)
    items = list(range(n_items)); pool = list(range(n_items, n_items + n_pool))
    enc = ImageEncoder((56, 28), blur=0.4)
    cfg = dict(areas=[dict(V1, local=dict(lr=0.5))], heads=[dict(source="input")],
               projection=dict(source="all", balance=True), replay=0.5, joint=False, arbiter=0.01)
    net = Cortex((56, 28, 1), 2, seed=semilla, **cfg)
    estado = dict(net=net)

    def x_de(a, b):
        return enc(np.concatenate([figs[a], figs[b]], axis=0)).reshape(56, 28, 1)

    def paso_de(red, aprende):
        def paso(a, b, igual, aprender):
            y = 0 if igual else 1
            if aprender and aprende:
                return bool(red.step(x_de(a, b), y) == y)
            return bool(red.forward(x_de(a, b)) == y)
        return paso

    def congelar():
        fz = copy.deepcopy(estado["net"])
        for ar in fz.areas:
            ar._cache = None
        return paso_de(fz, False)
    med = curva(paso_de(net, True), congelar, rng, items, pool, tam, max_ses)
    # Controles (no son parte del protocolo publicado): ¿«igual» generaliza más allá de la
    # identidad de los píxeles? Con la red final congelada, 100 ensayos cada uno con
    # caracteres de fondo no usados: (a) «igual» = dos dibujos distintos del mismo carácter;
    # (b) «igual» = la misma imagen con la de abajo corrida 2 píxeles a la derecha.
    fz = copy.deepcopy(estado["net"])
    for ar in fz.areas:
        ar._cache = None
    libres = list(dibujos)
    rng.shuffle(libres)
    n_c = N_TRANSFER if escala >= 1.0 else 20

    def ensayo_control(arriba, abajo, igual):
        x = enc(np.concatenate([arriba, abajo], axis=0)).reshape(56, 28, 1)
        return bool(fz.forward(x) == (0 if igual else 1))
    for nombre, hacer in (("otro_dibujo", lambda d: d[1]), ("corrido_2px", lambda d: np.roll(d[0], 2, axis=1))):
        cand = libres[:3 * n_c // 2] if len(libres) >= 3 * n_c // 2 else libres
        libres = libres[len(cand):]
        hits_i, hits_d = [], []
        for j in range(0, len(cand) - 2, 3):
            a, b, c = dibujos[cand[j]], dibujos[cand[j + 1]], dibujos[cand[j + 2]]
            hits_i.append(ensayo_control(a[0], hacer(a), True))
            hits_d.append(ensayo_control(b[0], c[0], False))
        med[f"control_{nombre}_igual"] = float(np.mean(hits_i)) if hits_i else None
        med[f"control_{nombre}_distinto"] = float(np.mean(hits_d)) if hits_d else None
        med[f"control_{nombre}"] = float(np.mean(hits_i + hits_d)) if hits_i else None
    k = max(tam)
    return dict(puntaje=med[f"transferencia_{k}"], medidas=med, segundos=time.time() - t0)


_COMP = {"katz2010_8figuras": "transferencia_8"}
_DESV = [
    f"criterio de 80 % en una sesión (máximo {MAX_SESIONES} sesiones) y 100 ensayos de transferencia sin premio sobre "
    "una copia congelada: elegidos por la batería (los de los artículos no se leyeron)",
    "los animales responden tocando una figura o un rectángulo; la red elige entre dos salidas",
]
PRUEBAS = [
    Prueba(
        id="igual_distinto", variante="imagenes", condicion="imagenes",
        metrica="exactitud de transferencia a figuras nunca vistas con el conjunto de 256 figuras",
        protocolo=("depth2 con dos salidas sobre las dos figuras apiladas (56 x 28); conjuntos de 8 a 256 caracteres de "
                   "Omniglot que se duplican; entrenamiento hasta 80 % y transferencia con figuras nuevas"),
        desviaciones=_DESV + ["figuras: caracteres de Omniglot en blanco y negro a 28 x 28, no fotografías en color",
                              "controles agregados por la batería (no publicados): «igual» con dos dibujos del mismo "
                              "carácter y con la copia corrida 2 píxeles, para ver si el concepto va más allá de la "
                              "identidad de los píxeles"],
        correr=correr_imagenes, version=2, semillas=(0, 1, 2),
        equivalente=True, motivo_equivalencia="la red ve las figuras como imágenes, igual que los animales",
        referencia=None, comparar=_COMP,
        firmas={},
    ),
    Prueba(
        id="igual_distinto", variante="simbolos", condicion="simbolos",
        metrica="exactitud de transferencia a símbolos nunca vistos con el conjunto de 256",
        protocolo="Thinker; conjuntos de 8 a 256 símbolos que se duplican; entrenamiento hasta 80 % y transferencia",
        desviaciones=_DESV + ["condición de entrada: símbolos; la igualdad de dos símbolos la detecta la compuerta "
                              "por diseño (detección de coincidencia), no la aprende la red"],
        correr=correr_simbolos, version=1, semillas=(0, 1, 2, 3, 4),
        equivalente=False, motivo_equivalencia="la identidad de los símbolos viene dada",
        referencia=None, comparar=_COMP,
        configs=CONFIGS_ELECCION, validar=validar_simbolos,
        criterio_validacion="exactitud en la décima sesión con el conjunto de 8 (semillas de validación)",
    ),
]
