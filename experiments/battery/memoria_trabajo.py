"""Memoria de trabajo: detección de cambio y capacidad K, con símbolos.

Protocolo publicado (Elmore et al. 2011): N cuadrados de colores (personas: 2 a 10; monos:
2, 4 y 6), una demora breve y vuelven dos, uno con otro color: hay que tocar el que cambió.
Capacidad K con el modelo de capacidad fija de Eng et al.

Lo que hace la batería, en el idioma del Thinker: cada ítem de la muestra llega como dos
percepciones seguidas, su lugar y su color (`observe(lugar)`, `observe(color)`), y después
un silencio (la demora). La prueba es de un solo ítem: un lugar de la muestra y un color
(el mismo o uno que no estaba en la muestra, con probabilidad 1/2), y la red responde SÍ
(cambió) o NO. Se entrena en línea con episodios de tamaño al azar y después una copia
congelada resuelve 100 episodios por tamaño. K por tamaño con la fórmula de Cowan (2001)
para un solo ítem, K = N (aciertos - falsas alarmas), y un K único ajustado por mínimos
cuadrados al modelo de capacidad fija (H - FA = min(K, N) / N).

Por qué un solo ítem y no dos: el Thinker responde con un símbolo que tiene en el foco o
con un juicio sobre su estado; para «tocar» el lugar que cambió entre dos tendría que
sostener uno y responder con el otro, algo que su entrada y salida no permiten. Es una
desviación de procedimiento que cambia la fórmula de K.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from .base import NoAplicable, Prueba
from .harlow import CONFIG, thinker

TAMANOS = (2, 4, 6, 8, 10)
LUGARES, COLORES = 10, 12
P_ENTRENO = 3000
N_PRUEBA = 100


def episodio(th, rng, N: int, aprender: bool = True) -> tuple[bool, bool]:
    """Un ensayo de detección de cambio. Devuelve (¿cambió?, ¿dijo que cambió?)."""
    SI, NO = LUGARES + COLORES, LUGARES + COLORES + 1
    lugares = rng.choice(LUGARES, size=N, replace=False)
    colores = LUGARES + rng.choice(COLORES, size=N, replace=False)
    th.reset()
    for l, c in zip(lugares, colores):
        th.observe(int(l)); th.observe(int(c))
    th.observe(None)                                      # la demora
    j = int(rng.integers(N))
    cambio = bool(rng.random() < 0.5)
    color = int(colores[j])
    if cambio:
        libres = sorted(set(range(LUGARES, LUGARES + COLORES)) - set(int(x) for x in colores))
        color = int(libres[int(rng.integers(len(libres)))])
    out = th.answer(int(lugares[j]), probe=color, explore=aprender)
    if aprender:
        th.feedback(target=SI if cambio else NO)
    return cambio, bool(out == SI)


def ajustar_K(hfa: dict) -> float:
    """K único del modelo de capacidad fija para un solo ítem: H - FA = min(K, N) / N."""
    grid = np.linspace(0.0, max(TAMANOS), 1001)
    err = [sum((hfa[N] - min(k, N) / N) ** 2 for N in hfa) for k in grid]
    return float(grid[int(np.argmin(err))])


CONFIGS = {"aprender_a_aprender": CONFIG,
           "compuerta": {k: v for k, v in CONFIG.items() if k != "prev_io"}}


def correr(semilla: int, escala: float = 1.0, config: dict | None = None, solo_entrenar: bool = False) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    th = thinker(LUGARES + COLORES + 2, semilla, config)
    n_ent = max(200, int(round(P_ENTRENO * escala)))
    n_pr = max(20, int(round(N_PRUEBA * escala)))
    ok = []
    for _ in range(n_ent):
        cambio, dijo = episodio(th, rng, int(rng.choice(TAMANOS)))
        ok.append(cambio == dijo)
    if solo_entrenar:
        return dict(entrenamiento=float(np.mean(ok[-500:])))
    fz = copy.deepcopy(th); fz.plastic = False
    erng = np.random.default_rng(20_000 + semilla)
    hfa, kn, acc = {}, {}, {}
    for N in TAMANOS:
        r = [episodio(fz, erng, N, aprender=False) for _ in range(n_pr)]
        cam = [d for c, d in r if c]; igu = [d for c, d in r if not c]
        H = float(np.mean(cam)) if cam else 0.0; FA = float(np.mean(igu)) if igu else 0.0
        hfa[N] = H - FA; kn[N] = N * (H - FA); acc[N] = float(np.mean([c == d for c, d in r]))
    K = ajustar_K(hfa)
    med = dict(K=K, exactitud_entrenamiento_ultimos500=float(np.mean(ok[-500:])))
    for N in TAMANOS:
        med[f"K_N{N}"] = kn[N]; med[f"exactitud_N{N}"] = acc[N]
    med["techo_de_la_prueba"] = bool(K >= max(TAMANOS) - 1e-9)
    return dict(puntaje=K, medidas=med, segundos=time.time() - t0)


def validar(config: dict, semilla: int, escala: float = 1.0) -> float:
    """Criterio de validación: exactitud en los últimos 500 episodios de entrenamiento."""
    return correr(semilla, escala, config, solo_entrenar=True)["entrenamiento"]


def correr_dos_items(semilla: int, escala: float = 1.0) -> dict:
    raise NoAplicable("el procedimiento de Elmore et al. (vuelven dos ítems y hay que tocar el que cambió) no se puede "
                      "presentar sin cambiar la arquitectura: el Thinker responde con un símbolo de su foco o con un juicio "
                      "sobre su estado, y para señalar el lugar que cambió entre dos tendría que sostener uno y responder "
                      "con el otro; la batería mide la versión de un solo ítem")


PRUEBAS = [
    Prueba(
        id="deteccion_de_cambio", variante="simbolos", condicion="simbolos",
        metrica="capacidad K (modelo de capacidad fija, prueba de un ítem)",
        protocolo=(f"{P_ENTRENO} episodios de entrenamiento en línea con N al azar en {TAMANOS}; {LUGARES} lugares y "
                   f"{COLORES} colores como símbolos; cada ítem llega como lugar y color seguidos; prueba de un ítem "
                   f"(cambia con probabilidad 1/2); {N_PRUEBA} episodios congelados por tamaño"),
        desviaciones=[
            "condición de entrada: símbolos que llegan de a uno, no una pantalla con todos los cuadrados a la vez "
            "(la red no tiene que codificar en paralelo una escena breve)",
            "prueba de un solo ítem (sí/no) con la fórmula de Cowan, no dos ítems y tocar el que cambió con la de Eng",
            "la memoria rápida del Thinker es una tabla exacta por episodio: si el control aprende a usarla, la "
            "capacidad la limita el diseño (o el techo de la prueba, N = 10), no la red",
            "la demora es un silencio: el Thinker no tiene escala de tiempo",
        ],
        correr=correr, version=1, semillas=tuple(range(8)),
        configs=CONFIGS, validar=validar, perfiles=True,
        criterio_validacion="exactitud en los últimos 500 episodios de entrenamiento (semillas de validación)",
        equivalente=False, motivo_equivalencia="otro procedimiento de prueba y memoria episódica exacta por diseño",
        techo=float(max(TAMANOS)),
        comparar={"elmore2011_rhesus_colores": "puntaje", "cowan2001": "puntaje"},
    ),
    Prueba(
        id="deteccion_de_cambio", variante="dos_items", condicion="simbolos",
        metrica="capacidad K con el procedimiento publicado (dos ítems, tocar el que cambió)",
        protocolo="el de Elmore et al. (2011)", desviaciones=[], correr=correr_dos_items, version=1, semillas=(0,),
        equivalente=False, motivo_equivalencia="no se pudo correr",
    ),
]
