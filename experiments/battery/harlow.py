"""Conjuntos de aprendizaje de Harlow (1949, 1950) con símbolos.

Protocolo publicado (Harlow 1950, resumen de la APA): 11 monos rhesus; 32 problemas
preliminares de 50 ensayos y después 200 problemas de 6 ensayos; en cada problema, dos
objetos nunca vistos, uno premiado, la posición cambia entre ensayos. Medida: acierto en
el ensayo 2 (el primero en el que se puede usar lo que pasó en el 1).

Lo que hace la batería: lo mismo, en línea (la red aprende mientras se mide, como los
monos), con 200 problemas más de 6 ensayos para comparar con la cifra «tras 400
problemas» de Warren (1965). Al final de los 200 y de los 400, una copia congelada
(`plastic = False`, sin exploración) resuelve 100 problemas nuevos: es la medida que usa
el README («pesos congelados»).

Red: `Thinker` con la configuración de aprender a aprender documentada en v0.3-dev
(memoria, habla interna, juicio, 10 pasos, compuerta y `prev_io`: el estado ve, conjugadas,
la última respuesta y si fue premiada). Los objetos llegan como dos ítems separados por la
compuerta, en el orden del ensayo. El resultado no se muestra como percepción: la única
información del resultado es la recompensa (que con dos objetos equivale al objetivo).
"""
from __future__ import annotations

import copy
import time

import numpy as np

from .base import NoAplicable, Prueba

CONFIG = dict(memory=True, speak=True, judge=True, max_steps=10, gate=True, prev_io=True)
# Configuraciones documentadas del Thinker entre las que se elige en validación (pruebas de
# elección con recompensa que no son la de Harlow): la de v0.3-dev por defecto (sólo DECIR,
# sin memoria rápida) y la de aprender a aprender.
CONFIGS_ELECCION = {"defecto": {}, "aprender_a_aprender": CONFIG}
# Perfiles de opciones de la rama razonamiento 3 (fusionadas en v0.3-dev 8fb949e, apagadas por
# defecto). Cada perfil se suma a cada configuración candidata de la prueba y la elección en
# validación se hace dentro del perfil. «R3+ts» agrega los conjuntos de tareas: con todo
# encendido rompen la multitarea (README de 8fb949e), por eso va aparte y marcado.
R3 = dict(prev_io=True, back=True, cogmap=True, modulators=True)
PERFILES = {"defecto": {}, "R3": R3, "R3+ts": dict(R3, task_sets=True)}
NOTA_PERFIL = {"defecto": "opciones documentadas de v0.3-dev, sin las de razonamiento 3",
               "R3": "cada configuración candidata + prev_io, back, cogmap y modulators (sin task_sets)",
               "R3+ts": "R3 + task_sets (marcado: con todo encendido rompe la multitarea, README de 8fb949e)"}
P_PRELIM, T_PRELIM = 32, 50
P_SEIS, T_SEIS = 400, 6
N_CONGELADO = 100


def thinker(n: int, seed: int, config: dict | None = None):
    from bioelastic.thinker import Thinker
    kw = dict(CONFIG if config is None else config)
    try:
        return Thinker(n, seed=seed, **kw)
    except TypeError as e:                    # un commit sin esas opciones
        raise NoAplicable(f"el Thinker de este commit no acepta la configuración: {e}") from e


def elegir(th, a: int, b: int, explore: bool) -> int:
    """Elección entre dos ítems. Con compuerta llegan como dos ítems separados (la red decide
    qué enfocar y qué sostener); sin compuerta, juntos en el foco (como B0 en
    exp_metalearning), porque sin compuerta el segundo término sólo se sostiene y el
    conocimiento no lo ve."""
    if getattr(th, "gate", False):
        return th.answer(a, probe=b, explore=explore)
    return th.answer((a, b), explore=explore)


class Simbolos:
    """Símbolos nunca vistos: cada pedido devuelve los siguientes de una permutación."""

    def __init__(self, n: int, rng):
        self.perm = rng.permutation(n); self.i = 0

    def tomar(self, k: int) -> list:
        out = [int(x) for x in self.perm[self.i:self.i + k]]
        if len(out) < k:
            raise RuntimeError("se acabaron los símbolos nuevos")
        self.i += k
        return out


def problema(th, rng, objs, T: int, explore: bool = True) -> list:
    """Un problema de discriminación: devuelve los aciertos por ensayo."""
    bueno = objs[int(rng.integers(2))]
    th.reset()
    hits = []
    for _ in range(T):
        a, b = (objs[i] for i in rng.permutation(2))
        out = elegir(th, a, b, explore)
        hits.append(bool(out == bueno))
        th.feedback(target=bueno)             # congelado no aprende; `prev_io` ve la recompensa
    return hits


def problema_percibido(th, rng, objs, simbolo, T: int, explore: bool = True) -> tuple[list, bool]:
    """Un problema con objetos vistos: `simbolo[o]` es lo que la percepción dice de cada
    objeto. Si los dos objetos reciben el mismo símbolo, la red no los distingue: su
    respuesta se resuelve al azar entre los dos (moneda del entorno) y la recompensa es la de
    ese objeto. Devuelve (aciertos, ¿colisión?)."""
    bueno = objs[int(rng.integers(2))]
    s = {o: simbolo[o] for o in objs}
    colision = s[objs[0]] == s[objs[1]]
    th.reset()
    hits = []
    for _ in range(T):
        a, b = (objs[i] for i in rng.permutation(2))
        out = elegir(th, s[a], s[b], explore)
        if colision:
            elegido = (a, b)[int(rng.integers(2))] if out == s[a] else None
        else:
            elegido = a if out == s[a] else (b if out == s[b] else None)
        ok = elegido == bueno
        hits.append(bool(ok))
        th.feedback(target=s[bueno], reward=1.0 if ok else -1.0)
    return hits, bool(colision)


def correr_imagenes(semilla: int, escala: float = 1.0, config: dict | None = None) -> dict:
    """Harlow con objetos vistos: caracteres de Omniglot que la percepción entrenada en MNIST
    (depth2) nombra con su vocabulario de 10 dígitos; el Thinker recibe ese nombre."""
    from . import omniglot, percepcion
    t0 = time.time()
    p_pre = max(8, int(round(P_PRELIM * escala)))
    p_seis = max(20, int(round(P_SEIS * escala)))
    n = percepcion.N_MNIST if escala >= 1.0 else max(1000, int(percepcion.N_MNIST * escala))
    net, _ = percepcion.cortex_mnist(semilla, n, log=percepcion.log_stderr)
    net = percepcion.congelar(net)
    E = percepcion.encoder()
    X, y, _ = omniglot.cargar_fondo()
    primero = np.r_[0, np.flatnonzero(np.diff(y)) + 1]
    figs = np.concatenate([X[primero], omniglot.cargar_evaluacion_primeras()])
    rng = np.random.default_rng(semilla)
    orden = [int(i) for i in rng.permutation(len(figs))]
    n_obj = 2 * (p_pre + p_seis)
    if n_obj > len(orden):
        raise NoAplicable("no alcanzan los caracteres de Omniglot para objetos nunca vistos")
    objs = orden[:n_obj]
    simbolo = {o: int(net.forward(E(figs[o]))) for o in objs}
    th = thinker(10, semilla, config)
    it = iter(range(0, n_obj, 2))
    pre, col_pre = [], []
    for _ in range(p_pre):
        j = next(it); h, c = problema_percibido(th, rng, objs[j:j + 2], simbolo, T_PRELIM)
        pre.append(h); col_pre.append(c)
    seis, col = [], []
    for _ in range(p_seis):
        j = next(it); h, c = problema_percibido(th, rng, objs[j:j + 2], simbolo, T_SEIS)
        seis.append(h); col.append(c)
    pre = np.array(pre, dtype=float); seis = np.array(seis, dtype=float); col = np.array(col)
    e2 = seis[:, 1]
    mitad = p_seis // 2
    q = max(1, p_seis // 8)
    med = dict(ensayo2_primeros8=float(pre[:8, 1].mean()), ensayo2_1a100=float(e2[:mitad // 2].mean()),
               ensayo2_101a200=float(e2[mitad // 2:mitad].mean()),
               ensayo2_233a344=float(e2[mitad:mitad + (112 * p_seis) // P_SEIS].mean()),
               ensayo2_351a400=float(e2[-q:].mean()),
               colisiones=float(np.mean(np.r_[col_pre, col])),
               ensayo2_sin_colision_101a200=float(e2[mitad // 2:mitad][~col[mitad // 2:mitad]].mean())
               if (~col[mitad // 2:mitad]).any() else None,
               simbolos_distintos=float(len(set(simbolo.values()))))
    return dict(puntaje=med["ensayo2_101a200"], medidas=med, segundos=time.time() - t0)


def congelado(th, rng, sup, n: int) -> float:
    fz = copy.deepcopy(th)
    fz.reset(); fz.plastic = False
    return float(np.mean([problema(fz, rng, sup.tomar(2), T_SEIS, explore=False)[1] for _ in range(n)]))


def correr(semilla: int, escala: float = 1.0, config: dict | None = None) -> dict:
    t0 = time.time()
    p_pre = max(8, int(round(P_PRELIM * escala)))
    p_seis = max(20, int(round(P_SEIS * escala)))
    n_cong = max(10, int(round(N_CONGELADO * escala)))
    rng = np.random.default_rng(semilla)
    n_obj = 2 * (p_pre + p_seis + 2 * n_cong)
    th = thinker(n_obj, semilla, config)
    sup = Simbolos(n_obj, rng)
    erng = np.random.default_rng(10_000 + semilla)           # la prueba congelada no toca el flujo
    pre = np.array([problema(th, rng, sup.tomar(2), T_PRELIM) for _ in range(p_pre)], dtype=float)
    seis = np.zeros((p_seis, T_SEIS))
    cong = {}
    mitad = p_seis // 2
    for i in range(p_seis):
        seis[i] = problema(th, rng, sup.tomar(2), T_SEIS)
        if i + 1 in (mitad, p_seis):
            cong[i + 1] = congelado(th, erng, sup, n_cong)
    e2 = seis[:, 1]
    q = max(1, p_seis // 8)                                   # 50 problemas con la escala completa
    med = dict(ensayo2_primeros8=float(pre[:8, 1].mean()),
               ensayo2_preliminares=float(pre[:, 1].mean()),
               ensayo2_1a100=float(e2[:mitad // 2].mean()),
               ensayo2_101a200=float(e2[mitad // 2:mitad].mean()),
               ensayo2_233a344=float(e2[mitad:mitad + (112 * p_seis) // P_SEIS].mean()),
               ensayo2_351a400=float(e2[-q:].mean()),
               ensayo1_101a200=float(seis[mitad // 2:mitad, 0].mean()),
               congelado_tras200=cong[mitad], congelado_tras400=cong[p_seis],
               curva_ensayo2_por25=[float(x) for x in e2[:len(e2) // 25 * 25].reshape(-1, 25).mean(1)]
               if len(e2) >= 25 else [])
    return dict(puntaje=med["ensayo2_101a200"], medidas=med, segundos=time.time() - t0,
                problemas=dict(preliminares=p_pre, seis=p_seis))


_COMPARAR = {"harlow1949_primeros8": "ensayo2_primeros8", "harlow1950_1a100": "ensayo2_1a100",
             "harlow1950_101a200": "ensayo2_101a200", "hodos1970_nino": "ensayo2_101a200",
             "harlow1949_ultimos112": "ensayo2_233a344",
             "warren1965_rhesus": "ensayo2_351a400", "warren1965_rata_ardilla": "ensayo2_351a400",
             "hodos1970_tupaya": "ensayo2_351a400", "thomas2012_chimpance": "ensayo2_101a200",
             "thomas2012_rata": "ensayo2_351a400"}
_EXPERIENCIA = {"ensayo2_101a200": 232, "ensayo2_351a400": 432, "ensayo2_1a100": 132, "ensayo2_primeros8": 8,
                "ensayo2_233a344": 344}
# Curva de Harlow con el mismo número de problemas: (tramo, medida de la red, referencia de los monos)
CURVA = [("problemas 1-8 (de 50 ensayos)", "ensayo2_primeros8", "harlow1949_primeros8"),
         ("problemas 33-132 (1-100 de 6 ensayos)", "ensayo2_1a100", "harlow1950_1a100"),
         ("problemas 133-232 (101-200 de 6 ensayos)", "ensayo2_101a200", "harlow1950_101a200"),
         ("problemas 233-344 (los últimos 112 de Harlow)", "ensayo2_233a344", "harlow1949_ultimos112"),
         ("problemas 383-432 (351-400 de 6 ensayos)", "ensayo2_351a400", "warren1965_rhesus")]

PRUEBAS = [
    Prueba(
        id="harlow", variante="simbolos", condicion="simbolos",
        metrica="acierto en el ensayo 2, problemas de 6 ensayos 101 a 200 (en línea, tras 32 preliminares de 50 ensayos)",
        protocolo=("32 problemas de 50 ensayos y 400 de 6 ensayos, dos símbolos nunca vistos por problema, uno premiado "
                   "al azar, orden al azar en cada ensayo; la red aprende en línea con su exploración; prueba congelada "
                   "con 100 problemas nuevos tras 200 y tras 400 problemas de 6 ensayos"),
        desviaciones=[
            "condición de entrada: símbolos con identidad perfecta, no objetos reales vistos en el aparato de Wisconsin; "
            "la red no tiene que reconocer el objeto ni separarlo de la posición",
            "el resultado llega como recompensa en el estado (`prev_io`), no como percepción del premio bajo el objeto",
            "sin los 112 problemas de discriminación con reversión de Harlow (su procedimiento no se pudo verificar): los "
            "problemas 233 a 344 de la red son de 6 ensayos, comparados con el ensayo 2 de los últimos 112 de Harlow",
            "los problemas 201 a 400 son una extensión de la batería para comparar con Warren (1965)",
            "los monos tuvieron entrenamiento previo con el aparato; la red empieza sin nada",
        ],
        correr=correr, version=2, semillas=(0, 1, 2, 3, 4), perfiles=True, config_base=CONFIG,
        equivalente=False,
        motivo_equivalencia="con símbolos la red no tiene que percibir ni reconocer los objetos",
        comparar=_COMPARAR, experiencia=_EXPERIENCIA, curva=CURVA,
    ),
    Prueba(
        id="harlow", variante="imagenes", condicion="imagenes",
        metrica="acierto en el ensayo 2, problemas de 6 ensayos 101 a 200 (en línea), con objetos vistos",
        protocolo=("el mismo calendario que con símbolos; cada objeto es un carácter de Omniglot nunca visto (una "
                   "imagen); la red depth2 entrenada en MNIST lo nombra con uno de sus 10 dígitos y el Thinker (misma "
                   "configuración) recibe ese nombre; si los dos objetos reciben el mismo nombre la elección se "
                   "resuelve al azar"),
        desviaciones=[
            "la percepción tiene un vocabulario de 10 símbolos (los dígitos con que se entrenó): objetos distintos "
            "pueden recibir el mismo nombre y entonces no se distinguen",
            "el mismo objeto se ve siempre con la misma imagen, como en el aparato de Wisconsin",
            "el resultado llega como recompensa en el estado (`prev_io`)",
            "sin los 112 problemas de reversión; problemas 201 a 400 como extensión",
        ],
        correr=correr_imagenes, version=2, semillas=(0, 1, 2), perfiles=True, config_base=CONFIG,
        equivalente=True,
        motivo_equivalencia="la red ve los objetos como imágenes; su percepción los nombra con un vocabulario chico",
        comparar=_COMPARAR, experiencia=_EXPERIENCIA, curva=CURVA,
    ),
]
