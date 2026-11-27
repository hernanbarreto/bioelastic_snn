"""Flexibilidad: reversión serial y reversión probabilística con volatilidad.

Reversión serial. Una red sin experiencia aprende a elegir entre dos símbolos fijos; al
llegar al criterio, se invierte cuál se premia, y así 10 veces con los mismos dos símbolos,
en una sola sesión continua (no hay `reset` entre fases: nadie le avisa a la red que hubo
una reversión). Medidas: errores hasta el criterio en cada fase y acierto en el ensayo 2
después de cada reversión. Firma de las especies que «aprenden a revertir» (córvidos, Bond
et al. 2007): menos errores en las últimas reversiones que en la primera. Las abejas con el
mismo par de olores empeoran de fase en fase (Mota y Giurfa 2010). No hay cifras
publicadas legibles: no hay índice.

Reversión volátil (Behrens et al. 2007). Dos opciones (azul y verde); en cada ensayo sólo
una es correcta. Fase estable: 120 ensayos con azul correcto el 75 %. Fase volátil: 170
ensayos en los que la opción más probable (80 %) cambia cada 30 o 40 ensayos. La mitad de
las semillas hace las fases en el orden inverso. A las elecciones de la red se les ajusta,
como a las personas, una regla delta con una tasa de aprendizaje por fase y una elección
sigmoide (máxima verosimilitud en una grilla). Firma humana: tasa mayor en la fase volátil
(t pareada (17) = 2,91; las medias están sólo en una figura: no hay índice).
"""
from __future__ import annotations

import time

import numpy as np

from .base import Prueba
from .harlow import CONFIGS_ELECCION, elegir, thinker

# ---------------------------------------------------------------- reversión serial
N_REV = 10
CRITERIO = (9, 10)            # 9 aciertos en los últimos 10 ensayos
MAX_FASE = 200


def azar_errores(max_fase: int = MAX_FASE, criterio=CRITERIO) -> float:
    """Errores esperados por fase de quien elige al azar (p = 1/2), con el mismo criterio y
    el mismo tope: cálculo exacto sobre los 2^10 historiales de los últimos 10 ensayos."""
    k, w = criterio
    m = (1 << w) - 1
    pc = np.array([bin(i).count("1") for i in range(1 << w)])
    q = np.zeros(1 << w); q[0] = 1.0
    err = 0.0
    idx = np.arange(1 << w)
    for t in range(1, max_fase + 1):
        err += 0.5 * q.sum()
        nq = np.zeros_like(q)
        np.add.at(nq, ((idx << 1) | 1) & m, 0.5 * q)
        np.add.at(nq, (idx << 1) & m, 0.5 * q)
        if t >= w:
            nq[pc >= k] = 0.0
        q = nq
    return float(err)


def correr_serial(semilla: int, escala: float = 1.0, config: dict | None = None) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    n_rev = max(3, int(round(N_REV * min(1.0, escala * 2))))
    th = thinker(2, semilla, config)                 # sólo existen los dos estímulos
    A, B = 0, 1
    bueno = A if rng.random() < 0.5 else B
    invalidas = 0
    th.reset()
    fases = []
    for fase in range(n_rev + 1):
        hist = []
        for _ in range(MAX_FASE):
            a, b = (A, B) if rng.random() < 0.5 else (B, A)
            out = elegir(th, a, b, True)
            invalidas += out not in (A, B)            # -1: no respondió (cuenta como error)
            hist.append(bool(out == bueno))
            th.feedback(target=bueno)
            if len(hist) >= CRITERIO[1] and sum(hist[-CRITERIO[1]:]) >= CRITERIO[0]:
                break
        llego = len(hist) >= CRITERIO[1] and sum(hist[-CRITERIO[1]:]) >= CRITERIO[0]
        fases.append(dict(ensayos=len(hist), errores=int(len(hist) - sum(hist)), llego=bool(llego),
                          ensayo2=bool(hist[1]) if len(hist) > 1 else None))
        bueno = B if bueno == A else A
    err = np.array([f["errores"] for f in fases[1:]], dtype=float)   # sólo reversiones
    k = min(3, len(err))
    x = np.arange(1, len(err) + 1, dtype=float)
    pend = float(np.polyfit(x, err, 1)[0]) if len(err) >= 2 else None
    med = dict(errores_discriminacion=float(fases[0]["errores"]),
               errores_reversion1=float(err[0]), errores_ultimas3=float(err[-k:].mean()),
               mejora_errores=float(err[0] - err[-k:].mean()),
               pendiente_errores=pend,
               ensayo2_tras_reversion=float(np.mean([f["ensayo2"] for f in fases[1:] if f["ensayo2"] is not None])),
               fases_al_criterio=float(np.mean([f["llego"] for f in fases])),
               mejora_con_reversiones=bool(pend is not None and pend < 0 and err[-k:].mean() < err[0]),
               respuestas_invalidas=float(invalidas / sum(f["ensayos"] for f in fases)),
               errores_por_fase=[f["errores"] for f in fases])
    return dict(puntaje=med["errores_ultimas3"], medidas=med, segundos=time.time() - t0)


# ---------------------------------------------------------------- reversión volátil
N_ESTABLE, P_ESTABLE = 120, 0.75
N_VOLATIL, P_VOLATIL = 170, 0.80


def calendario(rng, orden: int) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidad de que «azul» sea la correcta en cada ensayo y la fase (0 estable, 1
    volátil). `orden` 0: estable y después volátil; 1: al revés."""
    est = np.full(N_ESTABLE, P_ESTABLE)
    vol = []
    azul = rng.random() < 0.5
    while len(vol) < N_VOLATIL:
        largo = int(rng.choice([30, 40]))
        vol += [P_VOLATIL if azul else 1 - P_VOLATIL] * largo
        azul = not azul
    vol = np.array(vol[:N_VOLATIL])
    if orden == 0:
        return np.concatenate([est, vol]), np.r_[np.zeros(N_ESTABLE), np.ones(N_VOLATIL)].astype(int)
    return np.concatenate([vol, est]), np.r_[np.ones(N_VOLATIL), np.zeros(N_ESTABLE)].astype(int)


ALFAS = np.round(np.linspace(0.02, 1.0, 50), 4)
BETAS = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48])


def ajustar_delta(eleccion_azul: np.ndarray, azul_correcto: np.ndarray, fase: np.ndarray,
                  valida: np.ndarray | None = None) -> dict:
    """Regla delta con una tasa por fase y elección sigmoide, por máxima verosimilitud en
    una grilla: v <- v + alfa[fase] (o - v); P(azul) = 1 / (1 + exp(-beta (v - 0,5))). Los
    ensayos sin respuesta válida no entran en la verosimilitud (el resultado sí actualiza v)."""
    T = len(eleccion_azul)
    best = (np.inf, None)
    o = azul_correcto.astype(float); c = eleccion_azul.astype(float)
    w = np.ones(T) if valida is None else valida.astype(float)
    for a0 in ALFAS:
        for a1 in ALFAS:
            al = np.where(fase == 0, a0, a1)
            v = np.empty(T); x = 0.5
            for t in range(T):
                v[t] = x
                x = x + al[t] * (o[t] - x)
            for b in BETAS:
                p = 1.0 / (1.0 + np.exp(-b * (v - 0.5)))
                p = np.clip(p, 1e-6, 1 - 1e-6)
                nll = -float(np.sum(w * (c * np.log(p) + (1 - c) * np.log(1 - p))))
                if nll < best[0]:
                    best = (nll, (float(a0), float(a1), float(b)))
    a0, a1, b = best[1]
    return dict(alfa_estable=a0, alfa_volatil=a1, beta=b, nll=best[0])


def correr_volatil(semilla: int, escala: float = 1.0, config: dict | None = None) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    orden = semilla % 2
    prob, fase = calendario(rng, orden)
    if escala < 1.0:                                  # sólo para tests rápidos
        keep = np.r_[np.flatnonzero(fase == 0)[:40], np.flatnonzero(fase == 1)[:60]]
        prob, fase = prob[np.sort(keep)], fase[np.sort(keep)]
    th = thinker(2, semilla, config)
    AZUL, VERDE = 0, 1
    th.reset()
    elec, corr, ok, valida = [], [], [], []
    for t in range(len(prob)):
        y = AZUL if rng.random() < prob[t] else VERDE
        a, b = (AZUL, VERDE) if rng.random() < 0.5 else (VERDE, AZUL)
        out = elegir(th, a, b, True)
        elec.append(out == AZUL); corr.append(y == AZUL); valida.append(out in (AZUL, VERDE))
        ok.append(out == (AZUL if prob[t] > 0.5 else VERDE))     # eligió la más probable
        th.feedback(target=y)
    elec = np.array(elec); corr = np.array(corr); ok = np.array(ok, dtype=float)
    valida = np.array(valida)
    fit = ajustar_delta(elec, corr, fase, valida)
    med = dict(fit, orden=float(orden), recompensa=float(np.mean(np.array(elec) == np.array(corr))),
               diferencia_alfa=fit["alfa_volatil"] - fit["alfa_estable"],
               firma_volatil_mayor=bool(fit["alfa_volatil"] > fit["alfa_estable"]),
               elige_mas_probable_estable=float(ok[fase == 0].mean()),
               elige_mas_probable_volatil=float(ok[fase == 1].mean()),
               respuestas_invalidas=float(1.0 - valida.mean()))
    return dict(puntaje=med["diferencia_alfa"], medidas=med, segundos=time.time() - t0)


def validar_serial(config: dict, semilla: int, escala: float = 1.0) -> float:
    """Criterio de validación: errores por fase (discriminación y reversiones), menos es mejor."""
    return float(np.mean(correr_serial(semilla, escala, config)["medidas"]["errores_por_fase"]))


def validar_volatil(config: dict, semilla: int, escala: float = 1.0) -> float:
    """Criterio de validación: proporción de ensayos con la opción correcta (lo que premia la tarea)."""
    return correr_volatil(semilla, escala, config)["medidas"]["recompensa"]


PRUEBAS = [
    Prueba(
        id="reversion_serial", variante="simbolos", condicion="simbolos",
        metrica="errores hasta el criterio (9 de 10) en las últimas 3 reversiones; menos es mejor",
        protocolo=(f"dos símbolos fijos, discriminación hasta 9 aciertos en 10 ensayos (máximo {MAX_FASE} por fase) y "
                   f"{N_REV} reversiones seguidas en una sesión continua, sin aviso; red sin experiencia previa"),
        desviaciones=[
            "condición de entrada: símbolos, no estímulos visuales u olfativos",
            "criterio (9 de 10) y número de reversiones (10) elegidos por la batería: los de los artículos no se pudieron leer",
            "sin sesiones diarias: una sesión continua",
        ],
        correr=correr_serial, version=1, semillas=(0, 1, 2, 3, 4),
        configs=CONFIGS_ELECCION, validar=validar_serial, validar_mayor=False,
        criterio_validacion="errores por fase en semillas de validación (menos es mejor)",
        equivalente=False, motivo_equivalencia="símbolos en lugar de estímulos percibidos",
        referencia=None, mayor_es_mejor=False,
        azar=round(azar_errores(), 2),
        azar_derivacion=(f"errores esperados por fase de quien elige al azar con el mismo criterio y tope de {MAX_FASE} "
                         "ensayos (cálculo exacto); en el ensayo 2 tras una reversión el azar es 0,5"),
        firmas={"mejora_con_reversiones": "menos errores en las últimas reversiones que en la primera (córvidos; "
                                          "las abejas con el mismo par empeoran)"},
    ),
    Prueba(
        id="reversion_volatil", variante="simbolos", condicion="simbolos",
        metrica="tasa de aprendizaje ajustada en la fase volátil menos la de la fase estable",
        protocolo=("calendario de Behrens et al. (2007): 120 ensayos estables (75 %) y 170 volátiles (80 %, cambia cada "
                   "30 o 40), orden contrabalanceado por semilla; ajuste de regla delta + sigmoide por máxima verosimilitud"),
        desviaciones=[
            "sin magnitudes de premio: en el original cada opción tenía un premio al azar (0 a 100) que desacopla la "
            "elección de la probabilidad; acá sólo cuenta la probabilidad",
            "condición de entrada: dos símbolos, no colores en pantalla",
            "ajuste por máxima verosimilitud en una grilla, no la estimación bayesiana del original",
        ],
        correr=correr_volatil, version=1, semillas=(0, 1, 2, 3, 4, 5),
        configs=CONFIGS_ELECCION, validar=validar_volatil, semillas_validacion=(100, 101, 102, 103),
        criterio_validacion="proporción de elecciones correctas en semillas de validación",
        equivalente=False, motivo_equivalencia="sin magnitudes de premio y con símbolos",
        referencia=None,
        firmas={"firma_volatil_mayor": "tasa de aprendizaje mayor en la fase volátil (personas, Behrens et al. 2007)"},
    ),
]
