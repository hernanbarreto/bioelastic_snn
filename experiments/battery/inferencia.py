"""Inferencia transitiva con pares premiados (series de 5, 6 y 7 ítems).

Los ítems son símbolos en un orden oculto (A > B > C ...). En cada ensayo aparecen dos, en
orden al azar, y se premia el mayor; con dos opciones, el objetivo equivale a la
recompensa. Cada sesión es un episodio (`reset` al empezar), como las sesiones diarias de
los animales.

  serie 5  von Fersen et al. (1991), palomas: A+B-, B+C-, C+D-, D+E- hasta 80 % en cada par
           de una sesión; después 6 sesiones de prueba, cada una con dos presentaciones sin
           refuerzo del par BD (12 en total).
  serie 6  Siemann y Delius (1996), personas: A+B- ... E+F- hasta menos de 30 % de errores en
           cada par de una sesión; después BD, BE y CE sin refuerzo.
  serie 7  Jensen et al. (2017), personas: una sesión con una lista nueva; fase 1 sólo FG (20
           ensayos), fase 2 los adyacentes y los pares con F o G (150), fase 3 los 21 pares
           (210), con premio en cada respuesta correcta. Medida: exactitud por distancia
           simbólica en la fase 3 y efecto de distancia (pendiente).

Una prueba sin refuerzo se hace sobre una copia congelada de la red (`plastic = False`, sin
exploración): no cambia nada de la red que sigue entrenando, tampoco su estado. Una
respuesta que no es ninguno de los dos ítems cuenta como error y se informa aparte.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from .base import Prueba
from .harlow import CONFIGS_ELECCION, elegir, thinker

SESION = 10                # presentaciones de cada par por sesión de entrenamiento
MAX_SESIONES = 125         # las palomas de von Fersen llegaron al criterio en 125 sesiones
N_PRUEBA_6 = 10            # pruebas sin refuerzo por par en la serie de 6


def ensayo(th, x: int, y: int, mayor: int, rng, aprender: bool = True) -> tuple[bool, bool]:
    a, b = (x, y) if rng.random() < 0.5 else (y, x)
    out = elegir(th, a, b, aprender)
    if aprender:
        th.feedback(target=mayor)
    return bool(out == mayor), bool(out in (x, y))


def sonda(th, x: int, y: int, mayor: int, rng) -> tuple[bool, bool]:
    fz = copy.deepcopy(th)
    fz.plastic = False
    return ensayo(fz, x, y, mayor, rng, aprender=False)


def sesion(th, pares, rng, sondas=()) -> dict:
    """Una sesión de entrenamiento: cada par `SESION` veces en orden al azar; `sondas`
    (pares sin refuerzo) se intercalan en posiciones al azar. Devuelve aciertos por par."""
    orden = [p for p in pares for _ in range(SESION)]
    orden = [orden[i] for i in rng.permutation(len(orden))]
    pos = set(int(i) for i in rng.choice(len(orden), size=len(sondas), replace=False)) if sondas else set()
    th.reset()
    hits = {p: [] for p in pares}; prob = []
    sond = list(sondas)
    for i, (x, y) in enumerate(orden):
        if i in pos:
            u, v = sond.pop()
            prob.append(((u, v),) + sonda(th, u, v, u, rng))
        hits[(x, y)].append(ensayo(th, x, y, x, rng)[0])
    return dict(aciertos={p: float(np.mean(h)) for p, h in hits.items()}, sondas=prob)


def entrenar(th, pares, rng, criterio: float) -> tuple[int, bool, list]:
    """Sesiones hasta el criterio en cada par. Devuelve (sesiones, ¿llegó?, exactitud media
    en los pares de premisa por sesión)."""
    hist = []
    for s in range(1, MAX_SESIONES + 1):
        r = sesion(th, pares, rng)
        hist.append(float(np.mean(list(r["aciertos"].values()))))
        if all(v >= criterio for v in r["aciertos"].values()):
            return s, True, hist
    return MAX_SESIONES, False, hist


def items(rng, n):
    return [int(v) for v in rng.permutation(n)]          # items[0] es el mayor


def correr_5(semilla: int, escala: float = 1.0, config: dict | None = None, solo_entrenar: bool = False) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    A, B, C, D, E = items(rng, 5)
    th = thinker(5, semilla, config)
    pares = [(A, B), (B, C), (C, D), (D, E)]
    ses, llego, hist = entrenar(th, pares, rng, 0.8)
    if solo_entrenar:
        return dict(premisas=float(np.mean(hist[-10:])))
    bd, val = [], []
    ae = []
    for _ in range(6):
        r = sesion(th, pares, rng, sondas=[(B, D), (B, D)])
        bd += [h for _, h, _ in r["sondas"]]; val += [v for _, _, v in r["sondas"]]
    for _ in range(12):                                   # AE: no está en el experimento 1 (se informa aparte)
        ae.append(sonda(th, A, E, A, rng)[0])
    bdv = [h for h, v in zip(bd, val) if v]
    med = dict(eleccion_B_en_BD=float(np.mean(bd)), validas_BD=float(np.mean(val)),
               eleccion_B_entre_validas=float(np.mean(bdv)) if bdv else None,
               eleccion_A_en_AE=float(np.mean(ae)), sesiones_al_criterio=float(ses), llego_al_criterio=bool(llego),
               premisas_ultimas10=float(np.mean(hist[-10:])))
    return dict(puntaje=med["eleccion_B_en_BD"], medidas=med, segundos=time.time() - t0)


def correr_6(semilla: int, escala: float = 1.0, config: dict | None = None, solo_entrenar: bool = False) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    A, B, C, D, E, F = items(rng, 6)
    th = thinker(6, semilla, config)
    pares = [(A, B), (B, C), (C, D), (D, E), (E, F)]
    ses, llego, hist = entrenar(th, pares, rng, 0.7)
    if solo_entrenar:
        return dict(premisas=float(np.mean(hist[-10:])))
    th.reset()
    res = {}
    validos = []
    for nombre, (x, y) in (("BD", (B, D)), ("BE", (B, E)), ("CE", (C, E))):
        s = [sonda(th, x, y, x, rng) for _ in range(N_PRUEBA_6)]
        res[nombre] = (float(np.mean([h for h, _ in s])), float(np.mean([v for _, v in s])))
        validos += [h for h, v in s if v]
    med = dict(BD=res["BD"][0], BE=res["BE"][0], CE=res["CE"][0],
               media_BD_BE_CE=float(np.mean([res[k][0] for k in ("BD", "BE", "CE")])),
               validas=float(np.mean([res[k][1] for k in ("BD", "BE", "CE")])),
               correctas_entre_validas=float(np.mean(validos)) if validos else None,
               sesiones_al_criterio=float(ses), llego_al_criterio=bool(llego),
               premisas_ultimas10=float(np.mean(hist[-10:])))
    return dict(puntaje=med["media_BD_BE_CE"], medidas=med, segundos=time.time() - t0)


def correr_7(semilla: int, escala: float = 1.0, config: dict | None = None, solo_entrenar: bool = False) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    L = items(rng, 7)
    th = thinker(7, semilla, config)
    pos = {s: i for i, s in enumerate(L)}
    todos = [(L[i], L[j]) for i in range(7) for j in range(i + 1, 7)]
    F, G = L[5], L[6]
    adyac = [(L[i], L[i + 1]) for i in range(6)]
    fase2 = sorted(set(adyac) | {p for p in todos if F in p or G in p}, key=lambda p: (pos[p[0]], pos[p[1]]))
    th.reset()

    def bloque(pares, n):
        out = []
        for k in range(n):
            x, y = pares[int(rng.integers(len(pares)))]
            out.append(((x, y),) + ensayo(th, x, y, x, rng))
        return out
    bloque([(F, G)], 20)
    f2 = bloque(fase2, 150)
    if solo_entrenar:
        return dict(premisas=float(np.mean([h for _, h, _ in f2[-50:]])))
    vistos = set(fase2) | {(F, G)}
    primera = {}
    f3 = []
    for k in range(210):
        x, y = todos[int(rng.integers(len(todos)))]
        r = ensayo(th, x, y, x, rng)
        if (x, y) not in vistos and (x, y) not in primera:
            primera[(x, y)] = r
        f3.append(((x, y),) + r)
    dist = {d: [h for (p, h, _) in f3 if pos[p[1]] - pos[p[0]] == d] for d in range(1, 7)}
    acc = {d: float(np.mean(v)) if v else None for d, v in dist.items()}
    ds = [d for d in acc if acc[d] is not None]
    pend = float(np.polyfit(ds, [acc[d] for d in ds], 1)[0]) if len(ds) >= 2 else None
    lejos = [acc[d] for d in range(2, 7) if acc[d] is not None]
    pv = list(primera.values())
    med = dict(exactitud_fase3=float(np.mean([h for _, h, _ in f3])),
               primera_vez_no_vistos=float(np.mean([h for h, _ in pv])) if pv else None,
               primera_vez_validas=float(np.mean([v for _, v in pv])) if pv else None,
               pendiente_distancia=pend, efecto_distancia=bool(pend is not None and pend > 0),
               adyacentes_por_debajo=bool(acc[1] is not None and lejos and acc[1] < float(np.mean(lejos))),
               validas_fase3=float(np.mean([v for _, _, v in f3])),
               exactitud_por_distancia=[acc[d] for d in range(1, 7)])
    for d in range(1, 7):
        med[f"distancia_{d}"] = acc[d]
    return dict(puntaje=med["exactitud_fase3"], medidas=med, segundos=time.time() - t0)


_DESV = [
    "una respuesta que no es ninguno de los dos ítems cuenta como error (el animal no tiene esa opción); se informa "
    "también la proporción entre las respuestas válidas",
    "condición de entrada: símbolos con identidad perfecta, no figuras vistas (palomas: estímulos visuales en teclas; "
    "personas: polígonos o figuras en un juego de computadora)",
    "con dos opciones el objetivo equivale a la recompensa; la red recibe el objetivo",
    "la prueba sin refuerzo se hace sobre una copia congelada de la red, sin exploración",
]
def _validador(fn):
    def validar(config: dict, semilla: int, escala: float = 1.0) -> float:
        """Criterio de validación: exactitud en los pares entrenados (premisas) al final del
        entrenamiento, en semillas de validación; nunca los pares de prueba."""
        return fn(semilla, escala, config, solo_entrenar=True)["premisas"]
    return validar


PRUEBAS = [
    Prueba(
        id="inferencia_transitiva_5", variante="simbolos", condicion="simbolos",
        metrica="proporción de elecciones de B en el par BD (12 pruebas sin refuerzo por semilla)",
        protocolo=f"sesiones de {SESION} presentaciones de cada par adyacente hasta 80 % en cada par (máximo "
                  f"{MAX_SESIONES}); después 6 sesiones con 2 pruebas BD intercaladas",
        desviaciones=_DESV + [f"largo de la sesión ({SESION} por par) elegido por la batería: el del artículo no se leyó"],
        correr=correr_5, version=2, semillas=tuple(range(8)),
        configs=CONFIGS_ELECCION, validar=_validador(correr_5), perfiles=True,
        criterio_validacion="exactitud en los pares entrenados al final del entrenamiento (semillas de validación)",
        equivalente=False, motivo_equivalencia="símbolos en lugar de estímulos visuales",
        comparar={"vonfersen1991_palomas": "puntaje", "benard2004_abejas": "puntaje"},
    ),
    Prueba(
        id="inferencia_transitiva_6", variante="simbolos", condicion="simbolos",
        metrica="proporción de elecciones correctas en BD, BE y CE (media de los tres; 10 pruebas por par)",
        protocolo=f"sesiones de {SESION} presentaciones de cada par adyacente hasta menos de 30 % de errores en cada par "
                  f"(máximo {MAX_SESIONES}); después {N_PRUEBA_6} pruebas sin refuerzo de BD, BE y CE",
        desviaciones=_DESV + ["las personas traen conocimiento previo del orden y de la consigna del juego; la red empieza sin nada",
                              f"{N_PRUEBA_6} pruebas por par y largo de sesión elegidos por la batería"],
        correr=correr_6, version=2, semillas=tuple(range(8)),
        configs=CONFIGS_ELECCION, validar=_validador(correr_6), perfiles=True,
        criterio_validacion="exactitud en los pares entrenados al final del entrenamiento (semillas de validación)",
        equivalente=False, motivo_equivalencia="símbolos en lugar de figuras; las personas traen la idea de orden",
        comparar={"siemann1996_humanos_BD": "BD", "siemann1996_humanos_BE": "BE", "siemann1996_humanos_CE": "CE"},
    ),
    Prueba(
        id="inferencia_transitiva_7", variante="simbolos", condicion="simbolos",
        metrica="exactitud en la fase 3 (los 21 pares, con premio); el efecto de distancia es la firma",
        protocolo="una sesión con la lista de 7: FG 20 ensayos, adyacentes y pares con F o G 150, los 21 pares 210",
        desviaciones=_DESV[:2] + ["una sesión, como las personas; los monos tuvieron decenas de sesiones con listas nuevas"],
        correr=correr_7, version=2, semillas=tuple(range(8)),
        configs=CONFIGS_ELECCION, validar=_validador(correr_7), perfiles=True,
        criterio_validacion="exactitud en los pares entrenados al final del entrenamiento (semillas de validación)",
        equivalente=False, motivo_equivalencia="símbolos en lugar de fotografías",
        referencia=None,
        firmas={"efecto_distancia": "pendiente positiva de la exactitud contra la distancia simbólica (monos rhesus sí; "
                                    "personas: no se ve en la exactitud, Jensen et al. 2017)",
                "adyacentes_por_debajo": "los pares adyacentes con menos exactitud que los no adyacentes, como describe "
                                         "Jensen et al. (2017) el efecto de distancia (firma agregada después de ver que la "
                                         "pendiente sola no distingue un techo)"},
    ),
]
