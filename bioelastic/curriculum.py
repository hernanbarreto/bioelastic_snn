"""Currículo: qué se presenta y en qué orden, nunca qué se enseña.

Un currículo elige sólo los datos de cada episodio: las etiquetas son siempre las
verdaderas, la arquitectura y las reglas de aprendizaje no cambian, y el presupuesto se
cuenta en muestras (aquí, columnas de la suma: dos imágenes cada una), así que un
currículo y el orden aleatorio ven la misma cantidad de datos.

Motivación (ver `exp_integration3.analisis`): en la suma módulo 10 con sumandos uniformes,
el dígito de la columna es independiente de cada sumando por separado (I(a; d) = 0); la
señal por ranura aparece sólo cuando la otra ranura ya se percibe bien. Restringir los
datos al principio la hace aparecer sin tocar las etiquetas:

  "mas0"      primero n + 0 (la respuesta es el propio dígito: identidad), después n + 1,
              n + {0,1,2} y todo (los chicos aprenden n+0 y n+1 antes que el resto).
  "conteo"    primero los números chicos y luego más grandes (Carey, 2004; Le Corre y
              Carey, 2007: «uno», «dos», «tres» antes del principio cardinal); sin vuelta
              del módulo, la suma depende de cada sumando por separado.
  "corto"     «empezar de a poco» (Elman, 1993): números de una cifra antes que largos,
              con todos los dígitos. Rohde y Plaut (1999) mostraron que en redes
              recurrentes no ayuda y a veces perjudica: se reporta igual.
  "aleatorio" el control: 1 a 4 cifras, todos los dígitos, desde el principio.
"""
from __future__ import annotations

import numpy as np

ALL = tuple(range(10))


class Curriculum:
    """Etapas por fracción del presupuesto: `stages` = [(fracción, parámetros)], en orden.
    `params(usado)` devuelve los parámetros de la etapa en la que cae `usado` (en las
    mismas unidades que `budget`); pasado el presupuesto, los de la última."""

    def __init__(self, stages, budget: float):
        if not stages:
            raise ValueError("un currículo necesita al menos una etapa")
        fr = np.array([float(f) for f, _ in stages])
        self.ends = np.cumsum(fr / fr.sum()) * float(budget)
        self.stages = [dict(p) for _, p in stages]
        self.budget = float(budget)

    def stage(self, used: float) -> int:
        return int(min(np.searchsorted(self.ends, float(used), side="right"), len(self.stages) - 1))

    def params(self, used: float) -> dict:
        return self.stages[self.stage(used)]


def sum_stages(kind: str) -> list:
    """Etapas de los currículos de suma (ver docstring del módulo). Cada etapa: dígitos
    posibles del primer sumando (`a`), del segundo (`b`) y rango de cifras `n` (inclusive)."""
    if kind == "aleatorio":
        return [(1.0, dict(a=ALL, b=ALL, n=(1, 4)))]
    if kind == "mas0":
        return [(0.15, dict(a=ALL, b=(0,), n=(1, 1))),
                (0.10, dict(a=ALL, b=(0, 1), n=(1, 1))),
                (0.10, dict(a=ALL, b=(0, 1, 2), n=(1, 2))),
                (0.65, dict(a=ALL, b=ALL, n=(1, 4)))]
    if kind == "conteo":
        return [(0.10, dict(a=(0, 1, 2), b=(0, 1, 2), n=(1, 1))),
                (0.10, dict(a=tuple(range(5)), b=tuple(range(5)), n=(1, 1))),
                (0.15, dict(a=tuple(range(7)), b=tuple(range(7)), n=(1, 2))),
                (0.65, dict(a=ALL, b=ALL, n=(1, 4)))]
    if kind == "corto":
        return [(0.25, dict(a=ALL, b=ALL, n=(1, 1))),
                (0.25, dict(a=ALL, b=ALL, n=(1, 2))),
                (0.50, dict(a=ALL, b=ALL, n=(1, 4)))]
    raise ValueError(f"currículo desconocido: {kind}")


def sample_sum(params: dict, rng) -> tuple:
    """Un episodio de suma con los parámetros de una etapa: (dígitos de a, dígitos de b),
    de la cifra menos significativa a la más."""
    lo, hi = params["n"]
    n = int(rng.integers(lo, hi + 1))
    a = np.array(params["a"])[rng.integers(len(params["a"]), size=n)]
    b = np.array(params["b"])[rng.integers(len(params["b"]), size=n)]
    return a.astype(np.int64), b.astype(np.int64)


def per_slot_information(a_digits, b_digits) -> float:
    """I(a; (a + b) mod 10) en bits con a y b uniformes e independientes sobre los
    dígitos dados (sin acarreo de entrada): cuánta señal da la respuesta sobre un sumando
    por separado. 0 con todos los dígitos; positiva si el rango es chico o b es fijo."""
    J = np.zeros((10, 10))
    for x in a_digits:
        for z in b_digits:
            J[int(x), (int(x) + int(z)) % 10] += 1
    J /= J.sum()
    px = J.sum(1, keepdims=True); py = J.sum(0, keepdims=True)
    nz = J > 0
    return float((J[nz] * np.log2(J[nz] / (px @ py)[nz])).sum())
