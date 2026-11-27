"""Mapa cognitivo con la estructura separada de lo sensorial.

En la Tolman-Eichenbaum Machine (Whittington et al., 2020) las células entorrinales llevan
una base de conocimiento *estructural* (dónde se está dentro de una estructura abstracta,
sin importar qué objetos hay) y las hipocampales la ligan con lo sensorial (qué objeto está
en cada lugar). En el `Thinker`, la memoria episódica (`FastMemory`) ya es la parte
hipocampal: cada evento liga un símbolo con el que lo siguió. Acá se agrega la parte
estructural, calculada sobre esas ligaduras sin mirar la identidad de los símbolos:

  * Las relaciones de a un símbolo (X -> Y) forman un grafo. Su estructura abstracta es
    cuántos fragmentos tiene (componentes conexas) y si alguna relación se ramifica
    (un símbolo con dos sucesores o dos predecesores distintos).
  * La señal global de completitud del mapa: «no hay mapa» (ninguna relación), «mapa a
    medio armar» (más de un fragmento, o ramificado) o «mapa completo» (un solo
    fragmento, sin ramas: una cadena o un ciclo). Es un escalar global, como una señal
    de confianza: el control lo ve en su estado (no ve los símbolos).

Por qué hace falta: con preguntas de aprendizaje mezcladas («¿qué sigue a X?» la primera vez
que aparece X, antes de haber visto su relación), el estado «X no tiene sucesor en la
memoria» se da tanto al principio (no hay forma de saber la respuesta) como al cerrar el
anillo o preguntar qué sigue al último (ahí se puede inferir: el principio de la cadena, o
FIN). Sin la señal, los fracasos inevitables del principio le enseñan al control que en
ese estado no conviene la estrategia de cierre. Con la señal, son dos estados distintos.

Conocimiento estructural aprendido entre episodios: el tamaño típico de un mapa completo.
Sin él, si la última relación todavía no se vio y su destino nunca apareció, el mapa parcial
es una cadena sin ramas, indistinguible de uno completo más corto (pasa en ~1 de cada 4
bloques con preguntas de aprendizaje, y ese 20 % de fracasos de la estrategia de cierre en el
estado «completo» bastaba para que el modelo interno no diera por fiable el juicio FIN). Al
terminar cada episodio con un mapa completo se actualiza el tamaño esperado (promedio
exponencial, sólo con plasticidad); un mapa sin ramas pero con menos nodos que el esperado
cuenta como a medio armar. Un mapa más grande que lo visto (L=8 tras entrenar con 5) sí
cuenta como completo.
"""
from __future__ import annotations

NONE, PARTIAL, COMPLETE = 0, 1, 2
N_CODES = 3


def map_code(events, min_nodes: int = 0) -> int:
    """Señal de completitud del mapa a partir de los eventos de la memoria episódica
    (clave: frozenset, símbolos: tupla, cuándo). Sólo cuentan las relaciones de a un
    símbolo. `min_nodes`: tamaño esperado de un mapa completo (0: sin expectativa)."""
    return map_stats(events, min_nodes)[0]


def map_stats(events, min_nodes: int = 0) -> tuple:
    """(señal de completitud, cantidad de nodos) del mapa."""
    succ: dict = {}
    pred: dict = {}
    parent: dict = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_edges = 0
    for k, toks, _ in events:
        if len(k) != 1 or len(toks) != 1:
            continue
        (a,) = k
        b = toks[0]
        if a == b:
            continue
        succ.setdefault(a, set()).add(b)
        pred.setdefault(b, set()).add(a)
        for x in (a, b):
            if x not in parent:
                parent[x] = x
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
        n_edges += 1
    if not n_edges:
        return NONE, 0
    n = len(parent)
    if any(len(v) > 1 for v in succ.values()) or any(len(v) > 1 for v in pred.values()):
        return PARTIAL, n
    roots = {find(x) for x in parent}
    return (COMPLETE if len(roots) == 1 and n >= min_nodes else PARTIAL), n


class SizePrior:
    """Tamaño esperado de un mapa completo, aprendido entre episodios."""

    def __init__(self, rate: float = 0.1):
        self.rate = float(rate)
        self.size = 0.0

    def expected(self) -> int:
        return int(round(self.size)) if self.size > 0 else 0

    def learn(self, events):
        """Al terminar un episodio: si el mapa quedó completo (sin exigir tamaño), su
        tamaño actualiza la expectativa."""
        code, n = map_stats(events)
        if code == COMPLETE:
            self.size = float(n) if self.size == 0 else self.size + self.rate * (n - self.size)
