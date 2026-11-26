"""Pruebas pensadas para romper al Thinker. Todo se evalúa en casos no vistos y por semilla.

Un solo «mundo» de N símbolos con un conocimiento permanente f (una permutación sin puntos
fijos), más dos símbolos de juicio (SÍ, NO). Cada tarea es una consigna distinta:

  C0 conocimiento     x -> f(x).
  C1 seguir+aplicar   el episodio liga k->...->v (1-2 saltos en entrenamiento); se pregunta
                      k y la respuesta es f(v). Memoria y conocimiento en la misma consulta.
                      Entrena con v en la primera mitad del vocabulario; evalúa con v en la
                      segunda (nunca compuestos) y con 3-4 saltos (nunca entrenados).
  C2 aplicar+recordar el episodio liga u->w; se pregunta x con f(x)=u; respuesta w.
                      Entrena con x en la primera mitad; evalúa en la segunda.
  C3 cadenas          hechos A->B, B->C ... desordenados, con distractores y símbolos sueltos
                      de ruido. Entrena 1-3 saltos; evalúa 5-8.
  C4 composición      f(f(x)); entrena x en la primera mitad, evalúa en la segunda.
  C5 transitiva       premisas X>Y adyacentes de un orden nuevo en cada episodio (pares
                      desordenados + distractores). Pregunta «¿X > Y?» (X en el foco, Y
                      como `probe`) -> SÍ/NO. Entrena órdenes de 3-4 elementos; evalúa
                      órdenes de 8 con distancias >= 4 (nunca vistas).
  C6 roles            «(evento, rol) -> relleno» para varios eventos; se pregunta (evento,
                      rol). Entrena 2-3 eventos; evalúa 5 (más interferencia).

Rupturas (cada una sobre la red entrenada en la condición normal, salvo que se indique):
  orden adverso       cadenas con los hechos presentados del último al primero, y con un
                      hecho corregido (A->B y después A->B': vale lo último).
  distractores Z->B   hechos que llevan *hacia* símbolos de la cadena desde fuera.
  etiquetas ruidosas  10 % de los objetivos de entrenamiento al azar (se entrena así).
  vocabulario grande  N = 256.
  multitarea          un solo Thinker con todas las consignas: mezcladas, en bloques
                      secuenciales (olvido) y en bloques en orden inverso (f al final).

  largo               cadenas de 12-16 saltos y órdenes de 16 (max_steps=20).
  ablaciones(_multi)  cada pieza nueva reemplazada por la de la versión anterior (o quitada).

  python experiments/exp_thinking_hard.py [episodios] [semillas] [grupo ...]
  grupos: solas ruido grande largo multi ablaciones ablaciones_multi
  (resultados en results/thinking_hard_<grupo>.csv)
"""
from __future__ import annotations

import sys
import time

import numpy as np

from common import CsvLog
from bioelastic.thinker import _NONE, FastMemory, Thinker


def derangement(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        f = rng.permutation(n)
        if not np.any(f == np.arange(n)):
            return f


class World:
    def __init__(self, n=32, seed=11):
        self.N = n
        self.YES, self.NO = n, n + 1
        self.S = n + 2                                     # símbolos de la red
        self.f = derangement(n, seed)
        self.half = n // 2


def _bind(net, pairs, rng, noise=0, order=None):
    """Presenta hechos (a, b) separados por silencios, más `noise` símbolos sueltos."""
    items = [tuple(p) for p in pairs] + [(int(s),) for s in rng.integers(net.N, size=noise)]
    idx = rng.permutation(len(items)) if order is None else order
    for i in idx:
        for t in items[i]:
            net.observe(int(t))
        net.observe(None)


def _pick(rng, pool, k, exclude=()):
    pool = np.setdiff1d(pool, np.asarray(exclude, dtype=np.int64))
    return rng.choice(pool, k, replace=False)


# ---------------------------------------------------------------- tareas
def c0(W, net, rng, learn, cue=0, x=None, noisy=0.0):
    x = int(rng.integers(W.N)) if x is None else x
    y = int(W.f[x])
    net.reset()
    out = net.answer(x, cue=cue, explore=learn)
    net.feedback(target=_noisy(W, rng, y, noisy))
    return out == y


def _noisy(W, rng, y, p):
    return int(rng.integers(W.N)) if p and rng.random() < p else y


def c1(W, net, rng, learn, cue=1, test=False, hops=None, noisy=0.0):
    hops = (int(rng.integers(1, 3)) if not test else int(rng.integers(3, 5))) if hops is None else hops
    lo, hi = (W.half, W.N) if test else (0, W.half)
    v = int(rng.integers(lo, hi))
    chain = list(_pick(rng, np.arange(W.N), hops, [v])) + [v]
    others = _pick(rng, np.arange(W.N), 6, chain)
    pairs = [(chain[i], chain[i + 1]) for i in range(hops)] + [(others[2 * i], others[2 * i + 1]) for i in range(3)]
    net.reset()
    _bind(net, pairs, rng)
    y = int(W.f[v])
    out = net.answer(int(chain[0]), cue=cue, explore=learn)
    net.feedback(target=_noisy(W, rng, y, noisy))
    return out == y


def c2(W, net, rng, learn, cue=2, test=False, noisy=0.0):
    lo, hi = (W.half, W.N) if test else (0, W.half)
    xs = _pick(rng, np.arange(lo, hi), 4)
    us = W.f[xs]
    ws = _pick(rng, np.arange(W.N), 4, np.concatenate([us, xs]))
    net.reset()
    _bind(net, list(zip(us, ws)), rng)
    i = int(rng.integers(4))
    y = int(ws[i])
    out = net.answer(int(xs[i]), cue=cue, explore=learn)
    net.feedback(target=_noisy(W, rng, y, noisy))
    return out == y


def c3(W, net, rng, learn, cue=3, hops=None, distractors=4, noise=0, adverse=None, noisy=0.0):
    hops = int(rng.integers(1, 4)) if hops is None else hops
    syms = rng.permutation(W.N)
    chain = [int(s) for s in syms[:hops + 1]]
    rest = syms[hops + 1:]
    pairs = [(chain[i], chain[i + 1]) for i in range(hops)]
    nd = min(distractors, (len(rest) - 2) // 2)
    pairs += [(int(rest[2 * i]), int(rest[2 * i + 1])) for i in range(nd)]
    order = None
    if adverse == "reverso":                       # el último hecho de la cadena primero
        order = list(range(hops))[::-1] + list(range(hops, len(pairs)))
    if adverse == "corregido":                     # A->B' primero, y después A->B (vale lo último)
        wrong = int(rest[-1])
        pairs = [(chain[0], wrong)] + pairs
        order = [0] + list(1 + rng.permutation(len(pairs) - 1))
    if adverse == "apuntan":                       # distractores Z->B que llegan a la cadena
        pairs += [(int(rest[-1 - i]), chain[1 + i]) for i in range(min(hops, 4))]
    net.reset()
    _bind(net, pairs, rng, noise=noise, order=order)
    out = net.answer(chain[0], cue=cue, explore=learn)
    net.feedback(target=_noisy(W, rng, chain[-1], noisy))
    return out == chain[-1]


def c4(W, net, rng, learn, cue=4, test=False, noisy=0.0):
    lo, hi = (W.half, W.N) if test else (0, W.half)
    x = int(rng.integers(lo, hi))
    y = int(W.f[W.f[x]])
    net.reset()
    out = net.answer(x, cue=cue, explore=learn)
    net.feedback(target=_noisy(W, rng, y, noisy))
    return out == y


def c5(W, net, rng, learn, cue=5, L=None, min_gap=1, distractors=3, noisy=0.0):
    L = int(rng.integers(3, 5)) if L is None else L
    syms = rng.permutation(W.N)
    order = [int(s) for s in syms[:L]]
    rest = syms[L:]
    pairs = [(order[i], order[i + 1]) for i in range(L - 1)]
    pairs += [(int(rest[2 * i]), int(rest[2 * i + 1])) for i in range(distractors)]
    while True:
        i, j = (int(t) for t in rng.choice(L, 2, replace=False))
        if abs(i - j) >= min_gap:
            break
    y = W.YES if i < j else W.NO
    net.reset()
    _bind(net, pairs, rng)
    out = net.answer(order[i], cue=cue, probe=order[j], explore=learn)
    t = y if not (noisy and rng.random() < noisy) else int(rng.choice([W.YES, W.NO]))
    net.feedback(target=t)
    return out == y


def c6(W, net, rng, learn, cue=6, events=None, roles=3, noisy=0.0):
    events = int(rng.integers(2, 4)) if events is None else events
    syms = rng.permutation(W.N)
    ev = syms[:events]; ro = syms[events:events + roles]
    fill = rng.choice(syms[events + roles:], (events, roles))     # rellenos (pueden repetirse)
    pairs = [(int(ev[e]), int(ro[r]), int(fill[e, r])) for e in range(events) for r in range(roles)]
    net.reset()
    for i in rng.permutation(len(pairs)):
        e, r, x = pairs[i]
        net.observe((e, r)); net.observe(x); net.observe(None)
    e, r = int(rng.integers(events)), int(rng.integers(roles))
    y = int(fill[e, r])
    out = net.answer((int(ev[e]), int(ro[r])), cue=cue, explore=learn)
    net.feedback(target=_noisy(W, rng, y, noisy))
    return out == y


TASKS = {0: c0, 1: c1, 2: c2, 3: c3, 4: c4, 5: c5, 6: c6}


def evals_for(W, task, cue=None):
    """Casos de evaluación (todos no vistos) de una tarea: [(nombre, fn(net, rng) -> bool)]."""
    k = task if cue is None else cue
    E = {
        0: [("f(x), todos los x", lambda n, r: c0(W, n, r, False, k))],
        1: [("v nunca compuesto, 1-2 saltos", lambda n, r: c1(W, n, r, False, k, test=True, hops=int(r.integers(1, 3)))),
            ("v nunca compuesto, 3-4 saltos", lambda n, r: c1(W, n, r, False, k, test=True))],
        2: [("x nunca compuesto", lambda n, r: c2(W, n, r, False, k, test=True))],
        3: [("5-8 saltos", lambda n, r: c3(W, n, r, False, k, hops=int(r.integers(5, 9)))),
            ("5-8 saltos, 16 distractores + 8 ruidos", lambda n, r: c3(W, n, r, False, k, hops=int(r.integers(5, 9)), distractors=16, noise=8)),
            ("3 saltos, orden reverso", lambda n, r: c3(W, n, r, False, k, hops=3, adverse="reverso")),
            ("3 saltos, hecho corregido", lambda n, r: c3(W, n, r, False, k, hops=3, adverse="corregido")),
            ("3 saltos, distractores que apuntan a la cadena", lambda n, r: c3(W, n, r, False, k, hops=3, adverse="apuntan"))],
        4: [("x nunca compuesto", lambda n, r: c4(W, n, r, False, k, test=True))],
        5: [("orden de 8, distancia >= 4", lambda n, r: c5(W, n, r, False, k, L=8, min_gap=4))],
        6: [("5 eventos", lambda n, r: c6(W, n, r, False, k, events=5))],
    }
    return E[task]


# ---------------------------------------------------------------- arnés
def make_thinker(W, n_cues=1, pair_memory=False, **kw):
    base = dict(memory=True, speak=True, judge=True, max_steps=10)
    base.update(kw)
    net = Thinker(W.S, n_cues=n_cues, **base)
    if pair_memory:
        net.fast = PairMemory(W.S)
    return net


def run(W, make, schedule, evals, seeds, n_test, noisy=0.0):
    """Entrena una red por semilla según `schedule[i]` ((tarea, consigna) o una función del
    rng que la devuelve) y evalúa. Devuelve ({caso: [acc por semilla]}, costo medio)."""
    out = {name: [] for name, _ in evals}
    cost = {"nodos_busqueda": [], "neuronas_say": []}
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        net = make(seed)
        for i in range(len(schedule)):
            task, cue = schedule[i](rng) if callable(schedule[i]) else schedule[i]
            TASKS[task](W, net, rng, True, cue, noisy=noisy)
        cost["nodos_busqueda"].append(net.expanded / len(schedule))
        cost["neuronas_say"].append(net.say.layer.n)
        net.reset(); net.plastic = False
        for name, fn in evals:
            out[name].append(float(np.mean([fn(net, rng) for _ in range(n_test)])))
    return out, {k: float(np.mean(v)) for k, v in cost.items()}


class PairMemory(FastMemory):
    """Ablación: la memoria rápida anterior (ligaduras sueltas símbolo -> símbolo, gana la
    más reciente de cualquier clave). Sin conjunción."""

    def recall(self, keys):
        if len(keys) == 0 or not self.events:
            return _NONE
        when: dict = {}
        for k, toks, t in self.events:
            if k & set(keys):
                for x in toks:
                    when[x] = max(when.get(x, 0), t)
        if not when:
            return _NONE
        idx = np.fromiter(when.keys(), dtype=np.int64, count=len(when))
        s = np.fromiter(when.values(), dtype=np.float32, count=len(when))
        return idx, s / s.max()


def plan_single(task, n):
    """Tarea sola. Las que usan f (C1, C2, C4) tienen dos consignas: 0 = f(x) y 1 = la tarea;
    el primer cuarto es sólo f(x) (currículo) y después se mezclan."""
    if task not in (1, 2, 4):
        return [(task, 0)] * n
    cur = n // 4
    return [(0, 0)] * cur + [lambda r, t=task: (t, 1) if r.random() < 0.5 else (0, 0)] * (n - cur)


def plan_mixed(tasks, n, cur):
    """Todas las consignas mezcladas (consigna = número de tarea), tras `cur` episodios de f(x)."""
    def draw(r):
        t = int(r.choice(tasks))
        return t, t
    return [(0, 0)] * cur + [draw] * (n - cur)


def plan_blocks(tasks, n_block, cur, reverse=False):
    """C0 primero; después cada tarea en su bloque, sin volver a las anteriores. Con
    `reverse`, orden adverso: de la última a la primera y C0 (el conocimiento f que usan
    C1, C2 y C4) al final."""
    out = []
    for t in tasks[1:]:
        out += [(t, t)] * n_block
    if reverse:
        return [x for t in tasks[1:][::-1] for x in [(t, t)] * n_block] + [(0, 0)] * cur
    return [(0, 0)] * cur + out


GROUPS = ("solas", "ruido", "grande", "largo", "multi", "ablaciones", "ablaciones_multi")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    which = sys.argv[3:] or GROUPS
    seeds = tuple(range(n_seeds))
    n_test = 200
    W = World(32)
    tasks = (0, 1, 2, 3, 4, 5, 6)
    ev_all = [(f"C{t} {name}", fn) for t in tasks for name, fn in evals_for(W, t, t)]

    for group in which:
        log = CsvLog(f"thinking_hard_{group}", ["prueba", "sistema", "caso", "acc_media", "acc_min",
                                                 "semillas", "nodos_busqueda", "neuronas_say", "segundos"])

        def go(test, system, Wd, make, plan, evals, noisy=0.0):
            t0 = time.time()
            res, cost = run(Wd, make, plan, evals, seeds, n_test, noisy=noisy)
            for name, accs in res.items():
                log.row(test, system, name, float(np.mean(accs)), float(np.min(accs)), len(accs),
                        cost["nodos_busqueda"], cost["neuronas_say"], float(time.time() - t0))

        if group == "solas":
            for task in (1, 2, 3, 4, 5, 6):
                two = task in (1, 2, 4)
                go(f"C{task} sola", "thinker", W, lambda s: make_thinker(W, 2 if two else 1, seed=s),
                   plan_single(task, n), evals_for(W, task, cue=1 if two else 0))
        if group == "ruido":
            for task in (3, 5, 6):
                go(f"C{task} etiquetas 10 % al azar", "thinker", W, lambda s: make_thinker(W, 1, seed=s),
                   plan_single(task, n), evals_for(W, task, 0), noisy=0.1)
        if group == "grande":
            WB = World(256)
            for task in (1, 3, 5, 6):
                two = task == 1
                go(f"C{task} vocabulario 256", "thinker", WB, lambda s: make_thinker(WB, 2 if two else 1, seed=s),
                   plan_single(task, n), evals_for(WB, task, cue=1 if two else 0))
        if group == "largo":
            WL = World(64)
            go("C3 cadenas largas (max_steps=20)", "thinker", WL, lambda s: make_thinker(WL, 1, seed=s, max_steps=20),
               plan_single(3, n), [("12-16 saltos", lambda net, r: c3(WL, net, r, False, 0, hops=int(r.integers(12, 17)), distractors=8))])
            go("C5 órdenes largos (max_steps=20)", "thinker", WL, lambda s: make_thinker(WL, 1, seed=s, max_steps=20),
               plan_single(5, n), [("orden de 16, distancia >= 8", lambda net, r: c5(WL, net, r, False, 0, L=16, min_gap=8))])
        if group == "multi":
            go("multitarea mezclada", "thinker", W, lambda s: make_thinker(W, 7, seed=s),
               plan_mixed(tasks, 7 * n // 2, n // 4), ev_all)
            go("multitarea en bloques (olvido)", "thinker", W, lambda s: make_thinker(W, 7, seed=s),
               plan_blocks(tasks, n // 2, n // 4), ev_all)
            go("multitarea en bloques, orden inverso (f al final)", "thinker", W, lambda s: make_thinker(W, 7, seed=s),
               plan_blocks(tasks, n // 2, n // 4, reverse=True), ev_all)
        if group == "ablaciones":
            # cada fila quita una pieza y vuelve a la versión anterior de esa pieza
            go("C3 sola", "con reloj y anticipación (v0)", W,
               lambda s: make_thinker(W, 1, seed=s, clock=True, lookahead=True), plan_single(3, n), evals_for(W, 3, 0)[:2])
            go("C5 sola", "establecido = hits>=1 (v0)", W, lambda s: make_thinker(W, 1, seed=s, say_sure=0.0),
               plan_single(5, n), evals_for(W, 5, 0))
            go("C6 sola", "memoria por pares (v0)", W, lambda s: make_thinker(W, 1, seed=s, pair_memory=True),
               plan_single(6, n), evals_for(W, 6, 0))
            for label, kw in (("búsqueda exhaustiva (v0)", dict(search="amplitud")),
                              ("crédito a lo percibido (v0)", dict(credit="percepcion"))):
                go("multitarea mezclada", label, W, lambda s, kw=kw: make_thinker(W, 7, seed=s, **kw),
                   plan_mixed(tasks, 7 * n // 2, n // 4), ev_all)
        if group == "ablaciones_multi":
            for label, kw in (("sin protección", dict(protect=False)),
                              ("pesos de control compartidos", dict(mixed=True))):
                go("multitarea mezclada", label, W, lambda s, kw=kw: make_thinker(W, 7, seed=s, **kw),
                   plan_mixed(tasks, 7 * n // 2, n // 4), ev_all)
            go("multitarea en bloques (olvido)", "sin fiabilidad en la búsqueda", W,
               lambda s: make_thinker(W, 7, seed=s, reliability=0.0), plan_blocks(tasks, n // 2, n // 4), ev_all)
        log.close()


if __name__ == "__main__":
    main()
