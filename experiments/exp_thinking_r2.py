"""Ronda 2 del Thinker: antes/después de los fallos, ablaciones de cada atajo sacado y
pruebas para romperlo. Todo en casos no vistos y con >= 5 semillas.

  antes      la configuración de la ronda 1: consolidate="paciencia", sin repetición,
             búsqueda con la red misma (model="perfecto"), sin confianza, sin la
             restricción reactiva y con el sostén dado por la pregunta.
  despues    la de ahora (consolidación por repetición, revisión por explicación,
             repetición de errores, modelo interno aprendido, confianza, búsqueda
             reactiva). El sostén aprendido (`gate`) se mide aparte.

Grupos (cada uno corre en <= 10 minutos en un núcleo):
  fallos        T4 sin currículo; bloques en orden inverso (f al final).
  registros     T3 (suma) con registros y protección a la vez.
  orden         órdenes de presentación adversos: directo, inverso, mezclado y bloques en
                orden al azar.
  abl_<pieza>   ablación de una pieza (modelo, repeticion, consolidacion, confianza,
                reactivo, trazas, credito) sobre T4 sin currículo, bloques inversos y
                transitiva sola con sostén aprendido.
  sosten        sostén aprendido (compuerta) contra sostén dado.
  contradiccion consignas que se contradicen sin estrategia que las reconcilie.
  cambio        cambio de regla a mitad del flujo (¿revisa lo aprendido y lo que depende?).
  vocabulario   vocabularios más grandes.
  presupuesto   presupuesto de búsqueda chico.

  python experiments/exp_thinking_r2.py <grupo> [<grupo> ...]
  (resultados en results/thinking_r2_<grupo>.csv)
"""
from __future__ import annotations

import sys
import time

import numpy as np

from common import CsvLog
import exp_thinking as T
import exp_thinking_hard as H
from bioelastic.thinker import Thinker

ANTES = dict(consolidate="paciencia", replay=0, model="perfecto", confidence=False,
             reactive=False, gate=False, replacing=False, credit="eslabon")
DESPUES: dict = {}
SEEDS = (0, 1, 2, 3, 4)
TASKS = (0, 1, 2, 3, 4, 5, 6)
HEADER = ["prueba", "sistema", "caso", "acc_media", "acc_min", "semillas", "nodos_busqueda",
          "neuronas_say", "segundos"]


def ev_all(W):
    return [(f"C{t} {name}", fn) for t in TASKS for name, fn in H.evals_for(W, t, t)]


def go(log, test, system, W, make, plan, evals, seeds=SEEDS, n_test=200):
    t0 = time.time()
    res, cost = H.run(W, make, plan, evals, seeds, n_test)
    for name, accs in res.items():
        log.row(test, system, name, float(np.mean(accs)), float(np.min(accs)), len(accs),
                cost["nodos_busqueda"], cost["neuronas_say"], float(time.time() - t0))


# ---------------------------------------------------------------- T4 (composición) y T3 (suma)
def t4(log, system, kw, n=6000, curriculum=False, seeds=SEEDS, f=None, N=16, test="T4 sin currículo"):
    """f(x) y f(f(x)); f(f(x)) sólo se entrena con x en la primera mitad."""
    t0 = time.time()
    f = T.F4 if f is None else f
    half = N // 2
    out = {"f(f(x)), x no vistos": [], "f(x), todos": []}
    nodes, neur = [], []
    for seed in seeds:
        rng = np.random.default_rng(100 + seed)
        net = Thinker(N, n_cues=2, speak=True, seed=seed, **kw)

        def ep(learn, x=None, cue=None):
            cue = int(rng.integers(2)) if cue is None else cue
            x = (int(rng.integers(N)) if cue == 0 else int(rng.integers(half))) if x is None else x
            y = int(f[x]) if cue == 0 else int(f[f[x]])
            net.reset(); o = net.answer(x, cue=cue, explore=learn); net.feedback(target=y)
            return o == y
        for i in range(n):
            ep(True, cue=0 if (curriculum and i < n // 4) else None)
        nodes.append(net.expanded / n); neur.append(net.say.layer.n)
        net.reset(); net.plastic = False
        out["f(f(x)), x no vistos"].append(float(np.mean([ep(False, x, 1) for x in range(half, N)])))
        out["f(x), todos"].append(float(np.mean([ep(False, x, 0) for x in range(N)])))
    for name, accs in out.items():
        log.row(test, system, name, float(np.mean(accs)), float(np.min(accs)), len(accs),
                float(np.mean(nodes)), float(np.mean(neur)), float(time.time() - t0))


def t3(log, system, kw, force_protect=None, n=2500, seeds=tuple(range(8))):
    """Suma binaria con acarreo en un registro: entrena 1-6 dígitos, evalúa 50 exactos."""
    t0 = time.time()
    accs, neur = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        net = Thinker(5, registers=1, seed=seed, **kw)
        if force_protect is not None:
            net.protect = force_protect
        for _ in range(n):
            T.t3_episode(net, rng, True, int(rng.integers(1, 7)))
        net.reset(); net.plastic = False
        accs.append(float(np.mean([T.t3_episode(net, rng, False, 50) for _ in range(50)])))
        neur.append(net.say.layer.n)
    log.row("T3 suma con registro", system, "50 dígitos exactos (fracción de semillas = 1.0: "
            f"{sum(a == 1.0 for a in accs)}/{len(accs)})", float(np.mean(accs)), float(np.min(accs)),
            len(accs), 0.0, float(np.mean(neur)), float(time.time() - t0))


def c5_single(log, system, W, kw, n=3000, seeds=SEEDS, test="C5 transitiva sola"):
    go(log, test, system, W, lambda s: H.make_thinker(W, 1, seed=s, **kw), H.plan_single(5, n),
       H.evals_for(W, 5, 0), seeds)


def blocks(log, system, W, kw, reverse=True, n=3000, test=None, budget=None):
    kw = dict(kw, **({} if budget is None else dict(budget=budget)))
    test = test or ("bloques, orden inverso (f al final)" if reverse else "bloques, orden directo")
    go(log, test, system, W, lambda s: H.make_thinker(W, 7, seed=s, **kw),
       H.plan_blocks(TASKS, n // 2, n // 4, reverse=reverse), ev_all(W))


def mixed(log, system, W, kw, n=3000, test="multitarea mezclada"):
    go(log, test, system, W, lambda s: H.make_thinker(W, 7, seed=s, **kw),
       H.plan_mixed(TASKS, 7 * n // 2, n // 4), ev_all(W))


def plan_random_blocks(n_block, cur, seed):
    """Bloques en un orden al azar (distinto por semilla de orden), con C0 en cualquier lugar."""
    order = list(np.random.default_rng(seed).permutation(TASKS))
    return [x for t in order for x in [(int(t), int(t))] * (cur if t == 0 else n_block)], order


# ---------------------------------------------------------------- rupturas
def contradiction(log, system, kw, n=4000, seeds=SEEDS, N=32):
    """Dos consignas con respuestas directas incompatibles sobre el mismo contenido:
    consigna 0: x -> f(x); consigna 1: x -> g(x), g sin relación con f (ninguna estrategia
    las reconcilia). Un solo conocimiento compartido no puede sostener las dos."""
    t0 = time.time()
    f = H.derangement(N, 21); g = H.derangement(N, 22)
    out = {"consigna 0 (f)": [], "consigna 1 (g)": [], "ambas a la vez": []}
    for seed in seeds:
        rng = np.random.default_rng(300 + seed)
        net = Thinker(N, n_cues=2, speak=True, seed=seed, **kw)

        def ep(learn, x, cue):
            y = int(f[x]) if cue == 0 else int(g[x])
            net.reset(); o = net.answer(x, cue=cue, explore=learn); net.feedback(target=y)
            return o == y
        for _ in range(n):
            ep(True, int(rng.integers(N)), int(rng.integers(2)))
        net.reset(); net.plastic = False
        a0 = [ep(False, x, 0) for x in range(N)]; a1 = [ep(False, x, 1) for x in range(N)]
        out["consigna 0 (f)"].append(float(np.mean(a0))); out["consigna 1 (g)"].append(float(np.mean(a1)))
        out["ambas a la vez"].append(float(np.mean(np.logical_and(a0, a1))))
    for name, accs in out.items():
        log.row("consignas contradictorias (f y g sin relación)", system, name, float(np.mean(accs)),
                float(np.min(accs)), len(accs), 0.0, 0.0, float(time.time() - t0))


def rule_change(log, system, kw, n=3000, n_after=1500, seeds=SEEDS, N=16):
    """Fase A: f(x) y f(f(x)) mezclados (con currículo). Fase B: la regla cambia a g y sólo se
    entrena g(x). ¿Se revisa el conocimiento? ¿La composición (nunca reentrenada) pasa a
    usar g? Se mide también la precisión en línea de los primeros 300 episodios de B."""
    t0 = time.time()
    f = T.F4; g = H.derangement(N, 31)
    out = {"g(x) tras el cambio": [], "g(g(x)) sin reentrenar, x no vistos": [],
           "f(x) todavía (debería ser 0)": [], "en línea, 300 primeros de g": [],
           "revisiones [misma consigna]": []}
    for seed in seeds:
        rng = np.random.default_rng(500 + seed)
        net = Thinker(N, n_cues=2, speak=True, seed=seed, **kw)

        def ep(learn, h, x=None, cue=None):
            cue = int(rng.integers(2)) if cue is None else cue
            x = (int(rng.integers(N)) if cue == 0 else int(rng.integers(N // 2))) if x is None else x
            y = int(h[x]) if cue == 0 else int(h[h[x]])
            net.reset(); o = net.answer(x, cue=cue, explore=learn); net.feedback(target=y)
            return o == y
        for i in range(n):
            ep(True, f, cue=0 if i < n // 4 else None)
        online = [ep(True, g, cue=0) for _ in range(n_after)]
        net.reset(); net.plastic = False
        out["g(x) tras el cambio"].append(float(np.mean([ep(False, g, x, 0) for x in range(N)])))
        out["g(g(x)) sin reentrenar, x no vistos"].append(
            float(np.mean([ep(False, g, x, 1) for x in range(N // 2, N)])))
        out["f(x) todavía (debería ser 0)"].append(float(np.mean([ep(False, f, x, 0) for x in range(N)])))
        out["en línea, 300 primeros de g"].append(float(np.mean(online[:300])))
        out["revisiones [misma consigna]"].append(float(net.revised[0]))
    for name, accs in out.items():
        log.row("cambio de regla a mitad del flujo", system, name, float(np.mean(accs)),
                float(np.min(accs)), len(accs), 0.0, 0.0, float(time.time() - t0))


# ---------------------------------------------------------------- grupos
def main():
    groups = sys.argv[1:] or ["fallos"]
    W = H.World(32)
    for group in groups:
        log = CsvLog(f"thinking_r2_{group}", HEADER)
        if group == "fallos":
            for system, kw in (("antes", ANTES), ("despues", DESPUES)):
                t4(log, system, kw)
            for system, kw in (("antes", ANTES), ("despues", DESPUES)):
                blocks(log, system, W, kw, reverse=True)
        if group == "registros":
            t3(log, "antes, sin protección (la ronda 1 la apagaba con registros)", ANTES)
            t3(log, "antes, con protección forzada", ANTES, force_protect=True)
            t3(log, "despues (con protección)", DESPUES)
        if group == "orden":
            blocks(log, "despues", W, DESPUES, reverse=False)
            mixed(log, "despues", W, DESPUES)
            for k in range(3):
                plan, order = plan_random_blocks(1500, 750, 40 + k)
                go(log, "bloques en orden al azar " + "-".join(f"C{t}" for t in order), "despues", W,
                   lambda s: H.make_thinker(W, 7, seed=s), plan, ev_all(W))
        if group.startswith("abl_"):
            piece = group[4:]
            kw = {"modelo": dict(model="perfecto"), "repeticion": dict(replay=0),
                  "consolidacion": dict(consolidate="paciencia"), "confianza": dict(confidence=False),
                  "reactivo": dict(reactive=False), "trazas": dict(replacing=False),
                  "credito": dict(credit="estricto")}[piece]
            label = {"modelo": "modelo perfecto (la red misma)",
                     "trazas": "trazas acumulativas",
                     "credito": "crédito estricto (nada al eslabón intermedio)"}.get(piece, "sin " + piece)
            t4(log, label, kw)
            blocks(log, label, W, kw, reverse=True)
            c5_single(log, label, W, dict(kw, gate=True), test="C5 transitiva sola, sostén aprendido")
        if group == "sosten":
            for label, kw in (("sostén dado (probe)", dict(gate=False)), ("sostén aprendido", dict(gate=True))):
                c5_single(log, label, W, kw)
            c5_single(log, "sostén aprendido", W, dict(gate=True), n=6000, test="C5 transitiva sola, 6000 episodios")
            mixed(log, "sostén aprendido", W, dict(gate=True))
        if group == "sosten_solas":
            for task in (1, 3, 6):
                two = task in (1, 2, 4)
                go(log, f"C{task} sola, sostén aprendido", "despues", W,
                   lambda s, two=two: H.make_thinker(W, 2 if two else 1, seed=s, gate=True),
                   H.plan_single(task, 3000), H.evals_for(W, task, cue=1 if two else 0))
            t3(log, "sostén aprendido (compuerta en cada dígito)", dict(gate=True), seeds=SEEDS)
        if group == "contradiccion":
            for system, kw in (("antes", ANTES), ("despues", DESPUES)):
                contradiction(log, system, kw)
        if group == "cambio":
            for system, kw in (("antes", ANTES), ("despues", DESPUES)):
                rule_change(log, system, kw)
        if group == "vocabulario":
            for N in (64, 128):
                t4(log, "despues", DESPUES, N=N, f=H.derangement(N, 41), n=6000 if N == 64 else 12000,
                   test=f"composición sin currículo, vocabulario {N}")
            WB = H.World(256)
            for task in (1, 4):
                go(log, f"C{task} sola, vocabulario 256", "despues", WB,
                   lambda s: H.make_thinker(WB, 2, seed=s), H.plan_single(task, 6000),
                   H.evals_for(WB, task, cue=1))
        if group.startswith("presupuesto"):
            budgets = (4, 16) if group == "presupuesto" else (8,)
            for b in budgets:
                blocks(log, f"presupuesto {b}", W, DESPUES, reverse=True, budget=b)
                go(log, "multitarea mezclada", f"presupuesto {b}", W,
                   lambda s, b=b: H.make_thinker(W, 7, seed=s, budget=b),
                   H.plan_mixed(TASKS, 7 * 3000 // 2, 3000 // 4), ev_all(W))
        log.close()


if __name__ == "__main__":
    main()
