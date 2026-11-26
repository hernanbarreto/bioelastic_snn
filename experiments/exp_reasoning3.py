"""Razonamiento, ronda 3: lo que dejó abierto el meta-aprendizaje (F1-F5).

Protocolo (el de `exp_metalearning`): en línea, primero se mide y después se aprende; un
problema o bloque es un episodio. Instantáneas congeladas (`plastic=False`, sin
exploración) sobre problemas nuevos con un generador aparte. ≥3 semillas por condición;
los hiperparámetros se eligen con semillas de validación (`VAL_SEEDS`), distintas de las de
prueba (`SEEDS`).

Grupos
  f1   conjuntos de tareas y causas latentes (`task_sets`): dos familias de reglas opuestas
       sin consigna (W -> A -> W), recuperación al volver, y con consigna transferencia a
       familias nuevas que comparten estructura (C-TS).
  f1v  validación de los hiperparámetros de F1 (semillas de validación).

  python experiments/exp_reasoning3.py <grupo> [<grupo> ...]   (OMP_NUM_THREADS=1)
  (resultados en results/razonamiento3_<grupo>.csv y .json)
"""
from __future__ import annotations

import copy
import json
import sys
import time

import numpy as np

from common import RESULTS, CsvLog
import exp_metalearning as M
from exp_metalearning import Supply, harlow_problem, ms, bins

SEEDS = (0, 1, 2)
VAL_SEEDS = (10, 11, 12)
HEADER = ["prueba", "sistema", "tramo", "medida", "media", "desvio", "semillas"]


def dump(name, obj):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"razonamiento3_{name}.json", "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=M._jsonable)


def make(**over):
    return M.make_base(**over)


# ---------------------------------------------------------------- F1: familias de reglas
FAMILIES = {"W": "objeto", "A": "alterna"}


def run_families(mk, schedule, P, checkpoints, evals, seeds=SEEDS, T=6, n_eval=50, perceive=False):
    """Familias de reglas de Harlow por la compuerta (los dos objetos como ítems separados),
    sin el resultado percibido: la regla rápida vive en el estado (`prev_io`).
    `schedule(i)` -> (familia, consigna) del problema i. En cada instantánea, red copiada,
    congelada y evaluada sin exploración en `evals` [(nombre, familia, consigna)], en
    orden, `n_eval` problemas nuevos cada una (la inferencia del conjunto de tareas sigue
    activa congelada: es estado). Devuelve instantáneas (ensayo 2 y ensayos 2-6), aciertos
    en línea por ensayo y el conjunto que regía en cada problema."""
    snaps = {name: [] for name, _, _ in evals}
    snaps_rest = {name: [] for name, _, _ in evals}
    snaps_first = {name: [] for name, _, _ in evals}
    online, active, secs, act_trials = [], [], [], []
    k = 2
    for seed in seeds:
        rng = np.random.default_rng(4000 + seed)
        n_obj = (P + len(checkpoints) * len(evals) * n_eval) * k
        net = mk(seed, n_obj + 2)
        sup = Supply(n_obj, rng)
        erng = np.random.default_rng(9000 + seed)
        cur = {name: [] for name, _, _ in evals}; rest = {name: [] for name, _, _ in evals}
        first = {name: [] for name, _, _ in evals}
        on = np.zeros((P, T)); act = np.full(P, -1); actt = np.full((P, T), -1)
        t0 = time.time()
        for i in range(P + 1):
            if i in checkpoints:
                fz = copy.deepcopy(net)
                fz.reset(); fz.plastic = False
                for name, fam, cue in evals:
                    a = np.array([harlow_problem(fz, erng, sup.take(k), T, FAMILIES[fam], perceive, cue=cue,
                                                 explore=False, split=True)[0] for _ in range(n_eval)])
                    cur[name].append(a[:, 1].mean()); rest[name].append(a[:, 1:].mean())
                    first[name].append(a[:2].ravel())
            if i == P:
                break
            fam, cue = schedule(i)
            if net.task_sets is not None:
                fb = net.feedback
                trace = []

                def fb_rec(*a, **kw):
                    fb(*a, **kw); trace.append(net.task_sets.active)
                net.feedback = fb_rec
            hit, _, _ = harlow_problem(net, rng, sup.take(k), T, FAMILIES[fam], perceive, cue=cue, split=True)
            on[i] = hit
            if net.task_sets is not None:
                del net.feedback
                act[i] = net.task_sets.active; actt[i] = trace
        secs.append(time.time() - t0)
        for name in cur:
            snaps[name].append(cur[name]); snaps_rest[name].append(rest[name]); snaps_first[name].append(first[name])
        online.append(on); active.append(act); act_trials.append(actt)
    return dict(snap={n: np.array(v) for n, v in snaps.items()},
                snap_rest={n: np.array(v) for n, v in snaps_rest.items()},
                snap_first={n: np.array(v) for n, v in snaps_first.items()},
                online=np.array(online), active=np.array(active), act_trials=np.array(act_trials), seconds=secs)


def ts_recovery(act_trials, start, ref):
    """Ensayos desde el cambio de familia (en el problema `start`) hasta que rige el conjunto
    `ref` y ya no cambia en los 3 problemas siguientes (-1 si no ocurre)."""
    out = []
    for a, r in zip(act_trials, ref):
        seq = a[start:start + 3].ravel()
        bad = np.flatnonzero(seq != r)
        out.append(0 if len(bad) == 0 else (int(bad[-1]) + 1 if bad[-1] < len(seq) - 1 else -1))
    return np.array(out, dtype=float)


def recovery(online, start, n=3):
    """Ensayos (dentro de los primeros `n` problemas desde `start`, contados desde el
    ensayo 2 de cada problema) hasta el primer acierto que ya no se pierde en el resto de
    esos problemas; y exactitud por ensayo de los primeros problemas."""
    out = []
    for on in online:
        seq = on[start:start + n, 1:].ravel()
        wrong = np.flatnonzero(~seq.astype(bool))
        out.append(0 if len(wrong) == 0 else int(wrong[-1]) + 1)
    return np.array(out, dtype=float)


def report_families(log, test, system, res, checkpoints, switches=(), ts_ref=()):
    sm = {}
    for name, arr in res["snap"].items():
        m, sd = ms(arr); mr, sr = ms(res["snap_rest"][name])
        for j, c in enumerate(checkpoints):
            log.row(test, system, f"instantánea tras {c} problemas", f"{name}: ensayo 2", float(m[j]), float(sd[j]), len(arr))
        fm = res["snap_first"][name].mean(0)              # instantáneas x (2 problemas x T ensayos)
        sm[name] = dict(checkpoints=checkpoints, trial2_mean=m, trial2_sd=sd, trial2_per_seed=arr,
                        trials2to6_mean=mr, trials2to6_sd=sr, first_two_problems_by_trial=fm)
    on = res["online"]
    for s, ref in ts_ref:                                   # recuperación por inferencia (conjunto que rige)
        if res["act_trials"].max() < 0:
            continue
        refs = [a[ref] for a in res["active"]]
        r = ts_recovery(res["act_trials"], s, refs)
        log.row(test, system, f"en línea, desde el problema {s}", "ensayos hasta que rige el conjunto de la familia",
                float(r.mean()), float(r.std()), len(r))
        sm[f"ts_recovery_from_{s}"] = r
    for s in switches:
        r = recovery(on, s)
        log.row(test, system, f"en línea, desde el problema {s}", "ensayos hasta recuperar (desde el ensayo 2)",
                float(r.mean()), float(r.std()), len(r))
        first = on[:, s:s + 5, 1].mean(1)
        log.row(test, system, f"en línea, problemas {s}-{s + 4}", "ensayo 2", float(first.mean()), float(first.std()), len(first))
        sm[f"recovery_from_{s}"] = r
        sm[f"online_trial2_{s}_{s + 4}"] = first
    sm["online_trial2_by25"] = np.array([bins(o[:, 1], 25) for o in on])
    sm["active_by25"] = res["active"][:, ::25]
    sm["seconds"] = res["seconds"]
    return sm


def group_f1():
    log = CsvLog("razonamiento3_f1", HEADER)
    summary = {}
    base = dict(gate=True, prev_io=True)
    ts = dict(base, task_sets=dict(TS_F1))
    # F1.1 sin consigna: W 400, A 400, W 200
    cps = [0, 25, 50, 100, 200, 400, 410, 425, 450, 500, 600, 800, 805, 810, 825, 850, 1000]
    sched = lambda i: ("W" if i < 400 or i >= 800 else "A", 0)       # noqa: E731
    ev = [("W (premia un objeto)", "W", 0), ("A (el premio alterna)", "A", 0)]
    conds = [("línea base: un solo control (prev_io)", make(**base)),
             ("conjuntos de tareas (task_sets)", make(**ts)),
             ("ablación: sin autocorrelación (hazard=0.5)", make(**dict(base, task_sets=dict(TS_F1, hazard=0.5)))),
             ("ablación: sin vigilar alternativas", make(**dict(base, task_sets=dict(TS_F1, explained=0.0))))]
    for label, mk in conds:
        res = run_families(mk, sched, 1000, cps, ev)
        summary["F1.1 " + label] = report_families(log, "F1.1 sin consigna W -> A -> W", label, res, cps,
                                                   switches=(400, 800), ts_ref=((400, 799), (800, 399)))
    # F1.2 con consigna: W (0) 300, A (1) 300, W' (2) 100, A' (3) 100; W' y A' comparten
    # la estructura de W y A con una consigna nunca vista
    cps2 = [0, 300, 600, 605, 610, 625, 700, 705, 710, 725, 800]
    fams = [("W", 0, 300), ("A", 1, 300), ("W", 2, 100), ("A", 3, 100)]
    starts = np.cumsum([0] + [n for _, _, n in fams])

    def sched2(i):
        j = int(np.searchsorted(starts, i, side="right")) - 1
        return fams[j][0], fams[j][1]
    ev2 = [("W (consigna 0)", "W", 0), ("A (consigna 1)", "A", 1), ("W' (consigna 2)", "W", 2), ("A' (consigna 3)", "A", 3)]
    conds2 = [("línea base: control conjugado con la consigna", make(n_cues=4, **base)),
              ("línea base: control compartido (mixed=True)", make(n_cues=4, mixed=True, **base)),
              ("conjuntos de tareas (task_sets)", make(n_cues=4, **ts))]
    for label, mk in conds2:
        res = run_families(mk, sched2, 800, cps2, ev2)
        summary["F1.2 " + label] = report_families(log, "F1.2 con consigna W, A, W', A'", label, res, cps2,
                                                   switches=(0, 300, 600, 700), ts_ref=((600, 299), (700, 599)))
    # F1.3 aprender a aprender: A desde cero contra A después de W (consignas distintas),
    # con un conjunto nuevo que arranca en cero o en el promedio encogido de los existentes
    cps3 = [0, 10, 25, 50, 100]
    for label, mk, pre in (("A desde cero", make(**base), 0),
                           ("A tras W, conjuntos de tareas (nuevo en cero)", make(n_cues=2, **ts), 300),
                           (f"A tras W, conjuntos de tareas (nuevo = {TS_SHRINK} x promedio)",
                            make(n_cues=2, **dict(base, task_sets=dict(TS_F1, shrink=TS_SHRINK))), 300)):
        sched3 = (lambda i, pre=pre: ("W", 0) if i < pre else ("A", 1 if pre else 0))
        res = run_families(mk, sched3, pre + 100, [pre + c for c in cps3], [("A", "A", 1 if pre else 0)])
        summary["F1.3 " + label] = report_families(log, "F1.3 aprender A (instantáneas desde que empieza A)", label,
                                                   res, cps3, switches=(pre,))
    log.close()
    dump("f1", summary)


TS_F1 = dict(alpha=0.5, hazard=0.03)                  # elegidos en validación (f1v)
TS_SHRINK = 0.5


def group_f1v():
    """Validación (semillas 10-12) de hazard y alpha en F1.1 y de shrink en F1.3."""
    log = CsvLog("razonamiento3_f1v", HEADER)
    summary = {}
    base = dict(gate=True, prev_io=True)
    cps = [0, 50, 400, 425, 450, 800, 805, 810]
    sched = lambda i: ("W" if i < 400 or i >= 800 else "A", 0)       # noqa: E731
    ev = [("W", "W", 0), ("A", "A", 0)]
    grid = [(0.03, al, True) for al in (0.5, 1.0, 2.0)] + [(0.01, 1.0, True)]
    for hz, al, gated in grid:
        name = f"hazard={hz} alpha={al}" + ("" if gated else " sin ganancia por creencia")
        res = run_families(make(**dict(base, task_sets=dict(alpha=al, hazard=hz, gated=gated))), sched, 900, cps, ev,
                           seeds=VAL_SEEDS)
        if True:
            sm = report_families(log, "F1v sin consigna", name, res, cps, switches=(400, 800))
            W, A = sm["W"]["trial2_mean"], sm["A"]["trial2_mean"]
            score = float(np.mean([W[2:].mean(), A[3:].mean()]))
            log.row("F1v sin consigna", name, "resumen", "media W tras 400 y A tras 425", score, 0.0, 3)
            summary[name] = dict(sm, score=score)
    for sh in ():
        res = run_families(make(n_cues=2, **dict(base, task_sets=dict(TS_F1, shrink=sh))),
                           lambda i: ("W", 0) if i < 300 else ("A", 1), 400, [300, 310, 325, 350],
                           [("A", "A", 1)], seeds=VAL_SEEDS)
        sm = report_families(log, "F1v A tras W", f"shrink={sh}", res, [0, 10, 25, 50], switches=(300,))
        summary[f"shrink={sh}"] = sm
    log.close()
    dump("f1v", summary)


GROUPS = {"f1": group_f1, "f1v": group_f1v}


def main():
    for g in sys.argv[1:] or ["f1"]:
        t0 = time.time()
        GROUPS[g]()
        print(f"[{g}] {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
