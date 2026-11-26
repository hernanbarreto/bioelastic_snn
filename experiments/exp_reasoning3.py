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
  f2   estructura del mapa (`cogmap`): cierre del anillo y FIN con preguntas de aprendizaje
       mezcladas, ablaciones y control con sólo premisas.
  f3   `back` sin daño: batería C0-C6 en multitarea (mezclada y en bloques), diagnóstico del
       atajo posicional en la transitiva y relación inversa en la línea.
  f4   neuromoduladores: regímenes S/V de Harlow y Harlow congelado; f4v, validación.

  Cada condición terminada queda en results/razonamiento3_<grupo>.parts.json y se salta al
  volver a correr (resiste reinicios); al final se escriben el CSV y el JSON del grupo.

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


class RowLog:
    """Como `CsvLog`, pero junta las filas en memoria (para guardarlas por condición)."""

    def __init__(self):
        self.rows = []

    def row(self, *vals):
        out = [round(v, 4) if isinstance(v, float) else v for v in vals]
        self.rows.append(out)
        print("  ".join(f"{h}={v}" for h, v in zip(HEADER, out)), flush=True)


class Store:
    """Resultados por condición que sobreviven a un reinicio: cada condición terminada se
    guarda en results/razonamiento3_<grupo>.parts.json (filas del CSV y resumen) y al volver
    a correr el grupo se salta. Al final se escriben el CSV y el JSON completos."""

    def __init__(self, group: str):
        self.group = group
        self.path = RESULTS / f"razonamiento3_{group}.parts.json"
        self.parts = json.load(open(self.path)) if self.path.exists() else {}

    def done(self, key) -> bool:
        return key in self.parts

    def get(self, key):
        return self.parts[key]["summary"]

    def put(self, key, log: RowLog, summary):
        self.parts[key] = dict(rows=log.rows, summary=json.loads(json.dumps(summary, default=M._jsonable)))
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.parts, f, ensure_ascii=False, default=M._jsonable)
        tmp.replace(self.path)

    def run(self, key, fn):
        """Corre `fn(log)` (que devuelve el resumen) si la condición no estaba hecha."""
        if self.done(key):
            print(f"[{self.group}] ya hecho: {key}", flush=True)
            return self.get(key)
        log = RowLog()
        summary = fn(log)
        self.put(key, log, summary)
        return self.get(key)

    def finish(self, summary=None):
        log = CsvLog(f"razonamiento3_{self.group}", HEADER)
        for part in self.parts.values():
            for r in part["rows"]:
                log._w.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in r])
        log.close()
        dump(self.group, summary if summary is not None else {k: v["summary"] for k, v in self.parts.items()})


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


# ---------------------------------------------------------------- F2: mapa con estructura
F2_B = 800


def group_f2():
    """Cierre del anillo y FIN con preguntas de aprendizaje mezcladas (la condición en que la
    ronda anterior quedó en 0), con la estructura del mapa (`cogmap`): completitud, tamaño
    esperado aprendido, posición estructural y el juicio como módulo candidato."""
    st = Store("f2")
    WB = 50
    tests = (("L=8 (más largo que lo entrenado)", dict(L=8)), ("R=1 (cada relación observada una vez)", dict(R=1)))
    conds = [("B1 + back (ronda anterior)", dict(back=True)),
             ("B1 + back + estructura del mapa (cogmap)", dict(back=True, cogmap=True)),
             ("ablación: sin posición estructural", dict(back=True, cogmap="sin_posicion")),
             ("ablación: sin tamaño esperado", dict(back=True, cogmap="sin_tamano")),
             ("ablación: sin el juicio como candidato", dict(back=True, cogmap="sin_juicio")),
             ("ablación: cogmap sin back", dict(cogmap=True))]
    for label, over in conds:
        key = f"F2.1 anillo | {label}"
        st.run(key, lambda log, over=over, label=label: M.report_line(
            log, "F2.1 anillo con preguntas de aprendizaje", label,
            M.run_line(make(n_cues=4, **over), F2_B, struct="anillo", perceive=True, tests=tests), WB))
    for label, over in conds:
        key = f"F2.2 línea FIN | {label}"
        st.run(key, lambda log, over=over, label=label: M.report_line(
            log, "F2.2 línea con FIN y preguntas de aprendizaje", label,
            M.run_line(make(n_cues=4, **over), F2_B, struct="linea", perceive=True, use_end=True, queries=(2, 3),
                       tests=(tests[0],)), WB))
    st.run("F2.3 anillo sólo premisas | cogmap", lambda log: M.report_line(
        log, "F2.3 anillo, sólo premisas (control)", "B1 + back + cogmap",
        M.run_line(make(n_cues=4, back=True, cogmap=True), 300, struct="anillo", perceive=True, premises=True,
                   tests=(tests[0],)), 25))
    st.finish()


# ---------------------------------------------------------------- F3: back sin daño
def run_battery(over, seeds=SEEDS, n=3000, plan="mezclada"):
    """Batería C0-C6 de `exp_thinking_hard` en multitarea (mezclada o en bloques), casos no
    vistos con todo congelado. Devuelve ({caso: [acc por semilla]}, ms por episodio, costo)."""
    import exp_thinking_hard as H
    W = H.World(32)
    tasks = (0, 1, 2, 3, 4, 5, 6)
    ev_all = [(f"C{t} {name}", fn) for t in tasks for name, fn in H.evals_for(W, t, t)]
    sched = H.plan_mixed(tasks, 7 * n // 2, n // 4) if plan == "mezclada" else H.plan_blocks(tasks, n // 2, n // 4)
    t0 = time.time(); c0 = time.process_time()
    out, cost = H.run(W, lambda s: H.make_thinker(W, 7, seed=s, **over), sched, ev_all, seeds, 200)
    n_ep = len(seeds) * (len(sched) + 200 * len(ev_all))
    ms_ep = (time.time() - t0) / n_ep * 1e3
    cost["ms_cpu"] = (time.process_time() - c0) / n_ep * 1e3      # CPU del proceso: no depende de la carga ajena
    return out, ms_ep, cost


def report_battery(log, test, label, res):
    out, ms_ep, cost = res
    for name, accs in out.items():
        log.row(test, label, "congelado, casos no vistos", name, float(np.mean(accs)), float(np.std(accs)), len(accs))
    log.row(test, label, "costo", "ms por episodio (entrenar + evaluar)", float(ms_ep), 0.0, len(SEEDS))
    log.row(test, label, "costo", "ms de CPU por episodio (entrenar + evaluar)", float(cost["ms_cpu"]), 0.0, len(SEEDS))
    allv = np.array([v for v in out.values()])
    log.row(test, label, "resumen", "mínimo sobre casos y semillas", float(allv.min()), 0.0, len(SEEDS))
    return dict(acc=out, ms_per_episode=ms_ep, cost=cost, min=float(allv.min()))


def c5_shortcut(log, label, over, seeds=SEEDS, n=3000):
    """Diagnóstico de F3: la transitiva sola (órdenes de 3-4), evaluada con órdenes de 8 y
    distancia >= 4. Exactitud en los casos SÍ y NO y fracción de respuestas que juzgan sin
    recorrer la cadena (el atajo posicional)."""
    import exp_thinking_hard as H
    W = H.World(32)
    res = {"YES": [], "NO": [], "juzga sin recorrer (SÍ)": [], "juzga sin recorrer (NO)": []}
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        net = H.make_thinker(W, 1, seed=seed, **over)
        for task, cue in H.plan_single(5, n):
            H.TASKS[task](W, net, rng, True, cue)
        net.reset(); net.plastic = False
        ok = {"YES": [], "NO": []}; imm = {"YES": [], "NO": []}
        for _ in range(400):
            L = 8; syms = rng.permutation(W.N); order = [int(x) for x in syms[:L]]; rest = syms[L:]
            pairs = [(order[i], order[i + 1]) for i in range(L - 1)] + [(int(rest[2 * i]), int(rest[2 * i + 1])) for i in range(3)]
            while True:
                i, j = (int(t) for t in rng.choice(L, 2, replace=False))
                if abs(i - j) >= 4:
                    break
            y = W.YES if i < j else W.NO
            net.reset(); H._bind(net, pairs, rng)
            out = net.answer(order[i], cue=0, probe=order[j], explore=False)
            k = "YES" if y == W.YES else "NO"
            ok[k].append(out == y); imm[k].append(len(net._path) == 1 or net._path[0] == 3)   # JUZGAR (o HABLAR) primero
        for k in ("YES", "NO"):
            res[k].append(float(np.mean(ok[k])))
            res[f"juzga sin recorrer ({'SÍ' if k == 'YES' else 'NO'})"].append(float(np.mean(imm[k])))
    for k, v in res.items():
        log.row("F3.2 transitiva sola, órdenes de 8 (diagnóstico)", label, "congelado", k, float(np.mean(v)), float(np.std(v)), len(v))
    return res


def group_f3():
    st = Store("f3")
    conds = [("Thinker por defecto (sin back)", {}),
             ("ronda anterior: back con predecesores siempre visibles", dict(back="automatico")),
             ("back: predecesores al retroceder (+ coincidencia con lo sostenido)", dict(back=True)),
             ("prev_io + back (nuevo)", dict(prev_io=True, back=True))]
    for label, over in conds:
        st.run("F3.1 " + label, lambda log, over=over, label=label: report_battery(
            log, "F3.1 batería C0-C6, multitarea mezclada", label, run_battery(over)))
    for label, over in conds[1:3]:
        st.run("F3.2 " + label, lambda log, over=over, label=label: c5_shortcut(log, label, over))
    tests = (("L=8 (más largo que lo entrenado)", dict(L=8)), ("R=1 (cada relación observada una vez)", dict(R=1)))
    for label, over in conds[1:3]:
        st.run("F3.3 " + label, lambda log, over=over, label=label: M.report_line(
            log, "F3.3 línea: relación inversa y transitiva", label,
            M.run_line(make(n_cues=4, **over), 300, struct="linea", perceive=True, tests=tests), 25))
    for label, over in (conds[0], conds[2]):
        st.run("F3.4 " + label, lambda log, over=over, label=label: report_battery(
            log, "F3.4 batería C0-C6, multitarea en bloques (olvido)", label, run_battery(over, plan="bloques")))
    st.finish()


# ---------------------------------------------------------------- F4: neuromoduladores
NM = {}                                                    # constantes elegidas en validación (f4v)
NM_KEYS = ("rate", "forget", "sure", "epsilon", "decay", "capture", "lr", "temperature", "volatility", "noise")


NE_OLD = dict(ne="fiabilidad", unexplained_gain=4.0)     # la primera señal de NE del control (ablación)


def _traced_reset(self):
    """`reset` que antes deja en `self.trace` la media de los meta-parámetros del episodio.
    Es un método ligado (no una clausura): `copy.deepcopy` lo liga a la copia, así una
    instantánea congelada no reinicia la red original."""
    tr = self.modulators.trace
    if tr:
        self.trace.append({k: float(np.mean([t[k] for t in tr])) for k in NM_KEYS})
        self.modulators.trace = []
    type(self).reset(self)


def make_nm(nm=None, **over):
    """Thinker con los 7 meta-parámetros regulados por neuromoduladores. Para medir, cada
    episodio deja en `net.trace` la media de los meta-parámetros que fijaron los canales."""
    import types
    nm = dict(NM) if nm is None else nm
    base = make(modulators=(nm or True), **over)

    def mk(seed, n):
        net = base(seed, n)
        net.trace = []
        net.modulators.keep_trace = True
        net.reset = types.MethodType(_traced_reset, net)
        return net
    return mk


def harlow_stability(mk, P=800, seeds=SEEDS, cps=(0, 10, 25, 50, 100, 200, 300, 400, 600, 800)):
    """F2.5 de la ronda anterior: Harlow por la compuerta, sin resultado percibido (`prev_io`),
    instantáneas congeladas; ensayo 2 sobre 50 problemas nuevos en cada instantánea."""
    cur, rest, on = M.run_curve(mk, lambda i: ("objeto", 0), P, list(cps), [("W", "objeto", 0)], seeds=seeds,
                                perceive=False)
    return cur["W"], on


def report_stability(log, name, label, arr, cps, start=50):
    m, sd = ms(arr)
    for j, c in enumerate(cps):
        log.row(name, label, f"instantánea tras {c} problemas", "ensayo 2", float(m[j]), float(sd[j]), len(arr))
    after = arr[:, list(cps).index(start):]
    log.row(name, label, f"instantáneas desde {start}", "mínimo por semilla", float(after.min(1).mean()),
            float(after.min(1).std()), len(arr))
    log.row(name, label, f"instantáneas desde {start}", "instantáneas < 0.95 (cantidad)",
            float((after < 0.95).sum()), 0.0, len(arr))
    return dict(checkpoints=list(cps), trial2_mean=m, trial2_sd=sd, per_seed=arr,
                min_after=after.min(1), dips=int((after < 0.95).sum()))


def group_f4():
    from bioelastic.metalearning import PARAMS
    st = Store("f4")
    P = 3000
    plus = ("exito", "sorpresa", "td", "novedad", "persistencia")
    conds = [("valores fijos a mano (Thinker, tasa 1)", make()),
             ("tasa fija 0.15", M.make_meta(fixed={"rate": 0.15})),
             ("tasa fija 0.3", M.make_meta(fixed={"rate": 0.3})),
             ("ronda anterior: los 7 aprendidos por perturbación", M.make_meta(meta=tuple(PARAMS), signals=plus)),
             ("neuromoduladores: los 7 adaptativos", make_nm()),
             ("ablación: sin memoria bayesiana (tasa 1)", make_nm(dict(NM, memory=None))),
             ("ablación: sólo la memoria bayesiana (control con valores fijos)", make_nm(dict(NM, control=False))),
             ("ablación: horizonte del episodio (5-HT sin acortar)", make_nm(dict(NM, horizon="episodio"))),
             ("ablación: NE = fiabilidad de la regla (primera versión)", make_nm(dict(NM, **NE_OLD)))]
    for label, mk in conds:
        st.run("F4.1 " + label, lambda log, mk=mk, label=label: M.report_regimes(
            log, "F4.1 harlow S/V, todo aprende", label, M.run_regimes(mk, P, pretrain=100)))
    s15 = st.get("F4.1 tasa fija 0.15"); s10 = st.get("F4.1 valores fijos a mano (Thinker, tasa 1)")
    s03 = st.get("F4.1 tasa fija 0.3")

    def oracle(log):
        best_s = [max(a, b, c) for a, b, c in zip(s15["S"]["frozen"], s10["S"]["frozen"], s03["S"]["frozen"])]
        best_v = [max(a, b, c) for a, b, c in zip(s15["V"]["frozen"], s10["V"]["frozen"], s03["V"]["frozen"])]
        both = [(a + b) / 2 for a, b in zip(best_s, best_v)]
        log.row("F4.1 harlow S/V, todo aprende", "óptimo por régimen (mejor tasa fija de cada uno, oráculo)",
                "congelado", "elige el premiado, promedio S y V", float(np.mean(both)), float(np.std(both)), len(both))
        return dict(S=best_s, V=best_v, both=both)
    st.run("F4.1 óptimo por régimen", oracle)
    cps = (0, 10, 25, 50, 100, 200, 300, 400, 600, 800)
    for label, mk in (("ronda anterior (prev_io)", make(gate=True, prev_io=True)),
                      ("neuromoduladores", make_nm(gate=True, prev_io=True)),
                      ("ablación: horizonte del episodio", make_nm(dict(NM, horizon="episodio"), gate=True, prev_io=True)),
                      ("ablación: sin temperatura de conducta (NE sólo en epsilon)",
                       make_nm(dict(NM, temp=(1.0, None)), gate=True, prev_io=True)),
                      ("ablación: NE = fiabilidad de la regla (primera versión)",
                       make_nm(dict(NM, **NE_OLD), gate=True, prev_io=True))):
        st.run("F4.2 " + label, lambda log, mk=mk, label=label: report_stability(
            log, "F4.2 harlow congelado (prev_io, sin resultado percibido)", label, harlow_stability(mk, cps=cps)[0], cps))
    # F4.3 con el control estabilizado: cierre del anillo y FIN con preguntas de aprendizaje (F2)
    # y batería C0-C6 con back (F3)
    tests = (("L=8 (más largo que lo entrenado)", dict(L=8)), ("R=1 (cada relación observada una vez)", dict(R=1)))
    for label, over in (("back + cogmap (F2)", dict(back=True, cogmap=True)),
                        ("back + cogmap + neuromoduladores", dict(back=True, cogmap=True, modulators=(dict(NM) or True))),
                        ("ablación: horizonte del episodio", dict(back=True, cogmap=True, modulators=dict(NM, horizon="episodio"))),
                        ("ablación: NE = fiabilidad de la regla (primera versión)",
                         dict(back=True, cogmap=True, modulators=dict(NM, **NE_OLD)))):
        st.run("F4.3 anillo | " + label, lambda log, over=over, label=label: M.report_line(
            log, "F4.3 anillo con preguntas de aprendizaje", label,
            M.run_line(make(n_cues=4, **over), F2_B, struct="anillo", perceive=True, tests=tests), 50))
        st.run("F4.3 FIN | " + label, lambda log, over=over, label=label: M.report_line(
            log, "F4.3 línea con FIN y preguntas de aprendizaje", label,
            M.run_line(make(n_cues=4, **over), F2_B, struct="linea", perceive=True, use_end=True, queries=(2, 3),
                       tests=(tests[0],)), 50))
    st.run("F4.4 batería | back + neuromoduladores", lambda log: report_battery(
        log, "F4.4 batería C0-C6, multitarea mezclada", "back + neuromoduladores",
        run_battery(dict(back=True, modulators=(dict(NM) or True)))))
    # (el RETROCEDER que no encuentra nada ahora deja el foco: se vuelve a medir la batería de F3)
    st.run("F4.4 batería | back", lambda log: report_battery(
        log, "F4.4 batería C0-C6, multitarea mezclada", "back (sin neuromoduladores)", run_battery(dict(back=True))))
    st.finish()


def group_f4v():
    """Validación (semillas 10-12) de las constantes de los canales: régimen S/V (1500
    problemas) y estabilidad de Harlow congelado (400 problemas). Las del control (temperatura
    mínima, n0) no cambian estas medidas (la memoria decide el régimen S/V y Harlow queda en
    1.00 con todas); de la memoria se elige prior_q=1.0 y phi=0.02, que son los valores por
    defecto de `BayesMemory`. (Una validación anterior, con la causa «otro» recibiendo parte
    del ruido y volatilidades hasta 0.4 en la grilla, dio 0.796 en S/V.)"""
    st = Store("f4v")
    grid = [dict(memory=dict(prior_q=2.0)), dict(memory=dict(prior_q=1.0)), dict(memory=dict(prior_q=4.0)),
            dict(memory=dict(prior_q=2.0, phi=0.01)), dict(memory=dict(prior_q=2.0, phi=0.04)),
            dict(memory=dict(prior_q=2.0), temp=(0.25, None)), dict(memory=dict(prior_q=2.0), n0=20.0)]
    for g in grid:
        name = json.dumps(g)

        def fn(log, g=g, name=name):
            sm = M.report_regimes(log, "F4v harlow S/V", name, M.run_regimes(make_nm(g), 1500, seeds=VAL_SEEDS,
                                                                                pretrain=100))
            arr, _ = harlow_stability(make_nm(g, gate=True, prev_io=True), P=400, seeds=VAL_SEEDS,
                                      cps=(25, 50, 100, 200, 300, 400))
            st_ = report_stability(log, "F4v harlow congelado", name, arr, (25, 50, 100, 200, 300, 400))
            score = float(np.mean(sm["frozen_mean_SV"])) + 0.5 * float(np.mean(st_["min_after"]))
            log.row("F4v resumen", name, "validación", "S/V + 0.5 x mínimo de Harlow", score, 0.0, len(VAL_SEEDS))
            return dict(sv=sm["frozen_mean_SV"], harlow_min=st_["min_after"], score=score)
        st.run(name, fn)
    st.finish()


def group_f4w():
    """Validación (semillas 10-12) de la señal de NE del control: fiabilidad de la regla
    (versión anterior: en un régimen ruidoso y estable NE queda alta para siempre y la conducta
    en línea sortea) contra incertidumbre inesperada (fallos por encima de los esperados).
    Medidas: régimen S/V en línea y congelado (1500 problemas), mínimo de Harlow congelado
    (400 problemas) y, para las finalistas, cierre del anillo y FIN (400 bloques)."""
    st = Store("f4w")
    grid = [dict(), dict(ne="inesperada"), dict(ne="inesperada", ne_gain=2.0),
            dict(ne="inesperada", unexplained_gain=0.0), dict(ne="inesperada", unexplained_gain=0.0, ne_gain=8.0),
            dict(ne="inesperada", unexplained_gain=0.0, ne_floor=0.05), dict(ne="inesperada", unexplained_gain=0.0, ne_floor=0.1),
            dict(ne="inesperada", unexplained_gain=0.0, beta=0.02)]
    for g in grid:
        name = json.dumps(g)

        def fn(log, g=g, name=name):
            sm = M.report_regimes(log, "F4w harlow S/V", name, M.run_regimes(make_nm(g), 1500, seeds=VAL_SEEDS,
                                                                                pretrain=100))
            arr, _ = harlow_stability(make_nm(g, gate=True, prev_io=True), P=400, seeds=VAL_SEEDS,
                                      cps=(25, 50, 100, 200, 300, 400))
            st_ = report_stability(log, "F4w harlow congelado", name, arr, (25, 50, 100, 200, 300, 400))
            online = float(np.mean(sm["S"]["online_last_third"]) + np.mean(sm["V"]["online_last_third"])) / 2
            frozen = float(np.mean(sm["frozen_mean_SV"]))
            score = frozen + online + 0.5 * float(np.mean(st_["min_after"]))
            log.row("F4w resumen", name, "validación", "S/V en línea (promedio S y V)", online, 0.0, len(VAL_SEEDS))
            log.row("F4w resumen", name, "validación", "S/V congelado + en línea + 0.5 x mínimo de Harlow", score, 0.0,
                    len(VAL_SEEDS))
            return dict(sv=sm["frozen_mean_SV"], online=online, harlow_min=st_["min_after"], score=score)
        st.run(name, fn)
    tests = (("L=8 (más largo que lo entrenado)", dict(L=8)),)
    for g in (dict(), dict(ne="inesperada", unexplained_gain=0.0)):
        name = json.dumps(g)
        over = dict(back=True, cogmap=True, modulators=(dict(NM, **g) or True))
        st.run("anillo " + name, lambda log, over=over, name=name: M.report_line(
            log, "F4w anillo con preguntas de aprendizaje", name,
            M.run_line(make(n_cues=4, **over), 400, seeds=VAL_SEEDS, struct="anillo", perceive=True, tests=tests), 50))
        st.run("FIN " + name, lambda log, over=over, name=name: M.report_line(
            log, "F4w línea con FIN y preguntas de aprendizaje", name,
            M.run_line(make(n_cues=4, **over), 400, seeds=VAL_SEEDS, struct="linea", perceive=True, use_end=True,
                       queries=(2, 3), tests=tests), 50))
    st.finish()


# ---------------------------------------------------------------- F5: todo encendido
OLD = dict(prev_io=False, back=False, task_sets=False, cogmap=False, modulators=False)   # defaults de la v0.3


def hard_cases(over, seeds=SEEDS, n=3000):
    """Los grupos de `exp_thinking_hard` (solas, ruido, grande, largo, multi), sin escribir
    sus CSV: [(prueba, {caso: [acc por semilla]})]."""
    import exp_thinking_hard as H
    W = H.World(32); WB = H.World(256); WL = H.World(64)
    tasks = (0, 1, 2, 3, 4, 5, 6)
    ev_all = [(f"C{t} {name}", fn) for t in tasks for name, fn in H.evals_for(W, t, t)]
    runs = []
    for task in (1, 2, 3, 4, 5, 6):
        two = task in (1, 2, 4)
        runs.append((f"C{task} sola", W, lambda s, two=two: H.make_thinker(W, 2 if two else 1, seed=s, **over),
                     H.plan_single(task, n), H.evals_for(W, task, cue=1 if two else 0), 0.0))
    for task in (3, 5, 6):
        runs.append((f"C{task} etiquetas 10 % al azar", W, lambda s: H.make_thinker(W, 1, seed=s, **over),
                     H.plan_single(task, n), H.evals_for(W, task, 0), 0.1))
    for task in (1, 3, 5, 6):
        two = task == 1
        runs.append((f"C{task} vocabulario 256", WB, lambda s, two=two: H.make_thinker(WB, 2 if two else 1, seed=s, **over),
                     H.plan_single(task, n), H.evals_for(WB, task, cue=1 if two else 0), 0.0))
    runs.append(("C3 cadenas largas (max_steps=20)", WL, lambda s: H.make_thinker(WL, 1, seed=s, max_steps=20, **over),
                 H.plan_single(3, n), [("12-16 saltos", lambda net, r: H.c3(WL, net, r, False, 0, hops=int(r.integers(12, 17)), distractors=8))], 0.0))
    runs.append(("C5 órdenes largos (max_steps=20)", WL, lambda s: H.make_thinker(WL, 1, seed=s, max_steps=20, **over),
                 H.plan_single(5, n), [("orden de 16, distancia >= 8", lambda net, r: H.c5(WL, net, r, False, 0, L=16, min_gap=8))], 0.0))
    runs.append(("multitarea en bloques, orden inverso (f al final)", W, lambda s: H.make_thinker(W, 7, seed=s, **over),
                 H.plan_blocks(tasks, n // 2, n // 4, reverse=True), ev_all, 0.0))
    return runs


def run_harlow_suite(over, seeds=SEEDS):
    """Harlow por la compuerta con resultado percibido (B1) + reversión y 3 objetos, y Harlow
    sin resultado percibido (`prev_io`): instantáneas congeladas."""
    rev = ("reversión en el ensayo 4 (T=8)", dict(T=8, reverse_at=3))
    three = ("3 objetos (entrenado con 2)", dict(k=3))
    res = M.run_harlow(make(gate=True, **over), 400, seeds=seeds, k=2, perceive=True, split=True, tests=(rev, three))
    return res


def group_f5():
    import exp_thinking_hard as H
    import exp_integration as EI
    import exp_integration2 as EI2
    st = Store("f5")
    systems = (("v0.3 (todo apagado)", OLD), ("todo encendido (nuevo por defecto)", {}))
    # F5.1 batería C0-C6 (multitarea mezclada y en bloques) y latencia
    for label, over in systems:
        st.run("F5.1 mezclada | " + label, lambda log, over=over, label=label: report_battery(
            log, "F5.1 batería C0-C6, multitarea mezclada", label, run_battery(over)))
        st.run("F5.1 bloques | " + label, lambda log, over=over, label=label: report_battery(
            log, "F5.1 batería C0-C6, multitarea en bloques (olvido)", label, run_battery(over, plan="bloques")))
    # F5.2 resto de exp_thinking_hard (todo encendido; la v0.3 está en results/thinking_hard_*.csv)
    for test, Wd, mk, plan, evals, noisy in hard_cases({}):
        def fn(log, test=test, Wd=Wd, mk=mk, plan=plan, evals=evals, noisy=noisy):
            t0 = time.time()
            out, cost = H.run(Wd, mk, plan, evals, SEEDS, 200, noisy=noisy)
            for name, accs in out.items():
                log.row("F5.2 " + test, "todo encendido", "congelado, casos no vistos", name, float(np.mean(accs)),
                        float(np.std(accs)), len(accs))
            log.row("F5.2 " + test, "todo encendido", "costo", "segundos", float(time.time() - t0), 0.0, len(SEEDS))
            return dict(acc=out, cost=cost)
        st.run("F5.2 " + test, fn)
    # F5.3 suma decimal con acarreo (símbolos perfectos, 1 registro, 12 000 episodios), 8 cifras
    for label, over in systems:
        def carry(log, over=over, label=label):
            datos = EI.Digitos(10)
            res = {n: [] for n in (1, 3, 8)}; secs = []
            for seed in SEEDS:
                rng = np.random.default_rng(1000 + seed)
                th = H.Thinker(21, registers=1, seed=seed, **over)
                t0 = time.time()
                EI2.entrenar_suma(th, EI2.Oraculo(), datos, rng, 12000, {}, {})
                secs.append(time.time() - t0)
                for n in res:
                    res[n].append(EI2.evaluar_suma(th, EI2.Oraculo(), datos, rng, n, 500)["exacta"])
            for n, v in res.items():
                log.row("F5.3 suma con acarreo (1 registro, 12 000 episodios)", label, "congelado", f"exacta, {n} cifras",
                        float(np.mean(v)), float(np.std(v)), len(v))
            log.row("F5.3 suma con acarreo (1 registro, 12 000 episodios)", label, "costo", "segundos de entrenamiento",
                    float(np.mean(secs)), float(np.std(secs)), len(secs))
            return {str(k): v for k, v in res.items()}
        st.run("F5.3 acarreo | " + label, carry)
    # F5.4 Harlow (B1), reversión, 3 objetos; Harlow sin resultado percibido (estabilidad)
    for label, over in systems:
        def harlow(log, over=over, label=label):
            res = run_harlow_suite(over)
            sm = M.report_harlow(log, "F5.4 harlow por la compuerta (B1)", label, res, 50)
            for tname, arr in res["tests"].items():
                sm["test " + tname] = M.report_by_trial(log, "F5.4 congelado, " + tname, label, arr)
            return sm
        st.run("F5.4 harlow B1 | " + label, harlow)
        cps = (0, 10, 25, 50, 100, 200, 300, 400, 600, 800)
        st.run("F5.4 harlow prev_io | " + label, lambda log, over=over, label=label: report_stability(
            log, "F5.4 harlow congelado (prev_io, sin resultado percibido)", label,
            harlow_stability(make(gate=True, **dict(over, prev_io=True)), cps=cps)[0], cps))
    # F5.5 transferencia de estructura (línea), anillo y FIN con preguntas de aprendizaje
    tests = (("L=8 (más largo que lo entrenado)", dict(L=8)), ("R=1 (cada relación observada una vez)", dict(R=1)))
    for label, over in systems:
        st.run("F5.5 línea | " + label, lambda log, over=over, label=label: M.report_line(
            log, "F5.5 transferencia de estructura (línea)", label,
            M.run_line(make(n_cues=4, **over), 300, struct="linea", perceive=True, tests=tests), 25))
        st.run("F5.5 anillo | " + label, lambda log, over=over, label=label: M.report_line(
            log, "F5.5 anillo con preguntas de aprendizaje", label,
            M.run_line(make(n_cues=4, **over), F2_B, struct="anillo", perceive=True, tests=tests), 50))
        st.run("F5.5 FIN | " + label, lambda log, over=over, label=label: M.report_line(
            log, "F5.5 línea con FIN y preguntas de aprendizaje", label,
            M.run_line(make(n_cues=4, **over), F2_B, struct="linea", perceive=True, use_end=True, queries=(2, 3),
                       tests=(tests[0],)), 50))
    # F5.6 familias de reglas (F1) y regímenes S/V (F4) con todo encendido
    cps = [0, 25, 50, 100, 200, 400, 410, 425, 450, 500, 600, 800, 805, 810, 825, 850, 1000]
    sched = lambda i: ("W" if i < 400 or i >= 800 else "A", 0)       # noqa: E731
    ev = [("W (premia un objeto)", "W", 0), ("A (el premio alterna)", "A", 0)]
    st.run("F5.6 familias sin consigna | todo encendido", lambda log: report_families(
        log, "F5.6 sin consigna W -> A -> W", "todo encendido", run_families(make(gate=True), sched, 1000, cps, ev), cps,
        switches=(400, 800), ts_ref=((400, 799), (800, 399))))
    fams = [("W", 0, 300), ("A", 1, 300), ("W", 2, 100), ("A", 3, 100)]
    starts = np.cumsum([0] + [n for _, _, n in fams])

    def sched2(i):
        j = int(np.searchsorted(starts, i, side="right")) - 1
        return fams[j][0], fams[j][1]
    cps2 = [0, 300, 600, 605, 610, 625, 700, 705, 710, 725, 800]
    ev2 = [("W (consigna 0)", "W", 0), ("A (consigna 1)", "A", 1), ("W' (consigna 2)", "W", 2), ("A' (consigna 3)", "A", 3)]
    st.run("F5.6 familias con consigna | todo encendido", lambda log: report_families(
        log, "F5.6 con consigna W, A, W', A'", "todo encendido", run_families(make(gate=True, n_cues=4), sched2, 800, cps2, ev2),
        cps2, switches=(0, 300, 600, 700), ts_ref=((600, 299), (700, 599))))
    st.run("F5.6 regímenes S/V | todo encendido", lambda log: M.report_regimes(
        log, "F5.6 harlow S/V, todo aprende", "todo encendido", M.run_regimes(make(), 3000, pretrain=100)))
    st.finish()


GROUPS = {"f1": group_f1, "f1v": group_f1v, "f2": group_f2, "f3": group_f3, "f4": group_f4, "f4v": group_f4v, "f4w": group_f4w, "f5": group_f5}


def main():
    for g in sys.argv[1:] or ["f1"]:
        t0 = time.time()
        GROUPS[g]()
        print(f"[{g}] {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
