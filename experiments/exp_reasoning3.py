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
    t0 = time.time()
    out, cost = H.run(W, lambda s: H.make_thinker(W, 7, seed=s, **over), sched, ev_all, seeds, 200)
    ms_ep = (time.time() - t0) / (len(seeds) * (len(sched) + 200 * len(ev_all))) * 1e3
    return out, ms_ep, cost


def report_battery(log, test, label, res):
    out, ms_ep, cost = res
    for name, accs in out.items():
        log.row(test, label, "congelado, casos no vistos", name, float(np.mean(accs)), float(np.std(accs)), len(accs))
    log.row(test, label, "costo", "ms por episodio (entrenar + evaluar)", float(ms_ep), 0.0, len(SEEDS))
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


GROUPS = {"f1": group_f1, "f1v": group_f1v, "f2": group_f2, "f3": group_f3}


def main():
    for g in sys.argv[1:] or ["f1"]:
        t0 = time.time()
        GROUPS[g]()
        print(f"[{g}] {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
