"""Aprender a aprender en el Thinker: tareas, línea base y mecanismos (F0-F4).

Protocolo: en línea, primero se mide y después se aprende (la respuesta con la que se mide
es la que el Thinker eligió, con su exploración). Un problema o bloque es un episodio
(`reset` al empezar); dentro, varios ensayos (`answer` + `feedback`). Además, al final de
cada corrida, una prueba con todo congelado (`plastic=False`, sin exploración) sobre
problemas nuevos.

Interfaz del resultado (cómo entra lo que pasó en un ensayo al siguiente):
  B0  sólo `feedback(target)`: lo que aprende el Thinker de un ensayo va a sus pesos.
  B1  además, el entorno muestra el resultado como percepción: `observe(pregunta)`,
      `observe(objetivo)`, silencio. Es la entrada «recompensa y acción previas» de
      meta-RL (Wang et al., 2018) en el idioma del Thinker; no toca el Thinker.

Tareas
  harlow    Harlow (1949) simbólico: en cada problema dos objetos nunca vistos, uno
            premiado, `T` ensayos; el orden (izquierda/derecha) cambia al azar en cada
            ensayo. Medida: exactitud en el ensayo 2 en función del número de problema.
            Modos: "objeto" (premia siempre el mismo objeto), "azar" (control sin
            estructura: el objetivo se sortea en cada ensayo), "ruido" (el premiado
            acierta con probabilidad p), "volatil" (el premiado cambia con probabilidad h
            tras cada ensayo), "posicion" (premia una posición fija, no un objeto),
            "alterna" (el premio alterna de objeto en cada ensayo).
  linea     Transferencia de estructura: en cada bloque, L símbolos nunca vistos en un
            orden oculto. Fase de aprendizaje: «¿qué sigue a X?» (consigna 0) sobre las
            relaciones adyacentes, R veces cada una, en orden al azar. Después, preguntas
            sobre relaciones nunca observadas, medidas en su primera aparición:
            «¿X antes que Y?» (consigna 1, distancia >= 2, SÍ/NO), «¿qué está dos después
            de X?» (consigna 2) y «¿qué viene antes de X?» (consigna 3, la relación
            inversa). Estructuras: "linea", "anillo" (se observan L-1 de las L aristas;
            la que falta se pregunta: cierre), "funcion" (control: X -> f(X) consistente
            pero f lleva fuera del bloque: no hay cadenas) y "azar" (control: objetivos
            sorteados en cada pregunta).

  python experiments/exp_metalearning.py <grupo> [<grupo> ...]   (OMP_NUM_THREADS=1)
  grupos: f0 f1
  (resultados en results/metaaprendizaje_<grupo>.csv y .json)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from common import RESULTS, CsvLog
from bioelastic.thinker import Thinker

SEEDS = (0, 1, 2)
BASE = dict(memory=True, speak=True, judge=True, max_steps=10)


# ---------------------------------------------------------------- utilidades
class Supply:
    """Símbolos nunca vistos: cada pedido devuelve los siguientes de una permutación."""

    def __init__(self, n: int, rng):
        self.perm = rng.permutation(n)
        self.i = 0

    def take(self, k: int) -> list:
        out = [int(x) for x in self.perm[self.i:self.i + k]]
        if len(out) < k:
            raise RuntimeError("se acabaron los símbolos nuevos")
        self.i += k
        return out


def ms(x, axis=0):
    """Media y desvío entre semillas (desvío poblacional, ddof=0)."""
    a = np.asarray(x, dtype=np.float64)
    return a.mean(axis=axis), a.std(axis=axis)


def bins(a, width):
    """Promedio por tramos de `width` filas (problemas o bloques)."""
    a = np.asarray(a, dtype=np.float64)
    n = len(a) // width
    return a[:n * width].reshape(n, width, *a.shape[1:]).mean(axis=1)


def dump(name, obj):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"metaaprendizaje_{name}.json", "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_jsonable)


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return np.round(x, 4).tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    raise TypeError(type(x))


# ---------------------------------------------------------------- Harlow
def harlow_problem(net, rng, objs, T=6, mode="objeto", perceive=True, cue=0, explore=True,
                   p=0.8, h=0.3, learn=True, split=False, reverse_at=None):
    """Un problema de discriminación. Devuelve (aciertos contra el objetivo de cada ensayo,
    aciertos contra el objeto que conviene elegir: el premiado en "ruido"/"volatil", y si
    lo más fuerte que devuelve la memoria episódica ante la pregunta es ese objeto).
    `split`: los objetos llegan como ítems separados (por la compuerta) en vez de juntos en
    el foco. `reverse_at`: en ese ensayo el premiado pasa a ser otro (reversión)."""
    k = len(objs)
    good = int(rng.integers(k))                       # índice del objeto (o posición) premiado
    alt = int(rng.integers(k))
    hit, best, mem = [], [], []
    net.reset()
    for t in range(T):
        if reverse_at is not None and t == reverse_at:
            good = (good + 1 + int(rng.integers(k - 1))) % k
        q = [objs[i] for i in rng.permutation(k)]
        if mode == "objeto":
            y = objs[good]
        elif mode == "azar":
            y = objs[int(rng.integers(k))]
        elif mode == "ruido":
            y = objs[good] if rng.random() < p else objs[(good + 1 + int(rng.integers(k - 1))) % k]
        elif mode == "volatil":
            y = objs[good]
        elif mode == "posicion":
            y = q[good]
        elif mode == "alterna":
            y = objs[(alt + t) % k]
        else:
            raise ValueError(mode)
        want = objs[good] if mode in ("ruido", "volatil") else y
        if net.fast is not None:
            mem.append(net._best(net.fast.recall(tuple(q))) == want)
        if split:
            out = net.answer(q[0], cue=cue, explore=explore, probe=q[1], extra=tuple((x,) for x in q[2:]))
        else:
            out = net.answer(tuple(q), cue=cue, explore=explore)
        hit.append(out == y)
        best.append(out == want)
        if learn:
            net.feedback(target=y)
        if perceive:
            net.observe(tuple(q)); net.observe(y); net.observe(None)
        if mode == "volatil" and rng.random() < h:
            good = (good + 1 + int(rng.integers(k - 1))) % k
    return hit, best, mem


def run_harlow(make, P, seeds=SEEDS, T=6, k=2, mode="objeto", perceive=True, n_test=100,
               test_mode=None, tests=(), **kw):
    """Entrena con P problemas por semilla (en línea) y prueba congelado con `n_test`
    problemas nuevos. `tests`: pruebas congeladas adicionales [(nombre, dict con k, T,
    mode, reverse_at...)], cada una con `n_test` problemas nuevos, en orden y sobre la red
    congelada al final del entrenamiento. Devuelve dict con curvas por semilla."""
    out = dict(online=[], online_best=[], frozen=[], seconds=[], answers=[])
    tests = list(tests)
    out["tests"] = {name: [] for name, _ in tests}
    kmax = max([k] + [t.get("k", k) for _, t in tests])
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        n_obj = (P + n_test) * k + n_test * kmax * len(tests)
        net = make(seed, n_obj + 2)
        sup = Supply(n_obj, rng)
        t0 = time.time()
        acc = np.zeros((P, T)); accb = np.zeros((P, T))
        for i in range(P):
            a, b, _ = harlow_problem(net, rng, sup.take(k), T, mode, perceive, **kw)
            acc[i] = a; accb[i] = b
        dt = time.time() - t0
        net.reset(); net.plastic = False
        # congelado: `feedback` se sigue llamando (con plastic=False no aprende nada, pero la
        # recompensa es entrada del estado con `prev_io`)
        fr = np.array([harlow_problem(net, rng, sup.take(k), T, test_mode or mode, perceive,
                                      explore=False, **kw)[1] for _ in range(n_test)])
        for name, t in tests:
            t = dict(t); kk = t.pop("k", k); TT = t.pop("T", T); mm = t.pop("mode", mode)
            kw2 = dict(kw); kw2.update(t)
            out["tests"][name].append(np.array([harlow_problem(net, rng, sup.take(kk), TT, mm, perceive,
                                                               explore=False, **kw2)[1] for _ in range(n_test)]))
        out["online"].append(acc); out["online_best"].append(accb); out["frozen"].append(fr)
        out["seconds"].append(dt); out["answers"].append(P * T)
    res = {key: np.array(v) for key, v in out.items() if key != "tests"}
    res["tests"] = {name: np.array(v) for name, v in out["tests"].items()}
    return res


# ---------------------------------------------------------------- estructura
def line_block(net, rng, syms, YES, NO, R=3, struct="linea", perceive=True, explore=True,
               learn=True, n_before=6, extra=None, queries=(1, 2, 3), end=None):
    """Un bloque de transferencia de estructura. `syms`: los L símbolos del bloque (orden
    oculto). `extra`: símbolos fuera del bloque (para "funcion"). Devuelve un dict con
    aciertos por tipo de pregunta (en su primera aparición) y errores de la fase de
    aprendizaje. `end`: símbolo FIN; si se da, en la línea también se pregunta qué sigue al
    último (objetivo FIN), la misma pregunta que en el anillo es el cierre."""
    L = len(syms)
    edges = [(i, i + 1) for i in range(L - 1)]
    if struct == "anillo":
        edges.append((L - 1, 0))
        rng.shuffle(edges)
        hidden = edges.pop()                              # la arista que no se observa
    else:
        hidden = None
    succ = {i: j for i, j in edges}
    if hidden is not None:
        succ[hidden[0]] = hidden[1]

    def target(i):
        if struct == "azar":
            return syms[int(rng.integers(L))]
        if struct == "funcion":
            return extra[i]
        return syms[succ[i]]

    def ask(tok, cue, y, probe=None):
        out = net.answer(tok, cue=cue, probe=probe, explore=explore)
        if learn:
            net.feedback(target=y)
        return out == y

    net.reset()
    trials = [i for i, _ in edges for _ in range(R)]
    rng.shuffle(trials)
    seq = []
    for i in trials:
        y = target(i)
        seq.append(ask(syms[i], 0, y))
        if perceive:
            net.observe(syms[i]); net.observe(y); net.observe(None)
    res = dict(errors=float(len(seq) - sum(seq)), ttc=float(_ttc(seq)))
    if hidden is not None:                                # cierre del anillo
        res["closure"] = float(ask(syms[hidden[0]], 0, syms[hidden[1]]))
    elif end is not None and struct == "linea":           # ¿qué sigue al último? FIN
        res["closure"] = float(ask(syms[L - 1], 0, end))
    order = list(range(L)) if struct != "anillo" else None
    if 1 in queries and order is not None:
        pairs = [(i, j) for i in range(L) for j in range(L) if abs(i - j) >= 2]
        ok = []
        for q in rng.permutation(len(pairs))[:n_before]:
            i, j = pairs[q]
            y = (YES if i < j else NO) if struct == "linea" else (YES if rng.random() < 0.5 else NO)
            ok.append(ask(syms[i], 1, y, probe=syms[j]))
        res["before"] = float(np.mean(ok))
    if 2 in queries:
        ok = []
        cand = [i for i in range(L) if i in succ and succ[i] in succ] if struct in ("linea", "anillo") else list(range(L - 2))
        for i in rng.permutation(cand)[:3]:
            i = int(i)
            y = syms[succ[succ[i]]] if struct in ("linea", "anillo") else syms[int(rng.integers(L))]
            ok.append(ask(syms[i], 2, y))
        res["two_step"] = float(np.mean(ok))
    if 3 in queries:
        pred = {j: i for i, j in succ.items()}
        cand = [j for j in range(L) if j in pred] if struct in ("linea", "anillo") else list(range(1, L))
        ok = []
        for j in rng.permutation(cand)[:3]:
            j = int(j)
            y = syms[pred[j]] if struct in ("linea", "anillo") else syms[int(rng.integers(L))]
            ok.append(ask(syms[j], 3, y))
        res["inverse"] = float(np.mean(ok))
    return res


def _ttc(seq):
    """Ensayos hasta el criterio: posición del último error + 1 (desde ahí, todo bien).
    0 si no hubo errores; len+1 si el último ensayo fue un error."""
    wrong = [t for t, ok in enumerate(seq) if not ok]
    if not wrong:
        return 0
    return wrong[-1] + 2 if wrong[-1] == len(seq) - 1 else wrong[-1] + 1


def run_line(make, B, seeds=SEEDS, L=5, R=3, struct="linea", perceive=True, n_test=50,
             test_struct=None, tests=(), schedule=None, use_end=False, **kw):
    """Entrena con B bloques por semilla y prueba congelado con `n_test` bloques nuevos.
    `tests`: pruebas congeladas adicionales [(nombre, dict con L, R, struct)].
    `schedule`: función bloque -> estructura (para cambiar de estructura a mitad)."""
    out = {}
    tests = list(tests)
    Lmax = max([L] + [t.get("L", L) for _, t in tests])
    for seed in seeds:
        rng = np.random.default_rng(2000 + seed)
        n = (B + n_test) * L * (2 if "funcion" in (struct, test_struct) else 1)
        n += n_test * len(tests) * Lmax * 2
        net = make(seed, n + (3 if use_end else 2))
        YES, NO = n, n + 1
        if use_end:
            kw["end"] = n + 2
        sup = Supply(n, rng)
        t0 = time.time()
        rows = []
        for b in range(B):
            st = schedule(b) if schedule is not None else struct
            syms = sup.take(L)
            extra = sup.take(L) if st == "funcion" else None
            rows.append(line_block(net, rng, syms, YES, NO, R, st, perceive, extra=extra, **kw))
        dt = time.time() - t0
        net.reset(); net.plastic = False
        for name, t in [("", dict(L=L, R=R, struct=test_struct or (schedule(B - 1) if schedule else struct)))] + tests:
            frows = []
            for _ in range(n_test):
                LL = t.get("L", L); ts = t.get("struct", struct)
                syms = sup.take(LL)
                extra = sup.take(LL) if ts == "funcion" else None
                frows.append(line_block(net, rng, syms, YES, NO, t.get("R", R), ts, perceive,
                                        explore=False, extra=extra, **kw))
            pre = "frozen_" + (name + " | " if name else "")
            for key in frows[0]:
                out.setdefault(pre + key, []).append(float(np.mean([r[key] for r in frows])))
        keys = sorted(set(k for r in rows for k in r))
        for key in keys:
            out.setdefault("online_" + key, []).append([r.get(key, np.nan) for r in rows])
        out.setdefault("seconds", []).append(dt)
        out.setdefault("say_neurons", []).append(net.say.layer.n)
        out.setdefault("decisions", []).append(net.stats())
    return {k: (np.array(v) if k != "decisions" else v) for k, v in out.items()}


# ---------------------------------------------------------------- informes
def report_harlow(log, name, system, res, width):
    """Filas: ensayo 2 (y 1) por tramo de problemas, media ± desvío entre semillas."""
    tr2 = np.array([bins(a[:, 1], width) for a in res["online"]])       # semillas x tramos
    tr1 = np.array([bins(a[:, 0], width) for a in res["online"]])
    m2, s2 = ms(tr2); m1, s1 = ms(tr1)
    for i in range(tr2.shape[1]):
        log.row(name, system, f"problemas {i * width}-{(i + 1) * width - 1}", "ensayo 2",
                float(m2[i]), float(s2[i]), len(tr2))
    fz = np.array([f[:, 1].mean() for f in res["frozen"]])
    fz1 = np.array([f[:, 0].mean() for f in res["frozen"]])
    log.row(name, system, "congelado, problemas nuevos", "ensayo 2", float(fz.mean()), float(fz.std()), len(fz))
    log.row(name, system, "congelado, problemas nuevos", "ensayo 1", float(fz1.mean()), float(fz1.std()), len(fz1))
    fine = np.array([bins(a[:100, 1], 10) for a in res["online"]])       # primeros 100, de a 10
    fm, fs = ms(fine)
    for i in range(min(3, len(fm))):
        log.row(name, system, f"problemas {i * 10}-{i * 10 + 9}", "ensayo 2", float(fm[i]), float(fs[i]), len(fine))
    return dict(trial2_bins_mean=m2, trial2_bins_sd=s2, trial1_bins_mean=m1, trial1_bins_sd=s1,
                trial2_first100_by10_mean=fm, trial2_first100_by10_sd=fs,
                frozen_trial2=fz, frozen_trial1=fz1,
                frozen_by_trial_mean=np.mean([f.mean(0) for f in res["frozen"]], 0),
                online_last_by_trial_mean=np.mean([a[-width:].mean(0) for a in res["online"]], 0),
                us_per_answer=float(np.mean(np.array(res["seconds"]) / np.array(res["answers"])) * 1e6),
                width=width)


def report_line(log, name, system, res, width):
    summ = {}
    for key in sorted(res):
        if key.startswith("online_"):
            b = np.array([np.nanmean(np.asarray(a, dtype=float)[:len(a) // width * width].reshape(-1, width), 1)
                          for a in res[key]])
            m, s = ms(b)
            summ[key + "_bins_mean"] = m; summ[key + "_bins_sd"] = s
            for i in (0, len(m) // 2, len(m) - 1):
                log.row(name, system, f"bloques {i * width}-{(i + 1) * width - 1}", key[7:],
                        float(m[i]), float(s[i]), len(b))
        elif key.startswith("frozen_"):
            m, s = ms(res[key])
            summ[key] = res[key]
            log.row(name, system, "congelado, bloques nuevos", key[7:], float(m), float(s), len(res[key]))
    summ["say_neurons"] = res["say_neurons"]
    summ["decisions"] = res.get("decisions")
    summ["width"] = width
    return summ


HEADER = ["prueba", "sistema", "tramo", "medida", "media", "desvio", "semillas"]


def make_base(**over):
    def make(seed, n):
        kw = dict(BASE); kw.update(over)
        return Thinker(n, seed=seed, **kw)
    return make


# ---------------------------------------------------------------- F0
def group_f0():
    log = CsvLog("metaaprendizaje_f0", HEADER)
    summary = {}
    P, W = 400, 50
    for perceive, label in ((False, "B0 sin resultado percibido"), (True, "B1 resultado percibido")):
        for mode, name in (("objeto", "harlow"), ("azar", "harlow control (azar)")):
            res = run_harlow(make_base(), P, mode=mode, perceive=perceive)
            summary[f"{name} | {label}"] = report_harlow(log, name, label, res, W)
    B, WB = 200, 25
    for perceive, label in ((False, "B0 sin resultado percibido"), (True, "B1 resultado percibido")):
        for struct, name in (("linea", "linea"), ("funcion", "linea control (funcion sin cadenas)"),
                             ("azar", "linea control (azar)")):
            res = run_line(make_base(n_cues=4), B, struct=struct, perceive=perceive)
            summary[f"{name} | {label}"] = report_line(log, name, label, res, WB)
    log.close()
    dump("f0", summary)


# ---------------------------------------------------------------- F1
REGIMES = {"S": ("ruido", dict(p=0.7)), "V": ("volatil", dict(h=0.2))}


def run_regimes(make, P, seeds=SEEDS, T=12, block=25, test_blocks=4, skip=5, pretrain=0,
                freeze_control=False, explore=True):
    """Problemas de Harlow con dos regímenes que se alternan en tramos de `block` problemas:
    S (estable y ruidoso: el premiado sale con p=0.7; conviene integrar) y V (volátil y
    determinista: el premiado cambia con h=0.2 tras cada ensayo; conviene quedarse con lo
    último). Medidas, ensayos 2..T: elegir el objeto premiado (no el objetivo ruidoso) y
    que lo más fuerte de la memoria episódica sea el premiado (lo que controla la tasa de
    ligadura, sin el ruido del control). Prueba congelada: `test_blocks` tramos alternados;
    se miden desde el problema `skip` de cada tramo (las señales necesitan unos problemas
    para reflejar el régimen).

    `pretrain`: problemas de Harlow determinista antes (para que el control aprenda a usar
    la memoria). `freeze_control`: después del preentrenamiento, el control deja de
    aprender (lr=0) y responde sin explorar: sólo aprende el meta-controlador."""
    out = dict(online=[], online_mem=[], regime=[], rate=[], params=[], frozen=[], frozen_mem=[],
               frozen_rate=[], seconds=[], noisy=[], noisy_rate=[])
    for seed in seeds:
        rng = np.random.default_rng(3000 + seed)
        n_obj = 2 * (pretrain + P + 2 * test_blocks * block)
        net = make(seed, n_obj + 2)
        sup = Supply(n_obj, rng)
        for _ in range(pretrain):
            harlow_problem(net, rng, sup.take(2), T, "objeto", True)
        if freeze_control:                      # la contracción L1 del control no depende de lr:
            net.lr = 0.0; net.decay = 0.0       # sin decay=0 el control se borraba solo (todo DECIR)
            explore = False
        if hasattr(net, "trace"):
            net.trace = []
        reg = ["S" if (i // block) % 2 == 0 else "V" for i in range(P)]
        acc = np.zeros(P); macc = np.zeros(P); rates = np.zeros(P); t0 = time.time()
        for i in range(P):
            mode, kw = REGIMES[reg[i]]
            _, best, mem = harlow_problem(net, rng, sup.take(2), T, mode, True, explore=explore, **kw)
            acc[i] = np.mean(best[1:]); macc[i] = np.mean(mem[1:])
            rates[i] = net.fast.rate if hasattr(net.fast, "rate") else 1.0
        out["seconds"].append(time.time() - t0)
        tr = getattr(net, "trace", [])[:P]                      # meta-parámetros de cada problema
        last = range(2 * P // 3, len(tr))
        out["params"].append({r: {k: float(np.mean([tr[i][k] for i in last if reg[i] == r]))
                                  for k in (tr[0] if tr else {})} for r in ("S", "V")})
        net.reset(); net.plastic = False
        fr = {"S": [], "V": []}; fm = {"S": [], "V": []}; frr = {"S": [], "V": []}
        for b in range(test_blocks):
            r = "S" if b % 2 == 0 else "V"
            mode, kw = REGIMES[r]
            for j in range(block):
                _, best, mem = harlow_problem(net, rng, sup.take(2), T, mode, True, explore=False, **kw)
                if j >= skip:
                    fr[r].append(np.mean(best[1:])); fm[r].append(np.mean(mem[1:]))
                    frr[r].append(net.fast.rate if hasattr(net.fast, "rate") else 1.0)
        if getattr(net, "meta", None) is not None:
            # la misma prueba congelada, pero el meta-controlador sigue sorteando con ruido (sin
            # aprender): ¿la trampa de lazo cerrado desaparece con la exploración?
            net.meta_noise_frozen = True
            nz = {"S": [], "V": []}; nzr = {"S": [], "V": []}
            for b in range(test_blocks):
                r = "S" if b % 2 == 0 else "V"
                mode, kw = REGIMES[r]
                for j in range(block):
                    _, best, _ = harlow_problem(net, rng, sup.take(2), T, mode, True, explore=False, **kw)
                    if j >= skip:
                        nz[r].append(np.mean(best[1:])); nzr[r].append(net.fast.rate)
            net.meta_noise_frozen = False
            out["noisy"].append({r: float(np.mean(v)) for r, v in nz.items()})
            out["noisy_rate"].append({r: float(np.mean(v)) for r, v in nzr.items()})
        out["online"].append(acc); out["online_mem"].append(macc); out["regime"].append(reg)
        out["rate"].append(rates)
        out["frozen"].append({r: float(np.mean(v)) for r, v in fr.items()})
        out["frozen_mem"].append({r: float(np.mean(v)) for r, v in fm.items()})
        out["frozen_rate"].append({r: float(np.mean(v)) for r, v in frr.items()})
    return out


def report_regimes(log, name, system, res):
    summ = {}
    P = len(res["online"][0])
    last = slice(2 * P // 3, P)

    def by(key, r):
        return [float(np.mean(a[last][np.array(g[last]) == r])) for a, g in zip(res[key], res["regime"])]
    for r in ("S", "V"):
        on, onm, rt = by("online", r), by("online_mem", r), by("rate", r)
        fz = [f[r] for f in res["frozen"]]; fzm = [f[r] for f in res["frozen_mem"]]
        fr = [f[r] for f in res["frozen_rate"]]
        for label, v, what in (("en línea, último tercio", on, "elige el premiado"),
                               ("congelado", fz, "elige el premiado"),
                               ("en línea, último tercio", onm, "memoria da el premiado"),
                               ("congelado", fzm, "memoria da el premiado"),
                               ("en línea, último tercio", rt, "tasa de ligadura"),
                               ("congelado", fr, "tasa de ligadura")):
            log.row(name, system, label, f"{what}, régimen {r}", float(np.mean(v)), float(np.std(v)), len(v))
        summ[r] = dict(online_last_third=on, online_mem_last_third=onm, frozen=fz, frozen_mem=fzm,
                       rate_online=rt, rate_frozen=fr)
    for key, what in (("frozen", "elige el premiado"), ("frozen_mem", "memoria da el premiado")):
        both = [float(np.mean([summ["S"][key][i], summ["V"][key][i]])) for i in range(len(res["frozen"]))]
        log.row(name, system, "congelado", f"{what}, promedio S y V", float(np.mean(both)), float(np.std(both)), len(both))
        summ[key + "_mean_SV"] = both
    if res["noisy"]:
        for r in ("S", "V"):
            v = [f[r] for f in res["noisy"]]; vr = [f[r] for f in res["noisy_rate"]]
            log.row(name, system, "congelado, meta-controlador con ruido", f"elige el premiado, régimen {r}", float(np.mean(v)), float(np.std(v)), len(v))
            log.row(name, system, "congelado, meta-controlador con ruido", f"tasa de ligadura, régimen {r}", float(np.mean(vr)), float(np.std(vr)), len(vr))
            summ[r]["frozen_meta_noise"] = v; summ[r]["rate_frozen_meta_noise"] = vr
        both = [float(np.mean([f["S"], f["V"]])) for f in res["noisy"]]
        log.row(name, system, "congelado, meta-controlador con ruido", "elige el premiado, promedio S y V", float(np.mean(both)), float(np.std(both)), len(both))
        summ["frozen_meta_noise_mean_SV"] = both
    summ["seconds"] = res["seconds"]
    summ["params_last_third"] = res["params"]
    for r in ("S", "V"):
        for k in (res["params"][0][r] if res["params"] else {}):
            if k != "rate":
                v = [p[r][k] for p in res["params"]]
                log.row(name, system, "en línea, último tercio", f"{k}, régimen {r}", float(np.mean(v)), float(np.std(v)), len(v))
    summ["rate_by_problem_mean"] = np.mean(np.array(res["rate"]), 0)
    return summ


def make_meta(**over):
    from bioelastic.metalearning import MetaThinker

    def make(seed, n):
        kw = dict(BASE); kw.update(over)
        return MetaThinker(n, seed=seed, **kw)
    return make


def group_f1():
    from bioelastic.metalearning import PARAMS
    log = CsvLog("metaaprendizaje_f1", HEADER)
    summary = {}
    P = 3000
    default = ("exito", "sorpresa", "td", "novedad")
    plus = default + ("persistencia",)
    # F1a: control preentrenado (100 problemas deterministas) y congelado; sólo aprende el
    # meta-controlador. Aísla si la tasa de ligadura se aprende y se condiciona al régimen.
    conds = [("Thinker actual (FastMemory)", make_base())]
    conds += [(f"tasa fija {a}", make_meta(fixed={"rate": a})) for a in (0.15, 0.3, 0.5, 1.0)]
    conds += [
        ("tasa aprendida, sin señales (sólo sesgo)", make_meta(meta=("rate",), signals=())),
        ("tasa aprendida; señales: éxito, sorpresa, |td|, novedad", make_meta(meta=("rate",), signals=default)),
        ("tasa aprendida; + persistencia de la sorpresa", make_meta(meta=("rate",), signals=plus)),
        ("tasa y olvido aprendidos; + persistencia", make_meta(meta=("rate", "forget"), signals=plus)),
    ]
    for label, make in conds:
        res = run_regimes(make, P, pretrain=100, freeze_control=True)
        summary["F1a " + label] = report_regimes(log, "F1a harlow S/V, control congelado", label, res)
    # F1b: todo aprende a la vez (control con exploración y meta-controlador)
    conds = [("Thinker actual (FastMemory)", make_base()),
             ("tasa fija 0.15", make_meta(fixed={"rate": 0.15})),
             ("tasa fija 1.0", make_meta(fixed={"rate": 1.0})),
             ("tasa aprendida; + persistencia", make_meta(meta=("rate",), signals=plus)),
             ("tasa, olvido y exploración aprendidos; + persistencia",
              make_meta(meta=("rate", "forget", "epsilon"), signals=plus)),
             ("los 7 meta-parámetros aprendidos; + persistencia", make_meta(meta=tuple(PARAMS), signals=plus))]
    for label, make in conds:
        res = run_regimes(make, P, pretrain=100)
        summary["F1b " + label] = report_regimes(log, "F1b harlow S/V, todo aprende", label, res)
    log.close()
    dump("f1", summary)


GROUPS = {"f0": group_f0, "f1": group_f1}


def main():
    for g in sys.argv[1:] or ["f0"]:
        t0 = time.time()
        GROUPS[g]()
        print(f"[{g}] {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
