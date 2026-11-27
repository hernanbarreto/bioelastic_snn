"""Consolidación, corrección y refuerzo del aprendizaje: mecanismos humanos, medidos.

Cada fase implementa (en la librería, como opción apagada por defecto) un mecanismo y lo
mide con al menos 3 semillas, protocolo online «probar y después enseñar» y conjuntos de
prueba separados de los de entrenamiento (los ejemplos de evaluación salen del split de
test de MNIST / Fashion-MNIST; el entrenamiento, del split de entrenamiento).

  f0  línea base: interferencia entre clases parecidas (CD, chainF) y flujo continuo
      reducido (escala 0,1 = 20 000 pasos), con las referencias MLP+repaso existentes.
  f1  dos sistemas explícitos (Complementary Learning Systems).
  f2  esquemas (Tse et al. 2007).
  f3  reconsolidación gobernada por error de predicción (Sevenster et al. 2013/2014).
  f4  repaso como autoexamen (efecto de testeo) y espaciado.
  f5  sueño: consolidación activa + renormalización sináptica global.
  f6  todo junto + ablaciones.

Resultados: results/consolidacion_<fase>.json (configuración, reducciones, filas por
semilla y agregados media ± desvío muestral) y results/consolidacion_<fase>.csv.

  OMP_NUM_THREADS=1 python experiments/exp_consolidation.py f0 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from common import RESULTS, CsvLog
from exp_continual import CortexAgent, build, load_data
from exp_continual import metrics as continual_metrics
from exp_continual import run as continual_run
from exp_interference import SCENARIOS, MID
from exp_vision import encoder

# ---------------------------------------------------------------------- referencias
C2 = "c2"                                   # mejor configuración de la ronda Continuo 2
CONT = "c2+tc=0.1+budget=500"               # la misma, en el flujo reducido (Continuo 2)
MLP = "mlp_replay"                          # MLP 784-256-20 + repaso balanceado (2000)
ERACE = "erace+bytes=8600000+iters=3"       # ER-ACE con reservorio igualado en bytes a c2


# ====================================================================== reanudación
CACHE = RESULTS / "consolidacion_cache"
_PHASE = {"name": "sin_fase"}


def cached(key: dict, compute):
    """Fila ya medida (misma fase y misma clave) o la mide y la agrega al registro
    results/consolidacion_cache/<fase>.jsonl. Así una corrida cortada por un reinicio
    retoma donde quedó sin repetir lo hecho."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{_PHASE['name']}.jsonl"
    k = json.dumps(key, sort_keys=True, default=str)
    if path.exists():
        for line in path.read_text().splitlines():
            d = json.loads(line)
            if d["key"] == k:
                return d["row"]
    row = compute()
    with path.open("a") as f:
        f.write(json.dumps({"key": k, "row": row}, default=float) + "\n")
    return row


# ====================================================================== validación
VAL_N = 500                                  # imágenes por clase reservadas del split de entrenamiento


def split_data(data, val_n: int = VAL_N):
    """{"val": (entrenamiento, validación), "test": (entrenamiento, test)}. La validación
    son `val_n` imágenes por clase del split de entrenamiento (permutación fija, igual para
    todas las semillas) que nunca entran al flujo; el test es el split oficial de test.
    Los hiperparámetros se eligen con "val"; los números que se informan, con "test"."""
    pools, test = data
    train, val = {}, {}
    for c, P in pools.items():
        perm = np.random.default_rng([1234, c]).permutation(len(P))
        val[c] = P[perm[:val_n]]; train[c] = P[perm[val_n:]]
    return {"val": (train, val), "test": (train, test)}


# ====================================================================== utilidades
def agg(rows: list[dict], by: tuple, keys: list[str]) -> list[dict]:
    """Media y desvío muestral (ddof=1) de `keys` por grupo `by`."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r[b] for b in by), []).append(r)
    out = []
    for g, rs in groups.items():
        d = dict(zip(by, g)); d["seeds"] = len(rs)
        for k in keys:
            v = np.array([r.get(k, np.nan) for r in rs], dtype=float)
            v = v[np.isfinite(v)]
            if len(v):
                d[k] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]
        out.append(d)
    return out


def fmt(d: dict, keys) -> str:
    return "  ".join(f"{k}={d[k][0]:.3f}±{d[k][1]:.3f}" for k in keys if k in d)


def save(phase: str, payload: dict, rows: list[dict], keys: list[str]):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"consolidacion_{phase}.json").write_text(json.dumps(payload, indent=1, default=float))
    head = [k for k in ("bench", "scenario", "model", "seed") if any(k in r for r in rows)]
    log = CsvLog(f"consolidacion_{phase}", head + keys)
    for r in rows:
        log._w.writerow([r.get(k, "") for k in head] +
                        [f"{r[k]:.4f}" if isinstance(r.get(k), float) else r.get(k, "") for k in keys])
    log.close()


def _parts(net):
    """(lento, rápido) de la última inferencia de un Cortex: la proyección (y las áreas de
    asociación que no son el almacén) y el área de memoria."""
    if hasattr(net, "parts"):
        p = net.parts()
        return p["slow"], p["fast"]
    Vm = net.heads[0]._last.V
    return net._V - Vm, Vm


# ====================================================================== interferencia
def run_scenario(sc: str, spec: str, seed: int, data, n_eval: int = 100, phases=None,
                 timing: bool = False, on_step=None, factory=None, extra_evals=(), keep_agent=False):
    """Escenario de exp_interference con el mismo flujo y los mismos conjuntos de prueba
    (mismo consumo del generador), registrando además la precisión del sistema lento y del
    rápido por separado y, con `timing`, la latencia de cada paso.

    phases: [(clases, pasos[, mapa de etiquetas])]; el mapa (dict clase -> etiqueta)
            cambia la etiqueta que se enseña y la que se evalúa desde esa fase."""
    pools, test = data
    enc = encoder()
    phases = phases or SCENARIOS[sc]
    classes = sorted({c for ph in phases for c in ph[0]})
    rng = np.random.default_rng([seed, 4242])
    ev = {c: [enc(test[c][i].astype(np.float32) / 255)
              for i in rng.choice(len(test[c]), n_eval, replace=False)] for c in classes}
    agent = (factory or build(spec))(seed)
    agent.on_phase(0)
    net = agent.net if isinstance(agent, CortexAgent) else None
    label = {c: c for c in classes}

    def evaluate():
        full, slow, fast = {}, {}, {}
        diag = []                                   # (margen del lento, lento ok, rápido ok, todo ok, consultó)
        for c, S in ev.items():
            y = label[c]; hf, hs, hq = [], [], []
            for s in S:
                V = agent.scores(s)
                hf.append(int(np.argmax(V)) == y)
                if net is not None:
                    Vs, Vq = _parts(net)
                    hs.append(int(np.argmax(Vs)) == y); hq.append(int(np.argmax(Vq)) == y)
                    e = np.exp((Vs - Vs.max()) / net.temperature); q = np.sort(e / e.sum())
                    diag.append((q[-1] - q[-2], hs[-1], hq[-1], hf[-1], bool(getattr(net, "_recalled", False))))
            full[c] = float(np.mean(hf))
            if hs:
                slow[c] = float(np.mean(hs)); fast[c] = float(np.mean(hq))
        return dict(full=full, slow=slow, fast=fast, diag=diag)

    hist = []
    extra = {}
    hits = []
    lat_f, lat_t = [], []
    t0 = time.time()
    step = 0
    for k, ph in enumerate(phases):
        cls, steps = ph[0], ph[1]
        if len(ph) > 2 and isinstance(ph[2], dict):
            label.update(ph[2])
        for t in range(1, steps + 1):
            c = int(rng.choice(cls))
            i = int(rng.integers(len(pools[c])))
            s = enc(pools[c][i].astype(np.float32) / 255)
            if net is not None and timing:
                a = time.perf_counter(); pred = net.forward(s); b = time.perf_counter()
                net.teach(label[c]); e = time.perf_counter()
                lat_f.append(b - a); lat_t.append(e - b)
            else:
                pred = agent.sup(s, label[c])
            hits.append(int(pred) == label[c])
            step += 1
            if on_step is not None:
                on_step(agent, k, t, step)
            if step in extra_evals:
                extra[step] = evaluate()
            if t == MID and k > 0:
                hist.append((k, t, evaluate()))
        hist.append((k, steps, evaluate()))
    t_last = sum(p[1] for p in phases[:-1])
    out = dict(hist=hist, stats=agent.stats(), seconds=time.time() - t0, extra=extra,
               preq_last500=float(np.mean(hits[t_last:t_last + 500])))
    if net is not None:
        out["counters"] = counters(net)
        out["causes"] = {k: int(sum(getattr(h, k, 0) for h in net.heads))
                         for k in ("relabeled", "protected", "new_causes", "superseded")}
    if timing:
        out["latency"] = latency(lat_f, lat_t)
    if keep_agent:
        out["agent"] = agent
    return out


def counters(net) -> dict:
    keys = ("replays", "rehearsals", "directed", "recalls", "assimilated", "labile", "sleeps",
            "sleep_replays", "tests", "test_errors", "released")
    return {k: int(getattr(net, k)) for k in keys if hasattr(net, k)}


def latency(f, t) -> dict:
    f = np.array(f) * 1e3; t = np.array(t) * 1e3; s = f + t
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")
    return dict(forward_p50=q(f, 50), forward_p99=q(f, 99), forward_max=float(f.max()) if len(f) else np.nan,
                teach_p50=q(t, 50), teach_p99=q(t, 99), teach_max=float(t.max()) if len(t) else np.nan,
                step_p50=q(s, 50), step_p99=q(s, 99), step_max=float(s.max()) if len(s) else np.nan,
                step_mean=float(s.mean()) if len(s) else np.nan)


def summarize(phases, hist, part: str = "full") -> dict:
    """old / new / all / min / drop / mid_old como en exp_interference, para una parte."""
    final = hist[-1][2][part]
    if not final:
        return {}
    first = set(phases[0][0]); last = set(phases[-1][0])
    m = dict(old=float(np.mean([final[c] for c in first])),
             new=float(np.mean([final[c] for c in last])),
             all=float(np.mean(list(final.values()))), min=float(min(final.values())))
    prev = {c: max(h[2][part][c] for h in hist[:-1]) for c in final if c not in last} if len(hist) > 1 else {}
    m["drop"] = float(np.mean([prev[c] - final[c] for c in prev])) if prev else 0.0
    m["mid_old"] = float(np.mean([hist[1][2][part][c] for c in first])) if len(hist) > 2 else float("nan")
    return m


def scenario_row(sc, spec, seed, r, phases=None) -> dict:
    phases = phases or SCENARIOS[sc]
    row = dict(bench="interf", scenario=sc, model=spec, seed=seed)
    row.update(summarize(phases, r["hist"]))
    for part in ("slow", "fast"):
        row.update({f"{part}_{k}": v for k, v in summarize(phases, r["hist"], part).items()
                    if k in ("old", "new", "all", "min")})
    diag = r["hist"][-1][2].get("diag")
    if diag:                                        # ¿la inseguridad del lento señala sus errores?
        d = np.array(diag, dtype=float)
        uns = d[:, 0] < 0.2
        row["unsure_frac"] = float(uns.mean())
        row["slow_acc_sure"] = float(d[~uns, 1].mean()) if (~uns).any() else float("nan")
        row["slow_acc_unsure"] = float(d[uns, 1].mean()) if uns.any() else float("nan")
        row["fast_acc_unsure"] = float(d[uns, 2].mean()) if uns.any() else float("nan")
        row["recall_frac"] = float(d[:, 4].mean())
    row["preq_last500"] = r.get("preq_last500", float("nan"))
    row.update(neurons=r["stats"]["neurons"], bytes=r["stats"]["bytes"], seconds=r["seconds"])
    row.update(r.get("counters", {}))
    row.update({f"lat_{k}": v for k, v in r.get("latency", {}).items()})
    return row


INTERF_KEYS = ["old", "new", "all", "min", "drop", "mid_old", "slow_old", "slow_new", "slow_all",
               "fast_old", "fast_new", "fast_all", "neurons", "bytes", "seconds", "replays"]


def interference(scenarios, specs, seeds, data, *, tag: str, log_rows: list, phases=None, split="full", **kw):
    for sc in scenarios:
        for spec in specs:
            for seed in seeds:
                key = dict(b="interf", sc=sc, spec=spec, seed=seed, split=split, phases=str(phases),
                           kw=str(sorted(kw.items())))
                row = cached(key, lambda sc=sc, spec=spec, seed=seed: scenario_row(
                    sc, spec, seed, run_scenario(sc, spec, seed, data, phases=phases, **kw), phases))
                log_rows.append(row)
                print(f"[{tag}] {sc} {spec} s{seed}: " +
                      " ".join(f"{k}={row[k]:.3f}" for k in ("old", "new", "all", "min", "slow_all", "fast_all")
                               if isinstance(row.get(k), float)) + f" {row['seconds']:.0f}s", flush=True)


# ====================================================================== flujo continuo
CONT_KEYS = ["final_clean_known", "preprobe_clean_known", "final_rot90", "final_occlude", "final_noise",
             "forget_clean", "forget_rot90", "forget_occlude", "savings", "fwt", "fwt_restricted",
             "probe_test_trained", "probe_test_fresh", "plasticity_ratio", "preq_A_reward",
             "preq_all_reward_noise", "sim_after_C", "sim_after_D", "sim_drop_D", "C_after_D",
             "D_after_D", "sim_final", "neurons_end", "bytes_end", "train_seconds"]


def continual(specs, seeds, data, *, scale: float, n_eval: int, log_rows: list, tag: str, split="full"):
    for spec in specs:
        for seed in seeds:
            def go(spec=spec, seed=seed):
                m = continual_metrics(continual_run(spec, seed, scale, n_eval, data))
                row = dict(bench="continual", scenario=f"x{scale:g}", model=spec, seed=seed)
                row.update({k: float(m[k]) for k in CONT_KEYS if k in m})
                return row
            row = cached(dict(b="continual", spec=spec, seed=seed, scale=scale, n_eval=n_eval, split=split), go)
            log_rows.append(row)
            print(f"[{tag}] continual {spec} s{seed}: " + fmt({k: [row[k], 0] for k in CONT_KEYS[:6] if k in row},
                                                              CONT_KEYS[:6]), flush=True)


def finish(phase: str, rows: list, extra: dict, cfg: dict):
    keys = sorted({k for r in rows for k in r if k not in ("bench", "scenario", "model", "seed")},
                  key=lambda k: (INTERF_KEYS + CONT_KEYS + [k]).index(k))
    keys = [k for k in keys if any(isinstance(r.get(k), (int, float)) for r in rows)]
    summary = agg(rows, ("bench", "scenario", "model"), keys)
    for d in summary:
        print(f"  {d['bench']:9s} {d['scenario']:7s} {d['model']:60s} n={d['seeds']} "
              + fmt(d, ["all", "old", "new", "min", "slow_all", "final_clean_known", "forget_clean",
                        "sim_after_D", "plasticity_ratio"]), flush=True)
    save(phase, dict(phase=phase, config=cfg, summary=summary, rows=rows, **extra), rows, keys)


# ====================================================================== fases
def phase_f0(a, data):
    rows: list = []
    interference(("CD", "chainF"), (C2, "d2", MLP, ERACE), a.seeds, data, tag="f0", log_rows=rows)
    continual((CONT, MLP), a.seeds, data, scale=0.1, n_eval=30, log_rows=rows, tag="f0")
    cfg = dict(seeds=a.seeds, interference=dict(scenarios={k: SCENARIOS[k] for k in ("CD", "chainF")},
                                                n_eval_per_class=100, mid_eval=MID),
               continual=dict(scale=0.1, steps=20000, n_eval_per_class=30),
               reductions="flujo continuo a escala 0,1 (20 000 pasos de 200 000) con constantes de "
                          "tiempo comprimidas (tc=0.1) y presupuesto de 500 neuronas, como en Continuo 2; "
                          "evaluación con 30 imágenes de test por clase y contexto")
    finish("f0", rows, {}, cfg)


STORED = "+top.stored_labels=true"
GATE = "+top.recall=gate"


def phase_f1(a, data):
    """Dos sistemas explícitos. El almacén rápido es el área de memoria con huella (ya era
    de un solo ensayo, finita y con olvido); se le agregan la etiqueta, el paso y el
    contexto de cada episodio, la recuperación por coseno de varios episodios y la
    compuerta por inseguridad del sistema lento."""
    rows: list = []
    specs = [C2, C2 + STORED, C2 + STORED + GATE, C2 + STORED + GATE + "+top.recall_margin=0.5",
             C2 + STORED + "+top.recall=always", C2 + STORED + GATE + "+top.recall_source=readout",
             C2 + GATE]
    interference(("CD",), specs, a.seeds, data, tag="f1", log_rows=rows, timing=True)
    continual((CONT + STORED + GATE,), a.seeds, data, scale=0.1, n_eval=30, log_rows=rows, tag="f1")
    cfg = dict(seeds=a.seeds, scenario=SCENARIOS["CD"], n_eval_per_class=100,
               continual=dict(scale=0.1, base=CONT, compare_with="consolidacion_f0.json"),
               unsure_definition="margen p1-p2 del sistema lento (softmax a T=0,25) < 0,2 en la última evaluación",
               reductions="sólo CD en interferencia (chainF queda para f6); flujo continuo a escala 0,1")
    finish("f1", rows, {}, cfg)


OWN = "+top.slow_error=own"


def phase_f1b(a, data):
    """Complemento de f1: el sistema lento aprende con su propio error (sin que la
    memoria le «explique» el ejemplo), solo y con la compuerta."""
    rows: list = []
    specs = [C2 + STORED + OWN, C2 + STORED + OWN + GATE + "+top.recall_source=readout",
             C2 + STORED + OWN + GATE]
    interference(("CD",), specs, a.seeds, data, tag="f1b", log_rows=rows, timing=True)
    continual((CONT + STORED + OWN,), a.seeds, data, scale=0.1, n_eval=30, log_rows=rows, tag="f1b")
    cfg = dict(seeds=a.seeds, scenario=SCENARIOS["CD"], n_eval_per_class=100,
               continual=dict(scale=0.1, base=CONT, compare_with="consolidacion_f0.json"),
               note="slow_error=own: la proyección aprende con p del sistema lento solo",
               reductions="sólo CD en interferencia; flujo continuo a escala 0,1")
    finish("f1b", rows, {}, cfg)


def tse_run(seed: int, data, prior: str, kind: str, modes=("measure", "gate", "always"),
            n_prior: int = 3000, session: int = 200, n_items: int = 5, trials: int = 3,
            spec: str = C2 + STORED + "+top.schema=measure"):
    """Prueba tipo Tse et al. (2007) sobre dígitos.

    Esquema: 3000 pasos con los dígitos 0-7. «consistent»: etiquetas verdaderas;
    «inconsistent»: la asignación imagen -> etiqueta se permuta al azar entre 0-7 cada
    `session` pasos (como la arena donde los pares sabor-lugar cambian).
    Ítems nuevos (`kind`): «new»: 5 imágenes de entrenamiento de cada dígito nuevo (8 y 9)
    con su etiqueta; «conflict» (control de McClelland 2013): 5 imágenes de 3 etiquetadas
    8 y 5 de 5 etiquetadas 9, que contradicen lo que el sistema ya sabe. Cada ensayo
    presenta los 10 ítems una vez. La red después del esquema se copia para cada modo
    (mismo estado previo): measure (= sin mecanismo), gate, always.

    Después de cada ensayo: acierto en los propios ítems y en imágenes de test de las
    clases nuevas (o de 3 y 5 con su etiqueta verdadera para «conflict»), del sistema
    completo y del lento solo (la «lesión» del almacén rápido), y acierto en lo viejo."""
    import copy
    pools, test = data
    enc = encoder()
    rng = np.random.default_rng([seed, 777])
    OLD, NEW = list(range(8)), [8, 9]
    ev_old = {c: [enc(test[c][i].astype(np.float32) / 255) for i in rng.choice(len(test[c]), 50, replace=False)]
              for c in OLD}
    ev_new = {c: [enc(test[c][i].astype(np.float32) / 255) for i in rng.choice(len(test[c]), 100, replace=False)]
              for c in NEW}
    src = {8: 8, 9: 9} if kind == "new" else {8: 3, 9: 5}
    items = [(enc(pools[src[y]][i].astype(np.float32) / 255), y)
             for y in NEW for i in rng.choice(len(pools[src[y]]), n_items, replace=False)]
    agent = build(spec)(seed)
    net = agent.net
    perm = np.arange(10)
    t0 = time.time()
    for t in range(n_prior):
        if prior == "inconsistent" and t % session == 0:
            perm[OLD] = rng.permutation(OLD)
        c = int(rng.choice(OLD)); i = int(rng.integers(len(pools[c])))
        agent.sup(enc(pools[c][i].astype(np.float32) / 255), int(perm[c]))
    prior_s = time.time() - t0
    rel = net.reliability

    def acc(model, S, y):
        hf, hs = [], []
        for s in S:
            hf.append(int(np.argmax(model.scores(s))) == y)
            hs.append(int(np.argmax(model.net.parts()["slow"])) == y)
        return float(np.mean(hf)), float(np.mean(hs))

    def evaluate(model):
        out = {}
        f, s = zip(*[acc(model, [x], y) for x, y in items])
        out["items_full"], out["items_slow"] = float(np.mean(f)), float(np.mean(s))
        if kind == "new":
            r = [acc(model, ev_new[c], c) for c in NEW]
        else:                                        # 3 y 5 con su etiqueta verdadera
            r = [acc(model, ev_old[c], c) for c in (3, 5)]
        out["probe_full"], out["probe_slow"] = (float(np.mean(v)) for v in zip(*r))
        r = [acc(model, ev_old[c], c) for c in OLD]
        out["old_full"], out["old_slow"] = (float(np.mean(v)) for v in zip(*r))
        return out

    base = evaluate(agent)
    res = []
    order_rng = np.random.default_rng([seed, 778])
    orders = [order_rng.permutation(len(items)) for _ in range(trials)]
    for mode in modes:
        a = copy.deepcopy(agent)
        a.net.schema = mode
        n0 = a.net.heads[0].layer.n; as0 = a.net.assimilated
        fits, parts = [], []
        for k in range(trials):
            for j in orders[k]:
                x, y = items[j]
                a.sup(x, y)
                fits.append(a.net.fit); parts.append(a.net.fit_parts)
            row = dict(bench="tse", scenario=f"{prior}/{kind}", model=mode, seed=seed, trial=k + 1,
                       reliability=rel, prior_seconds=prior_s,
                       neurons_added=a.net.heads[0].layer.n - n0, assimilated=a.net.assimilated - as0,
                       fit_mean=float(np.nanmean(fits)), cov_mean=float(np.nanmean([p[0] for p in parts])),
                       conflict_mean=float(np.nanmean([p[2] for p in parts])),
                       **evaluate(a))
            for key in ("old_full", "old_slow", "probe_full", "probe_slow"):
                row[f"d_{key}"] = row[key] - base[key]
            res.append(row)
            print(f"[f2] tse {prior}/{kind} {mode} s{seed} ensayo {k + 1}: " +
                  " ".join(f"{key}={row[key]:.3f}" for key in ("items_full", "items_slow", "probe_full",
                                                                "probe_slow", "old_full", "fit_mean")) +
                  f" asim={row['assimilated']} +n={row['neurons_added']}", flush=True)
    return res


TSE_KEYS = ["items_full", "items_slow", "probe_full", "probe_slow", "old_full", "old_slow", "d_old_full",
            "d_old_slow", "d_probe_full", "d_probe_slow", "reliability", "fit_mean", "cov_mean",
            "conflict_mean", "assimilated", "neurons_added"]


def phase_f2(a, data):
    rows: list = []
    for seed in a.seeds:
        for prior, kind in (("consistent", "new"), ("inconsistent", "new"), ("consistent", "conflict")):
            rows += tse_run(seed, data, prior, kind)
    tse = agg(rows, ("bench", "scenario", "model", "trial"), TSE_KEYS)
    for d in tse:
        print(f"  {d['scenario']:22s} {d['model']:8s} ensayo {d['trial']} " +
              fmt(d, ["items_full", "items_slow", "probe_full", "probe_slow", "d_old_full", "d_old_slow",
                      "reliability", "fit_mean", "assimilated", "neurons_added"]), flush=True)
    irows: list = []
    specs = [C2 + STORED + "+top.schema=measure", C2 + STORED + "+top.schema=gate",
             C2 + STORED + "+top.schema=gate+top.schema_misfit_lr=0", C2 + STORED + "+top.schema=always",
             C2 + STORED + "+head.separation_relax=200"]     # separación adaptativa (Continuo 2)
    interference(("CD",), specs, a.seeds, data, tag="f2", log_rows=irows)
    cfg = dict(seeds=a.seeds,
               tse=dict(prior_steps=3000, session=200, old=list(range(8)), new=[8, 9], items_per_class=5,
                        trials=3, conflict_sources={8: 3, 9: 5}, eval_old_per_class=50, eval_new_per_class=100,
                        spec=C2 + STORED + "+top.schema=measure", schema_fit=0.5, schema_boost=4.0,
                        note="el estado tras el esquema se copia para cada modo (comparación pareada)"),
               interference=dict(scenario=SCENARIOS["CD"]),
               adaptive_separation="head.separation_relax=200: margen x 200 / (200 + usos de la clase)",
               reductions="sólo CD en interferencia; dígitos 0-7 / 8-9 para la prueba tipo Tse")
    finish("f2", irows, dict(tse_rows=rows, tse_summary=tse), cfg)


DRIFT = [((10, 11, 12, 13), 3000), ((10, 11, 12, 13), 1500, {10: 12, 12: 10})]
# caso adversarial inverso: una clase conocida pasa a llamarse con un nombre recién aparecido
RENAME = [((10, 11, 12, 13), 3000), ((10, 11, 12, 13), 1500, {10: 18})]
RECON = {                                    # lectura de la memoria / proyección
    "fija/nada (c2)": C2,
    "nada/nada": C2 + "+no_consol",
    "recon/nada": C2 + "+readout.recon_lock=20",
    "fija/fija": C2 + "+proj.consolidation=5",
    "fija/recon": C2 + "+proj.recon_lock=50",
    "recon/recon+proto": C2 + "+readout.recon_lock=20+proj.recon_lock=50+head.recon_lock=20",
}


def recon_counters(net) -> dict:
    rs = [h.readout for h in net.heads] + ([net.proj] if net.proj is not None else [])
    return dict(opened=sum(r.opened for r in rs), relearned=sum(r.relearned for r in rs),
                blocked=sum(r.blocked for r in rs), locked_skips=sum(getattr(h, "locked_skips", 0) for h in net.heads),
                proj_locked_frac=float((net.proj.count[:net.proj.n] >= net.proj.recon_lock).mean())
                if net.proj is not None and net.proj.recon_lock > 0 else float("nan"))


def drift_row(spec, seed, r, changed=(10, 12), kept=(11, 13), scenario="swap10_12") -> dict:
    """Cambio de etiqueta (10 <-> 12, o 10 -> 18 con `changed=(10,)`) después de aprender C:
    ¿se actualiza lo que cambió sin destruir lo vecino?"""
    h0 = r["hist"][0][2]; hm = r["hist"][1][2]; hf = r["hist"][-1][2]
    row = dict(bench="drift", scenario=scenario, model=spec, seed=seed)
    for part in ("full", "slow", "fast"):
        if not hf[part]:
            continue
        row[f"{part}_swapped_end"] = float(np.mean([hf[part][c] for c in changed]))
        row[f"{part}_swapped_mid"] = float(np.mean([hm[part][c] for c in changed]))
        row[f"{part}_kept_end"] = float(np.mean([hf[part][c] for c in kept]))
        row[f"{part}_kept_drop"] = float(np.mean([h0[part][c] - hf[part][c] for c in kept]))
        row[f"{part}_before"] = float(np.mean(list(h0[part].values())))
    row.update(r.get("counters", {}), seconds=r["seconds"])
    return row


def phase_f3(a, data):
    rows: list = []
    extra = {}
    for name, spec in RECON.items():
        for seed in a.seeds:
            res = {}

            def grab(agent, k, t, step, res=res):
                if step == SCEN_STEPS["CD"]:
                    res.update(recon_counters(agent.net))
            r = run_scenario("CD", spec, seed, data, on_step=grab)
            row = scenario_row("CD", spec, seed, r); row["variant"] = name; row.update(res)
            rows.append(row)
            print(f"[f3] CD {name} s{seed}: " + " ".join(f"{k}={row[k]:.3f}" for k in
                  ("old", "new", "all", "min", "slow_all", "fast_all")) + f" abiertas={row.get('opened')}", flush=True)
    drows: list = []
    for name, spec in RECON.items():
        for seed in a.seeds:
            res = {}

            def grab(agent, k, t, step, res=res):
                if step == 4500:
                    res.update(recon_counters(agent.net))
            r = run_scenario("drift", spec, seed, data, phases=DRIFT, on_step=grab)
            row = drift_row(spec, seed, r); row["variant"] = name; row.update(res)
            drows.append(row)
            print(f"[f3] drift {name} s{seed}: " + " ".join(f"{k}={row[k]:.3f}" for k in
                  ("full_swapped_mid", "full_swapped_end", "full_kept_end", "full_kept_drop", "slow_swapped_end"))
                  + f" abiertas={row.get('opened')}", flush=True)
    dkeys = sorted({k for r in drows for k in r if isinstance(r[k], (int, float)) and k != "seed"})
    extra["drift_rows"] = drows
    extra["drift_summary"] = agg(drows, ("bench", "scenario", "model"), dkeys)
    for d in extra["drift_summary"]:
        print(f"  drift {d['model']:70s} " + fmt(d, ["full_swapped_mid", "full_swapped_end", "full_kept_end",
                                                      "full_kept_drop", "slow_swapped_end", "slow_kept_drop"]))
    cfg = dict(seeds=a.seeds, variants=RECON, interference=SCENARIOS["CD"], drift=DRIFT,
               recon=dict(readout_lock=20, proj_lock=50, proto_lock=20, window=1, kappa=2.0, beta=0.01, rate=1.0),
               note="«fija» = consolidación por conteo actual (consolidation=5); «nada» = consolidation=0; "
                    "umbrales de bloqueo elegidos a priori, sin barrido",
               reductions="sólo CD en interferencia; el cambio de etiqueta usa las clases de C")
    finish("f3", rows, extra, cfg)


UNI = "+top.replay_focus=uniform"
EQUAL = "+top.replay_mode=active+top.spacing=equal+top.space_base=200"
EXPAND = "+top.replay_mode=active+top.spacing=expanding+top.space_base=86+top.space_factor=2"
REVIEW = {
    "pasivo uniforme": C2 + STORED + UNI,
    "pasivo c2 (confusión)": C2 + STORED,
    "activo sin espaciado": C2 + STORED + UNI + "+top.replay_mode=active",
    "activo iguales": C2 + STORED + UNI + EQUAL,
    "activo expansivos": C2 + STORED + UNI + EXPAND,
    "agendado expansivo (aprende siempre)": C2 + STORED + UNI + EXPAND.replace("=active", "=scheduled"),
    "activo expansivos + confusión": C2 + STORED + EXPAND,
    "contraste dirigido": C2 + STORED + "+top.replay_contrast=true",
    "contraste + activos expansivos": C2 + STORED + EXPAND + "+top.replay_contrast=true",
}
CDM = [((10, 11, 12, 13), 3000), ((14, 15, 16, 17), 3000), ((0, 1, 2, 3, 4), 3000)]


def cdm_row(spec, seed, r) -> dict:
    """Retención larga: C y D después de 3000 pasos de dígitos sin repaso del flujo."""
    row = dict(bench="retention", scenario="CDM", model=spec, seed=seed)
    hD = r["hist"][-3][2]; hf = r["hist"][-1][2]        # fin de D, fin de M
    for part in ("full", "slow", "fast"):
        if not hf[part]:
            continue
        cd = (10, 11, 12, 13, 14, 15, 16, 17)
        row[f"{part}_CD_end"] = float(np.mean([hf[part][c] for c in cd]))
        row[f"{part}_C_end"] = float(np.mean([hf[part][c] for c in cd[:4]]))
        row[f"{part}_CD_afterD"] = float(np.mean([hD[part][c] for c in cd]))
        row[f"{part}_CD_loss"] = row[f"{part}_CD_afterD"] - row[f"{part}_CD_end"]
        row[f"{part}_M_end"] = float(np.mean([hf[part][c] for c in range(5)]))
    row.update(r.get("counters", {}), seconds=r["seconds"])
    return row


def phase_f4(a, data):
    rows: list = []
    for name, spec in REVIEW.items():
        for seed in a.seeds:
            r = run_scenario("CD", spec, seed, data)
            row = scenario_row("CD", spec, seed, r); row["variant"] = name
            rows.append(row)
            print(f"[f4] CD {name} s{seed}: " + " ".join(f"{k}={row[k]:.3f}" for k in
                  ("old", "new", "all", "min", "slow_all", "fast_all")) +
                  f" repasos={row.get('replays')} exámenes={row.get('tests')} fallos={row.get('test_errors')}", flush=True)
    for name in ("pasivo c2 (confusión)", "activo expansivos + confusión", "contraste dirigido"):
        for seed in a.seeds:                         # clases que llegan de a una (propuesta de Continuo 2)
            r = run_scenario("chainF", REVIEW[name], seed, data)
            row = scenario_row("chainF", REVIEW[name], seed, r); row["variant"] = name
            rows.append(row)
            print(f"[f4] chainF {name} s{seed}: " + " ".join(f"{k}={row[k]:.3f}" for k in
                  ("old", "new", "all", "min", "slow_all", "slow_old")), flush=True)
    lrows: list = []
    for name in ("pasivo uniforme", "activo sin espaciado", "activo iguales", "activo expansivos"):
        for seed in a.seeds:
            r = run_scenario("CDM", REVIEW[name], seed, data, phases=CDM)
            row = cdm_row(REVIEW[name], seed, r); row["variant"] = name
            lrows.append(row)
            print(f"[f4] CDM {name} s{seed}: " + " ".join(f"{k}={row[k]:.3f}" for k in
                  ("full_CD_afterD", "full_CD_end", "full_CD_loss", "slow_CD_end", "full_M_end")), flush=True)
    lkeys = sorted({k for r in lrows for k in r if isinstance(r[k], (int, float)) and k != "seed"})
    lsum = agg(lrows, ("bench", "scenario", "model"), lkeys)
    for d in lsum:
        print(f"  CDM {d['model']:90s} " + fmt(d, ["full_CD_afterD", "full_CD_end", "full_CD_loss",
                                                   "slow_CD_end", "slow_CD_loss", "full_M_end"]))
    cfg = dict(seeds=a.seeds, variants=REVIEW, interference=SCENARIOS["CD"], retention=CDM,
               budget="replay=2 repasos por paso en todas las variantes (mismo número de repasos)",
               spacing=dict(equal=dict(gap=200), expanding=dict(first=86, factor=2),
                            note="86 + 172 + 344 ~ 3 x 200: mismo retraso total hasta el tercer examen, "
                                 "como 1-5-9 vs 5-5-5 en Karpicke y Roediger 2007"),
               contrast="replay_contrast: el repaso dirigido a la confusión se vuelve un examen de a dos "
                        "(clase guardada contra la que se enseña) y aprende sólo si falla",
               reductions="CD; chainF sólo para 3 variantes; CDM (retención tras 3000 pasos de dígitos) para 4")
    finish("f4", rows, dict(retention_rows=lrows, retention_summary=lsum), cfg)


SLEEP = "+top.replay=1+top.sleep_every=500+top.sleep_replays=500"
MICRO = SLEEP + "+top.sleep_micro=2"
SLEEPS = {
    "despierto (replay=2)": C2 + STORED,
    "sueño en bloque": C2 + STORED + SLEEP,
    "sueño en micro-fases": C2 + STORED + MICRO,
    "micro + renormalización": C2 + STORED + MICRO + "+top.sleep_downscale=0.1",
    "micro + renorm. + poda": C2 + STORED + MICRO + "+top.sleep_downscale=0.1+top.sleep_prune=0.01",
    "micro + renorm. + poda + liberación": C2 + STORED + MICRO +
        "+top.sleep_downscale=0.1+top.sleep_prune=0.01+top.sleep_release=0.5",
    "por carga (250 episodios) + renorm.": C2 + STORED +
        "+top.replay=1+top.sleep_trigger=load+top.sleep_load=250+top.sleep_replays=500+top.sleep_micro=2"
        "+top.sleep_downscale=0.1",
}


def phase_f5(a, data):
    rows: list = []
    for name, spec in SLEEPS.items():
        for seed in a.seeds:
            r = run_scenario("CD", spec, seed, data, timing=True, extra_evals=(2999,))
            row = scenario_row("CD", spec, seed, r); row["variant"] = name
            pre = r["extra"][2999]["full"]; post = r["hist"][0][2]["full"]
            row["C_pre3000"] = float(np.mean([pre[c] for c in (10, 11, 12, 13)]))
            row["C_post3000"] = float(np.mean([post[c] for c in (10, 11, 12, 13)]))
            row["gain_3000"] = row["C_post3000"] - row["C_pre3000"]
            pre_s = r["extra"][2999]["slow"]; post_s = r["hist"][0][2]["slow"]
            row["slow_gain_3000"] = float(np.mean([post_s[c] - pre_s[c] for c in (10, 11, 12, 13)]))
            rows.append(row)
            print(f"[f5] CD {name} s{seed}: " + " ".join(f"{k}={row[k]:.3f}" for k in
                  ("old", "new", "all", "slow_all", "gain_3000", "preq_last500", "lat_step_p50",
                   "lat_step_p99", "lat_step_max")) +
                  f" sueños={row.get('sleeps')} rep_sueño={row.get('sleep_replays')} liberados={row.get('released')}",
                  flush=True)
    for name, spec in (("despierto, presupuesto 600", C2 + STORED + "+budget=600"),
                       ("sueño micro + renorm. + liberación, presupuesto 600", C2 + STORED + "+budget=600" + MICRO +
                        "+top.sleep_downscale=0.1+top.sleep_release=0.5")):
        for seed in a.seeds:
            r = run_scenario("CD", spec, seed, data)
            row = scenario_row("CD", spec, seed, r); row["variant"] = name
            rows.append(row)
            print(f"[f5] CD {name} s{seed}: " + " ".join(f"{k}={row[k]:.3f}" for k in
                  ("old", "new", "all", "slow_all", "preq_last500")) + f" liberados={row.get('released')}", flush=True)
    continual((CONT + STORED + MICRO + "+top.sleep_downscale=0.1",), a.seeds, data, scale=0.1, n_eval=30,
              log_rows=rows, tag="f5")
    cfg = dict(seeds=a.seeds, variants=SLEEPS, interference=SCENARIOS["CD"],
               budget="12 000 repasos en CD en todas las variantes: despierto 2 por paso; con sueño 1 por "
                      "paso despierto + 500 cada 500 pasos (12 sueños), salvo «por carga» (variable)",
               gain="acierto en C justo antes (paso 2999) y después (paso 3000) del sueño del paso 3000",
               capacity="preq_last500: acierto prequential en los primeros 500 pasos de D",
               latency="ms por paso (forward + teach), percentiles; con 4 agentes compartiendo 2 CPU: "
                       "comparar sólo dentro de esta tabla",
               reductions="CD con 3 semillas; flujo continuo a escala 0,1 para la variante micro + renorm.")
    finish("f5", rows, {}, cfg)


MECH = {        # variante de cada mecanismo que entra en «todo»: la mejor medida en su fase
    "F1": STORED + OWN + GATE + "+top.recall_source=readout",     # f1b: 0,762
    "F2": "+top.schema=gate",                                     # f2: 0,744
    "F3": "+proj.recon_lock=50",                                  # f3: 0,764
    "F4": UNI + EQUAL + "+top.replay_contrast=true",              # f4: iguales 0,776 + contraste
    "F5": MICRO + "+top.sleep_downscale=0.1",                     # f5
}
BEST = C2 + STORED + OWN + "+top.replay_contrast=true"            # sólo lo que no empeoró solo
CHOICE = ("cada mecanismo en la variante con mayor «all» en CD de su fase, aunque empeore "
          "solo; «mejores» junta sólo los que no empeoraron solos")


def f6_spec(base: str, drop: str | None = None) -> str:
    spec = base
    for k, m in MECH.items():
        if k != drop:
            spec += m
    return spec


def phase_f6(a, data):
    """Todos los mecanismos a la vez y ablaciones quitando uno por vez (CD), y «todo» y
    «mejores» en el flujo continuo reducido. Sin F5 el repaso despierto vuelve a 2 por
    paso (mismo presupuesto)."""
    rows: list = []
    variants = {"todo": f6_spec(C2)}
    for k in MECH:
        variants[f"todo - {k}"] = f6_spec(C2, k)
    variants["mejores"] = BEST
    for name, spec in variants.items():
        for seed in a.seeds:
            row = cached(dict(b="interf", sc="CD", spec=spec, seed=seed, split="full", timing=True),
                         lambda spec=spec, seed=seed: scenario_row(
                             "CD", spec, seed, run_scenario("CD", spec, seed, data, timing=True)))
            row = dict(row, variant=name)
            rows.append(row)
            print(f"[f6] CD {name} s{seed}: " + " ".join(f"{k}={row[k]:.3f}" for k in
                  ("old", "new", "all", "min", "slow_all", "fast_all", "lat_step_p99") if k in row), flush=True)
    cont = [variants["todo"].replace(C2, CONT, 1), BEST.replace(C2, CONT, 1)]
    continual(cont, a.seeds, data, scale=0.1, n_eval=30, log_rows=rows, tag="f6")
    cfg = dict(seeds=a.seeds, mechanisms=MECH, best=BEST, choice=CHOICE, variants=variants,
               reference="c2 y referencias en consolidacion_f0.json (mismos flujos y semillas)",
               reductions="CD para todo, las 5 ablaciones y «mejores»; flujo continuo x0,1 sólo para "
                          "todo y «mejores» (los mecanismos se rehacen en f7)")
    finish("f6", rows, {}, cfg)


# ====================================================================== referencia externa
SPLIT = {"mnist": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)],
         "fashion": [(10, 11), (12, 13), (14, 15), (16, 17), (18, 19)]}
PUBLISHED = {
    "mnist": dict(source="van de Ven, Tuytelaars y Tolias 2022, Nat. Mach. Intell. 4:1185, tabla 2 (Class-IL)",
                  protocol="MLP 2x400, 2000 iteraciones x lote 128 por contexto (~21 pasadas por tarea, no "
                           "un solo pase); repaso con 100 ejemplos por clase; 20 semillas; media ± e.e.m.",
                  values=dict(EWC=20.64, SI=21.20, ER=88.79, **{"A-GEM": 65.10}, Ninguno=19.89, Conjunto=98.17)),
    "fashion": dict(source="Sokar, Mocanu y Pechenizkiy 2021 (SpaceNet), Neural Networks, tabla 3",
                    protocol="MLP 2x400, 20 épocas por tarea (no un solo pase), cabeza única; no se encontró "
                             "una cifra publicada de ER ni de A-GEM en Split Fashion-MNIST por clases con un "
                             "protocolo comparable: la referencia de repaso es nuestra ER / ER-ACE con los "
                             "mismos bytes",
                    values=dict(EWC=19.47, SI=19.93, iCaRL=80.70)),
}


def split_run(ds: str, spec: str, seed: int, D, n_small: int = 100, iid: bool = False) -> dict:
    """Split-MNIST / Split-Fashion-MNIST por clases (van de Ven et al. 2022): 5 tareas de 2
    clases, un solo pase online por todas las imágenes de entrenamiento de cada tarea
    (sin las de validación), sin identidad de tarea en la prueba: la respuesta es el
    argmax entre las 10 clases del conjunto. Acierto final sobre todo el test de las 10
    clases y olvido (máximo previo - final, por tarea, con 100 imágenes por clase).
    iid=True: control sin secuencia, un solo pase por las mismas imágenes mezcladas (la
    exactitud por tarea se sigue informando por los pares originales; sin olvido)."""
    pools, test = D
    enc = encoder()
    tasks = SPLIT[ds]; cls = np.array([c for t in tasks for c in t])
    rng = np.random.default_rng([seed, 555])
    small = {c: [enc(test[c][i].astype(np.float32) / 255) for i in rng.choice(len(test[c]), n_small, replace=False)]
             for c in cls}
    agent = build(spec)(seed)
    agent.on_phase(0)
    pred = lambda s: int(cls[int(np.argmax(np.asarray(agent.scores(s))[cls]))])
    hist = []
    t0 = time.time()
    train = [tuple(cls)] if iid else tasks
    for k, t in enumerate(train):
        items = [(c, i) for c in t for i in range(len(pools[c]))]
        for o in rng.permutation(len(items)):
            c, i = items[o]
            agent.sup(enc(pools[c][i].astype(np.float32) / 255), int(c))
        seen = [c for tt in train[:k + 1] for c in tt]
        hist.append({int(c): float(np.mean([pred(s) == c for s in small[c]])) for c in seen})
    secs = time.time() - t0
    hits = []; per_task = []
    net = agent.net if isinstance(agent, CortexAgent) else None
    hs, hq = [], []                                  # sistema lento / rápido por separado (Cortex)
    for t in tasks:
        h = []
        for c in t:
            for x in test[c]:
                h.append(pred(enc(x.astype(np.float32) / 255)) == c)
                if net is not None:
                    Vs, Vq = _parts(net)
                    hs.append(int(cls[int(np.argmax(np.asarray(Vs)[cls]))]) == c)
                    hq.append(int(cls[int(np.argmax(np.asarray(Vq)[cls]))]) == c)
        per_task.append(float(np.mean(h))); hits += h
    task_acc = lambda d, t: float(np.mean([d[c] for c in t]))
    forget = [] if iid else [max(task_acc(h, t) for h in hist[k:-1]) - task_acc(hist[-1], t)
                             for k, t in enumerate(tasks[:-1])]
    st = agent.stats()
    row = dict(bench="split", scenario=ds, model=spec, seed=seed, acc=float(np.mean(hits)),
               forgetting=float(np.mean(forget)) if forget else float("nan"), seconds=secs, bytes=st["bytes"], neurons=st["neurons"])
    if "evicted" in st:
        row["evicted"] = int(st["evicted"])
    if hs:
        row["slow_acc"] = float(np.mean(hs)); row["fast_acc"] = float(np.mean(hq))
    row.update({f"task{k + 1}": v for k, v in enumerate(per_task)})
    return row


def split_bench(specs, seeds, D, *, rows, tag, split="test", match=None, datasets=None, iid=False):
    """Corre los modelos de `specs`; `match`: referencias (er, erace) igualadas en bytes al
    promedio del primer modelo; iid: control sin secuencia (ver split_run)."""
    extra = dict(iid=True) if iid else {}
    for ds in (datasets or SPLIT):
        for spec in specs:
            for seed in seeds:
                row = cached(dict(b="split", ds=ds, spec=spec, seed=seed, split=split, **extra),
                             lambda ds=ds, spec=spec, seed=seed: split_run(ds, spec, seed, D, iid=iid))
                rows.append(row)
                print(f"[{tag}] split {ds} {spec[:60]} s{seed}: acc={row['acc']:.4f} olvido={row['forgetting']:.3f} "
                      f"bytes={row['bytes']} {row['seconds']:.0f}s", flush=True)
        if match:
            nb = int(np.mean([r["bytes"] for r in rows if r["bench"] == "split" and r["scenario"] == ds
                              and r["model"] == specs[0]]))
            for base in match:
                sp = f"{base}+bytes={nb}+iters=3"
                for seed in seeds:
                    row = cached(dict(b="split", ds=ds, spec=sp, seed=seed, split=split, **extra),
                                 lambda ds=ds, sp=sp, seed=seed: split_run(ds, sp, seed, D, iid=iid))
                    rows.append(row)
                    print(f"[{tag}] split {ds} {sp} s{seed}: acc={row['acc']:.4f} olvido={row['forgetting']:.3f}",
                          flush=True)


def phase_f6s(a, data):
    """F6, referencia externa: Split-MNIST y Split-Fashion-MNIST por clases, un solo pase."""
    D = split_data(data)["test"]
    rows: list = []
    split_bench([C2], a.seeds, D, rows=rows, tag="f6s", match=("erace", "er"))
    keys = ["acc", "forgetting", "task1", "task2", "task3", "task4", "task5", "bytes", "neurons", "seconds"]
    summary = agg(rows, ("bench", "scenario", "model"), keys)
    for d in summary:
        print(f"  {d['scenario']:8s} {d['model']:45s} " + fmt(d, ["acc", "forgetting"]), flush=True)
    cfg = dict(seeds=a.seeds, tasks=SPLIT, published=PUBLISHED,
               protocol="un solo pase online por las imágenes de entrenamiento de cada tarea (sin las 500 por "
                        "clase de validación), test oficial completo; argmax entre las 10 clases del conjunto",
               note="ER y ER-ACE nuestras: MLP 784-256-20 con reservorio igualado en bytes a c2, lotes de 10, "
                    "3 iteraciones por lote; las cifras publicadas usan más pasadas (ver protocol)")
    save("f6_split", dict(phase="f6s", config=cfg, summary=summary, rows=rows), rows, keys)


CAP_BUDGETS = (1000, 2000, 4000, 8000)             # f6cap0 agrega 0 = sin tope (confirmación opcional)


def phase_f6cap(a, data):
    """Curva de capacidad en Split-Fashion-MNIST (semillas de test 3-5). En f6s c2 pierde
    contra el repaso y su almacén llega al tope de 2000 neuronas en las tres semillas: ¿es
    saturación (recursos) o arquitectura? Presupuesto del almacén 1000, 2000, 4000 y 8000
    neuronas, y sin tope (a 8000 todavía desaloja); ER y ER-ACE con los bytes equivalentes
    en cada punto. Comparte el registro de f6s (el punto de 2000 es c2 tal cual). Regla:
    si c2 sigue perdiendo cuando ya no satura, la pérdida es de arquitectura."""
    _PHASE["name"] = "f6s"
    D = split_data(data)["test"]
    rows: list = []
    budgets = CAP_BUDGETS + ((0,) if a.phase == "f6cap0" else ())
    for n in budgets:
        spec = C2 if n == 2000 else f"{C2}+budget={n}"
        pts: list = []
        split_bench([spec], a.seeds, D, rows=pts, tag="f6cap", match=("erace", "er"), datasets=("fashion",))
        for r in pts:
            fam = r["model"].split("+")[0]
            sat = (r["evicted"] > 0 if "evicted" in r else (n > 0 and r["neurons"] >= n)) if fam == "c2" else False
            rows.append(dict(r, budget=n, family=fam, saturated=float(sat)))
    keys = ["acc", "forgetting", "task1", "task2", "task3", "task4", "task5", "bytes", "neurons", "evicted",
            "saturated", "seconds"]
    summary = agg(rows, ("scenario", "budget", "family"), keys)
    for d in summary:
        print(f"  {d['budget']:5d} {d['family']:6s} " + fmt(d, ["acc", "forgetting", "neurons", "seconds"]), flush=True)
    gap = {n: float(np.mean([r["acc"] for r in rows if r["budget"] == n and r["family"] == "c2"])
                    - max(np.mean([r["acc"] for r in rows if r["budget"] == n and r["family"] == f])
                          for f in ("erace", "er"))) for n in budgets}
    cfg = dict(seeds=a.seeds, budgets=budgets, tasks=SPLIT["fashion"],
               protocol="como f6s: un solo pase online por tarea (sin las 500 por clase de validación), test "
                        "oficial completo, argmax entre las 10 clases; ER/ER-ACE: MLP 784-256-20, lotes de 10, "
                        "3 iteraciones, reservorio uint8 igualado en bytes al c2 de cada punto",
               rule=phase_f6cap.__doc__, gap_c2_minus_best_replay=gap,
               note="budget 0 = sin tope; saturated = 1 si el almacén desalojó alguna vez (en las filas de "
                    "2000, medidas en f6s sin el contador, si terminó con tantas neuronas como su presupuesto); "
                    "evicted = neuronas desalojadas por llegar al tope")
    save("f6_capacidad", dict(phase="f6cap", config=cfg, summary=summary, rows=rows), rows,
         ["budget", "family"] + keys)


def phase_f6iid(a, data):
    """Control sin secuencia en Split-Fashion-MNIST (semillas 3-5): un solo pase por las
    mismas imágenes de entrenamiento mezcladas, c2 y ER / ER-ACE con los mismos bytes.
    Separa el techo del aprendiz (lo que logra cada modelo sin olvido posible) de la
    dinámica continua: si c2 ya pierde contra el repaso sin secuencia, la brecha de f6s
    es del aprendiz y no del olvido."""
    _PHASE["name"] = "f6s"
    D = split_data(data)["test"]
    rows: list = []
    split_bench([C2], a.seeds, D, rows=rows, tag="f6iid", match=("erace", "er"), datasets=("fashion",), iid=True)
    for r in rows:
        r["family"] = r["model"].split("+")[0]; r["order"] = "iid"
    keys = ["acc", "slow_acc", "fast_acc", "task1", "task2", "task3", "task4", "task5", "bytes", "neurons",
            "evicted", "seconds"]
    summary = agg(rows, ("scenario", "order", "family"), keys)
    for d in summary:
        print(f"  iid {d['family']:6s} " + fmt(d, ["acc", "slow_acc", "fast_acc", "task1", "task3", "neurons",
                                                    "seconds"]), flush=True)
    cfg = dict(seeds=a.seeds, rule=phase_f6iid.__doc__,
               protocol="las mismas imágenes de f6s (sin las 500 por clase de validación) en un solo pase, "
                        "mezcladas; test oficial completo, argmax entre las 10 clases",
               compare="consolidacion_f6_split.json (secuencia por tareas)")
    save("f6_iid", dict(phase="f6iid", config=cfg, summary=summary, rows=rows), rows, ["order", "family"] + keys)


def _ridge_probe(Xtr, ytr, Xte, classes, seed: int, lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """Sonda lineal (ridge en forma dual, objetivos +-1 por clase) sobre filas dispersas;
    lambda elegido con el 20 % del entrenamiento de la sonda. Devuelve las predicciones."""
    import scipy.sparse as sps
    Xtr = sps.csr_matrix(Xtr); Xte = sps.csr_matrix(Xte)
    Y = np.where(ytr[:, None] == classes[None, :], 1.0, -1.0)
    perm = np.random.default_rng([seed, 31]).permutation(len(ytr)); nv = len(ytr) // 5
    va, tr = perm[:nv], perm[nv:]
    K = (Xtr @ Xtr.T).toarray().astype(np.float64)
    best = (-1.0, lams[0])
    for lam in lams:
        A = np.linalg.solve(K[np.ix_(tr, tr)] + lam * np.eye(len(tr)), Y[tr])
        acc = float(np.mean(classes[np.argmax(K[np.ix_(va, tr)] @ A, axis=1)] == ytr[va]))
        best = max(best, (acc, lam))
    A = np.linalg.solve(K + best[1] * np.eye(len(ytr)), Y)
    P = (Xte @ Xtr.T).toarray() @ A
    return classes[np.argmax(P, axis=1)], best[1]


def split_diag(ds: str, spec: str, seed: int, D, n_small: int = 100, n_tr: int = 300, n_te: int = 500,
               probes: bool = True) -> dict:
    """¿Dónde está el olvido de c2 en Split-Fashion? Mismo flujo que split_run (mismo
    generador: el modelo final es el mismo) y, al terminar cada tarea, una copia de los
    rasgos de V1 y del almacén. Al final, sobre las mismas imágenes de test (n_te por
    clase), exactitud de cada lectura (la proyección sobre V1 = sistema lento; la lectura
    del almacén = sistema rápido; el todo) contra una sonda lineal entrenada aparte
    (n_tr imágenes de entrenamiento por clase, las 10 clases balanceadas) sobre los
    mismos rasgos congelados: los del final y los de justo después de aprender cada tarea.
    Sonda final ~ sonda al aprender y lectura muy por debajo: sesgo de la lectura. Sonda
    final por debajo de la sonda al aprender: deriva de los rasgos. probes=False: sólo las
    lecturas y el sesgo de los errores (sin copias ni sondas)."""
    import copy
    pools, test = D
    enc = encoder()
    tasks = SPLIT[ds]; cls = np.array([c for t in tasks for c in t])
    rng = np.random.default_rng([seed, 555])
    small = {c: [enc(test[c][i].astype(np.float32) / 255) for i in rng.choice(len(test[c]), n_small, replace=False)]
             for c in cls}
    agent = build(spec)(seed)
    agent.on_phase(0)
    net = agent.net
    am = lambda V: int(cls[int(np.argmax(np.asarray(V)[cls]))])

    def three(s):
        V = agent.scores(s)
        Vs, Vq = _parts(net)
        return am(V), am(Vs), am(Vq)

    learn = {"full": {}, "slow": {}, "fast": {}}
    snaps = []
    t0 = time.time()
    for k, t in enumerate(tasks):
        items = [(c, i) for c in t for i in range(len(pools[c]))]
        for o in rng.permutation(len(items)):
            c, i = items[o]
            agent.sup(enc(pools[c][i].astype(np.float32) / 255), int(c))
        seen = [c for tt in tasks[:k + 1] for c in tt]
        for c in seen:                                   # (las mismas llamadas que split_run)
            r = np.array([three(s) for s in small[c]])
            if c in t:
                for j, name in enumerate(("full", "slow", "fast")):
                    learn[name][int(c)] = float(np.mean(r[:, j] == c))
        if probes:
            snaps.append((copy.deepcopy(net.areas[0]), copy.deepcopy(net.heads[0])))
    secs = time.time() - t0
    hits = [three(enc(x.astype(np.float32) / 255))[0] == c for c in cls for x in test[c]]
    prng = np.random.default_rng([seed, 777])
    tr = [(c, pools[c][i]) for c in cls for i in prng.choice(len(pools[c]), n_tr, replace=False)]
    te = [(c, test[c][i]) for c in cls for i in prng.choice(len(test[c]), n_te, replace=False)]
    ytr = np.array([c for c, _ in tr]); yte = np.array([c for c, _ in te])
    Xtr_in = [enc(x.astype(np.float32) / 255) for _, x in tr]
    Xte_in = [enc(x.astype(np.float32) / 255) for _, x in te]
    out = dict(full=[], slow=[], fast=[])
    for s in Xte_in:
        f, sl, fa = three(s)
        out["full"].append(f); out["slow"].append(sl); out["fast"].append(fa)
    out = {k: np.array(v) for k, v in out.items()}
    import scipy.sparse as sps

    def v1_rows(area, X):
        rows = []
        for x in X:
            code = area.encode(np.asarray(x, dtype=np.float32).reshape(net.input_shape))
            idx, val = net._proj_input([code])
            rows.append((idx, val))
        area._cache = None
        return rows

    def head_rows(head, X):
        rows, pred = [], []
        for x in X:
            v = head.layer.forward(*head._as_sparse(x))
            h = head._votes(v)
            rows.append((np.asarray(v.winners, dtype=np.int64), np.asarray(h, dtype=np.float32)))
            pred.append(am(head.readout.potentials(v.winners, h)))
        return rows, np.array(pred)

    def mat(rows, dim):
        ind = np.concatenate([r[0] for r in rows]); dat = np.concatenate([r[1] for r in rows])
        ptr = np.concatenate([[0], np.cumsum([len(r[0]) for r in rows])])
        return sps.csr_matrix((dat.astype(np.float64), ind, ptr), shape=(len(rows), dim))

    pr = {}
    pix = lambda X: [(np.flatnonzero(np.ravel(x)), np.ravel(x)[np.flatnonzero(np.ravel(x))] /
                      max(float(np.linalg.norm(np.ravel(x))), 1e-9)) for x in X]
    lams = {}
    head_readout = {}
    if probes:
        pr["pix"], lams["pix"] = _ridge_probe(mat(pix(Xtr_in), 784), ytr, mat(pix(Xte_in), 784), cls, seed)
    for k, (A, H) in enumerate(snaps):
        dim = int(net.proj.W.shape[1])
        p, lam = _ridge_probe(mat(v1_rows(A, Xtr_in), dim), ytr, mat(v1_rows(A, Xte_in), dim), cls, seed)
        pr[f"v1_{k + 1}"] = p; lams[f"v1_{k + 1}"] = lam
        hdim = int(H.layer.n) + 1
        rtr, _ = head_rows(H, Xtr_in); rte, hp = head_rows(H, Xte_in)
        p, lam = _ridge_probe(mat(rtr, hdim), ytr, mat(rte, hdim), cls, seed)
        pr[f"head_{k + 1}"] = p; lams[f"head_{k + 1}"] = lam
        head_readout[k + 1] = hp
    acc = lambda pred, c_set: float(np.mean(pred[np.isin(yte, c_set)] == yte[np.isin(yte, c_set)]))
    row = dict(bench="diag", scenario=ds, model=spec, seed=seed, acc=float(np.mean(hits)), seconds=secs,
               neurons=agent.stats()["neurons"], evicted=int(agent.stats().get("evicted", 0)), lambdas=lams)
    last = np.array(tasks[-1])
    for k, t in enumerate(tasks):
        tag = f"t{k + 1}"
        for name in ("full", "slow", "fast"):
            row[f"end_{name}_{tag}"] = acc(out[name], t)
            row[f"learn_{name}_{tag}"] = float(np.mean([learn[name][c] for c in t]))
            m = np.isin(yte, t) & (out[name] != yte)
            later = np.array([c for tt in tasks[k + 1:] for c in tt])
            row[f"errlater_{name}_{tag}"] = float(np.mean(np.isin(out[name][m], later))) if m.any() and len(later) else 0.0
            row[f"errlast_{name}_{tag}"] = float(np.mean(np.isin(out[name][m], last))) if m.any() else 0.0
        if not probes:
            continue
        row[f"probe_pix_{tag}"] = acc(pr["pix"], t)
        row[f"probe_v1_end_{tag}"] = acc(pr[f"v1_{len(tasks)}"], t)
        row[f"probe_v1_learn_{tag}"] = acc(pr[f"v1_{k + 1}"], t)
        row[f"probe_head_end_{tag}"] = acc(pr[f"head_{len(tasks)}"], t)
        row[f"probe_head_learn_{tag}"] = acc(pr[f"head_{k + 1}"], t)
        row[f"readout_head_end_{tag}"] = acc(head_readout[len(tasks)], t)
    old = [f"t{k + 1}" for k in range(len(tasks) - 1)]
    for key in [k[:-3] for k in list(row) if k.endswith("_t1")]:
        row[f"{key}_old"] = float(np.mean([row[f"{key}_{t}"] for t in old]))
    return row


DIAG_KEYS = ["acc", "end_full_old", "end_slow_old", "end_fast_old", "learn_full_old", "learn_slow_old",
             "learn_fast_old", "probe_v1_end_old", "probe_v1_learn_old", "probe_head_end_old",
             "probe_head_learn_old", "readout_head_end_old", "probe_pix_old", "errlater_full_old",
             "errlater_slow_old", "errlater_fast_old", "errlast_slow_old", "errlast_fast_old",
             "end_full_t5", "end_slow_t5", "end_fast_t5", "neurons", "evicted", "seconds"]


DIAG_SEEDS = (10, 11, 12)


def phase_f6diag(a, data):
    """Diagnóstico de la causa del olvido de c2 en Split-Fashion antes de tocar nada (ver
    split_diag). Corre en el split de validación (500 imágenes por clase fuera del
    entrenamiento; semillas 10, 11 y 12), para que lo que se decida arreglar a partir de
    acá no salga de mirar el test. Regla fijada antes de medir, sobre las tareas
    viejas (1-4): si la sonda sobre los rasgos finales conserva lo viejo (cae menos de 0,03
    respecto de la sonda justo después de aprenderlo) y la lectura queda muy por debajo de
    la sonda, el olvido es sesgo de la lectura hacia lo reciente; si la sonda también cae
    0,03 o más, es deriva de los rasgos. Se mira por separado el sistema lento (V1 +
    proyección) y el rápido (almacén + su lectura)."""
    D = split_data(data)["val"]
    rows: list = []
    for seed in DIAG_SEEDS:
        row = cached(dict(b="diag", ds="fashion", spec=C2, seed=seed, split="val"),
                     lambda seed=seed: split_diag("fashion", C2, seed, D))
        rows.append(row)
        print(f"[f6diag] s{seed}: acc={row['acc']:.4f} " + " ".join(f"{k}={row[k]:.3f}" for k in DIAG_KEYS[1:18]),
              flush=True)
    keys = DIAG_KEYS + [k for k in rows[0] if k.startswith(("end_", "learn_", "probe_", "readout_", "err"))
                        and not k.endswith("_old") and k not in DIAG_KEYS]
    summary = agg(rows, ("bench", "scenario", "model"), keys)
    for d in summary:
        print("  " + fmt(d, DIAG_KEYS[:18]), flush=True)
    cfg = dict(seeds=DIAG_SEEDS, split="val", rule=phase_f6diag.__doc__, method=split_diag.__doc__,
               probe="ridge dual con objetivos +-1, lambda en {1e-3..100} elegido con el 20 % del entrenamiento "
                     "de la sonda; 300 imágenes de entrenamiento por clase (las 10), 500 de test por clase",
               note="readout_head_end = la lectura del almacén sobre los mismos rasgos que su sonda (pase sin "
                    "contexto); end_fast = el sistema rápido dentro del modelo")
    save("f6_diagnostico", dict(phase="f6diag", config=cfg, summary=summary, rows=rows), rows, keys)


ACE = "+proj.ace=true"


def phase_f6fix(a, data):
    """Hipótesis que sale del diagnóstico (f6diag, en validación): los rasgos de V1 conservan
    lo viejo y la proyección lo pierde, con los errores yendo a clases más nuevas. La
    proyección de c2 ya no deprime a las clases ausentes (active_window=200), pero su error
    positivo usa la competencia contra todas las salidas: una clase ausente parecida baja
    p(objetivo) y la fila nueva crece sobre los rasgos compartidos. Arreglo local (Readout
    ace): para lo que llega, la competencia se resuelve sólo entre las clases presentes,
    como la entropía cruzada enmascarada de ER-ACE; el repaso sigue con el error completo.
    1) Validación (Split-Fashion, split de validación, semillas 10-12): c2 contra c2 + ace;
       se adopta si sube la exactitud media (regla fijada antes de mirar).
    2) Si se adopta, test una sola vez (semillas 3-5) en Split-Fashion y Split-MNIST, contra
       c2 y ER / ER-ACE con los mismos bytes de f6s (ace no agrega estado)."""
    _PHASE["name"] = "f6diag"
    V = split_data(data)["val"]
    vrows: list = []
    for spec in (C2, C2 + ACE):
        for seed in DIAG_SEEDS:
            key = dict(b="diag", ds="fashion", spec=spec, seed=seed, split="val")
            if spec != C2:
                key["probes"] = False
            row = cached(key, lambda spec=spec, seed=seed: split_diag("fashion", spec, seed, V, probes=(spec == C2)))
            vrows.append(row)
            print(f"[f6fix] val {spec} s{seed}: acc={row['acc']:.4f} " + " ".join(
                f"{k}={row[k]:.3f}" for k in ("end_full_old", "end_slow_old", "end_fast_old", "learn_slow_old",
                                             "errlater_slow_old", "end_full_t5", "end_slow_t5")), flush=True)
    m = lambda sp, k: float(np.mean([r[k] for r in vrows if r["model"] == sp]))
    adopt = m(C2 + ACE, "acc") > m(C2, "acc")
    vkeys = ["acc", "end_full_old", "end_slow_old", "end_fast_old", "learn_full_old", "learn_slow_old",
             "learn_fast_old", "errlater_full_old", "errlater_slow_old", "errlast_slow_old", "end_full_t5",
             "end_slow_t5", "end_fast_t5", "seconds"]
    vsum = agg(vrows, ("bench", "scenario", "model"), vkeys)
    for d in vsum:
        print(f"  val {d['model']:24s} " + fmt(d, vkeys[:10]), flush=True)
    out = dict(phase="f6fix", adopted=adopt, rule=phase_f6fix.__doc__, val_seeds=DIAG_SEEDS,
               validation=dict(summary=vsum, rows=vrows))
    rows: list = []
    if adopt:
        _PHASE["name"] = "f6s"
        T = split_data(data)["test"]
        split_bench([C2 + ACE], a.seeds, T, rows=rows, tag="f6fix")
        ref = json.loads((RESULTS / "consolidacion_f6_split.json").read_text())["rows"]
        rows += [dict(r, reference="f6s") for r in ref]
        keys = ["acc", "forgetting", "slow_acc", "fast_acc", "task1", "task2", "task3", "task4", "task5", "bytes",
                "neurons", "evicted", "seconds"]
        out["test"] = dict(seeds=a.seeds, summary=agg(rows, ("bench", "scenario", "model"), keys), rows=rows)
        for d in out["test"]["summary"]:
            print(f"  test {d['scenario']:8s} {d['model'][:40]:40s} " + fmt(d, ["acc", "forgetting"]), flush=True)
    (RESULTS / "consolidacion_f6_arreglo.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"[f6fix] adoptado: {adopt} (validación c2 {m(C2, 'acc'):.4f}, c2+ace {m(C2 + ACE, 'acc'):.4f})", flush=True)


# ====================================================================== F7
VAL_SEEDS = (10, 11)
CAUSES = "+head.causes=true+head.cause_tau=500"
F7_BASE = C2 + STORED + OWN


def pair_rows(spec, seeds, data, *, tag, split, rows, cd=True, swap=True, chain=False, rename=False):
    """CD y cambio de etiqueta (y chainF) con el mismo modelo y el split pedido (reanudable)."""
    def cd_row(spec, seed):
        r = run_scenario("CD", spec, seed, data)
        row = scenario_row("CD", spec, seed, r); row["split"] = split
        row.update({f"c_{k}": v for k, v in cause_counters(r).items()})
        return row

    def sw_row(spec, seed):
        r = run_scenario("drift", spec, seed, data, phases=DRIFT)
        row = drift_row(spec, seed, r); row["split"] = split
        row.update({f"c_{k}": v for k, v in cause_counters(r).items()})
        return row

    def ch_row(spec, seed):
        r = run_scenario("chainF", spec, seed, data)
        row = scenario_row("chainF", spec, seed, r); row["split"] = split
        row.update({f"c_{k}": v for k, v in cause_counters(r).items()})
        return row

    def rn_row(spec, seed):
        r = run_scenario("rename", spec, seed, data, phases=RENAME)
        row = drift_row(spec, seed, r, changed=(10,), kept=(11, 12, 13), scenario="rename10_18")
        row["split"] = split
        row.update({f"c_{k}": v for k, v in cause_counters(r).items()})
        return row

    for seed in seeds:
        if cd:
            row = cached(dict(b="interf", sc="CD", spec=spec, seed=seed, split=split), lambda: cd_row(spec, seed))
            rows.append(row)
            print(f"[{tag}] {split} CD {spec[len(C2):]} s{seed}: " + " ".join(
                f"{k}={row[k]:.3f}" for k in ("old", "new", "all", "min", "slow_all")) +
                f" causas={[row.get('c_' + k) for k in ('relabeled', 'protected', 'new_causes')]}", flush=True)
        if swap:
            row = cached(dict(b="drift", sc="swap", spec=spec, seed=seed, split=split), lambda: sw_row(spec, seed))
            rows.append(row)
            print(f"[{tag}] {split} cambio {spec[len(C2):]} s{seed}: " + " ".join(
                f"{k}={row[k]:.3f}" for k in ("full_swapped_mid", "full_swapped_end", "full_kept_end",
                                                "full_kept_drop")) +
                f" causas={[row.get('c_' + k) for k in ('relabeled', 'protected', 'new_causes')]}", flush=True)
        if chain:
            row = cached(dict(b="interf", sc="chainF", spec=spec, seed=seed, split=split), lambda: ch_row(spec, seed))
            rows.append(row)
            print(f"[{tag}] {split} chainF {spec[len(C2):]} s{seed}: " + " ".join(
                f"{k}={row[k]:.3f}" for k in ("old", "new", "all", "min")), flush=True)
        if rename:
            row = cached(dict(b="drift", sc="rename", spec=spec, seed=seed, split=split), lambda: rn_row(spec, seed))
            rows.append(row)
            print(f"[{tag}] {split} renombre {spec[len(C2):]} s{seed}: " + " ".join(
                f"{k}={row[k]:.3f}" for k in ("full_swapped_mid", "full_swapped_end", "full_kept_end")), flush=True)


def cause_counters(r) -> dict:
    return {k: r.get("causes", {}).get(k, 0) for k in ("relabeled", "protected", "new_causes", "superseded")}


def val_score(rows, spec) -> float:
    """Criterio fijado antes de mirar resultados: media de cinco números de validación,
    CD (todas, viejas) y cambio de etiqueta (a los 200 pasos, al final, vecinos al final)."""
    cd = [r for r in rows if r["bench"] == "interf" and r["model"] == spec]
    sw = [r for r in rows if r["bench"] == "drift" and r["model"] == spec]
    vals = [np.mean([r["all"] for r in cd]), np.mean([r["old"] for r in cd]),
            np.mean([r["full_swapped_mid"] for r in sw]), np.mean([r["full_swapped_end"] for r in sw]),
            np.mean([r["full_kept_end"] for r in sw])]
    return float(np.mean(vals))


SWEEP_A = {"sin causas": "",
           "causas": CAUSES,
           "causas, corrimiento 0.06": CAUSES + "+head.cause_shift=0.06",
           "causas, racha 2": CAUSES + "+head.cause_streak=2",
           "causas, ritmo 0.5": CAUSES + "+head.cause_rho=0.5",
           "causas sin proteger el lento": CAUSES + "+top.cause_slow=false",
           "sólo proteger (nunca reconsolidar)": CAUSES + "+head.cause_shift=-9"}
SWEEP_B = {
    "recuerdo de respaldo (F1)": "+top.recall=fallback+top.recall_gain=0.5",
    "esquema congelado (F2)": "+top.schema=gate+top.schema_frozen=500",
    "autoexamen con margen (F4)": UNI + EQUAL + "+top.active_margin=0.5",
    "contraste (F4)": "+top.replay_contrast=true",
    "sueño micro priorizado (F5)": MICRO + "+top.sleep_focus=weak",
}


def phase_f7a(a, data):
    """Selección en validación (semillas 10 y 11, split de validación): primero la
    variante de causas latentes, después cada arreglo encima de la mejor, y por último
    la suma de los arreglos que mejoraron el criterio."""
    D = split_data(data)["val"]
    rows: list = []
    scores = {}
    for name, mod in SWEEP_A.items():
        pair_rows(F7_BASE + mod, VAL_SEEDS, D, tag="f7a", split="val", rows=rows)
        scores[F7_BASE + mod] = val_score(rows, F7_BASE + mod)
        print(f"[f7a] puntaje {name}: {scores[F7_BASE + mod]:.4f}", flush=True)
    best_a = max((F7_BASE + m for m in SWEEP_A.values()), key=lambda sp: scores[sp])
    good = []
    for name, mod in SWEEP_B.items():
        sp = best_a + mod
        pair_rows(sp, VAL_SEEDS, D, tag="f7a", split="val", rows=rows)
        scores[sp] = val_score(rows, sp)
        print(f"[f7a] puntaje {name}: {scores[sp]:.4f} (base {scores[best_a]:.4f})", flush=True)
        if scores[sp] > scores[best_a]:
            good.append(mod)
    chosen = best_a
    if len(good) > 1:
        sp = best_a + "".join(good)
        pair_rows(sp, VAL_SEEDS, D, tag="f7a", split="val", rows=rows)
        scores[sp] = val_score(rows, sp)
        print(f"[f7a] puntaje combinado: {scores[sp]:.4f}", flush=True)
        cands = [best_a + m for m in good] + [sp]
        chosen = max(cands, key=lambda c: scores[c])
    elif good:
        chosen = best_a + good[0]
    keys = sorted({k for r in rows for k in r if isinstance(r[k], (int, float)) and k != "seed"})
    summary = agg(rows, ("bench", "scenario", "model"), keys)
    out = dict(phase="f7a", chosen=chosen, best_causes=best_a, improving_fixes=good, scores=scores,
               score_definition="media de: CD todas, CD viejas, cambio a los 200 pasos, cambio al final, "
                                "vecinos al final (todo en validación)",
               config=dict(val_seeds=VAL_SEEDS, val_per_class=VAL_N, sweep_a=SWEEP_A, sweep_b=SWEEP_B,
                           base=F7_BASE),
               summary=summary, rows=rows)
    (RESULTS / "consolidacion_f7_validacion.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"[f7a] elegido: {chosen}", flush=True)


def overlap(net, test, pairs=((10, 16), (12, 14), (10, 12), (14, 16), (10, 13)), n=200, seed=0) -> dict:
    """¿La representación superpone clases parecidas? Coseno entre las medias de clase
    (vectores normalizados) en píxeles y en el código de V1, y redundancia del diccionario
    de V1 (coseno máximo medio de cada rasgo con otro). Nota cruzada de la ronda de
    predicción: V1 muy redundante."""
    enc = encoder()
    rng = np.random.default_rng([seed, 99])
    cls = sorted({c for p in pairs for c in p})
    mp, mv = {}, {}
    for c in cls:
        P, Vv = [], []
        for i in rng.choice(len(test[c]), n, replace=False):
            x = enc(test[c][i].astype(np.float32) / 255)
            p = x.ravel(); P.append(p / max(np.linalg.norm(p), 1e-9))
            code = net.areas[0].encode(np.asarray(x, dtype=np.float32).reshape(net.input_shape)).ravel()
            net.areas[0]._cache = None
            Vv.append(code / max(np.linalg.norm(code), 1e-9))
        mp[c] = np.mean(P, axis=0); mv[c] = np.mean(Vv, axis=0)
    cosm = lambda a, b: float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))
    out = {f"pix_{a}_{b}": cosm(mp[a], mp[b]) for a, b in pairs}
    out.update({f"v1_{a}_{b}": cosm(mv[a], mv[b]) for a, b in pairs})
    A = net.areas[0]; W = A.W[:A.n][A.alive[:A.n]]
    if len(W) > 1:
        S = W @ W.T; np.fill_diagonal(S, -1.0)
        out["v1_redundancy"] = float(S.max(axis=1).mean())
    return out


FIXES = "+top.replay_contrast=true" + MICRO + "+top.sleep_focus=weak"
PARALLEL = "+head.cause_mode=parallel"
FINAL_CANDS = {"sin causas": "",
               "causas (reescribir)": CAUSES,
               "causas (reescribir), sin antigüedad": CAUSES + "+head.cause_est=0",
               "causas (paralelo)": CAUSES + PARALLEL,
               "causas (paralelo), sin antigüedad": CAUSES + PARALLEL + "+head.cause_est=0"}


def phase_f7a2(a, data):
    """Selección final en validación (semillas 10 y 11) con el objetivo duro como
    restricción y los cuatro casos a la vez: interferencia CD, cambio de etiqueta entre
    nombres establecidos (10 <-> 12), clases parecidas que llegan de a una (chainF) y el
    caso adversarial inverso, una clase que pasa a llamarse con un nombre recién aparecido
    (10 -> 18). El primer barrido (consolidacion_f7_validacion_barrido1.json) eligió por
    un promedio que cambiaba viejas de CD por velocidad y no miraba chainF. Regla:
      factible  CD todas >= sin causas - 0,01; CD viejas >= 0,80; chainF todas >= sin
                causas - 0,01; vecinos del cambio de etiqueta >= sin causas - 0,01
      se elige  el factible con mayor media de la exactitud de lo que cambió de nombre
                (a los 200 pasos y al final, en 10 <-> 12 y en 10 -> 18); si ninguno es
                factible, el de menor violación total. Después se prueban encima los dos
                arreglos que mejoraron en el primer barrido (contraste + sueño priorizado)
                y se quedan si siguen siendo factibles y mejoran ese objetivo."""
    D = split_data(data)["val"]
    rows: list = []
    table = {}

    def measure(name, sp):
        pair_rows(sp, VAL_SEEDS, D, tag="f7a2", split="val", rows=rows, chain=True, rename=True)
        mine = [r for r in rows if r["model"] == sp]
        cd = [r for r in mine if r["bench"] == "interf" and r["scenario"] == "CD"]
        ch = [r for r in mine if r["bench"] == "interf" and r["scenario"] == "chainF"]
        sw = [r for r in mine if r["bench"] == "drift" and r["scenario"] == "swap10_12"]
        rn = [r for r in mine if r["bench"] == "drift" and r["scenario"] == "rename10_18"]
        f = lambda rs, k: float(np.mean([r.get(k, 0.0) for r in rs]))
        t = dict(spec=sp, cd_all=f(cd, "all"), cd_old=f(cd, "old"), cd_new=f(cd, "new"),
                 chain_all=f(ch, "all"), chain_old=f(ch, "old"), swap_mid=f(sw, "full_swapped_mid"),
                 swap_end=f(sw, "full_swapped_end"), kept=f(sw, "full_kept_end"),
                 rename_mid=f(rn, "full_swapped_mid"), rename_end=f(rn, "full_swapped_end"),
                 rename_kept=f(rn, "full_kept_end"),
                 relabel_cd=f(cd, "c_relabeled"), relabel_chain=f(ch, "c_relabeled"),
                 superseded_cd=f(cd, "c_superseded"), superseded_chain=f(ch, "c_superseded"))
        t["objective"] = float(np.mean([t["swap_mid"], t["swap_end"], t["rename_mid"], t["rename_end"]]))
        table[name] = t
        print(f"[f7a2] {name}: " + " ".join(f"{k}={v:.3f}" for k, v in t.items() if k != "spec"), flush=True)

    def judge():
        base = table["sin causas"]
        for t in table.values():
            t["violation"] = (max(0.0, base["cd_all"] - 0.01 - t["cd_all"]) + max(0.0, 0.80 - t["cd_old"])
                              + max(0.0, base["chain_all"] - 0.01 - t["chain_all"])
                              + max(0.0, base["kept"] - 0.01 - t["kept"]))
            t["feasible"] = t["violation"] == 0.0
        feas = [n for n, t in table.items() if t["feasible"]]
        if feas:
            return feas, max(feas, key=lambda n: table[n]["objective"])
        return feas, min(table, key=lambda n: (table[n]["violation"], -table[n]["objective"]))

    for name, mod in FINAL_CANDS.items():
        measure(name, F7_BASE + mod)
    feas, best = judge()
    if best != "sin causas":
        measure(best + " + contraste + sueño priorizado", table[best]["spec"] + FIXES)
        feas, best = judge()
    keys = sorted({k for r in rows for k in r if isinstance(r[k], (int, float)) and k != "seed"})
    out = dict(phase="f7a2", chosen=table[best]["spec"], chosen_name=best, feasible=feas, table=table,
               rule=phase_f7a2.__doc__, first_sweep="consolidacion_f7_validacion_barrido1.json",
               config=dict(val_seeds=VAL_SEEDS, val_per_class=VAL_N, candidates=FINAL_CANDS, base=F7_BASE,
                           scenarios=dict(CD=SCENARIOS["CD"], chainF=SCENARIOS["chainF"], swap=DRIFT, rename=RENAME)),
               summary=agg(rows, ("bench", "scenario", "model"), keys), rows=rows)
    (RESULTS / "consolidacion_f7_validacion.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"[f7a2] factibles: {feas}; elegido: {best}: {table[best]['spec']}", flush=True)


def chosen_spec() -> str:
    return json.loads((RESULTS / "consolidacion_f7_validacion.json").read_text())["chosen"]


def ablations(spec: str) -> dict:
    """Quitar de a uno los componentes que el elegido agrega sobre c2."""
    mods = spec[len(C2):].split("+")[1:]
    groups = {"etiquetas guardadas": ["top.stored_labels=true"], "error propio del lento": ["top.slow_error=own"]}
    causes = [m for m in mods if m.startswith("head.cause") or m.startswith("top.cause_")]
    if causes:
        groups["causas latentes"] = causes
    for name, mod in SWEEP_B.items():
        parts = mod.split("+")[1:]
        if parts and all(p in mods for p in parts):
            groups[name] = parts
    out = {}
    for name, parts in groups.items():
        keep = [m for m in mods if m not in parts]
        if len(keep) < len(mods):
            out[f"sin {name}"] = C2 + "".join("+" + m for m in keep)
    return out


F7_TEST_SEEDS = (3, 4, 5)
CONTAMINATION = ("Semillas de test 0 y 1 contaminadas para F7: una elección anterior (primer barrido: "
                 "corrimiento 0,06 + contraste + sueño priorizado, sin cause_est) llegó a correr en test "
                 "con las semillas 0 (CD, cambio de etiqueta, chainF) y 1 (CD, cambio de etiqueta) y se "
                 "vieron esos números (CD 0,825 / 0,755; chainF 0,622) antes de descartarla; ese vistazo "
                 "motivó mirar chainF y agregar cause_est. Por eso todas las pruebas de F7 usan las "
                 "semillas nuevas 3, 4 y 5 (la 2 se descarta por prudencia) y corren una sola vez sobre "
                 "lo elegido en validación (semillas 10 y 11).")


def phase_f7b(a, data):
    """Prueba (split oficial de test, semillas nuevas 3-5; ver CONTAMINATION) de la
    configuración elegida en validación, contra c2 y ER-ACE con los mismos bytes, con
    ablaciones: interferencia CD, cambio de etiqueta 10 <-> 12, chainF y renombre 10 -> 18
    con la misma configuración."""
    D = split_data(data)["test"]
    spec = chosen_spec()
    rows: list = []
    for sp in (spec, C2):
        pair_rows(sp, a.seeds, D, tag="f7b", split="test", rows=rows, chain=True, rename=True)
    nb = int(np.mean([r["bytes"] for r in rows if r["bench"] == "interf" and r["scenario"] == "CD"
                      and r["model"] == spec]))
    er = f"erace+bytes={nb}+iters=3"
    pair_rows(er, a.seeds, D, tag="f7b", split="test", rows=rows, chain=True, rename=True)
    abl = ablations(spec)
    for name, sp in abl.items():                     # (chainF sólo para el elegido y las referencias)
        pair_rows(sp, a.seeds, D, tag="f7b", split="test", rows=rows)
    ov = []
    s0 = a.seeds[0]
    for sp in (spec, C2):                            # diagnóstico de representación (primera semilla de test)
        def go(sp=sp):
            r = run_scenario("CD", sp, s0, D, keep_agent=True)
            return dict(bench="overlap", scenario="CD", model=sp, seed=s0, **overlap(r["agent"].net, D[1]))
        ov.append(cached(dict(b="overlap", spec=sp, seed=s0, split="test"), go))
        print(f"[f7b] superposición {sp[len(C2):]}: " + " ".join(f"{k}={v:.3f}" for k, v in ov[-1].items()
                                                                if isinstance(v, float)), flush=True)
    for r in rows:
        r.setdefault("split", "test")
    rows += ov
    cfg = dict(seeds=a.seeds, chosen=spec, erace=er, ablations=abl, contamination=CONTAMINATION,
               protocol="entrenamiento sin las 500 imágenes por clase de validación; evaluación en test, "
                        "una sola vez, sobre la configuración elegida en f7a2",
               rename=RENAME,
               drift=DRIFT, interference={k: SCENARIOS[k] for k in ("CD", "chainF")})
    finish("f7b", rows, {}, cfg)


def phase_f7x(a, data):
    """Referencia externa para el elegido: Split-MNIST y Split-Fashion-MNIST por clases
    (las de c2, ER y ER-ACE están en consolidacion_f6_split.json)."""
    D = split_data(data)["test"]
    spec = chosen_spec()
    rows: list = []
    split_bench([spec], a.seeds, D, rows=rows, tag="f7x", match=("erace",))
    keys = ["acc", "forgetting", "slow_acc", "fast_acc", "task1", "task2", "task3", "task4", "task5", "bytes",
            "neurons", "evicted", "seconds"]
    summary = agg(rows, ("bench", "scenario", "model"), keys)
    for d in summary:
        print(f"  {d['scenario']:8s} {d['model'][-60:]:60s} " + fmt(d, ["acc", "forgetting"]), flush=True)
    cfg = dict(seeds=a.seeds, chosen=spec, tasks=SPLIT, published=PUBLISHED,
               protocol="un solo pase online por tarea (sin las 500 por clase de validación), test oficial "
                        "completo; argmax entre las 10 clases del conjunto")
    save("f7_split", dict(phase="f7x", config=cfg, summary=summary, rows=rows), rows, keys)


def stream_spec() -> str:
    """Configuración para los flujos largos: la elegida en CD/cambio de etiqueta, con el
    ajuste de plasticidad elegido en validación (ver phase_f7k)."""
    p = RESULTS / "consolidacion_f7_plasticidad.json"
    return json.loads(p.read_text())["chosen"] if p.exists() else chosen_spec()


def phase_f7k(a, data):
    """Flujo continuo reducido (escala 0,1).
    1) Validación (semillas 10 y 11): la plasticidad tardía bajaba con slow_error="own"
       (f1b: 0,932 de una red nueva). Candidatos: el elegido, el elegido con sueño en
       micro-fases y renormalización, el elegido con el error conjunto y, si f6fix lo
       adoptó, el elegido con la competencia entre presentes (ace). Criterio fijado
       antes de mirar: final - olvido + min(plasticidad, 1).
    2) Test (semillas nuevas 3-5): el ganador contra c2 y ER-ACE con los mismos bytes."""
    S = split_data(data)
    base = chosen_spec()
    cands = {"elegido": base}
    if "sleep_every" in base:
        cands["elegido + renormalización"] = _with_mod(base, "top.sleep_downscale", 0.1)
    else:
        cands["elegido + sueño micro y renormalización"] = base + MICRO + "+top.sleep_downscale=0.1"
    if "top.slow_error=own" in base:
        cands["elegido con error conjunto"] = base.replace("+top.slow_error=own", "")
    fix = RESULTS / "consolidacion_f6_arreglo.json"
    if fix.exists() and json.loads(fix.read_text()).get("adopted"):    # arreglo de la lectura (f6fix)
        cands["elegido + competencia entre presentes (ace)"] = base + ACE
    vrows: list = []
    scores = {}
    for name, sp in cands.items():
        continual((sp.replace(C2, CONT, 1),), VAL_SEEDS, S["val"], scale=0.1, n_eval=30, log_rows=vrows,
                  tag="f7k-val", split="val")
        rs = [r for r in vrows if r["model"] == sp.replace(C2, CONT, 1)]
        scores[name] = float(np.mean([r["final_clean_known"] - r["forget_clean"] + min(r["plasticity_ratio"], 1.0)
                                      for r in rs]))
        print(f"[f7k] validación {name}: {scores[name]:.4f} " +
              " ".join(f"{k}={np.mean([r[k] for r in rs]):.3f}" for k in
                       ("final_clean_known", "forget_clean", "plasticity_ratio", "sim_after_D")), flush=True)
    best = max(cands, key=lambda k: scores[k])
    spec = cands[best]
    (RESULTS / "consolidacion_f7_plasticidad.json").write_text(json.dumps(dict(
        chosen=spec, chosen_name=best, candidates=cands, scores=scores, val_seeds=VAL_SEEDS,
        score="final_clean_known - forget_clean + min(plasticity_ratio, 1), flujo x0,1, split de validación",
        rows=vrows, summary=agg(vrows, ("bench", "scenario", "model"), CONT_KEYS)), indent=1, default=float))
    D = S["test"]
    rows: list = []
    cont = spec.replace(C2, CONT, 1)
    continual((cont, CONT), a.seeds, D, scale=0.1, n_eval=30, log_rows=rows, tag="f7k", split="test")
    nbc = int(np.mean([r["bytes_end"] for r in rows if r["bench"] == "continual" and r["model"] == cont]))
    continual((f"erace+bytes={nbc}+iters=3",), a.seeds, D, scale=0.1, n_eval=30, log_rows=rows, tag="f7k",
              split="test")
    cfg = dict(seeds=a.seeds, chosen=cont, chosen_name=best, validation_scores=scores, erace_bytes=nbc,
               reductions="flujo continuo a escala 0,1 con tc=0.1 y presupuesto 500 (como en f0); "
                          "entrenamiento sin las imágenes de validación")
    finish("f7k", rows, {}, cfg)


def _with_mod(spec: str, key: str, value) -> str:
    """spec con la opción `key` (p. ej. "top.replay" o "budget") reemplazada por `value`."""
    mods = [m for m in spec.split("+")[1:] if m.split("=")[0] != key]
    return "+".join([spec.split("+")[0]] + mods + [f"{key}={value}"])


def phase_f7s(a, data):
    """Escalado (regla del proyecto: «límite de hardware» sólo si la exactitud sigue
    subiendo con los recursos): CD con el elegido variando neuronas del almacén, repasos
    por paso y datos por fase (4 escalas cada uno, 3 semillas, test)."""
    D = split_data(data)["test"]
    spec = chosen_spec()
    rows: list = []
    axes = {"neuronas": [("budget", n, None) for n in (250, 500, 1000, 2000)],
            "repasos": [("top.replay", r, None) for r in (0.5, 1, 2, 4)],
            "datos": [(None, n, n) for n in (750, 1500, 3000, 6000)]}
    for axis, pts in axes.items():
        for key, val, steps in pts:
            sp = _with_mod(spec, key, val) if key else spec
            phases = [(cls, steps) for cls, _ in SCENARIOS["CD"]] if steps else None
            for seed in a.seeds:
                def go(sp=sp, seed=seed, phases=phases):
                    r = run_scenario("CD", sp, seed, D, phases=phases)
                    return scenario_row("CD", sp, seed, r, phases)
                row = cached(dict(b="scale", axis=axis, spec=sp, steps=steps, seed=seed, split="test"), go)
                row = dict(row, axis=axis, scale_value=val)
                rows.append(row)
                print(f"[f7s] {axis}={val} s{seed}: " + " ".join(f"{k}={row[k]:.3f}" for k in
                      ("all", "old", "new", "slow_all") if k in row) + f" n={row.get('neurons')}", flush=True)
    keys = ["all", "old", "new", "min", "slow_all", "fast_all", "neurons", "replays", "seconds"]
    summary = agg(rows, ("axis", "scale_value"), keys)
    for d in summary:
        print(f"  {d['axis']:8s} {d['scale_value']:>6} " + fmt(d, ["all", "old", "new", "slow_all", "neurons"]), flush=True)
    cfg = dict(seeds=a.seeds, chosen=spec, axes={k: [p[1] for p in v] for k, v in axes.items()},
               rule="si la curva sigue subiendo en la última escala, el límite es de recursos; si se aplana, "
                    "es de arquitectura",
               note="datos = pasos por fase de CD (750 a 6000); repasos por paso; presupuesto de neuronas del almacén")
    save("f7_escalado", dict(phase="f7s", config=cfg, summary=summary, rows=rows), rows, keys)


def phase_f7c(a, data):
    """Criterio del proyecto: flujo completo de 200 000 pasos, elegido contra ER-ACE con
    los mismos bytes (3 semillas)."""
    D = split_data(data)["test"]
    spec = stream_spec()
    rows: list = []
    continual((spec,), a.seeds, D, scale=1.0, n_eval=30, log_rows=rows, tag="f7c", split="test")
    nb = int(np.mean([r["bytes_end"] for r in rows if r["model"] == spec]))
    continual((f"erace+bytes={nb}+iters=3",), a.seeds, D, scale=1.0, n_eval=30, log_rows=rows, tag="f7c", split="test")
    cfg = dict(seeds=a.seeds, chosen=spec, erace_bytes=nb, scale=1.0, steps=200000,
               protocol="entrenamiento sin las 500 imágenes por clase de validación; evaluación en test "
                        "(30 imágenes por clase y contexto)")
    finish("f7c", rows, {}, cfg)


SCEN_STEPS = {k: sum(p[1] for p in v) for k, v in SCENARIOS.items()}

PHASES = {"f0": phase_f0, "f1": phase_f1, "f1b": phase_f1b, "f2": phase_f2, "f3": phase_f3, "f4": phase_f4, "f5": phase_f5, "f6": phase_f6, "f6s": phase_f6s, "f6cap": phase_f6cap, "f6iid": phase_f6iid, "f6diag": phase_f6diag, "f6fix": phase_f6fix, "f6cap0": phase_f6cap, "f7a": phase_f7a, "f7a2": phase_f7a2, "f7b": phase_f7b, "f7x": phase_f7x, "f7k": phase_f7k, "f7s": phase_f7s, "f7c": phase_f7c}


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    a = ap.parse_args()
    if a.seeds is None:                              # las pruebas de F7 usan semillas no contaminadas
        a.seeds = list(F7_TEST_SEEDS) if a.phase in ("f6s", "f6cap", "f6cap0", "f6iid", "f6diag", "f6fix", "f7b", "f7x", "f7k", "f7s", "f7c") else [0, 1, 2]
    _PHASE["name"] = a.phase
    PHASES[a.phase](a, load_data())


if __name__ == "__main__":
    main()
