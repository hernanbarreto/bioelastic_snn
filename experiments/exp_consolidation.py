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
                 timing: bool = False, on_step=None, factory=None, extra_evals=()):
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
    if timing:
        out["latency"] = latency(lat_f, lat_t)
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


def interference(scenarios, specs, seeds, data, *, tag: str, log_rows: list, phases=None, **kw):
    for sc in scenarios:
        for spec in specs:
            for seed in seeds:
                r = run_scenario(sc, spec, seed, data, phases=phases, **kw)
                row = scenario_row(sc, spec, seed, r, phases)
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


def continual(specs, seeds, data, *, scale: float, n_eval: int, log_rows: list, tag: str):
    for spec in specs:
        for seed in seeds:
            r = continual_run(spec, seed, scale, n_eval, data)
            m = continual_metrics(r)
            row = dict(bench="continual", scenario=f"x{scale:g}", model=spec, seed=seed)
            row.update({k: float(m[k]) for k in CONT_KEYS if k in m})
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


def drift_row(spec, seed, r) -> dict:
    """Cambio de etiqueta (10 <-> 12) después de aprender C: ¿se actualiza lo que cambió
    sin destruir lo vecino (11, 13)?"""
    h0 = r["hist"][0][2]; hm = r["hist"][1][2]; hf = r["hist"][-1][2]
    row = dict(bench="drift", scenario="swap10_12", model=spec, seed=seed)
    for part in ("full", "slow", "fast"):
        if not hf[part]:
            continue
        row[f"{part}_swapped_end"] = float(np.mean([hf[part][c] for c in (10, 12)]))
        row[f"{part}_swapped_mid"] = float(np.mean([hm[part][c] for c in (10, 12)]))
        row[f"{part}_kept_end"] = float(np.mean([hf[part][c] for c in (11, 13)]))
        row[f"{part}_kept_drop"] = float(np.mean([h0[part][c] - hf[part][c] for c in (11, 13)]))
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


SCEN_STEPS = {k: sum(p[1] for p in v) for k, v in SCENARIOS.items()}

PHASES = {"f0": phase_f0, "f1": phase_f1, "f1b": phase_f1b, "f2": phase_f2, "f3": phase_f3, "f4": phase_f4, "f5": phase_f5}


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    PHASES[a.phase](a, load_data())


if __name__ == "__main__":
    main()
