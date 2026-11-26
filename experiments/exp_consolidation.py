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


PHASES = {"f0": phase_f0, "f1": phase_f1, "f1b": phase_f1b}


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    PHASES[a.phase](a, load_data())


if __name__ == "__main__":
    main()
