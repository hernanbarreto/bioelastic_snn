"""Interferencia entre clases parecidas y pruebas para romper el aprendizaje continuo.

Escenarios cortos y focalizados (cada fase muestra sólo sus clases; las anteriores quedan
ausentes, sin repaso de datos del flujo):

  CD      Fashion C (remera, pantalón, pulóver, vestido) y después D (abrigo, sandalia,
          camisa, zapatilla): abrigo y camisa invaden a remera y pulóver.
  DC      el orden inverso.
  chainF  clases parecidas de a una en orden adverso: pulóver, abrigo, camisa, remera, vestido.
  chainM  dígitos parecidos de a dos en orden adverso: (3, 5), (8, 9), (4, 7).
  shift   cambio de distribución dentro de cada clase: los 10 dígitos, primero los trazos
          finos (poca tinta) y después sólo los gruesos. Se mide si se olvidan los finos.
  joint   referencia: C y D mezclados (lo que se alcanzaría sin interferencia secuencial).
  v1sat   V1 se llena con dígitos y después llegan prendas («new» = prendas); comparar con
  fresh   una red nueva que sólo ve las mismas prendas.

Métricas por corrida (precisión en test, 100 imágenes por clase): por clase al final de
cada fase; «old» = clases de la primera fase, «new» = las de la última, «all» = todas,
«min» = la peor clase al final; además neuronas y bytes.

  python experiments/exp_interference.py --scenarios CD --models d2 d2+episodic --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import CsvLog
from exp_continual import build, load_data
from exp_vision import encoder

SCENARIOS = {
    "CD": [((10, 11, 12, 13), 3000), ((14, 15, 16, 17), 3000)],
    "DC": [((14, 15, 16, 17), 3000), ((10, 11, 12, 13), 3000)],
    "chainF": [((12,), 1200), ((14,), 1200), ((16,), 1200), ((10,), 1200), ((13,), 1200)],
    "chainM": [((3, 5), 2000), ((8, 9), 2000), ((4, 7), 2000)],
    "shift": [(tuple(range(10)), 3000, "thin"), (tuple(range(10)), 3000, "thick")],
    "joint": [((10, 11, 12, 13, 14, 15, 16, 17), 6000)],
    # ¿el diccionario de V1 lleno limita lo nuevo? V1 se llena con dígitos (a los ~1600
    # pasos) y después llegan prendas; se compara con una red nueva que sólo ve prendas.
    "v1sat": [(tuple(range(10)), 4000), ((10, 11, 12, 13), 3000)],
    "fresh": [((10, 11, 12, 13), 3000)],
}
MID = 200                                   # también se evalúa a los MID pasos de cada fase


def ink_split(pools: dict, classes) -> dict:
    """{clase: (índices finos, índices gruesos)} por la mediana de tinta de cada clase."""
    out = {}
    for c in classes:
        ink = pools[c].reshape(len(pools[c]), -1).sum(axis=1)
        med = np.median(ink)
        out[c] = (np.flatnonzero(ink < med), np.flatnonzero(ink >= med))
    return out


def run(scenario: str, model: str, seed: int, data, n_eval: int = 100):
    pools, test = data
    enc = encoder()
    phases = SCENARIOS[scenario]
    classes = sorted({c for ph in phases for c in ph[0]})
    rng = np.random.default_rng([seed, 4242])
    ev = {}
    if scenario == "shift":
        tsplit = ink_split(test, classes); split = ink_split(pools, classes)
        for c in classes:
            for part, ids in zip(("thin", "thick"), tsplit[c]):
                ev[(c, part)] = [enc(test[c][i].astype(np.float32) / 255) for i in rng.choice(ids, n_eval // 2, replace=False)]
    else:
        split = None
        for c in classes:
            ev[(c, "all")] = [enc(test[c][i].astype(np.float32) / 255) for i in rng.choice(len(test[c]), n_eval, replace=False)]
    agent = build(model)(seed)
    agent.on_phase(0)

    def evaluate():
        out = {}
        for key, S in ev.items():
            out[key] = float(np.mean([np.argmax(agent.scores(s)) == key[0] for s in S]))
        return out

    hist = []
    t0 = time.time()
    for k, ph in enumerate(phases):
        cls, steps = ph[0], ph[1]
        part = ph[2] if len(ph) > 2 else None
        for t in range(1, steps + 1):
            c = int(rng.choice(cls))
            if part is None:
                i = int(rng.integers(len(pools[c])))
            else:
                ids = split[c][0 if part == "thin" else 1]
                i = int(ids[rng.integers(len(ids))])
            agent.sup(enc(pools[c][i].astype(np.float32) / 255), c)
            if t == MID and k > 0:
                hist.append((k, t, evaluate()))
        hist.append((k, steps, evaluate()))
    return hist, agent.stats(), time.time() - t0


def summarize(scenario, hist):
    phases = SCENARIOS[scenario]
    final = hist[-1][2]
    first = set(phases[0][0]); last = set(phases[-1][0])
    m = {}
    if scenario == "shift":
        m["old"] = np.mean([v for (c, p), v in final.items() if p == "thin"])
        m["new"] = np.mean([v for (c, p), v in final.items() if p == "thick"])
        m["all"] = np.mean(list(final.values()))
        m["min"] = min(np.mean([final[(c, p)] for p in ("thin", "thick")]) for c in phases[0][0])
        after1 = [h for h in hist if h[0] == 0][-1][2]
        m["drop"] = np.mean([after1[(c, "thin")] - final[(c, "thin")] for c in phases[0][0]])
        return m
    acc = {c: v for (c, _), v in final.items()}
    m["old"] = np.mean([acc[c] for c in first])
    m["new"] = np.mean([acc[c] for c in last])
    m["all"] = np.mean(list(acc.values()))
    m["min"] = min(acc.values())
    # olvido: máximo previo - final, sobre las clases que no están en la última fase
    prev = {c: max(h[2][(c, "all")] for h in hist[:-1]) for c in acc if c not in last} if len(hist) > 1 else {}
    m["drop"] = np.mean([prev[c] - acc[c] for c in prev]) if prev else 0.0
    m["mid_old"] = np.mean([hist[1][2][(c, "all")] for c in first]) if len(hist) > 2 else np.nan
    for c in (10, 12):
        if c in acc:
            m[f"c{c}"] = acc[c]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["CD"])
    ap.add_argument("--models", nargs="+", default=["d2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    data = load_data()
    keys = ["old", "new", "all", "min", "drop", "mid_old", "c10", "c12"]
    log = CsvLog(f"interference{a.tag}", ["scenario", "model", "seed"] + keys + ["neurons", "bytes", "seconds"])
    detail = CsvLog(f"interference{a.tag}_classes", ["scenario", "model", "seed", "phase", "step", "class", "part", "acc"])
    for sc in a.scenarios:
        for model in a.models:
            for seed in a.seeds:
                hist, st, sec = run(sc, model, seed, data)
                for k, t, acc in hist:
                    for (c, part), v in sorted(acc.items()):
                        detail._w.writerow([sc, model, seed, k, t, c, part, f"{v:.4f}"])   # sin imprimir
                detail._f.flush()
                m = summarize(sc, hist)
                log.row(sc, model, seed, *[float(m.get(k, np.nan)) for k in keys], st["neurons"], st["bytes"], sec)
    log.close(); detail.close()


if __name__ == "__main__":
    main()
