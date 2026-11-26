"""Escalado: memoria y cómputo por sinapsis cuando la red crece de 10^5 a 10^7 neuronas.

Una lámina de columnas elásticas independientes (1000 neuronas, fan-in 32 cada una). En
cada paso todas las columnas reciben una volea propia y resuelven su competencia TTFS.
Se mide si el costo por sinapsis depende del tamaño total. Con eso se extrapola a la
corteza humana (bioelastic/blueprint.py).

  python experiments/exp_scaling.py [max_neuronas]
"""
from __future__ import annotations

import resource
import sys
import time

import numpy as np

from common import CsvLog
from bioelastic import ElasticLayer
from bioelastic.blueprint import HUMAN, fmt, plan

COL, FAN_IN, D, ACTIVE = 1000, 32, 1024, 32


def make_column(rng) -> ElasticLayer:
    """Columna ya poblada con memorias aleatorias (lo que se mide es el costo, no la tarea)."""
    L = ElasticLayer(D, fan_in=FAN_IN, capacity=COL)
    start = rng.integers(0, D, size=(COL, 1))                # FAN_IN aferentes distintas por neurona
    L.pre[:] = (start + rng.permutation(D)[None, :FAN_IN]) % D
    w = rng.random((COL, FAN_IN), dtype=np.float32)
    L.w[:] = w / np.linalg.norm(w, axis=1, keepdims=True)
    L.fill[:] = FAN_IN
    L.n = COL
    return L


def main():
    n_max = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10_000_000
    rng = np.random.default_rng(0)
    log = CsvLog("scaling", ["neurons", "columns", "synapses", "synapse_bytes", "rss_bytes",
                             "seconds_per_step", "syn_per_core_second", "ns_per_synapse"])
    cols: list[ElasticLayer] = []
    sizes = [n for n in (100_000, 300_000, 1_000_000, 3_000_000, 10_000_000) if n <= n_max]
    for n_total in sizes:
        while len(cols) * COL < n_total:
            cols.append(make_column(rng))
        volleys = [(np.sort(rng.choice(D, ACTIVE, replace=False)),
                    rng.random(ACTIVE, dtype=np.float32) * 0.9 + 0.1) for _ in range(256)]
        sample = cols if len(cols) <= 2000 else [cols[i] for i in rng.choice(len(cols), 2000, replace=False)]
        best = np.inf
        for _ in range(3):                                    # mejor de 3: columnas al azar de toda la lámina
            t0 = time.perf_counter()
            for i, L in enumerate(sample):
                L.forward(*volleys[i & 255])
            best = min(best, (time.perf_counter() - t0) / len(sample))
        step = best * len(cols)                               # un paso = todas las columnas
        syn = len(cols) * COL * FAN_IN
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        log.row(len(cols) * COL, len(cols), syn, sum(L.nbytes() for L in cols), rss,
                float(step), float(syn / step), float(step / syn * 1e9))
    log.close()

    rate = syn / step
    human = CsvLog("human_scale", ["scale", "neurons", "synapses", "memory", "core_seconds_per_step"])
    for scale in (1e-6, 1e-4, 1e-2, 1.0):
        p = plan(scale, syn_per_core_second=rate)
        human.row(f"{scale:g}", fmt(p["neurons"]), fmt(p["synapses"]), fmt(p["memory_bytes"], "B"),
                  fmt(p["core_seconds_per_step"]))
    human.close()
    print(f"corteza humana: {fmt(HUMAN['neurons_cortex'])} neuronas")


if __name__ == "__main__":
    main()
