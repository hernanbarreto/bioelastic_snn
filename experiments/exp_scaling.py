"""Escalado: memoria real y cómputo cuando la red crece de 10^5 a 10^7 neuronas.

A. Lámina de columnas elásticas independientes (1000 neuronas, fan-in 32, D=1024). En
   cada paso todas las columnas reciben una volea propia (32 aferentes activas) y
   resuelven su competencia TTFS. *Medido*: el costo de una columna, sobre una muestra
   de hasta 2000 columnas al azar de toda la lámina. *Extrapolado*: el costo del paso
   completo (costo por columna x columnas). Memoria: lo que reserva cada columna
   (sinapsis, estadísticas por neurona, buffers de tamaño D) y el RSS del proceso.

B. Un área sola que crece (una columna de n neuronas). Aunque la volea tenga siempre
   32 aferentes activas, el pase denso cuesta n x fan-in: el costo por volea crece con la
   red. Con el índice invertido cuesta según las neuronas tocadas, que también crecen
   con n si D es fijo ("compartida"); sólo quedan acotadas si cada neurona toma sus
   aferentes de una vecindad y D crece con la red ("topográfica").

Costo por sinapsis (ns) y costo por paso no son lo mismo: el primero es constante, el
segundo crece con la red. Con eso se extrapola a la corteza humana (bioelastic/blueprint.py).

  python experiments/exp_scaling.py [max_neuronas_A] [max_neuronas_B]
"""
from __future__ import annotations

import resource
import sys
import time

import numpy as np

from common import CsvLog
from bioelastic import ElasticLayer
from bioelastic.blueprint import HUMAN, NOT_DEMONSTRATED, fmt, plan

COL, FAN_IN, D, ACTIVE = 1000, 32, 1024, 32


def populate(L: ElasticLayer, rng, n: int, D_: int, local: bool = False):
    """Columna ya poblada con memorias aleatorias (lo que se mide es el costo, no la tarea).
    FAN_IN aferentes distintas por neurona; con `local`, tomadas de una ventana de 1024
    alrededor de la posición de la neurona (conectividad topográfica)."""
    if local:
        start = (np.arange(n) * (D_ / n)).astype(np.int64)[:, None]
        L.pre[:n] = (start + rng.permutation(1024)[None, :FAN_IN]) % D_
    else:
        start = rng.integers(0, D_, size=(n, 1))
        L.pre[:n] = (start + rng.permutation(D_)[None, :FAN_IN]) % D_
    w = rng.random((n, FAN_IN), dtype=np.float32)
    L.w[:n] = w / np.linalg.norm(w, axis=1, keepdims=True)
    L.fill[:n] = FAN_IN
    L.n = n
    L.reindex()
    return L


def best_time(fn, items, repeats=3) -> float:
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        for it in items:
            fn(*it)
        best = min(best, (time.perf_counter() - t0) / len(items))
    return best


def sheet(n_max: int, rng):
    log = CsvLog("scaling", ["neurons", "columns", "synapses", "synapse_bytes", "memory_bytes",
                             "bytes_per_neuron", "rss_bytes", "measured_columns", "seconds_per_column",
                             "seconds_per_step", "syn_per_core_second", "ns_per_synapse",
                             "fixed_us_per_column"])
    tiny = populate(ElasticLayer(D, fan_in=FAN_IN, capacity=1), rng, 1, D)
    vol = [(np.sort(rng.choice(D, ACTIVE, replace=False)), rng.random(ACTIVE, dtype=np.float32) * 0.9 + 0.1)
           for _ in range(256)]
    fixed = best_time(tiny.forward, vol)                       # costo fijo (Python) de un pase
    cols: list[ElasticLayer] = []
    rate = None
    for n_total in [n for n in (100_000, 300_000, 1_000_000, 3_000_000, 10_000_000) if n <= n_max]:
        while len(cols) * COL < n_total:
            cols.append(populate(ElasticLayer(D, fan_in=FAN_IN, capacity=COL), rng, COL, D))
        sample = cols if len(cols) <= 2000 else [cols[i] for i in rng.choice(len(cols), 2000, replace=False)]
        best = np.inf
        for _ in range(3):                                    # mejor de 3
            t0 = time.perf_counter()
            for i, L in enumerate(sample):
                L.forward(*vol[i & 255])
            best = min(best, (time.perf_counter() - t0) / len(sample))
        step = best * len(cols)                               # extrapolado si la muestra < columnas
        syn = len(cols) * COL * FAN_IN
        mem = sum(L.memory()["total"] for L in cols)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        rate = syn / step
        log.row(len(cols) * COL, len(cols), syn, sum(L.nbytes() for L in cols), mem,
                float(mem / (len(cols) * COL)), rss, len(sample), float(best), float(step), float(rate),
                float(step / syn * 1e9), float(fixed * 1e6))
    log.close()
    return rate


def single_area(n_max: int, rng):
    log = CsvLog("scaling_area", ["connectivity", "neurons", "afferents", "active", "touched_fraction",
                                  "us_dense", "us_index", "ns_per_synapse_dense", "memory_bytes"])
    marginal = None
    for local in (False, True):
        for n in [n for n in (1_000, 10_000, 100_000, 300_000, 1_000_000) if n <= n_max]:
            D_ = max(4096, int(n * 1.024)) if local else D
            L = populate(ElasticLayer(D_, fan_in=FAN_IN, capacity=n), rng, n, D_, local)
            if local:                                         # la volea cae en una vecindad
                base = int(rng.integers(0, D_ - 1024))
                vol = [(np.sort(base + rng.choice(1024, ACTIVE, replace=False)),
                        rng.random(ACTIVE, dtype=np.float32) * 0.9 + 0.1) for _ in range(30)]
            else:
                vol = [(np.sort(rng.choice(D_, ACTIVE, replace=False)),
                        rng.random(ACTIVE, dtype=np.float32) * 0.9 + 0.1) for _ in range(30)]
            L.SPARSE_MIN_NEURONS = 10 ** 12
            dense = best_time(L.forward, vol)
            L.SPARSE_MIN_NEURONS = 1; L.SPARSE_MIN_SYNAPSES = 1; L.SPARSE_MAX_TOUCHED = np.inf
            L.SPARSE_MAX_ACTIVE = 1.0
            index = best_time(L.forward, vol)
            ptr = L._ix[0]
            touched = np.mean([len(np.unique(np.concatenate([L._ix[1][ptr[a]:ptr[a + 1]] for a in v[0]])))
                               for v in vol[:5]]) / n
            log.row("topográfica" if local else "compartida", n, D_, ACTIVE, float(touched),
                    float(dense * 1e6), float(index * 1e6), float(dense / (n * FAN_IN) * 1e9),
                    L.memory()["total"])
            if not local and n >= 100_000:
                marginal = dense / (n * FAN_IN)
            del L
    log.close()
    return marginal


def main():
    n_sheet = int(float(sys.argv[1])) if len(sys.argv) > 1 else 3_000_000
    n_area = int(float(sys.argv[2])) if len(sys.argv) > 2 else 300_000
    rng = np.random.default_rng(0)
    rate = sheet(n_sheet, rng)
    marginal = single_area(n_area, rng)

    human = CsvLog("human_scale", ["scale", "neurons", "synapses", "memory", "synapse_memory",
                                   "core_seconds_per_step", "core_seconds_per_step_marginal",
                                   "cores_for_real_time"])
    for scale in (1e-6, 1e-4, 1e-2, 1.0):
        p = plan(scale, syn_per_core_second=rate, index=True, slack=1.5, steps_per_second=100)
        q = plan(scale, syn_per_core_second=1.0 / marginal) if marginal else p
        human.row(f"{scale:g}", fmt(p["neurons"]), fmt(p["synapses"]), fmt(p["memory_total_bytes"], "B"),
                  fmt(p["memory_bytes"], "B"), fmt(p["core_seconds_per_step"]),
                  fmt(q["core_seconds_per_step"]), fmt(p["cores_for_real_time"]))
    human.close()
    print(f"corteza humana: {fmt(HUMAN['neurons_cortex'])} neuronas")
    print("medido: costo por columna y por sinapsis en un núcleo; extrapolado: el resto. "
          "Supuestos: índice, reserva 1.5x, 100 pasos por segundo simulado, todas las columnas activas.")
    for line in NOT_DEMONSTRATED:
        print(" no demostrado:", line)


if __name__ == "__main__":
    main()
