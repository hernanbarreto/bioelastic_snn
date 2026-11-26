"""Dónde no escala el diseño: costo de cada parte al crecer su parámetro de tamaño.

  search     Thinker._search: simulación interna de caminos tras cada error. Cada nodo
             se ramifica en SEGUIR y HABLAR: consultas a la memoria de largo plazo que
             crecen como 2^max_steps (exponencial).
  fastmem    FastMemory: matriz símbolo x símbolo. Memoria y borrado por episodio crecen
             como N^2 (cuadrático en el vocabulario).
  evict      ElasticLayer con presupuesto: cada neurona nueva desaloja a la de menor
             utilidad, que se busca recorriendo todas (O(n) por reclutamiento).
  maintain   poda: recorre todas las neuronas (O(n) cada `maintain_every` pasos).
  forward    pase denso de un área con entrada de imagen (aferentes compartidas por casi
             todas las neuronas): O(n x fan-in) por volea; el índice no lo evita.

  python experiments/exp_limits.py
"""
from __future__ import annotations

import time

import numpy as np

from common import CsvLog
from bioelastic import ElasticLayer
from bioelastic.layer import Volley
from bioelastic.thinker import FastMemory, Thinker


def best_time(fn, repeats=5, number=1) -> float:
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        best = min(best, (time.perf_counter() - t0) / number)
    return best


def search_cost(log):
    """Peor caso: todo símbolo tiene sucesor en la memoria y conocimiento establecido, y
    el objetivo es inalcanzable (la búsqueda recorre todos los niveles)."""
    N = 32
    for steps in (1, 2, 4, 6, 8, 10):
        net = Thinker(N, memory=True, speak=True, max_steps=steps, seed=0)
        f = np.roll(np.arange(N), 3)
        for x in range(N):                                   # conocimiento: f, ya establecido
            for _ in range(3):
                net.reset(); net.answer(x); net.feedback(target=int(f[x]))
        net.reset()
        for x in range(N):                                   # memoria: x -> x + 1
            net.observe(x); net.observe((x + 1) % N); net.observe(None)
        net.answer(0)
        calls = [0]
        peek = net.say.peek

        def counted(*a, **k):
            calls[0] += 1
            return peek(*a, **k)
        net.say.peek = counted
        net._search(-1)
        n_peek = calls[0]
        secs = best_time(lambda: net._search(-1), repeats=3)
        log.row("search", "max_steps", steps, "consultas por búsqueda", n_peek)
        log.row("search", "max_steps", steps, "ms por búsqueda", secs * 1e3)


def fastmem_cost(log):
    for N in (64, 256, 1024, 4096):
        m = FastMemory(N)
        log.row("fastmem", "símbolos", N, "MB de memoria", m.M.nbytes / 1e6)
        log.row("fastmem", "símbolos", N, "us por reset (un episodio)", best_time(m.reset, number=20) * 1e6)
        for t in range(64):
            m.bind((t % N,))
        log.row("fastmem", "símbolos", N, "us por recall", best_time(lambda: m.recall((3,)), number=200) * 1e6)


def evict_cost(log, rng):
    for n in (1_000, 10_000, 100_000):
        L = ElasticLayer(4096, fan_in=16, max_neurons=n, capacity=n)
        idx = np.sort(rng.choice(4096, 16, replace=False)); val = np.ones(16, np.float32)
        v = Volley(idx, val, 4.0, None, None, None, True)
        L.pre[:n] = rng.integers(0, 4096, size=(n, 16)); L.w[:n] = 0.25; L.fill[:n] = 16; L.n = n
        L.util[:n] = rng.random(n, dtype=np.float32); L.reindex()

        def recruit():
            L.tick(); L.recruit(v)
        log.row("evict", "neuronas", n, "us por reclutamiento con presupuesto lleno",
                best_time(recruit, number=50) * 1e6)
        log.row("maintain", "neuronas", n, "us por poda", best_time(L.maintain, number=5) * 1e6)


def forward_cost(log, rng):
    D, act = 784, 150                                         # como una imagen: 150 píxeles activos
    for n in (1_000, 4_000, 16_000):
        L = ElasticLayer(D, fan_in=256, capacity=n)
        for j in range(n):                                    # prototipos con 200 píxeles en el centro
            L.pre[j, :200] = rng.choice(np.arange(150, 650), 200, replace=False)
        w = rng.random((n, 200), dtype=np.float32)
        L.w[:n, :200] = w / np.linalg.norm(w, axis=1, keepdims=True); L.fill[:n] = 200; L.n = n
        L.reindex()
        vols = [(np.sort(rng.choice(np.arange(150, 650), act, replace=False)),
                 rng.random(act, dtype=np.float32) + 0.1) for _ in range(20)]
        secs = best_time(lambda: [L.forward(*v) for v in vols], repeats=3) / len(vols)
        log.row("forward", "neuronas (imagen)", n, "us por volea", secs * 1e6)


def main():
    rng = np.random.default_rng(0)
    log = CsvLog("limits", ["component", "parameter", "value", "metric", "measured"])
    search_cost(log)
    fastmem_cost(log)
    evict_cost(log, rng)
    forward_cost(log, rng)
    log.close()


if __name__ == "__main__":
    main()
