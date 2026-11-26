"""Batería de razonamiento: tareas que no se resuelven memorizando ni reconociendo.

Cada prueba se evalúa sobre casos que la red nunca vio y que sólo salen bien si
*compone* lo aprendido. Se comparan la red actual y prototipos de los mecanismos que
le faltan (armados acá con el mismo núcleo, todavía fuera del paquete).

  R1 ligadura     4 pares clave->valor nuevos en cada episodio; al final se pregunta una
                  clave. Memorizar no sirve: las ligaduras cambian siempre.
  R2 composición  una función f sobre 16 símbolos. Se pregunta f(f(x)) para x nunca
                  preguntados a dos saltos, y f(f(f(x))), que nunca se entrenó.
  R3 longitud     paridad de una secuencia de bits. Se entrena con largos 1-6 y se
                  evalúa con largos 10-20 y 100.

  python experiments/exp_reasoning.py
"""
from __future__ import annotations

import numpy as np

from common import CsvLog
from bioelastic import Cortex

TEST = 1000


def frame(*idx):
    return np.array(idx, dtype=np.int64), np.ones(len(idx), dtype=np.float32)


class SuccessorMemory:
    """Memoria rápida de un episodio: liga cada símbolo con el que vino después (Hebb de
    un disparo) y, ante un símbolo, recupera lo que lo siguió antes. Es la operación de
    una «induction head»: [A][B] ... [A] -> [B]."""

    def __init__(self, n: int):
        self.M = np.zeros((n, n), dtype=bool)
        self.prev = None

    def reset(self):
        self.M[:] = False
        self.prev = None

    def step(self, x: int) -> np.ndarray:
        recalled = np.flatnonzero(self.M[:, x])
        if self.prev is not None:
            self.M[x, self.prev] = True
        self.prev = x
        return recalled


# ---------------------------------------------------------------- R1 ligadura en contexto
def r1_episode(rng):
    keys = rng.permutation(8)[:4]
    vals = rng.integers(8, size=4)
    q = int(rng.integers(4))
    seq = [t for k, v in zip(keys, vals) for t in (int(k), 8 + int(v))] + [16, int(keys[q])]
    return seq, int(vals[q])


def r1(log):
    rng = np.random.default_rng(0)
    net = Cortex(17, 8, heads=[dict(context=12, delay=0.05), dict(tau=16.0)])

    def run(learn):
        seq, y = r1_episode(rng)
        net.reset_context()
        for s in seq[:-1]:
            net.forward(s)
        return (net.step(seq[-1], y) if learn else net.predict(seq[-1])) == y
    for _ in range(4000):
        run(True)
    log.row("R1 ligadura", "red actual (ventana + traza)", "claves con ligaduras nuevas",
            float(np.mean([run(False) for _ in range(TEST)])), 1 / 8)

    rng = np.random.default_rng(0)
    net = Cortex(34, 8, heads=[{}])
    mem = SuccessorMemory(17)

    def run_fast(learn):
        seq, y = r1_episode(rng)
        mem.reset()
        for s in seq[:-1]:
            mem.step(s)
        x = frame(seq[-1], *(17 + mem.step(seq[-1])))
        return (net.step(x, y) if learn else net.predict(x)) == y
    for _ in range(4000):
        run_fast(True)
    log.row("R1 ligadura", "+ memoria rápida de sucesores", "claves con ligaduras nuevas",
            float(np.mean([run_fast(False) for _ in range(TEST)])), 1 / 8)


# ---------------------------------------------------------------- R2 composición
def r2(log):
    rng = np.random.default_rng(1)
    while True:
        f = rng.permutation(16)
        if not np.any(f == np.arange(16)):
            break
    f2, f3 = f[f], f[f[f]]
    held = np.arange(8, 16)                                   # nunca preguntados a dos saltos

    net = Cortex(19, 16, heads=[dict(context=2, delay=0.2)])

    def ask(x, marker, y, learn):
        net.reset_context()
        net.forward(int(x))
        return (net.step(marker, int(y)) if learn else net.predict(marker)) == y
    for _ in range(8000):
        if rng.random() < 0.5:
            x = rng.integers(16); ask(x, 16, f[x], True)
        else:
            x = rng.integers(8); ask(x, 17, f2[x], True)
    log.row("R2 composición", "red actual", "f(f(x)), x no vistos a 2 saltos",
            float(np.mean([ask(x, 17, f2[x], False) for x in held])), 1 / 16)
    log.row("R2 composición", "red actual", "f(f(f(x))), nunca entrenado",
            float(np.mean([ask(x, 18, f3[x], False) for x in range(16)])), 1 / 16)

    net = Cortex(16, 16, heads=[{}])                          # sólo aprende un salto
    for _ in range(3000):
        x = int(rng.integers(16)); net.step(x, int(f[x]))

    def hops(x, k):
        for _ in range(k):
            x = net.predict(int(x))                           # la salida vuelve a entrar
        return x
    log.row("R2 composición", "+ bucle (la salida reentra)", "f(f(x)), x no vistos a 2 saltos",
            float(np.mean([hops(x, 2) == f2[x] for x in held])), 1 / 16)
    log.row("R2 composición", "+ bucle (la salida reentra)", "f(f(f(x))), nunca entrenado",
            float(np.mean([hops(x, 3) == f3[x] for x in range(16)])), 1 / 16)
    f10 = np.arange(16)
    for _ in range(10):
        f10 = f[f10]
    log.row("R2 composición", "+ bucle (la salida reentra)", "10 saltos, nunca entrenado",
            float(np.mean([hops(x, 10) == f10[x] for x in range(16)])), 1 / 16)


# ---------------------------------------------------------------- R3 generalización en longitud
def r3(log):
    rng = np.random.default_rng(2)
    net = Cortex(3, 2, heads=[dict(context=24, delay=0.03), dict(tau=32.0)])

    def run(n, learn):
        bits = rng.integers(2, size=n)
        y = int(bits.sum() % 2)
        net.reset_context()
        for b in bits:
            net.forward(int(b))
        return (net.step(2, y) if learn else net.predict(2)) == y
    for _ in range(6000):
        run(int(rng.integers(1, 7)), True)
    log.row("R3 longitud", "red actual (ventana + traza)", "largos 10-20 (entrenó con 1-6)",
            float(np.mean([run(int(rng.integers(10, 21)), False) for _ in range(TEST)])), 1 / 2)

    for name, every_step in (("+ bucle, sólo respuesta final", False),
                             ("+ bucle, pasos intermedios supervisados", True)):
        rng = np.random.default_rng(2)
        net = Cortex(4, 2, heads=[{}])

        def loop(n, learn):
            bits = rng.integers(2, size=n)
            state = 0; true = 0
            for i, b in enumerate(bits):
                true ^= int(b)
                pred = net.forward(frame(int(b), 2 + state))  # entra el bit y el estado propio
                if learn and (every_step or i == n - 1):
                    net.teach(true)
                state = true if (learn and every_step) else pred
            return pred == true
        for _ in range(6000):
            loop(int(rng.integers(1, 7)), True)
        log.row("R3 longitud", name, "largos 10-20 (entrenó con 1-6)",
                float(np.mean([loop(int(rng.integers(10, 21)), False) for _ in range(TEST)])), 1 / 2)
        log.row("R3 longitud", name, "largo 100",
                float(np.mean([loop(100, False) for _ in range(200)])), 1 / 2)


def main():
    log = CsvLog("reasoning", ["test", "system", "evaluated_on", "accuracy", "chance"])
    r1(log); r2(log); r3(log)
    log.close()


if __name__ == "__main__":
    main()
