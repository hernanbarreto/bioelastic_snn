"""Razonamiento con control aprendido: el Thinker contra pruebas donde mi código no decide nada.

En todas, la red recibe sólo la percepción y, al responder, el objetivo (o una
recompensa). Cuántos pasos pensar, cuándo usar la memoria y qué guardar en los
registros lo aprende ella. Se evalúa en casos que nunca vio.

  T1 ligadura    32 claves, 32 valores. Cada episodio liga 4 claves con valores nuevos.
                 La mitad de las preguntas son sobre esas claves (hay que recordar el
                 episodio) y la otra mitad sobre claves no ligadas (vale el dato
                 permanente f(clave), aprendido a lo largo de los episodios).
  T2 cadenas     hechos sueltos A->B, B->C, ... en desorden, más distractores. Se pregunta
                 A y la respuesta es el final de la cadena. Entrena con 1-2 saltos,
                 evalúa con 3 y 4.
  T3 suma        suma binaria con acarreo, dígito a dígito. Entrena con 1-6 dígitos,
                 evalúa con 10-20 y 100. El acarreo nunca se enseña.
  T4 composición f sobre 16 símbolos. «f(x)» se entrena para todo x, «f(f(x))» sólo para
                 x de 0 a 7; se evalúa f(f(x)) con x de 8 a 15. Currículo: el primer cuarto
                 del entrenamiento es sólo f(x); después, mezclado.

  python experiments/exp_thinking.py [episodios_por_tarea]
"""
from __future__ import annotations

import sys

import numpy as np

from common import CsvLog
from bioelastic import Cortex
from bioelastic.thinker import Thinker

TEST = 1000


def train_eval(net, episode, n_train, n_test, rng):
    for _ in range(n_train):
        episode(net, rng, True)
    net.reset()
    net.plastic = False
    acc = float(np.mean([episode(net, rng, False) for _ in range(n_test)]))
    net.plastic = True
    return acc


# ---------------------------------------------------------------- T1 ligadura
K, V = 32, 32
F1 = K + np.random.default_rng(10).permutation(V)              # dato permanente de cada clave


def t1_episode(net, rng, learn, bound_query=None):
    keys = rng.permutation(K)[:5]
    vals = K + rng.integers(V, size=4)
    net.reset()
    for k, v in zip(keys[:4], vals):
        net.observe(int(k)); net.observe(int(v)); net.observe(None)
    bound = rng.random() < 0.5 if bound_query is None else bound_query
    q = int(keys[rng.integers(4)]) if bound else int(keys[4])
    y = int(vals[list(keys[:4]).index(q)]) if bound else int(F1[q])
    out = net.answer(q, explore=learn)
    net.feedback(target=y)
    return out == y


def t1_cortex_episode(net, rng, learn, bound_query=None):
    keys = rng.permutation(K)[:5]
    vals = K + rng.integers(V, size=4)
    net.reset_context()
    for k, v in zip(keys[:4], vals):
        net.forward(int(k)); net.forward(int(v)); net.forward(K + V)
    bound = rng.random() < 0.5 if bound_query is None else bound_query
    q = int(keys[rng.integers(4)]) if bound else int(keys[4])
    y = int(vals[list(keys[:4]).index(q)]) if bound else int(F1[q])
    return (net.step(q, y) if learn else net.predict(q)) == y


# ---------------------------------------------------------------- T2 cadenas
N2 = 64


def t2_episode(net, rng, learn, hops=None):
    hops = int(rng.integers(1, 3)) if hops is None else hops
    syms = rng.permutation(N2)
    chain = syms[:hops + 1]
    pairs = [(int(chain[i]), int(chain[i + 1])) for i in range(hops)]
    rest = syms[hops + 1:]
    pairs += [(int(rest[2 * i]), int(rest[2 * i + 1])) for i in range(4)]   # distractores
    net.reset()
    for i in rng.permutation(len(pairs)):
        a, b = pairs[i]
        net.observe(a); net.observe(b); net.observe(None)
    out = net.answer(int(chain[0]), explore=learn)
    net.feedback(target=int(chain[-1]))
    return out == int(chain[-1])


# ---------------------------------------------------------------- T3 suma con acarreo
END = 4


def t3_episode(net, rng, learn, n=None, per_digit=False):
    n = int(rng.integers(1, 7)) if n is None else n
    a = rng.integers(2, size=n); b = rng.integers(2, size=n)
    net.reset()
    carry = 0; ok = []
    for i in range(n + 1):
        if i < n:
            s = int(a[i]) + int(b[i]) + carry
            digit, carry = s % 2, s // 2
            out = net.answer((int(a[i]), 2 + int(b[i])), explore=learn)
        else:
            digit = carry
            out = net.answer(END, explore=learn)
        net.feedback(target=digit)
        ok.append(out == digit)
    return float(np.mean(ok)) if per_digit else all(ok)


def t3_cortex_episode(net, rng, learn, n=None):
    n = int(rng.integers(1, 7)) if n is None else n
    a = rng.integers(2, size=n); b = rng.integers(2, size=n)
    net.reset_context()
    carry = 0; ok = []
    for i in range(n + 1):
        if i < n:
            s = int(a[i]) + int(b[i]) + carry
            digit, carry = s % 2, s // 2
            x = (np.array([int(a[i]), 2 + int(b[i])]), np.ones(2, dtype=np.float32))
        else:
            digit, x = carry, END
        ok.append((net.step(x, digit) if learn else net.predict(x)) == digit)
    return all(ok)


# ---------------------------------------------------------------- T4 composición
def make_f(seed=1):
    rng = np.random.default_rng(seed)
    while True:
        f = rng.permutation(16)
        if not np.any(f == np.arange(16)):
            return f


F4 = make_f()


def t4_episode(net, rng, learn, x=None, cue=None):
    if cue is None:
        cue = int(rng.integers(2))
    if x is None:
        x = int(rng.integers(16)) if cue == 0 else int(rng.integers(8))
    y = int(F4[x]) if cue == 0 else int(F4[F4[x]])
    net.reset()
    out = net.answer(x, cue=cue, explore=learn)
    net.feedback(target=y)
    return out == y


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    seeds = (0, 1, 2)
    log = CsvLog("thinking", ["test", "system", "evaluated_on", "accuracy_mean", "accuracy_min",
                              "seeds", "chance"])

    def row(test, system, on, accs, chance):
        log.row(test, system, on, float(np.mean(accs)), float(np.min(accs)), len(accs), chance)

    def thinker_runs(make, episode, train_kw_fn, evals):
        """Entrena una red por semilla y evalúa cada caso; devuelve {caso: [acc por semilla]}."""
        out = {name: [] for name, *_ in evals}
        for seed in seeds:
            rng = np.random.default_rng(100 + seed)
            net = make(seed)
            for i in range(n):
                episode(net, rng, True, **train_kw_fn(i))
            net.reset(); net.plastic = False
            for name, fn, count in evals:
                out[name].append(float(np.mean([fn(net, rng) for _ in range(count)])))
        return out

    # T1 ligadura
    for bq, name in ((True, "claves ligadas en el episodio"), (False, "claves no ligadas (dato permanente)")):
        accs = []
        for seed in seeds:
            rng = np.random.default_rng(100 + seed)
            net = Cortex(K + V + 1, K + V, heads=[dict(context=12, delay=0.05), dict(tau=16.0)], seed=seed)
            for _ in range(n):
                t1_cortex_episode(net, rng, True)
            accs.append(float(np.mean([t1_cortex_episode(net, rng, False, bq) for _ in range(TEST)])))
        row("T1 ligadura", "red v0.2 (ventana + traza)", name, accs, 1 / V)
    evals = [("claves ligadas en el episodio", lambda net, r: t1_episode(net, r, False, True), TEST),
             ("claves no ligadas (dato permanente)", lambda net, r: t1_episode(net, r, False, False), TEST)]
    for system, kw in (("thinker sin memoria", dict()), ("thinker", dict(memory=True))):
        res = thinker_runs(lambda s: Thinker(K + V, seed=s, **kw), t1_episode, lambda i: {}, evals)
        for name, accs in res.items():
            row("T1 ligadura", system, name, accs, 1 / V)

    # T2 cadenas
    evals = [(f"{h} saltos" + ("" if h <= 2 else " (nunca entrenado)"),
              (lambda h: lambda net, r: t2_episode(net, r, False, h))(h), TEST) for h in (1, 2, 3, 4)]
    res = thinker_runs(lambda s: Thinker(N2, memory=True, seed=s), t2_episode, lambda i: {}, evals)
    for name, accs in res.items():
        row("T2 cadenas", "thinker", name, accs, 1 / N2)

    # T3 suma con acarreo
    evals = [("1-6 dígitos, suma exacta", lambda net, r: t3_episode(net, r, False, int(r.integers(1, 7))), TEST),
             ("10-20 dígitos, suma exacta", lambda net, r: t3_episode(net, r, False, int(r.integers(10, 21))), TEST),
             ("100 dígitos, suma exacta", lambda net, r: t3_episode(net, r, False, 100), 200)]
    for system, R in (("thinker sin registros", 0), ("thinker", 1)):
        res = thinker_runs(lambda s: Thinker(5, registers=R, seed=s), t3_episode, lambda i: {}, evals)
        for (name, accs), chance in zip(res.items(), (2.0 ** -2, 2.0 ** -11, 2.0 ** -101)):
            row("T3 suma", system, name, accs, chance)
    accs = []
    for seed in seeds:
        rng = np.random.default_rng(100 + seed)
        net = Cortex(5, 2, heads=[dict(context=8, delay=0.1), dict(tau=8.0)], seed=seed)
        for _ in range(n):
            t3_cortex_episode(net, rng, True)
        accs.append(float(np.mean([t3_cortex_episode(net, rng, False, int(rng.integers(10, 21)))
                                   for _ in range(TEST)])))
    row("T3 suma", "red v0.2 (ventana + traza)", "10-20 dígitos, suma exacta", accs, 2.0 ** -11)

    # T4 composición (currículo: el primer cuarto, sólo f(x))
    evals = [("f(f(x)), x de 8 a 15 (no vistos)", lambda net, r: np.mean([t4_episode(net, r, False, x, 1) for x in range(8, 16)]), 1),
             ("f(x), todos", lambda net, r: np.mean([t4_episode(net, r, False, x, 0) for x in range(16)]), 1)]
    for system, kw in (("thinker sin habla interna", dict()), ("thinker", dict(speak=True))):
        res = thinker_runs(lambda s: Thinker(16, n_cues=2, seed=s, **kw), t4_episode,
                           lambda i: dict(cue=0 if i < n // 4 else None), evals)
        for name, accs in res.items():
            row("T4 composición", system, name, accs, 1 / 16)
    log.close()


if __name__ == "__main__":
    main()
