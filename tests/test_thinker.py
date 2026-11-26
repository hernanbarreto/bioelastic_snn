import numpy as np

from bioelastic.thinker import COPY, EMIT, FOLLOW, JUDGE, FastMemory, Thinker


def test_fast_memory_binds_successors_and_silence_breaks_adjacency():
    m = FastMemory(10)
    for t in (1, 2, None, 3, 4, None, 1, 5):
        m.bind(() if t is None else (t,))
    idx, val = m.recall((1,))
    assert set(idx) == {2, 5} and val[list(idx).index(5)] == 1.0     # el más reciente, más fuerte
    assert len(m.recall((2,))[0]) == 0                                # el silencio cortó 2 -> 3
    assert m.has_successor(3) and not m.has_successor(4)


def _bind_episode(net, rng, learn, bound=None):
    keys = rng.permutation(8)[:5]
    vals = 8 + rng.integers(8, size=4)
    net.reset()
    for k, v in zip(keys[:4], vals):
        net.observe(int(k)); net.observe(int(v)); net.observe(None)
    bound = rng.random() < 0.5 if bound is None else bound
    q = int(keys[rng.integers(4)]) if bound else int(keys[4])
    y = int(vals[list(keys[:4]).index(q)]) if bound else 8 + (q * 3) % 8
    out = net.answer(q, explore=learn)
    net.feedback(target=y)
    return out == y


def test_learns_when_to_use_episodic_memory_and_when_long_term_knowledge():
    rng = np.random.default_rng(0)
    net = Thinker(16, memory=True, seed=0)
    for _ in range(1500):
        _bind_episode(net, rng, True)
    net.reset(); net.plastic = False
    assert np.mean([_bind_episode(net, rng, False, True) for _ in range(200)]) > 0.98
    assert np.mean([_bind_episode(net, rng, False, False) for _ in range(200)]) > 0.98
    assert net.decisions[COPY] > 0


def _chain_episode(net, rng, learn, hops):
    syms = rng.permutation(32)
    chain = syms[:hops + 1]
    pairs = [(int(chain[i]), int(chain[i + 1])) for i in range(hops)]
    pairs += [(int(syms[-2 * i - 1]), int(syms[-2 * i - 2])) for i in range(3)]
    net.reset()
    for i in rng.permutation(len(pairs)):
        net.observe(pairs[i][0]); net.observe(pairs[i][1]); net.observe(None)
    out = net.answer(int(chain[0]), explore=learn)
    net.feedback(target=int(chain[-1]))
    return out == int(chain[-1])


def test_chains_longer_than_any_seen_in_training():
    rng = np.random.default_rng(1)
    net = Thinker(32, memory=True, seed=1)
    for _ in range(6000):
        _chain_episode(net, rng, True, int(rng.integers(1, 3)))
    net.reset(); net.plastic = False
    for hops in (3, 4):
        assert np.mean([_chain_episode(net, rng, False, hops) for _ in range(300)]) > 0.97
    assert net.decisions[FOLLOW] > 0


def _add_episode(net, rng, learn, n):
    a = rng.integers(2, size=n); b = rng.integers(2, size=n)
    net.reset()
    carry = 0; ok = True
    for i in range(n + 1):
        if i < n:
            s = int(a[i]) + int(b[i]) + carry
            digit, carry = s % 2, s // 2
            out = net.answer((int(a[i]), 2 + int(b[i])), explore=learn)
        else:
            digit = carry
            out = net.answer(4, explore=learn)
        net.feedback(target=digit)
        ok &= out == digit
    return ok


def test_carry_is_discovered_in_a_register_and_generalizes_in_length():
    for R, expect in ((0, False), (1, True)):
        rng = np.random.default_rng(2)
        net = Thinker(5, registers=R, seed=2)
        for _ in range(2500):
            _add_episode(net, rng, True, int(rng.integers(1, 7)))
        net.reset(); net.plastic = False
        acc = np.mean([_add_episode(net, rng, False, 50) for _ in range(50)])
        assert (acc == 1.0) if expect else (acc < 0.2)


def test_inner_speech_composes_knowledge_on_unseen_inputs():
    rng = np.random.default_rng(3)
    f = np.roll(np.arange(16), 5)
    net = Thinker(16, n_cues=2, speak=True, seed=3)

    def ep(x, cue, learn):
        y = int(f[x]) if cue == 0 else int(f[f[x]])
        net.reset()
        out = net.answer(int(x), cue=cue, explore=learn)
        net.feedback(target=y)
        return out == y
    for i in range(4000):
        cue = 0 if i < 1000 else int(rng.integers(2))
        ep(int(rng.integers(16)) if cue == 0 else int(rng.integers(8)), cue, True)
    net.reset(); net.plastic = False
    assert np.mean([ep(x, 1, False) for x in range(8, 16)]) == 1.0
    assert np.mean([ep(x, 0, False) for x in range(16)]) == 1.0


def test_hindsight_blames_strategy_and_protects_knowledge():
    net = Thinker(16, n_cues=2, speak=True, seed=4)
    f = np.roll(np.arange(16), 3)
    for x in range(16):                                    # conocimiento: f
        for _ in range(3):
            net.reset(); net.answer(x, cue=0); net.feedback(target=int(f[x]))
    net.Wc[:] = 0.0                                        # control sin experiencia: dice directo
    before = net.blame.copy()
    net.reset(); out = net.answer(2, cue=1)                # estrategia ingenua para f(f(x))
    assert out == int(f[2])
    net.feedback(target=int(f[f[2]]))
    assert net.blame[0] == before[0] + 1                   # error atribuido a la estrategia
    x = (np.array([2]), np.ones(1, dtype=np.float32))
    assert net.say.peek(x) == int(f[2])                    # el conocimiento no se pisó


# ---------------------------------------------------------------- mecanismos nuevos
def test_fast_memory_recall_is_conjunctive():
    m = FastMemory(20)
    for e, r, x in ((0, 2, 10), (0, 3, 11), (1, 2, 12), (1, 3, 10)):
        m.bind((e, r)); m.bind((x,)); m.bind(())
    # 10 quedó ligado a 0 (evento 1) y a 3 (evento 4) pero nunca a la conjunción (0, 3):
    # sumando ligaduras sueltas empataría con 11 y ganaría por ser más reciente.
    assert list(m.recall((0, 3))[0]) == [11]
    assert list(m.recall((0, 2))[0]) == [10]
    assert list(m.recall((1, 2))[0]) == [12]


def _ti_episode(net, rng, learn, L, min_gap=1):
    YES, NO = 32, 33
    syms = rng.permutation(32)
    order = [int(s) for s in syms[:L]]
    pairs = [(order[i], order[i + 1]) for i in range(L - 1)]
    pairs += [(int(syms[L + 2 * i]), int(syms[L + 2 * i + 1])) for i in range(3)]
    while True:
        i, j = (int(t) for t in rng.choice(L, 2, replace=False))
        if abs(i - j) >= min_gap:
            break
    net.reset()
    for k in rng.permutation(len(pairs)):
        net.observe(pairs[k][0]); net.observe(pairs[k][1]); net.observe(None)
    y = YES if i < j else NO
    out = net.answer(order[i], probe=order[j], explore=learn)
    net.feedback(target=y)
    return out == y


def test_transitive_inference_on_unseen_distances():
    """Premisas X>Y de un orden nuevo en cada episodio; entrena órdenes de 3-4, evalúa
    órdenes de 8 con distancias >= 4. El juicio sale del estado (¿lo recordado es lo
    sostenido?), no de los símbolos."""
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        net = Thinker(34, memory=True, judge=True, max_steps=10, seed=seed)
        for _ in range(1500):
            _ti_episode(net, rng, True, int(rng.integers(3, 5)))
        net.reset(); net.plastic = False
        assert np.mean([_ti_episode(net, rng, False, 8, 4) for _ in range(200)]) > 0.97
        assert net.decisions[JUDGE] > 0


def test_chains_generalize_without_clock_or_lookahead():
    """Sin reloj absoluto ni «¿el recuerdo sigue?»: seguir hasta que la memoria no
    devuelva nada y emitir el foco. Entrena 1-2 saltos y evalúa 6."""
    rng = np.random.default_rng(5)
    net = Thinker(32, memory=True, max_steps=8, seed=5)
    assert not net.clock and not net.lookahead
    for _ in range(3000):
        _chain_episode(net, rng, True, int(rng.integers(1, 3)))
    net.reset(); net.plastic = False
    assert np.mean([_chain_episode(net, rng, False, 6) for _ in range(300)]) > 0.97
    assert net.decisions[EMIT] > 0


def test_search_is_bounded_by_budget():
    rng = np.random.default_rng(6)
    net = Thinker(32, memory=True, speak=True, max_steps=10, budget=8, seed=6)
    for _ in range(200):
        before = net.expanded
        _chain_episode(net, rng, True, int(rng.integers(1, 4)))
        assert net.expanded - before <= 8


def test_consistent_knowledge_survives_one_contradiction():
    net = Thinker(16, seed=7)
    for _ in range(6):
        for x in range(4):
            net.reset(); net.answer(x); net.feedback(target=x + 8)
    net.reset(); assert net.answer(2) == 10
    net.feedback(target=3)                                 # una contradicción suelta
    x = (np.array([2]), np.ones(1, dtype=np.float32))
    assert net.say.peek(x) == 10


def test_one_thinker_learns_tasks_in_sequence_without_forgetting():
    """Un solo Thinker, tres consignas entrenadas en bloques (conocimiento, cadenas,
    transitiva). Al final se evalúan las tres en casos no vistos."""
    rng = np.random.default_rng(8)
    f = np.roll(np.arange(32), 7)
    net = Thinker(34, n_cues=3, memory=True, speak=True, judge=True, max_steps=10, seed=8)

    def know(learn):
        x = int(rng.integers(32)); net.reset()
        out = net.answer(x, cue=0, explore=learn); net.feedback(target=int(f[x]))
        return out == f[x]

    def chain(learn, hops):
        syms = rng.permutation(32); c = [int(s) for s in syms[:hops + 1]]
        pairs = [(c[i], c[i + 1]) for i in range(hops)] + [(int(syms[-1 - 2 * i]), int(syms[-2 - 2 * i])) for i in range(3)]
        net.reset()
        for k in rng.permutation(len(pairs)):
            net.observe(pairs[k][0]); net.observe(pairs[k][1]); net.observe(None)
        out = net.answer(c[0], cue=1, explore=learn); net.feedback(target=c[-1])
        return out == c[-1]

    def ti(learn, L, gap):
        YES, NO = 32, 33
        syms = rng.permutation(32); o = [int(s) for s in syms[:L]]
        while True:
            i, j = (int(t) for t in rng.choice(L, 2, replace=False))
            if abs(i - j) >= gap:
                break
        net.reset()
        for k in rng.permutation(L - 1):
            net.observe(o[k]); net.observe(o[k + 1]); net.observe(None)
        y = YES if i < j else NO
        out = net.answer(o[i], cue=2, probe=o[j], explore=learn); net.feedback(target=y)
        return out == y

    for _ in range(800):
        know(True)
    for _ in range(1500):
        chain(True, int(rng.integers(1, 4)))
    for _ in range(1500):
        ti(True, int(rng.integers(3, 5)), 1)
    net.reset(); net.plastic = False
    assert np.mean([know(False) for _ in range(200)]) > 0.97
    assert np.mean([chain(False, 6) for _ in range(200)]) > 0.97
    assert np.mean([ti(False, 8, 4) for _ in range(200)]) > 0.97


def test_repeated_consistent_contradiction_revises_knowledge():
    """Una contradicción suelta no pisa lo establecido; la misma corrección repetida sí."""
    net = Thinker(16, seed=9)
    for _ in range(6):
        for x in range(4):
            net.reset(); net.answer(x); net.feedback(target=x + 8)
    x = (np.array([1]), np.ones(1, dtype=np.float32))
    for target in (2, 5, 3):                               # objetivos que cambian: ruido episódico
        net.reset(); net.answer(1); net.feedback(target=target)
    assert net.say.peek(x) == 9
    for _ in range(12):                                    # el mismo objetivo, repetido
        net.reset(); net.answer(1); net.feedback(target=4)
    assert net.say.peek(x) == 4
