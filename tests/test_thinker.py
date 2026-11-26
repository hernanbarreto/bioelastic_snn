import numpy as np

from bioelastic.thinker import COPY, FOLLOW, FastMemory, Thinker


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
