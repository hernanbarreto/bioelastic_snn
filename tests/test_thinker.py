import numpy as np
import pytest

from bioelastic.thinker import COPY, EMIT, FOCUS, FOLLOW, HOLD, JUDGE, SAY, SPEAK, FastMemory, Thinker

# Regresión medida con conjuntos de tareas (por eso no están encendidos por defecto): estas
# pruebas pasan con `task_sets=False` y fallan con ellos. Queda a la vista (xfail estricto: si se
# arregla, la prueba avisa). La de consignas que se contradicen ya pasa con la verosimilitud
# Beta-Bernoulli y la pegajosidad por consigna (F5).
_TS = pytest.mark.parametrize("ts", [False, pytest.param(True, marks=pytest.mark.xfail(strict=True, reason="con conjuntos de tareas (por defecto desde F5) una consigna puede terminar en el conjunto de otra: un conjunto que no predice nada tiene verosimilitud 0.5 por ensayo y gana a uno que aprende con errores confiados, la creencia deriva y el control compartido se sobrescribe (abierto)"))])
_OLD = dict(prev_io=False, back=False, task_sets=False, cogmap=False, modulators=False)   # la v0.3


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


@pytest.mark.xfail(strict=True, reason="con los valores por defecto de F5 (prev_io + moduladores) el conocimiento f "
                   "de esta semilla no queda establecido en 7 pasadas y la búsqueda no encuentra el camino; con cogmap "
                   "encendido sí (en 4 semillas la pasada en que quedan los 16 era la misma que en la v0.3); en curso")
def test_hindsight_blames_strategy_and_protects_knowledge():
    net = Thinker(16, n_cues=2, speak=True, seed=4)
    f = np.roll(np.arange(16), 3)
    # conocimiento: f. Con crédito estricto, los episodios en que el control empezó por
    # HABLAR no escriben nada (el objetivo final no dice qué debía dar el eslabón
    # intermedio): con 3 repeticiones seguidas por símbolo sólo quedaba 1 de 16
    # establecido; con 6 pasadas en orden al azar, los 16 (v0.3). Con los valores por defecto de
    # F5 hacen falta 7 en esta semilla; en 4 semillas, la pasada en que quedan los 16 es en
    # promedio la misma (6.25 con la v0.3 y con F5).
    rng = np.random.default_rng(4)
    for _ in range(7):
        for x in rng.permutation(16):
            net.reset(); net.answer(int(x), cue=0); net.feedback(target=int(f[x]))
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
    """Cada búsqueda expande a lo sumo `budget` nodos. (Por episodio puede haber más de una:
    la retrospectiva, la repetición y, con `modulators`, la que pregunta si el error tenía
    explicación, que la retrospectiva reutiliza.)"""
    rng = np.random.default_rng(6)
    net = Thinker(32, memory=True, speak=True, max_steps=10, budget=8, seed=6)
    inner = net._search_in
    seen = []

    def bounded(*a, **k):
        before = net.expanded
        out = inner(*a, **k)
        seen.append(net.expanded - before)
        return out
    net._search_in = bounded
    for _ in range(200):
        _chain_episode(net, rng, True, int(rng.integers(1, 4)))
    assert seen and max(seen) <= 8


def test_consistent_knowledge_survives_one_contradiction():
    net = Thinker(16, seed=7)
    for _ in range(6):
        for x in range(4):
            net.reset(); net.answer(x); net.feedback(target=x + 8)
    net.reset(); assert net.answer(2) == 10
    net.feedback(target=3)                                 # una contradicción suelta
    x = (np.array([2]), np.ones(1, dtype=np.float32))
    assert net.say.peek(x) == 10


@pytest.mark.parametrize("ts", [False, True])
def test_one_thinker_learns_tasks_in_sequence_without_forgetting(ts):
    """Un solo Thinker, tres consignas entrenadas en bloques (conocimiento, cadenas,
    transitiva). Al final se evalúan las tres en casos no vistos."""
    rng = np.random.default_rng(8)
    f = np.roll(np.arange(32), 7)
    net = Thinker(34, n_cues=3, memory=True, speak=True, judge=True, max_steps=10, seed=8, task_sets=ts)

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


# ---------------------------------------------------------------- ronda 2: consolidación, revisión, repetición
def _f4_episode(net, f, rng, learn, x=None, cue=None):
    cue = int(rng.integers(2)) if cue is None else cue
    x = (int(rng.integers(16)) if cue == 0 else int(rng.integers(8))) if x is None else x
    y = int(f[x]) if cue == 0 else int(f[f[x]])
    net.reset()
    out = net.answer(x, cue=cue, explore=learn)
    net.feedback(target=y)
    return out == y


def test_composition_without_curriculum():
    """f(x) y f(f(x)) mezclados desde el principio (sin currículo). Antes, f(f(x)) se
    memorizaba sobre el mismo contenido que f(x) y lo pisaba: 0/5 semillas. Ahora la
    revisión por explicación deja f como conocimiento y f(f(x)) como estrategia."""
    rng = np.random.default_rng(100)
    f = np.array([5, 6, 10, 11, 8, 4, 7, 0, 1, 15, 2, 3, 13, 14, 9, 12])
    net = Thinker(16, n_cues=2, speak=True, seed=0)
    for _ in range(6000):
        _f4_episode(net, f, rng, True)
    net.reset(); net.plastic = False
    assert np.mean([_f4_episode(net, f, rng, False, x, 1) for x in range(8, 16)]) == 1.0
    assert np.mean([_f4_episode(net, f, rng, False, x, 0) for x in range(16)]) == 1.0
    assert net.revised[1] > 0                              # ganó la hipótesis que explica a la otra


def test_episodic_targets_are_not_consolidated_but_consistent_ones_are():
    """Marca y captura: un objetivo que cambia cada vez para el mismo contenido no se
    escribe en el conocimiento; el mismo objetivo repetido, sí (a la segunda)."""
    net = Thinker(16, seed=11)
    c = (np.array([3]), np.ones(1, dtype=np.float32))
    for t in (8, 9, 10, 11, 12, 13):                       # objetivos episódicos
        net.reset(); net.answer(3); net.feedback(target=t)
    assert net.say.layer.n == 0
    net.reset(); net.answer(3); net.feedback(target=7)
    assert net.say.layer.n == 0                            # una vez: sólo la marca
    net.reset(); net.answer(3); net.feedback(target=7)
    assert net.say.peek(c) == 7                            # dos veces seguidas: se consolida


def test_search_only_returns_paths_a_reactive_control_can_follow():
    """En el mismo estado, la misma acción: HABLAR, HABLAR, EMITIR no es una explicación
    válida si los dos estados tras HABLAR son iguales."""
    net = Thinker(16, n_cues=2, memory=True, speak=True, model="perfecto", seed=12)
    f = np.roll(np.arange(16), 3)
    for x in range(16):                                    # conocimiento f, escrito directo
        c = (np.array([x]), np.ones(1, dtype=np.float32))
        for _ in range(3):
            net.say.forward(c); net.say.teach(int(f[x]))
    assert all(net._know((np.array([x]), np.ones(1, dtype=np.float32)))[1] for x in range(16))
    net.Wc[:] = 0.0
    net.Wc[SAY] = -5.0                                     # el control casi nunca dice: lo barato es hablar y emitir
    net.reset(); net.answer(2, cue=1)
    net.reactive = False
    assert [a for _, _, a in net._search(int(f[f[2]]))[0]] == [SPEAK, SPEAK, EMIT]
    net.reactive = True
    paths = net._search(int(f[f[2]]))
    seq = [a for _, _, a in paths[0]]
    assert seq == [SPEAK, SAY]
    states = [s[0].tobytes() for s, _, _ in paths[0]]
    for i in range(len(states)):
        for j in range(i):
            if states[i] == states[j]:
                assert seq[i] == seq[j]


@_TS
def test_replay_propagates_new_knowledge_to_an_old_task(ts):
    """Orden adverso: f(f(x)) se entrena antes que f. Lo que no se pudo explicar entonces se
    repite cuando aparece f y el control de f(f(x)) se corrige sin volver a entrenarlo."""
    rng = np.random.default_rng(13)
    f = np.array([5, 6, 10, 11, 8, 4, 7, 0, 1, 15, 2, 3, 13, 14, 9, 12])
    net = Thinker(16, n_cues=2, speak=True, seed=13, task_sets=ts)
    for _ in range(1500):
        _f4_episode(net, f, rng, True, cue=1)
    for _ in range(1500):
        _f4_episode(net, f, rng, True, cue=0)
    net.reset(); net.plastic = False
    assert np.mean([_f4_episode(net, f, rng, False, x, 0) for x in range(16)]) == 1.0
    assert np.mean([_f4_episode(net, f, rng, False, x, 1) for x in range(8, 16)]) == 1.0
    assert net.replayed > 0


def test_learned_gate_decides_what_to_hold():
    """Compuerta aprendida (sin `probe` privilegiado): la pregunta «¿X > Y?» llega como dos
    ítems y la red aprende cuál enfocar y cuál sostener. (En 5 semillas generaliza en 4;
    ver experiments/exp_thinking_r2.py.)"""
    rng = np.random.default_rng(0)
    net = Thinker(34, memory=True, judge=True, speak=True, max_steps=10, gate=True, seed=0)
    for _ in range(3000):
        _ti_episode(net, rng, True, int(rng.integers(3, 5)))
    net.reset(); net.plastic = False
    assert np.mean([_ti_episode(net, rng, False, 8, 4) for _ in range(200)]) > 0.97
    held = net.decisions[HOLD]
    assert held > 0 and net.decisions[FOCUS] > 0
    _ti_episode(net, rng, False, 8, 4)
    assert sorted(net._path[:2]) == [FOCUS, HOLD]          # uno al foco y otro al sostén


def test_learned_gate_focuses_single_item_questions():
    rng = np.random.default_rng(1)
    net = Thinker(32, memory=True, gate=True, seed=1)
    for _ in range(3000):
        _chain_episode(net, rng, True, int(rng.integers(1, 3)))
    net.reset(); net.plastic = False
    assert np.mean([_chain_episode(net, rng, False, 4) for _ in range(200)]) > 0.97
    assert net._path[0] == FOCUS


# ---------------------------------------------------------------- ronda 2: pruebas para romperlo
def test_rule_change_midstream_revises_knowledge_and_the_strategy_that_depends_on_it():
    """La regla cambia (f -> g) y sólo se entrena g(x). El conocimiento se revisa (misma
    consigna, contradicción consistente) y la composición, nunca reentrenada, pasa a dar
    g(g(x)) en x no vistos: lo aprendido como estrategia se reutiliza sobre lo revisado."""
    rng = np.random.default_rng(14)
    f = np.array([5, 6, 10, 11, 8, 4, 7, 0, 1, 15, 2, 3, 13, 14, 9, 12])
    g = np.roll(np.arange(16), 7)
    net = Thinker(16, n_cues=2, speak=True, seed=14)
    for i in range(3000):
        _f4_episode(net, f, rng, True, cue=0 if i < 750 else None)
    for _ in range(1500):
        _f4_episode(net, g, rng, True, cue=0)
    net.reset(); net.plastic = False
    assert np.mean([_f4_episode(net, g, rng, False, x, 0) for x in range(16)]) == 1.0
    assert np.mean([_f4_episode(net, g, rng, False, x, 1) for x in range(8, 16)]) == 1.0
    assert net.revised[0] > 0


@_TS
def test_contradicting_cues_without_reconciling_strategy_do_not_corrupt_knowledge(ts):
    """Dos consignas exigen respuestas directas incompatibles (f y g sin relación) sobre el
    mismo contenido. Un único conocimiento compartido no puede sostener las dos: es un
    límite conocido. Lo que se prueba es que no queda basura: cada contenido termina con
    la respuesta de una de las dos consignas."""
    rng = np.random.default_rng(15)
    f = np.roll(np.arange(16), 3); g = np.roll(np.arange(16), 11)
    net = Thinker(16, n_cues=2, speak=True, seed=15, task_sets=ts)
    for _ in range(3000):
        x = int(rng.integers(16)); cue = int(rng.integers(2))
        net.reset(); net.answer(x, cue=cue, explore=True)
        net.feedback(target=int(f[x]) if cue == 0 else int(g[x]))
    outs = [net.say.peek((np.array([x]), np.ones(1, dtype=np.float32))) for x in range(16)]
    assert np.mean([o in (f[x], g[x]) for x, o in enumerate(outs)]) >= 0.9


# ---------------------------------------------------------------- percepción que duda (integración r2)
def _mod_sum_thinker(seed=0, passes=3, **kw):
    """Thinker que sabe (a, 10+b) -> (a+b) mod 10 con símbolos perfectos, establecido."""
    th = Thinker(20, seed=seed, **kw)
    rng = np.random.default_rng(seed)
    for _ in range(passes):
        for i in rng.permutation(100):
            a, b = divmod(int(i), 10)
            th.reset(); th.answer((a, 10 + b)); th.feedback(target=(a + b) % 10)
    return th


def test_input_credit_names_the_misperceived_slot_and_spares_knowledge():
    th = _mod_sum_thinker()
    th.reset()
    assert th.answer((8, 13)) == 1                      # la percepción vio 8 donde había un 3
    blame = th.blame.copy()
    th.feedback(target=6, alternatives=[[3, 5], []])    # segunda opción de cada ranura
    assert th.input_credit == [(0, 3)] and th.blamed_inputs == 1
    assert (th.blame == blame).all()                    # el Thinker no se culpa ni aprende
    assert not th.Ec.any() and not th.ev.any() and th._V is None
    for _ in range(3):                                  # aunque se repita, no se consolida
        th.reset(); th.answer((8, 13)); th.feedback(target=6, alternatives=[[3], []])
    assert th.say.peek((np.array([8, 13]), np.ones(2, dtype=np.float32))) == 1


def test_an_explanation_is_unique_only_if_the_others_are_ruled_out():
    """Si una alternativa no se conoce (nunca vio (5, 13)), «única» sería sólo «la única
    que conozco»: con `explain="refutadas"` (por defecto) no hay crédito; con "unica", sí."""
    for explain, credit in (("refutadas", []), ("unica", [(0, 3)])):
        th = Thinker(20, seed=0, explain=explain)
        rng = np.random.default_rng(0)
        for _ in range(3):
            for i in rng.permutation(100):
                a, b = divmod(int(i), 10)
                if a != 5:
                    th.reset(); th.answer((a, 10 + b)); th.feedback(target=(a + b) % 10)
        th.reset(); th.answer((8, 13))
        th.feedback(target=6, alternatives=[[3, 5], []])
        assert th.input_credit == credit


def test_two_explanations_give_no_input_credit():
    """Con todas las alternativas en las dos ranuras, cambiar cualquiera de las dos explica
    el error: no hay una sola explicación y el Thinker aprende como siempre."""
    th = _mod_sum_thinker()
    th.reset(); th.answer((8, 13))
    blame = th.blame.copy()
    th.feedback(target=6, alternatives=[list(range(10)), [10 + x for x in range(10)]])
    assert th.input_credit == [] and th.blame[1] == blame[1] + 1


def test_doubtful_perception_does_not_teach_the_thinker():
    th = _mod_sum_thinker(doubt=0.5)
    th.reset(); th.answer((8, 13), certainty=0.3)
    tags = dict(th._tags); rel = {k: list(v) for k, v in th.rel.items()}
    th.feedback(target=6)
    assert th.abstained == 1 and th._tags == tags and th.rel == rel and not th.Ec.any()
    th.reset(); th.answer((8, 13), certainty=0.9)       # segura: aprende (marca la contradicción)
    th.feedback(target=6)
    assert th.abstained == 1 and len(th._tags) == len(tags) + 1


def test_episodic_memory_keeps_the_certainty_of_what_it_bound():
    m = FastMemory(10)
    m.bind((1,), 0.9); m.bind((5,), 0.3); m.bind(()); m.bind((2,)); m.bind((6,))
    assert m.certainty((1,)) == 0.3 and m.certainty((2,)) == 1.0 and m.certainty((7,)) == 1.0
    th = Thinker(10, memory=True, seed=0, doubt=0.5)
    th.Wc[COPY, 0] = 5.0                                 # el control copia
    th.reset(); th.observe(1, certainty=0.95); th.observe(5, certainty=0.2); th.observe(None)
    assert th.answer(1, certainty=0.99) == 5 and th._trust == 0.2
    th.feedback(target=5)                                # acertó, pero con un valor dudoso
    assert th.abstained == 1


def _dec_episode(net, rng, learn, n):
    a = rng.integers(10, size=n); b = rng.integers(10, size=n)
    net.reset(); carry = 0; ok = True
    for i in range(n + 1):
        if i < n:
            s = int(a[i]) + int(b[i]) + carry; d, carry = s % 10, s // 10
            out = net.answer((int(a[i]), 10 + int(b[i])), explore=learn)
        else:
            d = carry; out = net.answer(20, explore=learn)
        if learn:
            net.feedback(target=d)
        ok &= out == d
    return ok


def test_one_register_carries_in_decimal_when_its_start_value_is_learned():
    """Suma decimal con un solo registro. Si el registro empieza en 0 (`reg_reset="cero"`)
    y el control adopta la convención 0 = «me llevo una», la primera columna (sin acarreo)
    la contradice y no converge (semilla 0: 0.01 con 6 dígitos). Si el valor inicial lo
    escribe el propio escritor (`"escrito"`), el acarreo sale (1.0). Con 5+5 semillas en
    experiments/exp_integration2.py acarreo. (Con la v0.3; con los valores por defecto de F5
    "cero" también converge en esta semilla: 0.95.)"""
    res = {}
    for reset in ("cero", "escrito"):
        rng = np.random.default_rng(1000)
        net = Thinker(21, registers=1, seed=0, reg_reset=reset, **_OLD)
        for _ in range(3500):
            _dec_episode(net, rng, True, int(rng.integers(1, 5)))
        net.reset(); net.plastic = False
        res[reset] = np.mean([_dec_episode(net, rng, False, 6) for _ in range(60)])
    assert res["cero"] < 0.3 and res["escrito"] > 0.9


def test_input_credit_can_blame_what_was_bound_before_the_question():
    """Memoria episódica: la clave 1 se ligó con un valor que se percibió mal (13 en vez de
    17, con 17 como segunda opción). Al preguntar por 1 el control copia 13; el objetivo es
    17. Sólo la pregunta como hipótesis no explica nada (ninguna otra clave da 17); con lo
    ligado antes, la única explicación es la observación del valor."""
    th = Thinker(20, memory=True, seed=0)
    th.Wc[COPY, 0] = 5.0                                 # el control copia
    th.reset()
    th.observe(1, alternatives=[[2]]); th.observe(13, alternatives=[[17]]); th.observe(None)
    th.observe(4, alternatives=[[5]]); th.observe(15, alternatives=[[16]]); th.observe(None)
    assert th.answer(1) == 13
    th.feedback(target=17, alternatives=[[2]])
    assert th.input_credit == [((1, 0), 17)] and th.blamed_inputs == 1
