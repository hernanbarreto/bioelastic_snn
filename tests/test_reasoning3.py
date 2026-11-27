"""Ronda 3 de razonamiento: conjuntos de tareas, mapa, back sin daño, neuromodulación."""
import numpy as np

from bioelastic.tasksets import TaskSet, TaskSets
from bioelastic.thinker import Thinker


# ---------------------------------------------------------------- F1: conjuntos de tareas
def test_task_sets_crp_prior_and_sticky_belief():
    T = TaskSets(alpha=1.0, hazard=0.05)
    w = np.zeros((2, 3)); v = np.zeros(3)
    T.add(TaskSet(w, v), cue=0); T.active = 0
    b = T.predict(0)
    assert b[0] > 0.9 and b[-1] < 0.1                       # la causa latente tiende a seguir
    # una consigna nunca vista arranca en el CRP: reutilizar lo popular o crear
    b1 = T.predict(1)
    assert abs(b1.sum() - 1) < 1e-9 and 0.2 < b1[-1] < 0.8
    # evidencia contra el que rige: gana «nuevo»
    for _ in range(4):
        T.predict(0)
        T.update(0, np.array([0.05, 0.5]))
    assert T.choose(0, allow_new=True) == 1
    assert T.choose(0, allow_new=False) == 0              # congelado no se crea nada


def test_task_set_likelihood_ignores_unconfident_predictions():
    T = TaskSets()
    T.add(TaskSet(np.zeros((2, 2)), np.zeros(2)), 0)
    T.sets[0].rho = 0.95
    assert T.likelihood(0, 3, 0.5, 3) == 0.5 == T.likelihood(0, 3, 0.5, 4)   # sin preferencia
    assert T.likelihood(0, 3, 1.0, 3) > 0.9 and T.likelihood(0, 3, 1.0, 4) < 0.1
    assert T.likelihood(0, -1, 1.0, 4) == 0.5                                 # sin predicción


def _harlow(net, rng, objs, family, cue=0, learn=True, T=6):
    good = int(rng.integers(2)); ok = []
    net.reset()
    for t in range(T):
        q = [objs[i] for i in rng.permutation(2)]
        y = objs[good] if family == "W" else objs[(good + t) % 2]
        out = net.answer(q[0], probe=q[1], cue=cue, explore=learn)
        ok.append(out == y)
        net.feedback(target=y)
    return ok


def _families(task_sets, seed=0, n_cues=1, sched=(("W", 0, 150), ("A", 0, 150))):
    rng = np.random.default_rng(seed)
    P = sum(n for _, _, n in sched)
    net = Thinker(2 * (P + 400) + 2, memory=True, speak=True, judge=True, max_steps=10, gate=True,
                  prev_io=True, n_cues=n_cues, task_sets=task_sets, seed=seed)
    objs = rng.permutation(2 * (P + 400))
    i = 0
    for fam, cue, n in sched:
        for _ in range(n):
            _harlow(net, rng, [int(objs[2 * i]), int(objs[2 * i + 1])], fam, cue)
            i += 1
    return net, rng, objs[2 * P:]


def _eval(net, rng, objs, family, cue=0, n=40, off=0):
    return np.array([_harlow(net, rng, [int(objs[2 * (off + j)]), int(objs[2 * (off + j) + 1])], family, cue,
                             learn=False) for j in range(n)])


def test_task_sets_keep_the_first_rule_and_learn_the_second_without_cue():
    """W (quedarse si ganó) y después A (cambiar si ganó), sin consigna: con conjuntos de
    tareas la segunda regla queda en un conjunto nuevo y la primera se conserva; congelado,
    la inferencia vuelve a W en el primer problema."""
    net, rng, objs = _families(dict(hazard=0.03))
    assert len(net.task_sets.sets) >= 2
    net.reset(); net.plastic = False
    W = net.Wc.copy()
    a = _eval(net, rng, objs, "A")
    w = _eval(net, rng, objs, "W", off=40)
    assert a[:, 1].mean() > 0.9 and w[:, 1].mean() > 0.9
    assert w[1:, 1].all()                                   # desde el segundo problema, todo bien
    assert np.array_equal(net.task_sets.sets[net.task_sets.active].Wc, net.Wc)
    assert any(np.array_equal(W, s.Wc) for s in net.task_sets.sets)   # congelado: nada cambió


def test_without_task_sets_the_second_rule_overwrites_the_first():
    net, rng, objs = _families(False)
    net.reset(); net.plastic = False
    w = _eval(net, rng, objs, "W")
    assert w[:, 1].mean() < 0.6


def test_task_sets_transfer_to_a_new_cue_with_the_same_structure():
    """Con consigna: W en la consigna 0. Una consigna nunca vista con la misma estructura
    reutiliza el conjunto (transferencia inmediata, congelado)."""
    net, rng, objs = _families(dict(hazard=0.03), n_cues=3, sched=(("W", 0, 150),))
    k = net.task_sets.active
    net.reset(); net.plastic = False
    w = _eval(net, rng, objs, "W", cue=2)
    assert w[:, 1].mean() > 0.9
    assert net.task_sets.active == k                        # el mismo conjunto, en otra consigna


def test_frozen_inference_never_creates_task_sets():
    net, rng, objs = _families(dict(hazard=0.03), sched=(("W", 0, 120),))
    K = len(net.task_sets.sets)
    net.reset(); net.plastic = False
    _eval(net, rng, objs, "A", n=20)
    assert len(net.task_sets.sets) == K


def test_task_sets_off_is_the_previous_thinker():
    outs = []
    for ts in (False, None):
        rng = np.random.default_rng(3)
        old = dict(back=False, cogmap=False, modulators=False, task_sets=False)   # la v0.3 con prev_io
        kw = old if ts is None else dict(old, task_sets=ts)
        net = Thinker(40, memory=True, speak=True, judge=True, gate=True, prev_io=True, seed=1, **kw)
        objs = rng.permutation(40)
        res = [_harlow(net, rng, [int(objs[2 * (i % 20)]), int(objs[2 * (i % 20) + 1])], "W") for i in range(60)]
        outs.append((res, net.Wc.copy()))
    assert outs[0][0] == outs[1][0] and np.array_equal(outs[0][1], outs[1][1])


# ---------------------------------------------------------------- F2: estructura del mapa
def _events(pairs):
    return [(frozenset([a]), (b,), t) for t, (a, b) in enumerate(pairs)]


def test_map_code_tells_partial_from_complete_maps():
    from bioelastic.cogmap import COMPLETE, NONE, PARTIAL, map_code
    assert map_code([]) == NONE
    assert map_code([(frozenset([1, 2]), (3,), 0)]) == NONE          # sólo relaciones de a un símbolo
    assert map_code(_events([(1, 2), (3, 4)])) == PARTIAL              # dos fragmentos
    assert map_code(_events([(1, 2), (2, 3), (3, 4)])) == COMPLETE     # una cadena
    assert map_code(_events([(1, 2), (2, 3), (3, 1)])) == COMPLETE     # un ciclo
    assert map_code(_events([(1, 2), (1, 3)])) == PARTIAL              # una rama
    assert map_code(_events([(1, 2), (2, 3), (1, 2)])) == COMPLETE     # repetir una relación no ramifica


def test_expected_size_makes_a_short_chain_partial():
    from bioelastic.cogmap import COMPLETE, PARTIAL, SizePrior, map_code
    sp = SizePrior()
    sp.learn(_events([(1, 2), (2, 3), (3, 4), (4, 5)]))
    assert sp.expected() == 5
    sp.learn(_events([(1, 2), (3, 4)]))                                 # un mapa a medio armar no enseña
    assert sp.expected() == 5
    short = _events([(1, 2), (2, 3), (3, 4)])
    assert map_code(short, sp.expected()) == PARTIAL
    assert map_code(_events([(i, i + 1) for i in range(7)]), sp.expected()) == COMPLETE   # más largo: completo


def test_cogmap_state_layout_and_size_learning_only_with_plasticity():
    a = Thinker(20, memory=True, judge=True, speak=True, back=True, cogmap=False, seed=0)
    b = Thinker(20, memory=True, judge=True, speak=True, back=True, cogmap=True, seed=0)
    c = Thinker(20, memory=True, judge=True, speak=True, back=True, cogmap="sin_posicion", seed=0)
    assert b.D_state == a.D_state + 3 + 36 + 12 * (b._n_act + 1) and c.D_state == a.D_state + 3
    for x in (1, 2, 3, 4):
        b.observe(x); b.observe(x + 1); b.observe(None)
    b.answer(5)
    assert b._mapc == 2
    b.plastic = False; b.reset()
    assert b._map_size.expected() == 0                                  # congelado no aprende el tamaño
    b.plastic = True
    for x in (1, 2, 3, 4):
        b.observe(x); b.observe(x + 1); b.observe(None)
    b.reset()
    assert b._map_size.expected() == 5


def test_judge_candidate_learns_a_target_tied_to_a_structural_position():
    """Si respondió la memoria episódica y no había camino, el juicio aprende el objetivo
    para ese estado con marca y captura: FIN (siempre el mismo al final de la cadena) se
    consolida y el modelo interno lo da por fiable; un objetivo que cambia cada vez, no."""
    from bioelastic.thinker import COPY, _key
    # (prueba de unidad del crédito al juicio: sin lo que también actúa en `feedback`)
    net = Thinker(30, memory=True, judge=True, speak=True, back=True, cogmap=True, seed=0,
                  prev_io=False, task_sets=False, modulators=False)
    FIN = 29

    def episode(target):
        net.reset()
        for x in (1, 2, 3):
            net.observe(x); net.observe(x + 1); net.observe(None)
        net.answer(4, explore=False)
        net._path[-1] = COPY                                            # respondió la memoria
        net.feedback(target=target)
        return ("judge", _key(net._base(net._fs)))
    k = episode(10); episode(11); episode(12)
    assert not net.M.predict(k)[1]
    episode(FIN); k = episode(FIN)
    out, sure = net.M.predict(k)
    assert out == FIN and sure


# ---------------------------------------------------------------- F3: back sin daño
def test_predecessors_are_seen_only_after_going_back():
    from bioelastic.thinker import BACK, _recall_back
    for back, always in ((True, False), ("automatico", True)):
        net = Thinker(20, memory=True, judge=True, speak=True, back=back, seed=0)
        for a, b in ((1, 2), (2, 3)):
            net.observe(a); net.observe(b); net.observe(None)
        f = (2,); bk = _recall_back(net.fast.events, f)
        hold = frozenset((1,))
        s0 = net._state(0, -1, 0, f, net._recall(f), hold, back=bk)[0] % net.D_base
        s1 = net._state(0, BACK, 1, f, net._recall(f), hold, back=bk)[0] % net.D_base
        o, k = net.o_back, (5 if back is True else 3)
        seen0 = [i for i in s0 if o <= i < o + k]; seen1 = [i - o for i in s1 if o <= i < o + k]
        assert bool(seen0) == always
        assert 1 in seen1                                                  # un predecesor
        if back is True:
            assert 4 in seen1                                              # y es lo sostenido


def test_going_back_from_the_first_keeps_the_focus_and_marks_the_failure():
    """Retroceder desde algo sin predecesor no vacía el foco: queda, con la marca de que la
    consulta falló (distinta de «retrocedí y llegué al primero»). "automatico" no cambia."""
    from bioelastic.thinker import BACK, FOLLOW, _Stay, _recall_back
    net = Thinker(20, memory=True, judge=True, speak=True, back=True, seed=0)
    for a, b in ((1, 2), (2, 3)):
        net.observe(a); net.observe(b); net.observe(None)
    f = (1,); bk = _recall_back(net.fast.events, f)
    c = net._content(f)
    (stay,) = net._successors(BACK, f, net._recall(f), c, None, bk)
    assert isinstance(stay, _Stay) and tuple(stay) == f
    moved = net._successors(BACK, (2,), net._recall((2,)), net._content((2,)), None, _recall_back(net.fast.events, (2,)))
    assert moved == [(1,)] and not isinstance(moved[0], _Stay)
    mark = net.o_back + 5
    s_stay = net._state(0, BACK, 1, stay, net._recall(f), frozenset(), back=bk)[0] % net.D_base
    s_moved = net._state(0, BACK, 1, moved[0], net._recall(f), frozenset(), back=bk)[0] % net.D_base
    assert mark in s_stay and mark not in s_moved
    assert set(s_stay) - {mark} == set(s_moved)                           # sólo la marca las distingue
    assert net._successors(FOLLOW, stay, net._recall(f), c, None, bk) == [(2,)]   # y se puede seguir desde ahí
    old = Thinker(20, memory=True, judge=True, speak=True, back="automatico", seed=0)
    assert old._successors(BACK, f, old._recall(f), c, None, bk) == []


def _c5_long(net, rng, W):
    import sys
    sys.path.insert(0, "experiments")
    import exp_thinking_hard as H
    L = 8; syms = rng.permutation(W.N); order = [int(x) for x in syms[:L]]
    pairs = [(order[i], order[i + 1]) for i in range(L - 1)]
    i, j = 1, 6                                                        # SÍ, X en el medio, distancia 5
    net.reset(); H._bind(net, pairs, rng)
    return net.answer(order[i], cue=0, probe=order[j], explore=False) == W.YES


def test_transitive_inference_with_back_has_no_positional_shortcut():
    """Con `back` la transitiva (entrenada con órdenes de 3-4) sigue resolviendo los SÍ de
    órdenes de 8: no aprende «si X tiene predecesor, NO»."""
    import sys
    sys.path.insert(0, "experiments")
    import exp_thinking_hard as H
    W = H.World(32)
    rng = np.random.default_rng(1000)
    net = H.make_thinker(W, 1, seed=0, back=True)
    for task, cue in H.plan_single(5, 2000):
        H.TASKS[task](W, net, rng, True, cue)
    net.reset(); net.plastic = False
    assert np.mean([_c5_long(net, rng, W) for _ in range(60)]) > 0.95


# ---------------------------------------------------------------- F4: neuromoduladores
def _bm_recall(m, key):
    from bioelastic.thinker import _recall_events
    idx, val = _recall_events(m.events, key)
    return int(idx[np.argmax(val)]) if len(idx) else -1


def _bm_problem(m, rng, a, b, regime, T=12):
    good = a; ok = []
    m.reset()
    for t in range(T):
        if t:
            ok.append(_bm_recall(m, (a, b)) == good)
        y = (good if rng.random() < 0.7 else (b if good == a else a)) if regime == "S" else good
        m.bind((a, b)); m.bind((y,)); m.bind(())
        if regime == "V" and rng.random() < 0.2:
            good = b if good == a else a
    return np.mean(ok)


def test_bayes_memory_lets_the_last_binding_win_without_evidence_of_noise():
    from bioelastic.modulators import BayesMemory
    m = BayesMemory(20)
    for k, v in ((1, 5), (1, 6), (2, 7)):                                # A->B' y después A->B
        m.bind((k,)); m.bind((v,)); m.bind(())
    assert _bm_recall(m, (1,)) == 6 and _bm_recall(m, (2,)) == 7


def test_bayes_memory_tells_noise_from_volatility():
    """Desenlaces ruidosos y estables (S) contra deterministas y volátiles (V): la memoria
    estima el ruido (ACh) y la volatilidad (NE) y con eso integra en S (como contar) y se
    queda con lo último en V."""
    from bioelastic.modulators import BayesMemory
    rng = np.random.default_rng(0)
    m = BayesMemory(200, prior_q=4.0)
    accS = np.mean([_bm_problem(m, rng, 2 * i, 2 * i + 1, "S") for i in range(40)][10:])
    nS, vS = m.noise(), m.volatility()
    accV = np.mean([_bm_problem(m, rng, 2 * i, 2 * i + 1, "V") for i in range(40, 80)][10:])
    nV, vV = m.noise(), m.volatility()
    assert nS > 0.2 and nV < 0.1 and vV > vS
    assert accS > 0.72 and accV > 0.75                                    # «lo último» en S da ~0.70


def _harlow_net(**kw):
    return Thinker(2 * 1200 + 2, memory=True, speak=True, judge=True, max_steps=10, gate=True,
                   prev_io=True, seed=0, **kw)


def test_neuromodulated_harlow_stays_solved_when_frozen():
    """Harlow sin resultado percibido (`prev_io`) con los canales: congelado en distintos
    momentos del entrenamiento, el ensayo 2 queda resuelto. La oscilación de la ronda anterior
    venía del δ de fin de episodio; con el horizonte de un ensayo quedan caídas raras (2 de 320
    instantáneas en 16 semillas, 27 de 160 con el horizonte del episodio), así que se pide que
    a lo sumo una instantánea de las seis esté por debajo de 0.95."""
    import copy

    def snapshots(**nm):
        net = _harlow_net(modulators=nm or True)
        rng = np.random.default_rng(0)
        objs = rng.permutation(2 * 1200)
        i = 0
        snaps = []
        for p in range(300):
            _harlow(net, rng, [int(objs[2 * i]), int(objs[2 * i + 1])], "W"); i += 1
            if (p + 1) % 50 == 0:
                fz = copy.deepcopy(net); fz.reset(); fz.plastic = False
                acc = []
                for _ in range(30):
                    acc.append(_harlow(fz, rng, [int(objs[2 * i]), int(objs[2 * i + 1])], "W", learn=False)[1]); i += 1
                snaps.append(np.mean(acc))
        return net, np.array(snaps)
    net, snaps = snapshots()
    assert (snaps < 0.95).sum() <= 1 and snaps.mean() >= 0.95, snaps
    assert net._V is None                                                  # horizonte de un ensayo
    assert net.modulators.params["epsilon"] < 0.03                         # nada inesperado: poca exploración


def test_ach_gain_ignores_outcomes_known_to_be_random():
    from bioelastic.modulators import Neuromodulators
    net = _harlow_net(modulators=True)
    nm = net.modulators
    nm.knowledge = False                                                   # (sin consultar el conocimiento: prueba de la señal)
    net.reset(); net.answer(5, probe=6, explore=False)
    net._ls = net._fs
    gains = []
    for t in range(200):                                                   # mismo estado, desenlace al azar
        net._ctl_gain = 1.0
        ok = t % 2 == 0
        net._out = 5
        nm.step(5 if ok else 6, None, (-1, 0.0))
        gains.append(nm.params["lr"] / net.lr)
    assert gains[0] > 0.9 and gains[-1] < 0.2


def test_candidate_ne_is_unexpected_uncertainty_not_noise():
    """La señal de NE candidata (`ne="inesperada"`): fallos por encima de los esperados. Con un
    desenlace ruidoso pero estable (acierta el 70 %) la exploración se apaga; cuando la
    estrategia deja de rendir lo que rendía, sube. La señal por defecto (fiabilidad de la
    regla) queda alta con el ruido. (La candidata no es el valor por defecto: con ella la
    transitiva de la batería deja de resolverse en 1 de 3 semillas.)"""
    rng = np.random.default_rng(0)

    def run(**nm_kw):
        net = _harlow_net(modulators=nm_kw or True)
        nm = net.modulators
        nm.knowledge = False                                               # (sin consultar el conocimiento: prueba de la señal)
        net.reset(); net.answer(5, probe=6, explore=False)
        net._ls = net._fs
        ne = []
        for t in range(700):
            net._ctl_gain = 1.0; net._out = 5
            ok = rng.random() < (0.7 if t < 600 else 0.0)                  # ruido estable y, al final, un cambio
            nm.step(5 if ok else 6, None, (5, 1.0))
            ne.append(nm.params["ne"])
        return np.array(ne)
    ne = run(ne="inesperada", unexplained_gain=0.0)
    assert ne[0] > 0.8                                                     # contexto nuevo: explora
    assert ne[300:600].mean() < 0.2                                        # ruido conocido: no explora
    assert ne[610:640].max() > 0.8                                         # dejó de rendir: explora
    assert ne[-1] < ne[610:640].max()                                      # y lo nuevo pasa a ser lo esperado
    assert run()[300:600].mean() > 0.4                                     # por defecto: sigue explorando


def test_regret_gives_full_gain_only_when_the_alternative_does_better():
    from bioelastic.modulators import Neuromodulators
    net = _harlow_net(modulators=dict(regret=0.15))
    nm = net.modulators
    nm.knowledge = False                                                   # (sin consultar el conocimiento: prueba de la señal)
    net.reset(); net.answer(5, probe=6, explore=False)
    net._ls = net._fs
    for t in range(100):                                                   # la acción elegida acierta la mitad
        net._ctl_gain = 1.0; net._out = 5
        nm.step(5 if t % 2 == 0 else 6, None, (-1, 0.0))
    low = nm.params["lr"] / net.lr
    net._ctl_gain = 1.0; nm.step(6, None, (-1, 0.0), explained=0.75)       # la alternativa rinde 0.75 > 0.5 + 0.15
    high = nm.params["lr"] / net.lr
    net._ctl_gain = 1.0; nm.step(6, None, (-1, 0.0), explained=0.55)       # no rinde bastante más
    assert low < 0.2 and high == 1.0 and nm.params["lr"] / net.lr < 0.2
    off = _harlow_net(modulators=True).modulators
    assert off.regret >= 1.0                                               # por defecto, apagado


# ---------------------------------------------------------------- F6: memoria para lo de afuera
def test_temporal_context_binds_a_sentence_as_a_whole():
    """Con `context` la clave de cada ligadura es todo lo percibido desde el último silencio:
    una oración queda ligada entera y se recupera por cualquier palabra que comparta con la
    pregunta. Con hechos de dos elementos no cambia nada."""
    from bioelastic.thinker import FastMemory
    for ctx, want in ((False, []), (True, [4])):
        m = FastMemory(10); m.context = ctx
        for w in (1, 2, 3, 4):
            m.bind((w,))
        m.bind(())
        got = sorted(int(x) for x, s in zip(*m.recall((9, 1))) if s == 1.0)
        assert (got == [2]) if not ctx else (4 in [int(x) for x in m.recall((9, 1))[0]] and got == want)
    a, b = FastMemory(10), FastMemory(10); b.context = True
    for m in (a, b):
        for x, y in ((1, 2), (3, 4)):
            m.bind((x,)); m.bind((y,)); m.bind(())
    assert [e[:2] for e in a.events] == [e[:2] for e in b.events]


def test_bayes_memory_lets_the_most_recent_fact_win_across_keys():
    """Dos hechos sobre lo mismo con claves distintas («María fue a la cocina», «María se
    movió al baño»): gana el más reciente, como en `FastMemory`. Dentro de una clave sigue
    decidiendo la creencia."""
    net = Thinker(30, memory=True, seed=0, context=True)
    for sent in ((1, 2, 3, 4), (5, 2, 3, 6), (1, 7, 3, 8)):
        for w in sent:
            net.observe(w)
        net.observe(None)
    assert net._best(net.fast.recall((9, 1))) == 8
    assert net._best(net.fast.recall((9, 5))) == 6
