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
        kw = {} if ts is None else dict(task_sets=ts)
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
    a = Thinker(20, memory=True, judge=True, speak=True, back=True, seed=0)
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
    net = Thinker(30, memory=True, judge=True, speak=True, back=True, cogmap=True, seed=0)
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
