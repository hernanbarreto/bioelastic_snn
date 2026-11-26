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
