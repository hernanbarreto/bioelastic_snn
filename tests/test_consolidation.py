"""Consolidación, corrección y refuerzo: dos sistemas (CLS), esquemas, reconsolidación,
autoexamen espaciado y sueño."""
import numpy as np

from bioelastic import BioElasticNet, Cortex, Readout


def _blob(rng, c):
    img = 0.05 * rng.random((8, 8, 1)).astype(np.float32)
    y, x = divmod(c, 2)
    img[y * 4:y * 4 + 4, x * 4:x * 4 + 4, 0] += rng.uniform(0.6, 1.0, (4, 4)).astype(np.float32)
    return np.clip(img, 0, 1)


def _cortex(**kw):
    return Cortex((8, 8, 1), 4, areas=[dict(rf=3, pool=2, max_features=16, theta=0.8)],
                  heads=[dict(source="input", episodic=True)], projection=dict(source="all"),
                  **{"replay": 1.0, "joint": False, **kw})


# ---------------------------------------------------------------- F1: dos sistemas
def test_episode_records_label_time_context_and_follow_swap_remove():
    rng = np.random.default_rng(0)
    D = 30
    net = BioElasticNet(D, 3, episodic=True, layer=dict(theta=0.99, fan_in=8))
    for i in range(280):                               # casi todas novedosas: crece la capacidad
        net.context_tag = i % 5
        x = np.zeros(D, dtype=np.float32); x[rng.choice(D, 10, replace=False)] = rng.uniform(0.2, 1.0, 10)
        net.step(x, i % 3)
    L, ep = net.layer, net.ep
    n = L.n
    assert n > 256 and len(ep["y"]) >= n
    assert set(np.unique(ep["y"][:n])) <= {0, 1, 2} and (ep["ctx"][:n] >= 0).all()
    e = net.episodes[:n].astype(np.float32)
    assert np.allclose(ep["norm"][:n], np.linalg.norm(e, axis=1) / 255.0, rtol=1e-5)
    last = {k: v[n - 1] for k, v in ep.items()}
    L.remove(2)                                        # la última pasa a la 2, con sus campos
    assert all(ep[k][2] == last[k] for k in ep)
    back = BioElasticNet.from_state(net.state())
    for k in ep:
        assert np.array_equal(back.ep[k][:L.n], ep[k][:L.n])


def test_stored_label_does_not_drift_with_the_readout():
    net = BioElasticNet(10, 3, episodic=True)
    x = np.zeros(10, dtype=np.float32); x[[1, 4, 7]] = [1.0, 0.8, 0.6]
    net.step(x, 0)
    net.readout.W[:, 0] = [0.0, 0.0, 2.0]               # la lectura derivó hacia la clase 2
    assert net.labels(stored=False)[0] == 2 and net.labels(stored=True)[0] == 0


def test_recall_returns_the_most_similar_episodes():
    rng = np.random.default_rng(1)
    D = 40
    net = BioElasticNet(D, 4, episodic=True, layer=dict(theta=0.95))
    X = []
    for i in range(40):
        x = np.zeros(D, dtype=np.float32); x[rng.choice(D, 8, replace=False)] = rng.uniform(0.3, 1.0, 8)
        X.append(x); net.step(x, i % 4)
    q = X[7]; idx = np.flatnonzero(q); j7 = int(np.flatnonzero(net.ep["t"][:net.layer.n] == 8)[0])
    ids, sim = net.recall(idx, q[idx], k=5)
    assert ids[0] == j7 and sim[0] > 0.99 and np.all(np.diff(sim) <= 1e-7)


def test_gate_consults_the_store_only_when_the_slow_system_is_unsure():
    rng = np.random.default_rng(3)
    net = _cortex(recall="gate", recall_margin=0.5)
    for _ in range(300):
        c = int(rng.integers(4)); net.step(_blob(rng, c), c)
    sure, unsure = 0, 0
    for _ in range(60):
        c = int(rng.integers(4))
        r0 = net.recalls
        net.forward(_blob(rng, c))
        parts = net.parts()
        S = parts["slow"]
        e = np.exp((S - S.max()) / net.temperature); q = np.sort(e / e.sum())
        if q[-1] - q[-2] >= 0.5:                       # seguro: responde sólo el lento
            assert net.recalls == r0 and np.array_equal(net._V, S); sure += 1
        else:                                          # inseguro: lento + episodios recuperados
            assert net.recalls == r0 + 1
            assert np.allclose(net._V, S + parts["fast"]) and np.isclose(parts["fast"].sum(), 1.0)
            unsure += 1
    assert sure > 0
    always = _cortex(recall="always")
    for _ in range(50):
        c = int(rng.integers(4)); always.step(_blob(rng, c), c)
    assert always.recalls >= 49                        # (en el primer paso el almacén está vacío)


def test_recall_off_is_the_default_and_leaves_inference_unchanged():
    rng = np.random.default_rng(4)
    a, b = _cortex(), _cortex(recall="off", stored_labels=False)
    for _ in range(150):
        c = int(rng.integers(4)); x = _blob(rng, c)
        assert a.step(x, c) == b.step(x, c)
    assert np.array_equal(a._V, b._V) and a.recalls == 0


def test_replay_with_stored_labels_uses_the_label_of_the_episode():
    rng = np.random.default_rng(5)
    net = _cortex(stored_labels=True)
    for _ in range(200):
        c = int(rng.integers(4)); net.step(_blob(rng, c), c)
    mem = net.heads[0]
    n = mem.layer.n
    mem.readout.W[:, :n] = mem.readout.W[::-1, :n]      # todas las lecturas derivaron
    seen = []
    learn = net.proj.learn
    net.proj.learn = lambda idx, h, c, p=None, replay=False, **kw: (seen.append(c) if replay else None,
                                                                   learn(idx, h, c, p=p, replay=replay, **kw))
    for _ in range(40):
        net._replay_one()
    assert seen and set(seen) <= set(np.unique(mem.ep["y"][:mem.layer.n]))
    assert set(np.unique(mem.labels(True))) == set(np.unique(mem.ep["y"][:mem.layer.n]))


def test_slow_system_can_learn_from_its_own_error():
    rng = np.random.default_rng(17)
    nets = {m: _cortex(replay=0.0, slow_error=m) for m in ("joint", "own")}
    xs = [(_blob(rng, c), int(c)) for c in rng.integers(4, size=120)]
    for net in nets.values():
        for x, c in xs[:100]:
            net.step(x, c)
    own = nets["own"]
    x, c = xs[100]
    own.forward(x)
    S = own.proj.potentials(*own._pv)
    e = np.exp((S - S.max()) / own.temperature)
    ref = own.proj.W.copy()
    ref[:, own._pv[0]] += (own.proj.lr * (np.eye(4)[c] - e / e.sum()))[:, None].astype(np.float32) * own._pv[1][None, :]
    own.teach(c)
    assert np.allclose(own.proj.W, ref, atol=1e-6)      # error del lento solo, sin el voto de la memoria
    j = nets["joint"]
    assert not np.array_equal(j.proj.W, own.proj.W)


def test_cortex_with_two_systems_saves_and_loads(tmp_path):
    rng = np.random.default_rng(6)
    net = _cortex(recall="gate", stored_labels=True)
    for _ in range(120):
        c = int(rng.integers(4)); net.step(_blob(rng, c), c)
    net.save(tmp_path / "n.npz")
    back = Cortex.load(tmp_path / "n.npz")
    assert back.recall == "gate" and back.stored_labels
    x = _blob(rng, 2)
    assert back.forward(x) == net.forward(x) and np.allclose(back._V, net._V)


# ---------------------------------------------------------------- F2: esquemas
def test_schema_measure_has_no_effect_and_fit_is_a_fraction():
    rng = np.random.default_rng(7)
    a, b = _cortex(), _cortex(schema="measure", schema_known=5)
    fits = []
    for _ in range(200):
        c = int(rng.integers(4)); x = _blob(rng, c)
        assert a.step(x, c) == b.step(x, c)
        fits.append(b.fit)
    assert np.array_equal(a.proj.W, b.proj.W) and a.heads[0].layer.n == b.heads[0].layer.n
    f = np.array(fits[50:])
    assert np.all((f >= 0) & (f <= 1)) and b.reliability > 0.5 and b.assimilated == 0
    cov, rel, conf = b.fit_parts
    assert 0 <= cov <= 1 and 0 <= conf <= 1


def test_schema_gate_assimilates_fitting_items_without_neurogenesis():
    rng = np.random.default_rng(8)
    net = _cortex(schema="gate", schema_fit=0.3, schema_boost=4.0, schema_known=5)
    for _ in range(300):
        c = int(rng.integers(3)); net.step(_blob(rng, c), c)     # esquema: clases 0-2
    assert net.reliability > 0.6 and net.assimilated > 0
    mem = net.heads[0]
    # un ítem que encaja: se asimila con tasa x4 y no crea neurona
    for _ in range(50):
        c = int(rng.integers(3)); x = _blob(rng, c)
        net.forward(x)
        W0 = net.proj.W.copy(); n0 = mem.layer.n; a0 = net.assimilated
        replay, net.replay = net.replay, 0.0                       # aislar el aprendizaje directo
        net.teach(c)
        net.replay = replay
        if net.assimilated > a0:
            assert mem.layer.n == n0
            break
    else:
        raise AssertionError("ningún ítem encajó")
    assert np.abs(net.proj.W - W0).sum() > 0                     # el lento sí aprendió (x4, ver abajo)
    # sin esquema fiable no se asimila: la clase nueva queda en el almacén
    net.reliability = 0.0
    n0 = mem.layer.n; a0 = net.assimilated
    for _ in range(20):
        net.step(_blob(rng, 3), 3)
    assert net.assimilated == a0 and mem.layer.n > n0


def test_lr_scale_multiplies_the_readout_update():
    idx = np.array([0, 1]); h = np.array([0.6, 0.8], dtype=np.float32)
    a, b = Readout(3, consolidation=0.0), Readout(3, consolidation=0.0)
    for r in (a, b):
        r.add_input(cls=0); r.add_input(cls=1)
    p = np.array([0.5, 0.3, 0.2])
    a.learn(idx, h, 2, p=p); b.learn(idx, h, 2, p=p, lr_scale=4.0)
    W0 = np.zeros((3, 2), dtype=np.float32); W0[0, 0] = 1.0; W0[1, 1] = 1.0
    assert np.allclose(b.W[:, :2] - W0, 4.0 * (a.W[:, :2] - W0), atol=1e-6)
    c = Readout(3, consolidation=0.0); c.add_input(cls=0)
    c.learn(np.array([0]), np.ones(1, dtype=np.float32), 1, lr_scale=0.0)
    assert c.W[1, 0] == 0.0 and c.t == 1                         # sin cambio, pero cuenta como actividad


def test_adaptive_separation_relaxes_with_class_use():
    a = np.zeros(60, dtype=np.float32); a[:20] = 1.0
    b = a.copy(); b[14:20] = 0.0; b[20:26] = 1.0
    rng = np.random.default_rng(0)
    nets = {r: BioElasticNet(60, 2, separation=0.05, separation_relax=r, layer=dict(theta=0.6))
            for r in (0.0, 5.0)}
    for net in nets.values():
        for _ in range(20):
            net.step(np.clip(a + 0.05 * rng.random(60), 0, 1).astype(np.float32), 0)
        net._uses[1] = 50.0                            # la clase 1 ya está establecida
        n0 = net.layer.n
        net.step(np.clip(b + 0.05 * rng.random(60), 0, 1).astype(np.float32), 1)
        net.vig_new = net.layer.vig[n0] if net.layer.vig is not None and net.layer.n > n0 else 0.0
    assert nets[5.0].vig_new < nets[0.0].vig_new


# ---------------------------------------------------------------- F3: reconsolidación
def _consolidated(lock=5.0, **kw):
    r = Readout(3, consolidation=0.0, recon_lock=lock, **kw)
    for c in range(3):
        r.add_input(cls=c)
    h = np.ones(1, dtype=np.float32)
    for _ in range(int(lock) + 3):                     # cada entrada aprende su clase hasta bloquearse
        for c in range(3):
            r.learn(np.array([c]), h, c)
    assert (r.count[:3] >= lock).all()
    return r


def test_reconsolidation_without_prediction_error_changes_nothing():
    r = _consolidated()
    W0 = r.W.copy(); o0 = r.opened
    idx = np.array([0]); h = np.ones(1, dtype=np.float32)
    for _ in range(20):                                # recuperación correcta y sin sorpresa
        p = r.probs(r.potentials(idx, h))
        assert int(np.argmax(p)) == 0
        r.learn(idx, h, 0)
    assert np.array_equal(r.W, W0) and r.opened == o0 and r.blocked >= 20


def test_reconsolidation_with_error_updates_only_the_retrieved_memory_and_relocks():
    r = _consolidated(recon_window=1, recon_rate=1.0)
    W0 = r.W.copy()
    idx = np.array([0]); h = np.ones(1, dtype=np.float32)
    r.learn(idx, h, 1)                                 # error: la entrada 0 predice 0, el objetivo es 1
    assert r.opened == 1 and r.relearned == 1
    d = r.W - W0
    assert d[1, 0] > 0 and d[0, 0] < 0 and d[2, 0] == 0.0   # sólo filas del error (objetivo y predicha)
    assert np.array_equal(r.W[:, 1:], W0[:, 1:])       # lo vecino (otras entradas) no cambia
    W1 = r.W.copy()
    p = r.probs(r.potentials(idx, h))
    if int(np.argmax(p)) == 1:                         # si ya predice 1 y sin sorpresa: re-consolidada
        r.learn(idx, h, 1)
        assert np.array_equal(r.W, W1)


def test_free_inputs_learn_normally_and_surprise_opens_the_window():
    r = Readout(2, consolidation=0.0, recon_lock=3.0, recon_kappa=1.5)
    r.add_input(cls=0)
    h = np.ones(1, dtype=np.float32); idx = np.array([0])
    plain = Readout(2, consolidation=0.0); plain.add_input(cls=0)
    for _ in range(3):                                 # antes del bloqueo: igual que sin reconsolidación
        r.learn(idx, h, 0); plain.learn(idx, h, 0)
    assert np.array_equal(r.W[:, :1], plain.W[:, :1])
    for _ in range(30):                                # media de sorpresa baja
        r.learn(idx, h, 0)
    o = r.opened
    p = np.array([0.55, 0.45])                         # acierta pero con mucha más sorpresa que su media
    r.learn(idx, h, 0, p=p)
    assert r.opened == o + 1


def test_labile_state_follows_swap_remove():
    r = Readout(2, recon_lock=1.0)
    for c in (0, 1, 0):
        r.add_input(cls=c)
    r.labile_until[:3] = [5, 6, 7]
    r.remove_input(0, 2)
    assert r.labile_until[0] == 7 and r.n == 2


def test_prototype_lock_skips_ltp_only_when_the_area_was_right():
    rng = np.random.default_rng(9)
    net = BioElasticNet(20, 2, recon_lock=3)
    a = np.zeros(20, dtype=np.float32); a[:6] = 1.0
    for _ in range(10):
        net.step(np.clip(a + 0.05 * rng.random(20), 0, 1).astype(np.float32), 0)
    assert net.locked_skips > 0
    j = 0
    w0 = net.layer.w[j].copy()
    net.step(np.clip(a + 0.05 * rng.random(20), 0, 1).astype(np.float32), 0)
    assert np.array_equal(net.layer.w[j], w0)          # acierto: el prototipo consolidado no se mueve


# ---------------------------------------------------------------- F4: autoexamen espaciado
def test_active_replay_learns_only_from_errors_and_spends_the_same_budget():
    rng = np.random.default_rng(10)
    xs = [(_blob(rng, c), c) for c in rng.integers(4, size=200)]
    nets = {m: _cortex(replay=2.0, stored_labels=True, replay_mode=m, spacing="expanding", space_base=20)
            for m in ("passive", "active")}
    for m, net in nets.items():
        for x, c in xs:
            net.step(x, int(c))
    assert nets["passive"].replays == nets["active"].replays   # mismo presupuesto
    act = nets["active"]
    assert act.tests == act.replays and 0 < act.test_errors < act.tests
    mem = act.heads[0]
    ep = mem.ep; n = mem.layer.n
    tested = ep["tests"][:n] > 0
    assert tested.any() and np.all(ep["gap"][:n][tested] >= 20)
    # un examen que acierta no cambia la proyección; uno que falla sí
    W0 = act.proj.W.copy(); e0 = act.test_errors
    act._replay_scheduled()
    assert (act.test_errors > e0) == (not np.array_equal(act.proj.W, W0))


def test_spacing_expands_or_keeps_the_interval_and_forgetting_gets_priority():
    rng = np.random.default_rng(11)
    for spacing, gaps in (("equal", (30, 30, 30)), ("expanding", (60, 120, 120))):
        net = _cortex(replay=0.0, stored_labels=True, replay_mode="active", spacing=spacing,
                      space_base=30, space_factor=2.0)
        for _ in range(150):
            c = int(rng.integers(4)); net.step(_blob(rng, c), c)
        mem = net.heads[0]; ep = mem.ep; n = mem.layer.n
        net.proj.learn = lambda *a, **k: None          # el examen no cambia nada: resultados estables
        lab = net._labels(mem)
        passing = []
        for j in range(n):                             # recuerdos que el lento ya acierta
            idx, h, _ = net._reactivate(mem.episode(j), "input")
            if int(np.argmax(net.proj.potentials(idx, h))) == lab[j]:
                passing.append(j)
        j = passing[0]
        ep["due"][:n] = 10 ** 9; ep["fail"][:n] = 0.0
        ep["due"][j] = net._steps; ep["gap"][j] = 30.0; ep["ok"][j] = 1
        net._replay_scheduled()                        # vencido y acierta
        assert ep["gap"][j] == gaps[0] and ep["due"][j] == net._steps + gaps[0]
        net._steps = int(ep["due"][j])
        net._replay_scheduled()                        # vencido otra vez
        assert ep["gap"][j] == gaps[1]
        net._replay_scheduled()                        # adelantado (nada vencido): no se estira
        assert ep["gap"][j] == gaps[2]
        # un recuerdo que se sabía y falla pasa adelante de todos los vencidos
        k = passing[1]
        ep["due"][:n] = net._steps - 5; ep["gap"][:n] = 30.0
        ep["fail"][:n] = 0.0; ep["fail"][k] = 1.0
        assert net._pick_scheduled(mem, lab) == k


def test_contrast_replay_only_touches_the_two_competing_rows_and_only_on_failure():
    rng = np.random.default_rng(12)
    net = _cortex(replay=2.0, replay_focus="confusion", confusion_p=0.0, replay_contrast=True,
                  stored_labels=True)
    for _ in range(200):
        c = int(rng.integers(4)); net.step(_blob(rng, c), c)
    assert net.tests > 0 and net.directed > 0
    mem = net.heads[0]
    j = 0; x = mem.episode(j); c = int(mem.ep["y"][j]); rival = (c + 1) % 4
    W0 = net.proj.W.copy(); e0 = net.test_errors
    net.active_margin = 2.0                            # imposible de superar: siempre aprende
    net._contrast(x, c, "input", rival)
    changed = np.flatnonzero(np.abs(net.proj.W - W0).sum(axis=1) > 0)
    assert net.test_errors == e0 + 1 and set(changed) <= {c, rival} and len(changed)
    net.active_margin = -2.0                           # siempre aprueba: no cambia nada
    W1 = net.proj.W.copy()
    net._contrast(x, c, "input", rival)
    assert np.array_equal(net.proj.W, W1)

