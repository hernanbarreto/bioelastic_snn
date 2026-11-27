"""Modo serial del Thinker (escalón E, razonamiento composicional): un ancla sobre la oración leída
que persiste entre respuestas, miradas a palabras cercanas, el orden visto por el control,
significados por palabra en lo de largo plazo, FIN, la cuenta de respuestas en la posición
(contador) y el crédito transituacional con exclusividad mutua."""
import numpy as np

from bioelastic.thinker import _GLANCE, BACK, FOLLOW, JUDGE, RETURN, SAY, Thinker, _ptr, _Ptr

SER = dict(glance=(-3, 0), ctl=(-4, 4), dwell=24, prior=3.0, xsit=0.3, gist=False, realign=True)


def _serial(n=10, **kw):
    base = dict(memory=True, judge=True, copy=False, back=True, prev_io=False, cogmap=False, modulators=False,
                serial=dict(SER))
    return Thinker(n, **dict(base, **kw))


def _read(net, words):
    for w in words:
        net.observe(w)
    net.observe(None)


# ---------------------------------------------------------------------------------------- ancla
def test_pointer_walks_the_sentence_in_order_even_with_repeated_words():
    net = _serial()
    _read(net, [1, 2, 1, 3])
    f = net._serial_focus()
    assert isinstance(f, _Ptr) and tuple(f) == (1,) and (f.p, f.d, f.ret) == (0, 0, 0)
    # la cadena de la memoria episódica es ambigua con palabras repetidas: tras «1» vinieron 2 y 3
    assert set(net._recall((1,))[0].tolist()) == {2, 3}
    # el ancla no: recorre la oración en orden, posición por posición, hasta pasar el final
    seen = []
    while FOLLOW in net._allowed(0, True, f):
        (f,) = net._successors(FOLLOW, f, None, None)
        seen.append((f.p, tuple(f)))
    assert seen == [(1, (2,)), (2, (1,)), (3, (3,)), (4, ())]
    end = net._allowed(0, True, f)
    assert SAY not in end and JUDGE in end                         # pasado el final: no hay palabra que decir
    (b,) = net._successors(BACK, f, None, None)
    assert (b.p, tuple(b)) == (3, (3,))
    (r,) = net._successors(RETURN, b, None, None)
    assert (r.p, r.d, r.ret, tuple(r)) == (0, 0, 1, (1,))          # volver deja la marca de que volvió
    start = net._allowed(0, True, _ptr(f.s, 0))
    assert BACK not in start and RETURN not in start and FOLLOW in start
    assert not any(a >= _GLANCE for a in start)                    # no hay palabras antes del principio
    assert set(net._allowed(0, True, _ptr(f.s, 3))) >= {_GLANCE, _GLANCE + 1, _GLANCE + 2}


def test_pointer_persists_between_answers_and_restarts_with_a_new_sentence():
    net = _serial()
    _read(net, [1, 2, 3])
    net.answer(None)
    p0 = net._ptr
    net.answer(None)
    p1 = net._ptr
    assert p1.s is p0.s                                            # la misma oración
    if not any(a in (FOLLOW, BACK, RETURN) for a in net._path):    # respondió sin mover el ancla:
        assert (p1.p, p1.d) == (p0.p, p0.d + 1)                    # una respuesta más en esa posición
    _read(net, [4, 5])
    f = net._serial_focus()
    assert tuple(f.s) == ((4,), (5,)) and (f.p, f.d) == (0, 0)


def test_the_control_sees_the_order_and_the_knowledge_sees_single_words():
    """El multiconjunto ya no confunde: «1 2» y «2 1» dan estados distintos al control (cada
    palabra con su distancia al ancla), mientras que lo de largo plazo recibe la palabra sola y
    su significado vale en cualquier oración."""
    net = _serial()
    a, b = _ptr(((1,), (2,)), 0), _ptr(((2,), (1,)), 1)            # el ancla en «1» en las dos
    sa = set(net._state(0, -1, 0, a, net._recall(a), frozenset())[0].tolist())
    sb = set(net._state(0, -1, 0, b, net._recall(b), frozenset())[0].tolist())
    assert sa != sb
    assert net._glance_content(a, 0)[0].tolist() == net._glance_content(b, 0)[0].tolist() == [1]
    assert net._glance_content(b, -1)[0].tolist() == [2]           # mirar una palabra atrás: «2»


def test_serial_off_keeps_the_previous_layout():
    """Sin la opción, la disposición del contenido, del estado y de las acciones es la de antes
    (números de 10eed10)."""
    for kw, (dc, ds, na) in ((dict(memory=True, judge=True), (10, 198, 9)),
                            (dict(memory=True, judge=True, back=False), (10, 179, 8)),
                            (dict(memory=True, judge=True, prev_io=False, cogmap=False, modulators=False), (10, 29, 9)),
                            (dict(n_cues=2, memory=True, registers=2), (14, 609, 9))):
        t = Thinker(10, **kw)
        assert (t.D_content, t.D_state, t._n_act) == (dc, ds, na), kw
        assert RETURN not in t.actions


def test_the_serial_judgment_sees_only_the_cue():
    """El juicio del modo serial dice «terminé»; si viera la posición, aprendía a adivinar el
    primer símbolo de la salida por la posición (como el diseño anterior con la posición ordinal)."""
    net = _serial()
    _read(net, [1, 2])
    f = net._serial_focus()
    s = net._state(0, -1, 0, f, net._recall(f), frozenset())
    assert net._base(s)[0].tolist() == [0, net.o_cue]


# ---------------------------------------------------------------------------------------- FIN y significado por palabra
_LEX = dict(memory=True, judge=True, copy=False, back=False, prev_io=False, cogmap=False, modulators=False,
            max_steps=6, serial=dict(SER))


def _translate(net, seq, f, fin, learn, max_len=12):
    """Una oración y su salida `f(seq)` + FIN, un símbolo por respuesta (sin exploración: los
    errores los explica la búsqueda retrospectiva)."""
    _read(net, seq)
    gold = list(f(seq)) + [fin]
    outs = []
    for t in range(len(gold) if learn else max_len):
        o = net.answer(None)
        if learn:
            net.feedback(target=gold[t])
        outs.append(o)
        if not learn and o == fin:
            break
    return outs == gold


def _trained(f, W=6, n_train=300, seed=0, lens=(1, 4), **kw):
    rng = np.random.default_rng(seed)
    net = Thinker(2 * W + 1, seed=seed, **dict(_LEX, **kw))
    for _ in range(n_train):
        net.reset()
        _translate(net, list(rng.integers(0, W, int(rng.integers(lens[0], lens[1] + 1)))), f, 2 * W, True)
    net.plastic = False
    return net, rng


def _accuracy(net, rng, f, W=6, lengths=(5, 6), n=25):
    ok = []
    for L in lengths:
        for _ in range(n):
            net.reset()
            ok.append(_translate(net, list(rng.integers(0, W, L)), f, 2 * W, False))
    return float(np.mean(ok))


def _lexicon(net, W=6):
    """Cuántas palabras tienen establecido su significado correcto (w -> w + W)."""
    return sum(net._know(net._content((w,))) == (w + W, True) for w in range(W))


def test_fin_and_word_meanings_generalize_to_new_and_longer_sentences():
    """«a b c» -> «A B C FIN». Entrenado con 1 a 4 palabras, congelado: oraciones nuevas de 5 y 6
    palabras salen exactas, con FIN justo al final. Lo de largo plazo guarda un significado por
    palabra (pocas neuronas), no oraciones."""
    W = 6
    iconic = lambda seq: [w + W for w in seq]                      # noqa: E731
    net, rng = _trained(iconic, W)
    assert _accuracy(net, rng, iconic, W) >= 0.9
    assert _lexicon(net, W) == W and net.say.layer.n <= 3 * W


def test_the_output_order_is_learned_by_the_control_not_by_the_knowledge():
    """«a b c» -> «C B A FIN»: la salida no sigue el orden de la entrada (no es icónica). El
    crédito transituacional le enseña a lo de largo plazo el significado de la palabra que causó
    cada símbolo aunque el control haya mirado otra; el orden al revés lo aprende el control. Sin
    ese crédito (se corrige la palabra mirada, como antes) los significados quedan mezclados."""
    W = 6
    rev = lambda seq: [w + W for w in seq[::-1]]                   # noqa: E731
    net, rng = _trained(rev, W, n_train=400, lens=(2, 3))
    assert _lexicon(net, W) == W and net.say.layer.n == W
    assert _accuracy(net, rng, rev, W, lengths=(2, 3)) >= 0.8
    # con 100 oraciones (medido, semillas 0-2): con el crédito, el léxico completo en las tres (6
    # neuronas); sin él, 3 de 6 palabras en dos semillas (8-14 neuronas) y exacta 0.00-0.26 en ellas
    on = [_lexicon(_trained(rev, W, n_train=100, lens=(2, 3), seed=sd)[0], W) for sd in (0, 1, 2)]
    off = [_lexicon(_trained(rev, W, n_train=100, lens=(2, 3), seed=sd, serial=dict(SER, xsit=0.0))[0], W)
           for sd in (0, 1, 2)]
    assert min(on) == W and min(off) < W, (on, off)


def test_the_count_of_answers_in_a_position_repeats_and_stops():
    """Contador: «x twice» -> «X X», «x thrice» -> «X X X», «x» -> «X», y oraciones de dos frases
    con «and». La cuenta de respuestas del ancla en la posición decide cuántas veces repetir y
    cuándo seguir; se generaliza a combinaciones nuevas de frases conocidas."""
    U, TW, TH, AND = range(4), 4, 5, 6
    N = 13; FIN = 12
    meaning = {u: 7 + u for u in U}

    def out(seq):
        res, i = [], 0
        while i < len(seq):
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            k = 2 if nxt == TW else 3 if nxt == TH else 1
            res += [meaning[seq[i]]] * k
            i += (2 if k > 1 else 1) + 1                           # la frase y el «and» que sigue
        return res
    phrases = [(u,) for u in U] + [(u, TW) for u in U] + [(u, TH) for u in U]
    cmds = phrases + [p + (AND,) + q for p in phrases for q in phrases]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(cmds))
    train = [cmds[i] for i in order[:len(cmds) // 2]]
    test = [cmds[i] for i in order[len(cmds) // 2:]]
    net = Thinker(N, seed=0, **dict(_LEX, max_steps=8))
    for _ in range(4):
        for i in rng.permutation(len(train)):
            net.reset()
            _translate(net, list(train[i]), out, FIN, True)
    net.plastic = False
    ok = []
    for c in test:
        net.reset()
        ok.append(_translate(net, list(c), out, FIN, False))
    assert np.mean(ok) >= 0.8, np.mean(ok)


def test_mutual_exclusivity_gives_the_new_symbol_to_the_new_word():
    """Dos palabras que siempre aparecen juntas explican igual de bien un símbolo nuevo; si una ya
    significa otra cosa (establecido), la causa es la otra (exclusividad mutua)."""
    net = _serial(n=8)
    for _ in range(4):                                             # «1 2» -> 5 (siempre juntas)
        net._xs_count(((1,), (2,)), {5})
    for _ in range(4):                                             # otros episodios sin ellas
        net._xs_count(((3,),), {6})
    net._instr = ((1,), (2,))
    assert net._xs_cause(5) in (1, 2)
    for _ in range(3):                                             # «1» ya significa 7, establecido
        net.say.forward(net._content((1,)))
        net.say.teach(7)
    assert net._know(net._content((1,))) == (7, True)
    assert net._xs_cause(5) == 2
    assert net._xs_cause(6) is None                                # ninguna de las dos explica el 6
