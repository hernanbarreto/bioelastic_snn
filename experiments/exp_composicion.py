"""Escalón E, razonamiento composicional: diagnóstico y diseño nuevo de entrada, salida y memoria.

Datos: los de `exp_external` (SCAN, MLC y bAbI en `ext_data/`, con la fuente exacta en ese
archivo). Mismo protocolo: el entorno muestra lo que hay para leer palabra por palabra
(`observe`), pide la salida un símbolo por vez y da el objetivo de cada símbolo (`feedback`).

Grupos
  diag          diagnóstico del diseño anterior en SCAN simple con el 1 % de los comandos (semilla
                0, exploratorio): por qué no ajusta ni lo visto (a), por qué no dice FIN (b),
                cuántos prototipos quedan para cuántos contenidos (c) y el multiconjunto (d).
  scan_imitate  modo serial en SCAN al 1 %, en validación: peso de la imitación de la búsqueda.
  mini          modo serial en mini-SCAN (juguete propio), 3 semillas, imitate 1 y 3.
  scan_exp1     curva del experimento 1 (1-8 %, 3 semillas, mismas pasadas), test tocado una vez.

  python -I experiments/exp_composicion.py <grupo> [...]     (OMP_NUM_THREADS=1, nice -n 10)
  (resultados en results/composicion_<grupo>.csv y .json; cada condición terminada queda en
  results/composicion_<grupo>.parts.json y se salta al volver a correr)
"""
from __future__ import annotations

import copy
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import exp_external as X  # noqa: E402
import exp_metalearning as M  # noqa: E402
import exp_reasoning3 as R3  # noqa: E402
from bioelastic.thinker import ACTION_NAMES, SAY, Thinker  # noqa: E402
from common import RESULTS, CsvLog  # noqa: E402

NAMES = ACTION_NAMES + ("retroceder",)


class Store(R3.Store):
    """`exp_reasoning3.Store` con el prefijo de esta ronda (results/composicion_<grupo>.*)."""

    def __init__(self, group: str):
        self.group = group
        self.path = RESULTS / f"composicion_{group}.parts.json"
        self.parts = json.load(open(self.path)) if self.path.exists() else {}

    def finish(self, summary=None):
        log = CsvLog(f"composicion_{self.group}", R3.HEADER)
        for part in self.parts.values():
            for r in part["rows"]:
                log._w.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in r])
        log.close()
        obj = summary if summary is not None else {k: v["summary"] for k, v in self.parts.items()}
        with open(RESULTS / f"composicion_{self.group}.json", "w") as f:
            json.dump(obj, f, indent=1, ensure_ascii=False, default=M._jsonable)


# ====================================================================== diagnóstico (diseño anterior)
class _Probe(Thinker):
    """El Thinker de siempre, con contadores para el diagnóstico (no cambia lo que hace)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.n_correct = Counter()       # correcciones de conocimiento: marcadas / escritas
        self.say_keys = defaultdict(set)  # contenido de DECIR -> objetivos que recibió

    def _correct(self, mod, c, target, cue, episode, refresh):
        if mod == "say" and self.consolidate == "explicacion" and len(c[0]):
            from bioelastic.thinker import _key
            alt, n = self._tags.get((("say", _key(c)), cue), (None, 0))
            n = n + 1 if alt == target else 1
            self.n_correct["marcada (sin escribir)" if n < self.capture else "escrita"] += 1
        return super()._correct(mod, c, target, cue, episode, refresh)


def diag_scan(pct=1, seed=0, n_eval=200):
    """Corre el diseño anterior (BASE de exp_external) en SCAN simple al `pct` % y mide las
    cuatro causas en el código."""
    from bioelastic.thinker import _key
    d = X.SCAN / "simple_split" / "size_variations"
    train = X.scan_read(d / f"tasks_train_simple_p{pct}.txt")
    test = X.scan_read(d / f"tasks_test_simple_p{pct}.txt")
    rng = np.random.default_rng(seed)
    V = X.scan_vocab()
    EOS = V["<fin>"]
    net = _Probe(len(V), seed=seed, **X.BASE)
    fin_steps = Counter()                  # en los pasos cuyo objetivo es FIN: acción final y si dijo FIN
    final = Counter()                      # acción con que respondió cada símbolo (entrenamiento)
    outcome = Counter()                    # aciertos / errores, y de los errores: estrategia o conocimiento
    t0 = time.time()
    for i in rng.permutation(len(train)):
        cmd, acts = train[int(i)]
        net.reset()
        for w in cmd:
            net.observe(V[w])
        net.observe(None)
        q = tuple(V[w] for w in cmd)
        gold = [V[a] for a in acts] + [EOS]
        for t, y in enumerate(gold):
            b0 = net.blame.copy()
            out = net.answer(q + (V[f"<pos{t}>"],), explore=True)
            a = net._path[-1]
            if a == SAY:
                net.say_keys[_key(net._cf)].add(y)
            net.feedback(target=y)
            final[NAMES[a]] += 1
            db = net.blame - b0
            outcome["acierto" if out == y else ("error de estrategia" if db[0] else "error de conocimiento")] += 1
            if y == EOS:
                fin_steps[(NAMES[a], out == EOS)] += 1
    secs = time.time() - t0
    net.reset(); net.plastic = False
    # (c) cuántos contenidos distintos llegó a ver DECIR, cuántos con más de un objetivo, y
    # cuántos prototipos quedaron; fusión: contenidos con objetivos distintos que gana la
    # misma neurona
    keys = list(net.say_keys.items())
    multi = sum(len(v) > 1 for _, v in keys)
    win_targets = defaultdict(set)
    for k, ys in keys:
        idx = np.frombuffer(k, dtype=np.int64)
        v = net.say.layer.forward(idx, np.ones(len(idx), dtype=np.float32))
        if len(v.winners) and not v.novel:
            win_targets[int(v.winners[0])].update(ys)
    fused = sum(len(v) > 1 for v in win_targets.values())

    def evaluate(items):
        r = dict(exact=[], token=[], length_ok=[], no_fin=[])
        for cmd, acts in items:
            ok, outs = X.scan_item(net, cmd, acts, V, False, free=True)
            g = [V[a] for a in acts]
            r["exact"].append(ok)
            r["token"].append(np.mean([j < len(outs) and outs[j] == x for j, x in enumerate(g)]))
            r["length_ok"].append(len(outs) == len(g))
            r["no_fin"].append(len(outs) >= X.SCAN_MAX)
        return {k: float(np.mean(v)) for k, v in r.items()}
    seen = [train[int(i)] for i in rng.permutation(len(train))[:n_eval]]
    te = [test[int(i)] for i in rng.permutation(len(test))[:n_eval]]
    return dict(train=evaluate(seen), test=evaluate(te), secs=secs, final={k: int(v) for k, v in final.items()},
                outcome={k: int(v) for k, v in outcome.items()},
                corrections={k: int(v) for k, v in net.n_correct.items()},
                fin_steps={f"{a} | dijo FIN={ok}": int(n) for (a, ok), n in fin_steps.items()},
                say_contents=len(keys), say_contents_multi=multi, say_neurons=int(net.say.layer.n),
                winners_with_several_targets=fused, winners_used=len(win_targets),
                bag_ambiguity_train=X.scan_bag_ambiguity(train), bag_ambiguity_test=X.scan_bag_ambiguity(te))


def group_diag(pcts=(1,), seed=0):
    st = Store("diag")
    HEAD = "E0 diagnóstico del diseño anterior, SCAN simple"
    SYS = f"Thinker (valores por defecto de la punta + contexto temporal), semilla {seed} (exploratorio, 1 semilla)"
    for pct in pcts:
        def fn(log, pct=pct):
            r = diag_scan(pct, seed)
            T = f"{pct} %, una pasada"
            for part in ("train", "test"):
                for k, v in r[part].items():
                    log.row(HEAD, SYS, T + (", comandos vistos (200)" if part == "train" else ", test (200)"), k, v, 0.0, 1)
            n = sum(r["outcome"].values())
            for k, v in sorted(r["outcome"].items()):
                log.row(HEAD, SYS, T + ", símbolos del entrenamiento", f"fracción: {k}", v / n, 0.0, 1)
            for k, v in sorted(r["final"].items()):
                log.row(HEAD, SYS, T + ", acción con que respondió (entrenamiento)", k, v / n, 0.0, 1)
            nc = sum(r["corrections"].values())
            for k, v in sorted(r["corrections"].items()):
                log.row(HEAD, SYS, T + ", correcciones de DECIR", f"fracción {k} (de {nc})", v / max(nc, 1), 0.0, 1)
            nf = sum(r["fin_steps"].values())
            for k, v in sorted(r["fin_steps"].items()):
                log.row(HEAD, SYS, T + ", pasos cuyo objetivo es FIN", k, v / max(nf, 1), 0.0, 1)
            for k in ("say_contents", "say_contents_multi", "say_neurons", "winners_used", "winners_with_several_targets"):
                log.row(HEAD, SYS, T + ", memoria de largo plazo", k, float(r[k]), 0.0, 1)
            log.row(HEAD, SYS, T, "multiconjunto ambiguo (train)", r["bag_ambiguity_train"], 0.0, 1)
            return r
        st.run(f"diag scan {pct}% semilla {seed}", fn)
    st.finish()


# ====================================================================== modo serial
# El Thinker lee la oración palabra por palabra (`observe`) y después responde sin pregunta
# (`answer(None)`): la pregunta de cada respuesta es un ancla sobre la oración leída, que persiste
# entre respuestas del episodio (ver el docstring de `bioelastic.thinker`). El entorno ya no le
# muestra la posición ordinal del símbolo que toca. Se entrena sin exploración: los errores los
# explica la búsqueda retrospectiva.
SERIAL = dict(memory=True, judge=True, copy=False, speak=False, back=True, prev_io=False, cogmap=False,
              modulators=False, max_steps=10, budget=128,
              serial=dict(glance=(-3, 0), ctl=(-4, 4), dwell=10, prior=3.0, xsit=0.3, gist=False, realign=True,
                          imitate=3.0))
SERIAL_KEYS = ("glance", "ctl", "dwell", "prior", "xsit", "gist", "realign", "ret", "cls", "trial_td", "reuse", "td_actor", "imitate")


def cfg(**over):
    """Configuración serial con cambios: las claves de `serial` van dentro del dict."""
    c = dict(SERIAL)
    sc = dict(c["serial"])
    for k in list(over):
        if k in SERIAL_KEYS:
            sc[k] = over.pop(k)
    c.update(over)
    c["serial"] = sc
    return c


def scan_vocab_serial():
    words = ["jump", "walk", "run", "look", "turn", "left", "right", "opposite", "around", "twice", "thrice", "and", "after"]
    acts = ["I_JUMP", "I_WALK", "I_RUN", "I_LOOK", "I_TURN_LEFT", "I_TURN_RIGHT"]
    return {w: i for i, w in enumerate(words + acts + ["<fin>"])}


def serial_item(net, cmd, acts, V, learn, free=False, max_len=X.SCAN_MAX, log=None):
    """Un comando = un episodio: se lee palabra por palabra, con un silencio al final, y después
    se produce la salida un símbolo por vez hasta FIN. Aprendiendo, con el objetivo de cada
    símbolo; `free`: congelado y sin objetivo, corta en FIN o en `max_len` símbolos."""
    EOS = V["<fin>"]
    net.reset()
    for w in cmd:
        net.observe(V[w])
    net.observe(None)
    gold = [V[a] for a in acts] + [EOS]
    outs = []
    for t in range(max_len + 1 if free else len(gold)):
        out = net.answer(None, explore=learn and not free)
        if log is not None:
            log.append(("-".join(net.action_names()[a] for a in net._path), int(out), net._ptr.p, net._ptr.d - 1))
        if free:
            if out == EOS:
                return outs == gold[:-1], outs
            outs.append(out)
        else:
            net.feedback(target=gold[t])
            outs.append(out)
    return (outs == gold if not free else False), (outs if free else [o for o in outs if o != EOS])


def scan_eval(net, items, V, by_template=True):
    r = dict(exact=[], token=[], length_ok=[], no_fin=[], by={})
    for cmd, acts in items:
        ok, outs = serial_item(net, cmd, acts, V, False, free=True)
        g = [V[a] for a in acts]
        r["exact"].append(ok)
        r["token"].append(np.mean([j < len(outs) and outs[j] == x for j, x in enumerate(g)]))
        r["length_ok"].append(len(outs) == len(g))
        r["no_fin"].append(len(outs) > X.SCAN_MAX)
        if by_template:
            r["by"].setdefault(X.scan_template(cmd), []).append(ok)
            conj = "after" if "after" in cmd else ("and" if "and" in cmd else "sin conjunción")
            r["by"].setdefault(f"conjunción: {conj}", []).append(ok)
    return {k: (float(np.mean(v)) if k != "by" else {t: (float(np.mean(x)), len(x)) for t, x in v.items()}) for k, v in r.items()}


def scan_serial_run(pct, seed, conf=None, passes=1, n_test=1000, n_seen=300, val=False, explore=False, curve_every=0):
    """Experimento 1 de Lake y Baroni (2018) con el modo serial. `val`: se separa el 10 % del
    train (al azar, con la semilla) como validación; se entrena con el resto y se mide en esa
    validación (el test no se toca). Si no, se entrena con todo el train y se mide en una muestra
    al azar de `n_test` comandos del test (desviación declarada: el artículo usa el test entero)."""
    d = X.SCAN / "simple_split" / "size_variations"
    train = X.scan_read(d / f"tasks_train_simple_p{pct}.txt")
    rng = np.random.default_rng(seed)
    V = scan_vocab_serial()
    order = rng.permutation(len(train))
    if val:
        n_val = max(1, len(train) // 10)
        held = [train[int(i)] for i in order[:n_val]]
        train = [train[int(i)] for i in order[n_val:]]
    else:
        train = [train[int(i)] for i in order]
    net = Thinker(len(V), seed=seed, **(conf or SERIAL))
    t0 = time.time(); steps = 0; online = []; curve = []
    for ps in range(passes):
        for j in (rng.permutation(len(train)) if ps else range(len(train))):
            cmd, acts = train[int(j)]
            online.append(serial_item(net, cmd, acts, V, explore)[0]); steps += len(acts) + 1
        if val and curve_every and (ps + 1) % curve_every == 0 and ps + 1 < passes:
            fz = copy.deepcopy(net); fz.reset(); fz.plastic = False
            curve.append((ps + 1, scan_eval(fz, held, V, False)["exact"], scan_eval(fz, train[:100], V, False)["exact"]))
            print(f"  pasada {ps + 1}: validación {curve[-1][1]:.3f}, train (100) {curve[-1][2]:.3f}", flush=True)
    secs = time.time() - t0
    net.reset(); net.plastic = False
    seen = [train[int(i)] for i in rng.permutation(len(train))[:n_seen]]
    out = dict(seen=scan_eval(net, seen, V, False), n_train=len(train), passes=passes, train_seconds=secs,
               ms_per_step=secs / max(steps, 1) * 1e3, online_last=float(np.mean(online[-max(len(online) // 5, 1):])),
               stats=net.stats(), blame=net.blame.tolist(), curve=curve, explore=explore)
    if val:
        out["val"] = scan_eval(net, held, V)
    else:
        test = X.scan_read(d / f"tasks_test_simple_p{pct}.txt")
        te = [test[int(i)] for i in np.random.default_rng(10_000 + seed).permutation(len(test))[:n_test]]
        t1 = time.time()
        out["test"] = scan_eval(net, te, V)
        out["test_seconds"] = time.time() - t1
        out["n_test"] = len(te)
    return out


SCAN_PASSES = 20    # elegido en validación al 1 % (10/20/30 pasadas: 0.00/0.05/0.00, exploratorio)


def group_scan_exp1(pcts=(1, 2, 4, 8), seeds=(0, 1, 2), passes=SCAN_PASSES):
    """Experimento 1 de Lake y Baroni (2018) con el modo serial: el train de cada proporción,
    `passes` pasadas en línea (las mismas en todos los puntos de la curva; presentaciones =
    comandos x pasadas, el artículo usó 100 000), test congelado sobre una muestra fija de 1000
    comandos del test de esa proporción (desviación declarada: el artículo usa el test entero),
    tocado una vez. Una condición por (proporción, semilla)."""
    st = Store("scan_exp1")
    HEAD = "E2 SCAN, experimento 1 (proporción de comandos), modo serial"
    for pct in pcts:
        for sd in seeds:
            def fn(log, pct=pct, sd=sd):
                r = scan_serial_run(pct, sd, SERIAL, passes=passes, n_test=1000, n_seen=300)
                SYS = f"Thinker serial, semilla {sd}"
                T = f"{pct} % ({r['n_train']} comandos x {passes} pasadas = {r['n_train'] * passes} presentaciones)"
                for part, label in (("seen", "comandos vistos (300)"), ("test", "test (muestra de 1000)")):
                    for k in ("exact", "token", "length_ok", "no_fin"):
                        log.row(HEAD, SYS, T + ", " + label, k, float(r[part][k]), 0.0, 1)
                for t_, (v, n) in sorted(r["test"]["by"].items()):
                    log.row(HEAD, SYS, T + ", test por tipo", f"{t_} (n={n})", v, 0.0, 1)
                log.row(HEAD, SYS, T, "ms por símbolo (entrenamiento)", r["ms_per_step"], 0.0, 1)
                if pct in X.SCAN_EXP1_PUBLISHED:
                    log.row(HEAD, "seq2seq, mejor en conjunto (publicado, 100 000 presentaciones; no equivalente)",
                            f"{pct} %", "secuencia exacta, test completo", X.SCAN_EXP1_PUBLISHED[pct], 0.0, 0)
                return r
            st.run(f"scan exp1 {pct}% semilla {sd}", fn)
    st.finish()


def group_scan_imitate(gains=(1.0, 3.0, 5.0), pct=1, seed=0, passes=30, every=10):
    """Hipótesis del jefe de equipo: el control imita el camino que halló la búsqueda
    retrospectiva con más peso que el TD. SCAN al `pct` %, en validación (10 % del train, el
    test no se toca), una semilla (exploratorio), las mismas pasadas para cada peso."""
    st = Store("scan_imitate")
    HEAD = "E1 SCAN, imitación de la búsqueda (validación)"
    for g in gains:
        def fn(log, g=g):
            r = scan_serial_run(pct, seed, cfg(imitate=g), passes=passes, n_seen=100, val=True, curve_every=every)
            SYS = f"Thinker serial, imitate={g}, semilla {seed} (exploratorio, 1 semilla)"
            pres = r["n_train"] * passes
            for ps, v, tr in r["curve"]:
                T = f"{pct} %, {ps} pasadas ({r['n_train'] * ps} presentaciones)"
                log.row(HEAD, SYS, T, "secuencia exacta, validación", v, 0.0, 1)
                log.row(HEAD, SYS, T, "secuencia exacta, vistos (100)", tr, 0.0, 1)
            T = f"{pct} %, {passes} pasadas ({pres} presentaciones)"
            for part, label in (("seen", "vistos (100)"), ("val", "validación (" + str(len(r["val"]["by"]) and sum(n for _, n in [r["val"]["by"].get("conjunción: and", (0, 0)), r["val"]["by"].get("conjunción: after", (0, 0)), r["val"]["by"].get("conjunción: sin conjunción", (0, 0))])) + ")")):
                for k in ("exact", "token", "length_ok", "no_fin"):
                    log.row(HEAD, SYS, T + ", " + label, k, float(r[part][k]), 0.0, 1)
            n = sum(r["blame"])
            log.row(HEAD, SYS, T, "errores explicados por estrategia (fracción de los errores)", r["blame"][0] / max(n, 1), 0.0, 1)
            log.row(HEAD, SYS, T, "ms por símbolo (entrenamiento)", r["ms_per_step"], 0.0, 1)
            return r
        st.run(f"imitate {g} {pct}% semilla {seed}", fn)
    st.finish()


# ====================================================================== mini-SCAN (juguete propio)
# Gramática chica para iterar rápido sobre el mecanismo, no una prueba externa: U en {a, b, c, d}
# -> {A, B, C, D}; V = U | U left (L U) | U twice (U U); C = V | V and V | V after V (V2 V1).
MINI_U = ("a", "b", "c", "d")


def mini_commands():
    import itertools
    P = []
    for u in MINI_U:
        P += [((u,), (u.upper(),)), ((u, "left"), ("L", u.upper())), ((u, "twice"), (u.upper(),) * 2)]
    cmds = list(P)
    for (c1, o1), (c2, o2) in itertools.product(P, P):
        cmds += [(c1 + ("and",) + c2, o1 + o2), (c1 + ("after",) + c2, o2 + o1)]
    return cmds


def mini_run(seed=0, frac=0.3, passes=15, conf=None):
    """Entrena con una fracción al azar de los 300 comandos (sin exploración) y mide exacta,
    congelado, en lo visto y en el resto."""
    words = list(MINI_U) + ["left", "twice", "and", "after"]
    V = {w: i for i, w in enumerate(words + ["A", "B", "C", "D", "L", "<fin>"])}
    rng = np.random.default_rng(seed)
    cmds = mini_commands()
    idx = rng.permutation(len(cmds))
    k = int(frac * len(cmds))
    tr, te = [cmds[i] for i in idx[:k]], [cmds[i] for i in idx[k:]]
    net = Thinker(len(V), seed=seed, **(conf or SERIAL))
    for _ in range(passes):
        for i in rng.permutation(len(tr)):
            serial_item(net, *tr[i], V, True, max_len=12)
    net.reset(); net.plastic = False
    ev = lambda items: float(np.mean([serial_item(net, c, o, V, False, free=True, max_len=12)[0] for c, o in items]))  # noqa: E731
    return dict(train=ev(tr), test=ev(te), n_train=len(tr), passes=passes)


def group_mini(seeds=(0, 1, 2), gains=(1.0, 3.0)):
    st = Store("mini")
    HEAD = "E1 mini-SCAN (gramática de juguete propia, 30 % del train, 15 pasadas)"
    for g in gains:
        def fn(log, g=g):
            res = [mini_run(sd, conf=cfg(imitate=g)) for sd in seeds]
            for k in ("train", "test"):
                v = [r[k] for r in res]
                log.row(HEAD, f"Thinker serial, imitate={g}", f"{res[0]['n_train']} comandos x 15 pasadas",
                        f"secuencia exacta, {k}", float(np.mean(v)), float(np.std(v)), len(v))
            return dict(train=[r["train"] for r in res], test=[r["test"] for r in res])
        st.run(f"mini imitate {g}", fn)
    st.finish()


GROUPS = {"diag": group_diag, "scan_exp1": group_scan_exp1, "scan_imitate": group_scan_imitate,
          "mini": group_mini}


def main():
    for g in sys.argv[1:] or ["diag"]:
        t0 = time.time()
        GROUPS[g]()
        print(f"[{g}] {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
