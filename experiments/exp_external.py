"""Razonamiento medido afuera (F6): bAbI, SCAN y episodios de pocos ejemplos (MLC).

Los datos no se versionan: se bajan de GitHub a `ext_data/` (fuera de `data/`), cada uno en
su carpeta, y acá queda la fuente exacta:

  bAbI v1.2, 1 000 ejemplos por tarea (`en/`): espejo en github.com/bgavran/DNC, commit
      668a2f1ecc676748c68cedab78dd198aca190d3b, carpeta
      src/task_implementations/bAbI/tasks_1-20_v1-2/en (Weston et al., 2015; el original
      estaba en fb.ai/babi, que el sandbox no alcanza). md5 de
      qa1_single-supporting-fact_train.txt: b3d2196a7dd5d8c65b2ddc17a34d76bd.
        git clone --depth 1 --filter=blob:none --sparse https://github.com/bgavran/DNC.git ext_data/babi_repo
        git -C ext_data/babi_repo sparse-checkout set src/task_implementations/bAbI/tasks_1-20_v1-2/en
  SCAN: github.com/brendenlake/SCAN, commit c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f
      (simple_split, add_prim_split/…jump, length_split).
        git clone --depth 1 https://github.com/brendenlake/SCAN.git ext_data/scan
  MLC (Lake y Baroni, 2023): github.com/brendenlake/MLC, commit
      34837ccb06c855bb29558ad99bb1cef70debd559 (data_human/few_shot: el episodio de la tarea
      de pocos ejemplos y las respuestas de las personas). Sólo se leen los .txt.
        git clone --depth 1 https://github.com/brendenlake/MLC.git ext_data/mlc

Protocolo común: nada de la tarea metido a mano. El entorno sólo hace tres cosas, las mismas
en las tres pruebas: (1) muestra lo que hay para leer palabra por palabra, con un silencio al
final de cada oración (`observe`); (2) pregunta mostrando las palabras de la pregunta
(`answer`); (3) da el objetivo (`feedback`). Cuando la respuesta es una secuencia, pide un
símbolo por vez hasta que el Thinker dice FIN (o se pasa del largo máximo) y le muestra, junto
con la pregunta, la posición ordinal del símbolo que toca (un código de orden serial, como el
de las células de tiempo; no depende de la tarea).

  python -I experiments/exp_external.py <grupo> [...]     (OMP_NUM_THREADS=1)
  grupos: babi  scan  mlc
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from bioelastic.thinker import ACTION_NAMES, Thinker  # noqa: E402
from exp_reasoning3 import SEEDS, Store  # noqa: E402

EXT = HERE.parent / "ext_data"
BABI = EXT / "babi_repo" / "src" / "task_implementations" / "bAbI" / "tasks_1-20_v1-2" / "en"
SCAN = EXT / "scan"
MLC = EXT / "mlc" / "data_human" / "few_shot"
NAMES = ACTION_NAMES + ("retroceder",)
BASE = dict(memory=True, speak=True, judge=True, max_steps=10, context=True)


def path_name(path) -> str:
    return "-".join(NAMES[a][:3] for a in path)


# ====================================================================== bAbI
def babi_files():
    """{tarea: (nombre, train, test)} con los archivos del espejo."""
    out = {}
    for f in sorted(BABI.glob("qa*_train.txt")):
        t = int(f.name[2:].split("_")[0])
        out[t] = (f.name.split("_", 1)[1].rsplit("_", 1)[0], f, f.with_name(f.name.replace("_train", "_test")))
    return dict(sorted(out.items()))


def babi_read(path):
    """Historias: [[("s", palabras) | ("q", palabras, respuesta)]]. Minúsculas, sin puntuación;
    una respuesta con comas (listas, caminos) es un solo símbolo, como en las redes de memoria."""
    stories, cur = [], None
    for line in path.read_text().splitlines():
        num, _, text = line.partition(" ")
        if int(num) == 1:
            cur = []; stories.append(cur)
        if "\t" in text:
            q, a, _ = text.split("\t")
            cur.append(("q", tuple(q.lower().replace("?", " ").split()), a.strip().lower()))
        else:
            cur.append(("s", tuple(text.lower().replace(".", " ").split())))
    return stories


def babi_vocab(*sets):
    words = sorted({w for st in sets for story in st for item in story for w in item[1]}
                   | {item[2] for st in sets for story in st for item in story if item[0] == "q"})
    return {w: i for i, w in enumerate(words)}


def babi_story(net, story, V, learn, stats=None):
    """Una historia = un episodio. Devuelve los aciertos de sus preguntas."""
    net.reset()
    ok = []
    for item in story:
        if item[0] == "s":
            for w in item[1]:
                net.observe(V[w])
            net.observe(None)
            continue
        y = V[item[2]]
        out = net.answer(tuple(V[w] for w in item[1]), explore=learn)
        net.feedback(target=y)
        ok.append(out == y)
        if stats is not None:
            stats.append((out == y, path_name(net._path)))
    return ok


def babi_task(task, seed, epochs=5, over=None, limit=None):
    """Una tarea: hasta `epochs` pasadas en línea sobre el 90 % de las historias de
    entrenamiento; tras cada pasada, exactitud congelada en el 10 % de validación; la mejor
    instantánea se evalúa una vez en el test. Devuelve un dict con las medidas."""
    name, ftr, fte = babi_files()[task]
    train, test = babi_read(ftr), babi_read(fte)
    if limit:
        train, test = train[:limit], test[:limit]
    V = babi_vocab(train, test)
    n_val = max(1, len(train) // 10)
    tr, va = train[:-n_val], train[-n_val:]
    net = Thinker(len(V), seed=seed, **dict(BASE, **(over or {})))
    rng = np.random.default_rng(seed)
    best, best_acc, curve = None, -1.0, []
    t0 = time.time(); n_q = 0
    for ep in range(epochs):
        online = []
        for i in rng.permutation(len(tr)):
            online += babi_story(net, tr[int(i)], V, True)
        n_q += len(online)
        fz = copy.deepcopy(net); fz.reset(); fz.plastic = False
        acc = float(np.mean([x for s in va for x in babi_story(fz, s, V, False)]))
        curve.append((float(np.mean(online)), acc))
        if acc > best_acc:
            best, best_acc = fz, acc
    secs = time.time() - t0
    stats = []
    fz = copy.deepcopy(best); fz.reset(); fz.plastic = False
    test_acc = float(np.mean([x for s in test for x in babi_story(fz, s, V, False, stats)]))
    paths = {}
    for ok, p in stats:
        a = paths.setdefault(p, [0, 0]); a[0] += int(ok); a[1] += 1
    top = sorted(paths.items(), key=lambda kv: -kv[1][1])[:4]
    from collections import Counter
    ans_tr = Counter(it[2] for st_ in train for it in st_ if it[0] == "q")
    ans_te = Counter(it[2] for st_ in test for it in st_ if it[0] == "q")
    chance = dict(uniforme=1.0 / len(ans_tr), mayoritaria=ans_te.most_common(1)[0][1] / sum(ans_te.values()))
    return dict(name=name, test=test_acc, val=best_acc, curve=curve, ms_per_question=secs / max(n_q, 1) * 1e3, chance=chance,
                paths={p: v for p, v in top}, say_neurons=int(net.say.layer.n),
                task_sets=len(net.task_sets.sets) if net.task_sets is not None else 0)


# cifras publicadas con 1 000 ejemplos por tarea (error de test, %), Sukhbaatar et al. 2015,
# tabla 1 (arxiv.org/abs/1503.08895): MemNN con supervisión fuerte (Weston et al. 2015), LSTM,
# MemN2N BoW y MemN2N PE LS RN (por tarea, 3 saltos, 100 épocas, mejor de 10 corridas).
BABI_PUBLISHED = {
    "MemNN (supervisión fuerte)": [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 15.0, 9.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 35.0, 5.0, 64.0, 0.0],
    "LSTM": [50.0, 80.0, 80.0, 39.0, 30.0, 52.0, 51.0, 55.0, 36.0, 56.0, 38.0, 26.0, 6.0, 73.0, 79.0, 77.0, 49.0, 48.0, 92.0, 9.0],
    "MemN2N BoW": [0.6, 17.6, 71.0, 32.0, 18.3, 8.7, 23.5, 11.4, 21.1, 22.8, 4.1, 0.3, 10.5, 1.3, 24.3, 52.0, 45.4, 48.1, 89.7, 0.1],
    "MemN2N PE LS RN": [0.0, 8.3, 40.3, 2.8, 13.1, 7.6, 17.3, 10.0, 13.2, 15.1, 0.9, 0.2, 0.4, 1.7, 0.0, 1.3, 51.0, 11.1, 82.8, 0.0],
}


def group_babi(tasks=None, seeds=SEEDS, epochs=3):
    st = Store("f6_babi")
    files = babi_files()
    HEAD = "F6c bAbI (1 000 ejemplos por tarea, entrenamiento por tarea)"
    for t in (tasks or files):
      for sd in seeds:
        def fn(log, t=t, sd=sd):
            res = [babi_task(t, sd, epochs)]
            acc = [r["test"] for r in res]
            name = f"{t}: {res[0]['name']}"
            log.row(HEAD, f"Thinker (valores por defecto + contexto temporal), semilla {sd}", "test, congelado", name,
                    float(np.mean(acc)), 0.0, 1)
            for label, errs in BABI_PUBLISHED.items():
                log.row(HEAD, label + " (publicado)", "test", name, 1.0 - errs[t - 1] / 100.0, 0.0, 0)
            for kind, v in res[0]["chance"].items():
                log.row(HEAD, f"azar ({kind})", "test", name, float(v), 0.0, 0)
            log.row(HEAD, "Thinker (todo encendido + contexto temporal)", "costo", name + " | ms por pregunta",
                    float(np.mean([r["ms_per_question"] for r in res])), 0.0, len(res))
            return dict(test=acc, val=[r["val"] for r in res], curve=[r["curve"] for r in res],
                        paths=[r["paths"] for r in res], say_neurons=[r["say_neurons"] for r in res],
                        task_sets=[r["task_sets"] for r in res], name=res[0]["name"], chance=res[0]["chance"])
        st.run(f"babi {t} semilla {sd}", fn)
    accs = np.array([[st.get(f"babi {t} semilla {sd}")["test"][0] for sd in seeds] for t in files
                     if all(st.done(f"babi {t} semilla {sd}") for sd in seeds)])
    if len(accs) == len(files) and len(seeds) == len(SEEDS):
        def summary(log):
            m = accs.mean(0)                                 # por semilla
            log.row(HEAD, "Thinker (todo encendido + contexto temporal)", "test, congelado", "error medio (20 tareas)",
                    float(1 - m.mean()), float(m.std()), len(m))
            failed = (accs.mean(1) < 0.95).sum()
            log.row(HEAD, "Thinker (todo encendido + contexto temporal)", "test, congelado", "tareas falladas (error > 5 %)",
                    float(failed), 0.0, len(m))
            for label, errs in BABI_PUBLISHED.items():
                log.row(HEAD, label + " (publicado)", "test", "error medio (20 tareas)", float(np.mean(errs)) / 100.0, 0.0, 0)
                log.row(HEAD, label + " (publicado)", "test", "tareas falladas (error > 5 %)",
                        float(sum(e > 5.0 for e in errs)), 0.0, 0)
            return dict(mean_error=float(1 - m.mean()), failed=int(failed))
        st.run("babi resumen", summary)
    st.finish()


# ====================================================================== secuencias
def emit(net, question, target, V, learn, max_len, free=False):
    """El Thinker produce una secuencia: un símbolo por respuesta, con la pregunta y la
    posición ordinal a la vista, hasta decir FIN. Aprendiendo (o midiendo en línea), se le da
    el objetivo de cada posición y se sigue hasta el largo del objetivo; `free`: congelado,
    sin objetivo, corta cuando dice FIN. Devuelve los símbolos producidos (sin FIN)."""
    EOS = V["<fin>"]
    gold = list(target) + [EOS]
    outs = []
    for t in range(max_len):
        out = net.answer(tuple(question) + (V[f"<pos{t}>"],), explore=learn)
        if free:
            if out == EOS:
                break
            outs.append(out)
            continue
        y = gold[t]
        net.feedback(target=y)
        if out != EOS:
            outs.append(out)
        elif y != EOS:
            outs.append(-1)                                  # cortó antes de tiempo
        if y == EOS:
            break
    return outs


# ====================================================================== SCAN
def scan_read(path):
    out = []
    for line in Path(path).read_text().splitlines():
        cmd, act = line[4:].split(" OUT: ")
        out.append((tuple(cmd.split()), tuple(act.split())))
    return out


SCAN_SPLITS = {
    "simple": ("simple_split/tasks_train_simple.txt", "simple_split/tasks_test_simple.txt"),
    "add jump": ("add_prim_split/tasks_train_addprim_jump.txt", "add_prim_split/tasks_test_addprim_jump.txt"),
    "length": ("length_split/tasks_train_length.txt", "length_split/tasks_test_length.txt"),
}
# cifras publicadas, exactitud de secuencia exacta en el test (%). seq2seq: Lake y Baroni 2018
# (arxiv.org/abs/1711.00350), mejor arquitectura de cada experimento y la mejor en conjunto
# (LSTM de 2 capas x 200), 100 000 presentaciones, media de 5 corridas. MLC: Lake y Baroni
# 2023 (Nature 623, tabla 2), 100 - error; en length reportan 100 % de error.
SCAN_PUBLISHED = {
    "simple": {"seq2seq (mejor)": 99.8, "seq2seq (mejor en conjunto)": 99.7, "MLC": 99.98},
    "add jump": {"seq2seq (mejor)": 1.2, "seq2seq (mejor en conjunto)": 0.08, "MLC": 99.78},
    "length": {"seq2seq (mejor)": 20.8, "seq2seq (mejor en conjunto)": 13.8, "MLC": 0.0},
}
SCAN_MAX = 49


def scan_vocab():
    words = ["jump", "walk", "run", "look", "turn", "left", "right", "opposite", "around", "twice", "thrice", "and", "after"]
    acts = ["I_JUMP", "I_WALK", "I_RUN", "I_LOOK", "I_TURN_LEFT", "I_TURN_RIGHT"]
    toks = words + acts + ["<fin>"] + [f"<pos{t}>" for t in range(SCAN_MAX + 1)]
    return {w: i for i, w in enumerate(toks)}


def scan_item(net, cmd, acts, V, learn, free=False):
    """Un comando = un episodio: se lee palabra por palabra y después se produce la secuencia."""
    net.reset()
    for w in cmd:
        net.observe(V[w])
    net.observe(None)
    q = tuple(V[w] for w in cmd)
    outs = emit(net, q, [V[a] for a in acts], V, learn, SCAN_MAX if free else len(acts) + 1, free)
    return outs == [V[a] for a in acts], outs


def scan_features(cmd, acts):
    """Tipos de ítem para el diagnóstico."""
    f = []
    n = len(acts)
    f.append("largo 1-4" if n <= 4 else "largo 5-12" if n <= 12 else "largo 13-22" if n <= 22 else "largo 23-48")
    f.append("con conjunción (and/after)" if ("and" in cmd or "after" in cmd) else "sin conjunción")
    if "twice" in cmd or "thrice" in cmd:
        f.append("con repetición (twice/thrice)")
    if "around" in cmd or "opposite" in cmd:
        f.append("con around/opposite")
    if "jump" in cmd:
        f.append("con jump")
    return f


def scan_run(split, seed, n_train, n_test=500, over=None):
    ftr, fte = SCAN_SPLITS[split]
    train, test = scan_read(SCAN / ftr), scan_read(SCAN / fte)
    rng = np.random.default_rng(seed)
    V = scan_vocab()
    net = Thinker(len(V), seed=seed, **dict(BASE, **(over or {})))
    idx = rng.permutation(len(train))[:n_train]               # n_train None: el train completo
    if split == "add jump" and n_train is not None:           # (submuestra: que no falte la primitiva sola)
        jump = [i for i, (c, _) in enumerate(train) if c == ("jump",)][:1]
        idx = np.concatenate([jump, idx[:max(n_train - 1, 0)]]).astype(int)
        rng.shuffle(idx)
    t0 = time.time(); online = []; steps = 0; windows = []; tw = time.time()
    for j, i in enumerate(idx):
        cmd, acts = train[int(i)]
        ok, _ = scan_item(net, cmd, acts, V, True)
        online.append(ok); steps += len(acts) + 1
        if (j + 1) % 1000 == 0:                               # costo por comando, por tramo (no promediado desde el inicio)
            windows.append((time.time() - tw) / 1000 * 1e3); tw = time.time()
    secs = time.time() - t0
    net.reset(); net.plastic = False
    te = [test[int(i)] for i in rng.permutation(len(test))[:n_test]]
    exact, tok, by = [], [], {}
    t1 = time.time()
    for cmd, acts in te:
        ok, outs = scan_item(net, cmd, acts, V, False, free=True)
        gold = [V[a] for a in acts]
        exact.append(ok)
        tok.append(np.mean([i < len(outs) and outs[i] == g for i, g in enumerate(gold)]))
        for f in scan_features(cmd, acts):
            by.setdefault(f, []).append(ok)
        by.setdefault("largo producido = largo correcto", []).append(len(outs) == len(gold))
    return dict(exact=float(np.mean(exact)), token=float(np.mean(tok)), online_last=float(np.mean(online[-max(len(online) // 5, 1):])),
                by={k: float(np.mean(v)) for k, v in by.items()}, ms_per_step=secs / max(steps, 1) * 1e3,
                say_neurons=int(net.say.layer.n), n_train=len(idx), n_test=len(te), ms_per_command_windows=windows,
                train_seconds=secs, test_seconds=time.time() - t1)


def group_scan_curve(sizes=(100, 300, 1000), seeds=SEEDS, n_test=200, splits=("simple",)):
    """Curva con submuestras del train (desviación de protocolo: el artículo entrena con el train
    completo). Quedó medida para «simple» antes de pasar al protocolo completo."""
    st = Store("f6_scan")
    HEAD = "F6a SCAN"
    SYS = "Thinker (todo encendido + contexto temporal)"
    for split in splits:
        for n in sizes:
            def fn(log, split=split, n=n):
                res = [scan_run(split, s, n, n_test=n_test) for s in seeds]
                for key, what in (("exact", "secuencia exacta"), ("token", "símbolos en su posición"),
                                  ("online_last", "secuencia exacta en línea, último quinto del entrenamiento")):
                    v = [r[key] for r in res]
                    log.row(HEAD + f", {split}", SYS, f"{n} comandos de entrenamiento", what, float(np.mean(v)), float(np.std(v)), len(v))
                log.row(HEAD + f", {split}", SYS, f"{n} comandos de entrenamiento", "ms por símbolo",
                        float(np.mean([r["ms_per_step"] for r in res])), 0.0, len(res))
                return dict(exact=[r["exact"] for r in res], token=[r["token"] for r in res], by=[r["by"] for r in res],
                            online_last=[r["online_last"] for r in res], say_neurons=[r["say_neurons"] for r in res])
            st.run(f"scan {split} {n}", fn)
    st.finish()


def group_scan(seeds=SEEDS, splits=("add jump", "length", "simple")):
    """Protocolo del artículo: el train completo de la división (una sola pasada en línea) y el
    test completo, congelado. Una condición por semilla (con una sola semilla las filas dicen
    «exploratorio, 1 semilla»; el promedio entre semillas lo hace `export`)."""
    st = Store("f6_scan")
    HEAD = "F6a SCAN"
    for split in splits:
        for sd in seeds:
            def fn(log, split=split, sd=sd):
                r = scan_run(split, sd, None, n_test=None)
                SYS = f"Thinker (valores por defecto + contexto temporal), semilla {sd}"
                T = f"train completo ({r['n_train']} comandos, una pasada), test completo ({r['n_test']})"
                for key, what in (("exact", "secuencia exacta"), ("token", "símbolos en su posición"),
                                  ("online_last", "secuencia exacta en línea, último quinto del entrenamiento")):
                    log.row(HEAD + f", {split}", SYS, T, what, float(r[key]), 0.0, 1)
                for f_, v in sorted(r["by"].items()):
                    log.row(HEAD + f", {split}", SYS, T + ", por tipo de ítem", f_, float(v), 0.0, 1)
                for label, acc in SCAN_PUBLISHED[split].items():
                    log.row(HEAD + f", {split}", label + " (publicado)", "train completo", "secuencia exacta", acc / 100.0, 0.0, 0)
                log.row(HEAD + f", {split}", SYS, T, "ms por símbolo (entrenamiento)", float(r["ms_per_step"]), 0.0, 1)
                return r
            st.run(f"scan {split} completo semilla {sd}", fn)
    st.finish()


def scan_template(cmd):
    """Plantilla de un comando para el diagnóstico."""
    if len(cmd) == 1:
        return "primitiva sola"
    if "and" in cmd or "after" in cmd:
        return "con and/after"
    if "twice" in cmd or "thrice" in cmd:
        return "con twice/thrice (sin conjunción)"
    return "con left/right/around/opposite (sin conjunción ni repetición)"


def scan_bag_ambiguity(cmds):
    """Fracción de comandos cuyo multiconjunto de palabras (lo que ve el contenido del Thinker:
    las palabras de la pregunta sin orden) es el de otro comando con otra salida."""
    from collections import defaultdict
    outs = defaultdict(set)
    for c, a in cmds:
        outs[tuple(sorted(c))].add(a)
    return float(np.mean([len(outs[tuple(sorted(c))]) > 1 for c, _ in cmds]))


def scan_size_run(pct, seed, passes=1, n_test=1000, n_train_eval=300):
    """Experimento 1 de Lake y Baroni (2018): entrenar con el pct % de los comandos (archivos
    simple_split/size_variations del repositorio) y probar con el resto. Diagnóstico: exactitud
    en lo ya visto, por plantilla, errores de largo y de FIN."""
    d = SCAN / "simple_split" / "size_variations"
    train = scan_read(d / f"tasks_train_simple_p{pct}.txt")
    test = scan_read(d / f"tasks_test_simple_p{pct}.txt")
    rng = np.random.default_rng(seed)
    V = scan_vocab()
    net = Thinker(len(V), seed=seed, **BASE)
    t0 = time.time(); steps = 0
    for _ in range(passes):
        for i in rng.permutation(len(train)):
            cmd, acts = train[int(i)]
            scan_item(net, cmd, acts, V, True); steps += len(acts) + 1
    secs = time.time() - t0
    net.reset(); net.plastic = False

    def evaluate(items):
        r = dict(exact=[], token=[], first=[], length_ok=[], no_fin=[], by={})
        for cmd, acts in items:
            ok, outs = scan_item(net, cmd, acts, V, False, free=True)
            gold = [V[a] for a in acts]
            r["exact"].append(ok)
            r["token"].append(np.mean([i < len(outs) and outs[i] == g for i, g in enumerate(gold)]))
            r["first"].append(bool(outs) and outs[0] == gold[0])
            r["length_ok"].append(len(outs) == len(gold))
            r["no_fin"].append(len(outs) >= SCAN_MAX)
            r["by"].setdefault(scan_template(cmd), []).append(ok)
        return {k: (float(np.mean(v)) if k != "by" else {t: (float(np.mean(x)), len(x)) for t, x in v.items()}) for k, v in r.items()}
    seen = [train[int(i)] for i in rng.permutation(len(train))[:n_train_eval]]
    te = [test[int(i)] for i in rng.permutation(len(test))[:n_test]]
    return dict(train=evaluate(seen), test=evaluate(te), n_train=len(train), n_test_total=len(test), n_test=len(te),
                passes=passes, ms_per_step=secs / max(steps, 1) * 1e3, say_neurons=int(net.say.layer.n),
                bag_ambiguity_test=scan_bag_ambiguity(te), bag_ambiguity_all=scan_bag_ambiguity(train + test))


# Lake y Baroni (2018), experimento 1, texto (arxiv.org/pdf/1711.00350v2): con el modelo mejor en
# conjunto (LSTM 2 x 200, sin atención, dropout 0.5) y 100 000 presentaciones (los comandos de
# entrenamiento sorteados con reposición), «about 5% correct» con 1 %, «about 54%» con 2 % y
# «about 93%» con 4 %. Las demás proporciones sólo están en la figura 3 (no se leen cifras).
SCAN_EXP1_PUBLISHED = {1: 0.05, 2: 0.54, 4: 0.93}


def group_scan_sizes(pcts=(1, 2, 4, 8), seeds=(0,), passes=(1,)):
    """Experimento 1 de Lake y Baroni (2018) con 1-8 % de los comandos, diagnóstico incluido.
    El test es una muestra al azar de 1000 comandos del test de cada variación (desviación
    declarada: el artículo usa el test completo)."""
    st = Store("f6_scan_exp1")
    HEAD = "F6a SCAN, experimento 1 (proporción de comandos)"
    for pct in pcts:
        for ps in passes:
            for sd in seeds:
                def fn(log, pct=pct, ps=ps, sd=sd):
                    r = scan_size_run(pct, sd, ps)
                    SYS = f"Thinker (valores por defecto + contexto temporal), semilla {sd}" + (" (exploratorio, 1 semilla)" if len(seeds) == 1 else "")
                    T = f"{pct} % ({r['n_train']} comandos), {ps} pasada(s)"
                    for part in ("train", "test"):
                        for key, what in (("exact", "secuencia exacta"), ("token", "símbolos en su posición"), ("first", "primer símbolo"),
                                          ("length_ok", "largo correcto"), ("no_fin", "nunca dijo FIN")):
                            log.row(HEAD, SYS, T + (", comandos ya vistos" if part == "train" else ", test (muestra de 1000)"), what,
                                    float(r[part][key]), 0.0, 1)
                    for t_, (v, n) in sorted(r["test"]["by"].items()):
                        log.row(HEAD, SYS, T + ", test por plantilla", f"{t_} (n={n})", v, 0.0, 1)
                    log.row(HEAD, SYS, T, "comandos del test con el mismo multiconjunto de palabras que otro con otra salida",
                            r["bag_ambiguity_test"], 0.0, 1)
                    log.row(HEAD, SYS, T, "ms por símbolo (entrenamiento)", r["ms_per_step"], 0.0, 1)
                    if pct in SCAN_EXP1_PUBLISHED:
                        log.row(HEAD, "seq2seq, mejor en conjunto (publicado, 100 000 presentaciones)", f"{pct} %",
                                "secuencia exacta, test completo", SCAN_EXP1_PUBLISHED[pct], 0.0, 0)
                    return r
                st.run(f"scan exp1 {pct}% {ps} pasadas semilla {sd}", fn)
    st.finish()


# ====================================================================== pocos ejemplos (MLC)
def mlc_read(path):
    """Un episodio: (ejemplos de estudio, consultas), cada uno [(palabras, salida)]."""
    sup, qry, mode = [], [], None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("*"):
            mode = line.strip("*")
            continue
        if not line.startswith("IN:") or mode not in ("SUPPORT", "QUERY"):
            continue
        a, b = line[4:].split(" OUT:")
        (sup if mode == "SUPPORT" else qry).append((tuple(a.split()), tuple(b.split())))
    return sup, qry


MLC_PRIMS = ("1", "2", "3", "DAX")
MLC_MAX = 8


def mlc_random_episode(rng, n_support=14, n_query=10):
    """Episodio de meta-entrenamiento de la misma familia que el de las personas: 4 primitivas
    con su símbolo de salida y 3 palabras de función con reglas sorteadas (una de un argumento
    que repite, una de dos argumentos atómicos con un patrón de 3, una de dos argumentos que
    ordena dos tramos). Los nombres de las palabras y de las salidas se sortean en cada
    episodio: lo único que se puede aprender entre episodios es a leer las reglas de los
    ejemplos. No se usa la gramática del episodio de prueba (se sortea otra)."""
    words = [f"w{i}" for i in rng.permutation(12)[:7]]
    outs = [f"o{i}" for i in rng.permutation(8)[:4]]
    prim = dict(zip(words[:4], outs))
    f1, f2, f3 = words[4:]
    rep = int(rng.integers(2, 4))                                  # f1: u1 f1 -> u1 x rep
    pats = [(0, 1, 0), (1, 0, 1), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
    pat = pats[int(rng.integers(len(pats)))]                       # f2: u1 f2 u2 -> patrón sobre (u1, u2)
    swap = bool(rng.integers(2))                                   # f3: x1 f3 x2 -> x2 x1 (o x1 x2)

    def atom():
        p = words[int(rng.integers(4))]
        return (p,), (prim[p],)

    def phrase(depth=1):
        k = int(rng.integers(3)) if depth else 0
        if k == 0:
            return atom()
        if k == 1:
            (a,), (oa,) = atom()
            return (a, f1), (oa,) * rep
        (a,), (oa,) = atom(); (b,), (ob,) = atom()
        return (a, f2, b), tuple((oa, ob)[i] for i in pat)

    def sentence():
        if rng.random() < 0.5:
            return phrase()
        (l, lo), (r, ro) = phrase(), phrase()
        return l + (f3,) + r, (ro + lo) if swap else (lo + ro)
    sup = [((p,), (o,)) for p, o in prim.items()]
    seen = {s[0] for s in sup}
    while len(sup) < n_support:
        x = sentence()
        if x[0] not in seen and len(x[1]) <= MLC_MAX - 2:
            sup.append(x); seen.add(x[0])
    qry = []
    while len(qry) < n_query:
        x = sentence()
        if x[0] not in seen and len(x[0]) > 1 and len(x[1]) <= MLC_MAX - 2:
            qry.append(x); seen.add(x[0])
    return sup, qry


def mlc_vocab():
    toks = ([f"w{i}" for i in range(12)] + [f"o{i}" for i in range(8)] + ["1", "2", "3", "DAX", "after", "thrice", "surround"]
            + [f"out:{p}" for p in MLC_PRIMS] + ["<sep>", "<fin>"] + [f"<pos{t}>" for t in range(MLC_MAX + 1)])
    return {w: i for i, w in enumerate(toks)}


def mlc_episode(net, sup, qry, V, learn, out_of=lambda o: o, free=False):
    """Un episodio de pocos ejemplos: se leen los ejemplos de estudio (palabras, separador,
    salida, silencio) y se responden las consultas. Devuelve [(palabras, producido, objetivo)]."""
    net.reset()
    for words, out in sup:
        for w in words:
            net.observe(V[w])
        net.observe(V["<sep>"])
        for o in out:
            net.observe(V[out_of(o)])
        net.observe(None)
    res = []
    for words, out in qry:
        gold = [V[out_of(o)] for o in out]
        q = tuple(V[w] for w in words)
        outs = emit(net, q, gold, V, learn, MLC_MAX if free else len(gold) + 1, free)
        res.append((words, outs, gold))
    return res


def mlc_errors(words, outs, gold, V):
    """Tipo de error como en Lake y Baroni (2023): uno-a-uno (mismo largo que el comando y cada
    primitiva en su lugar con su símbolo) e icónico (en consultas con «after», la salida es el
    objetivo al revés)."""
    prim_out = {p: V[f"out:{p}"] for p in MLC_PRIMS}
    one = len(outs) == len(words) and all(o == prim_out[w] for w, o in zip(words, outs) if w in prim_out)
    iconic = ("after" in words) and list(outs)[::-1] == list(gold)
    return one, iconic


def mlc_human():
    """Exactitud de las personas por consulta y tipos de error (de sus respuestas publicadas)."""
    gold = dict(mlc_read(MLC / "val_gold" / "mini_scan.txt")[1])
    V = mlc_vocab()
    acc, n_err, n_one, n_after_err, n_icon = {}, 0, 0, 0, 0
    for f in sorted((MLC / "val_behavior").glob("*.txt")):
        _, qry = mlc_read(f)
        words = qry[0][0]
        ok = [out == gold[words] for _, out in qry]
        acc[" ".join(words)] = float(np.mean(ok))
        for (_, out), good in zip(qry, ok):
            if good:
                continue
            outs = [V[f"out:{o}"] for o in out if o in MLC_PRIMS]
            one, iconic = mlc_errors(words, outs if len(outs) == len(out) else [-1] * len(out), [V[f"out:{o}"] for o in gold[words]], V)
            n_err += 1; n_one += int(one)
            if "after" in words:
                n_after_err += 1; n_icon += int(iconic)
    return dict(acc=acc, mean=float(np.mean(list(acc.values()))), one2one=n_one / max(n_err, 1),
                iconic=n_icon / max(n_after_err, 1), n_err=n_err)


def mlc_run(seed, n_meta, over=None):
    V = mlc_vocab()
    rng = np.random.default_rng(seed)
    net = Thinker(len(V), seed=seed, **dict(BASE, **(over or {})))
    online = []
    t0 = time.time()
    for _ in range(n_meta):
        sup, qry = mlc_random_episode(rng)
        res = mlc_episode(net, sup, qry, V, True)
        online.append(np.mean([o == g for _, o, g in res]))
    secs = time.time() - t0
    net.reset(); net.plastic = False
    # meta-prueba 1: episodios nuevos de la misma familia (gramáticas sorteadas)
    held = []
    for _ in range(30):
        sup, qry = mlc_random_episode(rng)
        held += [o == g for _, o, g in mlc_episode(net, sup, qry, V, False, free=True)]
    # meta-prueba 2: el episodio de las personas (palabras y gramática nunca vistas)
    sup, qry = mlc_read(MLC / "val_gold" / "mini_scan.txt")
    res = mlc_episode(net, sup, qry, V, False, out_of=lambda o: f"out:{o}", free=True)
    acc = {" ".join(w): float(o == g) for w, o, g in res}
    errs = [mlc_errors(w, o, g, V) + (("after" in w),) for w, o, g in res if o != g]
    inv = {i: w for w, i in V.items()}
    return dict(human_task=acc, human_task_mean=float(np.mean(list(acc.values()))), held_out=float(np.mean(held)),
                online_last=float(np.mean(online[-max(n_meta // 5, 1):])) if online else 0.0,
                one2one=float(np.mean([e[0] for e in errs])) if errs else float("nan"),
                iconic=float(np.mean([e[1] for e in errs if e[2]])) if any(e[2] for e in errs) else float("nan"),
                outputs={" ".join(w): " ".join(inv.get(x, "?") for x in o) for w, o, g in res},
                ms_per_episode=secs / max(n_meta, 1) * 1e3)


def group_mlc(sizes=(0, 30, 100, 300), seeds=SEEDS):
    st = Store("f6_mlc")
    HEAD = "F6b pocos ejemplos (tarea de Lake y Baroni 2023)"
    SYS = "Thinker (todo encendido + contexto temporal)"

    def human(log):
        h = mlc_human()
        log.row(HEAD, "personas (respuestas publicadas, n=25)", "episodio de prueba", "secuencia exacta, media de las 10 consultas", h["mean"], 0.0, 0)
        log.row(HEAD, "personas (respuestas publicadas, n=25)", "episodio de prueba", "errores uno-a-uno (fracción de los errores)", h["one2one"], 0.0, 0)
        log.row(HEAD, "personas (respuestas publicadas, n=25)", "episodio de prueba", "errores icónicos (de los errores con «after»)", h["iconic"], 0.0, 0)
        # publicado en el artículo: 80.7 % exacto; 24.4 % de los errores uno-a-uno; 23.3 % icónicos;
        # MLC muestreando: 82.4 %, 56.3 % y 13.8 %; MLC con decodificación voraz: 100 %
        for label, a, o, i in (("personas (publicado en el artículo)", 0.807, 0.244, 0.233), ("MLC, muestreo (publicado)", 0.824, 0.563, 0.138),
                               ("MLC, voraz (publicado)", 1.0, float("nan"), float("nan"))):
            log.row(HEAD, label, "episodio de prueba", "secuencia exacta", a, 0.0, 0)
            log.row(HEAD, label, "episodio de prueba", "errores uno-a-uno (fracción de los errores)", o, 0.0, 0)
            log.row(HEAD, label, "episodio de prueba", "errores icónicos (de los errores con «after»)", i, 0.0, 0)
        return h
    st.run("mlc personas", human)
    for n in sizes:
        def fn(log, n=n):
            res = [mlc_run(s, n) for s in seeds]
            for key, what in (("human_task_mean", "secuencia exacta, episodio de las personas"),
                              ("held_out", "secuencia exacta, episodios nuevos de la familia de entrenamiento"),
                              ("online_last", "secuencia exacta en línea, último quinto del meta-entrenamiento"),
                              ("one2one", "errores uno-a-uno (fracción de los errores)"),
                              ("iconic", "errores icónicos (de los errores con «after»)")):
                v = [r[key] for r in res]
                log.row(HEAD, SYS, f"{n} episodios de meta-entrenamiento", what, float(np.nanmean(v)), float(np.nanstd(v)), len(v))
            for q in res[0]["human_task"]:
                v = [r["human_task"][q] for r in res]
                log.row(HEAD, SYS, f"{n} episodios, por consulta", q, float(np.mean(v)), float(np.std(v)), len(v))
            return dict(human_task=[r["human_task"] for r in res], held_out=[r["held_out"] for r in res],
                        online_last=[r["online_last"] for r in res], one2one=[r["one2one"] for r in res],
                        iconic=[r["iconic"] for r in res], outputs=[r["outputs"] for r in res],
                        ms_per_episode=[r["ms_per_episode"] for r in res])
        st.run(f"mlc {n}", fn)
    st.finish()


# ====================================================================== resumen para la batería comparada
CITES = {
    "SCAN": "Lake, B. M. y Baroni, M. (2018). Generalization without systematicity: On the compositional skills "
            "of sequence-to-sequence recurrent networks. ICML 2018. arXiv:1711.00350. Exactitud de secuencia exacta en "
            "el test: «mejor» = la mejor arquitectura de cada experimento; «mejor en conjunto» = LSTM 2 capas x 200, sin "
            "atención, dropout 0.5; 100 000 presentaciones; media de 5 corridas.",
    "MLC": "Lake, B. M. y Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural "
           "network. Nature 623, 115-121. Tabla 2 (error de MLC en SCAN) y texto principal (tarea de pocos ejemplos: "
           "personas 80.7 % exacto, 24.4 % de sus errores uno-a-uno, 23.3 % de los errores en consultas con «after» "
           "icónicos; MLC muestreando 82.4 %, 56.3 % y 13.8 %).",
    "bAbI": "Sukhbaatar, S., Szlam, A., Weston, J. y Fergus, R. (2015). End-To-End Memory Networks. NeurIPS 2015. "
            "arXiv:1503.08895, tabla 1 (1 000 ejemplos de entrenamiento por tarea, error de test por tarea): MemNN con "
            "supervisión fuerte y LSTM (de Weston et al. 2015, arXiv:1502.05698), MemN2N BoW y MemN2N PE LS RN (3 saltos, "
            "entrenamiento por tarea, la mejor de 10 corridas por error de entrenamiento). La tabla se leyó del PDF de arXiv.",
}
PROTOCOL = {
    "SCAN": "En línea, una sola pasada: cada comando de entrenamiento (submuestra al azar del train de la división) se "
            "lee palabra por palabra y el Thinker produce la secuencia un símbolo por vez (pregunta = palabras del comando "
            "+ posición ordinal), con el objetivo de cada posición como única señal. Test congelado, sin objetivo, hasta "
            "FIN, sobre una muestra al azar del test. Desviaciones respecto del artículo: entrenamiento con 250-4000 "
            "comandos vistos una vez (no 100 000 presentaciones del train completo); test sobre una muestra; en add jump "
            "la primitiva «jump» sola entra una vez.",
    "MLC": "Meta-entrenamiento en línea con episodios de una familia de gramáticas propia (4 primitivas, una palabra de "
           "función que repite, una de dos argumentos con un patrón de 3, una que ordena dos tramos; nombres de palabras "
           "y salidas sorteados en cada episodio). Prueba congelada en el episodio de las personas (14 ejemplos de "
           "estudio, 10 consultas), con palabras y salidas nunca vistas. Desviaciones: otra familia de gramáticas que "
           "la de MLC (100 000 episodios); respuesta por posición ordinal; las personas respondieron con círculos de "
           "colores, acá son símbolos de salida.",
    "bAbI": "Por tarea: hasta 3 pasadas en línea sobre el 90 % de las historias de entrenamiento (1 000 preguntas por "
            "tarea), la mejor pasada elegida por exactitud congelada en el 10 % restante y evaluada una vez en el test "
            "(1 000 preguntas). Cada historia es un episodio: las oraciones se leen palabra por palabra (silencio al "
            "final), la pregunta se muestra entera y se da la respuesta correcta como objetivo. Respuestas de varias "
            "palabras (listas, caminos) son un símbolo. Sin hechos de apoyo (supervisión débil, como MemN2N).",
}


def export(path=None):
    """Junta los resultados de F6 en un JSON por división / tarea para la batería comparada:
    media, desvío, semillas, azar, protocolo, desviaciones y cita de cada cifra publicada."""
    import json
    from common import RESULTS
    out = {"fuente_datos": {
        "SCAN": "github.com/brendenlake/SCAN @ c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f",
        "MLC": "github.com/brendenlake/MLC @ 34837ccb06c855bb29558ad99bb1cef70debd559 (data_human/few_shot)",
        "bAbI": "github.com/bgavran/DNC @ 668a2f1ecc676748c68cedab78dd198aca190d3b, "
                "src/task_implementations/bAbI/tasks_1-20_v1-2/en (bAbI v1.2, 1k)"},
        "citas": CITES, "protocolo": PROTOCOL, "SCAN": {}, "MLC": {}, "bAbI": {}}
    import inspect
    import subprocess
    sig = inspect.signature(Thinker).parameters
    out["opciones"] = dict(
        thinker_por_defecto={k: sig[k].default for k in ("prev_io", "back", "task_sets", "cogmap", "modulators", "context")},
        base_del_experimento=BASE,
        commit=subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        nota="Las opciones encendidas son las de la punta de agent/razonamiento3 más BASE (memory, speak, judge, max_steps=10, context=True).")
    out["azar"] = {"SCAN": "secuencia exacta: ~0 (hasta 48 símbolos de 6 acciones)",
                   "MLC": "secuencia exacta: ~0 (2-6 símbolos de 4)", "bAbI": "por tarea: uniforme sobre las respuestas del train y clase mayoritaria del test (filas «azar»)"}
    for group, key in (("f6_scan", "SCAN"), ("f6_mlc", "MLC"), ("f6_babi", "bAbI")):
        f = RESULTS / f"razonamiento3_{group}.parts.json"
        if not f.exists():
            continue
        parts = json.load(open(f))
        for name, part in parts.items():
            rows = [dict(prueba=r[0], sistema=r[1], tramo=r[2], medida=r[3], media=r[4], desvio=r[5], semillas=r[6])
                    for r in part["rows"]]
            out[key][name] = dict(filas=rows, resumen=part["summary"])
    (RESULTS / "razonamiento3_f6_externo.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("escrito", RESULTS / "razonamiento3_f6_externo.json")


GROUPS = {"babi": group_babi, "scan": group_scan, "mlc": group_mlc, "export": export, "scan_curva": group_scan_curve,
          "scan0": lambda: group_scan(seeds=(0,)), "scan12": lambda: group_scan(seeds=(1, 2)),
          "babi0": lambda: group_babi(seeds=(0,)), "babi12": lambda: group_babi(seeds=(1, 2)),
          "babi_todas": lambda: group_babi(seeds=SEEDS), "scan_exp1": group_scan_sizes,
          "scan_exp1_pasadas": lambda: group_scan_sizes(pcts=(1,), passes=(1, 3))}


def main():
    for g in sys.argv[1:] or ["babi"]:
        t0 = time.time()
        GROUPS[g]()
        print(f"[{g}] {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
