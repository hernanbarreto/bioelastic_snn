"""Codificación predictiva en la jerarquía sensorial: diagnóstico por nivel y profundidad.

Una pasada online (test-then-train) sobre las primeras `--train` imágenes del orden de la
semilla; validación = primeras `--val` de las últimas 5 000 de entrenamiento; test = primeras
`--test` del test oficial (se evalúa una sola vez, al final de cada corrida). Nada aprende en
la evaluación.

Por nivel (0 = píxeles, i = área sensorial i) se miden:
  sonda     lector lineal de 3 factores propio de cada nivel (misma regla que la proyección
            directa), entrenado online en la misma pasada y que no influye en la red; se
            informa su exactitud prequential (ventanas), en validación y en test
  dens      fracción de unidades activas del código
  dicc      rasgos vivos, rasgos efectivos (exp de la entropía del uso en validación) y
            coseno máximo medio entre rasgos (redundancia dentro del nivel)
  cka       CKA lineal entre los códigos de dos niveles (redundancia entre niveles, sobre
            `--cka` imágenes de validación)
  errores   solapamiento de errores de las sondas de dos niveles (|ambas fallan| / |alguna
            falla|) y exactitud del oráculo (alguna acierta)
  pe        error de predicción descendente del área (fracción de la energía de su entrada
            que su predicción no explica), si el área es predictiva

  python experiments/exp_predictive.py run mnist --depths 1,2,3 --seeds 0,1,2 --tag f0
         [--train 10000] [--val 2000] [--test 5000] [--perturb 1000]
         [--pred '{"lr": 0.05}'] [--fb '{"iters": 1}'] [--cond nombre]
  python experiments/exp_predictive.py summarize f0
  python experiments/exp_predictive.py cnn mnist --seeds 0,1,2 --tag f3
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from common import RESULTS
from exp_vision import DEEP_B, V1L, V2L, encoder, sheet_shape, split
from perturb import perturb
from bioelastic import Cortex, Readout

RUNS = RESULTS / "predictivo_runs"
V3L = dict(rf=3, pool=1, max_features=256, theta=0.5, floor=0.2, k=4, local=dict(lr=0.5))
AREAS = (V1L, V2L, V3L)
DEFAULT_TRAIN = dict(mnist=10_000, fashion=10_000, cifar10=5_000)


def build(depth: int, pred: dict | None = None, fb: dict | None = None,
          pred_levels: str = "all", override: dict | None = None) -> dict:
    """Configuración de `Cortex` por profundidad (1 = sólo el área de asociación sobre
    píxeles; 2..4 = 1..3 áreas sensoriales con error local + memoria + proyección + árbitro).
    `pred`: opción `predictive` de las áreas; `pred_levels`: "all" o lista "1,2" (1-based)."""
    if depth == 1:
        return dict(heads=[{}])
    areas = [dict(s) for s in AREAS[:depth - 1]]
    for k, v in (override or {}).items():                 # {"2": {...}}: cambia el área 2 (1-based)
        if int(k) - 1 < len(areas):
            areas[int(k) - 1].update(v)
    if pred is not None:
        which = range(len(areas)) if pred_levels == "all" else \
            [int(v) - 1 for v in str(pred_levels).split(",")]
        for i in which:
            if i < len(areas):
                areas[i]["predictive"] = dict(pred)
    cfg = dict(areas=areas, **DEEP_B)
    if fb:
        cfg["feedback"] = dict(fb)
    return cfg


# ---------------------------------------------------------------- sondas por nivel
def unit_sparse_vec(v: np.ndarray):
    v = np.asarray(v, dtype=np.float32).ravel()
    idx = np.flatnonzero(v)
    h = v[idx]
    n = float(np.sqrt(h @ h)) if len(h) else 0.0
    return idx, (h / np.float32(max(n, 1e-6))).astype(np.float32)


class Probes:
    """Un lector lineal de 3 factores por nivel, aparte de la red (no le devuelve nada)."""

    def __init__(self, dims, C: int = 10, lr: float = 0.5):
        self.R = []
        for d in dims:
            R = Readout(C, capacity=int(d), consolidation=0.0, lr=lr)
            R.n = int(d)
            self.R.append(R)
        self._in = None

    def predict(self, levels) -> list[int]:
        self._in = [unit_sparse_vec(L) for L in levels]
        return [int(np.argmax(R.potentials(i, h))) if len(i) else -1
                for R, (i, h) in zip(self.R, self._in)]

    def learn(self, target: int):
        for R, (i, h) in zip(self.R, self._in):
            R.learn(i, h, int(target))


def levels_of(net: Cortex, x) -> list:
    return [x] + list(net._codes)


def sub_names(net: Cortex) -> list[str]:
    """Sondas parciales de las áreas con canales de paso: rasgos propios y canales de paso."""
    out = []
    for i, a in enumerate(net.areas):
        if getattr(a, "passthrough", False):
            out += [f"L{i + 1}_own", f"L{i + 1}_pass", f"L{i + 1}_bal"]
    return out


def _unit(v):
    n = float(np.sqrt((v * v).sum()))
    return v / np.float32(n) if n > 0 else v


def sub_levels(net: Cortex) -> list:
    out = []
    for a, c in zip(net.areas, net._codes):
        if getattr(a, "passthrough", False):
            own, pas = c[:, :, :a.F], c[:, :, a.F:]
            # código completo con las dos mitades de igual norma: la sonda completa sin
            # normalizar queda dominada por los canales de paso (tienen mucha más energía)
            out += [own, pas, np.concatenate([_unit(own), _unit(pas)], axis=2)]
    return out


def sub_probes(net: Cortex):
    dims = []
    for a in net.areas:
        if getattr(a, "passthrough", False):
            dims += [a.ph * a.pw * a.F, a.ph * a.pw * (a.Fo - a.F), a.ph * a.pw * a.Fo]
    return Probes(dims) if dims else None


def area_pe(net: Cortex) -> list[float]:
    return [float(getattr(a, "pe", np.nan)) for a in net.areas]


def parts_of(net: Cortex) -> dict:
    """Potenciales de cada componente de la red tras `forward` (sin aprender)."""
    out = {}
    if net.heads:
        out["memory"] = net.heads[0]._last.V
    if net.proj is not None:
        out["proj"] = net.proj.potentials(*net._pv)
    for i, a in enumerate(net.areas):
        z = a.potentials()
        if z is not None:
            out[f"local{i + 1}"] = z
    return out


# ---------------------------------------------------------------- medidas de redundancia
def cka(A, B) -> float:
    """CKA lineal entre dos conjuntos de códigos (filas = imágenes)."""
    K = np.asarray((A @ A.T).todense() if sparse.issparse(A) else A @ A.T, dtype=np.float64)
    L = np.asarray((B @ B.T).todense() if sparse.issparse(B) else B @ B.T, dtype=np.float64)
    for M in (K, L):
        M -= M.mean(0, keepdims=True); M -= M.mean(1, keepdims=True)
    den = np.sqrt((K * K).sum() * (L * L).sum())
    return float((K * L).sum() / den) if den > 0 else float("nan")


def code_maxcorr(S, m, diag, a) -> float:
    """Media, sobre los rasgos vivos usados, de la máxima correlación de su actividad (por
    posición del código) con la de otro rasgo: redundancia de lo que el nivel transmite.
    Se usa la coactivación sin centrar normalizada (coseno entre canales de actividad)."""
    alive = np.flatnonzero(a.alive & (diag[:a.F] > 0))
    if len(alive) < 2:
        return float("nan")
    Sa = S[np.ix_(alive, alive)]
    d = np.sqrt(np.diag(Sa))
    Cn = Sa / np.outer(d, d)
    np.fill_diagonal(Cn, -1.0)
    return float(Cn.max(axis=1).mean())


def dict_stats(a, usage: np.ndarray) -> dict:
    W = a.W[a.alive]
    if len(W) > 1:
        G = W @ W.T; np.fill_diagonal(G, -1.0)
        maxcos = float(G.max(axis=1).mean()); red90 = int((G > 0.9).sum() // 2)
    else:
        maxcos = float("nan"); red90 = 0
    p = usage[usage > 0].astype(np.float64)
    eff = float(np.exp(-(p / p.sum() * np.log(p / p.sum())).sum())) if len(p) else 0.0
    return dict(alive=int(a.alive.sum()), eff_features=eff, mean_maxcos=maxcos, pairs_cos90=red90)


# ---------------------------------------------------------------- evaluación
def evaluate(net: Cortex, probes: Probes, E, X, y, collect: int = 0, extra=None, subs=None) -> dict:
    """Exactitud de la red, de sus componentes y de las sondas; pe y densidad por nivel.
    `collect`: guarda los códigos de las primeras `collect` imágenes (CKA) y el uso de rasgos.
    `extra(net, x, i)`: medida adicional por imagen (se acumula en una lista)."""
    for a in net.areas:
        a._cache = None
    nL = len(net.areas) + 1
    hits = {"net": 0}
    phits = np.zeros(nL); dens = np.zeros(nL); pe = np.zeros(len(net.areas))
    pe_parts = np.zeros((len(net.areas), 2))
    correct = np.zeros((len(y), nL), dtype=bool)
    sub_hits = np.zeros(len(subs.R)) if subs is not None else None
    rows = [[] for _ in range(nL)]
    usage = [np.zeros(a.out_shape[2]) for a in net.areas]
    co = [None for _ in net.areas] if collect else []    # coactivación por posición (dispersa)
    npos = [0 for _ in net.areas]
    ext = []
    t0 = time.time()
    for i, (x0, t) in enumerate(zip(X, y)):
        x = E(x0)
        pred = net.predict(x)
        hits["net"] += int(pred == t)
        for k, V in parts_of(net).items():
            hits[k] = hits.get(k, 0) + int(int(np.argmax(V)) == t)
        lv = levels_of(net, x)
        pp = probes.predict(lv)
        correct[i] = np.array(pp) == t
        if subs is not None:
            sub_hits += np.array(subs.predict(sub_levels(net))) == t
        for j, L in enumerate(lv):
            dens[j] += np.count_nonzero(L) / L.size
        pe += np.nan_to_num(area_pe(net))
        pe_parts += np.nan_to_num(np.array([getattr(a, "pe_parts", (np.nan, np.nan)) for a in net.areas],
                                           dtype=float).reshape(-1, 2))
        if i < collect:
            for j, L in enumerate(lv):
                idx, h = unit_sparse_vec(L)
                rows[j].append(sparse.csr_matrix((h, (np.zeros(len(idx), int), idx)), shape=(1, L.size)))
        if collect:
            for j, c in enumerate(net._codes):
                usage[j] += (c.reshape(-1, c.shape[-1]) > 0).sum(axis=0)
                if i < collect:
                    Cs = sparse.csr_matrix(c.reshape(-1, c.shape[-1]))
                    S = (Cs.T @ Cs).tocsr()
                    co[j] = S if co[j] is None else co[j] + S
                    npos[j] += Cs.shape[0]
        if extra is not None:
            ext.append(extra(net, x, i))
    secs = time.time() - t0
    n = len(y)
    out = dict(acc={k: v / n for k, v in hits.items()},
               probe=(correct.mean(axis=0)).tolist(), density=(dens / n).tolist(),
               pe=[float(v) if a.__dict__.get("pc") else float("nan") for v, a in zip(pe / n, net.areas)]
               if any(a.__dict__.get("pc") for a in net.areas) else [],
               pe_active_silent=(pe_parts / max(n, 1)).tolist(),
               ms_per_image=1000.0 * secs / max(n, 1))
    if subs is not None:                                  # sondas parciales (propios / de paso)
        out["probe_sub"] = dict(zip(sub_names(net), (sub_hits / max(n, 1)).tolist()))
    if collect:
        M = [sparse.vstack(r).tocsr() for r in rows]
        out["cka"] = {f"{i}-{j}": cka(M[i], M[j]) for i in range(nL) for j in range(i + 1, nL)}
        ov = {}
        for i in range(nL):
            for j in range(i + 1, nL):
                wi, wj = ~correct[:, i], ~correct[:, j]
                either = (wi | wj).sum()
                ov[f"{i}-{j}"] = dict(jaccard_err=float((wi & wj).sum() / max(either, 1)),
                                      oracle=float(1.0 - (wi & wj).mean()))
        out["err_overlap"] = ov
        out["dict"] = [dict_stats(a, u[:a.F]) for a, u in zip(net.areas, usage)]
        for d, a, S, m in zip(out["dict"], net.areas, co, npos):   # redundancia de la actividad
            if S is None:
                d["code_maxcorr"] = float("nan"); continue
            S = S[:a.F, :a.F].toarray().astype(np.float32)
            d["code_maxcorr"] = code_maxcorr(S, m, S.diagonal(), a)
    if extra is not None:
        out["extra"] = ext
    return out


# ---------------------------------------------------------------- completado descendente
class feedback_off:
    """Contexto: la misma red sin el bucle de inferencia (un solo pase ascendente)."""

    def __init__(self, net: Cortex):
        self.net = net; self.saved = None

    def __enter__(self):
        if getattr(self.net, "fb", None) is not None:
            self.saved = self.net.fb["iters"]; self.net.fb["iters"] = 0
        return self.net

    def __exit__(self, *exc):
        if self.saved is not None:
            self.net.fb["iters"] = self.saved


def _cos(u, v):
    nu, nv = float(np.sqrt(u @ u)), float(np.sqrt(v @ v))
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else 0.0


def completion(net: Cortex, E, Xc, Xo) -> dict:
    """¿Completa la predicción de arriba lo que falta? Para cada imagen ocluida se compara el
    código del primer área, dentro de las posiciones cuyo campo receptivo toca el oclusor, con
    el código ascendente de la imagen limpia (referencia). Con y sin el bucle de
    retroalimentación, misma red y mismos pesos. También el coseno global del área de arriba."""
    a = net.areas[0]
    ry, rx = a.rf; sy, sx = a.stride; py, px = a.pool
    span_y = (py - 1) * sy + ry; span_x = (px - 1) * sx + rx       # píxeles que ve una compleja
    yy = np.arange(a.ph) * py * sy; xx = np.arange(a.pw) * px * sx
    out = dict(region_ff=[], region_fb=[], top_ff=[], top_fb=[], region_clean_fb=[])
    for xc, xo in zip(Xc, Xo):
        d = np.asarray(xc) != np.asarray(xo)
        d = d.any(axis=-1) if d.ndim == 3 else d
        pts = np.argwhere(d)
        if len(pts) == 0:
            continue
        (r0, c0), (r1, c1) = pts.min(0), pts.max(0)
        my = (yy <= r1) & (yy + span_y - 1 >= r0); mx = (xx <= c1) & (xx + span_x - 1 >= c0)
        region = my[:, None] & mx[None, :]
        with feedback_off(net):
            net.forward(E(xc)); ref = [c.copy() for c in net._codes]
            net.forward(E(xo)); off = [c.copy() for c in net._codes]
        net.forward(E(xo)); on = [c.copy() for c in net._codes]
        net.forward(E(xc)); clean_on = [c.copy() for c in net._codes]
        out["region_ff"].append(_cos(off[0][region].ravel(), ref[0][region].ravel()))
        out["region_fb"].append(_cos(on[0][region].ravel(), ref[0][region].ravel()))
        out["region_clean_fb"].append(_cos(clean_on[0][region].ravel(), ref[0][region].ravel()))
        out["top_ff"].append(_cos(off[-1].ravel(), ref[-1].ravel()))
        out["top_fb"].append(_cos(on[-1].ravel(), ref[-1].ravel()))
    return {k: float(np.mean(v)) if v else None for k, v in out.items()} | dict(n=len(out["region_ff"]))


# ---------------------------------------------------------------- una corrida
def saccade(img, rng, jitter: int = 2, rot: float = 0.0):
    """Una micro-sacada: la imagen corrida hasta `jitter` píxeles (relleno con 0) y, si `rot`,
    girada hasta `rot` grados. Entrada y salida uint8 (H, W[, C])."""
    from perturb import _shift
    dy, dx = (int(v) for v in rng.integers(-jitter, jitter + 1, size=2))
    out = _shift(np.asarray(img), dy, dx)
    if rot:
        from scipy.ndimage import rotate
        out = rotate(out, float(rng.uniform(-rot, rot)), axes=(0, 1), reshape=False, order=1,
                     mode="constant").astype(np.uint8)
    return out


def run_one(name: str, depth: int, seed: int, n_train: int, n_val: int, n_test: int,
            n_perturb: int, cfg: dict, cond: str, tag: str, log_every: int = 1000,
            n_cka: int = 1000, extra_eval=None, views: int = 1, jitter: int = 2,
            rot: float = 0.0, perturbs=("noise", "occlude")) -> dict:
    """Con `views` > 1 cada imagen de entrenamiento se presenta como una secuencia de
    `views` vistas (la original y micro-sacadas); cada vista cuenta como una muestra del
    presupuesto `n_train` (se ven n_train / views imágenes distintas)."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(name, seed)
    E = encoder(name)
    net = Cortex(sheet_shape(name), 10, seed=seed, **cfg)
    vrng = np.random.default_rng(seed + 12345)
    dims = [int(np.prod(sheet_shape(name)))] + [int(np.prod(a.out_shape)) for a in net.areas]
    probes = Probes(dims)
    subs = sub_probes(net)
    w_sub = np.zeros(len(subs.R)) if subs is not None else None
    nL = len(dims)
    timeline = []
    w_net = 0; w_probe = np.zeros(nL); w_pe = np.zeros(len(net.areas)); w_n = 0
    w_loc = np.zeros(len(net.areas))                      # aciertos de los lectores locales
    preq = 0
    t0 = time.time()
    for i in range(n_train):
        img, v = divmod(i, views)
        if v == 0 and views > 1:
            net.new_sequence()                            # otra imagen: otra secuencia
        x = E(Xtr[img] if v == 0 else saccade(Xtr[img], vrng, jitter, rot)); t = int(ytr[img])
        pred = net.forward(x)
        pp = probes.predict(levels_of(net, x))
        pe = area_pe(net)
        w_loc += [float(z is not None and int(np.argmax(z)) == t) for z in net._zs]
        if subs is not None:
            w_sub += np.array(subs.predict(sub_levels(net))) == t
        net.teach(t)
        probes.learn(t)
        if subs is not None:
            subs.learn(t)
        w_net += int(pred == t); preq += int(pred == t)
        w_probe += np.array(pp) == t
        w_pe += np.nan_to_num(pe); w_n += 1
        if (i + 1) % log_every == 0 or i + 1 == n_train:
            st = net.stats()
            row = dict(step=i + 1, acc=w_net / w_n, probe=(w_probe / w_n).tolist(),
                       local=(w_loc / w_n).tolist(),
                       **({"probe_sub": (w_sub / w_n).tolist()} if subs is not None else {}),
                       features=[s["features"] for s in st["areas"]],
                       secs=round(time.time() - t0, 1))
            if any(getattr(a, "con", None) for a in net.areas):
                row["phi"] = [round(float(a.phi), 4) for a in net.areas]
            if any(a.__dict__.get("pc") for a in net.areas):
                row["pe"] = [float(v) if a.__dict__.get("pc") else float("nan")
                             for v, a in zip(w_pe / w_n, net.areas)]
                row["gain"] = [float(getattr(a, "gain_avg", np.nan)) for a in net.areas]
            timeline.append(row)
            print(f"[{tag} {name} d{depth} {cond} s{seed}] {json.dumps(row)}", flush=True)
            w_net = 0; w_probe[:] = 0; w_pe[:] = 0; w_n = 0; w_loc[:] = 0
            if subs is not None:
                w_sub[:] = 0
    train_secs = time.time() - t0
    val = evaluate(net, probes, E, Xva[:n_val], yva[:n_val], collect=min(n_cka, n_val), subs=subs)
    test = evaluate(net, probes, E, Xte[:n_test], yte[:n_test], extra=extra_eval, subs=subs) \
        if n_test else {}
    pert = {}
    loop = getattr(net, "fb", None) is not None and net.fb["iters"] > 0
    for p in perturbs if n_perturb else ():
        Xp = perturb(Xte[:n_perturb], p)
        pert[p] = evaluate(net, probes, E, Xp, yte[:n_perturb], subs=subs)
        if loop:                                          # misma red, sin el bucle descendente
            with feedback_off(net):
                pert[p + "_fb_off"] = evaluate(net, probes, E, Xp, yte[:n_perturb])
    if n_perturb and loop:
        pert["completion"] = completion(net, E, Xte[:min(n_perturb, 500)],
                                        perturb(Xte[:n_perturb], "occlude")[:min(n_perturb, 500)])
    if n_test and loop:
        with feedback_off(net):
            pert["clean_fb_off"] = evaluate(net, probes, E, Xte[:n_perturb], yte[:n_perturb])
    st = net.stats(); cost = net.cost()
    res = dict(tag=tag, name=name, depth=depth, cond=cond, seed=seed, n_train=n_train,
               n_val=n_val, n_test=n_test, n_perturb=n_perturb, config=cfg,
               preq_acc=preq / n_train, train_secs=train_secs,
               ms_per_step=1000.0 * train_secs / n_train,
               val=val, test=test, perturb=pert, timeline=timeline,
               features=[s["features"] for s in st["areas"]],
               macs_per_inference=cost["macs_per_inference"],
               memory_bytes=cost["memory_bytes"], synapses=st["synapses"],
               views=views, jitter=jitter, rot=rot, distinct_images=n_train // views,
               reductions=("una pasada sobre n_train imágenes (no 55 000/45 000); validación = "
                           "primeras n_val de las últimas 5 000 de entrenamiento; test = primeras "
                           "n_test del test oficial; perturbaciones sobre las primeras n_perturb"))
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{tag}_{name}_d{depth}_{cond}_s{seed}.json"
    out.write_text(json.dumps(res, default=float))
    return res


# ---------------------------------------------------------------- techo: sondas en una CNN
def cnn_probe(name: str, seeds, n_train: int, n_val: int, tag: str):
    """Control de techo: ¿la segunda capa de una CNN entrenada con backprop (una pasada, mismo
    presupuesto) le gana a la primera con la misma sonda lineal de 3 factores en una pasada?
    Si ni el backprop logra que la capa 2 supere a la 1 en este régimen, «sonda de V2 > V1»
    no es alcanzable sólo con cambiar las reglas locales."""
    from cnn_backprop import ARCH, CNN
    E = encoder(name)
    side, _, channels = sheet_shape(name)
    S = lambda X: np.stack([E(x) for x in X]).astype(np.float32)
    out = []
    for seed in seeds:
        (Xtr, ytr), (Xva, yva), _ = split(name, seed)
        Str, Sva = S(Xtr[:n_train]), S(Xva[:n_val])
        best = None
        for lr in (3e-4, 1e-3, 3e-3):                      # tasa elegida en validación
            net = CNN(ARCH["cnn2"], side=side, channels=channels, lr=lr, seed=seed)
            for i in range(0, len(Str), 10):
                net.train_batch(Str[i:i + 10], ytr[i:i + 10].astype(np.int64))
            acc = float(np.mean(net.predict(Sva) == yva[:n_val]))
            if best is None or acc > best[0]:
                best = (acc, lr, net)
        acc, lr, net = best

        def layers(X):
            feats = []
            for k, W, b in net.convs:                         # mismas operaciones que CNN.forward
                from numpy.lib.stride_tricks import sliding_window_view
                B = len(X)
                cols = sliding_window_view(X, (k, k), axis=(1, 2))
                oh, ow = cols.shape[1], cols.shape[2]
                cols = cols.transpose(0, 1, 2, 4, 5, 3).reshape(B * oh * ow, -1)
                A = np.maximum((cols @ W + b).reshape(B, oh, ow, -1), 0.0)
                ph, pw = oh // 2, ow // 2
                X = A[:, :ph * 2, :pw * 2].reshape(B, ph, 2, pw, 2, -1).max(axis=(2, 4))
                feats.append(X.reshape(B, -1))
            return feats
        dims = None; probes = None
        for i in range(0, len(Str), 100):                     # una pasada, test-then-train
            F = layers(Str[i:i + 100])
            if probes is None:
                dims = [f.shape[1] for f in F]; probes = Probes([Str[0].size] + dims)
            for j in range(len(F[0])):
                probes.predict([Str[i + j]] + [f[j] for f in F]); probes.learn(int(ytr[i + j]))
        hits = np.zeros(len(dims) + 1)
        for i in range(0, len(Sva), 100):
            F = layers(Sva[i:i + 100])
            for j in range(len(F[0])):
                hits += np.array(probes.predict([Sva[i + j]] + [f[j] for f in F])) == int(yva[i + j])
        row = dict(name=name, seed=seed, lr=lr, n_train=n_train, cnn_val=acc,
                   probe_pixels=hits[0] / len(Sva), probe_layer1=hits[1] / len(Sva),
                   probe_layer2=hits[2] / len(Sva), dims=dims)
        print(row, flush=True)
        out.append(row)
    (RESULTS / f"predictivo_{tag}_cnnprobe_{name}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


# ---------------------------------------------------------------- estímulo inesperado
def _auroc(neg, pos) -> float:
    """Probabilidad de que un positivo (novedoso) tenga más error que un negativo."""
    neg = np.asarray(neg); pos = np.asarray(pos)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def surprise(name: str, depth: int, seed: int, cfg: dict, cond: str, tag: str,
             n_a: int = 3000, n_b: int = 2000, n_held: int = 500, window: int = 100) -> dict:
    """Flujo con un cambio inesperado: primero sólo las clases 0-4, después sólo las 5-9.
    Se registra el error de predicción por nivel a lo largo del flujo; justo antes del cambio
    se mide (sin aprender) si el error distingue imágenes de clases nunca vistas (AUROC)."""
    (Xtr, ytr), (Xva, yva), _ = split(name, seed)
    E = encoder(name)
    A = np.flatnonzero(ytr < 5)[:n_a]; B = np.flatnonzero(ytr >= 5)[:n_b]
    net = Cortex(sheet_shape(name), 10, seed=seed, **cfg)
    rows = []; pe_steps = []; hits = []
    t0 = time.time()

    def held_pe(mask):
        idx = np.flatnonzero(mask)[:n_held]
        for a in net.areas:
            a._cache = None
        out = []
        for i in idx:
            net.forward(E(Xva[i])); out.append(area_pe(net))
        return np.array(out)

    held = None
    for step, i in enumerate(np.concatenate([A, B])):
        if step == len(A):                                    # justo antes del cambio
            fam, nov = held_pe(yva < 5), held_pe(yva >= 5)
            held = dict(pe_familiar=fam.mean(0).tolist(), pe_novel=nov.mean(0).tolist(),
                        auroc=[_auroc(fam[:, k], nov[:, k]) for k in range(fam.shape[1])])
            print(f"[{tag} sorpresa {cond} s{seed}] antes del cambio: {held}", flush=True)
        pred = net.forward(E(Xtr[i])); t = int(ytr[i])
        pe_steps.append(area_pe(net)); hits.append(pred == t)
        net.teach(t)
        if (step + 1) % window == 0:
            rows.append(dict(step=step + 1, acc=float(np.mean(hits[-window:])),
                             pe=np.mean(pe_steps[-window:], axis=0).tolist(),
                             gain=[float(getattr(a, "gain_avg", np.nan)) for a in net.areas],
                             features=[int(a.alive.sum()) for a in net.areas],
                             created=[int(a.created) for a in net.areas]))
    pe = np.array(pe_steps)
    nA = len(A)
    res = dict(tag=tag, name=name, depth=depth, cond=cond, seed=seed, n_a=nA, n_b=len(B),
               config=cfg, held=held, timeline=rows,
               pe_before=pe[nA - 500:nA].mean(0).tolist(),
               pe_first100_after=pe[nA:nA + 100].mean(0).tolist(),
               pe_last500_after=pe[-500:].mean(0).tolist(),
               acc_first500_after=float(np.mean(hits[nA:nA + 500])),
               acc_last500_after=float(np.mean(hits[-500:])),
               created_after=[int(a.created) for a in net.areas],
               secs=time.time() - t0)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{tag}_{name}_d{depth}_{cond}_s{seed}.json").write_text(json.dumps(res, default=float))
    print(f"[{tag} sorpresa {cond} s{seed}] " + json.dumps({k: res[k] for k in (
        "pe_before", "pe_first100_after", "pe_last500_after", "acc_first500_after")}), flush=True)
    return res


# ---------------------------------------------------------------- cambio de dominio tardío
def shift(seed: int, cfg: dict, cond: str, tag: str, name_a: str = "mnist", name_b: str = "fashion",
          n_a: int = 6000, n_b: int = 6000, n_val: int = 1000, n_test: int = 2000,
          window: int = 500) -> dict:
    """Flujo no estacionario: primero `n_a` imágenes del dominio A (clases 0-9), después `n_b`
    del dominio B (clases 10-19), una sola pasada. ¿Sigue aprendiendo la red después de
    consolidar? Se mide la exactitud en lo nuevo (B), el olvido de lo viejo (A antes y después)
    y las sondas por nivel en cada dominio. Validación y test separados por dominio."""
    (Xa, ya), (Xva_a, yva_a), (Xte_a, yte_a) = split(name_a, seed)
    (Xb, yb), (Xva_b, yva_b), (Xte_b, yte_b) = split(name_b, seed)
    E = encoder(name_a)
    shape = sheet_shape(name_a)
    net = Cortex(shape, 20, seed=seed, **cfg)
    dims = [int(np.prod(shape))] + [int(np.prod(a.out_shape)) for a in net.areas]
    probes = Probes(dims, C=20)
    subs = sub_probes(net)
    if subs is not None:
        subs = Probes([R.n for R in subs.R], C=20)
    rows = []; hits = []; phits = []
    t0 = time.time()

    def ev(X, y, off, n):
        r = evaluate(net, probes, E, X[:n], y[:n].astype(np.int64) + off, subs=subs)
        return dict(acc=r["acc"], probe=r["probe"], probe_sub=r.get("probe_sub", {}))

    at_switch = None
    for step in range(n_a + n_b):
        if step == n_a:                                       # justo antes del cambio
            at_switch = dict(A_val=ev(Xva_a, yva_a, 0, n_val),
                             phi=[float(getattr(a, "phi", np.nan)) for a in net.areas],
                             features=[int(a.alive.sum()) for a in net.areas],
                             created=[int(a.created) for a in net.areas])
            print(f"[{tag} cambio {cond} s{seed}] antes del cambio: A {at_switch['A_val']['acc']['net']:.4f} "
                  f"phi {at_switch['phi']}", flush=True)
        if step < n_a:
            x = E(Xa[step]); t = int(ya[step])
        else:
            x = E(Xb[step - n_a]); t = int(yb[step - n_a]) + 10
        pred = net.forward(x)
        pp = probes.predict(levels_of(net, x))
        if subs is not None:
            subs.predict(sub_levels(net))
        net.teach(t); probes.learn(t)
        if subs is not None:
            subs.learn(t)
        hits.append(pred == t); phits.append(np.array(pp) == t)
        if (step + 1) % window == 0:
            rows.append(dict(step=step + 1, acc=float(np.mean(hits[-window:])),
                             probe=np.mean(phits[-window:], axis=0).tolist(),
                             phi=[float(getattr(a, "phi", np.nan)) for a in net.areas],
                             reopens=[int(getattr(a, "reopens", 0)) for a in net.areas],
                             features=[int(a.alive.sum()) for a in net.areas],
                             created=[int(a.created) for a in net.areas]))
            print(f"[{tag} cambio {cond} s{seed}] {json.dumps(rows[-1])}", flush=True)
    end = dict(A_val=ev(Xva_a, yva_a, 0, n_val), B_val=ev(Xva_b, yva_b, 10, n_val),
               A_test=ev(Xte_a, yte_a, 0, n_test), B_test=ev(Xte_b, yte_b, 10, n_test))
    res = dict(tag=tag, name=f"{name_a}>{name_b}", depth=len(net.areas) + 1, cond=cond, seed=seed,
               n_a=n_a, n_b=n_b, n_val=n_val, n_test=n_test, config=cfg, at_switch=at_switch, end=end,
               timeline=rows,
               forgetting_val=at_switch["A_val"]["acc"]["net"] - end["A_val"]["acc"]["net"],
               acc_B_first1000=float(np.mean(hits[n_a:n_a + 1000])),
               acc_B_last1000=float(np.mean(hits[-1000:])),
               created_after=[int(a.created) - c for a, c in zip(net.areas, at_switch["created"])],
               reopens=[int(getattr(a, "reopens", 0)) for a in net.areas],
               secs=time.time() - t0)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{tag}_shift_d{res['depth']}_{cond}_s{seed}.json").write_text(json.dumps(res, default=float))
    print(f"[{tag} cambio {cond} s{seed}] fin: A {end['A_test']['acc']['net']:.4f} B {end['B_test']['acc']['net']:.4f} "
          f"olvido(val) {res['forgetting_val']:.4f} reaperturas {res['reopens']}", flush=True)
    return res


def summarize_shift(tag: str):
    runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob(f"{tag}_shift_*.json"))]
    groups = {}
    for r in runs:
        groups.setdefault(r["cond"], []).append(r)
    table = []
    for cond, rs in sorted(groups.items()):
        L = rs[0]["depth"]
        row = dict(cond=cond, seeds=sorted(r["seed"] for r in rs), n_a=rs[0]["n_a"], n_b=rs[0]["n_b"],
                   A_before_val=_ms([r["at_switch"]["A_val"]["acc"]["net"] for r in rs]),
                   A_after_val=_ms([r["end"]["A_val"]["acc"]["net"] for r in rs]),
                   forgetting_val=_ms([r["forgetting_val"] for r in rs]),
                   A_test=_ms([r["end"]["A_test"]["acc"]["net"] for r in rs]),
                   B_test=_ms([r["end"]["B_test"]["acc"]["net"] for r in rs]),
                   B_val=_ms([r["end"]["B_val"]["acc"]["net"] for r in rs]),
                   acc_B_first1000=_ms([r["acc_B_first1000"] for r in rs]),
                   acc_B_last1000=_ms([r["acc_B_last1000"] for r in rs]),
                   phi_at_switch=[_ms([r["at_switch"]["phi"][i] for r in rs]) for i in range(L - 1)],
                   reopens=[_ms([r["reopens"][i] for r in rs]) for i in range(L - 1)],
                   created_after=[_ms([r["created_after"][i] for r in rs]) for i in range(L - 1)])
        for dom in ("A_test", "B_test"):
            row[f"{dom}_probe"] = [_ms([r["end"][dom]["probe"][i] for r in rs]) for i in range(L)]
            keys = sorted(rs[0]["end"][dom]["probe_sub"])
            row[f"{dom}_probe_sub"] = {k: _ms([r["end"][dom]["probe_sub"][k] for r in rs]) for k in keys}
        row["timeline_steps"] = [t["step"] for t in rs[0]["timeline"]]
        row["timeline_acc"] = [float(np.mean([r["timeline"][k]["acc"] for r in rs]))
                               for k in range(len(row["timeline_steps"]))]
        row["timeline_phi"] = [np.mean([r["timeline"][k]["phi"] for r in rs], axis=0).tolist()
                               for k in range(len(row["timeline_steps"]))]
        table.append(row)
    (RESULTS / f"predictivo_{tag}_shift.json").write_text(json.dumps(table, indent=1, default=float))
    fmt = lambda m: "" if m is None else f"{m[0]:.4f}±{m[1]:.4f}"
    with (RESULTS / f"predictivo_{tag}_shift.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cond", "n_seeds", "A_before_val", "A_after_val", "forgetting_val", "A_test", "B_test",
                    "acc_B_first1000", "acc_B_last1000", "phi_at_switch", "reopens", "created_after",
                    "B_test_probe_by_level", "B_test_probe_sub", "A_test_probe_by_level"])
        for r in table:
            j = lambda k: " | ".join(fmt(m) for m in r[k])
            w.writerow([r["cond"], len(r["seeds"]), fmt(r["A_before_val"]), fmt(r["A_after_val"]),
                        fmt(r["forgetting_val"]), fmt(r["A_test"]), fmt(r["B_test"]),
                        fmt(r["acc_B_first1000"]), fmt(r["acc_B_last1000"]), j("phi_at_switch"),
                        j("reopens"), j("created_after"), j("B_test_probe"),
                        " | ".join(f"{k}:{fmt(v)}" for k, v in r["B_test_probe_sub"].items()),
                        j("A_test_probe")])
            print(r["cond"], "A", fmt(r["A_test"]), "B", fmt(r["B_test"]), "olvido", fmt(r["forgetting_val"]),
                  "B primeras 1000", fmt(r["acc_B_first1000"]), "phi", j("phi_at_switch"),
                  "sondas B", j("B_test_probe"))
    return table


def summarize_surprise(tag: str):
    """Resumen del estímulo inesperado por condición: media, desvío y n entre semillas."""
    runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob(f"{tag}_*.json"))]
    groups = {}
    for r in runs:
        groups.setdefault((r["name"], r["depth"], r["cond"]), []).append(r)
    table = []
    for (name, depth, cond), rs in sorted(groups.items()):
        L = depth - 1
        row = dict(name=name, depth=depth, cond=cond, seeds=sorted(r["seed"] for r in rs),
                   n_a=rs[0]["n_a"], n_b=rs[0]["n_b"])
        for k in ("pe_before", "pe_first100_after", "pe_last500_after"):
            row[k] = [_ms([r[k][i] for r in rs]) for i in range(L)]
        row["jump_ratio"] = [_ms([r["pe_first100_after"][i] / max(r["pe_before"][i], 1e-9) for r in rs])
                             for i in range(L)]
        row["recovery_ratio"] = [_ms([r["pe_last500_after"][i] / max(r["pe_first100_after"][i], 1e-9)
                                      for r in rs]) for i in range(L)]
        row["auroc_novel"] = [_ms([r["held"]["auroc"][i] for r in rs]) for i in range(L)]
        row["pe_familiar"] = [_ms([r["held"]["pe_familiar"][i] for r in rs]) for i in range(L)]
        row["pe_novel"] = [_ms([r["held"]["pe_novel"][i] for r in rs]) for i in range(L)]
        row["acc_first500_after"] = _ms([r["acc_first500_after"] for r in rs])
        row["acc_last500_after"] = _ms([r["acc_last500_after"] for r in rs])
        row["created_after"] = [_ms([r["created_after"][i] for r in rs]) for i in range(L)]
        row["timeline_steps"] = [t["step"] for t in rs[0]["timeline"]]
        row["timeline_pe"] = [np.mean([r["timeline"][k]["pe"] for r in rs], axis=0).tolist()
                              for k in range(len(row["timeline_steps"]))]
        row["timeline_acc"] = [float(np.mean([r["timeline"][k]["acc"] for r in rs]))
                               for k in range(len(row["timeline_steps"]))]
        table.append(row)
    (RESULTS / f"predictivo_{tag}.json").write_text(json.dumps(table, indent=1, default=float))
    fmt = lambda m: "" if m is None else f"{m[0]:.4f}±{m[1]:.4f}"
    with (RESULTS / f"predictivo_{tag}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "depth", "cond", "n_seeds", "pe_before", "pe_first100_after",
                    "pe_last500_after", "jump_ratio", "recovery_ratio", "auroc_novel",
                    "acc_first500_after", "acc_last500_after", "created_after"])
        for r in table:
            j = lambda k: " | ".join(fmt(m) for m in r[k])
            w.writerow([r["name"], r["depth"], r["cond"], len(r["seeds"]), j("pe_before"),
                        j("pe_first100_after"), j("pe_last500_after"), j("jump_ratio"),
                        j("recovery_ratio"), j("auroc_novel"), fmt(r["acc_first500_after"]),
                        fmt(r["acc_last500_after"]), j("created_after")])
            print(r["cond"], "salto", j("jump_ratio"), "recuperación", j("recovery_ratio"),
                  "auroc", j("auroc_novel"), "acc tras el cambio", fmt(r["acc_first500_after"]))
    return table


# ---------------------------------------------------------------- resumen
def _ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return None
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)]


def summarize(tag: str, out: str | None = None):
    """Agrupa las corridas `tag_*` por (datos, profundidad, condición): media, desvío, n.
    `tag` puede ser una lista separada por comas ("f0,f3"); `out` nombra el resumen."""
    runs = [json.loads(p.read_text()) for t in tag.split(",") for p in sorted(RUNS.glob(f"{t}_*.json"))]
    tag = out or tag.replace(",", "_")
    groups = {}
    for r in runs:
        groups.setdefault((r["name"], r["depth"], r["cond"]), []).append(r)
    table = []
    for (name, depth, cond), rs in sorted(groups.items()):
        nL = depth if depth > 1 else 1
        row = dict(name=name, depth=depth, cond=cond, seeds=sorted(r["seed"] for r in rs),
                   n_train=rs[0]["n_train"], n_test=rs[0]["n_test"],
                   preq_acc=_ms([r["preq_acc"] for r in rs]),
                   val_acc=_ms([r["val"]["acc"]["net"] for r in rs]),
                   test_acc=_ms([r["test"]["acc"]["net"] for r in rs if r["test"]]),
                   ms_per_step=_ms([r["ms_per_step"] for r in rs]),
                   ms_per_image=_ms([r[k]["ms_per_image"] for r in rs
                                     for k in (("test",) if r["test"] else ("val",))]),
                   macs_per_inference=_ms([r["macs_per_inference"] for r in rs]),
                   features=[_ms([r["features"][i] for r in rs]) for i in range(depth - 1)])
        for part in ("val", "test"):
            for k in sorted({k for r in rs if r[part] for k in r[part]["acc"]}):
                row[f"{part}_{k}"] = _ms([r[part]["acc"].get(k) for r in rs if r[part]])
        for part in ("val", "test"):
            keys = sorted({k for r in rs if r[part] for k in r[part].get("probe_sub", {})})
            if keys:
                row[f"{part}_probe_sub"] = {k: _ms([r[part]["probe_sub"].get(k) for r in rs
                                                    if r[part] and "probe_sub" in r[part]]) for k in keys}
        row["val_probe"] = [_ms([r["val"]["probe"][i] for r in rs]) for i in range(nL)]
        row["test_probe"] = [_ms([r["test"]["probe"][i] for r in rs if r["test"]]) for i in range(nL)]
        if rs[0]["val"].get("pe"):
            row["val_pe"] = [_ms([r["val"]["pe"][i] for r in rs]) for i in range(depth - 1)]
        row["density"] = [_ms([r["val"]["density"][i] for r in rs]) for i in range(nL)]
        if rs[0]["test"] and rs[0]["test"].get("pe"):
            row["test_pe"] = [_ms([r["test"]["pe"][i] for r in rs]) for i in range(depth - 1)]
        if "cka" in rs[0]["val"]:
            row["cka"] = {k: _ms([r["val"]["cka"][k] for r in rs]) for k in rs[0]["val"]["cka"]}
            row["err_jaccard"] = {k: _ms([r["val"]["err_overlap"][k]["jaccard_err"] for r in rs])
                                  for k in rs[0]["val"]["err_overlap"]}
            row["oracle"] = {k: _ms([r["val"]["err_overlap"][k]["oracle"] for r in rs])
                             for k in rs[0]["val"]["err_overlap"]}
            row["dict"] = [{k: _ms([r["val"]["dict"][i][k] for r in rs])
                            for k in rs[0]["val"]["dict"][i]} for i in range(depth - 1)]
        for p in ("noise", "occlude", "shift3", "noise_fb_off", "occlude_fb_off", "shift3_fb_off",
                  "clean_fb_off"):
            if all(p in r["perturb"] for r in rs):
                row[f"{p}_acc"] = _ms([r["perturb"][p]["acc"]["net"] for r in rs])
                row[f"{p}_probe"] = [_ms([r["perturb"][p]["probe"][i] for r in rs]) for i in range(nL)]
                if rs[0]["perturb"][p].get("pe"):
                    row[f"{p}_pe"] = [_ms([r["perturb"][p]["pe"][i] for r in rs])
                                      for i in range(depth - 1)]
        if all("completion" in r["perturb"] for r in rs):
            row["completion"] = {k: _ms([r["perturb"]["completion"][k] for r in rs])
                                 for k in rs[0]["perturb"]["completion"] if k != "n"}
        if rs[0]["test"] and "pe_active_silent" in rs[0]["test"] and rs[0]["test"].get("pe"):
            row["test_pe_active_silent"] = [[_ms([r["test"]["pe_active_silent"][i][j] for r in rs])
                                             for j in range(2)] for i in range(depth - 1)]
        # evolución temporal (media entre semillas) de sondas y pe por nivel
        steps = [t["step"] for t in rs[0]["timeline"]]
        row["timeline_steps"] = steps
        row["timeline_acc"] = [float(np.mean([r["timeline"][k]["acc"] for r in rs]))
                               for k in range(len(steps))]
        row["timeline_probe"] = [np.mean([r["timeline"][k]["probe"] for r in rs], axis=0).tolist()
                                 for k in range(len(steps))]
        if "pe" in rs[0]["timeline"][0]:
            row["timeline_pe"] = [np.mean([r["timeline"][k]["pe"] for r in rs], axis=0).tolist()
                                  for k in range(len(steps))]
        table.append(row)
    (RESULTS / f"predictivo_{tag}.json").write_text(json.dumps(table, indent=1, default=float))
    fmt = lambda m: "" if m is None else f"{m[0]:.4f}±{m[1]:.4f}"
    with (RESULTS / f"predictivo_{tag}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "depth", "cond", "n_seeds", "n_train", "test_acc", "val_acc",
                    "preq_acc", "test_probe_by_level", "val_probe_by_level", "val_pe_by_level",
                    "noise_acc", "occlude_acc", "shift3_acc", "ms_per_step", "ms_per_image",
                    "macs_per_inference"])
        for r in table:
            w.writerow([r["name"], r["depth"], r["cond"], len(r["seeds"]), r["n_train"],
                        fmt(r["test_acc"]), fmt(r["val_acc"]), fmt(r["preq_acc"]),
                        " | ".join(fmt(m) for m in r["test_probe"]),
                        " | ".join(fmt(m) for m in r["val_probe"]),
                        " | ".join(fmt(m) for m in r.get("val_pe", [])),
                        fmt(r.get("noise_acc")), fmt(r.get("occlude_acc")), fmt(r.get("shift3_acc")),
                        fmt(r["ms_per_step"]), fmt(r["ms_per_image"]), fmt(r["macs_per_inference"])])
    for r in table:
        print(r["name"], r["depth"], r["cond"], "val", fmt(r["val_acc"]), "test", fmt(r["test_acc"]),
              "val probes", " | ".join(fmt(m) for m in r["val_probe"]),
              "pe", " | ".join(fmt(m) for m in r.get("val_pe", [])))
    return table


# ---------------------------------------------------------------- CNN de referencia
def cnn(name: str, seeds, n_train: int, n_val: int, n_test: int, n_perturb: int, tag: str):
    """CNN con backprop (experiments/cnn_backprop.py), mismo presupuesto y mismas imágenes de
    validación y test que las corridas de este script; tasa elegida en validación."""
    from cnn_backprop import ARCH, CNN
    E = encoder(name)
    side, _, channels = sheet_shape(name)
    S = lambda X: np.stack([E(x) for x in X]).astype(np.float32)
    out = []
    for seed in seeds:
        (Xtr, ytr), (Xva, yva), (Xte, yte) = split(name, seed)
        Str, Sva, Ste = S(Xtr[:n_train]), S(Xva[:n_val]), S(Xte[:n_test])
        for model in ARCH:
            best = None
            for lr in (3e-4, 1e-3, 3e-3):
                net = CNN(ARCH[model], side=side, channels=channels, lr=lr, seed=seed)
                t0 = time.time()
                for i in range(0, len(Str), 10):
                    net.train_batch(Str[i:i + 10], ytr[i:i + 10].astype(np.int64))
                secs = time.time() - t0
                acc = float(np.mean(net.predict(Sva) == yva[:n_val]))
                if best is None or acc > best[0]:
                    best = (acc, lr, net, secs)
            acc, lr, net, secs = best
            row = dict(name=name, model=model, seed=seed, lr=lr, n_train=n_train, val_acc=acc,
                       test_acc=float(np.mean(net.predict(Ste) == yte[:n_test])),
                       train_secs=secs)
            for p in ("noise", "occlude") if n_perturb else ():
                Sp = S(perturb(Xte[:n_perturb], p))
                row[f"{p}_acc"] = float(np.mean(net.predict(Sp) == yte[:n_perturb]))
            print(row, flush=True)
            out.append(row)
    path = RESULTS / f"predictivo_{tag}_cnn_{name}.json"
    path.write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize", "cnn", "surprise", "summarize_surprise",
                                    "shift", "summarize_shift", "cnnprobe"])
    ap.add_argument("name", nargs="?", default="mnist")
    ap.add_argument("--depths", default="1,2,3")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--test", type=int, default=5000)
    ap.add_argument("--perturb", type=int, default=1000)
    ap.add_argument("--pred", default="", help="JSON de la opción `predictive` de las áreas")
    ap.add_argument("--pred-levels", default="all")
    ap.add_argument("--override", default="", help='JSON {"2": {...}}: cambia parámetros de un área')
    ap.add_argument("--fb", default="", help="JSON de la opción `feedback` de Cortex")
    ap.add_argument("--cond", default="ff")
    ap.add_argument("--tag", default="f0")
    ap.add_argument("--skip-existing", action="store_true", help="no repite corridas ya guardadas")
    ap.add_argument("--out", default="", help="nombre del resumen (summarize)")
    ap.add_argument("--views", type=int, default=1, help="vistas por imagen (micro-sacadas)")
    ap.add_argument("--jitter", type=int, default=2, help="desplazamiento máximo de una vista (px)")
    ap.add_argument("--rot", type=float, default=0.0, help="giro máximo de una vista (grados)")
    ap.add_argument("--perturbs", default="noise,occlude")
    args = ap.parse_args()
    if args.cmd == "summarize":
        summarize(args.name, args.out or None)
        return
    if args.cmd == "summarize_surprise":
        summarize_surprise(args.name)
        return
    if args.cmd == "summarize_shift":
        summarize_shift(args.name)
        return
    n_train = args.train or DEFAULT_TRAIN[args.name]
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.cmd == "cnn":
        cnn(args.name, seeds, n_train, args.val, args.test, args.perturb, args.tag)
        return
    if args.cmd == "cnnprobe":
        cnn_probe(args.name, seeds, n_train, args.val, args.tag)
        return
    pred = json.loads(args.pred) if args.pred else None
    fb = json.loads(args.fb) if args.fb else None
    over = json.loads(args.override) if args.override else None
    if args.cmd == "shift":                               # `name` no se usa: dígitos -> prendas
        for depth in (int(d) for d in args.depths.split(",")):
            for seed in seeds:
                if args.skip_existing and (RUNS / f"{args.tag}_shift_d{depth}_{args.cond}_s{seed}.json").exists():
                    continue
                shift(seed, build(depth, pred, fb, args.pred_levels, over), args.cond, args.tag,
                      n_a=args.train or 6000, n_b=args.train or 6000)
        return
    if args.cmd == "surprise":
        for depth in (int(d) for d in args.depths.split(",")):
            for seed in seeds:
                if args.skip_existing and (RUNS / f"{args.tag}_{args.name}_d{depth}_{args.cond}_s{seed}.json").exists():
                    continue
                surprise(args.name, depth, seed, build(depth, pred, fb, args.pred_levels, over),
                         args.cond, args.tag)
        return
    for depth in (int(d) for d in args.depths.split(",")):
        for seed in seeds:
            cfg = build(depth, pred, fb, args.pred_levels, over)
            if depth == 1 and (pred or fb):
                continue                                  # sin áreas no hay predicción
            if args.skip_existing and (RUNS / f"{args.tag}_{args.name}_d{depth}_{args.cond}_s{seed}.json").exists():
                continue
            run_one(args.name, depth, seed, n_train, args.val, args.test, args.perturb, cfg,
                    args.cond, args.tag, views=args.views, jitter=args.jitter, rot=args.rot,
                    perturbs=tuple(args.perturbs.split(",")))


if __name__ == "__main__":
    main()
