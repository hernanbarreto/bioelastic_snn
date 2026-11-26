"""Diagnóstico de la profundidad sensorial: qué aprende cada área y qué información pierde.

Protocolo: las áreas aprenden sin supervisión sobre las primeras `--fit` imágenes de un orden
(semilla) de las primeras 55 000 de entrenamiento; después se congelan. Se mide sobre la
validación (las últimas 5 000 de entrenamiento); el test no se toca.

Por nivel (píxeles, V1, V2 y controles):
  online   lectura de 3 factores en una pasada (la misma regla que la proyección directa)
  ridge    decodificador lineal offline (cota de la información linealmente disponible)
  pos      decodificabilidad del desplazamiento horizontal (-2..2) de la imagen
  dens     fracción de unidades activas en el código; vivos, uso efectivo y redundancia

  python experiments/diag_depth.py mnist|fashion [--fit 10000] [--n 4000] [--seed 0]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy import sparse

from common import CsvLog
from bioelastic import FeatureArea, ImageEncoder, Readout
from bioelastic.datasets import load_fashion, load_mnist

V1 = dict(rf=5, pool=2, max_features=256, theta=0.95, floor=0.3, k=8)
V2 = dict(rf=3, pool=2, max_features=256, theta=0.5, floor=0.2, k=4)


def encoder():
    enc = ImageEncoder((28, 28), blur=0.4)
    return lambda x: enc(x).reshape(28, 28, 1)


def split(name: str, seed: int):
    """(Xtr, ytr) = primeras 55 000 en el orden de la semilla; (Xva, yva) = últimas 5 000."""
    Xtr, ytr, _, _ = load_mnist() if name == "mnist" else load_fashion()
    order = np.random.default_rng(seed).permutation(55_000)
    return Xtr[order], ytr[order], Xtr[55_000:60_000], ytr[55_000:60_000]


def train_stack(specs, sheets, shape=(28, 28, 1)):
    areas = []
    for s in specs:
        a = FeatureArea(shape, **s); areas.append(a); shape = a.out_shape
    for X in sheets:
        for a in areas:
            X = a.encode(X); a.learn()
    return areas


def pool_code(code, ph, pw):
    """Máximo adaptativo de una lámina (h, w, F) a (ph, pw, F) (control de resolución)."""
    h, w, F = code.shape
    ys = np.linspace(0, h, ph + 1).astype(int); xs = np.linspace(0, w, pw + 1).astype(int)
    out = np.zeros((ph, pw, F), dtype=np.float32)
    for i in range(ph):
        for j in range(pw):
            out[i, j] = code[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].max(axis=(0, 1))
    return out


def share(a: FeatureArea, **over):
    """Otra área que comparte el diccionario de `a` pero cambia la lámina 2/3."""
    spec = dict(rf=a.rf, stride=a.stride, pool=a.pool, max_features=a.F, theta=a.theta,
                floor=a.floor, k=a.k)
    spec.update(over)
    b = FeatureArea(a.in_shape, **spec)
    b.W = a.W; b.alive = a.alive; b.n = a.n; b.plastic = False
    return b


def clone_cache(a: FeatureArea, X):
    a.encode(X); c = a._cache; a._cache = None
    return c


def unit(M):
    n = np.sqrt((M * M).sum(axis=1, keepdims=True))
    return M / np.maximum(n, 1e-6)


def unit_sparse(M):
    """CSR -> CSR con filas de norma 1 (como entran a la proyección)."""
    M = sparse.csr_matrix(M, dtype=np.float32)
    n = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    return sparse.diags(1.0 / np.maximum(n, 1e-6)).astype(np.float32) @ M


def online_readout(Ztr, ytr, Zva, yva, C=10):
    """Una pasada de la regla de 3 factores (como `Cortex.proj`) y exactitud en validación."""
    R = Readout(C, capacity=Ztr.shape[1], consolidation=0.0, lr=0.5); R.n = Ztr.shape[1]
    for i, y in enumerate(ytr):
        lo, hi = Ztr.indptr[i], Ztr.indptr[i + 1]
        R.learn(Ztr.indices[lo:hi], Ztr.data[lo:hi], int(y))
    return float(np.mean(np.asarray(Zva @ R.W[:, :Zva.shape[1]].T).argmax(1) == yva))


def gram(A, B, chunk=250):
    """A @ B.T denso, por tramos de B (acota la memoria aunque los códigos sean densos)."""
    out = np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    for i in range(0, B.shape[0], chunk):
        out[:, i:i + chunk] = A @ B[i:i + chunk].T.toarray()
    return out


def ridge(Ztr, ytr, Zva, yva, C=10, lams=(1e-2, 1e-1, 1.0, 10.0)):
    """Decodificador lineal por mínimos cuadrados regularizados (forma dual: n << d)."""
    Y = np.eye(C, dtype=np.float32)[ytr]
    K = gram(Ztr, Ztr); Kva = gram(Zva, Ztr)
    best = 0.0
    for lam in lams:
        A = np.linalg.solve(K + lam * np.eye(len(K), dtype=np.float32), Y)
        best = max(best, float(np.mean((Kva @ A).argmax(1) == yva)))
    return best


def dict_stats(a: FeatureArea, use):
    W = a.W[a.alive]
    G = W @ W.T; np.fill_diagonal(G, -1.0)
    use = use.reshape(-1, a.F).sum(axis=0).astype(np.float64)
    p = use[use > 0] / use.sum()
    return dict(alive=int(a.alive.sum()), eff=float(np.exp(-(p * np.log(p)).sum())),
                maxcos=float(G.max(axis=1).mean()), red90=float((G > 0.9).sum() / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="mnist")
    ap.add_argument("--fit", type=int, default=10_000)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    E = encoder()
    Xtr, ytr, Xva, yva = split(args.name, args.seed)
    t0 = time.time()
    v1, v2 = train_stack([V1, V2], (E(x) for x in Xtr[:args.fit]))
    print(f"áreas entrenadas en {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(100 + args.seed)
    # Control: V2 con rasgos impresos de parches V1 al azar, sin promediar (sin aprendizaje).
    rnd = FeatureArea(v1.out_shape, **V2)
    pats = []
    for x in Xtr[rng.integers(0, args.fit, size=v2.n)]:
        Pm, norms, valid = clone_cache(v2, v1.encode(E(x)))[:3]
        j = rng.choice(np.flatnonzero(valid)); pats.append(Pm[j] / norms[j])
    rnd.W[:v2.n] = np.stack(pats); rnd.n = v2.n; rnd.alive[:v2.n] = True
    # Controles de pérdida de información en V1 (mismo diccionario, otra lámina 2/3).
    v1_all = share(v1, k=v1.F)                             # sin k-WTA
    v1_np = share(v1, pool=1)                              # sin pooling

    def levels(X):
        S = E(X)
        c1 = v1.encode(S); c2 = v2.encode(c1)
        return dict(pix=S, v1=c1, v2=c2, v1_5x5=pool_code(c1, 5, 5), v2rand=rnd.encode(c1),
                    v1_nokwta=v1_all.encode(S), v1_nopool=v1_np.encode(S))

    def collect(Xs):
        raw = {}
        for x in Xs:
            for k, v in levels(x).items():
                raw.setdefault(k, []).append(sparse.csr_matrix(v.reshape(1, -1)))
        raw = {k: sparse.vstack(v).tocsr() for k, v in raw.items()}
        out = {k: unit_sparse(v) for k, v in raw.items()}
        out["v1v2"] = unit_sparse(sparse.hstack([raw["v1"], raw["v2"]]))   # como proj "all"
        out["use"] = {k: np.asarray((raw[k] > 0).sum(0)).ravel() for k in ("v1", "v2")}
        out["share_v2"] = float(np.mean(np.asarray(raw["v2"].multiply(raw["v2"]).sum(1)).ravel() /
                                        np.maximum(np.asarray(sparse.hstack([raw["v1"], raw["v2"]]).power(2).sum(1)).ravel(), 1e-9)))
        return out

    ntr = args.n
    tr = collect(Xtr[args.fit - ntr:args.fit])            # imágenes ya vistas por las áreas
    va = collect(Xva[:ntr])
    sh = rng.integers(-2, 3, size=ntr)                    # desplazamiento horizontal
    shz = collect([np.roll(x, int(d), axis=1) for x, d in zip(Xva[:ntr], sh)])
    half = ntr // 2
    print(f"códigos en {time.time() - t0:.0f}s; fracción de la norma^2 de V1+V2 que aporta V2: "
          f"{va['share_v2']:.3f}", flush=True)
    log = CsvLog(f"diag_depth_{args.name}_s{args.seed}",
                 ["level", "dims", "density", "online", "ridge", "pos", "alive", "eff_features",
                  "mean_maxcos", "pairs_cos90"])
    for k in ("pix", "v1", "v1_nokwta", "v1_nopool", "v1_5x5", "v2", "v2rand", "v1v2"):
        Ztr, Zva, Zsh = tr[k], va[k], shz[k]
        dens = Zva.nnz / np.prod(Zva.shape)
        on = online_readout(Ztr, ytr[args.fit - ntr:args.fit], Zva, yva[:ntr])
        rd = ridge(Ztr, ytr[args.fit - ntr:args.fit], Zva, yva[:ntr])
        pos = ridge(Zsh[:half], sh[:half] + 2, Zsh[half:], sh[half:] + 2, C=5)
        ds = dict_stats(v1 if k == "v1" else v2, va["use"][k]) if k in ("v1", "v2") else {}
        log.row(k, Ztr.shape[1], dens, on, rd, pos, ds.get("alive", ""), ds.get("eff", ""),
                ds.get("maxcos", ""), ds.get("red90", ""))
    log.close()


if __name__ == "__main__":
    main()
