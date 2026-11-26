"""Exploración rápida a nivel de áreas (sin memoria ni proyección): cuánto cambia el error
local los rasgos y qué tan bien lee cada lector local la validación.

  python experiments/explore_area.py fashion --lrs 0,0.05,0.5,5 [--train 5000] [--v2]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import CsvLog
from bioelastic import FeatureArea
from exp_vision import V1, V2, encoder, sheet_shape, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="fashion")
    ap.add_argument("--lrs", default="0,0.05,0.5,5")
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v2", default="", help="JSON con cambios a V2 (vacío: sólo V1)")
    ap.add_argument("--v1", default="", help="JSON con cambios a V1")
    ap.add_argument("--residual", action="store_true", help="error local residual entre áreas")
    ap.add_argument("--tag", default="", help="sufijo del CSV (no pisa corridas anteriores)")
    args = ap.parse_args()
    import json
    (Xtr, ytr), (Xva, yva), _ = split(args.name, args.seed)
    E = encoder(args.name)
    Str = [E(x) for x in Xtr[:args.train]]; Sva = [E(x) for x in Xva[:args.val]]
    log = CsvLog(f"explore_area_{args.name}{args.tag}", ["lr", "v1", "v2", "residual", "local1", "local2",
                                               "sum12", "w_drift1", "w_drift2", "secs"])
    ref = None
    for lr in (float(v) for v in args.lrs.split(",")):
        specs = [dict(V1, **(json.loads(args.v1) if args.v1 else {}))]
        if args.v2:
            specs.append(dict(V2, **json.loads(args.v2)))
        areas, shape = [], sheet_shape(args.name)
        for s in specs:
            a = FeatureArea(shape, **s, local=dict(lr=lr, residual=args.residual), n_classes=10,
                            seed=args.seed)
            areas.append(a); shape = a.out_shape
        t0 = time.time()
        for x, y in zip(Str, ytr):
            for a in areas:
                x = a.encode(x)
            prior = np.zeros(10, dtype=np.float32)
            pots = [a.potentials() for a in areas]
            for a, z in zip(areas, pots):
                a.learn(int(y), prior); prior = prior + z
        secs = time.time() - t0
        hits = np.zeros(len(areas)); hsum = np.zeros(len(areas))
        for x, y in zip(Sva, yva):
            z = np.zeros(10, dtype=np.float32)
            for i, a in enumerate(areas):
                x = a.encode(x); V = a.potentials(); z = z + V
                hits[i] += int(np.argmax(V)) == y; hsum[i] += int(np.argmax(z)) == y
            for a in areas:
                a._cache = None
        acc = hits / len(Sva); accsum = hsum / len(Sva)
        Ws = [a.W[:a.n].copy() for a in areas]
        if ref is None:
            ref = Ws
        drift = [float(np.mean(np.sum(w[:min(len(w), len(r))] * r[:min(len(w), len(r))], 1)))
                 for w, r in zip(Ws, ref)]
        log.row(lr, args.v1, args.v2, int(args.residual), acc[0], acc[1] if len(acc) > 1 else "",
                accsum[-1] if len(acc) > 1 else "", drift[0],
                drift[1] if len(drift) > 1 else "", round(secs, 1))
    log.close()


if __name__ == "__main__":
    main()
