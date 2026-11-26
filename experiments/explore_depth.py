"""Exploración en validación (subconjuntos): variantes de error local, proyección y V2.

Nunca toca el test. Cada variante se entrena con las primeras `--train` imágenes del orden
de la semilla y se mide sobre las primeras `--val` de la validación (últimas 5 000 de
entrenamiento). Informa la red completa y cada componente por separado.

  python experiments/explore_depth.py mnist --variants a,b,c [--seeds 0] [--train 10000]
"""
from __future__ import annotations

import argparse

from common import CsvLog
from exp_vision import DEEP, V1, V2, run

B = dict(DEEP, projection=dict(source="all", balance=True))


def L(spec, **local):
    return dict(spec, local=dict(dict(lr=0.02), **local))


VARIANTS = {
    "d2_v02": dict(areas=[V1], **DEEP),
    "d3_v02": dict(areas=[V1, V2], **DEEP),
    "d3_bal": dict(areas=[V1, V2], **B),                           # sólo proyección balanceada
    "d2_loc": dict(areas=[L(V1)], **B),
    "d3_loc": dict(areas=[L(V1), L(V2)], **B),
    "d3_loc_nobal": dict(areas=[L(V1), L(V2)], **DEEP),            # error local sin balance
    "d3_loc_v2only": dict(areas=[V1, L(V2)], **B),                 # error local sólo en V2
    "d2_loc005": dict(areas=[L(V1, lr=0.005)], **B),
    "d3_loc005": dict(areas=[L(V1, lr=0.005), L(V2, lr=0.005)], **B),
    "d2_loc05": dict(areas=[L(V1, lr=0.05)], **B),
    "d3_loc05": dict(areas=[L(V1, lr=0.05), L(V2, lr=0.05)], **B),
    "d3_loc05_v2k16": dict(areas=[L(V1, lr=0.05), L(dict(V2, k=16, floor=0.1), lr=0.05)], **B),
    "d3_loc05_v2f512": dict(areas=[L(V1, lr=0.05), L(dict(V2, max_features=512), lr=0.05)], **B),
    "d3_loc_v1t90": dict(areas=[L(dict(V1, theta=0.9)), L(V2)], **B),
    # árbitro + error local fuerte (lr 0.5, el mejor en explore_area)
    "d2_arb": dict(areas=[L(V1, lr=0.5)], arbiter=0.01, **B),
    "d3_arb": dict(areas=[L(V1, lr=0.5), L(V2, lr=0.5)], arbiter=0.01, **B),
    "d2_arb_lr0": dict(areas=[L(V1, lr=0.0)], arbiter=0.01, **B),      # sólo lector local
    "d3_arb_lr0": dict(areas=[L(V1, lr=0.0), L(V2, lr=0.0)], arbiter=0.01, **B),
    "d3_arb_norep": dict(areas=[L(V1, lr=0.5), L(V2, lr=0.5)], arbiter=0.01,
                         **dict(B, replay=0.0)),                         # sin repaso
    "d3_arb_nomem": dict(areas=[L(V1, lr=0.5), L(V2, lr=0.5)], arbiter=0.01,
                         **dict(B, heads=[])),                           # sin memoria (ni repaso)
    "d2_sum": dict(areas=[L(V1, lr=0.5)], **B),                         # suma sin árbitro
    "d3_sum": dict(areas=[L(V1, lr=0.5), L(V2, lr=0.5)], **B),
    "d2_rnd": dict(areas=[L(V1, feedback="random")], **B),         # lector local fijo aleatorio
    "d3_rnd": dict(areas=[L(V1, feedback="random"), L(V2, feedback="random")], **B),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="mnist")
    ap.add_argument("--variants", default="d2_v02,d3_v02,d2_loc,d3_loc")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--train", type=int, default=10_000)
    ap.add_argument("--val", type=int, default=5000)
    ap.add_argument("--tag", default="", help="sufijo del CSV (no pisa corridas anteriores)")
    args = ap.parse_args()
    log = CsvLog(f"explore_{args.name}_n{args.train}{args.tag}",
                 ["variant", "seed", "preq_acc", "val_acc", "val_parts", "features", "secs"])
    for v in args.variants.split(","):
        for seed in (int(s) for s in args.seeds.split(",")):
            net, preq, val, _, secs = run(args.name, v, seed, args.train, args.val, False,
                                          tag="_explore", cfg=VARIANTS[v], combos=True)
            parts = " ".join(f"{k}:{x:.4f}" for k, x in val.items() if k != "net")
            log.row(v, seed, preq, val["net"], parts, net.stats()["features"], round(secs, 1))
    log.close()


if __name__ == "__main__":
    main()
