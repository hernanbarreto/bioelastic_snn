"""Resumen de la evaluación final: media y desvío entre semillas por modelo.

  python experiments/summarize_depth.py results/fashion_final_summary.csv [...]
"""
from __future__ import annotations

import csv
import sys

import numpy as np

COLS = ("val_acc", "test_acc", "test_shift3", "test_noise", "test_occlude")


def summarize(path):
    rows = list(csv.DictReader(open(path)))
    models = list(dict.fromkeys(r["model"] for r in rows))
    print(path)
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        cells = []
        for c in COLS:
            v = [float(r[c]) for r in rs if r.get(c) not in (None, "")]
            if v:
                cells.append(f"{c}={100 * np.mean(v):.2f}±{100 * np.std(v):.2f}")
        seeds = ",".join(r["seed"] for r in rs)
        print(f"  {m:<14} semillas={seeds:<6} " + "  ".join(cells))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
