"""Figura de la v0.3 a partir de los CSV de results/ (3 semillas, media y desvío).

  python experiments/plot_v03.py   ->   results/v03.png
"""
from __future__ import annotations

import csv
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from common import RESULTS

INK, INK2, GRID, SURFACE = "#191919", "#666666", "#e8e8e8", "#ffffff"
BLUE, ORANGE, AQUA, GRAY = "#257de0", "#f47230", "#00b894", "#9c9c9c"
W = 0.36                                   # ancho de barra
GAP = dict(edgecolor=SURFACE, linewidth=2)  # separación de 2 px entre barras vecinas


def read(name):
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


def stats(rows, key, **where):
    v = [float(r[key]) for r in rows if all(r[k] == w for k, w in where.items())]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def pm(s):
    """'0.8988±0.0075' -> (0.8988, 0.0075)"""
    m, d = s.split("±")
    return float(m), float(d)


def style(ax, title, ylabel=None, pct=True):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=10, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    if pct:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))


def bars(ax, x, vals, errs, color, label):
    ax.bar(x, vals, W, yerr=errs, color=color, label=label, error_kw=dict(ecolor=INK2, lw=1, capsize=2),
           **GAP)


def legend(ax, **kw):
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, handlelength=1.4, **kw)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    (a, b), (c, d) = axes

    # A. profundidad: exactitud de test por niveles, contra la CNN de una pasada
    datasets = (("MNIST", "mnist"), ("Fashion-MNIST", "fashion"))
    for i, (name, key) in enumerate(datasets):
        rows = read(f"{key}_final_summary.csv")
        cnn = read(f"{key}_final_cnn_summary.csv")
        ms = [stats(rows, "test_acc", model=f"depth{k}") for k in (1, 2, 3)]
        ax = a if i == 0 else b
        xs = range(3)                          # eje recortado: puntos, no barras
        ax.errorbar(list(xs), [m for m, _ in ms], yerr=[s for _, s in ms], fmt="o", ms=8, color=BLUE,
                    mec=SURFACE, mew=2, ecolor=INK2, elinewidth=1, capsize=3,
                    label="BioElastic, 1 pasada online")
        best = max((stats(cnn, "test_acc", model=m) for m in ("cnn1", "cnn2")), key=lambda t: t[0])
        ax.axhline(best[0], color=ORANGE, lw=2, label="mejor CNN con backprop, 1 pasada")
        for x, (m, _) in zip(xs, ms):
            ax.annotate(f"{m:.1%}", (x, m), xytext=(12, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK)
        ax.set_xlim(-0.5, 2.6)
        ax.set_xticks(list(xs), ["1 nivel", "2 niveles", "3 niveles"])
        lo = 0.94 if key == "mnist" else 0.78
        ax.set_ylim(lo, 1.0 if key == "mnist" else 0.92)
        style(ax, f"{name}: exactitud de test por profundidad")
        legend(ax, loc="lower right")

    # C. robustez en Fashion-MNIST: limpio, ruido, oclusión
    rows = read("fashion_final_summary.csv"); cnn = read("fashion_final_cnn_summary.csv")
    cases = (("test_acc", "limpio"), ("test_noise", "ruido"), ("test_occlude", "oclusión"),
             ("test_shift3", "desplazado 3 px"))
    xs = list(range(len(cases)))
    w3 = W * 2 / 3
    series = ((stats, rows, "depth2", BLUE, "BioElastic, 2 niveles"),
              (stats, cnn, "cnn1", ORANGE, "CNN de 1 capa conv, 1 pasada"),
              (stats, cnn, "cnn2", AQUA, "CNN de 2 capas conv, 1 pasada"))
    for j, (f, src, model, color, label) in enumerate(series):
        vals = [f(src, k, model=model) for k, _ in cases]
        c.bar([x + (j - 1) * w3 for x in xs], [m for m, _ in vals], w3, yerr=[s for _, s in vals],
              color=color, label=label, error_kw=dict(ecolor=INK2, lw=1, capsize=2), **GAP)
    c.set_xticks(xs, [t for _, t in cases]); c.set_ylim(0, 1)
    style(c, "Fashion-MNIST: test con perturbaciones nunca vistas")
    c.legend(frameon=False, fontsize=8, labelcolor=INK, handlelength=1.4, loc="upper right")

    # D. aprendizaje continuo: flujo no estacionario de 200 000 pasos
    agg = {r["model"]: r for r in read("continual_agg.csv") if r["scale"] == "1.0000"}
    models = (("d2", "BioElastic"), ("mlp_replay", "MLP + replay"), ("slda", "SLDA"),
              ("knn_buffer", "kNN + buffer"))
    xs = list(range(len(models)))
    fin = [pm(agg[m]["final_clean_known"]) for m, _ in models]
    fgt = [pm(agg[m]["forget_clean"]) for m, _ in models]
    bars(d, [x - W / 2 for x in xs], [m for m, _ in fin], [s for _, s in fin], BLUE, "exactitud final")
    bars(d, [x + W / 2 for x in xs], [m for m, _ in fgt], [s for _, s in fgt], GRAY, "olvido")
    d.set_xticks(xs, [t for _, t in models]); d.set_ylim(0, 1)
    style(d, "Flujo continuo de 200 000 pasos (3 semillas)")
    legend(d, loc="upper right")

    fig.text(0.01, 0.005, "Media de 3 semillas; barras de error: desvío. Una sola pasada online "
             "(se predice antes de aprender).", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(RESULTS / "v03.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
