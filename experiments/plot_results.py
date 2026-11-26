"""Figura resumen a partir de los CSV de results/.

  python experiments/plot_results.py   ->   results/overview.png
"""
from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from common import RESULTS, ROOT

INK, INK2, GRID, SURFACE = "#191919", "#666666", "#e8e8e8", "#ffffff"
BIO, REF, REF2 = "#257de0", "#f47230", "#9c9c9c"      # BioElastic / referencia / referencia secundaria


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: [r[k] for r in rows] for k in rows[0]}


def num(col):
    return [float(v) for v in col]


def style(ax, title, xlabel, pct=True):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    if pct:
        ax.set_ylim(0, 1.04)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))


def end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", fontsize=8.5,
                color=INK, va="center", annotation_clip=False)


def legend(ax, **kw):
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, handlelength=1.6, **kw)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "lines.solid_capstyle": "round"})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    (a, b, c), (d, e, f) = axes

    # A. sintético del bioNet original
    sup = read(RESULTS / "synthetic_C10_supervised.csv")
    rew = read(RESULTS / "synthetic_C10_reward.csv")
    leg = read(ROOT / "legacy" / "bioNet_online_B_log.csv")
    x = num(sup["step"])
    a.plot(x, num(sup["acc_eval"]), color=BIO, lw=2, label="BioElastic, supervisado")
    a.plot(x, num(rew["acc_eval"]), color=BIO, lw=2, ls=(0, (4, 2)), label="BioElastic, sólo recompensa")
    a.plot(num(leg["step"]), num(leg["acc_eval"]), color=REF, lw=2, label="bioNet original")
    end_label(a, x[-1], num(sup["acc_eval"])[-1], f"{num(sup['acc_eval'])[-1]:.1%}", dy=5)
    end_label(a, x[-1], num(rew["acc_eval"])[-1], f"{num(rew['acc_eval'])[-1]:.1%}", dy=-7)
    end_label(a, x[-1], num(leg["acc_eval"])[-1], f"{num(leg['acc_eval'])[-1]:.1%}")
    style(a, "Tarea sintética del bioNet original (10 clases)", "pasos online")
    legend(a, loc="lower right", bbox_to_anchor=(1.0, 0.08))

    # B, C. MNIST online
    m = read(RESULTS / "mnist_online.csv")
    x = num(m["step"])
    b.plot(x, num(m["acc_win"]), color=BIO, lw=2)
    end_label(b, x[-1], num(m["acc_win"])[-1], f"{num(m['acc_win'])[-1]:.1%}")
    style(b, "MNIST: exactitud online (ventana de 2k)", "imágenes vistas")
    b.set_ylim(0.6, 1.01)
    c.plot(x, num(m["created"]), color=REF2, lw=2, label="creadas (acumulado)")
    c.plot(x, num(m["neurons"]), color=BIO, lw=2, label="vivas")
    end_label(c, x[-1], num(m["created"])[-1], f"{int(num(m['created'])[-1]):,}")
    end_label(c, x[-1], num(m["neurons"])[-1], f"{int(num(m['neurons'])[-1]):,}")
    style(c, "MNIST: neuronas ocultas", "imágenes vistas", pct=False)
    c.set_ylim(0, None); legend(c, loc="upper left")

    # D. texto
    t = read(RESULTS / "text_online.csv")
    x = num(t["byte"])
    d.plot(x, num(t["acc_3gram"]), color=REF2, lw=2, label="n-gramas orden 3")
    d.plot(x, num(t["acc_5gram"]), color=REF, lw=2, label="n-gramas orden 5 (backoff)")
    d.plot(x, num(t["acc_win"]), color=BIO, lw=2, label="BioElastic")
    end_label(d, x[-1], num(t["acc_5gram"])[-1], f"{num(t['acc_5gram'])[-1]:.1%}", dy=6)
    end_label(d, x[-1], num(t["acc_win"])[-1], f"{num(t['acc_win'])[-1]:.1%}", dy=-4)
    end_label(d, x[-1], num(t["acc_3gram"])[-1], f"{num(t['acc_3gram'])[-1]:.1%}", dy=-8)
    style(d, "Texto: acierto del próximo byte (ventana 20k)", "bytes leídos")
    d.set_ylim(0.2, 0.62); legend(d, loc="lower right")

    # E, F. incremental por clases
    ev = read(RESULTS / "incremental_eval.csv")
    tasks = ["0-1", "2-3", "4-5", "6-7", "8-9"]
    for model, color, name in (("mlp_sgd", REF, "MLP + SGD"), ("bioelastic", BIO, "BioElastic")):
        ys = []
        for k in range(1, 6):
            row = next(i for i in range(len(ev["model"]))
                       if ev["model"][i] == model and int(ev["after_task"][i]) == k)
            ys.append(sum(float(ev[f"acc_task{j}"][row]) for j in range(1, k + 1)) / k)
        e.plot(range(1, 6), ys, color=color, lw=2, marker="o", ms=9, mec=SURFACE, mew=2, label=name)
        end_label(e, 5, ys[-1], f"{ys[-1]:.1%}")
    style(e, "Incremental: exactitud en los dígitos ya vistos", "dígitos presentados hasta la tarea")
    e.set_xticks(range(1, 6)); e.set_xticklabels(tasks); e.set_xlim(0.7, 5.3)
    legend(e, loc="lower left", bbox_to_anchor=(0.0, 0.04))

    g = read(RESULTS / "incremental_growth.csv")
    x = num(g["step"]); task = [int(v) for v in g["task"]]
    f.plot(x, num(g["neurons"]), color=BIO, lw=2)
    end_label(f, x[-1], num(g["neurons"])[-1], f"{int(num(g['neurons'])[-1]):,}")
    style(f, "Incremental: neuronas ocultas por tarea", "imágenes vistas", pct=False)
    f.set_ylim(0, None)
    top = f.get_ylim()[1]
    starts = [0.0] + [x[i] for i in range(1, len(x)) if task[i] != task[i - 1]]
    for s, name in zip(starts, tasks):
        if s:
            f.axvline(s, color=GRID, lw=1)
        f.text(s + 600, top * 0.97, name, fontsize=8.5, color=INK2, va="top")

    fig.tight_layout(w_pad=3.0, h_pad=2.2)
    out = RESULTS / "overview.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
