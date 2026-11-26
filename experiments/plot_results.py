"""Figuras a partir de los CSV de results/.

  python experiments/plot_results.py   ->   results/overview.png, results/scaling.png
"""
from __future__ import annotations

import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from common import RESULTS
from bioelastic.blueprint import BRAIN_SIM_REFERENCE, HUMAN, plan

INK, INK2, GRID, SURFACE = "#191919", "#666666", "#e8e8e8", "#ffffff"
BLUE, ORANGE, AQUA, GRAY = "#257de0", "#f47230", "#00b894", "#9c9c9c"
TASK_SIZES = (2115, 2042, 1874, 1986, 1983)     # imágenes de test de MNIST por tarea (0-1, 2-3, ...)


def read(name):
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


def col(rows, key, **where):
    return [float(r[key]) for r in rows if all(r[k] == v for k, v in where.items())]


def kfmt(v, _=None):
    for div, suf in ((1e6, "M"), (1e3, "k")):
        if abs(v) >= div:
            return f"{v / div:g}{suf}"
    return f"{v:g}"


def style(ax, title, xlabel, pct=True):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    if pct:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))


def end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(7, dy), textcoords="offset points", fontsize=8.5,
                color=INK, va="center", annotation_clip=False)


def legend(ax, **kw):
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, handlelength=1.8, **kw)


def overview():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8))
    (a, b, c), (d, e, f) = axes

    # A. profundidad
    curves = []
    for name, file, color in (("MNIST", "mnist_summary.csv", BLUE),
                              ("Fashion-MNIST", "fashion_summary.csv", ORANGE)):
        rows = read(file)
        curves.append((name, color, [col(rows, "test_acc", model=f"depth{k}")[0] for k in (1, 2, 3)], 9))
    rows = read("fsdd_summary.csv")
    curves.append(("Voz (FSDD)", AQUA,
                   [col(rows, "test_acc", model=f"depth{k}", order_seed="mean")[0] for k in (1, 2, 3)], -14))
    for name, color, ys, dy in curves:
        a.plot((1, 2, 3), ys, color=color, lw=2, marker="o", ms=9, mec=SURFACE, mew=2, label=name)
        for k, y in zip((1, 2, 3), ys):
            a.annotate(f"{y:.1%}", (k, y), xytext=(0, dy), textcoords="offset points",
                       ha="center", fontsize=8.5, color=INK)
    style(a, "Exactitud en test según la profundidad", "niveles de la jerarquía")
    a.set_xticks((1, 2, 3)); a.set_xlim(0.7, 3.3); a.set_ylim(0.70, 1.03)
    legend(a, loc="lower right")

    # B. incremental: exactitud sobre lo ya visto
    ev = read("incremental_eval.csv")
    tasks = ["0-1", "2-3", "4-5", "6-7", "8-9"]
    spec = [("mlp_sgd", ORANGE, "-", "MLP con backprop", 0),
            ("depth2_no_replay", AQUA, "-", "2 niveles sin repaso", 0),
            ("depth1", BLUE, (0, (4, 2)), "1 nivel", -6),
            ("depth2", BLUE, "-", "2 niveles con repaso", 6)]
    for model, color, ls, name, dy in spec:
        ys = []
        for k in range(1, 6):
            r = next(r for r in ev if r["model"] == model and int(r["after_task"]) == k)
            seen = TASK_SIZES[:k]
            ys.append(sum(float(r[f"acc_task{j + 1}"]) * n for j, n in enumerate(seen)) / sum(seen))
        b.plot(range(1, 6), ys, color=color, lw=2, ls=ls, marker="o", ms=8, mec=SURFACE, mew=2, label=name)
        end_label(b, 5, ys[-1], f"{ys[-1]:.1%}", dy=dy)
    style(b, "Incremental: exactitud en los dígitos ya vistos", "dígitos presentados hasta la tarea")
    b.set_xticks(range(1, 6)); b.set_xticklabels(tasks); b.set_xlim(0.7, 5.3); b.set_ylim(0, 1.05)
    legend(b, loc="lower left", bbox_to_anchor=(0.0, 0.02))

    # C. incremental: crecimiento
    g = read("incremental_growth.csv")
    x = col(g, "step", model="depth2"); y = col(g, "assoc_neurons", model="depth2")
    c.plot(x, y, color=BLUE, lw=2)
    end_label(c, x[-1], y[-1], f"{int(y[-1]):,}")
    style(c, "Incremental: neuronas de la memoria", "imágenes vistas", pct=False)
    c.set_ylim(0, None)
    top = c.get_ylim()[1]
    rows = [r for r in g if r["model"] == "depth2"]
    starts = [0.0] + [float(rows[i]["step"]) for i in range(1, len(rows)) if rows[i]["task"] != rows[i - 1]["task"]]
    for s0, name in zip(starts, tasks):
        if s0:
            c.axvline(s0, color=GRID, lw=1)
        c.text(s0 + 600, top * 0.98, name, fontsize=8.5, color=INK2, va="top")

    # D. texto
    t = read("text_online.csv")
    x = col(t, "byte")
    for key, color, name, dy in (("acc_3gram", GRAY, "n-gramas orden 3", -8),
                                 ("acc_5gram", ORANGE, "n-gramas orden 5 (backoff)", 6),
                                 ("acc_win", BLUE, "BioElastic", -4)):
        y = col(t, key)
        d.plot(x, y, color=color, lw=2, label=name)
        end_label(d, x[-1], y[-1], f"{y[-1]:.1%}", dy=dy)
    style(d, "Texto: acierto del próximo byte (ventana 20k)", "bytes leídos")
    d.set_ylim(0.2, 0.62); legend(d, loc="lower right")

    # E. memoria de trabajo
    m = read("memory.csv")
    for model, color, name in (("window", GRAY, "sólo ventana de 8 pasos"),
                               ("window+trace", BLUE, "ventana + área de traza lenta")):
        x = col(m, "episode", model=model); y = col(m, "acc_win", model=model)
        e.plot(x, y, color=color, lw=2, label=name)
        end_label(e, x[-1], y[-1], f"{y[-1]:.1%}")
    e.axhline(0.25, color=GRID, lw=1)
    e.text(5450, 0.275, "azar", fontsize=8.5, color=INK2)
    style(e, "Memoria: recordar una clave tras 10-30 pasos", "episodios")
    e.set_ylim(0, 1.05); legend(e, loc="center right")

    # F. tarea sintética del bioNet original
    sup = read("synthetic_C10_supervised.csv"); rew = read("synthetic_C10_reward.csv")
    ref = read("reference_bionet_original.csv")
    x = col(sup, "step")
    f.plot(x, col(sup, "acc_eval"), color=BLUE, lw=2, label="BioElastic, supervisado")
    f.plot(x, col(rew, "acc_eval"), color=BLUE, lw=2, ls=(0, (4, 2)), label="BioElastic, sólo recompensa")
    f.plot(col(ref, "step"), col(ref, "acc_eval"), color=ORANGE, lw=2, label="bioNet original")
    end_label(f, x[-1], col(sup, "acc_eval")[-1], f"{col(sup, 'acc_eval')[-1]:.1%}", dy=5)
    end_label(f, x[-1], col(rew, "acc_eval")[-1], f"{col(rew, 'acc_eval')[-1]:.1%}", dy=-7)
    end_label(f, x[-1], col(ref, "acc_eval")[-1], f"{col(ref, 'acc_eval')[-1]:.1%}")
    style(f, "Tarea sintética del bioNet original (10 clases)", "pasos online")
    f.set_ylim(0, 1.05); legend(f, loc="lower right", bbox_to_anchor=(1.0, 0.08))

    fig.tight_layout(w_pad=3.2, h_pad=2.4)
    fig.savefig(RESULTS / "overview.png", dpi=150)
    print(RESULTS / "overview.png")


def scaling():
    s = read("scaling.csv")
    n = col(s, "neurons"); syn = col(s, "synapses")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    y = col(s, "ns_per_synapse")
    a.plot(n, y, color=BLUE, lw=2, marker="o", ms=9, mec=SURFACE, mew=2)
    for xi, yi in zip(n, y):
        a.annotate(f"{yi:.1f}", (xi, yi), xytext=(0, 9), textcoords="offset points", ha="center",
                   fontsize=8.5, color=INK)
    a.set_xscale("log")
    style(a, "Costo por sinapsis al crecer la red (ns, un núcleo)", "neuronas instanciadas", pct=False)
    a.set_ylim(0, max(y) * 1.6)

    mem = col(s, "memory_bytes") if "memory_bytes" in s[0] else col(s, "synapse_bytes")
    human = plan(1.0, index=True, slack=1.5)                 # memoria total, no sólo sinapsis
    human_syn = human["synapses"]
    per_syn = human["memory_total_bytes"] / human_syn
    b.plot([syn[0], human_syn], [syn[0] * per_syn, human_syn * per_syn], color=GRID, lw=2, zorder=1)
    b.scatter(syn, mem, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    b.scatter([human_syn], [human_syn * per_syn], s=70, color=ORANGE, edgecolor=SURFACE,
              linewidth=2, zorder=3)
    k = BRAIN_SIM_REFERENCE
    b.scatter([k["synapses"]], [1e15], s=70, color=GRAY, edgecolor=SURFACE, linewidth=2, zorder=3)
    b.annotate("medido en esta máquina", (syn[-1], mem[-1]), xytext=(8, -12), textcoords="offset points",
               fontsize=8.5, color=INK)
    b.annotate(f"corteza humana\n(extrapolado, {per_syn:.0f} B/sinapsis en total)", (human_syn, human_syn * per_syn),
               xytext=(-10, -6), textcoords="offset points", fontsize=8.5, color=INK, ha="right", va="top")
    b.annotate("K computer 2013\n(simulación real, ~1 PB)", (k["synapses"], 1e15), xytext=(-10, 4),
               textcoords="offset points", fontsize=8.5, color=INK, ha="right", va="bottom")
    b.set_xscale("log"); b.set_yscale("log")
    style(b, "Memoria real según el número de sinapsis", "sinapsis", pct=False)
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{int(round(math.log10(v)))}}}$"))
    units = {1e6: "1 MB", 1e9: "1 GB", 1e12: "1 TB", 1e15: "1 PB"}
    b.set_yticks(list(units)); b.set_yticklabels(list(units.values()))
    b.set_ylim(3e6, 3e16)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(RESULTS / "scaling.png", dpi=150)
    print(RESULTS / "scaling.png")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "lines.solid_capstyle": "round"})
    overview()
    if (RESULTS / "scaling.csv").exists():
        scaling()
