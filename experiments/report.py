"""Tablas de métricas (Markdown) a partir de los CSV de results/. Es lo que cita el README.

  python experiments/report.py
"""
from __future__ import annotations

import csv

from common import RESULTS
from bioelastic.blueprint import fmt


def read(name):
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


def pct(x):
    return f"{100 * float(x):.1f} %"


def get(rows, key, **where):
    return next(r[key] for r in rows if all(r[k] == v for k, v in where.items()))


def main():
    print("### Exactitud en test por profundidad (una pasada online)\n")
    print("| Tarea | 1 nivel | 2 niveles | 3 niveles | MLP con backprop |")
    print("|---|---|---|---|---|")
    for name, file in (("MNIST", "mnist_summary.csv"), ("Fashion-MNIST", "fashion_summary.csv")):
        r = read(file)
        mlp = next(x for x in r if x["model"].startswith("mlp"))
        print(f"| {name} | " + " | ".join(pct(get(r, "test_acc", model=f"depth{k}")) for k in (1, 2, 3))
              + f" | {pct(mlp['test_acc'])} |")
    r = read("fsdd_summary.csv")
    cells = []
    for k in (1, 2, 3):
        accs = [float(x["test_acc"]) for x in r if x["model"] == f"depth{k}" and x["order_seed"] != "mean"]
        cells.append(f"{pct(get(r, 'test_acc', model=f'depth{k}', order_seed='mean'))} "
                     f"({100 * min(accs):.1f}-{100 * max(accs):.1f})")
    print("| Voz, palabra completa (FSDD, 4 órdenes) | " + " | ".join(cells) + " | — |")
    accs = [float(x["test_acc"]) for x in r if x["model"] == "stream" and x["order_seed"] != "mean"]
    print(f"\nVoz frame a frame (1 nivel, streaming): {pct(get(r, 'test_acc', model='stream', order_seed='mean'))} "
          f"({100 * min(accs):.1f}-{100 * max(accs):.1f})")

    print("\n### Tamaño de las redes\n")
    print("| Tarea | Modelo | Rasgos | Neuronas de asociación | Unidades | Sinapsis | s de entrenamiento |")
    print("|---|---|---|---|---|---|---|")
    for name, file in (("MNIST", "mnist_summary.csv"), ("Fashion-MNIST", "fashion_summary.csv")):
        for x in read(file):
            print(f"| {name} | {x['model']} | {x['features'] or '—'} | {int(float(x['assoc_neurons'])):,} | "
                  f"{int(float(x['units'])):,} | {int(float(x['synapses'])):,} | {x['train_seconds'] or '—'} |")

    print("\n### Incremental por clases (MNIST, 5 tareas, sin volver a ver datos)\n")
    ev = [x for x in read("incremental_eval.csv") if x["after_task"] == "5"]
    print("| Modelo | Los 10 dígitos | Tarea 1 (0-1) | Tarea 5 (8-9) |")
    print("|---|---|---|---|")
    for x in ev:
        print(f"| {x['model']} | {pct(x['acc_all'])} | {pct(x['acc_task1'])} | {pct(x['acc_task5'])} |")

    print("\n### Texto (próximo byte, promedio de los últimos 100k bytes)\n")
    t = read("text_online.csv")[-5:]
    mean = lambda k: sum(float(x[k]) for x in t) / len(t)
    print(f"BioElastic {pct(mean('acc_win'))} · 3-gramas {pct(mean('acc_3gram'))} · "
          f"5-gramas {pct(mean('acc_5gram'))} · neuronas {int(float(t[-1]['neurons'])):,} · "
          f"memoria de sinapsis {fmt(float(t[-1]['synapse_bytes']), 'B')} · contextos del 5-grama {int(float(t[-1]['ctx_5gram'])):,}")

    print("\n### Memoria de trabajo (últimos 500 episodios)\n")
    m = read("memory.csv")
    for model in ("window", "window+trace"):
        last = [x for x in m if x["model"] == model][-1]
        print(f"{model}: {pct(last['acc_win'])} ({last['assoc_neurons']} neuronas)")

    print("\n### Tarea sintética del bioNet original (16k pasos)\n")
    for C in (10, 20):
        for mode in ("supervised", "reward"):
            x = read(f"synthetic_C{C}_{mode}.csv")
            print(f"C={C} {mode}: {pct(x[-1]['acc_eval'])} (paso 1000: {pct(x[0]['acc_eval'])}; "
                  f"{x[-1]['neurons']} neuronas)")
    print(f"bioNet original: {pct(read('reference_bionet_original.csv')[-1]['acc_eval'])}")

    if (RESULTS / "scaling.csv").exists():
        print("\n### Escalado medido\n")
        print("| Neuronas | Sinapsis | Memoria de sinapsis | ns por sinapsis | Sinapsis/s por núcleo |")
        print("|---|---|---|---|---|")
        for x in read("scaling.csv"):
            print(f"| {fmt(float(x['neurons']))} | {fmt(float(x['synapses']))} | "
                  f"{fmt(float(x['synapse_bytes']), 'B')} | {float(x['ns_per_synapse']):.2f} | "
                  f"{fmt(float(x['syn_per_core_second']))} |")
        print("\n| Escala (corteza humana = 1) | Neuronas | Sinapsis | Memoria | Núcleo-segundos por paso |")
        print("|---|---|---|---|---|")
        for x in read("human_scale.csv"):
            print(f"| {x['scale']} | {x['neurons']} | {x['synapses']} | {x['memory']} | {x['core_seconds_per_step']} |")


if __name__ == "__main__":
    main()
