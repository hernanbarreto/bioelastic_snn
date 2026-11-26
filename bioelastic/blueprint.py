"""Plano de escala: qué significa "capacidad humana" y qué cuesta llegar con esta arquitectura.

Los números del cerebro y de los modelos de referencia vienen de la literatura (ver
`SOURCES`). Las funciones convierten una escala (1.0 = corteza humana completa) en
neuronas, sinapsis, memoria y cómputo, usando costos *medidos* en esta implementación
(`experiments/exp_scaling.py`). Todo lo que sale de `plan` es extrapolado: ver
`NOT_DEMONSTRATED`.

Dos magnitudes distintas que no hay que confundir:
  * costo por sinapsis evaluada (ns): constante, no depende del tamaño de la red.
  * costo por paso (núcleo-segundos): sinapsis evaluadas por paso x costo por sinapsis.
    Con el pase denso, cada columna que recibe una volea evalúa todas sus sinapsis, así
    que crece linealmente con la red. El índice invertido lo reduce sólo si la entrada
    es dispersa respecto del número de aferentes de la columna.

    python -m bioelastic.blueprint            # tabla de requisitos por escala
"""
from __future__ import annotations

HUMAN = dict(
    neurons_total=86e9,                 # encéfalo completo
    neurons_cortex=16e9,                # corteza cerebral
    neurons_cerebellum=69e9,            # cerebelo
    synapses_neocortex=1.5e14,          # 0.15 x 10^15
    synapses_per_cortical_neuron=7000,
    cortical_areas=360,                 # 180 por hemisferio
    cortical_laminae=6,
    visual_hierarchy_levels=10,         # macaco: niveles corticales de la jerarquía visual
)

SOURCES = {
    "neurons": "Azevedo et al. 2009, J Comp Neurol — https://pubmed.ncbi.nlm.nih.gov/19226510/",
    "synapses": "Pakkenberg et al. 2003, Exp Gerontol — https://pubmed.ncbi.nlm.nih.gov/12543266/",
    "synapses_per_neuron": "Drachman 2005, Neurology (BioNumbers 112055) — "
                           "https://bionumbers.hms.harvard.edu/bionumber.aspx?id=112055",
    "areas": "Glasser et al. 2016, Nature — https://pmc.ncbi.nlm.nih.gov/articles/PMC4990127",
    "hierarchy": "Felleman & Van Essen 1991, Cereb Cortex — "
                 "https://chd.ucsd.edu/_files/fall2008/Felleman.1991.CC_Part1.pdf",
    "k_computer": "RIKEN 2013 — https://www.riken.jp/en/pr/press/2013/20130802_1/",
    "llama": "Meta Llama 3.1 (config.json) — "
             "https://huggingface.co/neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8",
}

# Transformers de pesos abiertos (las arquitecturas de los modelos cerrados no son públicas).
LLM_REFERENCE = [
    dict(name="Llama 3.1 70B", layers=80, width=8192, parameters=70e9),
    dict(name="Llama 3.1 405B", layers=126, width=16384, parameters=405e9),
]

# Mayor simulación de red de spikes de propósito general publicada con NEST en 2013.
BRAIN_SIM_REFERENCE = dict(name="K computer + NEST (2013)", neurons=1.73e9, synapses=1.04e13,
                           processors=82944, seconds_per_biological_second=2400,
                           bytes_per_synapse=24)

# ---------------------------------------------------------------- memoria real (ElasticLayer)
BYTES_PER_SYNAPSE = 8        # sólo la sinapsis: aferente int32 + peso float32
INDEX_BYTES_PER_SYNAPSE = 12 # índice invertido: neurona int32 + posición int64 (si se usa)
BYTES_PER_NEURON = 45        # estadísticas: fill, wins, hits (int32), born, util_t, hoff_t (int64),
                             # util, hoff (float32), marca de fila cambiada (bool)
READOUT_BYTES_PER_NEURON = 4 # por salida (peso float32), más 4 del contador de consolidación
SCRATCH_BYTES_PER_AFFERENT = 8   # por columna: volea densa float32 + búsquedas int32 (tamaño D)
INDEX_BYTES_PER_AFFERENT = 8     # por columna con índice: puntero int64 por aferente

NOT_DEMONSTRATED = (
    "Que la red aprenda bien a esa escala: sólo se midió el costo de inferencia con pesos al "
    "azar; precisión, estabilidad del aprendizaje y del crecimiento no se midieron por encima "
    "de ~10^4 neuronas por área.",
    "El costo de comunicar spikes entre núcleos o máquinas: no está medido ni contado; el "
    "plano supone que cada columna tiene su entrada en memoria local.",
    "Que el costo por sinapsis medido con fan-in 32 valga con fan-in 7000 (otra forma de "
    "acceso a memoria).",
    "El costo del aprendizaje: STDP es O(fan-in) por neurona que aprende, pero el desalojo "
    "con presupuesto y la poda recorren todas las neuronas de la columna (O(n)).",
    "Que columnas de `column_size` neuronas con `afferents_per_column` aferentes sean una "
    "organización válida: es un supuesto del plano, no un resultado.",
    "Que la entrada real sea tan dispersa como para que el índice invertido evite el pase "
    "denso: con aferentes compartidas por muchas neuronas (imagen, texto) el costo sigue "
    "siendo proporcional a las neuronas tocadas, que crecen con la red.",
)


def bytes_per_neuron(fan_in: int, *, n_outputs: int = 0, index: bool = False,
                     slack: float = 1.0) -> float:
    """Memoria real de una neurona de `ElasticLayer` (más su columna en la capa de salida).

    slack   reserva por crecimiento: la capacidad se duplica al llenarse (1.0 a 2.0)."""
    syn = fan_in * (BYTES_PER_SYNAPSE + (INDEX_BYTES_PER_SYNAPSE if index else 0))
    out = (READOUT_BYTES_PER_NEURON * n_outputs + 4) if n_outputs else 0
    return slack * (syn + BYTES_PER_NEURON + out)


def plan(scale: float = 1.0, *, fan_in: int | None = None, column_size: int = 1000,
         bytes_per_synapse: int = BYTES_PER_SYNAPSE, syn_per_core_second: float | None = None,
         active_fraction: float = 1.0, afferents_per_column: int | None = None,
         index: bool = False, slack: float = 1.0, steps_per_second: float | None = None) -> dict:
    """Requisitos de una red con `scale` x la corteza humana (extrapolado).

    fan_in               sinapsis por neurona (por defecto, las 7000 biológicas)
    column_size          neuronas por columna elástica (unidad de reparto entre máquinas)
    syn_per_core_second  sinapsis evaluadas por segundo y por núcleo (medido)
    active_fraction      fracción de columnas que recibe una volea en un paso (pase denso:
                         todas sus sinapsis se evalúan)
    afferents_per_column D de cada columna (por defecto 10 x fan_in: supuesto)
    index                si cada columna mantiene el índice invertido
    slack                reserva por crecimiento de la capacidad (1.0 a 2.0)
    steps_per_second     pasos por segundo de tiempo simulado (para núcleos en tiempo real)

    `memory_bytes` es sólo la memoria de las sinapsis (8 B/sinapsis); `memory_total_bytes`
    suma estadísticas por neurona, índice, buffers por columna y reserva.
    """
    fan_in = int(fan_in or HUMAN["synapses_per_cortical_neuron"])
    neurons = HUMAN["neurons_cortex"] * scale
    synapses = neurons * fan_in
    columns = neurons / column_size
    D = int(afferents_per_column or 10 * fan_in)
    per_col = D * (SCRATCH_BYTES_PER_AFFERENT + (INDEX_BYTES_PER_AFFERENT if index else 0))
    out = dict(scale=scale, areas=HUMAN["cortical_areas"], laminae=HUMAN["cortical_laminae"],
               neurons=neurons, neurons_per_area=neurons / HUMAN["cortical_areas"],
               columns=columns, fan_in=fan_in, synapses=synapses,
               memory_bytes=synapses * bytes_per_synapse,
               neuron_overhead_bytes=neurons * slack * BYTES_PER_NEURON,
               index_bytes=synapses * slack * INDEX_BYTES_PER_SYNAPSE if index else 0.0,
               column_buffer_bytes=columns * per_col)
    out["memory_total_bytes"] = (slack * synapses * bytes_per_synapse + out["neuron_overhead_bytes"]
                                 + out["index_bytes"] + out["column_buffer_bytes"])
    if syn_per_core_second:
        out["ns_per_synapse"] = 1e9 / syn_per_core_second
        out["synapses_per_step"] = synapses * active_fraction
        out["core_seconds_per_step"] = out["synapses_per_step"] / syn_per_core_second
        if steps_per_second:
            out["cores_for_real_time"] = out["core_seconds_per_step"] * steps_per_second
    return out


def fmt(x: float, unit: str = "") -> str:
    for div, suf in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            return f"{x / div:.3g} {suf}{unit}"
    return f"{x:.3g} {unit}".strip()


def main():
    import csv
    from pathlib import Path
    measured = None
    f = Path(__file__).resolve().parent.parent / "results" / "scaling.csv"
    if f.exists():
        rows = list(csv.DictReader(f.open()))
        measured = float(rows[-1]["syn_per_core_second"])
    print(f"{'escala':>8} {'neuronas':>10} {'sinapsis':>10} {'mem. sinapsis':>14} {'mem. total':>11} "
          f"{'núcleo-s por paso':>18} {'núcleos (100 pasos/s)':>22}")
    for scale in (1e-6, 1e-4, 1e-2, 1.0):
        p = plan(scale, syn_per_core_second=measured, index=True, slack=1.5, steps_per_second=100)
        cs = fmt(p["core_seconds_per_step"]) if measured else "(falta medir)"
        rt = fmt(p["cores_for_real_time"]) if measured else ""
        print(f"{scale:>8g} {fmt(p['neurons']):>10} {fmt(p['synapses']):>10} "
              f"{fmt(p['memory_bytes'], 'B'):>14} {fmt(p['memory_total_bytes'], 'B'):>11} {cs:>18} {rt:>22}")
    print("\nMedido: costo por sinapsis en un núcleo. Extrapolado: todo lo demás.")
    print("Supuestos de la tabla: índice invertido, reserva 1.5x, D = 10 x fan-in por columna, "
          "todas las columnas activas en cada paso, 100 pasos por segundo simulado.")
    print("No demostrado:")
    for line in NOT_DEMONSTRATED:
        print(" -", line)


if __name__ == "__main__":
    main()
