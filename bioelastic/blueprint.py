"""Plano de escala: qué significa "capacidad humana" y qué cuesta llegar con esta arquitectura.

Los números del cerebro y de los modelos de referencia vienen de la literatura (ver
`SOURCES`). Las funciones convierten una escala (1.0 = corteza humana completa) en
neuronas, sinapsis, memoria y cómputo, usando el costo por sinapsis *medido* en esta
implementación (`experiments/exp_scaling.py`).

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

BYTES_PER_SYNAPSE = 8        # esta implementación: aferente int32 + peso float32


def plan(scale: float = 1.0, *, fan_in: int | None = None, column_size: int = 1000,
         bytes_per_synapse: int = BYTES_PER_SYNAPSE, syn_per_core_second: float | None = None,
         active_fraction: float = 1.0) -> dict:
    """Requisitos de una red con `scale` x la corteza humana.

    fan_in              sinapsis por neurona (por defecto, las 7000 biológicas)
    column_size         neuronas por columna elástica (unidad de reparto entre máquinas)
    syn_per_core_second sinapsis evaluadas por segundo y por núcleo (medido)
    active_fraction     fracción de columnas que recibe spikes en un paso
    """
    fan_in = int(fan_in or HUMAN["synapses_per_cortical_neuron"])
    neurons = HUMAN["neurons_cortex"] * scale
    synapses = neurons * fan_in
    out = dict(scale=scale, areas=HUMAN["cortical_areas"], laminae=HUMAN["cortical_laminae"],
               neurons=neurons, neurons_per_area=neurons / HUMAN["cortical_areas"],
               columns=neurons / column_size, fan_in=fan_in, synapses=synapses,
               memory_bytes=synapses * bytes_per_synapse)
    if syn_per_core_second:
        out["core_seconds_per_step"] = synapses * active_fraction / syn_per_core_second
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
    print(f"{'escala':>8} {'neuronas':>10} {'sinapsis':>10} {'memoria':>10} {'núcleo-s por paso':>18}")
    for scale in (1e-6, 1e-4, 1e-2, 1.0):
        p = plan(scale, syn_per_core_second=measured)
        cs = fmt(p["core_seconds_per_step"]) if measured else "(falta medir)"
        print(f"{scale:>8g} {fmt(p['neurons']):>10} {fmt(p['synapses']):>10} "
              f"{fmt(p['memory_bytes'], 'B'):>10} {cs:>18}")


if __name__ == "__main__":
    main()
