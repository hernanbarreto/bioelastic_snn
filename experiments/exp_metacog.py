"""Metacognición y neuromodulación en el Cortex solo, sobre el flujo continuo no estacionario.

Las mismas opciones que se prueban en L3 (exp_integration3.py, F2 y F3), en la percepción
sola, sobre el flujo de `bioelastic/stream.py` (MNIST + Fashion, clases que aparecen y
vuelven, fases sólo con recompensa o sin etiquetas, contextos que cambian) reducido con
`--scale` (0.05: 10 000 pasos) para la CPU disponible. Red: la `d2` de exp_continual.py
(V1 + memoria sobre los píxeles + proyección + repaso), dos votantes.

Variantes (ablación):
  base          sin nada (las señales se leen igual: confianza cruda, conflicto)
  cal           calibrador online de la confianza (`metacog`); no cambia el aprendizaje
  hiper         + hipercorrección (tasa x (1 + g * confianza) en los errores)
  hiper_plano   control: la misma ganancia media en los errores, sin ponderar por confianza
  ne            + neuromodulación: NE (sorpresa inesperada) sube la plasticidad, libera
                consolidación y habilita crecimiento ante errores
  ne_ach        + ACh: la incertidumbre esperada baja la tasa (no perseguir el ruido)

Medidas (prequential: cada señal se lee antes de saber si acertó). El foco: separar
«me equivoqué en algo conocido» (incertidumbre esperada) de «esto es nuevo o distinto»
(novedad, cambio). AUROC por caso: (1) error entre entradas de (clase, contexto) ya vistas
>= 100 veces; (2) entrada nueva (< 10 vistas) vs conocida; (3) entre los errores, nuevo vs
conocido. Señales: 1 - confianza (cruda/calibrada), conflicto, -margen, 1 - familiaridad
(`Cortex.familiarity`) y NE.
  AUROC         detección del propio error antes del feedback con 1 - confianza
                (cruda / calibrada), conflicto entre votantes y margen
  ECE           error de calibración (10 bins) de la confianza cruda y calibrada, y la
                curva de confiabilidad (en el JSON)
  recuperación  acierto en los primeros W pasos de cada fase supervisada y pasos hasta
                volver a `nivel` (cambios de distribución: clases nuevas, contexto nuevo,
                clases que vuelven)
  olvido        acierto final en test por grupo (lo que quedó de lo anterior)

  python experiments/exp_metacog.py --seed S [--variante V] [--scale 0.05]
  python experiments/exp_metacog.py --resumen
Escribe results/integracion3_metacog_<variante>_seed<S>.csv (formato largo) y .json (curvas).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import RESULTS, CsvLog
from bioelastic import Cortex
from bioelastic.metacog import auroc, ece
from bioelastic.stream import GROUPS, KNOWN, SCHEDULE, ContinualStream, steps_to, window_acc
from exp_continual import D2, load_data
from exp_vision import encoder

HEADER = ["prueba", "variante", "semilla", "metrica", "valor", "referencia", "nota"]
N_NUEVO, N_CONOCIDO = 10, 100     # «nuevo»: < 10 vistas de (clase, contexto); «conocido»: >= 100
MC = dict(lr=0.05)
VARIANTES = {
    "base": {},
    "cal": dict(metacog=MC),
    "hiper": dict(metacog=MC, hypercorrection=1.0),
    "hiper_plano": dict(metacog=MC, hypercorrection=1.0, hypercorrection_mode="plano"),
    "ne": dict(metacog=MC, neuromod=dict(ne_gain=1.0, ach_gain=0.0)),
    "ne_ach": dict(metacog=MC, neuromod=dict(ne_gain=1.0, ach_gain=1.0)),
}


# ====================================================================== corrida
def correr(var: str, seed: int, scale: float, n_eval: int, data, out, extra):
    pools, test = data
    enc = encoder()
    stream = ContinualStream(pools, scale=scale, seed=seed)
    bounds = stream.bounds(); T = len(stream)
    net = Cortex((28, 28, 1), 20, seed=seed, **D2, **VARIANTES[var])
    rec = {k: np.zeros(T, dtype=np.float32) for k in ("ok", "conf", "raw", "conflict", "margin", "ne", "ach", "gain",
                                                      "fam")}
    mode = np.zeros(T, dtype=np.int8)                      # 0 sup, 1 reward, 2 unlab
    vistas = np.zeros(T, dtype=np.int32)                   # presentaciones previas de (clase, contexto)
    cuenta: dict = {}
    t0 = time.time()
    for ev in stream:
        s = enc(ev.x)
        if ev.mode == "sup":
            a = net.forward(s)
        elif ev.mode == "reward":
            a = net.forward(s, explore=True)
        else:
            a = net.forward(s)
        sg = net.signals()                                 # antes de saber si acertó
        if ev.mode == "sup":
            net.teach(ev.y)
        elif ev.mode == "reward":
            net.reinforce(1.0 if a == ev.y else -1.0)
        else:
            net.observe(s)
        t = ev.t
        rec["ok"][t] = a == ev.y; rec["conf"][t] = sg["confidence"]; rec["raw"][t] = sg["raw_confidence"]
        rec["conflict"][t] = sg["conflict"]; rec["margin"][t] = sg["margin"]
        rec["ne"][t] = net.unexpected_uncertainty; rec["ach"][t] = net.expected_uncertainty
        rec["gain"][t] = net.lr_gain; rec["fam"][t] = sg["familiarity"]
        mode[t] = ("sup", "reward", "unlab").index(ev.mode)
        k = (ev.y, ev.ctx); vistas[t] = cuenta.get(k, 0); cuenta[k] = vistas[t] + 1
    out("metacog", var, "segundos", round(time.time() - t0, 1), "", f"{T} pasos (scale {scale})")
    ok = rec["ok"].astype(bool)
    for nombre, m in (("sup", mode == 0), ("recompensa", mode == 1)):
        if not m.any():
            continue
        out("metacog", var, f"acierto prequential {nombre}", float(ok[m].mean()), "", "")
        for sen, sc in (("1-confianza cruda", 1 - rec["raw"]), ("1-confianza calibrada", 1 - rec["conf"]),
                        ("conflicto", rec["conflict"]), ("-margen", -rec["margin"])):
            out("metacog", var, f"AUROC error {nombre}: {sen}", auroc(sc[m], ~ok[m]), 0.5, "azar")
    # F2: ¿la señal separa «me equivoqué en algo conocido» de «esto es nuevo o distinto»?
    nuevo = vistas < N_NUEVO                               # primeras presentaciones de (clase, contexto)
    conocido = vistas >= N_CONOCIDO
    con_fb = mode < 2
    senales = {"1-confianza cruda": 1 - rec["raw"], "1-confianza calibrada": 1 - rec["conf"],
               "conflicto": rec["conflict"], "-margen": -rec["margin"], "1-familiaridad": 1 - rec["fam"],
               "NE": rec["ne"]}
    m1 = con_fb & conocido
    m2 = con_fb & (nuevo | conocido)
    m3 = con_fb & ~ok & (nuevo | conocido)
    out("metacog", var, "error en lo conocido (tasa)", float(1 - ok[m1].mean()), "", f">= {N_CONOCIDO} vistas")
    out("metacog", var, "pasos nuevos", int((con_fb & nuevo).sum()), "", f"< {N_NUEVO} vistas de (clase, contexto)")
    for sen, sc in senales.items():
        out("metacog", var, f"AUROC (1) error en lo conocido: {sen}", auroc(sc[m1], ~ok[m1]), 0.5, "azar")
        out("metacog", var, f"AUROC (2) nuevo vs conocido: {sen}", auroc(sc[m2], nuevo[m2]), 0.5, "azar")
        out("metacog", var, f"AUROC (3) entre errores, nuevo vs conocido: {sen}", auroc(sc[m3], nuevo[m3]), 0.5, "azar")
    e1, _ = ece(rec["conf"][m1], ok[m1])
    out("metacog", var, "ECE calibrada en lo conocido", e1, 0.0, "")
    ms = mode == 0
    half = np.arange(T) >= T // 2
    for nombre, c in (("cruda", rec["raw"]), ("calibrada", rec["conf"])):
        e, curva = ece(c[ms], ok[ms])
        out("metacog", var, f"ECE {nombre} (sup)", e, 0.0, "")
        e2_, _ = ece(c[ms & half], ok[ms & half])
        out("metacog", var, f"ECE {nombre} (sup, 2.a mitad)", e2_, 0.0, "")
        extra[f"confiabilidad_{nombre}"] = curva
    # recuperación tras cada cambio de distribución (fases supervisadas salvo la primera)
    W = max(100, int(round(4000 * scale)))
    rec_w, pasos = [], []
    for k, (a0, a1) in enumerate(bounds):
        ph = SCHEDULE[k]
        if k == 0 or ph.mode != "sup":
            continue
        w = window_acc(ok, a0, W)
        n = steps_to(ok, a0, a1, 0.7, 50)
        out("metacog", var, f"recuperación {ph.name}: acierto primeros {W}", w, "", "")
        out("metacog", var, f"recuperación {ph.name}: pasos hasta 0.7", n, a1 - a0, "largo de la fase")
        rec_w.append(w); pasos.append(n)
    out("metacog", var, f"recuperación media: acierto primeros {W}", float(np.mean(rec_w)), "", "fases sup salvo la 1.a")
    out("metacog", var, "recuperación media: pasos hasta 0.7", float(np.mean(pasos)), "", "")
    out("metacog", var, "NE medio / máximo", float(rec["ne"].mean()), float(rec["ne"].max()), "máximo en referencia")
    out("metacog", var, "liberaciones de consolidación", net.releases, "", "")
    out("metacog", var, "ganancia media de la tasa", float(rec["gain"][mode < 2].mean()), 1.0, "")
    # NE alrededor de los cambios: media en los primeros W pasos de cada fase vs el resto
    ini = np.zeros(T, dtype=bool)
    for a0, _ in bounds[1:]:
        ini[a0:a0 + W] = True
    if rec["ne"].max() > 0:
        out("metacog", var, "NE al inicio de fase / resto", float(rec["ne"][ini].mean()), float(rec["ne"][~ini].mean()),
            "resto en referencia")
    # lo que quedó al final (test): por grupo, contexto limpio
    rng = np.random.default_rng([seed, 31337])
    for g in ("A", "B", "C", "D"):
        hits = []
        for c in GROUPS[g]:
            for i in rng.choice(len(test[c]), n_eval, replace=False):
                hits.append(net.forward(enc(test[c][i].astype(np.float32) / 255.0)) == c)
        out("metacog", var, f"final test grupo {g}", float(np.mean(hits)), "", "")
    extra["ne_curva"] = [float(x) for x in rec["ne"][::max(1, T // 500)]]
    extra["acierto_curva"] = [float(ok[i:i + max(1, T // 200)].mean()) for i in range(0, T, max(1, T // 200))]
    extra["bounds"] = bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variante", default=None)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--n-eval", type=int, default=20)
    a = ap.parse_args()
    data = load_data()
    for var in ([a.variante] if a.variante else list(VARIANTES)):
        nombre = f"integracion3_metacog_{var}_seed{a.seed}"
        log = CsvLog(nombre, HEADER)
        extra: dict = {}
        correr(var, a.seed, a.scale, a.n_eval, data, lambda *r: log.row(r[0], r[1], a.seed, *r[2:]), extra)
        log.close()
        (RESULTS / f"{nombre}.json").write_text(json.dumps(extra))


if __name__ == "__main__":
    main()
