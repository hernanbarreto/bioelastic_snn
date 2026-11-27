"""Lenguaje: BLiMP (pares mínimos gramaticales) después de entrenar con un corpus chico.

Protocolo publicado (Warstadt et al. 2020): 67 paradigmas de 1 000 pares mínimos; un modelo
acierta un par si da más probabilidad a la oración aceptable (oración completa). Personas:
20 validadores juzgan 5 pares de cada paradigma en elección forzada; acuerdo individual
88,6 %.

Lo que hace la batería:
  * corpus: habla dirigida a niños (AO-CHILDES, el que usa BabyBERTa; en GitHub). El corpus
    de BabyLM (10 M de palabras) no tiene espejo en GitHub. La red aprende en línea, una
    pasada, la predicción del próximo byte sobre los primeros `N_BYTES` bytes, con la
    configuración publicada de exp_text (8 bytes de retardos);
  * BLiMP: 50 pares por paradigma (los mismos para todas las semillas, sorteados con semilla
    0), normalizados al formato del corpus (minúsculas, puntuación separada por un espacio);
    cada oración empieza tras un salto de línea y termina con uno. Log-probabilidad con la
    distribución de la decisión de la red (`readout.probs` sobre los potenciales del último
    paso: lectura interna, sin aprender). El prefijo común de las dos oraciones aporta lo
    mismo a las dos y no se calcula;
  * control con los mismos datos: un modelo de 5-gramas de bytes con interpolación de
    Witten-Bell entrenado con los mismos bytes (el criterio de abandono F del README).

Entrenamiento por tramos de `TRAMO` bytes con guardado y recarga (`BioElasticNet.save/load`)
para que una corrida cortada se retome: es determinista con tramos fijos, aunque no idéntico
bit a bit a una pasada sin cortes (al recargar, el contexto de 8 bytes empieza vacío).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .base import NoAplicable, Prueba, Pendiente, vencido

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "ext_data" / "corpus" / "babyberta" / "aochildes.txt"
CORPUS_SHA1 = "277c8fd2da6417da66a490321369718cc5476a22"
BLIMP = ROOT / "ext_data" / "blimp_git" / "repo" / "data"
CACHE = ROOT / "ext_data" / "cache"
CFG = dict(context=8, delay=0.1, layer=dict(max_neurons=30000, fan_in=16))     # exp_text
N_BYTES = 600_000
TRAMO = 100_000
PARES = 50
ORDEN_NGRAMA = 5


# ---------------------------------------------------------------- datos
def corpus(n: int) -> bytes:
    if not CORPUS.exists():
        raise NoAplicable(f"falta el corpus {CORPUS} (BabyBERTa, data/corpora/aochildes.txt)")
    raw = CORPUS.read_bytes()
    if hashlib.sha1(raw).hexdigest() != CORPUS_SHA1:
        raise NoAplicable("el corpus no coincide con el archivo verificado (sha1)")
    return raw[:n]


def normalizar(s: str) -> bytes:
    s = re.sub(r"([.,?!;:])", r" \1", s.lower())
    return re.sub(r"\s+", " ", s).strip().encode("utf-8")


def pares_blimp(n: int = PARES) -> list[dict]:
    if not BLIMP.exists():
        raise NoAplicable(f"faltan los datos de BLiMP ({BLIMP})")
    out = []
    rng = np.random.default_rng(0)
    for p in sorted(BLIMP.glob("*.jsonl")):
        filas = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i in sorted(rng.choice(len(filas), size=min(n, len(filas)), replace=False)):
            r = filas[int(i)]
            out.append(dict(uid=r["UID"], campo=r["field"], termino=r["linguistics_term"],
                            buena=normalizar(r["sentence_good"]), mala=normalizar(r["sentence_bad"])))
    return out


# ---------------------------------------------------------------- 5-gramas de bytes (control)
class NgramaWB:
    """n-gramas de bytes con interpolación de Witten-Bell hasta la uniforme sobre 256."""

    def __init__(self, orden: int = ORDEN_NGRAMA):
        self.k = orden - 1
        self.c = [defaultdict(Counter) for _ in range(self.k + 1)]

    def entrenar(self, b: bytes):
        for i in range(len(b)):
            for o in range(self.k + 1):
                if i - o < 0:
                    break
                self.c[o][b[i - o:i]][b[i]] += 1
        self._tot = [{h: sum(v.values()) for h, v in t.items()} for t in self.c]

    def logp(self, ctx: bytes, y: int) -> float:
        p = 1.0 / 256.0
        for o in range(0, self.k + 1):
            if o > len(ctx):
                break
            h = ctx[len(ctx) - o:] if o else b""
            cnt = self.c[o].get(h)
            if not cnt:
                continue
            n = self._tot[o][h]; t = len(cnt)
            p = (cnt.get(y, 0) + t * p) / (n + t)
        return float(np.log(p))

    def oracion(self, s: bytes) -> float:
        b = b"\n" + s + b"\n"
        return sum(self.logp(b[max(0, i - self.k):i], b[i]) for i in range(1, len(b)))


# ---------------------------------------------------------------- red
def _huella() -> str:
    import bioelastic
    h = hashlib.sha1()
    for p in sorted(Path(bioelastic.__file__).resolve().parent.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:10]


def entrenar(semilla: int, n: int, log=None):
    """La red de texto aprende en línea los primeros `n` bytes del corpus, por tramos con
    guardado. Devuelve (red, info del entrenamiento)."""
    from bioelastic import BioElasticNet, TextEncoder
    base = CACHE / f"texto_{_huella()}_s{semilla}_n{n}"
    data = TextEncoder()(corpus(n))
    patron = re.compile(re.escape(base.name) + r"_(\d+)\.npz$")
    hechos = sorted(int(m.group(1)) for p in (CACHE.glob(base.name + "_*.npz") if CACHE.exists() else [])
                    if (m := patron.match(p.name)))
    hecho = max(hechos) if hechos else 0
    if hecho:
        net = BioElasticNet.load(base.parent / f"{base.name}_{hecho}.npz")
        info = json.loads((base.parent / f"{base.name}_{hecho}.json").read_text())
    else:
        net = BioElasticNet(256, 256, seed=semilla, **CFG)
        info = dict(aciertos=0, segundos=0.0)
    rate = 3e-3                                                # s/byte supuesto hasta medir el primer tramo
    while hecho < n - 1:
        ini, fin = hecho, min(n - 1, hecho + TRAMO)
        if vencido(margen=rate * (fin - ini) * 1.2):
            raise Pendiente(f"entrenamiento de texto en {ini}/{n} bytes")
        t0 = time.time(); ok = 0
        for t in range(ini, fin):
            ok += int(net.forward(int(data[t])) == data[t + 1]); net.teach(int(data[t + 1]))
        dt = time.time() - t0
        rate = dt / max(1, fin - ini)
        info = dict(aciertos=info["aciertos"] + ok, segundos=info["segundos"] + dt)
        CACHE.mkdir(parents=True, exist_ok=True)
        (base.parent / f"{base.name}_{fin}.json").write_text(json.dumps(info))
        tmp = base.parent / f"{base.name}_{fin}.tmp.npz"
        net.save(tmp); tmp.replace(base.parent / f"{base.name}_{fin}.npz")
        for h in hechos:                                       # sólo se conserva el último tramo
            (base.parent / f"{base.name}_{h}.npz").unlink(missing_ok=True)
            (base.parent / f"{base.name}_{h}.json").unlink(missing_ok=True)
        hechos = [fin]; hecho = fin
        if log:
            log(f"    texto s{semilla}: {fin}/{n} bytes ({dt:.0f} s, próximo byte {ok / max(1, fin - ini):.3f})")
        if fin < n - 1:                                        # al recargar, el contexto empieza vacío
            net = BioElasticNet.load(base.parent / f"{base.name}_{fin}.npz")
    return net, dict(info, exactitud_en_linea=info["aciertos"] / max(1, n - 1))


def _logp_red(net, ctx: bytes, s: bytes) -> float:
    """Log-probabilidad de los bytes de `s` después de `ctx` (sólo se suman los de `s`)."""
    net.reset_context()
    lp = 0.0
    b = ctx + s
    for i in range(len(b) - 1):
        net.forward(int(b[i]))
        if i + 1 >= len(ctx):
            p = net.readout.probs(net._last.V)
            lp += float(np.log(max(float(p[b[i + 1]]), 1e-12)))
    return lp


def par_red(net, buena: bytes, mala: bytes) -> tuple[float, float]:
    """Log-probabilidades de las dos oraciones desde el primer byte distinto (el prefijo
    común suma lo mismo a las dos); la red tiene 8 bytes de contexto, que se vuelven a dar."""
    a = b"\n" + buena + b"\n"; m = b"\n" + mala + b"\n"
    c = 0
    while c < min(len(a), len(m)) and a[c] == m[c]:
        c += 1
    ini = max(0, c - CFG["context"])
    return _logp_red(net, a[ini:c], a[c:]), _logp_red(net, m[ini:c], m[c:])


def correr(semilla: int, escala: float = 1.0) -> dict:
    t0 = time.time()
    n = N_BYTES if escala >= 1.0 else max(5000, int(N_BYTES * escala))
    pares = pares_blimp(PARES if escala >= 1.0 else 2)
    net, info = entrenar(semilla, n, log=_log)
    cache = CACHE / f"blimp_{_huella()}_s{semilla}_n{n}_p{len(pares)}.json"
    hechos = json.loads(cache.read_text()) if cache.exists() else {}
    ng = NgramaWB(); ng.entrenar(corpus(n))
    por_uid = defaultdict(list)
    for i, pr in enumerate(pares):
        por_uid[pr["uid"]].append(i)
    t_par = None
    for uid, idx in por_uid.items():
        if uid in hechos:
            continue
        if t_par is not None and vencido(margen=t_par * len(idx) * 1.5):
            cache.write_text(json.dumps(hechos))
            raise Pendiente(f"BLiMP: {len(hechos)}/{len(por_uid)} paradigmas")
        t1 = time.time()
        res = []
        for i in idx:
            g, b = par_red(net, pares[i]["buena"], pares[i]["mala"])
            res.append([bool(g > b), bool(ng.oracion(pares[i]["buena"]) > ng.oracion(pares[i]["mala"]))])
        hechos[uid] = res
        t_par = (time.time() - t1) / max(1, len(idx))
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(hechos))
    red = np.array([r[0] for uid in por_uid for r in hechos[uid]])
    ngr = np.array([r[1] for uid in por_uid for r in hechos[uid]])
    por_paradigma_red = {uid: float(np.mean([r[0] for r in hechos[uid]])) for uid in por_uid}
    por_paradigma_ng = {uid: float(np.mean([r[1] for r in hechos[uid]])) for uid in por_uid}
    campo = {pr["uid"]: pr["campo"] for pr in pares}; term = {pr["uid"]: pr["termino"] for pr in pares}
    med = dict(exactitud=float(np.mean(list(por_paradigma_red.values()))),
               exactitud_5gramas_mismos_datos=float(np.mean(list(por_paradigma_ng.values()))),
               exactitud_por_pares=float(red.mean()),
               supera_5gramas=bool(np.mean(list(por_paradigma_red.values())) > np.mean(list(por_paradigma_ng.values()))),
               bytes_entrenamiento=float(n), exactitud_en_linea=info.get("exactitud_en_linea"),
               pares=float(len(red)))
    for c in sorted(set(campo.values())):
        us = [u for u in por_uid if campo[u] == c]
        med[f"campo_{c}"] = float(np.mean([por_paradigma_red[u] for u in us]))
        med[f"campo_{c}_5gramas"] = float(np.mean([por_paradigma_ng[u] for u in us]))
    for c in sorted(set(term.values())):
        us = [u for u in por_uid if term[u] == c]
        med[f"categoria_{c}"] = float(np.mean([por_paradigma_red[u] for u in us]))
    return dict(puntaje=med["exactitud"], medidas=med, segundos=time.time() - t0)


def _log(m):
    import sys
    print(m, file=sys.stderr, flush=True)


PRUEBAS = [
    Prueba(
        id="blimp", variante="texto", condicion="texto",
        metrica=f"proporción de pares con más probabilidad para la oración aceptable (media de los 67 paradigmas, "
                f"{PARES} pares cada uno)",
        protocolo=(f"red de texto de exp_text (bytes, 8 de contexto) entrenada en línea con {N_BYTES} bytes de AO-CHILDES; "
                   f"{PARES} pares por paradigma, oración completa; control: 5-gramas de bytes (Witten-Bell) con los "
                   "mismos bytes"),
        desviaciones=[
            f"corpus: {N_BYTES} bytes (unas 140 000 palabras) de habla dirigida a niños, no los 10 M de palabras de "
            "BabyLM (sin espejo en GitHub) ni los corpus de los modelos de la tabla 3 (de 83 M a 8 000 M)",
            "las oraciones de BLiMP se pasan a minúsculas y con la puntuación separada, el formato del corpus",
            f"{PARES} pares por paradigma (las personas: 5 pares por paradigma y 20 validadores)",
            "la probabilidad se lee de la decisión de la red (`readout.probs`), que no fue hecha para puntuar oraciones",
            "las personas juzgan aceptabilidad con toda su competencia; la red sólo predice bytes",
        ],
        correr=correr, version=1, semillas=(0, 1, 2),
        equivalente=True,
        motivo_equivalencia="la misma prueba de pares mínimos sobre texto; cambia muchísimo la cantidad de datos",
        comparar={"5gramas": "puntaje", "lstm": "puntaje", "txl": "puntaje", "gpt2": "puntaje",
                  "babylm_opt": "puntaje", "babylm_roberta": "puntaje", "babylm_t5": "puntaje"},
        firmas={"supera_5gramas": "supera a 5-gramas de bytes entrenados con los mismos datos (criterio F del README)"},
    ),
]
