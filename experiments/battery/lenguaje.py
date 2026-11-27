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


class KneserNey:
    """n-gramas con interpolación de Kneser-Ney (Chen y Goodman 1999): descuento por orden
    D = n1 / (n1 + 2 n2), conteos de continuación en los órdenes bajos y la uniforme sobre
    `V` símbolos al fondo. Las secuencias son `bytes` (símbolos = bytes) o tuplas de enteros
    (palabras). Con `orden_max` se cuenta una vez y se evalúa cualquier orden <= orden_max."""

    def __init__(self, orden_max: int, V: int):
        self.K, self.V = int(orden_max), int(V)

    @staticmethod
    def _cat(h, w):
        return h + bytes([w]) if isinstance(h, bytes) else h + (w,)

    def entrenar(self, secuencias):
        K = self.K
        self.c = [None] + [defaultdict(int) for _ in range(K)]
        for seq in secuencias:
            for i in range(len(seq)):
                for k in range(1, K + 1):
                    if i - k + 1 < 0:
                        break
                    self.c[k][seq[i - k + 1:i + 1]] += 1
        self.cont = [None] + [defaultdict(int) for _ in range(K)]
        for k in range(2, K + 1):
            for g in self.c[k]:
                self.cont[k - 1][g[1:]] += 1
        self.orden(K)
        return self

    def orden(self, K: int):
        """Usa el modelo de orden K (conteos crudos en K, de continuación debajo)."""
        self.k = int(K)
        self.ef = [None] + [self.cont[k] if k < self.k else self.c[k] for k in range(1, self.k + 1)]
        self.tot = [None]; self.tipos = [None]; self.D = [None]
        for k in range(1, self.k + 1):
            tot, tipos = defaultdict(int), defaultdict(int)
            n1 = n2 = 0
            for g, v in self.ef[k].items():
                tot[g[:-1]] += v; tipos[g[:-1]] += 1
                n1 += v == 1; n2 += v == 2
            self.tot.append(tot); self.tipos.append(tipos)
            self.D.append(n1 / (n1 + 2 * n2) if n1 + n2 else 0.5)
        return self

    def logp(self, hist, w) -> float:
        p = 1.0 / self.V
        for k in range(1, self.k + 1):
            if k - 1 > len(hist):
                break
            h = hist[len(hist) - (k - 1):] if k > 1 else hist[:0]
            tot = self.tot[k].get(h)
            if not tot:
                continue
            c = self.ef[k].get(self._cat(h, w), 0)
            D = self.D[k]
            p = max(c - D, 0.0) / tot + D * self.tipos[k][h] / tot * p
        return float(np.log(p))

    def secuencia(self, seq, desde: int = 1) -> float:
        """Suma de log-probabilidades de seq[desde:], cada símbolo dado los k-1 anteriores."""
        return sum(self.logp(seq[max(0, i - self.k + 1):i], seq[i]) for i in range(desde, len(seq)))


N_VALIDACION = 60_000            # bytes del corpus después de los de entrenamiento: elegir el orden
ORDEN_MAX_BYTES, ORDEN_MAX_PALABRAS = 10, 6


def _palabras(raw: bytes, vocab=None):
    lineas = [l.split() for l in raw.decode("utf-8", errors="replace").split("\n") if l.strip()]
    if vocab is None:
        cnt = Counter(w for l in lineas for w in l)
        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        for w, c in sorted(cnt.items()):
            if c >= 2:
                vocab[w] = len(vocab)
    return lineas, vocab


def _ids(palabras, vocab, K):
    return (1,) * (K - 1) + tuple(vocab.get(w, 0) for w in palabras) + (2,)


def lineas_base(n: int, pares: list) -> dict:
    """n-gramas con los mismos `n` bytes del corpus, de bytes y de palabras (Kneser-Ney), con el
    orden elegido por la entropía cruzada en los `N_VALIDACION` bytes siguientes (que la red
    tampoco vio); y el de Witten-Bell de bytes de orden 5. Devuelve, por modelo, el orden, la
    validación y el resultado por par (1 acierta, 0 falla, 0,5 empate) en el orden de `pares`."""
    clave = hashlib.sha1(json.dumps([CORPUS_SHA1, n, N_VALIDACION, [p["buena"].hex() for p in pares[:50]],
                                     len(pares), ORDEN_MAX_BYTES, ORDEN_MAX_PALABRAS, 3]).encode()).hexdigest()[:10]
    ruta = CACHE / f"blimp_lineas_base_n{n}_{clave}.json"
    if ruta.exists():
        return json.loads(ruta.read_text())
    todo = CORPUS.read_bytes()
    raw, val = todo[:n], todo[n:n + N_VALIDACION]
    out = {}

    def resultado(g, b):
        return 1.0 if g > b else (0.5 if g == b else 0.0)
    # bytes
    kb = KneserNey(ORDEN_MAX_BYTES, 256).entrenar([raw])
    hval = {}
    for K in range(1, ORDEN_MAX_BYTES + 1):
        kb.orden(K)
        hval[K] = -kb.secuencia(val) / max(1, len(val) - 1) / np.log(2)
    Kb = min(hval, key=hval.get)
    kb.orden(Kb)
    out["kn_bytes"] = dict(orden=Kb, validacion_bits_por_byte={str(k): v for k, v in hval.items()},
                           por_par=[resultado(kb.secuencia(b"\n" + p["buena"] + b"\n"),
                                              kb.secuencia(b"\n" + p["mala"] + b"\n")) for p in pares])
    # palabras
    lin, vocab = _palabras(raw)
    lval, _ = _palabras(val, vocab)
    hval = {}
    modelos = {}
    for K in range(1, ORDEN_MAX_PALABRAS + 1):
        km = KneserNey(K, len(vocab)).entrenar([_ids(l, vocab, K) for l in lin])
        modelos[K] = km
        tot = sum(km.secuencia(_ids(l, vocab, K), desde=K - 1 if K > 1 else 0) for l in lval)
        nw = sum(len(l) + 1 for l in lval)
        hval[K] = -tot / nw / np.log(2)
    Kw = min(hval, key=hval.get)
    km = modelos[Kw]

    def lp_pal(sent: bytes):
        return km.secuencia(_ids(sent.decode().split(), vocab, Kw), desde=Kw - 1 if Kw > 1 else 0)
    oov = [np.mean([w not in vocab for w in p["buena"].decode().split()]) for p in pares]
    out["kn_palabras"] = dict(orden=Kw, validacion_bits_por_palabra={str(k): v for k, v in hval.items()},
                              vocabulario=len(vocab), palabras_fuera_del_vocabulario=float(np.mean(oov)),
                              por_par=[resultado(lp_pal(p["buena"]), lp_pal(p["mala"])) for p in pares])
    wb = NgramaWB(); wb.entrenar(raw)
    out["wb5_bytes"] = dict(orden=5, por_par=[resultado(wb.oracion(p["buena"]), wb.oracion(p["mala"])) for p in pares])
    CACHE.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(out))
    return out


def _comparar_pares(red: np.ndarray, ng: np.ndarray) -> dict:
    """Por par mínimo: en cuántos acierta sólo la red y en cuántos sólo el n-grama (empates
    del n-grama como medio acierto), con la prueba del signo (binomial bilateral)."""
    from scipy.stats import binomtest
    solo_red = int(np.sum((red == 1) & (ng == 0))); solo_ng = int(np.sum((red == 0) & (ng == 1)))
    p = binomtest(solo_red, solo_red + solo_ng, 0.5).pvalue if solo_red + solo_ng else 1.0
    return dict(solo_red=solo_red, solo_ngrama=solo_ng, p_signo=float(p))


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


def par_red(net, buena: bytes, mala: bytes) -> tuple[float, float, int, int]:
    """Log-probabilidades de las dos oraciones desde el primer byte distinto (el prefijo
    común suma lo mismo a las dos); la red tiene 8 bytes de contexto, que se vuelven a dar.
    Devuelve también cuántos bytes se sumaron de cada una."""
    a = b"\n" + buena + b"\n"; m = b"\n" + mala + b"\n"
    c = 0
    while c < min(len(a), len(m)) and a[c] == m[c]:
        c += 1
    ini = max(0, c - CFG["context"])
    return _logp_red(net, a[ini:c], a[c:]), _logp_red(net, m[ini:c], m[c:]), len(a) - c, len(m) - c


FACTORES_T = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)       # múltiplos de la temperatura de la decisión


def bits_red(net, datos: bytes) -> tuple[float, float, dict]:
    """Entropía cruzada (bits por byte) y acierto del próximo byte de la red, sin aprender,
    sobre bytes que no vio. Además, los bits con la temperatura de la decisión multiplicada
    por cada factor de `FACTORES_T` (para elegir uno en validación: la red no ajustó su
    temperatura para la verosimilitud; sólo afecta a la lectura, no a la red)."""
    net.reset_context()
    lp, ok = 0.0, 0
    lpf = np.zeros(len(FACTORES_T))
    T = float(net.readout.temperature)
    for i in range(len(datos) - 1):
        pred = net.forward(int(datos[i]))
        V = net._last.V
        z = (V - V.max()) / T
        y = datos[i + 1]
        for j, f in enumerate(FACTORES_T):
            zf = z / f
            lpf[j] += (zf[y] - np.log(np.exp(zf).sum())) / np.log(2)
        ok += int(pred == y)
    m = max(1, len(datos) - 1)
    por_t = {str(f): float(-lpf[j] / m) for j, f in enumerate(FACTORES_T)}
    return por_t["1.0"], ok / m, por_t


def bits_kn(tramos: dict, n: int, orden: int) -> dict:
    """Lo mismo para n-gramas de bytes de Kneser-Ney entrenados con los primeros `n` bytes: bits
    por byte en cada tramo y acierto del próximo byte (en una de cada 20 posiciones)."""
    kb = KneserNey(orden, 256).entrenar([CORPUS.read_bytes()[:n]])
    out = {}
    for nombre, datos in tramos.items():
        lp, ok, m = 0.0, 0, 0
        for i in range(1, len(datos)):
            h = datos[max(0, i - orden + 1):i]
            lp += kb.logp(h, datos[i]) / np.log(2)
            if i % 20 == 0:
                ok += int(max(range(256), key=lambda y: kb.logp(h, y)) == datos[i]); m += 1
        out[nombre] = [-lp / max(1, len(datos) - 1), ok / max(1, m)]
    return out


def _signo(k: int, n: int) -> float:
    from scipy.stats import binomtest
    return float(binomtest(k, n, 0.5).pvalue) if n else 1.0


def correr(semilla: int, escala: float = 1.0) -> dict:
    t0 = time.time()
    n = N_BYTES if escala >= 1.0 else max(5000, int(N_BYTES * escala))
    pares = pares_blimp(PARES if escala >= 1.0 else 2)
    net, info = entrenar(semilla, n, log=_log)
    cache = CACHE / f"blimp2_{_huella()}_s{semilla}_n{n}_p{len(pares)}.json"
    hechos = json.loads(cache.read_text()) if cache.exists() else {}
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
        hechos[uid] = [list(par_red(net, pares[i]["buena"], pares[i]["mala"])) for i in idx]
        t_par = (time.time() - t1) / max(1, len(idx))
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(hechos))
    # bits por byte en lo no visto: los 60 000 bytes de validación de los n-gramas y otros
    # 60 000 después (prueba limpia: el orden de los n-gramas no se eligió con ellos)
    cb = CACHE / f"bits2_{_huella()}_s{semilla}_n{n}.json"
    bits = json.loads(cb.read_text()) if cb.exists() else {}
    todo = CORPUS.read_bytes()
    tramos = dict(validacion=todo[n:n + N_VALIDACION], prueba=todo[n + N_VALIDACION:n + 2 * N_VALIDACION])
    if escala < 1.0:
        tramos = {k: v[:3000] for k, v in tramos.items()}
    for nombre, datos in tramos.items():
        if nombre not in bits:
            if vencido(margen=len(datos) * 4e-3):
                raise Pendiente(f"bits por byte de la red ({nombre})")
            bits[nombre] = list(bits_red(net, datos))
            cb.write_text(json.dumps(bits))
    base = lineas_base(n, pares)
    ckn = CACHE / f"bits_kn_n{n}_o{base['kn_bytes']['orden']}_{len(tramos['prueba'])}.json"
    if ckn.exists():
        bkn = json.loads(ckn.read_text())
    else:
        bkn = bits_kn(tramos, n, int(base["kn_bytes"]["orden"]))
        ckn.write_text(json.dumps(bkn))
    filas = [r for uid in por_uid for r in hechos[uid]]
    red = np.array([1.0 if g > b else (0.5 if g == b else 0.0) for g, b, _, _ in filas])
    lg = np.array([la for _, _, la, _ in filas]); lm = np.array([lb for _, _, _, lb in filas])
    norm = np.array([1.0 if g / max(la, 1) > b / max(lb, 1) else 0.0 for g, b, la, lb in filas])
    uids = [pr["uid"] for pr in pares]
    modelos = {"red": red}
    for nombre in ("kn_bytes", "kn_palabras", "wb5_bytes"):
        modelos[nombre] = np.array(base[nombre]["por_par"], dtype=float)

    def por_paradigma(v):
        return {u: float(np.mean([x for x, uu in zip(v, uids) if uu == u])) for u in por_uid}
    pp = {m: por_paradigma(v) for m, v in modelos.items()}
    media = {m: float(np.mean(list(d.values()))) for m, d in pp.items()}
    campo = {pr["uid"]: pr["campo"] for pr in pares}; term = {pr["uid"]: pr["termino"] for pr in pares}
    mejor_ng = max(media["kn_bytes"], media["kn_palabras"])

    def sobre_azar(v):                         # prueba del signo contra 0,5 (sin los empates)
        k = int(np.sum(v == 1)); m = int(np.sum(v != 0.5))
        return _signo(k, m), k / max(1, m)
    p_red, _ = sobre_azar(red)
    p_kb, _ = sobre_azar(modelos["kn_bytes"]); p_kp, _ = sobre_azar(modelos["kn_palabras"])
    informativo = bool((p_kb < 0.05 and media["kn_bytes"] > 0.5) or (p_kp < 0.05 and media["kn_palabras"] > 0.5)
                       or (p_red < 0.05 and media["red"] > 0.5))
    corta = (lg < lm) | (lm < lg)
    pref_corta = np.where(lg < lm, 1.0, 0.0)[corta]          # «prefiero la más corta» acierta si la buena es más corta
    med = dict(exactitud=media["red"], exactitud_kn_bytes=media["kn_bytes"], orden_kn_bytes=float(base["kn_bytes"]["orden"]),
               exactitud_kn_palabras=media["kn_palabras"], orden_kn_palabras=float(base["kn_palabras"]["orden"]),
               exactitud_wb5_bytes=media["wb5_bytes"],
               palabras_fuera_del_vocabulario=base["kn_palabras"]["palabras_fuera_del_vocabulario"],
               p_signo_red_contra_azar=p_red, p_signo_kn_bytes_contra_azar=p_kb, p_signo_kn_palabras_contra_azar=p_kp,
               blimp_informativo=informativo,
               supera_ngramas=bool(media["red"] > mejor_ng),
               paradigmas_red_sobre_mejor_ngrama=float(np.mean([pp["red"][u] > max(pp["kn_bytes"][u], pp["kn_palabras"][u])
                                                                for u in por_uid])),
               exactitud_por_pares=float(red.mean()), bytes_entrenamiento=float(n),
               exactitud_en_linea=info.get("exactitud_en_linea"), pares=float(len(red)),
               # diagnóstico de la suma de log-probabilidades (no cambia el protocolo)
               pares_distinta_longitud=float(corta.mean()),
               exactitud_misma_longitud=float(red[~corta].mean()) if (~corta).any() else None,
               exactitud_distinta_longitud=float(red[corta].mean()) if corta.any() else None,
               exactitud_prefiere_la_mas_corta=float(pref_corta.mean()) if corta.any() else None,
               exactitud_red_por_byte_diagnostico=float(norm.mean()),
               bits_por_byte_validacion=bits["validacion"][0], acierto_byte_validacion=bits["validacion"][1],
               bits_por_byte_prueba=bits["prueba"][0], acierto_byte_prueba=bits["prueba"][1],
               factor_temperatura_validacion=float(min(bits["validacion"][2], key=bits["validacion"][2].get)),
               bits_por_byte_prueba_temperatura_validada=bits["prueba"][2][min(bits["validacion"][2],
                                                                               key=bits["validacion"][2].get)],
               bits_por_byte_kn_validacion=bkn["validacion"][0], bits_por_byte_kn_prueba=bkn["prueba"][0],
               acierto_byte_kn_prueba=bkn["prueba"][1],
               red_menos_bits_que_kn=bool(bits["prueba"][0] < bkn["prueba"][0]),
               red_menos_bits_que_kn_temperatura_validada=bool(
                   bits["prueba"][2][min(bits["validacion"][2], key=bits["validacion"][2].get)] < bkn["prueba"][0]))
    # el mismo desglose para la red y para cada n-grama (control del diagnóstico): por la
    # relación de largo en bytes entre la buena y la mala, y paradigmas bajo el azar
    lg_c = np.array([len(pr["buena"]) for pr in pares]); lm_c = np.array([len(pr["mala"]) for pr in pares])
    debajo = {}
    for nombre, v in modelos.items():
        for rel, msk in (("buena_corta", lg_c < lm_c), ("igual_largo", lg_c == lm_c), ("buena_larga", lg_c > lm_c)):
            vv = v[msk]
            k = int(np.sum(vv == 1)); m = int(np.sum(vv != 0.5))
            med[f"exactitud_{rel}_{nombre}"] = float(vv.mean()) if len(vv) else None
            med[f"p_signo_{rel}_{nombre}"] = _signo(k, m)
        bajo = []
        for u in por_uid:
            vv = np.array([x for x, uu in zip(v, uids) if uu == u])
            k = int(np.sum(vv == 1)); m = int(np.sum(vv != 0.5))
            if m and k / m < 0.5 and _signo(k, m) < 0.05:
                bajo.append(u)
        debajo[nombre] = bajo
        med[f"paradigmas_bajo_azar_n_{nombre}"] = float(len(bajo))
    med["pares_igual_largo"] = float(np.sum(lg_c == lm_c))
    med["paradigmas_bajo_azar"] = debajo["red"]
    med["paradigmas_bajo_azar_kn_bytes"] = debajo["kn_bytes"]
    med["paradigmas_bajo_azar_comunes_red_kn_bytes"] = float(len(set(debajo["red"]) & set(debajo["kn_bytes"])))
    med["igual_largo_red_bajo_kn_bytes"] = bool(med["exactitud_igual_largo_red"] < med["exactitud_igual_largo_kn_bytes"])
    med["neuronas"] = float(net.stats()["neurons"]); med["max_neuronas"] = float(CFG["layer"]["max_neurons"])
    for nombre in ("kn_bytes", "kn_palabras"):
        for k, v in _comparar_pares(red, modelos[nombre]).items():
            med[f"pares_{k}_{nombre}"] = float(v)
    for c in sorted(set(term.values())):
        us = [u for u in por_uid if term[u] == c]
        for m in modelos:
            med[f"categoria_{c}_{m}"] = float(np.mean([pp[m][u] for u in us]))
        med[f"categoria_{c}_red_supera"] = bool(med[f"categoria_{c}_red"] > max(med[f"categoria_{c}_kn_bytes"],
                                                                                  med[f"categoria_{c}_kn_palabras"]))
    for c in sorted(set(campo.values())):
        us = [u for u in por_uid if campo[u] == c]
        for m in modelos:
            med[f"campo_{c}_{m}"] = float(np.mean([pp[m][u] for u in us]))
    refs = [dict(id="ngrama_kn_bytes_mismos_datos", sistema=f"n-gramas de bytes, Kneser-Ney, orden {base['kn_bytes']['orden']} "
                                                            "(elegido en validación), mismos bytes",
                 valor=media["kn_bytes"], tipo_valor="media", verificacion="medida por la batería"),
            dict(id="ngrama_kn_palabras_mismos_datos", sistema=f"n-gramas de palabras, Kneser-Ney, orden "
                                                               f"{base['kn_palabras']['orden']} (elegido en validación), "
                                                               "mismos bytes",
                 valor=media["kn_palabras"], tipo_valor="media", verificacion="medida por la batería"),
            dict(id="ngrama_wb5_bytes_mismos_datos", sistema="5-gramas de bytes, Witten-Bell, mismos bytes",
                 valor=media["wb5_bytes"], tipo_valor="media", verificacion="medida por la batería")]
    return dict(puntaje=med["exactitud"], medidas=med, segundos=time.time() - t0, referencias_locales=refs,
                validacion_ngramas=dict(kn_bytes=base["kn_bytes"]["validacion_bits_por_byte"],
                                        kn_palabras=base["kn_palabras"]["validacion_bits_por_palabra"]))


def _log(m):
    import sys
    print(m, file=sys.stderr, flush=True)


PRUEBAS = [
    Prueba(
        id="blimp", variante="texto", condicion="texto",
        metrica=f"proporción de pares con más probabilidad para la oración aceptable (media de los 67 paradigmas, "
                f"{PARES} pares cada uno)",
        protocolo=(f"red de texto de exp_text (bytes, 8 de contexto) entrenada en línea con {N_BYTES} bytes de AO-CHILDES; "
                   f"{PARES} pares por paradigma, oración completa; líneas base con los mismos bytes: n-gramas de bytes y "
                   f"de palabras con Kneser-Ney (orden elegido por entropía cruzada en los {N_VALIDACION} bytes "
                   "siguientes del corpus) y 5-gramas de bytes de Witten-Bell; comparación por par mínimo (prueba del "
                   "signo), por paradigma y por fenómeno"),
        desviaciones=[
            f"corpus: {N_BYTES} bytes (unas 140 000 palabras) de habla dirigida a niños, no los 10 M de palabras de "
            "BabyLM (sin espejo en GitHub) ni los corpus de los modelos de la tabla 3 (de 83 M a 8 000 M)",
            "las oraciones de BLiMP se pasan a minúsculas y con la puntuación separada, el formato del corpus",
            f"{PARES} pares por paradigma (las personas: 5 pares por paradigma y 20 validadores)",
            "la probabilidad se lee de la decisión de la red (`readout.probs`), que no fue hecha para puntuar oraciones",
            "las personas juzgan aceptabilidad con toda su competencia; la red sólo predice bytes",
            "la red de texto es determinista dado el corpus: la semilla sólo cambia la exploración, que el "
            "entrenamiento no usa; las semillas 0 y 1 dan la misma red bit a bit, así que no hay variabilidad entre "
            "semillas que medir (n = 1 en los hechos)",
        ],
        correr=correr, version=5, semillas=(0, 1),
        variabilidad="determinista: la red de texto no usa la semilla para aprender y el corpus va siempre en el mismo "
                     "orden; los intervalos salen del bootstrap por bloques del tramo de prueba (en la curva)",
        equivalente=False,
        motivo_equivalencia=("la cifra humana publicada es una referencia no equivalente: las personas no aprendieron "
                             "con 600 KB; la comparación que decide (criterio F) es contra n-gramas con los mismos datos"),
        comparar={"5gramas": "puntaje", "lstm": "puntaje", "txl": "puntaje", "gpt2": "puntaje",
                  "babylm_opt": "puntaje", "babylm_roberta": "puntaje", "babylm_t5": "puntaje"},
        firmas={"blimp_informativo": "algún modelo (la red o un n-grama con los mismos datos) supera al azar con la "
                                     "prueba del signo (p < 0,05): si no, BLiMP no discrimina a esta escala",
                "supera_ngramas": "BLiMP: supera al mejor n-grama de Kneser-Ney con los mismos bytes (sólo cuenta si "
                                  "BLiMP es informativo a esta escala)",
                "red_menos_bits_que_kn": "menos bits por byte que el n-grama de Kneser-Ney de orden 8 con los mismos "
                                         "bytes, en 60 000 bytes que ninguno vio: la comparación del criterio F que sí "
                                         "discrimina a esta escala",
                "red_menos_bits_que_kn_temperatura_validada": "lo mismo con la temperatura de la lectura elegida en "
                                                              "los 60 000 bytes de validación (la red no la ajusta)"},
    ),
]
