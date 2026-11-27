"""Curva de escalado del lenguaje: la red de texto y n-gramas de Kneser-Ney con 0,2, 0,6 y
1,8 MB de AO-CHILDES (regla de hardware contra arquitectura del README: tres escalas).

En cada escala se mide, para la red (3 semillas) y para KN entrenado con los mismos bytes:
  * bits por byte (y acierto del próximo byte) en un tramo fijo que ningún modelo vio: los
    últimos 60 000 bytes del corpus (KN elige su orden con los 60 000 anteriores);
  * BLiMP con los mismos 3 350 pares que la prueba principal;
  * neuronas en uso contra `max_neurons` (si la red llega al tope, una parte plana de la
    curva es del tope de capacidad, no del mecanismo).

La red de 0,2 MB se entrena desde cero; la de 0,6 MB es la de la prueba principal; la de 1,8
MB sigue la misma pasada en línea desde el punto de control final de la de 0,6 MB. Si la red
llega a `max_neurons` antes de 1,8 MB, se agrega una serie con el tope triplicado y una
semilla (marcada: `tope_triple`).
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from . import lenguaje as L
from .base import Pendiente, Prueba, vencido

ESCALAS = (200_000, 600_000, 1_800_000)
N_FIN = 60_000


def datos_barajados(k: int, n: int) -> bytes:
    """Una «semilla de datos»: las emisiones del corpus (sin el tramo fijo del final) en un
    orden al azar con semilla k, y los primeros n bytes. Es otra condición de datos: mezcla
    las edades, que en AO-CHILDES vienen en orden."""
    todo = L.CORPUS.read_bytes()[:-2 * N_FIN]
    lineas = todo.split(b"\n")
    orden = np.random.default_rng(1000 + k).permutation(len(lineas))
    return b"\n".join(lineas[i] for i in orden)[:n]


def tramos_fijos(corto: bool = False) -> dict:
    todo = L.CORPUS.read_bytes()
    t = dict(validacion=todo[-2 * N_FIN:-N_FIN], prueba=todo[-N_FIN:])
    return {k: v[:3000] for k, v in t.items()} if corto else t


def _cfg(tope: int | None):
    cfg = json.loads(json.dumps(L.CFG))
    if tope:
        cfg["layer"]["max_neurons"] = int(tope)
    return cfg


def entrenar_serie(semilla: int, n: int, cfg: dict, etiqueta: str, desde: tuple | None = None, log=None,
                   crudo: bytes | None = None):
    """Como `lenguaje.entrenar`, con otra configuración (`etiqueta` en el nombre de la cache),
    otros datos (`crudo`) y, si `desde` = (nombre de la corrida previa, posición), continuando
    su pasada."""
    from bioelastic import BioElasticNet, TextEncoder
    base = L.CACHE / f"texto_{L._huella()}_s{semilla}_n{n}{etiqueta}"
    data = TextEncoder()(L.corpus(n) if crudo is None else crudo)
    patron = re.compile(re.escape(base.name) + r"_(\d+)\.npz$")
    hechos = sorted(int(m.group(1)) for p in (L.CACHE.glob(base.name + "_*.npz") if L.CACHE.exists() else [])
                    if (m := patron.match(p.name)))
    hecho = max(hechos) if hechos else 0
    if hecho:
        net = BioElasticNet.load(base.parent / f"{base.name}_{hecho}.npz")
        info = json.loads((base.parent / f"{base.name}_{hecho}.json").read_text())
    elif desde is not None:
        previo, pos = desde
        net = BioElasticNet.load(L.CACHE / f"{previo}_{pos}.npz")
        info = json.loads((L.CACHE / f"{previo}_{pos}.json").read_text())
        hecho = pos
    else:
        net = BioElasticNet(256, 256, seed=semilla, **cfg)
        info = dict(aciertos=0, segundos=0.0)
    rate = 4e-3
    while hecho < n - 1:
        ini, fin = hecho, min(n - 1, hecho + L.TRAMO)
        if vencido(margen=rate * (fin - ini) * 1.2):
            raise Pendiente(f"curva: {base.name} en {ini}/{n} bytes")
        t0 = time.time(); ok = 0
        for t in range(ini, fin):
            ok += int(net.forward(int(data[t])) == data[t + 1]); net.teach(int(data[t + 1]))
        dt = time.time() - t0
        rate = dt / max(1, fin - ini)
        info = dict(aciertos=info["aciertos"] + ok, segundos=info["segundos"] + dt)
        L.CACHE.mkdir(parents=True, exist_ok=True)
        (base.parent / f"{base.name}_{fin}.json").write_text(json.dumps(info))
        tmp = base.parent / f"{base.name}_{fin}.tmp.npz"
        net.save(tmp); tmp.replace(base.parent / f"{base.name}_{fin}.npz")
        for h in hechos:
            (base.parent / f"{base.name}_{h}.npz").unlink(missing_ok=True)
            (base.parent / f"{base.name}_{h}.json").unlink(missing_ok=True)
        hechos = [fin]; hecho = fin
        if log:
            log(f"    curva {base.name}: {fin}/{n} bytes ({dt:.0f} s, neuronas {net.stats()['neurons']})")
        if fin < n - 1:
            net = BioElasticNet.load(base.parent / f"{base.name}_{fin}.npz")
    return net, base.name


def red_en(semilla: int, n: int, tope: int | None = None, log=None):
    """La red de la serie (principal o con tope) entrenada con los primeros n bytes."""
    if tope is None and n <= L.N_BYTES:                       # 0,2 y 0,6 MB: la corrida de siempre
        net, _ = L.entrenar(semilla, n, log=log)
        return net, f"texto_{L._huella()}_s{semilla}_n{n}"
    etq = f"_tope{tope}" if tope else ""
    i = ESCALAS.index(n)
    if i == 0:
        return entrenar_serie(semilla, n, _cfg(tope), etq, None, log)
    previo_n = ESCALAS[i - 1]
    _, previo = red_en(semilla, previo_n, tope, log)
    return entrenar_serie(semilla, n, _cfg(tope), f"{etq}_desde{previo_n}", (previo, previo_n - 1), log)


def bits_por_posicion_red(net, datos: bytes) -> np.ndarray:
    """Bits de cada byte del tramo (a partir del segundo) según la red, sin aprender."""
    net.reset_context()
    out = np.empty(len(datos) - 1, dtype=np.float64)
    for i in range(len(datos) - 1):
        net.forward(int(datos[i]))
        p = net.readout.probs(net._last.V)
        out[i] = -np.log2(max(float(p[datos[i + 1]]), 1e-12))
    return out


def bits_por_posicion_kn(kb, datos: bytes) -> np.ndarray:
    if hasattr(kb, "logp_posiciones"):
        return -kb.logp_posiciones(datos)[1:] / np.log(2)
    return np.array([-kb.logp(datos[max(0, i - kb.k + 1):i], datos[i]) / np.log(2) for i in range(1, len(datos))])


class KneserNeyCompacto:
    """El mismo modelo que `lenguaje.KneserNey` sobre bytes (interpolación de Kneser-Ney con
    descuento por orden n1 / (n1 + 2 n2), conteos de continuación debajo del orden usado y la
    uniforme sobre `V` al fondo), con tablas ordenadas de numpy en lugar de diccionarios. Cada
    n-grama de hasta 10 bytes es un entero de 6 bits por byte: los bytes vistos al entrenar se
    numeran de 1 a 63 y el 0 queda para los no vistos, que así nunca coinciden con un n-grama
    del entrenamiento. Con 1,8 MB los diccionarios pasan de 1 GB; esto usa decenas de MB y da
    las mismas probabilidades (ver tests/test_battery.py)."""

    BITS = 6

    def __init__(self, orden_max: int, V: int = 256):
        if orden_max * self.BITS > 63:
            raise ValueError("orden demasiado alto para 6 bits por byte en 64 bits")
        self.K, self.V = int(orden_max), int(V)

    def _gramas(self, cod: np.ndarray, k: int) -> np.ndarray:
        """El entero de cada k-grama de `cod`, en el orden de su último byte."""
        m = len(cod) - k + 1
        if m <= 0:
            return np.zeros(0, dtype=np.uint64)
        g = np.zeros(m, dtype=np.uint64)
        for j in range(k):
            g = (g << np.uint64(self.BITS)) | cod[j:j + m]
        return g

    def _cod(self, seq: bytes) -> np.ndarray:
        return self.mapa[np.frombuffer(bytes(seq), dtype=np.uint8)]

    def entrenar(self, secuencias):
        vistos = sorted(set().union(*[set(s) for s in secuencias]))
        if len(vistos) > 2 ** self.BITS - 1:
            raise ValueError(f"{len(vistos)} bytes distintos: no entran en {self.BITS} bits")
        self.mapa = np.zeros(256, dtype=np.uint64)
        self.mapa[vistos] = np.arange(1, len(vistos) + 1, dtype=np.uint64)
        cods = [self._cod(s) for s in secuencias]
        self.c = [None] + [np.unique(np.concatenate([self._gramas(c, k) for c in cods]), return_counts=True)
                           for k in range(1, self.K + 1)]
        mask = [np.uint64((1 << (self.BITS * k)) - 1) for k in range(self.K + 1)]
        self.cont = [None] + [np.unique(self.c[k + 1][0] & mask[k], return_counts=True) for k in range(1, self.K)]
        self.orden(self.K)
        return self

    def orden(self, K: int):
        self.k = int(K)
        self.ef = [None] + [self.cont[k] if k < self.k else self.c[k] for k in range(1, self.k + 1)]
        self.hist, self.tot, self.tipos, self.D = [None], [None], [None], [None]
        for k in range(1, self.k + 1):
            claves, v = self.ef[k]
            pref = claves >> np.uint64(self.BITS)
            h, ini = np.unique(pref, return_index=True)
            self.hist.append(h)
            self.tot.append(np.add.reduceat(v, ini).astype(np.float64) if len(v) else np.zeros(0))
            self.tipos.append(np.diff(np.r_[ini, len(v)]).astype(np.float64))
            n1 = int(np.sum(v == 1)); n2 = int(np.sum(v == 2))
            self.D.append(n1 / (n1 + 2 * n2) if n1 + n2 else 0.5)
        return self

    @staticmethod
    def _buscar(claves: np.ndarray, q: np.ndarray):
        i = np.searchsorted(claves, q)
        i = np.minimum(i, max(len(claves) - 1, 0))
        ok = (claves[i] == q) if len(claves) else np.zeros(len(q), dtype=bool)
        return i, ok

    def logp_posiciones(self, seq: bytes) -> np.ndarray:
        """log p de cada byte de `seq` dado los k-1 anteriores (el primero, sin historia)."""
        cod = self._cod(seq)
        n = len(cod)
        p = np.full(n, 1.0 / self.V)
        for k in range(1, self.k + 1):
            if n < k:
                break
            g = self._gramas(cod, k)                          # posiciones k-1 .. n-1
            h = g >> np.uint64(self.BITS)
            ih, okh = self._buscar(self.hist[k], h)
            if not okh.any():
                continue
            ig, okg = self._buscar(self.ef[k][0], g)
            c = np.where(okg, self.ef[k][1][ig], 0).astype(np.float64)
            tot = self.tot[k][ih]; tip = self.tipos[k][ih]
            D = self.D[k]
            q = p[k - 1:]
            nuevo = np.maximum(c - D, 0.0) / tot + D * tip / tot * q
            p[k - 1:] = np.where(okh, nuevo, q)
        return np.log(p)

    def logp(self, hist, w) -> float:
        return float(self.logp_posiciones(bytes(hist) + bytes([w]))[-1]) if len(hist) < self.k else \
            float(self.logp_posiciones(bytes(hist[len(hist) - self.k + 1:]) + bytes([w]))[-1])

    def secuencia(self, seq, desde: int = 1) -> float:
        return float(np.sum(self.logp_posiciones(seq)[desde:]))


BLOQUE, B_REMUESTRAS = 1000, 2000


def bootstrap_bloques(x: np.ndarray, semilla: int = 0) -> dict:
    """Intervalo del 95 % de la media por bootstrap por bloques contiguos de `BLOQUE` bytes
    (respeta la dependencia entre bytes vecinos)."""
    nb = len(x) // BLOQUE
    medias = x[:nb * BLOQUE].reshape(nb, BLOQUE).mean(axis=1)
    rng = np.random.default_rng(semilla)
    rem = medias[rng.integers(nb, size=(B_REMUESTRAS, nb))].mean(axis=1)
    return dict(media=float(x.mean()), ic95=[float(np.quantile(rem, 0.025)), float(np.quantile(rem, 0.975))],
                bloques=int(nb))


def evaluar_red(net, clave: str, pares: list, corto: bool = False) -> dict:
    """Bits por byte en los tramos fijos y BLiMP, con caches por clave (reanudable)."""
    cb = L.CACHE / f"bitsfin2_{clave}{'_corto' if corto else ''}.json"
    bits = json.loads(cb.read_text()) if cb.exists() else {}
    for nombre, datos in tramos_fijos(corto).items():
        if nombre not in bits:
            if vencido(margen=len(datos) * 5e-3):
                raise Pendiente(f"curva: bits por byte de {clave}")
            bits[nombre] = list(L.bits_red(net, datos)); cb.write_text(json.dumps(bits))
    cache = L.CACHE / f"blimp2_{clave.replace('texto_', '')}_p{len(pares)}.json"
    hechos = json.loads(cache.read_text()) if cache.exists() else {}
    por_uid: dict = {}
    for i, pr in enumerate(pares):
        por_uid.setdefault(pr["uid"], []).append(i)
    t_par = None
    for uid, idx in por_uid.items():
        if uid in hechos:
            continue
        if t_par is not None and vencido(margen=t_par * len(idx) * 1.5):
            raise Pendiente(f"curva: BLiMP de {clave} en {len(hechos)}/{len(por_uid)}")
        t1 = time.time()
        hechos[uid] = [list(L.par_red(net, pares[i]["buena"], pares[i]["mala"])) for i in idx]
        t_par = (time.time() - t1) / max(1, len(idx))
        cache.write_text(json.dumps(hechos))
    filas = [r for uid in por_uid for r in hechos[uid]]
    red = np.array([1.0 if g > b else (0.5 if g == b else 0.0) for g, b, _, _ in filas])
    uids = [pares[i]["uid"] for uid in por_uid for i in por_uid[uid]]
    pp = [float(np.mean([x for x, u in zip(red, uids) if u == uid])) for uid in por_uid]
    st = net.stats()
    f_val = min(bits["validacion"][2], key=bits["validacion"][2].get)
    cp = L.CACHE / f"bitsposfin_{clave}{'_corto' if corto else ''}.npy"
    if not cp.exists():
        if vencido(margen=len(tramos_fijos(corto)["prueba"]) * 3e-3):
            raise Pendiente(f"curva: bits por posición de {clave}")
        np.save(cp, bits_por_posicion_red(net, tramos_fijos(corto)["prueba"]))
    return dict(bits_prueba=bits["prueba"][0], acierto_byte_prueba=bits["prueba"][1],
                bits_por_posicion=str(cp),
                bits_validacion=bits["validacion"][0], factor_temperatura=float(f_val),
                bits_prueba_temperatura_validada=bits["prueba"][2][f_val], blimp=float(np.mean(pp)),
                p_signo_blimp=L._signo(int(np.sum(red == 1)), int(np.sum(red != 0.5))),
                neuronas=float(st["neurons"]), creadas=float(st["created"]), podadas=float(st["pruned"]),
                desalojadas=float(st.get("evicted", 0)))


def kn_en(n: int, pares: list, corto: bool = False, crudo: bytes | None = None, etiqueta: str = "") -> dict:
    """KN de bytes con los mismos n bytes: orden elegido con el tramo fijo de validación,
    bits por byte en el de prueba (también por posición) y BLiMP. Determinista."""
    ruta = L.CACHE / f"kn_escala_n{n}{etiqueta}_p{len(pares)}{'_corto' if corto else ''}.json"
    if ruta.exists():
        return json.loads(ruta.read_text())
    tr = tramos_fijos(corto)
    kb = KneserNeyCompacto(L.ORDEN_MAX_BYTES, 256).entrenar([L.CORPUS.read_bytes()[:n] if crudo is None else crudo])
    hval = {}
    for K in range(1, L.ORDEN_MAX_BYTES + 1):
        kb.orden(K)
        hval[K] = -kb.secuencia(tr["validacion"]) / max(1, len(tr["validacion"]) - 1) / np.log(2)
    K = min(hval, key=hval.get)
    kb.orden(K)
    bits = -kb.secuencia(tr["prueba"]) / max(1, len(tr["prueba"]) - 1) / np.log(2)
    cp = ruta.with_suffix(".npy")
    np.save(cp, bits_por_posicion_kn(kb, tr["prueba"]))
    v = np.array([1.0 if (g := kb.secuencia(b"\n" + p["buena"] + b"\n")) > (b := kb.secuencia(b"\n" + p["mala"] + b"\n"))
                  else (0.5 if g == b else 0.0) for p in pares])
    uids = [p["uid"] for p in pares]
    blimp = float(np.mean([np.mean([x for x, u in zip(v, uids) if u == uid]) for uid in dict.fromkeys(uids)]))
    out = dict(orden=K, bits_prueba=float(bits), bits_por_posicion=str(cp),
               validacion={str(k): float(h) for k, h in hval.items()}, blimp=blimp,
               p_signo_blimp=L._signo(int(np.sum(v == 1)), int(np.sum(v != 0.5))))
    L.CACHE.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(out))
    return out


def correr(semilla: int, escala: float = 1.0, tope: int | None = None) -> dict:
    t0 = time.time()
    escalas = ESCALAS if escala >= 1.0 else (6000, 12000, 24000)
    pares = L.pares_blimp(L.PARES if escala >= 1.0 else 2)
    med, curva = {}, []
    for n in escalas:
        if escala < 1.0:
            net, _ = L.entrenar(semilla, n)
            clave = f"texto_{L._huella()}_s{semilla}_n{n}"
        else:
            net, clave = red_en(semilla, n, tope, log=L._log)
        r = evaluar_red(net, clave, pares, corto=escala < 1.0)
        del net                                        # la memoria de la red no hace falta para KN
        k = kn_en(n, pares, corto=escala < 1.0)
        br = np.load(r.pop("bits_por_posicion")); bk_ = np.load(k["bits_por_posicion"])
        r["bootstrap_red"] = bootstrap_bloques(br, 1); r["bootstrap_kn"] = bootstrap_bloques(bk_, 2)
        r["bootstrap_red_menos_kn"] = bootstrap_bloques(br - bk_, 3)
        mb = n / 1e6
        for c, v in r.items():
            if isinstance(v, dict):
                med[f"{c}_media_{mb:g}MB"] = v["media"]
                med[f"{c}_ic95_inf_{mb:g}MB"] = v["ic95"][0]; med[f"{c}_ic95_sup_{mb:g}MB"] = v["ic95"][1]
                continue
            med[f"{c}_{mb:g}MB"] = v
        med[f"kn_bits_prueba_{mb:g}MB"] = k["bits_prueba"]; med[f"kn_orden_{mb:g}MB"] = float(k["orden"])
        med[f"kn_blimp_{mb:g}MB"] = k["blimp"]; med[f"kn_p_signo_blimp_{mb:g}MB"] = k["p_signo_blimp"]
        curva.append(dict(bytes=n, red=r, kn=k))
    # variabilidad por muestra de datos, sólo en la escala menor (otra condición: edades mezcladas)
    if tope is None and escala >= 1.0:
        n0 = escalas[0]
        for kd in (1, 2):
            crudo = datos_barajados(kd, n0)
            net, nombre = entrenar_serie(semilla, n0, _cfg(None), f"_barajado{kd}", None, L._log, crudo=crudo)
            r = evaluar_red(net, nombre, pares)
            del net
            k = kn_en(n0, pares, crudo=crudo, etiqueta=f"_barajado{kd}")
            br = np.load(r.pop("bits_por_posicion")); bk_ = np.load(k["bits_por_posicion"])
            med[f"barajado{kd}_bits_red_{n0 / 1e6:g}MB"] = r["bits_prueba"]
            med[f"barajado{kd}_bits_kn_{n0 / 1e6:g}MB"] = k["bits_prueba"]
            med[f"barajado{kd}_red_menos_kn_{n0 / 1e6:g}MB"] = float((br - bk_).mean())
            med[f"barajado{kd}_blimp_red_{n0 / 1e6:g}MB"] = r["blimp"]; med[f"barajado{kd}_blimp_kn_{n0 / 1e6:g}MB"] = k["blimp"]
            med[f"barajado{kd}_neuronas_{n0 / 1e6:g}MB"] = r["neuronas"]
    tope_n = (tope or L.CFG["layer"]["max_neurons"])
    b = [c["red"]["bits_prueba"] for c in curva]; bk = [c["kn"]["bits_prueba"] for c in curva]
    bt = [c["red"]["bits_prueba_temperatura_validada"] for c in curva]
    med.update(max_neuronas=float(tope_n),
               satura_antes_del_final=bool(any(c["red"]["neuronas"] >= 0.99 * tope_n for c in curva[:-1])),
               satura=bool(any(c["red"]["neuronas"] >= 0.99 * tope_n for c in curva)),
               curva_baja_bits=bool(b[0] > b[1] > b[2]), red_menos_bits_que_kn_al_final=bool(b[-1] < bk[-1]),
               mejora_bits_ultimo_tramo=float(b[1] - b[2]), mejora_bits_kn_ultimo_tramo=float(bk[1] - bk[2]),
               curva_bits=b, curva_bits_kn=bk, curva_bits_temperatura_validada=bt,
               curva_red_menos_kn=[c["red"]["bootstrap_red_menos_kn"]["media"] for c in curva],
               curva_red_menos_kn_ic95=[c["red"]["bootstrap_red_menos_kn"]["ic95"] for c in curva],
               red_menos_bits_que_kn_al_final_temperatura_validada=bool(bt[-1] < bk[-1]), curva_blimp=[c["red"]["blimp"] for c in curva],
               curva_blimp_kn=[c["kn"]["blimp"] for c in curva], curva_neuronas=[c["red"]["neuronas"] for c in curva])
    return dict(puntaje=b[-1], medidas=med, segundos=time.time() - t0, curva=curva)


def correr_tope_triple(semilla: int, escala: float = 1.0) -> dict:
    return correr(semilla, escala, tope=3 * L.CFG["layer"]["max_neurons"])


N_EXPLORATORIO = 5_400_000


def correr_tope_triple_exploratorio(semilla: int, escala: float = 1.0) -> dict:
    """5,4 MB con el tope triplicado, continuando la pasada de la serie de tope triple desde
    1,8 MB. Exploratorio y condicional: sólo si con 1,8 MB esa serie todavía baja los bits
    por byte y no llegó al tope (si no, la prueba es «no aplicable hoy» con el motivo)."""
    from .base import NoAplicable
    if escala < 1.0:
        raise NoAplicable("exploratorio: sólo a escala completa")
    tope = 3 * L.CFG["layer"]["max_neurons"]
    pares = L.pares_blimp(L.PARES)
    n18 = ESCALAS[-1]
    puntos = []
    for n in ESCALAS[1:]:                                   # 0,6 y 1,8 MB de la serie, de las caches
        net, clave = red_en(semilla, n, tope, log=L._log)
        r = evaluar_red(net, clave, pares)
        puntos.append(dict(bytes=n, bits=r["bits_prueba"], neuronas=r["neuronas"], clave=clave))
        del net
    baja = puntos[1]["bits"] < puntos[0]["bits"]
    satura = puntos[1]["neuronas"] >= 0.99 * tope
    if not baja or satura:
        raise NoAplicable(f"con 1,8 MB la serie de tope triple {'ya no baja los bits por byte' if not baja else ''}"
                          f"{' y ' if not baja and satura else ''}{'llegó al tope de neuronas' if satura else ''}: "
                          f"no corresponde extenderla a 5,4 MB")
    net, nombre = entrenar_serie(semilla, N_EXPLORATORIO, _cfg(tope), f"_tope{tope}_desde{n18}",
                                 (puntos[1]["clave"], n18 - 1), L._log)
    t0 = time.time()
    r = evaluar_red(net, nombre, pares)
    del net
    k = kn_en(N_EXPLORATORIO, pares)
    br = np.load(r.pop("bits_por_posicion")); bk_ = np.load(k["bits_por_posicion"])
    med = {f"{c}_5.4MB": v for c, v in r.items()}
    for c, x in (("bootstrap_red", br), ("bootstrap_kn", bk_), ("bootstrap_red_menos_kn", br - bk_)):
        b = bootstrap_bloques(x, dict(bootstrap_red=1, bootstrap_kn=2, bootstrap_red_menos_kn=3)[c])
        med[f"{c}_media_5.4MB"] = b["media"]; med[f"{c}_ic95_inf_5.4MB"] = b["ic95"][0]
        med[f"{c}_ic95_sup_5.4MB"] = b["ic95"][1]
    med.update({"kn_bits_prueba_5.4MB": k["bits_prueba"], "kn_orden_5.4MB": float(k["orden"]),
                "kn_blimp_5.4MB": k["blimp"], "bits_prueba_1.8MB": puntos[1]["bits"],
                "neuronas_1.8MB": puntos[1]["neuronas"]})
    med.update(max_neuronas=float(tope),
               sigue_bajando=bool(r["bits_prueba"] < puntos[1]["bits"]), satura=bool(r["neuronas"] >= 0.99 * tope),
               red_menos_bits_que_kn=bool(r["bits_prueba"] < k["bits_prueba"]))
    return dict(puntaje=r["bits_prueba"], medidas=med, segundos=time.time() - t0)


_DESV = [
    "corpus de habla dirigida a niños, muy repetitivo: a estas escalas BLiMP no discrimina (los n-gramas quedan en "
    "el azar) y la comparación que decide es la de bits por byte",
    "el tramo fijo no visto es el final del corpus, el mismo para todas las escalas",
    "la red de 1,8 MB continúa la pasada de la de 0,6 MB desde su punto de control final (el contexto de 8 bytes "
    "empieza vacío al recargar)",
]
PRUEBAS = [
    Prueba(
        id="blimp", variante="curva_escalado", condicion="texto",
        metrica="bits por byte de la red en el tramo fijo no visto con 1,8 MB (menos es mejor); la curva es el resultado",
        protocolo=("red de texto de exp_text con 0,2, 0,6 y 1,8 MB de AO-CHILDES (una pasada en línea) y KN de bytes con "
                   "los mismos datos; bits por byte en los últimos 60 000 bytes del corpus, BLiMP y neuronas en cada escala"),
        desviaciones=_DESV + ["una corrida por escala: la red de texto es determinista dado el corpus (las semillas dan "
                              "la misma red bit a bit)"], correr=correr, version=1, semillas=(0,),
        equivalente=False, motivo_equivalencia="curva de diagnóstico (hardware contra arquitectura), sin referencia humana",
        referencia=None, azar=8.0, azar_derivacion="8 bits por byte: distribución uniforme sobre 256 bytes",
        mayor_es_mejor=False,
        firmas={"curva_baja_bits": "los bits por byte de la red bajan en las tres escalas",
                "red_menos_bits_que_kn_al_final": "la red predice mejor que KN con los mismos 1,8 MB",
                "satura": "la red llega a max_neurons en alguna escala (el tope de capacidad limita la curva)"},
    ),
    Prueba(
        id="blimp", variante="curva_escalado_tope_triple", condicion="texto",
        metrica="bits por byte con 1,8 MB, con max_neurons triplicado (serie marcada, una semilla)",
        protocolo="la misma curva con el tope de neuronas triplicado (90 000), desde cero",
        desviaciones=_DESV + ["serie marcada: sólo si la serie principal llega al tope antes de 1,8 MB; una semilla"],
        correr=correr_tope_triple, version=1, semillas=(0,),
        equivalente=False, motivo_equivalencia="diagnóstico del tope de capacidad", referencia=None, azar=8.0,
        azar_derivacion="8 bits por byte: distribución uniforme sobre 256 bytes", mayor_es_mejor=False,
        firmas={"curva_baja_bits": "los bits por byte bajan en las tres escalas", "satura": "llega al tope"},
    ),
    Prueba(
        id="blimp", variante="curva_tope_triple_5_4MB_exploratorio", condicion="texto",
        metrica="bits por byte con 5,4 MB y max_neurons triplicado (exploratorio, una semilla)",
        protocolo=("la serie de tope triple sigue su pasada en línea de 1,8 a 5,4 MB; mismo tramo fijo, BLiMP y KN de "
                   "bytes con los mismos 5,4 MB; sólo si con 1,8 MB esa serie todavía baja y no llegó al tope"),
        desviaciones=_DESV + ["exploratorio: una semilla, un punto más, condicionado a lo que dio la serie de tope "
                              "triple con 1,8 MB"],
        correr=correr_tope_triple_exploratorio, version=1, semillas=(0,),
        equivalente=False, motivo_equivalencia="exploratorio (hardware contra arquitectura)", referencia=None, azar=8.0,
        azar_derivacion="8 bits por byte: distribución uniforme sobre 256 bytes", mayor_es_mejor=False,
        firmas={"sigue_bajando": "los bits por byte bajan de 1,8 a 5,4 MB", "satura": "llega al tope",
                "red_menos_bits_que_kn": "la red predice mejor que KN con los mismos 5,4 MB"},
    ),
]
