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


def tramos_fijos(corto: bool = False) -> dict:
    todo = L.CORPUS.read_bytes()
    t = dict(validacion=todo[-2 * N_FIN:-N_FIN], prueba=todo[-N_FIN:])
    return {k: v[:3000] for k, v in t.items()} if corto else t


def _cfg(tope: int | None):
    cfg = json.loads(json.dumps(L.CFG))
    if tope:
        cfg["layer"]["max_neurons"] = int(tope)
    return cfg


def entrenar_serie(semilla: int, n: int, cfg: dict, etiqueta: str, desde: tuple | None = None, log=None):
    """Como `lenguaje.entrenar`, con otra configuración (`etiqueta` en el nombre de la cache)
    y, si `desde` = (nombre de la corrida previa, posición), continuando su pasada."""
    from bioelastic import BioElasticNet, TextEncoder
    base = L.CACHE / f"texto_{L._huella()}_s{semilla}_n{n}{etiqueta}"
    data = TextEncoder()(L.corpus(n))
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
    return dict(bits_prueba=bits["prueba"][0], acierto_byte_prueba=bits["prueba"][1],
                bits_validacion=bits["validacion"][0], factor_temperatura=float(f_val),
                bits_prueba_temperatura_validada=bits["prueba"][2][f_val], blimp=float(np.mean(pp)),
                p_signo_blimp=L._signo(int(np.sum(red == 1)), int(np.sum(red != 0.5))),
                neuronas=float(st["neurons"]), creadas=float(st["created"]), podadas=float(st["pruned"]),
                desalojadas=float(st.get("evicted", 0)))


def kn_en(n: int, pares: list, corto: bool = False) -> dict:
    """KN de bytes con los mismos n bytes: orden elegido con el tramo fijo de validación,
    bits por byte en el de prueba y BLiMP. Determinista: una sola vez por escala."""
    ruta = L.CACHE / f"kn_escala_n{n}_p{len(pares)}{'_corto' if corto else ''}.json"
    if ruta.exists():
        return json.loads(ruta.read_text())
    tr = tramos_fijos(corto)
    kb = L.KneserNey(L.ORDEN_MAX_BYTES, 256).entrenar([L.CORPUS.read_bytes()[:n]])
    hval = {}
    for K in range(1, L.ORDEN_MAX_BYTES + 1):
        kb.orden(K)
        hval[K] = -kb.secuencia(tr["validacion"]) / max(1, len(tr["validacion"]) - 1) / np.log(2)
    K = min(hval, key=hval.get)
    kb.orden(K)
    bits = -kb.secuencia(tr["prueba"]) / max(1, len(tr["prueba"]) - 1) / np.log(2)
    v = np.array([1.0 if (g := kb.secuencia(b"\n" + p["buena"] + b"\n")) > (b := kb.secuencia(b"\n" + p["mala"] + b"\n"))
                  else (0.5 if g == b else 0.0) for p in pares])
    uids = [p["uid"] for p in pares]
    blimp = float(np.mean([np.mean([x for x, u in zip(v, uids) if u == uid]) for uid in dict.fromkeys(uids)]))
    out = dict(orden=K, bits_prueba=float(bits), validacion={str(k): float(h) for k, h in hval.items()}, blimp=blimp,
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
        k = kn_en(n, pares, corto=escala < 1.0)
        mb = n / 1e6
        for c, v in r.items():
            med[f"{c}_{mb:g}MB"] = v
        med[f"kn_bits_prueba_{mb:g}MB"] = k["bits_prueba"]; med[f"kn_orden_{mb:g}MB"] = float(k["orden"])
        med[f"kn_blimp_{mb:g}MB"] = k["blimp"]; med[f"kn_p_signo_blimp_{mb:g}MB"] = k["p_signo_blimp"]
        curva.append(dict(bytes=n, red=r, kn=k))
    tope_n = (tope or L.CFG["layer"]["max_neurons"])
    b = [c["red"]["bits_prueba"] for c in curva]; bk = [c["kn"]["bits_prueba"] for c in curva]
    bt = [c["red"]["bits_prueba_temperatura_validada"] for c in curva]
    med.update(max_neuronas=float(tope_n),
               satura_antes_del_final=bool(any(c["red"]["neuronas"] >= 0.99 * tope_n for c in curva[:-1])),
               satura=bool(any(c["red"]["neuronas"] >= 0.99 * tope_n for c in curva)),
               curva_baja_bits=bool(b[0] > b[1] > b[2]), red_menos_bits_que_kn_al_final=bool(b[-1] < bk[-1]),
               mejora_bits_ultimo_tramo=float(b[1] - b[2]), mejora_bits_kn_ultimo_tramo=float(bk[1] - bk[2]),
               curva_bits=b, curva_bits_kn=bk, curva_bits_temperatura_validada=bt,
               red_menos_bits_que_kn_al_final_temperatura_validada=bool(bt[-1] < bk[-1]), curva_blimp=[c["red"]["blimp"] for c in curva],
               curva_blimp_kn=[c["kn"]["blimp"] for c in curva], curva_neuronas=[c["red"]["neuronas"] for c in curva])
    return dict(puntaje=b[-1], medidas=med, segundos=time.time() - t0, curva=curva)


def correr_tope_triple(semilla: int, escala: float = 1.0) -> dict:
    return correr(semilla, escala, tope=3 * L.CFG["layer"]["max_neurons"])


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
        desviaciones=_DESV, correr=correr, version=1, semillas=(0, 1, 2),
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
]
