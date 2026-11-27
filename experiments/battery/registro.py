"""Lista de pruebas de la batería, corrida reanudable de cada una y armado del informe."""
from __future__ import annotations

import inspect
import time

import numpy as np

from . import base as _base
from . import esquema
from .base import NoAplicable, Pendiente


def pruebas() -> list:
    from . import harlow, inferencia, memoria_trabajo, reversion
    out = [*harlow.PRUEBAS, *reversion.PRUEBAS, *inferencia.PRUEBAS, *memoria_trabajo.PRUEBAS]
    for mod in ("omniglot", "igual_distinto", "metacognicion", "lenguaje"):
        try:
            m = __import__(f"{__package__}.{mod}", fromlist=["PRUEBAS"])
        except ModuleNotFoundError as e:
            if e.name != f"{__package__}.{mod}":
                raise
            continue
        out += list(m.PRUEBAS)
    return out


def _acepta_config(fn) -> bool:
    return "config" in inspect.signature(fn).parameters


def seleccionar(p, almacen, escala: float, log=print) -> dict | None:
    """Elige la configuración de la red en validación (ver `base.Prueba.configs`)."""
    if not p.configs:
        return None
    punt = {}
    for nombre, cfg in p.configs.items():
        clave = f"_validacion__{p.clave}__{nombre}"
        vals = []
        for s in p.semillas_validacion:
            c = almacen.cargar(clave, s, p.version, escala)
            if c is None:
                t0 = time.time()
                try:
                    v = float(p.validar(cfg, s, escala))
                    c = dict(semilla=s, version=p.version, escala=escala, valor=v, segundos=time.time() - t0)
                except NoAplicable as e:
                    c = dict(semilla=s, version=p.version, escala=escala, valor=None, no_aplicable=str(e))
                almacen.guardar(clave, s, c)
                log(f"  validación {p.clave} {nombre} s{s}: {c['valor']}")
            vals.append(c["valor"])
        m, sd = esquema.media_desvio(vals)
        punt[nombre] = dict(media=m, desvio=sd, por_semilla=vals)
    validas = {k: v for k, v in punt.items() if v["media"] is not None}
    if not validas:
        return dict(elegida=None, puntajes=punt, criterio=p.criterio_validacion,
                    semillas=list(p.semillas_validacion))
    signo = 1.0 if p.validar_mayor else -1.0
    elegida = max(validas, key=lambda k: (signo * validas[k]["media"], -list(p.configs).index(k)))
    return dict(elegida=elegida, puntajes=punt, criterio=p.criterio_validacion,
                semillas=list(p.semillas_validacion), opciones={k: p.configs[k] for k in p.configs})


def correr(p, almacen, escala: float = 1.0, semillas=None, limite: float | None = None, log=print):
    """Corre (o recupera) cada semilla de la prueba `p`. `limite`: hora (time.time()) a partir
    de la cual no se empiezan semillas nuevas; lo que falte queda para la próxima corrida.
    Devuelve (crudos, selección, completa)."""
    sel = seleccionar(p, almacen, escala, log)
    cfg_nombre = sel["elegida"] if sel else None
    cfg = p.configs[cfg_nombre] if (sel and cfg_nombre) else None
    crudos, completa = [], True
    _base.LIMITE = limite
    for s in (semillas if semillas is not None else p.semillas):
        c = almacen.cargar(p.clave, s, p.version, escala)
        if c is not None and cfg_nombre is not None and c.get("config") != cfg_nombre:
            c = None                                  # cambió la configuración elegida
        if c is None:
            if limite is not None and time.time() > limite:
                completa = False
                continue
            t0 = time.time()
            try:
                if sel is not None and cfg_nombre is None:
                    raise NoAplicable("ninguna configuración documentada corre en este commit")
                r = p.correr(s, escala, cfg) if _acepta_config(p.correr) else p.correr(s, escala)
                c = dict(r, semilla=s, version=p.version, escala=escala, config=cfg_nombre)
            except Pendiente as e:
                log(f"  {p.clave} s{s}: queda para la próxima corrida ({e})")
                completa = False
                break                                 # se acabó el tiempo de esta corrida
            except NoAplicable as e:
                c = dict(semilla=s, version=p.version, escala=escala, config=cfg_nombre, no_aplicable=str(e),
                         segundos=time.time() - t0)
            almacen.guardar(p.clave, s, c)
            log(f"  {p.clave} s{s}: puntaje={c.get('puntaje')} ({c.get('segundos', 0):.1f} s)"
                + (f" NO APLICABLE: {c['no_aplicable']}" if c.get("no_aplicable") else ""))
        crudos.append(c)
    return crudos, sel, completa


def controles(p, almacen, escala: float, sel: dict | None, semillas=None, limite=None, log=print) -> dict:
    """Las otras configuraciones documentadas, corridas en las mismas semillas de prueba
    como control (no cambian la elegida ni el puntaje principal): muestran cuánto depende
    el resultado de la configuración."""
    out = {}
    if not sel or not sel.get("elegida"):
        return out
    for nombre, cfg in p.configs.items():
        if nombre == sel["elegida"]:
            continue
        clave = f"_control__{p.clave}__{nombre}"
        vals = []
        for s in (semillas if semillas is not None else p.semillas):
            c = almacen.cargar(clave, s, p.version, escala)
            if c is None:
                if limite is not None and time.time() > limite:
                    continue
                t0 = time.time()
                try:
                    r = p.correr(s, escala, cfg)
                    c = dict(r, semilla=s, version=p.version, escala=escala, config=nombre)
                except NoAplicable as e:
                    c = dict(semilla=s, version=p.version, escala=escala, config=nombre, no_aplicable=str(e),
                             segundos=time.time() - t0)
                almacen.guardar(clave, s, c)
                log(f"  control {p.clave} {nombre} s{s}: puntaje={c.get('puntaje')}")
            vals.append(c)
        m, sd = esquema.media_desvio([c.get("puntaje") for c in vals])
        det = {}
        for k in sorted({k for c in vals for k, v in c.get("medidas", {}).items() if isinstance(v, (int, float, bool))}):
            mk, sk = esquema.media_desvio([c.get("medidas", {}).get(k) for c in vals])
            det[k] = dict(media=mk, desvio=sk)
        out[nombre] = dict(media=m, desvio=sd, por_semilla=[c.get("puntaje") for c in vals],
                           semillas=[c["semilla"] for c in vals], detalle=det)
    return out


def fila(p, crudos, sel, catalogo, commit, ctrl: dict | None = None) -> dict:
    f = esquema.fila_resultado(p, crudos, catalogo, commit)
    if sel is not None:
        f["configuracion"] = dict(sel, controles=ctrl or {})
    elif crudos:
        f["configuracion"] = dict(elegida="única documentada para esta prueba")
    return f


def media(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None
