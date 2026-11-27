"""Percepción compartida por varias pruebas: la red `depth2` publicada entrenada en MNIST.

`depth2` es la configuración de exp_vision (V1 con error local, memoria sobre la entrada,
proyección balanceada, árbitro), entrenada en una sola pasada en línea con las primeras
`n` imágenes del orden de la semilla (el mismo orden que exp_vision: permutación de las
55 000 de entrenamiento). Se guarda en ext_data/cache (no versionado) por commit medido.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[2] / "ext_data" / "cache"

# Copia literal de exp_vision.CONFIGS["depth2"].
V1L = dict(rf=5, pool=2, max_features=256, theta=0.95, floor=0.3, k=8, local=dict(lr=0.5))
DEPTH2 = dict(areas=[V1L], heads=[dict(source="input")], projection=dict(source="all", balance=True),
              replay=0.5, joint=False, arbiter=0.01)
N_MNIST = 20_000


from .version import ARCHIVOS_PERCEPCION, huella, huella_equivalente


def _huella_red() -> str:
    """Identifica el código de percepción importado (para no mezclar caches entre commits). Si
    esos archivos son idénticos entre dos commits, la red entrenada también lo es (el
    entrenamiento es determinista) y la cache se comparte; el Thinker no entra."""
    import bioelastic
    raiz = Path(bioelastic.__file__).resolve().parent
    return huella_equivalente(huella((n, (raiz / n).read_bytes() if (raiz / n).exists() else b"")
                                     for n in ARCHIVOS_PERCEPCION))


def encoder():
    from bioelastic import ImageEncoder
    enc = ImageEncoder((28, 28), blur=0.4)
    return lambda x: enc(x).reshape(28, 28, 1)


def mnist():
    from bioelastic.datasets import load_mnist
    return load_mnist()


def cortex_mnist(semilla: int, n: int = N_MNIST, log=None, config: dict | None = None):
    """La red depth2 entrenada con las primeras `n` imágenes del orden de la semilla (con
    opciones extra de `config`, p. ej. `metacog`). Devuelve (red, info); info es None si salió
    de la cache. Con `config`, la cache incluye el código entero (no sólo la percepción)."""
    from bioelastic import Cortex
    if config:
        import bioelastic, json as _json
        raiz = Path(bioelastic.__file__).resolve().parent
        hc = hashlib.sha1(_json.dumps(config, sort_keys=True).encode()
                          + b"".join(p.read_bytes() for p in sorted(raiz.glob("*.py")))).hexdigest()[:10]
        ruta = CACHE / f"mnist_depth2_c{hc}_s{semilla}_n{n}.npz"
    else:
        ruta = CACHE / f"mnist_depth2_{_huella_red()}_s{semilla}_n{n}.npz"
    if ruta.exists():
        return Cortex.load(ruta), None
    Xtr, ytr, _, _ = mnist()
    order = np.random.default_rng(semilla).permutation(len(Xtr) - 5000)[:n]
    E = encoder()
    net = Cortex((28, 28, 1), 10, seed=semilla, **DEPTH2, **(config or {}))
    t0 = time.time(); ok = 0
    for j, i in enumerate(order):
        ok += int(net.step(E(Xtr[i]), int(ytr[i])) == ytr[i])
        if log and (j + 1) % 5000 == 0:
            log(f"    mnist depth2 s{semilla}: {j + 1}/{n} ({time.time() - t0:.0f} s)")
    info = dict(imagenes=n, exactitud_en_linea=ok / n, segundos=time.time() - t0)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(ruta.stem + ".tmp.npz")         # con `metacog`, `save` guarda los calibradores
    net.save(tmp)
    tmp.replace(ruta)
    return net, info



def congelar(net):
    """Prepara la red para medir sin aprender (como exp_vision antes de evaluar)."""
    for a in net.areas:
        a._cache = None
    return net


def confianza(net) -> float:
    """Probabilidad de la decisión elegida, leída de la red sin tocarla. En v0.3-dev no hay
    una API pública de confianza (`Cortex.signals()` está en la rama integracion3); esto lee
    `Cortex._p()`, la misma distribución con la que la red explora."""
    return float(np.max(net._p()))


def log_stderr(m):
    print(m, file=sys.stderr, flush=True)
