"""Metacognición: sensibilidad de la confianza a los propios aciertos con dificultad igualada.

Protocolo publicado (Fleming et al. 2010, texto y material suplementario): 32 personas;
decisión perceptiva de dos alternativas (en cuál de dos intervalos apareció el parche de
mayor contraste); una escalera de 1 arriba y 2 abajo sobre el contraste (pasos de 3 %)
mantiene el acierto cerca de 71 %; 600 ensayos en 6 bloques de 100 (se analizan los
bloques 2 a 6); confianza de 1 a 6 después de cada decisión. Sensibilidad metacognitiva:
área bajo la curva ROC de tipo 2 (la confianza separa aciertos de errores). Rango entre
personas: 0,55 a 0,75 (no se publicó la media).

Lo que hace la batería: la red `depth2` entrenada en MNIST (ver `percepcion`) clasifica
dígitos del test con el contraste degradado y ruido gaussiano fijo; una escalera de 1
arriba y 2 abajo sobre el contraste (pasos de 0,03) iguala la dificultad; 600 ensayos en 6
bloques, sin aprendizaje; la confianza es la probabilidad de la decisión (`Cortex._p()`).
AUROC de tipo 2 en los bloques 2 a 6, con la confianza continua y discretizada en 6
niveles (cortes en los sextiles del bloque 1).

Firmas de ratas (Kepecs et al. 2008; Lak et al. 2014), medidas con estímulos constantes
en 5 niveles de contraste alrededor del umbral: la confianza de los aciertos crece con la
facilidad y la de los errores baja (relación opuesta), y la exactitud crece con la
confianza de forma monótona.
"""
from __future__ import annotations

import time

import numpy as np

from .base import NoAplicable, Prueba
from . import percepcion

RUIDO = 0.3
PASO = 0.03
N_ENSAYOS, N_BLOQUE = 600, 100
NIVELES = (0.5, 0.75, 1.0, 1.5, 2.0)          # múltiplos del contraste final de la escalera
N_NIVEL = 200


def estimulo(x: np.ndarray, c: float, rng) -> np.ndarray:
    s = np.clip(c * (x.astype(np.float32) / 255.0) + rng.normal(0.0, RUIDO, x.shape), 0.0, 1.0)
    return np.round(255.0 * s).astype(np.uint8)


def auroc(conf: np.ndarray, ok: np.ndarray) -> float | None:
    """Área bajo la ROC de tipo 2: P(confianza de un acierto > la de un error), empates 1/2."""
    a = conf[ok]; e = conf[~ok]
    if len(a) == 0 or len(e) == 0:
        return None
    from scipy.stats import rankdata
    r = rankdata(np.r_[a, e])
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(e)))


def _conf(net, config) -> float:
    """Con un perfil de opciones (p. ej. `metacog`), la confianza pública de la red
    (`Cortex.confidence()`: calibrada si hay `metacog`); si no, la probabilidad de la decisión."""
    if config:
        return float(net.confidence())
    return percepcion.confianza(net)


def correr(semilla: int, escala: float = 1.0, config: dict | None = None, validacion: bool = False) -> dict:
    t0 = time.time()
    n = percepcion.N_MNIST if escala >= 1.0 else max(1000, int(percepcion.N_MNIST * escala))
    try:
        net, info = percepcion.cortex_mnist(semilla, n, log=percepcion.log_stderr, config=config)
    except TypeError as e:
        raise NoAplicable(f"el Cortex de este commit no acepta {sorted(config or {})}: {e}") from e
    net = percepcion.congelar(net)
    Xtr, ytr, Xte, yte = percepcion.mnist()
    if validacion:                                  # las últimas 5 000 de entrenamiento, nunca el test
        Xte, yte = Xtr[len(Xtr) - 5000:], ytr[len(ytr) - 5000:]
    E = percepcion.encoder()
    rng = np.random.default_rng((40_000 if validacion else 30_000) + semilla)
    n_tr = N_ENSAYOS if escala >= 1.0 else 120
    c, racha = 1.0, 0
    conf, ok, cs = [], [], []
    for t in range(n_tr):
        i = int(rng.integers(len(Xte)))
        pred = net.forward(E(estimulo(Xte[i], c, rng)))
        hit = bool(pred == yte[i])
        conf.append(_conf(net, config)); ok.append(hit); cs.append(c)
        if hit:
            racha += 1
            if racha == 2:
                c, racha = max(PASO, c - PASO), 0
        else:
            c, racha = min(1.0, c + PASO), 0
    conf = np.array(conf); ok = np.array(ok); cs = np.array(cs)
    nb = N_BLOQUE if escala >= 1.0 else 20
    an = slice(nb, None)
    cortes = np.quantile(conf[:nb], np.linspace(0, 1, 7)[1:-1])
    nivel6 = np.digitize(conf, cortes)
    med = dict(auroc2=auroc(conf[an], ok[an]), auroc2_6niveles=auroc(nivel6[an].astype(float), ok[an]),
               exactitud_bloques2a6=float(ok[an].mean()), contraste_final=float(np.mean(cs[-nb:])),
               confianza_media=float(conf[an].mean()))
    # firmas con estímulos constantes
    c0 = med["contraste_final"]
    cc, ce, todos_c, todos_ok = [], [], [], []
    n_niv = N_NIVEL if escala >= 1.0 else 30
    for m in NIVELES:
        cn = float(np.clip(c0 * m, PASO, 1.0))
        cf, hs = [], []
        for _ in range(n_niv):
            i = int(rng.integers(len(Xte)))
            hs.append(bool(net.forward(E(estimulo(Xte[i], cn, rng))) == yte[i]))
            cf.append(_conf(net, config))
        cf = np.array(cf); hs = np.array(hs)
        cc.append(float(cf[hs].mean()) if hs.any() else None)
        ce.append(float(cf[~hs].mean()) if (~hs).any() else None)
        todos_c += list(cf); todos_ok += list(hs)
        med[f"exactitud_nivel_{m}"] = float(hs.mean())
    lv = np.arange(len(NIVELES), dtype=float)

    def pend(v):
        k = [i for i, x in enumerate(v) if x is not None]
        return float(np.polyfit(lv[k], [v[i] for i in k], 1)[0]) if len(k) >= 2 else None
    pc, pe = pend(cc), pend(ce)
    tc = np.array(todos_c); to = np.array(todos_ok)
    q = np.quantile(tc, [0.2, 0.4, 0.6, 0.8])
    quint = np.digitize(tc, q)
    acc_q = [float(to[quint == j].mean()) if (quint == j).any() else None for j in range(5)]
    from scipy.stats import spearmanr
    k = [j for j in range(5) if acc_q[j] is not None]
    rho = float(spearmanr(k, [acc_q[j] for j in k])[0]) if len(k) >= 3 else None
    med.update(pendiente_confianza_aciertos=pc, pendiente_confianza_errores=pe,
               firma_opuesta=bool(pc is not None and pe is not None and pc > 0 and pe < 0),
               exactitud_por_quintil_de_confianza=acc_q, rho_confianza_exactitud=rho,
               firma_calibracion=bool(rho is not None and rho >= 0.9 and acc_q[-1] > acc_q[0]),
               confianza_aciertos_por_nivel=cc, confianza_errores_por_nivel=ce)
    if info:
        med.update(mnist_imagenes=float(info["imagenes"]), mnist_exactitud_en_linea=info["exactitud_en_linea"])
    return dict(puntaje=med["auroc2"], medidas=med, segundos=time.time() - t0)


MC = dict(lr=0.05)                     # la de integracion3 (exp_metacog.py)
CONFIGS_METACOG = {"cal": dict(metacog=MC), "ne": dict(metacog=MC, neuromod=dict(ne_gain=1.0, ach_gain=0.0))}


def validar(config: dict, semilla: int, escala: float = 1.0) -> float:
    """Criterio de validación: AUROC de tipo 2 con el mismo protocolo sobre las imágenes de
    validación (las últimas 5 000 de entrenamiento), con semillas de validación."""
    return correr(semilla, escala, config, validacion=True)["medidas"]["auroc2"]


def correr_espera(semilla: int, escala: float = 1.0) -> dict:
    raise NoAplicable("la apuesta de Kepecs y Lak (cuánto tiempo espera la rata el premio antes de abandonar) necesita "
                      "una acción de esperar o abandonar y una escala de tiempo: en v0.3-dev la percepción decide en un "
                      "paso y no tiene ninguna de las dos; las firmas se miden con la probabilidad de la decisión (variante "
                      "«imagenes») y la rama integracion3 mide una apuesta con su API nueva")


PRUEBAS = [
    Prueba(
        id="auroc_tipo2", variante="imagenes", condicion="imagenes",
        metrica="AUROC de tipo 2 (confianza continua) en los bloques 2 a 6, con acierto igualado cerca de 71 %",
        protocolo=(f"depth2 entrenada con {percepcion.N_MNIST} imágenes de MNIST; 600 ensayos con dígitos del test, contraste "
                   f"por escalera 1 arriba 2 abajo (paso {PASO}) y ruido gaussiano fijo (desvío {RUIDO}); sin aprendizaje; "
                   "confianza = probabilidad de la decisión"),
        desviaciones=[
            "tarea: clasificar un dígito entre 10 (azar 10 %), no elegir entre dos intervalos (azar 50 %); la escalera "
            "iguala el acierto en ~71 % igual que en el original",
            "la confianza no es un juicio aparte: es la probabilidad de la decisión de la red (`Cortex._p()`, lectura "
            "interna; la API pública `signals()` está en integracion3)",
            f"la red se entrenó con {percepcion.N_MNIST} imágenes de MNIST (la configuración publicada usa 55 000)",
            "la referencia humana es un rango entre sujetos: el índice se informa contra los dos extremos",
        ],
        correr=correr, version=1, semillas=(0, 1, 2),
        equivalente=True,
        motivo_equivalencia="decisión perceptiva sobre imágenes con dificultad igualada; cambia la tarea (10 clases)",
        firmas={"firma_opuesta": "confianza de aciertos que sube y de errores que baja con la facilidad (ratas)",
                "firma_calibracion": "exactitud que crece con la confianza (ratas)"},
    ),
    Prueba(
        id="auroc_tipo2", variante="apuesta_espera", condicion="imagenes",
        metrica="firmas de la confianza en el tiempo de espera (ratas)",
        protocolo="el de Kepecs et al. (2008) y Lak et al. (2014)", desviaciones=[], correr=correr_espera, version=1,
        semillas=(0,), equivalente=False, motivo_equivalencia="no se pudo correr", referencia=None,
    ),
    Prueba(
        id="auroc_tipo2", variante="imagenes+metacog", condicion="imagenes",
        metrica="AUROC de tipo 2 con la confianza calibrada de la red (`metacog`), bloques 2 a 6, acierto igualado",
        protocolo=("el mismo que «imagenes», con depth2 + `metacog` (calibrador en línea de integracion3) entrenada con "
                   f"{percepcion.N_MNIST} imágenes; la confianza es `Cortex.confidence()`; variante elegida en validación "
                   "(cal: sólo el calibrador; ne: + neuromodulación NE)"),
        desviaciones=[
            "tarea: clasificar un dígito entre 10, no elegir entre dos intervalos; la escalera iguala el acierto en ~71 %",
            "el calibrador aprende durante el entrenamiento y queda quieto en la prueba (sin respuesta del resultado)",
            f"la red se entrenó con {percepcion.N_MNIST} imágenes de MNIST (la configuración publicada usa 55 000)",
            "la referencia humana es un rango entre sujetos: el índice se informa contra los dos extremos",
            "perfil metacog: opciones de integracion3, fusionadas en v0.3-dev 1b23db0 (apagadas por defecto)",
        ],
        correr=correr, version=1, semillas=(0, 1, 2), perfil="metacog",
        configs=CONFIGS_METACOG, validar=validar, semillas_validacion=(100, 101),
        criterio_validacion="AUROC de tipo 2 en imágenes de validación (semillas 100 y 101)",
        equivalente=True,
        motivo_equivalencia="decisión perceptiva sobre imágenes con dificultad igualada; cambia la tarea (10 clases)",
        firmas={"firma_opuesta": "confianza de aciertos que sube y de errores que baja con la facilidad (ratas)",
                "firma_calibracion": "exactitud que crece con la confianza (ratas)"},
    ),
]
