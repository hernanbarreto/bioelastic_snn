"""Metacognición: confianza calibrada, detección del propio error y sorpresa.

Piezas reutilizables (las usan `Cortex` y `Thinker` cuando se activan sus opciones, y
cualquier sistema que combine varias partes):

  Calibrator       regresión logística online que aprende P(acierto) a partir de señales
                   internas leídas antes del desenlace (confianza, conflicto, ...). Es la
                   confianza calibrada: lo que la red cree de sí misma, ajustado a cuán a
                   menudo acierta cuando cree eso.
  SurpriseTracker  incertidumbre esperada e inesperada (Yu y Dayan, 2005). La sorpresa de
                   cada desenlace (-log P(desenlace) según la propia confianza) se promedia
                   lento: ACh, el ruido de fondo que ya se espera; y rápido: si el promedio
                   rápido excede al lento en más de lo que un promedio rápido suele variar
                   (desvío de la sorpresa * sqrt(fast / (2 - fast))), NE: un cambio en el
                   mundo que la incertidumbre esperada no explica.
  auroc, ece       cuán bien una señal predice el propio error (AUROC: 0.5 = azar) y cuán
                   calibrada está una confianza (ECE y curva de confiabilidad).
"""
from __future__ import annotations

import numpy as np


def logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


class Calibrator:
    """P(acierto | señales) por regresión logística online (SGD normalizado sobre la
    log-verosimilitud: el paso se divide por 1 + |f|^2, así un logit grande no lo dispara).

    `features(...)` arma el vector [1, logit(confianza), señales extra...]; los pesos
    empiezan en la identidad (P = confianza cruda) y se ajustan con cada desenlace."""

    def __init__(self, n_extra: int = 0, lr: float = 0.05):
        self.w = np.zeros(2 + int(n_extra)); self.w[1] = 1.0
        self.lr = float(lr)
        self.n = 0

    def features(self, confidence: float, *extra) -> np.ndarray:
        return np.array([1.0, logit(confidence), *[float(x) for x in extra]])

    def predict(self, f: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-float(self.w @ f))))

    def update(self, f: np.ndarray, correct: bool) -> float:
        """Aprende del desenlace; devuelve la predicción que tenía antes de verlo."""
        p = self.predict(f)
        self.w += self.lr * (float(correct) - p) * f / (1.0 + float(f @ f))
        self.n += 1
        return p


class SurpriseTracker:
    """Sorpresa de los desenlaces: media lenta (incertidumbre esperada, ACh), su varianza y
    media rápida. `unexpected` (NE) = max(0, z) con z = (rápida - lenta) / (desvío lento *
    sqrt(fast / (2 - fast))): cuántos desvíos típicos de un promedio rápido se apartó de lo
    esperado. En cero durante `warmup` desenlaces (sin historia no hay «inesperado»)."""

    def __init__(self, slow: float = 0.002, fast: float = 0.05, warmup: int = 200, var0: float = 1.0):
        self.slow, self.fast = float(slow), float(fast)
        self.warmup = int(warmup)
        self.expected = 0.0; self.var = float(var0); self.recent = 0.0
        self.unexpected = 0.0; self.last = 0.0
        self.n = 0

    @staticmethod
    def surprise(confidence: float, correct: bool) -> float:
        c = min(max(float(confidence), 1e-4), 1.0 - 1e-4)
        return float(-np.log(c if correct else 1.0 - c))

    def update(self, s: float) -> float:
        """Registra una sorpresa y devuelve NE."""
        self.last = float(s)
        if self.n == 0:
            self.expected = self.recent = self.last
        self.n += 1
        d = self.last - self.expected
        self.expected += self.slow * d
        self.var += self.slow * (d * d - self.var)
        self.recent += self.fast * (self.last - self.recent)
        sd = float(np.sqrt(self.var + 1e-6) * np.sqrt(self.fast / (2.0 - self.fast)))
        ne = max(0.0, (self.recent - self.expected) / (sd + 1e-3))
        self.unexpected = 0.0 if self.n < self.warmup else ne
        return self.unexpected


def auroc(score, positive) -> float:
    """Área bajo la curva ROC de `score` para separar los `positive` (Mann-Whitney, con
    rango medio en los empates). nan si falta alguna de las dos clases."""
    score = np.asarray(score, dtype=np.float64); pos = np.asarray(positive, dtype=bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s)); r[order] = ranks
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ece(confidence, correct, bins: int = 10):
    """Error de calibración esperado (bins iguales en [0, 1]) y la curva de confiabilidad
    [(confianza media, acierto, n)] por bin no vacío."""
    conf = np.asarray(confidence, dtype=np.float64); ok = np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return float("nan"), []
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, int(bins) - 1)
    e = 0.0; curve = []
    for b in range(int(bins)):
        m = idx == b
        if m.any():
            e += m.mean() * abs(conf[m].mean() - ok[m].mean())
            curve.append((float(conf[m].mean()), float(ok[m].mean()), int(m.sum())))
    return float(e), curve
