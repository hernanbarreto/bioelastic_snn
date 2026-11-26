"""Perturbaciones que nunca aparecen en el entrenamiento (pruebas para romper la red).

  shift3   la imagen entera corrida 3 píxeles en una de 8 direcciones (relleno con 0)
  noise    ruido gaussiano aditivo (sigma = 0.2 del rango) en todos los píxeles
  occlude  un cuadrado de 8x8 borrado en una posición al azar del centro 20x20

Semilla fija: todas las redes ven exactamente las mismas imágenes perturbadas.
"""
from __future__ import annotations

import numpy as np

PERTURBATIONS = ("shift3", "noise", "occlude")


def _shift(img, dy, dx):
    out = np.zeros_like(img)
    H, W = img.shape[:2]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = img[ys, xs]
    return out


def perturb(X: np.ndarray, kind: str, seed: int = 1234) -> np.ndarray:
    """X uint8 (N, H, W[, C]) -> X perturbada (mismo tipo y forma)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    if kind == "shift3":
        dirs = [(dy, dx) for dy in (-3, 0, 3) for dx in (-3, 0, 3) if (dy, dx) != (0, 0)]
        pick = rng.integers(len(dirs), size=len(X))
        return np.stack([_shift(x, *dirs[d]) for x, d in zip(X, pick)])
    if kind == "noise":
        Y = X.astype(np.float32) / 255.0 + rng.normal(0.0, 0.2, X.shape).astype(np.float32)
        return (np.clip(Y, 0.0, 1.0) * 255).astype(np.uint8)
    if kind == "occlude":
        Y = X.copy()
        side = X.shape[1]; lo = (side - 20) // 2
        pos = rng.integers(lo, lo + 20 - 8 + 1, size=(len(X), 2))
        for y, (r, c) in zip(Y, pos):
            y[r:r + 8, c:c + 8] = 0
        return Y
    raise ValueError(kind)
