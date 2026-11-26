"""Carga de datos para los experimentos (descarga a ./data la primera vez)."""
from __future__ import annotations

import gzip
import subprocess
import urllib.request
import wave
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"

_MNIST = "https://raw.githubusercontent.com/fgnt/mnist/master/"
_SHAKESPEARE = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
                "data/tinyshakespeare/input.txt")
_FSDD = "https://github.com/Jakobovski/free-spoken-digit-dataset.git"


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load_mnist():
    """(Xtr, ytr, Xte, yte) con X en uint8 (N, 28, 28)."""
    def idx(name, offset, shape):
        raw = gzip.open(_fetch(_MNIST + name, DATA / "mnist" / name)).read()
        return np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape(shape)
    return (idx("train-images-idx3-ubyte.gz", 16, (-1, 28, 28)),
            idx("train-labels-idx1-ubyte.gz", 8, (-1,)),
            idx("t10k-images-idx3-ubyte.gz", 16, (-1, 28, 28)),
            idx("t10k-labels-idx1-ubyte.gz", 8, (-1,)))


def load_text() -> bytes:
    return _fetch(_SHAKESPEARE, DATA / "tinyshakespeare.txt").read_bytes()


def load_fsdd():
    """Free Spoken Digit Dataset. Devuelve lista de (onda float32, dígito, locutor, índice)
    y la frecuencia de muestreo."""
    root = DATA / "fsdd"
    if not (root / "recordings").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", _FSDD, str(root)], check=True)
    out, sr = [], 8000
    for p in sorted((root / "recordings").glob("*.wav")):
        digit, speaker, k = p.stem.split("_")
        with wave.open(str(p), "rb") as w:
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            if w.getnchannels() > 1:
                pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
        out.append((pcm.astype(np.float32) / 32768.0, int(digit), speaker, int(k)))
    return out, sr


def bionet_stream(n_features=256, n_classes=10, k_on=40, sep=0.6, noise=0.20,
                  distractor_ratio=0.35, seed=11):
    """Generador sintético idéntico al de bioNet_online_B (misma semilla, mismo orden)."""
    rng = np.random.default_rng(seed)
    stride = max(1, (n_features - k_on) // n_classes)
    protos = np.zeros((n_classes, n_features), dtype=np.float32)
    for c in range(n_classes):
        protos[c, np.arange(c * stride, c * stride + k_on) % n_features] = 1.0
    while True:
        y = int(rng.integers(0, n_classes))
        x = sep * protos[y] + (1.0 - sep) * protos[int(rng.integers(0, n_classes))]
        n_dist = int(distractor_ratio * k_on)
        if n_dist > 0:
            idxd = rng.integers(0, n_features, size=n_dist)
            x[idxd] = np.maximum(x[idxd], 0.3)
        x = np.clip(x + noise * rng.random(n_features).astype(np.float32), 0.0, 1.0)
        yield x.astype(np.float32), y
