"""Encoders: traducen cada modalidad al formato común (voleas de latencia).

Todos devuelven fuerzas en [0, 1] (0 = sin spike, spike en t = 1 - fuerza).
La red no sabe de dónde viene la volea.
"""
from __future__ import annotations

import numpy as np


class ImageEncoder:
    """Retina. mode="intensity": latencia = brillo. mode="onoff": centro-periferia
    (diferencia de gaussianas) separada en canales ON y OFF."""

    def __init__(self, shape, mode: str = "intensity", threshold: float = 0.1,
                 sigma: float = 1.0, blur: float = 0.0):
        self.shape = tuple(shape)
        self.mode = mode
        self.threshold = float(threshold)
        self.sigma = float(sigma)
        self.blur = float(blur)
        n = int(np.prod(self.shape))
        self.n_inputs = n if mode == "intensity" else 2 * n

    def __call__(self, img) -> np.ndarray:
        x = np.asarray(img, dtype=np.float32).reshape(self.shape)
        mx = float(x.max())
        if mx > 0:
            x = x / mx
        if self.blur > 0 or self.mode == "onoff":
            from scipy.ndimage import gaussian_filter
        if self.blur > 0:
            x = gaussian_filter(x, self.blur)
            x = x / max(float(x.max()), 1e-8)
        if self.mode == "intensity":
            s = x.ravel().copy()
        else:
            dog = gaussian_filter(x, self.sigma) - gaussian_filter(x, 2.0 * self.sigma)
            on = np.maximum(dog, 0.0); off = np.maximum(-dog, 0.0)
            s = np.concatenate([on.ravel(), off.ravel()])
            s = s / max(float(s.max()), 1e-8)
        s[s < self.threshold] = 0.0
        return s


class AudioEncoder:
    """Cóclea: banco de filtros mel sobre STFT, compresión logarítmica y latencia.

    Devuelve (F, n_bands). Con n_frames fijo, normaliza la duración del segmento.

    Entradas degeneradas: las muestras no finitas (NaN, inf) se toman como silencio; una
    entrada vacía, o sin potencia por encima del piso numérico (`silence_power`, muy por
    debajo de un bit de audio de 16 bits), es silencio y devuelve ceros (sin spikes). La
    normalización es relativa al máximo de cada segmento, así que sin ese piso el silencio
    total se codificaría como la máxima actividad en todas las bandas.
    """

    def __init__(self, sr: int = 8000, n_bands: int = 32, frame_ms: float = 25.0,
                 hop_ms: float = 10.0, fmin: float = 80.0, fmax: float | None = None,
                 dyn_range_db: float = 50.0, n_frames: int | None = None,
                 trim_db: float = 30.0, silence_power: float = 1e-10):
        self.sr = int(sr)
        self.n_bands = int(n_bands)
        self.win = int(round(sr * frame_ms / 1000.0))
        self.hop = int(round(sr * hop_ms / 1000.0))
        self.n_fft = 1 << int(np.ceil(np.log2(self.win)))
        self.dyn = float(dyn_range_db)
        self.n_frames = n_frames
        self.trim_db = float(trim_db)
        self.silence_power = float(silence_power)
        self.n_inputs = self.n_bands
        self.fb = self._mel_filterbank(fmin, fmax or sr / 2.0)
        self.window = np.hanning(self.win).astype(np.float32)

    def _mel_filterbank(self, fmin, fmax):
        mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
        imel = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        pts = imel(np.linspace(mel(fmin), mel(fmax), self.n_bands + 2))
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sr)
        fb = np.zeros((self.n_bands, len(freqs)), dtype=np.float32)
        for b in range(self.n_bands):
            lo, ce, hi = pts[b], pts[b + 1], pts[b + 2]
            up = (freqs - lo) / max(ce - lo, 1e-6)
            down = (hi - freqs) / max(hi - ce, 1e-6)
            fb[b] = np.clip(np.minimum(up, down), 0.0, None)
        return fb

    def __call__(self, wave) -> np.ndarray:
        x = np.asarray(wave, dtype=np.float32).reshape(-1)
        if len(x) == 0:
            return np.zeros((self.n_frames or 1, self.n_bands), dtype=np.float32)
        finite = np.isfinite(x)
        if not finite.all():
            x = np.where(finite, x, np.float32(0.0))
        x = x - x.mean()
        if len(x) < self.win:
            x = np.pad(x, (0, self.win - len(x)))
        n = 1 + (len(x) - self.win) // self.hop
        ix = np.arange(self.win)[None, :] + self.hop * np.arange(n)[:, None]
        spec = np.abs(np.fft.rfft(x[ix] * self.window, n=self.n_fft, axis=1)) ** 2
        power = spec @ self.fb.T
        if not float(power.max()) > self.silence_power:               # silencio: ningún spike
            return np.zeros((self.n_frames or n, self.n_bands), dtype=np.float32)
        logmel = 10.0 * np.log10(power + 1e-10)                      # (F, bands) en dB
        energy = logmel.max(axis=1)
        live = np.flatnonzero(energy > energy.max() - self.trim_db)  # recorta silencio
        logmel = logmel[live[0]:live[-1] + 1]
        if self.n_frames is not None:                                # normaliza duración
            src = np.linspace(0, len(logmel) - 1, self.n_frames)
            lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, len(logmel) - 1)
            a = (src - lo)[:, None]
            logmel = (1 - a) * logmel[lo] + a * logmel[hi]
        s = (logmel - (logmel.max() - self.dyn)) / self.dyn          # rango dinámico -> [0, 1]
        return np.clip(s, 0.0, 1.0).astype(np.float32)


class TextEncoder:
    """Texto como flujo de bytes UTF-8: 256 aferentes, un spike por símbolo.
    Sirve para cualquier idioma sin vocabulario previo."""

    n_inputs = 256

    def __call__(self, text) -> np.ndarray:
        if isinstance(text, str):
            text = text.encode("utf-8")
        return np.frombuffer(text, dtype=np.uint8)


class PopulationEncoder:
    """Sensores escalares (IoT, telemetría): cada magnitud excita `n_cells` neuronas
    con campos receptivos gaussianos solapados."""

    def __init__(self, n_features: int, n_cells: int = 8, lo=0.0, hi=1.0,
                 threshold: float = 0.05):
        self.F = int(n_features); self.P = int(n_cells)
        self.lo = np.broadcast_to(np.asarray(lo, dtype=np.float32), (self.F,))
        self.hi = np.broadcast_to(np.asarray(hi, dtype=np.float32), (self.F,))
        self.centers = np.linspace(0.0, 1.0, self.P, dtype=np.float32)
        self.sigma = 1.0 / (self.P - 1) if self.P > 1 else 1.0
        self.threshold = float(threshold)
        self.n_inputs = self.F * self.P

    def __call__(self, x) -> np.ndarray:
        u = (np.asarray(x, dtype=np.float32) - self.lo) / np.maximum(self.hi - self.lo, 1e-8)
        u = np.clip(u, 0.0, 1.0)
        s = np.exp(-0.5 * ((u[:, None] - self.centers[None, :]) / self.sigma) ** 2)
        s[s < self.threshold] = 0.0
        return s.ravel().astype(np.float32)


class LatencyEncoder:
    """Vector ya normalizado en [0, 1] -> latencia directa, con jitter y pérdida de
    spikes opcionales (mismo modelo de ruido que el bioNet original)."""

    def __init__(self, n_inputs: int, jitter: float = 0.0, drop_p: float = 0.0, seed: int = 0):
        self.n_inputs = int(n_inputs)
        self.jitter = float(jitter); self.drop_p = float(drop_p)
        self.rng = np.random.default_rng(seed)

    def __call__(self, x) -> np.ndarray:
        s = np.asarray(x, dtype=np.float32).copy()
        m = s > 0
        if self.jitter > 0:
            s[m] = np.clip(s[m] + self.rng.normal(0.0, self.jitter, int(m.sum())), 0.0, 1.0)
        if self.drop_p > 0:
            s[m & (self.rng.random(s.shape) < self.drop_p)] = 0.0
        return s
