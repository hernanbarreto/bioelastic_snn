import numpy as np

from bioelastic import AudioEncoder, ImageEncoder, LatencyEncoder, PopulationEncoder, TextEncoder


def test_image_encoder_range_and_channels():
    img = np.zeros((28, 28)); img[10:18, 12:16] = 200
    s = ImageEncoder((28, 28))(img)
    assert s.shape == (784,) and s.min() >= 0 and s.max() == 1.0
    assert (s > 0).sum() == 32
    onoff = ImageEncoder((28, 28), mode="onoff")
    s2 = onoff(img)
    assert s2.shape == (1568,) and onoff.n_inputs == 1568
    assert s2[:784].any() and s2[784:].any()           # ON adentro, OFF en el borde


def test_audio_encoder_is_tonotopic_and_normalizes_length():
    sr = 8000; t = np.arange(sr // 2) / sr
    enc = AudioEncoder(sr=sr, n_bands=24)
    peaks = [int(enc(np.sin(2 * np.pi * f * t)).mean(axis=0).argmax()) for f in (300, 800, 1600, 3000)]
    assert peaks == sorted(peaks) and len(set(peaks)) == 4
    fr = enc(np.sin(2 * np.pi * 500 * t))
    assert fr.ndim == 2 and fr.shape[1] == 24 and 0 <= fr.min() and fr.max() <= 1
    assert AudioEncoder(sr=sr, n_bands=24, n_frames=16)(np.sin(2 * np.pi * 500 * t)).shape == (16, 24)


def test_audio_encoder_trims_silence():
    sr = 8000; t = np.arange(sr // 4) / sr
    tone = np.sin(2 * np.pi * 600 * t)
    padded = np.concatenate([np.zeros(sr // 2), tone, np.zeros(sr // 2)]) + 1e-5
    enc = AudioEncoder(sr=sr)
    assert abs(len(enc(padded)) - len(enc(tone))) <= 4


def test_text_encoder_is_byte_level_and_language_free():
    b = TextEncoder()("señal Ω")
    assert b.dtype == np.uint8 and len(b) == len("señal Ω".encode("utf-8")) and b.max() < 256


def test_population_encoder_peak_tracks_value():
    enc = PopulationEncoder(n_features=2, n_cells=8, lo=[0, -10], hi=[1, 10])
    s = enc([0.0, 10.0]).reshape(2, 8)
    assert s[0].argmax() == 0 and s[1].argmax() == 7
    assert enc.n_inputs == 16


def test_latency_encoder_noise_model():
    x = np.full(1000, 0.5, dtype=np.float32)
    s = LatencyEncoder(1000, jitter=0.05, drop_p=0.1, seed=0)(x)
    assert 0.05 < np.mean(s == 0) < 0.15
    assert 0.03 < s[s > 0].std() < 0.07
    assert np.array_equal(LatencyEncoder(1000)(x), x)
