"""Audio I/O and noise mixing (16 kHz mono float32)."""
from __future__ import annotations

import numpy as np

SR = 16000


def load_audio(path, sr: int = SR) -> np.ndarray:
    import soundfile as sf

    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        import torch
        import torchaudio.functional as AF

        wav = AF.resample(torch.from_numpy(wav), file_sr, sr).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


def power(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) ** 2) + 1e-12)


def fit_length(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(noise) >= n:
        start = int(rng.integers(0, len(noise) - n + 1))
        return noise[start : start + n]
    reps = int(np.ceil(n / max(len(noise), 1)))
    return np.tile(noise, reps)[:n]


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add noise so that 10*log10(P_signal / P_noise) == snr_db."""
    noise = fit_length(noise, len(signal), rng)
    scale = np.sqrt(power(signal) / (power(noise) * 10 ** (snr_db / 10)))
    out = signal + scale * noise
    peak = np.abs(out).max()
    if peak > 0.999:  # avoid clipping; keeps the SNR unchanged
        out = out / peak * 0.999
    return out.astype(np.float32)


def make_babble(utterances: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """Sum several real speech utterances (equal power) into babble noise."""
    acc = np.zeros(n, dtype=np.float64)
    for u in utterances:
        u = fit_length(u, n, rng)
        acc += u / np.sqrt(power(u))
    return acc.astype(np.float32)
