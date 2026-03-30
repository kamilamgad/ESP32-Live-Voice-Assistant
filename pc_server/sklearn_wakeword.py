from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import stft


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_WINDOW_SECONDS = 1.2
DEFAULT_THRESHOLD = 0.7


def normalize_pcm(samples: np.ndarray, target_peak: int = 28000) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.int16)
    if samples.size == 0:
        return samples
    peak = int(np.max(np.abs(samples)))
    if peak <= 0:
        return samples
    gain = min(target_peak / peak, 8.0)
    scaled = np.clip(samples.astype(np.float32) * gain, -32768, 32767)
    return scaled.astype(np.int16)


def fit_audio_window(samples: np.ndarray, sample_rate: int, window_seconds: float) -> np.ndarray:
    target_samples = int(sample_rate * window_seconds)
    if target_samples <= 0:
        raise ValueError("window_seconds must be positive")
    samples = np.asarray(samples, dtype=np.int16)
    if samples.size == target_samples:
        return samples
    if samples.size > target_samples:
        start = max((samples.size - target_samples) // 2, 0)
        return samples[start : start + target_samples]
    padded = np.zeros(target_samples, dtype=np.int16)
    start = (target_samples - samples.size) // 2
    padded[start : start + samples.size] = samples
    return padded


def resize_2d(values: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    if values.shape == (target_rows, target_cols):
        return values
    row_positions = np.linspace(0, values.shape[0] - 1, target_rows)
    lower_rows = np.floor(row_positions).astype(np.int32)
    upper_rows = np.ceil(row_positions).astype(np.int32)
    row_weights = (row_positions - lower_rows).astype(np.float32)
    resized_rows = np.vstack(
        [
            values[lower] * (1.0 - weight) + values[upper] * weight
            for lower, upper, weight in zip(lower_rows, upper_rows, row_weights)
        ]
    )
    col_positions = np.linspace(0, resized_rows.shape[1] - 1, target_cols)
    resized = np.vstack(
        [np.interp(col_positions, np.arange(resized_rows.shape[1]), row) for row in resized_rows]
    )
    return resized


def extract_feature_vector(
    samples: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> np.ndarray:
    window = fit_audio_window(normalize_pcm(samples), sample_rate, window_seconds).astype(np.float32) / 32768.0
    _, _, spectrum = stft(
        window,
        fs=sample_rate,
        nperseg=400,
        noverlap=240,
        nfft=512,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(spectrum)
    usable_bins = min(128, magnitude.shape[0])
    log_spec = np.log1p(magnitude[:usable_bins, :])
    resized = resize_2d(log_spec, 64, 48)
    energy = np.array(
        [
            float(np.mean(np.abs(window))),
            float(np.std(window)),
            float(np.max(np.abs(window))),
        ],
        dtype=np.float32,
    )
    return np.concatenate([resized.astype(np.float32).ravel(), energy], axis=0)


def segment_samples(samples: np.ndarray, sample_rate: int, window_seconds: float, hop_seconds: float) -> list[np.ndarray]:
    window_samples = int(sample_rate * window_seconds)
    hop_samples = max(int(sample_rate * hop_seconds), 1)
    samples = np.asarray(samples, dtype=np.int16)
    if samples.size <= window_samples:
        return [fit_audio_window(samples, sample_rate, window_seconds)]
    windows: list[np.ndarray] = []
    for start in range(0, samples.size - window_samples + 1, hop_samples):
        windows.append(samples[start : start + window_samples])
    if not windows:
        windows.append(fit_audio_window(samples, sample_rate, window_seconds))
    return windows


class SklearnWakeWordModel:
    def __init__(self, model_path: str | Path) -> None:
        model_file = Path(model_path)
        payload: dict[str, Any] = pickle.loads(model_file.read_bytes())
        self.pipeline = payload["pipeline"]
        self.sample_rate = int(payload.get("sample_rate", DEFAULT_SAMPLE_RATE))
        self.window_seconds = float(payload.get("window_seconds", DEFAULT_WINDOW_SECONDS))
        self.threshold = float(payload.get("threshold", DEFAULT_THRESHOLD))

    def score(self, pcm_bytes: bytes) -> float:
        if not pcm_bytes:
            return 0.0
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        if samples.size == 0:
            return 0.0
        windows = segment_samples(samples, self.sample_rate, self.window_seconds, hop_seconds=0.2)
        features = np.vstack(
            [extract_feature_vector(window, sample_rate=self.sample_rate, window_seconds=self.window_seconds) for window in windows]
        )
        return float(self.pipeline.predict_proba(features)[:, 1].max())
