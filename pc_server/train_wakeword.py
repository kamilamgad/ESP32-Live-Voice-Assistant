from __future__ import annotations

import argparse
import json
import pickle
import wave
from pathlib import Path

import av
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_wakeword import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_SECONDS,
    extract_feature_vector,
    fit_audio_window,
    normalize_pcm,
    segment_samples,
)


def decode_audio(path: Path) -> np.ndarray:
    container = av.open(str(path))
    stream = next(s for s in container.streams if s.type == "audio")
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=DEFAULT_SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            chunk = resampled.to_ndarray()
            if chunk.ndim == 2:
                chunk = chunk[0]
            chunks.append(np.asarray(chunk, dtype=np.int16))
    flushed = resampler.resample(None)
    if flushed:
        for frame in flushed:
            chunk = frame.to_ndarray()
            if chunk.ndim == 2:
                chunk = chunk[0]
            chunks.append(np.asarray(chunk, dtype=np.int16))
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return normalize_pcm(np.concatenate(chunks).astype(np.int16, copy=False))


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(DEFAULT_SAMPLE_RATE)
        wav_file.writeframes(audio.astype(np.int16, copy=False).tobytes())


def convert_directory(source_dir: Path, output_dir: Path) -> list[Path]:
    converted: list[Path] = []
    for source_path in sorted(p for p in source_dir.iterdir() if p.is_file()):
        audio = decode_audio(source_path)
        if audio.size == 0:
            continue
        output_path = output_dir / f"{source_path.stem}.wav"
        write_wav(output_path, audio)
        converted.append(output_path)
    return converted


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        return np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)


def split_paths(paths: list[Path], test_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if len(paths) < 2:
        return paths[:], paths[:]
    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths))
    rng.shuffle(indices)
    test_count = max(1, int(round(len(paths) * test_ratio)))
    if test_count >= len(paths):
        test_count = len(paths) - 1
    test_indices = set(indices[:test_count].tolist())
    train = [path for idx, path in enumerate(paths) if idx not in test_indices]
    test = [path for idx, path in enumerate(paths) if idx in test_indices]
    return train, test


def augment_positive(samples: np.ndarray, rng: np.random.Generator, copies: int = 10) -> list[np.ndarray]:
    target_samples = int(DEFAULT_SAMPLE_RATE * DEFAULT_WINDOW_SECONDS)
    centered = fit_audio_window(samples, DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW_SECONDS)
    augmented = [centered]
    base = centered.astype(np.float32)
    for _ in range(copies):
        shift = int(rng.integers(-1200, 1200))
        shifted = np.roll(base, shift)
        gain = float(rng.uniform(0.85, 1.15))
        noise = rng.normal(0.0, 250.0, size=target_samples).astype(np.float32)
        mixed = np.clip(shifted * gain + noise, -32768, 32767).astype(np.int16)
        augmented.append(normalize_pcm(mixed))
    return augmented


def sample_negative_windows(samples: np.ndarray, rng: np.random.Generator, max_windows: int = 48) -> list[np.ndarray]:
    windows = segment_samples(samples, DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW_SECONDS, hop_seconds=0.4)
    energies = np.array([float(np.mean(np.abs(window))) for window in windows], dtype=np.float32)
    if windows and energies.max() > 0:
        order = np.argsort(-energies)
        selected = [windows[int(idx)] for idx in order[: max_windows // 2]]
        if len(windows) > len(selected):
            remaining = [windows[idx] for idx in range(len(windows)) if idx not in set(order[: max_windows // 2].tolist())]
            if remaining:
                rng.shuffle(remaining)
                selected.extend(remaining[: max_windows - len(selected)])
        return selected[:max_windows]
    return windows[:max_windows]


def mine_hard_negative_features(
    pipeline: Pipeline,
    negative_paths: list[Path],
    limit_per_file: int = 8,
) -> np.ndarray:
    mined: list[np.ndarray] = []
    for path in negative_paths:
        samples = load_wav(path)
        windows = sample_negative_windows(samples, np.random.default_rng(0), max_windows=96)
        if not windows:
            continue
        features = np.vstack([extract_feature_vector(window) for window in windows]).astype(np.float32)
        scores = pipeline.predict_proba(features)[:, 1]
        hard_indices = np.argsort(-scores)[:limit_per_file]
        for idx in hard_indices:
            mined.append(features[int(idx)])
    if not mined:
        return np.empty((0, 64 * 48 + 3), dtype=np.float32)
    return np.vstack(mined).astype(np.float32)


def build_training_examples(positive_paths: list[Path], negative_paths: list[Path], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[int] = []

    for path in positive_paths:
        samples = load_wav(path)
        for clip in augment_positive(samples, rng):
            features.append(extract_feature_vector(clip))
            labels.append(1)

    for path in negative_paths:
        samples = load_wav(path)
        for clip in sample_negative_windows(samples, rng):
            features.append(extract_feature_vector(clip))
            labels.append(0)

    silence = np.zeros(int(DEFAULT_SAMPLE_RATE * DEFAULT_WINDOW_SECONDS), dtype=np.int16)
    for _ in range(24):
        features.append(extract_feature_vector(silence))
        labels.append(0)

    return np.vstack(features).astype(np.float32), np.array(labels, dtype=np.int32)


def score_clip(pipeline: Pipeline, path: Path, label: int) -> dict:
    samples = load_wav(path)
    if label == 1:
        centered = fit_audio_window(samples, DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW_SECONDS)
        windows = [centered, np.roll(centered, 400), np.roll(centered, -400)]
    else:
        windows = segment_samples(samples, DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW_SECONDS, hop_seconds=0.2)
    features = np.vstack([extract_feature_vector(window) for window in windows]).astype(np.float32)
    probability = float(pipeline.predict_proba(features)[:, 1].max())
    return {"path": str(path), "label": label, "score": probability}


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> dict:
    best = {
        "threshold": 0.7,
        "balanced_accuracy": -1.0,
        "false_positives": 0,
        "false_negatives": 0,
    }
    for threshold in np.linspace(0.2, 0.95, 31):
        predictions = (scores >= threshold).astype(np.int32)
        balanced = float(balanced_accuracy_score(labels, predictions))
        false_positives = int(((predictions == 1) & (labels == 0)).sum())
        false_negatives = int(((predictions == 0) & (labels == 1)).sum())
        current = (balanced, -false_positives, -false_negatives)
        previous = (best["balanced_accuracy"], -best["false_positives"], -best["false_negatives"])
        if current > previous:
            best = {
                "threshold": float(threshold),
                "balanced_accuracy": balanced,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }
    return best


def choose_runtime_threshold(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    if negative_scores.size == 0:
        return 0.95
    safe_threshold = float(min(0.999, max(0.95, float(negative_scores.max()) + 0.001)))
    if positive_scores.size and safe_threshold > float(positive_scores.max()):
        return float(max(0.95, float(positive_scores.max()) - 0.001))
    return safe_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local wake-word classifier from positive/negative clips.")
    parser.add_argument("--positives", required=True, type=Path)
    parser.add_argument("--negatives", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "wakeword_dobby",
    )
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    prepared_dir = output_dir / "prepared"
    positives_wav = convert_directory(args.positives.resolve(), prepared_dir / "positives")
    negatives_wav = convert_directory(args.negatives.resolve(), prepared_dir / "negatives")

    if not positives_wav or not negatives_wav:
        raise SystemExit("Need at least one positive and one negative clip.")

    pos_train, pos_test = split_paths(positives_wav, args.test_ratio, args.seed)
    neg_train, neg_test = split_paths(negatives_wav, args.test_ratio, args.seed + 1)
    x_train, y_train = build_training_examples(pos_train, neg_train, args.seed)

    def build_pipeline() -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        C=3.0,
                        solver="liblinear",
                        random_state=args.seed,
                    ),
                ),
            ]
        )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    hard_negative_features = mine_hard_negative_features(pipeline, neg_train)
    if hard_negative_features.size:
        x_train = np.vstack([x_train, hard_negative_features])
        y_train = np.concatenate([y_train, np.zeros(hard_negative_features.shape[0], dtype=np.int32)])
        pipeline = build_pipeline()
        pipeline.fit(x_train, y_train)

    holdout_records = [score_clip(pipeline, path, 1) for path in pos_test]
    holdout_records.extend(score_clip(pipeline, path, 0) for path in neg_test)
    holdout_scores = np.array([record["score"] for record in holdout_records], dtype=np.float32)
    holdout_labels = np.array([record["label"] for record in holdout_records], dtype=np.int32)
    threshold_info = choose_threshold(holdout_scores, holdout_labels)

    final_x, final_y = build_training_examples(positives_wav, negatives_wav, args.seed)
    final_pipeline = build_pipeline()
    final_pipeline.fit(final_x, final_y)
    final_hard_negative_features = mine_hard_negative_features(final_pipeline, negatives_wav)
    if final_hard_negative_features.size:
        final_x = np.vstack([final_x, final_hard_negative_features])
        final_y = np.concatenate([final_y, np.zeros(final_hard_negative_features.shape[0], dtype=np.int32)])
        final_pipeline = build_pipeline()
        final_pipeline.fit(final_x, final_y)

    full_positive_records = [score_clip(final_pipeline, path, 1) for path in positives_wav]
    full_negative_records = [score_clip(final_pipeline, path, 0) for path in negatives_wav]
    runtime_threshold = choose_runtime_threshold(
        np.array([record["score"] for record in full_positive_records], dtype=np.float32),
        np.array([record["score"] for record in full_negative_records], dtype=np.float32),
    )

    model_path = output_dir / "dobby_wakeword.pkl"
    model_payload = {
        "pipeline": final_pipeline,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "window_seconds": DEFAULT_WINDOW_SECONDS,
        "threshold": runtime_threshold,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(pickle.dumps(model_payload))

    report = {
        "prepared_positive_count": len(positives_wav),
        "prepared_negative_count": len(negatives_wav),
        "train_positive_count": len(pos_train),
        "train_negative_count": len(neg_train),
        "test_positive_count": len(pos_test),
        "test_negative_count": len(neg_test),
        "training_examples": int(len(y_train)),
        "final_training_examples": int(len(final_y)),
        "holdout_auc": float(roc_auc_score(holdout_labels, holdout_scores)),
        "holdout": threshold_info,
        "holdout_scores": holdout_records,
        "runtime_threshold": runtime_threshold,
        "full_positive_scores": full_positive_records,
        "full_negative_scores": full_negative_records,
        "recommended_env": {
            "CUSTOM_WAKEWORD_MODEL_PATH": str(model_path),
            "CUSTOM_WAKEWORD_THRESHOLD": runtime_threshold,
        },
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["recommended_env"], indent=2))
    print(f"Report saved to {output_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()
