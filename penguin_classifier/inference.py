"""Validated model loading and a single prediction path for CLI and Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from penguin_classifier import CLASS_NAMES, METADATA_NAME, MODEL_NAME
from penguin_classifier.preprocessing import prepare_image


class ArtifactError(ValueError):
    """Saved model and metadata are missing or incompatible."""


def load_classifier(directory: str | Path):
    import tensorflow as tf

    directory = Path(directory)
    model_path, metadata_path = directory / MODEL_NAME, directory / METADATA_NAME
    if not model_path.is_file() or not metadata_path.is_file():
        raise ArtifactError(
            f"Model or metadata missing in {directory}. Run train_image_classifier.py first."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        names = metadata["class_names"]
        size = metadata["image_size"]
        if (
            metadata["schema_version"] != 1
            or names != list(CLASS_NAMES)
            or len(names) != len(set(names))
            or not all(isinstance(n, str) and n for n in names)
            or not isinstance(size, list)
            or len(size) != 2
            or any(not isinstance(n, int) or n < 32 for n in size)
            or metadata["preprocessing"] != "rgb_float32_0_255; in_model_rescale_to_-1_1"
        ):
            raise ValueError("Invalid metadata schema or class mapping")
        model = tf.keras.models.load_model(model_path, compile=False)
        if tuple(model.input_shape[1:]) != (*size, 3) or model.output_shape[-1] != len(names):
            raise ValueError("Model shape does not match metadata")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ArtifactError(f"Invalid model artifacts: {exc}") from exc
    return model, metadata


def predict(model, metadata: dict, source: str | Path | bytes) -> dict:
    batch = prepare_image(source, tuple(metadata["image_size"]))
    scores = np.asarray(model({"rgb_pixels_0_255": batch}, training=False))[0]
    names = metadata["class_names"]
    if (
        scores.shape != (len(names),)
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or not np.isclose(scores.sum(), 1, atol=1e-3)
    ):
        raise ArtifactError("Model returned invalid class probabilities")
    index = int(np.argmax(scores))
    return {
        "prediction": names[index],
        "confidence": float(scores[index]),
        "scores": {name: float(score) for name, score in zip(names, scores)},
    }
