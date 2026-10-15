from pathlib import Path

from app import artifact_status


def test_missing_model_message(tmp_path: Path):
    model, metadata, error = artifact_status(tmp_path)
    assert model is None and metadata is None
    assert "train_image_classifier.py" in error
