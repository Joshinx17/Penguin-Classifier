import json
import os

import pytest

from penguin_classifier.inference import ArtifactError, load_classifier
from penguin_classifier.inference import predict


def test_missing_artifacts(tmp_path):
    with pytest.raises(ArtifactError, match="missing"):
        load_classifier(tmp_path)


def test_bad_metadata(tmp_path):
    (tmp_path / "penguin_image_classifier.keras").write_bytes(b"placeholder")
    (tmp_path / "metadata.json").write_text(json.dumps({"schema_version": 1, "class_names": []}))
    with pytest.raises(ArtifactError, match="Invalid"):
        load_classifier(tmp_path)


@pytest.mark.slow
def test_saved_model_inference_when_available():
    directory = os.environ.get("PENGUIN_TEST_MODEL_DIR")
    image = os.environ.get("PENGUIN_TEST_IMAGE")
    if not directory or not image:
        pytest.skip("Set PENGUIN_TEST_MODEL_DIR and PENGUIN_TEST_IMAGE to run integration test")
    model, metadata = load_classifier(directory)
    result = predict(model, metadata, image)
    assert set(result["scores"]) == set(metadata["class_names"])
    assert 0 <= result["confidence"] <= 1
