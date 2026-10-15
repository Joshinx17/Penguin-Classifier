import numpy as np
import pytest


@pytest.mark.slow
def test_model_shape_frozen_and_fine_tuning():
    tf = pytest.importorskip("tensorflow")
    from penguin_classifier.model import build_model, enable_fine_tuning

    model, backbone = build_model(3, (96, 96), weights=None)
    assert not backbone.trainable
    scores = model(np.full((1, 96, 96, 3), 255, dtype=np.float32), training=False).numpy()
    assert scores.shape == (1, 3)
    assert np.isfinite(scores).all()
    assert np.isclose(scores.sum(), 1)
    assert any(isinstance(layer, tf.keras.layers.Rescaling) for layer in model.layers)
    normalizer = model.get_layer("mobilenet_normalization")
    endpoints = normalizer(tf.constant([0.0, 255.0])).numpy()
    assert np.allclose(endpoints, [-1.0, 1.0])
    enable_fine_tuning(model, backbone, 1e-5, 10)
    assert any(layer.trainable for layer in backbone.layers)
    assert not any(
        layer.trainable
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    )
