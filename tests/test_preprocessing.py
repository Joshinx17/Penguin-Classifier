from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from penguin_classifier.preprocessing import ImageInputError, prepare_image


def test_rgb_shape_range_and_single_normalization():
    buffer = BytesIO()
    Image.new("L", (40, 50), 255).save(buffer, format="PNG")
    batch = prepare_image(buffer.getvalue(), (32, 32))
    assert batch.shape == (1, 32, 32, 3)
    assert batch.dtype == np.float32
    assert np.all(batch == 255)  # Keras model alone converts to [-1, 1].


def test_invalid_image():
    with pytest.raises(ImageInputError):
        prepare_image(b"not an image", (32, 32))
