"""Models accept RGB float32 pixels in [0, 255]; normalization lives inside Keras."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from penguin_classifier.data import MAX_IMAGE_BYTES, MAX_PIXELS

MAX_UPLOAD_BYTES = MAX_IMAGE_BYTES


class ImageInputError(ValueError):
    """An input image is invalid or too large."""


def prepare_image(source: str | Path | bytes, image_size: tuple[int, int]) -> np.ndarray:
    if isinstance(source, bytes):
        if len(source) > MAX_UPLOAD_BYTES:
            raise ImageInputError("Image exceeds the 10 MB upload limit")
        stream = BytesIO(source)
    else:
        path = Path(source)
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ImageInputError("Use a JPG or PNG image")
        if not path.is_file():
            raise ImageInputError(f"Image not found: {path}")
        if path.stat().st_size > MAX_UPLOAD_BYTES:
            raise ImageInputError("Image exceeds the 10 MB limit")
        stream = path.open("rb")
    try:
        with stream, Image.open(stream) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ImageInputError("Image contents must be JPEG or PNG")
            if image.width < 32 or image.height < 32 or image.width * image.height > MAX_PIXELS:
                raise ImageInputError(
                    "Image dimensions must be at least 32×32 and at most 30 million pixels"
                )
            rgb = image.convert("RGB").resize(
                (image_size[1], image_size[0]), Image.Resampling.BILINEAR
            )
            return np.asarray(rgb, dtype=np.float32)[None, ...]
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageInputError("Could not read the image") from exc
