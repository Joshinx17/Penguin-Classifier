"""MobileNetV2 transfer model. Input preprocessing is saved in the model."""

from __future__ import annotations


def build_model(
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    weights: str | None = "imagenet",
    dropout: float = 0.25,
    learning_rate: float = 5e-4,
):
    import tensorflow as tf

    if num_classes < 2 or min(image_size) < 32 or not 0 <= dropout < 1 or learning_rate <= 0:
        raise ValueError("Invalid model configuration")
    if weights not in {"imagenet", None}:
        raise ValueError("weights must be imagenet or None")
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3), include_top=False, weights=weights
    )
    backbone.trainable = False
    inputs = tf.keras.Input((*image_size, 3), name="rgb_pixels_0_255")
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_normalization")(inputs)
    x = backbone(x, training=False)  # Keep BatchNorm statistics fixed even during fine-tuning.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        tf.keras.optimizers.Adam(learning_rate),
        "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, backbone


def enable_fine_tuning(model, backbone, learning_rate: float, unfreeze_last: int = 30) -> None:
    import tensorflow as tf

    if learning_rate <= 0 or learning_rate > 1e-4 or unfreeze_last < 1:
        raise ValueError("Fine-tuning needs 0 < learning rate <= 1e-4 and positive unfreeze_last")
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-unfreeze_last:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    model.compile(
        tf.keras.optimizers.Adam(learning_rate),
        "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
