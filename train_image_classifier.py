import argparse
from pathlib import Path

import tensorflow as tf


IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a transfer-learning penguin image classifier."
    )
    parser.add_argument(
        "--data-dir",
        default="data/penguins",
        help="Folder containing one subfolder per penguin class.",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory where the trained model and labels are saved.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use ImageNet transfer learning weights, or train from scratch with none.",
    )
    return parser.parse_args()


def build_datasets(data_dir, batch_size, validation_split):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    train_ds = train_ds.prefetch(AUTOTUNE)
    validation_ds = validation_ds.prefetch(AUTOTUNE)
    return train_ds, validation_ds, class_names


def build_model(num_classes, weights):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights=None if weights == "none" else weights,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}. Expected one subfolder per class."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, validation_ds, class_names = build_datasets(
        data_dir,
        args.batch_size,
        args.validation_split,
    )
    model = build_model(num_classes=len(class_names), weights=args.weights)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model_path = output_dir / "penguin_image_classifier.keras"
    class_names_path = output_dir / "class_names.txt"
    model.save(model_path)
    class_names_path.write_text("\n".join(class_names), encoding="utf-8")

    best_accuracy = max(history.history.get("val_accuracy", [0.0]))
    print(f"Saved model to {model_path}")
    print(f"Saved class names to {class_names_path}")
    print(f"Best validation accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()
