import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


IMAGE_SIZE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a penguin class from an image.")
    parser.add_argument("image", help="Path to the penguin image.")
    parser.add_argument(
        "--model",
        default="models/penguin_image_classifier.keras",
        help="Path to the trained Keras model.",
    )
    parser.add_argument(
        "--classes",
        default="models/class_names.txt",
        help="Path to the class names text file.",
    )
    return parser.parse_args()


def load_image(path):
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def load_class_names(path):
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model)
    class_names = load_class_names(args.classes)
    probabilities = model.predict(load_image(args.image), verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    print(f"Prediction: {class_names[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index]:.2%}")
    print()
    print("All class probabilities:")
    for class_name, probability in sorted(
        zip(class_names, probabilities),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"- {class_name}: {probability:.2%}")


if __name__ == "__main__":
    main()
