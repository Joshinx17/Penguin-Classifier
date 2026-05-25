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
    
    image_path = Path(args.image)
    model_path = Path(args.model)
    classes_path = Path(args.classes)

    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return

    if not model_path.exists():
        print(f"Error: Trained model file not found: {model_path}")
        print("Please train a model first using train_image_classifier.py.")
        return

    if not classes_path.exists():
        print(f"Error: Class names file not found: {classes_path}")
        print("Please train a model first to generate the class names file.")
        return

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error: Failed to load Keras model: {e}")
        return

    try:
        class_names = load_class_names(classes_path)
    except Exception as e:
        print(f"Error: Failed to load class names: {e}")
        return

    try:
        img_tensor = load_image(image_path)
    except Exception as e:
        print(f"Error: Failed to load or process image {image_path}: {e}")
        return

    probabilities = model.predict(img_tensor, verbose=0)[0]

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
