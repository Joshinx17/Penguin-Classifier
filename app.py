from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image


MODEL_PATH = Path("models/penguin_image_classifier.keras")
CLASS_NAMES_PATH = Path("models/class_names.txt")
IMAGE_SIZE = (224, 224)


st.set_page_config(page_title="Penguin Image Classifier")
st.title("Penguin Image Classifier")
st.markdown(
    "Upload a penguin image and the deep learning model will predict which "
    "penguin class it belongs to."
)


def read_class_names():
    if not CLASS_NAMES_PATH.exists():
        return None
    return [
        line.strip()
        for line in CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, (
            "No trained model found. Train one first with "
            "`python train_image_classifier.py --data-dir data/penguins`."
        )

    try:
        import tensorflow as tf
    except ImportError:
        return None, "TensorFlow is not installed. Run `pip install -r requirements.txt`."

    return tf.keras.models.load_model(MODEL_PATH), None


def prepare_image(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


class_names = read_class_names()
model, model_error = load_model()

with st.sidebar:
    st.header("Model")
    st.write(f"Model file: `{MODEL_PATH}`")
    if class_names:
        st.write("Classes")
        st.write(", ".join(class_names))

uploaded_file = st.file_uploader(
    "Choose a penguin image",
    type=("jpg", "jpeg", "png", "webp"),
)

if model_error:
    st.warning(model_error)
    st.info(
        "Put training images in folders by class, for example "
        "`data/penguins/Adelie`, `data/penguins/Chinstrap`, and "
        "`data/penguins/Gentoo`, then run the training command."
    )

if uploaded_file is None:
    st.stop()

image = Image.open(uploaded_file)
st.image(image, caption="Uploaded image", use_container_width=True)

if model is None:
    st.stop()

if not class_names:
    st.error(f"Class label file not found: `{CLASS_NAMES_PATH}`")
    st.stop()

input_batch = prepare_image(image)
probabilities = model.predict(input_batch, verbose=0)[0]
predicted_index = int(np.argmax(probabilities))
predicted_class = class_names[predicted_index]
confidence = float(probabilities[predicted_index])

st.subheader("Prediction")
st.write(f"**{predicted_class}**")
st.progress(confidence)
st.caption(f"Confidence: {confidence:.1%}")

st.subheader("Class probabilities")
probability_table = sorted(
    zip(class_names, probabilities),
    key=lambda item: item[1],
    reverse=True,
)
st.dataframe(
    {
        "Penguin": [name for name, _ in probability_table],
        "Probability": [f"{score:.1%}" for _, score in probability_table],
    },
    hide_index=True,
    use_container_width=True,
)
