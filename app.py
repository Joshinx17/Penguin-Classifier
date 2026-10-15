"""Streamlit interface for a locally trained penguin photograph classifier."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import streamlit as st

from penguin_classifier import CLASS_NAMES, METADATA_NAME, MODEL_NAME
from penguin_classifier.inference import ArtifactError, load_classifier, predict
from penguin_classifier.preprocessing import ImageInputError, MAX_UPLOAD_BYTES, prepare_image

LOG = logging.getLogger(__name__)
MODEL_DIR = Path(os.environ.get("PENGUIN_MODEL_DIR", "models/real-run"))
SPECIES = {
    "Adelie": {
        "scientific": "Pygoscelis adeliae",
        "field_mark": "White eye ring",
        "habitat": "Antarctic coast",
        "diet": "Mainly krill and fish",
    },
    "Chinstrap": {
        "scientific": "Pygoscelis antarcticus",
        "field_mark": "Thin black line under the chin",
        "habitat": "Antarctic Peninsula and nearby islands",
        "diet": "Mainly krill",
    },
    "Gentoo": {
        "scientific": "Pygoscelis papua",
        "field_mark": "White head patch and orange bill",
        "habitat": "Subantarctic and Antarctic islands",
        "diet": "Krill, fish, and squid",
    },
}


@st.cache_resource
def cached_classifier(model_dir: str, model_mtime: float, metadata_mtime: float):
    return load_classifier(model_dir)


def artifact_status(directory: Path):
    model_path, metadata_path = directory / MODEL_NAME, directory / METADATA_NAME
    if not model_path.is_file() or not metadata_path.is_file():
        return (
            None,
            None,
            "A trained model and metadata are needed. Run `python train_image_classifier.py --data-dir data/penguins --output-dir models/real-run`, or set PENGUIN_MODEL_DIR to a trained artifact directory.",
        )
    try:
        model, metadata = cached_classifier(
            str(directory), model_path.stat().st_mtime, metadata_path.stat().st_mtime
        )
        return model, metadata, None
    except (ArtifactError, OSError, ValueError):
        LOG.exception("Could not load model artifacts")
        return (
            None,
            None,
            "Model artifacts could not be loaded. Check that the model and metadata came from the same training run.",
        )


def main() -> None:
    st.set_page_config(page_title="Penguin Classifier", page_icon="🐧", layout="wide")
    st.markdown(
        """<style>
      .stApp {background: #f4f8fa; color: #123443}
      .block-container {max-width: 1050px; padding-top: 2rem}
      h1, h2, h3 {color: #123443}
      div[data-testid="stMetric"] {background: #e4f2f4; border-radius: 12px; padding: 1rem}
    </style>""",
        unsafe_allow_html=True,
    )
    st.title("🐧 Penguin Classifier")
    st.caption("Identify Adelie, Chinstrap, or Gentoo penguins from a photograph")
    st.write(
        "A MobileNetV2 image classifier trained on labeled photos. It only distinguishes these three classes; a non-penguin image will still receive a class score."
    )

    with st.sidebar:
        st.header("Model")
        model_dir = MODEL_DIR
        st.code(str(model_dir))
        st.write("Classes: " + ", ".join(CLASS_NAMES))
        st.caption("Scores are model outputs, not calibrated probabilities of correctness.")

    model, metadata, error = artifact_status(model_dir)
    if error:
        st.warning(error)
    elif metadata:
        st.success("Model ready")
        st.caption(
            f"{metadata['architecture']} · {metadata['image_size'][0]}×{metadata['image_size'][1]} RGB · {'demo illustrations only' if metadata['demo_only'] else 'photograph dataset'}"
        )
        if not metadata["demo_only"] and metadata.get("evaluation"):
            st.metric("Held-out test accuracy", f"{metadata['evaluation']['test_accuracy']:.1%}")

    st.header("Analyze an image")
    uploaded = st.file_uploader("JPG or PNG, up to 10 MB", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        content = uploaded.getvalue()
        try:
            prepare_image(content, (224, 224))  # Validate before rendering or inference.
        except ImageInputError as exc:
            st.error(str(exc))
            return
        if len(content) > MAX_UPLOAD_BYTES:
            st.error("Image exceeds the 10 MB upload limit")
            return
        left, right = st.columns([1, 1])
        with left:
            st.image(content, caption="Uploaded image", use_container_width=True)
        with right:
            if model is None:
                st.info("Train and load a model to see a prediction.")
            else:
                try:
                    with st.spinner("Classifying image…"):
                        result = predict(model, metadata, content)
                    st.subheader(result["prediction"])
                    st.metric("Model confidence score", f"{result['confidence']:.1%}")
                    for name, score in sorted(
                        result["scores"].items(), key=lambda item: item[1], reverse=True
                    ):
                        st.write(f"{name}: {score:.1%}")
                        st.progress(score)
                    if result["confidence"] < 0.5:
                        st.info(
                            "The top score is below the 50% heuristic threshold. This threshold has not been calibrated."
                        )
                    st.caption(
                        "A high score is not a guarantee. Check the image and field marks independently."
                    )
                except (ArtifactError, ImageInputError, OSError, ValueError):
                    LOG.exception("Prediction failed")
                    st.error(
                        "Could not classify this image. Try another JPG or PNG or check the model artifacts."
                    )
    st.divider()
    st.header("Field guide")
    for name, info in SPECIES.items():
        with st.expander(f"{name} · {info['scientific']}"):
            st.write(f"**Field mark:** {info['field_mark']}")
            st.write(f"**Habitat:** {info['habitat']}")
            st.write(f"**Diet:** {info['diet']}")
    st.caption(
        "Field guide text is curated reference information, separate from model output. Sources are listed in README.md."
    )


if __name__ == "__main__":
    main()
