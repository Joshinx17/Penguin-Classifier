import time
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# Setup page configuration with a premium theme
st.set_page_config(
    page_title="Penguin Classifier - Deep Learning Portal",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("models/penguin_image_classifier.keras")
CLASS_NAMES_PATH = Path("models/class_names.txt")
IMAGE_SIZE = (224, 224)

# Custom premium CSS styling (Glassmorphism + Ice Theme)
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Playfair+Display:ital,wght@0,600;1,400&display=swap');

/* Main Background and Fonts */
.stApp {
    background: radial-gradient(circle at 50% 50%, #0d1b2a 0%, #010811 100%);
    color: #e0e1dd;
    font-family: 'Outfit', sans-serif;
}

/* Glassmorphism containers */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: transform 0.3s ease, border-color 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 180, 216, 0.4);
}

/* Header design */
.main-header {
    background: linear-gradient(90deg, #00b4d8 0%, #90e0ef 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    margin-bottom: 5px;
    letter-spacing: -1px;
}
.subtitle {
    font-size: 1.15rem;
    color: #90e0ef;
    margin-bottom: 30px;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* Custom metrics and highlights */
.highlight-text {
    font-weight: 800;
    font-size: 2.2rem;
    color: #00b4d8;
    margin: 10px 0;
    text-shadow: 0 0 15px rgba(0, 180, 216, 0.3);
}

/* Prediction Bars */
.prob-label {
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
}
.prob-bar-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    height: 12px;
    width: 100%;
    margin-bottom: 15px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.prob-bar-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #00b4d8, #90e0ef);
    box-shadow: 0 0 10px rgba(0, 180, 216, 0.5);
    transition: width 0.8s cubic-bezier(0.1, 0.8, 0.2, 1);
}

/* File Uploader override */
div[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.02);
    border: 2px dashed rgba(0, 180, 216, 0.2);
    border-radius: 15px;
    padding: 20px;
    transition: border-color 0.3s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #00b4d8;
}

/* Buttons style */
.stButton>button {
    background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 10px 24px !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(0, 180, 216, 0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton>button:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 6px 20px rgba(0, 180, 216, 0.5) !important;
}

/* Quick Select Images styling */
.img-option {
    border: 2px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.img-option:hover {
    border-color: #00b4d8;
    transform: scale(1.05);
}

/* Custom list items for species */
.trait-badge {
    display: inline-block;
    background: rgba(0, 180, 216, 0.15);
    border: 1px solid rgba(0, 180, 216, 0.3);
    color: #90e0ef;
    padding: 4px 10px;
    border-radius: 30px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Penguin species educational database
SPECIES_DATABASE = {
    "Adelie": {
        "scientific_name": "Pygoscelis adeliae",
        "badges": ["Antarctic Continent", "Pebble Nests", "Medium Size"],
        "description": (
            "Adelie penguins are medium-sized birds characterized by a black back and head, "
            "a white chest, and a very distinctive white ring around their eyes. They are the most "
            "widespread penguin species on the Antarctic coast."
        ),
        "fun_fact": (
            "They build their nests entirely out of pebbles. Because pebbles are scarce in Antarctica, "
            "Adelies frequently steal pebbles from neighboring nests when the owners aren't looking!"
        ),
        "diet": "Krill, silverfish, and glacial squid.",
        "conservation": "Least Concern"
    },
    "Chinstrap": {
        "scientific_name": "Pygoscelis antarcticus",
        "badges": ["South Shetland Islands", "Bold & Aggressive", "Helmet Strap Line"],
        "description": (
            "Easily identified by a narrow black band under their chin that looks like a helmet strap, "
            "Chinstrap penguins have a white face and a stark black back. They form massive colonies "
            "on rocky islands around the Antarctic Peninsula."
        ),
        "fun_fact": (
            "Chinstrap penguins are often considered the boldest and most aggressive of all penguin species. "
            "They can dive up to 70 meters (230 feet) deep and make hundreds of short dives a day to catch krill."
        ),
        "diet": "Primarily krill, with some small fish.",
        "conservation": "Least Concern"
    },
    "Gentoo": {
        "scientific_name": "Pygoscelis papua",
        "badges": ["Sub-Antarctic Islands", "Speed Swimmers", "Orange-Red Bill"],
        "description": (
            "Gentoo penguins are distinguished by a bright white stripe across the top of their heads "
            "(like a bonnet or headband), a vibrant orange-red bill, and a long brush-like tail that "
            "sweeps behind them. They are the third-largest penguin species."
        ),
        "fun_fact": (
            "Gentoos are the fastest underwater swimming birds on Earth! They can reach speeds of up to "
            "36 km/h (22 mph) under water, which helps them easily evade seals and killer whales."
        ),
        "diet": "Krill, fish, and squid.",
        "conservation": "Least Concern"
    }
}


def read_class_names():
    if not CLASS_NAMES_PATH.exists():
        return None
    return [
        line.strip()
        for line in CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# Cache model with file modification time dependency (prevents lock on failure)
@st.cache_resource
def load_model_cached(model_path_str, file_mtime):
    try:
        import tensorflow as tf
    except ImportError:
        return None, "TensorFlow is not installed in the environment."
    try:
        model = tf.keras.models.load_model(model_path_str)
        return model, None
    except Exception as e:
        return None, f"Failed to load the trained model file: {e}"


def get_model_and_error():
    if not MODEL_PATH.exists():
        return None, (
            "No trained model found. Train one first with:\n"
            "`python train_image_classifier.py --data-dir data/penguins`"
        )
    mtime = MODEL_PATH.stat().st_mtime
    return load_model_cached(str(MODEL_PATH), mtime)


def prepare_image(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


# Main Web Dashboard Layout
# Title Header
st.markdown('<div class="main-header">PENGUIN CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep learning transfer classifier built with MobileNetV2</div>', unsafe_allow_html=True)

# Fetch Model status
class_names = read_class_names()
model, model_error = get_model_and_error()

# Sidebar: Status & Controls
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/0/08/Chinstrap_Penguin_on_iceberg.jpg",
        use_container_width=True,
    )
    st.markdown("### 🛠️ Model Environment")

    if model_error:
        st.error("🔴 Model Inactive")
        st.warning("Please train a model using your dataset first.")
    else:
        st.success("🟢 Model Active & Cached")
        st.info(f"Loaded: `{MODEL_PATH.name}`")

    if class_names:
        st.markdown("#### Supported Classes:")
        for cls in class_names:
            st.markdown(f"- **{cls}**")
    else:
        st.markdown("*No classes registered yet.*")

    st.markdown("---")
    st.markdown("### 📚 Project Status")
    st.markdown(
        "This project uses transfer learning. A base MobileNetV2 network "
        "trained on ImageNet is frozen, and a dense output layer is trained "
        "on custom penguin data."
    )

# Content area
if model_error:
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 5px solid #ff4b4b;">
            <h3 style="color:#ff4b4b;margin-top:0;">⚠️ Model File Missing</h3>
            <p>{model_error}</p>
            <p><strong>To fix this and get running in 1 minute:</strong></p>
            <ol>
                <li>Run <code>python download_sample_data.py</code> to download sample penguin photos.</li>
                <li>Run <code>python train_image_classifier.py --epochs 5</code> to train the classifier.</li>
                <li>Refresh this page, and the application will load automatically.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Quick Select options (If sample data exists)
quick_samples = {
    "Adelie": Path("data/penguins/Adelie/sample-1.jpg"),
    "Chinstrap": Path("data/penguins/Chinstrap/sample-1.jpg"),
    "Gentoo": Path("data/penguins/Gentoo/sample-1.jpg"),
}

selected_sample_path = None

# Show sample images clicker if available
available_samples = {k: v for k, v in quick_samples.items() if v.exists()}

st.markdown("### 📷 Select a Penguin Image")

col_upload, col_samples = st.columns([2, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a custom penguin image",
        type=("jpg", "jpeg", "png", "webp"),
        help="Upload a crisp photo of a single penguin for best results.",
    )

with col_samples:
    if available_samples:
        st.markdown("**OR choose one of our verified samples:**")
        sample_cols = st.columns(len(available_samples))
        for idx, (species, path) in enumerate(available_samples.items()):
            with sample_cols[idx]:
                st.image(str(path), caption=species, use_container_width=True)
                if st.button(f"Analyze {species}", key=f"btn_{species}"):
                    selected_sample_path = path
    else:
        st.markdown(
            "*(💡 Tip: run `python download_sample_data.py` to add quick-click sample photos here)*"
        )

# Select image source
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif selected_sample_path is not None:
    image = Image.open(selected_sample_path)

if image is None:
    st.markdown(
        """
        <div style='text-align: center; padding: 40px; color: #888;'>
            <p style='font-size: 1.2rem;'>Select a quick-click sample or upload an image to begin the analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Execution and presentation
st.markdown("---")
col_img, col_results = st.columns([1, 1])

with col_img:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.image(image, caption="Active Image Analysis", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_results:
    if model is None:
        st.markdown(
            """
            <div class="glass-card" style="border-left: 5px solid #e07a5f;">
                <h4 style="color:#e07a5f;margin-top:0;">Inference Halted</h4>
                <p>A penguin photo is selected, but the neural network classifier is offline. Please train the model to enable live predictions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    if not class_names:
        st.error("Class labels are missing. Make sure models/class_names.txt exists.")
        st.stop()

    with st.spinner("🧠 Feedforwarding image through MobileNetV2 backbone..."):
        input_batch = prepare_image(image)
        probabilities = model.predict(input_batch, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index])

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Classification Results")
    st.markdown(f"The model predicts the species is:")
    st.markdown(f'<div class="highlight-text">{predicted_class}</div>', unsafe_allow_html=True)
    st.markdown(f"**Confidence Score**: `{confidence:.2%}`")
    st.markdown('</div>', unsafe_allow_html=True)

    # Probabilities section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Prediction Confidence Distribution")

    sorted_probs = sorted(
        zip(class_names, probabilities),
        key=lambda x: x[1],
        reverse=True,
    )

    for cls_name, prob in sorted_probs:
        pct = prob * 100
        st.markdown(
            f"""
            <div class="prob-label">
                <span>{cls_name}</span>
                <span>{prob:.1%}</span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar-fill" style="width: {pct}%;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# Species Spotlight Card (Educational Details)
if predicted_class in SPECIES_DATABASE:
    info = SPECIES_DATABASE[predicted_class]
    st.markdown('<div class="glass-card" style="margin-top: 10px;">', unsafe_allow_html=True)
    st.markdown(f"## 📚 Species Spotlight: **{predicted_class} Penguin**")
    st.markdown(f"*Scientific Name: `{info['scientific_name']}`*")

    # Badges row
    badges_html = "".join([f'<span class="trait-badge">{badge}</span>' for badge in info["badges"]])
    st.markdown(badges_html, unsafe_allow_html=True)
    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    col_info_l, col_info_r = st.columns(2)
    with col_info_l:
        st.markdown(f"**Description**:\n{info['description']}")
        st.markdown(f"**Primary Diet**:\n{info['diet']}")
    with col_info_r:
        st.markdown(
            f"""
            <div style="background: rgba(0, 180, 216, 0.05); padding: 15px; border-radius: 12px; border-left: 4px solid #00b4d8;">
                <h5 style="margin-top:0; color:#00b4d8;">💡 Did You Know?</h5>
                <p style="font-size:0.95rem; margin-bottom:0; font-style:italic;">"{info['fun_fact']}"</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**Conservation Status**: `{info['conservation']}`")

    st.markdown('</div>', unsafe_allow_html=True)
