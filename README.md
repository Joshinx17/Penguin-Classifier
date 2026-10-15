# Penguin Classifier

A small, reproducible MobileNetV2 project for classifying **photographs** of Adelie, Chinstrap, and Gentoo penguins. The repository includes a dataset validator, stratified training pipeline, held-out evaluation, command-line predictor, and Streamlit app. It does **not** include a verified photographic dataset or a validated production model.

## How it works

```text
Verified labeled photos → validation → seeded train/validation/test split
  → training-only augmentation → MobileNetV2 + classification head
  → best checkpoint + metadata → held-out report → CLI / Streamlit
```

The input is a 224×224 RGB `float32` image with pixel values from 0 to 255, resized with Pillow bilinear interpolation. A `Rescaling(1/127.5, offset=-1)` layer **inside the saved model** converts to the MobileNetV2 range. Both inference surfaces call the same `predict` function. The ImageNet backbone is frozen for head training. Optional fine-tuning unfreezes its last layers while keeping Batch Normalization frozen and called in inference mode.

## Setup

Use Python 3.11–3.13 (3.11 is used by CI and intended for hosting):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The main dependencies are TensorFlow/Keras (MobileNetV2), Pillow (image decoding), scikit-learn (stratified splits and metrics), NumPy (arrays), Matplotlib (curves), and Streamlit (UI). `requirements.txt` mirrors the application dependencies for hosts that install from it.

## Dataset

Provide permission-cleared, **individually verified** penguin photographs:

```text
data/penguins/Adelie/*.jpg
data/penguins/Chinstrap/*.jpg
data/penguins/Gentoo/*.jpg
```

At least seven distinct images per class are needed for the default three-way split. The validator rejects corrupt files, unsupported formats, unexpected classes/files, empty or missing classes, and exact SHA-256 duplicates. It reports counts and imbalance. It cannot reliably detect visually similar frames or images of the same animal; group those manually before training. A useful real dataset should have far more than seven photos per class and a clearly licensed source. The project does not automatically scrape or relabel photographs.

Add `data/penguins/.dataset-source.json` after verifying the photos and their license:

```json
{
  "source_url": "https://your-real-source.example/dataset",
  "license": "Actual license and attribution for these photographs",
  "image_labels_verified": true
}
```

Replace the example values with the actual source and license. Check whether that license permits the intended training and display. If photos originate from a video, individual, or shared scene, keep that group within one split; the current splitter only prevents **exact** duplicates.

Validate without training:

```bash
python train_image_classifier.py --data-dir data/penguins --validate-only
```

The old `download_sample_data.py` actually generated drawings. It now creates an explicitly marked demo dataset in a new directory and never overwrites existing data:

```bash
python download_sample_data.py --output-dir data/demo_penguins
python train_image_classifier.py --data-dir data/demo_penguins --allow-demo-data --weights none --epochs 1 --output-dir models/demo-run --report-dir reports/demo-run
```

This checks the pipeline only. Its predictions and test metrics are **not evidence of photographic species accuracy**. The 45 local `data/penguins/sample-*.jpg` files found during this audit are also drawings with no provenance file, and the old `models/` artifact has no compatible metadata; both are ignored by Git and preserved.

## Train and evaluate

After preparing real data:

```bash
python train_image_classifier.py --data-dir data/penguins --output-dir models/real-run --report-dir reports/real-run --epochs 10 --batch-size 32 --seed 42
```

The first ImageNet run may download weights. Use `--weights none` only for an offline pipeline check; its frozen backbone has random features and cannot establish useful image recognition. Optional fine-tuning: `--fine-tune-epochs 3 --fine-tune-learning-rate 0.00001`. Use a **new empty** output and report directory for every run. The chosen `.keras` checkpoint, `metadata.json`, `evaluation.json`, `history.json`, and `training_curves.png` are generated. Metadata records class order, input/preprocessing contract, seed, configuration, source, split hashes/counts, and held-out metrics. Test images are used only after model selection. Head training uses early stopping on validation loss.

The default split is approximately 70/15/15, stratified with a fixed seed. Small data may produce unstable scores. Macro precision, recall, F1, class support, and confusion matrix are saved along with accuracy. No real photographic metrics are claimed here because a verified dataset is absent.

Re-evaluate the saved model against the same test images (the command refuses a changed split):

```bash
python evaluate_model.py --data-dir data/penguins --model-dir models/real-run --report-dir reports/real-recheck
```

## Predict and run the app

```bash
python predict_image.py path/to/photo.jpg --model-dir models/real-run
python predict_image.py path/to/photo.jpg --model-dir models/real-run --json --top-k 3
streamlit run app.py
```

The app loads `models/real-run` by default. Set `PENGUIN_MODEL_DIR` in the process environment to use a different trusted artifact directory. The app validates JPEG/PNG content and dimensions, limits uploads to 10 MB, displays all class scores, and handles missing artifacts. Scores are uncalibrated softmax outputs. An image of another species will still receive one of the three labels; no out-of-distribution detector is included. The 50% low-score notice is only a heuristic. The field guide is reference text, not generated by the neural network.

## Quality checks

```bash
ruff check .
ruff format --check .
pytest -q
```

CI runs lint, format, and fast tests on Python 3.11. The `slow` test builds a real MobileNetV2 model locally; CI skips it. See [development guide](docs/development.md).

## Deploy

Streamlit Community Cloud can install from `requirements.txt` and run `app.py` with Python 3.11. The host needs the paired `.keras` and `metadata.json` in a selected artifact directory. This repository ignores generated models, so deployment requires a deliberate, trusted artifact delivery step after real training; the app never downloads model files. Check memory and storage needs with the selected host before deploying TensorFlow and a trained model. Streamlit [documents resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app) and [Python selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy). `.streamlit/config.toml` caps uploads at 10 MB. There is no deployed URL yet.

## Limitations and troubleshooting

- No verified photographic dataset or supported accuracy claim is bundled. Source, individual, scene, geography, lighting, and camera bias may affect results.
- If artifacts are missing, train a real model and point the sidebar or CLI to its directory. Old `class_names.txt` artifacts lack the required schema and cannot be treated as validated models.
- If a split fails, add more independent images per class. Exact hash detection cannot identify near-duplicates.
- If ImageNet weights cannot download, check network access or use `--weights none` for an architecture smoke check; the backbone is frozen with random weights.
- If the host runs out of memory, train elsewhere and deploy only the selected model and metadata.

Next improvements: obtain a clearly licensed photo dataset, group near-duplicates/individuals before splitting, assess calibration and out-of-distribution behavior, and add a real app screenshot after validation.

Field-guide facts were checked against [British Antarctic Survey's penguin overview](https://legacy.bas.ac.uk/about_antarctica/wildlife/birds/penguins/index.php), [Adelie](https://legacy.bas.ac.uk/about_antarctica/wildlife/birds/penguins/adelie.php), [Chinstrap](https://legacy.bas.ac.uk/about_antarctica/wildlife/birds/penguins/chinstrap.php), and [Animal Diversity Web's Gentoo account](https://animaldiversity.org/accounts/Pygoscelis_papua/). These sources do not license reuse of their photos; no source images were copied. This repository currently has no software license; the owner should choose one before inviting reuse.
