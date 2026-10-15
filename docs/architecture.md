# Architecture

The root scripts are human-facing entry points. `penguin_classifier/data.py` validates source folders and creates deterministic, stratified splits. `preprocessing.py` reads JPEG/PNG, converts to RGB, resizes bilinearly, and returns float32 pixels in 0–255. `model.py` embeds normalization as a Keras layer before MobileNetV2. This prevents the CLI and app from accidentally skipping it or applying it twice.

Training uses augmentation only on the training dataset. Validation loss controls early stopping and checkpoint selection. The test dataset is evaluated once after selection. `evaluation.py` writes metrics and learning curves. `inference.py` checks schema, class ordering, input/output shapes, and probability vectors. The CLI and Streamlit app both call it; the app caches a model keyed by artifact modification times.

`evaluate_model.py` replays the saved seed and checks every split hash before scoring a saved checkpoint, so changing the dataset cannot silently produce a new test result.

Artifacts are a native `.keras` model plus JSON metadata schema version 1. Metadata includes exact ordered classes, image dimensions, preprocessing contract, training settings, provenance, split SHA-256 hashes, and evaluation. A missing or incompatible pair raises `ArtifactError`. Input problems raise `ImageInputError`. The app turns these into user-facing messages while logging technical failures.

Exact duplicate detection uses file hashes. It does not detect crop, resize, near-duplicate video frames, or the same individual across photos; users must group such material before splitting.
