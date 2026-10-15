# Development

Use Python 3.11–3.13. Install with `python -m pip install -e ".[dev]"`. Run `ruff check .`, `ruff format --check .`, and `pytest -q`. The fast CI suite excludes `pytest -q -m slow`, which builds a real network without downloading weights. No database or secrets are needed.

To reproduce training, record the exact command, data source/license file, seed, and generated metadata. Use an empty artifact and report directory per run. Start with `python train_image_classifier.py --validate-only --data-dir data/penguins`. Then run the training command in the README. The metadata stores split hashes, so the chosen records can be audited. Hold out the test split until model selection is finished.

For a smoke check, generate demo illustrations into a new path with `download_sample_data.py` and train with `--allow-demo-data --weights none --epochs 1`. This checks code and artifact flow only. Never report its metrics as species recognition quality. Verify a finished artifact using `predict_image.py` and `streamlit run app.py`.

New code should keep the class order in `penguin_classifier.CLASS_NAMES` and the preprocessing contract in the model. Add focused tests for new error cases. Do not commit datasets, model files, secrets, or reports. Main dependency rationale is in the README.
