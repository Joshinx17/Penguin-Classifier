# Model card

**Intended use:** Educational three-class classification of photographs showing an Adelie, Chinstrap, or Gentoo penguin.

**Unsupported use:** Wildlife population decisions, arbitrary animal recognition, absence/presence of penguins, or calibrated certainty estimates.

**Architecture:** MobileNetV2 without its ImageNet head, frozen by default, with global average pooling, dropout, and a three-class softmax head. RGB float32 pixel input is normalized from 0–255 to about −1–1 inside the saved model. Optional fine-tuning keeps Batch Normalization frozen.

**Data:** The repository includes no verified licensed photograph dataset. Local 45-image illustrations are demo fixtures with unknown external provenance. A training run must record source URL, license, and verification status. The data validator checks integrity but cannot validate species labels scientifically.

**Evaluation:** Seeded stratified train/validation/test split. Validation loss chooses the checkpoint; held-out test metrics include accuracy, macro precision/recall/F1, class-wise report, support, and confusion matrix. No validated photographic results are currently available. Demo results must never be reported as field performance.

**Risks:** Class imbalance, source leakage, background shortcuts, habitat/camera bias, near-duplicate photos, and shift to unseen conditions. Softmax confidence is uncalibrated. Non-penguin photos still receive one of the three labels. A real data collection and independent evaluation are required before practical use.
