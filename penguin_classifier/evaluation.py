"""Held-out classification metrics and training curves."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate(model, dataset, class_names: list[str], report_dir: Path) -> dict:
    actual, predicted = [], []
    for images, labels in dataset:
        probabilities = model(images, training=False).numpy()
        actual.extend(labels.numpy().tolist())
        predicted.extend(np.argmax(probabilities, axis=1).tolist())
    indices = list(range(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=indices, average="macro", zero_division=0
    )
    report = {
        "test_count": len(actual),
        "test_accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=indices).tolist(),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=indices,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def save_history(history: dict, report_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (report_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for key in (metric, f"val_{metric}"):
            if key in history:
                ax.plot(history[key], label=key)
        ax.set(xlabel="Epoch", ylabel=metric.title())
        ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "training_curves.png", dpi=140)
    plt.close(fig)
