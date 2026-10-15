"""Re-evaluate a saved model against its recorded held-out test split."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from penguin_classifier.data import DatasetError, split_records, split_summary, validate_dataset
from penguin_classifier.evaluation import evaluate
from penguin_classifier.inference import ArtifactError, load_classifier
from train_image_classifier import make_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.report_dir.exists() and any(args.report_dir.iterdir()):
        parser.error("Report directory must be new or empty")
    try:
        model, metadata = load_classifier(args.model_dir)
        records, _ = validate_dataset(args.data_dir)
        seed = metadata["training"]["seed"]
        splits = split_records(records, seed)
        if split_summary(splits) != metadata["splits"]:
            raise DatasetError("Dataset or recorded split changed; refusing evaluation")
        test_ds = make_dataset(
            splits["test"],
            metadata["training"]["batch_size"],
            tuple(metadata["image_size"]),
            False,
            seed,
        )
        report = evaluate(model, test_ds, metadata["class_names"], args.report_dir)
    except (ArtifactError, DatasetError, KeyError, OSError, ValueError) as exc:
        logging.error("Evaluation failed: %s", exc)
        return 2
    print(f"Held-out test accuracy: {report['test_accuracy']:.3f}")
    print(f"Macro F1: {report['macro_f1']:.3f}; samples: {report['test_count']}")
    print(f"Report: {args.report_dir / 'evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
