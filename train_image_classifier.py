"""Train and evaluate a MobileNetV2 classifier on labeled penguin photographs."""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from penguin_classifier import CLASS_NAMES, METADATA_NAME, MODEL_NAME
from penguin_classifier.data import DatasetError, split_records, split_summary, validate_dataset
from penguin_classifier.preprocessing import prepare_image

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/penguins"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--allow-demo-data",
        action="store_true",
        help="Permit generated illustrations for a pipeline smoke test only",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfreeze-last", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    return parser.parse_args()


def make_dataset(records, batch_size: int, image_size: tuple[int, int], training: bool, seed: int):
    import tensorflow as tf
    import numpy as np

    paths = [(str(record.path), CLASS_NAMES.index(record.label)) for record in records]

    def images():
        for path, label in paths:
            yield prepare_image(path, image_size)[0], np.int32(label)

    ds = tf.data.Dataset.from_generator(
        images,
        output_signature=(tf.TensorSpec((*image_size, 3), tf.float32), tf.TensorSpec((), tf.int32)),
    )
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training:
        augment = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ]
        )
        ds = ds.map(lambda image, label: (augment(image, training=True), label))
    return ds.apply(
        tf.data.experimental.assert_cardinality(math.ceil(len(paths) / batch_size))
    ).prefetch(1)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if (
        args.epochs < 1
        or args.fine_tune_epochs < 0
        or args.batch_size < 1
        or args.image_size < 32
        or args.learning_rate <= 0
        or args.unfreeze_last < 1
        or (args.fine_tune_epochs and args.weights == "none")
    ):
        LOG.error("Invalid training configuration; fine-tuning requires ImageNet weights")
        return 2
    try:
        records, summary = validate_dataset(args.data_dir)
        print(json.dumps(summary, indent=2))
        if args.validate_only:
            return 0
        source_path = args.data_dir / ".dataset-source.json"
        demo = (args.data_dir / ".synthetic-demo.json").exists() or not source_path.exists()
        if demo and not args.allow_demo_data:
            raise DatasetError(
                "Dataset provenance is missing or data is generated. Add .dataset-source.json for verified photos, or use --allow-demo-data for a smoke test only."
            )
        provenance = (
            json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else None
        )
        if provenance and not all(
            provenance.get(key) for key in ("source_url", "license", "image_labels_verified")
        ):
            raise DatasetError(
                ".dataset-source.json must include source_url, license, and image_labels_verified=true"
            )
        splits = split_records(records, args.seed)
    except DatasetError as exc:
        LOG.error("%s", exc)
        return 2
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        LOG.error(
            "Output directory is not empty: %s. Choose a new directory to protect existing artifacts.",
            args.output_dir,
        )
        return 2
    if args.report_dir.exists() and any(args.report_dir.iterdir()):
        LOG.error("Report directory is not empty: %s. Choose a new directory.", args.report_dir)
        return 2
    import tensorflow as tf
    from penguin_classifier.evaluation import evaluate, save_history
    from penguin_classifier.model import build_model, enable_fine_tuning

    tf.keras.utils.set_random_seed(args.seed)
    size = (args.image_size, args.image_size)
    datasets = {
        key: make_dataset(value, args.batch_size, size, key == "train", args.seed)
        for key, value in splits.items()
    }
    model, backbone = build_model(
        len(CLASS_NAMES),
        size,
        None if args.weights == "none" else args.weights,
        learning_rate=args.learning_rate,
    )
    args.output_dir.mkdir(parents=True)
    args.report_dir.mkdir(parents=True)
    checkpoint = args.output_dir / MODEL_NAME
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        datasets["train"],
        validation_data=datasets["validation"],
        epochs=args.epochs,
        callbacks=callbacks,
    ).history
    head_best_loss = min(history["val_loss"])
    selected_stage = "head"
    if args.fine_tune_epochs:
        enable_fine_tuning(model, backbone, args.fine_tune_learning_rate, args.unfreeze_last)
        fine_checkpoint = args.output_dir / "fine_tuned_candidate.keras"
        fine_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                fine_checkpoint, monitor="val_loss", save_best_only=True
            ),
        ]
        fine = model.fit(
            datasets["train"],
            validation_data=datasets["validation"],
            epochs=args.fine_tune_epochs,
            callbacks=fine_callbacks,
        ).history
        if min(fine["val_loss"]) < head_best_loss:
            checkpoint = fine_checkpoint
            selected_stage = "fine_tuned"
        history = {
            key: history.get(key, []) + fine.get(key, []) for key in set(history) | set(fine)
        }
    best = tf.keras.models.load_model(checkpoint, compile=False)
    if checkpoint.name != MODEL_NAME:
        best.save(args.output_dir / MODEL_NAME)
    report = evaluate(best, datasets["test"], list(CLASS_NAMES), args.report_dir)
    save_history(history, args.report_dir)
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "class_names": list(CLASS_NAMES),
        "image_size": list(size),
        "preprocessing": "rgb_float32_0_255; in_model_rescale_to_-1_1",
        "architecture": "MobileNetV2",
        "weights": args.weights,
        "selected_stage": selected_stage,
        "demo_only": demo,
        "provenance": provenance,
        "dataset": summary,
        "splits": split_summary(splits),
        "training": {
            "epochs": args.epochs,
            "fine_tune_epochs": args.fine_tune_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "fine_tune_learning_rate": args.fine_tune_learning_rate,
            "unfreeze_last": args.unfreeze_last,
            "seed": args.seed,
        },
        "evaluation": report,
    }
    (args.output_dir / METADATA_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOG.info(
        "Saved selected model and metadata to %s; held-out test accuracy %.3f",
        args.output_dir,
        report["test_accuracy"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
