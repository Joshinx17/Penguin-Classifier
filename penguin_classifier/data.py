"""Validate labeled photographs and make deterministic, stratified splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from penguin_classifier import CLASS_NAMES

EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_PIXELS = 30_000_000
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class DatasetError(ValueError):
    """The dataset cannot safely be used for training."""


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str
    digest: str


def validate_dataset(root: Path, *, min_per_class: int = 7) -> tuple[list[ImageRecord], dict]:
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"Dataset directory does not exist: {root}")
    if min_per_class < 1:
        raise DatasetError("min_per_class must be positive")
    found = {p.name for p in root.iterdir() if p.is_dir()}
    missing = sorted(set(CLASS_NAMES) - found)
    unexpected = sorted(found - set(CLASS_NAMES))
    rejected: list[dict[str, str]] = []
    records: list[ImageRecord] = []
    seen: dict[str, Path] = {}
    for name in CLASS_NAMES:
        folder = root / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_dir():
                continue
            reason = None
            if path.suffix.lower() not in EXTENSIONS:
                reason = "unsupported format or unexpected file"
            elif path.stat().st_size > MAX_IMAGE_BYTES:
                reason = "image exceeds 10 MB limit"
            else:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        if (
                            image.width < 32
                            or image.height < 32
                            or image.width * image.height > MAX_PIXELS
                        ):
                            reason = "invalid dimensions"
                        else:
                            image.convert("RGB").load()
                except (OSError, ValueError, UnidentifiedImageError):
                    reason = "unreadable image"
            if reason is None:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen:
                    reason = f"duplicate of {seen[digest]}"
                else:
                    seen[digest] = path
                    records.append(ImageRecord(path, name, digest))
            if reason:
                rejected.append({"path": str(path), "reason": reason})
    for path in root.iterdir():
        if path.is_file() and path.name not in {".synthetic-demo.json", ".dataset-source.json"}:
            rejected.append({"path": str(path), "reason": "unexpected file at dataset root"})
    counts = {name: sum(r.label == name for r in records) for name in CLASS_NAMES}
    summary = {
        "root": str(root),
        "counts": counts,
        "total": len(records),
        "rejected": rejected,
        "missing_classes": missing,
        "unexpected_classes": unexpected,
        "imbalance_ratio": round(max(counts.values()) / min(counts.values()), 2)
        if min(counts.values())
        else None,
    }
    errors = []
    if missing:
        errors.append(f"missing classes: {', '.join(missing)}")
    if unexpected:
        errors.append(f"unexpected classes: {', '.join(unexpected)}")
    if rejected:
        errors.append(f"{len(rejected)} rejected image(s)")
    if any(count < min_per_class for count in counts.values()):
        errors.append(f"each class needs at least {min_per_class} valid images for splitting")
    if errors:
        raise DatasetError("; ".join(errors) + "\n" + json.dumps(summary, indent=2))
    return records, summary


def split_records(records: list[ImageRecord], seed: int = 42) -> dict[str, list[ImageRecord]]:
    """Stratify 70/15/15; duplicates are rejected before this function is called."""
    from sklearn.model_selection import train_test_split

    labels = [record.label for record in records]
    if any(Counter(labels)[name] < 7 for name in CLASS_NAMES):
        raise DatasetError("At least seven images per class are needed for three splits")
    train, rest = train_test_split(records, test_size=0.30, stratify=labels, random_state=seed)
    val, test = train_test_split(
        rest, test_size=0.50, stratify=[r.label for r in rest], random_state=seed
    )
    return {"train": train, "validation": val, "test": test}


def split_summary(splits: dict[str, list[ImageRecord]]) -> dict:
    return {
        name: {
            "count": len(items),
            "classes": dict(Counter(r.label for r in items)),
            "sha256": sorted(r.digest for r in items),
        }
        for name, items in splits.items()
    }
