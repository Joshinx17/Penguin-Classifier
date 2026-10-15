from pathlib import Path

import pytest
from PIL import Image

from penguin_classifier import CLASS_NAMES
from penguin_classifier.data import DatasetError, split_records, validate_dataset


def make_data(root: Path, count: int = 10) -> None:
    for class_index, name in enumerate(CLASS_NAMES):
        folder = root / name
        folder.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (40, 40), (class_index * 70, index * 10, 5)).save(
                folder / f"{index}.png"
            )


def test_validation_and_split(tmp_path):
    make_data(tmp_path)
    records, summary = validate_dataset(tmp_path)
    assert summary["counts"] == dict.fromkeys(CLASS_NAMES, 10)
    first = split_records(records, seed=5)
    second = split_records(records, seed=5)
    assert {key: [r.digest for r in value] for key, value in first.items()} == {
        key: [r.digest for r in value] for key, value in second.items()
    }
    assert len({r.digest for group in first.values() for r in group}) == len(records)


def test_missing_class_and_corrupt_image(tmp_path):
    make_data(tmp_path)
    (tmp_path / "Gentoo" / "0.png").write_bytes(b"broken")
    with pytest.raises(DatasetError, match="rejected image"):
        validate_dataset(tmp_path)
    for path in (tmp_path / "Gentoo").iterdir():
        path.unlink()
    (tmp_path / "Gentoo").rmdir()
    with pytest.raises(DatasetError, match="missing classes"):
        validate_dataset(tmp_path)


def test_duplicate_rejected(tmp_path):
    make_data(tmp_path)
    (tmp_path / "Adelie" / "0.png").replace(tmp_path / "Adelie" / "copy.png")
    (tmp_path / "Adelie" / "0.png").write_bytes((tmp_path / "Adelie" / "copy.png").read_bytes())
    with pytest.raises(DatasetError, match="duplicate"):
        validate_dataset(tmp_path)
