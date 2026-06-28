from __future__ import annotations

import pytest

from mdra.data.split_utils import (
    read_split_dir,
    sample_subset,
    split_sample_ids,
    validate_ratios,
    write_split_dir,
)


def test_split_ratio_and_reproducibility():
    samples = [f"scene/sample_{index:02d}" for index in range(10)]
    first = split_sample_ids(
        samples,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
    )
    second = split_sample_ids(
        samples,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
    )

    assert {name: len(values) for name, values in first.items()} == {
        "train": 7,
        "val": 2,
        "test": 1,
    }
    assert first == second
    assert len(set(first["train"]) | set(first["val"]) | set(first["test"])) == 10
    assert not (set(first["train"]) & set(first["val"]))


def test_invalid_ratio_is_rejected():
    with pytest.raises(ValueError):
        validate_ratios(0.7, 0.2, 0.2)


def test_split_files_are_not_overwritten(tmp_path):
    splits = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}
    metadata = {"split_type": "fixed/unified split", "is_official_split": False}
    write_split_dir(tmp_path / "split", splits, metadata)

    with pytest.raises(FileExistsError):
        write_split_dir(tmp_path / "split", splits, metadata)

    loaded = read_split_dir(tmp_path / "split")
    assert loaded == splits


def test_subset_sampling():
    values = [str(index) for index in range(20)]
    subset = sample_subset(values, 5, seed=7)
    assert len(subset) == 5
    assert subset == sample_subset(values, 5, seed=7)
    with pytest.raises(ValueError):
        sample_subset(values, 21, seed=7)

