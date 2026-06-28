from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

from mdra.utils.io_utils import read_nonempty_lines, save_json, write_text
from mdra.utils.path_utils import require_writable_targets, safe_mkdir


SPLIT_NAMES = ("train", "val", "test")


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(value < 0 or value > 1 for value in ratios):
        raise ValueError("split ratios must each be within [0, 1]")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"split ratios must sum to 1.0, got {sum(ratios):.12f}")
    if train_ratio == 0 or val_ratio == 0:
        raise ValueError("train and validation ratios must be positive")


def split_sample_ids(
    sample_ids: Iterable[str],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    validate_ratios(train_ratio, val_ratio, test_ratio)
    ordered = sorted(dict.fromkeys(sample_ids))
    if not ordered:
        raise ValueError("cannot split an empty sample list")
    rng = random.Random(seed)
    rng.shuffle(ordered)

    total = len(ordered)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    if train_count == 0 or val_count == 0 or (test_ratio > 0 and test_count == 0):
        raise ValueError(
            f"sample count {total} is too small for ratios "
            f"{train_ratio}/{val_ratio}/{test_ratio}"
        )
    return {
        "train": ordered[:train_count],
        "val": ordered[train_count : train_count + val_count],
        "test": ordered[train_count + val_count :],
    }


def sample_subset(sample_ids: Iterable[str], count: int, seed: int) -> list[str]:
    values = sorted(dict.fromkeys(sample_ids))
    if count < 0:
        raise ValueError("subset count must be non-negative")
    if count > len(values):
        raise ValueError(f"requested {count} samples but split only contains {len(values)}")
    rng = random.Random(seed)
    rng.shuffle(values)
    return values[:count]


def read_split_dir(split_dir: str | Path) -> dict[str, list[str]]:
    root = Path(split_dir)
    result: dict[str, list[str]] = {}
    for name in SPLIT_NAMES:
        path = root / f"{name}.txt"
        if path.is_file():
            result[name] = read_nonempty_lines(path)
        elif name in {"train", "val"}:
            raise FileNotFoundError(f"required split file does not exist: {path}")
        else:
            result[name] = []
    return result


def write_split_dir(
    output_dir: str | Path,
    splits: dict[str, list[str]],
    metadata: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    root = Path(output_dir)
    paths = [root / f"{name}.txt" for name in SPLIT_NAMES] + [root / "split.json"]
    require_writable_targets(paths, overwrite=overwrite)
    safe_mkdir(root)
    for name in SPLIT_NAMES:
        content = "\n".join(splits.get(name, []))
        if content:
            content += "\n"
        write_text(root / f"{name}.txt", content, overwrite=overwrite)
    save_json(metadata, root / "split.json", overwrite=overwrite)
    return root

