from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from mdra.utils.io_utils import load_yaml


DEFAULT_CLASS_NAMES = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]


def load_visible_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    config.setdefault("experiment_id", Path(path).stem)
    config.setdefault("modality", "visible")
    config.setdefault("model", "yolov8s.pt")
    config.setdefault("epochs", 2)
    config.setdefault("batch", 4)
    config.setdefault("imgsz", 640)
    config.setdefault("device", "0")
    config.setdefault("workers", 4)
    config.setdefault("amp", True)
    config.setdefault("seed", 42)
    config.setdefault("deterministic", True)
    config.setdefault("link_mode", "symlink")
    config.setdefault("predict_after_train", True)
    config.setdefault("class_names", DEFAULT_CLASS_NAMES)
    if config["modality"] != "visible":
        raise NotImplementedError(
            "phase 6.1 only implements modality=visible; infrared and fusion modes are reserved"
        )
    if not isinstance(config["class_names"], list) or not config["class_names"]:
        raise ValueError("class_names must be a non-empty YAML list")
    return config


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    return resolved


def last_results_row(path: str | Path) -> dict[str, str] | None:
    results_path = Path(path)
    if not results_path.is_file():
        return None
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None

