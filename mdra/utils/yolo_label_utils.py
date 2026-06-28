from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def read_yolo_label(path: str | Path) -> list[dict[str, float | int]]:
    """Strictly parse a YOLO detection label file."""
    label_path = Path(path)
    records: list[dict[str, float | int]] = []
    with label_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{label_path}:{line_number}: expected 5 columns, got {len(parts)}"
                )
            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
            except ValueError as exc:
                raise ValueError(f"{label_path}:{line_number}: non-numeric value") from exc
            records.append(
                {
                    "line": line_number,
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "area_norm": width * height,
                }
            )
    return records


def validate_yolo_label(
    label_path: str | Path,
    *,
    num_classes: int | None = None,
    check_corners: bool = True,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate YOLO class/box rows and return machine-readable diagnostics."""
    path = Path(label_path)
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "valid": False,
        "empty": False,
        "num_objects": 0,
        "records": [],
        "errors": [],
    }
    if not path.is_file():
        result["errors"].append("label file does not exist")
        return result

    try:
        records = read_yolo_label(path)
    except (OSError, UnicodeError, ValueError) as exc:
        result["errors"].append(str(exc))
        return result

    result["records"] = records
    result["num_objects"] = len(records)
    result["empty"] = len(records) == 0

    for record in records:
        line = int(record["line"])
        class_id = int(record["class_id"])
        values = [
            float(record["x_center"]),
            float(record["y_center"]),
            float(record["width"]),
            float(record["height"]),
        ]
        if class_id < 0:
            result["errors"].append(f"line {line}: class_id must be non-negative")
        if num_classes is not None and class_id >= num_classes:
            result["errors"].append(
                f"line {line}: class_id {class_id} is outside [0, {num_classes - 1}]"
            )
        if not all(math.isfinite(value) for value in values):
            result["errors"].append(f"line {line}: bbox contains non-finite values")
            continue
        if not all(-tolerance <= value <= 1.0 + tolerance for value in values):
            result["errors"].append(f"line {line}: bbox fields must be within [0, 1]")
        x_center, y_center, width, height = values
        if width <= 0 or height <= 0:
            result["errors"].append(f"line {line}: bbox width and height must be positive")
        if check_corners and width > 0 and height > 0:
            x1, x2 = x_center - width / 2, x_center + width / 2
            y1, y2 = y_center - height / 2, y_center + height / 2
            if x1 < -tolerance or y1 < -tolerance or x2 > 1 + tolerance or y2 > 1 + tolerance:
                result["errors"].append(
                    f"line {line}: bbox corners exceed normalized image bounds"
                )

    result["valid"] = not result["errors"]
    return result

