from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .path_utils import safe_mkdir


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_text(
    path: str | Path,
    text: str,
    *,
    overwrite: bool = False,
) -> Path:
    target = Path(path)
    safe_mkdir(target.parent)
    mode = "w" if overwrite else "x"
    with target.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return target


def save_json(
    obj: Any,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
        default=_json_default,
    )
    return write_text(path, payload + "\n", overwrite=overwrite)


def save_csv(
    rows: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    fieldnames: Sequence[str] | None = None,
    overwrite: bool = False,
) -> Path:
    materialized = list(rows)
    if fieldnames is None:
        ordered: list[str] = []
        for row in materialized:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered

    target = Path(path)
    safe_mkdir(target.parent)
    mode = "w" if overwrite else "x"
    with target.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return target


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def save_yaml(
    obj: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    payload = yaml.safe_dump(
        dict(obj),
        allow_unicode=True,
        sort_keys=False,
    )
    return write_text(path, payload, overwrite=overwrite)


def read_nonempty_lines(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]

