from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from mdra.utils.path_utils import resolve_path
from mdra.utils.yolo_label_utils import validate_yolo_label


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_SUFFIXES = {".txt"}

DIR_CANDIDATES = {
    "visible": (
        "visible",
        "Visible",
        "VIS",
        "vis",
        "rgb",
        "RGB",
        "images/visible",
        "images/vis",
        "images/rgb",
    ),
    "infrared": (
        "infrared",
        "Infrared",
        "IR",
        "ir",
        "thermal",
        "Thermal",
        "images/infrared",
        "images/ir",
        "images/thermal",
    ),
    "labels": ("labels", "Labels", "annotations/labels", "Annotations/labels"),
}


def resolve_dataset_dir(
    data_root: str | Path,
    explicit: str | Path | None,
    kind: str,
) -> Path:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist or is not a directory: {root}")
    if explicit is not None:
        target = resolve_path(root, explicit)
        if not target.is_dir():
            raise FileNotFoundError(f"{kind} directory does not exist: {target}")
        return target

    matches = [root / candidate for candidate in DIR_CANDIDATES[kind] if (root / candidate).is_dir()]
    unique = list(dict.fromkeys(path.resolve() for path in matches))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise FileNotFoundError(
            f"could not auto-detect {kind} directory below {root}; pass the corresponding --*-dir argument"
        )
    choices = ", ".join(str(path) for path in unique)
    raise ValueError(f"multiple candidate {kind} directories found: {choices}; pass an explicit directory")


def sample_id_for(path: Path, base_dir: Path) -> str:
    return path.relative_to(base_dir).with_suffix("").as_posix()


def index_files(
    directory: str | Path,
    suffixes: Iterable[str],
) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    base = Path(directory).resolve()
    allowed = {suffix.lower() for suffix in suffixes}
    grouped: dict[str, list[Path]] = {}
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            grouped.setdefault(sample_id_for(path, base), []).append(path.resolve())
    index = {sample_id: paths[0] for sample_id, paths in grouped.items()}
    duplicates = {sample_id: paths for sample_id, paths in grouped.items() if len(paths) > 1}
    return index, duplicates


def _display_path(path: Path | None, data_root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(data_root).as_posix()
    except ValueError:
        return str(path)


def _inspect_image(path: Path) -> tuple[tuple[int, int] | None, str | None]:
    try:
        with Image.open(path) as image:
            size = image.size
            image.verify()
        return size, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def audit_m3fd(
    *,
    data_root: str | Path,
    vis_dir: str | Path | None = None,
    ir_dir: str | Path | None = None,
    label_dir: str | Path | None = None,
    num_classes: int | None = 6,
    check_box_corners: bool = True,
) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    visible = resolve_dataset_dir(root, vis_dir, "visible")
    infrared = resolve_dataset_dir(root, ir_dir, "infrared")
    labels = resolve_dataset_dir(root, label_dir, "labels")

    vis_index, vis_duplicates = index_files(visible, IMAGE_SUFFIXES)
    ir_index, ir_duplicates = index_files(infrared, IMAGE_SUFFIXES)
    label_index, label_duplicates = index_files(labels, LABEL_SUFFIXES)
    sample_ids = sorted(set(vis_index) | set(ir_index) | set(label_index))

    issue_counter: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    valid_sample_ids: list[str] = []

    for sample_id in sample_ids:
        vis_path = vis_index.get(sample_id)
        ir_path = ir_index.get(sample_id)
        label_path = label_index.get(sample_id)
        issues: list[str] = []

        if sample_id in vis_duplicates:
            issues.append("duplicate_visible_stem")
        if sample_id in ir_duplicates:
            issues.append("duplicate_infrared_stem")
        if sample_id in label_duplicates:
            issues.append("duplicate_label_stem")
        if vis_path is None:
            issues.append("missing_visible")
        if ir_path is None:
            issues.append("missing_infrared")
        if label_path is None:
            issues.append("missing_label")

        vis_size: tuple[int, int] | None = None
        ir_size: tuple[int, int] | None = None
        if vis_path is not None:
            vis_size, error = _inspect_image(vis_path)
            if error:
                issues.append(f"invalid_visible:{error}")
        if ir_path is not None:
            ir_size, error = _inspect_image(ir_path)
            if error:
                issues.append(f"invalid_infrared:{error}")
        if vis_size and ir_size and vis_size != ir_size:
            issues.append("visible_infrared_size_mismatch")

        label_validation: dict[str, Any] | None = None
        if label_path is not None:
            label_validation = validate_yolo_label(
                label_path,
                num_classes=num_classes,
                check_corners=check_box_corners,
            )
            for error in label_validation["errors"]:
                issues.append(f"invalid_label:{error}")

        for issue in issues:
            issue_counter[issue.split(":", 1)[0]] += 1
        is_valid = len(issues) == 0
        if is_valid:
            valid_sample_ids.append(sample_id)

        records.append(
            {
                "sample_id": sample_id,
                "valid": is_valid,
                "visible_path": _display_path(vis_path, root),
                "infrared_path": _display_path(ir_path, root),
                "label_path": _display_path(label_path, root),
                "visible_width": vis_size[0] if vis_size else "",
                "visible_height": vis_size[1] if vis_size else "",
                "infrared_width": ir_size[0] if ir_size else "",
                "infrared_height": ir_size[1] if ir_size else "",
                "num_objects": label_validation["num_objects"] if label_validation else "",
                "empty_label": label_validation["empty"] if label_validation else "",
                "issues": " | ".join(issues),
            }
        )

    summary = {
        "total_union_samples": len(sample_ids),
        "visible_images": len(vis_index),
        "infrared_images": len(ir_index),
        "label_files": len(label_index),
        "valid_samples": len(valid_sample_ids),
        "invalid_samples": len(sample_ids) - len(valid_sample_ids),
        "issue_counts": dict(sorted(issue_counter.items())),
        "num_classes_checked": num_classes,
        "box_corner_check_enabled": check_box_corners,
    }
    return {
        "data_root": str(root),
        "visible_dir": str(visible),
        "infrared_dir": str(infrared),
        "label_dir": str(labels),
        "summary": summary,
        "valid_sample_ids": valid_sample_ids,
        "records": records,
        "duplicate_files": {
            "visible": {key: [str(path) for path in value] for key, value in vis_duplicates.items()},
            "infrared": {key: [str(path) for path in value] for key, value in ir_duplicates.items()},
            "labels": {key: [str(path) for path in value] for key, value in label_duplicates.items()},
        },
    }

