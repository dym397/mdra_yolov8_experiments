from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from mdra.data.m3fd import IMAGE_SUFFIXES, LABEL_SUFFIXES, index_files, resolve_dataset_dir
from mdra.data.split_utils import read_split_dir
from mdra.utils.io_utils import save_json, save_yaml
from mdra.utils.path_utils import safe_mkdir


def _materialize(source: Path, destination: Path, mode: str) -> None:
    safe_mkdir(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"dataset-view destination already exists: {destination}")
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"unsupported link mode: {mode}")


def build_visible_yolo_view(
    *,
    data_root: str | Path,
    vis_dir: str | Path | None,
    label_dir: str | Path | None,
    split_dir: str | Path,
    output_dir: str | Path,
    class_names: Iterable[str],
    link_mode: str = "symlink",
) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    class_names = list(class_names)
    visible = resolve_dataset_dir(root, vis_dir, "visible")
    labels = resolve_dataset_dir(root, label_dir, "labels")
    vis_index, vis_duplicates = index_files(visible, IMAGE_SUFFIXES)
    label_index, label_duplicates = index_files(labels, LABEL_SUFFIXES)
    if vis_duplicates or label_duplicates:
        raise ValueError("duplicate sample stems detected; run check_m3fd_pairs.py first")

    splits = read_split_dir(split_dir)
    view_root = Path(output_dir).resolve()
    if view_root.exists():
        raise FileExistsError(f"YOLO dataset view already exists: {view_root}")
    safe_mkdir(view_root)

    counts: dict[str, int] = {}
    first_val_image: Path | None = None
    for split_name, sample_ids in splits.items():
        counts[split_name] = len(sample_ids)
        for sample_id in sample_ids:
            if sample_id not in vis_index:
                raise FileNotFoundError(f"visible image not found for sample_id={sample_id}")
            if sample_id not in label_index:
                raise FileNotFoundError(f"label not found for sample_id={sample_id}")
            source_image = vis_index[sample_id]
            source_label = label_index[sample_id]
            image_destination = view_root / "images" / split_name / f"{sample_id}{source_image.suffix.lower()}"
            label_destination = view_root / "labels" / split_name / f"{sample_id}.txt"
            _materialize(source_image, image_destination, link_mode)
            _materialize(source_label, label_destination, link_mode)
            if split_name == "val" and first_val_image is None:
                first_val_image = image_destination

    yaml_payload: dict[str, Any] = {
        "path": str(view_root),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    if counts.get("test", 0):
        yaml_payload["test"] = "images/test"
    data_yaml = save_yaml(yaml_payload, view_root / "data.yaml")
    manifest = {
        "data_root": str(root),
        "visible_dir": str(visible),
        "label_dir": str(labels),
        "source_split_dir": str(Path(split_dir).resolve()),
        "link_mode": link_mode,
        "counts": counts,
        "class_names": class_names,
    }
    save_json(manifest, view_root / "manifest.json")
    return {
        "view_root": view_root,
        "data_yaml": data_yaml,
        "counts": counts,
        "first_val_image": first_val_image,
    }
