#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mdra.data.m3fd import IMAGE_SUFFIXES, LABEL_SUFFIXES, index_files, resolve_dataset_dir
from mdra.data.split_utils import read_split_dir
from mdra.utils.io_utils import save_csv, save_json
from mdra.utils.path_utils import require_writable_targets, safe_mkdir
from mdra.utils.yolo_label_utils import validate_yolo_label


DEFAULT_CLASS_NAMES = ("People", "Car", "Bus", "Motorcycle", "Lamp", "Truck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute M3FD class, bbox-area, empty-label, and scene statistics.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--vis-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--class-names", nargs="+", default=list(DEFAULT_CLASS_NAMES))
    parser.add_argument("--small-area", type=float, default=32**2, help="Small/medium pixel-area boundary.")
    parser.add_argument("--medium-area", type=float, default=96**2, help="Medium/large pixel-area boundary.")
    parser.add_argument("--scene-metadata", type=Path, default=None, help="Optional CSV with sample_id and scene columns.")
    parser.add_argument("--no-check-box-corners", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_scene_map(path: Path | None) -> tuple[dict[str, str], str]:
    if path is None:
        return {}, "not provided; all scenes reported as unknown"
    if not path.is_file():
        raise FileNotFoundError(f"scene metadata does not exist: {path}")
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"sample_id", "scene"}.issubset(reader.fieldnames):
            raise ValueError("scene metadata CSV must contain sample_id and scene columns")
        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            scene = (row.get("scene") or "unknown").strip() or "unknown"
            if sample_id:
                mapping[sample_id] = scene
    return mapping, str(path.resolve())


def size_group(area_pixels: float, small_area: float, medium_area: float) -> str:
    if area_pixels < small_area:
        return "small"
    if area_pixels < medium_area:
        return "medium"
    return "large"


def make_plots(
    *,
    class_rows: list[dict[str, Any]],
    bbox_rows: list[dict[str, Any]],
    class_names: list[str],
    plot_dir: Path,
    overwrite: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    safe_mkdir(plot_dir)
    class_plot = plot_dir / "class_counts.png"
    area_plot = plot_dir / "bbox_area_hist.png"
    require_writable_targets([class_plot, area_plot], overwrite=overwrite)

    totals = Counter()
    for row in class_rows:
        totals[int(row["class_id"])] += int(row["count"])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(class_names)))
    ax.bar(x, [totals[index] for index in x], color="#4472C4")
    ax.set_xticks(x, class_names, rotation=30, ha="right")
    ax.set_ylabel("Instance count")
    ax.set_title("M3FD class distribution")
    fig.tight_layout()
    fig.savefig(class_plot, dpi=180)
    plt.close(fig)

    areas = [float(row["area_pixels"]) for row in bbox_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    if areas:
        ax.hist(areas, bins=50, color="#70AD47", edgecolor="white")
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "No bounding boxes", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Bounding-box area (original-image pixels)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("M3FD bounding-box area distribution")
    fig.tight_layout()
    fig.savefig(area_plot, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.small_area <= 0 or args.medium_area <= args.small_area:
        raise ValueError("require 0 < small-area < medium-area")

    data_root = args.data_root.expanduser().resolve()
    visible_dir = resolve_dataset_dir(data_root, args.vis_dir, "visible")
    label_dir = resolve_dataset_dir(data_root, args.label_dir, "labels")
    vis_index, vis_duplicates = index_files(visible_dir, IMAGE_SUFFIXES)
    label_index, label_duplicates = index_files(label_dir, LABEL_SUFFIXES)
    if vis_duplicates or label_duplicates:
        raise ValueError("duplicate sample stems detected; run check_m3fd_pairs.py first")
    splits = read_split_dir(args.split_dir)
    scene_map, scene_source = load_scene_map(args.scene_metadata)

    output_dir = args.output_root / "dataset_stats"
    plot_dir = output_dir / "plots"
    outputs = [
        output_dir / "stats_summary.json",
        output_dir / "class_counts.csv",
        output_dir / "bbox_area_stats.csv",
        output_dir / "image_target_counts.csv",
        output_dir / "scene_counts.csv",
        plot_dir / "class_counts.png",
        plot_dir / "bbox_area_hist.png",
    ]
    require_writable_targets(outputs, overwrite=args.overwrite)
    safe_mkdir(output_dir)

    class_counts: dict[str, Counter[int]] = defaultdict(Counter)
    size_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scene_counts: dict[str, Counter[str]] = defaultdict(Counter)
    empty_counts: Counter[str] = Counter()
    bbox_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []

    for split_name, sample_ids in splits.items():
        for sample_id in sample_ids:
            image_path = vis_index.get(sample_id)
            label_path = label_index.get(sample_id)
            if image_path is None or label_path is None:
                raise FileNotFoundError(
                    f"sample {sample_id!r} from {split_name}.txt is missing visible image or label"
                )
            validation = validate_yolo_label(
                label_path,
                num_classes=len(args.class_names),
                check_corners=not args.no_check_box_corners,
            )
            if not validation["valid"]:
                raise ValueError(f"invalid label for {sample_id}: {validation['errors']}")
            with Image.open(image_path) as image:
                width, height = image.size
            object_count = int(validation["num_objects"])
            if object_count == 0:
                empty_counts[split_name] += 1
            scene = scene_map.get(sample_id, "unknown")
            scene_counts[split_name][scene] += 1
            image_rows.append(
                {
                    "split": split_name,
                    "sample_id": sample_id,
                    "width": width,
                    "height": height,
                    "num_objects": object_count,
                    "empty_label": object_count == 0,
                    "scene": scene,
                }
            )
            for object_index, record in enumerate(validation["records"]):
                class_id = int(record["class_id"])
                area_norm = float(record["area_norm"])
                area_pixels = area_norm * width * height
                group = size_group(area_pixels, args.small_area, args.medium_area)
                class_counts[split_name][class_id] += 1
                size_counts[split_name][group] += 1
                bbox_rows.append(
                    {
                        "split": split_name,
                        "sample_id": sample_id,
                        "object_index": object_index,
                        "class_id": class_id,
                        "class_name": args.class_names[class_id],
                        "image_width": width,
                        "image_height": height,
                        "area_norm": area_norm,
                        "area_pixels": area_pixels,
                        "size_group": group,
                    }
                )

    class_rows = [
        {
            "split": split_name,
            "class_id": class_id,
            "class_name": class_name,
            "count": class_counts[split_name][class_id],
        }
        for split_name in splits
        for class_id, class_name in enumerate(args.class_names)
    ]
    scene_rows = [
        {"split": split_name, "scene": scene, "sample_count": count}
        for split_name in splits
        for scene, count in sorted(scene_counts[split_name].items())
    ]
    summary = {
        "split_type": "fixed/unified split",
        "is_official_split": False,
        "data_root": str(data_root),
        "visible_dir": str(visible_dir),
        "label_dir": str(label_dir),
        "split_dir": str(args.split_dir.resolve()),
        "sample_counts": {name: len(values) for name, values in splits.items()},
        "object_counts": {
            name: sum(class_counts[name].values()) for name in splits
        },
        "empty_label_counts": dict(empty_counts),
        "size_thresholds_pixels": {
            "small_lt": args.small_area,
            "medium_lt": args.medium_area,
            "standard": "COCO pixel-area thresholds unless overridden",
        },
        "size_counts": {name: dict(size_counts[name]) for name in splits},
        "class_names": args.class_names,
        "scene_metadata_source": scene_source,
        "scene_note": (
            "Scene is unknown when no explicit metadata is provided; no subjective image-content "
            "classification is performed."
        ),
        "scene_counts": {name: dict(scene_counts[name]) for name in splits},
    }

    save_json(summary, output_dir / "stats_summary.json", overwrite=args.overwrite)
    save_csv(
        class_rows,
        output_dir / "class_counts.csv",
        fieldnames=("split", "class_id", "class_name", "count"),
        overwrite=args.overwrite,
    )
    save_csv(
        bbox_rows,
        output_dir / "bbox_area_stats.csv",
        fieldnames=(
            "split",
            "sample_id",
            "object_index",
            "class_id",
            "class_name",
            "image_width",
            "image_height",
            "area_norm",
            "area_pixels",
            "size_group",
        ),
        overwrite=args.overwrite,
    )
    save_csv(
        image_rows,
        output_dir / "image_target_counts.csv",
        fieldnames=("split", "sample_id", "width", "height", "num_objects", "empty_label", "scene"),
        overwrite=args.overwrite,
    )
    save_csv(
        scene_rows,
        output_dir / "scene_counts.csv",
        fieldnames=("split", "scene", "sample_count"),
        overwrite=args.overwrite,
    )
    make_plots(
        class_rows=class_rows,
        bbox_rows=bbox_rows,
        class_names=args.class_names,
        plot_dir=plot_dir,
        overwrite=args.overwrite,
    )

    print("Dataset statistics completed:")
    print(f"  sample counts: {summary['sample_counts']}")
    print(f"  object counts: {summary['object_counts']}")
    print(f"  size counts:   {summary['size_counts']}")
    print(f"  scene source:  {scene_source}")
    print(f"Saved to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
