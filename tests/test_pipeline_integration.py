from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from mdra.data.yolo_view import build_visible_yolo_view


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_script(name: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / name), *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_data_audit_split_stats_and_subset_pipeline(tmp_path):
    data_root = tmp_path / "M3FD"
    for directory in ("visible", "infrared", "labels"):
        (data_root / directory).mkdir(parents=True)

    for index in range(10):
        sample = f"sample_{index:02d}"
        Image.new("RGB", (64, 48), color=(index, 20, 40)).save(
            data_root / "visible" / f"{sample}.png"
        )
        Image.new("L", (64, 48), color=index + 20).save(
            data_root / "infrared" / f"{sample}.png"
        )
        (data_root / "labels" / f"{sample}.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
        )

    output_root = tmp_path / "outputs"
    split_dir = tmp_path / "fixed_split"
    tiny_dir = tmp_path / "tiny_split"
    common = [
        "--data-root",
        str(data_root),
        "--vis-dir",
        "visible",
        "--ir-dir",
        "infrared",
        "--label-dir",
        "labels",
    ]
    run_script(
        "check_m3fd_pairs.py",
        *common,
        "--output-root",
        str(output_root),
        "--fail-on-error",
    )
    run_script(
        "make_fixed_split.py",
        *common,
        "--output-dir",
        str(split_dir),
        "--train-ratio",
        "0.6",
        "--val-ratio",
        "0.2",
        "--test-ratio",
        "0.2",
        "--seed",
        "42",
    )
    run_script(
        "dataset_stats.py",
        "--data-root",
        str(data_root),
        "--split-dir",
        str(split_dir),
        "--vis-dir",
        "visible",
        "--label-dir",
        "labels",
        "--output-root",
        str(output_root),
    )
    run_script(
        "create_tiny_subset.py",
        "--split-dir",
        str(split_dir),
        "--output-dir",
        str(tiny_dir),
        "--num-train",
        "2",
        "--num-val",
        "1",
        "--seed",
        "42",
    )

    pair_report = json.loads(
        (output_root / "data_checks" / "m3fd_pair_check_report.json").read_text(
            encoding="utf-8"
        )
    )
    split_report = json.loads((split_dir / "split.json").read_text(encoding="utf-8"))
    stats_report = json.loads(
        (output_root / "dataset_stats" / "stats_summary.json").read_text(encoding="utf-8")
    )
    tiny_report = json.loads((tiny_dir / "split.json").read_text(encoding="utf-8"))
    yolo_view = build_visible_yolo_view(
        data_root=data_root,
        vis_dir="visible",
        label_dir="labels",
        split_dir=tiny_dir,
        output_dir=tmp_path / "yolo_view",
        class_names=["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"],
        link_mode="copy",
    )

    assert pair_report["summary"]["valid_samples"] == 10
    assert split_report["counts"] == {"train": 6, "val": 2, "test": 2}
    assert stats_report["sample_counts"] == {"train": 6, "val": 2, "test": 2}
    assert stats_report["scene_counts"] == {
        "train": {"unknown": 6},
        "val": {"unknown": 2},
        "test": {"unknown": 2},
    }
    assert tiny_report["counts"] == {"train": 2, "val": 1, "test": 0}
    assert yolo_view["counts"] == {"train": 2, "val": 1, "test": 0}
    assert yolo_view["data_yaml"].is_file()
    assert yolo_view["first_val_image"].is_file()
    assert (output_root / "dataset_stats" / "plots" / "class_counts.png").is_file()
    assert (output_root / "dataset_stats" / "plots" / "bbox_area_hist.png").is_file()
