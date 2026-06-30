from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from mdra.data.paired_dataset import PairedM3FDDataset, letterbox_pair, paired_collate_fn


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "M3FD"
    for name in ("Vis", "Ir", "labels"):
        (root / name).mkdir(parents=True)
    for index in range(2):
        sample_id = f"{index:05d}"
        visible = np.zeros((40, 80, 3), dtype=np.uint8)
        visible[..., 1] = 100 + index
        infrared = np.full((40, 80), 150 + index, dtype=np.uint8)
        cv2.imwrite(str(root / "Vis" / f"{sample_id}.png"), visible)
        cv2.imwrite(str(root / "Ir" / f"{sample_id}.png"), infrared)
        (root / "labels" / f"{sample_id}.txt").write_text(
            "0 0.500000 0.500000 0.250000 0.500000\n", encoding="utf-8"
        )
    (root / "train.txt").write_text("00000\n00001\n", encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("mode", "channels"),
    [("visible", 3), ("infrared", 3), ("early_fusion", 4), ("lcmf", 4), ("lcmf_p2", 4)],
)
def test_paired_dataset_modalities(tmp_path: Path, mode: str, channels: int) -> None:
    root = _dataset_root(tmp_path)
    dataset = PairedM3FDDataset(
        data_root=root,
        split_file=root / "train.txt",
        input_mode=mode,
        imgsz=64,
        vis_dir="Vis",
        ir_dir="Ir",
        label_dir="labels",
        augment=False,
    )
    sample = dataset[0]
    assert sample["img"].shape == (channels, 64, 64)
    assert sample["img"].dtype == torch.float32
    assert sample["bboxes"].shape == (1, 4)
    assert torch.all((sample["bboxes"] >= 0) & (sample["bboxes"] <= 1))


def test_letterbox_pair_preserves_shared_geometry() -> None:
    visible = np.zeros((40, 80, 3), dtype=np.uint8)
    infrared = np.zeros((40, 80), dtype=np.uint8)
    boxes = np.asarray([[0.5, 0.5, 0.25, 0.5]], dtype=np.float32)
    vis_out, ir_out, transformed, meta = letterbox_pair(visible, infrared, boxes, 64)
    assert vis_out.shape[:2] == ir_out.shape[:2] == (64, 64)
    assert meta.ratio == pytest.approx(0.8)
    assert meta.pad_top == 16
    assert transformed[0].tolist() == pytest.approx([0.5, 0.5, 0.25, 0.25])


def test_paired_collate_builds_ultralytics_loss_batch(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    dataset = PairedM3FDDataset(
        data_root=root,
        split_file=root / "train.txt",
        input_mode="lcmf",
        imgsz=64,
        vis_dir="Vis",
        ir_dir="Ir",
        label_dir="labels",
    )
    batch = paired_collate_fn([dataset[0], dataset[1]])
    assert batch["img"].shape == (2, 4, 64, 64)
    assert batch["cls"].shape == (2, 1)
    assert batch["bboxes"].shape == (2, 4)
    assert batch["batch_idx"].tolist() == [0, 1]
