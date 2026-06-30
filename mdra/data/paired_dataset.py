from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from mdra.data.m3fd import IMAGE_SUFFIXES, LABEL_SUFFIXES, index_files, resolve_dataset_dir
from mdra.utils.io_utils import read_nonempty_lines
from mdra.utils.yolo_label_utils import read_yolo_label


INPUT_MODES = ("visible", "infrared", "early_fusion", "early_fusion_p2", "lcmf", "lcmf_p2")


@dataclass(frozen=True)
class LetterboxMeta:
    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    ratio: float
    pad_left: int
    pad_top: int


def _resize_and_pad(image: np.ndarray, size: int, ratio: float, left: int, top: int) -> np.ndarray:
    height, width = image.shape[:2]
    resized_width = max(1, int(round(width * ratio)))
    resized_height = max(1, int(round(height * ratio)))
    interpolation = cv2.INTER_LINEAR if ratio > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    bottom = size - resized_height - top
    right = size - resized_width - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114)


def letterbox_pair(
    visible_rgb: np.ndarray,
    infrared_gray: np.ndarray,
    boxes_xywhn: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LetterboxMeta]:
    """Apply one deterministic letterbox transform to both modalities and normalized boxes."""
    if visible_rgb.shape[:2] != infrared_gray.shape[:2]:
        raise ValueError(
            f"paired image size mismatch: visible={visible_rgb.shape[:2]}, infrared={infrared_gray.shape[:2]}"
        )
    height, width = visible_rgb.shape[:2]
    ratio = min(size / height, size / width)
    resized_width = max(1, int(round(width * ratio)))
    resized_height = max(1, int(round(height * ratio)))
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    visible_out = _resize_and_pad(visible_rgb, size, ratio, left, top)
    infrared_out = _resize_and_pad(infrared_gray, size, ratio, left, top)

    transformed = boxes_xywhn.astype(np.float32, copy=True)
    if transformed.size:
        centers_x = transformed[:, 0] * width * ratio + left
        centers_y = transformed[:, 1] * height * ratio + top
        widths = transformed[:, 2] * width * ratio
        heights = transformed[:, 3] * height * ratio
        transformed = np.stack(
            (centers_x / size, centers_y / size, widths / size, heights / size), axis=1
        ).astype(np.float32)
        transformed[:, 0:2] = np.clip(transformed[:, 0:2], 0.0, 1.0)
        transformed[:, 2:4] = np.clip(transformed[:, 2:4], 1e-6, 1.0)

    meta = LetterboxMeta(
        original_shape=(height, width),
        resized_shape=(size, size),
        ratio=ratio,
        pad_left=left,
        pad_top=top,
    )
    return visible_out, infrared_out, transformed, meta


def _read_pair(visible_path: Path, infrared_path: Path) -> tuple[np.ndarray, np.ndarray]:
    visible_bgr = cv2.imread(str(visible_path), cv2.IMREAD_COLOR)
    infrared = cv2.imread(str(infrared_path), cv2.IMREAD_GRAYSCALE)
    if visible_bgr is None:
        raise OSError(f"failed to read visible image: {visible_path}")
    if infrared is None:
        raise OSError(f"failed to read infrared image: {infrared_path}")
    visible_rgb = cv2.cvtColor(visible_bgr, cv2.COLOR_BGR2RGB)
    return visible_rgb, infrared


class PairedM3FDDataset(Dataset):
    """M3FD paired detection dataset with synchronized, modality-safe spatial transforms."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        split_file: str | Path,
        input_mode: str,
        imgsz: int = 640,
        vis_dir: str | Path | None = None,
        ir_dir: str | Path | None = None,
        label_dir: str | Path | None = None,
        augment: bool = False,
        hflip_prob: float = 0.5,
        max_samples: int | None = None,
    ) -> None:
        if input_mode not in INPUT_MODES:
            raise ValueError(f"unsupported input_mode={input_mode!r}; expected one of {INPUT_MODES}")
        if imgsz <= 0 or imgsz % 32 != 0:
            raise ValueError("imgsz must be a positive multiple of 32")
        if not 0.0 <= hflip_prob <= 1.0:
            raise ValueError("hflip_prob must be within [0, 1]")

        self.data_root = Path(data_root).expanduser().resolve()
        self.visible_dir = resolve_dataset_dir(self.data_root, vis_dir, "visible")
        self.infrared_dir = resolve_dataset_dir(self.data_root, ir_dir, "infrared")
        self.label_dir = resolve_dataset_dir(self.data_root, label_dir, "labels")
        self.input_mode = input_mode
        self.imgsz = int(imgsz)
        self.augment = bool(augment)
        self.hflip_prob = float(hflip_prob)

        sample_ids = read_nonempty_lines(split_file)
        if max_samples is not None:
            sample_ids = sample_ids[: int(max_samples)]
        if not sample_ids:
            raise ValueError(f"split contains no samples: {split_file}")

        visible_index, visible_duplicates = index_files(self.visible_dir, IMAGE_SUFFIXES)
        infrared_index, infrared_duplicates = index_files(self.infrared_dir, IMAGE_SUFFIXES)
        label_index, label_duplicates = index_files(self.label_dir, LABEL_SUFFIXES)
        duplicate_ids = set(visible_duplicates) | set(infrared_duplicates) | set(label_duplicates)
        missing: list[str] = []
        samples: list[tuple[str, Path, Path, Path]] = []
        for sample_id in sample_ids:
            if sample_id in duplicate_ids:
                missing.append(f"{sample_id}: duplicate stem")
                continue
            visible_path = visible_index.get(sample_id)
            infrared_path = infrared_index.get(sample_id)
            label_path = label_index.get(sample_id)
            if visible_path is None or infrared_path is None or label_path is None:
                missing.append(
                    f"{sample_id}: vis={visible_path is not None}, ir={infrared_path is not None}, "
                    f"label={label_path is not None}"
                )
                continue
            samples.append((sample_id, visible_path, infrared_path, label_path))
        if missing:
            preview = "\n  - ".join(missing[:20])
            raise ValueError(f"split contains invalid paired samples:\n  - {preview}")
        self.samples = samples

    @property
    def channels(self) -> int:
        return 3 if self.input_mode in {"visible", "infrared"} else 4

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id, visible_path, infrared_path, label_path = self.samples[index]
        visible, infrared = _read_pair(visible_path, infrared_path)
        records = read_yolo_label(label_path)
        classes = np.asarray([int(row["class_id"]) for row in records], dtype=np.float32).reshape(-1, 1)
        boxes = np.asarray(
            [
                [float(row["x_center"]), float(row["y_center"]), float(row["width"]), float(row["height"])]
                for row in records
            ],
            dtype=np.float32,
        ).reshape(-1, 4)

        visible, infrared, boxes, meta = letterbox_pair(visible, infrared, boxes, self.imgsz)
        flipped = self.augment and random.random() < self.hflip_prob
        if flipped:
            visible = np.ascontiguousarray(visible[:, ::-1])
            infrared = np.ascontiguousarray(infrared[:, ::-1])
            if boxes.size:
                boxes[:, 0] = 1.0 - boxes[:, 0]

        infrared_channel = infrared[..., None]
        if self.input_mode == "visible":
            image = visible
        elif self.input_mode == "infrared":
            image = np.repeat(infrared_channel, 3, axis=2)
        else:
            image = np.concatenate((visible, infrared_channel), axis=2)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
        return {
            "img": image_tensor,
            "cls": torch.from_numpy(classes),
            "bboxes": torch.from_numpy(boxes),
            "sample_id": sample_id,
            "visible_path": str(visible_path),
            "infrared_path": str(infrared_path),
            "original_shape": meta.original_shape,
            "ratio_pad": (meta.ratio, (meta.pad_left, meta.pad_top)),
            "flipped": flipped,
        }


def paired_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    images = torch.stack([sample["img"] for sample in batch], dim=0)
    classes: list[torch.Tensor] = []
    boxes: list[torch.Tensor] = []
    batch_indices: list[torch.Tensor] = []
    for index, sample in enumerate(batch):
        count = int(sample["cls"].shape[0])
        if count:
            classes.append(sample["cls"])
            boxes.append(sample["bboxes"])
            batch_indices.append(torch.full((count,), index, dtype=torch.long))
    return {
        "img": images,
        "cls": torch.cat(classes, dim=0) if classes else torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.cat(boxes, dim=0) if boxes else torch.zeros((0, 4), dtype=torch.float32),
        "batch_idx": torch.cat(batch_indices, dim=0) if batch_indices else torch.zeros((0,), dtype=torch.long),
        "sample_id": [sample["sample_id"] for sample in batch],
        "visible_path": [sample["visible_path"] for sample in batch],
        "infrared_path": [sample["infrared_path"] for sample in batch],
        "original_shape": [sample["original_shape"] for sample in batch],
        "ratio_pad": [sample["ratio_pad"] for sample in batch],
    }
