from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from ultralytics.utils.ops import non_max_suppression


def xywhn_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    output = boxes.clone()
    output[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width
    output[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height
    output[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width
    output[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height
    return output


def process_batch(
    detections: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    iouv: torch.Tensor,
) -> torch.Tensor:
    """Return a detection-by-IoU-threshold correctness matrix using one-to-one matching."""
    correct = torch.zeros((detections.shape[0], iouv.numel()), dtype=torch.bool, device=detections.device)
    if detections.numel() == 0 or gt_boxes.numel() == 0:
        return correct
    iou = box_iou(gt_boxes, detections[:, :4])
    class_match = gt_classes[:, None] == detections[:, 5]
    for threshold_index, threshold in enumerate(iouv):
        matches = torch.where((iou >= threshold) & class_match)
        if matches[0].numel() == 0:
            continue
        match_data = torch.cat(
            (torch.stack(matches, dim=1), iou[matches[0], matches[1]][:, None]), dim=1
        ).detach().cpu().numpy()
        if match_data.shape[0] > 1:
            match_data = match_data[np.argsort(-match_data[:, 2])]
            match_data = match_data[np.unique(match_data[:, 1], return_index=True)[1]]
            match_data = match_data[np.argsort(-match_data[:, 2])]
            match_data = match_data[np.unique(match_data[:, 0], return_index=True)[1]]
        correct[match_data[:, 1].astype(int), threshold_index] = True
    return correct


@torch.no_grad()
def evaluate_detector(
    model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
    class_names: list[str],
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    amp: bool = False,
    plot: bool = False,
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    model.eval()
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    image_count = 0
    target_count = 0
    inference_seconds = 0.0
    confusion = (
        ConfusionMatrix(nc=len(class_names), conf=0.25, iou_thres=0.45, task="detect")
        if plot
        else None
    )

    for batch in dataloader:
        images = batch["img"].to(device, non_blocking=True)
        start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        cpu_start = time.perf_counter() if device.type == "cpu" else None
        if start is not None:
            start.record()
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            output = model(images)
        if end is not None:
            end.record()
            torch.cuda.synchronize(device)
            inference_seconds += start.elapsed_time(end) / 1000.0
        elif cpu_start is not None:
            inference_seconds += time.perf_counter() - cpu_start
        prediction = output[0] if isinstance(output, tuple) else output
        detections = non_max_suppression(
            prediction,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=300,
            nc=len(class_names),
        )
        height, width = images.shape[2:]
        for image_index, detection in enumerate(detections):
            mask = batch["batch_idx"] == image_index
            gt_classes = batch["cls"][mask].view(-1).to(device)
            gt_boxes = xywhn_to_xyxy(batch["bboxes"][mask].to(device), width, height)
            if confusion is not None:
                confusion.process_batch(detection if detection.shape[0] else None, gt_boxes, gt_classes)
            target_count += int(gt_classes.numel())
            image_count += 1
            if detection.shape[0]:
                correct = process_batch(detection, gt_boxes, gt_classes, iouv)
                stats.append(
                    (
                        correct.cpu().numpy(),
                        detection[:, 4].detach().cpu().numpy(),
                        detection[:, 5].detach().cpu().numpy(),
                        gt_classes.cpu().numpy(),
                    )
                )
            else:
                stats.append(
                    (
                        np.zeros((0, iouv.numel()), dtype=bool),
                        np.zeros((0,), dtype=np.float32),
                        np.zeros((0,), dtype=np.float32),
                        gt_classes.cpu().numpy(),
                    )
                )

    if not stats:
        raise RuntimeError("validation dataloader produced no batches")
    true_positives = np.concatenate([item[0] for item in stats], axis=0)
    confidences = np.concatenate([item[1] for item in stats], axis=0)
    predicted_classes = np.concatenate([item[2] for item in stats], axis=0)
    target_classes = np.concatenate([item[3] for item in stats], axis=0)

    per_class: dict[str, dict[str, float | int]] = {
        name: {
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "mAP75": 0.0,
            "mAP50_95": 0.0,
            "targets": 0,
        }
        for name in class_names
    }
    for class_index, name in enumerate(class_names):
        per_class[name]["targets"] = int((target_classes == class_index).sum())

    if confidences.size:
        target_dir = Path(save_dir) if save_dir else Path(".")
        if plot:
            target_dir.mkdir(parents=True, exist_ok=True)
        result = ap_per_class(
            true_positives,
            confidences,
            predicted_classes,
            target_classes,
            plot=plot,
            save_dir=target_dir,
            names={index: name for index, name in enumerate(class_names)},
        )
        _, _, precision, recall, _, ap, unique_classes = result[:7]
        for row, class_index in enumerate(unique_classes.tolist()):
            name = class_names[int(class_index)]
            per_class[name].update(
                {
                    "precision": float(precision[row]),
                    "recall": float(recall[row]),
                    "mAP50": float(ap[row, 0]),
                    "mAP75": float(ap[row, 5]),
                    "mAP50_95": float(ap[row].mean()),
                }
            )
        mean_precision = float(np.mean(precision)) if precision.size else 0.0
        mean_recall = float(np.mean(recall)) if recall.size else 0.0
        map50 = float(ap[:, 0].mean()) if ap.size else 0.0
        map75 = float(ap[:, 5].mean()) if ap.size else 0.0
        map50_95 = float(ap.mean()) if ap.size else 0.0
    else:
        mean_precision = mean_recall = map50 = map75 = map50_95 = 0.0

    confusion_values = None
    if confusion is not None:
        target_dir = Path(save_dir) if save_dir else Path(".")
        target_dir.mkdir(parents=True, exist_ok=True)
        confusion.plot(normalize=True, save_dir=target_dir, names=tuple(class_names))
        confusion.plot(normalize=False, save_dir=target_dir, names=tuple(class_names))
        confusion_values = confusion.matrix.tolist()
        labels = list(class_names) + ["background"]
        with (target_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["predicted\\true", *labels])
            for label, row in zip(labels, confusion.matrix.tolist()):
                writer.writerow([label, *row])
        with (target_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["class", "targets", "precision", "recall", "mAP50", "mAP75", "mAP50_95"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for name, values in per_class.items():
                writer.writerow({"class": name, **values})

    return {
        "images": image_count,
        "targets": target_count,
        "precision": mean_precision,
        "recall": mean_recall,
        "mAP50": map50,
        "mAP75": map75,
        "mAP50_95": map50_95,
        "per_class": per_class,
        "confusion_matrix": confusion_values,
        "inference_seconds": inference_seconds,
        "milliseconds_per_image": 1000.0 * inference_seconds / max(image_count, 1),
    }
