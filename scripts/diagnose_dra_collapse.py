#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from _bootstrap import PROJECT_ROOT
from mdra.data.edge_targets import build_dra_supervision
from mdra.engine.b_trainer import build_b_eval_dataloader, resolve_device
from mdra.models.baselines import load_b_checkpoint_model
from mdra.utils.io_utils import save_json
from mdra.utils.path_utils import unique_experiment_dir


THRESHOLDS = (0.0, 0.01, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose sparse-target collapse in a trained DRA head.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vis-dir", type=Path, default=None)
    parser.add_argument("--ir-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--visualize-images", type=int, default=16)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


def _empty_stats() -> dict[str, Any]:
    return {
        "images": 0,
        "empty_images": 0,
        "supervised_cells": 0,
        "target_sum": 0.0,
        "positive_counts": {str(value): 0 for value in THRESHOLDS},
        "zero_l1_per_image": [],
    }


def _update_target_stats(stats: dict[str, Any], target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    selected = target[mask > 0.5].float()
    stats["images"] += 1
    if selected.numel() == 0:
        stats["empty_images"] += 1
        stats["zero_l1_per_image"].append(0.0)
        return {"cells": 0, "zero_l1": 0.0, **{f"positive_{v}": 0.0 for v in THRESHOLDS}}
    cells = int(selected.numel())
    target_sum = float(selected.sum())
    zero_l1 = target_sum / cells
    stats["supervised_cells"] += cells
    stats["target_sum"] += target_sum
    stats["zero_l1_per_image"].append(zero_l1)
    result = {"cells": cells, "zero_l1": zero_l1}
    for threshold in THRESHOLDS:
        count = int((selected > threshold).sum())
        stats["positive_counts"][str(threshold)] += count
        result[f"positive_{threshold}"] = count / cells
    return result


def _finalize_target_stats(stats: dict[str, Any]) -> dict[str, Any]:
    cells = max(int(stats["supervised_cells"]), 1)
    per_image = stats.pop("zero_l1_per_image")
    stats["global_zero_prediction_l1"] = stats["target_sum"] / cells
    stats["mean_per_image_zero_prediction_l1"] = sum(per_image) / max(len(per_image), 1)
    stats["positive_ratios"] = {
        key: value / cells for key, value in stats.pop("positive_counts").items()
    }
    return stats


def _new_prediction_stats() -> dict[str, Any]:
    return {
        "cells": 0,
        "abs_error_sum": 0.0,
        "prediction_sum": 0.0,
        "prediction_positive_counts": {str(value): 0 for value in THRESHOLDS},
        "per_image_l1": [],
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_x2": 0.0,
        "sum_y2": 0.0,
        "sum_xy": 0.0,
    }


def _update_prediction_stats(
    stats: dict[str, Any], prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    selected = mask > 0.5
    x = prediction[selected].float()
    y = target[selected].float()
    if x.numel() == 0:
        stats["per_image_l1"].append(0.0)
        return {"trained_l1": 0.0, "prediction_mean": 0.0}
    error = (x - y).abs()
    count = int(x.numel())
    stats["cells"] += count
    stats["abs_error_sum"] += float(error.sum())
    stats["prediction_sum"] += float(x.sum())
    stats["per_image_l1"].append(float(error.mean()))
    stats["sum_x"] += float(x.sum())
    stats["sum_y"] += float(y.sum())
    stats["sum_x2"] += float((x * x).sum())
    stats["sum_y2"] += float((y * y).sum())
    stats["sum_xy"] += float((x * y).sum())
    for threshold in THRESHOLDS:
        stats["prediction_positive_counts"][str(threshold)] += int((x > threshold).sum())
    return {"trained_l1": float(error.mean()), "prediction_mean": float(x.mean())}


def _finalize_prediction_stats(stats: dict[str, Any], zero_l1: float) -> dict[str, Any]:
    count = max(int(stats["cells"]), 1)
    mean_x = stats["sum_x"] / count
    mean_y = stats["sum_y"] / count
    covariance = stats["sum_xy"] / count - mean_x * mean_y
    variance_x = max(stats["sum_x2"] / count - mean_x * mean_x, 0.0)
    variance_y = max(stats["sum_y2"] / count - mean_y * mean_y, 0.0)
    denominator = math.sqrt(variance_x * variance_y)
    global_l1 = stats["abs_error_sum"] / count
    per_image = stats.pop("per_image_l1")
    result = {
        "supervised_cells": stats["cells"],
        "global_trained_prediction_l1": global_l1,
        "mean_per_image_trained_prediction_l1": sum(per_image) / max(len(per_image), 1),
        "global_prediction_mean": stats["prediction_sum"] / count,
        "target_prediction_pearson": covariance / denominator if denominator > 1e-12 else None,
        "trained_to_zero_l1_ratio": global_l1 / zero_l1 if zero_l1 > 0 else None,
        "prediction_positive_ratios": {
            key: value / count for key, value in stats["prediction_positive_counts"].items()
        },
    }
    return result


def _to_numpy(tensor: torch.Tensor):
    return tensor.detach().float().cpu().clamp(0, 1).numpy()


def _save_visualization(
    output_dir: Path,
    sample_id: str,
    image: torch.Tensor,
    full_target: torch.Tensor,
    target_mask: torch.Tensor,
    target_aware_target: torch.Tensor,
    prediction: torch.Tensor,
) -> str:
    error = (prediction - full_target).abs()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    panels = (
        (image[:3].permute(1, 2, 0), "Visible RGB", None),
        (image[3], "Infrared", "gray"),
        (full_target[0], "FullEdge target (P2)", "magma"),
        (prediction[0], "Trained DRA prediction", "magma"),
        (error[0], "Absolute error", "magma"),
        (target_mask[0], "TargetAware hard mask", "gray"),
        (target_aware_target[0], "TargetAware target (P2)", "magma"),
        ((prediction * target_mask)[0], "Prediction inside target mask", "magma"),
    )
    for axis, (panel, title, cmap) in zip(axes.flat, panels):
        axis.imshow(_to_numpy(panel), cmap=cmap, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    target = output_dir / f"{sample_id}.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return str(target)


def main() -> int:
    args = parse_args()
    if args.batch <= 0 or args.workers < 0 or args.visualize_images < 0:
        raise ValueError("batch must be positive; workers and visualize-images must be non-negative")
    model, checkpoint = load_b_checkpoint_model(args.checkpoint)
    if getattr(model, "dra_head", None) is None:
        raise ValueError("checkpoint does not contain a DRA head; use the training checkpoint, not best_inference.pt")
    config = dict(checkpoint["config"])
    config.update(
        {
            "data_root": str(args.data_root.expanduser().resolve()),
            "split_dir": str(args.split_dir.expanduser().resolve()),
            "vis_dir": str(args.vis_dir) if args.vis_dir else None,
            "ir_dir": str(args.ir_dir) if args.ir_dir else None,
            "label_dir": str(args.label_dir) if args.label_dir else None,
            "device": args.device,
            "workers": args.workers,
            "batch": args.batch,
        }
    )
    device = resolve_device(args.device)
    loader = build_b_eval_dataloader(config, split_name=args.split, max_samples=args.max_images)
    model = model.to(device).eval()
    output_dir = unique_experiment_dir(
        args.output_root / "dra_diagnostics", f"{config['experiment_id']}_{args.split}"
    )
    plot_dir = output_dir / "prediction_maps"
    plot_dir.mkdir(parents=True, exist_ok=False)
    mode_stats = {"full_edge": _empty_stats(), "target_aware": _empty_stats()}
    prediction_stats = _new_prediction_stats()
    per_image_rows: list[dict[str, Any]] = []
    visualization_paths: list[str] = []
    processed = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True)
            gpu_batch = {
                **batch,
                "bboxes": batch["bboxes"].to(device, non_blocking=True),
                "batch_idx": batch["batch_idx"].to(device, non_blocking=True),
            }
            _, prediction = model.forward_dra_diagnostic(images)
            output_size = tuple(prediction.shape[-2:])
            supervision = {
                mode: build_dra_supervision(
                    images,
                    gpu_batch,
                    output_size=output_size,
                    mode=mode,
                    bbox_expansion=float(config["dra_bbox_expansion"]),
                    edge_fusion=str(config["dra_edge_fusion"]),
                    edge_alpha=float(config["dra_edge_alpha"]),
                )
                for mode in ("full_edge", "target_aware")
            }
            for index, sample_id in enumerate(batch["sample_id"]):
                row: dict[str, Any] = {"sample_id": sample_id}
                for mode in ("full_edge", "target_aware"):
                    values = _update_target_stats(
                        mode_stats[mode], supervision[mode].target[index], supervision[mode].loss_mask[index]
                    )
                    row.update({f"{mode}_{key}": value for key, value in values.items()})
                row.update(
                    _update_prediction_stats(
                        prediction_stats,
                        prediction[index],
                        supervision["full_edge"].target[index],
                        supervision["full_edge"].loss_mask[index],
                    )
                )
                per_image_rows.append(row)
                if len(visualization_paths) < args.visualize_images:
                    visualization_paths.append(
                        _save_visualization(
                            plot_dir,
                            sample_id,
                            images[index],
                            supervision["full_edge"].target[index],
                            supervision["target_aware"].loss_mask[index],
                            supervision["target_aware"].target[index],
                            prediction[index],
                        )
                    )
                processed += 1
    finalized_modes = {mode: _finalize_target_stats(stats) for mode, stats in mode_stats.items()}
    prediction_summary = _finalize_prediction_stats(
        prediction_stats, finalized_modes["full_edge"]["global_zero_prediction_l1"]
    )
    summary = {
        "status": "completed",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "split": args.split,
        "images": processed,
        "edge_thresholds": list(THRESHOLDS),
        "supervision": finalized_modes,
        "trained_full_edge_dra": prediction_summary,
        "visualizations": visualization_paths,
        "interpretation_rule": (
            "A prediction/target correlation near zero together with prediction mean near the target mean and "
            "trained-to-zero L1 ratio near 1 indicates a near-constant or near-zero sparse-target solution."
        ),
    }
    save_json(summary, output_dir / "summary.json")
    with (output_dir / "per_image.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
