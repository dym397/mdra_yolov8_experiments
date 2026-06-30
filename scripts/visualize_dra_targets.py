#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _bootstrap import PROJECT_ROOT
from mdra.data.edge_targets import build_dra_supervision
from mdra.engine.b_trainer import build_b_eval_dataloader, resolve_device
from mdra.experiments.baselines import load_b_experiment_config, resolve_config_paths
from mdra.utils.io_utils import save_json
from mdra.utils.path_utils import unique_experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DRA supervision before training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vis-dir", type=Path, default=None)
    parser.add_argument("--ir-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


def _numpy_image(tensor: torch.Tensor):
    return tensor.detach().float().cpu().clamp(0, 1).numpy()


def main() -> int:
    args = parse_args()
    config = resolve_config_paths(load_b_experiment_config(args.config), PROJECT_ROOT)
    config.update(
        {
            "data_root": str(args.data_root.expanduser().resolve()),
            "split_dir": str(args.split_dir.expanduser().resolve()),
            "vis_dir": str(args.vis_dir) if args.vis_dir else None,
            "ir_dir": str(args.ir_dir) if args.ir_dir else None,
            "label_dir": str(args.label_dir) if args.label_dir else None,
            "device": args.device,
            "workers": 0,
            "batch": min(4, args.max_images),
        }
    )
    mode = "target_aware" if "target_aware" in config["variant"] else "full_edge"
    loader = build_b_eval_dataloader(config, split_name=args.split, max_samples=args.max_images)
    output_dir = unique_experiment_dir(
        args.output_root / "dra_target_visualizations", config["experiment_id"]
    )
    device = resolve_device(args.device)
    saved: list[str] = []
    count = 0
    for batch in loader:
        images = batch["img"].to(device)
        boxes = batch["bboxes"].to(device)
        indices = batch["batch_idx"].to(device)
        supervision = build_dra_supervision(
            images,
            {**batch, "bboxes": boxes, "batch_idx": indices},
            output_size=(images.shape[-2] // 4, images.shape[-1] // 4),
            mode=mode,
            bbox_expansion=float(config["dra_bbox_expansion"]),
            edge_fusion=str(config["dra_edge_fusion"]),
            edge_alpha=float(config["dra_edge_alpha"]),
        )
        for index, sample_id in enumerate(batch["sample_id"]):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
            panels = (
                (images[index, :3].permute(1, 2, 0), "Visible RGB", None),
                (images[index, 3], "Infrared", "gray"),
                (supervision.edge_visible[index, 0], "Visible Sobel", "magma"),
                (supervision.edge_infrared[index, 0], "Infrared Sobel", "magma"),
                (supervision.edge_multimodal[index, 0], "Multimodal edge", "magma"),
                (supervision.input_mask[index, 0], f"Input mask ({mode})", "gray"),
                (supervision.target[index, 0], "Aligned P2 target", "magma"),
                (supervision.loss_mask[index, 0], "Hard P2 loss mask", "gray"),
            )
            for axis, (panel, title, cmap) in zip(axes.flat, panels):
                axis.imshow(_numpy_image(panel), cmap=cmap, vmin=0, vmax=1)
                axis.set_title(title)
                axis.axis("off")
            target = output_dir / f"{sample_id}.png"
            fig.savefig(target, dpi=150)
            plt.close(fig)
            saved.append(str(target))
            count += 1
            if count >= args.max_images:
                break
        if count >= args.max_images:
            break
    report = {
        "experiment_id": config["experiment_id"],
        "mode": mode,
        "split": args.split,
        "images": count,
        "bbox_expansion": config["dra_bbox_expansion"],
        "edge_fusion": config["dra_edge_fusion"],
        "normalization": "fixed Sobel (abs(Gx)+abs(Gy))/8",
        "saved_files": saved,
    }
    save_json(report, output_dir / "report.json")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
