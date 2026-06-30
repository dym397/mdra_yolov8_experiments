#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mdra.engine.b_trainer import build_b_eval_dataloader, resolve_device
from mdra.engine.metrics import evaluate_detector
from mdra.models.baselines import load_b_checkpoint_model
from mdra.utils.io_utils import save_json
from mdra.utils.path_utils import unique_experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an MDRA B1-B5 checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, checkpoint = load_b_checkpoint_model(args.checkpoint)
    config = dict(checkpoint["config"])
    config["device"] = args.device
    if args.batch is not None:
        config["batch"] = args.batch
    if args.workers is not None:
        config["workers"] = args.workers
    device = resolve_device(args.device)
    model.to(device)
    eval_loader = build_b_eval_dataloader(config, split_name=args.split)
    run_dir = unique_experiment_dir(
        args.output_root / "b_validations", args.checkpoint.stem + f"_{args.split}_validation"
    )
    metrics = evaluate_detector(
        model,
        eval_loader,
        device=device,
        class_names=list(config["class_names"]),
        conf_thres=float(config["conf_thres"]),
        iou_thres=float(config["iou_thres"]),
        amp=bool(config.get("amp", True)) and device.type == "cuda",
        plot=True,
        save_dir=run_dir,
    )
    summary = {
        "status": "completed",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "variant": config["variant"],
        "split": args.split,
        "device": str(device),
        "metrics": metrics,
    }
    save_json(summary, run_dir / "summary.json")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
