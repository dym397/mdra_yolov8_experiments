#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from _bootstrap import PROJECT_ROOT
from mdra.engine.b_trainer import resolve_device
from mdra.experiments.baselines import load_b_experiment_config
from mdra.models.baselines import build_b_model
from mdra.utils.io_utils import save_json
from mdra.utils.path_utils import unique_experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Params/GFLOPs/latency/FPS for a B1-B5 model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_b_experiment_config(args.config)
    model, report = build_b_model(
        config["variant"], nc=config["nc"], class_names=config["class_names"], pretrained=None
    )
    channels = 3 if config["variant"] in {"visible", "infrared"} else 4
    if args.batch <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("batch and iterations must be positive; warmup must be non-negative")
    dummy = torch.zeros(args.batch, channels, args.imgsz, args.imgsz)
    model.eval()
    flops = None
    error = None
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = float(macs * 2)
    except Exception as exc:  # profiling should still report parameters
        error = f"{type(exc).__name__}: {exc}"

    device = resolve_device(args.device)
    model = model.to(device)
    timed_input = dummy.to(device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(timed_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(args.iterations):
            model(timed_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    images = args.batch * args.iterations
    latency_ms = 1000.0 * elapsed / images
    output = {
        "variant": config["variant"],
        "input_shape": list(dummy.shape),
        "parameters": report["parameters"],
        "GFLOPs": flops / 1e9 if flops is not None else None,
        "device": str(device),
        "batch": args.batch,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "latency_ms_per_image": latency_ms,
        "FPS": images / elapsed,
        "profiling_error": error,
        "note": (
            "FLOPs count uses two operations per MAC when thop is available. "
            "FPS is end-to-end model forward throughput for the requested batch and device; "
            "report batch=1 for the paper latency table."
        ),
    }
    target_dir = unique_experiment_dir(args.output_root / "b_profiles", config["experiment_id"])
    target = target_dir / "profile.json"
    save_json(output, target)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
