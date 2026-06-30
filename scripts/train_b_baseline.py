#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mdra.engine.b_trainer import BExperimentTrainer, dry_run_b_pipeline
from mdra.experiments.baselines import apply_b_overrides, load_b_experiment_config, resolve_config_paths
from mdra.utils.io_utils import save_json
from mdra.utils.path_utils import safe_mkdir, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a unified B1-B5 M3FD baseline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vis-dir", type=Path, default=None)
    parser.add_argument("--ir-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--effective-batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="One real batch through load/forward/loss/backward on --device; does not run an epoch.",
    )
    return parser.parse_args()


def resolved_config(args: argparse.Namespace) -> dict:
    config = load_b_experiment_config(args.config)
    config = apply_b_overrides(
        config,
        {
            "experiment_id": args.experiment_id,
            "epochs": args.epochs,
            "batch": args.batch,
            "effective_batch": args.effective_batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "workers": args.workers,
            "seed": args.seed,
            "amp": args.amp,
            "pretrained": str(args.pretrained) if args.pretrained else None,
        },
    )
    config = resolve_config_paths(config, PROJECT_ROOT)
    config.update(
        {
            "data_root": str(args.data_root.expanduser().resolve()),
            "split_dir": str(args.split_dir.expanduser().resolve()),
            "output_root": str(args.output_root.expanduser().resolve()),
            "vis_dir": str(args.vis_dir) if args.vis_dir else None,
            "ir_dir": str(args.ir_dir) if args.ir_dir else None,
            "label_dir": str(args.label_dir) if args.label_dir else None,
        }
    )
    return config


def main() -> int:
    args = parse_args()
    config = resolved_config(args)
    if args.dry_run:
        result = dry_run_b_pipeline(config)
        output = safe_mkdir(args.output_root / "b_dry_runs") / f"{config['experiment_id']}_{timestamp()}.json"
        save_json(result, output)
        print(result)
        print(f"Saved dry-run report: {output}")
        return 0
    trainer = BExperimentTrainer(
        config,
        project_root=PROJECT_ROOT,
        output_root=args.output_root,
        resume=args.resume,
    )
    summary = trainer.train()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
