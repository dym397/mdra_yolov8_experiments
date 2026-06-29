#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mdra.data.yolo_view import build_visible_yolo_view
from mdra.experiments.visible import apply_overrides, last_results_row, load_visible_config
from mdra.utils.env_utils import collect_env_info, format_env_report
from mdra.utils.io_utils import save_json, save_yaml, write_text
from mdra.utils.logging_utils import setup_logging
from mdra.utils.path_utils import unique_experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the phase-6.1 YOLOv8s Visible baseline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vis-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--modality", choices=("visible", "infrared", "early_fusion", "lcmf"), default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="Enable AMP.")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP.")
    parser.set_defaults(amp=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default=None)
    parser.add_argument("--resume", type=Path, default=None, metavar="LAST_PT", help="Resume the original Ultralytics run from last.pt.")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install project requirements before training."
        ) from exc
    return YOLO


def _save_run_context(run_dir: Path, config: dict[str, Any]) -> None:
    env_info = collect_env_info()
    save_json(env_info, run_dir / "environment.json")
    write_text(run_dir / "environment.txt", format_env_report(env_info) + "\n")
    save_yaml(config, run_dir / "resolved_config.yaml")
    write_text(run_dir / "command.txt", shlex.join(sys.argv) + "\n")


def _resume(args: argparse.Namespace, config: dict[str, Any]) -> int:
    checkpoint = args.resume.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
    log_dir = unique_experiment_dir(
        args.output_root / "resume_logs",
        f"{config['experiment_id']}_resume",
    )
    logger = setup_logging("train_visible_resume", log_file=log_dir / "resume.log", verbose=args.verbose)
    resume_context = dict(config)
    resume_context["resume_checkpoint"] = str(checkpoint)
    resume_context["resume_note"] = (
        "True resume intentionally continues the original Ultralytics run directory; "
        "this separate directory stores the resume request and environment only."
    )
    _save_run_context(log_dir, resume_context)
    YOLO = _import_yolo()
    logger.info("Resuming original run from %s", checkpoint)
    model = YOLO(str(checkpoint))
    model.train(resume=True)
    save_json({"status": "resume_completed", "checkpoint": str(checkpoint)}, log_dir / "summary.json")
    logger.info("Resume completed; audit log saved to %s", log_dir)
    return 0


def _train_kwargs(config: dict[str, Any], data_yaml: Path, run_dir: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(config["epochs"]),
        "batch": int(config["batch"]),
        "imgsz": int(config["imgsz"]),
        "device": str(config["device"]),
        "workers": int(config["workers"]),
        "amp": bool(config["amp"]),
        "seed": int(config["seed"]),
        "deterministic": bool(config.get("deterministic", True)),
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
        "save": True,
        "plots": True,
        "verbose": bool(config.get("ultralytics_verbose", True)),
    }
    optional_keys = (
        "patience",
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "close_mosaic",
        "cache",
    )
    for key in optional_keys:
        if key in config and config[key] is not None:
            kwargs[key] = config[key]
    return kwargs


def main() -> int:
    args = parse_args()
    config = load_visible_config(args.config)
    config = apply_overrides(
        config,
        {
            "experiment_id": args.experiment_id,
            "modality": args.modality,
            "model": args.model,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "workers": args.workers,
            "amp": args.amp,
            "seed": args.seed,
            "link_mode": args.link_mode,
        },
    )
    if config["modality"] != "visible":
        raise NotImplementedError("phase 6.1 only implements the Visible baseline")
    if args.skip_predict:
        config["predict_after_train"] = False
    config.update(
        {
            "config_source": str(args.config.resolve()),
            "data_root": str(args.data_root.expanduser().resolve()),
            "split_dir": str(args.split_dir.expanduser().resolve()),
            "visible_dir_argument": str(args.vis_dir) if args.vis_dir else None,
            "label_dir_argument": str(args.label_dir) if args.label_dir else None,
        }
    )

    if args.resume is not None:
        return _resume(args, config)

    run_dir = unique_experiment_dir(args.output_root / "experiments", str(config["experiment_id"]))
    logger = setup_logging("train_visible", log_file=run_dir / "run.log", verbose=args.verbose)
    logger.info("Created unique run directory: %s", run_dir)
    _save_run_context(run_dir, config)

    logger.info("Building a standard YOLO dataset view using %s links", config["link_mode"])
    view = build_visible_yolo_view(
        data_root=args.data_root,
        vis_dir=args.vis_dir,
        label_dir=args.label_dir,
        split_dir=args.split_dir,
        output_dir=run_dir / "dataset_view",
        class_names=config["class_names"],
        link_mode=str(config["link_mode"]),
    )
    logger.info("Dataset-view counts: %s", view["counts"])

    YOLO = _import_yolo()
    model = YOLO(str(config["model"]))
    train_kwargs = _train_kwargs(config, view["data_yaml"], run_dir)
    save_json(train_kwargs, run_dir / "train_arguments.json")
    logger.info("Starting YOLOv8s Visible training")
    model.train(**train_kwargs)

    weights_dir = run_dir / "weights"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"
    results_csv = run_dir / "results.csv"
    if not best_path.is_file() and last_path.is_file():
        best_path = last_path

    prediction_dir: Path | None = None
    if config.get("predict_after_train", True):
        source = view["first_val_image"]
        if source is None:
            raise RuntimeError("validation split is empty; cannot run post-training prediction")
        if not best_path.is_file():
            raise FileNotFoundError(f"trained checkpoint not found: {best_path}")
        prediction_dir = run_dir / "predictions" / "single_batch"
        logger.info("Running single-image prediction smoke test: %s", source)
        YOLO(str(best_path)).predict(
            source=str(source),
            save=True,
            project=str(run_dir / "predictions"),
            name="single_batch",
            exist_ok=False,
            device=str(config["device"]),
            imgsz=int(config["imgsz"]),
            verbose=False,
        )

    summary = {
        "status": "completed",
        "experiment_id": config["experiment_id"],
        "run_dir": str(run_dir),
        "modality": "visible",
        "fixed_or_unified_split": str(args.split_dir.resolve()),
        "dataset_view_counts": view["counts"],
        "best_checkpoint": str(best_path) if best_path.is_file() else None,
        "last_checkpoint": str(last_path) if last_path.is_file() else None,
        "results_csv": str(results_csv) if results_csv.is_file() else None,
        "last_results_row": last_results_row(results_csv),
        "prediction_dir": str(prediction_dir) if prediction_dir else None,
    }
    save_json(summary, run_dir / "run_summary.json")
    logger.info("Training completed. Summary: %s", run_dir / "run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
