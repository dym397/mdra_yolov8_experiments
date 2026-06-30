#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.utils.ops import non_max_suppression

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mdra.data.paired_dataset import PairedM3FDDataset, paired_collate_fn
from mdra.engine.b_trainer import resolve_device
from mdra.models.baselines import load_b_checkpoint_model
from mdra.utils.io_utils import save_json
from mdra.utils.path_utils import unique_experiment_dir


COLORS = (
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save prediction visualizations for an MDRA B1-B5 checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    return parser.parse_args()


def _display_image(image: torch.Tensor) -> np.ndarray:
    """Convert the letterboxed model tensor to an OpenCV BGR canvas."""
    rgb = image[:3].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((rgb * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def _draw(canvas: np.ndarray, detection: torch.Tensor, names: list[str]) -> None:
    for row in detection.detach().cpu().numpy():
        x1, y1, x2, y2, confidence, class_id = row[:6]
        class_index = int(class_id)
        color = COLORS[class_index % len(COLORS)]
        point1 = (int(round(x1)), int(round(y1)))
        point2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(canvas, point1, point2, color, 2, cv2.LINE_AA)
        label = f"{names[class_index]} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_top = max(point1[1] - text_height - 7, 0)
        cv2.rectangle(canvas, (point1[0], label_top), (point1[0] + text_width + 4, point1[1]), color, -1)
        cv2.putText(
            canvas,
            label,
            (point1[0] + 2, max(point1[1] - 4, text_height + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    if args.batch <= 0 or args.max_images <= 0:
        raise ValueError("batch and max-images must be positive")
    model, checkpoint = load_b_checkpoint_model(args.checkpoint)
    config = dict(checkpoint["config"])
    if args.data_root is not None:
        config["data_root"] = str(args.data_root.expanduser().resolve())
    if args.split_dir is not None:
        config["split_dir"] = str(args.split_dir.expanduser().resolve())
    device = resolve_device(args.device)
    model = model.to(device).eval()
    dataset = PairedM3FDDataset(
        data_root=config["data_root"],
        split_file=Path(config["split_dir"]) / f"{args.split}.txt",
        input_mode=config["variant"],
        imgsz=int(config["imgsz"]),
        vis_dir=config.get("vis_dir"),
        ir_dir=config.get("ir_dir"),
        label_dir=config.get("label_dir"),
        augment=False,
        hflip_prob=0.0,
        max_samples=args.max_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=paired_collate_fn,
    )
    run_dir = unique_experiment_dir(
        args.output_root / "b_predictions", f"{config['experiment_id']}_{args.split}"
    )
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True)
    names = list(config["class_names"])
    records: list[dict] = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(config.get("amp", True)) and device.type == "cuda"):
                output = model(images)
            prediction = output[0] if isinstance(output, tuple) else output
            detections = non_max_suppression(
                prediction,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                max_det=300,
                nc=len(names),
            )
            for index, detection in enumerate(detections):
                canvas = _display_image(batch["img"][index])
                _draw(canvas, detection, names)
                sample_id = batch["sample_id"][index]
                target = image_dir / f"{sample_id}.jpg"
                if not cv2.imwrite(str(target), canvas):
                    raise OSError(f"failed to write prediction image: {target}")
                records.append(
                    {
                        "sample_id": sample_id,
                        "image": str(target),
                        "detections": [
                            {
                                "xyxy": [float(value) for value in row[:4]],
                                "confidence": float(row[4]),
                                "class_id": int(row[5]),
                                "class_name": names[int(row[5])],
                            }
                            for row in detection.detach().cpu().tolist()
                        ],
                    }
                )

    summary = {
        "status": "completed",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "variant": config["variant"],
        "split": args.split,
        "device": str(device),
        "conf_thres": args.conf_thres,
        "iou_thres": args.iou_thres,
        "images": len(records),
        "predictions": records,
    }
    save_json(summary, run_dir / "predictions.json")
    print(f"Saved {len(records)} visualizations to {image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
