#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mdra.data.m3fd import audit_m3fd
from mdra.data.split_utils import split_sample_ids, write_split_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible fixed/unified M3FD split.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vis-dir", type=Path, default=None)
    parser.add_argument("--ir-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--allow-invalid", action="store_true", help="Split only valid samples even when the audit finds invalid entries.")
    parser.add_argument("--no-check-box-corners", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_m3fd(
        data_root=args.data_root,
        vis_dir=args.vis_dir,
        ir_dir=args.ir_dir,
        label_dir=args.label_dir,
        num_classes=args.num_classes,
        check_box_corners=not args.no_check_box_corners,
    )
    invalid_count = audit["summary"]["invalid_samples"]
    if invalid_count and not args.allow_invalid:
        raise RuntimeError(
            f"data audit found {invalid_count} invalid samples; run check_m3fd_pairs.py and fix them, "
            "or explicitly pass --allow-invalid to split only valid samples"
        )

    splits = split_sample_ids(
        audit["valid_sample_ids"],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    metadata = {
        "split_type": "fixed/unified split",
        "is_official_split": False,
        "generated_at": datetime.now().astimezone().isoformat(),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "visible_dir": audit["visible_dir"],
        "infrared_dir": audit["infrared_dir"],
        "label_dir": audit["label_dir"],
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "counts": {name: len(values) for name, values in splits.items()},
        "source_audit_summary": audit["summary"],
        "samples": splits,
    }
    output_dir = write_split_dir(args.output_dir, splits, metadata, overwrite=args.overwrite)
    print("Created fixed/unified split (not labeled as official):")
    for name, values in splits.items():
        print(f"  {name}: {len(values)}")
    print(f"Saved to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

