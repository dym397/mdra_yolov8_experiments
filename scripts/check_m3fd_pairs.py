#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mdra.data.m3fd import audit_m3fd
from mdra.utils.io_utils import save_csv, save_json
from mdra.utils.path_utils import require_writable_targets, safe_mkdir


REPORT_FIELDS = (
    "sample_id",
    "valid",
    "visible_path",
    "infrared_path",
    "label_path",
    "visible_width",
    "visible_height",
    "infrared_width",
    "infrared_height",
    "num_objects",
    "empty_label",
    "issues",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit M3FD visible/infrared/YOLO-label pairing.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vis-dir", type=Path, default=None, help="Visible directory, absolute or relative to data-root.")
    parser.add_argument("--ir-dir", type=Path, default=None, help="Infrared directory, absolute or relative to data-root.")
    parser.add_argument("--label-dir", type=Path, default=None, help="Label directory, absolute or relative to data-root.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--no-check-box-corners", action="store_true", help="Only check the five normalized fields, not derived box corners.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true", help="Return exit code 2 if invalid samples are found.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_root / "data_checks"
    json_path = output_dir / "m3fd_pair_check_report.json"
    csv_path = output_dir / "m3fd_pair_check_report.csv"
    invalid_path = output_dir / "missing_or_invalid_files.csv"
    require_writable_targets([json_path, csv_path, invalid_path], overwrite=args.overwrite)
    safe_mkdir(output_dir)

    report = audit_m3fd(
        data_root=args.data_root,
        vis_dir=args.vis_dir,
        ir_dir=args.ir_dir,
        label_dir=args.label_dir,
        num_classes=args.num_classes,
        check_box_corners=not args.no_check_box_corners,
    )
    invalid_rows = [row for row in report["records"] if not row["valid"]]
    save_json(report, json_path, overwrite=args.overwrite)
    save_csv(report["records"], csv_path, fieldnames=REPORT_FIELDS, overwrite=args.overwrite)
    save_csv(invalid_rows, invalid_path, fieldnames=REPORT_FIELDS, overwrite=args.overwrite)

    summary = report["summary"]
    print("M3FD pair audit summary")
    print(f"  valid samples:   {summary['valid_samples']}")
    print(f"  invalid samples: {summary['invalid_samples']}")
    print(f"  visible images:  {summary['visible_images']}")
    print(f"  infrared images: {summary['infrared_images']}")
    print(f"  label files:     {summary['label_files']}")
    print(f"  issue counts:    {summary['issue_counts']}")
    print(f"Reports saved to: {output_dir}")
    if args.fail_on_error and summary["invalid_samples"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

