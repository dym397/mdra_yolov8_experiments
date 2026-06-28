#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mdra.utils.env_utils import collect_env_info, format_env_report
from mdra.utils.io_utils import save_json, write_text
from mdra.utils.path_utils import safe_mkdir, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Linux, Python, CUDA, GPU, and package information.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory for experiment outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = safe_mkdir(args.output_root / "env_reports")
    stamp = timestamp()
    text_path = output_dir / f"env_report_{stamp}.txt"
    json_path = output_dir / f"env_report_{stamp}.json"
    counter = 1
    while text_path.exists() or json_path.exists():
        text_path = output_dir / f"env_report_{stamp}_{counter:03d}.txt"
        json_path = output_dir / f"env_report_{stamp}_{counter:03d}.json"
        counter += 1

    info = collect_env_info()
    report = format_env_report(info)
    write_text(text_path, report + "\n")
    save_json(info, json_path)
    print(report)
    print(f"Saved text report: {text_path}")
    print(f"Saved JSON report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

