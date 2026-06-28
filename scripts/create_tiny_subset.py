#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mdra.data.split_utils import read_split_dir, sample_subset, write_split_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic tiny or small subsets from a fixed split.")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-train", type=int, required=True)
    parser.add_argument("--num-val", type=int, required=True)
    parser.add_argument("--num-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = read_split_dir(args.split_dir)
    subset = {
        "train": sample_subset(source["train"], args.num_train, args.seed),
        "val": sample_subset(source["val"], args.num_val, args.seed + 1),
        "test": sample_subset(source["test"], args.num_test, args.seed + 2),
    }
    metadata = {
        "split_type": "fixed/unified subset",
        "is_official_split": False,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_split_dir": str(args.split_dir.resolve()),
        "seed": args.seed,
        "counts": {name: len(values) for name, values in subset.items()},
        "samples": subset,
    }
    output_dir = write_split_dir(args.output_dir, subset, metadata, overwrite=args.overwrite)
    print("Created deterministic subset:")
    for name, values in subset.items():
        print(f"  {name}: {len(values)}")
    print(f"Saved to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

