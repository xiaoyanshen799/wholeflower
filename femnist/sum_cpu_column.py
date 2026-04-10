#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sum numeric values from a CSV column."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/xiaoyan/wholeflower/femnist/speech_comm/target_86_1.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--column",
        default="cpu",
        help="Column name to sum.",
    )
    return parser.parse_args()


def parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input CSV does not exist: {args.input}")

    total = 0.0
    count = 0

    with args.input.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if args.column not in (reader.fieldnames or []):
            raise SystemExit(
                f"Column '{args.column}' not found in {args.input}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            value = parse_float(row.get(args.column))
            if value is None:
                continue
            total += value
            count += 1

    print(f"input={args.input}")
    print(f"column={args.column}")
    print(f"count={count}")
    print(f"sum={total:.12f}")


if __name__ == "__main__":
    main()
