"""Sum CPUQuota values in a client CPU plan CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sum cpu_quota values from a plan CSV.")
    parser.add_argument("plan_csv", help="Path to plan CSV")
    parser.add_argument("--mode", default=None, help="Only include rows with this mode")
    parser.add_argument("--quota-column", default="cpu_quota")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = Path(args.plan_csv)
    total = 0.0
    rows = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if args.quota_column not in (reader.fieldnames or []):
            raise SystemExit(f"{path} missing column {args.quota_column!r}")
        for row in reader:
            if args.mode is not None and row.get("mode") != args.mode:
                continue
            raw = str(row.get(args.quota_column, "")).strip().rstrip("%")
            if not raw:
                continue
            total += float(raw)
            rows += 1

    print(f"file={path}")
    if args.mode is not None:
        print(f"mode={args.mode}")
    print(f"clients={rows}")
    print(f"cpu_quota_sum={total:.2f}%")
    print(f"cpu_core_sum={total / 100.0:.4f}")
    if rows:
        print(f"cpu_quota_avg={total / rows:.2f}%")


if __name__ == "__main__":
    main()
