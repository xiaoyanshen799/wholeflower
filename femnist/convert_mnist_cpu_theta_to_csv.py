#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CPU_RE = re.compile(r"^cpu\s+([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)
LINE_RE = re.compile(
    r"Client\s+([^\s]+)\s+(?:num_examples\s+([0-9]+)\s+)?=>\s+theta\s*=\s*([0-9.]+)\s*,\s*k\s*=\s*([0-9.]+)",
    re.IGNORECASE,
)


def _normalize_ip(addr: str) -> str:
    # Accept formats like "ipv4:127.0.0.1:58434" or "127.0.0.1:58434"
    if addr.startswith("ipv4:"):
        addr = addr.split("ipv4:", 1)[1]
    if ":" in addr:
        addr = addr.rsplit(":", 1)[0]
    return addr


def parse_log(path: Path):
    current_cpu = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            cpu_m = CPU_RE.match(line)
            if cpu_m:
                current_cpu = cpu_m.group(1)
                continue
            m = LINE_RE.search(line)
            if not m:
                continue
            raw_addr, num_examples, theta, k = m.groups()
            ip = _normalize_ip(raw_addr)
            yield {
                "num_examples": num_examples,
                "theta": theta,
                "k": k,
                "cpu": current_cpu,
            }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert mnist_cpu_theta.log to CSV (columns: ip, theta, k, cpu)."
    )
    parser.add_argument("input", type=Path, help="Path to mnist_cpu_theta.log")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: same name with .csv)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_suffix(".csv")

    rows = list(parse_log(args.input))
    if not rows:
        raise SystemExit(f"No rows parsed from {args.input}")

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["num_examples", "theta", "k", "cpu"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
