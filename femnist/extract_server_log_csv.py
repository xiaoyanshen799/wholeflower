#!/usr/bin/env python3
"""Extract per-client per-round durations from a server log into CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def make_record_re(metric_name: str) -> re.Pattern[str]:
    return re.compile(
        rf"Round\s+(?P<server_round>\d+)\s+client\s+"
        rf"(?:partition_id|client_id)=(?P<client_id>[^\s]+).*?"
        rf"{re.escape(metric_name)}=(?P<duration>{FLOAT_RE}|n/a)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse server-side Flower logs and export per-client durations as CSV."
        )
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        required=True,
        help="Path to the server log file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <log-file stem>_<metric>.csv",
    )
    parser.add_argument(
        "--metric",
        default="train_runtime",
        help="Metric name to export. Default: train_runtime",
    )
    return parser.parse_args()


def extract_rows(text: str, metric_name: str) -> list[tuple[int, str, float]]:
    clean_text = ANSI_RE.sub("", text)
    record_re = make_record_re(metric_name)
    rows: list[tuple[int, str, float]] = []

    for match in record_re.finditer(clean_text):
        duration_text = match.group("duration")
        if duration_text == "n/a":
            continue
        rows.append(
            (
                int(match.group("server_round")),
                match.group("client_id"),
                float(duration_text),
            )
        )

    rows.sort(key=lambda item: (item[0], item[1]))
    return rows


def main() -> int:
    args = parse_args()
    log_file = args.log_file.resolve()
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else log_file.with_name(f"{log_file.stem}_{args.metric}.csv")
    )

    text = log_file.read_text(encoding="utf-8", errors="replace")
    rows = extract_rows(text, args.metric)
    if not rows:
        raise SystemExit(
            f"No '{args.metric}' records found in {log_file}. "
            "Expected lines like: Round <n> client partition_id=<id> "
            f"{args.metric}=<value>"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["server_round", "client_id", "duration"])
        for server_round, client_id, duration in rows:
            writer.writerow([server_round, client_id, duration])

    print(f"Wrote {len(rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
