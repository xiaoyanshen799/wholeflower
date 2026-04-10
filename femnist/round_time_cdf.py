#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an empirical CDF of per-round time, where each round time is the "
            "slowest training time observed in that round."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Input CSV. Supports either a round log (for example speech_commom_target_85_1.csv) "
            "or a client summary snapshot (for example client_num_examples_target_85_1.csv)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "round-log", "summary-files"),
        default="auto",
        help=(
            "auto: detect from columns; round-log: group by round and take max(time); "
            "summary-files: treat matching *_N.csv files as per-round snapshots and take max(theta_latest)."
        ),
    )
    parser.add_argument(
        "--round-col",
        default="server_round",
        help="Round column used in round-log mode.",
    )
    parser.add_argument(
        "--time-col",
        default="client_train_s",
        help="Training time column used in round-log mode.",
    )
    parser.add_argument(
        "--client-col",
        default="client_id",
        help="Client identifier column used for exclusion filtering.",
    )
    parser.add_argument(
        "--summary-time-col",
        default="theta_latest",
        help="Per-client time column used in summary-files mode.",
    )
    parser.add_argument(
        "--exclude-client",
        action="append",
        default=None,
        help="Client identifier(s) to exclude. Repeat the flag or pass a comma-separated list.",
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=None,
        help="Optional minimum time filter applied before round max aggregation.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Optional maximum time filter applied before round max aggregation.",
    )
    parser.add_argument(
        "--round-times-out",
        type=Path,
        default=None,
        help="Optional CSV path for per-round slowest times.",
    )
    parser.add_argument(
        "--cdf-out",
        type=Path,
        default=None,
        help="Optional CSV path for the empirical CDF.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


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


def keep_time(value: float, min_time: float | None, max_time: float | None) -> bool:
    if min_time is not None and value < min_time:
        return False
    if max_time is not None and value > max_time:
        return False
    return True


def parse_excluded_clients(values: list[str] | None) -> set[str]:
    excluded: set[str] = set()
    for raw in values or []:
        for token in raw.split(","):
            token = token.strip()
            if token:
                excluded.add(token)
    return excluded


def should_exclude_row(
    row: dict[str, str],
    client_col: str,
    excluded_clients: set[str],
) -> bool:
    if not excluded_clients:
        return False
    return str(row.get(client_col, "")).strip() in excluded_clients


def infer_mode(
    mode: str,
    fieldnames: list[str],
    round_col: str,
    time_col: str,
    summary_time_col: str,
) -> str:
    if mode != "auto":
        return mode
    if round_col in fieldnames and time_col in fieldnames:
        return "round-log"
    if summary_time_col in fieldnames:
        return "summary-files"
    raise SystemExit(
        "Could not auto-detect mode from CSV columns. "
        f"Columns: {fieldnames}"
    )


def find_summary_snapshot_files(path: Path) -> list[Path]:
    match = re.match(r"^(?P<prefix>.*_)(?P<index>\d+)(?P<suffix>\.csv)$", path.name)
    if not match:
        return [path]
    pattern = f"{match.group('prefix')}*{match.group('suffix')}"
    candidates = []
    for candidate in path.parent.glob(pattern):
        suffix_match = re.match(
            rf"^{re.escape(match.group('prefix'))}(?P<index>\d+){re.escape(match.group('suffix'))}$",
            candidate.name,
        )
        if suffix_match:
            candidates.append((int(suffix_match.group("index")), candidate))
    if not candidates:
        return [path]
    candidates.sort(key=lambda item: item[0])
    return [candidate for _, candidate in candidates]


def round_sort_key(raw_round: str) -> tuple[int, float | str]:
    value = parse_float(raw_round)
    if value is not None:
        return (0, value)
    return (1, raw_round)


def build_round_times_from_round_log(
    rows: list[dict[str, str]],
    round_col: str,
    time_col: str,
    client_col: str,
    excluded_clients: set[str],
    min_time: float | None,
    max_time: float | None,
) -> list[tuple[str, float, str]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if should_exclude_row(row, client_col, excluded_clients):
            continue
        round_id = str(row.get(round_col, "")).strip()
        value = parse_float(row.get(time_col))
        if not round_id or value is None:
            continue
        if not keep_time(value, min_time, max_time):
            continue
        grouped.setdefault(round_id, []).append(value)

    round_times = []
    for round_id in sorted(grouped, key=round_sort_key):
        values = grouped[round_id]
        if values:
            round_times.append((round_id, max(values), "row-max"))
    return round_times


def build_round_times_from_summary_files(
    paths: list[Path],
    summary_time_col: str,
    client_col: str,
    excluded_clients: set[str],
    min_time: float | None,
    max_time: float | None,
) -> list[tuple[str, float, str]]:
    round_times: list[tuple[str, float, str]] = []
    for path in paths:
        _, rows = read_csv_rows(path)
        values = []
        for row in rows:
            if should_exclude_row(row, client_col, excluded_clients):
                continue
            value = parse_float(row.get(summary_time_col))
            if value is None:
                continue
            if not keep_time(value, min_time, max_time):
                continue
            values.append(value)
        if not values:
            continue
        round_label = path.stem
        match = re.match(r"^(?P<prefix>.*_)(?P<index>\d+)$", path.stem)
        if match:
            round_label = match.group("index")
        round_times.append((round_label, max(values), path.name))
    return round_times


def empirical_cdf(values: list[float]) -> list[tuple[float, float]]:
    ordered = sorted(values)
    total = len(ordered)
    return [(value, index / total) for index, value in enumerate(ordered, start=1)]


def nearest_rank(values: list[float], p: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(p * len(ordered)))
    return ordered[rank - 1]


def default_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}")


def write_round_times(path: Path, rows: list[tuple[str, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", "round_time", "source"])
        writer.writerows(rows)


def write_cdf(path: Path, rows: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "cdf"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input CSV does not exist: {args.input}")

    fieldnames, rows = read_csv_rows(args.input)
    mode = infer_mode(
        args.mode,
        fieldnames,
        args.round_col,
        args.time_col,
        args.summary_time_col,
    )
    excluded_clients = parse_excluded_clients(args.exclude_client)

    if mode == "round-log":
        round_times = build_round_times_from_round_log(
            rows,
            round_col=args.round_col,
            time_col=args.time_col,
            client_col=args.client_col,
            excluded_clients=excluded_clients,
            min_time=args.min_time,
            max_time=args.max_time,
        )
    else:
        snapshot_files = find_summary_snapshot_files(args.input)
        round_times = build_round_times_from_summary_files(
            snapshot_files,
            summary_time_col=args.summary_time_col,
            client_col=args.client_col,
            excluded_clients=excluded_clients,
            min_time=args.min_time,
            max_time=args.max_time,
        )

    if not round_times:
        raise SystemExit("No valid per-round times were produced.")

    round_time_values = [value for _, value, _ in round_times]
    cdf_rows = empirical_cdf(round_time_values)

    round_times_out = args.round_times_out or default_output_path(args.input, "_round_times.csv")
    cdf_out = args.cdf_out or default_output_path(args.input, "_round_time_cdf.csv")

    write_round_times(round_times_out, round_times)
    write_cdf(cdf_out, cdf_rows)

    print(f"mode={mode}")
    if excluded_clients:
        print(f"excluded_clients={','.join(sorted(excluded_clients))}")
    print(f"rounds={len(round_times)}")
    print(f"min={min(round_time_values):.6f}")
    print(f"max={max(round_time_values):.6f}")
    print(f"mean={mean(round_time_values):.6f}")
    print(f"p90={nearest_rank(round_time_values, 0.90):.6f}")
    print(f"p95={nearest_rank(round_time_values, 0.95):.6f}")
    print(f"p99={nearest_rank(round_time_values, 0.99):.6f}")
    print(f"round_times_csv={round_times_out}")
    print(f"cdf_csv={cdf_out}")


if __name__ == "__main__":
    main()
