"""Plot FEMNIST timing series from two PACER runs.

Example:

    python femnist/plot_two_run_client_times.py
    python femnist/plot_two_run_client_times.py --num-examples 623
    python femnist/plot_two_run_client_times.py --mode client --num-examples 623

By default the script compares:

    femnist/logs/pacer_times_235.csv
    femnist/logs/pacer_times_normal_243.csv

The default mode compares each run's per-round slowest client time.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_A = SCRIPT_DIR / "logs" / "pacer_times_235.csv"
DEFAULT_CSV_B = SCRIPT_DIR / "logs" / "pacer_times_normal_243.csv"


def parse_num_examples(value: str) -> int:
    return int(float(value.strip()))


def parse_seconds(row: dict[str, str], duration_column: str, include_comm: bool) -> float:
    try:
        seconds = float(row[duration_column])
    except KeyError as exc:
        raise ValueError(f"Missing required duration column {duration_column!r}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid {duration_column} value {row[duration_column]!r}") from exc

    if include_comm:
        for column in ("server_to_client_ms", "client_to_server_ms"):
            raw = row.get(column, "")
            if raw.strip():
                seconds += float(raw) / 1000.0
    return seconds


def common_num_examples(csv_a: Path, csv_b: Path) -> list[int]:
    return sorted(read_num_examples(csv_a) & read_num_examples(csv_b))


def read_num_examples(csv_path: Path) -> set[int]:
    values: set[int] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "num_examples" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} is missing required column 'num_examples'")
        for row in reader:
            raw = row.get("num_examples", "")
            if raw.strip():
                values.add(parse_num_examples(raw))
    return values


def read_series(
    csv_path: Path,
    *,
    num_examples: int,
    duration_column: str,
    include_comm: bool,
) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"server_round", "num_examples", duration_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing required column(s): {missing_text}")

        for row in reader:
            raw_examples = row.get("num_examples", "")
            if not raw_examples.strip():
                continue
            if parse_num_examples(raw_examples) != num_examples:
                continue

            raw_round = row.get("server_round", "")
            if not raw_round.strip():
                continue
            server_round = int(float(raw_round))
            seconds = parse_seconds(row, duration_column, include_comm)
            points.append((server_round, seconds))

    points.sort(key=lambda item: item[0])
    return points


def read_round_worst_series(
    csv_path: Path,
    *,
    duration_column: str,
    include_comm: bool,
) -> list[tuple[int, float]]:
    by_round: dict[int, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"server_round", duration_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing required column(s): {missing_text}")

        for row in reader:
            raw_round = row.get("server_round", "")
            if not raw_round.strip():
                continue
            raw_duration = row.get(duration_column, "")
            if not raw_duration.strip():
                continue

            server_round = int(float(raw_round))
            seconds = parse_seconds(row, duration_column, include_comm)
            by_round[server_round] = max(seconds, by_round.get(server_round, seconds))

    return sorted(by_round.items(), key=lambda item: item[0])


def infer_label(csv_path: Path) -> str:
    match = re.search(r"(\d+)$", csv_path.stem)
    if match:
        return f"run {match.group(1)}"
    return csv_path.stem


def default_output_path(
    csv_a: Path,
    csv_b: Path,
    *,
    mode: str,
    num_examples: int | None,
) -> Path:
    if mode == "client":
        name = f"compare_num_examples_{num_examples}_{csv_a.stem}_vs_{csv_b.stem}.png"
    else:
        name = f"compare_round_worst_{csv_a.stem}_vs_{csv_b.stem}.png"
    return csv_a.parent / name


def plot_two_runs(
    series_a: list[tuple[int, float]],
    series_b: list[tuple[int, float]],
    *,
    label_a: str,
    label_b: str,
    title: str,
    ylabel: str,
    x_axis: str,
    reference_line: float,
    output: Path,
    dpi: int,
) -> None:
    if not series_a:
        raise ValueError("No plottable rows found in first CSV")
    if not series_b:
        raise ValueError("No plottable rows found in second CSV")

    fig, ax = plt.subplots(figsize=(10, 5))
    rounds_a, values_a = zip(*series_a)
    rounds_b, values_b = zip(*series_b)
    if x_axis == "index":
        x_values_a = list(range(1, len(series_a) + 1))
        x_values_b = list(range(1, len(series_b) + 1))
        xlabel = "Round index"
    else:
        x_values_a = rounds_a
        x_values_b = rounds_b
        xlabel = "Round"

    ax.plot(
        x_values_a,
        values_a,
        marker="o",
        markersize=4,
        linewidth=1.2,
        color="tab:blue",
        label=label_a,
    )
    ax.plot(
        x_values_b,
        values_b,
        marker="s",
        markersize=4,
        linewidth=1.2,
        color="tab:orange",
        label=label_b,
    )
    ax.axhline(
        reference_line,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"D = {reference_line:.2f}s",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot timing over rounds from two PACER timing CSV files. The "
            "default mode compares each server_round's slowest client."
        )
    )
    parser.add_argument("--csv-a", type=Path, default=DEFAULT_CSV_A)
    parser.add_argument("--csv-b", type=Path, default=DEFAULT_CSV_B)
    parser.add_argument(
        "--mode",
        choices=["worst-round", "client"],
        default="worst-round",
        help=(
            "worst-round: per server_round max client time; client: same "
            "num_examples over rounds. Default: worst-round."
        ),
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        help="Client identifier for --mode client, e.g. 623 or 1170.",
    )
    parser.add_argument(
        "--duration-column",
        default="client_train_s",
        help="Timing column to plot in seconds. Default: client_train_s.",
    )
    parser.add_argument(
        "--include-comm",
        action="store_true",
        help=(
            "Add server_to_client_ms and client_to_server_ms to duration-column "
            "after converting them to seconds."
        ),
    )
    parser.add_argument("--label-a", help="Legend label for --csv-a.")
    parser.add_argument("--label-b", help="Legend label for --csv-b.")
    parser.add_argument(
        "--x-axis",
        choices=["server_round", "index"],
        default="server_round",
        help=(
            "Use CSV server_round values on the x-axis, or use 1..N observation "
            "index for each run. Default: server_round."
        ),
    )
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--reference-line",
        type=float,
        default=253,
        help="Value for the red dashed horizontal line. Default: 247.5.",
    )
    parser.add_argument(
        "--list-common",
        action="store_true",
        help="Print common num_examples values in both CSVs and exit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    csv_a = args.csv_a.expanduser().resolve()
    csv_b = args.csv_b.expanduser().resolve()

    if args.list_common:
        common = common_num_examples(csv_a, csv_b)
        print("common num_examples:")
        print(", ".join(str(value) for value in common))
        return

    if args.mode == "client":
        common = common_num_examples(csv_a, csv_b)
        if args.num_examples is None:
            print("common num_examples:")
            print(", ".join(str(value) for value in common))
            print("Pass --num-examples VALUE to generate the plot.")
            return

        if args.num_examples not in common:
            raise SystemExit(
                f"num_examples={args.num_examples} is not present in both CSV files. "
                f"Common values: {', '.join(str(value) for value in common)}"
            )

        series_a = read_series(
            csv_a,
            num_examples=args.num_examples,
            duration_column=args.duration_column,
            include_comm=args.include_comm,
        )
        series_b = read_series(
            csv_b,
            num_examples=args.num_examples,
            duration_column=args.duration_column,
            include_comm=args.include_comm,
        )
        title = f"num_examples={args.num_examples}: two PACER runs"
        warning_context = f"num_examples={args.num_examples}"
    else:
        series_a = read_round_worst_series(
            csv_a,
            duration_column=args.duration_column,
            include_comm=args.include_comm,
        )
        series_b = read_round_worst_series(
            csv_b,
            duration_column=args.duration_column,
            include_comm=args.include_comm,
        )
        title = "Per-round slowest client time: two PACER runs"
        warning_context = "per-round worst time"

    # label_a = args.label_a or infer_label(csv_a)
    label_a = "CPU = 9.8 cores (20 clients)"
    label_b = "CPU = 9.1 cores (20 clients)"
    # label_b = args.label_b or infer_label(csv_b)
    for label, series in ((label_a, series_a), (label_b, series_b)):
        if series:
            rounds = [server_round for server_round, _ in series]
            print(
                f"{label}: {len(series)} points, "
                f"server_round {min(rounds)}..{max(rounds)}"
            )
        else:
            print(f"{label}: 0 points")
    if len(series_a) != len(series_b):
        print(
            "Warning: the two runs have different point counts for "
            f"{warning_context}."
        )

    output = (
        args.output.expanduser().resolve()
        if args.output
        else default_output_path(
            csv_a,
            csv_b,
            mode=args.mode,
            num_examples=args.num_examples,
        )
    )
    ylabel = "T_Round (s)" if args.include_comm else f"{args.duration_column} (s)"
    plot_two_runs(
        series_a,
        series_b,
        label_a=label_a,
        label_b=label_b,
        title=title,
        ylabel=ylabel,
        x_axis=args.x_axis,
        reference_line=args.reference_line,
        output=output,
        dpi=args.dpi,
    )
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
