#!/usr/bin/env python3
"""Fit logistic CDFs for time and token metrics extracted from stream.log."""

from __future__ import annotations

import argparse
from collections import defaultdict
import re
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TIMING_RE = re.compile(rf"t_local_round_s\s*=\s*({FLOAT_RE})")
CLIENT_TIME_RE = re.compile(
    rf"Round\s+(\d+)\s+client\s+partition_id=(\d+).*?"
    rf"train_runtime=({FLOAT_RE}|n/a).*?"
    rf"t_local_round_s=({FLOAT_RE}|n/a)"
)


def make_metric_re(metric_name: str) -> re.Pattern[str]:
    return re.compile(rf"'{re.escape(metric_name)}'\s*:\s*'?({FLOAT_RE})'?")


def make_json_re(metric_name: str) -> re.Pattern[str]:
    return re.compile(rf'"{re.escape(metric_name)}"\s*:\s*({FLOAT_RE})')


def logistic_cdf(t: np.ndarray, theta: float, k: float) -> np.ndarray:
    """2-parameter logistic CDF."""
    return 1.0 / (1.0 + np.exp(-(t - theta) / k))


def empirical_cdf_y(n: int) -> np.ndarray:
    """Return stable plotting positions for fitting a continuous CDF."""
    return np.arange(1, n + 1, dtype=float) / float(n + 1)


def extract_timing_values(text: str, source: str) -> tuple[list[float], str]:
    clean_text = ANSI_RE.sub("", text)
    metric_re = make_metric_re("t_local_round_s")

    def from_timing() -> list[float]:
        return [float(m.group(1)) for m in TIMING_RE.finditer(clean_text)]

    def from_metric() -> list[float]:
        values: list[float] = []
        for line in clean_text.splitlines():
            if "Aggregated MetricRecord" not in line:
                continue
            match = metric_re.search(line)
            if match:
                values.append(float(match.group(1)))
        return values

    if source == "timing":
        return from_timing(), "timing"
    if source == "metric":
        return from_metric(), "metric"

    timing_values = from_timing()
    if timing_values:
        return timing_values, "timing"

    metric_values = from_metric()
    if metric_values:
        return metric_values, "metric"

    return [], "none"


def extract_metric_values(
    text: str,
    metric_name: str,
    trace_metric_name: str | None = None,
) -> tuple[list[float], str]:
    clean_text = ANSI_RE.sub("", text)
    metric_re = make_metric_re(metric_name)

    values: list[float] = []
    for line in clean_text.splitlines():
        if "Aggregated MetricRecord" not in line:
            continue
        match = metric_re.search(line)
        if match:
            values.append(float(match.group(1)))
    if values:
        return values, "metric"

    if trace_metric_name is not None:
        json_re = make_json_re(trace_metric_name)
        trace_values = [float(m.group(1)) for m in json_re.finditer(clean_text)]
        if trace_values:
            return trace_values, "trace"

    return [], "none"


def fit_logistic(values: list[float]) -> tuple[float, float, str]:
    if len(values) < 3:
        raise ValueError(f"need at least 3 values, got {len(values)}")

    x_data = np.sort(np.asarray(values, dtype=float))
    if float(np.max(x_data)) == float(np.min(x_data)):
        raise ValueError("all values are identical; cannot fit logistic CDF")

    y_data = empirical_cdf_y(len(x_data))

    theta0 = float(np.median(x_data))
    k0 = float((np.percentile(x_data, 75) - np.percentile(x_data, 25)) / 4.0)
    p0 = [theta0, max(k0, 0.1)]

    x_span = float(np.max(x_data) - np.min(x_data))
    k_upper = max(300.0, x_span, 10.0 * p0[1])
    bounds = ([float(np.min(x_data)), 0.005], [float(np.max(x_data)), k_upper])

    popt, _ = curve_fit(
        logistic_cdf,
        x_data,
        y_data,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
        loss="soft_l1",
    )
    theta_hat, k_hat = popt
    fit_status = "ok"
    if abs(float(k_hat) - k_upper) <= max(1e-6, 1e-3 * k_upper):
        fit_status = f"ok_hit_k_upper_bound({k_upper:.3f})"
    return float(theta_hat), float(k_hat), fit_status


def try_fit_logistic(values: list[float]) -> tuple[tuple[float, float] | None, str]:
    try:
        theta_hat, k_hat, fit_status = fit_logistic(values)
        return (theta_hat, k_hat), fit_status
    except (ValueError, RuntimeError) as exc:
        return None, str(exc)


def extract_client_time_series(
    text: str,
    metric_name: str,
) -> dict[int, list[tuple[int, float]]]:
    """Extract per-client per-round timing series from server-side logs."""
    clean_text = ANSI_RE.sub("", text)
    per_client: dict[int, list[tuple[int, float]]] = defaultdict(list)

    metric_index = {
        "train_runtime": 3,
        "t_local_round_s": 4,
    }[metric_name]

    for match in CLIENT_TIME_RE.finditer(clean_text):
        round_id = int(match.group(1))
        partition_id = int(match.group(2))
        value_str = match.group(metric_index)
        if value_str == "n/a":
            continue
        per_client[partition_id].append((round_id, float(value_str)))

    for partition_id in per_client:
        per_client[partition_id].sort(key=lambda item: item[0])
    return dict(sorted(per_client.items()))


def plot_fit(
    values: list[float],
    metric_label: str,
    plot_title: str,
    plot_file: Path,
    fit_params: tuple[float, float] | None,
    fit_status: str,
) -> None:
    """Plot empirical CDF points and fitted logistic CDF curve if available."""
    x_data = np.sort(np.asarray(values, dtype=float))
    y_data = np.arange(1, len(x_data) + 1, dtype=float) / float(len(x_data))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        x_data,
        y_data,
        s=22,
        alpha=0.85,
        label="Empirical points",
        color="#1f77b4",
    )

    if fit_params is not None:
        theta, k = fit_params
        x_grid = np.linspace(float(np.min(x_data)), float(np.max(x_data)), 300)
        y_fit = logistic_cdf(x_grid, theta, k)
        ax.plot(
            x_grid,
            y_fit,
            linewidth=2.2,
            label=f"Fitted logistic CDF (theta={theta:.3f}, k={k:.3f})",
            color="#d62728",
        )
    else:
        ax.axvline(
            float(x_data[0]),
            linewidth=2.0,
            color="#d62728",
            alpha=0.8,
            label=f"No logistic fit: {fit_status}",
        )

    ax.set_title(plot_title)
    ax.set_xlabel(metric_label)
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)


def plot_client_cdfs(
    series_by_client: dict[int, list[tuple[int, float]]],
    metric_label: str,
    plot_title: str,
    plot_file: Path,
) -> dict[int, tuple[tuple[float, float] | None, str]]:
    """Plot per-client empirical CDFs and logistic fits on a single figure."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fit_summary: dict[int, tuple[tuple[float, float] | None, str]] = {}

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, (partition_id, series) in enumerate(series_by_client.items()):
        values = [value for _, value in series]
        if not values:
            continue

        color = colors[idx % len(colors)] if colors else None
        x_data = np.sort(np.asarray(values, dtype=float))
        y_data = np.arange(1, len(x_data) + 1, dtype=float) / float(len(x_data))
        fit_params, fit_status = try_fit_logistic(values)
        fit_summary[partition_id] = (fit_params, fit_status)

        ax.scatter(
            x_data,
            y_data,
            s=18,
            alpha=0.45,
            color=color,
            label=f"client {partition_id} empirical",
        )

        if fit_params is not None:
            theta, k = fit_params
            x_grid = np.linspace(float(np.min(x_data)), float(np.max(x_data)), 300)
            y_fit = logistic_cdf(x_grid, theta, k)
            ax.plot(
                x_grid,
                y_fit,
                linewidth=2.0,
                color=color,
                label=f"client {partition_id} fit",
            )

    ax.set_title(plot_title)
    ax.set_xlabel(metric_label)
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2 if len(series_by_client) > 4 else 1, fontsize=9)
    fig.tight_layout()

    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)
    return fit_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract t_local_round_s and train_total_input_tokens from stream.log "
            "and plot their empirical/logistic CDFs."
        )
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(
            "/home/xiaoyan/wholeflower/flowertune-llm-medical/.flower-process-runtime/20260326_002840/logs/superlink.log"
        ),
        help="Path to stream.log.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "timing", "metric"],
        default="auto",
        help="Where to extract t_local_round_s from.",
    )
    parser.add_argument(
        "--plot-file",
        type=Path,
        default=None,
        help="Output figure path for t_local_round_s. Default: <log-file stem>_logistic_fit.png",
    )
    parser.add_argument(
        "--token-plot-file",
        type=Path,
        default=None,
        help=(
            "Output figure path for train_total_input_tokens. "
            "Default: <log-file stem>_train_total_input_tokens_logistic_fit.png"
        ),
    )
    parser.add_argument(
        "--client-time-metric",
        choices=["train_runtime", "t_local_round_s"],
        default="train_runtime",
        help="Per-client time metric to plot as round curves.",
    )
    parser.add_argument(
        "--client-time-plot-file",
        type=Path,
        default=None,
        help=(
            "Output figure path for per-client round-vs-time curves. "
            "Default: <log-file stem>_per_client_<metric>.png"
        ),
    )
    return parser.parse_args()


def print_metric_summary(
    name: str,
    values: list[float],
    used_source: str,
    fit_params: tuple[float, float] | None,
    fit_status: str,
    plot_file: Path,
) -> None:
    print(f"{name}_source={used_source}")
    print(f"{name}_num_values={len(values)}")
    if fit_params is not None:
        print(f"{name}_theta={fit_params[0]:.6f}")
        print(f"{name}_k={fit_params[1]:.6f}")
    else:
        print(f"{name}_theta=NA")
        print(f"{name}_k=NA")
    print(f"{name}_fit_status={fit_status}")
    print(f"{name}_plot_file={plot_file}")


def print_client_series_summary(
    series_by_client: dict[int, list[tuple[int, float]]],
    metric_name: str,
    plot_file: Path,
    fit_summary: dict[int, tuple[tuple[float, float] | None, str]] | None = None,
) -> None:
    print(f"client_time_metric={metric_name}")
    print(f"client_time_num_clients={len(series_by_client)}")
    print(f"client_time_plot_file={plot_file}")
    for partition_id, series in series_by_client.items():
        values = [value for _, value in series]
        if not values:
            continue
        fit_params, fit_status = (
            fit_summary.get(partition_id, (None, "NA"))
            if fit_summary is not None
            else (None, "NA")
        )
        theta = f"{fit_params[0]:.6f}" if fit_params is not None else "NA"
        k = f"{fit_params[1]:.6f}" if fit_params is not None else "NA"
        print(
            f"client_{partition_id}_{metric_name}_count={len(values)} "
            f"mean={np.mean(values):.6f} min={np.min(values):.6f} max={np.max(values):.6f} "
            f"theta={theta} k={k} fit_status={fit_status}"
        )


def main() -> int:
    args = parse_args()
    text = args.log_file.read_text(encoding="utf-8", errors="ignore")

    time_values, time_source = extract_timing_values(text, args.source)
    if not time_values:
        raise SystemExit(
            f"No t_local_round_s values found in {args.log_file} (source={args.source})."
        )

    token_values, token_source = extract_metric_values(
        text,
        metric_name="train_total_input_tokens",
        trace_metric_name="total_input_tokens",
    )
    if not token_values:
        raise SystemExit(
            f"No train_total_input_tokens values found in {args.log_file}."
        )

    time_fit, time_fit_status = try_fit_logistic(time_values)
    token_fit, token_fit_status = try_fit_logistic(token_values)
    client_time_series = extract_client_time_series(text, args.client_time_metric)

    time_plot_file = (
        args.plot_file
        if args.plot_file is not None
        else args.log_file.with_name(f"{args.log_file.stem}_logistic_fit.png")
    )
    token_plot_file = (
        args.token_plot_file
        if args.token_plot_file is not None
        else args.log_file.with_name(
            f"{args.log_file.stem}_train_total_input_tokens_logistic_fit.png"
        )
    )
    client_time_plot_file = (
        args.client_time_plot_file
        if args.client_time_plot_file is not None
        else args.log_file.with_name(
            f"{args.log_file.stem}_per_client_{args.client_time_metric}.png"
        )
    )

    plot_fit(
        values=time_values,
        metric_label="t_local_round_s (seconds)",
        plot_title="t_local_round_s Logistic CDF Fit",
        plot_file=time_plot_file,
        fit_params=time_fit,
        fit_status=time_fit_status,
    )
    plot_fit(
        values=token_values,
        metric_label="train_total_input_tokens",
        plot_title="train_total_input_tokens Logistic CDF Fit",
        plot_file=token_plot_file,
        fit_params=token_fit,
        fit_status=token_fit_status,
    )
    client_fit_summary = None
    if client_time_series:
        client_fit_summary = plot_client_cdfs(
            series_by_client=client_time_series,
            metric_label=args.client_time_metric,
            plot_title=f"Per-client {args.client_time_metric} Logistic CDF Fit",
            plot_file=client_time_plot_file,
        )

    print(f"log_file={args.log_file}")
    print_metric_summary(
        name="time",
        values=time_values,
        used_source=time_source,
        fit_params=time_fit,
        fit_status=time_fit_status,
        plot_file=time_plot_file,
    )
    print_metric_summary(
        name="token",
        values=token_values,
        used_source=token_source,
        fit_params=token_fit,
        fit_status=token_fit_status,
        plot_file=token_plot_file,
    )
    if client_time_series:
        print_client_series_summary(
            series_by_client=client_time_series,
            metric_name=args.client_time_metric,
            plot_file=client_time_plot_file,
            fit_summary=client_fit_summary,
        )
    else:
        print(
            f"client_time_metric={args.client_time_metric} "
            "client_time_num_clients=0 client_time_plot_file=NA"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
