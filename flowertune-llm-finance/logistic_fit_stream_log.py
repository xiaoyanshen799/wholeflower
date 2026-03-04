#!/usr/bin/env python3
"""Fit a 2-parameter logistic CDF for t_local_round_s values in stream.log."""

from __future__ import annotations

import argparse
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
METRIC_RE = re.compile(rf"'t_local_round_s'\s*:\s*'?({FLOAT_RE})'?")


def logistic_cdf(t: np.ndarray, theta: float, k: float) -> np.ndarray:
    """2-parameter logistic CDF."""
    return 1.0 / (1.0 + np.exp(-(t - theta) / k))


def extract_values(text: str, source: str) -> tuple[list[float], str]:
    clean_text = ANSI_RE.sub("", text)

    def from_timing() -> list[float]:
        return [float(m.group(1)) for m in TIMING_RE.finditer(clean_text)]

    def from_metric() -> list[float]:
        values: list[float] = []
        for line in clean_text.splitlines():
            if "Aggregated MetricRecord" not in line:
                continue
            match = METRIC_RE.search(line)
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


def fit_logistic(values: list[float]) -> tuple[float, float]:
    if len(values) < 3:
        raise ValueError(f"need at least 3 values, got {len(values)}")

    x_data = np.sort(np.asarray(values, dtype=float))
    if float(np.max(x_data)) == float(np.min(x_data)):
        raise ValueError("all values are identical; cannot fit logistic CDF")

    y_data = np.arange(1, len(x_data) + 1, dtype=float) / float(len(x_data))

    theta0 = float(np.median(x_data))
    k0 = float((np.percentile(x_data, 75) - np.percentile(x_data, 25)) / 4.0)
    p0 = [theta0, max(k0, 0.1)]

    bounds = ([float(np.min(x_data)), 0.005], [float(np.max(x_data)), 300.0])

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
    return float(theta_hat), float(k_hat)


def plot_fit(values: list[float], theta: float, k: float, plot_file: Path) -> None:
    """Plot empirical CDF points and fitted logistic CDF curve."""
    x_data = np.sort(np.asarray(values, dtype=float))
    y_data = np.arange(1, len(x_data) + 1, dtype=float) / float(len(x_data))

    x_grid = np.linspace(float(np.min(x_data)), float(np.max(x_data)), 300)
    y_fit = logistic_cdf(x_grid, theta, k)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        x_data,
        y_data,
        s=22,
        alpha=0.85,
        label="Empirical points",
        color="#1f77b4",
    )
    ax.plot(
        x_grid,
        y_fit,
        linewidth=2.2,
        label=f"Fitted logistic CDF (theta={theta:.3f}, k={k:.3f})",
        color="#d62728",
    )
    ax.set_title("t_local_round_s Logistic CDF Fit")
    ax.set_xlabel("t_local_round_s (seconds)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract t_local_round_s from stream.log and fit logistic CDF."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(
            "/home/xiaoyan/wholeflower/flowertune-llm-finance/exp_logs/MPS100/stream.log"
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
        help="Output figure path (PNG). Default: <log-file stem>_logistic_fit.png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = args.log_file.read_text(encoding="utf-8", errors="ignore")
    values, used_source = extract_values(text, args.source)

    if not values:
        raise SystemExit(
            f"No t_local_round_s values found in {args.log_file} (source={args.source})."
        )

    theta, k = fit_logistic(values)
    plot_file = (
        args.plot_file
        if args.plot_file is not None
        else args.log_file.with_name(f"{args.log_file.stem}_logistic_fit.png")
    )
    plot_fit(values, theta, k, plot_file)
    print(f"log_file={args.log_file}")
    print(f"source={used_source}")
    print(f"num_values={len(values)}")
    print(f"theta={theta:.6f}")
    print(f"k={k:.6f}")
    print(f"plot_file={plot_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
