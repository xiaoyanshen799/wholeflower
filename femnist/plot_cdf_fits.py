#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def make_empirical_cdf(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.sort(times)
    cdf = np.arange(1, len(times) + 1, dtype=float) / float(len(times))
    return times, cdf


def model_fixed(theta: float, a_theta: float):
    def f(t, lam):
        return (1.0 + np.exp(-(t - theta) / a_theta)) ** (-lam)
    return f


def logistic_cdf(t: np.ndarray, theta: float, k: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(t - theta) / k))


def model_client(theta_i: float, k_i: float):
    def f(t):
        return 1.0 / (1.0 + np.exp(-(t - theta_i) / k_i))
    return f


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def parse_exclude_clients(values: list[str] | None) -> set[str]:
    excluded: set[str] = set()
    for raw in values or []:
        for token in raw.split(","):
            token = token.strip()
            if token:
                excluded.add(token)
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-client CDF fits and overall product CDFs."
    )
    parser.add_argument(
        "--csv",
        default="/home/xiaoyan/wholeflower/femnist/logs/speech_commom_target_85_3.csv",
        type=Path,
        help="Input comm_times.csv",
    )
    parser.add_argument(
        "--theta-target",
        type=float,
        required=True,
        help="Fixed theta_target for model1.",
    )
    parser.add_argument(
        "--a-theta",
        type=float,
        default=None,
        help="Fixed a_theta for model1. If omitted, use k(theta)=k_slope*theta+k_intercept.",
    )
    parser.add_argument(
        "--k-slope",
        type=float,
        default=0.0059,
        help="Slope for k(theta)=k_slope*theta+k_intercept.",
    )
    parser.add_argument(
        "--k-intercept",
        type=float,
        default=0.3,
        help="Intercept for k(theta)=k_slope*theta+k_intercept.",
    )
    parser.add_argument(
        "--time-col",
        default="client_train_s",
        help="Column name for duration.",
    )
    parser.add_argument(
        "--client-col",
        default="num_examples",
        help="Column name used as the stable client identifier.",
    )
    parser.add_argument(
        "--num-examples-col",
        default="num_examples",
        help="Column name for num_examples in comm_times.csv.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Optional max time filter.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum samples per client to plot/fit.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/cdf_fits"),
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("logs/lambda_fits.csv"),
        help="Output CSV for lambda fits.",
    )
    parser.add_argument(
        "--exclude-client",
        action="append",
        default=None,
        help="Client identifier(s) to skip. Repeat the flag or pass a comma-separated list.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.time_col not in df.columns or args.client_col not in df.columns:
        raise SystemExit("Missing required columns in comm_times.csv.")
    excluded_clients = parse_exclude_clients(args.exclude_client)
    if excluded_clients:
        client_keys = df[args.client_col].astype(str).str.strip()
        df = df.loc[~client_keys.isin(excluded_clients)].copy()
    if df.empty:
        raise SystemExit("No rows left in CSV after applying excluded clients.")

    a_theta = args.a_theta
    if a_theta is None:
        a_theta = args.k_slope * args.theta_target + args.k_intercept
    model1 = model_fixed(args.theta_target, a_theta)

    out_dir = args.out_dir
    client_dir = out_dir / "clients"
    client_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    lam_values = []
    client_plot_rows = []
    all_times = df[args.time_col].to_numpy(dtype=float)
    all_times = all_times[np.isfinite(all_times)]
    if args.max_time is not None:
        all_times = all_times[all_times <= args.max_time]
    if len(all_times) == 0:
        raise SystemExit("No valid times in CSV.")

    for client, sub in df.groupby(args.client_col):
        client_str = str(client)
        if client_str in excluded_clients:
            continue

        times = sub[args.time_col].to_numpy(dtype=float)
        if args.max_time is not None:
            times = times[times <= args.max_time]
        times = times[np.isfinite(times)]
        if len(times) < args.min_samples:
            continue

        x, y = make_empirical_cdf(times)

        # Fit lambda for model1; emphasize right half of the CDF (larger t / higher y)
        try:
            weights = np.where(y >= 0.6, 3.0, 1.0)  # boost tail influence
            sigma = 1.0 / np.sqrt(weights)
            popt, _ = curve_fit(
                model1,
                x,
                y,
                p0=[1.0],
                bounds=([0.0], [np.inf]),
                sigma=sigma,
                absolute_sigma=False,
                maxfev=20000,
            )
            lam = float(popt[0])
        except Exception:
            lam = float("nan")

        client_num_examples = None
        if args.num_examples_col in sub.columns:
            try:
                client_num_examples = int(float(sub[args.num_examples_col].iloc[0]))
            except Exception:
                client_num_examples = None

        theta_i = k_i = None
        try:
            theta0 = float(np.median(x))
            k0 = float(max((np.percentile(x, 75) - np.percentile(x, 25)) / 4.0, 0.1))
            popt, _ = curve_fit(
                logistic_cdf,
                x,
                y,
                p0=[theta0, k0],
                bounds=([float(x.min()), 0.05], [float(x.max()), 1000.0]),
                maxfev=20000,
            )
            theta_i, k_i = map(float, popt)
        except Exception:
            theta_i = k_i = None

        rows.append(
            {
                "client_id": client,
                "num_examples": client_num_examples,
                "lambda": lam,
                "theta_i": theta_i,
                "k_i": k_i,
                "n": len(times),
            }
        )
        if np.isfinite(lam):
            lam_values.append(lam)
        print(
            f"[Lambda Fit] client={client} num_examples={client_num_examples} "
            f"lambda={lam:.6f} n={len(times)}"
        )

        client_plot_rows.append(
            {
                "client": client,
                "x": x,
                "y": y,
                "theta_i": theta_i,
                "k_i": k_i,
                "lambda": lam,
            }
        )

        # Plot per-client
        plt.figure(figsize=(7, 5))
        plt.plot(x, y, "o", markersize=3, alpha=0.7, label="empirical")

        # model1 curve
        x_grid = np.linspace(x.min(), x.max(), 200)
        if np.isfinite(lam):
            y1 = model1(x_grid, lam)
            plt.plot(x_grid, y1, "r--", linewidth=2, label=f"fixed θ,λ={lam:.3f}")

        # model2 curve
        if np.isfinite(lam):
            y_lam = model1(x_grid, lam)
            plt.plot(x_grid, y_lam, "r--", linewidth=2, label=f"fixed θ,λ={lam:.3f}")

        plt.xlabel("t")
        plt.ylabel("CDF")
        plt.title(f"Client {client}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(client_dir / f"{sanitize_name(str(client))}.png", dpi=150)
        plt.close()

    # Save lambda fits
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    if lam_values:
        print(
            f"[Lambda Range] min={min(lam_values):.6f} max={max(lam_values):.6f} "
            f"mean={float(np.mean(lam_values)):.6f}"
        )

    # Overall product CDFs
    t_grid = np.linspace(all_times.min(), all_times.max(), 400)

    prod_model1 = np.ones_like(t_grid)
    prod_model2 = np.ones_like(t_grid)
    any_model2 = False
    for r in rows:
        lam = r["lambda"]
        if np.isfinite(lam):
            prod_model1 *= model1(t_grid, lam)
        if r["theta_i"] is not None and r["k_i"] is not None and r["k_i"] > 0:
            any_model2 = True
            prod_model2 *= model_client(r["theta_i"], r["k_i"])(t_grid)

    # Empirical max per round
    if "server_round" in df.columns:
        max_per_round = (
            df.groupby("server_round")[args.time_col].max().to_numpy(dtype=float)
        )
        max_per_round = max_per_round[np.isfinite(max_per_round)]
        if args.max_time is not None:
            max_per_round = max_per_round[max_per_round <= args.max_time]
        x_max, y_max = make_empirical_cdf(max_per_round)
    else:
        x_max = y_max = None

    plt.figure(figsize=(12, 8))
    colours = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(client_plot_rows), 1)))
    for colour, plot_row in zip(colours, client_plot_rows):
        client = plot_row["client"]
        x = plot_row["x"]
        y = plot_row["y"]
        plt.plot(
            x,
            y,
            "o",
            markersize=2.5,
            alpha=1,
            color=colour,
            label=f"{client} empirical",
        )
        lam = plot_row["lambda"]
        if np.isfinite(lam):
            plt.plot(
                t_grid,
                model1(t_grid, lam),
                "--",
                linewidth=1.5,
                alpha=0.85,
                color=colour,
                label=f"{client} lam-fit",
            )

    plt.plot(t_grid, prod_model1, "r--", linewidth=2.5, label="Product CDF (fixed θ,λ)")
    if x_max is not None:
        plt.plot(x_max, y_max, "ko", markersize=3.5, label="Empirical max per round")
    plt.xlabel("t")
    plt.ylabel("CDF")
    plt.title("Overall CDF comparison with per-client empirical points and λ fits")
    plt.grid(alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) <= 30:
        plt.legend(fontsize=8, ncol=2)
    else:
        plt.legend(handles[-2:], labels[-2:], fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "overall_cdfs.png", dpi=150)
    plt.close()

    print(f"Saved per-client plots to {client_dir}")
    print(f"Saved overall plot to {out_dir / 'overall_cdfs.png'}")
    print(f"Saved lambda fits to {args.out_csv}")


if __name__ == "__main__":
    main()
