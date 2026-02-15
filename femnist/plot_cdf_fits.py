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


def model_client(theta_i: float, k_i: float):
    def f(t):
        return 1.0 / (1.0 + np.exp(-(t - theta_i) / k_i))
    return f


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-client CDF fits and overall product CDFs."
    )
    parser.add_argument(
        "--csv",
        default="/home/ubuntu/wholeflower/logs/comm_times.csv",
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
        default=0.0911,
        help="Intercept for k(theta)=k_slope*theta+k_intercept.",
    )
    parser.add_argument(
        "--theta-k-csv",
        default="/home/ubuntu/wholeflower/femnist/mnist_cpu_theta.csv",
        type=Path,
        help="CSV with columns id,theta,k (id is num_examples_train).",
    )
    parser.add_argument(
        "--time-col",
        default="client_train_s",
        help="Column name for duration.",
    )
    parser.add_argument(
        "--client-col",
        default="client_id",
        help="Column name for client id.",
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
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.time_col not in df.columns or args.client_col not in df.columns:
        raise SystemExit("Missing required columns in comm_times.csv.")

    theta_k = pd.read_csv(args.theta_k_csv)
    if not {"id", "theta", "k"}.issubset(theta_k.columns):
        raise SystemExit("theta_k_csv must contain columns: id, theta, k")
    theta_k_map = {
        int(row["id"]): (float(row["theta"]), float(row["k"]))
        for _, row in theta_k.iterrows()
    }

    a_theta = args.a_theta
    if a_theta is None:
        a_theta = args.k_slope * args.theta_target + args.k_intercept
    model1 = model_fixed(args.theta_target, a_theta)

    out_dir = args.out_dir
    client_dir = out_dir / "clients"
    client_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    lam_values = []
    all_times = df[args.time_col].to_numpy(dtype=float)
    all_times = all_times[np.isfinite(all_times)]
    if args.max_time is not None:
        all_times = all_times[all_times <= args.max_time]
    if len(all_times) == 0:
        raise SystemExit("No valid times in CSV.")

    for client, sub in df.groupby(args.client_col):
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

        num_examples = None
        if args.num_examples_col in sub.columns:
            try:
                num_examples = int(float(sub[args.num_examples_col].iloc[0]))
            except Exception:
                num_examples = None

        theta_i = k_i = None
        if num_examples is not None and num_examples in theta_k_map:
            theta_i, k_i = theta_k_map[num_examples]

        rows.append(
            {
                "client_id": client,
                "num_examples": num_examples,
                "lambda": lam,
                "theta_i": theta_i,
                "k_i": k_i,
                "n": len(times),
            }
        )
        if np.isfinite(lam):
            lam_values.append(lam)
        print(
            f"[Lambda Fit] client={client} num_examples={num_examples} "
            f"lambda={lam:.6f} n={len(times)}"
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
        if theta_i is not None and k_i is not None and k_i > 0:
            m2 = model_client(theta_i, k_i)
            y2 = m2(x_grid)
            plt.plot(x_grid, y2, "g-.", linewidth=2, label="per-client θ,k")

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

    plt.figure(figsize=(8, 6))
    plt.plot(t_grid, prod_model1, "r--", linewidth=2, label="Product CDF (fixed θ,λ)")
    if any_model2:
        plt.plot(t_grid, prod_model2, "g-.", linewidth=2, label="Product CDF (per-client θ,k)")
    if x_max is not None:
        plt.plot(x_max, y_max, "ko", markersize=3, label="Empirical max per round")
    plt.xlabel("t")
    plt.ylabel("CDF")
    plt.title("Overall CDF comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "overall_cdfs.png", dpi=150)
    plt.close()

    print(f"Saved per-client plots to {client_dir}")
    print(f"Saved overall plot to {out_dir / 'overall_cdfs.png'}")
    print(f"Saved lambda fits to {args.out_csv}")


if __name__ == "__main__":
    main()
