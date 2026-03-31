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


def fit_client_theta_k(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    theta0 = float(np.median(x))
    k0 = float((np.percentile(x, 75) - np.percentile(x, 25)) / 4.0)
    p0 = [theta0, max(k0, 0.1)]
    bounds = ([float(x.min()), 0.05], [float(x.max()), 300.0])
    popt, _ = curve_fit(
        lambda t, theta_i, k_i: model_client(theta_i, k_i)(t),
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )
    return float(popt[0]), float(popt[1])


def parse_error_stats(spec: str) -> list[tuple[str, float | None]]:
    stats: list[tuple[str, float | None]] = []
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token == "mean":
            stats.append(("mean", None))
            continue
        if token.startswith("p"):
            token = token[1:]
        if token.endswith("%"):
            token = token[:-1]
        value = float(token)
        if value > 1.0:
            value /= 100.0
        if not 0.0 < value < 1.0:
            raise ValueError(f"Invalid quantile: {raw}")
        label = f"p{int(round(value * 100))}" if np.isclose(value * 100, round(value * 100)) else f"p{value:g}"
        stats.append((label, value))
    if not stats:
        raise ValueError("No valid error stats requested.")
    return stats


def fixed_model_quantile(theta: float, a_theta: float, lam_total: float, p: float) -> float:
    if lam_total <= 0.0:
        raise ValueError("lam_total must be positive.")
    inner = p ** (-1.0 / lam_total) - 1.0
    return float(theta - a_theta * np.log(inner))


def fixed_model_mean(theta: float, a_theta: float, lam_total: float) -> float:
    upper = fixed_model_quantile(theta, a_theta, lam_total, 0.9999)
    t_eval = np.linspace(0.0, max(upper, theta + 10.0 * a_theta), 5000)
    cdf = model_fixed(theta, a_theta)(t_eval, lam_total)
    return float(np.trapz(1.0 - cdf, t_eval))


def relative_error(predicted: float, actual: float) -> float:
    if np.isclose(actual, 0.0):
        return float("nan")
    return float(abs(predicted - actual) / abs(actual))


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-client CDF fits and overall product CDFs."
    )
    parser.add_argument(
        "--csv",
        default="/home/xiaoyan/wholeflower/flowertune-llm-medical/.flower-process-runtime/20260326_002840/logs/output.csv",
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
        default=2,
        help="Intercept for k(theta)=k_slope*theta+k_intercept.",
    )
    parser.add_argument(
        "--theta-k-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns id,theta,k (id is num_examples_train).",
    )
    parser.add_argument(
        "--time-col",
        default="client_train_s",
        help="Column name for duration.",
    )
    parser.add_argument(
        "--client-col",
        default="num_examples",
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
    parser.add_argument(
        "--error-stats",
        default="p90,p95,mean",
        help="Comma-separated stats for fixed-theta overall errors, e.g. p90,p95,mean.",
    )
    parser.add_argument(
        "--error-out-csv",
        type=Path,
        default=Path("logs/fixed_theta_error_summary.csv"),
        help="Output CSV for fixed-theta overall error summary.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.time_col not in df.columns or args.client_col not in df.columns:
        raise SystemExit("Missing required columns in comm_times.csv.")

    theta_k_map = {}
    if args.theta_k_csv is not None:
        theta_k = pd.read_csv(args.theta_k_csv)
        if not {"id", "theta", "k"}.issubset(theta_k.columns):
            raise SystemExit("theta_k_csv must contain columns: id, theta, k")
        theta_k_map = {
            int(row["id"]): (float(row["theta"]), float(row["k"]))
            for _, row in theta_k.iterrows()
        }
        print(f"Loaded theta/k mapping from {args.theta_k_csv}")
    else:
        print("No theta/k mapping CSV provided; fitting per-client theta/k from data.")

    a_theta = args.a_theta
    if a_theta is None:
        a_theta = args.k_slope * args.theta_target + args.k_intercept
    model1 = model_fixed(args.theta_target, a_theta)

    out_dir = args.out_dir
    client_dir = out_dir / "clients"
    client_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    lam_values = []
    empirical_client_cdfs: dict[object, tuple[np.ndarray, np.ndarray]] = {}
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
        empirical_client_cdfs[client] = (x, y)

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
        theta_k_source = "none"
        if theta_k_map:
            if num_examples is not None and num_examples in theta_k_map:
                theta_i, k_i = theta_k_map[num_examples]
                theta_k_source = "csv"
        else:
            try:
                theta_i, k_i = fit_client_theta_k(x, y)
                theta_k_source = "fit"
            except Exception:
                theta_i = k_i = None

        rows.append(
            {
                "client_id": client,
                "num_examples": num_examples,
                "lambda": lam,
                "theta_i": theta_i,
                "k_i": k_i,
                "theta_k_source": theta_k_source,
                "time_min": float(x.min()),
                "time_max": float(x.max()),
                "n": len(times),
            }
        )
        if np.isfinite(lam):
            lam_values.append(lam)
        print(
            f"[Lambda Fit] client={client} num_examples={num_examples} "
            f"lambda={lam:.6f} n={len(times)}"
        )
        if theta_i is not None and k_i is not None:
            print(
                f"[Theta/K {theta_k_source}] client={client} num_examples={num_examples} "
                f"theta={theta_i:.6f} k={k_i:.6f}"
            )
        else:
            print(
                f"[Theta/K {theta_k_source}] client={client} num_examples={num_examples} "
                "theta/k unavailable"
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
    client_labels_in_legend = len(rows) <= 12
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(rows), 1)))
    for idx, r in enumerate(rows):
        client = r["client_id"]
        theta_i = r["theta_i"]
        k_i = r["k_i"]
        lam = r["lambda"]
        client_mask = t_grid >= float(r["time_min"])
        x_emp, y_emp = empirical_client_cdfs[client]
        color = colors[idx % len(colors)]
        point_label = f"Client {client} empirical" if client_labels_in_legend else None
        plt.plot(
            x_emp,
            y_emp,
            "o",
            markersize=3,
            alpha=0.5,
            color=color,
            label=point_label,
        )
        if theta_i is not None and k_i is not None and k_i > 0:
            y_client = model_client(theta_i, k_i)(t_grid[client_mask])
            label = f"Client {client} fit" if client_labels_in_legend else None
            plt.plot(
                t_grid[client_mask],
                y_client,
                linewidth=1.0,
                alpha=0.35,
                color=color,
                label=label,
            )
        elif np.isfinite(lam):
            y_client = model1(t_grid[client_mask], lam)
            label = (
                f"Client {client} fixed θ,λ" if client_labels_in_legend else None
            )
            plt.plot(
                t_grid[client_mask],
                y_client,
                linewidth=1.0,
                alpha=0.25,
                color=color,
                label=label,
            )

    product_start = max(float(r["time_min"]) for r in rows) if rows else float(t_grid.min())
    product_mask = t_grid >= product_start
    plt.plot(
        t_grid[product_mask],
        prod_model1[product_mask],
        "r--",
        linewidth=2,
        label="Product CDF (fixed θ,λ)",
    )
    if any_model2:
        plt.plot(
            t_grid[product_mask],
            prod_model2[product_mask],
            "g-.",
            linewidth=2,
            label="Product CDF (per-client θ,k)",
        )
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

    if "server_round" in df.columns and x_max is not None and len(max_per_round) > 0:
        error_rows = []
        lam_total = float(np.nansum([r["lambda"] for r in rows if np.isfinite(r["lambda"])]))
        requested_stats = parse_error_stats(args.error_stats)
        for label, quantile in requested_stats:
            if label == "mean":
                actual_value = float(np.mean(max_per_round))
                predicted_value = fixed_model_mean(args.theta_target, a_theta, lam_total)
            else:
                assert quantile is not None
                actual_value = float(np.quantile(max_per_round, quantile))
                predicted_value = fixed_model_quantile(
                    args.theta_target,
                    a_theta,
                    lam_total,
                    quantile,
                )
            abs_err = float(abs(predicted_value - actual_value))
            rel_err = relative_error(predicted_value, actual_value)
            error_rows.append(
                {
                    "stat": label,
                    "actual": actual_value,
                    "predicted_fixed_theta_lambda": predicted_value,
                    "abs_err": abs_err,
                    "rel_err": rel_err,
                }
            )
            print(
                f"[Fixed Theta Error] {label}: actual={actual_value:.6f} "
                f"predicted={predicted_value:.6f} abs_err={abs_err:.6f} "
                f"rel_err={rel_err:.6f}"
            )
        args.error_out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(error_rows).to_csv(args.error_out_csv, index=False)
        print(f"Saved fixed-theta error summary to {args.error_out_csv}")
    else:
        print("Skipped fixed-theta error summary because server_round data is unavailable.")

    print(f"Saved per-client plots to {client_dir}")
    print(f"Saved overall plot to {out_dir / 'overall_cdfs.png'}")
    print(f"Saved lambda fits to {args.out_csv}")


if __name__ == "__main__":
    main()
