"""Fit only gamma for a fixed-theta/k generalized logistic CDF.

Model:

    F(t) = 1 / (1 + exp(-(t - theta) / k)) ** gamma

The script fits one client at a time. Use ``--match-column num_examples`` when
the CSV client_id is an ephemeral address such as ``ipv4:127.0.0.1:46622``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares


def generalized_logistic_cdf(t: np.ndarray, theta: float, k: float, gamma: float) -> np.ndarray:
    if k <= 0.0:
        raise ValueError("k must be positive")
    z = np.clip(-(np.asarray(t, dtype=float) - theta) / k, -700.0, 700.0)
    base = 1.0 + np.exp(z)
    return np.power(base, -gamma)


def gamma_model(theta: float, k: float):
    def model(t: np.ndarray, gamma: float) -> np.ndarray:
        return generalized_logistic_cdf(t, theta, k, gamma)

    return model


def read_client_values(
    csv_path: Path,
    *,
    match_column: str,
    client_id: str,
    duration_column: str,
    max_duration: float | None,
) -> tuple[np.ndarray, dict[str, str]]:
    values: list[float] = []
    first_row: dict[str, str] | None = None
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        for column in (match_column, duration_column):
            if column not in fieldnames:
                raise ValueError(f"{csv_path} missing required column {column!r}")
        for row in reader:
            if str(row.get(match_column, "")).strip() != str(client_id):
                continue
            raw_value = row.get(duration_column, "")
            if raw_value == "":
                continue
            duration = float(raw_value)
            if max_duration is not None and duration > max_duration:
                continue
            values.append(duration)
            if first_row is None:
                first_row = dict(row)
    if first_row is None or not values:
        raise ValueError(
            f"No rows found for {match_column}={client_id!r} with nonempty {duration_column!r}"
        )
    arr = np.asarray(sorted(values), dtype=float)
    return arr, first_row


def fit_gamma(
    durations: np.ndarray,
    *,
    theta: float,
    k: float,
    gamma_init: float,
    gamma_min: float,
    gamma_max: float,
    tail_min_cdf: float,
    tail_weight_power: float,
    objective: str,
    eps: float,
) -> tuple[float, np.ndarray, float, int]:
    y_emp = np.arange(1, len(durations) + 1, dtype=float) / float(len(durations))
    fit_mask = y_emp >= tail_min_cdf
    if not np.any(fit_mask):
        raise ValueError(f"--tail-min-cdf={tail_min_cdf} selected no samples")

    x_fit = durations[fit_mask]
    y_fit_emp = y_emp[fit_mask]
    weights = np.power(np.clip(y_fit_emp, eps, 1.0), tail_weight_power)

    if objective == "cdf":
        def residual(params: np.ndarray) -> np.ndarray:
            y_pred = generalized_logistic_cdf(x_fit, theta, k, float(params[0]))
            return (y_pred - y_fit_emp) * weights

        result = least_squares(
            residual,
            x0=np.asarray([gamma_init], dtype=float),
            bounds=([gamma_min], [gamma_max]),
            loss="soft_l1",
            max_nfev=20000,
        )
        gamma = float(result.x[0])
    elif objective == "log-survival":
        def residual(params: np.ndarray) -> np.ndarray:
            y_pred = generalized_logistic_cdf(x_fit, theta, k, float(params[0]))
            pred_survival = np.clip(1.0 - y_pred, eps, 1.0)
            emp_survival = np.clip(1.0 - y_fit_emp, eps, 1.0)
            return (np.log(pred_survival) - np.log(emp_survival)) * weights

        result = least_squares(
            residual,
            x0=np.asarray([gamma_init], dtype=float),
            bounds=([gamma_min], [gamma_max]),
            loss="soft_l1",
            max_nfev=20000,
        )
        gamma = float(result.x[0])
    else:
        popt, _ = curve_fit(
            gamma_model(theta, k),
            x_fit,
            y_fit_emp,
            p0=[gamma_init],
            bounds=([gamma_min], [gamma_max]),
            maxfev=20000,
            loss="soft_l1",
        )
        gamma = float(popt[0])

    y_fit = generalized_logistic_cdf(durations, theta, k, gamma)
    sse = float(np.sum((y_emp - y_fit) ** 2))
    return gamma, y_emp, sse, int(fit_mask.sum())


def write_summary(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "csv",
        "match_column",
        "client_id",
        "duration_column",
        "num_samples",
        "theta",
        "k",
        "gamma",
        "sse",
        "tail_min_cdf",
        "tail_weight_power",
        "objective",
        "fit_samples",
        "duration_min",
        "duration_median",
        "duration_max",
        "num_examples",
        "raw_client_id",
    ]
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            if existing_fieldnames != fieldnames:
                existing_rows = list(reader)
                with path.open("w", newline="", encoding="utf-8") as out_f:
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()
                    for existing in existing_rows:
                        writer.writerow({name: existing.get(name, "") for name in fieldnames})
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def plot_fit(
    path: Path,
    durations: np.ndarray,
    y_emp: np.ndarray,
    *,
    theta: float,
    k: float,
    gamma: float,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_grid = np.linspace(float(durations.min()), float(durations.max()), 400)
    y_grid = generalized_logistic_cdf(x_grid, theta, k, gamma)
    plt.figure(figsize=(8, 5))
    plt.plot(durations, y_emp, "o", markersize=4, alpha=0.7, label="empirical CDF")
    plt.plot(x_grid, y_grid, "-", linewidth=2.0, label=f"gamma fit = {gamma:.6g}")
    plt.xlabel("duration")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit gamma for one client with fixed theta and k.")
    parser.add_argument("--csv", default="logs/pacer_times_240.csv", help="Timing CSV path")
    parser.add_argument("--client-id", required=True, help="Client id value to match")
    parser.add_argument(
        "--match-column",
        default="num_examples",
        help="Column used to identify the client, e.g. num_examples or client_id",
    )
    parser.add_argument("--duration-column", default="client_train_s")
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument("--gamma-init", type=float, default=1.0)
    parser.add_argument("--gamma-min", type=float, default=1e-6)
    parser.add_argument("--gamma-max", type=float, default=1_000_000.0)
    parser.add_argument(
        "--tail-min-cdf",
        type=float,
        default=0.0,
        help="Only fit points with empirical CDF >= this value, e.g. 0.8 for upper tail",
    )
    parser.add_argument(
        "--tail-weight-power",
        type=float,
        default=0.0,
        help="Extra tail emphasis: residual weights are empirical_cdf ** this value",
    )
    parser.add_argument(
        "--objective",
        choices=["cdf", "log-survival", "curve-fit"],
        default="cdf",
        help="Use log-survival to emphasize right-tail completion probability",
    )
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--out", default=None, help="Append one-row summary CSV")
    parser.add_argument("--plot", default=None, help="Optional output plot path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    csv_path = Path(args.csv)
    durations, first_row = read_client_values(
        csv_path,
        match_column=args.match_column,
        client_id=args.client_id,
        duration_column=args.duration_column,
        max_duration=args.max_duration,
    )
    gamma, y_emp, sse, fit_samples = fit_gamma(
        durations,
        theta=args.theta,
        k=args.k,
        gamma_init=args.gamma_init,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        tail_min_cdf=args.tail_min_cdf,
        tail_weight_power=args.tail_weight_power,
        objective=args.objective,
        eps=args.eps,
    )
    summary = {
        "csv": str(csv_path),
        "match_column": args.match_column,
        "client_id": args.client_id,
        "duration_column": args.duration_column,
        "num_samples": len(durations),
        "theta": args.theta,
        "k": args.k,
        "gamma": gamma,
        "sse": sse,
        "tail_min_cdf": args.tail_min_cdf,
        "tail_weight_power": args.tail_weight_power,
        "objective": args.objective,
        "fit_samples": fit_samples,
        "duration_min": float(durations.min()),
        "duration_median": float(np.median(durations)),
        "duration_max": float(durations.max()),
        "num_examples": first_row.get("num_examples", ""),
        "raw_client_id": first_row.get("client_id", ""),
    }
    print(
        "gamma-fit "
        f"{args.match_column}={args.client_id} n={len(durations)} "
        f"fit_samples={fit_samples} theta={args.theta:.6g} k={args.k:.6g} "
        f"gamma={gamma:.8g} sse={sse:.8g} "
        f"tail_min_cdf={args.tail_min_cdf:.3g} objective={args.objective} "
        f"duration=[{durations.min():.4f}, {np.median(durations):.4f}, {durations.max():.4f}]"
    )
    if args.out:
        write_summary(Path(args.out), summary)
        print(f"appended summary to {args.out}")
    if args.plot:
        title = f"{args.match_column}={args.client_id}, theta={args.theta:g}, k={args.k:g}"
        plot_fit(Path(args.plot), durations, y_emp, theta=args.theta, k=args.k, gamma=gamma, title=title)
        print(f"saved plot to {args.plot}")


if __name__ == "__main__":
    main()
