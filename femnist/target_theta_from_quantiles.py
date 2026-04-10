#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_THETA_K_CSV = Path("/home/xiaoyan/wholeflower/femnist/mnist_cpu_theta.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit k ~= a * theta from mnist_cpu_theta.csv, then infer a shared target theta "
            "for identical logistic clients from desired overall quantile times."
        )
    )
    parser.add_argument(
        "--theta-k-csv",
        type=Path,
        default=DEFAULT_THETA_K_CSV,
        help="CSV with columns theta and k.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=20,
        help="Number of identical clients in the product/max CDF.",
    )
    parser.add_argument(
        "--p90-time",
        type=float,
        default=None,
        help="Desired overall p90 time in seconds.",
    )
    parser.add_argument(
        "--p95-time",
        type=float,
        default=None,
        help="Desired overall p95 time in seconds.",
    )
    parser.add_argument(
        "--p99-time",
        type=float,
        default=None,
        help="Desired overall p99 time in seconds.",
    )
    return parser.parse_args()


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
    if not math.isfinite(value):
        return None
    return value


def load_theta_k_pairs(path: Path) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "theta" not in (reader.fieldnames or []) or "k" not in (reader.fieldnames or []):
            raise SystemExit(f"{path} must contain 'theta' and 'k' columns.")
        for row in reader:
            theta = parse_float(row.get("theta"))
            k = parse_float(row.get("k"))
            if theta is None or k is None:
                continue
            if theta <= 0.0:
                continue
            pairs.append((theta, k))
    if not pairs:
        raise SystemExit(f"No valid theta/k pairs found in {path}.")
    return pairs


def fit_a_through_origin(pairs: list[tuple[float, float]]) -> float:
    numerator = sum(theta * k for theta, k in pairs)
    denominator = sum(theta * theta for theta, _ in pairs)
    if denominator <= 0.0:
        raise SystemExit("Could not fit a because sum(theta^2) <= 0.")
    return numerator / denominator


def quantile_coefficient(probability: float, num_clients: int, a_value: float) -> tuple[float, float, float]:
    if not (0.0 < probability < 1.0):
        raise ValueError(f"Probability must be in (0, 1), got {probability}")
    single_client_cdf = probability ** (1.0 / num_clients)
    logit = math.log(single_client_cdf / (1.0 - single_client_cdf))
    coeff = 1.0 + a_value * logit
    if coeff <= 0.0:
        raise ValueError(
            f"Coefficient became non-positive for p={probability}, n={num_clients}, a={a_value}"
        )
    return single_client_cdf, logit, coeff


def implied_theta(target_time: float, coeff: float) -> float:
    return target_time / coeff


def fit_joint_theta(targets: list[tuple[float, float]], num_clients: int, a_value: float) -> float:
    coeffs: list[float] = []
    times: list[float] = []
    for probability, target_time in targets:
        _, _, coeff = quantile_coefficient(probability, num_clients, a_value)
        coeffs.append(coeff)
        times.append(target_time)
    numerator = sum(coeff * target_time for coeff, target_time in zip(coeffs, times))
    denominator = sum(coeff * coeff for coeff in coeffs)
    if denominator <= 0.0:
        raise SystemExit("Could not fit joint theta because sum(coeff^2) <= 0.")
    return numerator / denominator


def predicted_quantile_time(theta: float, probability: float, num_clients: int, a_value: float) -> float:
    _, _, coeff = quantile_coefficient(probability, num_clients, a_value)
    return theta * coeff


def main() -> None:
    args = parse_args()
    if not args.theta_k_csv.exists():
        raise SystemExit(f"theta-k CSV does not exist: {args.theta_k_csv}")
    if args.num_clients <= 0:
        raise SystemExit("--num-clients must be positive.")

    raw_targets = [
        (0.90, args.p90_time),
        (0.95, args.p95_time),
        (0.99, args.p99_time),
    ]
    targets = [(probability, target_time) for probability, target_time in raw_targets if target_time is not None]
    if not targets:
        raise SystemExit("Provide at least one of --p90-time, --p95-time, or --p99-time.")

    pairs = load_theta_k_pairs(args.theta_k_csv)
    a_value = fit_a_through_origin(pairs)

    print(f"theta_k_csv={args.theta_k_csv}")
    print(f"num_points={len(pairs)}")
    print(f"num_clients={args.num_clients}")
    print(f"a={a_value:.12f}")
    print()
    print("Per-Quantile Theta")
    print("probability  target_time  implied_theta  implied_k")
    print("-----------  -----------  -------------  ---------")
    for probability, target_time in targets:
        _, _, coeff = quantile_coefficient(probability, args.num_clients, a_value)
        theta = implied_theta(target_time, coeff)
        print(
            f"{probability:11.2%}  "
            f"{target_time:11.4f}  "
            f"{theta:13.4f}  "
            f"{(a_value * theta):9.4f}"
        )

    if len(targets) == 1:
        target_theta = implied_theta(
            targets[0][1],
            quantile_coefficient(targets[0][0], args.num_clients, a_value)[2],
        )
    else:
        target_theta = fit_joint_theta(targets, args.num_clients, a_value)

    print()
    print(f"target_theta={target_theta:.12f}")
    print(f"target_k={a_value * target_theta:.12f}")
    print()
    print("Predicted Quantiles From target_theta")
    print("probability  target_time  predicted_time  error")
    print("-----------  -----------  --------------  ------")
    for probability, target_time in targets:
        predicted_time = predicted_quantile_time(target_theta, probability, args.num_clients, a_value)
        print(
            f"{probability:11.2%}  "
            f"{target_time:11.4f}  "
            f"{predicted_time:14.4f}  "
            f"{(predicted_time - target_time):6.4f}"
        )


if __name__ == "__main__":
    main()
