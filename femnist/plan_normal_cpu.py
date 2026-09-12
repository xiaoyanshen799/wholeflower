"""Generate per-client CPUQuota plans for target speed distributions.

The input ``cputheta.csv`` is expected to contain one row per client and CPU
anchor with columns:

    id,theta,k,cpu

For the current CIFAR-10 runs, ``id`` is the number of train examples reported
by the server, not the logical client id. This script maps it back to
``client_XXXXX.npz`` by matching the 90/10 train split size.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

try:
    from scipy.optimize import curve_fit

    HAVE_SCIPY = True
except Exception:
    curve_fit = None
    HAVE_SCIPY = False


def power_shift(cpu: float | np.ndarray, alpha: float, beta: float, gamma: float) -> float | np.ndarray:
    cpu_arr = np.asarray(cpu, dtype=float)
    return alpha * np.power(cpu_arr, -beta) + gamma


@dataclass
class ThetaCpuModel:
    model_id: str
    client_id: int
    num_examples: int
    alpha: float
    beta: float
    gamma: float

    def theta_at(self, cpu: float) -> float:
        return float(power_shift(cpu, self.alpha, self.beta, self.gamma))

    def cpu_for_theta(self, theta: float) -> float:
        if self.alpha <= 0.0 or self.beta <= 0.0 or theta <= self.gamma:
            return math.inf
        return float((self.alpha / (theta - self.gamma)) ** (1.0 / self.beta))


def parse_float_or_auto(raw: str) -> float | None:
    if raw.lower() == "auto":
        return None
    return float(raw)


def fit_theta_cpu(model_id: str, client_id: int, num_examples: int, observations: list[tuple[float, float]]) -> ThetaCpuModel:
    observations = sorted(observations)
    cpus = np.asarray([item[0] for item in observations], dtype=float)
    thetas = np.asarray([item[1] for item in observations], dtype=float)
    if cpus.size < 2:
        raise ValueError(f"Client {client_id}: need at least two CPU/theta observations")

    if HAVE_SCIPY and cpus.size >= 3:
        gamma_upper = max(float(thetas.min()) * 0.999, 1e-6)
        gamma0 = max(0.0, float(thetas.min()) * 0.25)
        alpha0 = max(1e-6, float(thetas.max() - gamma0) * float(cpus.min()))
        try:
            params, _ = curve_fit(
                power_shift,
                cpus,
                thetas,
                p0=[alpha0, 1.0, gamma0],
                bounds=([0.0, 0.0, 0.0], [np.inf, 10.0, gamma_upper]),
                maxfev=20000,
            )
            alpha, beta, gamma = [float(x) for x in params]
            return ThetaCpuModel(model_id, client_id, num_examples, alpha, max(beta, 1e-9), gamma)
        except Exception:
            pass

    safe_cpu = np.clip(cpus, 1e-9, np.inf)
    safe_theta = np.clip(thetas, 1e-9, np.inf)
    slope, intercept = np.polyfit(np.log(safe_cpu), np.log(safe_theta), 1)
    return ThetaCpuModel(
        model_id=model_id,
        client_id=client_id,
        num_examples=num_examples,
        alpha=float(math.exp(intercept)),
        beta=max(1e-9, -float(slope)),
        gamma=0.0,
    )


def load_theta_observations(path: Path) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "theta", "cpu"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required column(s): {', '.join(sorted(missing))}")
        for row in reader:
            model_id = str(row["id"]).strip()
            grouped[model_id].append((float(row["cpu"]), float(row["theta"])))
    return grouped


def load_num_examples_to_client(data_dir: Path, max_clients: int | None = None) -> dict[int, int]:
    mapping: dict[int, int] = {}
    files = sorted(data_dir.glob("client_*.npz"))
    if max_clients is not None:
        files = files[:max_clients]
    for file in files:
        raw_cid = file.stem.replace("client_", "")
        cid = int(raw_cid)
        with np.load(file) as npz:
            num_samples = int(npz["y_train"].shape[0])
        split = int(0.9 * num_samples)
        split = min(max(split, 1), num_samples - 1) if num_samples >= 2 else num_samples
        if split in mapping:
            raise ValueError(
                f"Cannot map by num_examples because {split} appears for both "
                f"client {mapping[split]} and client {cid}"
            )
        mapping[split] = cid
    return mapping


def build_models(args: argparse.Namespace) -> list[ThetaCpuModel]:
    grouped = load_theta_observations(Path(args.theta_cpu_csv))
    example_to_client = (
        load_num_examples_to_client(Path(args.data_dir), args.clients)
        if args.id_kind == "num_examples"
        else {}
    )
    models: list[ThetaCpuModel] = []
    for model_id, observations in grouped.items():
        if args.id_kind == "cid":
            cid = int(model_id)
            num_examples = -1
        else:
            num_examples = int(float(model_id))
            if num_examples not in example_to_client:
                raise ValueError(
                    f"Could not map cputheta id={model_id} to a client. "
                    "Use --id-kind cid if the id column already contains client ids."
                )
            cid = example_to_client[num_examples]
        models.append(fit_theta_cpu(model_id, cid, num_examples, observations))
    models = sorted(models, key=lambda model: model.client_id)
    if args.id_kind == "cid" and args.clients is not None:
        models = models[: args.clients]
    return models


def normal_quantiles(n: int, mean: float, std: float) -> list[float]:
    normal = NormalDist(mu=mean, sigma=std)
    return [normal.inv_cdf((rank + 0.5) / n) for rank in range(n)]


def exponential_quantiles(n: int, mean: float) -> list[float]:
    """Quantiles for t_i ~ Exp(lambda), where mean = std = 1 / lambda."""
    scale = mean
    return [-scale * math.log(1.0 - ((rank + 0.5) / n)) for rank in range(n)]


def target_thetas(n: int, mean: float, std: float, distribution: str) -> list[float]:
    if distribution == "normal":
        return normal_quantiles(n, mean, std)
    if distribution == "exponential":
        return exponential_quantiles(n, mean)
    if distribution == "homogeneous":
        return [mean for _ in range(n)]
    raise ValueError(f"Unsupported distribution: {distribution}")


def command_plan(args: argparse.Namespace) -> None:
    models = build_models(args)
    if not models:
        raise ValueError("No client models were loaded")

    mode = args.mode or args.distribution
    ref_thetas = np.asarray([model.theta_at(args.ref_cpu) for model in models], dtype=float)
    target_mean = parse_float_or_auto(args.target_mean)
    target_std = parse_float_or_auto(args.target_std)
    if target_mean is None:
        target_mean = float(ref_thetas.mean())
    if target_std is None:
        target_std = float(ref_thetas.std(ddof=1)) if len(ref_thetas) > 1 else max(target_mean * 0.05, 1.0)
        target_std = max(target_std, max(target_mean * args.min_auto_std_fraction, 1e-6))
    if target_std <= 0.0:
        raise ValueError("--target-std must be positive")

    targets = target_thetas(len(models), target_mean, target_std, args.distribution)
    ranked_models = sorted(models, key=lambda model: model.theta_at(args.ref_cpu))

    rows = []
    for model, target_theta in zip(ranked_models, targets):
        raw_cpu = model.cpu_for_theta(target_theta)
        if not math.isfinite(raw_cpu):
            cpu = args.cpu_max
            clamped = "yes"
        else:
            cpu = min(max(raw_cpu, args.cpu_min), args.cpu_max)
            clamped = "yes" if abs(cpu - raw_cpu) > 1e-9 else "no"
        rows.append(
            {
                "mode": mode,
                "distribution": args.distribution,
                "client_id": str(model.client_id),
                "cpu_quota": f"{cpu * 100.0:.{args.quota_decimals}f}",
                "cpu": f"{cpu:.6f}",
                "target_theta": f"{target_theta:.6f}",
                "predicted_theta": f"{model.theta_at(cpu):.6f}",
                "raw_cpu": f"{raw_cpu:.6f}" if math.isfinite(raw_cpu) else "inf",
                "clamped": clamped,
                "model_id": model.model_id,
                "num_examples": str(model.num_examples),
                "alpha": f"{model.alpha:.10f}",
                "beta": f"{model.beta:.10f}",
                "gamma": f"{model.gamma:.10f}",
            }
        )

    rows.sort(key=lambda row: int(row["client_id"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "distribution",
        "client_id",
        "cpu_quota",
        "cpu",
        "target_theta",
        "predicted_theta",
        "raw_cpu",
        "clamped",
        "model_id",
        "num_examples",
        "alpha",
        "beta",
        "gamma",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    predicted = np.asarray([float(row["predicted_theta"]) for row in rows], dtype=float)
    quotas = np.asarray([float(row["cpu_quota"]) for row in rows], dtype=float)
    clamped_count = sum(1 for row in rows if row["clamped"] == "yes")
    print(f"Wrote {len(rows)} client CPU assignments to {out}")
    print(f"mode={mode}")
    print(f"distribution={args.distribution}")
    print(f"target mean/std theta = {target_mean:.4f}/{target_std:.4f}")
    print(f"predicted mean/std theta = {predicted.mean():.4f}/{predicted.std(ddof=1):.4f}")
    print(f"CPUQuota min/mean/max = {quotas.min():.2f}%/{quotas.mean():.2f}%/{quotas.max():.2f}%")
    print(f"clamped assignments = {clamped_count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit per-client theta(cpu) curves and assign CPUQuota values to target speed distributions."
    )
    parser.add_argument("--theta-cpu-csv", default="logs/cputheta.csv")
    parser.add_argument("--data-dir", default="data_partitions_cifar10")
    parser.add_argument("--out", default="logs/normal_cpu_plan.csv")
    parser.add_argument("--mode", default=None, help="Plan mode written to CSV; defaults to --distribution")
    parser.add_argument(
        "--distribution",
        choices=["normal", "exponential", "homogeneous"],
        default="normal",
        help="Target distribution for predicted client theta values",
    )
    parser.add_argument("--id-kind", choices=["num_examples", "cid"], default="num_examples")
    parser.add_argument("--clients", type=int, default=None, help="Only map the first N client partition files")
    parser.add_argument("--ref-cpu", type=float, default=0.6, help="Reference CPU fraction used for rank matching")
    parser.add_argument("--cpu-min", type=float, default=0.1)
    parser.add_argument("--cpu-max", type=float, default=1.0)
    parser.add_argument("--quota-decimals", type=int, default=2)
    parser.add_argument("--target-mean", default="auto", help="'auto' or target normal mean theta")
    parser.add_argument("--target-std", default="auto", help="'auto' or target normal std theta")
    parser.add_argument(
        "--min-auto-std-fraction",
        type=float,
        default=0.05,
        help="When --target-std=auto, use at least this fraction of the mean",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command_plan(args)


if __name__ == "__main__":
    main()
