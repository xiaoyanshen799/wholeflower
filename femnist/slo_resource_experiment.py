"""FedPacer SLO resource experiments.

The Table 4 workflow is:

1. Generate a fixed Vanilla FedAvg CPU plan from the paper's nominal per-step
   speed distributions.
2. Run vanilla unchanged and use its empirical round-time p90/p95 as the SLO
   deadline D.
3. With that D, derive optimistic oracle and FedPacer CPU allocations without
   changing FedAvg training semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import numpy as np

try:
    from scipy.optimize import curve_fit

    HAVE_SCIPY = True
except Exception:
    curve_fit = None
    HAVE_SCIPY = False


def logistic_cdf(t: float | np.ndarray, theta: float, k: float) -> float | np.ndarray:
    z = -(np.asarray(t) - theta) / max(k, 1e-9)
    return 1.0 / (1.0 + np.exp(z))


def logit(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def power_shift(cpu: float | np.ndarray, alpha: float, beta: float, gamma: float) -> float | np.ndarray:
    cpu_arr = np.asarray(cpu, dtype=float)
    return alpha * np.power(cpu_arr, -beta) + gamma


@dataclass
class LogisticFit:
    theta: float
    k: float
    samples: int


@dataclass
class ClientResourceModel:
    client_id: str
    alpha: float
    beta: float
    gamma: float
    k_over_theta: float
    observed: list[tuple[float, LogisticFit]]

    def theta_at(self, cpu_quota: float) -> float:
        return float(power_shift(cpu_quota, self.alpha, self.beta, self.gamma))

    def k_at(self, cpu_quota: float) -> float:
        return max(1e-6, self.k_over_theta * self.theta_at(cpu_quota))

    def cdf_at(self, deadline: float, cpu_quota: float) -> float:
        return float(logistic_cdf(deadline, self.theta_at(cpu_quota), self.k_at(cpu_quota)))

    def cpu_for_theta(self, theta_target: float) -> float:
        if self.alpha <= 0.0 or self.beta <= 0.0 or theta_target <= self.gamma:
            return math.inf
        return float((self.alpha / (theta_target - self.gamma)) ** (1.0 / self.beta))


def fit_logistic(samples: list[float]) -> LogisticFit:
    values = np.asarray(sorted(samples), dtype=float)
    if values.size == 0:
        raise ValueError("cannot fit logistic distribution without samples")

    cdf = np.arange(1, values.size + 1, dtype=float) / float(values.size)
    theta0 = float(np.median(values))
    iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
    k0 = max(iqr / (2.0 * math.log(3.0)), 0.05)

    if HAVE_SCIPY and values.size >= 3:
        lower = [max(1e-9, float(values.min()) * 0.5), 1e-6]
        upper = [max(float(values.max()) * 1.5, theta0 + 1.0), max(float(values.max()) * 2.0, 1.0)]
        try:
            params, _ = curve_fit(
                logistic_cdf,
                values,
                cdf,
                p0=[theta0, k0],
                bounds=(lower, upper),
                maxfev=20000,
            )
            theta, k = float(params[0]), float(params[1])
            return LogisticFit(theta=theta, k=max(k, 1e-6), samples=int(values.size))
        except Exception:
            pass

    return LogisticFit(theta=theta0, k=k0, samples=int(values.size))


def fit_resource_model(client_id: str, observed: list[tuple[float, LogisticFit]]) -> ClientResourceModel:
    observed = sorted(observed, key=lambda item: item[0])
    cpus = np.asarray([item[0] for item in observed], dtype=float)
    thetas = np.asarray([item[1].theta for item in observed], dtype=float)
    ratios = [item[1].k / item[1].theta for item in observed if item[1].theta > 0.0]
    k_over_theta = float(np.median(ratios)) if ratios else 0.01

    if HAVE_SCIPY and cpus.size >= 3:
        gamma0 = max(0.0, float(thetas.min()) * 0.5)
        alpha0 = max(1e-6, float(thetas.max() - gamma0) * float(cpus.min()))
        beta0 = 1.0
        try:
            params, _ = curve_fit(
                power_shift,
                cpus,
                thetas,
                p0=[alpha0, beta0, gamma0],
                bounds=([0.0, 0.0, 0.0], [np.inf, 10.0, max(float(thetas.min()) * 0.999, 1e-6)]),
                maxfev=20000,
            )
            alpha, beta, gamma = [float(x) for x in params]
            return ClientResourceModel(client_id, alpha, max(beta, 1e-9), gamma, k_over_theta, observed)
        except Exception:
            pass

    gamma = 0.0
    safe_cpu = np.clip(cpus, 1e-9, np.inf)
    safe_theta = np.clip(thetas, 1e-9, np.inf)
    slope, intercept = np.polyfit(np.log(safe_cpu), np.log(safe_theta), 1)
    beta = max(1e-9, -float(slope))
    alpha = float(math.exp(intercept))
    return ClientResourceModel(client_id, alpha, beta, gamma, k_over_theta, observed)


def parse_cpu_quotas(raw: str) -> list[float]:
    quotas = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not quotas:
        raise ValueError("at least one CPU quota is required")
    return quotas


def cpu_label(cpu_quota: float) -> str:
    return ("%g" % cpu_quota).replace(".", "p")


def client_sort_key(client_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(client_id))
    except ValueError:
        return (1, client_id)


def normalize_dataset_name(name: str) -> str:
    dataset = name.lower().replace("_", "-")
    if dataset in {"cifar", "cifar10", "cifar-10"}:
        return "cifar10"
    if dataset in {"mnist", "fmnist", "fashionmnist", "fashion-mnist", "femnist"}:
        return "mnist"
    return dataset


@dataclass(frozen=True)
class Table4Setting:
    dataset: str
    clients: int
    heterogeneity: str
    distribution: str
    mean_step_s: float | None = None
    std_step_s: float | None = None
    exp_lambda: float | None = None


TABLE4_SETTINGS: tuple[Table4Setting, ...] = (
    Table4Setting("mnist", 5, "normal", "normal", mean_step_s=0.1, std_step_s=0.02),
    Table4Setting("mnist", 10, "normal", "normal", mean_step_s=0.1, std_step_s=0.03),
    Table4Setting("mnist", 10, "homogeneous", "homogeneous", mean_step_s=0.1, std_step_s=0.0),
    Table4Setting("mnist", 20, "normal", "normal", mean_step_s=0.1, std_step_s=0.04),
    Table4Setting("mnist", 10, "exponential", "exponential", exp_lambda=3.5),
    Table4Setting("cifar10", 10, "normal", "normal", mean_step_s=12.5, std_step_s=3.0),
)


def resolve_table4_setting(dataset: str, clients: int, heterogeneity: str) -> Table4Setting:
    key = normalize_dataset_name(dataset)
    heterogeneity = heterogeneity.lower()
    for setting in TABLE4_SETTINGS:
        if setting.dataset == key and setting.clients == clients and setting.heterogeneity == heterogeneity:
            return setting

    same_dataset = [setting for setting in TABLE4_SETTINGS if setting.dataset == key]
    same_heterogeneity = [setting for setting in same_dataset if setting.heterogeneity == heterogeneity]
    if len(same_heterogeneity) == 1:
        return same_heterogeneity[0]

    normal_settings = [setting for setting in same_dataset if setting.heterogeneity == "normal"]
    if heterogeneity == "homogeneous" and normal_settings:
        base = min(normal_settings, key=lambda setting: abs(setting.clients - clients))
        return Table4Setting(key, clients, "homogeneous", "homogeneous", mean_step_s=base.mean_step_s, std_step_s=0.0)
    if heterogeneity == "exponential" and normal_settings:
        base = min(normal_settings, key=lambda setting: abs(setting.clients - clients))
        return Table4Setting(
            key,
            clients,
            "exponential",
            "exponential",
            mean_step_s=base.mean_step_s,
            std_step_s=base.std_step_s,
        )

    supported = ", ".join(
        f"{setting.dataset}/N={setting.clients}/{setting.heterogeneity}" for setting in TABLE4_SETTINGS
    )
    raise ValueError(
        f"No Table 4 speed setting for dataset={dataset!r}, clients={clients}, "
        f"heterogeneity={heterogeneity!r}. Supported settings: {supported}"
    )


def normal_quantiles(n: int, mean: float, std: float) -> list[float]:
    if std <= 0.0:
        return [mean for _ in range(n)]
    normal = NormalDist(mu=mean, sigma=std)
    return [normal.inv_cdf((rank + 0.5) / n) for rank in range(n)]


def shifted_exponential_quantiles(n: int, mean: float, std: float) -> list[float]:
    scale = max(std, 1e-12)
    shift = mean - scale
    return [shift - scale * math.log(1.0 - ((rank + 0.5) / n)) for rank in range(n)]


def rate_exponential_quantiles(n: int, rate: float) -> list[float]:
    if rate <= 0.0:
        raise ValueError("exponential lambda/rate must be positive")
    return [-math.log(1.0 - ((rank + 0.5) / n)) / rate for rank in range(n)]


def table4_step_times(
    *,
    dataset: str,
    clients: int,
    heterogeneity: str,
    mean_step_s: float | None = None,
    std_step_s: float | None = None,
    exp_lambda: float | None = None,
    min_step_s: float = 1e-6,
) -> tuple[list[float], Table4Setting]:
    setting = resolve_table4_setting(dataset, clients, heterogeneity)
    mean = setting.mean_step_s if mean_step_s is None else mean_step_s
    std = setting.std_step_s if std_step_s is None else std_step_s
    lam = setting.exp_lambda if exp_lambda is None else exp_lambda

    if setting.distribution == "homogeneous":
        if mean is None:
            raise ValueError("homogeneous Table 4 setting requires a mean step time")
        values = [mean for _ in range(clients)]
    elif setting.distribution == "normal":
        if mean is None or std is None:
            raise ValueError("normal Table 4 setting requires mean and std step times")
        values = normal_quantiles(clients, mean, std)
    elif setting.distribution == "exponential":
        if lam is not None:
            values = rate_exponential_quantiles(clients, lam)
        elif mean is not None and std is not None:
            values = shifted_exponential_quantiles(clients, mean, std)
        else:
            raise ValueError("exponential Table 4 setting requires lambda or mean/std")
    else:
        raise ValueError(f"Unsupported Table 4 distribution: {setting.distribution}")

    values = [max(min_step_s, float(value)) for value in values]
    values.sort()
    return values, setting


def load_num_examples_to_client(
    data_dir: Path,
    max_clients: int | None = None,
    *,
    allow_duplicates: bool = False,
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    duplicates: set[int] = set()
    files = sorted(data_dir.glob("client_*.npz"))
    if max_clients is not None and max_clients > 0:
        files = files[:max_clients]
    for file in files:
        cid = int(file.stem.replace("client_", ""))
        with np.load(file) as npz:
            if "y_train" in npz:
                num_samples = int(npz["y_train"].shape[0])
            elif "targets" in npz:
                num_samples = int(npz["targets"].shape[0])
            else:
                raise ValueError(f"{file} does not contain y_train or targets")
        split = int(0.9 * num_samples)
        split = min(max(split, 1), num_samples - 1) if num_samples >= 2 else num_samples
        if split in mapping:
            if allow_duplicates:
                duplicates.add(split)
                continue
            raise ValueError(
                f"Cannot map by num_examples because {split} appears for both "
                f"client {mapping[split]} and client {cid}"
            )
        mapping[split] = cid
    for duplicate in duplicates:
        mapping.pop(duplicate, None)
    return mapping


def canonical_client_id(
    raw_id: str,
    *,
    id_kind: str,
    num_examples_to_client: dict[int, int] | None = None,
    clients: int | None = None,
) -> str:
    raw_id = str(raw_id).strip()
    if id_kind == "auto":
        try:
            numeric_id = int(float(raw_id))
        except ValueError:
            return raw_id
        if num_examples_to_client is not None and numeric_id in num_examples_to_client:
            return str(num_examples_to_client[numeric_id])
        if clients is not None and clients > 0:
            if 0 <= numeric_id < clients:
                return str(numeric_id)
            raise ValueError(
                f"id={numeric_id} does not look like a logical client id for "
                f"N={clients} and was not found in the num_examples mapping"
            )
        return str(numeric_id)
    if id_kind == "cid":
        return str(int(float(raw_id)))
    if id_kind == "num_examples":
        if num_examples_to_client is None:
            raise ValueError("num_examples_to_client mapping is required when id_kind='num_examples'")
        num_examples = int(float(raw_id))
        if num_examples not in num_examples_to_client:
            raise ValueError(
                f"Could not map num_examples={num_examples} to a client id. "
                "Use --id-kind cid if the id column already stores logical client ids."
            )
        return str(num_examples_to_client[num_examples])
    return raw_id


def normalize_cpu_quota(value: float, unit: str) -> float:
    if unit == "percent":
        return float(value)
    if unit == "fraction":
        return float(value) * 100.0
    if unit != "auto":
        raise ValueError(f"Unsupported CPU unit: {unit}")
    return float(value) * 100.0 if value <= 2.0 else float(value)


def load_models_from_theta_cpu_csv(
    path: Path,
    *,
    data_dir: Path,
    id_kind: str,
    clients: int | None,
    cpu_unit: str,
    default_k_over_theta: float,
) -> list[ClientResourceModel]:
    num_examples_to_client = (
        load_num_examples_to_client(data_dir, clients, allow_duplicates=id_kind == "auto")
        if id_kind in {"auto", "num_examples"}
        else None
    )
    grouped_raw: dict[str, list[tuple[float, float, float | None]]] = defaultdict(list)
    observed_ratios: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "theta", "cpu"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required column(s): {', '.join(sorted(missing))}")
        for row in reader:
            try:
                cid = canonical_client_id(
                    row["id"],
                    id_kind=id_kind,
                    num_examples_to_client=num_examples_to_client,
                    clients=clients,
                )
            except ValueError:
                continue
            cpu_quota = normalize_cpu_quota(float(row["cpu"]), cpu_unit)
            theta = float(row["theta"])
            raw_k = str(row.get("k", "")).strip()
            k_value: float | None = None
            if raw_k:
                k_value = max(float(raw_k), 1e-6)
                if theta > 0.0:
                    observed_ratios.append(k_value / theta)
            grouped_raw[cid].append((cpu_quota, theta, k_value))

    fallback_ratio = (
        float(np.median(observed_ratios))
        if observed_ratios
        else max(float(default_k_over_theta), 1e-9)
    )
    grouped: dict[str, list[tuple[float, LogisticFit]]] = defaultdict(list)
    for cid, observations in grouped_raw.items():
        for cpu_quota, theta, k_value in observations:
            k = k_value if k_value is not None else max(theta * fallback_ratio, 1e-6)
            grouped[cid].append((cpu_quota, LogisticFit(theta=theta, k=k, samples=0)))

    models = [
        fit_resource_model(cid, observations)
        for cid, observations in sorted(grouped.items(), key=lambda item: client_sort_key(item[0]))
        if len(observations) >= 2
    ]
    if not models:
        raise RuntimeError(f"no usable theta/cpu models loaded from {path}")
    return models


def default_theta_cpu_csv(dataset: str | None) -> Path | None:
    if not dataset:
        return None
    base_dir = Path(__file__).resolve().parent
    dataset_key = normalize_dataset_name(dataset)
    if dataset_key == "cifar10":
        return base_dir / "logs" / "cputheta.csv"
    if dataset_key == "mnist":
        return base_dir / "logs" / "mnistdata" / "cputheta.csv"
    return None


def resolve_theta_cpu_csv_arg(args: argparse.Namespace) -> str:
    raw = str(getattr(args, "theta_cpu_csv", "") or "").strip()
    if raw.lower() in {"none", "warmup"}:
        return ""
    if raw and raw.lower() != "auto":
        return raw

    dataset = getattr(args, "dataset", None)
    candidate = default_theta_cpu_csv(dataset)
    if candidate is not None and candidate.exists():
        return str(candidate)
    if raw.lower() == "auto":
        raise RuntimeError(
            f"could not resolve --theta-cpu-csv auto for dataset={dataset!r}; "
            "pass --theta-cpu-csv explicitly or use --theta-cpu-csv warmup"
        )
    return ""


def stop_units(prefix: str) -> None:
    cmd = (
        "units=$(systemctl list-units "
        + "'" + prefix + "*' --no-legend --plain 2>/dev/null | awk '{print $1}'); "
        + "if [ -n \"$units\" ]; then sudo systemctl stop $units || true; fi"
    )
    subprocess.run(["bash", "-lc", cmd], check=False)


def failed_units(prefix: str) -> list[str]:
    result = subprocess.run(
        [
            "systemctl",
            "list-units",
            f"{prefix}*",
            "--all",
            "--no-legend",
            "--plain",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    failed = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[2] == "failed":
            failed.append(parts[0])
    return failed


def run_server_and_clients(
    *,
    base_dir: Path,
    python_bin: str,
    data_dir: Path,
    dataset: str,
    clients: int,
    rounds: int,
    model: str,
    server_bind: str,
    server_connect: str,
    csv_path: Path,
    log_path: Path,
    unit_prefix: str,
    cpu_quota: float | None = None,
    plan_csv: Path | None = None,
    plan_mode: str | None = None,
    reporting_fraction: float = 1.0,
    local_epochs: int = 1,
    local_steps: int = 20,
    batch_size: int = 64,
    client_lr: float = 0.003,
    server_lr: float = 1.0,
    server_momentum: float = 0.0,
    startup_wait: float = 8.0,
    timeout_seconds: float | None = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()

    server_cmd = [
        python_bin,
        "-m",
        "run_server",
        "--dataset",
        dataset,
        "--strategy",
        "custom-fedavgm",
        "--rounds",
        str(rounds),
        "--clients",
        str(clients),
        "--reporting-fraction",
        str(reporting_fraction),
        "--model",
        model,
        "--data-dir",
        str(data_dir),
        "--local-epochs",
        str(local_epochs),
        "--local-steps",
        str(local_steps),
        "--batch-size",
        str(batch_size),
        "--client-lr",
        str(client_lr),
        "--server-lr",
        str(server_lr),
        "--server-momentum",
        str(server_momentum),
        "--address",
        server_bind,
        "--csv-path",
        str(csv_path),
        "--downlink-num-bits",
        "0",
    ]

    with log_path.open("w", encoding="utf-8") as log_file:
        server = subprocess.Popen(
            server_cmd,
            cwd=base_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        time.sleep(startup_wait)
        if server.poll() is not None:
            raise RuntimeError(f"server exited early; see {log_path}")

        env = os.environ.copy()
        env["PY"] = python_bin
        env["DATASET"] = dataset
        env["MODEL"] = model
        env["LR"] = str(client_lr)
        env["LOCAL_STEPS"] = str(local_steps)
        env["BATCH_SIZE"] = str(batch_size)
        dataset_key = dataset.lower()
        env["NUM_CLASSES"] = (
            "10"
            if dataset_key in {"cifar", "cifar10", "cifar-10", "fmnist", "fashionmnist", "fashion-mnist", "mnist"}
            else "62"
        )
        env["UNIT_PREFIX"] = unit_prefix
        if cpu_quota is not None:
            env["CPU_QUOTA"] = str(cpu_quota)
        if plan_csv is not None:
            env["PLAN_CSV"] = str(plan_csv)
        if plan_mode is not None:
            env["PLAN_MODE"] = plan_mode

        launch_cmd = [
            "bash",
            "launch_clients.sh",
            str(data_dir),
            server_connect,
            str(clients),
            str(cpu_quota if cpu_quota is not None else 0),
        ]
        launch_result = subprocess.run(
            launch_cmd,
            cwd=base_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_file.write("\n--- launch_clients.sh ---\n")
        log_file.write(launch_result.stdout)
        log_file.flush()
        if launch_result.returncode != 0:
            os.killpg(server.pid, signal.SIGTERM)
            stop_units(unit_prefix)
            raise RuntimeError(f"client launch failed; see {log_path}")

        time.sleep(5.0)
        failed = failed_units(unit_prefix)
        if failed:
            os.killpg(server.pid, signal.SIGTERM)
            stop_units(unit_prefix)
            sample = ", ".join(failed[:5])
            raise RuntimeError(
                f"{len(failed)} client systemd unit(s) failed shortly after launch "
                f"({sample}); see {log_path} and `systemctl status {sample.split(', ')[0]}`"
            )

        timeout = timeout_seconds if timeout_seconds is not None else max(900.0, rounds * 300.0)
        try:
            server.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(server.pid, signal.SIGTERM)
            stop_units(unit_prefix)
            raise RuntimeError(f"server timed out after {timeout:.0f}s; see {log_path}") from exc
        finally:
            stop_units(unit_prefix)

        if server.returncode != 0:
            raise RuntimeError(f"server failed with exit code {server.returncode}; see {log_path}")


def command_warmup(args: argparse.Namespace) -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    quotas = parse_cpu_quotas(args.cpu_quotas)
    metadata = {
        "cpu_quotas": quotas,
        "clients": args.clients,
        "rounds": args.rounds,
        "dataset": args.dataset,
        "model": args.model,
        "local_epochs": args.local_epochs,
        "local_steps": args.local_steps,
        "batch_size": args.batch_size,
        "client_lr": args.client_lr,
        "data_dir": str(data_dir),
        "q_default": args.q,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    for quota in quotas:
        label = cpu_label(quota)
        print(f"[warmup] CPUQuota={quota}% rounds={args.rounds}")
        run_server_and_clients(
            base_dir=base_dir,
            python_bin=args.python,
            data_dir=data_dir,
            dataset=args.dataset,
            clients=args.clients,
            rounds=args.rounds,
            model=args.model,
            server_bind=args.server_bind,
            server_connect=args.server_connect,
            csv_path=out_dir / f"comm_times_cpu{label}.csv",
            log_path=out_dir / f"warmup_cpu{label}.log",
            unit_prefix=f"fl_client_slo_warmup_{run_id}_{label}_",
            cpu_quota=quota,
            reporting_fraction=args.reporting_fraction,
            local_epochs=args.local_epochs,
            local_steps=args.local_steps,
            batch_size=args.batch_size,
            client_lr=args.client_lr,
            server_lr=args.server_lr,
            server_momentum=args.server_momentum,
            startup_wait=args.startup_wait,
            timeout_seconds=args.timeout,
        )


def load_warmup_fits(
    warmup_dir: Path,
    drop_rounds: int,
    min_samples: int,
    *,
    id_column: str = "client_id",
    id_kind: str = "raw",
    data_dir: Path | None = None,
    clients: int | None = None,
) -> dict[str, list[tuple[float, LogisticFit]]]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    pattern = re.compile(r"comm_times_cpu(?P<cpu>[0-9]+(?:p[0-9]+)?)\.csv$")
    num_examples_to_client = (
        load_num_examples_to_client(data_dir, clients, allow_duplicates=id_kind == "auto")
        if id_kind in {"auto", "num_examples"} and data_dir is not None
        else None
    )

    for csv_file in sorted(warmup_dir.glob("comm_times_cpu*.csv")):
        match = pattern.match(csv_file.name)
        if not match:
            continue
        quota = float(match.group("cpu").replace("p", "."))
        with csv_file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    server_round = int(float(row.get("server_round", "0") or 0))
                except ValueError:
                    server_round = 0
                if server_round <= drop_rounds:
                    continue
                raw_id = str(row.get(id_column, "")).strip()
                if not raw_id:
                    continue
                try:
                    cid = canonical_client_id(
                        raw_id,
                        id_kind=id_kind,
                        num_examples_to_client=num_examples_to_client,
                        clients=clients,
                    )
                except ValueError:
                    continue
                raw_time = row.get("client_train_s", "")
                if raw_time == "":
                    continue
                try:
                    duration = float(raw_time)
                except ValueError:
                    continue
                if duration > 0.0 and math.isfinite(duration):
                    grouped[(cid, quota)].append(duration)

    by_client: dict[str, list[tuple[float, LogisticFit]]] = defaultdict(list)
    for (cid, quota), samples in grouped.items():
        if len(samples) < min_samples:
            continue
        by_client[cid].append((quota, fit_logistic(samples)))

    return {
        cid: sorted(fits, key=lambda item: item[0])
        for cid, fits in by_client.items()
        if len(fits) >= 2
    }


def round_probability(models: Iterable[ClientResourceModel], deadline: float, quotas: dict[str, float]) -> float:
    prob = 1.0
    for model in models:
        prob *= model.cdf_at(deadline, quotas[model.client_id])
    return float(prob)


def load_client_models_from_args(args: argparse.Namespace) -> list[ClientResourceModel]:
    theta_cpu_csv = resolve_theta_cpu_csv_arg(args)
    if theta_cpu_csv:
        return load_models_from_theta_cpu_csv(
            Path(theta_cpu_csv).resolve(),
            data_dir=Path(getattr(args, "data_dir", "data_partitions")).resolve(),
            id_kind=getattr(args, "id_kind", "auto"),
            clients=getattr(args, "clients", None) or None,
            cpu_unit=getattr(args, "theta_cpu_unit", "auto"),
            default_k_over_theta=getattr(args, "default_k_over_theta", 0.01),
        )

    warmup_dir = Path(getattr(args, "warmup_dir", "logs/slo_warmup")).resolve()
    fits = load_warmup_fits(
        warmup_dir,
        getattr(args, "drop_rounds", 1),
        getattr(args, "min_samples", 3),
        id_column=getattr(args, "fit_id_column", "client_id"),
        id_kind=getattr(args, "id_kind", "raw"),
        data_dir=Path(getattr(args, "data_dir", "data_partitions")).resolve()
        if hasattr(args, "data_dir")
        else None,
        clients=getattr(args, "clients", None) or None,
    )
    if not fits:
        raise RuntimeError(f"no usable warmup fits found in {warmup_dir}")
    return [
        fit_resource_model(cid, obs)
        for cid, obs in sorted(fits.items(), key=lambda item: client_sort_key(item[0]))
    ]


def solve_common_theta_exact(models: list[ClientResourceModel], deadline: float, q: float) -> float:
    def product_at(theta: float) -> float:
        value = 1.0
        for model in models:
            k = max(1e-6, model.k_over_theta * theta)
            value *= float(logistic_cdf(deadline, theta, k))
        return value

    low = 1e-6
    high = max(deadline * 2.0, max(model.theta_at(max(cpu for cpu, _ in model.observed)) for model in models) * 2.0)
    while product_at(high) > q:
        high *= 2.0
        if high > deadline * 1000.0:
            break
    for _ in range(100):
        mid = (low + high) / 2.0
        if product_at(mid) >= q:
            low = mid
        else:
            high = mid
    return low


def estimate_reference_a(models: list[ClientResourceModel]) -> float:
    thetas: list[float] = []
    scales: list[float] = []
    for model in models:
        for _, fit in model.observed:
            if fit.theta > 0.0 and fit.k > 0.0:
                thetas.append(float(fit.theta))
                scales.append(float(fit.k))
    if not thetas:
        return 0.01
    theta_arr = np.asarray(thetas, dtype=float)
    scale_arr = np.asarray(scales, dtype=float)
    denom = float(np.dot(theta_arr, theta_arr))
    if denom <= 0.0:
        return max(1e-6, float(np.median(scale_arr / np.clip(theta_arr, 1e-9, np.inf))))
    return max(1e-6, float(np.dot(theta_arr, scale_arr) / denom))


def parse_reference_a(raw: str, models: list[ClientResourceModel]) -> float:
    if str(raw).lower() == "auto":
        return estimate_reference_a(models)
    value = float(raw)
    if value <= 0.0:
        raise ValueError("reference coefficient a must be positive")
    return value


def solve_power_law_theta(deadline: float, q: float, gamma_sum: float, reference_a: float) -> float:
    gamma_sum = max(float(gamma_sum), 1e-9)
    den = 1.0 + reference_a * logit(q ** (1.0 / gamma_sum))
    if den <= 0.0:
        raise RuntimeError("power-law theta denominator is non-positive; choose a looser SLO")
    return float(deadline / den)


def quotas_for_theta(
    models: list[ClientResourceModel],
    theta_target: float,
    *,
    cpu_min: float,
    cpu_max: float,
) -> tuple[dict[str, float], dict[str, float], bool]:
    raw = {model.client_id: model.cpu_for_theta(theta_target) for model in models}
    quotas = {
        cid: min(cpu_max, max(cpu_min, quota if math.isfinite(quota) else cpu_max))
        for cid, quota in raw.items()
    }
    feasible = all(math.isfinite(raw[model.client_id]) and raw[model.client_id] <= cpu_max for model in models)
    return raw, quotas, feasible


def predicted_round_probability(
    models: list[ClientResourceModel],
    deadline: float,
    quotas: dict[str, float],
) -> float:
    prob = 1.0
    for model in models:
        prob *= model.cdf_at(deadline, quotas[model.client_id])
    return float(prob)


def choose_oracle_reference(
    models: list[ClientResourceModel],
    vanilla_quotas: dict[str, float],
    q: float,
    *,
    mode: str,
) -> ClientResourceModel:
    key_fn = lambda model: (
        model.theta_at(vanilla_quotas[model.client_id])
        + model.k_at(vanilla_quotas[model.client_id]) * logit(q)
    )
    if mode == "fastest-aligned":
        return min(models, key=key_fn)
    if mode == "slowest-client":
        return max(models, key=key_fn)
    raise ValueError(f"unsupported oracle mode: {mode}")


def solve_oracle_theta(
    models: list[ClientResourceModel],
    *,
    deadline: float,
    q: float,
    vanilla_quotas: dict[str, float],
    oracle_mode: str,
) -> tuple[float, ClientResourceModel, float]:
    reference = choose_oracle_reference(models, vanilla_quotas, q, mode=oracle_mode)
    reference_quota = vanilla_quotas[reference.client_id]
    reference_theta = reference.theta_at(reference_quota)
    reference_a = reference.k_at(reference_quota) / max(reference_theta, 1e-9)
    if oracle_mode == "fastest-aligned":
        theta = solve_power_law_theta(deadline, q, len(models), reference_a)
    else:
        theta = solve_power_law_theta(deadline, q, 1.0, reference_a)
    return theta, reference, reference_a


def solve_fedpacer_power_law(
    models: list[ClientResourceModel],
    *,
    deadline: float,
    q: float,
    reference_a: float,
    cpu_min: float,
    cpu_max: float,
    feedback_iterations: int,
    gamma_min: float,
    gamma_max: float,
    tol: float,
) -> tuple[float, dict[str, float], dict[str, float], dict[str, float], list[dict[str, float]], bool]:
    gamma_by_client = {model.client_id: 1.0 for model in models}
    history: list[dict[str, float]] = []
    previous_theta: float | None = None
    final_raw: dict[str, float] = {}
    final_quotas: dict[str, float] = {}
    final_feasible = True

    iterations = max(0, feedback_iterations)
    for iteration in range(iterations + 1):
        gamma_sum = sum(gamma_by_client.values())
        theta = solve_power_law_theta(deadline, q, gamma_sum, reference_a)
        final_raw, final_quotas, final_feasible = quotas_for_theta(
            models,
            theta,
            cpu_min=cpu_min,
            cpu_max=cpu_max,
        )
        actual_prob = predicted_round_probability(models, deadline, final_quotas)
        ref_prob = float(logistic_cdf(deadline, theta, max(reference_a * theta, 1e-9)))
        history.append(
            {
                "iteration": float(iteration),
                "theta_target": theta,
                "gamma_sum": gamma_sum,
                "reference_client_cdf": ref_prob,
                "actual_model_probability": actual_prob,
                "total_cpu_quota": sum(final_quotas.values()),
            }
        )
        if iteration == iterations:
            return theta, final_raw, final_quotas, gamma_by_client, history, final_feasible

        log_ref = math.log(min(max(ref_prob, 1e-12), 1.0 - 1e-12))
        updated: dict[str, float] = {}
        for model in models:
            client_prob = model.cdf_at(deadline, final_quotas[model.client_id])
            log_client = math.log(min(max(client_prob, 1e-12), 1.0 - 1e-12))
            gamma = log_client / log_ref if log_ref != 0.0 else 1.0
            updated[model.client_id] = min(gamma_max, max(gamma_min, gamma))

        if previous_theta is not None and abs(theta - previous_theta) <= tol:
            gamma_by_client = updated
            theta = solve_power_law_theta(deadline, q, sum(gamma_by_client.values()), reference_a)
            final_raw, final_quotas, final_feasible = quotas_for_theta(
                models,
                theta,
                cpu_min=cpu_min,
                cpu_max=cpu_max,
            )
            return theta, final_raw, final_quotas, gamma_by_client, history, final_feasible

        previous_theta = theta
        gamma_by_client = updated

    raise AssertionError("unreachable")


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of an empty sequence")
    try:
        return float(np.quantile(np.asarray(values, dtype=float), q, method="linear"))
    except TypeError:
        return float(np.quantile(np.asarray(values, dtype=float), q, interpolation="linear"))


def round_times_from_csv(csv_path: Path, *, drop_rounds: int = 0) -> list[float]:
    by_round: dict[int, list[float]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                server_round = int(float(row.get("server_round", "0") or 0))
                duration = float(row.get("client_train_s", "") or "nan")
            except ValueError:
                continue
            if server_round > drop_rounds and math.isfinite(duration):
                by_round[server_round].append(duration)
    return [max(values) for _, values in sorted(by_round.items()) if values]


def read_plan_rows(plan_csv: Path, mode: str | None = None) -> list[dict[str, str]]:
    with plan_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if mode is None:
        return rows
    return [row for row in rows if row.get("mode") == mode]


def normalize_plan_client_id(raw: object) -> str:
    value = str(raw).strip()
    try:
        return str(int(float(value)))
    except ValueError:
        return value


def load_plan_quotas(plan_csv: Path, mode: str) -> dict[str, float]:
    rows = read_plan_rows(plan_csv, mode)
    if not rows:
        raise RuntimeError(f"{plan_csv} has no rows for mode={mode!r}")
    quotas = {}
    for row in rows:
        quotas[normalize_plan_client_id(row["client_id"])] = float(row["cpu_quota"])
    return quotas


def plan_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    preferred = [
        "mode",
        "client_id",
        "cpu_quota",
        "cpu",
        "deadline",
        "q",
        "theta_target",
        "predicted_theta",
        "predicted_k",
        "target_step_s",
        "target_theta",
        "raw_cpu_quota",
        "clamped",
        "feasible",
        "slo_model_probability",
        "actual_model_probability",
        "total_cpu_quota",
        "total_cpu_cores",
        "reference_a",
        "gamma",
        "gamma_sum",
        "oracle_reference_client",
        "oracle_reference_a",
        "alpha",
        "beta",
        "gamma_floor",
        "k_over_theta",
    ]
    seen = set()
    fieldnames = []
    for name in preferred:
        if any(name in row for row in rows):
            fieldnames.append(name)
            seen.add(name)
    for row in rows:
        for name in row:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)
    return fieldnames


def write_plan_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty plan")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = plan_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def mode_rows_for_quotas(
    *,
    mode: str,
    models: list[ClientResourceModel],
    quotas: dict[str, float],
    raw_quotas: dict[str, float] | None = None,
    deadline: float | None = None,
    q: float | None = None,
    theta_target: float | None = None,
    target_thetas: dict[str, float] | None = None,
    target_steps: dict[str, float] | None = None,
    reference_a: float | None = None,
    gamma_by_client: dict[str, float] | None = None,
    gamma_sum: float | None = None,
    oracle_reference_client: str = "",
    oracle_reference_a: float | None = None,
    feasible: bool = True,
) -> list[dict[str, object]]:
    actual_prob = predicted_round_probability(models, deadline, quotas) if deadline is not None else None
    total_cpu = sum(quotas.values())
    rows: list[dict[str, object]] = []
    for model in models:
        cid = model.client_id
        quota = quotas[cid]
        raw_quota = raw_quotas[cid] if raw_quotas and cid in raw_quotas else quota
        predicted_theta = model.theta_at(quota)
        predicted_k = model.k_at(quota)
        rows.append(
            {
                "mode": mode,
                "client_id": cid,
                "cpu_quota": f"{quota:.4f}",
                "cpu": f"{quota / 100.0:.6f}",
                "deadline": f"{deadline:.6f}" if deadline is not None else "",
                "q": f"{q:.6f}" if q is not None else "",
                "theta_target": f"{theta_target:.6f}" if theta_target is not None else "",
                "predicted_theta": f"{predicted_theta:.6f}",
                "predicted_k": f"{predicted_k:.6f}",
                "target_step_s": f"{target_steps[cid]:.6f}" if target_steps and cid in target_steps else "",
                "target_theta": f"{target_thetas[cid]:.6f}" if target_thetas and cid in target_thetas else "",
                "raw_cpu_quota": f"{raw_quota:.6f}" if math.isfinite(raw_quota) else "inf",
                "clamped": "yes" if abs(quota - raw_quota) > 1e-9 else "no",
                "feasible": str(feasible),
                "slo_model_probability": f"{actual_prob:.8f}" if actual_prob is not None else "",
                "actual_model_probability": f"{actual_prob:.8f}" if actual_prob is not None else "",
                "total_cpu_quota": f"{total_cpu:.4f}",
                "total_cpu_cores": f"{total_cpu / 100.0:.6f}",
                "reference_a": f"{reference_a:.8f}" if reference_a is not None else "",
                "gamma": f"{gamma_by_client[cid]:.8f}" if gamma_by_client and cid in gamma_by_client else "",
                "gamma_sum": f"{gamma_sum:.8f}" if gamma_sum is not None else "",
                "oracle_reference_client": oracle_reference_client,
                "oracle_reference_a": f"{oracle_reference_a:.8f}" if oracle_reference_a is not None else "",
                "alpha": f"{model.alpha:.10f}",
                "beta": f"{model.beta:.10f}",
                "gamma_floor": f"{model.gamma:.10f}",
                "k_over_theta": f"{model.k_over_theta:.10f}",
            }
        )
    return rows


def command_plan_slo(args: argparse.Namespace) -> None:
    models = load_client_models_from_args(args)
    q = args.q

    if args.deadline == "auto":
        deadline_cpu = args.deadline_cpu
        per_client_tail = []
        for model in models:
            theta = model.theta_at(deadline_cpu)
            k = model.k_at(deadline_cpu)
            per_client_tail.append((theta + k * logit(q), model.client_id))
        deadline, slowest_client = max(per_client_tail, key=lambda item: item[0])
        print(
            f"[plan] auto deadline D={deadline:.4f}s from slowest client {slowest_client} "
            f"at CPUQuota={deadline_cpu}% and client-level q={q}"
        )
    else:
        deadline = float(args.deadline)
        slowest_client = ""

    n_clients = len(models)
    fastest_ref = min(models, key=lambda model: model.k_over_theta)
    q_root = q ** (1.0 / n_clients)
    oracle_den = 1.0 + fastest_ref.k_over_theta * logit(q_root)
    if oracle_den <= 0.0:
        raise RuntimeError("oracle target denominator is non-positive; choose a looser SLO")
    theta_oracle = deadline / oracle_den
    theta_fedpacer = solve_common_theta_exact(models, deadline, q)

    rows = []
    summaries = {}
    for mode, theta_target in [("oracle", theta_oracle), ("fedpacer", theta_fedpacer)]:
        quotas_raw = {model.client_id: model.cpu_for_theta(theta_target) for model in models}
        quotas = {
            cid: min(args.cpu_max, max(args.cpu_min, quota if math.isfinite(quota) else args.cpu_max))
            for cid, quota in quotas_raw.items()
        }
        feasible = all(math.isfinite(quotas_raw[model.client_id]) and quotas_raw[model.client_id] <= args.cpu_max for model in models)
        actual_prob = round_probability(models, deadline, quotas)
        if mode == "oracle":
            slo_model_prob = float(logistic_cdf(deadline, theta_target, fastest_ref.k_over_theta * theta_target)) ** n_clients
        else:
            slo_model_prob = actual_prob
        total_cpu = sum(quotas.values())
        summaries[mode] = {
            "theta_target": theta_target,
            "slo_model_probability": slo_model_prob,
            "actual_model_probability": actual_prob,
            "total_cpu_quota": total_cpu,
            "feasible": feasible,
        }
        for model in models:
            raw_quota = quotas_raw[model.client_id]
            quota = quotas[model.client_id]
            rows.append(
                {
                    "mode": mode,
                    "client_id": model.client_id,
                    "cpu_quota": f"{quota:.4f}",
                    "raw_cpu_quota": f"{raw_quota:.4f}" if math.isfinite(raw_quota) else "inf",
                    "deadline": f"{deadline:.6f}",
                    "q": f"{q:.6f}",
                    "theta_target": f"{theta_target:.6f}",
                    "slo_model_probability": f"{slo_model_prob:.8f}",
                    "actual_model_probability": f"{actual_prob:.8f}",
                    "total_cpu_quota": f"{total_cpu:.4f}",
                    "feasible": str(feasible),
                    "alpha": f"{model.alpha:.8f}",
                    "beta": f"{model.beta:.8f}",
                    "gamma": f"{model.gamma:.8f}",
                    "k_over_theta": f"{model.k_over_theta:.8f}",
                    "fastest_ref_client": fastest_ref.client_id,
                    "auto_slowest_client": slowest_client,
                }
            )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "deadline": deadline,
        "q": q,
        "clients": n_clients,
        "deadline_mode": args.deadline,
        "deadline_cpu": args.deadline_cpu,
        "fastest_ref_client": fastest_ref.client_id,
        "auto_slowest_client": slowest_client,
        "modes": summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[plan] wrote {out_path}")
    print(f"[plan] wrote {summary_path}")
    print(json.dumps(summary, indent=2))


def build_table4_vanilla_rows(
    *,
    models: list[ClientResourceModel],
    dataset: str,
    clients: int,
    heterogeneity: str,
    local_steps: int,
    cpu_min: float,
    cpu_max: float,
    ref_cpu: float,
    mean_step_s: float | None = None,
    std_step_s: float | None = None,
    exp_lambda: float | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    step_times, setting = table4_step_times(
        dataset=dataset,
        clients=clients,
        heterogeneity=heterogeneity,
        mean_step_s=mean_step_s,
        std_step_s=std_step_s,
        exp_lambda=exp_lambda,
    )
    if len(models) != clients:
        raise ValueError(f"loaded {len(models)} client models, but --clients={clients}")

    ranked_models = sorted(models, key=lambda model: model.theta_at(ref_cpu))
    target_thetas: dict[str, float] = {}
    target_steps: dict[str, float] = {}
    raw_quotas: dict[str, float] = {}
    quotas: dict[str, float] = {}
    for model, step_time in zip(ranked_models, step_times):
        target_theta = float(step_time) * float(local_steps)
        raw_quota = model.cpu_for_theta(target_theta)
        quota = min(cpu_max, max(cpu_min, raw_quota if math.isfinite(raw_quota) else cpu_max))
        target_thetas[model.client_id] = target_theta
        target_steps[model.client_id] = float(step_time)
        raw_quotas[model.client_id] = raw_quota
        quotas[model.client_id] = quota

    rows = mode_rows_for_quotas(
        mode="vanilla",
        models=models,
        quotas=quotas,
        raw_quotas=raw_quotas,
        target_thetas=target_thetas,
        target_steps=target_steps,
        feasible=all(math.isfinite(raw_quotas[model.client_id]) and raw_quotas[model.client_id] <= cpu_max for model in models),
    )
    summary = {
        "dataset": dataset,
        "clients": clients,
        "heterogeneity": heterogeneity,
        "table4_setting": {
            "dataset": setting.dataset,
            "clients": setting.clients,
            "heterogeneity": setting.heterogeneity,
            "distribution": setting.distribution,
            "mean_step_s": setting.mean_step_s,
            "std_step_s": setting.std_step_s,
            "exp_lambda": setting.exp_lambda,
        },
        "local_steps": local_steps,
        "step_time_min": min(step_times),
        "step_time_mean": float(np.mean(step_times)),
        "step_time_std": float(np.std(step_times, ddof=1)) if len(step_times) > 1 else 0.0,
        "step_time_max": max(step_times),
        "target_theta_min": min(target_thetas.values()),
        "target_theta_mean": float(np.mean(list(target_thetas.values()))),
        "target_theta_max": max(target_thetas.values()),
        "total_cpu_quota": sum(quotas.values()),
        "total_cpu_cores": sum(quotas.values()) / 100.0,
    }
    return rows, summary


def command_table4_plan(args: argparse.Namespace) -> None:
    models = load_client_models_from_args(args)
    rows, summary = build_table4_vanilla_rows(
        models=models,
        dataset=args.dataset,
        clients=args.clients,
        heterogeneity=args.heterogeneity,
        local_steps=args.local_steps,
        cpu_min=args.cpu_min,
        cpu_max=args.cpu_max,
        ref_cpu=args.ref_cpu,
        mean_step_s=args.mean_step_s,
        std_step_s=args.std_step_s,
        exp_lambda=args.exp_lambda,
    )
    out_path = Path(args.out).resolve()
    write_plan_csv(out_path, rows)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[table4-plan] wrote {out_path}")
    print(f"[table4-plan] wrote {summary_path}")
    print(json.dumps(summary, indent=2))


def build_baseline_rows_from_deadline(
    *,
    models: list[ClientResourceModel],
    vanilla_rows: list[dict[str, str]],
    deadline: float,
    q: float,
    oracle_mode: str,
    cpu_min: float,
    cpu_max: float,
    reference_a: float,
    fedpacer_feedback_iterations: int,
    gamma_min: float,
    gamma_max: float,
    gamma_tol: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    model_ids = {model.client_id for model in models}
    vanilla_quotas = {normalize_plan_client_id(row["client_id"]): float(row["cpu_quota"]) for row in vanilla_rows}
    missing = sorted(model_ids - set(vanilla_quotas), key=client_sort_key)
    if missing:
        raise RuntimeError(f"vanilla plan is missing client_id(s): {', '.join(missing)}")

    vanilla_raw = {cid: quota for cid, quota in vanilla_quotas.items()}
    vanilla_target_thetas = {
        normalize_plan_client_id(row["client_id"]): float(row["target_theta"])
        for row in vanilla_rows
        if row.get("target_theta")
    }
    vanilla_target_steps = {
        normalize_plan_client_id(row["client_id"]): float(row["target_step_s"])
        for row in vanilla_rows
        if row.get("target_step_s")
    }

    oracle_theta, oracle_ref, oracle_a = solve_oracle_theta(
        models,
        deadline=deadline,
        q=q,
        vanilla_quotas=vanilla_quotas,
        oracle_mode=oracle_mode,
    )
    oracle_raw, oracle_quotas, oracle_feasible = quotas_for_theta(
        models,
        oracle_theta,
        cpu_min=cpu_min,
        cpu_max=cpu_max,
    )

    (
        fedpacer_theta,
        fedpacer_raw,
        fedpacer_quotas,
        fedpacer_gamma,
        fedpacer_history,
        fedpacer_feasible,
    ) = solve_fedpacer_power_law(
        models,
        deadline=deadline,
        q=q,
        reference_a=reference_a,
        cpu_min=cpu_min,
        cpu_max=cpu_max,
        feedback_iterations=fedpacer_feedback_iterations,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        tol=gamma_tol,
    )

    rows: list[dict[str, object]] = []
    rows.extend(
        mode_rows_for_quotas(
            mode="vanilla",
            models=models,
            quotas=vanilla_quotas,
            raw_quotas=vanilla_raw,
            deadline=deadline,
            q=q,
            target_thetas=vanilla_target_thetas,
            target_steps=vanilla_target_steps,
        )
    )
    rows.extend(
        mode_rows_for_quotas(
            mode="oracle",
            models=models,
            quotas=oracle_quotas,
            raw_quotas=oracle_raw,
            deadline=deadline,
            q=q,
            theta_target=oracle_theta,
            oracle_reference_client=oracle_ref.client_id,
            oracle_reference_a=oracle_a,
            feasible=oracle_feasible,
        )
    )
    rows.extend(
        mode_rows_for_quotas(
            mode="fedpacer",
            models=models,
            quotas=fedpacer_quotas,
            raw_quotas=fedpacer_raw,
            deadline=deadline,
            q=q,
            theta_target=fedpacer_theta,
            reference_a=reference_a,
            gamma_by_client=fedpacer_gamma,
            gamma_sum=sum(fedpacer_gamma.values()),
            feasible=fedpacer_feasible,
        )
    )

    cpu_vanilla = sum(vanilla_quotas.values()) / 100.0
    cpu_oracle = sum(oracle_quotas.values()) / 100.0
    cpu_fedpacer = sum(fedpacer_quotas.values()) / 100.0
    summary = {
        "deadline": deadline,
        "q": q,
        "clients": len(models),
        "oracle_mode": oracle_mode,
        "reference_a": reference_a,
        "cpu_vanilla": cpu_vanilla,
        "cpu_oracle": cpu_oracle,
        "cpu_fedpacer": cpu_fedpacer,
        "oracle_gap_percent": ((cpu_fedpacer - cpu_oracle) / cpu_oracle * 100.0) if cpu_oracle > 0.0 else math.nan,
        "vanilla_fedavg_gap_percent": ((cpu_vanilla - cpu_fedpacer) / cpu_vanilla * 100.0) if cpu_vanilla > 0.0 else math.nan,
        "oracle": {
            "theta_target": oracle_theta,
            "reference_client": oracle_ref.client_id,
            "reference_a": oracle_a,
            "mode": oracle_mode,
            "actual_model_probability": predicted_round_probability(models, deadline, oracle_quotas),
            "feasible": oracle_feasible,
        },
        "fedpacer": {
            "theta_target": fedpacer_theta,
            "gamma_sum": sum(fedpacer_gamma.values()),
            "actual_model_probability": predicted_round_probability(models, deadline, fedpacer_quotas),
            "feedback_iterations": fedpacer_feedback_iterations,
            "history": fedpacer_history,
            "feasible": fedpacer_feasible,
        },
    }
    return rows, summary


def command_plan_baselines(args: argparse.Namespace) -> None:
    models = load_client_models_from_args(args)
    vanilla_plan = Path(args.vanilla_plan).resolve()
    vanilla_rows = read_plan_rows(vanilla_plan, args.vanilla_mode)
    if not vanilla_rows:
        raise RuntimeError(f"{vanilla_plan} has no rows for mode={args.vanilla_mode!r}")

    if args.deadline == "from-vanilla":
        if not args.vanilla_run_csv:
            raise ValueError("--vanilla-run-csv is required when --deadline=from-vanilla")
        round_times = round_times_from_csv(Path(args.vanilla_run_csv).resolve(), drop_rounds=args.drop_rounds)
        if not round_times:
            raise RuntimeError(f"no round times found in {args.vanilla_run_csv}")
        deadline = quantile(round_times, args.q)
        deadline_source = {
            "type": "vanilla_run_quantile",
            "csv": str(Path(args.vanilla_run_csv).resolve()),
            "rounds": len(round_times),
            "quantile": args.q,
            "round_time_p50": quantile(round_times, 0.50),
            "round_time_p90": quantile(round_times, 0.90),
            "round_time_p95": quantile(round_times, 0.95),
            "round_time_max": max(round_times),
        }
    else:
        deadline = float(args.deadline)
        deadline_source = {"type": "manual"}

    reference_a = parse_reference_a(args.reference_a, models)
    rows, summary = build_baseline_rows_from_deadline(
        models=models,
        vanilla_rows=vanilla_rows,
        deadline=deadline,
        q=args.q,
        oracle_mode=args.oracle_mode,
        cpu_min=args.cpu_min,
        cpu_max=args.cpu_max,
        reference_a=reference_a,
        fedpacer_feedback_iterations=args.fedpacer_feedback_iterations,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        gamma_tol=args.gamma_tol,
    )
    summary["deadline_source"] = deadline_source
    summary["vanilla_plan"] = str(vanilla_plan)

    out_path = Path(args.out).resolve()
    write_plan_csv(out_path, rows)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[plan-baselines] wrote {out_path}")
    print(f"[plan-baselines] wrote {summary_path}")
    print(json.dumps(summary, indent=2))


def command_pipeline_table4(args: argparse.Namespace) -> None:
    models = load_client_models_from_args(args)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")

    vanilla_plan = out_dir / f"table4_vanilla_plan_{run_id}.csv"
    vanilla_rows, vanilla_plan_summary = build_table4_vanilla_rows(
        models=models,
        dataset=args.dataset,
        clients=args.clients,
        heterogeneity=args.heterogeneity,
        local_steps=args.local_steps,
        cpu_min=args.cpu_min,
        cpu_max=args.cpu_max,
        ref_cpu=args.ref_cpu,
        mean_step_s=args.mean_step_s,
        std_step_s=args.std_step_s,
        exp_lambda=args.exp_lambda,
    )
    write_plan_csv(vanilla_plan, vanilla_rows)
    vanilla_plan.with_suffix(".summary.json").write_text(
        json.dumps(vanilla_plan_summary, indent=2),
        encoding="utf-8",
    )

    print(f"[pipeline] running vanilla FedAvg with Table 4 CPU plan: {vanilla_plan}")
    vanilla_csv = out_dir / f"vanilla_{run_id}.csv"
    run_server_and_clients(
        base_dir=Path(__file__).resolve().parent,
        python_bin=args.python,
        data_dir=Path(args.data_dir).resolve(),
        dataset=args.dataset,
        clients=args.clients,
        rounds=args.vanilla_rounds,
        model=args.model,
        server_bind=args.server_bind,
        server_connect=args.server_connect,
        csv_path=vanilla_csv,
        log_path=out_dir / f"vanilla_{run_id}.log",
        unit_prefix=f"fl_client_slo_vanilla_{run_id}_",
        plan_csv=vanilla_plan,
        plan_mode="vanilla",
        reporting_fraction=args.reporting_fraction,
        local_epochs=args.local_epochs,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        client_lr=args.client_lr,
        server_lr=args.server_lr,
        server_momentum=args.server_momentum,
        startup_wait=args.startup_wait,
        timeout_seconds=args.timeout,
    )
    round_times = round_times_from_csv(vanilla_csv, drop_rounds=args.drop_rounds)
    if not round_times:
        raise RuntimeError(f"vanilla run produced no round timing rows: {vanilla_csv}")
    deadline = quantile(round_times, args.q)
    reference_a = parse_reference_a(args.reference_a, models)
    rows, summary = build_baseline_rows_from_deadline(
        models=models,
        vanilla_rows=vanilla_rows,
        deadline=deadline,
        q=args.q,
        oracle_mode=args.oracle_mode,
        cpu_min=args.cpu_min,
        cpu_max=args.cpu_max,
        reference_a=reference_a,
        fedpacer_feedback_iterations=args.fedpacer_feedback_iterations,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        gamma_tol=args.gamma_tol,
    )
    summary["deadline_source"] = {
        "type": "pipeline_vanilla_run_quantile",
        "csv": str(vanilla_csv),
        "rounds": len(round_times),
        "quantile": args.q,
        "round_time_p50": quantile(round_times, 0.50),
        "round_time_p90": quantile(round_times, 0.90),
        "round_time_p95": quantile(round_times, 0.95),
        "round_time_max": max(round_times),
    }
    summary["vanilla_plan"] = str(vanilla_plan)
    summary["vanilla_plan_summary"] = vanilla_plan_summary

    final_plan = out_dir / f"slo_plan_from_vanilla_p{int(args.q * 100):02d}_{run_id}.csv"
    write_plan_csv(final_plan, rows)
    final_plan.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[pipeline] wrote {final_plan}")
    print(json.dumps(summary, indent=2))


def load_plan_deadline(plan_csv: Path) -> tuple[float, float, int]:
    with plan_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty plan CSV: {plan_csv}")
    modes = {row["mode"] for row in rows}
    clients = len({row["client_id"] for row in rows if row["mode"] == next(iter(modes))})
    return float(rows[0]["deadline"]), float(rows[0]["q"]), clients


def summarize_run(csv_path: Path, deadline: float, window: int) -> dict[str, float | int]:
    round_max = round_times_from_csv(csv_path)
    if not round_max:
        return {"rounds": 0, "overall_success_rate": 0.0, "last_window_success_rate": 0.0, "min_window_success_rate": 0.0}

    successes = [1.0 if value <= deadline else 0.0 for value in round_max]
    window_rates = [
        sum(successes[idx : idx + window]) / float(window)
        for idx in range(0, len(successes) - window + 1)
    ]
    if not window_rates:
        window_rates = [sum(successes) / float(len(successes))]
    return {
        "rounds": len(round_max),
        "overall_success_rate": sum(successes) / float(len(successes)),
        "last_window_success_rate": window_rates[-1],
        "min_window_success_rate": min(window_rates),
        "max_round_time": max(round_max),
        "round_time_p50": quantile(round_max, 0.50),
        "round_time_p90": quantile(round_max, 0.90),
        "round_time_p95": quantile(round_max, 0.95),
    }


def command_run(args: argparse.Namespace) -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve()
    plan_csv = Path(args.plan).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    deadline, q, clients_in_plan = load_plan_deadline(plan_csv)
    clients = args.clients or clients_in_plan
    run_id = time.strftime("%Y%m%d_%H%M%S")

    for mode in args.modes.split(","):
        mode = mode.strip()
        if not mode:
            continue
        print(f"[run] mode={mode} rounds={args.rounds} D={deadline:.4f}s q={q}")
        csv_path = out_dir / f"{mode}_{run_id}.csv"
        run_server_and_clients(
            base_dir=base_dir,
            python_bin=args.python,
            data_dir=data_dir,
            dataset=args.dataset,
            clients=clients,
            rounds=args.rounds,
            model=args.model,
            server_bind=args.server_bind,
            server_connect=args.server_connect,
            csv_path=csv_path,
            log_path=out_dir / f"{mode}_{run_id}.log",
            unit_prefix=f"fl_client_slo_{mode}_{run_id}_",
            plan_csv=plan_csv,
            plan_mode=mode,
            reporting_fraction=args.reporting_fraction,
            local_epochs=args.local_epochs,
            local_steps=args.local_steps,
            batch_size=args.batch_size,
            client_lr=args.client_lr,
            server_lr=args.server_lr,
            server_momentum=args.server_momentum,
            startup_wait=args.startup_wait,
            timeout_seconds=args.timeout,
        )
        summary = summarize_run(csv_path, deadline, args.window)
        summary_path = csv_path.with_suffix(".summary.json")
        payload = {"mode": mode, "deadline": deadline, "q": q, "window": args.window, **summary}
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))


def add_model_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--theta-cpu-csv",
        default="",
        help="Optional id,theta,k,cpu CSV. Use 'auto' for dataset defaults or 'warmup' to force --warmup-dir.",
    )
    parser.add_argument("--theta-cpu-unit", choices=["auto", "fraction", "percent"], default="auto")
    parser.add_argument(
        "--default-k-over-theta",
        type=float,
        default=0.01,
        help="Fallback k/theta ratio when a theta/cpu CSV has an empty k column.",
    )
    parser.add_argument("--warmup-dir", default="logs/slo_warmup")
    parser.add_argument("--data-dir", default="data_partitions")
    parser.add_argument("--id-kind", choices=["auto", "raw", "cid", "num_examples"], default="auto")
    parser.add_argument("--fit-id-column", default="client_id")
    parser.add_argument("--drop-rounds", type=int, default=1)
    parser.add_argument("--min-samples", type=int, default=3)


def add_resource_bound_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cpu-min", type=float, default=1.0, help="Minimum systemd CPUQuota percent")
    parser.add_argument("--cpu-max", type=float, default=100.0, help="Maximum systemd CPUQuota percent")


def add_table4_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--heterogeneity", choices=["normal", "homogeneous", "exponential"], default="normal")
    parser.add_argument("--local-steps", type=int, default=20)
    parser.add_argument("--ref-cpu", type=float, default=60.0, help="CPUQuota percent used to rank clients by native speed")
    parser.add_argument("--mean-step-s", type=float, default=None, help="Override Table 4 mean step time")
    parser.add_argument("--std-step-s", type=float, default=None, help="Override Table 4 std step time")
    parser.add_argument("--exp-lambda", type=float, default=None, help="Override Table 4 exponential lambda/rate")


def add_fedavg_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--server-bind", default="0.0.0.0:8081")
    parser.add_argument("--server-connect", default="127.0.0.1:8081")
    parser.add_argument("--reporting-fraction", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--client-lr", type=float, default=0.01)
    parser.add_argument("--server-lr", type=float, default=1.0, help="1.0 with momentum 0.0 gives FedAvg-equivalent aggregation")
    parser.add_argument("--server-momentum", type=float, default=0.0, help="0.0 gives FedAvg-equivalent aggregation")
    parser.add_argument("--startup-wait", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=None)


def add_baseline_planning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--q", type=float, default=0.90)
    parser.add_argument(
        "--oracle-mode",
        choices=["slowest-client", "fastest-aligned"],
        default="slowest-client",
        help="slowest-client uses the vanilla slowest client's single-client SLO; fastest-aligned is the optimistic paper-style lower bound.",
    )
    parser.add_argument("--reference-a", default="auto", help="'auto' or the FedPacer pooled k/theta coefficient a")
    parser.add_argument("--fedpacer-feedback-iterations", type=int, default=4)
    parser.add_argument("--gamma-min", type=float, default=1e-3)
    parser.add_argument("--gamma-max", type=float, default=1e3)
    parser.add_argument("--gamma-tol", type=float, default=1e-3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm up, fit, plan, and run FEMNIST SLO resource experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    warmup = sub.add_parser("warmup")
    warmup.add_argument("--data-dir", default="data_partitions")
    warmup.add_argument("--clients", type=int, default=20)
    warmup.add_argument("--cpu-quotas", default="40,60,80")
    warmup.add_argument("--rounds", type=int, default=10)
    warmup.add_argument("--dataset", default="femnist")
    warmup.add_argument("--model", default="cnn", choices=["cnn", "tf_example", "resnet18", "resnet20", "mobilenet_v2_075", "mobilenet_v2_100"])
    warmup.add_argument("--q", type=float, default=0.95)
    warmup.add_argument("--out-dir", default="logs/slo_warmup")
    warmup.add_argument("--python", default=sys.executable)
    warmup.add_argument("--server-bind", default="0.0.0.0:8081")
    warmup.add_argument("--server-connect", default="127.0.0.1:8081")
    warmup.add_argument("--reporting-fraction", type=float, default=1.0)
    warmup.add_argument("--local-epochs", type=int, default=1)
    warmup.add_argument("--local-steps", type=int, default=20)
    warmup.add_argument("--batch-size", type=int, default=64)
    warmup.add_argument("--client-lr", type=float, default=0.003)
    warmup.add_argument("--server-lr", type=float, default=1.0)
    warmup.add_argument("--server-momentum", type=float, default=0.0)
    warmup.add_argument("--startup-wait", type=float, default=8.0)
    warmup.add_argument("--timeout", type=float, default=None)
    warmup.set_defaults(func=command_warmup)

    plan = sub.add_parser("plan-slo")
    add_model_source_args(plan)
    plan.add_argument("--deadline", default="auto", help="'auto' or a deadline in seconds")
    plan.add_argument("--deadline-cpu", type=float, default=95.0, help="CPUQuota used to auto-select D from the slowest client")
    plan.add_argument("--q", type=float, default=0.95)
    plan.add_argument("--clients", type=int, default=0)
    add_resource_bound_args(plan)
    plan.add_argument("--out", default="logs/slo_plan.csv")
    plan.set_defaults(func=command_plan_slo)

    table4 = sub.add_parser("table4-plan", help="Generate the fixed vanilla FedAvg CPU plan from Table 4 speeds.")
    add_model_source_args(table4)
    add_table4_args(table4)
    add_resource_bound_args(table4)
    table4.add_argument("--out", default="logs/table4_vanilla_plan.csv")
    table4.set_defaults(func=command_table4_plan, theta_cpu_csv="auto")

    baselines = sub.add_parser(
        "plan-baselines",
        help="Use a vanilla run p-quantile deadline to derive vanilla/oracle/FedPacer CPU allocations.",
    )
    add_model_source_args(baselines)
    add_resource_bound_args(baselines)
    add_baseline_planning_args(baselines)
    baselines.add_argument("--dataset", default="", help="Dataset name used only for --theta-cpu-csv auto")
    baselines.add_argument("--clients", type=int, default=0)
    baselines.add_argument("--vanilla-plan", default="logs/table4_vanilla_plan.csv")
    baselines.add_argument("--vanilla-mode", default="vanilla")
    baselines.add_argument("--vanilla-run-csv", default="")
    baselines.add_argument("--deadline", default="from-vanilla", help="'from-vanilla' or a deadline in seconds")
    baselines.add_argument("--out", default="logs/slo_plan_from_vanilla.csv")
    baselines.set_defaults(func=command_plan_baselines)

    pipeline = sub.add_parser(
        "pipeline-table4",
        help="Generate Table 4 vanilla plan, run vanilla FedAvg, then derive oracle/FedPacer allocations from vanilla p90.",
    )
    add_model_source_args(pipeline)
    add_table4_args(pipeline)
    add_resource_bound_args(pipeline)
    add_baseline_planning_args(pipeline)
    add_fedavg_run_args(pipeline)
    pipeline.add_argument("--model", default="resnet18", choices=["cnn", "tf_example", "resnet18", "resnet20", "mobilenet_v2_075", "mobilenet_v2_100"])
    pipeline.add_argument("--vanilla-rounds", type=int, default=50)
    pipeline.add_argument("--out-dir", default="logs/slo_table4_pipeline")
    pipeline.set_defaults(func=command_pipeline_table4, theta_cpu_csv="auto")

    run = sub.add_parser("run")
    run.add_argument("--plan", default="logs/slo_plan.csv")
    run.add_argument("--data-dir", default="data_partitions")
    run.add_argument("--clients", type=int, default=0)
    run.add_argument("--rounds", type=int, default=50)
    run.add_argument("--modes", default="vanilla,oracle,fedpacer")
    run.add_argument("--dataset", default="femnist")
    run.add_argument("--model", default="cnn", choices=["cnn", "tf_example", "resnet18", "resnet20", "mobilenet_v2_075", "mobilenet_v2_100"])
    run.add_argument("--out-dir", default="logs/slo_runs")
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--server-bind", default="0.0.0.0:8081")
    run.add_argument("--server-connect", default="127.0.0.1:8081")
    run.add_argument("--reporting-fraction", type=float, default=1.0)
    run.add_argument("--local-epochs", type=int, default=1)
    run.add_argument("--local-steps", type=int, default=20)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--client-lr", type=float, default=0.003)
    run.add_argument("--server-lr", type=float, default=1.0)
    run.add_argument("--server-momentum", type=float, default=0.0)
    run.add_argument("--startup-wait", type=float, default=8.0)
    run.add_argument("--timeout", type=float, default=None)
    run.add_argument("--window", type=int, default=15)
    run.set_defaults(func=command_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
