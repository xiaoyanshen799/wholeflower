"""Validate the calibration workload before launching federated stages."""

import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
from pathlib import Path


DEFAULTS = {"epochs": 5, "server_local_epochs": None, "batch_size": 8, "lr": 0.001,
            "num_classes": 10, "server_bind_address": "0.0.0.0:8081",
            "client_server_address": "127.0.0.1:8081", "server_cpu_affinity": "",
            "server_csv_path": "", "reporting_fraction": 1.0, "downlink_num_bits": 0,
            "strategy": "custom-fedavgm", "server_lr": 0.01, "server_momentum": 0.9,
            "mps_enable": 0, "enable_cpu_affinity": False,
            "seed": 42, "scan_rounds": 50, "scan_discard": 1,
            "validation_rounds": 30, "validation_discard": 1,
            "tolerance": 0.03, "min_cpu": 0.05, "max_cpu": 1.0, "cpu_step": 0.001,
            "max_iterations": 5, "startup_timeout_s": 600, "stage_timeout_s": 86400,
            "training_mode": "epochs", "local_steps": None, "heterogeneity": None}
SCAN_CPUS = (0.3, 0.5, 0.7, 0.9)


def physical_cpus():
    allowed = sorted(os.sched_getaffinity(0))
    chosen, seen = [], set()
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = ((topology / "physical_package_id").read_text().strip(),
                   (topology / "core_id").read_text().strip())
        except OSError:
            key = ("unknown", cpu)
        if key not in seen:
            seen.add(key)
            chosen.append(cpu)
    return chosen


def prepare_config(raw, project_dir, simulate=False):
    allowed = set(DEFAULTS) | {"dataset", "model", "data_dir", "output_dir", "client_ids", "cpu_ids"}
    if set(raw) - allowed:
        raise ValueError(f"Unknown config keys: {sorted(set(raw) - allowed)}")
    cfg = {**DEFAULTS, **raw}
    from fixed_step_training import SUPPORTED_DATASETS, positive_steps
    if cfg["training_mode"] not in ("epochs", "steps"):
        raise ValueError("training_mode must be epochs or steps")
    if cfg["training_mode"] == "steps":
        positive_steps(cfg["local_steps"])
        if cfg.get("dataset") not in SUPPORTED_DATASETS:
            raise ValueError("Fixed-step mode currently requires array-backed training data")
    elif cfg["local_steps"] is not None:
        raise ValueError("Set training_mode=steps when providing local_steps")
    for key in ("dataset", "model", "data_dir", "output_dir"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"Config requires a nonempty {key}")
    if cfg["strategy"] not in ("fedavg", "fedavgm", "custom-fedavgm", "fedcs", "tifl"):
        raise ValueError("Unsupported strategy")
    if cfg["server_local_epochs"] is None:
        cfg["server_local_epochs"] = cfg["epochs"]
    for key in ("epochs", "server_local_epochs", "batch_size", "num_classes", "scan_rounds",
                "validation_rounds", "max_iterations"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("downlink_num_bits", "mps_enable"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if cfg["downlink_num_bits"] not in (0, 8, 16):
        raise ValueError("downlink_num_bits must be 0, 8 or 16")
    if cfg["mps_enable"] not in (0, 1):
        raise ValueError("mps_enable must be 0 or 1")
    if type(cfg["enable_cpu_affinity"]) is not bool:
        raise ValueError("enable_cpu_affinity must be true or false")
    for key in ("server_bind_address", "client_server_address"):
        if not isinstance(cfg[key], str) or ":" not in cfg[key]:
            raise ValueError(f"{key} must be host:port")
    for key in ("server_cpu_affinity", "server_csv_path"):
        if not isinstance(cfg[key], str):
            raise ValueError(f"{key} must be a string")
    for key in ("seed", "scan_discard", "validation_discard"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if cfg["seed"] > 2**32 - 1:
        raise ValueError("seed must fit in uint32")
    for phase in ("scan", "validation"):
        if cfg[f"{phase}_rounds"] - cfg[f"{phase}_discard"] < 3:
            raise ValueError(f"{phase} needs at least three retained rounds")
    for key in ("lr", "server_lr", "reporting_fraction", "tolerance", "min_cpu", "max_cpu", "cpu_step",
                "startup_timeout_s", "stage_timeout_s"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if (isinstance(cfg["server_momentum"], bool)
            or not isinstance(cfg["server_momentum"], (int, float))
            or not math.isfinite(cfg["server_momentum"])
            or cfg["server_momentum"] < 0):
        raise ValueError("server_momentum must be finite and nonnegative")
    if not 0 < cfg["reporting_fraction"] <= 1:
        raise ValueError("reporting_fraction must be in (0, 1]")
    if not 0 < cfg["tolerance"] < 1:
        raise ValueError("tolerance must be a fraction between 0 and 1")
    if not 0.05 <= cfg["min_cpu"] <= 0.3 < 0.9 <= cfg["max_cpu"] <= 1:
        raise ValueError("Require 0.05 <= min_cpu <= 0.3 and 0.9 <= max_cpu <= 1 (20ms quota period)")
    if cfg["cpu_step"] < 0.001 or cfg["cpu_step"] > cfg["max_cpu"] - cfg["min_cpu"]:
        raise ValueError("cpu_step must be >= 0.001 and smaller than the CPU range")
    if cfg["stage_timeout_s"] <= cfg["startup_timeout_s"]:
        raise ValueError("stage_timeout_s must exceed startup_timeout_s")
    project = Path(project_dir).resolve()
    cfg["project_dir"] = str(project)
    cfg["python"] = os.path.abspath(sys.executable)  # Keep the venv symlink, not its system target.
    for key in ("data_dir", "output_dir"):
        path = Path(cfg[key]).expanduser()
        cfg[key] = str((project / path).resolve() if not path.is_absolute() else path.resolve())
    cfg["simulation"] = simulate
    partitions = {}
    if not simulate:
        for path in Path(cfg["data_dir"]).glob("client_*.npz"):
            match = re.fullmatch(r"client_(\d+)\.npz", path.name)
            if match:
                cid = str(int(match[1]))
                if path.name != f"client_{int(cid):05d}.npz":
                    raise ValueError(f"run_client.py requires zero-padded partition filenames: {path.name}")
                if cid in partitions:
                    raise ValueError(f"Duplicate partition ID {cid}")
                stat = path.stat()
                partitions[cid] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    ids = cfg.get("client_ids")
    if ids is None:
        ids = ["0", "1", "2"] if simulate else sorted(partitions, key=int)
    if not isinstance(ids, list) or not ids or any(not str(cid).isdigit() for cid in ids):
        raise ValueError("client_ids must be nonempty numeric IDs, or provide a valid partition directory")
    ids = [str(int(cid)) for cid in ids]
    if len(set(ids)) != len(ids) or max(map(int, ids)) + cfg["seed"] > 2**32 - 1:
        raise ValueError("Duplicate IDs or client seed exceeds uint32")
    if not simulate and set(ids) - set(partitions):
        raise ValueError(f"Missing partitions: {sorted(set(ids) - set(partitions))}")
    cpu_ids = cfg.get("cpu_ids")
    if cpu_ids is None:
        cpu_ids = list(range(len(ids))) if simulate else physical_cpus()[:len(ids)]
    if (not isinstance(cpu_ids, list) or len(cpu_ids) != len(ids)
            or any(type(cpu) is not int or cpu < 0 for cpu in cpu_ids)
            or len(set(cpu_ids)) != len(cpu_ids)):
        raise ValueError("cpu_ids must assign a distinct logical CPU to every client; not enough physical cores?")
    if not simulate and set(cpu_ids) - os.sched_getaffinity(0):
        raise ValueError("cpu_ids includes CPUs unavailable to this process")
    cfg["client_ids"], cfg["cpu_ids"] = ids, cpu_ids
    from .heterogeneity import prepare_heterogeneity
    cfg["heterogeneity"] = prepare_heterogeneity(cfg["heterogeneity"], cfg)
    cfg["partitions"] = {cid: partitions[cid] for cid in ids} if not simulate else {}
    runtime = {}
    for name in ("numpy", "scipy", "tensorflow", "flwr"):
        try:
            runtime[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            runtime[name] = "unavailable"
    cfg["runtime"] = runtime
    sources = list((project / "warmup").glob("*.py"))
    sources += [project / name for name in ("run_client.py", "client.py", "launch_clients.sh",
                                           "warmup_control.py", "warmup_launch.py", "fedavgm/models.py",
                                           "fedavgm/dataset.py", "run_server.py", "strategy.py",
                                           "fedavgm/server.py", "fixed_step_training.py")]
    cfg["source_hashes"] = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(sources) if path.is_file()}
    return cfg


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
