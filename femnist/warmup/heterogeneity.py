"""Persisted speed-distribution initialization before the existing theta loop."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .measurement import atomic_json, load_cpu_map
from .model import adjust_cpu, quantize_cpu
from .pacer_target import derive_pacer_target
from .speed import fit_speed_cpu_model


INITIAL_STAGE = "heterogeneous_initial"


def prepare_heterogeneity(raw, cfg):
    if raw is None:
        return {"enabled": False}
    if not isinstance(raw, dict):
        raise ValueError("heterogeneity must be an object")
    defaults = {"enabled": True, "reference_cpu": 0.5, "baseline_mode": "reference_client",
                "reference_client_id": cfg["client_ids"][0], "distribution": "normal",
                "mean": 1.0, "variance": None, "seed": cfg["seed"],
                "initial_rounds": 30, "initial_discard": 1, "out_of_range": "error"}
    if set(raw) - set(defaults):
        raise ValueError(f"Unknown heterogeneity keys: {sorted(set(raw) - set(defaults))}")
    h = {**defaults, **raw}
    if type(h["enabled"]) is not bool:
        raise ValueError("heterogeneity.enabled must be boolean")
    if not h["enabled"]:
        return {"enabled": False}
    if cfg.get("training_mode") != "steps":
        raise ValueError("Speed heterogeneity requires training_mode=steps")
    if cfg["reporting_fraction"] != 1.0:
        raise ValueError("Speed calibration requires reporting_fraction=1.0 for complete per-client rounds")
    if h["reference_cpu"] != 0.5 or h["baseline_mode"] != "reference_client":
        raise ValueError("Use reference_client at reference_cpu=0.5")
    if not str(h["reference_client_id"]).isdigit():
        raise ValueError("Invalid reference_client_id")
    h["reference_client_id"] = str(int(h["reference_client_id"]))
    if h["reference_client_id"] not in cfg["client_ids"]:
        raise ValueError("Reference client is not in client_ids")
    if h["distribution"] not in ("normal", "homogeneous", "exponential"):
        raise ValueError("distribution must be normal, homogeneous or exponential")
    if isinstance(h["mean"], bool) or not isinstance(h["mean"], (int, float)) or not math.isfinite(h["mean"]) or h["mean"] <= 0:
        raise ValueError("heterogeneity.mean must be finite and positive")
    if h["variance"] is None:
        h["variance"] = (h["mean"] * h["mean"] if h["distribution"] == "exponential"
                         else 0.0 if h["distribution"] == "homogeneous" else 0.04)
    variance = h["variance"]
    if isinstance(variance, bool) or not isinstance(variance, (int, float)) or not math.isfinite(variance) or variance < 0:
        raise ValueError("heterogeneity.variance must be finite and nonnegative")
    if h["distribution"] == "homogeneous" and variance != 0:
        raise ValueError("homogeneous requires variance=0")
    if h["distribution"] == "exponential" and not math.isclose(variance, h["mean"] * h["mean"], rel_tol=1e-10):
        raise ValueError("Pure exponential requires variance=mean**2; variance cannot be independent")
    for key in ("seed", "initial_rounds", "initial_discard"):
        if type(h[key]) is not int or h[key] < 0:
            raise ValueError(f"heterogeneity.{key} must be a nonnegative integer")
    if h["seed"] >= 2**32 or h["initial_rounds"] - h["initial_discard"] < 3:
        raise ValueError("Invalid heterogeneity seed or fewer than three retained initial rounds")
    if h["out_of_range"] != "error":
        raise ValueError("Only out_of_range=error is supported; speeds are never silently clipped or redrawn")
    return h


def sample_speed_factors(settings, client_ids):
    rng = np.random.default_rng(settings["seed"])
    ordered = sorted(client_ids, key=int)
    mean, variance = settings["mean"], settings["variance"]
    distribution = settings["distribution"]
    if distribution == "homogeneous":
        values = np.full(len(ordered), mean)
    elif distribution == "normal":
        values = rng.normal(mean, math.sqrt(variance), len(ordered))
    elif distribution == "exponential":
        values = rng.exponential(mean, len(ordered))
    else:
        raise ValueError("Unknown speed distribution")
    return dict(zip(ordered, map(float, values)))


def read_plan(path):
    from .config import fingerprint

    plan = json.loads(Path(path).read_text())
    if plan.get("checksum") != fingerprint({key: value for key, value in plan.items() if key != "checksum"}):
        raise ValueError("Saved heterogeneity plan checksum mismatch")
    return plan


def build_speed_plan(cfg, models, measured_reference, path):
    from .config import fingerprint

    h = cfg["heterogeneity"]
    key = fingerprint({"settings": h, "clients": cfg["client_ids"], "reference_speed": measured_reference,
                       "models": {cid: model.to_dict() for cid, model in models.items()},
                       "bounds": [cfg["min_cpu"], cfg["max_cpu"], cfg["cpu_step"]],
                       "local_steps": cfg["local_steps"], "batch_size": cfg["batch_size"]})
    path = Path(path)
    if path.exists():
        plan = read_plan(path)
        if plan["input_key"] != key:
            raise ValueError("Speed profiles or initialization settings changed on resume")
        return plan
    factors = sample_speed_factors(h, cfg["client_ids"])
    rows = []
    for cid in cfg["client_ids"]:
        factor = factors[cid]
        speed = measured_reference * factor
        model = models[cid]
        low, high = model.speed(cfg["min_cpu"]), model.speed(cfg["max_cpu"])
        error = ""
        cpu = raw_cpu = predicted = None
        if not math.isfinite(speed) or speed <= 0:
            error = "nonpositive_or_nonfinite_speed"
        elif not low <= speed <= high:
            error = "speed_outside_cpu_bounds"
        else:
            raw_cpu = model.inverse(speed)
            cpu = quantize_cpu(raw_cpu, cfg["min_cpu"], cfg["max_cpu"], cfg["cpu_step"])
            predicted = model.speed(cpu)
        rows.append({"client_id": cid, "factor": factor, "requested_speed_steps_per_s": speed,
                     "minimum_speed_steps_per_s": low, "maximum_speed_steps_per_s": high,
                     "unquantized_cpu": raw_cpu, "cpu": cpu, "predicted_speed_steps_per_s": predicted,
                     "extrapolated": cpu is not None and (cpu < 0.3 or cpu > 0.9), "error": error})
    plan = {"input_key": key, "reference_client_id": h["reference_client_id"], "reference_cpu": 0.5,
            "reference_speed_steps_per_s": measured_reference, "settings": h,
            "factor_sample_mean": float(np.mean(list(factors.values()))),
            "factor_sample_variance": float(np.var(list(factors.values()))), "variance_ddof": 0,
            "clients": rows, "feasible": all(not row["error"] for row in rows)}
    if any(not math.isfinite(row["factor"]) or not math.isfinite(row["requested_speed_steps_per_s"]) for row in rows):
        raise ValueError("Distribution overflow; reduce mean/variance")
    plan["checksum"] = fingerprint(plan)
    atomic_json(path, plan)
    return plan


def target_source_key(output, cfg, plan):
    from .config import fingerprint
    from .controller import batch_key, read_records

    stage = Path(output) / "stages" / INITIAL_STAGE
    complete = json.loads((stage / "complete.json").read_text())
    attempt = stage / complete["attempt"]
    if not attempt.resolve().is_relative_to(stage.resolve()):
        raise ValueError("Invalid initial attempt path")
    if complete["stage_id"] != f"{INITIAL_STAGE}/{attempt.name}":
        raise ValueError("Initial stage identity mismatch")
    cpu_map = load_cpu_map(attempt / "cpu_config.csv")
    requested = {row["client_id"]: row["cpu"] for row in plan["clients"]}
    if set(cpu_map) != set(cfg["client_ids"]) or any(cpu_map[cid][0] != requested[cid] for cid in cfg["client_ids"]):
        raise ValueError("Initial CPU allocation differs from the saved speed plan")
    clients = [{"client_id": cid, "cpu": cpu_map[cid][0], "cpu_affinity": cpu_map[cid][1]} for cid in cfg["client_ids"]]
    h = cfg["heterogeneity"]
    if complete["key"] != batch_key(clients, h["initial_rounds"], h["initial_discard"], cfg):
        raise ValueError("Initial workload differs from the saved completion record")
    job = json.loads((attempt / "job.json").read_text())
    for field in ("dataset", "model", "local_steps", "batch_size", "seed", "lr", "training_mode",
                  "data_dir", "num_classes", "enable_cpu_affinity", "server_cpu_affinity"):
        if job.get(field) != cfg.get(field):
            raise ValueError(f"Initial job workload changed: {field}")
    if (job.get("stage_id") != complete["stage_id"] or job.get("clients") != clients
            or job.get("rounds") != h["initial_rounds"] or job.get("discard") != h["initial_discard"]):
        raise ValueError("Initial job identity changed")
    hashes = {}
    for client, cpu_id in zip(clients, cfg["cpu_ids"]):
        expected_affinity = str(cpu_id) if cfg["enable_cpu_affinity"] else "-"
        if client["cpu_affinity"] != expected_affinity:
            raise ValueError("Initial affinity differs from the workload")
        source = attempt / f"client_{client['client_id']}.jsonl"
        read_records(source, complete["stage_id"], client, h["initial_rounds"], h["initial_discard"],
                     cfg["local_steps"], cfg["batch_size"], cfg["seed"])
        hashes[client["client_id"]] = hashlib.sha256(source.read_bytes()).hexdigest()
    return fingerprint({"plan_checksum": plan["checksum"], "stage_key": complete["key"], "raw": hashes}), attempt


def validate_speed_target(output, cfg):
    plan = read_plan(Path(output) / "heterogeneity_plan.json")
    if not plan["feasible"] or plan["settings"] != cfg["heterogeneity"]:
        raise ValueError("Invalid saved speed plan")
    key, attempt = target_source_key(output, cfg, plan)
    target = json.loads((Path(output) / "target.json").read_text())
    saved_fits = json.loads((attempt / "fits.json").read_text())
    if set(saved_fits) != set(cfg["client_ids"]):
        raise ValueError("Incomplete initial timing fits")
    thetas = {cid: fit["theta_s"] for cid, fit in saved_fits.items()}
    if any(not math.isfinite(value) or value <= 0 for value in thetas.values()):
        raise ValueError("Invalid initial theta")
    slowest = max(cfg["client_ids"], key=thetas.get)
    expected = derive_pacer_target(thetas[slowest], len(cfg["client_ids"]))
    common_valid = (target.get("source_key") == key and target.get("source_stage") == INITIAL_STAGE
                    and target.get("slowest_client_id") == slowest)
    current_valid = (target.get("definition") == "rounded_deadline_from_max_fitted_theta_at_heterogeneous_initial"
                     and all(target.get(name) == value for name, value in expected.items()))
    # Keep --export-last usable for completed pre-deadline runs. New runs never
    # create this legacy form because initialize_from_speed_plan writes the
    # rounded-deadline definition above.
    legacy_valid = (target.get("definition") == "max_fitted_theta_at_heterogeneous_initial"
                    and target.get("theta_target_s") == thetas[slowest])
    if not common_valid or not (current_valid or legacy_valid):
        raise ValueError("Fixed target differs from its initial measurement source")
    return target


def initialize_from_speed_plan(calibration, theta_models, points):
    from .controller import write_csv

    cfg, output = calibration.cfg, calibration.output
    speed_models = {cid: fit_speed_cpu_model([
        (cpu, calibration.batch_speeds[f"scan_{round(cpu * 100):02d}"][cid]["speed_steps_per_s"])
        for cpu, _ in points[cid]]) for cid in cfg["client_ids"]}
    atomic_json(output / "speed_cpu_models.json", {cid: model.to_dict() for cid, model in speed_models.items()})
    h = cfg["heterogeneity"]
    reference = calibration.batch_speeds["scan_50"][h["reference_client_id"]]["speed_steps_per_s"]
    plan = build_speed_plan(cfg, speed_models, reference, output / "heterogeneity_plan.json")
    if not plan["feasible"]:
        failures = [f"{row['client_id']}: {row['error']}" for row in plan["clients"] if row["error"]]
        raise ValueError(f"Infeasible speed distribution ({'; '.join(failures)}); see heterogeneity_plan.json")
    cpus = {row["client_id"]: row["cpu"] for row in plan["clients"]}
    write_csv(output / "heterogeneous_cpu_config.csv", [
        {**row, "cpu_affinity": calibration.affinity[row["client_id"]]} for row in plan["clients"]])
    fits = calibration.batch(INITIAL_STAGE, cpus, h["initial_rounds"], h["initial_discard"])
    write_csv(output / "heterogeneous_initial_measurements.csv", [
        {"client_id": row["client_id"], "cpu": row["cpu"],
         "requested_speed_steps_per_s": row["requested_speed_steps_per_s"],
         "measured_speed_steps_per_s": calibration.batch_speeds[INITIAL_STAGE][row["client_id"]]["speed_steps_per_s"],
         "speed_relative_error": calibration.batch_speeds[INITIAL_STAGE][row["client_id"]]["speed_steps_per_s"] / row["requested_speed_steps_per_s"] - 1,
         **fits[row["client_id"]].to_dict()} for row in plan["clients"]])
    slowest = max(cfg["client_ids"], key=lambda cid: fits[cid].theta_s)
    source_key, _ = target_source_key(output, cfg, plan)
    target_path = output / "target.json"
    if not target_path.exists():
        pacer_target = derive_pacer_target(fits[slowest].theta_s, len(cfg["client_ids"]))
        atomic_json(target_path, {**pacer_target, "slowest_client_id": slowest,
                                  "source_stage": INITIAL_STAGE, "source_key": source_key,
                                  "reference_client_id": h["reference_client_id"],
                                  "definition": "rounded_deadline_from_max_fitted_theta_at_heterogeneous_initial"})
    target = validate_speed_target(output, cfg)["theta_target_s"]
    history = {cid: [*points[cid], (cpus[cid], fits[cid].theta_s)] for cid in cfg["client_ids"]}
    for cid in cfg["client_ids"]:
        if abs(fits[cid].theta_s / target - 1) > cfg["tolerance"]:
            cpus[cid] = adjust_cpu(theta_models[cid], cpus[cid], fits[cid].theta_s, target,
                                   history[cid], cfg["min_cpu"], cfg["max_cpu"], cfg["cpu_step"])
    return cpus, target, history
