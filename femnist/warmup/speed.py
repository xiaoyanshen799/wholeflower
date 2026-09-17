"""Measured throughput and CPU-speed models (not Logistic location parameters)."""

from dataclasses import asdict, dataclass
import math

import numpy as np

from fixed_step_training import TIMING_DEFINITION
from .model import fit_cpu_model


def validate_step_record(record, steps, batch_size=None, seed=None):
    metrics = record.get("metrics", {})
    if (record.get("timing_definition") != TIMING_DEFINITION
            or metrics.get("training_mode") != "steps"
            or metrics.get("timing_definition") != TIMING_DEFINITION):
        raise ValueError("Measurement is not a fixed-step workload")
    for field in ("local_steps_requested", "local_steps_used"):
        if type(metrics.get(field)) is not int or metrics[field] != steps:
            raise ValueError(f"Invalid {field}: expected {steps}, got {metrics.get(field)}")
    batch = metrics.get("batch_size_used")
    if type(batch) is not int or batch < 1 or (batch_size is not None and batch != batch_size):
        raise ValueError("Fixed-step batch size differs from the workload")
    if metrics.get("processed_examples") != steps * batch:
        raise ValueError("Invalid fixed-step processed example count")
    if seed is not None and metrics.get("step_seed") != seed:
        raise ValueError("Fixed-step sampling seed differs from the workload")
    duration = record.get("train_time_s", 0)
    speed = metrics.get("steps_per_second", 0)
    if (not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0
            or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0
            or not math.isclose(speed, steps / duration, rel_tol=1e-9)):
        raise ValueError("Invalid fixed-step speed or duration")


def summarize_speed(records):
    if len(records) < 3:
        raise ValueError("Speed measurement needs at least three retained rounds")
    steps = records[0].get("metrics", {}).get("local_steps_used")
    batch = records[0].get("metrics", {}).get("batch_size_used")
    for row in records:
        validate_step_record(row, steps, batch)
    times = np.asarray([row["train_time_s"] for row in records], dtype=float)
    speeds = steps / times
    return {"n_samples": len(records), "local_steps": steps, "batch_size": batch,
            "total_steps": steps * len(records), "total_train_time_s": float(times.sum()),
            "mean_train_time_s": float(times.mean()),
            "speed_steps_per_s": float(steps * len(records) / times.sum()),
            "round_speed_mean": float(speeds.mean()), "round_speed_median": float(np.median(speeds)),
            "round_speed_variance": float(speeds.var()), "statistic": "total_steps/total_train_time_s"}


@dataclass(frozen=True)
class SpeedCpuModel:
    a_s_per_step: float
    beta: float
    floor_s_per_step: float
    relative_rmse: float

    def speed(self, cpu):
        return 1 / (self.a_s_per_step * cpu ** -self.beta + self.floor_s_per_step)

    def inverse(self, speed):
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Requested speed must be finite and positive")
        residual = 1 / speed - self.floor_s_per_step
        return (self.a_s_per_step / residual) ** (1 / self.beta) if residual > 0 else math.inf

    def to_dict(self):
        return asdict(self)


def fit_speed_cpu_model(points):
    if any(not math.isfinite(speed) or speed <= 0 for _, speed in points):
        raise ValueError("Invalid CPU-speed anchors")
    model = fit_cpu_model([(cpu, 1 / speed) for cpu, speed in points])
    return SpeedCpuModel(model.mu, model.beta, model.theta_floor_s, model.relative_rmse)
