"""Deterministic synthetic timings for exercising the complete control flow."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import logistic


def simulated_stage(job):
    for index, client in enumerate(job["clients"]):
        cpu = client["cpu"]
        theta = (2.0 + index * 2.0) * cpu ** -0.8 + 0.3
        # A reproducible validation-only slowdown exercises selective adjustment.
        if job["stage_id"].startswith("validate_") and index == 0:
            theta *= 1.08
        n = job["rounds"] - 1
        samples = theta + 0.02 * theta * logistic.ppf((np.arange(n) + 0.5) / n)
        samples = [theta * 5, *samples]
        output = Path(job["output_dir"]) / f"client_{client['client_id']}.jsonl"
        with output.open("w") as handle:
            for r, value in enumerate(samples, 1):
                row = {"stage_id": job["stage_id"], "client_id": client["client_id"], "round": r,
                       "train_time_s": float(value), "cpu_requested": cpu, "cpu_actual": cpu,
                       "cpu_affinity": [int(client["cpu_affinity"])], "simulation": True}
                handle.write(json.dumps(row) + "\n")
