"""Deterministic synthetic timings for exercising the complete control flow."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import logistic


def simulated_stage(job):
    for index, client in enumerate(job["clients"]):
        cpu = client["cpu"]
        theta = (2.0 + index * 2.0) * cpu ** -0.8 + 0.3
        if job.get("training_mode") == "steps":
            theta = ((4.0 + index * 0.02) * cpu ** -0.8 + 0.3) * job["local_steps"] / 20
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
                       "cpu_affinity": [] if client["cpu_affinity"] in (None, "", "-") else list(map(int, client["cpu_affinity"].split(","))),
                       "simulation": True}
                if job.get("training_mode") == "steps":
                    from fixed_step_training import TIMING_DEFINITION
                    steps, batch = job["local_steps"], job["batch_size"]
                    row.update(timing_definition=TIMING_DEFINITION,
                               metrics={"training_mode": "steps", "local_steps_requested": steps,
                                        "local_steps_used": steps, "batch_size_used": batch,
                                        "processed_examples": steps * batch, "steps_per_second": steps / float(value),
                                        "step_seed": job["seed"], "timing_definition": TIMING_DEFINITION})
                handle.write(json.dumps(row) + "\n")
