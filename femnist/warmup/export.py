"""Export an existing complete validation batch without launching or refitting it."""

import csv
import fcntl
import json
import math
import re
from pathlib import Path

from .config import fingerprint
from .controller import Calibration, read_timings
from .measurement import atomic_json, load_cpu_map
from .model import TimingFit


def export_last(output_dir):
    output = Path(output_dir).expanduser().resolve()
    if not output.is_dir():
        raise ValueError(f"Output directory does not exist: {output}")
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Calibration is still running; cannot export its results") from error
        manifest = json.loads((output / "manifest.json").read_text())
        latest = output / "latest_config.json"
        cfg = json.loads(latest.read_text()) if latest.exists() else manifest["config"]
        operational = {"max_iterations", "startup_timeout_s", "stage_timeout_s"}
        if fingerprint({key: value for key, value in cfg.items() if key not in operational}) != manifest["key"]:
            raise ValueError("Stored configuration does not match its manifest")
        stages = []
        for stage in (output / "stages").glob("validate_*"):
            match = re.fullmatch(r"validate_(\d+)", stage.name)
            if match and (stage / "complete.json").is_file():
                stages.append((int(match[1]), stage))
        if not stages:
            raise ValueError("No complete validation batch is available to export")
        iteration, stage = max(stages)
        complete = json.loads((stage / "complete.json").read_text())
        if not re.fullmatch(r"attempt_\d+", complete["attempt"]):
            raise ValueError("Invalid completed attempt path")
        attempt = stage / complete["attempt"]
        stage_id = f"{stage.name}/{attempt.name}"
        if complete["stage_id"] != stage_id:
            raise ValueError("Completed validation identity mismatch")
        cpu_map = load_cpu_map(attempt / "cpu_config.csv")
        if set(cpu_map) != set(cfg["client_ids"]):
            raise ValueError("Validation CPU map does not contain the complete client roster")
        clients = [{"client_id": cid, "cpu": cpu_map[cid][0], "cpu_affinity": cpu_map[cid][1]}
                   for cid in cfg["client_ids"]]
        rounds, discard = cfg["validation_rounds"], cfg["validation_discard"]
        if fingerprint({"clients": clients, "rounds": rounds, "discard": discard}) != complete["key"]:
            raise ValueError("Validation CPU allocation does not match the completion record")
        for client, cpu_id in zip(clients, cfg["cpu_ids"]):
            if client["cpu_affinity"] != str(cpu_id):
                raise ValueError("Validation affinity differs from the stored configuration")
        if not cfg["simulation"]:
            job = json.loads((attempt / "job.json").read_text())
            fields = ("dataset", "model", "data_dir", "epochs", "batch_size", "lr", "num_classes", "seed")
            if (job["stage_id"] != stage_id or job["clients"] != clients or job["rounds"] != rounds
                    or any(job[key] != cfg[key] for key in fields)):
                raise ValueError("Validation job differs from the stored workload")
        saved_fits = json.loads((attempt / "fits.json").read_text())
        if set(saved_fits) != set(cfg["client_ids"]):
            raise ValueError("Validation fit results have an incomplete roster")
        fits = {}
        for client in clients:
            cid = client["client_id"]
            samples = read_timings(attempt / f"client_{cid}.jsonl", stage_id, client, rounds, discard)
            fit = TimingFit(**saved_fits[cid])
            values = (fit.theta_s, fit.k_s, fit.ks_distance, fit.empirical_p90_s)
            if (not all(math.isfinite(value) for value in values) or fit.theta_s <= 0 or fit.k_s <= 0
                    or not 0 <= fit.ks_distance <= 1 or fit.empirical_p90_s <= 0 or fit.n_samples != len(samples)):
                raise ValueError(f"Invalid saved fit for client {cid}")
            fits[cid] = fit
        target = json.loads((output / "target.json").read_text())["theta_target_s"]
        if not math.isfinite(target) or target <= 0:
            raise ValueError("Invalid saved target")
        with (output / "validation_history.csv").open(newline="") as handle:
            history = [row for row in csv.DictReader(handle) if int(row["iteration"]) == iteration]
        if len(history) != len(clients) or {row["client_id"] for row in history} != set(cfg["client_ids"]):
            raise ValueError("Validation history does not contain a complete, unique client roster")
        for row in history:
            cid = row["client_id"]
            expected = {"cpu": cpu_map[cid][0], "theta_s": fits[cid].theta_s, "k_s": fits[cid].k_s,
                        "theta_target_s": target, "relative_error": (fits[cid].theta_s - target) / target,
                        "n_samples": fits[cid].n_samples}
            if any(not math.isclose(float(row[key]), value, rel_tol=1e-10, abs_tol=1e-12)
                   for key, value in expected.items()):
                raise ValueError(f"Saved fits, target and validation history disagree for client {cid}")
        # Preserve the old failure status and keep the original timing/model artifacts unchanged.
        status = output / "status.json"
        previous = output / "status.before_export.json"
        if status.exists() and not previous.exists():
            atomic_json(previous, json.loads(status.read_text()))
        cfg = {**cfg, "output_dir": str(output)}
        reason = "max_iterations" if iteration >= cfg["max_iterations"] else "manual_last_validation"
        rows = Calibration(cfg).finish({cid: cpu_map[cid][0] for cid in cfg["client_ids"]},
                                       fits, target, iteration, export_reason=reason)
        atomic_json(output / "last_export.json", {"stage_id": stage_id, "validation_iteration": iteration,
                                                  "source": str(attempt), "retrained": False,
                                                  "refitted": False, "converged": all(row["passed"] for row in rows)})
        return rows
