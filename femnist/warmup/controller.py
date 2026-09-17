"""Profile CPU quotas, choose a fixed target, and validate until every client passes."""

import csv
import fcntl
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import time
from pathlib import Path

from .config import SCAN_CPUS, fingerprint
from .measurement import atomic_json
from .model import adjust_cpu, fit_cpu_model, fit_logistic, quantize_cpu
from .pacer_target import build_pacer_control


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_records(path, stage_id, client, rounds, discard, expected_steps=None, batch_size=None, seed=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    recorded_rounds = [row.get("round") for row in rows]
    if (any(type(r) is not int or r < 1 or r > rounds for r in recorded_rounds)
            or recorded_rounds != sorted(recorded_rounds)
            or len(recorded_rounds) != len(set(recorded_rounds))
            ):
        raise ValueError(f"{path}: duplicated, unordered or out-of-range rounds (expected 1..{rounds})")
    retained_rounds = [r for r in recorded_rounds if r > discard]
    missing_retained = [r for r in range(discard + 1, rounds + 1) if r not in retained_rounds]
    if missing_retained:
        raise ValueError(f"{path}: missing retained rounds {missing_retained} (discard={discard}, expected through {rounds})")
    affinity = None if client.get("cpu_affinity") in (None, "", "-") else sorted(map(int, client["cpu_affinity"].split(",")))
    for row in rows:
        if row.get("stage_id") != stage_id or row.get("client_id") != client["client_id"]:
            raise ValueError(f"{path}: wrong stage or client identity")
        for field in ("cpu_requested", "cpu_actual"):
            if not math.isclose(row.get(field, -1), client["cpu"], rel_tol=0.001, abs_tol=1e-6):
                raise ValueError(f"{path}: CPU quota changed during measurement")
        if affinity is not None and row.get("cpu_affinity") != affinity:
            raise ValueError(f"{path}: CPU affinity changed during measurement")
        duration = row.get("train_time_s", float("nan"))
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{path}: invalid training duration")
        if expected_steps is not None:
            from .speed import validate_step_record
            validate_step_record(row, expected_steps, batch_size, seed)
    return rows


def read_timings(path, stage_id, client, rounds, discard, expected_steps=None, batch_size=None, seed=None):
    rows = read_records(path, stage_id, client, rounds, discard, expected_steps, batch_size, seed)
    return [row["train_time_s"] for row in rows if row["round"] > discard]


def batch_key(clients, rounds, discard, cfg):
    value = {"clients": clients, "rounds": rounds, "discard": discard}
    if cfg.get("training_mode") == "steps":
        from fixed_step_training import TIMING_DEFINITION
        value["workload"] = {"training_mode": "steps", "local_steps": cfg["local_steps"],
                             "batch_size": cfg["batch_size"], "step_seed": cfg["seed"],
                             "timing_definition": TIMING_DEFINITION}
    return fingerprint(value)


def _split_host_port(address):
    host, port = address.rsplit(":", 1)
    return host or "127.0.0.1", int(port)


def _wait_for_server(address, process, timeout):
    host, port = _split_host_port(address)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"Server exited before accepting clients (exit {code})")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Server did not listen on {address} within {timeout}s")


def _port_is_open(address):
    host, port = _split_host_port(address)
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _stop_client_scopes(client_ids):
    units = [f"fl_client_{cid}.scope" for cid in client_ids]
    if not units:
        return
    command = ["systemctl", "stop", *units]
    if os.geteuid() != 0:
        command = ["sudo", "-n", *command]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, check=False)


def external_stage(job):
    output = Path(job["output_dir"])
    job_path = output / "job.json"
    atomic_json(job_path, job)
    server_csv = job.get("server_csv_path") or str(output / "server_metrics.csv")
    server_command = [
        job["python"], "-m", "run_server",
        "--dataset", job["dataset"], "--model", job["model"],
        "--strategy", job["strategy"],
        "--clients", str(len(job["clients"])), "--rounds", str(job["rounds"]),
        "--reporting-fraction", str(job["reporting_fraction"]),
        "--address", job["server_bind_address"],
        "--client-lr", str(job["lr"]), "--batch-size", str(job["batch_size"]),
        "--local-epochs", str(job["server_local_epochs"]),
        "--server-lr", str(job["server_lr"]),
        "--server-momentum", str(job["server_momentum"]),
        "--downlink-num-bits", str(job["downlink_num_bits"]),
        "--csv-path", server_csv,
    ]
    if job.get("training_mode") == "steps":
        server_command += ["--local-steps", str(job["local_steps"]), "--step-seed", str(job["seed"])]
    if job.get("server_cpu_affinity"):
        server_command = ["taskset", "-c", job["server_cpu_affinity"], *server_command]
    server_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "FLWR_TELEMETRY_ENABLED": "0"}
    client_env = {
        **os.environ,
        "PY": job["python"],
        "DATASET": job["dataset"],
        "MODEL": job["model"],
        "BATCH_SIZE": str(job["batch_size"]),
        "LR": str(job["lr"]),
        "LOCAL_EPOCHS": str(job["epochs"]),
        "LOCAL_STEPS": str(job["local_steps"]) if job.get("training_mode") == "steps" else "",
        "STEP_SEED": str(job["seed"]) if job.get("training_mode") == "steps" else "",
        "NUM_CLASSES": str(job["num_classes"]),
        "MPS_ENABLE": str(job["mps_enable"]),
        "CLIENT_LOG_DIR": str(output),
        "CPU_MAP_CSV": str(job["cpu_map_csv"]),
        "CPU_MAP_ONLY": "1",
        "ENABLE_CPU_AFFINITY": "1" if job["enable_cpu_affinity"] else "0",
        "BIND_CLIENT_TO_CPU": "0",
        "MEASUREMENT_RUN_ID": job["stage_id"],
    }
    launcher_command = [
        "bash", str(Path(job["project_dir"]) / "launch_clients.sh"),
        job["data_dir"], job["client_server_address"], str(len(job["clients"])),
    ]
    server = None
    launcher = None
    clients_started = False
    try:
        if _port_is_open(job["client_server_address"]):
            raise RuntimeError(f"{job['client_server_address']} is already accepting connections; stop the old server first")
        with (output / "server.log").open("w") as server_log:
            server = subprocess.Popen(server_command, cwd=job["project_dir"], env=server_env,
                                      stdout=server_log, stderr=subprocess.STDOUT)
        _wait_for_server(job["client_server_address"], server, job["startup_timeout_s"])
        with (output / "launcher.log").open("w") as log:
            launcher = subprocess.Popen(launcher_command, cwd=job["project_dir"], env=client_env,
                                        stdout=log, stderr=subprocess.STDOUT)
            clients_started = True
            stage_started = time.monotonic()
            while True:
                server_code = server.poll()
                launcher_code = launcher.poll()
                if launcher_code not in (None, 0) and server_code is None:
                    raise RuntimeError(f"Client launcher failed (exit {launcher_code}); see {output / 'launcher.log'}")
                if server_code is not None:
                    if server_code:
                        raise RuntimeError(f"Server failed (exit {server_code}); see {output / 'server.log'}")
                    break
                if time.monotonic() - stage_started > job["stage_timeout_s"] + 60:
                    raise TimeoutError("Warm-up federated stage exceeded stage_timeout_s")
                time.sleep(1)
            if launcher.poll() is None:
                launcher.wait(timeout=60)
            elif launcher.returncode:
                raise RuntimeError(f"Client launcher failed (exit {launcher.returncode}); see {output / 'launcher.log'}")
        for client in job["clients"]:
            timing_path = output / f"client_{client['client_id']}.jsonl"
            if not timing_path.exists():
                raise RuntimeError(f"Missing client timing file: {timing_path}")
    finally:
        if clients_started:
            _stop_client_scopes([client["client_id"] for client in job["clients"]])
        if server is not None and server.poll() is None:
            server.send_signal(signal.SIGTERM)
            server.wait(timeout=30)
        if launcher is not None and launcher.poll() is None:
            launcher.send_signal(signal.SIGTERM)
            launcher.wait(timeout=30)


class Calibration:
    def __init__(self, cfg, stage_runner=external_stage):
        self.cfg, self.stage_runner = cfg, stage_runner
        self.output = Path(cfg["output_dir"])
        self.cid_list = cfg["client_ids"]
        self.batch_speeds = {}
        if cfg["enable_cpu_affinity"]:
            self.affinity = dict(zip(self.cid_list, map(str, cfg["cpu_ids"])))
        else:
            self.affinity = dict.fromkeys(self.cid_list, "-")

    def batch(self, name, cpus, rounds, discard):
        directory = self.output / "stages" / name
        directory.mkdir(parents=True, exist_ok=True)
        clients = [{"client_id": cid, "cpu": cpus[cid], "cpu_affinity": self.affinity[cid]}
                   for cid in self.cid_list]
        key = batch_key(clients, rounds, discard, self.cfg)
        complete = directory / "complete.json"
        if complete.exists():
            cached = json.loads(complete.read_text())
            if cached["key"] != key:
                raise ValueError(f"Resume input mismatch in {name}")
            attempt = directory / cached["attempt"]
            stage_id = cached["stage_id"]
            print(f"Reuse {name}", flush=True)
        else:
            attempts = [int(p.name.split("_")[-1]) for p in directory.glob("attempt_[0-9]*") if p.is_dir()]
            attempt = directory / f"attempt_{max(attempts, default=0) + 1:03d}"
            attempt.mkdir()
            stage_id = f"{name}/{attempt.name}"
            cpu_map_csv = attempt / "requested_cpu_config.csv"
            job = {**self.cfg, "stage_id": stage_id, "output_dir": str(attempt),
                   "clients": clients, "rounds": rounds, "discard": discard, "cpu_map_csv": str(cpu_map_csv)}
            write_csv(cpu_map_csv, clients)
            write_csv(attempt / "cpu_config.csv", clients)
            atomic_json(attempt / "job.json", job)
            print(f"Run {name}: {rounds} rounds, CPU {cpus}", flush=True)
            self.stage_runner(job)
        fits = {}
        speeds = {}
        steps = self.cfg.get("local_steps") if self.cfg.get("training_mode") == "steps" else None
        for client in clients:
            records = read_records(attempt / f"client_{client['client_id']}.jsonl", stage_id,
                                   client, rounds, discard, steps, self.cfg["batch_size"], self.cfg["seed"])
            retained = [row for row in records if row["round"] > discard]
            fits[client["client_id"]] = fit_logistic([row["train_time_s"] for row in retained])
            if steps is not None:
                from .speed import summarize_speed
                speeds[client["client_id"]] = summarize_speed(retained)
        if speeds:
            atomic_json(attempt / "speeds.json", speeds)
            self.batch_speeds[name] = speeds
        atomic_json(attempt / "fits.json", {cid: fit.to_dict() for cid, fit in fits.items()})
        atomic_json(complete, {"key": key, "stage_id": stage_id, "attempt": attempt.name,
                               "rounds": rounds, "discard": discard})
        from .logs import export_stage
        export_stage(self.output, directory, self.cfg)
        return fits

    def run(self, resume=False):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another calibration is using this output directory") from error
            manifest = self.output / "manifest.json"
            operational = {"max_iterations", "startup_timeout_s", "stage_timeout_s"}
            key = fingerprint({key: value for key, value in self.cfg.items() if key not in operational})
            if manifest.exists():
                if not resume:
                    raise ValueError("Output already used; choose a new output_dir or pass --resume")
                if json.loads(manifest.read_text())["key"] != key:
                    raise ValueError("Resume config, partition metadata or runtime version changed; use a new output_dir")
            else:
                if any(p.name != ".lock" for p in self.output.iterdir()):
                    raise ValueError("Output is nonempty without a manifest; choose a new output_dir")
                atomic_json(manifest, {"key": key, "config": self.cfg})
            atomic_json(self.output / "latest_config.json", self.cfg)
            try:
                # A failed resumed audit must not leave a stale launchable result.
                for name in ("final_cpu_config.csv", "simulated_cpu_config.csv", "launch_final_clients.sh",
                             "pacer-control.json"):
                    (self.output / name).unlink(missing_ok=True)
                atomic_json(self.output / "status.json", {"status": "running", "simulation": self.cfg["simulation"]})
                return self._calibrate()
            except BaseException as error:
                atomic_json(self.output / "status.json", {"status": "failed", "error": str(error),
                                                          "simulation": self.cfg["simulation"]})
                raise

    def _calibrate(self):
        cfg = self.cfg
        points = {cid: [] for cid in self.cid_list}
        profile_rows = []
        speed_rows = []
        for cpu in SCAN_CPUS:
            fits = self.batch(f"scan_{round(cpu * 100):02d}", dict.fromkeys(self.cid_list, cpu),
                              cfg["scan_rounds"], cfg["scan_discard"])
            for cid, fit in fits.items():
                points[cid].append((cpu, fit.theta_s))
                profile_rows.append({"client_id": cid, "cpu": cpu, **fit.to_dict()})
                if cfg.get("training_mode") == "steps":
                    speed_rows.append({"client_id": cid, "cpu": cpu,
                                       **self.batch_speeds[f"scan_{round(cpu * 100):02d}"][cid]})
            write_csv(self.output / "profiles.csv", profile_rows)
            if speed_rows:
                write_csv(self.output / "speed_profiles.csv", speed_rows)
        models = {cid: fit_cpu_model(points[cid]) for cid in self.cid_list}
        atomic_json(self.output / "cpu_models.json", {cid: model.to_dict() for cid, model in models.items()})
        if (cfg.get("heterogeneity") or {}).get("enabled"):
            from .heterogeneity import initialize_from_speed_plan
            cpus, target, history = initialize_from_speed_plan(self, models, points)
        else:
            # Legacy initialization still uses the measured 90% timing anchor.
            slowest = max(self.cid_list, key=lambda cid: points[cid][-1][1])
            target = points[slowest][-1][1]
            atomic_json(self.output / "target.json", {"theta_target_s": target, "slowest_client_id": slowest,
                                                     "reference_cpu": 0.9, "definition": "max_fitted_theta_at_90_percent"})
            cpus = {cid: quantize_cpu(models[cid].inverse(target), cfg["min_cpu"], cfg["max_cpu"], cfg["cpu_step"])
                    for cid in self.cid_list}
            history = {cid: list(points[cid]) for cid in self.cid_list}
        write_csv(self.output / "initial_cpu_config.csv", [
            {"client_id": cid, "cpu": cpus[cid], "cpu_affinity": self.affinity[cid],
             "extrapolated": cpus[cid] < 0.3 or cpus[cid] > 0.9} for cid in self.cid_list])
        validation_rows = []
        for iteration in range(1, cfg["max_iterations"] + 1):
            fits = self.batch(f"validate_{iteration:03d}", cpus, cfg["validation_rounds"], cfg["validation_discard"])
            failing = []
            for cid, fit in fits.items():
                error = (fit.theta_s - target) / target
                history[cid].append((cpus[cid], fit.theta_s))
                validation_rows.append({"iteration": iteration, "client_id": cid, "cpu": cpus[cid],
                                        "theta_s": fit.theta_s, "k_s": fit.k_s, "theta_target_s": target,
                                        "relative_error": error, "passed": abs(error) <= cfg["tolerance"],
                                        "n_samples": fit.n_samples})
                if abs(error) > cfg["tolerance"]:
                    failing.append(cid)
            write_csv(self.output / "validation_history.csv", validation_rows)
            atomic_json(self.output / "status.json", {"status": "validating", "iteration": iteration,
                                                       "failing_clients": failing, "theta_target_s": target,
                                                       "simulation": cfg["simulation"]})
            print(f"Validation {iteration}: failing={failing}, target={target:.6f}s", flush=True)
            if not failing:
                return self.finish(cpus, fits, target, iteration)
            if iteration == cfg["max_iterations"]:
                return self.finish(cpus, fits, target, iteration, export_reason="max_iterations")
            changes = []
            for cid in failing:
                previous = cpus[cid]
                try:
                    cpus[cid] = adjust_cpu(models[cid], previous, fits[cid].theta_s, target,
                                           history[cid], cfg["min_cpu"], cfg["max_cpu"], cfg["cpu_step"])
                except ValueError as error:
                    raise RuntimeError(f"Client {cid} cannot meet fixed target within CPU bounds: {error}") from error
                changes.append({"client_id": cid, "old_cpu": previous, "new_cpu": cpus[cid]})
            atomic_json(self.output / "stages" / f"validate_{iteration:03d}" / "adjustments.json", changes)

    def finish(self, cpus, fits, target, iteration, export_reason="validated"):
        failing = [cid for cid in self.cid_list
                   if abs((fits[cid].theta_s - target) / target) > self.cfg["tolerance"]]
        converged = not failing
        rows = [{"client_id": cid, "cpu": cpus[cid], "cpu_quota_percent": round(cpus[cid] * 100, 8),
                 "cpu_affinity": self.affinity[cid], "theta_target_s": target,
                 "measured_theta_s": fits[cid].theta_s, "relative_error": (fits[cid].theta_s - target) / target,
                 "validation_iteration": iteration, "simulation": self.cfg["simulation"],
                 "passed": cid not in failing, "converged": converged,
                 "export_reason": export_reason} for cid in self.cid_list]
        if self.cfg.get("training_mode") == "steps":
            for row in rows:
                row.update(training_mode="steps", local_steps=self.cfg["local_steps"],
                           batch_size=self.cfg["batch_size"], step_seed=self.cfg["seed"])
        filename = "simulated_cpu_config.csv" if self.cfg["simulation"] else "final_cpu_config.csv"
        write_csv(self.output / filename, rows)
        target_record = json.loads((self.output / "target.json").read_text())
        if (not self.cfg["simulation"]
                and target_record.get("definition") == "rounded_deadline_from_max_fitted_theta_at_heterogeneous_initial"):
            atomic_json(self.output / "pacer-control.json",
                        build_pacer_control(self.output, self.cid_list, target_record))
        if not self.cfg["simulation"]:
            env = {"PY": self.cfg["python"], "DATASET": self.cfg["dataset"], "MODEL": self.cfg["model"],
                   "BATCH_SIZE": str(self.cfg["batch_size"]), "LOCAL_EPOCHS": str(self.cfg["epochs"]),
                   "LR": str(self.cfg["lr"]), "NUM_CLASSES": str(self.cfg["num_classes"]),
                   "MPS_ENABLE": str(self.cfg["mps_enable"]),
                   "ENABLE_CPU_AFFINITY": "1" if self.cfg["enable_cpu_affinity"] else "0",
                   "BIND_CLIENT_TO_CPU": "0",
                   "LOCAL_STEPS": str(self.cfg["local_steps"]) if self.cfg.get("training_mode") == "steps" else "",
                   "STEP_SEED": str(self.cfg["seed"]) if self.cfg.get("training_mode") == "steps" else "",
                   "CPU_MAP_ONLY": "1", "CPU_MAP_CSV": str(self.output / filename)}
            command = " ".join(shlex.quote(f"{key}={value}") for key, value in env.items())
            warning = ""
            if not converged:
                message = f"WARNING: CPU calibration has not converged; clients {', '.join(failing)} exceed tolerance. Using the last measured configuration."
                warning = f"printf '%s\\n' {shlex.quote(message)} >&2\n"
            script = ("#!/usr/bin/env bash\nset -euo pipefail\n"
                      ': "${1:?Usage: bash launch_final_clients.sh HOST:PORT (server must already be running)}"\n'
                      f"{warning}"
                      f"exec env {command} bash {shlex.quote(str(Path(self.cfg['project_dir']) / 'launch_clients.sh'))} "
                      f"{shlex.quote(self.cfg['data_dir'])} \"$1\" 0\n")
            (self.output / "launch_final_clients.sh").write_text(script)
        status = "converged" if converged else "max_iterations_reached" if export_reason == "max_iterations" else "exported_unconverged"
        atomic_json(self.output / "status.json", {"status": status, "theta_target_s": target,
                                                 "iteration": iteration, "cpu_config": filename,
                                                 "simulation": self.cfg["simulation"], "converged": converged,
                                                 "failing_clients": failing, "export_reason": export_reason})
        if not converged:
            print(f"WARNING: exported last measured CPU config, but clients {failing} remain outside tolerance.", flush=True)
        return rows
