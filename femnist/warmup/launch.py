"""Launch one measured batch using owned transient systemd services."""

import argparse
import json
import os
import pwd
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

from .measurement import atomic_json, load_cpu_map, validate_distinct_affinity


def client_command(job, client, output):
    cid = client["client_id"]
    return [
        "taskset", "-c", client["cpu_affinity"], job["python"],
        str(Path(job["project_dir"]) / "run_client.py"), "--cid", cid,
        "--data-dir", job["data_dir"], "--dataset", job["dataset"], "--model", job["model"],
        "--epochs", str(job["epochs"]), "--batch-size", str(job["batch_size"]),
        "--lr", str(job["lr"]), "--num-classes", str(job["num_classes"]),
        "--uplink-num-bits", "0", "--local-only", "--local-rounds", str(job["rounds"]),
        "--local-seed", str(job["seed"] + int(cid)),
        "--local-timing-jsonl", str(output / f"client_{cid}.jsonl"),
        "--local-stage-id", job["stage_id"], "--local-cpu-fraction", str(client["cpu"]),
        "--local-cpu-affinity", client["cpu_affinity"],
        "--local-ready-file", str(output / f"ready_{cid}.json"),
        "--local-start-file", str(output / "start.json"),
        "--local-start-timeout", str(job["startup_timeout_s"]),
        "--log-file", str(output / f"training_{cid}.log"),
    ]


def service_command(job, client, output, unit, prefix):
    command = [*prefix, "systemd-run", "--quiet", "--collect", "--wait", "--pipe",
               "--no-ask-password", "--service-type=exec", "--expand-environment=no",
               f"--unit={unit}", f"--uid={pwd.getpwuid(os.getuid()).pw_name}",
               f"--working-directory={job['project_dir']}", "-p", "CPUAccounting=yes",
               "-p", f"CPUQuota={client['cpu'] * 100:.2f}%", "-p", "CPUQuotaPeriodSec=20ms",
               "-p", "TimeoutStopSec=5s", "-p", f"RuntimeMaxSec={job['stage_timeout_s']}"]
    environment = {"CUDA_VISIBLE_DEVICES": "", "SKIP_TF_GPU_MEMORY_GROWTH": "1",
                   "FLWR_TELEMETRY_ENABLED": "0", "TF_CPP_MIN_LOG_LEVEL": "2",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                   "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1",
                   "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": str(job["seed"] + int(client["client_id"]))}
    command.extend(f"--setenv={key}={value}" for key, value in environment.items())
    return command + client_command(job, client, output)


def launch_job(job):
    for tool in ("systemd-run", "systemctl", "taskset"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Required program is unavailable: {tool}")
    prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
    if prefix:
        subprocess.run([*prefix, "true"], check=True, timeout=10)
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "start.json").exists():
        raise ValueError("Refusing to reuse a previous batch start barrier")
    processes, handles, units = [], [], []
    nonce = uuid.uuid4().hex[:12]
    started = time.monotonic()
    try:
        for client in job["clients"]:
            cid = client["client_id"]
            unit = f"flower-warmup-{nonce}-c{cid}.service"
            units.append(unit)
            handle = (output / f"process_{cid}.log").open("w")
            handles.append(handle)
            command = service_command(job, client, output, unit, prefix)
            processes.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT))
        atomic_json(output / "units.json", {"units": units})
        ready = False
        while time.monotonic() - started < job["stage_timeout_s"]:
            codes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError("A client failed; inspect process_<cid>.log")
            if not ready:
                ready_paths = [output / f"ready_{c['client_id']}.json" for c in job["clients"]]
                if all(path.exists() for path in ready_paths):
                    for path, client in zip(ready_paths, job["clients"]):
                        record = json.loads(path.read_text())
                        if record.get("stage_id") != job["stage_id"] or record.get("client_id") != client["client_id"]:
                            raise ValueError("Mismatched client readiness record")
                    atomic_json(output / "start.json", {"stage_id": job["stage_id"]})
                    ready = True
                elif any(code is not None for code in codes):
                    raise RuntimeError("A client exited before the startup barrier")
                elif time.monotonic() - started > job["startup_timeout_s"]:
                    raise TimeoutError("Clients did not all initialize before startup timeout")
            if ready and all(code == 0 for code in codes):
                return
            time.sleep(0.1)
        raise TimeoutError("Warm-up batch exceeded stage_timeout_s")
    finally:
        # Stop only the unique units created by this invocation, including on Ctrl-C.
        previous_handlers = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
        cleanup_errors = []
        try:
            if units:
                try:
                    stopped = subprocess.run([*prefix, "systemctl", "stop", *units], timeout=15,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if stopped.returncode != 0 and any(process.poll() is None for process in processes):
                        cleanup_errors.append("systemctl stop failed while clients were running; check sudo permission")
                except (subprocess.TimeoutExpired, OSError):
                    try:
                        subprocess.run([*prefix, "systemctl", "kill", "--kill-whom=all", "--signal=SIGKILL", *units],
                                       timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except (subprocess.TimeoutExpired, OSError) as error:
                        cleanup_errors.append(str(error))
            for process in processes:
                try:
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError) as error:
                    cleanup_errors.append(str(error))
        finally:
            for handle in handles:
                handle.close()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        if cleanup_errors:
            raise RuntimeError(f"Owned-unit cleanup failed; inspect {output / 'units.json'}: {cleanup_errors}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job", type=Path)
    group.add_argument("--cpu-map", type=Path)
    parser.add_argument("--require-distinct-affinity", action="store_true")
    args = parser.parse_args()
    if args.cpu_map:
        mapping = load_cpu_map(args.cpu_map)
        if args.require_distinct_affinity:
            validate_distinct_affinity(mapping)
        for cid, (cpu, affinity) in mapping.items():
            print(f"{cid}\t{cpu:.10f}\t{affinity}")
    else:
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt("Warm-up launcher interrupted")
        signal.signal(signal.SIGTERM, interrupted)
        launch_job(json.loads(args.job.read_text()))
