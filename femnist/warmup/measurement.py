"""Full-precision timing records and CPU allocation checks in the client."""

import csv
import json
import math
import os
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_cpu_map(path):
    result = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "client_id" not in reader.fieldnames:
            raise ValueError("CPU map requires a client_id column")
        column = "cpu" if "cpu" in reader.fieldnames else "new_cpu" if "new_cpu" in reader.fieldnames else None
        if column is None:
            raise ValueError("CPU map requires cpu (0-1 fraction) or new_cpu")
        for row in reader:
            cid = str(int(row["client_id"]))
            cpu = float(row[column])
            if int(cid) < 0 or cid in result or not math.isfinite(cpu) or not 0 < cpu <= 1:
                raise ValueError(f"Invalid/duplicate CPU assignment for client {cid}")
            affinity = row.get("cpu_affinity", "").strip()
            if affinity == "-":
                affinity = ""
            if affinity and any(not part.isdigit() for part in affinity.split(",")):
                raise ValueError(f"Invalid CPU affinity for client {cid}")
            result[cid] = (cpu, affinity or "-")
    return result


def validate_distinct_affinity(mapping):
    seen = set()
    for cid, (_, affinity) in mapping.items():
        if not affinity.isdigit() or int(affinity) in seen:
            raise ValueError(f"Calibrated client {cid} needs its own single logical CPU; got {affinity}")
        seen.add(int(affinity))


def cpu_snapshot():
    affinity = sorted(os.sched_getaffinity(0))
    thread_masks = []
    for task in Path("/proc/self/task").iterdir():
        try:
            thread_masks.append(tuple(sorted(os.sched_getaffinity(int(task.name)))))
        except ProcessLookupError:
            continue
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            root = Path("/sys/fs/cgroup").resolve()
            group = (root / line.split(":", 2)[2].lstrip("/")).resolve()
            if not group.is_relative_to(root):
                raise ValueError("Invalid cgroup path")
            quota, period = (group / "cpu.max").read_text().split()
            if quota == "max":
                raise ValueError("Client CPU quota is unlimited")
            quota, period = int(quota), int(period)
            if quota <= 0 or period <= 0:
                raise ValueError("Invalid cgroup quota")
            snapshot = {"cpu_actual": quota / period, "quota_us": quota, "period_us": period,
                        "cpu_affinity": affinity, "cgroup": str(group), "thread_count": len(thread_masks),
                        "thread_cpu_affinities": [list(mask) for mask in sorted(set(thread_masks))]}
            frequencies = {}
            for cpu in affinity:
                try:
                    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
                    frequencies[str(cpu)] = float(path.read_text()) / 1000.0
                except (OSError, ValueError):
                    continue
            snapshot["bound_cpu_frequencies_mhz"] = frequencies
            stats = dict(line.split() for line in (group / "cpu.stat").read_text().splitlines())
            for key in ("usage_usec", "throttled_usec", "nr_periods", "nr_throttled"):
                if key in stats:
                    snapshot[f"cgroup_{key}"] = int(stats[key])
            return snapshot
    raise ValueError("Managed warm-up requires Linux cgroup v2")


class MeasurementSession:
    def __init__(self, output, stage_id, client_id, cpu=None, affinity=None):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists():
            raise ValueError(f"Refusing to mix old measurements into {self.output}")
        self.stage_id, self.client_id = stage_id, str(client_id)
        self.cpu = cpu
        self.affinity = sorted(int(v) for v in affinity.split(",")) if affinity else None

    def check_resources(self):
        if self.cpu is None:
            return {}
        snapshot = cpu_snapshot()
        if not math.isclose(snapshot["cpu_actual"], self.cpu, rel_tol=0.001, abs_tol=1e-6):
            raise ValueError(f"CPU quota mismatch: requested={self.cpu}, actual={snapshot['cpu_actual']}")
        if self.affinity is not None and snapshot["cpu_affinity"] != self.affinity:
            raise ValueError(f"CPU affinity mismatch: {snapshot['cpu_affinity']} != {self.affinity}")
        if self.affinity is not None and any(mask != self.affinity for mask in snapshot.get("thread_cpu_affinities", [])):
            raise ValueError(f"Worker thread affinity mismatch: {snapshot['thread_cpu_affinities']}")
        return snapshot

    def wait_for_start(self, ready_file, start_file, timeout):
        resources = self.check_resources()
        atomic_json(ready_file, {"stage_id": self.stage_id, "client_id": self.client_id, **resources})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if Path(start_file).exists():
                command = json.loads(Path(start_file).read_text())
                if command.get("stage_id") != self.stage_id:
                    raise ValueError("Wrong warm-up start barrier")
                return
            time.sleep(0.05)
        raise TimeoutError("Timed out waiting for all warm-up clients")

    def record(self, server_round, duration, **extra):
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Nonfinite or nonpositive training duration")
        record = {"stage_id": self.stage_id, "client_id": self.client_id, "round": server_round,
                  "train_time_s": float(duration), "cpu_requested": self.cpu,
                  **self.check_resources(), **extra}
        with self.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
