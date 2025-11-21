#!/usr/bin/env python3
"""Monitor Flower client processes locally or via SSH and record runtime metrics."""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import math
import os
import signal
import string
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

CPUFREQ_ROOT = Path("/sys/devices/system/cpu")
psutil = None  # Lazy import for local operations
log_write_lock = threading.Lock()


def ensure_psutil() -> None:
    global psutil
    if psutil is not None:
        return
    try:
        import psutil as _psutil  # type: ignore[import]
    except ImportError:
        print("psutil is required. Install with `sudo apt install python3-psutil`.", file=sys.stderr)
        sys.exit(1)
    psutil = _psutil


def read_cpu_freq_mhz_local() -> Tuple[float, List[float]]:
    freqs: List[float] = []
    for freq_file in CPUFREQ_ROOT.glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try:
            raw = freq_file.read_text().strip()
        except OSError:
            continue
        if not raw:
            continue
        try:
            freqs.append(float(raw) / 1000.0)
        except ValueError:
            continue
    if not freqs:
        return float("nan"), []
    return sum(freqs) / len(freqs), freqs


def _read_kernel_stack(pid_value: int) -> Optional[str]:
    stack_file = Path(f"/proc/{pid_value}/stack")
    try:
        raw = stack_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None
    return "\n".join(lines[:8])


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor CPU usage spikes for one process.")
    parser.add_argument("--pattern", type=str, default="client", help="Substring to match in process command line (default: client)")
    parser.add_argument("--threshold", type=float, default=80.0, help="CPU %% trigger for the target process")
    parser.add_argument("--other-threshold", type=float, default=50.0, help="CPU %% threshold for other processes to record")
    parser.add_argument("--interval", type=float, default=0.2, help="Sampling interval in seconds")
    parser.add_argument("--log", type=str, default="monitor-log.txt", help="Path to log file (JSON lines)")
    parser.add_argument("--max-duration", type=float, default=0.0, help="Maximum monitoring duration (0 = unlimited)")
    parser.add_argument("--cooldown", type=float, default=1.0, help="Seconds below threshold before marking training end")
    parser.add_argument("--log-idle", action="store_true", help="Log samples even when CPU utilisation is below threshold")
    parser.add_argument("--ssh-host", dest="ssh_hosts", action="append", help="Hostname or IP for SSH monitoring (can repeat)")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH port")
    parser.add_argument("--ssh-user", type=str, help="SSH username (default: current user)")
    parser.add_argument("--ssh-password", type=str, help="SSH password (optional if keys configured)")
    parser.add_argument("--ssh-key", type=str, help="SSH private key path (optional)")
    return parser.parse_args(argv)


def find_matching_pid_local(pattern: str) -> Tuple[int, str]:
    ensure_psutil()
    pattern_lower = pattern.lower()
    candidates: List[Tuple[float, int, str]] = []
    current_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "cmdline", "name", "create_time"]):
        try:
            pid = proc.info["pid"]
            if pid == current_pid:
                continue
            cmdline_list = proc.info.get("cmdline") or []
            command = " ".join(cmdline_list) if cmdline_list else (proc.info.get("name") or "")
            if not command or pattern_lower not in command.lower():
                continue
            create_time = proc.info.get("create_time") or 0.0
            candidates.append((create_time, pid, command))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not candidates:
        raise ValueError(f"No process matching pattern '{pattern}' found locally.")
    candidates.sort()
    _, pid, command = candidates[0]
    return pid, command


def _monitor_loop(
    proc,
    pid: int,
    command: str,
    args: argparse.Namespace,
    emit,
    read_wchan_fn,
    read_freq_fn,
    read_stack_fn,
) -> None:
    try:
        prev_ctx = proc.num_ctx_switches()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        prev_ctx = None
    try:
        prev_io = proc.io_counters()
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
        prev_io = None

    start_time = time.time()
    active = False
    active_start: Optional[float] = None
    last_active_seen: Optional[float] = None

    while True:
        if args.max_duration and time.time() - start_time > args.max_duration:
            break
        if not proc.is_running():
            break

        time.sleep(args.interval)

        try:
            proc_cpu = proc.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break

        timestamp = time.time()
        ts_iso = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat()
        is_above_threshold = proc_cpu >= args.threshold

        try:
            status = proc.status()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            status = None

        wchan = read_wchan_fn(pid)
        kernel_stack = None
        if wchan and "futex" in str(wchan):
            kernel_stack = read_stack_fn(pid)

        ctx_vol_delta = ctx_invol_delta = None
        try:
            ctx = proc.num_ctx_switches()
            if prev_ctx is not None:
                ctx_vol_delta = max(0, ctx.voluntary - prev_ctx.voluntary)
                ctx_invol_delta = max(0, ctx.involuntary - prev_ctx.involuntary)
            prev_ctx = ctx
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            ctx_vol_delta = ctx_invol_delta = None

        io_read_bytes_delta = io_write_bytes_delta = None
        io_read_count_delta = io_write_count_delta = None
        try:
            io_c = proc.io_counters()
            if prev_io is not None and io_c is not None:
                if hasattr(io_c, "read_bytes") and hasattr(prev_io, "read_bytes"):
                    io_read_bytes_delta = max(0, io_c.read_bytes - prev_io.read_bytes)
                if hasattr(io_c, "write_bytes") and hasattr(prev_io, "write_bytes"):
                    io_write_bytes_delta = max(0, io_c.write_bytes - prev_io.write_bytes)
                if hasattr(io_c, "read_count") and hasattr(prev_io, "read_count"):
                    io_read_count_delta = max(0, io_c.read_count - prev_io.read_count)
                if hasattr(io_c, "write_count") and hasattr(prev_io, "write_count"):
                    io_write_count_delta = max(0, io_c.write_count - prev_io.write_count)
            prev_io = io_c
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            io_read_bytes_delta = io_write_bytes_delta = None
            io_read_count_delta = io_write_count_delta = None

        rss_bytes = vms_bytes = None
        try:
            mem = proc.memory_info()
            rss_bytes = int(mem.rss)
            vms_bytes = int(mem.vms)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            rss_bytes = vms_bytes = None

        num_threads = None
        try:
            num_threads = proc.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            num_threads = None

        if is_above_threshold:
            if not active:
                active = True
                active_start = timestamp
                last_active_seen = timestamp
                emit(
                    {
                        "timestamp_iso": ts_iso,
                        "timestamp_epoch": round(timestamp, 6),
                        "pid": pid,
                        "command": command,
                        "event": "start",
                        "proc_cpu_percent": round(proc_cpu, 2),
                        "status": status,
                        "wchan": wchan,
                        "num_threads": num_threads,
                        "kernel_stack": kernel_stack,
                    }
                )
            else:
                last_active_seen = timestamp
        else:
            if active and last_active_seen is not None and timestamp - last_active_seen >= args.cooldown:
                duration = (last_active_seen - active_start) if active_start is not None else None
                emit(
                    {
                        "timestamp_iso": ts_iso,
                        "timestamp_epoch": round(timestamp, 6),
                        "pid": pid,
                        "command": command,
                        "event": "end",
                        "active_duration_s": round(duration, 6) if duration is not None else None,
                        "status": status,
                        "wchan": wchan,
                        "num_threads": num_threads,
                        "kernel_stack": kernel_stack,
                    }
                )
                active = False
                active_start = None
                last_active_seen = None

        if not (args.log_idle or is_above_threshold or active):
            continue

        avg_freq, freqs = read_freq_fn()
        competing: List[Tuple[int, str, float]] = []
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            if p.pid == pid:
                continue
            try:
                cpu_val = p.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if cpu_val >= args.other_threshold:
                name = p.info.get("name") or ""
                if not name and p.info.get("cmdline"):
                    name = " ".join(p.info["cmdline"])[:80]
                competing.append((p.pid, name, cpu_val))

        entry = {
            "timestamp_iso": ts_iso,
            "timestamp_epoch": round(timestamp, 6),
            "pid": pid,
            "command": command,
            "event": None,
            "proc_cpu_percent": round(proc_cpu, 2),
            "avg_freq_mhz": None if math.isnan(avg_freq) else round(avg_freq, 2),
            "freqs_mhz": [round(val, 2) for val in freqs],
            "other_processes": competing,
            "session_active": active,
            "above_threshold": is_above_threshold,
            "status": status,
            "wchan": wchan,
            "num_threads": num_threads,
            "ctx_switches_voluntary_delta": ctx_vol_delta,
            "ctx_switches_involuntary_delta": ctx_invol_delta,
            "io_read_bytes_delta": io_read_bytes_delta,
            "io_write_bytes_delta": io_write_bytes_delta,
            "io_read_count_delta": io_read_count_delta,
            "io_write_count_delta": io_write_count_delta,
            "rss_bytes": rss_bytes,
            "vms_bytes": vms_bytes,
            "kernel_stack": kernel_stack,
        }
        emit(entry)


def monitor_single_local(pid: int, command: str, args: argparse.Namespace, log_path: Path) -> int:
    ensure_psutil()
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"Process {pid} does not exist.", file=sys.stderr)
        return 1

    stop = False

    def _sig_handler(signum, frame):  # noqa: ANN001
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    for p in psutil.process_iter():
        try:
            p.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    def read_wchan_local(pid_value: int) -> Optional[str]:
        try:
            return Path(f"/proc/{pid_value}/wchan").read_text().strip()
        except OSError:
            return None

    def read_stack_local(pid_value: int) -> Optional[str]:
        return _read_kernel_stack(pid_value)

    with log_path.open("a") as logfile:
        def emit(entry: dict) -> None:
            entry_with_host = dict(entry)
            entry_with_host.setdefault("mode", "local")
            entry_with_host.setdefault("host", "local")
            line = json.dumps(entry_with_host, ensure_ascii=False)
            with log_write_lock:
                logfile.write(line + os.linesep)
                logfile.flush()

        proc.cpu_percent(None)
        _monitor_loop(proc, pid, command, args, emit, read_wchan_local, read_cpu_freq_mhz_local, read_stack_local)
    return 0


REMOTE_TEMPLATE = string.Template(
    """
import datetime as dt
import json
import math
import pathlib
import sys
import time

try:
    import psutil  # type: ignore[import]
except ImportError:
    sys.stderr.write("psutil is required on the remote host. Install with `sudo apt install python3-psutil`.\\n")
    sys.exit(1)

CPUFREQ_ROOT = pathlib.Path("/sys/devices/system/cpu")


def read_cpu_freq_mhz():
    freqs = []
    for freq_file in CPUFREQ_ROOT.glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try:
            raw = freq_file.read_text().strip()
        except OSError:
            continue
        if not raw:
            continue
        try:
            freqs.append(float(raw) / 1000.0)
        except ValueError:
            continue
    if not freqs:
        return float("nan"), []
    return sum(freqs) / len(freqs), freqs

pattern = ${PATTERN}
threshold = ${THRESHOLD}
other_threshold = ${OTHER_THRESHOLD}
interval = ${INTERVAL}
max_duration = ${MAX_DURATION}
cooldown = ${COOLDOWN}
log_idle = ${LOG_IDLE}


def select_target_pid():
    candidates = []
    for proc_iter in psutil.process_iter(["pid", "cmdline", "name", "create_time"]):
        try:
            cmdline_list = proc_iter.info.get("cmdline") or []
            command = " ".join(cmdline_list) if cmdline_list else (proc_iter.info.get("name") or "")
            if not command or pattern not in command.lower():
                continue
            create_time = proc_iter.info.get("create_time") or 0.0
            candidates.append((create_time, proc_iter.pid, command))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not candidates:
        return None, None
    candidates.sort()
    _, pid_val, command_val = candidates[0]
    return pid_val, command_val

pid, command = select_target_pid()
if pid is None:
    sys.stderr.write(f"No process matching pattern '{pattern}' found.\\n")
    sys.exit(1)


def read_wchan(pid_value: int):
    try:
        with open(f"/proc/{pid_value}/wchan", "r", encoding="utf-8") as wf:
            return wf.read().strip()
    except OSError:
        return None


def read_kernel_stack(pid_value: int):
    try:
        with open(f"/proc/{pid_value}/stack", "r", encoding="utf-8") as sf:
            data = sf.read().strip()
    except OSError:
        return None
    if not data:
        return None
    lines = [line.strip() for line in data.splitlines() if line.strip()]
    if not lines:
        return None
    return "\\n".join(lines[:8])


try:
    proc = psutil.Process(pid)
except psutil.NoSuchProcess:
    sys.stderr.write(f"Process {pid} does not exist.\\n")
    sys.exit(1)

for proc_iter in psutil.process_iter():
    try:
        proc_iter.cpu_percent(None)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

try:
    prev_ctx = proc.num_ctx_switches()
except (psutil.NoSuchProcess, psutil.AccessDenied):
    prev_ctx = None
try:
    prev_io = proc.io_counters()
except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
    prev_io = None

start_time = time.time()
active = False
active_start = None
last_active_seen = None

while True:
    if max_duration and time.time() - start_time > max_duration:
        break
    if not proc.is_running():
        break

    time.sleep(interval)

    try:
        proc_cpu = proc.cpu_percent(None)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        break

    timestamp = time.time()
    ts_iso = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat()
    is_above_threshold = proc_cpu >= threshold

    try:
        status = proc.status()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        status = None

    wchan = read_wchan(pid)
    kernel_stack = read_kernel_stack(pid) if wchan and "futex" in wchan else None

    ctx_vol_delta = ctx_invol_delta = None
    try:
        ctx = proc.num_ctx_switches()
        if prev_ctx is not None:
            ctx_vol_delta = max(0, ctx.voluntary - prev_ctx.voluntary)
            ctx_invol_delta = max(0, ctx.involuntary - prev_ctx.involuntary)
        prev_ctx = ctx
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        ctx_vol_delta = ctx_invol_delta = None

    io_read_bytes_delta = io_write_bytes_delta = None
    io_read_count_delta = io_write_count_delta = None
    try:
        io_c = proc.io_counters()
        if prev_io is not None and io_c is not None:
            if hasattr(io_c, "read_bytes") and hasattr(prev_io, "read_bytes"):
                io_read_bytes_delta = max(0, io_c.read_bytes - prev_io.read_bytes)
            if hasattr(io_c, "write_bytes") and hasattr(prev_io, "write_bytes"):
                io_write_bytes_delta = max(0, io_c.write_bytes - prev_io.write_bytes)
            if hasattr(io_c, "read_count") and hasattr(prev_io, "read_count"):
                io_read_count_delta = max(0, io_c.read_count - prev_io.read_count)
            if hasattr(io_c, "write_count") and hasattr(prev_io, "write_count"):
                io_write_count_delta = max(0, io_c.write_count - prev_io.write_count)
        prev_io = io_c
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
        io_read_bytes_delta = io_write_bytes_delta = None
        io_read_count_delta = io_write_count_delta = None

    rss_bytes = vms_bytes = None
    try:
        mem = proc.memory_info()
        rss_bytes = int(mem.rss)
        vms_bytes = int(mem.vms)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        rss_bytes = vms_bytes = None

    num_threads = None
    try:
        num_threads = proc.num_threads()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        num_threads = None

    if is_above_threshold:
        if not active:
            active = True
            active_start = timestamp
            last_active_seen = timestamp
            entry = {
                "timestamp_iso": ts_iso,
                "timestamp_epoch": round(timestamp, 6),
                "pid": pid,
                "command": command,
                "event": "start",
                "mode": "remote",
                "host": ${HOST},
                "proc_cpu_percent": round(proc_cpu, 2),
                "status": status,
                "wchan": wchan,
                "num_threads": num_threads,
                "kernel_stack": kernel_stack,
            }
            print(json.dumps(entry, ensure_ascii=False))
            sys.stdout.flush()
        else:
            last_active_seen = timestamp
    else:
        if active and last_active_seen is not None and timestamp - last_active_seen >= cooldown:
            duration = (last_active_seen - active_start) if active_start is not None else None
            entry = {
                "timestamp_iso": ts_iso,
                "timestamp_epoch": round(timestamp, 6),
                "pid": pid,
                "command": command,
                "event": "end",
                "mode": "remote",
                "host": ${HOST},
                "active_duration_s": round(duration, 6) if duration is not None else None,
                "status": status,
                "wchan": wchan,
                "num_threads": num_threads,
                "kernel_stack": kernel_stack,
            }
            print(json.dumps(entry, ensure_ascii=False))
            sys.stdout.flush()
            active = False
            active_start = None
            last_active_seen = None

    if not (log_idle or is_above_threshold or active):
        continue

    avg_freq, freqs = read_cpu_freq_mhz()
    competing = []
    for proc_iter in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc_iter.pid == pid:
            continue
        try:
            cpu_val = proc_iter.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if cpu_val >= other_threshold:
            name = proc_iter.info.get("name") or ""
            if not name and proc_iter.info.get("cmdline"):
                name = " ".join(proc_iter.info["cmdline"])[:80]
            competing.append([proc_iter.pid, name, cpu_val])

    entry = {
        "timestamp_iso": ts_iso,
        "timestamp_epoch": round(timestamp, 6),
        "pid": pid,
        "command": command,
        "event": None,
        "proc_cpu_percent": round(proc_cpu, 2),
        "avg_freq_mhz": None if math.isnan(avg_freq) else round(avg_freq, 2),
        "freqs_mhz": [round(val, 2) for val in freqs],
        "other_processes": competing,
        "mode": "remote",
        "host": ${HOST},
        "session_active": active,
        "above_threshold": is_above_threshold,
        "status": status,
        "wchan": wchan,
        "num_threads": num_threads,
        "ctx_switches_voluntary_delta": ctx_vol_delta,
        "ctx_switches_involuntary_delta": ctx_invol_delta,
        "io_read_bytes_delta": io_read_bytes_delta,
        "io_write_bytes_delta": io_write_bytes_delta,
        "io_read_count_delta": io_read_count_delta,
        "io_write_count_delta": io_write_count_delta,
        "rss_bytes": rss_bytes,
        "vms_bytes": vms_bytes,
        "kernel_stack": kernel_stack,
    }
    print(json.dumps(entry, ensure_ascii=False))
    sys.stdout.flush()
"""
)


def build_remote_script(args: argparse.Namespace, host_label: str) -> str:
    mapping = {
        "PATTERN": repr(args.pattern),
        "THRESHOLD": args.threshold,
        "OTHER_THRESHOLD": args.other_threshold,
        "INTERVAL": args.interval,
        "MAX_DURATION": args.max_duration,
        "COOLDOWN": args.cooldown,
        "LOG_IDLE": "True" if args.log_idle else "False",
        "HOST": repr(host_label),
    }
    script = REMOTE_TEMPLATE.substitute(mapping)
    return textwrap.dedent(script)

def monitor_remote_host(host: str, args: argparse.Namespace, log_path: Path, results: List[int]) -> None:
    try:
        import paramiko  # type: ignore[import]
    except ImportError:
        print("paramiko is required for SSH monitoring. Install with `pip install paramiko`.", file=sys.stderr)
        results.append(1)
        return

    user = args.ssh_user or getpass.getuser()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host,
            port=args.ssh_port,
            username=user,
            password=args.ssh_password or None,
            key_filename=args.ssh_key or None,
            timeout=10.0,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"Failed to connect to {host}: {exc}", file=sys.stderr)
        results.append(1)
        return

    remote_script = build_remote_script(args, host)
    command = f"python3 - <<'PY'\n{remote_script}\nPY\n"

    stdin, stdout, stderr = client.exec_command(command)
    stdin.close()

    try:
        with log_path.open("a") as log_file:
            while True:
                line = stdout.readline()
                if not line:
                    if stdout.channel.exit_status_ready():
                        break
                    time.sleep(0.1)
                    continue
                with log_write_lock:
                    log_file.write(line)
                    log_file.flush()
        err_output = stderr.read().decode(errors="ignore")
        if err_output.strip():
            print(err_output, file=sys.stderr)
        exit_status = stdout.channel.recv_exit_status()
    finally:
        client.close()
    results.append(exit_status)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    log_path = Path(args.log).expanduser().resolve()
    if not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.touch()

    if args.ssh_hosts:
        results: List[int] = []
        threads: List[threading.Thread] = []
        for host in args.ssh_hosts:
            t = threading.Thread(target=monitor_remote_host, args=(host, args, log_path, results), daemon=False)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        if not results:
            return 1
        return 0 if all(status == 0 for status in results) else 1

    try:
        pid, command = find_matching_pid_local(args.pattern)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return monitor_single_local(pid, command, args, log_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
