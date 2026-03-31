#!/usr/bin/env python3
"""Launch a Flower deployment runtime with one OS process per client.

This launcher starts:
1. one `flower-superlink` process
2. one `flower-supernode` process per client/partition
3. one `flwr run` submission against the local SuperLink

Each SuperNode receives its own environment, which allows per-client GPU/MPS
configuration through environment variables such as CUDA_VISIBLE_DEVICES or
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from typing import Iterable


DEFAULT_CLIENTS_BY_APP = {
    "flowertune-llm-medical": 20,
    "flowertune-llm-general-nlp": 20,
    "flowertune-llm-code": 10,
    "flowertune-llm-finance": 50,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Flower App via SuperLink/SuperNode deployment runtime so that "
            "each client gets its own OS process."
        )
    )
    parser.add_argument(
        "--app-dir",
        type=Path,
        default=Path.cwd(),
        help="Flower app directory containing pyproject.toml",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=None,
        help="Total number of client partitions/supernodes to launch",
    )
    parser.add_argument(
        "--run-config",
        action="append",
        default=[],
        help="Forwarded to `flwr run --run-config` (can be repeated)",
    )
    parser.add_argument(
        "--client-env-file",
        type=Path,
        default=None,
        help=(
            "JSON file mapping client ids to extra env vars. "
            'Use "*" for defaults applied to every client.'
        ),
    )
    parser.add_argument(
        "--fleet-api-address",
        default="127.0.0.1:9092",
        help="Address used by SuperNodes to reach SuperLink Fleet API",
    )
    parser.add_argument(
        "--control-api-address",
        default="127.0.0.1:9093",
        help="Address used by `flwr run` to reach SuperLink Control API",
    )
    parser.add_argument(
        "--serverappio-api-address",
        default="127.0.0.1:9091",
        help="Address used internally by SuperLink to launch ServerApp",
    )
    parser.add_argument(
        "--clientappio-base-port",
        type=int,
        default=10094,
        help="Base port used to assign one ClientAppIO address per SuperNode",
    )
    parser.add_argument(
        "--connection-name",
        default="local-process",
        help="Temporary SuperLink connection name written into FLWR_HOME/config.toml",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=None,
        help="Directory where launcher state/logs are written",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Pass `--stream` to `flwr run`",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only start SuperLink/SuperNodes and keep them alive until Ctrl-C",
    )
    return parser.parse_args(argv)


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(
            f"Required command not found in PATH: {name}. "
            "Activate the environment where Flower is installed first."
        )
    return path


def _guess_num_clients(app_dir: Path) -> int:
    current = app_dir.resolve()
    for candidate in (current.name, current.parent.name):
        if candidate in DEFAULT_CLIENTS_BY_APP:
            return DEFAULT_CLIENTS_BY_APP[candidate]
    raise SystemExit(
        "--num-clients is required because the app directory is not one of the "
        f"known defaults: {', '.join(sorted(DEFAULT_CLIENTS_BY_APP))}"
    )


def _parse_host_port(address: str) -> tuple[str, int]:
    host, sep, port = address.rpartition(":")
    if not sep:
        raise SystemExit(f"Invalid address, expected HOST:PORT, got: {address}")
    return host, int(port)


def _wait_for_port(address: str, timeout_s: float) -> None:
    host, port = _parse_host_port(address)
    deadline = time.time() + timeout_s
    last_error: OSError | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {address}: {last_error}")


def _load_client_envs(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Client env file must contain a JSON object: {path}")
    envs: dict[str, dict[str, str]] = {}
    for key, value in payload.items():
        if key.startswith("_"):
            continue
        if not isinstance(value, dict):
            raise SystemExit(f"Client env entry must be an object for key {key!r} in {path}")
        envs[str(key)] = {str(k): str(v) for k, v in value.items()}
    return envs


def _client_env(base_env: dict[str, str], overrides: dict[str, dict[str, str]], cid: int) -> dict[str, str]:
    env = dict(base_env)
    env.update(overrides.get("*", {}))
    env.update(overrides.get(str(cid), {}))
    env["FLOWER_CLIENT_ID"] = str(cid)
    return env


def _write_flwr_config(flwr_home: Path, connection_name: str, control_api_address: str) -> None:
    flwr_home.mkdir(parents=True, exist_ok=True)
    config_path = flwr_home / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                f"[superlink.{connection_name}]",
                f'address = "{control_api_address}"',
                "insecure = true",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _terminate_process_tree(proc: subprocess.Popen[bytes], name: str) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=5)
    print(f"[launcher] force-killed {name} (pid={proc.pid})", file=sys.stderr)


def _start_logged_process(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> tuple[subprocess.Popen[bytes], object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, handle


def _assert_running(processes: Iterable[tuple[str, subprocess.Popen[bytes], Path]]) -> None:
    for name, proc, log_path in processes:
        code = proc.poll()
        if code is not None:
            raise RuntimeError(
                f"{name} exited early with code {code}. See log: {log_path}"
            )


def main(
    argv: list[str] | None = None,
    *,
    default_app_dir: Path | None = None,
    default_num_clients: int | None = None,
) -> int:
    args = _parse_args(argv)
    if default_app_dir is not None and args.app_dir == Path.cwd():
        args.app_dir = default_app_dir
    args.app_dir = args.app_dir.resolve()
    if not (args.app_dir / "pyproject.toml").exists():
        raise SystemExit(f"pyproject.toml not found under app dir: {args.app_dir}")

    if args.num_clients is None:
        args.num_clients = default_num_clients or _guess_num_clients(args.app_dir)

    if args.client_env_file is None:
        candidate = args.app_dir / "client_envs.json"
        args.client_env_file = candidate if candidate.exists() else None
    elif not args.client_env_file.is_absolute():
        args.client_env_file = (args.app_dir / args.client_env_file).resolve()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    runtime_root = (
        args.runtime_root.resolve()
        if args.runtime_root is not None
        else args.app_dir / ".flower-process-runtime" / timestamp
    )
    logs_dir = runtime_root / "logs"
    flwr_home = runtime_root / "flwr-home"
    _write_flwr_config(flwr_home, args.connection_name, args.control_api_address)

    base_env = dict(os.environ)
    base_env["FLWR_HOME"] = str(flwr_home)
    client_envs = _load_client_envs(args.client_env_file)

    flwr_bin = _require_binary("flwr")
    superlink_bin = _require_binary("flower-superlink")
    supernode_bin = _require_binary("flower-supernode")

    managed_processes: list[tuple[str, subprocess.Popen[bytes], Path, object]] = []

    def cleanup() -> None:
        while managed_processes:
            name, proc, _log_path, handle = managed_processes.pop()
            _terminate_process_tree(proc, name)
            handle.close()

    atexit.register(cleanup)

    superlink_log = logs_dir / "superlink.log"
    superlink_cmd = [
        superlink_bin,
        "--insecure",
        "--fleet-api-address",
        args.fleet_api_address,
        "--control-api-address",
        args.control_api_address,
        "--serverappio-api-address",
        args.serverappio_api_address,
        "--database",
        str(runtime_root / "superlink-state.sqlite"),
        "--storage-dir",
        str(runtime_root / "ffs"),
    ]
    superlink_proc, superlink_handle = _start_logged_process(
        superlink_cmd,
        cwd=args.app_dir,
        env=base_env,
        log_path=superlink_log,
    )
    managed_processes.append(("superlink", superlink_proc, superlink_log, superlink_handle))

    _wait_for_port(args.fleet_api_address, timeout_s=30.0)
    _wait_for_port(args.control_api_address, timeout_s=30.0)

    supernode_status: list[tuple[str, subprocess.Popen[bytes], Path]] = []
    for cid in range(args.num_clients):
        client_log = logs_dir / f"supernode_{cid:03d}.log"
        client_addr = f"127.0.0.1:{args.clientappio_base_port + cid}"
        node_cfg = f"partition-id={cid} num-partitions={args.num_clients}"
        cmd = [
            supernode_bin,
            "--insecure",
            "--superlink",
            args.fleet_api_address,
            "--clientappio-api-address",
            client_addr,
            "--node-config",
            node_cfg,
        ]
        proc, handle = _start_logged_process(
            cmd,
            cwd=args.app_dir,
            env=_client_env(base_env, client_envs, cid),
            log_path=client_log,
        )
        managed_processes.append((f"supernode-{cid}", proc, client_log, handle))
        supernode_status.append((f"supernode-{cid}", proc, client_log))

    time.sleep(2.0)
    _assert_running([("superlink", superlink_proc, superlink_log), *supernode_status])

    print(f"[launcher] app_dir={args.app_dir}")
    print(f"[launcher] runtime_root={runtime_root}")
    if args.client_env_file is not None:
        print(f"[launcher] client_env_file={args.client_env_file}")
    print(f"[launcher] launched {args.num_clients} client processes")
    print(f"[launcher] logs={logs_dir}")

    if args.no_run:
        print(
            "[launcher] federation is up. Press Ctrl-C to stop it. "
            f"Use FLWR_HOME={flwr_home} to submit runs manually."
        )
        try:
            while True:
                _assert_running([("superlink", superlink_proc, superlink_log), *supernode_status])
                time.sleep(2.0)
        except KeyboardInterrupt:
            print("[launcher] stopping runtime")
            return 0

    run_cmd = [flwr_bin, "run", ".", args.connection_name]
    for override in args.run_config:
        run_cmd.extend(["--run-config", override])
    if args.stream:
        run_cmd.append("--stream")

    print(f"[launcher] submitting run via connection {args.connection_name}")
    run_proc = subprocess.run(
        run_cmd,
        cwd=str(args.app_dir),
        env=base_env,
        check=False,
    )
    return run_proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
