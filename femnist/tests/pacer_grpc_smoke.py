"""Three real gRPC clients, eight rounds and one external target update.

Run from femnist: ../venv/bin/python tests/pacer_grpc_smoke.py
Only synthetic model values/timings are used; no dataset or resource change.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def control():
    return dict(run_id="grpc-smoke", config_version=1, effective_round=1,
                theta_target_s=4.5, deadline_s=4.8, ref_a=0.02,
                q=0.9, q_lower=0.85, q_upper=0.95,
                required_client_ids=["0", "1", "2"],
                gamma_window=4, gamma_min_samples=3, violation_patience=2)


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def run_server(args):
    import numpy as np
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import Server
    from flwr.server.client_manager import SimpleClientManager
    from flwr.server.fleet.grpc_bidi.grpc_server import start_grpc_server
    from flwr.server.strategy import FedAvg
    from pacer.external import FileControlSource, JsonlSink
    from pacer.server import PacerStrategy

    class ExternalPublisher(JsonlSink):
        def emit(self, event):
            super().emit(event)
            if event.get("event") == "round_observation" and event["server_round"] == 4:
                updated = {**control(), "config_version": 2, "effective_round": 5, "theta_target_s": 4.4}
                atomic_json(args.directory / "control.json", updated)

    sink = ExternalPublisher(args.directory / "rounds.jsonl")
    source = FileControlSource(args.directory / "control.json", sink)
    base = FedAvg(fraction_fit=1, fraction_evaluate=0, min_fit_clients=3,
                  min_available_clients=3, accept_failures=False,
                  initial_parameters=ndarrays_to_parameters([np.zeros(1)]),
                  on_fit_config_fn=lambda r: {"local_epochs": 1, "batch_size": 1})
    manager = SimpleClientManager()
    server = Server(client_manager=manager, strategy=PacerStrategy(base, source, sink))
    grpc_server = start_grpc_server(client_manager=manager, server_address=args.address)
    try:
        server.fit(num_rounds=8, timeout=15)
        np.testing.assert_allclose(parameters_to_ndarrays(server.parameters)[0], [8 * 14 / 6])
        server.disconnect_all_clients(timeout=5)
    finally:
        grpc_server.stop(grace=0).wait(timeout=5)


def run_client(args):
    import numpy as np
    import flwr as fl
    from pacer.client import PacerNumPyClient
    from pacer.external import JsonlSink

    class SyntheticClient(fl.client.NumPyClient):
        def __init__(self):
            self.version = None
            self.index = 0

        def get_parameters(self, config):
            return [np.zeros(1)]

        def fit(self, parameters, config):
            if self.version != config["pacer.config_version"]:
                self.version, self.index = config["pacer.config_version"], 0
            self.index += 1
            theta, a = config["pacer.theta_target_s"], config["pacer.ref_a"]
            quantile = (self.index - 0.5) / 4
            f = quantile ** (1 / (1.0 + args.cid * 0.2))
            duration = theta + a * theta * np.log(f / (1 - f))
            return [parameters[0] + args.cid + 1], args.cid + 1, {"train_time": float(duration)}

    client = PacerNumPyClient(SyntheticClient(), str(args.cid), JsonlSink(args.directory / f"client-{args.cid}.jsonl"))
    fl.client.start_numpy_client(server_address=args.address, client=client)


def orchestrate():
    processes, handles = [], []
    with tempfile.TemporaryDirectory(prefix="pacer-grpc-") as temporary:
        directory = Path(temporary)
        atomic_json(directory / "control.json", control())
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        address = f"127.0.0.1:{port}"
        env = {**os.environ, "FLWR_TELEMETRY_ENABLED": "0", "TF_CPP_MIN_LOG_LEVEL": "3",
               "CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}

        def launch(role, cid=None):
            output = (directory / f"{role}-{cid}.log").open("w")
            handles.append(output)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--role", role,
                   "--directory", str(directory), "--address", address]
            if cid is not None:
                cmd.extend(["--cid", str(cid)])
            proc = subprocess.Popen(cmd, cwd=directory, env=env, stdout=output, stderr=subprocess.STDOUT)
            processes.append(proc)
            return proc

        try:
            server = launch("server")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("Smoke server exited before listening")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("Smoke server did not start within 30 seconds")
            for cid in range(3):
                launch("client", cid)
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline and any(p.poll() is None for p in processes):
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError("A smoke process failed")
                time.sleep(0.1)
            if any(p.poll() != 0 for p in processes):
                raise RuntimeError("Smoke processes failed or timed out")
            rounds = [json.loads(line) for line in (directory / "rounds.jsonl").read_text().splitlines()]
            rounds = [row for row in rounds if row["event"] == "round_observation"]
            assert len(rounds) == 8
            assert [row["config_version"] for row in rounds] == [1] * 4 + [2] * 4
            for idx in (0, 1, 4, 5):
                assert rounds[idx]["state"] == "insufficient_feedback", rounds[idx]
            for idx in (2, 3, 6, 7):
                assert rounds[idx]["n_valid"] == 3 and "p_hat" in rounds[idx], rounds[idx]
            for cid in range(3):
                records = [json.loads(line) for line in (directory / f"client-{cid}.jsonl").read_text().splitlines()]
                feedback = [row for row in records if row["event"] == "client_feedback"]
                assert len(feedback) == 8
                assert [row["pacer.n_samples"] for row in feedback] == [1, 2, 3, 4] * 2
            print("PASS: 3 real gRPC clients, 8 rounds, external target reload at round 5, gamma windows reset, weighted model unchanged")
            print(json.dumps([{"round": r["server_round"], "version": r["config_version"], "state": r["state"]} for r in rounds]))
        except BaseException:
            for handle in handles:
                handle.flush()
            for log in directory.glob("*.log"):
                print(f"{log.name}:\n{log.read_text()[-6000:]}", file=sys.stderr)
            raise
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.terminate()
            for proc in processes:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            for handle in handles:
                handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["server", "client"])
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--address")
    parser.add_argument("--cid", type=int)
    args = parser.parse_args()
    if args.role == "server":
        run_server(args)
    elif args.role == "client":
        run_client(args)
    else:
        orchestrate()
