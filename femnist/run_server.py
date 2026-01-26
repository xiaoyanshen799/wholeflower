import argparse
from datetime import datetime
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from pathlib import Path

import flwr as fl
from omegaconf import OmegaConf

from fedavgm.dataset import cifar10, fmnist, mnist, femnist_dataset
from fedavgm.models import (
    cnn,
    model_to_parameters,
    mobilenet_v2_075,
    mobilenet_v2_100,
    resnet20_keras,
    tf_example,
)
from fedavgm.server import get_on_fit_config, get_evaluate_fn
from strategy import QuantizedFedAvgM
from baseline_selection import FedCSStrategy, TiFLStrategy
import grpc
from concurrent import futures
max_message_length: int = 536_870_912
SRV_OPTS = [
    ("grpc.keepalive_time_ms", 20_000),
    ("grpc.keepalive_timeout_ms", 5_000),
    ("grpc.max_concurrent_streams", 1024),
    ("grpc.max_send_message_length", max_message_length),
    ("grpc.max_receive_message_length", max_message_length),
    ("grpc.http2.min_ping_interval_without_data_ms", 20_000),
    ("grpc.http2.max_pings_without_data", 0),
]

_real_srv = grpc.server
def _srv(executor=None, options=None, *a, **k):
    if executor is None:
        executor = futures.ThreadPoolExecutor(max_workers=64)
    opts = list(options) if options else []
    return _real_srv(executor, options=opts + SRV_OPTS, *a, **k)

grpc.server = _srv


def _load_dataset(name: str):
    """Return (x_train, y_train, x_test, y_test, input_shape, num_classes)."""
    name = name.lower()
    if name in {"cifar", "cifar10", "cifar-10"}:
        return cifar10(num_classes=10, input_shape=[32, 32, 3])
    if name in {"mnist"}:
        return mnist(num_classes=10, input_shape=[28, 28, 1])
    if name in {"fmnist", "fashionmnist", "fashion-mnist"}:
        return fmnist(num_classes=10, input_shape=[28, 28, 1])
    if name in {"femnist"}:
        return femnist_dataset()
    raise ValueError(f"Unsupported dataset '{name}'.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run Flower server for FedAvg/FedAvgM experiments.")
    parser.add_argument("--dataset", default="cifar10", help="cifar10 | mnist | fmnist | femnist")
    parser.add_argument(
        "--strategy",
        default="custom-fedavgm",
        choices=["fedavg", "fedavgm", "custom-fedavgm", "fedcs", "tifl"],
        help="Aggregation strategy",
    )
    parser.add_argument("--rounds", type=int, default=50, help="Total federated rounds")
    parser.add_argument("--clients", type=int, default=10, help="Expected total number of clients")
    parser.add_argument("--reporting-fraction", type=float, default=1.0, help="Fraction of clients selected per round")
    parser.add_argument("--server-lr", type=float, default=0.05, help="Server learning rate (FedAvgM)")
    parser.add_argument(
        "--model",
        default="resnet20",
        choices=["cnn", "tf_example", "resnet20", "mobilenet_v2_075", "mobilenet_v2_100"],
        help="Model architecture",
    )
    parser.add_argument("--server-momentum", type=float, default=0.9, help="Server momentum (FedAvgM)")
    parser.add_argument("--local-epochs", type=int, default=1, help="Client local epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Client batch size")
    parser.add_argument("--client-lr", type=float, default=0.05, help="Client learning rate (to build initial model)")
    parser.add_argument("--address", default="0.0.0.0:8081", help="Server bind address, e.g. 0.0.0.0:8081")
    parser.add_argument("--csv-path", default="logs/comm_times.csv", help="CSV file to log per-round per-client timing metrics")
    parser.add_argument(
        "--downlink-num-bits",
        type=int,
        choices=[0, 8, 16],
        default=8,
        help="Quantization bit-width for server-to-client payloads (0 disables downlink compression)",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=0,
        help="If >0, evaluate server model on a random subset of this many test samples per round",
    )
    parser.add_argument(
        "--eval-sample-seed",
        type=int,
        default=None,
        help="Random seed used when sampling evaluation examples (ignored when --eval-sample-size=0)",
    )
    parser.add_argument(
        "--round-deadline",
        type=float,
        default=180.0,
        help="FedCS: per-round deadline T_round in seconds.",
    )
    parser.add_argument(
        "--fedcs-default-train",
        "--fedcs-default-num-samples",
        type=int,
        default=300,
        help="FedCS: default local sample count when no data-dir is provided.",
    )
    parser.add_argument(
        "--profile-http-port",
        type=int,
        default=0,
        help="If set (>0), start an HTTP endpoint for clients to probe bandwidth.",
    )
    parser.add_argument(
        "--profile-http-host",
        default="127.0.0.1",
        help="Host used to construct profile URL passed to clients (e.g., 127.0.0.1).",
    )
    parser.add_argument(
        "--profile-bytes",
        type=int,
        default=262144,
        help="Bytes transferred in download/upload probes when profiling bandwidth.",
    )
    parser.add_argument(
        "--fedcs-properties-timeout",
        type=float,
        default=60.0,
        help="FedCS: timeout (seconds) for per-client get_properties during Resource Request.",
    )
    parser.add_argument(
        "--tifl-tiers",
        type=int,
        default=4,
        help="TiFL: number of latency tiers to build.",
    )
    parser.add_argument(
        "--tifl-interval",
        type=int,
        default=20,
        help="TiFL: periodic re-profiling interval (rounds); 0 disables.",
    )
    parser.add_argument(
        "--tifl-fast-bias",
        type=float,
        default=0.6,
        help="TiFL (legacy, ignored): fast-tier bias from the old heuristic implementation.",
    )
    parser.add_argument(
        "--tifl-sync-rounds",
        type=int,
        default=5,
        help="TiFL: number of profiling rounds used for tiering (sync_rounds).",
    )
    parser.add_argument(
        "--tifl-profile-warmup-rounds",
        type=int,
        default=2,
        help="TiFL: warmup profiling rounds to ignore (e.g., first-round overhead).",
    )
    parser.add_argument(
        "--tifl-tmax",
        type=float,
        default=60.0,
        help="TiFL: Tmax seconds for profiling timeout/capping.",
    )
    parser.add_argument(
        "--tifl-prob-update-interval",
        type=int,
        default=20,
        help="TiFL: probability update interval I in Algorithm 2.",
    )
    parser.add_argument(
        "--tifl-credits",
        type=int,
        default=None,
        help="TiFL: Credits per tier (Algorithm 2); omit for unlimited.",
    )

    args = parser.parse_args()

    profile_url = None
    if args.profile_http_port and args.profile_http_port > 0:
        profile_url = f"http://{args.profile_http_host}:{args.profile_http_port}"

        class _ProfileHandler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != "/download":
                    self.send_response(404)
                    self.end_headers()
                    return
                qs = parse_qs(parsed.query)
                n = int(qs.get("bytes", ["0"])[0] or "0")
                n = max(0, min(n, 5 * 1024 * 1024))
                payload = b"\0" * n
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):  # noqa: N802
                if self.path != "/upload":
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers.get("Content-Length", "0") or "0")
                _ = self.rfile.read(length) if length > 0 else b""
                self.send_response(204)
                self.end_headers()

            def log_message(self, fmt, *args2):  # silence default http.server logging
                return

        httpd = HTTPServer(("0.0.0.0", int(args.profile_http_port)), _ProfileHandler)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        logging.info("[Profile] HTTP probe server started at %s", profile_url)

    # ------------------------------------------------------------------
    # Build initial model to obtain parameter shapes
    # ------------------------------------------------------------------
    x_train, y_train, x_test, y_test, input_shape, num_classes = _load_dataset(args.dataset)
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"server_{timestamp}.log"
    with log_file.open("w", encoding="utf-8") as f:
        f.write(
            "dataset={dataset} model={model} rounds={rounds} clients={clients} "
            "local_epochs={local_epochs} batch_size={batch_size} reporting_fraction={reporting_fraction} "
            "server_lr={server_lr} server_momentum={server_momentum} downlink_num_bits={downlink_num_bits} "
            "eval_sample_size={eval_sample_size}\n".format(
                dataset=args.dataset,
                model=args.model,
                rounds=args.rounds,
                clients=args.clients,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                reporting_fraction=args.reporting_fraction,
                server_lr=args.server_lr,
                server_momentum=args.server_momentum,
                downlink_num_bits=args.downlink_num_bits,
                eval_sample_size=args.eval_sample_size,
            )
        )
    # Also mirror Python logging output into the per-run server log file.
    file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    model_builders = {
        "cnn": cnn,
        "tf_example": tf_example,
        "resnet20": resnet20_keras,
        "mobilenet_v2_075": mobilenet_v2_075,
        "mobilenet_v2_100": mobilenet_v2_100,
    }
    build_model = model_builders[args.model]

    model = build_model(input_shape, num_classes, args.client_lr)
    initial_parameters = model_to_parameters(model)

    # ------------------------------------------------------------------
    # Fit/eval configuration helpers
    # ------------------------------------------------------------------
    cfg = OmegaConf.create({"local_epochs": args.local_epochs, "batch_size": args.batch_size})
    fit_config_fn = get_on_fit_config(cfg)
    sample_size = args.eval_sample_size if args.eval_sample_size > 0 else None
    evaluate_fn = get_evaluate_fn(
        model,
        x_test,
        y_test,
        args.rounds,
        num_classes,
        log_path=str(log_file),
        sample_size=sample_size,
        sample_seed=args.eval_sample_seed,
    )

    # ------------------------------------------------------------------
    # Select strategy
    # ------------------------------------------------------------------
    if args.strategy == "fedavg":
        from flwr.server.strategy import FedAvg

        strategy = FedAvg(
            fraction_fit=args.reporting_fraction,
            fraction_evaluate=0.0,
            min_available_clients=args.clients,
            on_fit_config_fn=fit_config_fn,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
        )
    elif args.strategy == "fedavgm":
        from flwr.server.strategy import FedAvgM

        strategy = FedAvgM(
            fraction_fit=args.reporting_fraction,
            fraction_evaluate=0.0,
            min_available_clients=args.clients,
            on_fit_config_fn=fit_config_fn,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            server_learning_rate=args.server_lr,
            server_momentum=args.server_momentum,
        )
    elif args.strategy == "fedcs":
        strategy = FedCSStrategy(
            fraction_fit=args.reporting_fraction,
            fraction_evaluate=0.0,
            min_available_clients=args.clients,
            on_fit_config_fn=fit_config_fn,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            server_learning_rate=args.server_lr,
            server_momentum=args.server_momentum,
            csv_log_path=args.csv_path,
            downlink_quantization_enabled=args.downlink_num_bits != 0,
            downlink_quantization_bits=args.downlink_num_bits if args.downlink_num_bits != 0 else 8,
            round_deadline_s=args.round_deadline,
            resource_request_fraction=args.reporting_fraction,
            profile_url=profile_url,
            profile_bytes=args.profile_bytes,
            properties_timeout_s=args.fedcs_properties_timeout,
        )
    elif args.strategy == "tifl":
        strategy = TiFLStrategy(
            fraction_fit=args.reporting_fraction,
            fraction_evaluate=0.0,
            min_available_clients=args.clients,
            on_fit_config_fn=fit_config_fn,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            server_learning_rate=args.server_lr,
            server_momentum=args.server_momentum,
            csv_log_path=args.csv_path,
            downlink_quantization_enabled=args.downlink_num_bits != 0,
            downlink_quantization_bits=args.downlink_num_bits if args.downlink_num_bits != 0 else 8,
            num_tiers=args.tifl_tiers,
            sync_rounds=args.tifl_sync_rounds,
            profile_warmup_rounds=args.tifl_profile_warmup_rounds,
            tmax_s=args.tifl_tmax,
            prob_update_interval=args.tifl_prob_update_interval,
            credits_per_tier=args.tifl_credits,
            reprofiling_interval_rounds=args.tifl_interval,
        )
    else:  # custom-fedavgm (default)
        strategy = QuantizedFedAvgM(
            fraction_fit=args.reporting_fraction,
            fraction_evaluate=0.0,
            min_available_clients=args.clients,
            on_fit_config_fn=fit_config_fn,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            server_learning_rate=args.server_lr,
            server_momentum=args.server_momentum,
            csv_log_path=args.csv_path,
            downlink_quantization_enabled=args.downlink_num_bits != 0,
            downlink_quantization_bits=args.downlink_num_bits if args.downlink_num_bits != 0 else 8,
        )

    # ------------------------------------------------------------------
    # Start Flower server
    # ------------------------------------------------------------------
    print(f">>> Starting Flower server on {args.address} with strategy {strategy}…")
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main() 
