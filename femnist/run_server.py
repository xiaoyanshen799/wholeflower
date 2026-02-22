import argparse
from datetime import datetime
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from pathlib import Path

os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")

import flwr as fl
from omegaconf import OmegaConf
import tensorflow as tf
from flwr.server.client_manager import SimpleClientManager

from fedavgm.dataset import (
    cifar10,
    femnist_dataset,
    fmnist,
    load_stackoverflow_meta,
    mnist,
    shakespeare,
    speech_commands,
    stackoverflow,
)
from fedavgm.models import (
    cnn,
    model_to_parameters,
    mobilenet_v2_075,
    mobilenet_v2_100,
    resnet18_refl,
    resnet18_keras,
    resnet34_keras,
    resnet20_keras,
    cifar10_resnet,
    tf_example,
    char_lstm2,
    speech_cnn_small,
    embed_avg_mlp,
    embed_bilstm_mlp,
    embed_avg_mlp_bce,
    embed_bilstm_mlp_bce,
)
from fedavgm.server import get_on_fit_config, get_evaluate_fn
from strategy import CsvFedAvg, QuantizedFedAvgM
from refl_strategy import REFLAsyncSiloStrategy
from async_server import TrueAsyncServer
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


def _configure_tf_gpu_memory_growth() -> None:
    """Enable on-demand GPU memory allocation for TensorFlow."""
    if os.environ.get("FORCE_TF_CPU", "0") == "1":
        try:
            tf.config.set_visible_devices([], "GPU")
            logging.info("FORCE_TF_CPU=1, hiding all TensorFlow GPUs")
        except RuntimeError as exc:
            logging.warning("Failed to force CPU mode: %s", exc)
        return

    if os.environ.get("SKIP_TF_GPU_MEMORY_GROWTH", "0") == "1":
        logging.info("Skipping TensorFlow GPU memory-growth probe")
        return

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            logging.warning("Failed to enable memory growth for %s: %s", gpu, exc)


def _resolve_path(candidates):
    """Return first existing Path from candidates."""
    for cand in candidates:
        if cand is None:
            continue
        p = Path(cand)
        if p.exists():
            return p
    return None


def _load_dataset(name: str, *, shakespeare_dir=None, speech_dir=None, stackoverflow_dir=None):
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
    if name in {"shakespeare"}:
        base = _resolve_path(
            [
                shakespeare_dir,
                Path("data_partitions_shakespeare"),
                Path(__file__).resolve().parent / "data_partitions_shakespeare",
            ]
        )
        if base is None:
            raise FileNotFoundError("Could not find Shakespeare partitions (try --data-dir-shakespeare)")
        return shakespeare(num_classes=65536, input_shape=[80], data_dir=base)
    if name in {"speech_commands", "speech"}:
        base = _resolve_path(
            [
                speech_dir,
                Path("data_partitions_speech_commands"),
                Path(__file__).resolve().parent / "data_partitions_speech_commands",
            ]
        )
        if base is None:
            raise FileNotFoundError("Could not find Speech Commands partitions (try --data-dir-speech)")
        return speech_commands(num_classes=12, input_shape=[98, 64, 1], data_dir=base)
    if name in {"stackoverflow", "stack_overflow", "stack-overflow"}:
        if stackoverflow_dir is not None and not Path(stackoverflow_dir).exists():
            raise FileNotFoundError(f"StackOverflow partitions not found: {Path(stackoverflow_dir)}")
        base = _resolve_path(
            [
                stackoverflow_dir,
                Path("data_partitions_stackoverflow"),
                Path(__file__).resolve().parent / "data_partitions_stackoverflow",
            ]
        )
        if base is None:
            raise FileNotFoundError("Could not find StackOverflow partitions (try --data-dir-stackoverflow)")
        return stackoverflow(num_classes=0, input_shape=[0], data_dir=str(base))
    raise ValueError(f"Unsupported dataset '{name}'.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run Flower server for FedAvg/FedAvgM experiments.")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=[
            "cifar10",
            "mnist",
            "fmnist",
            "femnist",
            "shakespeare",
            "speech_commands",
            "stackoverflow",
        ],
        help="cifar10 | mnist | fmnist | femnist | shakespeare | speech_commands | stackoverflow",
    )
    parser.add_argument(
        "--strategy",
        default="fedavg",
        choices=[
            "fedavg",
            "fedavg-csv",
            "fedavgm",
            "custom-fedavgm",
            "fedyogi",
            "fedcs",
            "tifl",
            "refl",
            "refl-async",
        ],
        help="Aggregation strategy",
    )
    parser.add_argument("--rounds", type=int, default=50, help="Total federated rounds")
    parser.add_argument("--clients", type=int, default=3, help="Expected total number of clients")
    parser.add_argument("--reporting-fraction", type=float, default=1.0, help="Fraction of clients selected per round")
    parser.add_argument("--server-lr", type=float, default=0.01, help="Server learning rate (FedAvgM)")
    parser.add_argument(
        "--model",
        default="resnet18_refl",
        choices=[
            "cnn",
            "tf_example",
            "resnet18_refl",
            "resnet18",
            "resnet34",
            "resnet20",
            "cifar10_resnet",
            "mobilenet_v2_075",
            "mobilenet_v2_100",
            "char_lstm2",
            "speech_cnn_small",
            "embed_avg_mlp",
            "embed_bilstm_mlp",
            "embed_avg_mlp_bce",
            "embed_bilstm_mlp_bce",
        ],
        help="Model architecture",
    )
    parser.add_argument("--server-momentum", type=float, default=0.9, help="Server momentum (FedAvgM)")
    parser.add_argument("--local-epochs", type=int, default=1, help="Client local epochs")
    parser.add_argument("--batch-size", type=int, default=10, help="Client batch size")
    parser.add_argument("--client-lr", type=float, default=0.01, help="Client learning rate (to build initial model)")
    parser.add_argument(
        "--client-lr-decay-factor",
        type=float,
        default=0.98,
        help="Per-round client LR decay factor (REFL-style, applied every --client-lr-decay-every rounds).",
    )
    parser.add_argument(
        "--client-lr-decay-every",
        type=int,
        default=10,
        help="Apply client LR decay every N rounds; 0 disables decay.",
    )
    parser.add_argument(
        "--client-lr-min",
        type=float,
        default=5e-5,
        help="Minimum client LR after decay.",
    )
    parser.add_argument("--yogi-eta", type=float, default=0.01, help="FedYogi: server eta")
    parser.add_argument("--yogi-tau", type=float, default=1e-3, help="FedYogi: tau for adaptive denominator")
    parser.add_argument("--yogi-beta1", type=float, default=0.9, help="FedYogi: beta1")
    parser.add_argument("--yogi-beta2", type=float, default=0.99, help="FedYogi: beta2")
    parser.add_argument("--address", default="0.0.0.0:8081", help="Server bind address, e.g. 0.0.0.0:8081")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force TensorFlow server process to run on CPU (hide all GPUs).",
    )
    parser.add_argument("--csv-path", default="logs/comm_times.csv", help="CSV file to log per-round per-client timing metrics")
    parser.add_argument("--data-dir-shakespeare", type=Path, default=None, help="Path to Shakespeare partitions (optional)")
    parser.add_argument("--data-dir-speech", type=Path, default=None, help="Path to Speech Commands partitions (optional)")
    parser.add_argument(
        "--data-dir-stackoverflow",
        type=Path,
        default=None,
        help="Path to StackOverflow partitions (optional)",
    )
    parser.add_argument(
        "--downlink-num-bits",
        type=int,
        choices=[0, 8, 16],
        default=0,
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
    parser.add_argument(
        "--refl-stale-factor",
        type=float,
        default=-4.0,
        help="REFL stale factor mode (>1, 1, -1, -2, -3, -4). -4 matches REFL stale-aware weighting.",
    )
    parser.add_argument(
        "--refl-stale-beta",
        type=float,
        default=0.35,
        help="REFL stale_beta used when stale_factor=-4.",
    )
    parser.add_argument(
        "--refl-scale-coff",
        type=float,
        default=1.0,
        help="REFL scale coefficient for stale_factor=-4.",
    )
    parser.add_argument(
        "--refl-stale-update",
        type=int,
        default=-1,
        help="REFL max stale rounds to accept (-1 means no expiry).",
    )
    parser.add_argument(
        "--refl-disable-full-selection",
        action="store_true",
        help="REFL: disable full client selection and fall back to fraction_fit sampling.",
    )
    parser.add_argument(
        "--refl-server-update",
        choices=["fedavg", "fedavgm"],
        default="fedavg",
        help="REFL: server-side update rule after stale-aware weighting.",
    )
    parser.add_argument(
        "--refl-use-num-examples",
        action="store_true",
        help="REFL: multiply stale-aware importance by num_examples (disabled by default to match REFL).",
    )
    parser.add_argument(
        "--async-round-deadline",
        type=float,
        default=30.0,
        help="REFL-Async: aggregation deadline in seconds; collect arrived updates until this cut.",
    )
    parser.add_argument(
        "--async-min-fraction",
        type=float,
        default=0.8,
        help="REFL-Async: minimum fraction of currently available clients before aggregation.",
    )
    parser.add_argument(
        "--async-min-results",
        type=int,
        default=0,
        help="REFL-Async: minimum arrived updates to aggregate (0 uses async-min-fraction).",
    )
    parser.add_argument(
        "--async-poll-interval",
        type=float,
        default=0.5,
        help="REFL-Async: poll interval in seconds while waiting for new client results.",
    )
    parser.add_argument(
        "--async-max-workers",
        type=int,
        default=64,
        help="REFL-Async: maximum server worker threads for concurrent client RPCs.",
    )

    args = parser.parse_args()
    if args.cpu_only:
        os.environ["FORCE_TF_CPU"] = "1"
    _configure_tf_gpu_memory_growth()

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
    x_train, y_train, x_test, y_test, input_shape, num_classes = _load_dataset(
        args.dataset,
        shakespeare_dir=args.data_dir_shakespeare,
        speech_dir=args.data_dir_speech,
        stackoverflow_dir=args.data_dir_stackoverflow,
    )
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"server_{timestamp}.log"
    with log_file.open("w", encoding="utf-8") as f:
        f.write(
            "dataset={dataset} model={model} rounds={rounds} clients={clients} "
            "local_epochs={local_epochs} batch_size={batch_size} reporting_fraction={reporting_fraction} "
            "server_lr={server_lr} server_momentum={server_momentum} "
            "client_lr={client_lr} client_lr_decay_factor={client_lr_decay_factor} "
            "client_lr_decay_every={client_lr_decay_every} client_lr_min={client_lr_min} "
            "downlink_num_bits={downlink_num_bits} "
            "eval_sample_size={eval_sample_size} refl_server_update={refl_server_update} "
            "refl_use_num_examples={refl_use_num_examples} refl_stale_factor={refl_stale_factor} "
            "refl_stale_beta={refl_stale_beta} refl_scale_coff={refl_scale_coff}\n".format(
                dataset=args.dataset,
                model=args.model,
                rounds=args.rounds,
                clients=args.clients,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                reporting_fraction=args.reporting_fraction,
                server_lr=args.server_lr,
                server_momentum=args.server_momentum,
                client_lr=args.client_lr,
                client_lr_decay_factor=args.client_lr_decay_factor,
                client_lr_decay_every=args.client_lr_decay_every,
                client_lr_min=args.client_lr_min,
                downlink_num_bits=args.downlink_num_bits,
                eval_sample_size=args.eval_sample_size,
                refl_server_update=args.refl_server_update,
                refl_use_num_examples=args.refl_use_num_examples,
                refl_stale_factor=args.refl_stale_factor,
                refl_stale_beta=args.refl_stale_beta,
                refl_scale_coff=args.refl_scale_coff,
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
        "resnet18_refl": resnet18_refl,
        "resnet18": resnet18_keras,
        "resnet34": resnet34_keras,
        "resnet20": resnet20_keras,
        "cifar10_resnet": cifar10_resnet,
        "mobilenet_v2_075": mobilenet_v2_075,
        "mobilenet_v2_100": mobilenet_v2_100,
        "char_lstm2": char_lstm2,
        "speech_cnn_small": speech_cnn_small,
        "embed_avg_mlp": embed_avg_mlp,
        "embed_bilstm_mlp": embed_bilstm_mlp,
        "embed_avg_mlp_bce": embed_avg_mlp_bce,
        "embed_bilstm_mlp_bce": embed_bilstm_mlp_bce,
    }
    build_model = model_builders[args.model]

    if args.dataset.lower() == "stackoverflow":
        is_multilabel = getattr(y_test, "ndim", 1) == 2
        if is_multilabel:
            allowed = {"embed_avg_mlp_bce", "embed_bilstm_mlp_bce"}
            if args.model not in allowed:
                raise SystemExit(
                    "This StackOverflow partition looks multi-label (y_test is 2-D). "
                    "Please use --model embed_avg_mlp_bce or embed_bilstm_mlp_bce."
                )
        else:
            allowed = {"embed_avg_mlp", "embed_bilstm_mlp"}
            if args.model not in allowed:
                raise SystemExit(
                    "This StackOverflow partition looks single-label (y_test is 1-D). "
                    "Please use --model embed_avg_mlp or embed_bilstm_mlp."
                )
        base = _resolve_path(
            [
                args.data_dir_stackoverflow,
                Path("data_partitions_stackoverflow"),
                Path(__file__).resolve().parent / "data_partitions_stackoverflow",
            ]
        )
        if base is None:
            raise FileNotFoundError("Could not find StackOverflow partitions (try --data-dir-stackoverflow)")
        meta = load_stackoverflow_meta(base)
        model = build_model(
            input_shape,
            num_classes,
            args.client_lr,
            vocab_size=int(meta["vocab_size_total"]),
        )
    else:
        if args.model in {
            "embed_avg_mlp",
            "embed_bilstm_mlp",
            "embed_avg_mlp_bce",
            "embed_bilstm_mlp_bce",
        }:
            raise SystemExit(
                "--model embed_* are only supported with --dataset stackoverflow"
            )
        model = build_model(input_shape, num_classes, args.client_lr)
    initial_parameters = model_to_parameters(model)

    # ------------------------------------------------------------------
    # Fit/eval configuration helpers
    # ------------------------------------------------------------------
    cfg = OmegaConf.create(
        {
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "client_lr": args.client_lr,
            "client_lr_decay_factor": args.client_lr_decay_factor,
            "client_lr_decay_every_rounds": args.client_lr_decay_every,
            "client_lr_min": args.client_lr_min,
        }
    )
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
    server_impl = None
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
    elif args.strategy == "fedavg-csv":
        strategy = CsvFedAvg(
            fraction_fit=args.reporting_fraction,
            fraction_evaluate=0.0,
            min_available_clients=args.clients,
            on_fit_config_fn=fit_config_fn,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            csv_log_path=args.csv_path,
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
    elif args.strategy == "fedyogi":
        from flwr.server.strategy import FedYogi

        strategy = FedYogi(
            fraction_fit=args.reporting_fraction,
            fraction_evaluate=0.0,
            min_available_clients=args.clients,
            on_fit_config_fn=fit_config_fn,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            eta=args.yogi_eta,
            tau=args.yogi_tau,
            beta_1=args.yogi_beta1,
            beta_2=args.yogi_beta2,
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
            downlink_quantization_bits=args.downlink_num_bits if args.downlink_num_bits != 0 else 0,
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
            downlink_quantization_bits=args.downlink_num_bits if args.downlink_num_bits != 0 else 0,
            num_tiers=args.tifl_tiers,
            sync_rounds=args.tifl_sync_rounds,
            profile_warmup_rounds=args.tifl_profile_warmup_rounds,
            tmax_s=args.tifl_tmax,
            prob_update_interval=args.tifl_prob_update_interval,
            credits_per_tier=args.tifl_credits,
            reprofiling_interval_rounds=args.tifl_interval,
        )
    elif args.strategy == "refl":
        strategy = REFLAsyncSiloStrategy(
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
            downlink_quantization_bits=args.downlink_num_bits if args.downlink_num_bits != 0 else 0,
            stale_factor=args.refl_stale_factor,
            stale_beta=args.refl_stale_beta,
            scale_coff=args.refl_scale_coff,
            stale_update=args.refl_stale_update,
            refl_server_update=args.refl_server_update,
            use_num_examples=args.refl_use_num_examples,
            full_selection=not args.refl_disable_full_selection,
        )
    elif args.strategy == "refl-async":
        strategy = REFLAsyncSiloStrategy(
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
            downlink_quantization_bits=args.downlink_num_bits if args.downlink_num_bits != 0 else 0,
            stale_factor=args.refl_stale_factor,
            stale_beta=args.refl_stale_beta,
            scale_coff=args.refl_scale_coff,
            stale_update=args.refl_stale_update,
            refl_server_update=args.refl_server_update,
            use_num_examples=args.refl_use_num_examples,
            full_selection=not args.refl_disable_full_selection,
        )
        server_impl = TrueAsyncServer(
            client_manager=SimpleClientManager(),
            strategy=strategy,
            round_deadline_s=args.async_round_deadline,
            min_results=args.async_min_results,
            min_results_fraction=args.async_min_fraction,
            poll_interval_s=args.async_poll_interval,
        )
        server_impl.set_max_workers(args.async_max_workers)
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
            downlink_quantization_bits=args.downlink_num_bits if args.downlink_num_bits != 0 else 0,
        )

    # ------------------------------------------------------------------
    # Start Flower server
    # ------------------------------------------------------------------
    print(f">>> Starting Flower server on {args.address} with strategy {strategy}…")
    if server_impl is None:
        fl.server.start_server(
            server_address=args.address,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
    else:
        fl.server.start_server(
            server_address=args.address,
            server=server_impl,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
        )

if __name__ == "__main__":
    main() 
