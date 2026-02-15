import argparse
from pathlib import Path

import flwr as fl
from omegaconf import OmegaConf

from fedavgm.dataset import cifar10, fmnist, load_stackoverflow_meta, stackoverflow
from fedavgm.models import cnn, model_to_parameters
from fedavgm.server import get_on_fit_config, get_evaluate_fn
from fedavgm.strategy import CustomFedAvgM
import grpc
from concurrent import futures

SRV_OPTS = [
    ("grpc.keepalive_time_ms", 20_000),
    ("grpc.keepalive_timeout_ms", 5_000),
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


def _resolve_path(candidates):
    """Return first existing Path from candidates."""
    for cand in candidates:
        if cand is None:
            continue
        p = Path(cand)
        if p.exists():
            return p
    return None


def _load_dataset(name: str, *, stackoverflow_dir=None):
    """Return (x_train, y_train, x_test, y_test, input_shape, num_classes)."""
    name = name.lower()
    if name in {"cifar", "cifar10", "cifar-10"}:
        return cifar10(num_classes=10, input_shape=[32, 32, 3])
    if name in {"fmnist", "fashionmnist", "fashion-mnist"}:
        return fmnist(num_classes=10, input_shape=[28, 28])
    if name in {"stackoverflow", "stack_overflow", "stack-overflow"}:
        if stackoverflow_dir is not None and not Path(stackoverflow_dir).exists():
            raise FileNotFoundError(f"StackOverflow partitions not found: {Path(stackoverflow_dir)}")
        base = _resolve_path(
            [
                stackoverflow_dir,
                Path("data_partitions_stackoverflow"),
                Path("femnist") / "data_partitions_stackoverflow",
            ]
        )
        if base is None:
            raise FileNotFoundError("Could not find StackOverflow partitions (try --data-dir-stackoverflow)")
        return stackoverflow(num_classes=0, input_shape=[0], data_dir=str(base))
    raise ValueError(f"Unsupported dataset '{name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flower server for FedAvg/FedAvgM experiments.")
    parser.add_argument("--dataset", default="cifar10", help="cifar10 | fmnist | stackoverflow")
    parser.add_argument("--strategy", default="custom-fedavgm", choices=["fedavg", "fedavgm", "custom-fedavgm"], help="Aggregation strategy")
    parser.add_argument("--rounds", type=int, default=50, help="Total federated rounds")
    parser.add_argument("--clients", type=int, default=10, help="Expected total number of clients")
    parser.add_argument("--reporting-fraction", type=float, default=0.1, help="Fraction of clients selected per round")
    parser.add_argument("--server-lr", type=float, default=1.0, help="Server learning rate (FedAvgM)")
    parser.add_argument(
        "--model",
        default="resnet20",
        choices=[
            "cnn",
            "tf_example",
            "resnet20",
            "embed_avg_mlp",
            "embed_bilstm_mlp",
            "embed_avg_mlp_bce",
            "embed_bilstm_mlp_bce",
        ],
        help="Model architecture",
    )
    parser.add_argument("--server-momentum", type=float, default=0.9, help="Server momentum (FedAvgM)")
    parser.add_argument("--local-epochs", type=int, default=1, help="Client local epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Client batch size")
    parser.add_argument("--client-lr", type=float, default=0.01, help="Client learning rate (to build initial model)")
    parser.add_argument("--address", default="0.0.0.0:8081", help="Server bind address, e.g. 0.0.0.0:8081")
    parser.add_argument("--csv-path", default="logs/comm_times.csv", help="CSV file to log per-round per-client timing metrics")
    parser.add_argument("--data-dir-stackoverflow", type=Path, default=None, help="Path to StackOverflow partitions (optional)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build initial model to obtain parameter shapes
    # ------------------------------------------------------------------
    x_train, y_train, x_test, y_test, input_shape, num_classes = _load_dataset(
        args.dataset, stackoverflow_dir=args.data_dir_stackoverflow
    )
    if args.dataset.lower() == "stackoverflow":
        is_multilabel = getattr(y_test, "ndim", 1) == 2
        if is_multilabel and args.model not in {"embed_avg_mlp_bce", "embed_bilstm_mlp_bce"}:
            raise SystemExit(
                "This StackOverflow partition looks multi-label (y_test is 2-D). "
                "Please use --model embed_avg_mlp_bce or embed_bilstm_mlp_bce."
            )
        if (not is_multilabel) and args.model not in {"embed_avg_mlp", "embed_bilstm_mlp"}:
            raise SystemExit(
                "This StackOverflow partition looks single-label (y_test is 1-D). "
                "Please use --model embed_avg_mlp or embed_bilstm_mlp."
            )
        if args.model == "embed_avg_mlp":
            from fedavgm.models import embed_avg_mlp as build_model
        elif args.model == "embed_bilstm_mlp":
            from fedavgm.models import embed_bilstm_mlp as build_model
        elif args.model == "embed_avg_mlp_bce":
            from fedavgm.models import embed_avg_mlp_bce as build_model
        else:
            from fedavgm.models import embed_bilstm_mlp_bce as build_model

        base = _resolve_path(
            [
                args.data_dir_stackoverflow,
                Path("data_partitions_stackoverflow"),
                Path("femnist") / "data_partitions_stackoverflow",
            ]
        )
        if base is None:
            raise FileNotFoundError("Could not find StackOverflow partitions (try --data-dir-stackoverflow)")
        meta = load_stackoverflow_meta(base)
        model = build_model(
            input_shape, num_classes, args.client_lr, vocab_size=int(meta["vocab_size_total"])
        )
    else:
        if args.model.startswith("embed_"):
            raise SystemExit("--model embed_* are only supported with --dataset stackoverflow")
        if args.model == "cnn":
            from fedavgm.models import cnn as build_model
        elif args.model == "tf_example":
            from fedavgm.models import tf_example as build_model
        else:
            from fedavgm.models import resnet20_keras as build_model
        model = build_model(input_shape, num_classes, args.client_lr)
    initial_parameters = model_to_parameters(model)

    # ------------------------------------------------------------------
    # Fit/eval configuration helpers
    # ------------------------------------------------------------------
    cfg = OmegaConf.create({"local_epochs": args.local_epochs, "batch_size": args.batch_size})
    fit_config_fn = get_on_fit_config(cfg)
    evaluate_fn = get_evaluate_fn(model, x_test, y_test, args.rounds, num_classes)

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
    else:  # custom-fedavgm (default)
        strategy = CustomFedAvgM(
            fraction_fit=args.reporting_fraction,
            fraction_evaluate=0.0,
            min_available_clients=args.clients,
            on_fit_config_fn=fit_config_fn,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            server_learning_rate=args.server_lr,
            server_momentum=args.server_momentum,
            csv_log_path=args.csv_path,
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
