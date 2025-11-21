import argparse
from datetime import datetime
from pathlib import Path

import flwr as fl
from omegaconf import OmegaConf

from fedavgm.dataset import cifar10, fmnist, femnist_dataset
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
    if name in {"fmnist", "fashionmnist", "fashion-mnist"}:
        return fmnist(num_classes=10, input_shape=[28, 28, 1])
    if name in {"femnist"}:
        return femnist_dataset()
    raise ValueError(f"Unsupported dataset '{name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flower server for FedAvg/FedAvgM experiments.")
    parser.add_argument("--dataset", default="cifar10", help="cifar10 | fmnist")
    parser.add_argument("--strategy", default="custom-fedavgm", choices=["fedavg", "fedavgm", "custom-fedavgm"], help="Aggregation strategy")
    parser.add_argument("--rounds", type=int, default=50, help="Total federated rounds")
    parser.add_argument("--clients", type=int, default=10, help="Expected total number of clients")
    parser.add_argument("--reporting-fraction", type=float, default=0.1, help="Fraction of clients selected per round")
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

    args = parser.parse_args()

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
            "server_lr={server_lr} server_momentum={server_momentum} downlink_num_bits={downlink_num_bits}\n".format(
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
            )
        )
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
    evaluate_fn = get_evaluate_fn(model, x_test, y_test, args.rounds, num_classes, log_path=str(log_file))

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
