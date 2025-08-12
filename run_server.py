import argparse

import flwr as fl
from omegaconf import OmegaConf

from fedavgm.dataset import cifar10, fmnist
from fedavgm.models import cnn, model_to_parameters
from fedavgm.server import get_on_fit_config, get_evaluate_fn
from fedavgm.strategy import CustomFedAvgM
import os

def configure_grpc_keepalive():
    os.environ["GRPC_KEEPALIVE_TIME_MS"] = "20000"  # 20s：空闲多久发一次 ping
    os.environ["GRPC_KEEPALIVE_TIMEOUT_MS"] = "5000"  # 5s：ping 超时阈值
    os.environ["GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS"] = "1"  # 无活动流也允许 keepalive
    # HTTP/2 层最小 ping 间隔（收/发）
    os.environ["GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS"] = "20000"
    os.environ["GRPC_ARG_HTTP2_MIN_SENT_PING_INTERVAL_WITHOUT_DATA_MS"] = "20000"


def _load_dataset(name: str):
    """Return (x_train, y_train, x_test, y_test, input_shape, num_classes)."""
    name = name.lower()
    if name in {"cifar", "cifar10", "cifar-10"}:
        return cifar10(num_classes=10, input_shape=[32, 32, 3])
    if name in {"fmnist", "fashionmnist", "fashion-mnist"}:
        return fmnist(num_classes=10, input_shape=[28, 28])
    raise ValueError(f"Unsupported dataset '{name}'.")


def main() -> None:
    # Configure gRPC keepalive before creating any channels
    configure_grpc_keepalive()
    parser = argparse.ArgumentParser(description="Run Flower server for FedAvg/FedAvgM experiments.")
    parser.add_argument("--dataset", default="cifar10", help="cifar10 | fmnist")
    parser.add_argument("--strategy", default="custom-fedavgm", choices=["fedavg", "fedavgm", "custom-fedavgm"], help="Aggregation strategy")
    parser.add_argument("--rounds", type=int, default=50, help="Total federated rounds")
    parser.add_argument("--clients", type=int, default=10, help="Expected total number of clients")
    parser.add_argument("--reporting-fraction", type=float, default=0.1, help="Fraction of clients selected per round")
    parser.add_argument("--server-lr", type=float, default=1.0, help="Server learning rate (FedAvgM)")
    parser.add_argument("--model", default="resnet20", choices=["cnn", "tf_example", "resnet20"], help="Model architecture")
    parser.add_argument("--server-momentum", type=float, default=0.9, help="Server momentum (FedAvgM)")
    parser.add_argument("--local-epochs", type=int, default=1, help="Client local epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Client batch size")
    parser.add_argument("--client-lr", type=float, default=0.01, help="Client learning rate (to build initial model)")
    parser.add_argument("--address", default="0.0.0.0:8081", help="Server bind address, e.g. 0.0.0.0:8081")
    parser.add_argument("--csv-path", default="logs/comm_times.csv", help="CSV file to log per-round per-client timing metrics")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build initial model to obtain parameter shapes
    # ------------------------------------------------------------------
    x_train, y_train, x_test, y_test, input_shape, num_classes = _load_dataset(args.dataset)
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