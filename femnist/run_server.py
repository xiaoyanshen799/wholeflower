import argparse
import sys
from datetime import datetime
from pathlib import Path

import flwr as fl

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
    from fedavgm.dataset import cifar10, fmnist, femnist_dataset

    name = name.lower()
    if name in {"cifar", "cifar10", "cifar-10"}:
        return cifar10(num_classes=10, input_shape=[32, 32, 3])
    if name in {"fmnist", "fashionmnist", "fashion-mnist"}:
        return fmnist(num_classes=10, input_shape=[28, 28, 1])
    if name in {"femnist"}:
        return femnist_dataset()
    raise ValueError(f"Unsupported dataset '{name}'.")


def _is_ixi(name: str) -> bool:
    return name.lower() in {"ixi", "fed-ixi", "fed_ixi", "fedixi"}


def _is_isic(name: str) -> bool:
    return name.lower() in {"isic", "isic2019", "fed-isic2019", "fed_isic2019"}


def _is_cifar100(name: str) -> bool:
    return name.lower() in {"cifar100", "fed-cifar100", "fed_cifar100", "cifar100-resnet"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flower server for FedAvg/FedAvgM experiments.")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        help="cifar10 | fmnist | femnist | ixi | isic | cifar100",
    )
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
    parser.add_argument("--data-dir", default="data_partitions_cifar100", help="Partition directory for CIFAR-100 NPZ files")
    parser.add_argument("--data-root", default="data/fed_ixi", help="Fed-IXI dataset root (IXI_sample folder inside)")
    parser.add_argument("--isic-preprocessed-dir", default="ISIC_2019_Training_Input_preprocessed", help="ISIC preprocessed image folder (relative to --data-root)")
    parser.add_argument("--isic-split-csv", default=None, help="Path to ISIC train_test_split file")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights for vision backbones",
    )
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

    args = parser.parse_args()

    if _is_ixi(args.dataset) and "--batch-size" not in sys.argv:
        args.batch_size = 2
    if _is_isic(args.dataset) and "--batch-size" not in sys.argv:
        args.batch_size = 32
    if _is_cifar100(args.dataset) and "--batch-size" not in sys.argv:
        args.batch_size = 32

    # ------------------------------------------------------------------
    # Build initial model to obtain parameter shapes
    # ------------------------------------------------------------------
    use_ixi = _is_ixi(args.dataset)
    use_isic = _is_isic(args.dataset)
    use_cifar100 = _is_cifar100(args.dataset)
    if use_ixi:
        import torch
        from flwr.common import ndarrays_to_parameters

        from ixi_flower import evaluate_ixi, get_model_parameters
        from ixi_model import UNet3D

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet3D(in_channels=1, out_channels=2, base=8).to(device)
        initial_parameters = ndarrays_to_parameters(get_model_parameters(model))
        input_shape = (1, 48, 60, 48)
        num_classes = 2
        x_train = y_train = x_test = y_test = None
    elif use_isic:
        import torch
        from flwr.common import ndarrays_to_parameters

        from isic_flower import evaluate_isic, get_model_parameters
        from isic_model import build_efficientnet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_efficientnet(num_classes=8, pretrained=True).to(device)
        initial_parameters = ndarrays_to_parameters(get_model_parameters(model))
        input_shape = (3, 200, 200)
        num_classes = 8
        x_train = y_train = x_test = y_test = None
    elif use_cifar100:
        import torch
        from flwr.common import ndarrays_to_parameters

        from cifar100_flower import evaluate_cifar100, get_model_parameters
        from cifar100_model import build_resnet152

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_resnet152(num_classes=100, pretrained=not args.no_pretrained).to(device)
        initial_parameters = ndarrays_to_parameters(get_model_parameters(model))
        input_shape = (3, 224, 224)
        num_classes = 100
        x_train = y_train = x_test = y_test = None
    else:
        x_train, y_train, x_test, y_test, input_shape, num_classes = _load_dataset(args.dataset)
        model = None
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"server_{timestamp}.log"
    with log_file.open("w", encoding="utf-8") as f:
        model_name = "resnet152_imagenet224" if use_cifar100 else args.model
        f.write(
            "dataset={dataset} model={model} rounds={rounds} clients={clients} "
            "local_epochs={local_epochs} batch_size={batch_size} reporting_fraction={reporting_fraction} "
            "server_lr={server_lr} server_momentum={server_momentum} downlink_num_bits={downlink_num_bits} "
            "eval_sample_size={eval_sample_size}\n".format(
                dataset=args.dataset,
                model=model_name,
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
    if not use_ixi and not use_isic and not use_cifar100:
        from fedavgm.models import (
            cnn,
            model_to_parameters,
            mobilenet_v2_075,
            mobilenet_v2_100,
            resnet20_keras,
            tf_example,
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
    if use_ixi or use_isic or use_cifar100:
        import time

        def fit_config_fn(server_round: int):  # pylint: disable=unused-argument
            return {
                "local_epochs": args.local_epochs,
                "batch_size": args.batch_size,
                "confit_time": time.time(),
                "server_send_time": time.time(),
            }
    else:
        from omegaconf import OmegaConf
        from fedavgm.server import get_on_fit_config, get_evaluate_fn

        cfg = OmegaConf.create({"local_epochs": args.local_epochs, "batch_size": args.batch_size})
        fit_config_fn = get_on_fit_config(cfg)
    sample_size = args.eval_sample_size if args.eval_sample_size > 0 else None
    if use_ixi:
        from flwr.common import parameters_to_ndarrays

        def evaluate_fn(server_round: int, parameters, config):  # pylint: disable=unused-argument
            if hasattr(parameters, "tensors"):
                params_list = parameters_to_ndarrays(parameters)
            else:
                params_list = parameters
            loss, dice = evaluate_ixi(
                model,
                params_list,
                data_root=args.data_root,
                device=device,
                batch_size=1,
                num_workers=0,
                sample_size=sample_size,
                sample_seed=args.eval_sample_seed,
            )
            if log_file:
                entry = (
                    f"round={server_round} loss={loss:.6f} dice={dice:.6f} "
                    f"samples={sample_size or 'all'}\n"
                )
                Path(log_file).open("a", encoding="utf-8").write(entry)
            return loss, {"dice": dice}
    elif use_isic:
        from flwr.common import parameters_to_ndarrays

        def evaluate_fn(server_round: int, parameters, config):  # pylint: disable=unused-argument
            if hasattr(parameters, "tensors"):
                params_list = parameters_to_ndarrays(parameters)
            else:
                params_list = parameters
            loss, bal_acc = evaluate_isic(
                model,
                params_list,
                data_root=args.data_root,
                device=device,
                split_csv=args.isic_split_csv,
                preprocessed_dir=args.isic_preprocessed_dir,
                batch_size=args.batch_size,
                num_workers=0,
                sample_size=sample_size,
                sample_seed=args.eval_sample_seed,
            )
            if log_file:
                entry = (
                    f"round={server_round} loss={loss:.6f} balanced_accuracy={bal_acc:.6f} "
                    f"samples={sample_size or 'all'}\n"
                )
                Path(log_file).open("a", encoding="utf-8").write(entry)
            return loss, {"balanced_accuracy": bal_acc}
    elif use_cifar100:
        from flwr.common import parameters_to_ndarrays

        def evaluate_fn(server_round: int, parameters, config):  # pylint: disable=unused-argument
            if hasattr(parameters, "tensors"):
                params_list = parameters_to_ndarrays(parameters)
            else:
                params_list = parameters
            loss, accuracy = evaluate_cifar100(
                model,
                params_list,
                data_dir=args.data_dir,
                device=device,
                batch_size=args.batch_size,
                num_workers=0,
                sample_size=sample_size,
                sample_seed=args.eval_sample_seed,
            )
            if log_file:
                entry = (
                    f"round={server_round} loss={loss:.6f} accuracy={accuracy:.6f} "
                    f"samples={sample_size or 'all'}\n"
                )
                Path(log_file).open("a", encoding="utf-8").write(entry)
            return loss, {"accuracy": accuracy}
    else:
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
