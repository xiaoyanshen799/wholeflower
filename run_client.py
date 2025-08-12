import argparse
import pathlib

import numpy as np
import flwr as fl
import logging

from fedavgm.client import FlowerClient
import grpc

keepalive_opts = [
    ('grpc.keepalive_time_ms', 20000),
    ('grpc.keepalive_timeout_ms', 5000),
    ('grpc.keepalive_permit_without_calls', 1),
    ('grpc.http2.min_time_between_pings_ms', 20000),
    ('grpc.http2.min_ping_interval_without_data_ms', 20000),
]



def _load_partition(data_dir: pathlib.Path, cid: int) -> tuple[np.ndarray, np.ndarray]:
    """Load x_train, y_train arrays for the given client id."""
    pad = max(5, len(str(cid)))  # filenames padded with at least 5 digits
    file = data_dir / f"client_{cid:0{pad}d}.npz"
    if not file.exists():
        raise FileNotFoundError(f"Partition file not found: {file}")
    with np.load(file) as npz:
        return npz["x_train"], npz["y_train"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flower client.")
    parser.add_argument("--cid", type=int, required=True, help="Client ID (0-indexed)")
    parser.add_argument("--server", required=True, help="Server address host:port")
    parser.add_argument("--data-dir", default="data_partitions", help="Directory containing partition files")
    parser.add_argument("--num-classes", type=int, default=10, help="Total number of classes in dataset")
    parser.add_argument("--model", default="resnet20", choices=["cnn", "tf_example", "resnet20"], help="Model architecture")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for local model")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round (overrides server config if desired)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for local training")

    args = parser.parse_args()


    # ------------------------------------------------------------------
    # Configure logging to a file per client
    # ------------------------------------------------------------------
    log_path = pathlib.Path(args.data_dir) / f"client_{args.cid:05d}.log"
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    data_dir = pathlib.Path(args.data_dir)
    x_full, y_full = _load_partition(data_dir, args.cid)
    # Normalize to [0,1] to match server-side preprocessing
    if x_full.dtype != np.float32:
        x_full = x_full.astype(np.float32)
    if x_full.max() > 1.0:
        x_full /= 255.0

    # 90-10 split into train / validation (same rule as baseline)
    split_idx = int(0.9 * len(x_full))
    x_train, y_train = x_full[:split_idx], y_full[:split_idx]
    x_val, y_val = x_full[split_idx:], y_full[split_idx:]

    num_classes = args.num_classes
    input_shape = list(x_full.shape[1:])

    # Build Hydra-style config dict for FlowerClient
    target_map = {
        "cnn": "fedavgm.models.cnn",
        "tf_example": "fedavgm.models.tf_example",
        "resnet20": "fedavgm.models.resnet20_keras",
    }

    model_cfg = {
        "_target_": target_map[args.model],
        "input_shape": input_shape,
        "num_classes": int(num_classes),
        "learning_rate": args.lr,
    }

    client = FlowerClient(x_train, y_train, x_val, y_val, model_cfg, num_classes)
    channel = grpc.insecure_channel(args.server, options=keepalive_opts)
    print(f">>> Client {args.cid} connecting to {args.server}…")
    fl.client.start_numpy_client(client=client, grpc_channel=channel)


if __name__ == "__main__":
    main() 