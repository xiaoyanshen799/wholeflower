import argparse
import pathlib

import flwr as fl
import logging
import numpy as np
import tensorflow as tf

# Force TensorFlow to use a single worker thread so training time reflects
# actual compute instead of thread-pool throttling on constrained devices.
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from client import FlowerClient

# --- add: gRPC keepalive monkey patch (client) ---
import grpc

KA_OPTS = [
    ("grpc.keepalive_time_ms", 20_000),
    ("grpc.keepalive_timeout_ms", 5_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.min_time_between_pings_ms", 20_000),
    ("grpc.http2.max_pings_without_data", 0),
]

_real_ic = grpc.insecure_channel
def _ic(target, options=None, *a, **k):
    opts = list(options) if options else []
    return _real_ic(target, options=opts + KA_OPTS, *a, **k)
grpc.insecure_channel = _ic

#（如果将来用到 TLS，也补上 secure_channel）
_real_sc = grpc.secure_channel
def _sc(target, creds, options=None, *a, **k):
    opts = list(options) if options else []
    return _real_sc(target, creds, options=opts + KA_OPTS, *a, **k)
grpc.secure_channel = _sc
# --- end add ---

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
    parser.add_argument("--server", help="Server address host:port")
    parser.add_argument("--data-dir", default="data_partitions1", help="Directory containing partition files")
    parser.add_argument("--num-classes", type=int, default=10, help="Total number of classes in dataset")
    parser.add_argument(
        "--model",
        default="resnet20",
        choices=["cnn", "tf_example", "resnet20", "mobilenet_v2_075", "mobilenet_v2_100"],
        help="Model architecture",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for local model")
    parser.add_argument(
        "--epochs",
        "--local-epochs",
        type=int,
        default=None,
        help="Local epochs per round (fallback to server config if not set)",
    )
    parser.add_argument(
        "--batch-size",
        "--local-batch-size",
        type=int,
        default=None,
        help="Batch size for local training (fallback to server config if not set)",
    )
    parser.add_argument(
        "--uplink-num-bits",
        type=int,
        choices=[0, 8, 16],
        default=8,
        help="Quantization bit-width for client-to-server payloads (0 disables uplink compression)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Run a single local training pass to measure time (no server connection)",
    )
    parser.add_argument(
        "--local-rounds",
        type=int,
        default=1,
        help="Number of repeated local-only rounds to run when --local-only is set",
    )
    parser.add_argument(
        "--profile-down-mbps",
        type=float,
        default=None,
        help="FedCS profile: average downlink throughput (Mbps) reported via get_properties.",
    )
    parser.add_argument(
        "--profile-up-mbps",
        type=float,
        default=None,
        help="FedCS profile: average uplink throughput (Mbps) reported via get_properties.",
    )
    parser.add_argument(
        "--profile-compute-sps",
        type=float,
        default=None,
        help="FedCS profile: compute capability (samples/sec) reported via get_properties.",
    )

    args = parser.parse_args()
    if not args.local_only and not args.server:
        parser.error("--server is required unless --local-only is set")
    print(f"--- Client {args.cid}: Parsed arguments.")

    # ------------------------------------------------------------------
    # Configure logging to a file per client
    # ------------------------------------------------------------------
    log_path = pathlib.Path(args.data_dir) / f"client_{args.cid:05d}.log"
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print(f"--- Client {args.cid}: Loading partition...")
    data_dir = pathlib.Path(args.data_dir)
    x_full, y_full = _load_partition(data_dir, args.cid)
    print(f"--- Client {args.cid}: Partition loaded. Shape: {x_full.shape}")

    # Normalize to [0,1] to match server-side preprocessing
    if x_full.dtype != np.float32:
        x_full = x_full.astype(np.float32)
    if x_full.max() > 1.0:
        x_full /= 255.0
    if x_full.ndim == 3:
        x_full = np.expand_dims(x_full, axis=-1)

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
        "mobilenet_v2_075": "fedavgm.models.mobilenet_v2_075",
        "mobilenet_v2_100": "fedavgm.models.mobilenet_v2_100",
    }

    model_cfg = {
        "_target_": target_map[args.model],
        "input_shape": input_shape,
        "num_classes": int(num_classes),
        "learning_rate": args.lr,
    }

    print(f"--- Client {args.cid}: Initializing FlowerClient (and TF model)...")
    uplink_bits = args.uplink_num_bits
    # Optional: allow overriding profile values via CLI, otherwise the client will
    # profile itself (compute) and optionally probe bandwidth when requested by server.
    profile = {}
    if args.profile_down_mbps is not None:
        profile["b_down_bps"] = float(args.profile_down_mbps) * 1_000_000.0
    if args.profile_up_mbps is not None:
        profile["b_up_bps"] = float(args.profile_up_mbps) * 1_000_000.0
    if args.profile_compute_sps is not None:
        profile["f_samples_per_s"] = float(args.profile_compute_sps)
    client = FlowerClient(
        x_train,
        y_train,
        x_val,
        y_val,
        model_cfg,
        num_classes,
        local_epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        enable_compression=uplink_bits != 0,
        quantization_bits=uplink_bits if uplink_bits != 0 else 8,
        profile=profile,
    )
    print(f"--- Client {args.cid}: FlowerClient initialized successfully.")

    if args.local_only:
        # Run one local training pass to measure time without contacting server.
        rounds = max(1, args.local_rounds)
        print(f">>> Client {args.cid} running local-only pre-train for {rounds} round(s)…")
        params = client.get_parameters({})
        # Fall back to sensible defaults if overrides are not provided.
        epochs = args.epochs if args.epochs is not None else 1
        batch_size = args.batch_size if args.batch_size is not None else 64
        total_time = 0.0
        for r in range(1, rounds + 1):
            _, _, metrics = client.fit(
                params,
                {"local_epochs": epochs, "batch_size": batch_size},
            )
            params = client.get_parameters({})
            train_time = metrics.get("train_time")
            if isinstance(train_time, (int, float)):
                total_time += float(train_time)
                print(
                    f"[Round {r}] train_time={train_time:.2f}s on {len(x_train)} samples "
                    f"(epochs={epochs}, batch_size={batch_size})"
                )
            else:
                print(f"[Round {r}] training finished.")
        if rounds > 1 and total_time > 0:
            print(f"Total train time over {rounds} round(s): {total_time:.2f}s")
        return

    print(f">>> Client {args.cid} connecting to {args.server}…")
    fl.client.start_numpy_client(server_address=args.server, client=client)
    print(f"--- Client {args.cid}: Flower client finished.")


if __name__ == "__main__":
    main() 
