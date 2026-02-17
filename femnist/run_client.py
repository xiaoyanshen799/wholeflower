import argparse
import pathlib
import os

os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")

import flwr as fl
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path

# Force TensorFlow to use a single worker thread so training time reflects
# actual compute instead of thread-pool throttling on constrained devices.
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def _configure_tf_gpu_memory_growth() -> None:
    """Enable on-demand GPU memory allocation for TensorFlow."""
    # On some container/runtime stacks (e.g. K3s + HAMi), probing GPU devices
    # can block indefinitely. Allow skipping this step via env var.
    if os.environ.get("SKIP_TF_GPU_MEMORY_GROWTH", "0") == "1":
        logging.info("Skipping TensorFlow GPU memory-growth probe")
        return
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            logging.warning("Failed to enable memory growth for %s: %s", gpu, exc)

from client import FlowerClient
from fedavgm.dataset import load_stackoverflow_meta, remap_shakespeare, remap_stackoverflow_labels

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
    with np.load(file, allow_pickle=True) as npz:
        # Speech Commands partitions store 'files'/'labels'
        if "files" in npz and "labels" in npz:
            return npz["files"], npz["labels"]
        return npz["x_train"], npz["y_train"]


def _load_label_map(data_dir: Path) -> list[str]:
    path = data_dir / "label_map.txt"
    if not path.exists():
        raise FileNotFoundError(f"Label map not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _speech_log_mel(path: tf.Tensor, label: tf.Tensor, sample_rate: int = 16000, mel_bins: int = 64):
    audio = tf.io.read_file(path)
    wav, _ = tf.audio.decode_wav(audio, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    wav = tf.cast(wav, tf.float32)
    desired = sample_rate
    wav_len = tf.shape(wav)[0]
    wav = tf.cond(
        wav_len < desired,
        lambda: tf.pad(wav, [[0, desired - wav_len]]),
        lambda: wav[:desired],
    )
    stft = tf.signal.stft(wav, frame_length=256, frame_step=128, fft_length=256)
    spec = tf.abs(stft)
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=mel_bins,
        num_spectrogram_bins=spec.shape[-1],
        sample_rate=sample_rate,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )
    mel = tf.tensordot(tf.square(spec), mel_matrix, 1)
    log_mel = tf.math.log(mel + 1e-6)
    log_mel = tf.expand_dims(log_mel, -1)
    return log_mel, label


def _build_speech_ds(files, labels, data_root: Path, batch_size: int, training: bool) -> tf.data.Dataset:
    file_paths = tf.constant([str(data_root / Path(f)) for f in files])
    labels = tf.constant(labels, dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if training:
        ds = ds.shuffle(len(files))
    ds = ds.map(_speech_log_mel, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> None:
    _configure_tf_gpu_memory_growth()
    parser = argparse.ArgumentParser(description="Run Flower client.")
    parser.add_argument("--cid", type=int, required=True, help="Client ID (0-indexed)")
    parser.add_argument("--server", help="Server address host:port")
    parser.add_argument("--data-dir", default="data_partitions1", help="Directory containing partition files")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "mnist", "fmnist", "femnist", "shakespeare", "speech_commands", "stackoverflow"],
        help="Dataset name (controls preprocessing/model defaults)",
    )
    parser.add_argument("--num-classes", type=int, default=10, help="Total number of classes in dataset")
    parser.add_argument(
        "--model",
        default="resnet20",
        choices=[
            "cnn",
            "tf_example",
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
    print(f"--- Client {args.cid}: Partition loaded. Shape: {getattr(x_full, 'shape', 'n/a')}")

    dataset_name = args.dataset.lower()
    train_ds_fn = None
    val_ds_fn = None
    train_size = None
    val_size = None
    use_sparse_labels = None

    if dataset_name == "shakespeare":
        # Keep integer codepoints; remap to contiguous ids for Embedding
        x_full, vocab_size = remap_shakespeare(x_full, str(data_dir))
        y_full, _ = remap_shakespeare(y_full, str(data_dir))
        split_idx = int(0.9 * len(x_full))
        x_train, y_train = x_full[:split_idx], y_full[:split_idx]
        x_val, y_val = x_full[split_idx:], y_full[split_idx:]
        num_classes = vocab_size
        input_shape = [x_full.shape[1]]
        use_sparse_labels = True
    elif dataset_name == "speech_commands":
        label_map = _load_label_map(Path(args.data_dir))
        num_classes = len(label_map)
        files = np.asarray(x_full)
        labels = np.asarray(y_full, dtype=np.int64)
        split_idx = int(0.9 * len(files))
        train_files, val_files = files[:split_idx], files[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        train_size = len(train_files)
        val_size = len(val_files)
        # Build dataset creators that batch on the fly (streaming from disk)
        def _train_fn(batch_size, training=True):
            return _build_speech_ds(train_files, train_labels, data_dir, batch_size, training)

        def _val_fn(batch_size, training=False):
            return _build_speech_ds(val_files, val_labels, data_dir, batch_size, training)

        train_ds_fn = _train_fn
        val_ds_fn = _val_fn
        # Keep input shape aligned with server-side speech_commands defaults.
        # Avoid probing a sample at startup (can block on some runtime stacks).
        input_shape = [98, 64, 1]
        use_sparse_labels = True
        # Placeholders for legacy array interface (unused when dataset fns provided)
        x_train = np.empty((0,), dtype=object)
        y_train = np.empty((0,), dtype=np.int64)
        x_val = np.empty((0,), dtype=object)
        y_val = np.empty((0,), dtype=np.int64)
    elif dataset_name == "stackoverflow":
        meta = load_stackoverflow_meta(data_dir)
        if getattr(y_full, "ndim", 1) == 2:
            # Multi-label (multi-hot) targets
            y_full = y_full.astype(np.float32, copy=False)
            num_classes = int(y_full.shape[1])
            use_sparse_labels = True  # avoid categorical one-hot conversion
        else:
            y_full, num_classes = remap_stackoverflow_labels(y_full, meta.get("tag_vocab_size", 0))
            use_sparse_labels = True
        split_idx = int(0.9 * len(x_full))
        x_train, y_train = x_full[:split_idx].astype(np.int32, copy=False), y_full[:split_idx]
        x_val, y_val = x_full[split_idx:].astype(np.int32, copy=False), y_full[split_idx:]
        input_shape = [int(x_full.shape[1])]
    else:
        # Image-style datasets
        if x_full.dtype != np.float32:
            x_full = x_full.astype(np.float32)
        if x_full.max() > 1.0:
            x_full /= 255.0
        if x_full.ndim == 3:
            x_full = np.expand_dims(x_full, axis=-1)

        split_idx = int(0.9 * len(x_full))
        x_train, y_train = x_full[:split_idx], y_full[:split_idx]
        x_val, y_val = x_full[split_idx:], y_full[split_idx:]

        num_classes = args.num_classes
        input_shape = list(x_full.shape[1:])

    # Build Hydra-style config dict for FlowerClient
    target_map = {
        "cnn": "fedavgm.models.cnn",
        "tf_example": "fedavgm.models.tf_example",
        "resnet18": "fedavgm.models.resnet18_keras",
        "resnet34": "fedavgm.models.resnet34_keras",
        "resnet20": "fedavgm.models.resnet20_keras",
        "cifar10_resnet": "fedavgm.models.cifar10_resnet",
        "mobilenet_v2_075": "fedavgm.models.mobilenet_v2_075",
        "mobilenet_v2_100": "fedavgm.models.mobilenet_v2_100",
        "char_lstm2": "fedavgm.models.char_lstm2",
        "speech_cnn_small": "fedavgm.models.speech_cnn_small",
        "embed_avg_mlp": "fedavgm.models.embed_avg_mlp",
        "embed_bilstm_mlp": "fedavgm.models.embed_bilstm_mlp",
        "embed_avg_mlp_bce": "fedavgm.models.embed_avg_mlp_bce",
        "embed_bilstm_mlp_bce": "fedavgm.models.embed_bilstm_mlp_bce",
    }

    model_cfg = {
        "_target_": target_map[args.model],
        "input_shape": input_shape,
        "num_classes": int(num_classes),
        "learning_rate": args.lr,
    }
    if dataset_name == "stackoverflow":
        if args.model not in {
            "embed_avg_mlp",
            "embed_bilstm_mlp",
            "embed_avg_mlp_bce",
            "embed_bilstm_mlp_bce",
        }:
            raise SystemExit(
                "For --dataset stackoverflow, please use --model "
                "embed_avg_mlp/embed_bilstm_mlp (single-label) or "
                "embed_avg_mlp_bce/embed_bilstm_mlp_bce (multi-label)"
            )
        is_multilabel = getattr(y_full, "ndim", 1) == 2
        if is_multilabel and not args.model.endswith("_bce"):
            raise SystemExit(
                "This StackOverflow partition looks multi-label (y is 2-D). "
                "Please use --model embed_avg_mlp_bce or embed_bilstm_mlp_bce."
            )
        if (not is_multilabel) and args.model.endswith("_bce"):
            raise SystemExit(
                "This StackOverflow partition looks single-label (y is 1-D). "
                "Please use --model embed_avg_mlp or embed_bilstm_mlp."
            )
        meta = load_stackoverflow_meta(data_dir)
        model_cfg["vocab_size"] = int(meta["vocab_size_total"])

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
        train_dataset_fn=train_ds_fn,
        val_dataset_fn=val_ds_fn,
        train_size=train_size,
        val_size=val_size,
        use_sparse_labels=use_sparse_labels,
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
                sample_count = train_size if train_size is not None else len(x_train)
                print(
                    f"[Round {r}] train_time={train_time:.2f}s on {sample_count} samples "
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
