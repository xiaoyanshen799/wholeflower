"""Dataset utilities for federated learning."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

from fedavgm.common import create_lda_partitions


def cifar10(num_classes, input_shape):
    """Prepare the CIFAR-10.

    This method considers CIFAR-10 for creating both train and test sets. The sets are
    already normalized.
    """
    print(f">>> [Dataset] Loading CIFAR-10. {num_classes} | {input_shape}.")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def fmnist(num_classes, input_shape):
    """Prepare the FMNIST.

    This method considers FMNIST for creating both train and test sets. The sets are
    already normalized.
    """
    print(f">>> [Dataset] Loading FMNIST. {num_classes} | {input_shape}.")
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Expand channel dimension to (H, W, 1)
    if x_train.ndim == 3:
        x_train = x_train[..., None]
    if x_test.ndim == 3:
        x_test = x_test[..., None]
    # Use the requested input_shape/num_classes (caller controls these)
    input_shape = x_train.shape[1:]
    num_classes = int(num_classes)

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def mnist(num_classes, input_shape):
    """Prepare the MNIST.

    This method considers MNIST for creating both train and test sets. The sets are
    normalized and reshaped to (H, W, 1).
    """
    print(f">>> [Dataset] Loading MNIST. {num_classes} | {input_shape}.")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    if x_train.ndim == 3:
        x_train = x_train[..., None]
    if x_test.ndim == 3:
        x_test = x_test[..., None]
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def femnist_dataset(
    data_dir: str = "data_partitions",
    train_file: str = "train_femnist.npz",
    test_file: str = "test_femnist.npz",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int], int]:
    """Load federated FEMNIST centralized splits from NPZ files."""
    base = Path(data_dir)
    test_path = base / test_file
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    x_train = np.empty((0, 28, 28, 1), dtype=np.float32)
    y_train = np.empty((0,), dtype=np.int64)
    train_path = base / train_file
    if train_path.exists():
        with np.load(train_path) as npz:
            x_train = npz["x_train"].astype(np.float32)
            y_train = npz["y_train"].astype(np.int64)

    with np.load(test_path) as npz:
        x_test = npz["x_test"].astype(np.float32)
        y_test = npz["y_test"].astype(np.int64)

    def _prepare_images(images: np.ndarray, name: str) -> np.ndarray:
        """Ensure image arrays are 4-D floats in [0, 1]."""
        if images.ndim == 4:
            processed = images
        elif images.ndim == 3:
            processed = images[..., None]
        elif images.ndim == 2:
            side = int(np.sqrt(images.shape[1]))
            if side * side != images.shape[1]:
                raise ValueError(
                    f"Cannot reshape {name} of shape {images.shape} into square images."
                )
            processed = images.reshape((images.shape[0], side, side, 1))
        else:
            raise ValueError(
                f"Expected {name} to be 2-D, 3-D, or 4-D, got array with shape {images.shape}"
            )

        processed = processed.astype(np.float32, copy=False)
        if processed.size and processed.max() > 1.0:
            processed /= 255.0
        return processed

    x_train = _prepare_images(x_train, "x_train")
    x_test = _prepare_images(x_test, "x_test")

    input_shape = tuple(x_test.shape[1:])
    train_max = int(y_train.max()) if y_train.size else -1
    test_max = int(y_test.max()) if y_test.size else -1
    num_classes = max(train_max, test_max) + 1
    num_classes = max(num_classes, 62)

    return x_train, y_train, x_test, y_test, input_shape, num_classes


# ------------------------------------------------------------------------------
# Shakespeare (character-level) utilities
# ------------------------------------------------------------------------------

def _ensure_path(pathlike) -> Path:
    path = Path(pathlike)
    if not path.exists():
        raise FileNotFoundError(f"Directory or file not found: {path}")
    return path


def _build_lookup_from_values(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create a sorted vocabulary array and lookup table mapping original ids to compact ids."""
    vocab = np.array(sorted(int(v) for v in np.unique(values)), dtype=np.int32)
    if vocab.size == 0 or vocab[0] != 0:
        vocab = np.insert(vocab, 0, 0)
    lookup = np.full(int(vocab[-1]) + 1, -1, dtype=np.int32)
    lookup[vocab] = np.arange(vocab.size, dtype=np.int32)
    return vocab, lookup


def _remap_array(arr: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1)
    max_val = int(flat.max()) if flat.size else 0
    if max_val >= lookup.size:
        raise ValueError(f"Encountered value {max_val} outside of lookup range {lookup.size}")
    remapped = lookup[flat]
    if np.any(remapped < 0):
        missing = np.unique(flat[remapped < 0])
        raise ValueError(f"Found unmapped values: {missing}")
    return remapped.reshape(arr.shape).astype(np.int32, copy=False)


@lru_cache(maxsize=None)
def load_shakespeare_lookup(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load all Shakespeare partitions to construct a shared vocab and lookup table."""
    directory = _ensure_path(data_dir)
    all_values = []
    for npz_path in sorted(directory.glob("client_*.npz")):
        with np.load(npz_path) as npz:
            all_values.append(npz["x_train"].ravel())
            all_values.append(npz["y_train"].ravel())
    test_path = directory / "test_shakespeare.npz"
    if test_path.exists():
        with np.load(test_path) as npz:
            all_values.append(npz["x_test"].ravel())
            all_values.append(npz["y_test"].ravel())
    if not all_values:
        raise ValueError(f"No Shakespeare partitions found in {directory}")
    concatenated = np.concatenate(all_values)
    return _build_lookup_from_values(concatenated)


def remap_shakespeare(arr: np.ndarray, data_dir: str) -> Tuple[np.ndarray, int]:
    """Remap an array of raw codepoints to contiguous ids."""
    vocab, lookup = load_shakespeare_lookup(data_dir)
    remapped = _remap_array(arr, lookup)
    return remapped, len(vocab)


def shakespeare(num_classes, input_shape, data_dir: str = "data_partitions_shakespeare"):
    """Load centralized test split for Shakespeare (train arrays empty)."""
    data_dir = str(Path(data_dir))
    path = _ensure_path(Path(data_dir) / "test_shakespeare.npz")
    data = np.load(path)
    x_test_raw = data["x_test"]
    y_test_raw = data["y_test"]

    x_test, vocab_size = remap_shakespeare(x_test_raw, data_dir)
    y_test, _ = remap_shakespeare(y_test_raw, data_dir)

    seq_len = int(x_test.shape[1])

    x_train = np.empty((0, seq_len), dtype=np.int32)
    y_train = np.empty((0,), dtype=np.int32)
    input_shape = [seq_len]
    num_classes = vocab_size
    return x_train, y_train, x_test, y_test, input_shape, num_classes


# ------------------------------------------------------------------------------
# Stack Overflow tag prediction utilities (top-tags subset + token ids)
# ------------------------------------------------------------------------------

def _count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def load_stackoverflow_meta(data_dir: str | Path) -> Dict[str, int]:
    """Load vocab/tag sizes from meta.json and/or vocab files.

    Returns:
        dict with keys: vocab_size, tag_vocab_size, vocab_size_total, label_mode
    """
    base = Path(data_dir)
    meta_path = base / "meta.json"
    meta: Dict[str, int] = {}
    label_mode: str | None = None
    if meta_path.exists():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            for k in ("vocab_size", "tag_vocab_size", "seq_len"):
                if k in raw and raw[k] is not None:
                    meta[k] = int(raw[k])
            if "label_mode" in raw and raw["label_mode"] is not None:
                label_mode = str(raw["label_mode"])
        except Exception:
            meta = {}

    if "vocab_size" not in meta:
        meta["vocab_size"] = int(_count_nonempty_lines(base / "word_vocab.txt"))
    if "tag_vocab_size" not in meta:
        meta["tag_vocab_size"] = int(_count_nonempty_lines(base / "tag_vocab.txt"))

    meta["vocab_size_total"] = int(meta["vocab_size"]) + 2  # PAD=0, OOV=1
    meta["label_mode"] = label_mode or "single"
    return meta


def remap_stackoverflow_labels(y: np.ndarray, tag_vocab_size: int) -> Tuple[np.ndarray, int]:
    """Remap labels to 0..K-1 when labels are stored as 1..K (common for top-tags subset)."""
    y = np.asarray(y, dtype=np.int32)
    if y.size == 0:
        return y, int(tag_vocab_size)
    if y.min() >= 1 and y.max() <= int(tag_vocab_size):
        return (y - 1).astype(np.int32, copy=False), int(tag_vocab_size)
    # Fallback: treat labels as already 0-indexed (may include 0 = unknown)
    return y, int(y.max()) + 1


def stackoverflow(num_classes, input_shape, data_dir: str = "data_partitions_stackoverflow"):
    """Load centralized test split for StackOverflow tag prediction (train arrays empty)."""
    base = Path(data_dir)
    test_path = _ensure_path(base / "test_stackoverflow.npz")
    with np.load(test_path) as npz:
        x_test = npz["x_test"].astype(np.int32, copy=False)
        y_test = npz["y_test"]

    meta = load_stackoverflow_meta(base)
    if y_test.ndim == 2:
        # Multi-label (multi-hot) targets
        y_test = y_test.astype(np.float32, copy=False)
        inferred_classes = int(y_test.shape[1])
    else:
        y_test = y_test.astype(np.int32, copy=False)
        y_test, inferred_classes = remap_stackoverflow_labels(y_test, meta.get("tag_vocab_size", 0))

    seq_len = int(x_test.shape[1])
    input_shape = [seq_len]
    num_classes = int(inferred_classes)

    x_train = np.empty((0, seq_len), dtype=np.int32)
    if y_test.ndim == 2:
        y_train = np.empty((0, num_classes), dtype=np.float32)
    else:
        y_train = np.empty((0,), dtype=np.int32)
    return x_train, y_train, x_test, y_test, input_shape, num_classes


# ------------------------------------------------------------------------------
# Speech Commands utilities (manifests + log-mel features for evaluation)
# ------------------------------------------------------------------------------

def _load_label_map(data_dir: Path) -> list[str]:
    path = data_dir / "label_map.txt"
    if not path.exists():
        raise FileNotFoundError(f"Label map not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_log_mel(
    path: Path,
    sample_rate: int = 16000,
    mel_bins: int = 32,
    frame_length: int = 400,
    frame_step: int = 160,
    fft_length: int = 512,
) -> np.ndarray:
    """Return log-mel spectrogram for one audio file as a numpy array."""
    audio = tf.io.read_file(str(path))
    wav, sr = tf.audio.decode_wav(audio, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    wav = tf.cast(wav, tf.float32)

    # Pad/crop to 1 second
    desired = sample_rate
    wav_len = tf.shape(wav)[0]
    wav = tf.cond(
        wav_len < desired,
        lambda: tf.pad(wav, [[0, desired - wav_len]]),
        lambda: wav[:desired],
    )

    stft = tf.signal.stft(
        wav,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
    )
    spectrogram = tf.abs(stft)
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=mel_bins,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=sample_rate,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )
    mel_spectrogram = tf.tensordot(tf.square(spectrogram), mel_matrix, 1)
    log_mel = tf.math.log(mel_spectrogram + 1e-6)
    # Per-example CMVN keeps train/eval preprocessing aligned.
    mean = tf.reduce_mean(log_mel)
    std = tf.math.reduce_std(log_mel)
    log_mel = (log_mel - mean) / (std + 1e-6)
    log_mel = tf.expand_dims(log_mel, -1)
    return log_mel.numpy().astype(np.float32)


def speech_commands(
    num_classes,
    input_shape,
    data_dir: str = "data_partitions_speech_commands",
    mel_bins: int = 32,
):
    """Load Speech Commands manifests and compute test log-mel features for evaluation."""
    base = Path(data_dir)
    manifest = _ensure_path(base / "test_speech_commands.npz")
    label_map = _load_label_map(base)
    with np.load(manifest) as npz:
        rel_files = npz["files"]
        labels = npz["labels"].astype(np.int64)

    abs_files = [base / Path(f) for f in rel_files]
    # Compute log-mel features for centralized evaluation
    x_test = np.stack([_load_log_mel(p, mel_bins=mel_bins) for p in abs_files], axis=0)
    y_test = labels

    input_shape = x_test.shape[1:]
    num_classes = len(label_map)

    x_train = np.empty((0, *input_shape), dtype=np.float32)
    y_train = np.empty((0,), dtype=np.int64)
    return x_train, y_train, x_test, y_test, input_shape, num_classes


def partition(x_train, y_train, num_clients, concentration):
    """Create non-iid partitions.

    The partitions uses a LDA distribution based on concentration.
    """
    print(
        f">>> [Dataset] {num_clients} clients, non-iid concentration {concentration}..."
    )
    dataset = [x_train, y_train]
    partitions, _ = create_lda_partitions(
        dataset,
        num_partitions=num_clients,
        # concentration=concentration * num_classes,
        concentration=concentration,
        seed=1234,
    )
    return partitions
