"""Dataset utilities for federated learning."""

from pathlib import Path
from typing import Tuple

import numpy as np
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
