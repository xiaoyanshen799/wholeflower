"""Flower utilities for CIFAR-100 with ResNet-152."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import flwr as fl
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from cifar100_model import build_resnet152
from compression import ErrorFeedbackQuantizer, maybe_unpack_quantized

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_cifar100_transform(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class NpzImageDataset(Dataset):
    """Simple dataset wrapper for NPZ image arrays."""

    def __init__(self, images: np.ndarray, targets: np.ndarray, train: bool) -> None:
        self.images = images
        self.targets = targets.astype(np.int64, copy=False)
        self.transform = build_cifar100_transform(train=train)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        image = Image.fromarray(self.images[idx])
        image = self.transform(image)
        target = int(self.targets[idx])
        return image, target


def _load_partition(data_dir: str | Path, cid: int) -> tuple[np.ndarray, np.ndarray]:
    data_path = Path(data_dir)
    pad = max(5, len(str(cid)))
    file = data_path / f"client_{cid:0{pad}d}.npz"
    if not file.exists():
        fallback = data_path / f"client_{cid}.npz"
        if not fallback.exists():
            raise FileNotFoundError(f"Partition file not found: {file}")
        file = fallback
    with np.load(file) as npz:
        return npz["x_train"], npz["y_train"]


def _load_test(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    file = Path(data_dir) / "test.npz"
    if not file.exists():
        raise FileNotFoundError(f"Test split file not found: {file}")
    with np.load(file) as npz:
        return npz["x_test"], npz["y_test"]


def _split_train_val(
    images: np.ndarray, targets: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_samples = int(images.shape[0])
    if num_samples < 2:
        raise ValueError("Each CIFAR-100 client partition must contain at least 2 samples")
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(0.9 * num_samples)
    if split <= 0:
        split = 1
    if split >= num_samples:
        split = num_samples - 1
    train_idx = indices[:split]
    val_idx = indices[split:]
    return images[train_idx], targets[train_idx], images[val_idx], targets[val_idx]


def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [tensor.detach().cpu().numpy() for tensor in model.state_dict().values()]


def set_model_parameters(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    expected_keys = list(state_dict.keys())
    if len(params) != len(expected_keys):
        raise ValueError("Parameter length mismatch for CIFAR-100 model")
    new_state = {}
    for key, array in zip(expected_keys, params):
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        new_state[key] = torch.from_numpy(array)
    model.load_state_dict(new_state, strict=True)


def evaluate_cifar100(
    model: torch.nn.Module,
    parameters: List[np.ndarray],
    data_dir: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
    sample_size: Optional[int] = None,
    sample_seed: Optional[int] = None,
) -> tuple[float, float]:
    set_model_parameters(model, parameters)
    model.eval()

    test_x, test_y = _load_test(data_dir)
    dataset: Dataset = NpzImageDataset(test_x, test_y, train=False)
    if sample_size is not None and sample_size < len(dataset):
        generator = torch.Generator().manual_seed(sample_seed or 0)
        perm = torch.randperm(len(dataset), generator=generator).tolist()
        dataset = Subset(dataset, perm[:sample_size])

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == targets).sum().item())
            total_count += int(targets.size(0))
    loss_avg = total_loss / max(len(loader), 1)
    accuracy = float(total_correct / max(total_count, 1))
    return loss_avg, accuracy


class Cifar100FlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient for CIFAR-100 using ResNet-152."""

    def __init__(
        self,
        *,
        data_dir: str,
        cid: int,
        device: torch.device,
        batch_size: int,
        num_workers: int,
        seed: int,
        learning_rate: float = 1e-3,
        local_epochs_override: Optional[int] = None,
        batch_size_override: Optional[int] = None,
        pretrained: bool = True,
        enable_compression: bool = False,
        quantization_bits: int = 8,
    ) -> None:
        self.device = device
        self.learning_rate = learning_rate
        self._local_epochs_override = local_epochs_override
        self._batch_size_override = batch_size_override

        images, targets = _load_partition(data_dir, cid)
        train_x, train_y, val_x, val_y = _split_train_val(images, targets, seed=seed)

        train_ds = NpzImageDataset(train_x, train_y, train=True)
        val_ds = NpzImageDataset(val_x, val_y, train=False)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        self.model = build_resnet152(num_classes=100, pretrained=pretrained).to(self.device)
        self._quantizer = (
            ErrorFeedbackQuantizer(num_bits=quantization_bits, error_feedback=True)
            if enable_compression
            else None
        )

    def get_parameters(self, config):  # pylint: disable=unused-argument
        return get_model_parameters(self.model)

    def _set_parameters(self, parameters: List[np.ndarray]) -> None:
        params, _ = maybe_unpack_quantized(list(parameters))
        set_model_parameters(self.model, params)

    def fit(self, parameters, config):
        self._set_parameters(parameters)

        epochs = (
            self._local_epochs_override
            if self._local_epochs_override is not None
            else config.get("local_epochs", 1)
        )
        batch_size = (
            self._batch_size_override
            if self._batch_size_override is not None
            else config.get("batch_size", self.train_loader.batch_size)
        )
        server_send_time = config.get("server_send_time", None)
        server_to_client_ms = None
        now = time.time()
        if isinstance(server_send_time, (int, float)):
            server_to_client_ms = max(0.0, (now - server_send_time) * 1000.0)
        metrics: Dict[str, float] = {}

        if batch_size != self.train_loader.batch_size:
            train_dataset = self.train_loader.dataset
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.train_loader.num_workers,
                pin_memory=self.device.type == "cuda",
            )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()

        start = time.time()
        for _ in range(int(epochs)):
            for images, targets in self.train_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
        duration = time.time() - start
        logging.info("[CIFAR100 Client] Training finished in %.2f seconds", duration)

        weights = get_model_parameters(self.model)
        if isinstance(server_to_client_ms, (int, float)):
            metrics["server_to_client_ms"] = float(server_to_client_ms)
        metrics["train_time"] = float(duration)
        metrics["client_fit_end_time"] = float(time.time())
        if self._quantizer is not None:
            packed, report = self._quantizer.encode(weights)
            metrics.update(report.as_metrics())
            metrics["quant_bits"] = float(self._quantizer.num_bits)
            metrics["quant_applied"] = 1.0
            return packed, len(self.train_loader.dataset), metrics

        metrics["quant_applied"] = 0.0
        return weights, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):  # pylint: disable=unused-argument
        self._set_parameters(parameters)
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(images)
                loss = criterion(logits, targets)
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                total_correct += int((preds == targets).sum().item())
                total_count += int(targets.size(0))
        loss_avg = total_loss / max(len(self.val_loader), 1)
        accuracy = float(total_correct / max(total_count, 1))
        return loss_avg, len(self.val_loader.dataset), {"accuracy": accuracy}
