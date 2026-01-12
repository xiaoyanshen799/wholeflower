"""Flower utilities for Fed-IXI using a PyTorch 3D U-Net."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from compression import ErrorFeedbackQuantizer, maybe_unpack_quantized
from ixi_dataset import FedIXITinyDataset
from ixi_model import UNet3D


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    denominator = pred.sum(dim=dims) + target.sum(dim=dims)
    return (2 * intersection + eps) / (denominator + eps)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    foreground = probs[:, 1]
    target_fg = target[:, 0]
    loss = 1 - dice_score(foreground, target_fg).mean()
    return loss


def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [tensor.detach().cpu().numpy() for tensor in model.state_dict().values()]


def set_model_parameters(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    expected_keys = list(state_dict.keys())
    if len(params) != len(expected_keys):
        raise ValueError("Parameter length mismatch for IXI model")
    new_state = {}
    for key, array in zip(expected_keys, params):
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        new_state[key] = torch.from_numpy(array)
    model.load_state_dict(new_state, strict=True)


def create_ixi_loaders(
    data_root: str,
    center: Optional[str | int],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    dataset = FedIXITinyDataset(root=data_root, split="train", center=center)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader


class IXIFlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient for Fed-IXI using PyTorch."""

    def __init__(
        self,
        *,
        data_root: str,
        center: Optional[str | int],
        device: torch.device,
        batch_size: int,
        num_workers: int,
        seed: int,
        learning_rate: float = 1e-3,
        local_epochs_override: Optional[int] = None,
        batch_size_override: Optional[int] = None,
        enable_compression: bool = False,
        quantization_bits: int = 8,
    ) -> None:
        self.device = device
        self.model = UNet3D(in_channels=1, out_channels=2, base=8).to(self.device)
        self.train_loader, self.val_loader = create_ixi_loaders(
            data_root=data_root,
            center=center,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )
        self._local_epochs_override = local_epochs_override
        self._batch_size_override = batch_size_override
        self.learning_rate = learning_rate
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
            else config.get("batch_size", 2)
        )
        server_send_time = config.get("server_send_time", None)
        server_to_client_ms = None
        now = time.time()
        if isinstance(server_send_time, (int, float)):
            server_to_client_ms = max(0.0, (now - server_send_time) * 1000.0)
        metrics: Dict[str, float] = {}

        start = time.time()
        # Recreate loader if batch size override changes
        if batch_size != self.train_loader.batch_size:
            train_dataset = self.train_loader.dataset
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.train_loader.num_workers,
            )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()

        
        for _ in range(int(epochs)):
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = dice_loss(logits, labels)
                loss.backward()
                optimizer.step()
        duration = time.time() - start
        logging.info("[IXI Client] Training finished in %.2f seconds", duration)

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
        total_loss = 0.0
        total_dice = 0.0
        count = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = dice_loss(logits, labels)
                probs = torch.softmax(logits, dim=1)[:, 1]
                dice = dice_score(probs, labels[:, 0]).mean().item()
                total_loss += loss.item()
                total_dice += dice
                count += 1
        loss_avg = total_loss / max(count, 1)
        dice_avg = total_dice / max(count, 1)
        return loss_avg, len(self.val_loader.dataset), {"dice": dice_avg}


def evaluate_ixi(
    model: torch.nn.Module,
    parameters: List[np.ndarray],
    data_root: str,
    device: torch.device,
    batch_size: int = 1,
    num_workers: int = 0,
    sample_size: Optional[int] = None,
    sample_seed: Optional[int] = None,
) -> Tuple[float, float]:
    set_model_parameters(model, parameters)
    model.eval()

    dataset = FedIXITinyDataset(root=data_root, split="test", center=None)
    if sample_size is not None and sample_size < len(dataset):
        generator = torch.Generator().manual_seed(sample_seed or 0)
        subset, _ = random_split(
            dataset, [sample_size, len(dataset) - sample_size], generator=generator
        )
        dataset = subset

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    total_loss = 0.0
    total_dice = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = dice_loss(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]
            dice = dice_score(probs, labels[:, 0]).mean().item()
            total_loss += loss.item()
            total_dice += dice
            count += 1
    return total_loss / max(count, 1), total_dice / max(count, 1)
