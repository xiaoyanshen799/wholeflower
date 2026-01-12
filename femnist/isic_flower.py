"""Flower utilities for Fed-ISIC2019 using EfficientNet."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import flwr as fl
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from compression import ErrorFeedbackQuantizer, maybe_unpack_quantized
from isic_dataset import Isic2019Dataset
from isic_model import build_efficientnet
from isic_transforms import build_isic_transform


def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [tensor.detach().cpu().numpy() for tensor in model.state_dict().values()]


def set_model_parameters(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    expected_keys = list(state_dict.keys())
    if len(params) != len(expected_keys):
        raise ValueError("Parameter length mismatch for ISIC model")
    new_state = {}
    for key, array in zip(expected_keys, params):
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        new_state[key] = torch.from_numpy(array)
    model.load_state_dict(new_state, strict=True)


def evaluate_isic(
    model: torch.nn.Module,
    parameters: List[np.ndarray],
    data_root: str,
    device: torch.device,
    split_csv: Optional[str] = None,
    preprocessed_dir: str = "ISIC_2019_Training_Input_preprocessed",
    batch_size: int = 64,
    num_workers: int = 0,
    sample_size: Optional[int] = None,
    sample_seed: Optional[int] = None,
) -> tuple[float, float]:
    set_model_parameters(model, parameters)
    model.eval()

    dataset = Isic2019Dataset(
        data_root=data_root,
        split="test",
        pooled=True,
        split_csv=split_csv,
        preprocessed_dir=preprocessed_dir,
        transform=build_isic_transform(train=False),
    )
    if sample_size is not None and sample_size < len(dataset):
        generator = torch.Generator().manual_seed(sample_seed or 0)
        dataset, _ = torch.utils.data.random_split(
            dataset, [sample_size, len(dataset) - sample_size], generator=generator
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
    loss_avg = total_loss / max(len(loader), 1)
    bal_acc = balanced_accuracy_score(all_targets, all_preds) if all_targets else 0.0
    return loss_avg, bal_acc


class IsicFlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient for Fed-ISIC2019."""

    def __init__(
        self,
        *,
        data_root: str,
        center: int,
        device: torch.device,
        batch_size: int,
        num_workers: int,
        seed: int,
        learning_rate: float = 1e-3,
        local_epochs_override: Optional[int] = None,
        batch_size_override: Optional[int] = None,
        split_csv: Optional[str] = None,
        preprocessed_dir: str = "ISIC_2019_Training_Input_preprocessed",
        enable_compression: bool = False,
        quantization_bits: int = 8,
    ) -> None:
        self.device = device
        self.learning_rate = learning_rate
        self._local_epochs_override = local_epochs_override
        self._batch_size_override = batch_size_override

        train_ds = Isic2019Dataset(
            data_root=data_root,
            split="train",
            center=center,
            pooled=False,
            preprocessed_dir=preprocessed_dir,
            split_csv=split_csv,
            transform=build_isic_transform(train=True),
        )
        val_ds = Isic2019Dataset(
            data_root=data_root,
            split="test",
            center=center,
            pooled=False,
            preprocessed_dir=preprocessed_dir,
            split_csv=split_csv,
            transform=build_isic_transform(train=False),
        )

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

        self.model = build_efficientnet(num_classes=8, pretrained=True).to(self.device)
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
        logging.info("[ISIC Client] Training finished in %.2f seconds", duration)

        weights = get_model_parameters(self.model)
        if isinstance(server_to_client_ms, (int, float)):
            metrics["server_to_client_ms"] = float(server_to_client_ms)
        metrics["train_time"] = float(duration)

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
        all_targets: List[int] = []
        all_preds: List[int] = []
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(images)
                loss = criterion(logits, targets)
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_targets.extend(targets.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())
        loss_avg = total_loss / max(len(self.val_loader), 1)
        bal_acc = balanced_accuracy_score(all_targets, all_preds) if all_targets else 0.0
        return loss_avg, len(self.val_loader.dataset), {"balanced_accuracy": bal_acc}
