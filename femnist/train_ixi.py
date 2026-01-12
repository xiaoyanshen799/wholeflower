#!/usr/bin/env python3
"""Train a 3D U-Net on Fed-IXI (IXI Tiny) data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ixi_dataset import FedIXITinyDataset
from ixi_model import UNet3D


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = dice_loss(logits, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[:, 1]
            dice = dice_score(probs, labels[:, 0]).mean().item()
        total_loss += loss.item()
        total_dice += dice
        count += 1
    return total_loss / max(count, 1), total_dice / max(count, 1)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 3D U-Net on IXI Tiny.")
    parser.add_argument("--data-root", default="data/fed_ixi", help="Dataset root path")
    parser.add_argument("--center", default=None, help="Center name or id (Guys/HH/IOP or 0/1/2)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default=None, help="cpu or cuda (auto if unset)")
    parser.add_argument("--save-dir", default="checkpoints/ixi", help="Checkpoint directory")
    args = parser.parse_args()

    set_seed(args.seed)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_ds = FedIXITinyDataset(
        root=args.data_root,
        split="train",
        center=args.center,
    )
    test_ds = FedIXITinyDataset(
        root=args.data_root,
        split="test",
        center=args.center,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = UNet3D(in_channels=1, out_channels=2, base=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_dice = evaluate(model, test_loader, device)
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "test_loss": test_loss,
            "test_dice": test_dice,
        }
        history.append(entry)
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"test_loss={test_loss:.4f} test_dice={test_dice:.4f}"
        )
        if test_dice > best_dice:
            best_dice = test_dice
            torch.save(model.state_dict(), save_dir / "best.pt")

    torch.save(model.state_dict(), save_dir / "last.pt")
    (save_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
