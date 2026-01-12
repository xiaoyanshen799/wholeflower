#!/usr/bin/env python3
"""Train EfficientNet on Fed-ISIC2019."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from isic_dataset import Isic2019Dataset
from isic_model import build_efficientnet
from isic_transforms import build_isic_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    all_targets = []
    all_preds = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
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
    avg_loss = total_loss / max(len(loader), 1)
    bal_acc = balanced_accuracy_score(all_targets, all_preds) if all_targets else 0.0
    return avg_loss, bal_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientNet on ISIC2019.")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Dataset root (contains ISIC_2019_Training_Input_preprocessed)",
    )
    parser.add_argument(
        "--split-csv",
        default=None,
        help="Path to train_test_split file (auto-downloads if missing)",
    )
    parser.add_argument(
        "--center",
        type=int,
        default=None,
        help="Center id (0-5); leave unset for pooled training",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/isic",
        help="Directory to save checkpoints",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    pooled = args.center is None
    train_ds = Isic2019Dataset(
        data_root=args.data_root,
        split="train",
        center=args.center,
        pooled=pooled,
        split_csv=args.split_csv,
        transform=build_isic_transform(train=True),
    )
    test_ds = Isic2019Dataset(
        data_root=args.data_root,
        split="test",
        center=args.center,
        pooled=pooled,
        split_csv=args.split_csv,
        transform=build_isic_transform(train=False),
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_efficientnet(num_classes=8, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_bal_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        test_loss, bal_acc = evaluate(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "balanced_accuracy": bal_acc,
            }
        )
        print(
            f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} balanced_acc={bal_acc:.4f}"
        )

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            torch.save(model.state_dict(), output_dir / "best.pt")

    torch.save(model.state_dict(), output_dir / "last.pt")
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
