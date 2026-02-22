#!/usr/bin/env python3
"""Prepare federated CIFAR-100 partitions in NPZ format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner


def _pick_key(data, candidates: tuple[str, ...]) -> str:
    if hasattr(data, "column_names"):
        available = list(data.column_names)
    elif hasattr(data, "keys"):
        available = list(data.keys())
    else:
        raise TypeError(f"Unsupported data container type: {type(data)!r}")

    for key in candidates:
        if key in available:
            return key
    raise KeyError(f"None of keys {candidates} found in dataset columns: {available}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CIFAR-100 federated partitions.")
    parser.add_argument("--num-clients", type=int, default=10, help="Number of client partitions")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--partition-by",
        default="fine_label",
        choices=["fine_label", "label"],
        help="Label column used by Dirichlet partitioning",
    )
    parser.add_argument(
        "--out-dir",
        default="data_partitions_cifar100",
        help="Output folder for client_XXXXX.npz and test.npz",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    partitioner = DirichletPartitioner(
        num_partitions=args.num_clients,
        partition_by=args.partition_by,
        alpha=args.alpha,
        min_partition_size=0.0002,
        seed=args.seed,
    )

    fds = FederatedDataset(
        dataset="cifar100",
        partitioners={"train": partitioner, "test": 1},
    )

    pad = max(5, len(str(args.num_clients)))
    for cid in range(args.num_clients):
        ds = fds.load_partition(cid, "train").with_format("numpy")
        x_key = _pick_key(ds, ("img", "image", "pixel_values"))
        y_key = _pick_key(ds, ("label", "labels", "fine_label"))
        np.savez_compressed(
            out_dir / f"client_{cid:0{pad}d}.npz",
            x_train=ds[x_key],
            y_train=ds[y_key],
        )

    test_ds = fds.load_split("test").with_format("numpy")
    x_key = _pick_key(test_ds, ("img", "image", "pixel_values"))
    y_key = _pick_key(test_ds, ("label", "labels", "fine_label"))
    np.savez_compressed(
        out_dir / "test.npz",
        x_test=test_ds[x_key],
        y_test=test_ds[y_key],
    )

    print(
        f"Saved {args.num_clients} CIFAR-100 client partitions and test set to {out_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
