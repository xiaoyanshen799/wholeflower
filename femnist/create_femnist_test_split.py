"""Utility to carve out a centralized FEMNIST test split from client NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


def collect_clients(data_dir: Path) -> List[Path]:
    """Return sorted list of FEMNIST client NPZ files."""
    return sorted(data_dir.glob("client_*.npz"))


def carve_from_client(
    file_path: Path,
    fraction: float,
    max_per_client: int,
    min_train_after: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove a slice from the tail of a client file and return it for the test set."""
    with np.load(file_path) as npz:
        x = npz["x_train"]
        y = npz["y_train"]

    total = len(y)
    if total == 0:
        return np.empty((0,) + x.shape[1:], dtype=x.dtype), np.empty((0,), dtype=y.dtype)

    proposed = max(1, int(round(total * fraction)))
    take = min(proposed, max_per_client)
    take = min(take, max(0, total - min_train_after))

    if take <= 0:
        return np.empty((0,) + x.shape[1:], dtype=x.dtype), np.empty((0,), dtype=y.dtype)

    x_test_slice = x[-take:]
    y_test_slice = y[-take:]
    x_remain = x[:-take]
    y_remain = y[:-take]

    np.savez_compressed(file_path, x_train=x_remain, y_train=y_remain)
    print(
        f"[Split] {file_path.name}: total={total} -> kept={len(y_remain)} moved={take}"
    )
    return x_test_slice, y_test_slice


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split off centralized FEMNIST test samples from client NPZ files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_partitions"),
        help="Directory containing client_*.npz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("femnist/data_partitions/test_femnist.npz"),
        help="Destination NPZ for centralized test set.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction (0-1] of each client's samples to move into the test split.",
    )
    parser.add_argument(
        "--max-per-client",
        type=int,
        default=120,
        help="Upper bound on the number of samples to pull from a single client.",
    )
    parser.add_argument(
        "--min-train-after",
        type=int,
        default=200,
        help="Ensure at least this many samples remain in each client after carving.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report how many samples would be moved without modifying files.",
    )

    args = parser.parse_args()

    clients = collect_clients(args.data_dir)
    if not clients:
        raise SystemExit(f"No client_*.npz files found in {args.data_dir}")

    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for client_file in clients:
        with np.load(client_file) as npz:
            total = len(npz["y_train"])

        proposed = max(1, int(round(total * args.fraction)))
        take = min(proposed, args.max_per_client)
        take = min(take, max(0, total - args.min_train_after))

        if take <= 0:
            print(f"[Skip] {client_file.name}: total={total}, nothing moved")
            continue

        if args.dry_run:
            print(f"[DryRun] {client_file.name}: would move {take} samples")
            continue

        x_slice, y_slice = carve_from_client(
            client_file,
            args.fraction,
            args.max_per_client,
            args.min_train_after,
        )
        if x_slice.size == 0:
            continue
        all_x.append(x_slice)
        all_y.append(y_slice)

    if args.dry_run:
        print("[DryRun] Dry run complete; no files were modified.")
        return

    if not all_x:
        raise SystemExit("No samples were moved; test split would be empty.")

    x_test = np.concatenate(all_x, axis=0)
    y_test = np.concatenate(all_y, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x_test=x_test, y_test=y_test)
    print(
        f"[Done] Saved centralized test set to {args.output} "
        f"({x_test.shape[0]} samples)"
    )


if __name__ == "__main__":
    main()
