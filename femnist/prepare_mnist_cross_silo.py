#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def _allocate_counts(total: int, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Allocate integer counts that sum to total according to weights."""
    if total <= 0:
        return np.zeros_like(weights, dtype=int)
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones_like(weights, dtype=float)
    raw = weights / weights.sum() * total
    counts = np.floor(raw).astype(int)
    remainder = total - counts.sum()
    if remainder > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        for i in range(remainder):
            counts[order[i % len(order)]] += 1
    return counts


def _class_partition_indices(
    y: np.ndarray,
    num_clients: int,
    num_classes: int,
    nmin: int,
    nmax: int,
    mu: float,
    sigma: float,
    rng: np.random.Generator,
) -> List[List[int]]:
    class_indices = {c: np.where(y == c)[0].tolist() for c in range(num_classes)}
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    while True:
        class_partition = {c: [] for c in range(num_classes)}
        for i in range(num_clients):
            ni = int(rng.integers(nmin, nmax + 1))
            classes = rng.permutation(num_classes)[:ni]
            for c in classes:
                class_partition[c].append(i)
        if all(len(v) > 0 for v in class_partition.values()):
            break

    for c in range(num_classes):
        idxs = class_indices[c]
        rng.shuffle(idxs)
        clients = class_partition[c]
        k = len(clients)
        weights = rng.normal(loc=mu, scale=sigma, size=k)
        counts = _allocate_counts(len(idxs), weights, rng)

        start = 0
        for cnt, client_id in zip(counts, clients):
            if cnt <= 0:
                continue
            client_indices[client_id].extend(idxs[start : start + cnt])
            start += cnt
        if start < len(idxs):
            leftover = idxs[start:]
            for idx in leftover:
                client_indices[clients[int(rng.integers(0, k))]].append(idx)

    return client_indices


def _dual_dirichlet_indices(
    y: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha1: float,
    alpha2: float,
    rng: np.random.Generator,
) -> List[List[int]]:
    class_indices = {c: np.where(y == c)[0].tolist() for c in range(num_classes)}
    class_counts = np.array([len(class_indices[c]) for c in range(num_classes)], dtype=float)
    p1 = np.full(num_clients, 1.0 / num_clients)
    p2 = class_counts / class_counts.sum()

    alpha1_vec = alpha1 * p1
    alpha2_vec = alpha2 * p2

    client_wts = rng.dirichlet(alpha1_vec)
    class_wts = rng.dirichlet(alpha2_vec, size=num_clients)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idxs = class_indices[c]
        rng.shuffle(idxs)
        denom = float(np.dot(client_wts, class_wts[:, c]))
        if denom <= 0:
            denom = 1e-12
        weights = client_wts * class_wts[:, c]
        counts = _allocate_counts(len(idxs), weights / denom, rng)

        start = 0
        for client_id, cnt in enumerate(counts):
            if cnt <= 0:
                continue
            client_indices[client_id].extend(idxs[start : start + cnt])
            start += cnt
        if start < len(idxs):
            leftover = idxs[start:]
            for idx in leftover:
                client_indices[int(rng.integers(0, num_clients))].append(idx)

    return client_indices


def _summarize(y: np.ndarray, client_indices: Sequence[Sequence[int]], num_classes: int) -> None:
    for cid, idxs in enumerate(client_indices):
        labels = y[list(idxs)]
        counts = np.bincount(labels, minlength=num_classes)
        nonzero = np.count_nonzero(counts)
        total = counts.sum()
        print(f"client {cid:02d}: samples={total}, classes={nonzero}, counts={counts.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MNIST and partition it into cross-silo client splits (FedCompass)."
    )
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument(
        "--strategy",
        choices=["class", "dual_dirichlet"],
        default="class",
        help="Partition strategy from FedCompass (class or dual_dirichlet).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data_partitions_mnist"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mu", type=float, default=10.0, help="Class partition: normal mean (mu).")
    parser.add_argument("--sigma", type=float, default=3.0, help="Class partition: normal std (sigma).")
    parser.add_argument("--nmin", type=int, default=None)
    parser.add_argument("--nmax", type=int, default=None)
    parser.add_argument("--alpha1", type=float, default=None, help="Dual Dirichlet: alpha1.")
    parser.add_argument("--alpha2", type=float, default=0.5, help="Dual Dirichlet: alpha2.")
    parser.add_argument(
        "--partition-test",
        action="store_true",
        help="Also partition the MNIST test set into per-client splits.",
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10"],
        default="mnist",
        help="Dataset to download/partition.",
    )

    args = parser.parse_args()

    from tensorflow.keras.datasets import mnist, cifar10

    if args.dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # CIFAR10 labels are shape (N, 1)
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

    num_classes = int(np.max(y_train)) + 1  # should be 10 for CIFAR10


    rng = np.random.default_rng(args.seed)
    if args.strategy == "class":
        if args.nmin is None or args.nmax is None:
            if args.num_clients == 5:
                nmin, nmax = 5, 6
            else:
                nmin, nmax = 3, 5
        else:
            nmin, nmax = args.nmin, args.nmax

        client_indices = _class_partition_indices(
            y_train,
            num_clients=args.num_clients,
            num_classes=num_classes,
            nmin=nmin,
            nmax=nmax,
            mu=args.mu,
            sigma=args.sigma,
            rng=rng,
        )
    else:
        alpha1 = args.alpha1 if args.alpha1 is not None else float(args.num_clients)
        client_indices = _dual_dirichlet_indices(
            y_train,
            num_clients=args.num_clients,
            num_classes=num_classes,
            alpha1=alpha1*5,
            alpha2=args.alpha2,
            rng=rng,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pad = max(5, len(str(args.num_clients)))
    for cid, idxs in enumerate(client_indices):
        idxs = np.array(idxs, dtype=int)
        np.savez_compressed(
            args.out_dir / f"client_{cid:0{pad}d}.npz",
            x_train=x_train[idxs],
            y_train=y_train[idxs],
        )

    np.savez_compressed(
        args.out_dir / "test.npz",
        x_test=x_test,
        y_test=y_test,
    )

    if args.partition_test:
        rng_test = np.random.default_rng(args.seed + 1)
        if args.strategy == "class":
            if args.nmin is None or args.nmax is None:
                if args.num_clients == 5:
                    nmin, nmax = 5, 6
                else:
                    nmin, nmax = 3, 5
            else:
                nmin, nmax = args.nmin, args.nmax

            test_indices = _class_partition_indices(
                y_test,
                num_clients=args.num_clients,
                num_classes=num_classes,
                nmin=nmin,
                nmax=nmax,
                mu=args.mu,
                sigma=args.sigma,
                rng=rng_test,
            )
        else:
            alpha1 = args.alpha1 if args.alpha1 is not None else float(args.num_clients)
            test_indices = _dual_dirichlet_indices(
                y_test,
                num_clients=args.num_clients,
                num_classes=num_classes,
                alpha1=alpha1*5,
                alpha2=args.alpha2,
                rng=rng_test,
            )

        for cid, idxs in enumerate(test_indices):
            idxs = np.array(idxs, dtype=int)
            path = args.out_dir / f"client_{cid:0{pad}d}.npz"
            with np.load(path) as data:
                payload = dict(data)
            payload["x_test"] = x_test[idxs]
            payload["y_test"] = y_test[idxs]
            np.savez_compressed(path, **payload)

    print(f"Saved {args.num_clients} client partitions + test set to {args.out_dir.resolve()}")
    print("Train split summary:")
    _summarize(y_train, client_indices, num_classes)


if __name__ == "__main__":
    main()
