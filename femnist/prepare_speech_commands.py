#!/usr/bin/env python3
"""Download and partition Speech Commands v0.02 into client NPZ manifests.

The script mirrors the ergonomics of prepare_mnist_cross_silo.py: it
creates client_XXXXX.npz shards plus a held-out test split.  Each NPZ
contains **paths** to .wav files (not precomputed features) and integer
labels so that training can stream directly from disk.
"""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request

import numpy as np

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DEFAULT_CLASSES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "cat",
    "dog",
]


def _download_archive(dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    print(f">>> Downloading Speech Commands archive to {dst} ...")
    with urllib.request.urlopen(DATA_URL) as resp, dst.open("wb") as fh:
        block = 1024 * 1024
        while True:
            chunk = resp.read(block)
            if not chunk:
                break
            fh.write(chunk)
    return dst


def _extract(archive: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".extracted"
    if marker.exists():
        return target
    print(f">>> Extracting {archive.name} to {target} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=target)
    marker.touch()
    return target


def _collect_files(root: Path, classes: List[str]) -> Tuple[List[Path], List[int]]:
    files: List[Path] = []
    labels: List[int] = []
    class_to_id: Dict[str, int] = {c: i for i, c in enumerate(classes)}
    for cls in classes:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Expected class directory missing: {cls_dir}")
        wavs = sorted(cls_dir.glob("*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No .wav files found in {cls_dir}")
        files.extend(wavs)
        labels.extend([class_to_id[cls]] * len(wavs))
    return files, labels


def _allocate_counts(total: int, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
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


def _partition_by_class(labels: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator) -> List[List[int]]:
    """Dirichlet per-class split -> controllable non-IID (smaller alpha = more skew)."""
    labels = np.asarray(labels)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for cls in np.unique(labels):
        cls_idxs = np.where(labels == cls)[0]
        rng.shuffle(cls_idxs)
        # Dirichlet draw for this class
        props = rng.dirichlet([alpha] * num_clients)
        counts = _allocate_counts(len(cls_idxs), props, rng)
        start = 0
        for cid, cnt in enumerate(counts):
            client_indices[cid].extend(cls_idxs[start : start + cnt])
            start += cnt
    # Shuffle each client subset to mix classes
    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
    return client_indices


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and partition Speech Commands v0.02")
    ap.add_argument("--out-dir", type=Path, default=Path("data_partitions_speech_commands"))
    ap.add_argument("--num-clients", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-split", type=float, default=0.8, help="Fraction of per-class files used for train (rest -> test)")
    ap.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES, help="Subset of classes to keep")
    ap.add_argument("--alpha", type=float, default=0.8, help="Dirichlet concentration for non-IID split (smaller = more skew)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    out_dir = args.out_dir
    raw_dir = out_dir / "raw"
    archive = raw_dir / "speech_commands_v0.02.tar.gz"
    _download_archive(archive)
    extracted_root = _extract(archive, raw_dir)
    candidate = extracted_root / "speech_commands_v0.02"
    dataset_root = candidate if candidate.exists() else extracted_root

    # Collect the requested class files
    classes = list(args.classes)
    files, labels = _collect_files(dataset_root, classes)
    files = np.array(files)
    labels = np.array(labels, dtype=np.int32)

    # Stratified per-class split into train/test
    train_mask = np.zeros(len(files), dtype=bool)
    for cls_idx, cls in enumerate(classes):
        cls_indices = np.where(labels == cls_idx)[0]
        rng.shuffle(cls_indices)
        split = int(len(cls_indices) * args.train_split)
        train_mask[cls_indices[:split]] = True
    train_files, train_labels = files[train_mask], labels[train_mask]
    test_files, test_labels = files[~train_mask], labels[~train_mask]

    print(f"Collected {len(train_files)} train and {len(test_files)} test audio files across {len(classes)} classes.")

    # Partition training files across clients (Dirichlet per-class for controllable non-IID)
    alpha = max(1e-4, float(args.alpha))
    client_indices = _partition_by_class(train_labels, args.num_clients, alpha, rng)

    pad = max(5, len(str(args.num_clients)))
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = []
    for cid, idxs in enumerate(client_indices):
        files_c = train_files[idxs]
        labels_c = train_labels[idxs]
        rel_files = [str(Path(f).relative_to(out_dir)) for f in files_c]
        np.savez_compressed(out_dir / f"client_{cid:0{pad}d}.npz", files=np.array(rel_files), labels=labels_c)
        sizes.append((len(files_c), np.bincount(labels_c, minlength=len(classes))))

    # Save test manifest and label map
    rel_test = [str(Path(f).relative_to(out_dir)) for f in test_files]
    np.savez_compressed(out_dir / "test_speech_commands.npz", files=np.array(rel_test), labels=test_labels)

    label_map_path = out_dir / "label_map.txt"
    label_map_path.write_text("\n".join(classes), encoding="utf-8")

    print(f"Saved {args.num_clients} client manifests and test set to {out_dir.resolve()}")
    for cid, (n, counts) in enumerate(sizes):
        print(f"client {cid:02d}: samples={n}, class_counts={counts.tolist()}")


if __name__ == "__main__":
    main()
