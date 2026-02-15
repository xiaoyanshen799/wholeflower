#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _num_examples_from_npz(path: Path) -> int:
    """
    Return the number of examples stored in a client npz file.

    Different data preparation scripts save slightly different keys. We try a
    few common ones in order and raise a helpful error if none are found.
    """
    with np.load(path, allow_pickle=True) as d:
        if "x_train" in d:  # original FEMNIST format
            return int(d["x_train"].shape[0])
        if "files" in d:  # speech commands partition stores file paths + labels
            return int(d["files"].shape[0])
        if "labels" in d:  # fallback: labels length should match examples
            return int(d["labels"].shape[0])
    raise KeyError(
        f"Could not determine number of examples in {path.name}; expected one of "
        "'x_train', 'files', or 'labels' in the archive."
    )


def count_client_samples(data_dir: Path) -> dict[int, int]:
    counts = {}
    for f in sorted(data_dir.glob("client_*.npz")):
        cid_str = f.stem.replace("client_", "")
        try:
            cid = int(cid_str)
        except ValueError:
            continue
        n = _num_examples_from_npz(f)
        counts[cid] = n
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map local client_*.npz sizes to client ids."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with client_*.npz")
    parser.add_argument("--csv-out", type=Path, required=True, help="Output CSV path")
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Train split ratio used in run_client (default: 0.9).",
    )
    args = parser.parse_args()

    counts = count_client_samples(args.data_dir)
    if not counts:
        raise SystemExit(f"No client_*.npz found in {args.data_dir}")

    with args.csv_out.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(
            f_out, fieldnames=["client_id", "num_examples_total", "num_examples_train"]
        )
        writer.writeheader()
        for cid in sorted(counts):
            total = counts[cid]
            train_n = int(total * args.train_split)
            writer.writerow(
                {
                    "client_id": cid,
                    "num_examples_total": total,
                    "num_examples_train": train_n,
                }
            )

    print(f"Wrote {args.csv_out}")


if __name__ == "__main__":
    main()
