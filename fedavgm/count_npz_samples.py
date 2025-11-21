#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional

import numpy as np


def infer_sample_count(npz_file: np.lib.npyio.NpzFile) -> int:
    preferred_keys = ["x", "X", "images", "data"]

    for key in preferred_keys:
        if key in npz_file.files:
            arr = npz_file[key]
            count = _count_from_array(arr)
            if count is not None:
                return count

    counts = []
    for key in npz_file.files:
        arr = npz_file[key]
        count = _count_from_array(arr)
        if count is not None:
            counts.append(count)

    if counts:
        return int(max(counts))

    return 0


def _count_from_array(arr) -> Optional[int]:
    try:
        shape = getattr(arr, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
        return int(len(arr))
    except Exception:
        return None


def count_npz_in_directory(directory_path: str) -> list[tuple[str, int]]:
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    results: list[tuple[str, int]] = []
    for entry in os.scandir(directory_path):
        if entry.is_file() and entry.name.endswith(".npz"):
            npz_path = entry.path
            try:
                with np.load(npz_path, allow_pickle=False) as npz:
                    count = infer_sample_count(npz)
                results.append((entry.name, count))
            except Exception as e:
                results.append((entry.name, -1))
                print(
                    f"Warning: failed to read {entry.name}: {e}",
                    file=sys.stderr,
                )

    results.sort(key=lambda x: x[0])
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Count samples in each .npz file in a directory and print 'filename count' per line."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="@data_partitions1/",
        help="Directory containing client .npz files (default: @data_partitions1/)",
    )
    args = parser.parse_args()

    results = count_npz_in_directory(args.directory)
    for filename, count in results:
        print(f"{filename}\t{count}")


if __name__ == "__main__":
    main()


