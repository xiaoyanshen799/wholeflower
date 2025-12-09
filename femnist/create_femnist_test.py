#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def collect_client_files(data_dir: Path):
    """Return sorted list of client_*.npz files."""
    files = sorted(data_dir.glob("client_*.npz"))
    if not files:
        raise SystemExit(f"No client_*.npz found in {data_dir}")
    return files


def combine_client_tests(client_files):
    """Load all clients' test splits and concatenate."""
    all_x = []
    all_y = []

    for f in client_files:
        with np.load(f) as d:
            if "x_test" not in d or "y_test" not in d:
                raise ValueError(
                    f"File {f} does not contain x_test/y_test. "
                    f"Ensure you saved test splits in your client NPZ."
                )

            x_te = d["x_test"]
            y_te = d["y_test"]

            if len(x_te) == 0:
                print(f"[Skip] {f.name}: no test samples")
                continue

            all_x.append(x_te)
            all_y.append(y_te)
            print(f"[OK] {f.name}: loaded {len(x_te)} test samples")

    if not all_x:
        raise SystemExit("No test samples found in any client NPZ!")

    x_test = np.concatenate(all_x, axis=0)
    y_test = np.concatenate(all_y, axis=0)
    return x_test, y_test


def save_test_npz(output_path: Path, x_test: np.ndarray, y_test: np.ndarray):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, x_test=x_test, y_test=y_test)
    print(f"\n[Done] Saved centralized test set to {output_path}")
    print(f"Total samples: {x_test.shape[0]}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-client FEMNIST x_test/y_test into a single centralized test NPZ."
    )
    parser.add_argument(
        "--client-dir",
        type=Path,
        required=True,
        help="Directory containing client_*.npz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("centralized_femnist_test.npz"),
        help="Output NPZ path."
    )

    args = parser.parse_args()

    client_files = collect_client_files(args.client_dir)
    x_test, y_test = combine_client_tests(client_files)
    save_test_npz(args.output, x_test, y_test)


if __name__ == "__main__":
    main()
