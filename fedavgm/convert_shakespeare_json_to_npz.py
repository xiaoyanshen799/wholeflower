from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path


def encode_str_batch(str_list, seq_len: int = 80) -> np.ndarray:
    arr = np.zeros((len(str_list), seq_len), dtype=np.uint16)
    for i, s in enumerate(str_list):
        # store Unicode code points (covers quotes/dashes etc.)
        cp = [ord(ch) for ch in s[:seq_len]]
        arr[i, : len(cp)] = np.asarray(cp, dtype=np.uint16)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert per-client Shakespeare JSON to NPZ (x_train,y_train)")
    ap.add_argument("--src", type=Path, default=Path("data_partitions2"))
    ap.add_argument("--dst", type=Path, default=Path("data_partitions2"))
    args = ap.parse_args()

    src = args.src
    dst = args.dst
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("client_*.json"))
    pad = max(5, len(str(len(files) - 1)))
    for idx, f in enumerate(files):
        d = json.load(open(f))
        ud = d["user_data"]
        Xs = ud["x"]
        Ys = ud["y"]
        x_arr = encode_str_batch(Xs, seq_len=80)
        y_arr = np.asarray([ord(ch) for ch in Ys], dtype=np.uint16)
        out = dst / f"client_{idx:0{pad}d}.npz"
        np.savez_compressed(out, x_train=x_arr, y_train=y_arr)

    # Optional: centralized test not created here
    print("Converted", len(files), "clients to NPZ in", str(dst))


if __name__ == "__main__":
    main()


