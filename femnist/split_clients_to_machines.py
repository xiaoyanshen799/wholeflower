#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def split_clients(
    src_dir: Path,
    dst_root: Path,
    num_machines: int,
) -> None:
    # 1) 收集并排序 client_*.npz
    clients = sorted(src_dir.glob("client_*.npz"))
    if not clients:
        raise SystemExit(f"No client_*.npz found in {src_dir}")
    total = len(clients)
    print(f"Found {total} client files")

    # 2) 计算每台机器应分多少个
    base = total // num_machines        # 每台至少这么多
    rem = total % num_machines          # 前 rem 台多 1 个
    print(f"Target: {num_machines} machines, base={base}, extra={rem}")

    dst_root.mkdir(parents=True, exist_ok=True)

    idx = 0
    for m in range(num_machines):
        # 当前机器这个组的大小
        group_size = base + (1 if m < rem else 0)
        machine_dir = dst_root / f"machine_{m:02d}"
        machine_dir.mkdir(parents=True, exist_ok=True)

        group = clients[idx : idx + group_size]
        idx += group_size

        print(f"[Machine {m:02d}] assigning {len(group)} clients")
        for f in group:
            shutil.copy2(f, machine_dir / f.name)

    print("\nDone. Per-machine folders created under:", dst_root)


def main():
    parser = argparse.ArgumentParser(
        description="Split FEMNIST client_*.npz into N machine directories."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        required=True,
        help="Directory containing client_*.npz",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("machines"),
        help="Output root directory for per-machine splits",
    )
    parser.add_argument(
        "--num-machines",
        type=int,
        default=24,
        help="Number of machines to split across",
    )
    args = parser.parse_args()

    split_clients(args.src_dir, args.dst_root, args.num_machines)


if __name__ == "__main__":
    main()
