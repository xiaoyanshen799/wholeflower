from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    shard_path: Path
    num_samples: int


def discover_users(all_data_dir: Path) -> List[UserRecord]:
    """Scan LEAF FEMNIST all_data shards and collect (user, shard, num_samples)."""
    user_records: List[UserRecord] = []

    shard_files = sorted(all_data_dir.glob("all_data_*.json"))
    if not shard_files:
        raise FileNotFoundError(f"No shards found in {all_data_dir}")

    for shard in shard_files:
        with open(shard, "r") as f:
            data = json.load(f)
        users = data.get("users", [])
        num_samples = data.get("num_samples", [])
        if len(users) != len(num_samples):
            raise ValueError(
                f"Shard {shard} has mismatched users/num_samples lengths: "
                f"{len(users)} vs {len(num_samples)}"
            )
        for uid, n in zip(users, num_samples):
            user_records.append(UserRecord(user_id=uid, shard_path=shard, num_samples=int(n)))

    return user_records


def build_bins(user_records: List[UserRecord], bin_size: int) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], List[UserRecord]]]:
    """Create non-overlapping [lo, hi) bins and assign users to bins by num_samples."""
    if not user_records:
        return [], {}

    max_samples = max(u.num_samples for u in user_records)
    num_bins = max(1, math.ceil((max_samples + 1) / bin_size))

    bins: List[Tuple[int, int]] = []
    for i in range(num_bins):
        lo = i * bin_size
        hi = (i + 1) * bin_size
        bins.append((lo, hi))

    bin_map: Dict[Tuple[int, int], List[UserRecord]] = {b: [] for b in bins}

    for u in user_records:
        idx = min(u.num_samples // bin_size, num_bins - 1)
        b = bins[idx]
        bin_map[b].append(u)

    return bins, bin_map


def allocate_per_bin_targets(bin_map: Dict[Tuple[int, int], List[UserRecord]], total: int, rng: random.Random) -> Dict[Tuple[int, int], int]:
    """Allocate how many users to draw from each bin, proportional to bin sizes.

    Uses Hamilton's method (largest remainder). If a bin has fewer available users
    than allocated, re-distribute the deficit to other bins with capacity.
    """
    bins = list(bin_map.keys())
    sizes = np.array([len(bin_map[b]) for b in bins], dtype=float)
    total_clients = int(sizes.sum())
    if total_clients == 0:
        return {b: 0 for b in bins}

    quotas = (sizes / total_clients) * total
    base = np.floor(quotas).astype(int)
    rem = quotas - base
    allocated = base.copy()

    # Distribute remaining slots by largest remainders
    remaining = total - int(base.sum())
    order = list(range(len(bins)))
    order.sort(key=lambda i: rem[i], reverse=True)
    for i in order:
        if remaining <= 0:
            break
        allocated[i] += 1
        remaining -= 1

    # If any bin is over-allocated beyond its capacity, reclaim and redistribute
    surplus = 0
    for i, b in enumerate(bins):
        cap = len(bin_map[b])
        if allocated[i] > cap:
            surplus += allocated[i] - cap
            allocated[i] = cap

    if surplus > 0:
        # Create a list of bins with remaining capacity and distribute randomly
        capacity_left = [len(bin_map[b]) - allocated[i] for i, b in enumerate(bins)]
        candidates = [i for i, c in enumerate(capacity_left) if c > 0]
        while surplus > 0 and candidates:
            i = rng.choice(candidates)
            allocated[i] += 1
            capacity_left[i] -= 1
            surplus -= 1
            if capacity_left[i] == 0:
                candidates.remove(i)

    return {b: int(allocated[i]) for i, b in enumerate(bins)}


def select_users(bin_map: Dict[Tuple[int, int], List[UserRecord]], per_bin_targets: Dict[Tuple[int, int], int], rng: random.Random) -> List[UserRecord]:
    selected: List[UserRecord] = []
    for b, users in bin_map.items():
        k = int(per_bin_targets.get(b, 0))
        if k <= 0 or not users:
            continue
        # sample without replacement
        chosen = rng.sample(users, k=min(k, len(users)))
        selected.extend(chosen)
    return selected


def load_selected_user_data(selected: List[UserRecord]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load user_data for the selected users, grouped by shard for efficiency."""
    by_shard: Dict[Path, List[UserRecord]] = defaultdict(list)
    for rec in selected:
        by_shard[rec.shard_path].append(rec)

    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for shard_path, recs in by_shard.items():
        with open(shard_path, "r") as f:
            data = json.load(f)
        user_data = data["user_data"]
        for rec in recs:
            ud = user_data[rec.user_id]
            x_flat: List[List[int]] = ud["x"]
            y_list: List[int] = ud["y"]
            x = np.asarray(x_flat, dtype=np.uint8)
            # reshape to (n, 28, 28, 1) from (n, 784)
            if x.ndim != 2 or x.shape[1] != 784:
                raise ValueError(f"Unexpected shape for user {rec.user_id}: {x.shape}")
            x = x.reshape((-1, 28, 28, 1))
            y = np.asarray(y_list, dtype=np.int64)
            result[rec.user_id] = (x, y)

    return result


def save_npz_partitions(output_dir: Path, selected_data: Dict[str, Tuple[np.ndarray, np.ndarray]], seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Assign sequential cids 0..N-1 in a deterministic order
    users_sorted = sorted(selected_data.keys())
    pad = max(5, len(str(len(users_sorted) - 1)))
    for cid, uid in enumerate(users_sorted):
        x, y = selected_data[uid]
        out_file = output_dir / f"client_{cid:0{pad}d}.npz"
        np.savez_compressed(out_file, x_train=x, y_train=y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample 20 FEMNIST clients proportionally by sample-count bins and export NPZs.")
    parser.add_argument("--leaf-root", type=Path, default=Path(__file__).parent / "leaf" / "data" / "femnist" / "data" / "all_data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data_partitions1")
    parser.add_argument("--bin-size", type=int, default=100, help="Bin width for sample counts (e.g., 100 => 0-100, 100-200, ...)")
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1) Discover all users with their sample counts
    user_records = discover_users(args.leaf_root)

    # 2) Build bins and print stats
    bins, bin_map = build_bins(user_records, args.bin_size)
    print("Total users:", len(user_records))
    print("Bin counts:")
    for lo, hi in bins:
        cnt = len(bin_map[(lo, hi)])
        print(f"  {lo}-{hi}: {cnt}")

    # 3) Allocate proportional targets and sample users
    per_bin_targets = allocate_per_bin_targets(bin_map, args.num_clients, rng)
    print("Per-bin targets:", per_bin_targets)
    selected = select_users(bin_map, per_bin_targets, rng)

    # If due to rounding/capacity issues we didn't reach the target, fill globally
    if len(selected) < args.num_clients:
        already = set((r.user_id for r in selected))
        remaining_pool = [r for r in user_records if r.user_id not in already]
        need = args.num_clients - len(selected)
        selected.extend(rng.sample(remaining_pool, k=need))

    # 4) Load selected users' data
    selected_data = load_selected_user_data(selected)

    # 5) Save NPZ partitions with sequential cids
    save_npz_partitions(args.output_dir, selected_data, args.seed)

    # 6) Print summary of selected users
    print("Selected users (sorted):")
    users_sorted = sorted(selected_data.items(), key=lambda kv: kv[0])
    for idx, (uid, (x, y)) in enumerate(users_sorted):
        print(f"  cid={idx:02d} uid={uid} num_samples={len(x)}")


if __name__ == "__main__":
    main()


