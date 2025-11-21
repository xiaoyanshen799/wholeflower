from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


SPEAKER_RE = re.compile(r"^\s{0,6}([A-Z][A-Za-z'\- ]+)\.\s+(.*)")


def parse_raw_shakespeare(raw_path: Path) -> Dict[str, List[str]]:
    """Parse raw Shakespeare text into per-speaker lists of lines.

    Heuristic: a speaker line starts with optional spaces, then a capitalized
    name (can include spaces, hyphen, apostrophe), followed by a period and a
    space, then text. Continuation lines are appended until next speaker.
    """
    text = raw_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    speaker_to_lines: Dict[str, List[str]] = defaultdict(list)
    current = None

    for ln in lines:
        m = SPEAKER_RE.match(ln)
        if m:
            current = m.group(1).strip().upper()
            content = m.group(2).strip()
            if content:
                speaker_to_lines[current].append(content)
            continue
        # continuation
        if current:
            s = ln.strip()
            if s:
                speaker_to_lines[current].append(s)

    # Drop very short speakers (fewer than 2 lines)
    speaker_to_lines = {k: v for k, v in speaker_to_lines.items() if len(v) >= 2}
    return speaker_to_lines


def make_seq_xy(lines: List[str], seq_len: int = 80) -> Tuple[List[str], List[str]]:
    text = " ".join(lines)
    # normalize whitespace
    text = re.sub(r"\s+", " ", text)
    X: List[str] = []
    Y: List[str] = []
    if len(text) <= seq_len:
        return X, Y
    for i in range(0, len(text) - seq_len):
        X.append(text[i : i + seq_len])
        Y.append(text[i + seq_len])
    return X, Y


@dataclass(frozen=True)
class UserRec:
    uid: str
    num_samples: int


def build_bins(records: List[UserRec], bin_size: int) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], List[UserRec]]]:
    if not records:
        return [], {}
    mx = max(r.num_samples for r in records)
    bins: List[Tuple[int, int]] = []
    for lo in range(0, ((mx // bin_size) + 1) * bin_size, bin_size):
        bins.append((lo, lo + bin_size))
    bmap: Dict[Tuple[int, int], List[UserRec]] = {b: [] for b in bins}
    for r in records:
        idx = min(r.num_samples // bin_size, len(bins) - 1)
        bmap[bins[idx]].append(r)
    return bins, bmap


def allocate(bmap: Dict[Tuple[int, int], List[UserRec]], total: int, rng: random.Random) -> Dict[Tuple[int, int], int]:
    bins = list(bmap.keys())
    sizes = [len(bmap[b]) for b in bins]
    S = sum(sizes)
    if S == 0:
        return {b: 0 for b in bins}
    quotas = [s * total / S for s in sizes]
    base = [math.floor(q) for q in quotas]
    rem = [q - b for q, b in zip(quotas, base)]
    left = total - sum(base)
    order = sorted(range(len(bins)), key=lambda i: rem[i], reverse=True)
    for i in order:
        if left <= 0:
            break
        base[i] += 1
        left -= 1
    # cap by capacity, redistribute if needed
    surplus = 0
    for i, b in enumerate(bins):
        cap = len(bmap[b])
        if base[i] > cap:
            surplus += base[i] - cap
            base[i] = cap
    if surplus > 0:
        caps = [len(bmap[b]) - base[i] for i, b in enumerate(bins)]
        candidates = [i for i, c in enumerate(caps) if c > 0]
        while surplus > 0 and candidates:
            i = rng.choice(candidates)
            base[i] += 1
            caps[i] -= 1
            surplus -= 1
            if caps[i] == 0:
                candidates.remove(i)
    return {bins[i]: base[i] for i in range(len(bins))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample 20 Shakespeare clients proportionally by 0-50 bins and export per-client JSONs.")
    ap.add_argument("--raw-path", type=Path, default=Path("leaf/data/shakespeare/data/raw_data/raw_data.txt"))
    ap.add_argument("--output-dir", type=Path, default=Path("data_partitions2"))
    ap.add_argument("--bin-size", type=int, default=50)
    ap.add_argument("--num-clients", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    speakers = parse_raw_shakespeare(args.raw_path)
    # Build user dataset
    user_to_xy: Dict[str, Tuple[List[str], List[str]]] = {}
    for uid, lines in speakers.items():
        X, Y = make_seq_xy(lines, seq_len=80)
        if len(X) > 0:
            user_to_xy[uid] = (X, Y)

    records = [UserRec(uid=u, num_samples=len(y)) for u, (_, y) in user_to_xy.items()]
    print("Total users with samples:", len(records))
    if not records:
        print("No users found. Aborting.")
        return
    bins, bmap = build_bins(records, args.bin_size)
    print("Bin counts:")
    for lo, hi in bins:
        print(f"  {lo}-{hi}: {len(bmap[(lo, hi)])}")
    per_bin = allocate(bmap, args.num_clients, rng)
    selected: List[UserRec] = []
    for b, k in per_bin.items():
        if k <= 0:
            continue
        pool = bmap[b]
        if pool:
            chosen = rng.sample(pool, k=min(k, len(pool)))
            selected.extend(chosen)
    if len(selected) < args.num_clients:
        left_pool = [r for r in records if r.uid not in {s.uid for s in selected}]
        need = args.num_clients - len(selected)
        if left_pool:
            selected.extend(rng.sample(left_pool, k=min(need, len(left_pool))))

    # Save per-client JSONs
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pad = max(5, len(str(max(0, len(selected) - 1))))
    selected_sorted = sorted(selected, key=lambda r: r.uid)
    for cid, rec in enumerate(selected_sorted):
        X, Y = user_to_xy[rec.uid]
        payload = {
            "user": rec.uid,
            "num_samples": rec.num_samples,
            "user_data": {"x": X, "y": Y},
        }
        with (out / f"client_{cid:0{pad}d}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    print("Selected users:")
    for cid, rec in enumerate(selected_sorted):
        print(f"  cid={cid:02d} uid={rec.uid} num_samples={rec.num_samples}")


if __name__ == "__main__":
    main()


