#!/usr/bin/env python3
"""Inspect token lengths for all samples in a FlowerTune partition."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from flowertune_finance.dataset import (
    formatting_prompts_func,
    load_data,
    replace_keys,
    get_tokenizer_and_data_collator_and_propt_formatting,
)

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


def load_defaults(project_root: Path) -> dict:
    pyproject_path = project_root / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    cfg = data["tool"]["flwr"]["app"]["config"]
    static_cfg = cfg["static"]
    return replace_keys(
        {
            "model": {"name": cfg["model"]["name"]},
            "train": {"seq-length": cfg["train"]["seq-length"]},
            "static": {"dataset": {"name": static_cfg["dataset"]["name"]}},
        }
    )


def build_prompt(instruction: str, response: str) -> str:
    return formatting_prompts_func(
        {"instruction": [instruction], "response": [response]}
    )[0]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    defaults = load_defaults(project_root)

    parser = argparse.ArgumentParser(
        description="Export token lengths for every sample in a partition."
    )
    parser.add_argument(
        "--partition-id",
        type=int,
        default=0,
        help="Partition ID to inspect.",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=1,
        help="Number of partitions used when loading the dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=defaults["static"]["dataset"]["name"],
        help="Dataset name passed to FederatedDataset.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=defaults["model"]["name"],
        help="Tokenizer/model name.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=defaults["train"]["seq_length"],
        help="Training sequence length used for truncation.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="CSV output path. Default: partition_token_lengths_p<id>_of_<n>.csv",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print every sample length to stdout in addition to writing CSV.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of rows to print when --print-all is not set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_file = (
        args.output_file
        if args.output_file is not None
        else Path.cwd()
        / f"partition_token_lengths_p{args.partition_id}_of_{args.num_partitions}.csv"
    )

    trainset = load_data(args.partition_id, args.num_partitions, args.dataset_name)
    tokenizer, _, _ = get_tokenizer_and_data_collator_and_propt_formatting(
        args.model_name
    )

    rows: list[dict[str, int | str]] = []
    raw_lengths: list[int] = []
    effective_lengths: list[int] = []

    for idx, sample in enumerate(trainset):
        prompt = build_prompt(sample["instruction"], sample["response"])
        raw_token_len = len(
            tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        )
        effective_token_len = min(raw_token_len, args.seq_length)
        rows.append(
            {
                "sample_idx": idx,
                "raw_token_len": raw_token_len,
                "effective_token_len": effective_token_len,
                "was_truncated": int(raw_token_len > args.seq_length),
                "instruction_chars": len(sample["instruction"]),
                "response_chars": len(sample["response"]),
            }
        )
        raw_lengths.append(raw_token_len)
        effective_lengths.append(effective_token_len)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "raw_token_len",
                "effective_token_len",
                "was_truncated",
                "instruction_chars",
                "response_chars",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"partition_id={args.partition_id}")
    print(f"num_partitions={args.num_partitions}")
    print(f"num_samples={len(rows)}")
    print(f"seq_length={args.seq_length}")
    print(f"raw_token_len_min={min(raw_lengths)}")
    print(f"raw_token_len_max={max(raw_lengths)}")
    print(f"raw_token_len_mean={statistics.mean(raw_lengths):.2f}")
    print(f"effective_token_len_mean={statistics.mean(effective_lengths):.2f}")
    print(f"num_truncated={sum(1 for x in raw_lengths if x > args.seq_length)}")
    print(f"output_file={output_file}")

    preview_rows = rows if args.print_all else rows[: max(args.preview, 0)]
    for row in preview_rows:
        print(
            "sample_idx={sample_idx} raw_token_len={raw_token_len} "
            "effective_token_len={effective_token_len} was_truncated={was_truncated}".format(
                **row
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
