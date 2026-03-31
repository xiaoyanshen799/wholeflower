"""flowertune-finance: A Flower / FlowerTune app."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

FDS = None  # Cache FederatedDataset


def formatting_prompts_func(example):
    """Construct prompts."""
    output_texts = []
    # Constructing a standard Alpaca
    # (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )
    for i in range(len(example["instruction"])):
        text = (
            f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n"
            f"### Response: {example['response'][i]}"
        )
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str):
    """Get tokenizer, data_collator and prompt formatting."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    return tokenizer, data_collator, formatting_prompts_func


def formatting(dataset):
    """Format dataset."""
    dataset["instruction"] = dataset["instruction"] + " " + dataset["input"]
    return dataset


def reformat(dataset, llm_task):
    """Reformat datasets."""
    dataset = dataset.rename_column("output", "response")
    if llm_task in ["finance", "code"]:
        dataset = dataset.map(formatting, remove_columns=["input"])
    if llm_task == "medical":
        dataset = dataset.remove_columns(["instruction"])
        dataset = dataset.rename_column("input", "instruction")
    return dataset


def add_sample_indices(dataset):
    """Attach a stable sample index to each row in the current partition."""
    if "sample_idx" in dataset.column_names:
        return dataset
    return dataset.add_column("sample_idx", list(range(len(dataset))))


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_name: str,
    split: str = "train",
    eval_fraction: float = 0.1,
    split_seed: int = 42,
):
    """Load partition data and return a deterministic train/test subset."""
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    client_dataset = FDS.load_partition(partition_id, "train")
    client_dataset = reformat(client_dataset, llm_task="finance")
    client_dataset = add_sample_indices(client_dataset)

    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    if len(client_dataset) < 2 or eval_fraction <= 0.0:
        return client_dataset

    test_size = max(1, int(round(len(client_dataset) * eval_fraction)))
    test_size = min(test_size, len(client_dataset) - 1)
    split_datasets = client_dataset.train_test_split(
        test_size=test_size,
        seed=split_seed,
        shuffle=True,
    )
    return split_datasets[split]


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
