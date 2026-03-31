"""flowertune-medical: A Flower / FlowerTune app."""

from datasets import DatasetDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import ContinuousPartitioner, IidPartitioner
from flwr_datasets.partitioner.partitioner import Partitioner
import numpy as np
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

FDS = None  # Cache FederatedDataset


class UnequalContinuousPartitioner(Partitioner):
    """Continuous non-IID partitioner with unequal capped partition sizes."""

    def __init__(
        self,
        num_partitions: int,
        partition_by: str,
        strictness: float,
        min_partition_size: int,
        max_partition_size: int | None,
        size_skew: float,
        shuffle: bool = True,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        if not 0 <= strictness <= 1:
            raise ValueError("strictness must be between 0 and 1")
        if num_partitions <= 0:
            raise ValueError("num_partitions must be greater than 0")
        if min_partition_size <= 0:
            raise ValueError("min_partition_size must be greater than 0")
        if max_partition_size is not None and max_partition_size < min_partition_size:
            raise ValueError("max_partition_size must be >= min_partition_size")
        if size_skew <= 0:
            raise ValueError("size_skew must be greater than 0")

        self._num_partitions = num_partitions
        self._partition_by = partition_by
        self._strictness = strictness
        self._min_partition_size = min_partition_size
        self._max_partition_size = max_partition_size
        self._size_skew = size_skew
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_indices_determined = False

    def load_partition(self, partition_id: int):
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    def _determine_partition_sizes(self, dataset_size: int) -> list[int]:
        if self._min_partition_size * self._num_partitions > dataset_size:
            raise ValueError(
                "Dataset is too small for the requested min_partition_size and num_partitions."
            )

        target_total = dataset_size
        if self._max_partition_size is not None:
            target_total = min(target_total, self._max_partition_size * self._num_partitions)

        base_sizes = [self._min_partition_size] * self._num_partitions
        remaining = target_total - sum(base_sizes)
        if remaining <= 0:
            return base_sizes

        weights = np.linspace(1.0, self._size_skew, self._num_partitions, dtype=float)
        weights = weights / weights.sum()
        extras = np.floor(weights * remaining).astype(int)
        sizes = [base + int(extra) for base, extra in zip(base_sizes, extras)]

        remainder = target_total - sum(sizes)
        order = np.argsort(-weights)
        for idx in range(remainder):
            sizes[int(order[idx % self._num_partitions])] += 1

        if self._max_partition_size is not None:
            sizes = [min(size, self._max_partition_size) for size in sizes]

        return sizes

    def _determine_partition_id_to_indices_if_needed(self) -> None:
        if self._partition_id_to_indices_determined:
            return

        dataset_size = self.dataset.num_rows
        partition_sizes = self._determine_partition_sizes(dataset_size)
        target_total = sum(partition_sizes)

        all_indices = np.arange(dataset_size)
        if target_total < dataset_size:
            selected_indices = self._rng.choice(
                all_indices, size=target_total, replace=False
            )
            selected_indices = np.sort(selected_indices)
        else:
            selected_indices = all_indices

        property_values = np.array(
            [self.dataset[int(idx)][self._partition_by] for idx in selected_indices],
            dtype=np.float32,
        )

        std = np.std(property_values)
        if std < 1e-6:
            blended_values = self._rng.normal(loc=0.0, scale=1.0, size=len(property_values))
        else:
            standardized_values = (property_values - np.mean(property_values)) / std
            noise = self._rng.normal(loc=0.0, scale=1.0, size=len(standardized_values))
            blended_values = (
                self._strictness * standardized_values + (1 - self._strictness) * noise
            )

        sorted_local = np.argsort(blended_values)
        sorted_indices = selected_indices[sorted_local]

        start = 0
        for partition_id, size in enumerate(partition_sizes):
            end = start + size
            indices = sorted_indices[start:end].tolist()
            if self._shuffle:
                self._rng.shuffle(indices)
            self._partition_id_to_indices[partition_id] = indices
            start = end

        self._partition_id_to_indices_determined = True


class FixedSizeContinuousPartitioner(Partitioner):
    """Continuous non-IID partitioner with explicit partition sizes."""

    def __init__(
        self,
        partition_sizes: list[int],
        partition_by: str,
        strictness: float,
        shuffle: bool = True,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        if not partition_sizes:
            raise ValueError("partition_sizes must not be empty")
        if any(size <= 0 for size in partition_sizes):
            raise ValueError("All partition_sizes must be greater than 0")
        if not 0 <= strictness <= 1:
            raise ValueError("strictness must be between 0 and 1")

        self._partition_sizes = [int(size) for size in partition_sizes]
        self._partition_by = partition_by
        self._strictness = strictness
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_indices_determined = False

    def load_partition(self, partition_id: int):
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        return len(self._partition_sizes)

    def _determine_partition_id_to_indices_if_needed(self) -> None:
        if self._partition_id_to_indices_determined:
            return

        dataset_size = self.dataset.num_rows
        target_total = sum(self._partition_sizes)
        if target_total > dataset_size:
            raise ValueError(
                f"Sum of partition_sizes ({target_total}) exceeds dataset size ({dataset_size})."
            )

        all_indices = np.arange(dataset_size)
        if target_total < dataset_size:
            selected_indices = self._rng.choice(
                all_indices, size=target_total, replace=False
            )
            selected_indices = np.sort(selected_indices)
        else:
            selected_indices = all_indices

        property_values = np.array(
            [self.dataset[int(idx)][self._partition_by] for idx in selected_indices],
            dtype=np.float32,
        )
        std = np.std(property_values)
        if std < 1e-6:
            blended_values = self._rng.normal(loc=0.0, scale=1.0, size=len(property_values))
        else:
            standardized_values = (property_values - np.mean(property_values)) / std
            noise = self._rng.normal(loc=0.0, scale=1.0, size=len(standardized_values))
            blended_values = (
                self._strictness * standardized_values + (1 - self._strictness) * noise
            )

        sorted_local = np.argsort(blended_values)
        sorted_indices = selected_indices[sorted_local]

        start = 0
        for partition_id, size in enumerate(self._partition_sizes):
            end = start + size
            indices = sorted_indices[start:end].tolist()
            if self._shuffle:
                self._rng.shuffle(indices)
            self._partition_id_to_indices[partition_id] = indices
            start = end

        self._partition_id_to_indices_determined = True


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


def _add_input_length_feature(dataset_dict: DatasetDict) -> DatasetDict:
    """Add a continuous feature used for non-IID partitioning."""

    def _map_input_length(example):
        return {"partition_input_length": float(len(example["input"]))}

    dataset_dict["train"] = dataset_dict["train"].map(_map_input_length)
    return dataset_dict


def _build_partitioner(num_partitions: int, partitioning_config):
    """Build a dataset partitioner from runtime config."""
    mode = str(partitioning_config.get("mode", "iid")).lower()
    if mode == "iid":
        return IidPartitioner(num_partitions=num_partitions), None
    if mode == "input_length":
        strictness = float(partitioning_config.get("strictness", 1.0))
        seed = partitioning_config.get("seed", 42)
        explicit_partition_sizes = partitioning_config.get("partition_sizes", None)
        if explicit_partition_sizes not in (None, ""):
            if isinstance(explicit_partition_sizes, str):
                normalized = explicit_partition_sizes.strip().strip("[]")
                partition_sizes = [
                    int(size.strip())
                    for size in normalized.split(",")
                    if size.strip()
                ]
            else:
                partition_sizes = [int(size) for size in explicit_partition_sizes]
            if len(partition_sizes) != num_partitions:
                raise ValueError(
                    "partitioning.partition_sizes length must match num_partitions. "
                    f"Got {len(partition_sizes)} vs {num_partitions}."
                )
            return (
                FixedSizeContinuousPartitioner(
                    partition_sizes=partition_sizes,
                    partition_by="partition_input_length",
                    strictness=strictness,
                    shuffle=True,
                    seed=None if seed is None else int(seed),
                ),
                _add_input_length_feature,
            )
        unequal_partitions = bool(partitioning_config.get("unequal_partitions", False))
        min_partition_size = int(partitioning_config.get("min_partition_size", 2000))
        max_partition_size = partitioning_config.get("max_partition_size", None)
        max_partition_size = (
            None if max_partition_size in (None, "", "none") else int(max_partition_size)
        )
        size_skew = float(partitioning_config.get("size_skew", 2.0))
        partitioner_cls = (
            UnequalContinuousPartitioner if unequal_partitions else ContinuousPartitioner
        )
        return (
            partitioner_cls(
                num_partitions=num_partitions,
                partition_by="partition_input_length",
                strictness=strictness,
                shuffle=True,
                seed=None if seed is None else int(seed),
                **(
                    {
                        "min_partition_size": min_partition_size,
                        "max_partition_size": max_partition_size,
                        "size_skew": size_skew,
                    }
                    if unequal_partitions
                    else {}
                ),
            ),
            _add_input_length_feature,
        )
    raise ValueError(
        f"Unsupported partitioning.mode={mode!r}. Supported values: 'iid', 'input_length'."
    )


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_name: str,
    partitioning_config=None,
):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    if partitioning_config is None:
        partitioning_config = {}
    if FDS is None:
        partitioner, preprocessor = _build_partitioner(
            num_partitions=num_partitions,
            partitioning_config=partitioning_config,
        )
        FDS = FederatedDataset(
            dataset=dataset_name,
            preprocessor=preprocessor,
            partitioners={"train": partitioner},
        )
    client_trainset = FDS.load_partition(partition_id, "train")
    client_trainset = reformat(client_trainset, llm_task="medical")
    return client_trainset


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
