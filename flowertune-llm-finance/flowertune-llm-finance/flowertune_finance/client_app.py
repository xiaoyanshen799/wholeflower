"""flowertune-finance: A Flower / FlowerTune app."""

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from math import exp, isfinite

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import torch
from transformers import TrainingArguments
from trl import SFTTrainer

from flowertune_finance.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    replace_keys,
)
from flowertune_finance.models import cosine_annealing, get_model

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class BatchTraceRecorder:
    """Collect token statistics for batches actually consumed by training_step."""

    step_sample_indices: list[list[int]] = field(default_factory=list)
    step_sample_input_tokens: list[list[int]] = field(default_factory=list)
    step_sample_target_tokens: list[list[int]] = field(default_factory=list)
    step_input_tokens: list[int] = field(default_factory=list)
    step_target_tokens: list[int] = field(default_factory=list)

    def record(
        self,
        sample_indices: list[int],
        sample_input_tokens: list[int],
        sample_target_tokens: list[int],
    ) -> None:
        self.step_sample_indices.append(sample_indices)
        self.step_sample_input_tokens.append(sample_input_tokens)
        self.step_sample_target_tokens.append(sample_target_tokens)
        self.step_input_tokens.append(sum(sample_input_tokens))
        self.step_target_tokens.append(sum(sample_target_tokens))

    @property
    def total_input_tokens(self) -> int:
        return sum(self.step_input_tokens)

    @property
    def total_target_tokens(self) -> int:
        return sum(self.step_target_tokens)


class TraceAnnotatingCollator:
    """Wrap the training data collator and attach per-sample trace metadata."""

    _MODEL_KEYS = {
        "input_ids",
        "attention_mask",
        "labels",
        "token_type_ids",
        "special_tokens_mask",
    }

    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        sample_indices = [int(feature["sample_idx"]) for feature in features]
        sample_input_tokens = [len(feature["input_ids"]) for feature in features]
        sanitized_features = [
            {key: value for key, value in feature.items() if key in self._MODEL_KEYS}
            for feature in features
        ]
        batch = self.base_collator(sanitized_features)
        sample_target_tokens = (batch["labels"] != -100).sum(dim=1).to(torch.int64)
        batch["trace_sample_idx"] = torch.tensor(sample_indices, dtype=torch.int64)
        batch["trace_sample_input_tokens"] = torch.tensor(
            sample_input_tokens, dtype=torch.int64
        )
        batch["trace_sample_target_tokens"] = sample_target_tokens
        return batch


class TracingSFTTrainer(SFTTrainer):
    """SFTTrainer that records only batches actually used by training_step."""

    def __init__(self, *args, trace_recorder: BatchTraceRecorder, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_recorder = trace_recorder

    def training_step(self, model, inputs, num_items_in_batch=None):
        sample_indices = inputs.pop("trace_sample_idx", None)
        sample_input_tokens = inputs.pop("trace_sample_input_tokens", None)
        sample_target_tokens = inputs.pop("trace_sample_target_tokens", None)
        if (
            sample_indices is not None
            and sample_input_tokens is not None
            and sample_target_tokens is not None
        ):
            self.trace_recorder.record(
                sample_indices.detach().cpu().tolist(),
                sample_input_tokens.detach().cpu().tolist(),
                sample_target_tokens.detach().cpu().tolist(),
            )
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Parse config
    server_round = int(msg.content["config"]["server-round"])
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    training_arguments = TrainingArguments(**cfg.train.training_arguments)
    training_arguments.remove_unused_columns = False
    use_dynamic_data_seed = cfg.train.get("dynamic_data_seed", False)

    # Let's get the client partition
    trainset = load_data(partition_id, 50, cfg.static.dataset.name)
    if use_dynamic_data_seed:
        base_seed = int(
            training_arguments.data_seed
            if training_arguments.data_seed is not None
            else training_arguments.seed
        )
        training_arguments.data_seed = base_seed + server_round - 1
    else:
        samples_per_round = (
            training_arguments.per_device_train_batch_size
            * training_arguments.gradient_accumulation_steps
            * training_arguments.max_steps
        )
        if samples_per_round > 0 and len(trainset) > 0:
            num_full_chunks = len(trainset) // samples_per_round
            if num_full_chunks > 0:
                chunk_idx = (server_round - 1) % num_full_chunks
                start = chunk_idx * samples_per_round
                end = start + samples_per_round
                trainset = trainset.select(list(range(start, end)))
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    # Load the model and initialize it with the received weights
    model = get_model(cfg.model)
    local_round_start = time.perf_counter()
    set_peft_model_state_dict(model, msg.content["arrays"].to_torch_state_dict())

    # Set learning rate for current round
    new_lr = cosine_annealing(
        server_round,
        num_rounds,
        cfg.train.learning_rate_max,
        cfg.train.learning_rate_min,
    )

    training_arguments.learning_rate = new_lr
    training_arguments.output_dir = msg.content["config"]["save_path"]
    trace_recorder = BatchTraceRecorder()
    trace_collator = TraceAnnotatingCollator(data_collator)

    # Construct trainer
    trainer = TracingSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=cfg.train.seq_length,
        train_dataset=trainset,
        formatting_func=formatting_prompts_func,
        data_collator=trace_collator,
        trace_recorder=trace_recorder,
    )

    # Do local training
    results = trainer.train()
    t_local_round_s = time.perf_counter() - local_round_start
    trace_payload = {
        "server_round": server_round,
        "partition_id": int(partition_id),
        "round_slice_size": len(trainset),
        "num_steps_traced": len(trace_recorder.step_sample_indices),
        "total_input_tokens": trace_recorder.total_input_tokens,
        "total_target_tokens": trace_recorder.total_target_tokens,
        "step_input_tokens": trace_recorder.step_input_tokens,
        "step_target_tokens": trace_recorder.step_target_tokens,
        "step_sample_indices": trace_recorder.step_sample_indices,
        "step_sample_input_tokens": trace_recorder.step_sample_input_tokens,
        "step_sample_target_tokens": trace_recorder.step_sample_target_tokens,
    }
    print(f"ROUND_BATCH_TRACE {json.dumps(trace_payload, separators=(',', ':'))}")

    # Construct and return reply Message
    model_record = ArrayRecord(get_peft_model_state_dict(model))
    metrics = {
        "train_loss": results.training_loss,
        "num-examples": len(trainset),
        "t_local_round_s": t_local_round_s,
        "train_total_input_tokens": trace_recorder.total_input_tokens,
        "train_total_target_tokens": trace_recorder.total_target_tokens,
        "train_num_steps_traced": len(trace_recorder.step_sample_indices),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the global model on a deterministic held-out local test split."""
    partition_id = context.node_config["partition-id"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    training_arguments = TrainingArguments(**cfg.train.training_arguments)
    training_arguments.remove_unused_columns = False
    if training_arguments.per_device_eval_batch_size is None:
        training_arguments.per_device_eval_batch_size = (
            training_arguments.per_device_train_batch_size
        )

    testset = load_data(
        partition_id,
        50,
        cfg.static.dataset.name,
        split="test",
        eval_fraction=cfg.eval.test_fraction,
    )
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    model = get_model(cfg.model)
    set_peft_model_state_dict(model, msg.content["arrays"].to_torch_state_dict())

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=cfg.train.seq_length,
        train_dataset=None,
        eval_dataset=testset,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )
    results = trainer.evaluate(eval_dataset=testset)
    test_loss = float(results["eval_loss"])

    metrics = {
        "test_loss": test_loss,
        "num-examples": len(testset),
    }
    if isfinite(test_loss):
        metrics["test_perplexity"] = exp(test_loss)
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
