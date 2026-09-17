"""Fixed optimizer-update workloads, independent of the client's partition size."""

import math
import time

import numpy as np


TIMING_DEFINITION = "model_fit_fixed_steps_v1"
SUPPORTED_DATASETS = {"cifar10", "mnist", "fmnist", "femnist", "stackoverflow"}


def positive_steps(value):
    if type(value) is not int or value < 1:
        raise ValueError("local_steps must be a positive integer")
    return value


def resolve_steps(override, supplied):
    for value in (override, supplied):
        if value is not None:
            positive_steps(value)
    if override is not None and supplied is not None and override != supplied:
        raise ValueError(f"Client/server local_steps mismatch: {override} != {supplied}")
    return override if override is not None else supplied


def resolve_seed(override, supplied=None):
    for value in (override, supplied):
        if value is not None and (type(value) is not int or not 0 <= value < 2**32):
            raise ValueError("step_seed must fit in uint32")
    if override is not None and supplied is not None and override != supplied:
        raise ValueError("Client/server step_seed mismatch")
    return override if override is not None else supplied if supplied is not None else 42


def with_fixed_steps(base_config_fn, steps, seed):
    positive_steps(steps)
    seed = resolve_seed(seed)

    def config_fn(server_round):
        return {**base_config_fn(server_round), "training_mode": "steps",
                "local_steps": steps, "step_seed": seed, "server_round": server_round}

    return config_fn


def make_step_dataset(x, y, batch_size, seed, client_id, server_round):
    import tensorflow as tf

    if type(batch_size) is not int or batch_size < 1 or len(x) == 0 or len(x) != len(y):
        raise ValueError("Fixed steps require nonempty paired arrays and a positive batch size")
    seed = resolve_seed(seed)
    if int(client_id) < 0 or int(server_round) < 1:
        raise ValueError("Fixed steps require a nonnegative client ID and a positive round")
    round_seed = int(np.random.SeedSequence([seed, int(client_id), int(server_round)])
                     .generate_state(1)[0] % (2**31 - 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(len(x), seed=round_seed, reshuffle_each_iteration=True)
    # Repeat before batching so small partitions still produce full batches.
    dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options), round_seed


def fit_fixed_steps(model, x, y, steps, batch_size, seed, client_id, server_round):
    positive_steps(steps)
    dataset, round_seed = make_step_dataset(x, y, batch_size, seed, client_id, server_round)
    before = int(model.optimizer.iterations.numpy())
    started = time.perf_counter()
    model.fit(dataset, epochs=1, steps_per_epoch=steps, verbose=False)
    duration = time.perf_counter() - started
    used = int(model.optimizer.iterations.numpy()) - before
    if used != steps:
        raise ValueError(f"Expected {steps} optimizer updates, actually executed {used}")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid fixed-step training duration")
    return duration, {"training_mode": "steps", "local_steps_requested": steps,
                      "local_steps_used": used, "batch_size_used": batch_size,
                      "processed_examples": used * batch_size, "steps_per_second": used / duration,
                      "step_seed": seed, "step_round_seed": round_seed,
                      "timing_definition": TIMING_DEFINITION}
