"""Define the Flower Server and function to instantiate it."""

from pathlib import Path

from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import time
from omegaconf import DictConfig


def get_on_fit_config(config: DictConfig):
    """Generate the function for config.

    The config dict is sent to the client fit() method.
    """

    def fit_config_fn(server_round: int):  # pylint: disable=unused-argument
        # option to use scheduling of learning rate based on round
        # if server_round > 50:
        #     lr = config.lr / 10
        return {
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
            "confit_time": time.time(),
        }

    return fit_config_fn


def get_evaluate_fn(
    model,
    x_test,
    y_test,
    num_rounds,
    num_classes,
    log_path: str | None = None,
    sample_size: int | None = None,
    sample_seed: int | None = None,
):
    """Generate the function for server global model evaluation.

    The method evaluate_fn runs after global model aggregation.
    """
    rng = None
    sample_size = int(sample_size) if sample_size else None
    if sample_size:
        effective_size = min(sample_size, len(x_test))
        if effective_size < len(x_test):
            rng = np.random.default_rng(sample_seed)
            sample_size = effective_size
        else:
            sample_size = None  # Evaluating on full dataset, no need to sample

    # Helper: detect if model is compiled with sparse categorical loss
    def _uses_sparse_categorical_loss(loss_obj) -> bool:
        try:
            if isinstance(loss_obj, str):
                name = loss_obj.lower()
            else:
                name = loss_obj.__class__.__name__.lower()
        except Exception:
            name = str(loss_obj).lower()
        return ("sparse" in name) and ("categorical" in name)

    def _multilabel_eval(logits: np.ndarray, y_true: np.ndarray) -> tuple[float, dict[str, float]]:
        y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
        logits_tf = tf.convert_to_tensor(logits, dtype=tf.float32)
        loss_tf = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_tf, logits=logits_tf)
        )

        probs = tf.sigmoid(logits_tf)
        y_pred = tf.cast(probs >= 0.5, tf.float32)
        y_true_bin = tf.cast(y_true_tf >= 0.5, tf.float32)

        tp = tf.reduce_sum(y_pred * y_true_bin)
        fp = tf.reduce_sum(y_pred * (1.0 - y_true_bin))
        fn = tf.reduce_sum((1.0 - y_pred) * y_true_bin)
        micro_f1 = (2.0 * tp) / tf.maximum(2.0 * tp + fp + fn, 1e-9)

        metrics: dict[str, float] = {"micro_f1": float(micro_f1.numpy())}
        for k in (5, 10):
            k = int(k)
            topk = tf.math.top_k(logits_tf, k=k).indices  # [B, k]
            hits = tf.gather(y_true_bin, topk, batch_dims=1)  # [B, k]
            hits_count = tf.reduce_sum(hits, axis=1)
            true_count = tf.reduce_sum(y_true_bin, axis=1)
            recall = tf.reduce_mean(hits_count / tf.maximum(true_count, 1.0))
            metrics[f"recall@{k}"] = float(recall.numpy())

        avg_labels = tf.reduce_mean(tf.reduce_sum(y_true_bin, axis=1))
        metrics["avg_labels"] = float(avg_labels.numpy())

        return float(loss_tf.numpy()), metrics

    def evaluate_fn(
        server_round: int, parameters, config
    ):  # pylint: disable=unused-argument
        # if server_round == num_rounds:  # evaluates global model just on the last round
            # instantiate the model
        model.set_weights(parameters)

        x_eval = x_test
        y_eval = y_test
        if sample_size and rng is not None:
            indices = rng.choice(len(x_test), size=sample_size, replace=False)
            x_eval = x_test[indices]
            y_eval = y_test[indices]

        if getattr(y_eval, "ndim", 1) == 2:
            logits = model.predict(x_eval, batch_size=256, verbose=False)
            loss, metrics = _multilabel_eval(logits, y_eval)
            if log_path:
                subset = len(x_eval)
                entry = (
                    f"round={server_round} loss={loss:.6f} micro_f1={metrics['micro_f1']:.6f} "
                    f"recall@5={metrics['recall@5']:.6f} recall@10={metrics['recall@10']:.6f} "
                    f"avg_labels={metrics['avg_labels']:.3f} samples={subset}\n"
                )
                Path(log_path).open("a", encoding="utf-8").write(entry)
            return loss, metrics

        # Choose label format based on compiled loss
        if _uses_sparse_categorical_loss(model.loss):
            labels = y_eval
        else:
            labels = to_categorical(y_eval, num_classes=num_classes)

        loss, accuracy = model.evaluate(x_eval, labels, verbose=False)

        if log_path:
            subset = len(x_eval)
            entry = (
                f"round={server_round} loss={loss:.6f} accuracy={accuracy:.6f} "
                f"samples={subset}\n"
            )
            Path(log_path).open("a", encoding="utf-8").write(entry)

        return loss, {"accuracy": accuracy}

        # return None

    return evaluate_fn
