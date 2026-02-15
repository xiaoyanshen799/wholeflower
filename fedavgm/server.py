"""Define the Flower Server and function to instantiate it."""

import time
from omegaconf import DictConfig
from keras.utils import to_categorical
import tensorflow as tf


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
            "server_send_time": time.time(),
        }

    return fit_config_fn


def get_evaluate_fn(model, x_test, y_test, num_rounds, num_classes):
    """Generate the function for server global model evaluation.

    The method evaluate_fn runs after global model aggregation.
    """

    def _uses_sparse_labels(loss_obj) -> bool:
        """Return True when model expects sparse integer labels."""
        try:
            name = loss_obj if isinstance(loss_obj, str) else loss_obj.__class__.__name__
        except Exception:
            name = str(loss_obj)
        name = name.lower()
        # Common keras aliases: 'sparsecategoricalcrossentropy', 'sparse_categorical_crossentropy'
        return "sparse" in name and "categorical" in name

    def _multilabel_eval(logits, y_true):
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

        metrics = {"micro_f1": float(micro_f1.numpy())}
        for k in (5, 10):
            topk = tf.math.top_k(logits_tf, k=int(k)).indices
            hits = tf.gather(y_true_bin, topk, batch_dims=1)
            hits_count = tf.reduce_sum(hits, axis=1)
            true_count = tf.reduce_sum(y_true_bin, axis=1)
            recall = tf.reduce_mean(hits_count / tf.maximum(true_count, 1.0))
            metrics[f"recall@{int(k)}"] = float(recall.numpy())

        return float(loss_tf.numpy()), metrics

    def evaluate_fn(
        server_round: int, parameters, config
    ):  # pylint: disable=unused-argument
        # if server_round == num_rounds:  # evaluates global model just on the last round
            # instantiate the model
        model.set_weights(parameters)

        if getattr(y_test, "ndim", 1) == 2:
            logits = model.predict(x_test, batch_size=256, verbose=False)
            loss, metrics = _multilabel_eval(logits, y_test)
            return loss, metrics

        # Keep label format aligned with the model loss to avoid bogus metrics
        if _uses_sparse_labels(model.loss):
            y_eval = y_test
        else:
            y_eval = to_categorical(y_test, num_classes=num_classes)

        loss, accuracy = model.evaluate(x_test, y_eval, verbose=False)

        return loss, {"accuracy": accuracy}

        # return None

    return evaluate_fn
