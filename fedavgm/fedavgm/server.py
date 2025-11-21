"""Define the Flower Server and function to instantiate it."""

import logging
import numpy as np
from keras.utils import to_categorical
from keras.losses import SparseCategoricalCrossentropy
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


def get_evaluate_fn(model, x_test, y_test, num_rounds, num_classes):
    """Generate the function for server global model evaluation.

    The method evaluate_fn runs after global model aggregation.
    """

    def evaluate_fn(
        server_round: int, parameters, config
    ):  # pylint: disable=unused-argument
        # if server_round == num_rounds:  # evaluates global model just on the last round
        model.set_weights(parameters)

        # If the model uses sparse categorical loss, do not one-hot the labels
        use_sparse = isinstance(getattr(model, "loss", None), SparseCategoricalCrossentropy) or (
            getattr(getattr(model, "loss", None), "name", "").startswith("sparse_categorical_crossentropy")
        )
        if use_sparse:
            y_eval = y_test
        else:
            y_eval = to_categorical(y_test, num_classes=num_classes)

        loss, accuracy = model.evaluate(x_test, y_eval, verbose=False)
        perplexity = float(np.exp(min(loss, 50.0)))

        # Manual diagnostics to understand why accuracy might stay at 0
        sample_size = min(len(x_test), 256)
        preds = model.predict(x_test[:sample_size], verbose=False)
        pred_labels = np.argmax(preds, axis=1)
        if use_sparse:
            true_labels = y_test[:sample_size].reshape(-1)
        else:
            true_labels = np.argmax(y_eval[:sample_size], axis=1)
        manual_acc = float(np.mean(pred_labels == true_labels))

        top5 = (
            float(
                np.mean(
                    np.any(np.argsort(preds, axis=1)[:, -5:] == true_labels[:, None], axis=1)
                )
            )
            if preds.ndim == 2 and preds.shape[1] >= 5 and sample_size > 0
            else 0.0
        )

        weights = model.get_weights()
        weight_norms = [float(np.linalg.norm(w)) for w in weights if isinstance(w, np.ndarray)]
        true_label_dist = np.bincount(true_labels, minlength=num_classes)[:10].tolist()

        print(
            "[Server][Eval] round={} loss={:.6f} ppl={:.6f} keras_acc={:.6f} manual_acc={:.6f} top5={:.6f} "
            "preds_type={} preds_shape={} num_classes={} pred_label_max={} "
            "pred_label_dist={} true_label_dist={} first10_preds={} first10_true={} weight_norms={}".format(
                server_round,
                loss,
                perplexity,
                float(accuracy),
                manual_acc,
                top5,
                type(preds),
                preds.shape,
                num_classes,
                int(pred_labels.max()) if pred_labels.size else None,
                np.bincount(pred_labels, minlength=num_classes)[:10].tolist(),
                true_label_dist,
                pred_labels[:10].tolist(),
                true_labels[:10].tolist(),
                ",".join(f"{n:.2e}" for n in weight_norms[:6]),
            )
        )

        return loss, {"accuracy": accuracy, "perplexity": perplexity, "top5_accuracy": top5}

        # return None

    return evaluate_fn
