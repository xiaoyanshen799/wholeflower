"""Define the Flower Server and function to instantiate it."""

from pathlib import Path

from keras.utils import to_categorical
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


def get_evaluate_fn(model, x_test, y_test, num_rounds, num_classes, log_path: str | None = None):
    """Generate the function for server global model evaluation.

    The method evaluate_fn runs after global model aggregation.
    """

    def evaluate_fn(
        server_round: int, parameters, config
    ):  # pylint: disable=unused-argument
        # if server_round == num_rounds:  # evaluates global model just on the last round
            # instantiate the model
        model.set_weights(parameters)

        y_test_cat = to_categorical(y_test, num_classes=num_classes)
        loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=False)

        if log_path:
            entry = f"round={server_round} loss={loss:.6f} accuracy={accuracy:.6f}\n"
            Path(log_path).open("a", encoding="utf-8").write(entry)

        return loss, {"accuracy": accuracy}

        # return None

    return evaluate_fn
