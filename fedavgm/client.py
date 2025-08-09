"""Define the Flower Client and function to instantiate it."""

import math
import logging

import flwr as fl
from hydra.utils import instantiate
from keras.utils import to_categorical


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client."""

    # pylint: disable=too-many-arguments
    def __init__(self, x_train, y_train, x_val, y_val, model, num_classes) -> None:
        # local model
        self.model = instantiate(model)

        # local dataset
        self.x_train, self.y_train = x_train, to_categorical(
            y_train, num_classes=num_classes
        )
        self.x_val, self.y_val = x_val, to_categorical(y_val, num_classes=num_classes)

    def get_parameters(self, config):
        """Return the parameters of the current local model."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Implement distributed fit function for a given client."""
        self.model.set_weights(parameters)

        # Measure training time
        import time  # local import to avoid slowing module import

        # Communication time (server -> client): from server timestamp to now
        server_send_time = config.get("server_send_time", None)
        now = time.time()
        server_to_client_ms = None
        if isinstance(server_send_time, (int, float)):
            server_to_client_ms = max(0.0, (now - server_send_time) * 1000.0)

        t_start = time.perf_counter()
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=config["local_epochs"],
            batch_size=config["batch_size"],
            verbose=False,
        )
        duration = time.perf_counter() - t_start

        # Optional: log locally for easy inspection
        logging.info(f"[Client] Training finished in {duration:.2f} seconds")

        # Return training duration as custom metric
        metrics = {"train_time": duration}
        if server_to_client_ms is not None:
            metrics["server_to_client_ms"] = server_to_client_ms
        metrics["client_fit_end_time"] = time.time()

        return self.model.get_weights(), len(self.x_train), metrics

    def evaluate(self, parameters, config):
        """Implement distributed evaluation for a given client."""
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=False)
        logging.info("Client %s  val_acc=%.4f  loss=%.4f", config.get("cid", "?"), acc, loss)
        return loss, len(self.x_val), {"accuracy": acc}


def generate_client_fn(partitions, model, num_classes):
    """Generate the client function that creates the Flower Clients."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        full_x_train_cid, full_y_train_cid = partitions[int(cid)]

        # Use 10% of the client's training data for validation
        split_idx = math.floor(len(full_x_train_cid) * 0.9)
        x_train_cid, y_train_cid = (
            full_x_train_cid[:split_idx],
            full_y_train_cid[:split_idx],
        )
        x_val_cid, y_val_cid = (
            full_x_train_cid[split_idx:],
            full_y_train_cid[split_idx:],
        )

        return FlowerClient(
            x_train_cid, y_train_cid, x_val_cid, y_val_cid, model, num_classes
        )

    return client_fn
