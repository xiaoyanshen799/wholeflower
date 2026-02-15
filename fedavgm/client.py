"""Define the Flower Client and function to instantiate it."""

import math
import logging

import flwr as fl
from hydra.utils import instantiate
from keras.utils import to_categorical


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        model,
        num_classes,
        *,
        train_dataset_fn=None,
        val_dataset_fn=None,
        train_size=None,
        val_size=None,
        use_sparse_labels: bool | None = None,
    ) -> None:
        # local model
        self.model = instantiate(model)

        # local dataset
        self._use_dataset = train_dataset_fn is not None
        self._train_dataset_fn = train_dataset_fn
        self._val_dataset_fn = val_dataset_fn
        self._train_size = train_size if train_size is not None else (len(x_train) if x_train is not None else 0)
        self._val_size = val_size if val_size is not None else (len(x_val) if x_val is not None else 0)

        if use_sparse_labels is None:
            # Detect sparse vs categorical from model loss
            loss_obj = self.model.loss
            try:
                lname = loss_obj.lower() if isinstance(loss_obj, str) else loss_obj.__class__.__name__.lower()
            except Exception:
                lname = str(loss_obj).lower()
            use_sparse = ("sparse" in lname) and ("categorical" in lname)
            use_binary = ("binary" in lname) and ("crossentropy" in lname)
        else:
            use_sparse = bool(use_sparse_labels)
            loss_obj = self.model.loss
            try:
                lname = loss_obj.lower() if isinstance(loss_obj, str) else loss_obj.__class__.__name__.lower()
            except Exception:
                lname = str(loss_obj).lower()
            use_binary = ("binary" in lname) and ("crossentropy" in lname)

        if self._use_dataset:
            self.x_train = x_train
            self.y_train = y_train
            self.x_val = x_val
            self.y_val = y_val
        else:
            if use_binary:
                self.x_train, self.y_train = x_train, y_train.astype("float32", copy=False)
                self.x_val, self.y_val = x_val, y_val.astype("float32", copy=False)
            else:
                self.x_train, self.y_train = x_train, (
                    y_train if use_sparse else to_categorical(y_train, num_classes=num_classes)
                )
                self.x_val, self.y_val = x_val, (
                    y_val if use_sparse else to_categorical(y_val, num_classes=num_classes)
                )

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
        if self._use_dataset:
            train_ds = self._train_dataset_fn(config["batch_size"], training=True)
            val_ds = self._val_dataset_fn(config["batch_size"], training=False) if self._val_dataset_fn else None
            self.model.fit(
                train_ds,
                epochs=config["local_epochs"],
                verbose=False,
                validation_data=val_ds,
            )
            num_train = self._train_size
        else:
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=config["local_epochs"],
                batch_size=config["batch_size"],
                verbose=False,
            )
            num_train = len(self.x_train)
        duration = time.perf_counter() - t_start

        # Optional: log locally for easy inspection
        logging.info(f"[Client] Training finished in {duration:.2f} seconds")

        # Return training duration as custom metric
        metrics = {"train_time": duration}
        if server_to_client_ms is not None:
            metrics["server_to_client_ms"] = server_to_client_ms
        metrics["client_fit_end_time"] = time.time()

        return self.model.get_weights(), num_train, metrics

    def evaluate(self, parameters, config):
        """Implement distributed evaluation for a given client."""
        self.model.set_weights(parameters)
        if self._use_dataset:
            val_ds = self._val_dataset_fn(config.get("batch_size", 32), training=False) if self._val_dataset_fn else None
            loss, acc = self.model.evaluate(val_ds, verbose=False)
            num_val = self._val_size
        else:
            loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=False)
            num_val = len(self.x_val)
        logging.info("Client %s  val_acc=%.4f  loss=%.4f", config.get("cid", "?"), acc, loss)
        return loss, num_val, {"accuracy": acc}


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
