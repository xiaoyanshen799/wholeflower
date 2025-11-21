"""Define the Flower Client and function to instantiate it."""

import math
import logging
from typing import Dict, Optional

import flwr as fl
from hydra.utils import instantiate
from keras.utils import to_categorical

from fedavgm.compression import ErrorFeedbackQuantizer, maybe_unpack_quantized


# Keeps residual error buffers per logical client id so that repeated
# instantiations (e.g., in simulation) can continue error feedback.
CLIENT_RESIDUALS: Dict[str, Dict[str, object]] = {}


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
        cid: Optional[str] = None,
        enable_compression: bool = True,
        quantization_bits: int = 8,
        error_feedback: bool = True,
        local_epochs_override: Optional[int] = None,
        batch_size_override: Optional[int] = None,
    ) -> None:
        # Peek model target to decide label format
        target = model.get("_target_", "") if isinstance(model, dict) else ""
        use_sparse = target.endswith("char_rnn")

        # local model
        self.model = instantiate(model)

        # Compression/quantization setup
        self.cid = str(cid) if cid is not None else None
        self.enable_compression = enable_compression
        self._quantizer = (
            ErrorFeedbackQuantizer(num_bits=quantization_bits, error_feedback=error_feedback)
            if enable_compression
            else None
        )
        if self.cid and self._quantizer:
            prior_state = CLIENT_RESIDUALS.get(self.cid)
            if prior_state and prior_state.get('num_bits') == self._quantizer.num_bits and prior_state.get('error_feedback', True) == self._quantizer.error_feedback:
                residuals = prior_state.get('residuals')
                if residuals is not None:
                    self._quantizer.set_state(residuals)

        # local dataset
        self.x_train = x_train
        self.x_val = x_val
        if use_sparse:
            # Keep integer labels for sparse_categorical_crossentropy
            self.y_train = y_train
            self.y_val = y_val
        else:
            self.y_train = to_categorical(y_train, num_classes=num_classes)
            self.y_val = to_categorical(y_val, num_classes=num_classes)

        self._local_epochs_override = local_epochs_override
        self._batch_size_override = batch_size_override

    def get_parameters(self, config):
        """Return the parameters of the current local model."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Implement distributed fit function for a given client."""
        received_parameters, was_quantized = maybe_unpack_quantized(list(parameters))
        if was_quantized:
            logging.info("[Client] Received quantized payload; dequantizing before training")
        else:
            received_parameters = parameters

        self.model.set_weights(received_parameters)

        # Measure training time
        import time  # local import to avoid slowing module import

        # Communication time (server -> client): from server timestamp to now
        server_send_time = config.get("server_send_time", None)
        server_wait_time = config.get("server_wait_time", None)
        config_time = config.get("confit_time", None)
        now = time.time()
        server_to_client_ms = None
        if isinstance(server_send_time, (int, float)):
            server_to_client_ms = max(0.0, (now - server_send_time) * 1000.0)

        t_start = time.time()
        epochs = self._local_epochs_override
        if epochs is None:
            epochs = config.get("local_epochs")

        batch_size = self._batch_size_override
        if batch_size is None:
            batch_size = config.get("batch_size")

        logging.info(
            "[Client] Starting local training: epochs=%s batch_size=%s",
            epochs,
            batch_size,
        )

        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=False,
        )
        duration = time.time() - t_start

        # Optional: log locally for easy inspection
        logging.info(f"[Client] Training finished in {duration:.2f} seconds")

        # Return training duration as custom metric
        metrics = {
            "train_time": float(duration),
            "local_epochs_used": float(epochs) if epochs is not None else None,
            "batch_size_used": float(batch_size) if batch_size is not None else None,
        }
        if isinstance(server_to_client_ms, (int, float)):
            metrics["server_to_client_ms"] = float(server_to_client_ms)
        if isinstance(server_wait_time, (int, float)):
            metrics["server_wait_time"] = float(server_wait_time)
        if isinstance(config_time, (int, float)):
            metrics["config_time"] = float(config_time)
        metrics["client_fit_end_time"] = float(time.time())
        metrics = {k: v for k, v in metrics.items() if v is not None}
        # server_arrival_time intentionally omitted here to avoid None

        weights = [w.astype("float32", copy=True) for w in self.model.get_weights()]

        if self.enable_compression and self._quantizer:
            packed_weights, report = self._quantizer.encode(weights)
            metrics.update(report.as_metrics())
            metrics["quant_bits"] = float(self._quantizer.num_bits)
            metrics["quant_applied"] = 1.0
            payload_bytes = sum(arr.nbytes for arr in packed_weights)
            logging.info(
                "[Client] Upload payload %.3f MB (%.3f MiB) [quantized]",
                payload_bytes / 1_000_000,
                payload_bytes / (1024 ** 2),
            )
            if self.cid:
                state = self._quantizer.get_state()
                if state is not None:
                    CLIENT_RESIDUALS[self.cid] = {
                        'num_bits': self._quantizer.num_bits,
                        'error_feedback': self._quantizer.error_feedback,
                        'residuals': state,
                    }
            return packed_weights, len(self.x_train), metrics

        metrics["quant_applied"] = 0.0
        payload_bytes = sum(arr.nbytes for arr in weights)
        logging.info(
            "[Client] Upload payload %.3f MB (%.3f MiB) [float32]",
            payload_bytes / 1_000_000,
            payload_bytes / (1024 ** 2),
        )
        return weights, len(self.x_train), metrics

    def evaluate(self, parameters, config):
        """Implement distributed evaluation for a given client."""
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=False)
        logging.info("Client %s  val_acc=%.4f  loss=%.4f", config.get("cid", "?"), acc, loss)
        return loss, len(self.x_val), {"accuracy": acc}


def generate_client_fn(partitions, model, num_classes, quantization_cfg=None):
    """Generate the client function that creates the Flower Clients."""

    if quantization_cfg is None:
        quant_cfg = {}
    else:
        quant_cfg = {key: quantization_cfg[key] for key in quantization_cfg}
    enable_compression = bool(quant_cfg.get('enabled', True))
    quant_bits = int(quant_cfg.get('num_bits', 8))
    quant_error_feedback = bool(quant_cfg.get('error_feedback', True))

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
            x_train_cid,
            y_train_cid,
            x_val_cid,
            y_val_cid,
            model,
            num_classes,
            cid=cid,
            enable_compression=enable_compression,
            quantization_bits=quant_bits,
            error_feedback=quant_error_feedback,
        )

    return client_fn
