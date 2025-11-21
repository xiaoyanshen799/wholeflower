"""Define the Flower Client and function to instantiate it."""

import logging
import math
from pathlib import Path
import resource
from typing import Optional, Tuple

import flwr as fl
from hydra.utils import instantiate
from keras.utils import to_categorical


def _read_cpu_freq_mhz() -> Optional[float]:
    """Return average scaling_cur_freq across CPUs, or None if unavailable."""
    freq_values = []
    cpu_root = Path("/sys/devices/system/cpu")
    for cpu_dir in cpu_root.glob("cpu[0-9]*"):
        freq_file = cpu_dir / "cpufreq" / "scaling_cur_freq"
        if not freq_file.is_file():
            continue
        try:
            raw = freq_file.read_text().strip()
        except OSError:
            continue
        if not raw:
            continue
        try:
            freq_values.append(float(raw) / 1000.0)
        except ValueError:
            continue
    if not freq_values:
        return None
    return sum(freq_values) / float(len(freq_values))


def _read_schedstat() -> Optional[Tuple[int, int, int]]:
    sched_file = Path("/proc/self/schedstat")
    if not sched_file.is_file():
        return None
    try:
        raw = sched_file.read_text().strip()
    except OSError:
        return None
    if not raw:
        return None
    parts = raw.split()
    if len(parts) < 3:
        return None
    try:
        run_ns = int(parts[0])
        runqueue_ns = int(parts[1])
        timeslices = int(parts[2])
    except ValueError:
        return None
    return run_ns, runqueue_ns, timeslices


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
        local_epochs_override: Optional[int] = None,
        batch_size_override: Optional[int] = None,
    ) -> None:
        # local model
        self.model = instantiate(model)

        # local dataset
        self.x_train, self.y_train = x_train, to_categorical(
            y_train, num_classes=num_classes
        )
        self.x_val, self.y_val = x_val, to_categorical(y_val, num_classes=num_classes)

        self._local_epochs_override = local_epochs_override
        self._batch_size_override = batch_size_override

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
        server_wait_time = config.get("server_wait_time", None)
        config_time = config.get("confit_time", None)
        now = time.time()
        server_to_client_ms = None
        if isinstance(server_send_time, (int, float)):
            server_to_client_ms = max(0.0, (now - server_send_time) * 1000.0)

        t_start = time.time()
        cpu_freq_start = _read_cpu_freq_mhz()
        cpu_usage_start = resource.getrusage(resource.RUSAGE_SELF)
        cpu_time_start = float(cpu_usage_start.ru_utime + cpu_usage_start.ru_stime)
        sched_start = _read_schedstat()
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
        cpu_freq_end = _read_cpu_freq_mhz()
        cpu_usage_end = resource.getrusage(resource.RUSAGE_SELF)
        cpu_time_end = float(cpu_usage_end.ru_utime + cpu_usage_end.ru_stime)
        sched_end = _read_schedstat()

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
        if cpu_freq_start is not None:
            metrics["cpu_freq_start_mhz"] = float(cpu_freq_start)
        if cpu_freq_end is not None:
            metrics["cpu_freq_end_mhz"] = float(cpu_freq_end)
        metrics["cpu_time_start_s"] = cpu_time_start
        metrics["cpu_time_end_s"] = cpu_time_end
        metrics["cpu_time_elapsed_s"] = cpu_time_end - cpu_time_start
        if sched_start and sched_end:
            run_delta = max(0, sched_end[0] - sched_start[0])
            runqueue_delta = max(0, sched_end[1] - sched_start[1])
            timeslice_delta = max(0, sched_end[2] - sched_start[2])
            metrics["sched_run_time_ns"] = float(run_delta)
            metrics["sched_runqueue_time_ns"] = float(runqueue_delta)
            metrics["sched_runqueue_time_s"] = float(runqueue_delta) / 1_000_000_000.0
            metrics["sched_timeslices"] = float(timeslice_delta)
        metrics = {k: v for k, v in metrics.items() if v is not None}
        # server_arrival_time intentionally omitted here to avoid None

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
