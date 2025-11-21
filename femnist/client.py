"""Flower client with optional uplink quantization for FEMNIST experiments."""

from __future__ import annotations

import logging
import math
import resource
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import flwr as fl
from hydra.utils import instantiate
from keras.utils import to_categorical

from compression import ErrorFeedbackQuantizer, maybe_unpack_quantized

CLIENT_RESIDUALS: Dict[str, Dict[str, object]] = {}


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
            # scaling_cur_freq is reported in kHz on Linux.
            freq_values.append(float(raw) / 1000.0)
        except ValueError:
            continue
    if not freq_values:
        return None
    return sum(freq_values) / float(len(freq_values))


def _read_schedstat() -> Optional[Tuple[int, int, int]]:
    """Return (run_time_ns, runqueue_time_ns, timeslices) from /proc/self/schedstat."""
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
    """Client wrapper that handles model setup and uplink compression."""

    def __init__(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        model_cfg,
        num_classes: int,
        *,
        cid: Optional[str] = None,
        enable_compression: bool = True,
        quantization_bits: int = 8,
        error_feedback: bool = True,
        local_epochs_override: Optional[int] = None,
        batch_size_override: Optional[int] = None,
    ) -> None:
        target = model_cfg.get("_target_", "") if isinstance(model_cfg, dict) else ""
        use_sparse = target.endswith("char_rnn")

        self.model = instantiate(model_cfg)

        self.cid = str(cid) if cid is not None else None
        self.enable_compression = enable_compression
        self._quantizer = (
            ErrorFeedbackQuantizer(num_bits=quantization_bits, error_feedback=error_feedback)
            if enable_compression
            else None
        )
        if self.cid and self._quantizer:
            prior_state = CLIENT_RESIDUALS.get(self.cid)
            if (
                prior_state
                and prior_state.get("num_bits") == self._quantizer.num_bits
                and prior_state.get("error_feedback", True) == self._quantizer.error_feedback
            ):
                residuals = prior_state.get("residuals")
                if residuals is not None:
                    self._quantizer.set_state(residuals)

        self.x_train = x_train
        self.x_val = x_val
        if use_sparse:
            self.y_train = y_train
            self.y_val = y_val
        else:
            self.y_train = to_categorical(y_train, num_classes=num_classes)
            self.y_val = to_categorical(y_val, num_classes=num_classes)

        self._local_epochs_override = local_epochs_override
        self._batch_size_override = batch_size_override

    def get_parameters(self, config):  # pylint: disable=unused-argument
        return self.model.get_weights()

    def fit(self, parameters, config):
        decode_start = time.time()
        received_parameters, was_quantized = maybe_unpack_quantized(list(parameters))
        if was_quantized:
            logging.info("[Client] Received quantized payload; dequantizing before training")
        else:
            received_parameters = parameters
        decode_time = time.time() - decode_start if was_quantized else 0.0

        self.model.set_weights(received_parameters)

        server_send_time = config.get("server_send_time", None)
        server_wait_time = config.get("server_wait_time", None)
        config_time = config.get("confit_time", None)

        server_to_client_ms = None
        now = time.time()
        if isinstance(server_send_time, (int, float)):
            server_to_client_ms = max(0.0, (now - server_send_time) * 1000.0)

        train_start = time.time()
        cpu_freq_start = _read_cpu_freq_mhz()
        cpu_usage_start = resource.getrusage(resource.RUSAGE_SELF)
        cpu_time_start = float(cpu_usage_start.ru_utime + cpu_usage_start.ru_stime)
        sched_start = _read_schedstat()
        epochs = (
            self._local_epochs_override
            if self._local_epochs_override is not None
            else config.get("local_epochs")
        )
        batch_size = (
            self._batch_size_override
            if self._batch_size_override is not None
            else config.get("batch_size")
        )

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
        train_end = time.time()
        train_duration = train_end - train_start
        cpu_freq_end = _read_cpu_freq_mhz()
        cpu_usage_end = resource.getrusage(resource.RUSAGE_SELF)
        cpu_time_end = float(cpu_usage_end.ru_utime + cpu_usage_end.ru_stime)
        sched_end = _read_schedstat()
        logging.info("[Client] Training finished in %.2f seconds", train_duration)

        metrics = {
            "local_epochs_used": float(epochs) if epochs is not None else None,
            "batch_size_used": float(batch_size) if batch_size is not None else None,
        }
        if isinstance(server_to_client_ms, (int, float)):
            metrics["server_to_client_ms"] = float(server_to_client_ms)
        if isinstance(server_wait_time, (int, float)):
            metrics["server_wait_time"] = float(server_wait_time)
        if isinstance(config_time, (int, float)):
            metrics["config_time"] = float(config_time)
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

        weights = [w.astype("float32", copy=True) for w in self.model.get_weights()]
        encode_time = 0.0

        if self.enable_compression and self._quantizer:
            encode_start = time.time()
            packed_weights, report = self._quantizer.encode(weights)
            encode_time = time.time() - encode_start
            metrics.update(report.as_metrics())
            metrics["quant_bits"] = float(self._quantizer.num_bits)
            metrics["quant_applied"] = 1.0
            payload_bytes = sum(arr.nbytes for arr in packed_weights)
            logging.info(
                "[Client] Upload payload %.3f MB (%.3f MiB) [quantized]",
                payload_bytes / 1_000_000,
                payload_bytes / (1024**2),
            )
            if self.cid:
                state = self._quantizer.get_state()
                if state is not None:
                    CLIENT_RESIDUALS[self.cid] = {
                        "num_bits": self._quantizer.num_bits,
                        "error_feedback": self._quantizer.error_feedback,
                        "residuals": state,
                    }
            fit_end = time.time()
            metrics["train_time"] = float(train_duration)
            metrics["edcode"] = float(decode_time + encode_time)
            metrics["client_fit_end_time"] = float(fit_end)
            return packed_weights, len(self.x_train), metrics

        metrics["quant_applied"] = 0.0
        payload_bytes = sum(arr.nbytes for arr in weights)
        logging.info(
            "[Client] Upload payload %.3f MB (%.3f MiB) [float32]",
            payload_bytes / 1_000_000,
            payload_bytes / (1024**2),
        )
        fit_end = time.time()
        metrics["train_time"] = float(train_duration)
        metrics["edcode"] = float(decode_time + encode_time)
        metrics["client_fit_end_time"] = float(fit_end)
        return weights, len(self.x_train), metrics

    def evaluate(self, parameters, config):  # pylint: disable=unused-argument
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=False)
        logging.info("Client %s  val_acc=%.4f  loss=%.4f", config.get("cid", "?"), acc, loss)
        return loss, len(self.x_val), {"accuracy": acc}


def generate_client_fn(
    partitions, model_cfg, num_classes: int, quantization_cfg: Optional[Dict[str, object]] = None
):
    if quantization_cfg is None:
        quant_cfg: Dict[str, object] = {}
    else:
        quant_cfg = {key: quantization_cfg[key] for key in quantization_cfg}

    enable_compression = bool(quant_cfg.get("enabled", True))
    quant_bits = int(quant_cfg.get("num_bits", 8))
    quant_error_feedback = bool(quant_cfg.get("error_feedback", True))

    def client_fn(cid: str) -> FlowerClient:
        full_x_train_cid, full_y_train_cid = partitions[int(cid)]

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
            model_cfg,
            num_classes,
            cid=cid,
            enable_compression=enable_compression,
            quantization_bits=quant_bits,
            error_feedback=quant_error_feedback,
        )

    return client_fn
