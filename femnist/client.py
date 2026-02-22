"""Flower client with optional uplink quantization for FEMNIST experiments."""

from __future__ import annotations

import logging
import math
import os
import resource
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import flwr as fl
from hydra.utils import instantiate
from keras.utils import to_categorical
import tensorflow as tf

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


def _set_optimizer_lr(model: tf.keras.Model, lr: float) -> bool:
    """Best-effort update of optimizer learning rate."""
    try:
        optimizer = model.optimizer
    except Exception:
        return False
    if optimizer is None:
        return False

    lr_obj = None
    if hasattr(optimizer, "learning_rate"):
        lr_obj = optimizer.learning_rate
    elif hasattr(optimizer, "lr"):
        lr_obj = optimizer.lr
    else:
        return False

    try:
        if hasattr(lr_obj, "assign"):
            lr_obj.assign(lr)
        else:
            tf.keras.backend.set_value(lr_obj, lr)
        return True
    except Exception:
        try:
            if hasattr(optimizer, "learning_rate"):
                optimizer.learning_rate = lr
            elif hasattr(optimizer, "lr"):
                optimizer.lr = lr
            else:
                return False
            return True
        except Exception:
            return False


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
        # FedCS/TiFL-style resource profile (optional, used by server-side selection)
        profile: Optional[Dict[str, float]] = None,
        train_dataset_fn=None,
        val_dataset_fn=None,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        use_sparse_labels: Optional[bool] = None,
    ) -> None:
        # Instantiate model first, then detect loss type (sparse vs categorical)
        self.model = instantiate(model_cfg)

        def _uses_sparse_categorical_loss(loss_obj) -> bool:
            try:
                if isinstance(loss_obj, str):
                    name = loss_obj.lower()
                else:
                    name = loss_obj.__class__.__name__.lower()
            except Exception:
                name = str(loss_obj).lower()
            return ("sparse" in name) and ("categorical" in name)

        def _uses_binary_crossentropy(loss_obj) -> bool:
            try:
                if isinstance(loss_obj, str):
                    name = loss_obj.lower()
                else:
                    name = loss_obj.__class__.__name__.lower()
            except Exception:
                name = str(loss_obj).lower()
            return ("binary" in name) and ("crossentropy" in name)

        if use_sparse_labels is None:
            use_sparse = _uses_sparse_categorical_loss(self.model.loss)
            use_binary = _uses_binary_crossentropy(self.model.loss)
        else:
            use_sparse = bool(use_sparse_labels)
            use_binary = _uses_binary_crossentropy(self.model.loss)

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

        self._use_dataset = train_dataset_fn is not None
        self._train_dataset_fn = train_dataset_fn
        self._val_dataset_fn = val_dataset_fn
        self._train_size = (
            train_size if train_size is not None else (len(x_train) if x_train is not None else None)
        )
        self._val_size = val_size if val_size is not None else (len(x_val) if x_val is not None else None)

        self.x_train = x_train
        self.x_val = x_val
        if self._use_dataset:
            self.y_train = None
            self.y_val = None
        else:
            if use_binary:
                # Multi-label targets are already multi-hot vectors.
                self.y_train = y_train.astype("float32", copy=False)
                self.y_val = y_val.astype("float32", copy=False)
            elif use_sparse:
                self.y_train = y_train
                self.y_val = y_val
            else:
                self.y_train = to_categorical(y_train, num_classes=num_classes)
                self.y_val = to_categorical(y_val, num_classes=num_classes)

        self._local_epochs_override = local_epochs_override
        self._batch_size_override = batch_size_override
        self._profile = dict(profile or {})
        self._profile_cache: Dict[str, float] = {}
        self._profile_cache_ts: float = 0.0

    def get_parameters(self, config):  # pylint: disable=unused-argument
        return self.model.get_weights()

    def fit(self, parameters, config):
        fit_start = time.time()
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
        client_lr = config.get("client_lr", None)
        if isinstance(client_lr, (int, float)):
            applied = _set_optimizer_lr(self.model, float(client_lr))
            if applied:
                logging.info("[Client] Applied per-round learning rate: %.8f", float(client_lr))
            else:
                logging.warning("[Client] Failed to apply per-round learning rate: %s", client_lr)

        logging.info(
            "[Client] Starting local training: epochs=%s batch_size=%s",
            epochs,
            batch_size,
        )

        if self._use_dataset:
            train_ds = self._train_dataset_fn(batch_size, training=True) if self._train_dataset_fn else None
            val_ds = self._val_dataset_fn(batch_size, training=False) if self._val_dataset_fn else None
            self.model.fit(
                train_ds,
                epochs=epochs,
                verbose=False,
                validation_data=val_ds,
            )
            num_train_examples = self._train_size if self._train_size is not None else 0
        else:
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=False,
            )
            num_train_examples = len(self.x_train)
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
            "client_lr": float(client_lr) if isinstance(client_lr, (int, float)) else None,
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
            metrics["fit_elapsed_s"] = float(fit_end - fit_start)
            return packed_weights, num_train_examples, metrics

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
        metrics["fit_elapsed_s"] = float(fit_end - fit_start)
        return weights, num_train_examples, metrics

    def evaluate(self, parameters, config):  # pylint: disable=unused-argument
        self.model.set_weights(parameters)
        if self._use_dataset:
            batch_size = config.get("batch_size", 32) if isinstance(config, dict) else 32
            val_ds = self._val_dataset_fn(batch_size, training=False) if self._val_dataset_fn else None
            loss, acc = self.model.evaluate(val_ds, verbose=False)
            val_len = self._val_size if self._val_size is not None else 0
        else:
            loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=False)
            val_len = len(self.x_val)
        logging.info("Client %s  val_acc=%.4f  loss=%.4f", config.get("cid", "?"), acc, loss)
        return loss, val_len, {"accuracy": acc}

    def get_properties(self, config):  # pylint: disable=unused-argument
        """Return resource profile for FedCS-style client selection.

        Keys are intentionally generic and compatible with common baselines:
        - b_down_bps / b_up_bps: downlink/uplink throughput in bits/sec
        - f_samples_per_s: compute capability in samples/sec
        - n_samples: local training samples available
        """
        # Cache to keep get_properties lightweight (server may call per-round).
        refresh_s = float(config.get("profile_refresh_s", 60.0)) if isinstance(config, dict) else 60.0
        now = time.time()
        if self._profile_cache and (now - self._profile_cache_ts) < refresh_s:
            return dict(self._profile_cache)

        profile: Dict[str, float] = dict(self._profile)
        train_count = (
            self._train_size if self._train_size is not None else (len(self.x_train) if self.x_train is not None else 0)
        )
        profile["n_samples"] = float(train_count)

        # Compute profiling: estimate samples/sec using a short GradientTape run (no weight update).
        if "f_samples_per_s" not in profile or profile.get("f_samples_per_s", 0.0) <= 0:
            prof_sps = self._profile_compute_samples_per_s(
                batch_size=int(config.get("profile_batch_size", 64)) if isinstance(config, dict) else 64,
                steps=int(config.get("profile_steps", 1)) if isinstance(config, dict) else 1,
                warmup_steps=int(config.get("profile_warmup_steps", 3)) if isinstance(config, dict) else 3,
                runs=int(config.get("profile_runs", 2)) if isinstance(config, dict) else 2,
                discard_first=bool(config.get("profile_discard_first", True)) if isinstance(config, dict) else True,
            )
            if prof_sps is not None and prof_sps > 0:
                profile["f_samples_per_s"] = float(prof_sps)

        # Bandwidth probing (optional): server can provide profile_url for HTTP probe endpoint.
        profile_url = None
        if isinstance(config, dict):
            profile_url = config.get("profile_url")
        if not profile_url:
            profile_url = os.environ.get("FEDCS_PROFILE_URL")

        probe_bytes = int(config.get("profile_bytes", 262144)) if isinstance(config, dict) else 262144
        probe_timeout = float(config.get("profile_timeout_s", 5.0)) if isinstance(config, dict) else 5.0
        if profile_url and probe_bytes > 0:
            down_bps, up_bps = self._probe_bandwidth_bps(
                str(profile_url),
                num_bytes=probe_bytes,
                timeout_s=probe_timeout,
            )
            if down_bps:
                profile["b_down_bps"] = float(down_bps)
            if up_bps:
                profile["b_up_bps"] = float(up_bps)

        self._profile_cache = dict(profile)
        self._profile_cache_ts = now
        logging.info("[Profile] %s", self._profile_cache)
        return profile

    def _profile_compute_samples_per_s(
        self,
        *,
        batch_size: int,
        steps: int,
        warmup_steps: int = 3,
        runs: int = 2,
        discard_first: bool = True,
    ) -> Optional[float]:
        """Estimate compute capability (samples/sec) without mutating weights."""
        batch_size = max(1, int(batch_size))
        steps = max(1, int(steps))
        warmup_steps = max(0, int(warmup_steps))
        runs = max(1, int(runs))
        if self._use_dataset or self.x_train is None:
            return None
        if len(self.x_train) < batch_size:
            return None

        x = tf.convert_to_tensor(self.x_train[:batch_size])
        y = tf.convert_to_tensor(self.y_train[:batch_size])
        if y.shape.rank is None:
            return None

        if y.shape.rank == 1:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        else:
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        def _one_step() -> None:
            with tf.GradientTape() as tape:
                preds = self.model(x, training=True)
                loss = loss_fn(y, preds)
            grads = tape.gradient(loss, self.model.trainable_variables)
            _ = [tf.reduce_sum(g) for g in grads if g is not None]

        # Warmup to avoid first-call tracing/initialization costs.
        for _ in range(warmup_steps):
            _one_step()

        sps_runs: List[float] = []
        last_elapsed = None
        for _ in range(runs):
            start = time.time()
            for _ in range(steps):
                _one_step()
            elapsed = time.time() - start
            last_elapsed = elapsed
            if elapsed > 0:
                sps_runs.append((steps * batch_size) / elapsed)

        if not sps_runs:
            return None
        if discard_first and len(sps_runs) >= 2:
            sps = sps_runs[-1]
        else:
            sps = sum(sps_runs) / float(len(sps_runs))

        # Estimate update time as t_ud = E * n / f (paper), using local_epochs from env if available.
        # If not set, assume 1 epoch for the estimate shown in logs.
        local_epochs_env = os.environ.get("FEDCS_LOCAL_EPOCHS")
        try:
            local_epochs = int(local_epochs_env) if local_epochs_env else 1
        except ValueError:
            local_epochs = 1
        est_t_ud = (local_epochs * float(len(self.x_train))) / float(sps) if sps > 0 else float("nan")

        logging.info(
            "[Profile][Compute] batch=%s steps=%s warmup=%s runs=%s discard_first=%s "
            "elapsed_last=%.3fs sps=%.3f n_samples=%s local_epochs=%s est_t_ud=%.3fs",
            batch_size,
            steps,
            warmup_steps,
            runs,
            discard_first,
            float(last_elapsed or 0.0),
            float(sps),
            len(self.x_train),
            local_epochs,
            float(est_t_ud),
        )

        return float(sps)

    def _probe_bandwidth_bps(
        self, base_url: str, *, num_bytes: int, timeout_s: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Probe down/up bandwidth to the server's optional profile HTTP endpoint."""
        import urllib.parse
        import urllib.request

        num_bytes = max(1, int(num_bytes))
        timeout_s = max(0.1, float(timeout_s))
        base_url = base_url.rstrip("/")

        down_bps = None
        up_bps = None
        try:
            q = urllib.parse.urlencode({"bytes": str(num_bytes)})
            url = f"{base_url}/download?{q}"
            t0 = time.time()
            data = urllib.request.urlopen(url, timeout=timeout_s).read()
            t1 = time.time()
            if data:
                down_bps = (len(data) * 8.0) / max(1e-6, (t1 - t0))
        except Exception as exc:
            logging.warning("[Profile] download probe failed: %s", exc)

        try:
            url = f"{base_url}/upload"
            payload = bytearray(num_bytes)
            req = urllib.request.Request(url, data=payload, method="POST")
            t0 = time.time()
            urllib.request.urlopen(req, timeout=timeout_s).read()
            t1 = time.time()
            up_bps = (num_bytes * 8.0) / max(1e-6, (t1 - t0))
        except Exception as exc:
            logging.warning("[Profile] upload probe failed: %s", exc)

        return down_bps, up_bps


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
