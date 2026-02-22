"""REFL-style strategy for cross-silo full-participation training.

This strategy follows REFL's stale-aware aggregation (SAA) logic:
1) full-participation client selection (IPS disabled for silo setting);
2) stale-update importance weighting with REFL's stale_factor options;
3) normalized weighted model aggregation (FedAvg-style by default).
"""

from __future__ import annotations

import csv
import logging
import math
import os
import time
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from flwr.common import (
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from compression import maybe_unpack_quantized
from strategy import QuantizedFedAvgM


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class REFLAsyncSiloStrategy(QuantizedFedAvgM):
    """REFL-style stale-aware aggregation for cross-silo full selection.

    Notes
    -----
    - IPS/priority selection is intentionally disabled for silo full participation.
    - Staleness is tracked using the request round attached by async_server.
    - Server aggregation defaults to REFL/FedAvg-style weighted averaging.
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
        evaluate_fn: Optional[
            Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        server_learning_rate: float = 0.01,
        server_momentum: float = 0.9,
        csv_log_path: Optional[str] = None,
        downlink_quantization_enabled: bool = False,
        downlink_quantization_bits: int = 0,
        downlink_error_feedback: bool = True,
        stale_factor: float = -4.0,
        stale_beta: float = 0.35,
        scale_coff: float = 10.0,
        stale_update: int = -1,
        use_num_examples: bool = False,
        refl_server_update: str = "fedavg",
        full_selection: bool = True,
        wait_for_clients_timeout_s: float = 30.0,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            server_learning_rate=server_learning_rate,
            server_momentum=server_momentum,
            csv_log_path=csv_log_path,
            downlink_quantization_enabled=downlink_quantization_enabled,
            downlink_quantization_bits=downlink_quantization_bits,
            downlink_error_feedback=downlink_error_feedback,
        )
        self.stale_factor = float(stale_factor)
        self.stale_beta = float(stale_beta)
        self.scale_coff = float(scale_coff) if float(scale_coff) > 0.0 else 10.0
        self.stale_update = int(stale_update)
        self.use_num_examples = bool(use_num_examples)
        self.refl_server_update = str(refl_server_update).strip().lower()
        if self.refl_server_update not in {"fedavg", "fedavgm"}:
            raise ValueError("refl_server_update must be one of {'fedavg', 'fedavgm'}")
        self.full_selection = bool(full_selection)
        self.wait_for_clients_timeout_s = float(max(0.0, wait_for_clients_timeout_s))

        self._last_success_round: Dict[str, int] = {}
        self._last_round_end_time: Optional[float] = None

    def __repr__(self) -> str:
        return (
            "REFLAsyncSiloStrategy("
            f"stale_factor={self.stale_factor}, "
            f"stale_beta={self.stale_beta}, "
            f"scale_coff={self.scale_coff}, "
            f"server_update={self.refl_server_update}, "
            f"use_num_examples={self.use_num_examples}, "
            f"full_selection={self.full_selection})"
        )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config: Dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = dict(self.on_fit_config_fn(server_round))

        now = time.time()
        config["server_send_time"] = float(now)
        config["server_wait_time"] = (
            float(max(0.0, now - self._last_round_end_time))
            if self._last_round_end_time is not None
            else 0.0
        )
        config["server_round"] = int(server_round)

        parameters_to_send = parameters
        payload_label = "float32"

        if self.downlink_quantizer is not None:
            float_weights = parameters_to_ndarrays(parameters)
            quantized_weights, _ = self.downlink_quantizer.encode(float_weights)
            parameters_to_send = ndarrays_to_parameters(quantized_weights)
            payload_label = f"int{self.downlink_quant_bits or self.downlink_quantizer.num_bits}"

        payload_bytes = sum(len(tensor) for tensor in parameters_to_send.tensors)
        payload_mb = payload_bytes / 1_000_000
        payload_mib = payload_bytes / (1024**2)

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        if self.wait_for_clients_timeout_s > 0:
            _ = client_manager.wait_for(min_num_clients, timeout=self.wait_for_clients_timeout_s)

        if self.full_selection:
            available = client_manager.all()
            clients = list(available.values())
        else:
            clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        if clients:
            logging.info(
                "[Server][REFL] Round %s: sending %s payload %.3f MB (%.3f MiB) to %s clients "
                "(available=%s full_selection=%s)",
                server_round,
                payload_label,
                payload_mb,
                payload_mib,
                len(clients),
                client_manager.num_available(),
                self.full_selection,
            )
        else:
            logging.warning("[Server][REFL] Round %s: no clients selected", server_round)

        fit_ins = FitIns(parameters_to_send, config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        assert self.initial_parameters is not None, "Initial parameters must be set."
        current_weights = parameters_to_ndarrays(self.initial_parameters)

        entries: List[Dict[str, object]] = []
        round_receive_time = time.time()
        for client_proxy, fit_res in results:
            cid = str(getattr(client_proxy, "cid", "?"))
            tensors = parameters_to_ndarrays(fit_res.parameters)
            tensors, was_quantized = maybe_unpack_quantized(tensors)

            metrics = dict(fit_res.metrics or {})
            metrics.setdefault("server_arrival_time", round_receive_time)
            if was_quantized:
                metrics["quant_received"] = 1.0
            fit_res.metrics = metrics

            request_round_metric = _safe_float(metrics.get("refl_request_round"))
            if request_round_metric is not None:
                stale_rounds = max(0, int(server_round) - int(request_round_metric))
            else:
                last_round = self._last_success_round.get(cid)
                stale_rounds = 0 if last_round is None else max(0, int(server_round) - int(last_round) - 1)
            expired = self.stale_update >= 0 and stale_rounds > self.stale_update

            entries.append(
                {
                    "cid": cid,
                    "proxy": client_proxy,
                    "fit_res": fit_res,
                    "weights": tensors,
                    "num_examples": float(max(1, fit_res.num_examples)),
                    "stale_rounds": float(stale_rounds),
                    "ratio": 0.0,
                    "expired": expired,
                    "importance": 1.0,
                    "weighted_examples": 0.0,
                }
            )

        active_entries = [e for e in entries if not bool(e["expired"])]
        if not active_entries:
            logging.warning(
                "[Server][REFL] Round %s: all updates expired by stale_update=%s, skipping aggregation",
                server_round,
                self.stale_update,
            )
            return None, {}
        for entry in entries:
            if bool(entry["expired"]):
                entry["importance"] = 0.0
                entry["weighted_examples"] = 0.0

        stale_entries = [e for e in active_entries if float(e["stale_rounds"]) > 0.0]
        new_entries = [e for e in active_entries if float(e["stale_rounds"]) <= 0.0]

        if stale_entries:
            if new_entries:
                # Keep the same algebraic form as REFL:
                # val1 += || update/(n+1) + param/(n+1) - param/n ||^2
                # val2 += || param/n ||^2
                new_count = float(len(new_entries))
                sum_new_weights: List[np.ndarray] = []
                for layer_idx in range(len(current_weights)):
                    layer = np.zeros_like(current_weights[layer_idx], dtype=np.float64)
                    for entry in new_entries:
                        layer += np.asarray(entry["weights"][layer_idx], dtype=np.float64)
                    sum_new_weights.append(layer)
            else:
                new_count = 0.0
                sum_new_weights = [np.asarray(layer, dtype=np.float64) for layer in current_weights]

            for entry in stale_entries:
                stale_weights = entry["weights"]
                if new_entries:
                    val1 = 0.0
                    val2 = 0.0
                    for w_stale, w_sum_new in zip(stale_weights, sum_new_weights):
                        stale_arr = np.asarray(w_stale, dtype=np.float64)
                        sum_arr = np.asarray(w_sum_new, dtype=np.float64)
                        lhs = stale_arr / (new_count + 1.0) + sum_arr / (new_count + 1.0) - sum_arr / new_count
                        rhs = sum_arr / new_count
                        val1 += float(np.linalg.norm(lhs) ** 2)
                        val2 += float(np.linalg.norm(rhs) ** 2)
                    entry["ratio"] = float(abs(val1 / max(val2, 1e-12)))
                else:
                    avg_new_norm_sq = float(
                        sum(np.linalg.norm(np.asarray(layer, dtype=np.float64)) ** 2 for layer in sum_new_weights)
                    )
                    avg_new_norm_sq = max(avg_new_norm_sq, 1e-12)
                    ratio_num = float(
                        sum(
                            np.linalg.norm(np.asarray(w_stale, dtype=np.float64) - w_avg_new) ** 2
                            for w_stale, w_avg_new in zip(stale_weights, sum_new_weights)
                        )
                    )
                    entry["ratio"] = float(abs(ratio_num / avg_new_norm_sq))

        max_ratio = max([float(e["ratio"]) for e in stale_entries], default=1.0)
        max_ratio = max(max_ratio, 1e-12)
        avg_stale = float(np.mean([float(e["stale_rounds"]) + 1.0 for e in stale_entries])) if stale_entries else 1.0

        for entry in active_entries:
            stale_rounds = float(entry["stale_rounds"])
            if stale_rounds <= 0:
                entry["importance"] = 1.0
                continue

            importance = 1.0
            if self.stale_factor > 1:
                importance /= self.stale_factor
            elif self.stale_factor == 1:
                pass
            elif self.stale_factor == -1:
                importance /= max(avg_stale, 1.0)
            elif self.stale_factor == -2:
                importance /= (stale_rounds + 1.0)
            elif self.stale_factor == -3:
                importance *= math.exp(-(stale_rounds + 1.0))
            else:
                ratio = float(entry["ratio"])
                importance *= (1.0 - self.stale_beta) / (stale_rounds + 1.0) + self.stale_beta * (
                    1.0 - (math.exp(-ratio / max_ratio) / self.scale_coff)
                )
            entry["importance"] = float(max(importance, 1e-12))

        coeffs: List[float] = []
        for entry in active_entries:
            base = float(entry["num_examples"]) if self.use_num_examples else 1.0
            coeff = base * float(entry["importance"])
            entry["weighted_examples"] = coeff
            coeffs.append(coeff)

        coeff_sum = float(sum(coeffs))
        if coeff_sum <= 0:
            coeffs = [1.0 for _ in active_entries]
            coeff_sum = float(len(active_entries))

        normalized = [float(c / coeff_sum) for c in coeffs]
        refl_avg: NDArrays = []
        for layer_idx in range(len(current_weights)):
            layer = np.zeros_like(current_weights[layer_idx], dtype=np.float64)
            for coeff, entry in zip(normalized, active_entries):
                layer += coeff * np.asarray(entry["weights"][layer_idx], dtype=np.float64)
            refl_avg.append(layer.astype(current_weights[layer_idx].dtype, copy=False))

        if self.refl_server_update == "fedavgm":
            pseudo_gradient: NDArrays = [w_curr - w_avg for w_curr, w_avg in zip(current_weights, refl_avg)]
            if self.momentum_vector is None or len(self.momentum_vector) != len(pseudo_gradient):
                self.momentum_vector = list(pseudo_gradient)
            else:
                self.momentum_vector = [
                    self.server_momentum * m_prev + g for m_prev, g in zip(self.momentum_vector, pseudo_gradient)
                ]

            aggregated_weights: NDArrays = [
                w - self.server_learning_rate * step for w, step in zip(current_weights, self.momentum_vector)
            ]
        else:
            aggregated_weights = refl_avg
            self.momentum_vector = None

        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)
        self.initial_parameters = parameters_aggregated

        for entry in entries:
            self._last_success_round[str(entry["cid"])] = int(server_round)
        self._last_round_end_time = time.time()

        self._write_csv(server_round=server_round, entries=entries)

        metrics_aggregated: Dict[str, Scalar] = {
            "refl_updates_total": float(len(entries)),
            "refl_updates_used": float(len(active_entries)),
            "refl_updates_expired": float(len(entries) - len(active_entries)),
            "refl_updates_new": float(len(new_entries)),
            "refl_updates_stale": float(len(stale_entries)),
            "refl_avg_importance": float(np.mean([float(e["importance"]) for e in active_entries])),
            "refl_avg_stale_rounds": float(np.mean([float(e["stale_rounds"]) for e in active_entries])),
            "refl_server_update_fedavgm": 1.0 if self.refl_server_update == "fedavgm" else 0.0,
        }
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated.update(self.fit_metrics_aggregation_fn(fit_metrics))
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def _write_csv(self, *, server_round: int, entries: List[Dict[str, object]]) -> None:
        if not self.csv_log_path:
            return

        csv_dir = os.path.dirname(self.csv_log_path) or "."
        os.makedirs(csv_dir, exist_ok=True)
        file_exists = os.path.exists(self.csv_log_path)
        header = [
            "server_round",
            "client_id",
            "num_examples",
            "server_to_client_ms",
            "server_wait_ms",
            "client_train_s",
            "edcode_s",
            "client_to_server_ms",
            "server_receive_time",
            "config_time",
            "cpu_freq_start_mhz",
            "cpu_freq_end_mhz",
            "cpu_time_start_s",
            "cpu_time_end_s",
            "cpu_time_elapsed_s",
            "sched_run_time_ns",
            "sched_runqueue_time_ns",
            "sched_runqueue_time_s",
            "sched_timeslices",
            "fit_elapsed_s",
            "quant_applied",
            "quant_received",
            "refl_stale_rounds",
            "refl_ratio",
            "refl_importance",
            "refl_weighted_examples",
            "refl_expired",
        ]

        with open(self.csv_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)

            server_receive_time = time.time()
            for entry in entries:
                fit_res = entry["fit_res"]
                m = fit_res.metrics or {}

                stc = _safe_float(m.get("server_to_client_ms"))
                swt = _safe_float(m.get("server_wait_time"))
                train_s = _safe_float(m.get("train_time"))
                edcode = _safe_float(m.get("edcode"))
                config_time = _safe_float(m.get("config_time"))
                fit_elapsed_s = _safe_float(m.get("fit_elapsed_s"))

                cpu_start = _safe_float(m.get("cpu_freq_start_mhz"))
                cpu_end = _safe_float(m.get("cpu_freq_end_mhz"))
                cpu_time_start = _safe_float(m.get("cpu_time_start_s"))
                cpu_time_end = _safe_float(m.get("cpu_time_end_s"))
                cpu_time_elapsed = _safe_float(m.get("cpu_time_elapsed_s"))
                sched_run_ns = _safe_float(m.get("sched_run_time_ns"))
                sched_runqueue_ns = _safe_float(m.get("sched_runqueue_time_ns"))
                sched_runqueue_s = _safe_float(m.get("sched_runqueue_time_s"))
                sched_timeslices = _safe_float(m.get("sched_timeslices"))

                quant_applied = _safe_float(m.get("quant_applied"))
                quant_received = _safe_float(m.get("quant_received"))

                c2s_ms = None
                client_fit_end_time = _safe_float(m.get("client_fit_end_time"))
                if client_fit_end_time is not None:
                    c2s_ms = max(0.0, (server_receive_time - client_fit_end_time) * 1000.0)

                writer.writerow(
                    [
                        server_round,
                        entry["cid"],
                        fit_res.num_examples,
                        f"{stc:.3f}" if stc is not None else "",
                        f"{swt:.3f}" if swt is not None else "",
                        f"{train_s:.3f}" if train_s is not None else "",
                        f"{edcode:.3f}" if edcode is not None else "",
                        f"{c2s_ms:.3f}" if c2s_ms is not None else "",
                        f"{server_receive_time:.3f}",
                        f"{time.time() - config_time:.3f}" if config_time is not None else "",
                        f"{cpu_start:.3f}" if cpu_start is not None else "",
                        f"{cpu_end:.3f}" if cpu_end is not None else "",
                        f"{cpu_time_start:.6f}" if cpu_time_start is not None else "",
                        f"{cpu_time_end:.6f}" if cpu_time_end is not None else "",
                        f"{cpu_time_elapsed:.6f}" if cpu_time_elapsed is not None else "",
                        f"{sched_run_ns:.0f}" if sched_run_ns is not None else "",
                        f"{sched_runqueue_ns:.0f}" if sched_runqueue_ns is not None else "",
                        f"{sched_runqueue_s:.6f}" if sched_runqueue_s is not None else "",
                        f"{sched_timeslices:.0f}" if sched_timeslices is not None else "",
                        f"{fit_elapsed_s:.6f}" if fit_elapsed_s is not None else "",
                        f"{quant_applied:.0f}" if quant_applied is not None else "",
                        f"{quant_received:.0f}" if quant_received is not None else "",
                        f"{float(entry['stale_rounds']):.3f}",
                        f"{float(entry['ratio']):.8f}",
                        f"{float(entry['importance']):.8f}",
                        f"{float(entry['weighted_examples']):.8f}",
                        "1" if bool(entry["expired"]) else "0",
                    ]
                )
