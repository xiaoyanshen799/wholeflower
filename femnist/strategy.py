"""FedAvgM strategy with optional downlink quantization for FEMNIST scripts."""

from __future__ import annotations

import csv
import logging
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
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from compression import ErrorFeedbackQuantizer, maybe_unpack_quantized


class CsvFedAvg(FedAvg):
    """FedAvg with per-client CSV logging."""

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        csv_log_path: Optional[str] = None,
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
        )
        self.csv_log_path = csv_log_path

    def __repr__(self) -> str:
        return f"CsvFedAvg(accept_failures={self.accept_failures})"

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        if self.csv_log_path and results:
            os.makedirs(os.path.dirname(self.csv_log_path), exist_ok=True)
            file_exists = os.path.exists(self.csv_log_path)
            with open(self.csv_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        [
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
                        ]
                    )
                server_receive_time = time.time()
                for client_proxy, fit_res in results:
                    cid = getattr(client_proxy, "cid", "?")
                    m = fit_res.metrics or {}
                    stc = m.get("server_to_client_ms", None)
                    swt = m.get("server_wait_time", None)
                    train_s = m.get("train_time", None)
                    edcode = m.get("edcode", None)
                    config_time = m.get("config_time", None)
                    cpu_start = m.get("cpu_freq_start_mhz")
                    cpu_end = m.get("cpu_freq_end_mhz")
                    cpu_time_start = m.get("cpu_time_start_s")
                    cpu_time_end = m.get("cpu_time_end_s")
                    cpu_time_elapsed = m.get("cpu_time_elapsed_s")
                    sched_run_ns = m.get("sched_run_time_ns")
                    sched_runqueue_ns = m.get("sched_runqueue_time_ns")
                    sched_runqueue_s = m.get("sched_runqueue_time_s")
                    sched_timeslices = m.get("sched_timeslices")
                    c2s_ms = None
                    server_arrival_time = m.get("server_arrival_time")
                    if "client_fit_end_time" in m and server_arrival_time:
                        c2s_ms = max(
                            0.0,
                            (server_arrival_time - float(m["client_fit_end_time"])) * 1000.0,
                        )
                    writer.writerow(
                        [
                            server_round,
                            cid,
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
                        ]
                    )

        return parameters_aggregated, metrics_aggregated


class QuantizedFedAvgM(FedAvg):
    """Re-implementation of FedAvgM with downlink quantization support."""

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.9,
        csv_log_path: Optional[str] = None,
        downlink_quantization_enabled: bool = True,
        downlink_quantization_bits: int = 8,
        downlink_error_feedback: bool = True,
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
        )
        self.server_learning_rate = server_learning_rate
        self.server_momentum = server_momentum
        self.momentum_vector: Optional[NDArrays] = None
        self.csv_log_path = csv_log_path
        self.initial_parameters = initial_parameters

        self.downlink_quantizer: Optional[ErrorFeedbackQuantizer] = None
        self.downlink_quant_bits: Optional[int] = None
        if downlink_quantization_enabled:
            self.downlink_quantizer = ErrorFeedbackQuantizer(
                num_bits=downlink_quantization_bits,
                error_feedback=downlink_error_feedback,
            )
            self.downlink_quant_bits = downlink_quantization_bits

    def __repr__(self) -> str:
        return f"QuantizedFedAvgM(accept_failures={self.accept_failures})"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config: Dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

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
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        if clients:
            logging.info(
                "[Server] Round %s: sending %s payload %.3f MB (%.3f MiB) to %s clients",
                server_round,
                payload_label,
                payload_mb,
                payload_mib,
                len(clients),
            )

        fit_ins = FitIns(parameters_to_send, config)
        return [(client, fit_ins) for client in clients]

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.initial_parameters

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

        weights_results = []
        for client_proxy, fit_res in results:
            received_bytes = sum(len(tensor) for tensor in fit_res.parameters.tensors)
            logging.info(
                "[Server] Round %s: received model payload %.3f MB from client %s",
                server_round,
                received_bytes / (1024 * 1024),
                getattr(client_proxy, "cid", "?"),
            )
            tensors = parameters_to_ndarrays(fit_res.parameters)
            tensors, was_quantized = maybe_unpack_quantized(tensors)
            if was_quantized:
                metrics = dict(fit_res.metrics or {})
                metrics.setdefault("quant_received", 1.0)
                fit_res.metrics = metrics
            weights_results.append((tensors, fit_res.num_examples))

        fedavg_result = aggregate(weights_results)

        assert self.initial_parameters is not None, "Initial parameters must be set."

        current_weights = parameters_to_ndarrays(self.initial_parameters)
        pseudo_gradient: NDArrays = [
            w_curr - w_avg for w_curr, w_avg in zip(current_weights, fedavg_result)
        ]

        if self.momentum_vector is None or len(self.momentum_vector) != len(pseudo_gradient):
            self.momentum_vector = list(pseudo_gradient)
        else:
            self.momentum_vector = [
                self.server_momentum * m_prev + g for m_prev, g in zip(self.momentum_vector, pseudo_gradient)
            ]

        momentum_step = self.momentum_vector

        grad_norm = float(np.sqrt(sum(np.linalg.norm(g) ** 2 for g in pseudo_gradient)))
        mom_norm = float(np.sqrt(sum(np.linalg.norm(m) ** 2 for m in momentum_step)))

        fedavgm_result = [
            w - self.server_learning_rate * step for w, step in zip(current_weights, momentum_step)
        ]

        weight_norm = float(np.sqrt(sum(np.linalg.norm(w) ** 2 for w in fedavgm_result)))
        last_weight_shape = fedavgm_result[-2].shape if len(fedavgm_result) >= 2 else None
        last_bias_shape = fedavgm_result[-1].shape if len(fedavgm_result) >= 1 else None

        if np.isnan(grad_norm) or np.isnan(weight_norm) or np.isnan(mom_norm):
            print(
                f"[Server][Agg] round={server_round} detected NaN "
                f"grad_norm={grad_norm} mom_norm={mom_norm} weight_norm={weight_norm} "
                f"last_weight_shape={last_weight_shape} last_bias_shape={last_bias_shape}"
            )
        else:
            print(
                f"[Server][Agg] round={server_round} "
                f"grad_norm={grad_norm:.3e} mom_norm={mom_norm:.3e} "
                f"weight_norm={weight_norm:.3e} "
                f"last_weight_shape={last_weight_shape} last_bias_shape={last_bias_shape}"
            )

        self.initial_parameters = ndarrays_to_parameters(fedavgm_result)

        parameters_aggregated = ndarrays_to_parameters(fedavgm_result)

        if self.csv_log_path:
            os.makedirs(os.path.dirname(self.csv_log_path), exist_ok=True)
            file_exists = os.path.exists(self.csv_log_path)
            with open(self.csv_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        [
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
                        ]
                    )
                server_receive_time = time.time()
                for client_proxy, fit_res in results:
                    cid = getattr(client_proxy, "cid", "?")
                    m = fit_res.metrics or {}
                    stc = m.get("server_to_client_ms", None)
                    swt = m.get("server_wait_time", None)
                    train_s = m.get("train_time", None)
                    edcode = m.get("edcode", None)
                    config_time = m.get("config_time", None)
                    cpu_start = m.get("cpu_freq_start_mhz")
                    cpu_end = m.get("cpu_freq_end_mhz")
                    cpu_time_start = m.get("cpu_time_start_s")
                    cpu_time_end = m.get("cpu_time_end_s")
                    cpu_time_elapsed = m.get("cpu_time_elapsed_s")
                    sched_run_ns = m.get("sched_run_time_ns")
                    sched_runqueue_ns = m.get("sched_runqueue_time_ns")
                    sched_runqueue_s = m.get("sched_runqueue_time_s")
                    sched_timeslices = m.get("sched_timeslices")
                    c2s_ms = None
                    server_arrival_time = m.get("server_arrival_time")
                    if "client_fit_end_time" in m and server_arrival_time:
                        c2s_ms = max(
                            0.0,
                            (server_arrival_time - float(m["client_fit_end_time"])) * 1000.0,
                        )
                    writer.writerow(
                        [
                            server_round,
                            cid,
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
                        ]
                    )

        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
