"""FedAvg strategy with per-client timing CSV logging for PyTorch runs."""

from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import FitIns, FitRes, MetricsAggregationFn, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class CSVFedAvg(FedAvg):
    def __init__(
        self,
        *,
        csv_log_path: Optional[str] = None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
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

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config: Dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = dict(self.on_fit_config_fn(server_round))

        now = float(time.time())
        config["server_send_time"] = now
        config["config_time"] = now

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        if self.csv_log_path and results:
            csv_dir = os.path.dirname(self.csv_log_path)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)
            file_exists = os.path.exists(self.csv_log_path)
            with open(self.csv_log_path, "a", newline="", encoding="utf-8") as f:
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

                for client_proxy, fit_res in results:
                    m = dict(fit_res.metrics or {})
                    server_receive_time = float(time.time())
                    m["server_arrival_time"] = server_receive_time

                    cid = getattr(client_proxy, "cid", "?")
                    stc = m.get("server_to_client_ms")
                    swt = m.get("server_wait_time")
                    train_s = m.get("train_time")
                    edcode = m.get("edcode")
                    config_time = m.get("config_time")
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
                    client_fit_end = m.get("client_fit_end_time")
                    if isinstance(client_fit_end, (int, float)):
                        c2s_ms = max(0.0, (server_receive_time - float(client_fit_end)) * 1000.0)

                    writer.writerow(
                        [
                            server_round,
                            cid,
                            fit_res.num_examples,
                            f"{float(stc):.3f}" if isinstance(stc, (int, float)) else "",
                            f"{float(swt):.3f}" if isinstance(swt, (int, float)) else "",
                            f"{float(train_s):.3f}" if isinstance(train_s, (int, float)) else "",
                            f"{float(edcode):.3f}" if isinstance(edcode, (int, float)) else "",
                            f"{float(c2s_ms):.3f}" if isinstance(c2s_ms, (int, float)) else "",
                            f"{server_receive_time:.3f}",
                            f"{(server_receive_time - float(config_time)):.3f}"
                            if isinstance(config_time, (int, float))
                            else "",
                            f"{float(cpu_start):.3f}" if isinstance(cpu_start, (int, float)) else "",
                            f"{float(cpu_end):.3f}" if isinstance(cpu_end, (int, float)) else "",
                            f"{float(cpu_time_start):.6f}" if isinstance(cpu_time_start, (int, float)) else "",
                            f"{float(cpu_time_end):.6f}" if isinstance(cpu_time_end, (int, float)) else "",
                            f"{float(cpu_time_elapsed):.6f}" if isinstance(cpu_time_elapsed, (int, float)) else "",
                            f"{float(sched_run_ns):.0f}" if isinstance(sched_run_ns, (int, float)) else "",
                            f"{float(sched_runqueue_ns):.0f}" if isinstance(sched_runqueue_ns, (int, float)) else "",
                            f"{float(sched_runqueue_s):.6f}" if isinstance(sched_runqueue_s, (int, float)) else "",
                            f"{float(sched_timeslices):.0f}" if isinstance(sched_timeslices, (int, float)) else "",
                        ]
                    )

        return super().aggregate_fit(server_round, results, failures)
