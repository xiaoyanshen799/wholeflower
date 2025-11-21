"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
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


class CustomFedAvgM(FedAvg):
    """Re-implmentation of FedAvgM.

    This implementation of FedAvgM diverges from original (Flwr v1.5.0) implementation.
    Here, the re-implementation introduces the Nesterov Accelerated Gradient (NAG),
    same as reported in the original FedAvgM paper:

    https://arxiv.org/pdf/1909.06335.pdf
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.9,
        csv_log_path: Optional[str] = None,
    ) -> None:
        """Federated Averaging with Momentum strategy.

        Implementation based on https://arxiv.org/pdf/1909.06335.pdf

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],
        Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters
            Initial global model parameters.
        server_learning_rate: float
            Server-side learning rate used in server-side optimization.
            Defaults to 1.0.
        server_momentum: float
            Server-side momentum factor used for FedAvgM. Defaults to 0.9.
        """
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

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvgM(accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        fedavg_result = aggregate(weights_results)  # parameters_aggregated from FedAvg

        # original implementation follows convention described in
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html

        # do the check for self.initial_parameters being set
        assert (
            self.initial_parameters is not None
        ), "Initial parameters must be set for CustomFedAvgM strategy"

        # remember that updates are the opposite of gradients
        pseudo_gradient: NDArrays = [
            x - y
            for x, y in zip(
                parameters_to_ndarrays(self.initial_parameters), fedavg_result
            )
        ]

        if server_round > 1:
            assert self.momentum_vector, "Momentum should have been created on round 1."

            self.momentum_vector = [
                self.server_momentum * v + w
                for w, v in zip(pseudo_gradient, self.momentum_vector)
            ]
        else:  # Round 1
            # Initialize server-side model
            assert (
                self.initial_parameters is not None
            ), "When using server-side optimization, model needs to be initialized."
            # Initialize momentum vector
            self.momentum_vector = pseudo_gradient

        # Applying Nesterov
        pseudo_gradient = [
            g + self.server_momentum * v
            for g, v in zip(pseudo_gradient, self.momentum_vector)
        ]

        # Federated Averaging with Server Momentum
        fedavgm_result = [
            w - self.server_learning_rate * v
            for w, v in zip(
                parameters_to_ndarrays(self.initial_parameters), pseudo_gradient
            )
        ]

        # Update current weights
        self.initial_parameters = ndarrays_to_parameters(fedavgm_result)

        parameters_aggregated = ndarrays_to_parameters(fedavgm_result)

        # Write per-client CSV metrics if requested
        if self.csv_log_path:
            import csv, os, time
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
                        ]
                    )
                server_receive_time = time.time()
                for client_proxy, fit_res in results:
                    cid = getattr(client_proxy, "cid", "?")
                    m = fit_res.metrics or {}
                    stc = m.get("server_to_client_ms", None)
                    swt = m.get("server_wait_time", None)
                    train_s = m.get("train_time", None)
                    print(f"train_s: {train_s}")
                    edcode = m.get("edcode", None)
                    config_time = m.get("config_time", None)
                    # Approximate client->server time using server_arrival_time
                    c2s_ms = None
                    server_arrival_time = m.get("server_arrival_time")
                    print(f"server_arrival_time: {server_arrival_time}")
                    if "client_fit_end_time" in m and server_arrival_time:
                        c2s_ms = max(0.0, (server_arrival_time - float(m["client_fit_end_time"])) * 1000.0)
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
                        ]
                    )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
