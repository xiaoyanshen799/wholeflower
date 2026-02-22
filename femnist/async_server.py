"""Async Flower server loop with stale-update carry-over across rounds.

This server implements a true asynchronous training loop at the orchestration
level:
- clients can keep training across aggregation boundaries (in-flight carry-over);
- each aggregation round uses updates received by a deadline, instead of waiting
  for all sampled clients;
- late updates are retained and can be aggregated in later rounds as stale.
"""

from __future__ import annotations

import concurrent.futures
import math
import time
import timeit
from dataclasses import dataclass
from logging import INFO, WARNING
from typing import Dict, List, Optional, Tuple

from flwr.common import Code, FitRes, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.history import History
from flwr.server.server import Server, fit_client


@dataclass
class _PendingFit:
    cid: str
    sent_round: int


class TrueAsyncServer(Server):
    """Server loop that aggregates on arrivals with deadline-based rounds."""

    def __init__(
        self,
        *,
        client_manager,
        strategy,
        round_deadline_s: float = 30.0,
        min_results: int = 0,
        min_results_fraction: float = 0.8,
        poll_interval_s: float = 0.5,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.round_deadline_s = float(max(0.0, round_deadline_s))
        self.min_results = int(max(0, min_results))
        self.min_results_fraction = float(min(max(min_results_fraction, 0.0), 1.0))
        self.poll_interval_s = float(max(0.05, poll_interval_s))

    def fit(self, num_rounds: int, timeout: float | None) -> tuple[History, float]:
        """Run asynchronous FL for a number of aggregation rounds."""
        history = History()

        log(INFO, "[INIT]")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        log(INFO, "Starting evaluation of initial global parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(INFO, "initial parameters (loss, other metrics): %s, %s", res[0], res[1])
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
        else:
            log(INFO, "Evaluation returned no results (`None`)")

        start_time = timeit.default_timer()
        pending: Dict[concurrent.futures.Future, _PendingFit] = {}
        in_flight_by_cid: Dict[str, concurrent.futures.Future] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for current_round in range(1, num_rounds + 1):
                log(INFO, "")
                log(INFO, "[ROUND %s]", current_round)

                launched = self._dispatch_new_fit_tasks(
                    server_round=current_round,
                    timeout=timeout,
                    executor=executor,
                    pending=pending,
                    in_flight_by_cid=in_flight_by_cid,
                )
                target = self._round_target()
                log(
                    INFO,
                    "[Async] round=%s launched=%s in_flight=%s target_results=%s deadline_s=%.2f",
                    current_round,
                    launched,
                    len(in_flight_by_cid),
                    target,
                    self.round_deadline_s,
                )

                results, failures = self._collect_round_results(
                    server_round=current_round,
                    target_results=target,
                    pending=pending,
                    in_flight_by_cid=in_flight_by_cid,
                )
                log(
                    INFO,
                    "aggregate_fit: received %s results and %s failures (pending after cut=%s)",
                    len(results),
                    len(failures),
                    len(pending),
                )

                if results or failures:
                    parameters_prime, fit_metrics = self.strategy.aggregate_fit(
                        current_round, results, failures
                    )
                    if parameters_prime:
                        self.parameters = parameters_prime
                    history.add_metrics_distributed_fit(server_round=current_round, metrics=fit_metrics)
                else:
                    log(
                        WARNING,
                        "[Async] round=%s produced no completed updates before deadline",
                        current_round,
                    )

                res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
                if res_cen is not None:
                    loss_cen, metrics_cen = res_cen
                    log(
                        INFO,
                        "fit progress: (%s, %s, %s, %s)",
                        current_round,
                        loss_cen,
                        metrics_cen,
                        timeit.default_timer() - start_time,
                    )
                    history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                    history.add_metrics_centralized(server_round=current_round, metrics=metrics_cen)

                res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
                if res_fed is not None:
                    loss_fed, evaluate_metrics_fed, _ = res_fed
                    if loss_fed is not None:
                        history.add_loss_distributed(server_round=current_round, loss=loss_fed)
                        history.add_metrics_distributed(
                            server_round=current_round, metrics=evaluate_metrics_fed
                        )

        end_time = timeit.default_timer()
        return history, end_time - start_time

    def _round_target(self) -> int:
        if self.min_results > 0:
            return self.min_results
        n_avail = max(1, int(self.client_manager().num_available()))
        return max(1, int(math.ceil(float(n_avail) * self.min_results_fraction)))

    def _dispatch_new_fit_tasks(
        self,
        *,
        server_round: int,
        timeout: float | None,
        executor: concurrent.futures.ThreadPoolExecutor,
        pending: Dict[concurrent.futures.Future, _PendingFit],
        in_flight_by_cid: Dict[str, concurrent.futures.Future],
    ) -> int:
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self.client_manager(),
        )
        if not client_instructions:
            log(INFO, "configure_fit: no clients selected, cancel")
            return 0

        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self.client_manager().num_available(),
        )

        launched = 0
        for client_proxy, fit_ins in client_instructions:
            cid = str(getattr(client_proxy, "cid", id(client_proxy)))
            if cid in in_flight_by_cid:
                continue

            fut = executor.submit(fit_client, client_proxy, fit_ins, timeout, server_round)
            pending[fut] = _PendingFit(cid=cid, sent_round=server_round)
            in_flight_by_cid[cid] = fut
            launched += 1
        return launched

    def _collect_round_results(
        self,
        *,
        server_round: int,
        target_results: int,
        pending: Dict[concurrent.futures.Future, _PendingFit],
        in_flight_by_cid: Dict[str, concurrent.futures.Future],
    ) -> tuple[List[Tuple[object, FitRes]], List[object]]:
        results: List[Tuple[object, FitRes]] = []
        failures: List[object] = []

        deadline_ts: Optional[float] = None
        if self.round_deadline_s > 0:
            deadline_ts = time.time() + self.round_deadline_s

        while True:
            self._drain_completed(
                pending=pending,
                in_flight_by_cid=in_flight_by_cid,
                results=results,
                failures=failures,
            )

            if deadline_ts is None:
                if len(results) >= target_results:
                    break
            else:
                now = time.time()
                if len(results) >= target_results and now >= deadline_ts:
                    break
                if now >= deadline_ts and results:
                    break

            if not pending:
                if results:
                    break
                break

            wait_timeout = self.poll_interval_s
            if deadline_ts is not None:
                remaining = deadline_ts - time.time()
                wait_timeout = max(0.0, min(wait_timeout, remaining))

            done, _ = concurrent.futures.wait(
                fs=set(pending.keys()),
                timeout=wait_timeout,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            if not done and deadline_ts is not None and time.time() >= deadline_ts and not results:
                concurrent.futures.wait(
                    fs=set(pending.keys()),
                    timeout=None,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

            if deadline_ts is not None and time.time() >= deadline_ts and results:
                self._drain_completed(
                    pending=pending,
                    in_flight_by_cid=in_flight_by_cid,
                    results=results,
                    failures=failures,
                )
                break

        stale_count = 0
        for _, fit_res in results:
            req_round = fit_res.metrics.get("refl_request_round") if fit_res.metrics else None
            if req_round is not None and int(req_round) < server_round:
                stale_count += 1
        log(
            INFO,
            "[Async] round=%s used_updates=%s stale_updates=%s failures=%s pending_carry=%s",
            server_round,
            len(results),
            stale_count,
            len(failures),
            len(pending),
        )
        return results, failures

    def _drain_completed(
        self,
        *,
        pending: Dict[concurrent.futures.Future, _PendingFit],
        in_flight_by_cid: Dict[str, concurrent.futures.Future],
        results: List[Tuple[object, FitRes]],
        failures: List[object],
    ) -> None:
        completed = [future for future in list(pending.keys()) if future.done()]
        if not completed:
            return

        for future in completed:
            meta = pending.pop(future)
            in_flight_by_cid.pop(meta.cid, None)

            try:
                client_proxy, fit_res = future.result()
                metrics = dict(fit_res.metrics or {})
                metrics["server_arrival_time"] = float(time.time())
                metrics["refl_request_round"] = float(meta.sent_round)
                fit_res.metrics = metrics

                if fit_res.status.code == Code.OK:
                    results.append((client_proxy, fit_res))
                else:
                    failures.append((client_proxy, fit_res))
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)
