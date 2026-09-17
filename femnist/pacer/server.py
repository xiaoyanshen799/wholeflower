"""Fixed-roster monitoring around an existing Flower aggregation strategy."""

import math

import numpy as np
from flwr.common import Code, FitIns
from flwr.server.strategy import Strategy

from .external import JsonlSink
from .protocol import PREFIX, integer, positive_number


class IncompleteRoundError(RuntimeError):
    """A required client's training result or protocol identity is missing."""


class RoundMonitor:
    def __init__(self):
        self.version = None
        self.direction = None
        self.consecutive = 0
        self.notified = None
        self.last_samples = {}

    def _reset_streak(self):
        self.direction = None
        self.consecutive = 0

    def observe(self, snapshot, server_round, results, selected, failures):
        version = (snapshot.run_id, snapshot.config_version)
        if self.version != version:
            self.version = version
            self._reset_streak()
            self.notified = None
            self.last_samples = {}
        expected = set(snapshot.required_client_ids)
        seen = set()
        seen_proxies = set()
        valid = {}
        invalid = {}
        identity_errors = []
        train_times = []
        rpc_times = {}
        barrier_times = []
        for proxy, res in results:
            metrics = res.metrics or {}
            cid = metrics.get(PREFIX + "client_id")
            if proxy.cid not in selected or proxy.cid in seen_proxies:
                identity_errors.append(f"Unexpected or duplicate connection {proxy.cid}")
            seen_proxies.add(proxy.cid)
            if not isinstance(cid, str) or cid not in expected or cid in seen:
                identity_errors.append(f"Unexpected, missing or duplicate client ID {cid!r}")
                continue
            seen.add(cid)
            if res.status.code != Code.OK:
                identity_errors.append(f"Non-OK training result for {cid}")
                continue
            try:
                if metrics.get(PREFIX + "run_id") != snapshot.run_id:
                    raise ValueError("Wrong run_id")
                for name, value in (("round", server_round), ("config_version", snapshot.config_version)):
                    if integer(metrics.get(PREFIX + name), name) != value:
                        raise ValueError(f"Wrong {name}")
            except ValueError as exc:
                identity_errors.append(f"{cid}: {exc}")
                continue
            try:
                train_times.append(positive_number(metrics.get("train_time"), "train_time"))
            except ValueError:
                pass
            try:
                rpc_times[cid] = positive_number(metrics.get(PREFIX + "client_rpc_elapsed_s"), "RPC duration")
                barrier_times.append(positive_number(metrics.get(PREFIX + "dispatch_barrier_elapsed_s"), "barrier duration"))
            except ValueError:
                pass
            try:
                if metrics.get(PREFIX + "feedback_valid") is not True:
                    raise ValueError(str(metrics.get(PREFIX + "feedback_reason", "missing_feedback")))
                if metrics.get(PREFIX + "resource_status") not in ("external_managed", "applied"):
                    raise ValueError("Resources are unconfirmed")
                if metrics.get(PREFIX + "timing_basis") != snapshot.timing_basis:
                    raise ValueError("Wrong timing basis")
                n = integer(metrics.get(PREFIX + "n_samples"), "n_samples")
                if not snapshot.gamma_min_samples <= n <= snapshot.gamma_window:
                    raise ValueError("Invalid feedback sample count")
                end = integer(metrics.get(PREFIX + "sample_end_round"), "sample_end_round")
                if not 0 <= server_round - end <= snapshot.feedback_max_age_rounds:
                    raise ValueError("Stale or future feedback")
                gamma = positive_number(metrics.get(PREFIX + "gamma"), "gamma")
                valid[cid] = (gamma, end)
            except ValueError as exc:
                invalid[cid] = str(exc)

        incomplete = bool(failures or identity_errors or seen != expected or seen_proxies != selected)
        report = {
            "event": "round_observation", "run_id": snapshot.run_id,
            "config_version": snapshot.config_version, "server_round": server_round,
            "theta_target_s": snapshot.theta_target_s, "deadline_s": snapshot.deadline_s,
            "q": snapshot.q, "q_lower": snapshot.q_lower, "q_upper": snapshot.q_upper,
            "timing_basis": snapshot.timing_basis, "n_expected": len(expected),
            "n_valid": len(valid), "invalid_feedback": invalid,
            "missing_clients": sorted(expected - seen), "identity_errors": identity_errors,
            "n_failures": len(failures), "recalibration_requested": False,
            "client_rpc_elapsed_s": rpc_times,
        }
        if len(barrier_times) == len(expected) and not incomplete:
            report["dispatch_barrier_elapsed_s"] = max(barrier_times)
        if len(train_times) == len(expected) and not incomplete:
            report["local_train_round_s"] = max(train_times)
            report["observed_deadline_met"] = max(train_times) <= snapshot.deadline_s
        if incomplete or set(valid) != expected:
            self._reset_streak()
            report.update(state="incomplete_round" if incomplete else "insufficient_feedback", consecutive_checks=0)
            return report, incomplete

        try:
            gamma_sum = math.fsum(value[0] for value in valid.values())
            if not math.isfinite(gamma_sum):
                raise ValueError("Nonfinite exponent sum")
        except (ValueError, OverflowError):
            self._reset_streak()
            report.update(state="invalid_gamma_sum", consecutive_checks=0)
            return report, False
        z = (snapshot.deadline_s - snapshot.theta_target_s) / (snapshot.ref_a * snapshot.theta_target_s)
        p_hat = math.exp(gamma_sum * -float(np.logaddexp(0.0, -z)))
        state = "below_band" if p_hat < snapshot.q_lower else "above_band" if p_hat > snapshot.q_upper else "within_band"
        fresh = all(end > self.last_samples.get(cid, 0) for cid, (_, end) in valid.items())
        report.update(gamma_sum=gamma_sum, p_hat=p_hat, state=state, fresh_feedback=fresh)
        if not fresh:
            self._reset_streak()
        else:
            self.last_samples = {cid: end for cid, (_, end) in valid.items()}
            if state == "within_band":
                self._reset_streak()
                self.notified = None
            else:
                if state != self.direction:
                    self.consecutive = 0
                self.direction = state
                self.consecutive += 1
                if self.consecutive >= snapshot.violation_patience and self.notified != state:
                    report["recalibration_requested"] = True
                    self.notified = state
        report["consecutive_checks"] = self.consecutive
        return report, False


class PacerStrategy(Strategy):
    def __init__(self, base_strategy, source, sink=None):
        self.base = base_strategy
        self.source = source
        self.sink = sink or JsonlSink()
        self.monitor = RoundMonitor()
        self._pending = None
        self.last_report = None

    def initialize_parameters(self, client_manager):
        return self.base.initialize_parameters(client_manager)

    def evaluate(self, server_round, parameters):
        return self.base.evaluate(server_round, parameters)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.base.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        return self.base.aggregate_evaluate(server_round, results, failures)

    def configure_fit(self, server_round, parameters, client_manager):
        if self._pending is not None:
            raise RuntimeError("Previous Pacer round has not been aggregated")
        snapshot = self.source.snapshot_for_round(server_round)
        instructions = self.base.configure_fit(server_round, parameters, client_manager)
        selected = {proxy.cid for proxy, _ in instructions}
        if len(instructions) != len(snapshot.required_client_ids) or len(selected) != len(instructions):
            raise IncompleteRoundError("Pacer requires exactly the configured number of distinct clients every round")
        self._pending = (server_round, snapshot, selected)
        fields = snapshot.to_wire(server_round)
        return [(proxy, FitIns(ins.parameters, {**ins.config, **fields})) for proxy, ins in instructions]

    def aggregate_fit(self, server_round, results, failures):
        if self._pending is None or self._pending[0] != server_round:
            raise RuntimeError("No matching configured Pacer round")
        _, snapshot, selected = self._pending
        self._pending = None
        report, incomplete = self.monitor.observe(snapshot, server_round, results, selected, failures)
        self.last_report = report
        self.sink.emit(report)
        if incomplete:
            raise IncompleteRoundError("Required training results are incomplete; see Pacer round log")
        if report["recalibration_requested"]:
            self.sink.emit({**report, "event": "recalibration_requested", "direction": report["state"]})
        parameters, metrics = self.base.aggregate_fit(server_round, results, failures)
        monitor_metrics = {
            PREFIX + key: report[key] for key in (
                "state", "n_valid", "n_expected", "config_version", "p_hat", "gamma_sum",
                "theta_target_s", "recalibration_requested", "local_train_round_s",
            ) if key in report
        }
        return parameters, {**metrics, **monitor_metrics}
