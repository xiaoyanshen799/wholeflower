"""Attach FedPacer feedback without altering the wrapped model payload."""

import json
from pathlib import Path

from flwr.client import NumPyClient

from .external import JsonlSink
from .feedback import GammaEstimator
from .protocol import PREFIX, Settings, integer


class PacerNumPyClient(NumPyClient):
    def __init__(self, inner, client_id, sink=None, resource_state_path=None):
        self.inner = inner
        self.client_id = str(client_id)
        self.sink = sink or JsonlSink()
        self.estimator = GammaEstimator()
        self.resource_state_path = Path(resource_state_path) if resource_state_path else None
        self._settings = None
        self._last_round = 0

    def get_parameters(self, config):
        return self.inner.get_parameters(config)

    def get_properties(self, config):
        return self.inner.get_properties(config)

    def evaluate(self, parameters, config):
        return self.inner.evaluate(parameters, config)

    def _resource_state(self, settings):
        if self.resource_state_path is None:
            return "external_managed", "external", None
        try:
            with self.resource_state_path.open(encoding="utf-8") as handle:
                state = json.load(handle)
            if not isinstance(state, dict):
                raise ValueError("Resource state must be an object")
            if (state.get("run_id"), state.get("config_version"), state.get("client_id")) != (
                settings.run_id, settings.config_version, self.client_id
            ):
                raise ValueError("Resource state is for a different run, configuration or client")
            integer(state.get("config_version"), "resource config_version")
            revision = integer(state.get("resource_version"), "resource_version")
            if state.get("status") != "applied":
                raise ValueError("External resource allocation is not applied")
            return "applied", revision, None
        except (OSError, ValueError) as exc:
            return "unconfirmed", "unconfirmed", str(exc)

    def fit(self, parameters, config):
        if not any(key.startswith(PREFIX) for key in config):
            if self._settings is not None:
                raise ValueError("Pacer command disappeared during an active run")
            return self.inner.fit(parameters, config)
        settings, server_round = Settings.from_wire(config)
        if self._settings is not None and settings.run_id == self._settings.run_id:
            if server_round <= self._last_round or settings.config_version < self._settings.config_version:
                raise ValueError("Duplicate round or old Pacer configuration")
            if settings.config_version == self._settings.config_version and settings != self._settings:
                raise ValueError("Pacer settings changed without a new version")
        if settings != self._settings:
            self.sink.emit({"event": "target_received", "client_id": self.client_id, **settings.to_wire(server_round)})
        self._settings = settings
        self._last_round = server_round
        resource_status, resource_revision, resource_error = self._resource_state(settings)

        weights, num_examples, metrics = self.inner.fit(parameters, config)
        metrics = dict(metrics)
        # Check again after training in case an external allocator changed quota mid-fit.
        after_status, after_revision, after_error = self._resource_state(settings)
        if (after_status, after_revision) != (resource_status, resource_revision):
            resource_error = "External resources changed during training"
        resource_error = resource_error or after_error
        if resource_error:
            self.estimator = GammaEstimator()
            estimate = None
            resource_status = "unconfirmed"
        else:
            workload = (
                metrics.get("local_epochs_used", config.get("local_epochs")),
                metrics.get("batch_size_used", config.get("batch_size")), num_examples,
            )
            estimate = self.estimator.observe(
                metrics.get("train_time"), server_round, settings, resource_revision, workload
            )
        feedback = {
            PREFIX + "client_id": self.client_id,
            PREFIX + "run_id": settings.run_id,
            PREFIX + "config_version": settings.config_version,
            PREFIX + "round": server_round,
            PREFIX + "sample_end_round": estimate.sample_end_round if estimate else server_round,
            PREFIX + "n_samples": estimate.n_samples if estimate else 0,
            PREFIX + "feedback_valid": estimate is not None and estimate.gamma is not None,
            PREFIX + "feedback_reason": estimate.reason if estimate else "resource_unconfirmed",
            PREFIX + "resource_status": resource_status,
            PREFIX + "timing_basis": settings.timing_basis,
        }
        if estimate is not None and estimate.gamma is not None:
            feedback[PREFIX + "gamma"] = estimate.gamma
            feedback[PREFIX + "fit_rmse"] = estimate.rmse
        self.sink.emit({"event": "client_feedback", **feedback, "resource_error": resource_error})
        return weights, num_examples, {**metrics, **feedback}
