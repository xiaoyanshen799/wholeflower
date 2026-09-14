"""Contract tests; no datasets, GPUs or extra profiling runs required."""

import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from flwr.client import NumPyClient
from flwr.common import Code, FitIns, FitRes, Parameters, Status, ndarrays_to_parameters, parameters_to_ndarrays, serde
from flwr.server.server import fit_clients
from flwr.server.strategy import FedAvg

from pacer.client import PacerNumPyClient
from pacer.external import FileControlSource, JsonlSink
from pacer.feedback import GammaEstimator
from pacer.protocol import ControlConfig, Settings
from pacer.server import IncompleteRoundError, PacerStrategy, RoundMonitor


def settings(**overrides):
    values = dict(run_id="test", config_version=1, theta_target_s=4.5,
                  deadline_s=4.8, ref_a=0.02, q=0.9, q_lower=0.85, q_upper=0.95,
                  required_client_ids=["0", "1", "2"], gamma_window=50, gamma_min_samples=3)
    return ControlConfig.from_dict({**values, **overrides})


class MemorySink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def responses(cfg, server_round, gammas=(1.0, 1.2, 0.8), sample_end=None):
    result = []
    for i, gamma in enumerate(gammas):
        metrics = {
            "pacer.client_id": str(i), "pacer.run_id": cfg.run_id,
            "pacer.config_version": cfg.config_version, "pacer.round": server_round,
            "pacer.gamma": gamma, "pacer.feedback_valid": True,
            "pacer.sample_end_round": server_round if sample_end is None else sample_end,
            "pacer.n_samples": 3, "pacer.resource_status": "external_managed",
            "pacer.timing_basis": cfg.timing_basis, "train_time": 4.5,
        }
        params = ndarrays_to_parameters([np.array([float(i)], dtype=np.float32)])
        result.append((SimpleNamespace(cid=str(i)), FitRes(Status(Code.OK, ""), params, i + 1, metrics)))
    return result


class PacerTests(unittest.TestCase):
    def test_scalar_wire_roundtrip(self):
        cfg = settings()
        payload = Parameters([b"opaque-quantized-payload"], "custom")
        ins = FitIns(payload, cfg.to_wire(1))
        restored = serde.fit_ins_from_proto(serde.fit_ins_to_proto(ins))
        self.assertEqual(restored, ins)
        parsed, server_round = Settings.from_wire(restored.config)
        self.assertEqual(parsed.theta_target_s, 4.5)
        self.assertEqual(server_round, 1)
        res = responses(cfg, 1)[0][1]
        self.assertEqual(serde.fit_res_from_proto(serde.fit_res_to_proto(res)), res)

    def test_invalid_settings(self):
        for update in ({"ref_a": 0}, {"q": float("nan")}, {"q_lower": 0.95},
                       {"config_version": True}, {"gamma_min_samples": 100},
                       {"required_client_ids": ["0", "0"]}, {"timing_basis": "end_to_end"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                settings(**update)

    def test_config_versions_and_future_activation(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "control.json"
            cfg = settings()
            path.write_text(json.dumps(cfg.to_dict()))
            sink = MemorySink()
            source = FileControlSource(path, sink)
            updated = replace(cfg, config_version=2, theta_target_s=4.4, effective_round=4)
            path.write_text(json.dumps(updated.to_dict()))
            self.assertEqual(source.snapshot_for_round(2), cfg)
            path.write_text("{broken json")
            self.assertEqual(source.snapshot_for_round(3), cfg)
            self.assertEqual(source.snapshot_for_round(4), updated)
            path.write_text(json.dumps(replace(updated, theta_target_s=1.0).to_dict()))
            self.assertEqual(source.snapshot_for_round(5), updated)
            path.write_text(json.dumps(cfg.to_dict()))
            self.assertEqual(source.snapshot_for_round(6), updated)
            self.assertTrue(any(e["event"] == "config_rejected" for e in sink.events))

    def test_gamma_recovery_and_reset(self):
        cfg = settings(gamma_window=200, gamma_min_samples=20)
        estimator = GammaEstimator()
        true_gamma = 1.7
        # Quantiles of F_star(t)^gamma provide a deterministic distribution fixture.
        u = (np.arange(200) + 0.5) / 200
        f = u ** (1 / true_gamma)
        times = cfg.theta_target_s + cfg.ref_a * cfg.theta_target_s * np.log(f / (1 - f))
        for r, value in enumerate(times, 1):
            estimate = estimator.observe(float(value), r, cfg)
        self.assertAlmostEqual(estimate.gamma, true_gamma, delta=0.03)
        duplicate = estimator.observe(4.5, 200, cfg)
        self.assertIsNone(duplicate.gamma)
        changed = estimator.observe(4.5, 201, replace(cfg, config_version=2))
        self.assertEqual(changed.n_samples, 1)
        self.assertIsNone(changed.gamma)
        changed = estimator.observe(4.6, 202, replace(cfg, config_version=2), "new-quota")
        self.assertEqual(changed.n_samples, 1)

    def test_invalid_or_constant_samples_are_not_ready(self):
        cfg = settings()
        estimator = GammaEstimator()
        for r in range(1, 4):
            estimate = estimator.observe(4.5, r, cfg)
        self.assertEqual(estimate.reason, "degenerate_samples")
        estimate = estimator.observe(float("nan"), 4, cfg)
        self.assertIsNone(estimate.gamma)
        self.assertEqual(estimate.n_samples, 0)

    def test_probability_and_band_boundaries(self):
        cfg = settings()
        report, incomplete = RoundMonitor().observe(cfg, 3, responses(cfg, 3), {"0", "1", "2"}, [])
        self.assertFalse(incomplete)
        self.assertEqual(report["state"], "within_band")
        self.assertAlmostEqual(report["p_hat"], 0.9001829592716926)
        self.assertEqual(report["gamma_sum"], 3.0)
        boundary = replace(cfg, q=report["p_hat"], q_lower=report["p_hat"], q_upper=report["p_hat"])
        report, _ = RoundMonitor().observe(boundary, 3, responses(boundary, 3), {"0", "1", "2"}, [])
        self.assertEqual(report["state"], "within_band")

    def test_patience_dedup_and_recovery(self):
        cfg = settings()
        monitor = RoundMonitor()
        events = []
        for r in range(3, 8):
            report, _ = monitor.observe(cfg, r, responses(cfg, r, (2, 2, 2)), {"0", "1", "2"}, [])
            events.append(report["recalibration_requested"])
        self.assertEqual(events, [False, False, True, False, False])
        monitor.observe(cfg, 8, responses(cfg, 8), {"0", "1", "2"}, [])
        for r in range(9, 12):
            report, _ = monitor.observe(cfg, r, responses(cfg, r, (2, 2, 2)), {"0", "1", "2"}, [])
        self.assertTrue(report["recalibration_requested"])

    def test_missing_feedback_and_cached_samples_do_not_trigger(self):
        cfg = settings()
        monitor = RoundMonitor()
        for r in range(3, 6):
            report, _ = monitor.observe(cfg, r, responses(cfg, r, (2, 2, 2), sample_end=3), {"0", "1", "2"}, [])
            self.assertFalse(report["recalibration_requested"])
        results = responses(cfg, 6)
        del results[0][1].metrics["pacer.gamma"]
        report, incomplete = monitor.observe(cfg, 6, results, {"0", "1", "2"}, [])
        self.assertFalse(incomplete)
        self.assertEqual(report["state"], "insufficient_feedback")
        self.assertNotIn("p_hat", report)

    def test_feedback_gap_does_not_repeat_the_same_request(self):
        cfg = settings(violation_patience=1)
        monitor = RoundMonitor()
        report, _ = monitor.observe(cfg, 3, responses(cfg, 3, (2, 2, 2)), {"0", "1", "2"}, [])
        self.assertTrue(report["recalibration_requested"])
        results = responses(cfg, 4)
        results[0][1].metrics["pacer.feedback_valid"] = False
        monitor.observe(cfg, 4, results, {"0", "1", "2"}, [])
        report, _ = monitor.observe(cfg, 5, responses(cfg, 5, (2, 2, 2)), {"0", "1", "2"}, [])
        self.assertFalse(report["recalibration_requested"])

    def test_resource_ack_and_mid_training_change(self):
        with tempfile.TemporaryDirectory() as root:
            state_path = Path(root) / "resource.json"
            state = dict(run_id="test", config_version=1, client_id="0", resource_version=1, status="applied")

            class Inner(NumPyClient):
                def fit(self, parameters, config):
                    if config.get("change_resource"):
                        state_path.write_text(json.dumps({**state, "resource_version": 2}))
                    return parameters, 5, {"train_time": 4.5}

            wrapper = PacerNumPyClient(Inner(), "0", MemorySink(), state_path)
            payload = [np.zeros(1)]
            _, _, metrics = wrapper.fit(payload, settings().to_wire(1))
            self.assertEqual(metrics["pacer.resource_status"], "unconfirmed")
            self.assertFalse(metrics["pacer.feedback_valid"])
            state_path.write_text(json.dumps(state))
            _, _, metrics = wrapper.fit(payload, settings().to_wire(2))
            self.assertEqual(metrics["pacer.n_samples"], 1)
            _, _, metrics = wrapper.fit(payload, {**settings().to_wire(3), "change_resource": True})
            self.assertEqual(metrics["pacer.feedback_reason"], "resource_unconfirmed")
            self.assertEqual(metrics["pacer.n_samples"], 0)
            _, _, metrics = wrapper.fit(payload, settings().to_wire(4))
            self.assertEqual(metrics["pacer.n_samples"], 1)

    def test_client_rejects_replayed_and_mutated_commands(self):
        class Inner(NumPyClient):
            def fit(self, parameters, config):
                return parameters, 1, {"train_time": 4.5}

        wrapper = PacerNumPyClient(Inner(), "0", MemorySink())
        wrapper.fit([], settings().to_wire(1))
        with self.assertRaises(ValueError):
            wrapper.fit([], settings().to_wire(1))
        with self.assertRaises(ValueError):
            wrapper.fit([], settings(theta_target_s=4.4).to_wire(2))
        with self.assertRaises(ValueError):
            wrapper.fit([], {})

    def test_identity_wrong_version_and_failed_training(self):
        cfg = settings()
        for key, value in (("pacer.client_id", "1"), ("pacer.config_version", 99), ("pacer.round", True)):
            results = responses(cfg, 3)
            results[0][1].metrics[key] = value
            _, incomplete = RoundMonitor().observe(cfg, 3, results, {"0", "1", "2"}, [])
            self.assertTrue(incomplete)
        results = responses(cfg, 3)
        results[0][1].status = Status(Code.FIT_NOT_IMPLEMENTED, "no fit")
        _, incomplete = RoundMonitor().observe(cfg, 3, results, {"0", "1", "2"}, [])
        self.assertTrue(incomplete)

    def test_strategy_preserves_weighted_aggregation_and_config_isolation(self):
        cfg = settings()
        base = FedAvg(fraction_fit=1, fraction_evaluate=0, min_fit_clients=3,
                      min_available_clients=3, on_fit_config_fn=lambda r: {"local_epochs": 1})
        source = SimpleNamespace(snapshot_for_round=lambda r: cfg)
        sink = MemorySink()
        wrapper = PacerStrategy(base, source, sink)
        clients = [SimpleNamespace(cid=str(i)) for i in range(3)]
        manager = SimpleNamespace(num_available=lambda: 3, sample=lambda **kw: clients)
        payload = ndarrays_to_parameters([np.array([0.0], dtype=np.float32)])
        instructions = wrapper.configure_fit(3, payload, manager)
        self.assertEqual(len({id(ins.config) for _, ins in instructions}), 3)
        self.assertTrue(all(ins.parameters is payload for _, ins in instructions))
        results = responses(cfg, 3)
        results[0][1].metrics["pacer.feedback_valid"] = False
        actual, metrics = wrapper.aggregate_fit(3, results, [])
        expected, _ = base.aggregate_fit(3, results, [])
        np.testing.assert_array_equal(parameters_to_ndarrays(actual)[0], parameters_to_ndarrays(expected)[0])
        self.assertEqual(metrics["pacer.state"], "insufficient_feedback")
        wrapper.configure_fit(4, payload, manager)
        with self.assertRaises(IncompleteRoundError):
            wrapper.aggregate_fit(4, responses(cfg, 4)[:2], [RuntimeError("disconnected")])

    def test_client_real_compressed_and_uncompressed_paths(self):
        from client import FlowerClient
        from compression import ErrorFeedbackQuantizer, maybe_unpack_quantized

        class Model:
            def set_weights(self, weights):
                self.weights = weights

            def get_weights(self):
                return self.weights

            def fit(self, *args, **kwargs):
                self.weights = [self.weights[0] + 1]

        weights = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        for compressed in (False, True):
            inner = FlowerClient.__new__(FlowerClient)
            inner.model = Model()
            inner._use_dataset = False
            inner.x_train = inner.y_train = np.zeros((3, 1))
            inner._local_epochs_override = inner._batch_size_override = None
            inner.enable_compression = compressed
            inner._quantizer = ErrorFeedbackQuantizer(8) if compressed else None
            inner.cid = "0"
            inner._measurement_session = None
            inner._measurement_round = 0
            wrapper = PacerNumPyClient(inner, "0", MemorySink())
            config = {**settings().to_wire(1), "local_epochs": 1, "batch_size": 1}
            payload, n, metrics = wrapper.fit(weights, config)
            decoded, was_quantized = maybe_unpack_quantized(payload)
            self.assertEqual(was_quantized, compressed)
            np.testing.assert_allclose(decoded[0], weights[0] + 1, atol=0.02)
            self.assertEqual(n, 3)
            self.assertEqual(metrics["quant_applied"], float(compressed))
            self.assertEqual(metrics["pacer.n_samples"], 1)
            self.assertFalse(metrics["pacer.feedback_valid"])

    def test_venv_parallel_dispatch_and_status(self):
        barrier = threading.Barrier(2)
        captured = []

        class Proxy:
            def __init__(self, cid, code=Code.OK):
                self.cid, self.code = cid, code

            def fit(self, ins, timeout):
                captured.append(ins.config)
                barrier.wait(timeout=5)
                time.sleep(0.01)
                return FitRes(Status(self.code, ""), ins.parameters, 1, {})

        shared = FitIns(Parameters([], "test"), {})
        results, failures = fit_clients([(Proxy("0"), shared), (Proxy("1"), shared)], 2, 5)
        self.assertFalse(failures)
        self.assertIsNot(captured[0], captured[1])
        self.assertEqual(shared.config, {})
        for _, res in results:
            self.assertGreaterEqual(res.metrics["server_collection_time"], res.metrics["server_arrival_time"])
            self.assertGreater(res.metrics["pacer.client_rpc_elapsed_s"], 0)
            self.assertGreaterEqual(res.metrics["pacer.dispatch_barrier_elapsed_s"], res.metrics["pacer.client_rpc_elapsed_s"])
        results, failures = fit_clients([(Proxy("0"), shared), (Proxy("1", Code.FIT_NOT_IMPLEMENTED), shared)], 2, 5)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(failures), 1)


if __name__ == "__main__":
    unittest.main()
