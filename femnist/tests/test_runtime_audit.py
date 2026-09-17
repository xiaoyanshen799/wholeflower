"""Resource and log audit tests with a fake model; no real training."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters

from client import FlowerClient
from strategy import AUDIT_CSV_COLUMNS, QuantizedFedAvgM
from warmup.config import prepare_config
from warmup.controller import Calibration
from warmup.logs import export_stage
from warmup.measurement import MeasurementSession, validate_distinct_affinity
from warmup.simulation import simulated_stage


class RuntimeAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def client(self, monitor=None):
        client = FlowerClient.__new__(FlowerClient)
        weights = [np.array([1.0], dtype=np.float32)]
        client.model = SimpleNamespace(set_weights=lambda _: None, get_weights=lambda: weights,
                                       fit=lambda *args, **kwargs: None)
        client._use_dataset = False
        client.x_train = client.y_train = np.zeros((3, 1))
        client._local_epochs_override = client._batch_size_override = None
        client.enable_compression = False
        client._quantizer = None
        client.cid = "14"
        client._measurement_session = monitor
        client._measurement_round = 0
        return client, weights

    def test_thread_affinity_mismatch_is_rejected(self):
        monitor = MeasurementSession(self.root / "rounds.jsonl", "audit", "14", 0.5, "14")
        snapshot = {"cpu_actual": 0.5, "cpu_affinity": [14], "thread_cpu_affinities": [[14], [14, 15]]}
        with patch("warmup.measurement.cpu_snapshot", return_value=snapshot), self.assertRaises(ValueError):
            monitor.check_resources()
        self.assertFalse(monitor.output.exists())

    def test_calibrated_map_rejects_shared_or_missing_cpu(self):
        validate_distinct_affinity({"0": (0.5, "0"), "1": (0.5, "1")})
        for mapping in ({"0": (0.5, "0"), "1": (0.5, "0")}, {"0": (0.5, "-")},
                        {"0": (0.5, "0,1")}):
            with self.assertRaises(ValueError):
                validate_distinct_affinity(mapping)

    def test_federated_audit_records_full_precision_identity_quota_and_throttling(self):
        monitor = MeasurementSession(self.root / "rounds.jsonl", "audit", "14", 0.5, "14")
        client, weights = self.client(monitor)
        before = {"cpu_actual": 0.5, "cpu_affinity": [14], "thread_cpu_affinities": [[14]],
                  "quota_us": 10000, "period_us": 20000, "thread_count": 3,
                  "cgroup_usage_usec": 100, "cgroup_throttled_usec": 100, "cgroup_nr_periods": 1,
                  "cgroup_nr_throttled": 1}
        after = {**before, "cgroup_usage_usec": 100100, "cgroup_throttled_usec": 50100,
                 "cgroup_nr_periods": 10, "cgroup_nr_throttled": 3}
        with patch("warmup.measurement.cpu_snapshot", side_effect=[before, after, after]), \
                patch("client._read_cpu_freq_mhz", return_value=1000), \
                patch("client.time.perf_counter", side_effect=[10.0, 10.123456789]):
            _, _, metrics = client.fit(weights, {"local_epochs": 1, "batch_size": 64,
                                                 "server_round": 7, "training_run_id": "server-a"})
        record = json.loads(monitor.output.read_text())
        self.assertEqual(record["client_id"], "14")
        self.assertEqual(record["round"], 7)
        self.assertEqual(record["cpu_affinity"], [14])
        self.assertAlmostEqual(record["train_time_s"], 0.123456789)
        self.assertEqual(metrics["partition_id"], "14")
        self.assertEqual(metrics["training_run_id"], "server-a")
        self.assertEqual(metrics["cgroup_throttled_s"], 0.05)
        self.assertEqual(metrics["cgroup_cpu_usage_s"], 0.1)
        self.assertEqual(metrics["cgroup_throttled_periods"], 2)
        self.assertTrue(metrics["resource_verified"])

    def test_probe_is_outside_training_timer(self):
        client, weights = self.client()
        events = []
        def probe():
            events.append("probe")
            return 1000
        def timer():
            events.append("timer")
            return len(events)
        client.model.fit = lambda *args, **kwargs: events.append("fit")
        with patch("client._read_cpu_freq_mhz", side_effect=probe), patch("client.time.perf_counter", side_effect=timer):
            client.fit(weights, {"local_epochs": 1, "batch_size": 64})
        self.assertEqual(events, ["probe", "timer", "fit", "timer", "probe"])

    def test_server_logs_partition_id_and_rejects_old_csv_schema(self):
        old = self.root / "old.csv"
        old.write_text("server_round,client_id\n1,old\n")
        with self.assertRaises(ValueError):
            QuantizedFedAvgM(csv_log_path=str(old))
        self.assertEqual(old.read_text(), "server_round,client_id\n1,old\n")
        path = self.root / "new.csv"
        payload = ndarrays_to_parameters([np.array([1.0], dtype=np.float32)])
        strategy = QuantizedFedAvgM(csv_log_path=str(path), initial_parameters=payload,
                                   downlink_quantization_enabled=False)
        metrics = {"partition_id": "14", "cpu_actual": 0.5, "cpu_affinity": "14",
                   "resource_verified": True, "train_time": 0.123456789}
        result = FitRes(Status(Code.OK, ""), payload, 143, metrics)
        strategy.aggregate_fit(3, [(SimpleNamespace(cid="ipv4:127.0.0.1:9999"), result)], [])
        with path.open() as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            self.assertEqual(reader.fieldnames[-len(AUDIT_CSV_COLUMNS):], AUDIT_CSV_COLUMNS)
        self.assertEqual(rows[0]["partition_id"], "14")
        self.assertEqual(rows[0]["client_id"], "ipv4:127.0.0.1:9999")
        self.assertEqual(float(rows[0]["client_train_s"]), 0.123456789)

    def test_stage_exports_keep_all_raw_rounds_and_mark_discarded_rows(self):
        cfg = prepare_config({"dataset": "femnist", "model": "cnn", "data_dir": "unused",
                              "output_dir": str(self.root / "run")}, self.root, simulate=True)
        self.assertEqual(cfg["max_iterations"], 5)
        Calibration(cfg, simulated_stage).run()
        output = Path(cfg["output_dir"])
        source = output / "stages/scan_30/attempt_001/client_0.jsonl"
        original = source.read_bytes()
        export_stage(output, output / "stages/scan_30", cfg)
        self.assertEqual(source.read_bytes(), original)
        with (output / "timing_exports/scan_30.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 150)
        self.assertEqual(sum(row["used_for_fit"] == "True" for row in rows), 147)
        self.assertEqual(sum(row["used_for_fit"] == "False" for row in rows), 3)
        self.assertEqual({row["client_id"] for row in rows}, {"0", "1", "2"})


if __name__ == "__main__":
    unittest.main()
