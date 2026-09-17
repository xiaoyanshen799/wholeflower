"""Exercise the real CLI/local loop with model and transport dependencies mocked."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from warmup.measurement import atomic_json


class ClientEntryTests(unittest.TestCase):
    def test_local_loop_writes_precision_metadata_and_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Path(__file__).resolve().parents[1] / "run_client.py"
            dependencies = {name: MagicMock() for name in ("flwr", "tensorflow", "client", "fedavgm.dataset", "grpc")}
            spec = importlib.util.spec_from_file_location("warmup_test_run_client", source)
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, dependencies):
                spec.loader.exec_module(module)
            client = dependencies["client"].FlowerClient.return_value
            durations = [10.987654321, 2.123456789, 2.234567891]
            client.fit.side_effect = [([], 9, {"train_time": value}) for value in durations]
            client.get_parameters.return_value = []
            atomic_json(root / "start.json", {"stage_id": "test"})
            args = ["run_client.py", "--cid", "0", "--dataset", "mnist", "--model", "cnn",
                    "--local-only", "--local-rounds", "3", "--epochs", "2", "--batch-size", "4",
                    "--local-seed", "42", "--local-stage-id", "test",
                    "--local-timing-jsonl", str(root / "timing.jsonl"),
                    "--local-cpu-fraction", "0.3", "--local-cpu-affinity", "2",
                    "--local-ready-file", str(root / "ready.json"), "--local-start-file", str(root / "start.json"),
                    "--log-file", str(root / "training.log")]
            with patch.object(sys, "argv", args), patch.object(module, "_load_partition") as partition, \
                    patch.object(module.logging, "basicConfig"), patch("warmup.measurement.cpu_snapshot") as snapshot:
                partition.return_value = (np.zeros((10, 28, 28), dtype=np.float32), np.zeros(10, dtype=int))
                snapshot.return_value = {"cpu_actual": 0.3, "cpu_affinity": [2]}
                module.main()
            rows = [json.loads(line) for line in (root / "timing.jsonl").read_text().splitlines()]
            self.assertEqual([row["train_time_s"] for row in rows], durations)
            self.assertEqual([row["round"] for row in rows], [1, 2, 3])
            self.assertTrue(all(row["epochs"] == 2 and row["batch_size"] == 4 and row["seed"] == 42 for row in rows))
            self.assertTrue(all(row["num_examples"] == 10 for row in rows))
            self.assertEqual(json.loads((root / "ready.json").read_text())["stage_id"], "test")
            dependencies["tensorflow"].keras.utils.set_random_seed.assert_called_once_with(42)
            dependencies["flwr"].client.start_numpy_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
