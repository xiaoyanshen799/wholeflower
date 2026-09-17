"""Tiny real TensorFlow fits, without datasets, network services or CPU quotas."""

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from client import FlowerClient
from fixed_step_training import fit_fixed_steps, make_step_dataset


class FixedStepTrainingTests(unittest.TestCase):
    def model(self):
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(2,)), tf.keras.layers.Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss="mse")
        self.addCleanup(tf.keras.backend.clear_session)
        return model

    def test_exact_updates_for_small_and_large_partitions(self):
        for size in (3, 21):
            model = self.model()
            x = np.ones((size, 2), dtype=np.float32)
            y = np.zeros((size, 1), dtype=np.float32)
            for server_round in (1, 2):
                with self.subTest(size=size, server_round=server_round):
                    before = int(model.optimizer.iterations.numpy())
                    duration, metrics = fit_fixed_steps(model, x, y, 5, 8, 42, "0", server_round)
                    self.assertEqual(int(model.optimizer.iterations.numpy()) - before, 5)
                    self.assertEqual(metrics["local_steps_used"], 5)
                    self.assertEqual(metrics["processed_examples"], 40)
                    self.assertEqual(metrics["steps_per_second"], 5 / duration)

    def test_full_batches_reproducible_per_round_and_client(self):
        x = np.arange(19, dtype=np.float32)
        def sample(cid, server_round):
            dataset, seed = make_step_dataset(x, x, 8, 42, cid, server_round)
            values = [batch.numpy() for batch, _ in dataset.take(5)]
            self.assertTrue(all(len(batch) == 8 for batch in values))
            return np.concatenate(values), seed
        first, seed = sample("0", 1)
        repeat, again = sample("0", 1)
        np.testing.assert_array_equal(first, repeat)
        self.assertEqual(seed, again)
        for cid, server_round in (("0", 2), ("1", 1)):
            changed, new_seed = sample(cid, server_round)
            self.assertNotEqual(seed, new_seed)
            self.assertFalse(np.array_equal(first, changed))

    def test_empty_data_and_incomplete_optimizer_updates_fail(self):
        model = self.model()
        with self.assertRaises(ValueError):
            fit_fixed_steps(model, np.empty((0, 2)), np.empty((0, 1)), 5, 8, 42, "0", 1)
        with patch.object(model, "fit"), self.assertRaisesRegex(ValueError, "actually executed 0"):
            fit_fixed_steps(model, np.ones((3, 2)), np.zeros((3, 1)), 5, 8, 42, "0", 1)

    def test_flower_client_steps_override_epochs_and_keep_partition_weight(self):
        client = FlowerClient.__new__(FlowerClient)
        client.model = self.model()
        client._use_dataset = False
        client.x_train = np.ones((3, 2), dtype=np.float32)
        client.y_train = np.zeros((3, 1), dtype=np.float32)
        client._local_epochs_override = 100
        client._batch_size_override = 8
        client._local_steps_override = 5
        client._step_seed = 42
        client.enable_compression = False
        client._quantizer = None
        client.cid = "0"
        client._measurement_session = None
        client._measurement_round = 0
        weights, num_examples, metrics = client.fit(client.model.get_weights(),
                                                     {"local_steps": 5, "local_epochs": 999,
                                                      "batch_size": 8, "step_seed": 42, "server_round": 1})
        self.assertEqual(num_examples, 3)
        self.assertEqual(int(client.model.optimizer.iterations.numpy()), 5)
        self.assertEqual(metrics["processed_examples"], 40)
        self.assertEqual(metrics["local_steps_used"], 5)
        self.assertNotIn("local_epochs_used", metrics)
        self.assertEqual(len(weights), 2)
        with self.assertRaisesRegex(ValueError, "local_steps mismatch"):
            client.fit(weights, {"local_steps": 6, "batch_size": 8})
        client._use_dataset = True
        with self.assertRaisesRegex(ValueError, "array-backed"):
            client.fit(weights, {"local_steps": 5, "batch_size": 8})


if __name__ == "__main__":
    unittest.main()
