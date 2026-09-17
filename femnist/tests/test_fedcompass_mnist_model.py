"""Checks for the MNIST model and local optimizer behavior from FedCompass."""

import unittest

import numpy as np
import tensorflow as tf

from client import FlowerClient
from fedavgm.models import fedcompass_mnist_cnn


class FedCompassMnistModelTests(unittest.TestCase):
    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_architecture_and_optimizer_match_paper(self):
        model = fedcompass_mnist_cnn((28, 28, 1), 10, 0.003)

        self.assertEqual(model.count_params(), 582_026)
        self.assertEqual(model.output_shape, (None, 10))
        self.assertEqual(
            [layer.__class__.__name__ for layer in model.layers],
            [
                "InputLayer",
                "Conv2D",
                "ReLU",
                "MaxPooling2D",
                "Conv2D",
                "ReLU",
                "MaxPooling2D",
                "Flatten",
                "Dense",
                "ReLU",
                "Dense",
            ],
        )
        self.assertIsInstance(model.optimizer, tf.keras.optimizers.Adam)
        self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), 0.003)
        self.assertTrue(model.loss.get_config()["from_logits"])

    def test_client_resets_adam_for_each_local_task(self):
        rng = np.random.default_rng(42)
        x = rng.random((4, 28, 28, 1), dtype=np.float32)
        y = np.array([0, 1, 2, 3], dtype=np.int64)
        client = FlowerClient(
            x,
            y,
            x[:2],
            y[:2],
            {
                "_target_": "fedavgm.models.fedcompass_mnist_cnn",
                "input_shape": [28, 28, 1],
                "num_classes": 10,
                "learning_rate": 0.003,
            },
            10,
            cid="0",
            enable_compression=False,
            local_steps_override=1,
            step_seed=42,
        )
        config = {
            "training_mode": "steps",
            "local_steps": 1,
            "local_epochs": 1,
            "batch_size": 2,
            "step_seed": 42,
            "server_round": 1,
        }

        parameters, _, _ = client.fit(client.model.get_weights(), config)
        first_optimizer = client.model.optimizer
        self.assertEqual(int(first_optimizer.iterations.numpy()), 1)
        config["server_round"] = 2
        client.fit(parameters, config)

        self.assertIsNot(client.model.optimizer, first_optimizer)
        self.assertEqual(int(client.model.optimizer.iterations.numpy()), 1)


if __name__ == "__main__":
    unittest.main()
