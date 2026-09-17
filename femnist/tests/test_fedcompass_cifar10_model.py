"""Checks for the CIFAR-10 model used in the FedCompass experiments."""

import unittest

import tensorflow as tf

from fedavgm.models import fedcompass_cifar10_resnet18


class FedCompassCifar10ModelTests(unittest.TestCase):
    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_architecture_and_optimizer_match_paper(self):
        model = fedcompass_cifar10_resnet18((32, 32, 3), 10, 0.1)

        trainable = sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights)
        self.assertEqual(trainable, 11_173_962)
        # Keras also exposes BatchNorm moving mean/variance as model state.
        self.assertEqual(model.count_params(), 11_183_562)
        self.assertEqual(model.output_shape, (None, 10))
        self.assertEqual(sum(isinstance(layer, tf.keras.layers.Add) for layer in model.layers), 8)
        self.assertEqual(
            sum(isinstance(layer, tf.keras.layers.BatchNormalization) for layer in model.layers),
            20,
        )
        stem = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D))
        self.assertEqual(stem.kernel_size, (3, 3))
        self.assertEqual(stem.strides, (1, 1))
        self.assertFalse(stem.use_bias)
        self.assertIsInstance(model.layers[-3], tf.keras.layers.AveragePooling2D)
        self.assertIsInstance(model.optimizer, tf.keras.optimizers.SGD)
        self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), 0.1)
        self.assertEqual(float(model.optimizer.momentum), 0.0)
        self.assertIsNone(model.optimizer.clipnorm)
        self.assertTrue(model.loss.get_config()["from_logits"])
        self.assertTrue(model._reset_optimizer_each_round)


if __name__ == "__main__":
    unittest.main()
