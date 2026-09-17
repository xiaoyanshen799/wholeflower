"""Regression checks for centralized server evaluation label handling."""

import unittest

import numpy as np
import tensorflow as tf

from fedavgm.server import get_evaluate_fn


class ServerEvaluationTests(unittest.TestCase):
    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_column_vector_class_ids_are_not_treated_as_multilabel(self):
        inputs = tf.keras.Input(shape=(2,))
        outputs = tf.keras.layers.Dense(3)(inputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer="sgd",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        x_test = np.zeros((4, 2), dtype=np.float32)
        y_test = np.array([[0], [1], [2], [1]], dtype=np.int32)
        evaluate = get_evaluate_fn(model, x_test, y_test, 1, 3)

        loss, metrics = evaluate(0, model.get_weights(), {})

        self.assertTrue(np.isfinite(loss))
        self.assertIn("accuracy", metrics)


if __name__ == "__main__":
    unittest.main()
