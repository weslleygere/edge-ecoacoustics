"""CPU-only checks; TensorFlow not required for these tests."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
from train import statistics, evaluate


class TrainUtilitiesTests(unittest.TestCase):
    def test_statistics_shape(self):
        x = np.arange(2*5*4, dtype=np.float32).reshape(2, 5, 4)
        z = statistics(x)
        self.assertEqual(z.shape, (2, 12))
        np.testing.assert_allclose(z[:, :4], x.mean(axis=1))

    def test_perfect_regression(self):
        y = np.array([[0.2, 0.1, 0.7], [0.5, 0.4, 0.6],
                      [0.9, 0.7, 0.3]], dtype=np.float32)
        scores = evaluate(y, y)
        self.assertAlmostEqual(scores['macro_mae'], 0.0)
        for target in ('bio', 'geo', 'antro'):
            self.assertAlmostEqual(scores[target]['r2'], 1.0)
            self.assertAlmostEqual(scores[target]['spearman'], 1.0)


if __name__ == '__main__':
    unittest.main()
