import numpy as np
import unittest
from igad.detector import IGADDetector
from igad.families import GammaFamily


class TestIGADDetector(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        self.X_ref = rng.gamma(shape=8.0, scale=0.5, size=300)
        self.detector = IGADDetector(family=GammaFamily)
        self.detector.fit(self.X_ref)

    def test_score_batch_returns_nonnegative_float(self):
        rng = np.random.default_rng(1)
        X = rng.gamma(shape=8.0, scale=0.5, size=200)
        score = self.detector.score_batch(X)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_score_batch_raises_before_fit(self):
        detector = IGADDetector(family=GammaFamily)
        rng = np.random.default_rng(2)
        X = rng.gamma(shape=8.0, scale=0.5, size=100)
        with self.assertRaises(RuntimeError):
            detector.score_batch(X)

    def test_predict_returns_int_given_threshold(self):
        rng = np.random.default_rng(3)
        X = rng.gamma(shape=8.0, scale=0.5, size=200)
        result = self.detector.predict(X, threshold=0.5)
        self.assertIsInstance(result, int)
        self.assertIn(result, (0, 1))

    def test_score_is_zero_when_batch_matches_reference(self):
        # Fit and score on the SAME array so MLE is identical in both calls.
        # |R_ref - R_batch| must be 0 up to floating-point rounding.
        rng = np.random.default_rng(42)
        X = rng.gamma(shape=8.0, scale=0.5, size=500)
        detector = IGADDetector(family=GammaFamily)
        detector.fit(X)
        score = detector.score_batch(X)
        self.assertLess(score, 1e-6)

    def test_score_is_positive_when_distributions_differ(self):
        rng = np.random.default_rng(5)
        X_anom = rng.lognormal(mean=1.327, sigma=0.343, size=300)
        score = self.detector.score_batch(X_anom)
        self.assertGreater(score, 0.0)

    def test_predict_threshold_zero_always_anomalous(self):
        """With threshold=0 and any non-zero score, predict should return 1."""
        rng = np.random.default_rng(5)
        X_anom = rng.lognormal(mean=1.327, sigma=0.343, size=300)
        result = self.detector.predict(X_anom, threshold=0.0)
        self.assertEqual(result, 1)

    def test_predict_threshold_boundary(self):
        """Score exactly at threshold should return 1 (>= semantics)."""
        rng = np.random.default_rng(5)
        X_anom = rng.lognormal(mean=1.327, sigma=0.343, size=300)
        score = self.detector.score_batch(X_anom)
        # At exactly the score value, predict should return 1
        self.assertEqual(self.detector.predict(X_anom, threshold=score), 1)
        # Just above the score, predict should return 0
        self.assertEqual(self.detector.predict(X_anom, threshold=score + 1e-10), 0)


if __name__ == "__main__":
    unittest.main()
