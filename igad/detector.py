"""
IGAD: Information-Geometric Anomaly Detection.

Anomaly score = |R(theta_ref) - R(theta_batch)|

where R is the scalar curvature of the Fisher-Rao manifold,
theta_ref is the MLE fitted to reference data, and
theta_batch is the MLE fitted to the incoming batch.

Reference: Damari, O. (2026). IGAD: Information-Geometric Anomaly Detection
           via scalar curvature of Fisher-Rao manifolds.
"""

import numpy as np

from .curvature import scalar_curvature


class IGADDetector:
    """
    Information-Geometric Anomaly Detector.

    The anomaly score for a batch z is:

        IGAD(z) = |R(theta_ref) - R(theta_batch)|

    where:
        theta_ref   = MLE(reference data)
        theta_batch = MLE(z)
        R(.)        = scalar curvature of the Fisher-Rao manifold

    Parameters
    ----------
    family : object
        Exponential family class with .log_partition(theta) and .mle(data).
    """

    def __init__(self, family):
        self.family = family
        self.theta_ref_ = None
        self.R_ref_ = None

    def fit(self, X: np.ndarray) -> "IGADDetector":
        """
        Fit reference distribution from training data.

        Computes and caches theta_ref = MLE(X) and R_ref = R(theta_ref).

        Parameters
        ----------
        X : array-like of shape (n,) or (n, k)
            Reference observations.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        self.theta_ref_ = self.family.mle(X)
        self.R_ref_ = scalar_curvature(
            self.family.log_partition, self.theta_ref_
        )
        return self

    def score_batch(self, X: np.ndarray) -> float:
        """
        Compute the IGAD anomaly score for a batch X.

        Score = |R(theta_ref) - R(theta_batch)|

        Higher scores indicate greater geometric deviation from the reference.

        Parameters
        ----------
        X : array-like of shape (n,) or (n, k)
            Batch of observations to score.

        Returns
        -------
        score : float
            Non-negative anomaly score. Zero means identical curvature to reference.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if self.theta_ref_ is None:
            raise RuntimeError("IGADDetector.fit() must be called before score_batch().")
        X = np.asarray(X, dtype=np.float64)
        theta_batch = self.family.mle(X)
        R_batch = scalar_curvature(self.family.log_partition, theta_batch)
        return float(abs(self.R_ref_ - R_batch))

    def predict(self, X: np.ndarray, threshold: float) -> int:
        """
        Binary anomaly prediction for a batch.

        Parameters
        ----------
        X : array-like of shape (n,) or (n, k)
            Batch of observations to classify.
        threshold : float
            Score threshold above which the batch is declared anomalous.

        Returns
        -------
        label : int
            1 if anomalous, 0 if normal.
        """
        return int(self.score_batch(X) >= threshold)
