"""
IGAD: Information-Geometric Anomaly Detection.

Anomaly score = R(theta_ref) - R(theta_hat(z))

where R is the scalar curvature of the Fisher-Rao manifold.
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Optional

from .curvature import scalar_curvature


class IGADDetector:
    """
    Information-Geometric Anomaly Detector.

    Parameters
    ----------
    family : object
        Exponential family with .log_partition(theta) and .mle(data).
    k_neighbors : int
        Number of neighbors for local parameter estimation.
    """

    def __init__(self, family, k_neighbors: int = 30):
        self.family = family
        self.k = k_neighbors
        self.theta_ref_ = None
        self.R_ref_ = None
        self.tree_ = None
        self.X_train_ = None

    def fit(self, X: np.ndarray) -> "IGADDetector":
        """Fit reference distribution from training data."""
        X = np.asarray(X, dtype=np.float64)
        self.X_train_ = X
        self.tree_ = KDTree(X)

        self.theta_ref_ = self.family.mle(X)
        self.R_ref_ = scalar_curvature(
            self.family.log_partition, self.theta_ref_
        )
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute IGAD anomaly scores.
        Higher = more anomalous.
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        scores = np.zeros(n)

        for i in range(n):
            dists, idxs = self.tree_.query(X[i], k=self.k)
            neighbors = self.X_train_[idxs.ravel()]

            try:
                theta_local = self.family.mle(neighbors)
                R_local = scalar_curvature(
                    self.family.log_partition, theta_local
                )
                scores[i] = self.R_ref_ - R_local
            except (np.linalg.LinAlgError, ValueError):
                scores[i] = np.inf

        return scores

    def predict(self, X: np.ndarray, contamination: float = 0.05) -> np.ndarray:
        """Binary anomaly prediction. 1 = anomaly, 0 = normal."""
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - contamination))
        return (scores >= threshold).astype(int)
