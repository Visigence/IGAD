"""
Concrete exponential families with analytical log-partition functions.
"""

import numpy as np
from scipy.special import gammaln, digamma, polygamma


class GammaFamily:
    """
    Gamma(alpha, beta) in natural parameters theta = (alpha-1, -beta).

    A(theta_1, theta_2) = log Gamma(theta_1+1) - (theta_1+1) log(-theta_2)

    Domain: theta_1 > -1, theta_2 < 0.
    """

    name = "Gamma"

    @staticmethod
    def log_partition(theta: np.ndarray) -> float:
        t1, t2 = theta[0], theta[1]
        alpha = t1 + 1.0
        lam = -t2
        return float(gammaln(alpha) - alpha * np.log(lam))

    @staticmethod
    def to_natural(alpha: float, beta: float) -> np.ndarray:
        return np.array([alpha - 1.0, -beta])

    @staticmethod
    def from_natural(theta: np.ndarray):
        return theta[0] + 1.0, -theta[1]

    @staticmethod
    def fisher_metric_analytical(theta: np.ndarray) -> np.ndarray:
        alpha = theta[0] + 1.0
        lam = -theta[1]
        g = np.array([
            [polygamma(1, alpha), 1.0 / lam],
            [1.0 / lam,          alpha / (lam**2)],
        ])
        return g

    @staticmethod
    def third_cumulant_analytical(theta: np.ndarray) -> np.ndarray:
        alpha = theta[0] + 1.0
        lam = -theta[1]
        T = np.zeros((2, 2, 2))

        # T_{000} = psi''(alpha)
        T[0, 0, 0] = polygamma(2, alpha)
        # T_{001} = T_{010} = T_{100} = d/d(theta_2) g_{00}
        #   g_{00} = psi'(alpha), no dependence on theta_2 => 0
        # (already zero)

        # T_{011} = T_{101} = T_{110} = d/d(theta_1) g_{01}
        #   g_{01} = 1/lam = -1/theta_2, so d/d(theta_1) = 0
        #   BUT g_{01} = 1/lam, and d/d(theta_2)(g_{00}) = 0
        #   Actually T_{ijk} = d^3 A / d theta_i d theta_j d theta_k
        #   T_{011} = d^3 A / d theta_0 d theta_1 d theta_1
        #   A = gammaln(t1+1) - (t1+1)*log(-t2)
        #   dA/dt2 = -(t1+1)/t2 = (t1+1)/lam
        #   d^2A/dt1 dt2 = 1/(-t2) = 1/lam
        #   d^3A/dt1 dt2 dt2 = 0  (no t2 dependence in 1/lam... wait)
        #   d/dt2 (1/lam) = d/dt2 (-1/t2) = 1/t2^2 = 1/lam^2
        #   So T_{011} = +1/lam^2
        T[0, 1, 1] = T[1, 0, 1] = T[1, 1, 0] = 1.0 / (lam**2)

        # T_{111} = d^3 A / d theta_2^3
        #   dA/dt2 = (t1+1)/(-t2) = alpha/lam
        #   d^2A/dt2^2 = alpha/t2^2 = alpha/lam^2
        #   d^3A/dt2^3 = -2*alpha/t2^3 = -2*alpha/(-lam)^3 = 2*alpha/lam^3
        T[1, 1, 1] = 2.0 * alpha / (lam**3)

        return T

    @staticmethod
    def mle(data: np.ndarray) -> np.ndarray:
        """MLE for Gamma from positive data via Newton iteration."""
        x = np.asarray(data).ravel()
        x = x[x > 0]
        mean_x = np.mean(x)
        mean_log_x = np.mean(np.log(x))
        s = np.log(mean_x) - mean_log_x

        alpha = 0.5 / s if s > 0 else 1.0
        for _ in range(50):
            f = np.log(alpha) - digamma(alpha) - s
            fp = 1.0 / alpha - polygamma(1, alpha)
            step = f / fp
            alpha = max(alpha - step, 1e-8)
            if abs(step) < 1e-12:
                break

        beta = alpha / mean_x
        return GammaFamily.to_natural(alpha, beta)


class PoissonFamily:
    """
    Poisson(lam) in natural parameter theta = log(lam).
    A(theta) = exp(theta).
    1D manifold => R = 0 identically. Included for validation only.
    """

    name = "Poisson"

    @staticmethod
    def log_partition(theta: np.ndarray) -> float:
        return float(np.exp(theta[0]))

    @staticmethod
    def to_natural(lam: float) -> np.ndarray:
        return np.array([np.log(lam)])

    @staticmethod
    def mle(data: np.ndarray) -> np.ndarray:
        return PoissonFamily.to_natural(np.mean(data))
