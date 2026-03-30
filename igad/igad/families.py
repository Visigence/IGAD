import numpy as np
from scipy.special import gammaln, digamma, polygamma

class GammaFamily:
    name = "Gamma"

    @staticmethod
    def log_partition(theta):
        t1, t2 = theta[0], theta[1]
        alpha = t1 + 1.0
        lam = -t2
        return float(gammaln(alpha) - alpha * np.log(lam))

    @staticmethod
    def to_natural(alpha, beta):
        return np.array([alpha - 1.0, -beta])

    @staticmethod
    def from_natural(theta):
        return theta[0] + 1.0, -theta[1]

    @staticmethod
    def fisher_metric_analytical(theta):
        alpha = theta[0] + 1.0
        lam = -theta[1]
        return np.array([
            [polygamma(1, alpha), 1.0/lam],
            [1.0/lam, alpha/(lam*lam)],
        ])

    @staticmethod
    def third_cumulant_analytical(theta):
        alpha = theta[0] + 1.0
        lam = -theta[1]
        T = np.zeros((2, 2, 2))
        T[0, 0, 0] = polygamma(2, alpha)
        T[0, 1, 1] = 1.0 / (lam * lam)
        T[1, 0, 1] = 1.0 / (lam * lam)
        T[1, 1, 0] = 1.0 / (lam * lam)
        T[1, 1, 1] = 2.0 * alpha / (lam * lam * lam)
        return T

    @staticmethod
    def mle(data):
        x = np.asarray(data).ravel()
        x = x[x > 0]
        mean_x = np.mean(x)
        mean_log_x = np.mean(np.log(x))
        s = np.log(mean_x) - mean_log_x
        alpha = 0.5 / s if s > 0 else 1.0
        for _ in range(50):
            f_val = np.log(alpha) - digamma(alpha) - s
            fp = 1.0 / alpha - polygamma(1, alpha)
            step = f_val / fp
            alpha = max(alpha - step, 1e-8)
            if abs(step) < 1e-12:
                break
        beta = alpha / mean_x
        return GammaFamily.to_natural(alpha, beta)

class PoissonFamily:
    name = "Poisson"

    @staticmethod
    def log_partition(theta):
        return float(np.exp(theta[0]))

    @staticmethod
    def to_natural(lam):
        return np.array([np.log(lam)])

    @staticmethod
    def mle(data):
        return PoissonFamily.to_natural(np.mean(data))
