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
        T[0, 0, 0] = polygamma(2, alpha)
        T[0, 1, 1] = T[1, 0, 1] = T[1, 1, 0] = 1.0 / (lam**2)
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


def _inv_digamma(y: float) -> float:
    """Numerical inverse of digamma via Newton's method (Minka 2000).

    Newton's method converges quadratically from the Minka initialisation,
    typically reaching machine precision in 4-6 iterations.  The loop is
    capped at 50 iterations for safety, but exits early once the step size
    is below 1e-12.
    """
    if y >= -2.22:
        x = np.exp(y) + 0.5
    else:
        x = -1.0 / (y + digamma(1.0))
    for _ in range(50):
        delta = (digamma(x) - y) / polygamma(1, x)
        x -= delta
        if x <= 0:
            x = 1e-8
        if abs(delta) < 1e-12:
            break
    return x


class DirichletFamily:
    """
    Dirichlet(alpha_1, ..., alpha_k) in natural parameters theta_i = alpha_i - 1.

    A(theta) = sum_i gammaln(theta_i + 1) - gammaln(sum_i (theta_i + 1))
             = sum_i gammaln(alpha_i) - gammaln(alpha_0)

    Domain: theta_i > -1  for all i  (i.e., alpha_i > 0).

    This is a (k-1)-dimensional exponential family embedded in R^k.
    The manifold has non-constant scalar curvature for k >= 3.

    Key structural property: for k >= 3, mean + marginal variances do NOT
    determine alpha uniquely. Pure concentration-profile shifts are detectable
    via curvature even when lower-order moments match.

    References:
        - Minka, T. (2000). Estimating a Dirichlet distribution. Technical report.
        - Amari & Nagaoka (2000). Methods of Information Geometry.
    """

    name = "Dirichlet"

    @staticmethod
    def log_partition(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=np.float64)
        alpha = theta + 1.0
        return float(np.sum(gammaln(alpha)) - gammaln(np.sum(alpha)))

    @staticmethod
    def to_natural(alpha: np.ndarray) -> np.ndarray:
        return np.asarray(alpha, dtype=np.float64) - 1.0

    @staticmethod
    def from_natural(theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=np.float64) + 1.0

    @staticmethod
    def fisher_metric_analytical(theta: np.ndarray) -> np.ndarray:
        alpha = np.asarray(theta, dtype=np.float64) + 1.0
        alpha0 = alpha.sum()
        k = len(alpha)
        tri_alpha = polygamma(1, alpha)   # shape (k,)
        tri_alpha0 = polygamma(1, alpha0) # scalar
        g = -tri_alpha0 * np.ones((k, k))
        np.fill_diagonal(g, tri_alpha - tri_alpha0)
        return g

    @staticmethod
    def third_cumulant_analytical(theta: np.ndarray) -> np.ndarray:
        """
        Analytical third cumulant tensor for Dirichlet(alpha).

        A(theta) = sum_i gammaln(alpha_i) - gammaln(alpha_0),
        so d^3 A / d theta_i d theta_j d theta_k equals:
          psi_2(alpha_i) - psi_2(alpha_0)   if i == j == k
          -psi_2(alpha_0)                   otherwise

        where psi_2 = polygamma(2, .) is the tetragamma function.
        """
        alpha = np.asarray(theta, dtype=np.float64) + 1.0
        alpha0 = alpha.sum()
        k = len(alpha)
        psi2_alpha  = polygamma(2, alpha)    # shape (k,)
        psi2_alpha0 = float(polygamma(2, alpha0))
        # All entries start at -psi2_alpha0 (the off-diagonal value)
        T = np.full((k, k, k), -psi2_alpha0)
        # Diagonal correction: T[i,i,i] = psi2_alpha[i] - psi2_alpha0
        for i in range(k):
            T[i, i, i] = float(psi2_alpha[i]) - psi2_alpha0
        return T

    @staticmethod
    def mle(data: np.ndarray, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
        """
        MLE for Dirichlet via fixed-point iteration (Minka 2000).

        Parameters
        ----------
        data : array of shape (n, k), rows are simplex observations (sum to 1).
        max_iter : maximum fixed-point iterations.
        tol : convergence tolerance on alpha change.

        Returns
        -------
        theta : natural parameters (alpha - 1), shape (k,).

        Raises
        ------
        ConvergenceError
            If the fitted alpha does not reproduce E[log x_k] within 1e-4.
        """
        from igad.exceptions import ConvergenceError

        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        n, k = data.shape

        data = np.clip(data, 1e-15, 1.0)
        mean_log_x = np.mean(np.log(data), axis=0)   # shape (k,) — sufficient stats

        # Moment-matching initialization
        mean_x = np.mean(data, axis=0)
        var_x  = np.var(data, axis=0)
        eps = 1e-10
        denom = var_x[0] + eps
        alpha0_mom = mean_x[0] * (1.0 - mean_x[0]) / denom - 1.0
        alpha0_mom = max(alpha0_mom, 1.0)
        alpha = mean_x * alpha0_mom
        alpha = np.clip(alpha, 1e-3, None)

        # Fixed-point iteration (Minka 2000)
        for iteration in range(max_iter):
            alpha_old = alpha.copy()
            alpha0 = alpha.sum()
            psi_alpha0 = digamma(alpha0)
            for i in range(k):
                target = psi_alpha0 + mean_log_x[i]
                alpha[i] = _inv_digamma(target)
            alpha = np.clip(alpha, 1e-8, None)
            if np.max(np.abs(alpha - alpha_old)) < tol:
                break

        # ── Constraint 1: MLE Convergence Gate ──────────────────────────────
        # Verify the fitted alpha reproduces the sufficient statistics E[log x_k].
        # If not, raise ConvergenceError — a silent bad fit corrupts all downstream
        # experiments.
        alpha0_hat = alpha.sum()
        expected_log = digamma(alpha) - digamma(alpha0_hat)
        residual = np.abs(expected_log - mean_log_x)
        if np.max(residual) > 1e-4:
            raise ConvergenceError(
                f"DirichletFamily MLE did not converge: "
                f"max sufficient-statistic residual = {np.max(residual):.2e} "
                f"(tolerance 1e-4). alpha_hat = {alpha}"
            )

        return DirichletFamily.to_natural(alpha)


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



class InverseGaussianFamily:
    """
    Inverse Gaussian (Wald) distribution in natural parameters.

    f(x; mu, lam) = sqrt(lam/(2*pi*x^3)) * exp(-lam*(x-mu)^2/(2*mu^2*x))

    Natural parameters:
        theta_1 = -lam / (2*mu^2)    < 0
        theta_2 = -lam / 2           < 0

    Sufficient statistics: T(x) = (x, 1/x)

    Log-partition function:
        A(theta) = -1/2 * log(-2*theta_2) - 2 * sqrt(theta_1 * theta_2)

    Derivation:
        Follows from the Laplace transform identity:
        integral_0^inf x^{-3/2} exp(-a*x - b/x) dx = sqrt(pi/b)*exp(-2*sqrt(a*b))
        for a,b > 0. Setting a=-theta_1, b=-theta_2 and absorbing (2*pi)^{-1/2}
        yields the expression above.

    MLE (closed form):
        mu_hat  = mean(x)
        lam_hat = 1 / mean(1/x_i - 1/mu_hat)

    References:
        Tweedie, M.C.K. (1957). Statistical properties of inverse Gaussian distributions.
        Folks & Chhikara (1978). The inverse Gaussian distribution and its applications.
        Chhikara & Folks (1989). The Inverse Gaussian Distribution. Marcel Dekker.
    """

    name = "InverseGaussian"

    @staticmethod
    def log_partition(theta: np.ndarray) -> float:
        t1, t2 = float(theta[0]), float(theta[1])
        if t1 >= 0 or t2 >= 0:
            return np.inf
        return -0.5 * np.log(-2.0 * t2) - 2.0 * np.sqrt(t1 * t2)

    @staticmethod
    def to_natural(mu: float, lam: float) -> np.ndarray:
        return np.array([-lam / (2.0 * mu ** 2), -lam / 2.0])

    @staticmethod
    def from_natural(theta: np.ndarray):
        t1, t2 = theta[0], theta[1]
        lam = -2.0 * t2
        mu  = np.sqrt(t2 / t1)
        return mu, lam

    @staticmethod
    def mle(data: np.ndarray) -> np.ndarray:
        from igad.exceptions import ConvergenceError

        x = np.asarray(data, dtype=np.float64).ravel()
        x = x[x > 0]
        if len(x) < 2:
            raise ConvergenceError("InverseGaussianFamily MLE: fewer than 2 "
                                   "positive observations.")

        mu_hat  = float(np.mean(x))
        inv_lam = float(np.mean(1.0 / x - 1.0 / mu_hat))
        if inv_lam <= 0:
            raise ConvergenceError(
                f"InverseGaussianFamily MLE: 1/lambda_hat = {inv_lam:.4e} <= 0. "
                f"Data may not follow an Inverse Gaussian distribution."
            )

        lam_hat = 1.0 / inv_lam

        # Convergence gate: verify E[1/X] = 1/mu + 1/lam (Folks & Chhikara 1978)
        expected_inv = 1.0 / mu_hat + 1.0 / lam_hat
        observed_inv = float(np.mean(1.0 / x))
        residual     = abs(expected_inv - observed_inv)

        if residual > 1e-4 * observed_inv:
            raise ConvergenceError(
                f"InverseGaussianFamily MLE convergence gate: "
                f"E[1/X] residual = {residual:.2e} "
                f"(expected={expected_inv:.6f}, observed={observed_inv:.6f}). "
                f"Fitted: mu={mu_hat:.4f}, lam={lam_hat:.4f}."
            )

        return InverseGaussianFamily.to_natural(mu_hat, lam_hat)

    @staticmethod
    def fisher_metric_analytical(theta: np.ndarray) -> np.ndarray:
        a = -float(theta[0])
        b = -float(theta[1])
        g11 = np.sqrt(b) / (2.0 * a ** 1.5)
        g12 = -1.0 / (2.0 * np.sqrt(a * b))
        g22 = 1.0 / (2.0 * b ** 2) + np.sqrt(a) / (2.0 * b ** 1.5)
        return np.array([[g11, g12],
                         [g12, g22]])

    @staticmethod
    def third_cumulant_analytical(theta: np.ndarray) -> np.ndarray:
        a = -float(theta[0])
        b = -float(theta[1])
        T = np.zeros((2, 2, 2))
        T[0, 0, 0] = 3.0 * np.sqrt(b) / (4.0 * a ** 2.5)
        T[0, 0, 1] = T[0, 1, 0] = T[1, 0, 0] = -1.0 / (4.0 * np.sqrt(b) * a ** 1.5)
        T[0, 1, 1] = T[1, 0, 1] = T[1, 1, 0] = -1.0 / (4.0 * np.sqrt(a) * b ** 1.5)
        T[1, 1, 1] = 1.0 / b ** 3 + 3.0 * np.sqrt(a) / (4.0 * b ** 2.5)
        return T