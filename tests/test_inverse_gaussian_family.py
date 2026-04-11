"""
tests/test_inverse_gaussian_family.py

Unit tests for InverseGaussianFamily — mirroring Dirichlet/Gamma coverage.
"""

import numpy as np
import pytest
from scipy.stats import invgauss

from igad.curvature import scalar_curvature, fisher_metric, third_cumulant_tensor
from igad.detector import IGADDetector
from igad.exceptions import ConvergenceError
from igad.families import InverseGaussianFamily


# ─────────────────────────────────────────────────────────────────────────────
# Class 1 — Natural parameter conversion
# ─────────────────────────────────────────────────────────────────────────────

class TestInverseGaussianParams:

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1.0), (2.0, 5.0), (0.5, 10.0), (3.0, 0.5),
    ])
    def test_natural_param_roundtrip(self, mu, lam):
        """from_natural(to_natural(mu, lam)) recovers (mu, lam)."""
        theta = InverseGaussianFamily.to_natural(mu, lam)
        mu_rec, lam_rec = InverseGaussianFamily.from_natural(theta)
        np.testing.assert_allclose(mu_rec, mu, rtol=1e-12)
        np.testing.assert_allclose(lam_rec, lam, rtol=1e-12)

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1.0), (2.0, 5.0), (0.5, 10.0),
    ])
    def test_natural_params_are_negative(self, mu, lam):
        """Both natural parameters must be strictly negative."""
        theta = InverseGaussianFamily.to_natural(mu, lam)
        assert theta[0] < 0, "theta_1 should be negative"
        assert theta[1] < 0, "theta_2 should be negative"


# ─────────────────────────────────────────────────────────────────────────────
# Class 2 — Log-partition function
# ─────────────────────────────────────────────────────────────────────────────

class TestInverseGaussianLogPartition:

    def test_log_partition_gradient_matches_expected_stats(self):
        """
        dA/dθ₁ = E[X] = mu, dA/dθ₂ = E[1/X] = 1/mu + 1/lam.
        Verified via central finite differences.
        """
        mu, lam = 2.0, 5.0
        theta = InverseGaussianFamily.to_natural(mu, lam)
        A = InverseGaussianFamily.log_partition

        eps = 1e-6
        grad = np.zeros(2)
        for i in range(2):
            t_fwd = theta.copy(); t_fwd[i] += eps
            t_bwd = theta.copy(); t_bwd[i] -= eps
            grad[i] = (A(t_fwd) - A(t_bwd)) / (2 * eps)

        # E[X] = mu
        np.testing.assert_allclose(grad[0], mu, rtol=1e-4)
        # E[1/X] = 1/mu + 1/lam
        np.testing.assert_allclose(grad[1], 1.0/mu + 1.0/lam, rtol=1e-4)

    def test_log_partition_is_convex(self):
        """Hessian (Fisher metric) must be positive definite."""
        for mu, lam in [(1.0, 1.0), (2.0, 5.0), (0.5, 10.0), (3.0, 0.5)]:
            theta = InverseGaussianFamily.to_natural(mu, lam)
            g = fisher_metric(InverseGaussianFamily.log_partition, theta)
            eigvals = np.linalg.eigvalsh(g)
            assert np.all(eigvals > 0), (
                "Fisher metric not PD for mu=%s, lam=%s, eigvals=%s"
                % (mu, lam, eigvals)
            )

    def test_log_partition_returns_inf_for_invalid_params(self):
        """A(theta) should return inf when theta_1 >= 0 or theta_2 >= 0."""
        assert InverseGaussianFamily.log_partition(np.array([0.0, -1.0])) == np.inf
        assert InverseGaussianFamily.log_partition(np.array([-1.0, 0.0])) == np.inf
        assert InverseGaussianFamily.log_partition(np.array([1.0, -1.0])) == np.inf


# ─────────────────────────────────────────────────────────────────────────────
# Class 3 — Fisher metric
# ─────────────────────────────────────────────────────────────────────────────

class TestInverseGaussianFisherMetric:

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1.0), (2.0, 5.0), (0.5, 10.0), (3.0, 0.5),
    ])
    def test_fisher_metric_matches_numerical(self, mu, lam):
        """Analytical Fisher metric matches numerical Hessian."""
        theta = InverseGaussianFamily.to_natural(mu, lam)
        g_ana = InverseGaussianFamily.fisher_metric_analytical(theta)
        g_num = fisher_metric(InverseGaussianFamily.log_partition, theta)
        np.testing.assert_allclose(g_ana, g_num, rtol=1e-3)

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1.0), (2.0, 5.0), (3.0, 0.5),
    ])
    def test_fisher_metric_is_symmetric(self, mu, lam):
        """g == gᵀ."""
        theta = InverseGaussianFamily.to_natural(mu, lam)
        g = InverseGaussianFamily.fisher_metric_analytical(theta)
        np.testing.assert_allclose(g, g.T, atol=1e-12)

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1.0), (2.0, 5.0), (0.5, 10.0), (3.0, 0.5),
    ])
    def test_fisher_metric_is_positive_definite(self, mu, lam):
        """All eigenvalues > 0."""
        theta = InverseGaussianFamily.to_natural(mu, lam)
        g = InverseGaussianFamily.fisher_metric_analytical(theta)
        eigvals = np.linalg.eigvalsh(g)
        assert np.all(eigvals > 0), (
            "Not PD for mu=%s, lam=%s" % (mu, lam)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class 4 — Third cumulant tensor
# ─────────────────────────────────────────────────────────────────────────────

class TestInverseGaussianThirdCumulant:

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1.0), (2.0, 5.0), (0.5, 10.0),
    ])
    def test_third_cumulant_matches_numerical(self, mu, lam):
        """Analytical T_{ijk} matches numerical finite-difference tensor."""
        theta = InverseGaussianFamily.to_natural(mu, lam)
        T_ana = InverseGaussianFamily.third_cumulant_analytical(theta)
        T_num = third_cumulant_tensor(InverseGaussianFamily.log_partition, theta)
        np.testing.assert_allclose(T_ana, T_num, rtol=5e-3, atol=1e-6)

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1.0), (2.0, 5.0),
    ])
    def test_third_cumulant_is_symmetric(self, mu, lam):
        """T_{ijk} is fully symmetric in all indices."""
        theta = InverseGaussianFamily.to_natural(mu, lam)
        T = InverseGaussianFamily.third_cumulant_analytical(theta)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    assert abs(T[i, j, k] - T[j, i, k]) < 1e-12
                    assert abs(T[i, j, k] - T[i, k, j]) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Class 5 — Scalar curvature
# ─────────────────────────────────────────────────────────────────────────────

class TestInverseGaussianCurvature:

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1.0), (2.0, 5.0), (0.5, 10.0), (3.0, 0.5),
    ])
    def test_curvature_is_finite(self, mu, lam):
        """R is finite for valid parameters."""
        theta = InverseGaussianFamily.to_natural(mu, lam)
        R = scalar_curvature(
            InverseGaussianFamily.log_partition, theta,
            family=InverseGaussianFamily,
        )
        assert np.isfinite(R), "Non-finite R at mu=%s, lam=%s" % (mu, lam)

    def test_curvature_is_constant(self):
        """
        R ≈ 1.0 for all IG parameters — the IG Fisher-Rao manifold has
        constant scalar curvature, analogous to the Gaussian failure mode.
        This means IGAD cannot detect IG-to-IG parameter shifts.
        """
        R_values = []
        for mu, lam in [(1.0, 1.0), (2.0, 5.0), (0.5, 10.0), (3.0, 0.5)]:
            theta = InverseGaussianFamily.to_natural(mu, lam)
            R = scalar_curvature(
                InverseGaussianFamily.log_partition, theta,
                family=InverseGaussianFamily,
            )
            R_values.append(R)
        # All curvatures should be approximately equal (constant curvature)
        for R in R_values:
            assert abs(R - R_values[0]) < 1e-6, (
                "IG curvature should be constant: %s" % R_values
            )

    def test_curvature_analytical_matches_numerical(self):
        """Curvature via analytical path matches numerical-only path."""
        theta = InverseGaussianFamily.to_natural(2.0, 5.0)
        R_ana = scalar_curvature(
            InverseGaussianFamily.log_partition, theta,
            family=InverseGaussianFamily,
        )
        R_num = scalar_curvature(
            InverseGaussianFamily.log_partition, theta,
        )
        np.testing.assert_allclose(R_ana, R_num, rtol=1e-2)


# ─────────────────────────────────────────────────────────────────────────────
# Class 6 — MLE
# ─────────────────────────────────────────────────────────────────────────────

class TestInverseGaussianMLE:

    def test_mle_recovers_parameters(self):
        """Fit n=5000 samples from InvGauss(mu=2, lam=5), recover within rtol=0.1."""
        rng = np.random.default_rng(42)
        mu_true, lam_true = 2.0, 5.0
        # scipy.stats.invgauss parameterisation: mu_scipy = mu/lam, scale = lam
        # => X ~ IG(mu, lam)
        data = invgauss.rvs(mu_true / lam_true, scale=lam_true, size=5000,
                            random_state=rng)
        theta_hat = InverseGaussianFamily.mle(data)
        mu_hat, lam_hat = InverseGaussianFamily.from_natural(theta_hat)
        np.testing.assert_allclose(mu_hat, mu_true, rtol=0.1)
        np.testing.assert_allclose(lam_hat, lam_true, rtol=0.1)

    def test_mle_raises_on_too_few_observations(self):
        """MLE must raise ConvergenceError with fewer than 2 positive observations."""
        with pytest.raises(ConvergenceError):
            InverseGaussianFamily.mle(np.array([3.0]))

    def test_mle_raises_when_inv_lam_nonpositive(self):
        """MLE raises ConvergenceError when 1/lambda estimator is non-positive."""
        # Construct data where mean(1/x - 1/mu_hat) <= 0
        # Constant data has 1/x_i = 1/mu_hat for all i, so inv_lam = 0
        data = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        with pytest.raises(ConvergenceError):
            InverseGaussianFamily.mle(data)

    def test_mle_convergence_gate_passes_on_good_data(self):
        """MLE does NOT raise ConvergenceError on genuine IG data."""
        rng = np.random.default_rng(7)
        mu_true, lam_true = 3.0, 10.0
        data = invgauss.rvs(mu_true / lam_true, scale=lam_true, size=2000,
                            random_state=rng)
        # Should not raise
        InverseGaussianFamily.mle(data)


# ─────────────────────────────────────────────────────────────────────────────
# Class 7 — IGAD detection
# ─────────────────────────────────────────────────────────────────────────────

class TestIGADInverseGaussianDetection:

    def test_igad_detector_with_inverse_gaussian(self):
        """
        IG has constant curvature (R ≈ 1.0), so IGAD scores are ~0 for all
        IG data regardless of parameters. This documents the IG failure mode
        (analogous to Gaussian) — IGAD cannot detect IG-to-IG shifts.
        """
        rng = np.random.default_rng(42)
        mu_ref, lam_ref = 2.0, 5.0

        data_ref = invgauss.rvs(mu_ref / lam_ref, scale=lam_ref, size=500,
                                random_state=rng)
        detector = IGADDetector(family=InverseGaussianFamily)
        detector.fit(data_ref)

        # Same-distribution batch: score should be near 0
        data_same = invgauss.rvs(mu_ref / lam_ref, scale=lam_ref, size=300,
                                 random_state=np.random.default_rng(1))
        score_same = detector.score_batch(data_same)
        assert score_same < 0.01, (
            "Score for same distribution should be near 0: %s" % score_same
        )

        # Shifted batch: IG has constant curvature, so score should also be ~0
        mu_anom, lam_anom = 4.0, 5.0
        data_anom = invgauss.rvs(mu_anom / lam_anom, scale=lam_anom, size=300,
                                 random_state=np.random.default_rng(2))
        score_anom = detector.score_batch(data_anom)
        assert score_anom < 0.01, (
            "IG constant curvature: shifted score should also be near 0: %s"
            % score_anom
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
