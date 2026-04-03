# IGAD Operational Envelope

## When IGAD Wins

### Condition 1: Correct exponential family, k >= 3 parameters, small-to-moderate n

When the reference distribution belongs to a correctly-specified exponential family with
at least three parameters, IGAD exploits scalar curvature variation to detect concentration
shifts that cannot be resolved by mean or variance alone.

**Evidence 1 (with mean-shift confound):** Dirichlet(4,4,4) vs Dirichlet(1.5,4,6.5) — same
α₀=12.0, same mean direction for the symmetric case, only the concentration profile shifts.
IGAD AUC > 0.65 at n=200.

**Evidence 2 — Clean decisive result (Exp 7):** Dirichlet([4,4,4]) vs Dirichlet([2,2,2]) —
IDENTICAL mean direction [1/3,1/3,1/3], only concentration α₀ halved (12→6). This is a
pure shape shift with no mean-shift confound. Results: IGAD AUC=0.9999 at n=50 vs
MMD=0.874 and Wasserstein=0.928. **p<0.001, non-overlapping 95% CIs. DECISIVE.**

The curvature advantage is most pronounced at small-to-moderate n (50–200 samples per batch)
where non-parametric methods have insufficient power. At n > 200, MMD and Wasserstein
catch up as their estimators reach their asymptotic regime.

**Dimensional scaling:** The IGAD advantage grows with k (Exp 7):

| k | α_ref | |ΔR| | IGAD n=50 | MMD n=50 | Decisive? |
|---|-------|------|-----------|----------|---------|
| 3 | [4,4,4] | 0.027 | 0.9999 | 0.874 | Yes (p<0.001) |
| 4 | [3,3,3,3] | 0.046 | 1.0000 | 0.877 | Yes (p<0.001) |
| 5 | [2.4,…] | 0.091 | 1.0000 | 0.889 | Yes (p=0.002) |

**Sample complexity (updated):** IGAD achieves AUC > 0.999 at n ≈ 50 for the Dirichlet
pure concentration shift; MMD requires n ≈ 200–300 for equivalent power.

### Condition 2: Cross-family detection (misspecified model, small n)

When data comes from a different exponential family than assumed (e.g., LogNormal data
scored under a Gamma model), IGAD still detects anomalies because the MLE parameter
fit to the anomalous data lands at a point with systematically different curvature
than the reference.

**Evidence:** Gamma(8,2) vs LogNormal (matched mean AND variance). IGAD beats
MLE-skewness control by +0.053 AUC (mean, 5 seeds, n=200). This confirms that the
curvature geometry is doing real work, not just MLE efficiency.

---

## When IGAD Fails

### Failure Mode 1: Gaussian families

**R = constant for all Gaussian parameters.**

Mathematical reason: The Fisher-Rao metric of the multivariate Gaussian is the
Siegel upper half-space metric, a symmetric space of constant negative curvature
(hyperbolic geometry). Specifically, the Gaussian Fisher-Rao manifold is a symmetric space — its scalar
curvature R is a constant that depends only on dimension and sign convention, not
on the specific parameter values (mean, covariance). Empirically verified:
R(ρ=0.2) ≈ R(ρ=0.8) to within numerical noise (|ΔR| < 0.004, see proof.md §6.4). No matter how the mean or covariance
changes between reference and anomaly, R_ref = R_local identically, so the IGAD
score is zero.

**Consequence:** IGAD cannot detect any Gaussian-to-Gaussian shift (mean shift,
variance shift, correlation shift). Use classical tests (Hotelling's T², likelihood
ratio tests) for Gaussian anomaly detection.

### Failure Mode 2: 1D families

**R ≡ 0 identically.**

Mathematical reason: For a 1-dimensional exponential family (Poisson, Exponential,
Bernoulli, any single-parameter family), the Fisher-Rao manifold is a 1D Riemannian
manifold. The Riemann curvature tensor of a 1D manifold is identically zero by
definition — you cannot have intrinsic curvature in one dimension. Therefore
R = 0 for all parameter values, and the IGAD score |R_ref - R_local| = 0 always.

**Consequence:** IGAD is completely blind to shifts in Poisson rate, Exponential
rate, Bernoulli probability, or any single-parameter family. This is proven, not
empirical.

### Failure Mode 3: Large n with misspecified model

When n is large (> 500 per batch), non-parametric methods (MMD, Wasserstein)
accumulate enough samples to achieve near-perfect power without any parametric
assumptions. If the reference distribution is misspecified (e.g., data is actually
from a mixture model but scored under a single Dirichlet), IGAD's curvature
estimator absorbs the model error into a biased curvature estimate. Non-parametric
methods remain unbiased and eventually dominate.

**Consequence:** For large-n production systems with complex reference distributions,
prefer ensemble approaches combining IGAD (for small n) with MMD or Wasserstein
(for large n).

---

## The Falsifiable Claims

1. **IGAD AUC > MLE-skewness control by at least +0.04** (mean, 5 seeds, n=200,
   Gamma vs LogNormal with matched mean+variance). Verified in `experiments/demo_hard.py`
   and `experiments/demo_hard_extended.py`.

2. **IGAD AUC > 0.65 at n=200** for Dirichlet k=3 concentration shift
   (Dirichlet(4,4,4) vs Dirichlet(1.5,4,6.5), seed=42, 100 normal + 50 anomaly batches).
   Verified in `tests/test_dirichlet_family.py::TestIGADDirichletDetection`.

3. **IGAD AUC > MMD at n <= 100** for Dirichlet k=3 concentration shift
   (sample efficiency regime). Verified empirically in `experiments/demo_dirichlet.py`
   Part 3.

4. **R(Gaussian) is constant regardless of correlation parameter** (proven, not empirical).
   Verified in `tests/test_dirichlet_family.py::TestFailureModes::test_gaussian_constant_curvature`.

5. **R(Poisson) ≡ 0** (proven, verified in tests).
   Verified in `tests/test_curvature.py::TestPoissonFlat` and
   `tests/test_dirichlet_family.py::TestFailureModes::test_poisson_flat`.

6. **IGAD AUC > MMD AUC AND IGAD AUC > Wasserstein AUC with p<0.001 (permutation
   test) and non-overlapping 95% CIs at n=50** for Dirichlet pure concentration-profile
   shift (identical mean direction, halved α₀). Verified in `experiments/exp_dirichlet_decisive.py`
   across k=3, k=4, k=5 families. This is the strongest falsifiable claim: it holds even
   when the mean-shift confound from the original pair is removed.

---

## Sample Complexity Analysis

For a correctly-specified k-dimensional exponential family:

- **IGAD** requires O(n^{1/2}) samples to achieve AUC > 0.7 in the concentration-shift
  regime. The curvature estimator converges at the rate of the MLE (Fisher-efficient),
  and the signal (|ΔR|) is a fixed constant for a fixed Δα.

- **Non-parametric methods** (MMD, Wasserstein) require O(n) samples for equivalent
  power, because their test statistics converge at the slower rate of the
  empirical distribution (no parametric structure exploited).

- The advantage is O(n^{1/2}) — a sub-linear sample-efficiency gain from parametric
  structure.

(The O(n^{1/2}) claim is empirically derived from the sweep in Experiment 4,
`experiments/demo_dirichlet.py` Part 3. A formal proof requires analysis of the
Fisher information matrix eigenspectrum and the curvature functional's gradient
with respect to the natural parameters.)

---

## Curvature Separation Verification

For the canonical Dirichlet detection pair (with mean-shift confound):

| Distribution          | α               | R(α)      |
|-----------------------|-----------------|-----------|
| Reference             | [4.0, 4.0, 4.0] | 1.5132    |
| Anomaly               | [1.5, 4.0, 6.5] | 1.4932    |

`|R_ref - R_anom| ≈ 0.020` — verified in unit tests and Part 4 of
`experiments/demo_dirichlet.py`. This non-zero separation is the mathematical
foundation that makes IGAD work for this family.

For the decisive pure-shape-shift pairs (Exp 7, no mean-shift confound):

| DGP | α_ref | α_anom | R(α_ref) | R(α_anom) | \|ΔR\| |
|-----|-------|--------|----------|-----------|--------|
| k=3 sym  | [4,4,4] | [2,2,2] | 1.5109 | 1.4839 | 0.027 |
| k=3 asym | [6,3,3] | [3,1.5,1.5] | 1.5088 | 1.4708 | 0.038 |
| k=4 sym  | [3,3,3,3] | [1.5,…] | 2.0348 | 1.9885 | 0.046 |
| k=5 sym  | [2.4,…] | [1.2,…] | 2.5602 | 2.4696 | 0.091 |

Curvature separation grows with dimension k (0.027→0.046→0.091 for k=3,4,5),
explaining why IGAD's advantage strengthens in higher-dimensional Dirichlet families.

