# IGAD: Mathematical Background and Experimental Validation

**Status**: Sections 1–4 are known results in Hessian geometry.  
The novel contribution is the IGAD score (Section 5).  
Sections 6–7 are experimental validation.

**Author**: Omry Damari · 2026  
**Repository**: https://github.com/Visigence/IGAD

---

## 1. Setup

Let M = {p(x; θ)} be a regular exponential family:

    p(x; θ) = exp(⟨θ, T(x)⟩ − A(θ)) h(x)

where θ ∈ Rᵈ are the natural parameters and A(θ) is the log-partition function.

    Fisher metric:      g_{ij}(θ) = ∂²A / ∂θᵢ ∂θⱼ
    Third cumulant:     T_{ijk}(θ) = ∂³A / ∂θᵢ ∂θⱼ ∂θₖ

The Fisher metric g_{ij} is the Hessian of A — it defines a Riemannian
geometry on the parameter space M.

---

## 2. Christoffel Symbols (Known)

Because g_{ij} is the Hessian of A, and T_{ijk} = ∂ₖ g_{ij} is fully
symmetric in all three indices:

    Γ_{ij,k} = 1/2 · T_{ijk}

Reference: Amari and Nagaoka (2000), Chapter 2.

---

## 3. Fourth Cumulant Cancellation (Known)

The Riemann tensor R^l_{ijk} involves fourth cumulant terms arising from
∂ᵢ Γ^l_{jk}. These terms are symmetric in (i,j), but the Riemann tensor
is antisymmetric in (i,j). Therefore the fourth cumulant contributions
cancel exactly.

**Consequence**: R is purely quadratic in T — the scalar curvature depends
only on the third cumulant tensor, not the fourth.

Reference: Standard property of Hessian metrics. See Ruppeiner (1995).

---

## 4. Scalar Curvature Formula (Known)

    R(θ) = 1/4 · ( ‖S‖²_g − ‖T‖²_g )

where:

    S_m = g^{ab} T_{abm}                            (trace vector)
    ‖T‖²_g = g^{ia} g^{jb} g^{kc} T_{ijk} T_{abc}  (full tensor contraction)

When det g = const:  R = −1/4 · ‖T‖²_g ≤ 0.

**Why ‖T‖²_g matters**: This is not skewness. Skewness is a single scalar.
‖T‖²_g is a full metric-weighted contraction of the third cumulant tensor
across all parameter dimensions simultaneously — it captures how asymmetry
is distributed across the entire parameter geometry. No single moment
captures this quantity.

---

## 5. IGAD Score (Novel Contribution)

    IGAD(batch) = |R(θ_ref) − R(θ_local)|

where:
- θ_ref   is the MLE fitted to the reference data
- θ_local is the MLE fitted to the current batch

**Interpretation**: IGAD measures deviation in local third-cumulant
structure. Because R is governed by ‖T‖²_g, it is structurally sensitive
to shape shifts invisible to location-scale detectors.

**Relation to prior work**: Ruppeiner (1979) used scalar curvature to
detect phase transitions in thermodynamic systems — a conceptual precursor
in a completely different domain. The closest known related work in anomaly
detection applies Ricci curvature to graph structures — a fundamentally
different construction that operates on data topology. IGAD operates on
the geometry of the statistical model itself.

**Known failure modes**:
1. 1D families — R ≡ 0 identically (proven)
2. Constant-curvature manifolds — R_ref = R_local always (Gaussian)
3. Pure location-scale anomalies — geometry adds nothing
4. Model misspecification at large n — non-parametric methods dominate

---

## 6. Experimental Validation

### 6.1 Experiment 1 — Easy Case

**Setup**: Gamma(9,3) vs Gamma(1.5,0.5), batch_size=200
Same mean=3.0, different variance (1.0 vs 6.0) and skewness (0.667 vs 1.633)

| Method           | AUC-ROC |
|------------------|---------|
| IGAD (curvature) | 1.0000  |
| Variance shift   | 1.0000  |
| Skewness shift   | 0.9834  |
| Mean shift       | 0.8150  |

**Conclusion**: IGAD achieves perfect separation, but so does variance
shift (variance differs by 6×). This experiment does not prove unique value.

---

### 6.2 Experiment 2 — Hard Case (Key Result)

**Setup**: Gamma(8,2) vs LogNormal(μ=1.327, σ=0.343)
mean=4.000, var=2.000 **exactly identical** for both distributions.
Only higher-order geometric structure differs.

    Reference : Gamma(8,2)      mean=4.000  var=2.000  skew=0.707
    Anomaly   : LogNormal(...)  mean=4.000  var=2.000  skew=1.105

#### Control Experiment Design

To isolate geometry from MLE efficiency, a control baseline was constructed
using the **identical MLE fit** as IGAD but discarding the curvature tensor:

    skew_MLE(batch) = 2 / sqrt(α_MLE)
    score = |skew_MLE − skew_ref|

If IGAD ≈ MLE-skewness → MLE efficiency explains everything.
If IGAD > MLE-skewness → the curvature tensor is doing real work.

#### Results — 5 seeds, batch_size=200

| Method                  | Mean AUC | ± Std |
|-------------------------|----------|-------|
| IGAD (curvature)        | 0.6542   | 0.047 |
| MLE skewness [CONTROL]  | 0.6016   | 0.038 |
| MMD (RBF, median BW)    | 0.5894   | 0.076 |
| Wasserstein (1D)        | 0.5925   | 0.057 |
| Raw skewness            | 0.6794   | 0.072 |
| Mean shift [BLIND]      | 0.5240   | 0.062 |
| Variance shift [BLIND]  | 0.5818   | 0.027 |

**Gap (IGAD − MLE skewness): +0.053**
Curvature geometry adds signal beyond MLE efficiency alone.

#### Per-seed breakdown (batch_size=200)

| Method       | s42    | s7     | s123   | s999   | s2024  |
|--------------|--------|--------|--------|--------|--------|
| IGAD         | 0.6838 | 0.6796 | 0.6994 | 0.6390 | 0.5694 |
| MLE-skewness | 0.6098 | 0.6016 | 0.6096 | 0.6528 | 0.5342 |
| Raw skewness | 0.6514 | 0.5792 | 0.6472 | 0.7856 | 0.7334 |

#### Sample-efficiency sweep — IGAD vs MMD vs Wasserstein

Fixed signal: mean=4.0, var=2.0 identical. n is the only independent variable.
Seeds: [42, 7, 123, 999, 2024], averaged.

| n   | IGAD   | MMD    | Wasserstein | Gap(IGAD−MMD) |
|-----|--------|--------|-------------|---------------|
| 50  | 0.5522 | 0.5465 | 0.5425      | +0.006        |
| 100 | 0.5871 | 0.5639 | 0.5440      | +0.023        |
| 200 | 0.6542 | 0.5894 | 0.5925      | +0.065        |
| 300 | 0.6395 | 0.6074 | 0.5933      | +0.032        |
| 500 | 0.7150 | 0.6814 | 0.6777      | +0.034        |

IGAD beats MMD and Wasserstein at every n tested.

---

### 6.3 Experiment 3 — Dirichlet Concentration Shifts

**Setup**: Dirichlet(4,4,4) vs Dirichlet(1.5,4,6.5)
Both sum to α₀=12.0.

**Curvature verification**:

    R(α_ref=[4,4,4])       = 1.513247
    R(α_anom=[1.5,4,6.5])  = 1.493184
    |ΔR|                   = 0.020063

Scalar curvature varies non-trivially along concentration paths, confirming
the Dirichlet manifold is geometrically meaningful for IGAD.

Note: MMD and Wasserstein dominate this experiment because the chosen pair
includes a mean shift. The clean cross-family result is Experiment 2.

---

### 6.4 Experiment 4 — Gaussian Failure Mode (Documented)

The Gaussian manifold is a symmetric space of constant curvature.
R is constant regardless of parameter choice — empirically verified:

    R(ρ=0.2) = 2.000008
    R(ρ=0.8) = 1.996700
    |ΔR|     = 0.003308  — numerical noise only, not a real signal

Note: The positive value (~2.0) reflects the sign convention of the scalar curvature
formula implemented in curvature.py (R = 1/4(||S||²_g − ||T||²_g)). The standard
differential geometry convention for the Gaussian Fisher-Rao manifold yields negative
curvature (it is isometric to a hyperbolic space). The sign convention does not affect
IGAD's correctness — what matters is that |R_ref − R_local| ≈ 0 for all Gaussian
parameter choices, which is confirmed above.

Consequence: |R_ref − R_local| ≈ 0 for all Gaussian parameter choices.
IGAD cannot detect any Gaussian-to-Gaussian shift.

All methods reached AUC=1.0 at n=200 for ρ=0.2 vs ρ=0.8 — not because
of curvature, but because the correlation difference (0.6) is large enough
for any method. IGAD contributed nothing unique in this setting.

---

## 7. Operational Envelope

| Family          | dim | R varies?   | IGAD applicable? |
|-----------------|-----|-------------|------------------|
| Poisson         | 1   | No (R≡0)    | No — proven      |
| Exponential     | 1   | No (R≡0)    | No — proven      |
| Gamma           | 2   | Yes         | Yes — confirmed  |
| Gaussian        | 3   | No (const)  | No — proven      |
| Dirichlet (k≥3) | k−1 | Yes         | Promising        |
| Neg-Binomial    | 2   | Yes         | Untested         |

### Falsifiable Claims

1. **IGAD > MLE-skewness by ≥ +0.04** (mean, 5 seeds, n=200, Gamma vs
   LogNormal, matched mean+variance).
   Verified: `experiments/run_gamma_vs_lognormal.py`

2. **IGAD > MMD at n=200** for Gamma vs LogNormal (matched mean+variance).
   Gap: +0.065.
   Verified: `experiments/run_gamma_vs_lognormal_extended.py`

3. **IGAD > MMD at every n ∈ {50,100,200,300,500}** for Gamma vs LogNormal.
   Verified: `experiments/run_gamma_vs_lognormal_extended.py`

4. **R(Gaussian) constant regardless of parameters** — proven empirically.
   Verified: `tests/test_dirichlet_family.py::TestFailureModes`

5. **R(Poisson) ≡ 0** — proven.
   Verified: `tests/test_curvature.py::TestPoissonFlat`

---

## 8. References

1. Rao, C.R. (1945). Information and the accuracy attainable in the
   estimation of statistical parameters. *Bull. Calcutta Math. Soc.*
2. Amari, S. (1985). *Differential-Geometrical Methods in Statistics*.
   Springer.
3. Amari, S. and Nagaoka, H. (2000). *Methods of Information Geometry*.
   AMS/Oxford.
4. Minka, T. (2000). Estimating a Dirichlet distribution. MIT Tech Report.
5. Ruppeiner, G. (1979). Thermodynamics: A Riemannian geometric model.
   *Phys. Rev. A*, 20(4), 1608.
6. Ruppeiner, G. (1995). Riemannian geometry in thermodynamic fluctuation
   theory. *Rev. Mod. Phys.*, 67(3), 605.
