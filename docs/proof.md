# Mathematical Background: Scalar Curvature of Exponential Families

**Status**: Sections 1-4 are **known results** in Hessian geometry.
The novel contribution is the IGAD score (Section 5).

## 1. Setup

Let M = {p(x;theta)} be a regular exponential family:

    p(x;theta) = exp(<theta, T(x)> - A(theta)) h(x)

Fisher metric:     g_{ij}(theta) = d^2 A / d theta_i d theta_j
Third cumulant:    T_{ijk}(theta) = d^3 A / d theta_i d theta_j d theta_k

## 2. Christoffel Symbols (Known)

Because g_{ij} is the Hessian of A, and T_{ijk} = d_k g_{ij} is fully symmetric:

    Gamma_{ij,k} = 1/2 T_{ijk}

Reference: Amari and Nagaoka (2000), Chapter 2.

## 3. Fourth Cumulant Cancellation (Known)

The Riemann tensor R^l_{ijk} involves fourth cumulant terms from d_i Gamma^l_{jk}.
These are symmetric in (i,j), but R is antisymmetric in (i,j).
Therefore they cancel exactly. R is purely quadratic in T.

Reference: Standard property of Hessian metrics. See Ruppeiner (1995).

## 4. Scalar Curvature Formula (Known)

    R(theta) = 1/4 * ( ||S||^2_g - ||T||^2_g )

where S_m = g^{ab} T_{abm} = d_m log det g.

When det g = const: R = -1/4 ||T||^2_g <= 0.

## 5. IGAD Score (Novel Contribution)

    IGAD(batch) = |R(theta_ref) - R(theta_local)|

where theta_ref is the global MLE and theta_local is the batch MLE.

Interpretation: IGAD measures deviation in local third-cumulant structure,
making it sensitive to distributional shape changes invisible to
location-scale detectors.

Conceptual precursor: Ruppeiner (1979) used scalar curvature to detect
phase transitions in thermodynamic systems.

Failure modes:
1. 1D families (R = 0 identically)
2. Constant-curvature manifolds
3. Pure location-scale anomalies
4. Model misspecification at large sample sizes

## References

- Amari, S. (1985). Differential-Geometrical Methods in Statistics.
- Amari, S. and Nagaoka, H. (2000). Methods of Information Geometry.
- Ruppeiner, G. (1979). Thermodynamics: A Riemannian geometric model. Phys. Rev. A.
- Ruppeiner, G. (1995). Riemannian geometry in thermodynamic fluctuation theory. Rev. Mod. Phys.

---

## 6. Known Failure Mode: Constant-Curvature Families

### Gaussian Manifold (Empirically Confirmed)

The scalar curvature of the multivariate Gaussian manifold is constant
(the manifold is isometric to hyperbolic space). Therefore:

    |R_ref - R_local| ≈ 0  for all parameter choices

Verified experimentally (experiments/demo_gaussian2d.py):

    rho_ref=0.20, rho_anom=0.80  →  |R_diff| = 0.003308
    rho_ref=0.50, rho_anom=0.70  →  |R_diff| = 0.000645
    rho_ref=0.50, rho_anom=0.55  →  |R_diff| = 0.000049

All baselines (IGAD, MLE-correlation, raw correlation) reached AUC=1.0
at n=200 for rho=0.2 vs rho=0.8 — not because of curvature, but because
the correlation difference (0.6) is large enough for any method to detect.
IGAD contributed nothing unique in this setting.

### What This Means

IGAD requires families where R(theta) varies meaningfully with parameters.
This holds when the third cumulant tensor T_{ijk} changes substantially
across the parameter space — which is the case for Gamma but not Gaussian.

### Families Where IGAD Is Applicable

| Family       | dim | R varies? | IGAD applicable? |
|---|---|---|---|
| Poisson      | 1   | No (R=0)  | No               |
| Exponential  | 1   | No (R=0)  | No               |
| Gamma        | 2   | Yes       | Yes (confirmed)  |
| Gaussian     | 3   | No (const)| No               |
| Dirichlet    | k-1 | Yes       | Promising        |
| Neg-Binomial | 2   | Yes       | Untested         |

### Next Step

Dirichlet(alpha_1, ..., alpha_k) with k>=3 is a d=k-1 family where
R varies with parameters and mean+variance do not determine all parameters.
This is the most promising direction for a strong d>=3 result.
