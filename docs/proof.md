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
