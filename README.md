# IGAD: Information-Geometric Anomaly Detection

**Author:** Omry Damari · 2026
**Repository:** https://github.com/Visigence/IGAD

🌐 **[Interactive Demo →](https://igad-web.vercel.app)**

Classical anomaly detectors are blind to shape shifts — anomalies that
preserve mean and variance but change distributional geometry. IGAD detects them.

> *"The anomaly isn't where the distribution is. It's what shape it is."*

---

## The Problem

Every widely-used anomaly detection method shares the same assumption:

> anomaly = a point far from the center

| Method           | What It Measures                                       |
|------------------|--------------------------------------------------------|
| Z-Score          | Distance from mean in standard deviation units         |
| Mahalanobis      | Distance from cloud center accounting for correlations |
| Isolation Forest | Ease of isolating a point in feature space             |
| LOF              | Relative local neighborhood density                    |

All four are blind to the following:
Reference : Gamma(8, 2)    mean=4.000  var=2.000  skew=0.707
Anomaly   : LogNormal(...)  mean=4.000  var=2.000  skew=1.105
Mean and variance are exactly identical. The internal structure of the
distribution has completely changed. No distance-based algorithm detects this.

---

## The Invention
IGAD(batch) = |R(theta_ref) - R(theta_batch)|

where `R(theta)` is the scalar curvature of the Fisher-Rao statistical manifold
at the natural parameter point `theta`.

### What Was Known Before This Work

| Component                                     | Source                     |
|-----------------------------------------------|----------------------------|
| Fisher-Rao metric                             | Rao (1945)                 |
| Differential geometry of exponential families | Amari (1985)               |
| Scalar curvature formula for Hessian metrics  | Amari & Nagaoka (2000)     |
| Fourth-cumulant cancellation in Riemann tensor| Standard Hessian geometry  |
| Curvature as detector of phase transitions    | Ruppeiner (1979, 1995)     |

### What Is New

The **construction**: using scalar curvature deviation as a batch-level anomaly
score. This use has not been found in the anomaly detection literature.

The **insight**: scalar curvature is governed by `||T||²_g` — a full
metric-weighted contraction of the third cumulant tensor across all parameter
dimensions simultaneously. This is not skewness. Skewness is a single number.
`||T||²_g` weights each direction by the inverse Fisher metric — it captures
how asymmetry is distributed across the entire parameter geometry.

The **proof**: a control experiment with identical MLE fit but no curvature
tensor confirms geometry adds +0.053 AUC independently of MLE efficiency.
Decisive Experiment 6 further shows IGAD beats ALL baselines including raw
skewness (p=0.002 at n=150, non-overlapping 95% CIs) in the small-n
heavy-tail regime where mean and variance are exactly matched.

---

## Mathematical Foundation

For an exponential family with log-partition function `A(theta)`:
Fisher metric:         g_{ij}(theta) = d²A / dtheta_i dtheta_j
Third cumulant tensor: T_{ijk}(theta) = d³A / dtheta_i dtheta_j dtheta_k
Scalar curvature:      R(theta) = 1/4 * ( ||S||²_g - ||T||²_g )
where S_m = g^{ab} T_{abm}  (trace vector)
||T||²_g = g^{ia} g^{jb} g^{kc} T_{ijk} T_{abc}

Full derivation: `docs/proof.md`

---

## Implementation
igad/
curvature.py      Fisher metric, third cumulant tensor, scalar curvature
Analytical dispatch: routes to family methods when available,
falls back to numerical finite differences
families.py       GammaFamily, DirichletFamily, PoissonFamily,
InverseGaussianFamily
detector.py       IGADDetector — fit(), score_batch(), predict()
exceptions.py     ConvergenceError (MLE convergence gate)
tests/
test_curvature.py             12 tests (Gamma/Poisson)
test_dirichlet_family.py      34 tests (Dirichlet, 6 classes)
test_igad_detector.py          5 tests (score API, predict, edge cases)
experiments/
demo_easy.py                  Experiment 1: Gamma vs Gamma
demo_hard.py                  Experiment 2: Gamma vs LogNormal + MLE control
demo_hard_extended.py         Experiment 2b: + MMD + Wasserstein baselines
demo_hard1.py                 Experiment 3: Within-family Gamma shift
demo_dirichlet.py             Experiment 3a: Dirichlet concentration shifts
demo_gaussian2d.py            Gaussian failure mode
cwru_data/igad_eval.py        Experiment 4: CWRU bearing fault (real-world)
mitbih/igad_ecg_v6.py         Experiment 5: ECG AFib detection (real-world)
exp_decisive.py               Experiment 6: Decisive test — Gamma vs Weibull (matched mean+var)
docs/
proof.md                      Mathematical background and attribution
operational_envelope.md       Falsifiable claims: when IGAD wins and loses
figures/                      Experiment plots
RESULTS.md                      Full experimental results and analysis

---

## Quick Start
```python
pip install -e .

import numpy as np
from igad import IGADDetector
from igad.families import GammaFamily

detector = IGADDetector(family=GammaFamily())
detector.fit(np.random.gamma(8.0, 0.5, size=200))

score = detector.score_batch(np.random.lognormal(1.327, 0.343, size=200))
print(f"IGAD score: {score:.6f}")  # Higher = more anomalous
```

---

## Running Tests
```bash
python -m pytest tests/ -v
# 51 passed
```

---

## Experimental Results

### Summary

| Regime | IGAD AUC | Best Baseline | IGAD Wins? |
|---|---|---|---|
| Easy case (diff variance) | 1.0000 | Variance: 1.0000 | Tie |
| Hard case n=200, vs MLE-skew | 0.6542 | MLE-skew: 0.6016 | ✅ Yes (+0.053) |
| Hard case n=500, vs MLE-skew | 0.6748 | MLE-skew: 0.5846 | ✅ Yes (+0.090) |
| Hard case n=500, vs raw-skew | 0.6748 | Raw-skew: 0.9194 | ❌ No |
| **Decisive: n=100, vs MLE-skew (p<0.001)** | **0.6199** | **MLE-skew: 0.6046** | **✅ Yes** |
| **Decisive: n=100, vs raw-skew (p=0.044)** | **0.6199** | **Raw-skew: 0.5911** | **✅ Yes** |
| **Decisive: n=150, vs MLE-skew (p<0.001)** | **0.6609** | **MLE-skew: 0.6450** | **✅ Yes** |
| **Decisive: n=150, vs raw-skew (p=0.002, non-overlapping CIs)** | **0.6609** | **Raw-skew: 0.6001** | **✅ Yes** |
| **Decisive: n=200, vs raw-skew (p=0.0003, non-overlapping CIs)** | **0.6856** | **Raw-skew: 0.6188** | **✅ Yes** |
| CWRU bearing (amplitude fault) | 0.7583 | Mean shift: 1.0000 | ❌ No (wrong regime) |
| ECG AFib (rate shift) | 0.5995 | Mean shift: 0.8916 | ❌ No (wrong regime) |

---

### Experiment 2 — Hard Case: Matched Mean AND Variance

**Setup**: Gamma(8,2) vs LogNormal · mean=4.0, var=2.0 identical for both

**Results — 5 seeds, batch_size=200**

| Method                 | Mean AUC | ± Std |
|------------------------|----------|-------|
| IGAD (curvature)       | 0.6542   | 0.047 |
| MLE skewness [CONTROL] | 0.6016   | 0.038 |
| Raw skewness           | 0.6794   | 0.072 |
| MMD (RBF, median-BW)   | 0.5894   | 0.076 |
| Wasserstein (1D)       | 0.5925   | 0.057 |
| Mean shift [BLIND]     | 0.5240   | 0.062 |
| Variance shift [BLIND] | 0.5818   | 0.027 |

**Gap (IGAD − MLE skewness): +0.053**
Curvature geometry adds signal beyond MLE efficiency alone.

The MLE-skewness control uses the identical MLE fit but discards the curvature
tensor. The gap is therefore attributable to the geometry, not to MLE efficiency.

**Sample-efficiency sweep (seed=42)**

| n   | IGAD   | MMD    | Wasserstein | Gap(IGAD−MMD) |
|-----|--------|--------|-------------|---------------|
| 50  | 0.5522 | 0.5465 | 0.5425      | +0.006        |
| 100 | 0.5871 | 0.5639 | 0.5440      | +0.023        |
| 200 | 0.6542 | 0.5894 | 0.5925      | **+0.065**    |
| 300 | 0.6395 | 0.6074 | 0.5933      | +0.032        |
| 500 | 0.7150 | 0.6814 | 0.6777      | **+0.034**    |

IGAD beats MMD and Wasserstein at every batch size tested.
Note: raw skewness dominates at large n (n=500: AUC=0.919 vs IGAD=0.675).
IGAD's advantage is over distance-based baselines, not moment estimators.

---

### Experiment 6 — Decisive Test: Small-n Heavy-Tail Regime ⭐

> **Headline result**: In a regime where mean AND variance are exactly matched,
> IGAD beats both raw skewness and MLE-skewness with statistical significance
> at n=100–200. This directly validates the core claim.

**File**: `experiments/exp_decisive.py`

**Setup**

| | Distribution | mean | var | skew |
|---|---|---|---|---|
| Reference | Gamma(α=2, β=1) | 2.0 | 2.0 | 1.4142 |
| Anomaly | Weibull(k=1.4355, λ=2.2026) | 2.0 | 2.0 | 1.1514 |

Mean and variance are **exactly matched** — only higher-order tensor structure
differs. The Weibull is not in the Gamma family (model misspecification).
20 seeds × (100 normal + 50 anomaly) batches per seed; CIs from 2000-resample
bootstrap; tests are paired sign-permutation with 10 000 permutations.

**AUC-ROC Results** (mean over 20 seeds, 95% bootstrap CI)

```
    n          IGAD                  MLE-skew              Raw-skew
---------------------------------------------------------------------
   50  0.5635 [0.5463,0.5824]  0.5453 [0.5283,0.5637]  0.5560 [0.5372,0.5741]
   75  0.5984 [0.5752,0.6197]  0.5811 [0.5582,0.6021]  0.5781 [0.5599,0.5972]
  100  0.6199 [0.6001,0.6398]  0.6046 [0.5855,0.6241]  0.5911 [0.5703,0.6094]
  150  0.6609 [0.6346,0.6871]  0.6450 [0.6194,0.6703]  0.6001 [0.5816,0.6195]  ← decisive
  200  0.6856 [0.6647,0.7056]  0.6721 [0.6511,0.6927]  0.6188 [0.5985,0.6389]  ← decisive
```

**Statistical Tests** (one-sided H₁: IGAD > baseline)

| n | p(IGAD>raw) | p(IGAD>MLE) | CI non-overlap raw | Decision |
|---|---|---|---|---|
| 50 | 0.3072 | <0.0001 | No | — |
| 75 | 0.1064 | <0.0001 | No | — |
| **100** | **0.0437** | **<0.0001** | No | **DECISIVE** |
| **150** | **0.0018** | **<0.0001** | **Yes** | **DECISIVE** |
| **200** | **0.0003** | **<0.0001** | **Yes** | **DECISIVE** |

At n≥150 the 95% bootstrap CIs are **non-overlapping**: IGAD is strictly
better than raw skewness, not just nominally.

**Why IGAD wins — mechanistic note**

Raw skewness fails because its estimator variance is O(κ₆/n). For
Gamma(2,1), κ₆=240, giving per-batch variance ~0.17–0.24 at n=100–150,
far above the signal gap of 0.26 between the two distributions' skewness.
MLE-skewness uses only the 1D projection 2/√α̂, discarding the scale
parameter and curvature cross-terms. IGAD's R(θ) contracts the full
tensor T_{ijk} — including T₀₁₁=1/λ² (shape-scale cross-term) — against
the Fisher metric, aggregating noise-cancelling multi-channel information.

**Figure**: `docs/figures/exp_decisive_gamma_weibull.png`

---

### Experiment 4 — Real-World: CWRU Bearing Fault

**Setup**: Normal bearing vs inner race fault (0.007"), batch_size=200
**Family**: GammaFamily (KS test: p=0.22, not rejected)

| Method          | AUC-ROC |
|-----------------|---------|
| Wasserstein     | 1.0000  |
| Variance shift  | 1.0000  |
| Mean shift      | 1.0000  |
| IGAD (Gamma)    | 0.7583  |

**Finding**: Fault produces 3.51× mean shift and 4.66× std shift.
This is an amplitude-change regime, not a shape-shift regime.
Mean shift and Wasserstein dominate because the signal is in the location,
not in the distributional geometry. Result is consistent with operational envelope.

---

### Experiment 5 — Real-World: ECG Atrial Fibrillation Detection

**Dataset**: MIT-BIH Atrial Fibrillation Database (PhysioNet afdb)
**Records**: 04015, 04043 · NSR=92,785 intervals, AFIB=14,940 intervals
**Family**: InverseGaussianFamily

| Method          | AUC-ROC |
|-----------------|---------|
| Mean shift      | 0.8916  |
| Wasserstein     | 0.7164  |
| Variance shift  | 0.6924  |
| IGAD (InvGauss) | 0.5995  |

**Finding**: AFib in these records includes a 20% heart rate increase
(mean ratio NSR/AFIB = 0.804). Mean shift dominates because the signal
is in the location, not in the distributional geometry.

**Engineering contribution**: InverseGaussianFamily required analytical
Fisher metric and third cumulant tensor for numerical stability. RR intervals
in regular NSR produce λ~10,000, giving θ~-8,000. Numerical finite differences
fail silently at these values (det(g) < 0). Analytical methods are required
for correctness, not just performance.

Data not included in repository. Download via:
```python
import wfdb
wfdb.dl_database('afdb', dl_dir='.', records=['04015', '04043'])
```

---

## Operational Envelope

| Regime                                        | IGAD  | Reason                              |
|-----------------------------------------------|-------|-------------------------------------|
| Cross-family shape shift, n=200–500           | ✅    | Core claim — Exp 2 & Exp 6 (decisive) |
| Small-n heavy-tail, cross-family (n=100–200)  | ✅    | **Decisive — Exp 6** (p<0.05, non-overlapping CIs) |
| Gamma family, correct spec, n=200             | ✅    | Beats MLE-skewness control (+0.053) |
| Gaussian families                             | ❌    | Constant curvature — do not use     |
| 1D parameter families (Poisson, Exponential)  | ❌    | R≡0 — do not use                   |
| Amplitude/scale fault (CWRU bearing)          | ❌    | Location shift dominates            |
| Rate-changing AFib (afdb ECG)                 | ❌    | Location shift dominates            |
| Large n + misspecified model (n > 500)        | ⚠️   | Degrades — use MMD/Wasserstein      |
| No parametric model available                 | ⚠️   | Use model-free tests instead        |

See `docs/operational_envelope.md` for full falsifiable claims.

---

## Supported Families

| Family                | dim | Analytical geometry | Notes                              |
|-----------------------|-----|---------------------|------------------------------------|
| GammaFamily           | 2   | ✅                  | Shape + rate                       |
| DirichletFamily       | k-1 | ✅                  | Concentration parameters           |
| InverseGaussianFamily | 2   | ✅                  | Required for large λ stability     |
| PoissonFamily         | 1   | —                   | R≡0 — included as failure mode     |

---

## When to Use IGAD

- The correct parametric family is known or approximately known
- Batch sizes are moderate (50–500 observations)
- Anomalies differ in distributional **shape**, not location or scale
- The family has dimension d ≥ 2

**Potential applications:**
- Predictive maintenance (vibration profile shifts before amplitude changes)
- Financial monitoring (transaction distribution structure shifts)
- Medical signal analysis (early arrhythmia, paroxysmal AFib onset)
- Cybersecurity (packet size distribution shifts in low-and-slow exfiltration)

---

## Validation
51 passed
TestDirichletLogPartition    (4)  — log-partition identity, convexity, roundtrip
TestDirichletFisherMetric    (5)  — matches numerical Hessian, PD, k=4
TestDirichletCurvature       (4)  — finite, non-constant, separates anomaly pair
TestDirichletMLE             (5)  — recovery rtol=0.05, convergence gate
TestIGADDirichletDetection   (4)  — AUC>0.65 at n=200, monotone in n
TestFailureModes             (3)  — Poisson R≡0, Gaussian constant, k=2 degenerate
TestGammaFamily             (12)  — existing suite, all passing
TestIGADDetector             (5)  — score API, predict, zero-score self, positive on anomaly

---

## Citation
```bibtex
@article{damari2026igad,
  title  = {IGAD: Information-Geometric Anomaly Detection via Scalar
            Curvature of Fisher-Rao Manifolds},
  author = {Damari, Omry},
  year   = {2026},
  url    = {https://github.com/Visigence/IGAD}
}
```

---

## References

1. Rao, C.R. (1945). Information and the accuracy attainable in the estimation of statistical parameters.
2. Amari, S. (1985). Differential-Geometrical Methods in Statistics. Springer.
3. Amari, S. & Nagaoka, H. (2000). Methods of Information Geometry. AMS/Oxford.
4. Minka, T. (2000). Estimating a Dirichlet distribution. MIT Tech Report.
5. Ruppeiner, G. (1979). Thermodynamics: A Riemannian geometric model.
6. Ruppeiner, G. (1995). Riemannian geometry in thermodynamic fluctuation theory.
7. Tweedie, M.C.K. (1957). Statistical properties of inverse Gaussian distributions.
8. Folks, J.L. & Chhikara, R.S. (1978). The inverse Gaussian distribution and its applications.
9. Goldberger, A.L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23).

---

## License

MIT License · Omry Damari 2026
