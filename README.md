# IGAD: Information-Geometric Anomaly Detection

**Classical anomaly detectors are blind to shape shifts — anomalies that preserve mean and variance but change distributional geometry. IGAD detects them.
Batch-level anomaly detection via scalar curvature deviation on Fisher-Rao statistical manifolds.**

## Core Idea

Every exponential family of probability distributions defines a Riemannian manifold
(the Fisher-Rao manifold). Its scalar curvature R(theta) is governed by the
**third cumulant tensor** — the multivariate generalization of skewness.

IGAD scores a data batch by measuring how much its fitted curvature deviates
from a reference distribution:
```
IGAD(batch) = |R(theta_ref) - R(theta_local)|
```

This makes IGAD a **shape-sensitive** anomaly detector: it catches distributional
changes that are invisible to methods based on mean or variance alone.

## What Is Novel vs. What Is Known

| Component | Status |
|---|---|
| Fisher-Rao metric, information geometry | Known (Amari, 1985; Amari & Nagaoka, 2000) |
| Scalar curvature formula for Hessian metrics | Known (Ruppeiner, 1995) |
| Fourth-cumulant cancellation in Riemann tensor | Known |
| R = 1/4 (norm(grad log det g)^2 - norm(T)^2) | Known identity |
| **Curvature deviation as batch anomaly score** | **Novel (this work)** |
| **Skewness-contrast interpretation** | **Novel (this work)** |

## Results

### Experiment 1: Easy Case — Gamma(9,3) vs Gamma(1.5,0.5)
Same mean (3.0), different variance and skewness.

| Method | AUC-ROC |
|---|---|
| **IGAD (curvature)** | **1.0000** |
| Batch variance shift | 1.0000 |
| Batch skewness shift | 0.9834 |
| Batch mean shift | 0.8150 |

IGAD achieves perfect separation, but so does the variance baseline
(variance differs by 6x). This alone does not prove unique value.

### Experiment 2: Hard Case — Gamma(8,2) vs LogNormal (Matched Mean AND Variance)
mean=4.0, var=2.0 for both. Only skewness differs (0.707 vs 1.105).

| Method | AUC-ROC |
|---|---|
| **IGAD (curvature)** | **0.6838** |
| Batch skewness shift | 0.6514 |
| Batch variance shift | 0.5860 |
| Batch mean shift | 0.5502 |

**IGAD wins when location-scale methods are blind.** Mean and variance
baselines are near chance. IGAD outperforms even direct skewness comparison.

### Scaling Behavior (Hard Case)
```
batch_size= 200  IGAD=0.6838  Skewness=0.6514  (IGAD wins)
batch_size= 500  IGAD=0.6748  Skewness=0.9194  (Skewness wins)
batch_size=1000  IGAD=0.7892  Skewness=0.9686  (Skewness wins)
```

IGAD's advantage is in the **small-sample regime** (n < 300 per batch).
At larger batch sizes, model-free skewness estimation dominates due to
model misspecification (fitting Gamma MLE to LogNormal data).

See [RESULTS.md](RESULTS.md) for complete experimental details.

## When to Use IGAD

IGAD is most valuable when:
- The correct parametric family is known or approximately known
- Batch sizes are moderate (50-300 observations)
- Anomalies differ in distributional shape, not just location or scale
- The family has dimension d >= 2 (1D manifolds have R=0)

**Potential applications:**
- Predictive maintenance (vibration profile shape changes)
- Financial monitoring (transaction distribution structure shifts)
- Medical signal analysis (waveform geometry changes)

## When NOT to Use IGAD

- Anomalies are simple outliers (far from center) — use Isolation Forest
- No parametric model is appropriate — use model-free tests
- Large batch sizes available (n > 500) — direct skewness comparison is simpler
- 1D parameter families — scalar curvature is identically zero

## Installation
```bash
pip install -e .
```

## Quick Start
```python
import numpy as np
from igad import IGADDetector
from igad.families import GammaFamily

# Fit reference from normal data
detector = IGADDetector(family=GammaFamily)
reference_data = np.random.gamma(9.0, 1/3.0, size=200)
detector.fit(reference_data)

# Score a new batch
test_batch = np.random.gamma(1.5, 1/0.5, size=200)
score = detector.score_batch(test_batch)
print(f"IGAD score: {score:.6f}")  # Higher = more anomalous
```

## Running Tests
```bash
python -m pytest tests/ -v
```

12 tests validate the curvature engine against analytical results:
- Poisson flatness (1D manifolds have R=0)
- Fisher metric: numerical vs analytical (3 parameter sets)
- Third cumulant tensor: numerical vs analytical (3 parameter sets)
- Tensor symmetry (2 parameter sets)
- Curvature variation across parameters
- Curvature finiteness (12 parameter combinations)
- Formula consistency (manual vs function)

## Running Experiments
```bash
python experiments/demo_easy.py   # Gamma vs Gamma (easy case)
python experiments/demo_hard.py   # Gamma vs LogNormal (hard case)
```

## Project Structure
```
igad/                    # Python package
  __init__.py
  curvature.py           # Fisher metric, third cumulant tensor, scalar curvature
  families.py            # GammaFamily, PoissonFamily (with analytical formulas)
  detector.py            # IGADDetector (batch-level scoring)
tests/
  test_curvature.py      # 12 validation tests
experiments/
  demo_easy.py           # Experiment 1: different skewness + variance
  demo_hard.py           # Experiment 2: matched mean AND variance
docs/
  proof.md               # Mathematical background
RESULTS.md               # Full experimental results with analysis
```

## Mathematical Background

The scalar curvature of a Fisher-Rao manifold for an exponential family is:
```
R(theta) = 1/4 * ( ||S||^2_g - ||T||^2_g )
```

where g is the Fisher metric (Hessian of the log-partition function),
T is the third cumulant tensor, and S is its trace vector.

This identity is known in Hessian geometry (see docs/proof.md for derivation
and attribution). The novel contribution is using |R(ref) - R(local)| as
a batch-level anomaly score sensitive to distributional shape.

## References

- Amari, S. (1985). Differential-Geometrical Methods in Statistics. Springer.
- Amari, S. & Nagaoka, H. (2000). Methods of Information Geometry. AMS/Oxford.
- Ruppeiner, G. (1995). Riemannian geometry in thermodynamic fluctuation theory. Rev. Mod. Phys.

## Author

Omry Damari

## License

MIT — see [LICENSE](LICENSE)
