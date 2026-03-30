# IGAD: Information-Geometric Anomaly Detection

**Information-Geometric Anomaly Detection via Fisher-Rao Scalar Curvature**

## Overview

IGAD is a Python library for detecting anomalies using information geometry, specifically leveraging the Fisher-Rao scalar curvature of statistical manifolds. This approach provides a principled, geometric framework for identifying outliers and anomalies in data.

## Key Features

- **Fisher-Rao Curvature**: Computes the scalar curvature of statistical manifolds
- **Anomaly Detection**: Detects points with unusual geometric properties
- **Multiple Distribution Families**: Support for Gaussian, Poisson, Gamma, and other exponential families
- **Third Cumulant Analysis**: Advanced tensor computations for distribution geometry

## Demonstrations

### Basic Demo
**The IGAD Advantage:** In a standard shape-anomaly scenario (matched means, altered higher-order moments), IGAD achieves perfect separability (AUC = 1.000). Notice that it even outperforms a direct, naive comparison of empirical skewness (AUC = 0.983). This empirically validates our Skewness Decomposition Theorem: the intrinsic Fisher-Rao geometry provides a more precise signal than crude statistical moments.
![IGAD Demo](igad_demo.png)

### Hard Cases Demo
**Pushing the Limits:** In highly noisy, challenging regimes where standard location-scale methods fail completely (AUC ~ 0.50), IGAD still manages to extract a meaningful geometric signal (AUC = 0.684), consistently maintaining its edge over basic skewness shifts.
![IGAD Hard Demo](igad_hard_demo.png)

## Real-World Use Cases

Because IGAD excels at detecting "shape anomalies" where the mean and variance remain unchanged, it is highly effective in complex real-world scenarios:

* **Predictive Maintenance:** In rotating machinery or industrial equipment, degradation often begins with subtle shifts in the *shape* of the acoustic or vibration noise profile, long before the overall vibration amplitude (mean/variance) spikes.
* **Financial Fraud Detection:** Sophisticated fraudsters often attempt to "blend in" by matching the normal transaction volumes and frequencies of a user. IGAD detects the subtle structural anomalies in the transaction distribution that expose the fraud.
* **Medical Signal Analysis:** Analyzing complex biological signals like ECG or EEG to identify microscopic, structural changes in the signal's geometry that indicate physiological anomalies, which standard amplitude-based monitors might miss.

## Why IGAD?

Most traditional anomaly detection algorithms (like Isolation Forest, Mahalanobis distance, or LOF) operate on a fundamental assumption: anomalies exist far from the center of the data. 

**But what if an anomaly is hiding in plain sight?** Consider a system degradation where the new data points share the exact same mean and variance as normal operations, but have a fundamentally different distributional shape (e.g., a shift in skewness). Distance-based detectors are completely blind to these shape-shifting anomalies.

**IGAD fills this gap.** By measuring the scalar curvature of the Fisher-Rao manifold, IGAD is inherently sensitive to the squared norm of the third cumulant tensor — the multivariate generalization of skewness. The IGAD score elegantly compares the local statistical geometry of a test point's neighborhood against the global reference geometry, allowing you to detect structural anomalies that leave location and scale unchanged.

**The Paradigm Shift:** IGAD is more than just a code library; it represents a fundamental shift in anomaly detection—moving from measuring the *distance* between points to measuring the *curvature* of the statistical space.

## The Mathematics of IGAD

### The Core Equation
IGAD leverages the Fisher-Rao scalar curvature ($R$) as an anomaly signal. The anomaly score is calculated as the difference between the global curvature of the reference system and the local curvature around the evaluated point $z$:

$$IGAD(z) = R(\theta_{ref}) - R(\hat{\theta}(z))$$

Where $\theta_{ref}$ represents the parameters of the global reference distribution, and $\hat{\theta}(z)$ represents the local maximum likelihood estimate of the parameters in the neighborhood of $z$.

### Skewness Decomposition Theorem
At the heart of IGAD is the Skewness Decomposition Theorem. It proves that the Fisher-Rao scalar curvature acts as a "mirror" to the squared norm of the Third Cumulant Tensor. This makes curvature the most mathematically precise metric for detecting shape shifts that remain invisible to standard distance tests.

### Geometric Invariance
Unlike distance-based metrics (such as Euclidean or Mahalanobis distances) which can drastically change depending on how your data is scaled or represented, IGAD is **geometrically invariant**. Because scalar curvature is an intrinsic property of the statistical manifold, IGAD measures the true essence of the distribution, entirely independent of the chosen mathematical parameterization.

## Installation

```bash
pip install -r requirements.txt
python setup.py install
```

### Requirements
- Python >= 3.9
- numpy >= 1.24
- scipy >= 1.10
- scikit-learn >= 1.2



## Usage: Full Anomaly Scoring Workflow

The true power of IGAD lies in comparing global reference curvature to local neighborhood curvature. Here is how to compute the IGAD anomaly score for test points:

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from igad.families import GammaFamily
from igad.curvature import scalar_curvature

# 1. Setup Data
X_train = np.random.gamma(shape=9.0, scale=3.0, size=(500, 1)) # Reference data
X_test = np.array([[3.0], [8.0], [27.0]]) # Points to evaluate

# 2. Compute Global Reference Geometry
# Fit MLE to the entire reference dataset
alpha_ref, beta_ref = GammaFamily.fit_mle(X_train)
theta_ref = GammaFamily.to_natural(alpha_ref, beta_ref)

# R_ref is the scalar curvature of the normal operating state
R_ref = scalar_curvature(GammaFamily.log_partition, theta_ref)

# 3. Local Geometry & Anomaly Scoring (k-NN)
k = 20
nn = NearestNeighbors(n_neighbors=k).fit(X_train)
distances, indices = nn.kneighbors(X_test)

igad_scores = []

for i, neighbors_idx in enumerate(indices):
    local_data = X_train[neighbors_idx]
    
    # Fit local MLE for the test point's neighborhood
    alpha_local, beta_local = GammaFamily.fit_mle(local_data)
    theta_local = GammaFamily.to_natural(alpha_local, beta_local)
    
    # Compute local scalar curvature
    R_local = scalar_curvature(GammaFamily.log_partition, theta_local)
    
    # The IGAD Score: Difference between global and local curvature
    score = R_ref - R_local
    igad_scores.append(score)

print(f"Global Reference Curvature: {R_ref:.4f}")
for pt, score in zip(X_test.flatten(), igad_scores):
    print(f"Point {pt:5.1f} | IGAD Score: {score:.4f} (Higher = More Anomalous)")
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

## Author

Omry Damari

## License

See [LICENSE](LICENSE) file for details.
