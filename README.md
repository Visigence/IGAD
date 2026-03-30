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
![IGAD Demo](igad_demo.png)

### Hard Cases Demo
![IGAD Hard Demo](igad_hard_demo.png)

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

## Usage

```python
from igad.families import GammaFamily
from igad.curvature import scalar_curvature, fisher_metric

# Define parameters
alpha, beta = 2.0, 1.0

# Convert to natural parameters
theta = GammaFamily.to_natural(alpha, beta)

# Compute Fisher metric
g = fisher_metric(GammaFamily.log_partition, theta)

# Compute scalar curvature
R = scalar_curvature(GammaFamily.log_partition, theta)

print(f"Scalar Curvature: {R}")
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
