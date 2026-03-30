import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EllipticEnvelope
from igad.detector import IGADDetector
from igad.families import GammaFamily
from igad.curvature import scalar_curvature

def run_demo(seed=42):
    rng = np.random.default_rng(seed)
    n_normal, n_anomaly = 500, 50
    alpha_n, beta_n = 9.0, 3.0
    X_normal = rng.gamma(alpha_n, 1.0/beta_n, size=n_normal)
    alpha_a, beta_a = 1.5, 0.5
    X_anomaly = rng.gamma(alpha_a, 1.0/beta_a, size=n_anomaly)
    X_test = np.concatenate([X_normal[:100], X_anomaly])
    y_true = np.concatenate([np.zeros(100), np.ones(n_anomaly)])

    print("=" * 60)
    print("IGAD Demo: Skewness-Based Anomaly Detection")
    print("=" * 60)
    print("Normal:  Gamma(9,3) mean=3.00 skew=0.667")
    print("Anomaly: Gamma(1.5,0.5) mean=3.00 skew=1.633")

    det = IGADDetector(family=GammaFamily, k_neighbors=25)
    det.fit(X_normal.reshape(-1, 1))

    scores_igad = np.zeros(len(X_test))
    for i, z in enumerate(X_test):
        d, idx = det.tree_.query(z.reshape(1), k=det.k)
        nb = det.X_train_[idx.ravel()].ravel()
        nb = nb[nb > 0]
        if len(nb) < 5:
            scores_igad[i] = np.inf
            continue
        try:
            tl = GammaFamily.mle(nb)
            scores_igad[i] = det.R_ref_ - scalar_curvature(GammaFamily.log_partition, tl)
        except Exception:
            scores_igad[i] = np.inf

    fm = np.isfinite(scores_igad)
    if not all(fm):
        mx = np.max(scores_igad[fm]) if any(fm) else 0
        scores_igad[~fm] = mx * 2
    auc_igad = roc_auc_score(y_true, scores_igad)

    try:
        ee = EllipticEnvelope(contamination=0.1)
        ee.fit(X_normal.reshape(-1, 1))
        sm = -ee.score_samples(X_test.reshape(-1, 1))
        auc_mahal = roc_auc_score(y_true, sm)
    except Exception:
        auc_mahal = 0.5

    mu, sig = np.mean(X_normal), np.std(X_normal)
    sz = np.abs(X_test - mu) / sig
    auc_z = roc_auc_score(y_true, sz)

    print()
    hdr = "Method" + " " * 19 + "AUC-ROC"
    print(hdr)
    print("-" * 37)
    results = [("IGAD (curvature)", auc_igad), ("Mahalanobis", auc_mahal), ("Z-score", auc_z)]
    for name, auc in results:
        print("%-25s %10.4f" % (name, auc))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(X_normal, bins=40, alpha=0.6, label="Normal", density=True)
    axes[0].hist(X_anomaly, bins=20, alpha=0.6, label="Anomaly", density=True)
    axes[0].set_title("Data Distributions")
    axes[0].legend()
    axes[1].scatter(range(len(scores_igad)), scores_igad, c=y_true, cmap="coolwarm", alpha=0.6, s=15)
    axes[1].set_title("IGAD Scores (AUC=%.3f)" % auc_igad)
    axes[2].scatter(range(len(sz)), sz, c=y_true, cmap="coolwarm", alpha=0.6, s=15)
    axes[2].set_title("Z-Score (AUC=%.3f)" % auc_z)
    plt.tight_layout()
    plt.savefig("igad_demo.png", dpi=150)
    print()
    print("Plot saved to igad_demo.png")

if __name__ == "__main__":
    run_demo()
