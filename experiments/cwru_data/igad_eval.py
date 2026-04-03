import numpy as np
from scipy import io, stats
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from igad import IGADDetector
from igad.families import GammaFamily
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

BATCH_SIZE = 200

def load_batches(path):
    mat = io.loadmat(path)
    keys = [k for k in mat.keys() if k.startswith('X') and 'DE' in k]
    signal = np.abs(mat[keys[0]].flatten())
    n = len(signal) // BATCH_SIZE
    return signal[:n * BATCH_SIZE].reshape(n, BATCH_SIZE)

print("טוען נתונים...")
normal_batches = load_batches("normal_0.mat")
faulty_batches = load_batches("inner_race_007.mat")

n = min(len(normal_batches), len(faulty_batches))
normal_batches = normal_batches[:n]
faulty_batches = faulty_batches[:n]
all_batches    = np.vstack([normal_batches, faulty_batches])
labels         = np.array([0]*n + [1]*n)

print(f"אצוות: {n} נורמל + {n} פגום = {len(all_batches)} סה״כ\n")

# --- אמן IGAD ---
detector = IGADDetector(family=GammaFamily)
detector.fit(normal_batches.flatten())
print("IGAD אומן ✓")

# --- ציון ---
ref = normal_batches[0]
print("מחשב ציונים...")

igad_scores = [detector.score_batch(b) for b in all_batches]
wass_scores = [wasserstein_distance(ref, b)           for b in all_batches]
mean_scores = [abs(np.mean(b) - np.mean(ref))         for b in all_batches]
var_scores  = [abs(np.var(b)  - np.var(ref))          for b in all_batches]
skew_scores = [abs(stats.skew(b) - stats.skew(ref))   for b in all_batches]

# --- AUC ---
results = {
    "IGAD (curvature)": roc_auc_score(labels, igad_scores),
    "Wasserstein":      roc_auc_score(labels, wass_scores),
    "Skewness":         roc_auc_score(labels, skew_scores),
    "Variance shift":   roc_auc_score(labels, var_scores),
    "Mean shift":       roc_auc_score(labels, mean_scores),
}

print("\n" + "="*45)
print(f"  {'שיטה':<22} {'AUC':>6}")
print("="*45)
for method, auc in sorted(results.items(), key=lambda x: -x[1]):
    tag = " ◄" if "IGAD" in method else ""
    print(f"  {method:<22} {auc:.4f}{tag}")
print("="*45)
gap = results["IGAD (curvature)"] - results["Wasserstein"]
print(f"\n  פער IGAD − Wasserstein: {gap:+.4f}")
if gap > 0.05:
    print("  → IGAD מנצח בבירור ✅")
elif gap > 0:
    print("  → IGAD מנצח מעט ✅")
elif gap > -0.05:
    print("  → שניהם דומים — שינוי לא רק צורה")
else:
    print("  → Wasserstein מנצח — בדוק assumption")
