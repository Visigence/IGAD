import numpy as np
from scipy import io, stats
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from igad import IGADDetector
from igad.families import GammaFamily
import urllib.request, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# הורד פגם גדול יותר — 0.021" (אותו סוג פגם, גודל שונה)
if not os.path.exists("inner_race_021.mat"):
    print("מוריד inner_race_021.mat...")
    urllib.request.urlretrieve(
        "https://engineering.case.edu/sites/default/files/209.mat",
        "inner_race_021.mat"
    )
    print(f"  ✓ ({os.path.getsize('inner_race_021.mat'):,} bytes)")

BATCH_SIZE = 200

def load_batches(path):
    mat = io.loadmat(path)
    keys = [k for k in mat.keys() if k.startswith('X') and 'DE' in k]
    signal = np.abs(mat[keys[0]].flatten())
    n = len(signal) // BATCH_SIZE
    return signal[:n * BATCH_SIZE].reshape(n, BATCH_SIZE)

ref_batches   = load_batches("inner_race_007.mat")  # ייחוס: פגם קטן
anom_batches  = load_batches("inner_race_021.mat")  # חריג: פגם גדול

n = min(len(ref_batches), len(anom_batches), 300)
ref_batches  = ref_batches[:n]
anom_batches = anom_batches[:n]

print(f"=== פגם קטן (007) vs פגם גדול (021) ===")
print(f"ref  mean={np.mean(ref_batches):.4f}  std={np.std(ref_batches):.4f}")
print(f"anom mean={np.mean(anom_batches):.4f}  std={np.std(anom_batches):.4f}")
print(f"יחס ממוצע: {np.mean(anom_batches)/np.mean(ref_batches):.2f}x")

all_batches = np.vstack([ref_batches, anom_batches])
labels      = np.array([0]*n + [1]*n)

detector = IGADDetector(family=GammaFamily)
detector.fit(ref_batches.flatten())

ref = ref_batches[0]
igad_scores = [detector.score_batch(b) for b in all_batches]
wass_scores = [wasserstein_distance(ref, b) for b in all_batches]
mean_scores = [abs(np.mean(b) - np.mean(ref)) for b in all_batches]

results = {
    "IGAD (curvature)": roc_auc_score(labels, igad_scores),
    "Wasserstein":      roc_auc_score(labels, wass_scores),
    "Mean shift":       roc_auc_score(labels, mean_scores),
}

print("\n" + "="*40)
for method, auc in sorted(results.items(), key=lambda x: -x[1]):
    tag = " ◄" if "IGAD" in method else ""
    print(f"  {method:<22} {auc:.4f}{tag}")
print("="*40)
gap = results["IGAD (curvature)"] - results["Wasserstein"]
print(f"\n  פער IGAD − Wasserstein: {gap:+.4f}")
