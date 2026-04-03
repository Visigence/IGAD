import numpy as np
import wfdb
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from igad import IGADDetector
from igad.families import GammaFamily
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

RECORD = "04015"
DB     = "afdb"

print(f"מוריד {RECORD} מ-PhysioNet...")
wfdb.dl_database(DB, dl_dir=".", records=[RECORD])
print("✓ הורדה הושלמה\n")

ann = wfdb.rdann(RECORD, "atr")
rec = wfdb.rdrecord(RECORD)
fs  = rec.fs

rr_samples = np.diff(ann.sample)
rr_sec     = rr_samples / fs
rhythms    = ann.aux_note[1:]

nsr_rr  = rr_sec[(np.array(rhythms) == '(N')    & (rr_sec > 0.3) & (rr_sec < 2.0)]
afib_rr = rr_sec[(np.array(rhythms) == '(AFIB') & (rr_sec > 0.3) & (rr_sec < 2.0)]

print(f"NSR  intervals: {len(nsr_rr):,}")
print(f"AFIB intervals: {len(afib_rr):,}")

print(f"\nNSR   mean={np.mean(nsr_rr):.4f}  std={np.std(nsr_rr):.4f}  skew={stats.skew(nsr_rr):.4f}")
print(f"AFIB  mean={np.mean(afib_rr):.4f}  std={np.std(afib_rr):.4f}  skew={stats.skew(afib_rr):.4f}")
print(f"יחס ממוצע: {np.mean(afib_rr)/np.mean(nsr_rr):.3f}x")

a, _, scale = stats.gamma.fit(nsr_rr, floc=0)
ks = stats.kstest(nsr_rr, lambda x: stats.gamma.cdf(x, a, scale=scale))
print(f"\nKS גאמא על NSR: p={ks.pvalue:.4f} {'✅' if ks.pvalue > 0.05 else '❌'}")

BATCH_SIZE = 100
def to_batches(x, bs):
    n = len(x) // bs
    return x[:n*bs].reshape(n, bs)

nsr_b  = to_batches(nsr_rr,  BATCH_SIZE)
afib_b = to_batches(afib_rr, BATCH_SIZE)
n      = min(len(nsr_b), len(afib_b), 200)
nsr_b  = nsr_b[:n]
afib_b = afib_b[:n]

all_b  = np.vstack([nsr_b, afib_b])
labels = np.array([0]*n + [1]*n)

detector = IGADDetector(family=GammaFamily)
detector.fit(nsr_b.flatten())

ref = nsr_b[0]
igad_scores = [detector.score_batch(b) for b in all_b]
wass_scores = [wasserstein_distance(ref, b) for b in all_b]
mean_scores = [abs(np.mean(b) - np.mean(ref)) for b in all_b]
var_scores  = [abs(np.var(b)  - np.var(ref))  for b in all_b]

results = {
    "IGAD (curvature)": roc_auc_score(labels, igad_scores),
    "Wasserstein":      roc_auc_score(labels, wass_scores),
    "Mean shift":       roc_auc_score(labels, mean_scores),
    "Variance shift":   roc_auc_score(labels, var_scores),
}

print("\n" + "="*42)
print(f"  {'שיטה':<22} {'AUC':>6}")
print("="*42)
for method, auc in sorted(results.items(), key=lambda x: -x[1]):
    tag = " ◄" if "IGAD" in method else ""
    print(f"  {method:<22} {auc:.4f}{tag}")
print("="*42)
gap = results["IGAD (curvature)"] - results["Wasserstein"]
print(f"\n  פער IGAD − Wasserstein: {gap:+.4f}")
if gap > 0.03:
    print("  → IGAD מנצח ✅")
elif gap > -0.03:
    print("  → שניהם דומים")
else:
    print("  → Wasserstein מנצח — שינוי עוצמה, לא צורה")
