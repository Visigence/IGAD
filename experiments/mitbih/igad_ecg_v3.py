import numpy as np
import wfdb
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from igad import IGADDetector
from igad.families import GammaFamily
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ann = wfdb.rdann("04015", "atr")
rec = wfdb.rdrecord("04015")
fs  = rec.fs

NSR_TAG  = "(N"
AFIB_TAG = "(AFIB"

print(f"fs={fs}Hz | annotations: {len(ann.aux_note)}")

# בנה מיפוי: כל RR מקבל את הקצב הנוכחי
rr_sec        = np.diff(ann.sample) / fs
current       = None
nsr_rr, afib_rr = [], []

for i, note in enumerate(ann.aux_note):
    if note in (NSR_TAG, AFIB_TAG):
        current = note
    if i < len(rr_sec) and current is not None:
        rr = rr_sec[i]
        if 0.3 < rr < 2.0:
            if current == NSR_TAG:
                nsr_rr.append(rr)
            else:
                afib_rr.append(rr)

nsr_rr  = np.array(nsr_rr)
afib_rr = np.array(afib_rr)

print(f"NSR  intervals: {len(nsr_rr):,}")
print(f"AFIB intervals: {len(afib_rr):,}")

if len(nsr_rr) < 100 or len(afib_rr) < 100:
    print("\n❌ לא מספיק נתונים ברשומה 04015 — מוריד רשומות נוספות")
    for record in ["04043", "04048", "04126"]:
        print(f"מוריד {record}...")
        wfdb.dl_database("afdb", dl_dir=".", records=[record])
        ann2 = wfdb.rdann(record, "atr")
        rec2 = wfdb.rdrecord(record)
        fs2  = rec2.fs
        rr2  = np.diff(ann2.sample) / fs2
        cur2 = None
        for i, note in enumerate(ann2.aux_note):
            if note in (NSR_TAG, AFIB_TAG):
                cur2 = note
            if i < len(rr2) and cur2 is not None:
                rr = rr2[i]
                if 0.3 < rr < 2.0:
                    if cur2 == NSR_TAG:
                        nsr_rr = np.append(nsr_rr, rr)
                    else:
                        afib_rr = np.append(afib_rr, rr)
        print(f"  NSR={len(nsr_rr):,}  AFIB={len(afib_rr):,}")
        if len(nsr_rr) >= 500 and len(afib_rr) >= 500:
            break

print(f"\nסה״כ NSR={len(nsr_rr):,}  AFIB={len(afib_rr):,}")
print(f"NSR   mean={np.mean(nsr_rr):.4f}  std={np.std(nsr_rr):.4f}  skew={stats.skew(nsr_rr):.4f}")
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
print(f"\nאצוות: {n} NSR + {n} AFIB")

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
    print("  → IGAD מנצח ✅ — שינוי צורה ללא שינוי עוצמה")
elif gap > -0.03:
    print("  → שניהם דומים")
else:
    print("  → Wasserstein מנצח — יש שינוי עוצמה גם כן")
