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

# התגיות האמיתיות (עם גרשיים)
NSR_TAG  = "'(N'"
AFIB_TAG = "'(AFIB'"

print(f"fs={fs}Hz | סה״כ annotations: {len(ann.aux_note)}")

# בנה מיפוי sample → קצב
# כל תגית תקפה עד התגית הבאה
rr_samples = np.diff(ann.sample)
rr_sec     = rr_samples / fs

nsr_rr  = []
afib_rr = []

current_rhythm = None
for i, note in enumerate(ann.aux_note):
    if note in (NSR_TAG, AFIB_TAG):
        current_rhythm = note
    if i < len(rr_sec) and current_rhythm is not None:
        rr = rr_sec[i]
        if 0.3 < rr < 2.0:
            if current_rhythm == NSR_TAG:
                nsr_rr.append(rr)
            elif current_rhythm == AFIB_TAG:
                afib_rr.append(rr)

nsr_rr  = np.array(nsr_rr)
afib_rr = np.array(afib_rr)

print(f"\nNSR  intervals: {len(nsr_rr):,}")
print(f"AFIB intervals: {len(afib_rr):,}")
print(f"\nNSR   mean={np.mean(nsr_rr):.4f}  std={np.std(nsr_rr):.4f}  skew={stats.skew(nsr_rr):.4f}")
print(f"AFIB  mean={np.mean(afib_rr):.4f}  std={np.std(afib_rr):.4f}  skew={stats.skew(afib_rr):.4f}")
print(f"יחס ממוצע: {np.mean(afib_rr)/np.mean(nsr_rr):.3f}x")

# בדיקת משפחה
a, _, scale = stats.gamma.fit(nsr_rr, floc=0)
ks = stats.kstest(nsr_rr, lambda x: stats.gamma.cdf(x, a, scale=scale))
print(f"\nKS גאמא על NSR: p={ks.pvalue:.4f} {'✅' if ks.pvalue > 0.05 else '❌'}")
print(f"shape α={a:.4f}")

# אצוות
BATCH_SIZE = 100
def to_batches(x, bs):
    n = len(x) // bs
    return x[:n*bs].reshape(n, bs)

nsr_b  = to_batches(nsr_rr,  BATCH_SIZE)
afib_b = to_batches(afib_rr, BATCH_SIZE)
n      = min(len(nsr_b), len(afib_b), 200)

if n == 0:
    print(f"\n❌ אצוות לא מספיקות — NSR:{len(nsr_b)} AFIB:{len(afib_b)}")
    print("צריך להוריד רשומה נוספת עם יותר נתונים")
else:
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
    for method, auc in sorted(results.items(), key=lambda x: -x[1]):
        tag = " ◄" if "IGAD" in method else ""
        print(f"  {method:<22} {auc:.4f}{tag}")
    print("="*42)
    gap = results["IGAD (curvature)"] - results["Wasserstein"]
    print(f"\n  פער IGAD − Wasserstein: {gap:+.4f}")
