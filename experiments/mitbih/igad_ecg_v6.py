import numpy as np
import wfdb
from scipy import stats, signal as sp_signal
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from igad import IGADDetector
from igad.families import InverseGaussianFamily
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def extract_rr(record_name):
    rec = wfdb.rdrecord(record_name)
    ann = wfdb.rdann(record_name, "atr")
    fs  = rec.fs
    ecg = rec.p_signal[:, 0]
    height   = np.percentile(ecg, 90)
    min_dist = int(0.25 * fs)
    peaks, _ = sp_signal.find_peaks(ecg, height=height, distance=min_dist)
    rr_sec   = np.diff(peaks) / fs
    rhythm   = []
    for p in peaks[1:]:
        idx = np.searchsorted(ann.sample, p, side='right') - 1
        rhythm.append(ann.aux_note[idx] if 0 <= idx < len(ann.aux_note) else "")
    rhythm = np.array(rhythm)
    nsr  = rr_sec[(rhythm == "(N")    & (rr_sec > 0.3) & (rr_sec < 2.0)]
    afib = rr_sec[(rhythm == "(AFIB") & (rr_sec > 0.3) & (rr_sec < 2.0)]
    print(f"  {record_name}: peaks={len(peaks):,} | NSR={len(nsr):,} AFIB={len(afib):,}")
    return nsr, afib

records = ["04015", "04043", "04048", "04126", "04746", "04908"]
nsr_all, afib_all = np.array([]), np.array([])

for r in records:
    if not os.path.exists(f"{r}.hea"):
        print(f"מוריד {r}...")
        wfdb.dl_database("afdb", dl_dir=".", records=[r])
    nsr, afib = extract_rr(r)
    nsr_all  = np.append(nsr_all,  nsr)
    afib_all = np.append(afib_all, afib)
    if len(nsr_all) >= 1000 and len(afib_all) >= 1000:
        break

print(f"\nסה״כ: NSR={len(nsr_all):,}  AFIB={len(afib_all):,}")
print(f"NSR   mean={np.mean(nsr_all):.4f}  std={np.std(nsr_all):.4f}  skew={stats.skew(nsr_all):.4f}")
print(f"AFIB  mean={np.mean(afib_all):.4f}  std={np.std(afib_all):.4f}  skew={stats.skew(afib_all):.4f}")
print(f"יחס ממוצע: {np.mean(afib_all)/np.mean(nsr_all):.3f}x")

# --- בדיקת משפחה: InverseGaussian ---
from scipy.stats import invgauss
mu_hat = np.mean(nsr_all)
lam_hat = 1.0 / np.mean(1.0/nsr_all - 1.0/mu_hat)
ks_ig = stats.kstest(nsr_all,
    lambda x: invgauss.cdf(x, mu=mu_hat/lam_hat, scale=lam_hat))
a, _, scale = stats.gamma.fit(nsr_all, floc=0)
ks_g = stats.kstest(nsr_all,
    lambda x: stats.gamma.cdf(x, a, scale=scale))

print(f"\nKS InverseGaussian על NSR: p={ks_ig.pvalue:.4f} {'✅' if ks_ig.pvalue > 0.05 else '❌'}")
print(f"KS Gamma         על NSR: p={ks_g.pvalue:.4f}  {'✅' if ks_g.pvalue > 0.05 else '❌'}")

if len(nsr_all) < 100 or len(afib_all) < 100:
    print("\n❌ לא מספיק נתונים")
else:
    BATCH_SIZE = 100
    def to_batches(x, bs):
        n = len(x) // bs
        return x[:n*bs].reshape(n, bs)

    nsr_b  = to_batches(nsr_all,  BATCH_SIZE)
    afib_b = to_batches(afib_all, BATCH_SIZE)
    n      = min(len(nsr_b), len(afib_b), 200)
    nsr_b  = nsr_b[:n];  afib_b = afib_b[:n]
    all_b  = np.vstack([nsr_b, afib_b])
    labels = np.array([0]*n + [1]*n)
    print(f"אצוות: {n} NSR + {n} AFIB")

    detector = IGADDetector(family=InverseGaussianFamily)
    detector.fit(nsr_b.flatten())

    ref = nsr_b[0]
    igad_scores = [detector.score_batch(b) for b in all_b]
    wass_scores = [wasserstein_distance(ref, b) for b in all_b]
    mean_scores = [abs(np.mean(b)-np.mean(ref)) for b in all_b]
    var_scores  = [abs(np.var(b) -np.var(ref))  for b in all_b]

    results = {
        "IGAD (InvGaussian)": roc_auc_score(labels, igad_scores),
        "Wasserstein":        roc_auc_score(labels, wass_scores),
        "Mean shift":         roc_auc_score(labels, mean_scores),
        "Variance shift":     roc_auc_score(labels, var_scores),
    }

    print("\n" + "="*44)
    print(f"  {'שיטה':<24} {'AUC':>6}")
    print("="*44)
    for method, auc in sorted(results.items(), key=lambda x: -x[1]):
        tag = " ◄" if "IGAD" in method else ""
        print(f"  {method:<24} {auc:.4f}{tag}")
    print("="*44)
    gap = results["IGAD (InvGaussian)"] - results["Wasserstein"]
    print(f"\n  פער IGAD − Wasserstein: {gap:+.4f}")
    print(f"  יחס ממוצע NSR/AFIB:     {np.mean(afib_all)/np.mean(nsr_all):.3f}x")
    if gap > 0.03:
        print("  → IGAD מנצח ✅ — שינוי גיאומטרי ללא שינוי עוצמה")
    elif gap > -0.03:
        print("  → שניהם דומים")
    else:
        print("  → Wasserstein מנצח — יש שינוי עוצמה מובהק")
