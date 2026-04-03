import numpy as np
import wfdb
from scipy import stats, signal as sp_signal
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from igad import IGADDetector
from igad.families import GammaFamily
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- טען אות גולמי + אנוטציות קצב ---
rec = wfdb.rdrecord("04015")
ann = wfdb.rdann("04015", "atr")
fs  = rec.fs  # 250 Hz
ecg = rec.p_signal[:, 0]  # ערוץ 1

print(f"fs={fs}Hz | אורך אות: {len(ecg):,} דגימות ({len(ecg)/fs/60:.1f} דקות)")

# --- זיהוי R-peaks בסיסי ---
# סף: שיאים מעל 0.5 * max, מרחק מינימלי 0.4 שניות
min_dist = int(0.4 * fs)
height   = 0.5 * np.max(ecg)
peaks, _ = sp_signal.find_peaks(ecg, height=height, distance=min_dist)
print(f"R-peaks שזוהו: {len(peaks):,}")

# --- חשב RR intervals ---
rr_samples = np.diff(peaks)
rr_sec     = rr_samples / fs

# --- מפה כל peak לקצב לפי אנוטציות ---
# ann.sample = נקודות שינוי קצב
# ann.aux_note = הקצב מאותה נקודה והלאה
rhythm_at_peak = []
for p in peaks[1:]:  # [1:] כי rr מתחיל מהפרש
    # מצא את אנוטציית הקצב האחרונה לפני peak זה
    idx = np.searchsorted(ann.sample, p, side='right') - 1
    if idx >= 0 and idx < len(ann.aux_note):
        rhythm_at_peak.append(ann.aux_note[idx])
    else:
        rhythm_at_peak.append("")

rhythm_at_peak = np.array(rhythm_at_peak)

# --- סנן לפי קצב ---
mask_nsr  = (rhythm_at_peak == "(N")  & (rr_sec > 0.3) & (rr_sec < 2.0)
mask_afib = (rhythm_at_peak == "(AFIB") & (rr_sec > 0.3) & (rr_sec < 2.0)

nsr_rr  = rr_sec[mask_nsr]
afib_rr = rr_sec[mask_afib]

print(f"\nקצבים שזוהו: {set(rhythm_at_peak)}")
print(f"NSR  intervals: {len(nsr_rr):,}")
print(f"AFIB intervals: {len(afib_rr):,}")

if len(nsr_rr) < 50 or len(afib_rr) < 50:
    print("\n⚠️  נתונים לא מספיקים — מוסיף רשומות")
    for record in ["04043", "04048", "04126", "04746", "04908"]:
        if not os.path.exists(f"{record}.hea"):
            print(f"  מוריד {record}...")
            wfdb.dl_database("afdb", dl_dir=".", records=[record])
        rec2 = wfdb.rdrecord(record)
        ann2 = wfdb.rdann(record, "atr")
        fs2  = rec2.fs
        ecg2 = rec2.p_signal[:, 0]
        h2   = 0.5 * np.max(ecg2)
        p2, _ = sp_signal.find_peaks(ecg2, height=h2, distance=int(0.4*fs2))
        rr2   = np.diff(p2) / fs2
        r2    = []
        for p in p2[1:]:
            idx = np.searchsorted(ann2.sample, p, side='right') - 1
            r2.append(ann2.aux_note[idx] if 0 <= idx < len(ann2.aux_note) else "")
        r2 = np.array(r2)
        nsr_rr  = np.append(nsr_rr,  rr2[(r2=="(N")   &(rr2>0.3)&(rr2<2.0)])
        afib_rr = np.append(afib_rr, rr2[(r2=="(AFIB")&(rr2>0.3)&(rr2<2.0)])
        print(f"  {record}: NSR={len(nsr_rr):,}  AFIB={len(afib_rr):,}")
        if len(nsr_rr) >= 500 and len(afib_rr) >= 500:
            break

print(f"\nסה״כ: NSR={len(nsr_rr):,}  AFIB={len(afib_rr):,}")
print(f"NSR   mean={np.mean(nsr_rr):.4f}  std={np.std(nsr_rr):.4f}  skew={stats.skew(nsr_rr):.4f}")
print(f"AFIB  mean={np.mean(afib_rr):.4f}  std={np.std(afib_rr):.4f}  skew={stats.skew(afib_rr):.4f}")
print(f"יחס ממוצע: {np.mean(afib_rr)/np.mean(nsr_rr):.3f}x")

if len(nsr_rr) < 100 or len(afib_rr) < 100:
    print("\n❌ עדיין לא מספיק נתונים — עצור כאן")
else:
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
    nsr_b  = nsr_b[:n];  afib_b = afib_b[:n]
    all_b  = np.vstack([nsr_b, afib_b])
    labels = np.array([0]*n + [1]*n)
    print(f"אצוות: {n} NSR + {n} AFIB")

    detector = IGADDetector(family=GammaFamily)
    detector.fit(nsr_b.flatten())

    ref = nsr_b[0]
    igad_scores = [detector.score_batch(b) for b in all_b]
    wass_scores = [wasserstein_distance(ref, b) for b in all_b]
    mean_scores = [abs(np.mean(b)-np.mean(ref)) for b in all_b]
    var_scores  = [abs(np.var(b) -np.var(ref))  for b in all_b]

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
        print("  → Wasserstein מנצח")
