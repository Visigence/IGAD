"""
ECG Experiment v7 — Rate-Matched AFib
Window: RR in [0.55, 0.80]s — mean ratio NSR/AFIB ~ 0.97 (near 1.0)
This is the shape-shift regime: rate is matched, only irregularity differs.
"""
import numpy as np
import wfdb
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance
from igad.detector import IGADDetector
from igad.families import InverseGaussianFamily

RECORDS   = ['experiments/mitbih/04015', 'experiments/mitbih/04043']
WINDOW    = (0.55, 0.80)
BATCH_N   = 100

# ── 1. Load RR intervals with rhythm labels ──────────────────────────────────
def load_rr_by_rhythm(record):
    qrs = wfdb.rdann(record, 'qrs')
    atr = wfdb.rdann(record, 'atr')
    _, fields = wfdb.rdsamp(record)
    fs = fields['fs']
    rhythm_changes = sorted([
        (atr.sample[i], atr.aux_note[i].strip('\x00').strip()[1:])
        for i in range(len(atr.sample))
        if atr.aux_note[i].strip('\x00').strip().startswith('(')
    ])
    peaks  = qrs.sample
    labels = []
    for p in peaks:
        label = 'UNKNOWN'
        for start, rhy in rhythm_changes:
            if p >= start:
                label = rhy
            else:
                break
        labels.append(label)
    rr     = np.diff(peaks) / fs
    labels = np.array(labels[:-1])
    return rr[labels == 'N'], rr[labels == 'AFIB']

nsr_all, afib_all = [], []
for rec in RECORDS:
    n, a = load_rr_by_rhythm(rec)
    nsr_all.append(n); afib_all.append(a)

nsr_raw  = np.concatenate(nsr_all)
afib_raw = np.concatenate(afib_all)

# ── 2. Apply rate window ─────────────────────────────────────────────────────
lo, hi  = WINDOW
nsr_w   = nsr_raw[(nsr_raw   > lo) & (nsr_raw   < hi)]
afib_w  = afib_raw[(afib_raw > lo) & (afib_raw  < hi)]

print(f"Rate-matched window {WINDOW}")
print(f"NSR   n={len(nsr_w):,}  mean={nsr_w.mean():.4f}  std={nsr_w.std():.4f}  skew={float(__import__('scipy.stats',fromlist=['skew']).skew(nsr_w)):.4f}")
print(f"AFIB  n={len(afib_w):,}  mean={afib_w.mean():.4f}  std={afib_w.std():.4f}  skew={float(__import__('scipy.stats',fromlist=['skew']).skew(afib_w)):.4f}")
print(f"Mean ratio NSR/AFIB: {nsr_w.mean()/afib_w.mean():.3f}")

# ── 3. Batch scoring ─────────────────────────────────────────────────────────
rng     = np.random.default_rng(42)
n_batch = min(len(nsr_w), len(afib_w)) // BATCH_N

nsr_batches  = [rng.choice(nsr_w,  BATCH_N, replace=False) for _ in range(n_batch)]
afib_batches = [rng.choice(afib_w, BATCH_N, replace=False) for _ in range(n_batch)]

ref_data = rng.choice(nsr_w, 500, replace=False)

detector = IGADDetector(InverseGaussianFamily()).fit(ref_data)

def score_batch_all(batch, ref):
    igad  = detector.score_batch(batch)
    wass  = wasserstein_distance(batch, ref)
    mshift = abs(batch.mean() - ref.mean())
    vshift = abs(batch.std()  - ref.std())
    return igad, wass, mshift, vshift

ref_sample = rng.choice(nsr_w, BATCH_N, replace=False)

scores = {'igad':[], 'wass':[], 'mean':[], 'var':[], 'label':[]}
for b in nsr_batches:
    ig, w, m, v = score_batch_all(b, ref_sample)
    scores['igad'].append(ig); scores['wass'].append(w)
    scores['mean'].append(m);  scores['var'].append(v)
    scores['label'].append(0)
for b in afib_batches:
    ig, w, m, v = score_batch_all(b, ref_sample)
    scores['igad'].append(ig); scores['wass'].append(w)
    scores['mean'].append(m);  scores['var'].append(v)
    scores['label'].append(1)

y = scores['label']
print(f"\n{'='*44}")
print(f"  {'Method':<24}  AUC")
print(f"{'='*44}")
for name, key in [('IGAD (InvGaussian)', 'igad'),
                  ('Wasserstein',        'wass'),
                  ('Mean shift',         'mean'),
                  ('Variance shift',     'var')]:
    auc = roc_auc_score(y, scores[key])
    marker = ' ◄' if key == 'igad' else ''
    print(f"  {name:<24}  {auc:.4f}{marker}")
print(f"{'='*44}")
