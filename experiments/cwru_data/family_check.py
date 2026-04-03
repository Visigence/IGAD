import numpy as np
from scipy import io, stats
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- טעינה ---
def load_signal(path):
    mat = io.loadmat(path)
    keys = [k for k in mat.keys() if k.startswith('X') and 'DE' in k]
    print(f"  מפתח: {keys[0]}")
    return mat[keys[0]].flatten()

print("טוען normal_0.mat...")
normal = load_signal("normal_0.mat")
print(f"  אורך אות: {len(normal):,} דגימות\n")

# --- אמפליטודה של אצווה ראשונה ---
batch = np.abs(normal[:200])

# --- KS-Test ---
a, _, scale  = stats.gamma.fit(batch, floc=0)
s, _, scale2 = stats.lognorm.fit(batch, floc=0)

ks_g = stats.kstest(batch, lambda x: stats.gamma.cdf(x, a, scale=scale))
ks_l = stats.kstest(batch, lambda x: stats.lognorm.cdf(x, s, scale=scale2))

print("=" * 45)
print(f"  גאמא      p={ks_g.pvalue:.4f}  {'✅ לא נדחית' if ks_g.pvalue > 0.05 else '❌ נדחית'}")
print(f"  לוגנורמל  p={ks_l.pvalue:.4f}  {'✅ לא נדחית' if ks_l.pvalue > 0.05 else '❌ נדחית'}")
print(f"  shape גאמא: α={a:.4f}")
print("=" * 45)

if ks_g.pvalue > 0.05:
    print("\n→ המשפחה: GammaFamily ✅ — ממשיכים")
elif ks_l.pvalue > 0.05:
    print("\n→ המשפחה: LogNormal — IGAD לא תומך עדיין")
else:
    print("\n→ שתיהן נדחו — IGAD לא מתאים לנתונים אלו")
