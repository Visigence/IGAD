import numpy as np
from scipy import io, stats
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from igad import IGADDetector
from igad.families import GammaFamily
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

BATCH_SIZE = 200

def load_signal(path):
    mat = io.loadmat(path)
    keys = [k for k in mat.keys() if k.startswith('X') and 'DE' in k]
    return np.abs(mat[keys[0]].flatten())

normal = load_signal("normal_0.mat")
faulty = load_signal("inner_race_007.mat")

# בדוק: האם ממוצע ושונות שונים?
print("=== השוואת סטטיסטיקות ===")
print(f"normal  mean={np.mean(normal):.6f}  std={np.std(normal):.6f}")
print(f"faulty  mean={np.mean(faulty):.6f}  std={np.std(faulty):.6f}")
print(f"יחס ממוצע:  {np.mean(faulty)/np.mean(normal):.2f}x")
print(f"יחס std:    {np.std(faulty)/np.std(normal):.2f}x")

# בנה אצוות עם ממוצע זהה מלאכותית (נרמול)
print("\n=== ניסוי עם נרמול — מחק הבדלי ממוצע/שונות ===")

def normalize_batch(b):
    return (b - np.mean(b)) / np.std(b)  # z-score

normal_batches = normal[:606*BATCH_SIZE].reshape(606, BATCH_SIZE)
faulty_batches = faulty[:606*BATCH_SIZE].reshape(606, BATCH_SIZE)

n = 200
normal_norm = np.array([normalize_batch(b) for b in normal_batches[:n]])
faulty_norm = np.array([normalize_batch(b) for b in faulty_batches[:n]])

# אחרי נרמול — ממוצע ושונות זהים לכולם
# רק צורה נשארת שונה
all_norm = np.vstack([normal_norm, faulty_norm])
labels   = np.array([0]*n + [1]*n)

# חייבים להחזיר לערכים חיוביים לגאמא
all_pos = np.abs(all_norm) + 0.001

detector = IGADDetector(family=GammaFamily)
detector.fit(np.abs(normal_norm).flatten() + 0.001)

ref = all_pos[0]
igad_scores = [detector.score_batch(b) for b in all_pos]
wass_scores = [wasserstein_distance(ref, b) for b in all_pos]

igad_auc = roc_auc_score(labels, igad_scores)
wass_auc = roc_auc_score(labels, wass_scores)

print(f"\nאחרי נרמול (ממוצע ושונות מנוטרלים):")
print("="*40)
print(f"  IGAD (curvature)  {igad_auc:.4f} ◄")
print(f"  Wasserstein       {wass_auc:.4f}")
print("="*40)
print(f"  פער: {igad_auc - wass_auc:+.4f}")
if igad_auc > wass_auc:
    print("  → IGAD מנצח בדיוק כשצריך ✅")
else:
    print("  → שניהם דומים — הצורה לא שונה מספיק")
