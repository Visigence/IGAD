import wfdb
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ann = wfdb.rdann("04015", "atr")

# הדפס את כל הערכים הייחודיים
unique = list(set(ann.aux_note))
print("כל תגיות הקצב בקובץ:")
for u in unique:
    count = ann.aux_note.count(u)
    print(f"  '{repr(u)}' — {count} פעמים")

print(f"\nסה״כ annotations: {len(ann.aux_note)}")
print(f"\n10 ראשונות:")
for i in range(min(10, len(ann.aux_note))):
    print(f"  [{i}] '{repr(ann.aux_note[i])}'")
