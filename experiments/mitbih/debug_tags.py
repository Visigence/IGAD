import wfdb, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

ann = wfdb.rdann("04015", "atr")

print("bytes של כל תגית ייחודית:")
seen = set()
for note in ann.aux_note:
    if note not in seen:
        seen.add(note)
        print(f"  repr: {repr(note)}")
        print(f"  bytes: {[hex(b) for b in note.encode('utf-8')]}")
        print()
