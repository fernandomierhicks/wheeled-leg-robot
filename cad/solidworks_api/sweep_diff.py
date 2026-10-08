"""Compare two 06_sweep.py CSVs: interfering pairs new / gone / changed at any pose (> TOL mm3).

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/sweep_diff.py out/sweep_wheel_1deg.csv out/sweep_1deg.csv

06 overwrites out/sweep_1deg.csv every run: copy it to a named file first.  "0 new,
0 gone, 0 changed" is the pass for a change that should not touch anything that moves.
"""
import csv
import sys

TOL = 0.01


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head, data = rows[0][1:], rows[1:]
    hips = [float(r[0]) for r in data]
    cols = {h: [float(r[i + 1]) for r in data] for i, h in enumerate(head)}
    return hips, cols


h0, a = load(sys.argv[1])
h1, b = load(sys.argv[2])
assert h0 == h1, "different pose lists"
new = [k for k in b if k not in a and max(b[k]) > TOL]
gone = [k for k in a if k not in b and max(a[k]) > TOL]
changed = []
for k in set(a) & set(b):
    d = [abs(x - y) for x, y in zip(a[k], b[k])]
    if max(d) > TOL:
        i = d.index(max(d))
        changed.append((k, h0[i], a[k][i], b[k][i]))
print(f"poses {len(h0)}; pairs old {len(a)}, new {len(b)}")
print(f"NEW ({len(new)}):" + "".join(f"\n  {k}: max {max(b[k]):.3f} mm3 at hip {h1[b[k].index(max(b[k]))]:+.0f}" for k in new))
print(f"GONE ({len(gone)}):" + "".join(f"\n  {k}: was max {max(a[k]):.3f}" for k in gone))
print(f"CHANGED ({len(changed)}):" + "".join(f"\n  {k}: hip {h:+.0f} {x:.3f} -> {y:.3f}" for k, h, x, y in changed))
