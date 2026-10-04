"""Step 6 of the SolidWorks automation ladder: collisions over the WHOLE hip stroke.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/06_sweep.py [step_deg]

Walks the hip from the retracted stop (-28) to the extended stop (+57) in
`step_deg` steps (default 1.0 -> 86 poses, ~4.5 min) and runs Interference
Detection at every pose, with each sub-assembly treated as one block -- so
screws inside their own part are not reported, only things that can move
relative to each other.

Then it sorts every interfering pair into:
  * FIT       -- same volume at every pose: a bearing in its seat, a screw
                 across a bolted joint.  Designed, not a collision.
  * COLLISION -- appears at only some poses, or changes volume with the hip.
Writes every pose x pair volume to out/sweep_<step>deg.csv.
"""
import os
import sys
import csv
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib
from importlib import import_module
interferences = import_module("05_interference").interferences

LO, HI = -28.0, 57.0
FIT_TOL = 0.01          # mm3; a fit's volume may wobble this much between poses


def main(step):
    sw, _ = swlib.connect()
    model = swlib.open_v5(sw)
    hip = swlib.HipDriver(model)
    hip0 = hip.hip()

    n = int(round((HI - LO) / step)) + 1
    angles = np.linspace(LO, HI, n)
    vol = {}                    # (pair) -> {pose index: mm3}
    got_angles = []
    t0 = time.time()
    for i, a in enumerate(angles):
        got = hip.set(float(a))
        got_angles.append(got)
        if abs(got - a) > 0.01:
            print(f"  !! hip {a:+.2f} commanded, {got:+.2f} reached")
        hits, _ = interferences(model, subassemblies=True)
        for v, names, fast, poss in hits:
            key = tuple(sorted(names))
            vol.setdefault(key, {})
            vol[key][i] = vol[key].get(i, 0.0) + v       # a pair can touch in several places
        if i % 10 == 0 or i == n - 1:
            el = time.time() - t0
            print(f"  pose {i + 1:3d}/{n}  hip {got:+7.2f}  {len(hits):3d} interferences  "
                  f"{el:6.0f} s elapsed, ~{el / (i + 1) * (n - i - 1):4.0f} s left", flush=True)
    hip.set(hip0)

    fits, coll = [], []
    for key, d in vol.items():
        v = np.array([d.get(i, 0.0) for i in range(n)])
        if len(d) == n and v.max() - v.min() <= FIT_TOL:
            fits.append((key, v.mean()))
        else:
            coll.append((key, v))

    print(f"\n{n} poses, {time.time() - t0:.0f} s total, {len(vol)} interfering pair(s)")
    print(f"\nFITS -- constant over the whole stroke ({len(fits)}):")
    for key, m in sorted(fits, key=lambda f: -f[1]):
        print(f"  {m:9.3f} mm3  {'  x  '.join(key)}")
    print(f"\nCOLLISIONS -- appear or change with the hip ({len(coll)}):")
    for key, v in sorted(coll, key=lambda c: -c[1].max()):
        on = np.nonzero(v > 0)[0]
        print(f"  max {v.max():9.3f} mm3 at hip {angles[v.argmax()]:+6.1f}, present "
              f"{angles[on[0]]:+6.1f}..{angles[on[-1]]:+6.1f} ({len(on)}/{n} poses)  "
              f"{'  x  '.join(key)}")

    out = os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"sweep_{step:g}deg.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = sorted(vol)
        w.writerow(["hip_deg"] + [" x ".join(k) for k in keys])
        for i, a in enumerate(got_angles):
            w.writerow([f"{a:.3f}"] + [f"{vol[k].get(i, 0.0):.4f}" for k in keys])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
