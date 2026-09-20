r"""Show ONLY what the styling added, against the untouched source.

The whole "paint on a pig" problem is invisible in a normal render because the
added material is the same colour as the part.  Here the source is neutral grey
and everything ADDED is red, so a thin fin standing off the original is obvious
at a glance -- and so is whether it reads as structure.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import paths
import render_color as RC
from build123d import import_step
from render3d import tessellate

SRC = (120, 126, 136)
ADD = (214, 69, 69)
VIEWS = [("end-on  (down X)", 0, 2), ("end-on  (down Y)", 0, 92),
         ("three-quarter", 62, -58), ("show face", 88, -90)]

if __name__ == "__main__":
    TAG = "glacier"
    if "--tag" in sys.argv:
        i = sys.argv.index("--tag"); TAG = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    paths.ensure_out()
    for part in (sys.argv[1:] or ["Femur"]):
        src = import_step(paths.part_step(part))
        sty = import_step(paths.styled_step(part, TAG))
        added = sty - src
        kept = sty & src
        av = 0.0 if added is None else added.volume
        kv = 0.0 if kept is None else kept.volume
        print(f"{part}: source {src.volume/1000:.2f} cm3 | styled "
              f"{sty.volume/1000:.2f} | ADDED {av/1000:.2f} | source kept "
              f"{kv/1000:.2f} ({100*kv/src.volume:.1f}%)")
        bodies, allV = [], []
        for shp, col in ((kept, SRC), (added, ADD)):
            if shp is None or shp.volume < 1:
                continue
            V, T, _ = tessellate(shp, 0.12)
            if len(T):
                bodies.append((V, T, col)); allV.append(V)
        if not bodies:
            print(f"  {part}: nothing to draw"); continue
        allV = np.vstack(allV)
        tiles = [(nm, "", RC.view(bodies, allV, (820, 700), elev=e, azim=a))
                 for nm, e, a in VIEWS]
        out = os.path.join(paths.RENDERS, f"added_{part.replace(' ', '_')}_{TAG}.png")
        RC.sheet(tiles, 2, out, header=f"{part} - what the styling ADDED",
                 sub="grey = untouched source   |   RED = added material. "
                     "Red standing alone as a thin wall is the failure mode.",
                 size=(820, 700))
        print(f"  wrote {out}")
