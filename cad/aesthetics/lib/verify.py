r"""Compare a styled part against its source, opening by opening.

    C:/Users/ferna/cadenv/Scripts/python.exe lib/verify.py Tibia [tag]

The invariant that keeps parts printable: material is only ever ADDED outside the
original silhouette, and the source solid is never re-cut.  If that holds, every
opening -- including counterbores and blind pockets -- is bit-identical, and this
reports "0 changed, worst deviation 0.000 mm3".
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from build123d import import_step
from keepout import openings
import manifold
from shputil import prism
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder


def vertical_cylinders(shape):
    out = set()
    for f in shape.faces():
        a = BRepAdaptor_Surface(f.wrapped)
        if a.GetType() != GeomAbs_Cylinder:
            continue
        cyl = a.Cylinder(); ax = cyl.Axis(); p = ax.Location()
        if abs(ax.Direction().Z()) > 0.99:
            out.add((round(cyl.Radius() * 2, 2), round(p.X(), 2), round(p.Y(), 2)))
    return out


def _load(path):
    """The WHOLE compound, never `solids()[0]`.

    A styled body is routinely SEVERAL solids, and taking the first silently
    measures a fragment.  That is trap 7, and it was live in this file: the Side
    panel read as 0.39 cm3 against a 92.01 cm3 source and reported nine openings
    "changed", purely because the styled STEP's first solid is a 27 x 15 x 4 mm
    offcut.  Volume on a Compound sums its solids, and the boolean intersections
    below work on compounds, so nothing else has to change.
    """
    c = import_step(path)
    n = len(c.solids())
    if n > 1:
        print(f"  (styled body is {n} solids; measuring all of them)")
    return c


def verify(part="Tibia", tag="arctic"):
    src = _load(paths.part_step(part))
    sty_file = paths.styled_step(part, tag)
    if not os.path.exists(sty_file):
        raise SystemExit(f"no styled part at {sty_file} -- run parts/{part.lower()}.py first")
    sty = _load(sty_file)
    bb, sb = src.bounding_box(), sty.bounding_box()
    print(f"=== {part} : styled vs source ===")
    print(f"source  {src.volume/1000:7.2f} cm3   {bb.size.X:6.1f} x {bb.size.Y:5.1f} x {bb.size.Z:5.1f}")
    print(f"styled  {sty.volume/1000:7.2f} cm3   {sb.size.X:6.1f} x {sb.size.Y:5.1f} x {sb.size.Z:5.1f}")
    print(f"        grew {sb.size.X-bb.size.X:+.1f} X, {sb.size.Y-bb.size.Y:+.1f} Y, "
          f"{sb.size.Z-bb.size.Z:+.1f} Z\n")

    # The stated invariant is that material is only ever ADDED outside the
    # original silhouette.  The opening test below does not cover it: the 4.4 mm
    # chamfer bevels the OUTER edge, and while growth exceeds the chamfer it
    # lands on added material and nothing is lost.  Once growth is clipped below
    # the chamfer -- which the assembly keep-out does -- the same bevel starts
    # eating the source, and every hole can still verify perfectly while the
    # part quietly loses mass.  Measure it directly.
    # Measured by INTERSECTION, not difference.  `src - sty` on multi-solid
    # compounds returns the whole source often enough to be useless -- it
    # reported "100.00% removed" for the Tibia and RobotMount while every one of
    # their openings verified clean.  A metric that falsely reads 100% is worse
    # than no metric, because it trains you to ignore it.
    keep = src & sty
    ev = src.volume - (0.0 if keep is None else keep.volume)
    if ev < -1.0 or ev > src.volume + 1.0:
        ev = float("nan")
    # This is TOTAL removal and most of it is deliberate: pockets, windows,
    # full-depth trapezoid cutouts, side-wall pockets and the accent engraving
    # are the approved look, not a defect.  It is reported for scale, not as a
    # pass/fail -- the removal that IS a defect is the envelope cutting the
    # source, and build() measures that one directly where `env` is in scope.
    if ev != ev:            # NaN: the boolean did not give a usable answer
        print("source material removed: (boolean unreliable on this compound)")
    else:
        print(f"source material removed: {ev:9.1f} mm3 "
              f"({100 * ev / src.volume:.2f}%)  -- pockets, cutouts, engraving "
              f"and chamfer combined; see build() for the envelope-only figure")
    print()

    ops = openings(src, bb.max.Z, bb.min.Z)
    worst, bad = 0.0, 0
    for p in sorted(ops, key=lambda p: -p.area):
        core = p.buffer(-0.35)
        if core.is_empty:
            continue
        pr = prism(core, bb.min.Z - 12, bb.max.Z + 40)
        if pr is None:
            continue
        a, b = (src & pr), (sty & pr)
        va = 0.0 if a is None else a.volume
        vb = 0.0 if b is None else b.volume
        worst = max(worst, abs(vb - va))
        if abs(vb - va) > 1.0:
            bad += 1
            c = p.centroid
            print(f"   CHANGED area={p.area:7.1f} at ({c.x:7.2f},{c.y:7.2f}): "
                  f"{va:8.1f} -> {vb:8.1f} mm3")
    print(f"{len(ops)} openings: {bad} changed, worst deviation {worst:.3f} mm3\n")

    cs, ct = vertical_cylinders(src), vertical_cylinders(sty)
    miss = sorted(cs - ct)
    print(f"vertical cylindrical faces: source {len(cs)}, styled {len(ct)}")
    print(f"bores missing from styled : {len(miss)}")
    for d, x, y in miss:
        print(f"    D{d:7.2f} at ({x:8.2f},{y:8.2f})")

    # THE TOPOLOGY GATE.  This file passing has been insufficient THREE times,
    # because it only ever compared VOLUMES through openings and never once
    # looked at topology.  A body can match the source hole for hole, to 0.000
    # mm3, and still be non-manifold (Parasolid shreds it), be five disconnected
    # solids (his "no floating bodies"), or contain a sealed bubble (unprintable).
    # All three shipped.  Checked on the FUSED part, which must be one solid, and
    # on each filament body, which may legitimately be several.
    print()
    ok = manifold.gate_report(sty.wrapped, f"{part} fused", one_solid=True)
    pdir = paths.print_dir(part, tag)
    for nm in ("white", "graphite", "accent"):
        f = os.path.join(pdir, f"{part}_{tag}_{nm}.step")
        if os.path.exists(f):
            ok &= manifold.gate_report(import_step(f).wrapped, f"{part} {nm}",
                                       one_solid=False)
    print(f"\ntopology gate: {'PASS' if ok else 'FAIL'}")
    return bad, worst, ok


if __name__ == "__main__":
    # lib/verify.py <Part> [tag]   -- tag defaults to whatever specs/<part>.json
    # was built under, so the locked concept verifies without extra arguments
    part = sys.argv[1] if len(sys.argv) > 1 else "Tibia"
    if len(sys.argv) > 2:
        tag = sys.argv[2]
    else:
        import json
        sf = os.path.join(paths.SPECS, f"{part.lower()}.json")
        tag = (json.load(open(sf)).get("tag", "arctic")
               if os.path.exists(sf) else "arctic")
    bad, worst, ok = verify(part, tag=tag)
    sys.exit(0 if (ok and bad == 0) else 1)
