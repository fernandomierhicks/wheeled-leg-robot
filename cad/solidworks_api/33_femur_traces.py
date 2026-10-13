"""Step 33: Femur -- raised grey/blue trace accents on the hip (AK45) end of the show face.

    cadenv/python 33_femur_traces.py [--dry] [--quick]

His call 2026-10-10: on the Femur's show face (z -5, the face the two hoses are on), at the
end with the 3 motor mounting holes, add raised graphite / blue traces.  3 mm keepout from the
mounting holes (nothing else needs one: the middle of the circular face is free).  Raised
accents are 0.6 mm (STANDING RULES, README).  ADDS material only: new bodies `RA_*` (not `RL_`:
28 build would delete them -- delete RA_* before re-running 28 build on Femur, re-run 33 after),
coloured, nothing merged, so no mate face changes.  Then 25 mirror-colours Femur + 16 chain.
`--dry` stops after the offline check and the preview PNG (out/relief/Femur_traces.png).
"""
import importlib
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
R = importlib.import_module("28_relief")
R._aes()
from shapely.affinity import rotate, scale, translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

A = (-93.79, 0.0)
R_DISC = 19.4
HOLES = [(-93.79, 13.5), (-105.48, -6.75), (-82.10, -6.75)]      # the 3 mounting holes (x, y), r 1.6 at the face
KEEP = 3.0                                                       # his keepout from a mounting hole's edge
H = 0.6                                                          # accent height (3 layers)
W = 1.2                                                          # trace width


def _cl(g, min_area=2.0, w=0.9):
    """Drop slivers and anything narrower than w (a 0.4 nozzle cannot print it)."""
    g = g.buffer(-w / 2, join_style=2).buffer(w / 2, join_style=2)
    return unary_union([p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and p.area >= min_area])


def flip(g):
    """The face's (u, v) is the part's (x, -y)."""
    return scale(g, 1, -1, origin=(0, 0))


def design(F):
    """-> {colour: shapely geometry in the part (x, y) plane}, already clear of everything."""
    white_xy = flip(F.footprint(kinds=["white"]))
    other_xy = flip(F.footprint(kinds=[k for k in R.KINDS if k != "white"]))
    disc = Point(*A).buffer(R_DISC - 1.2, 96)                      # stay 1.2 off the disc edge
    ground = white_xy.intersection(disc).difference(other_xy.buffer(0.8, join_style=2))
    keep = unary_union([Point(*h).buffer(1.6 + KEEP, 64) for h in HOLES] + [Point(*A).buffer(0.7 + 1.5, 32)])
    ground = ground.difference(keep)

    def polar(r, deg):
        return (A[0] + r * math.cos(math.radians(deg)), A[1] + r * math.sin(math.radians(deg)))

    blue, gfx = [], []
    for k, th in enumerate((30, 150, 270)):                        # three spokes between the holes
        s = 1 if k % 2 == 0 else -1
        pts = [polar(4.8, th), polar(9.5, th)]
        x, y = pts[-1]
        d = math.radians(th + s * 45)
        mid = (x + 2.6 * math.cos(d), y + 2.6 * math.sin(d))
        end = (mid[0] + 5.5 * math.cos(math.radians(th)), mid[1] + 5.5 * math.sin(math.radians(th)))
        blue.append(LineString(pts + [mid, end]).buffer(W / 2, cap_style=2, join_style=2, mitre_limit=2.0))
        ex, ey = end
        blue.append(Polygon([(ex - 1.3, ey), (ex, ey - 1.3), (ex + 1.3, ey), (ex, ey + 1.3)]))
        # raked blue comb ticks beside the spoke
        for j in range(3):
            c = polar(6.2 + 1.9 * j, th + s * 20)
            blue.append(rotate(box(c[0] - 0.5, c[1] - 2.0, c[0] + 0.5, c[1] + 2.0), th + s * 20 + 25, origin=c))
    # graphite raked plates in the three gaps between a spoke and a hole, outer ring
    for th in (60, 180, 300):
        base = Polygon([(10.5, -2.0), (16.2, -2.0), (17.4, 0.6), (17.4, 2.4), (10.5, 2.4)])
        gfx.append(translate(rotate(base, th, origin=(0, 0)), *A))
    blue_g = _cl(unary_union(blue).intersection(ground))
    gfx_g = _cl(unary_union(gfx).intersection(ground).difference(blue_g.buffer(0.8)))
    return {"blue": blue_g, "graphite": gfx_g}


def preview(F, parts, path):
    from PIL import Image, ImageDraw
    s = 14.0
    x0, y0, x1, y1 = A[0] - 24, -24, A[0] + 24, 24
    im = Image.new("RGB", (int((x1 - x0) * s), int((y1 - y0) * s)), (255, 255, 255))
    d = ImageDraw.Draw(im)
    T = lambda x, y: ((x - x0) * s, (y1 - y) * s)

    def poly(g, col):
        for p in getattr(g, "geoms", [g]):
            if p.geom_type == "Polygon" and not p.is_empty:
                d.polygon([T(*c) for c in p.exterior.coords], fill=col)
                for r in p.interiors:
                    d.polygon([T(*c) for c in r.coords], fill=(255, 255, 255))
    poly(flip(F.footprint(kinds=["white"])), (225, 228, 232))
    poly(flip(F.footprint(kinds=["graphite"])), (170, 174, 180))
    poly(flip(F.footprint(kinds=["blue"])), (170, 190, 235))
    d.ellipse([*T(A[0] - R_DISC, A[1] + R_DISC), *T(A[0] + R_DISC, A[1] - R_DISC)], outline=(0, 0, 0))
    for h in HOLES:
        d.ellipse([*T(h[0] - 1.6 - KEEP, h[1] + 1.6 + KEEP), *T(h[0] + 1.6 + KEEP, h[1] - 1.6 - KEEP)], outline=(220, 0, 0))
        d.ellipse([*T(h[0] - 1.6, h[1] + 1.6), *T(h[0] + 1.6, h[1] - 1.6)], fill=(0, 0, 0))
    poly(parts["graphite"], (64, 68, 76))
    poly(parts["blue"], (33, 115, 242))
    im.save(path)


def main():
    dry = "--dry" in sys.argv
    P = R.Part("Femur", built=True)
    F = R.Frame(P, (0, 0, -1), 5.0, u=(1, 0, 0), name="out")
    parts = design(F)
    os.makedirs(R.OUT, exist_ok=True)
    preview(F, parts, os.path.join(R.OUT, "Femur_traces.png"))
    ops = []
    for colour in ("blue", "graphite"):
        g = flip(parts[colour])                                      # back to the face's (u, v)
        for j, p in enumerate(getattr(g, "geoms", [g])):
            if p.geom_type == "Polygon" and p.area > 0:
                ops.append(R.raise_(F, p, H, colour, name=f"trace_{colour[0]}{j + 1}"))
    print(f"{len(ops)} raised shapes: blue {parts['blue'].area:.0f} mm2, graphite {parts['graphite'].area:.0f} mm2")
    state, cuts, log = R.apply(P, ops)
    new = [s for s in state if s["orig"] is None and s["solid"] is not None]
    for row in log:
        want = row["area"] * H
        print(f"  {row['name']:12s} {row['colour']:8s} {row['added']:7.2f} mm3 (prism {want:.2f})")
        if abs(row["added"] - want) > 0.02 * want + 0.05:
            raise SystemExit(f"{row['name']}: the body is not the full prism -- something stands on the face there")
    if any(cuts.values()):
        raise SystemExit("an accent op cut an existing body -- it must only add")
    if dry:
        return
    import swlib
    import swstyle as S
    import stepcolor
    from swlib import c, wrap, sld
    stepcolor.MIN_BODY_MM3 = 0.0
    step = os.path.join(R.OUT, "Femur_traces.step")
    stepcolor.write([(s["name"], s["solid"], tuple(int(round(x * 255)) for x in s["rgb"])) for s in new], step,
                    part="Femur_traces", say=lambda *a: None)
    sw, _ = swlib.connect()
    swlib.open_v5(sw)
    path = os.path.join(swlib.V5, R.PARTS["Femur"])
    doc = wrap(sw.OpenDoc6(path, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)[0], sld.IModelDoc2)
    sw.ActivateDoc3(doc.GetTitle(), False, 0, 0)
    old = [f for f in S._iter_features(doc) if f.Name.startswith("RA_")]
    if old:
        doc.ClearSelection2(True)
        for k, f in enumerate(old):
            f.Select2(k > 0, 0)
        doc.Extension.DeleteSelection2(c.swDelete_Absorbed | c.swDelete_Children)
        doc.EditRebuild3()
    n0 = len(S.bodies(doc))
    pdoc = wrap(doc, sld.IPartDoc)
    src = R._import_bodies(sw, doc, step)
    if len(src) != len(new):
        raise SystemExit(f"{len(src)} bodies in the STEP, expected {len(new)}")
    want = [(s["solid"].volume, s) for s in new]
    for v, body in sorted(src, key=lambda x: -x[0]):
        vv, s = min(want, key=lambda x: abs(x[0] - v))
        want = [w for w in want if w[1] is not s]
        names0 = {b.Name for b in S.bodies(doc)}
        f = wrap(pdoc.CreateFeatureFromBody3(body, False, c.swCreateFeatureBodyCheck), sld.IFeature)
        if f is None:
            raise SystemExit("SolidWorks refused an accent body")
        f.Name = f"RA_{s['name']}"
        nb = [b for b in S.bodies(doc) if b.Name not in names0]
        if len(nb) != 1 or abs(S.volume(nb[0]) - vv) > max(0.05, 1e-3 * vv):
            raise SystemExit(f"{f.Name}: {len(nb)} bodies, {S.volume(nb[0]) if nb else 0:.3f} vs {vv:.3f} mm3")
        S.colour(nb[0], list(s["rgb"]))
    doc.EditRebuild3()
    print(f"  bodies {n0} -> {len(S.bodies(doc))}; {len(new)} accents added, nothing removed")
    importlib.import_module("25_fasteners").mirror_colours(only=["Femur"])
    importlib.import_module("16_check_and_save").chain(
        sw, [R.PARTS["Femur"], r"Links\MirrorFemur.SLDPRT"], quick="--quick" in sys.argv)


if __name__ == "__main__":
    main()
