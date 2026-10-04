"""Step 15: the green retaining washers in GLACIER colours -- colour only.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/15_style_washers.py preview
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/15_style_washers.py [--restyle]
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/15_style_washers.py render

The three green parts that clamp the 6804 bearings and pins:

  BearingWasher       Links  44 OD, 26 bore, 2 mm plate + 1.5 mm lip on the
                             outer race; 6 M3 on r 19.  COUPLER-1 at the
                             coupler-body pivot F, screw heads outboard (+Z).
  SmallBearingWahser  Body   23 OD, 2 mm, solid centre, 4 M3 on r 7.  TWICE,
                             facing opposite ways: inside the BearingWasher's
                             bore at F (part -Y face outboard) and capping the
                             coupler-tibia pin at E (part +Y face outboard) --
                             so each face gets its own pattern.
  InsideFemurShaft    Body   29 flange + 20.05 shaft on the hip axis, inside
                             the body box; end face (part +Y) inboard.

The look is the links' own GLACIER circuit motif shrunk onto a disc: the part
turns GRAPHITE, with BLUE traces (45-degree doglegs, tick marks, a little
"chip" on each small washer's centre) and WHITE pads and vias, each face laid
out differently.  Everything stays off every screw-head footprint (M3 button
head r 2.85 + 0.5 mm) and inside the flat of each face (the 0.5 mm edge
fillets would otherwise be filled in by the inlay prism).

Nothing about the SHAPE changes: each colour region is cut out of the
original body and the very same sketch is extruded back into the groove as a
new body (the 07 multi-colour recipe).  The original body stays the one the
mates hold (gotcha 30); the bodies' union is the original solid, so no
collision or fit can change.  Checked here: volume partition exact, box
unchanged.  Saves each part; originals backed up to v5/_originals.

`preview` draws every face to out/renders/washer_patterns.png (no SolidWorks).
"""
import math
import os
import shutil
import sys
from importlib import import_module

from shapely.geometry import LineString, Point, Polygon, box as sbox
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib
import swstyle as S
from swlib import c, wrap, sld

ORIGINALS = os.path.join(swlib.V5, "_originals")
COLOURS = {"graphite": S.GROUPS["graphite"], "blue": S.GROUPS["blue"], "white": S.GROUPS["white"]}
W = 1.0        # mm, trace width (two 0.4 mm lines and a bit)
HEAD = 2.85    # mm, M3 button head radius (ISO 7380)
CLEAR = 0.5    # mm, kept free round a head footprint


# --- pattern vocabulary (sketch coords on the Top Plane: (x, y) = part (X, -Z)) --
def P(r, deg):
    a = math.radians(deg)
    return (r * math.cos(a), r * math.sin(a))


def trace(pts, w=W):
    """An angular trace through polar points (r, deg), flat ends, mitred corners."""
    return LineString([P(*p) for p in pts]).buffer(w / 2, cap_style=2, join_style=2,
                                                   mitre_limit=2.0)


def pad(r, deg, rad=0.8):
    return Point(P(r, deg)).buffer(rad, quad_segs=16)


def ticks(along_deg, radii, half=1.2, w=0.8):
    """Short bars across a radial trace at `along_deg`, one per radius."""
    out = []
    for r in radii:
        x, y = P(r, along_deg)
        dx, dy = P(half, along_deg + 90)
        out.append(LineString([(x - dx, y - dy), (x + dx, y + dy)]).buffer(w / 2, cap_style=2))
    return unary_union(out)


def tangent_ticks(r0, r1, degs, w=0.9):
    """Short radial bars crossing a tangential trace, one per angle."""
    return unary_union([LineString([P(r0, d), P(r1, d)]).buffer(w / 2, cap_style=2) for d in degs])


def chip(half, rot=0.0):
    sq = sbox(-half, -half, half, half)
    if rot:
        from shapely import affinity
        sq = affinity.rotate(sq, rot, origin=(0, 0))
    return sq


class Face:
    """One face's pattern.  `cut` overshoots the free face by 0.5 mm so it never
    ends on it; `inlay` is exactly the material removed."""

    def __init__(self, name, cut, inlay, holes, r_in, r_out, blue, white, top=(), bore=0.0):
        """White pads sit on top of blue traces; `top` is blue drawn over white
        (a bus across a white panel)."""
        self.name, self.cut, self.inlay = name, cut, inlay
        self.holes, self.bore = holes, bore           # holes: [(x, y, r_hole, r_keep)]
        allowed = Point(0, 0).buffer(r_out, quad_segs=64)
        if r_in > 0:
            allowed = allowed.difference(Point(0, 0).buffer(r_in, quad_segs=64))
        allowed = allowed.difference(unary_union([Point(x, y).buffer(k, quad_segs=32)
                                                  for x, y, _, k in holes]))
        self.allowed, self.r_out = allowed, r_out
        top = unary_union(list(top))
        white = unary_union(white).difference(top).intersection(allowed)
        blue = unary_union(blue).difference(white).union(top).intersection(allowed)
        self.blue = unary_union(S.polys(blue, 0.2))
        self.white = unary_union(S.polys(white, 0.2))


def ring_of(r, degs, r_hole, r_keep):
    return [P(r, d) + (r_hole, r_keep) for d in degs]


def faces():
    bw_holes = ring_of(19.02, range(30, 360, 60), 1.6, HEAD + CLEAR)
    sw_holes = ring_of(7.0, (0, 90, 180, 270), 1.55, HEAD + CLEAR)
    sh_holes = (ring_of(7.0, (0, 90, 180, 270), 2.8, 3.2)       # head counterbores
                + ring_of(7.0, (45, 135, 225, 315), 1.4, 1.8))  # tapped, M3 from the washer

    # BearingWasher, free face y=-2 (flat r 13.5..21.525): traces leave the
    # inner band (under no head) and dogleg out between the screws; one
    # sector carries an angular white panel with a blue two-line bus over it.
    panel = Polygon([P(17.2, 285), P(15.9, 300), P(17.2, 315), P(23.0, 316), P(23.0, 284)])
    bus = [LineString([P(r0, 300), P(23.0, 300)]).offset_curve(o).buffer(W / 2, cap_style=2)
           for r0, o in ((16.9, 0.0), (18.2, 1.9), (18.2, -1.9))]
    bearing = Face(
        "free", (-2.5, -1.4), (-2.0, -1.4), bw_holes, 13.9, 21.1, bore=13.0,
        blue=[trace([(14.7, -52), (14.7, -40), (14.7, -28), (14.7, -16), (15.0, -8),
                     (17.4, 0), (20.0, 0)]),
              trace([(14.7, 98), (14.7, 110), (15.2, 116), (17.2, 121), (19.6, 121)]),
              trace([(18.6, 163), (18.4, 170), (18.4, 190), (18.6, 197)]),
              tangent_ticks(17.1, 19.7, (174, 180, 186)),
              trace([(14.7, 203), (14.7, 216), (15.6, 228), (17.6, 236), (20.2, 236)]),
              trace([(16.8, 50), (18.0, 56), (19.8, 60)])],
        white=[pad(20.0, 0, 0.85), pad(14.7, -52), pad(14.7, 98), pad(19.6, 121, 0.85),
               pad(18.6, 163), pad(18.6, 197), pad(20.2, 236, 0.85), pad(19.8, 60),
               pad(16.8, 50, 0.65), panel],
        top=bus)

    # SmallBearingWahser: the free channels are the diagonals between the four
    # heads and the centre.  -Y face (seen at F, inside the BearingWasher):
    small_f = Face(
        "F_side", (-0.5, 0.6), (0.0, 0.6), sw_holes, 0.0, 10.6,
        blue=[chip(1.8),
              trace([(2.2, 45), (9.2, 45)]), trace([(2.2, 135), (7.0, 135)]),
              trace([(2.2, 315), (9.4, 315)]), ticks(315, (5.4, 6.6))],
        white=[pad(1.2, 135, 0.55), pad(9.2, 45, 0.85), pad(7.0, 135),
               pad(6.0, 225, 0.55), pad(8.6, 225, 0.55)])
    # +Y face (seen at E, capping the coupler-tibia pin): a diamond chip
    small_e = Face(
        "E_side", (1.4, 2.5), (1.4, 2.0), sw_holes, 0.0, 10.6,
        blue=[chip(1.75, rot=45),
              trace([(1.6, 45), (9.0, 45)]),
              trace([(1.6, 225), (5.6, 225), (7.6, 235)]),
              trace([(1.6, 315), (9.6, 315)]), ticks(315, (4.6, 5.8, 7.0))],
        white=[pad(1.0, 90, 0.55), pad(9.0, 45, 0.85), pad(7.6, 235),
               pad(8.4, 135, 0.6), pad(6.2, 135, 0.55)])

    # InsideFemurShaft end face y=10, inside the body box: just the chip
    shaft = Face(
        "end", (9.4, 10.5), (9.4, 10.0), sh_holes, 0.0, 9.6,
        blue=[chip(1.5), *[trace([(1.8, a), (4.3, a)]) for a in (45, 135, 225, 315)]],
        white=[pad(0.9, 135, 0.55), *[pad(4.3, a, 0.6) for a in (45, 135, 225, 315)]])

    return {
        "BearingWasher": (r"Links\BearingWasher.SLDPRT", 2269.1, [bearing]),
        "SmallBearingWahser": (r"Body\SmallBearingWahser.SLDPRT", 760.7, [small_f, small_e]),
        "InsideFemurShaft": (r"Body\InsideFemurShaft.SLDPRT", 3040.3, [shaft]),
    }


# --- preview (no SolidWorks) -----------------------------------------------
def preview(path):
    from PIL import Image, ImageDraw
    rgb = {k: tuple(int(v * 255) for v in col) for k, col in COLOURS.items()}
    bg = (170, 176, 186)
    tiles = [(p, f) for p, (_, _, fs) in faces().items() for f in fs]
    N = 520
    img = Image.new("RGB", (N * len(tiles), N + 40), bg)
    d = ImageDraw.Draw(img)
    for i, (part, f) in enumerate(tiles):
        s = (N / 2 - 20) / (f.r_out + 1.0)
        ox, oy = i * N + N / 2, N / 2 + 30

        def px(x, y):
            return (ox + x * s, oy - y * s)

        def circ(x, y, r, **kw):
            d.ellipse([px(x - r, y + r), px(x + r, y - r)], **kw)

        circ(0, 0, f.r_out + 0.5, fill=rgb["graphite"])
        if f.bore:
            circ(0, 0, f.bore, fill=bg)
        for col in ("blue", "white"):          # one mask per colour: holes stay holes
            mask = Image.new("L", img.size, 0)
            md = ImageDraw.Draw(mask)
            for p in S.polys(getattr(f, col), 0):
                md.polygon([px(x, y) for x, y in p.exterior.coords], fill=255)
                for h in p.interiors:
                    md.polygon([px(x, y) for x, y in h.coords], fill=0)
            img.paste(rgb[col], (0, 0), mask)
        for x, y, rh, rk in f.holes:
            circ(x, y, rk, outline=(200, 190, 60))
            circ(x, y, rh, fill=bg)
        d.text((i * N + 10, 8), f"{part} / {f.name}  blue {f.blue.area:.0f} mm2, "
                                f"white {f.white.area:.0f} mm2", fill=(235, 235, 235))
    img.save(path)
    print(f"  wrote {path}")


# --- SolidWorks -------------------------------------------------------------
def open_part(sw, rel):
    path = os.path.join(swlib.V5, rel)
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        if os.path.normcase(d.GetPathName()) == os.path.normcase(path):
            return d
    doc, err, warn = sw.OpenDoc6(path, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
    if doc is None:
        raise SystemExit(f"cannot open {path} (err {err})")
    return wrap(doc, sld.IModelDoc2)


def box(model):
    lo = [1e9] * 3
    hi = [-1e9] * 3
    for b in S.bodies(model):
        x = [v * 1000 for v in b.GetBodyBox()]
        lo = [min(a, v) for a, v in zip(lo, x[:3])]
        hi = [max(a, v) for a, v in zip(hi, x[3:])]
    return lo + hi


def style(sw, part, rel, v_src, fs, restyle):
    path = os.path.join(swlib.V5, rel)
    bak = os.path.join(ORIGINALS, rel)
    if not os.path.exists(bak):
        os.makedirs(os.path.dirname(bak), exist_ok=True)
        shutil.copy2(path, bak)
        print(f"  backed up the original to {bak}")
    model = open_part(sw, rel)
    sw.ActivateDoc3(model.GetTitle(), False, 0, 0)
    if any(f.Name.startswith("GL_") for f in S._iter_features(model)):
        if not restyle:
            print("  already styled (GL_ features present) -- pass --restyle to rebuild")
            return
        print(f"  --restyle: deleted {S.strip_styling(model, v_src)} GL_ features")
    bs = S.bodies(model)
    v0 = S.total_volume(model)
    if len(bs) != 1 or abs(v0 - v_src) > 0.1:
        raise SystemExit(f"  expected the original: 1 body of {v_src} mm3, got {len(bs)} / {v0:.1f}")
    box0 = box(model)
    made = {"blue": [], "white": []}
    for f in fs:
        for col in ("blue", "white"):
            g = getattr(f, col)
            if g.is_empty:
                continue
            body = max(S.bodies(model), key=S.volume)     # the original: by far the largest
            sk = S.sketch(model, g, name=f"GL_{f.name}_{col}", plane="Top Plane")
            S.cut(model, sk, *f.cut, name=f"GL_{f.name}_{col}_cut", scope=[body])
            new = S.tool(model, sk, *f.inlay, name=f"GL_{f.name}_{col}_inlay")
            for b in new:
                S.colour(b, COLOURS[col])
            made[col] += new
            sk.Select2(False, 0)
            model.BlankSketch()
            print(f"    {f.name:7s} {col:5s} {g.area:6.1f} mm2 -> {len(new)} bodies")
    model.ClearSelection2(True)
    main_body = max(S.bodies(model), key=S.volume)
    S.colour(main_body, COLOURS["graphite"])
    model.EditRebuild3()
    S.name_by_colour(model)

    tot = S.total_volume(model)
    by = {}
    for b in S.bodies(model):
        g = S.group_of(b)
        by[g] = by.get(g, 0.0) + S.volume(b)
    d_box = max(abs(a - b) for a, b in zip(box(model), box0))
    n = len(S.bodies(model))
    print(f"  original {v0:8.2f} mm3 -> " + " + ".join(f"{k} {v:.2f}" for k, v in sorted(by.items())) +
          f" = {tot:8.2f} ({tot - v0:+.4f}), {n} bodies, box moved {d_box:.4f} mm")
    if abs(tot - v0) > 1e-3 or n != 1 + len(made["blue"]) + len(made["white"]) or d_box > 0.01:
        raise SystemExit("  the colour split changed the part -- NOT saved")
    ok, err, warn = model.Extension.SaveAs3(path, c.swSaveAsCurrentVersion,
                                            c.swSaveAsOptions_Silent, None, None, 0, 0)
    if not ok:
        raise SystemExit(f"  save failed (err {err}, warn {warn})")
    print(f"  saved {rel}")


def check_and_export(sw, parts):
    """Mates in every open v5 assembly, then a Bambu 3MF per washer."""
    mate_errors = import_module("10_verify_styled").mate_errors
    export_3mf = import_module("11_check_and_export").export_3mf
    model = swlib.open_v5(sw)
    model.ForceRebuild3(False)
    bad = []
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        if d.GetType() == c.swDocASSEMBLY and os.path.normcase(d.GetPathName()).startswith(
                os.path.normcase(swlib.V5)):
            d.ForceRebuild3(False)
            bad += [(d.GetTitle(),) + b for b in mate_errors(d)]
    print(f"\nMATES, every open v5 assembly: {len(bad)} in error" + "".join(f"\n   {b}" for b in bad))
    for part, (rel, _, _) in parts.items():
        export_3mf(part, open_part(sw, rel))
    return not bad


# --- close-up pictures ------------------------------------------------------
JOINTS = {   # assembly mm: centre and half-size of the view box
    "F_coupler_body": ((-36.42, 37.54, 121.0), 34.0),
    "E_coupler_tibia": ((-186.10, -42.09, 146.0), 22.0),
}


def render(sw, parts):
    render12 = import_module("12_render")
    out = render12.OUT
    os.makedirs(out, exist_ok=True)
    model = swlib.open_v5(sw)
    sw.ActivateDoc3(model.GetTitle(), False, 0, 0)
    for name, ((x, y, z), h) in JOINTS.items():
        # Front only: in *Trimetric the same ViewZoomTo2 box framed the wrong spot
        # (a body corner).  Unverified why -- possibly the box is read in view
        # coordinates, which match the model's only in *Front
        for view in ("*Front",):
            model.Extension.SetUserPreferenceToggle(c.swViewDisplayHideAllTypes, 0, True)
            model.ShowNamedView2(view, -1)
            model.ViewZoomTo2((x - h) / 1000, (y - h) / 1000, (z - h) / 1000,
                              (x + h) / 1000, (y + h) / 1000, (z + h) / 1000)
            p = os.path.join(out, f"washer_{name}_{view.strip('*').lower()}.png")
            ok, err, warn = model.Extension.SaveAs3(
                p, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy, None, None, 0, 0)
            print(f"  {'ok ' if ok else 'ERR'} {os.path.relpath(p, HERE)}")
    for part, (rel, _, _) in parts.items():
        d = open_part(sw, rel)
        for view in ("*Top", "*Bottom", "*Trimetric"):     # the disc faces are +-Y
            render12.shot(sw, d, view, os.path.join(out, f"part_{part}_{view.strip('*').lower()}.png"))
    sw.ActivateDoc3(model.GetTitle(), False, 0, 0)


def main(args):
    parts = faces()
    if args[:1] == ["preview"]:
        preview(os.path.join(HERE, "out", "renders", "washer_patterns.png"))
        return
    sw, _ = swlib.connect()
    if args[:1] == ["render"]:
        render(sw, parts)
        return
    for part, (rel, v_src, fs) in parts.items():
        print(f"\n=== {part}")
        style(sw, part, rel, v_src, fs, "--restyle" in args)
    check_and_export(sw, parts)


if __name__ == "__main__":
    main(sys.argv[1:])
