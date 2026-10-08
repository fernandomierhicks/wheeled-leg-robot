"""Step 22: the wheel rim (Motor/Wheel Motor/Wheel) in GLACIER -- a chip on the hub.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/22_style_wheel.py preview [ctx.json, default out/styled/wheel_ctx.json]
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/22_style_wheel.py [--restyle]
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/22_style_wheel.py verify
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/22_style_wheel.py render

The rim's OUTBOARD face is part -Z (the wheel frame's z axis is robot -Z; the
rim is the outermost part of the robot).  What is there: a flat disc,
z -4.0527, r <= 20.91, on a 0.8 mm floor (the motor can sits against its
inner face), then a 45-degree cone out to the S-spokes' ends at z 0.  Four
holes, r 1.45, on r 17.00 at 45.681 + k*90 deg (the screws into the can; the
heads sit on this face).  A r 1 x 1 mm nub at the centre (his call: bury it).

The design (his pick 2026-10-06: B, the DARK wheel, to contrast the white links):

  RIM     the whole original body graphite.  Spokes, rim band, tyre ribs,
          floor, holes and everything inside the cup: unchanged.
  CHIP    white octagon, circumradius 11, 1.6 mm proud, 45-degree bevel,
          inside the screw circle (a vertex at each screw, 6.0 clear of its
          head seat).  A new body: the original stays the one mates hold.
  CROWN   graphite octagon, circumradius 7.2, turned 22.5 deg, on a column
          through the chip, +1.2 mm, 45-degree bevel; merged into the rim
          (it buries the nub).  Blue diamond die with four legs to the
          screws, a white pin-1 dot.
  FLOOR   flush 0.6 mm colour inlays on the flat round the chip: a white
          chevron bracket round every screw head (it reads as a component),
          a different blue trace on each of the four lobes between screws,
          white pads, a white chevron panel under a three-line bus.

Nothing raised goes round the screws: their seats (r 11.45 .. 22.55) fill the
face, and every plate shaped round them came out four-lobed -- flared lobes
read as a cross pattee, hooked ones as a swastika (both tried in preview).

Colour inlays: cut from the body they sit in, the same sketch extruded back as
a new body (the 07/15 recipe), so the bodies' union is the styled shape.
Every feature is GL_*; --restyle deletes them first (strip_styling proves the
original by volume).  After the checks pass it saves Wheel.SLDPRT AND every
assembly above it (WheelMotorASM, Tibia, ROBOT) through 16_check_and_save.chain.

Coordinates: part X, Y in mm (= Front Plane sketch coordinates).  The preview
draws the face AS SEEN FROM OUTBOARD (from -Z), i.e. X mirrored.
"""
import json
import math
import os
import sys

from shapely import affinity, wkt
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))

# --- the part as it is (measured on the v5 Wheel.SLDPRT, 2026-10-06) --------
Z_FACE = -4.0527            # outboard hub face (normal -Z)
R_FLAT = 20.91              # the flat ends here (0.6 fillet, then the 45-deg cone)
HOLE_R, HOLE_PCD_R = 1.45, 17.00
HOLE_A0 = 45.681            # deg, first hole; the others every 90
HOLES = [(HOLE_PCD_R * math.cos(math.radians(HOLE_A0 + 90 * k)),
          HOLE_PCD_R * math.sin(math.radians(HOLE_A0 + 90 * k))) for k in range(4)]
SEAT = HOLE_R + 3.5         # head / washer seat (the links' fastener rule)
ADD_CLEAR = 0.6             # additions stay this far off a seat

# --- the design -------------------------------------------------------------
# Nothing raised may go round the screws: their seats (r 11.45 .. 22.55) fill
# the face, and every plate shaped round them came out four-lobed -- flared
# lobes read as a cross pattee, hooked ones as a swastika (both tried in
# preview, 2026-10-06).  So the raised part stays INSIDE the screw circle (an
# octagonal "chip"), and the rest of the face is a flush circuit board.
LOBE_A0 = HOLE_A0 - 45.0    # between the screws
R_INLAY = R_FLAT - 0.5      # flush inlays stay on the flat (gotcha 34: not over the fillet)
GEM_R = 11.0                # raised octagon, circumradius; a vertex points at each screw (6.0 clear)
CROWN_R = 7.2               # second tier, circumradius, turned 22.5 deg against the gem
H1, H2 = 1.6, 1.2           # gem, crown height
BEVEL = 45.0                # deg, both tiers
INLAY = 0.6                 # colour inlay depth
MARGIN = 0.4                # inlays stay this far inside a top face
HEAD_KEEP = 2.85 + 0.5      # flush inlays stay off the M3 head footprint (as on the washers)
W = 1.0                     # trace width
COLOURS = {"white": (0.93, 0.94, 0.95), "graphite": (0.25, 0.27, 0.30), "blue": (0.13, 0.45, 0.95)}


def octagon(r, rot_deg):
    return Polygon([(r * math.cos(math.radians(rot_deg + 45 * k)),
                     r * math.sin(math.radians(rot_deg + 45 * k))) for k in range(8)])


def frame(a0_deg):
    """(u, v) in a frame turned a0 -> part (x, y)."""
    a = math.radians(a0_deg)
    ca, sa = math.cos(a), math.sin(a)
    return lambda pts: [(u * ca - v * sa, u * sa + v * ca) for u, v in pts]


def lobe(k):
    return frame(LOBE_A0 + 90 * k)


def screw(k):
    """Frame centred on screw k, u radial (out), v tangential."""
    f = frame(HOLE_A0 + 90 * k)
    return lambda pts: f([(HOLE_PCD_R + u, v) for u, v in pts])


def trace(f, pts, w=W):
    return LineString(f(pts)).buffer(w / 2, cap_style=2, join_style=2, mitre_limit=2.0)


def pad(f, u, v, r=0.85):
    return Point(f([(u, v)])[0]).buffer(r, quad_segs=16)


def tick(f, u, v, half=1.2, w=0.8):
    return LineString(f([(u, v - half), (u, v + half)])).buffer(w / 2, cap_style=2)


def slash(f, u, v, length=3.0, w=0.8):
    """A 45-degree vent slash centred on (u, v)."""
    d = length / 2 / math.sqrt(2)
    return LineString(f([(u - d, v - d), (u + d, v + d)])).buffer(w / 2, cap_style=2)


def design():
    keep = unary_union([Point(h).buffer(SEAT + ADD_CLEAR, quad_segs=32) for h in HOLES])
    gem = octagon(GEM_R, HOLE_A0)
    if not gem.within(Point(0, 0).buffer(R_FLAT)) or gem.intersects(keep):
        raise SystemExit("design: the gem leaves the flat or touches a screw seat -- fix the numbers")
    crown = octagon(CROWN_R, LOBE_A0 + 22.5)
    gem_top = gem.buffer(-H1 * math.tan(math.radians(BEVEL)), join_style=2)
    crown_top = crown.buffer(-H2 * math.tan(math.radians(BEVEL)), join_style=2)
    heads = unary_union([Point(h).buffer(HEAD_KEEP, quad_segs=32) for h in HOLES])
    floor_room = (Point(0, 0).buffer(R_INLAY, quad_segs=128)
                  .difference(gem.buffer(0.3, join_style=2)).difference(heads))
    crown_room = crown_top.buffer(-MARGIN, join_style=2)

    u0 = GEM_R * math.cos(math.radians(22.5)) + 0.8       # floor traces start just off the gem
    dark, accent = [], []          # on the floor: dark = graphite on a white wheel
    # every screw a "component": a square bracket round its head, open to the
    # rim, its two inner corners chamfered (seen from the hub it is a chevron)
    for k in range(4):
        s = screw(k)
        o, i, c = 4.6, 3.5, 1.4
        dark.append(Polygon(s([(6.0, -o), (-o + c, -o), (-o, -o + c), (-o, o - c), (-o + c, o), (6.0, o),
                               (6.0, i), (-i + c * 0.42, i), (-i, i - c * 0.42), (-i, -i + c * 0.42),
                               (-i + c * 0.42, -i), (6.0, -i)])))
    # between the screws, a different trace on every lobe
    L = [lobe(k) for k in range(4)]
    accent += [trace(L[0], [(u0, 0), (14.4, 0), (16.4, 2.0), (19.0, 2.0)]),
               tick(L[0], 12.2, 0), tick(L[0], 13.4, 0)]
    dark += [pad(L[0], 19.0, 2.0, 0.8)]
    accent += [trace(L[1], [(u0, 1.0), (15.0, 1.0), (17.0, 3.0), (18.8, 3.0)]),
               trace(L[1], [(u0, -1.0), (15.0, -1.0), (17.0, -3.0), (18.8, -3.0)])]
    dark += [pad(L[1], 18.8, 3.0, 0.75), pad(L[1], 18.8, -3.0, 0.75)]
    accent += [trace(L[2], [(u0, 0), (13.6, 0)]),
               slash(L[2], 15.4, 0), slash(L[2], 16.7, 0), slash(L[2], 18.0, 0)]
    dark += [pad(L[2], 13.6, 0, 0.8)]
    dark += [Polygon(L[3]([(14.0, -2.6), (17.2, -2.6), (19.2, -0.6), (19.2, 0.6),
                           (17.2, 2.6), (14.0, 2.6), (15.0, 0)]))]
    accent += [trace(L[3], [(u0, v), (18.6 - abs(v), v)], 0.8) for v in (-1.2, 0.0, 1.2)]
    # crown: a blue diamond chip with four legs towards the screws, a pin-1 dot
    chip = affinity.rotate(Polygon([(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]),
                           HOLE_A0 + 45, origin=(0, 0))
    legs = [trace(screw(k), [(-15.2, 0), (-12.0, 0)], 0.8) for k in range(4)]
    dot = [Point(lobe(0)([(1.1, 0.0)])[0]).buffer(0.5, quad_segs=16)]

    def split(under, over, room):
        """`over` is drawn on top of `under` (blue bus over a graphite panel)."""
        over = unary_union(over).intersection(room)
        under = unary_union(under).difference(over).intersection(room)
        return under, over

    f_dark, f_accent = split(dark, accent, floor_room)
    c_accent, c_dot = split([chip] + legs, dot, crown_room)
    c_accent = c_accent.difference(c_dot)
    return {"gem": gem, "gem_top": gem_top, "crown": crown, "crown_top": crown_top,
            "floor_dark": f_dark, "floor_accent": f_accent,
            "crown_accent": c_accent, "crown_dot": c_dot, "keep": keep}


# --- preview (no SolidWorks) -----------------------------------------------
def polys(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [p for q in getattr(g, "geoms", []) for p in polys(q)]


def _fill(img, d, geom, colour, px):
    from PIL import Image, ImageDraw
    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    for p in polys(geom):
        md.polygon([px(x, y) for x, y in p.exterior.coords], fill=255)
        for h in p.interiors:
            md.polygon([px(x, y) for x, y in h.coords], fill=0)
    img.paste(colour, (0, 0), mask)


def preview(path, ctx_path=None):
    from PIL import Image, ImageDraw
    g = design()
    ctx = {k: wkt.loads(v) for k, v in json.load(open(ctx_path)).items()} if ctx_path else {}
    rgb = {k: tuple(int(v * 255) for v in c_) for k, c_ in COLOURS.items()}
    shade = lambda c_, f: tuple(int(v * f) for v in c_)
    bg = (150, 156, 168)
    palettes = {
        "A: white wheel, graphite chip": {"rim": rgb["white"], "gem": rgb["graphite"], "crown": rgb["white"],
                                          "dark": rgb["graphite"]},
        "B: graphite wheel, white chip": {"rim": rgb["graphite"], "gem": rgb["white"], "crown": rgb["graphite"],
                                          "dark": rgb["white"]},
    }
    N, S_ = 900, 7.2               # tile px, px/mm for the full wheel view
    Z = 19.0                       # px/mm for the hub close-up
    img = Image.new("RGB", (N * 2, N * 2 + 30), bg)
    d = ImageDraw.Draw(img)
    for col, (title, pal) in enumerate(palettes.items()):
        for row, s in enumerate((S_, Z)):
            tile = Image.new("RGB", (N, N), bg)
            td = ImageDraw.Draw(tile)

            def px(x, y, s=s):
                return (N / 2 - x * s, N / 2 - y * s)        # seen from outboard: X mirrored

            if ctx:
                _fill(tile, td, ctx["spokes"], pal["rim"], px)
                _fill(tile, td, ctx["cone"], shade(pal["rim"], 0.82), px)
                _fill(tile, td, ctx["floor"], pal["rim"], px)
            _fill(tile, td, g["floor_dark"], pal["dark"], px)
            _fill(tile, td, g["floor_accent"], rgb["blue"], px)
            for h in HOLES:
                _fill(tile, td, Point(h).buffer(HOLE_R), (40, 40, 40), px)
                if row:          # M3 socket head
                    td.line([px(*q) for q in Point(h).buffer(2.75).exterior.coords], fill=(220, 190, 40), width=2)
            _fill(tile, td, g["gem"], shade(pal["gem"], 0.78), px)              # bevel
            _fill(tile, td, g["gem_top"], pal["gem"], px)
            _fill(tile, td, g["crown"], shade(pal["crown"], 0.80), px)
            _fill(tile, td, g["crown_top"], pal["crown"], px)
            _fill(tile, td, g["crown_accent"], rgb["blue"], px)
            _fill(tile, td, g["crown_dot"], pal["dark"], px)
            td.text((10, 6), title + ("  (hub close-up, yellow = M3 socket head)" if row else ""),
                    fill=(250, 250, 250))
            img.paste(tile, (col * N, row * N + 30))
    d.text((10, 8), f"Wheel hub, seen from OUTBOARD.  chip +{H1} mm, crown +{H2} mm, 45-deg bevels; "
                    f"flush inlays {INLAY} mm.", fill=(250, 250, 250))
    img.save(path)
    print(f"  wrote {path}")


# --- SolidWorks -------------------------------------------------------------
RIM = r"Motor\Wheel Motor\Wheel.SLDPRT"
ASM = r"Motor\Wheel Motor\WheelMotorASM.SLDASM"
V_SRC = 38815.66            # mm3, the rim before styling (tyre-lock ribs included)
V_NUB = 3.1428              # mm3, the centre nub the column swallows (measured, OCC)
WHEEL, GEM, CROWN, LIGHT = "graphite", "white", "graphite", "white"     # palette B
Z_TOP = Z_FACE - H1 - H2    # the crown's top face
OUT = os.path.join(HERE, "out")
FILAMENT = {"white": 1, "graphite": 2, "blue": 3}


def frustum(poly, h, n=64):
    """Volume of `poly` extruded h with the inward BEVEL draft (Simpson's rule;
    the area of a mitred inward offset is quadratic in the offset, so exact)."""
    t = math.tan(math.radians(BEVEL))
    f = lambda z: poly.buffer(-z * t, join_style=2).area
    hs = h / n
    return hs / 3 * (f(0) + f(h) + sum((4 if i % 2 else 2) * f(i * hs) for i in range(1, n)))


def _open(sw, rel, kind=None):
    import swlib
    from swlib import c, wrap, sld
    kind = kind or c.swDocPART
    path = os.path.join(swlib.V5, rel)
    d, err, warn = sw.OpenDoc6(path, kind, c.swOpenDocOptions_Silent, "", 0, 0)
    if d is None:
        raise SystemExit(f"cannot open {path} (err {err})")
    m = wrap(d, sld.IModelDoc2)
    sw.ActivateDoc3(m.GetTitle(), False, 0, 0)
    return m


def _box(S, model, bs=None):
    lo, hi = [1e9] * 3, [-1e9] * 3
    for b in bs or S.bodies(model):
        x = [v * 1000 for v in b.GetBodyBox()]
        lo = [min(a, v) for a, v in zip(lo, x[:3])]
        hi = [max(a, v) for a, v in zip(hi, x[3:])]
    return lo + hi


def _drafted(S, model, sk, base, h, merge, name):
    """Pad from `base` towards -Z (outboard), h tall, BEVEL inward draft.  No
    silent fallback to a square edge (swstyle.raised has one): refuse."""
    from swlib import wrap, sld
    f = S._extrude(model, sk, base, h, False, merge, BEVEL)
    if f is None:
        raise S.FeatureFailed(f"drafted pad {name} refused")
    f = wrap(f, sld.IFeature)
    f.Name = name
    return f


def _expect(what, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"    {what:34s} {got:10.3f}  expected {want:10.3f}  {'ok' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(f"{what}: {got:.4f} vs {want:.4f} -- NOT saved")


def build(sw, restyle):
    import swlib
    import swstyle as S
    from importlib import import_module
    from swlib import c, wrap, sld
    g = design()
    model = _open(sw, RIM)
    if any(f.Name.startswith("GL_") for f in S._iter_features(model)):
        if not restyle:
            raise SystemExit("Wheel already has GL_ features -- pass --restyle to rebuild")
        print(f"  --restyle: deleted {S.strip_styling(model, V_SRC)} GL_ features")
    bs = S.bodies(model)
    v0 = S.total_volume(model)
    if len(bs) != 1 or abs(v0 - V_SRC) > 0.1:
        raise SystemExit(f"expected the original rim: 1 body of {V_SRC} mm3, got {len(bs)} / {v0:.2f}")
    box0 = _box(S, model)
    a_c = g["crown"].area

    # 1. the crown's column + the crown, merged into the rim (the only body yet)
    sk = S.sketch(model, g["crown"], name="GL_ColumnSk", grow=0.0)
    S.boss(model, sk, Z_FACE - H1, Z_FACE + 0.1, name="GL_Column")       # 0.1 into the floor
    sk = S.sketch(model, g["crown"], name="GL_CrownSk", grow=0.0)
    _drafted(S, model, sk, Z_FACE - H1, H2, True, "GL_Crown")
    model.EditRebuild3()
    if len(S.bodies(model)) != 1:
        raise SystemExit(f"column/crown did not merge: {len(S.bodies(model))} bodies")
    v1 = S.total_volume(model)
    _expect("column + crown added (mm3)", v1 - v0, a_c * H1 - V_NUB + frustum(g["crown"], H2), 0.05)

    # 2. the chip: a new body, then the column's footprint cut out of it
    before = {S.id_(b) for b in S.bodies(model)}
    sk = S.sketch(model, g["gem"], name="GL_ChipSk", grow=0.0)
    _drafted(S, model, sk, Z_FACE, H1, False, "GL_Chip")
    gem = [b for b in S.bodies(model) if S.id_(b) not in before]
    if len(gem) != 1:
        raise SystemExit(f"the chip made {len(gem)} bodies")
    sk = S.sketch(model, g["crown"], name="GL_ChipHoleSk", grow=0.0)
    S.cut(model, sk, Z_FACE - H1 - 0.3, Z_FACE + 0.3, name="GL_ChipHole", scope=gem)
    model.EditRebuild3()
    rim = max(S.bodies(model), key=S.volume)
    gem = [b for b in S.bodies(model) if S.id_(b) != S.id_(rim)]
    if len(gem) != 1:
        raise SystemExit(f"after the chip: {len(S.bodies(model))} bodies, expected 2")
    gem = gem[0]
    _expect("chip body (mm3)", S.volume(gem), frustum(g["gem"], H1) - a_c * H1, 0.05)
    gb = _box(S, model, [gem])
    _expect("chip z from (outboard)", gb[2], Z_FACE - H1, 0.01)
    _expect("chip z to", gb[5], Z_FACE, 0.01)
    _expect("crown top z", _box(S, model, [rim])[2], Z_TOP, 0.01)
    v2 = S.total_volume(model)
    box2 = _box(S, model)

    # 3. colour inlays, each cut from its own body and extruded back
    S.colour(rim, COLOURS[WHEEL])
    S.colour(gem, COLOURS[GEM])
    made = 0
    for key, col, z in (("floor_dark", LIGHT, Z_FACE), ("floor_accent", "blue", Z_FACE),
                        ("crown_accent", "blue", Z_TOP), ("crown_dot", LIGHT, Z_TOP)):
        geom = g[key]
        if geom.is_empty:
            continue
        rim = max(S.bodies(model), key=S.volume)
        sk = S.sketch(model, geom, name=f"GL_{key}")
        S.cut(model, sk, z - 0.5, z + INLAY, name=f"GL_{key}_cut", scope=[rim])
        new = S.tool(model, sk, z, z + INLAY, name=f"GL_{key}_inlay")
        for b in new:
            S.colour(b, COLOURS[col])
        made += len(new)
        print(f"    {key:13s} {col:8s} {geom.area:6.1f} mm2 -> {len(new)} bodies")
    for f in S._iter_features(model):
        if f.Name.startswith("GL_") and f.GetTypeName2() == "ProfileFeature":
            model.ClearSelection2(True)
            f.Select2(False, 0)
            model.BlankSketch()
    model.ClearSelection2(True)
    model.EditRebuild3()
    S.name_by_colour(model)
    n = len(S.bodies(model))
    _expect("inlays: volume unchanged (mm3)", S.total_volume(model), v2, 1e-3)
    _expect("bodies", n, 2 + made, 0)
    d_box = max(abs(a - b) for a, b in zip(_box(S, model), box2))
    _expect("inlays: box moved (mm)", d_box, 0.0, 0.01)
    d_xy = max(abs(a - b) for i, (a, b) in enumerate(zip(_box(S, model), box0)) if i % 3 != 2)
    _expect("rim x/y extent moved (mm)", d_xy, 0.0, 0.01)
    by = {}
    for b in S.bodies(model):
        by[S.group_of(b)] = by.get(S.group_of(b), 0.0) + S.volume(b)
    print(f"  {v0:.2f} mm3 -> " + " + ".join(f"{k} {v:.2f}" for k, v in sorted(by.items())) +
          f" = {S.total_volume(model):.2f} ({S.total_volume(model) - v0:+.2f}), {n} bodies")

    # mates: the wheel's own assembly and the robot, before anything is saved
    mate_errors = import_module("10_verify_styled").mate_errors
    bad = []
    for rel, kind in ((ASM, c.swDocASSEMBLY), (os.path.relpath(swlib.ASM, swlib.V5), c.swDocASSEMBLY)):
        a = _open(sw, rel, kind)
        a.ForceRebuild3(False)
        bad += [(rel,) + e for e in mate_errors(a)]
    print(f"  mates in error (WheelMotorASM, ROBOT): {len(bad)}" + "".join(f"\n    {b}" for b in bad))
    if bad:
        raise SystemExit("mate errors -- NOT saved")
    # the rim AND every assembly above it (WheelMotorASM, Tibia, ROBOT), never the part alone
    import_module("16_check_and_save").chain(sw, [RIM])
    sw.ActivateDoc3(model.GetTitle(), False, 0, 0)
    export(sw, model)


def export(sw, model):
    """STEP (one solid per body) and a Bambu 3MF in PRINT orientation: cup
    down, hub face up (turned 180 deg about X), filament 1 white / 2 graphite /
    3 blue."""
    import numpy as np
    import swstyle as S
    from importlib import import_module
    from swlib import c
    sys.path.insert(0, os.path.join(HERE, "..", "aesthetics", "lib"))
    import export3mf
    body_mesh = import_module("11_check_and_export").body_mesh
    os.makedirs(os.path.join(OUT, "styled"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "print"), exist_ok=True)
    stp = os.path.join(OUT, "styled", "Wheel_glacier_native.step")
    ok, err, warn = model.Extension.SaveAs3(stp, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy,
                                            None, None, 0, 0)
    print(f"  {'ok ' if ok else 'ERR'} {os.path.relpath(stp, HERE)}")
    parts, by = [], {}
    for b in S.bodies(model):
        V, T = body_mesh(b)
        if V is None:
            continue
        fil = FILAMENT[S.group_of(b)]
        parts.append((b.Name, V * np.array([1.0, -1.0, -1.0]), T, fil))
        by[fil] = by.get(fil, 0) + 1
    out = os.path.join(OUT, "print", "Wheel_glacier_native.3mf")
    export3mf.write_3mf(parts, out, name="Wheel GLACIER (native SolidWorks), cup down")
    print(f"  3MF: {len(parts)} bodies (white x{by.get(1, 0)}, graphite x{by.get(2, 0)}, "
          f"blue x{by.get(3, 0)}) -> {os.path.relpath(out, HERE)}")


def verify(src_step):
    """Offline (OCC), styled STEP against the source rim: nothing removed,
    nothing changed inboard of the inlays, the holes empty, every head seat
    clear of additions."""
    from build123d import import_step, Box, Cylinder, Pos
    src = import_step(src_step)
    sty = import_step(os.path.join(OUT, "styled", "Wheel_glacier_native.step"))
    solids = sty.solids()
    fused = solids[0]
    for s_ in solids[1:]:
        fused = fused.fuse(s_)
    print(f"  styled: {len(solids)} solids, fused {fused.volume:.2f} mm3; source {src.volume:.2f}")
    inb = Pos(0, 0, Z_FACE + INLAY + 0.01 + 50) * Box(200, 200, 100)
    vol = lambda x: 0.0 if x is None else x.volume        # an empty boolean comes back None
    rows = [("source - styled (removed)", vol(src - fused), 0.0),
            ("added (styled - source)", vol(fused - src), None),
            ("inboard of the inlays, source", vol(src & inb), None),
            ("inboard of the inlays, styled", vol(fused & inb), vol(src & inb))]
    for x, y in HOLES:
        rows.append((f"hole ({x:6.2f},{y:6.2f}) material", vol(fused & Pos(x, y, 10) * Cylinder(HOLE_R - 0.01, 40)), 0.0))
        seat = Pos(x, y, Z_FACE - 5) * Cylinder(SEAT + ADD_CLEAR - 0.01, 10)
        rows.append((f"  its head seat, above the face", vol(fused & seat), 0.0))
    bad = 0
    for what, v, want in rows:
        flag = "" if want is None else ("ok" if abs(v - want) < 1e-3 else "FAIL")
        bad += flag == "FAIL"
        print(f"    {what:34s} {v:10.3f} mm3  {flag}")
    print("  VERIFY " + ("PASS" if not bad else f"FAIL ({bad})"))


def render(sw):
    """The part from outboard (and oblique), the cup side, and the wheel on the robot."""
    import swlib
    from importlib import import_module
    from swlib import c
    r12 = import_module("12_render")
    out = os.path.join(OUT, "renders")
    m = _open(sw, RIM)
    for view, tag, rot in (("*Back", "outboard", None), ("*Back", "outboard_oblique", (0.45, 0.55)),
                           ("*Trimetric", "cup", None)):
        m.Extension.SetUserPreferenceToggle(c.swViewDisplayHideAllTypes, 0, True)
        m.ShowNamedView2(view, -1)
        if rot:
            m.ActiveView.RotateAboutCenter(*rot)
        m.ViewZoomtofit2()
        p = os.path.join(out, f"wheel_{tag}.png")
        ok, err, warn = m.Extension.SaveAs3(p, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy, None, None, 0, 0)
        print(f"  {'ok ' if ok else 'ERR'} {os.path.relpath(p, HERE)}")
    robot = swlib.open_v5(sw)
    comps = swlib.components(robot)
    x, y, z = swlib.placement(comps["Tibia-1/WheelMotorASM-1/Wheel-1"])[:, 3]
    robot.Extension.SetUserPreferenceToggle(c.swViewDisplayHideAllTypes, 0, True)
    robot.ShowNamedView2("*Front", -1)          # robot +Z = outboard of the right leg
    h = 75.0
    robot.ViewZoomTo2((x - h) / 1000, (y - h) / 1000, (z - h) / 1000, (x + h) / 1000, (y + h) / 1000, (z + h) / 1000)
    p = os.path.join(out, "wheel_on_robot_front.png")
    ok, err, warn = robot.Extension.SaveAs3(p, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy, None, None, 0, 0)
    print(f"  {'ok ' if ok else 'ERR'} {os.path.relpath(p, HERE)}")
    r12.shot(sw, robot, "*Trimetric", os.path.join(out, "wheel_on_robot_trimetric.png"))


def main(args):
    if args[:1] == ["preview"]:
        ctx = args[1] if len(args) > 1 else os.path.join(OUT, "styled", "wheel_ctx.json")    # the source rim, sliced
        preview(os.path.join(OUT, "renders", "wheel_hub_preview.png"), ctx if os.path.exists(ctx) else None)
        return
    if args[:1] == ["verify"]:
        verify(args[1] if len(args) > 1 else os.path.join(OUT, "styled", "Wheel_source.step"))
        return
    sys.path.insert(0, HERE)
    import swlib
    sw, _ = swlib.connect()
    if args[:1] == ["render"]:
        render(sw)
        return
    build(sw, "--restyle" in args)


if __name__ == "__main__":
    main(sys.argv[1:])
