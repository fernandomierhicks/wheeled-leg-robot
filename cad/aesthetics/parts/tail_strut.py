"""The back support (tail strut) under the box -- a faceted keel that replaces the
v3 BackWheelSupport loft (2026-10-06, his call: "free to edit the shape
aggressively").  ROBOT coords, mm, Distance1 = 75 frame (X forward, Y up, Z to
the right leg), the same frame as box_facet_print.py.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/aesthetics/parts/tail_strut.py <out_dir> [--3mf <print_dir>]

writes TailStrut.step (colour bodies via lib/stepcolor), preview_tail_strut.png
(old vs new, in context) and the checks below.

KEPT EXACTLY (the old mates are re-made on these faces by geometry):
  * the top face, y = 15, against the floor (BottomPanel underside)
  * the 4 M3 clearance holes r 1.7 at x -147 / -112, z +-20; the heads seat on
    the pad underside at y = 5, as on the v3 part (10 mm grip)
  * the caster axle (-171.676, -32) along z: r 1.7 through the +z arm, r 1.4
    (self-tap) through the -z arm; the fork's inner faces at z = +-5
KEPT OUT OF:
  * below y = -35 (the v3 fork bottom; the caster's tyre reaches -39.5)
  * behind the tip-over line, caster back (-179.18, -32) -> bumper heel (-170, 15)
  * above y = 15 (the floor) and outside |z| 40 (the side walls are at 75, the legs beyond)
PRINT: pad down (top face on the bed); every face that looks up in the robot is
at least 45 deg off horizontal, so no supports (the window's floor is a V).

Parts of the shape:
  PAD     10 mm plate, chevron nose forward, chamfered heel, drafted sides
  KEEL    the blade from the pad to the fork: swept back edge and front edge,
          both ridged (two facets meeting on the centre line -- a crystal keel),
          a V window through it, all planar
  FORK    plain block round the axle, the caster's slot (faceted arch)
  COLOUR  graphite body; white chevron armour inlaid on the two back facets,
          a blue line on the ridge; white flags on the pad's flanks
"""
import os
import sys
import math
import numpy as np
from build123d import Box, Pos, Plane, Polyline, make_face, extrude, Cylinder, Axis, import_step, Keep, split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)
import stepcolor
import box_concept_facet as K

WHITE, GRAPHITE, DARK, BLUE = K.WHITE, K.GRAPHITE, K.DARK, K.BLUE
TPU_GREY = (0x6E, 0x73, 0x7A)
INPUT = os.path.join(HERE, "..", "input", "box")

# ------------------------------------------------------------------ fixed by the robot
Y_TOP, Y_SEAT = 15.0, 5.0                     # floor underside; screw-head seat (pad underside)
HOLES = [(-147.0, -20.0), (-147.0, 20.0), (-112.0, -20.0), (-112.0, 20.0)]   # (x, z)
R_CLEAR = 1.7
AXLE = (-171.676, -32.0)                      # caster axle (x, y), along z
R_AXLE_CLEAR, R_AXLE_TAP = 1.7, 1.4           # +z arm clearance, -z arm self-tap
FORK_IN = 5.0                                 # fork inner faces z = +-5 (caster is 8 wide)
CASTER_R = 7.5
Y_LOW = -35.0                                 # nothing lower
TIP = ((-179.18, -32.0), (-170.0, 15.0))      # tip-over line: nothing behind it

# ------------------------------------------------------------------ the design
PAD_HALF_W = 32.0
PAD_PLAN_HALF = [(-92.0, 0.0), (-100.0, PAD_HALF_W), (-150.0, PAD_HALF_W), (-153.0, 26.0), (-156.0, 0.0)]
PAD_NOSE = ((-100.0, Y_SEAT), (-92.0, 11.0))      # underside chamfers (side view): nose ...
PAD_HEEL = ((-151.0, Y_SEAT), (-156.0, 10.0))     # ... and heel (clear of the head seats at -108.7 / -150.3)
PAD_DRAFT = 2.0                               # pad sides inset this much at its underside
Y_KT = Y_SEAT + 0.5                           # keel top (0.5 into the pad)
KEEL_SIDE = [(-106.0, Y_KT), (-151.0, Y_KT), (-178.0, -28.0), (-176.0, Y_LOW), (-163.0, Y_LOW)]
KEEL_Z_TOP, KEEL_Z_LOW = 16.0, 16.0           # half-width at the pad / at the fork: the v3 fork's
                                              # outer faces were at +-16, so the same axle screw fits
BACK = ((-151.0, Y_KT), (-178.0, -28.0))      # the swept back edge (CCW)
FRONT = ((-163.0, Y_LOW), (-106.0, Y_KT))     # the front (under) edge (CCW)
# crystal facets: each corner of the keel cut by a triangle that is CUT deep
# (in the side view) at the pad and runs out to nothing at END -- no steps.
# BACK_CUT = the half-width: the back is a sharp V ridge at the pad.
BACK_CUT, BACK_END = 16.0, -22.0
FRONT_CUT, FRONT_END = 9.0, -20.0
WINDOW = [(-126.0, 0.0), (-147.0, 0.0), (-141.0, -12.5)]   # V window (x, y), through z
LINER = 1.4                                   # blue lining round the window, on both flanks
SLOT = [(-183.0, Y_LOW - 1), (-183.0, -24.0), (-178.5, -20.5), (-165.0, -20.5), (-160.5, -25.0),
        (-160.5, Y_LOW - 1)]                  # caster slot (x, y), |z| < FORK_IN
# fangs: a raked fin under each outer edge of the pad, |z| 25..30 (clear of the
# screw keys at |z| <= 23.3), pointing down and back
FANG = [(-102.0, Y_KT), (-150.0, Y_KT), (-160.0, -12.0)]
FANG_Z = (25.0, 30.0)                         # |z| at the pad ...
FANG_SPLAY = 4.0                              # ... and this much further out at the tip (a jaw from behind)
# teeth along the front (under) edge, raked forward: at edge length s from the fork
TEETH_S, TOOTH_L, TOOTH_H, TOOTH_Z = (27.0, 37.0, 47.0), 7.0, 4.5, (9.0, 4.0)
INLAY = 0.8                                   # colour inlay depth
FACET_INSET = 1.0                             # facet colour panels stay this far inside their facet
RIDGE_W = 1.4                                 # blue line on the back ridge, full width


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def mirror_half(half):
    rev = [(x, -z) for x, z in reversed(half)]
    if abs(half[-1][1]) < 1e-9:
        rev = rev[1:]
    if abs(half[0][1]) < 1e-9:
        rev = rev[:-1]
    return half + rev


def side_planes(prof):
    """Half-spaces of a convex (x, y) profile, constant in z."""
    P = K.ccw(prof)
    out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        e = b - a
        out.append(((a[0], a[1], 0.0), (e[1], -e[0], 0.0), f"side{i}"))
    return out


def edge_normal(a, b):
    """Outward normal (x, y) of the CCW profile edge a -> b."""
    e = np.subtract(b, a)
    return unit([e[1], -e[0]])


def z_half(y):
    return KEEL_Z_LOW + (KEEL_Z_TOP - KEEL_Z_LOW) * (y - Y_LOW) / (Y_KT - Y_LOW)


def x_on(edge, y):
    (x0, y0), (x1, y1) = edge
    return x0 + (y - y0) * (x1 - x0) / (y1 - y0)


def facet(edge, cut, y_end, sz):
    """One crystal facet on the corner (edge, side sz): the plane through the
    ridge at the top of the edge, the side corner `cut` inside the edge at the
    top, and the side corner at y_end.  Returns (three points, outward normal)."""
    (xa, ya), _ = edge
    top = edge[0] if edge[0][1] > edge[1][1] else edge[1]
    n2 = edge_normal(*edge)
    q1 = np.array([top[0], top[1], 0.0])
    q2 = np.array([top[0] - cut * n2[0], top[1] - cut * n2[1], sz * z_half(top[1])])
    q3 = np.array([x_on(edge, y_end), y_end, sz * z_half(y_end)])
    n = np.cross(q2 - q1, q3 - q1)
    out = np.array([n2[0], n2[1], 0.0])
    if n @ out < 0:
        n = -n
    return (q1, q2, q3), unit(n)


def facets():
    return [facet(BACK, BACK_CUT, BACK_END, sz) for sz in (1, -1)] + \
           [facet(FRONT, FRONT_CUT, FRONT_END, sz) for sz in (1, -1)]


def prism_z(prof_xy, z0, z1):
    f = make_face(Polyline(*[(float(x), float(y)) for x, y in prof_xy], close=True))
    return Pos(0, 0, z0) * extrude(f, amount=z1 - z0)


def pad():
    """Full plan at the top face, every side PAD_DRAFT inset at the underside.
    (K.frustum_walls is built narrow-end-up; this one is narrow-end-down.)"""
    P = K.ccw(mirror_half(PAD_PLAN_HALF))
    h = Y_TOP - Y_SEAT
    walls = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        m = unit([b[1] - a[1], -(b[0] - a[0])])          # outward in the (x, z) plan
        walls.append(((a[0], Y_TOP, a[1]), (h * m[0], -PAD_DRAFT, h * m[1]), "pad"))
    for (xa, ya), (xb, yb) in (PAD_NOSE, PAD_HEEL):
        n = unit([yb - ya, -(xb - xa)])
        if n[0] * (xa + 128) < 0:                         # point it away from the pad's middle
            n = -n
        walls.append(((xa, ya, 0.0), (n[0], n[1], 0.0), "chamfer"))
    return K.polyhedron(walls + [((0, Y_TOP, 0), (0, 1, 0), "top"), ((0, Y_SEAT, 0), (0, -1, 0), "seat")])


def keel():
    dz = (KEEL_Z_TOP - KEEL_Z_LOW) / (Y_KT - Y_LOW)
    planes = side_planes(KEEL_SIDE) + [((0, Y_KT, KEEL_Z_TOP), (0, -dz, 1), "zp"),
                                       ((0, Y_KT, -KEEL_Z_TOP), (0, -dz, -1), "zn")]
    planes += [(tuple(q[0]), tuple(n), "facet") for q, n in facets()]
    return K.polyhedron(planes)


def fork_cuts():
    slot = prism_z(SLOT, -FORK_IN, FORK_IN)
    clear = Pos(AXLE[0], AXLE[1], FORK_IN + 10) * Cylinder(R_AXLE_CLEAR, 20)
    tap = Pos(AXLE[0], AXLE[1], -FORK_IN - 10) * Cylinder(R_AXLE_TAP, 20)
    return [slot, clear, tap]


def holes():
    return [Pos(x, Y_TOP, z) * Cylinder(R_CLEAR, 40, rotation=(90, 0, 0)) for x, z in HOLES]


def fangs():
    out = []
    for sz in (1, -1):
        z0, z1 = FANG_Z
        dy = Y_KT - FANG[2][1]                            # pad to tip
        k = FANG_SPLAY / dy                               # dz per mm down
        pl = side_planes(FANG) + [((0, Y_KT, sz * z1), (0, k, sz), "o"), ((0, Y_KT, sz * z0), (0, -k, -sz), "i")]
        out.append(K.polyhedron(pl))
    return out


def teeth():
    """Raked-forward teeth on the front edge: base on the edge (2 mm into the
    keel), the forward face perpendicular to the edge, |z| tapering to the tip."""
    (x0, y0), (x1, y1) = FRONT
    d = unit([x1 - x0, y1 - y0])
    n = edge_normal(*FRONT)
    out = []
    for s0 in TEETH_S:
        b0 = np.array([x0, y0]) + s0 * d - 2.0 * n
        b1 = np.array([x0, y0]) + (s0 + TOOTH_L) * d - 2.0 * n
        tip = np.array([x0, y0]) + (s0 + TOOTH_L) * d + TOOTH_H * n
        zb, zt = TOOTH_Z
        pl = side_planes([tuple(b0), tuple(b1), tuple(tip)])
        for sz in (1, -1):              # z faces: zb at the base line, zt at the tip
            # plane through (b0, sz*zb), (b1, sz*zb), (tip, sz*zt)
            P0 = np.array([*b0, sz * zb]); P1 = np.array([*b1, sz * zb]); P2 = np.array([*tip, sz * zt])
            nn = np.cross(P1 - P0, P2 - P0)
            if nn[2] * sz < 0:
                nn = -nn
            pl.append((tuple(P0), tuple(nn), "tz"))
        out.append(K.polyhedron(pl))
    return out


def colour_tools():
    """graphite: the four crystal facets (inlaid panels); blue: the back ridge,
    the window lining on both flanks, a chevron on each pad flank."""
    from shapely.geometry import Polygon as SP
    dark = []
    for (q1, q2, q3), n in facets():
        u = unit(q3 - q1)
        v = np.cross(n, u)                   # K.carve's own local y (z_dir x x_dir)
        tri = SP([(0.0, 0.0), ((q2 - q1) @ u, (q2 - q1) @ v), ((q3 - q1) @ u, (q3 - q1) @ v)])
        tri = tri.buffer(-FACET_INSET, join_style=2)
        dark.append(K.carve((q1, u, v, n), list(tri.exterior.coords)[:-1], INLAY))
    blue = []
    (x0, y0), _ = BACK
    yb = BACK_END + 3.0                                         # stop above the fork
    x1, y1 = x_on(BACK, yb), yb
    nb = edge_normal(*BACK)
    t = 1.2                                                     # into the keel from the ridge
    band = [(x0 + 2 * nb[0], y0 + 2 * nb[1]), (x1 + 2 * nb[0], y1 + 2 * nb[1]),
            (x1 - t * nb[0], y1 - t * nb[1]), (x0 - t * nb[0], y0 - t * nb[1])]
    blue.append(prism_z(band, -RIDGE_W / 2, RIDGE_W / 2))
    from shapely.geometry import Polygon as SPg
    ring = SPg(WINDOW).buffer(LINER, join_style=2)
    for sz in (1, -1):                                          # flank z = +-16 is planar
        z0 = sz * KEEL_Z_TOP
        blue.append(prism_z(list(ring.exterior.coords)[:-1], min(z0 - sz * INLAY, z0 + sz * 2),
                            max(z0 - sz * INLAY, z0 + sz * 2)))
    # a blue chevron on each pad flank (|z| = 32 at the top, drafted), pointing back
    P = K.ccw(mirror_half(PAD_PLAN_HALF))
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        if abs(a[1]) < PAD_HALF_W - 0.1 or abs(b[1]) < PAD_HALF_W - 0.1:
            continue
        m = unit([b[1] - a[1], -(b[0] - a[0])])
        h = Y_TOP - Y_SEAT
        n = unit([h * m[0], -PAD_DRAFT, h * m[1]])
        o = np.array([max(a[0], b[0]), Y_TOP, a[1]])            # front-top corner of the flank
        fr = K.face_frame(o, n, np.array([-1.0, 0.0, 0.0]))
        vv = np.cross(fr[3], fr[1])
        sgn = 1.0 if vv[1] < 0 else -1.0                        # +v runs DOWN the flank
        chev = [(8.0, 3.4), (36.0, 3.4), (40.0, 5.3), (36.0, 7.2), (8.0, 7.2), (11.0, 5.3)]
        blue.append(K.carve(fr, [(x_, sgn * y_) for x_, y_ in chev], INLAY))
    return dark, blue


def build():
    fs, ts = fangs(), teeth()
    body = K.union([pad(), keel()] + fs + ts)
    for c in fork_cuts() + holes() + [prism_z(WINDOW, -40, 40)]:
        body = body - c
    dark, blue = colour_tools()
    Dt = K.union(dark + fs + ts)          # fangs and teeth are graphite through and through
    Bt = K.union(blue)
    Bl = body & Bt
    G = (body & Dt) - Bt
    W = body - Dt - Bt
    return {"TailStrut": [("white", W, WHITE), ("graphite", G, DARK), ("blue", Bl, BLUE)]}, body


# ------------------------------------------------------------------ checks
def checks(body):
    from shapely.geometry import Polygon, Point
    rows = []
    bb = body.bounding_box()
    rows.append(("lowest point y", bb.min.Y, f">= {Y_LOW}", bb.min.Y >= Y_LOW - 1e-6))
    rows.append(("highest point y", bb.max.Y, f"<= {Y_TOP}", bb.max.Y <= Y_TOP + 1e-6))
    rows.append(("half-width |z|", max(-bb.min.Z, bb.max.Z), "<= 40", max(-bb.min.Z, bb.max.Z) <= 40 + 1e-6))
    (x0, y0), (x1, y1) = TIP
    worst = -1e9
    for v in body.vertices():
        # signed distance BEHIND the tip line (positive = behind = bad)
        t = (v.Y - y0) / (y1 - y0)
        xl = x0 + t * (x1 - x0)
        if y0 - 10 <= v.Y <= y1:
            worst = max(worst, xl - v.X)
    rows.append(("behind the tip-over line (mm)", worst, "<= 0", worst <= 1e-6))
    caster = Pos(AXLE[0], AXLE[1], 0) * Cylinder(CASTER_R + 0.5, 9.0)
    ov = body & caster
    v = 0.0 if ov is None else ov.volume
    rows.append(("caster (r+0.5) overlap mm3", v, "0", v < 1e-6))
    for x, z in HOLES:
        hole = Pos(x, Y_TOP, z) * Cylinder(R_CLEAR - 0.01, 40, rotation=(90, 0, 0))
        ov = body & hole
        v = 0.0 if ov is None else ov.volume
        rows.append((f"hole ({x:.0f},{z:.0f}) material", v, "0", v < 1e-6))
        head = Pos(x, Y_SEAT - 10, z) * Cylinder(3.3, 20 - 0.01, rotation=(90, 0, 0))    # head + key, y -5..5
        ov = body & head
        v = 0.0 if ov is None else ov.volume
        rows.append((f"  head/key clearance below it", v, "0", v < 1e-6))
    return rows


def preview(parts, out):
    """Old vs new, in context (floor, back panel, bumper, caster)."""
    from render3d import tessellate
    from render_color import view, sheet
    ctx = []
    for f, rgb in (("v3_BottomPanel_d75", (0xC8, 0xCC, 0xD2)), ("FacetBack_d75", WHITE),
                   ("BumperBack_d75", TPU_GREY), ("v3_SupportWheel_d75", (0x30, 0x33, 0x38))):
        p = os.path.join(INPUT, f + ".step")
        if os.path.exists(p):
            ctx.append((import_step(p), rgb))
    old = import_step(os.path.join(INPUT, "v3_BackWheelSupport_d75.step"))

    def tb_of(items):
        tb, allV = [], []
        for shape, rgb in items:
            V, T, _ = tessellate(shape, 0.12)
            V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)
            tb.append((V, T, rgb))
            allV.append(V)
        return tb, np.vstack(allV)

    new_items = [(b, rgb) for n, b, rgb in parts["TailStrut"] if b is not None and b.solids()]
    tiles = []
    for name, items in (("v3 (now)", [(old, (0xB8, 0xB4, 0xE0))]), ("TAIL STRUT", new_items)):
        tb, allV = tb_of(ctx + items)
        # frame on the strut + caster + the panel's lower part, not the whole box
        foc = np.vstack([tb_of(items)[1], tb_of(ctx[-1:])[1]])
        foc = np.vstack([foc, foc.min(0) - [8, 8, 0], foc.max(0) + [8, 8, 30]])
        for t, e, a in (("rear", 6, 180), ("rear 3/4, low", -14, -145), ("side", 0.5, -90), ("under 3/4", -40, -120)):
            tiles.append((f"{name} -- {t}", "", view(tb, foc, (620, 520), e, a, light="camera")))
    sheet(tiles, 4, out, header="Back support: v3 vs TAIL STRUT", size=(620, 520))


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    parts, body = build()
    old = import_step(os.path.join(INPUT, "v3_BackWheelSupport_d75.step"))
    tot = 0.0
    for n, b, _ in parts["TailStrut"]:
        if b is None:
            continue
        v = sum(s.volume for s in b.solids())
        tot += v
        print(f"  {n:9s} {v / 1000:7.2f} cm3  {len(b.solids())} solid(s)")
    print(f"  total {tot / 1000:.2f} cm3 (body {body.volume / 1000:.2f}); v3 support {old.volume / 1000:.2f} cm3")
    bad = 0
    for what, v, want, ok in checks(body):
        bad += not ok
        print(f"    {what:32s} {v:10.3f}  want {want:8s} {'ok' if ok else 'FAIL'}")
    print("  CHECKS " + ("PASS" if not bad else f"FAIL ({bad})"))
    stepcolor.write([(n, b, rgb) for n, b, rgb in parts["TailStrut"] if b is not None and b.solids()],
                    os.path.join(out, "TailStrut.step"), part="TailStrut")
    preview(parts, os.path.join(out, "preview_tail_strut.png"))
    if "--3mf" in sys.argv:                       # print orientation: pad down (robot -y is up)
        import box_facet_print as BF
        BF.PRINT_UP = {"TailStrut": (0.0, -1.0, 0.0)}
        BF.write_print_3mf(parts, sys.argv[sys.argv.index("--3mf") + 1])
