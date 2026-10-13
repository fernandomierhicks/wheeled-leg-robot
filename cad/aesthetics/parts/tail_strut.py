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
PRINT (his call 2026-10-12): the Front plane (XY) on the bed -- a layer is a z slice,
so the keel's bending runs along the layers, and every colour costs a filament change
in each layer it appears in.  Hence colour boundaries at z = const (below) and the two
long pad flanks + the fangs' outer faces FLAT at z = +-32: the part sits on one of them.
Supports: under the keel flank (16 mm above the bed) and the upper fang.

Parts of the shape:
  PAD     10 mm plate, chevron nose forward, chamfered heel; nose / heel drafted,
          the long flanks flat
  KEEL    the blade from the pad to the fork: swept back edge and front edge,
          both ridged (two facets meeting on the centre line -- a crystal keel),
          a V window through it, all planar
  FORK    plain block round the axle, the caster's slot (faceted arch)
  COLOUR  white body (facets and teeth too: they read by shading); |z| >= BAND_Z
          graphite through -- the pad's outer band and the fangs, whole layers of one
          colour; blue: on the flat flanks the chevron + trapezoids cut TOP_PRESS deep on
          blue floors; on the keel flanks the window lining + trapezoids; the ridge line;
          and a seam -- the skin where the surface crosses |z| <= SEAM_H (the ridge
          line's layers).  Accents INLAY = 0.6 deep (3 layers), as on every part; every
          blue face is parallel to the bed, so the floors share a few layers.
          (2026-10-06..12 the facets / teeth were graphite and a hose ran on the
          under facet: ~230 of 340 layers multi-colour; now 39.)
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
PAD_DRAFT = 2.0                               # nose / heel sides inset this much at the pad underside
FLANK_DRAFT = 0.0                             # the long flanks (|z| = 32): flat, a face to print on
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
# fangs: a raked fin under each outer edge of the pad, |z| 25..32 (clear of the
# screw keys at |z| <= 23.3), pointing down and back
FANG = [(-102.0, Y_KT), (-150.0, Y_KT), (-160.0, -12.0)]
FANG_Z = (25.0, PAD_HALF_W)                   # |z| at the pad; the outer face stays flat at 32 (flush
FANG_SPLAY = 4.0                              # with the flank), the inner one splays out this much at the tip
# teeth along the front (under) edge, raked forward: at edge length s from the fork
TEETH_S, TOOTH_L, TOOTH_H, TOOTH_Z = (27.0, 37.0, 47.0), 7.0, 4.5, (9.0, 4.0)
INLAY = 0.6                                   # colour inlay depth (3 layers, his rule 2026-10-10)
RIDGE_W = 1.4                                 # blue line on the back ridge, full width
BAND_Z = 25.0                                 # |z| >= this: graphite through (pad band + fangs)
SEAM_H = 0.6                                  # the seam: |z| <= this (inside the ridge line's +-0.7)
# blue trapezoids on the two bed-parallel faces (his ask 2026-10-12), both sides, (x, y) mm:
#   TOP  = the flat flanks z = +-32 (graphite): cut TOP_PRESS deep on an INLAY blue floor -- the chevron too;
#   MID  = the keel flanks z = +-16 (white): flush INLAY here; 28 presses them like the window lining.
# Each is clipped to its face less ACC_MARGIN (edges, other colours, the axle head), and kept >= ACC_MIN_W.
TOP_PRESS = 2.0
ACC_MARGIN, ACC_MIN_W = 1.2, 0.9
FANG_ACC_MIN_Y = -8.8                         # fang thickness 7 - 0.229 (5.5 - y) >= press + floor + 1 + margin
CHEVRON = [(-108.0, 11.6), (-136.0, 11.6), (-140.0, 9.7), (-136.0, 7.8), (-108.0, 7.8), (-111.0, 9.7)]
TOP_SHAPES = [
    [(-102.4, 11.6), (-105.6, 11.6), (-106.4, 7.8), (-103.2, 7.8)],              # bracket ahead of the chevron
    [(-139.0, 11.6), (-147.8, 11.6), (-147.0, 7.8), (-139.0, 7.8), (-143.0, 9.7)],   # block behind it, a V echo
    [(-109.0, 6.6), (-134.0, 6.6), (-129.0, 1.6), (-120.0, 1.6)],                 # keel plate under it
]
FANG_BARS = [(1.2, 3.2, -1.0, 18.0), (4.4, 6.6, -4.0, 14.0)]    # off the fang's lower edge: (w0, w1, s0, s1)
MID_UPPER = ((-143.6, -7.08), (-173.15, -21.98))                # the keel flank's upper (back facet) edge
MID_LANE = (1.5, 27.0, 1.2, 3.0)                                # under it: (s0, s1, w0, w1)
MID_SLASHES = (8.0, 13.5, 19.0)                                 # raked slashes across the strip, at s
MID_PLATE = (23.0, 32.0, 4.4, 9.0)                              # a plate above the axle


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


def is_flank(a, b):
    """The plan edge a -> b is one of the two long flanks (|z| = PAD_HALF_W at both ends)."""
    return abs(a[1]) > PAD_HALF_W - 0.1 and abs(b[1]) > PAD_HALF_W - 0.1


def pad():
    """Full plan at the top face, the nose / heel sides PAD_DRAFT inset at the underside,
    the long flanks FLANK_DRAFT (flat).
    (K.frustum_walls is built narrow-end-up; this one is narrow-end-down.)"""
    P = K.ccw(mirror_half(PAD_PLAN_HALF))
    h = Y_TOP - Y_SEAT
    walls = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        m = unit([b[1] - a[1], -(b[0] - a[0])])          # outward in the (x, z) plan
        dr = FLANK_DRAFT if is_flank(a, b) else PAD_DRAFT
        walls.append(((a[0], Y_TOP, a[1]), (h * m[0], -dr, h * m[1]), "pad"))
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
        k = FANG_SPLAY / dy                               # dz per mm down (the inner face)
        pl = side_planes(FANG) + [((0, Y_KT, sz * z1), (0, 0, sz), "o"), ((0, Y_KT, sz * z0), (0, -k, -sz), "i")]
        out.append(K.polyhedron(pl))
    return out


def band_tool():
    """|z| >= BAND_Z, both sides, well past the part."""
    return [Pos(-135.0, -10.0, sz * (BAND_Z + 10.0)) * Box(140.0, 100.0, 20.0) for sz in (1, -1)]


def band_shape():
    """The pad's outer band + the fangs, |z| >= BAND_Z, the V window through them (as through
    everything): the part's every body out there (the keel and teeth stay inside)."""
    return (K.union([pad()] + fangs()) & K.union(band_tool())) - prism_z(WINDOW, -40, 40)


def band_bodies():
    """(graphite, blue): the band with every TOP accent cut TOP_PRESS deep, and their INLAY
    blue floors."""
    from shapely.ops import unary_union
    band = band_shape()
    reg = unary_union(top_accents(band))
    rec, fl = [], []
    for sz in (1, -1):
        zf = sz * PAD_HALF_W
        rec.append(prism_poly(reg, *sorted((zf - sz * TOP_PRESS, zf + sz * 3))))
        fl.append(prism_poly(reg, *sorted((zf - sz * TOP_PRESS, zf - sz * (TOP_PRESS + INLAY)))))
    floors = band & K.union(fl)
    return band - K.union(rec) - K.union(fl), floors


def _trap(p0, d, n, s0, s1, w0, w1, rake0, rake1):
    """A trapezoid in a line frame (origin p0, unit d along, unit n across): s0..s1 along at w0,
    each end moving rake * (w - w0) along d as w grows (45 deg = 1)."""
    from shapely.geometry import Polygon
    p0, d, n = (np.asarray(v, float) for v in (p0, d, n))
    P = lambda s, w: tuple(p0 + s * d + w * n)
    return Polygon([P(s0, w0), P(s1, w0), P(s1 + rake1 * (w1 - w0), w1), P(s0 + rake0 * (w1 - w0), w1)])


def _fit(shapes, use):
    """Each shape clipped to `use`, opened by ACC_MIN_W (nothing narrower), crumbs dropped."""
    from shapely.geometry import Polygon
    out = []
    for s in shapes:
        g = Polygon(s).intersection(use) if not hasattr(s, "geom_type") else s.intersection(use)
        g = g.buffer(-ACC_MIN_W / 2, join_style=2).buffer(ACC_MIN_W / 2, join_style=2)
        out += [p.simplify(0.01) for p in getattr(g, "geoms", [g]) if isinstance(p, Polygon) and p.area >= 4.0]
    return out


def top_accents(band):
    """(x, y) regions on the flat flanks z = +-32 (the same on both: the part is symmetric):
    the chevron + TOP_SHAPES + FANG_BARS, inside the face less ACC_MARGIN, the bars kept off the
    fang's thin tip (FANG_ACC_MIN_Y)."""
    from shapely.geometry import Polygon
    face = section_xy([band], PAD_HALF_W - 0.01)
    chev = Polygon(CHEVRON)
    use = face.buffer(-ACC_MARGIN, join_style=2).difference(chev.buffer(ACC_MARGIN, join_style=1))
    use = use.intersection(Polygon([(-200, FANG_ACC_MIN_Y), (-80, FANG_ACC_MIN_Y), (-80, 30), (-200, 30)]))
    h0, h1 = np.array(FANG[0]), np.array(FANG[2])                # the fang's lower edge
    dh = unit(h1 - h0)
    nh = np.array([dh[1], -dh[0]])
    nh = nh if nh[1] > 0 else -nh                                # up, into the fang
    hw = h0 + (-143.61 - h0[0]) / dh[0] * dh                     # where the window's back edge leaves it
    bars = [_trap(hw, dh, nh, s0, s1, w0, w1, -1.0, -1.0) for w0, w1, s0, s1 in FANG_BARS]
    return [chev] + _fit(TOP_SHAPES + bars, use)


def mid_accents(body):
    """(x, y) regions on the keel flanks z = +-16: a lane under the upper edge, slashes raked
    toward the fork, a plate above the axle -- inside the face less ACC_MARGIN, clear of the
    window lining and of the axle screw head (r 3.5 + margin)."""
    from shapely.geometry import Polygon, Point
    face = section_xy([body], KEEL_Z_TOP - 0.01).difference(section_xy([body], KEEL_Z_TOP + 0.01).buffer(0.01))
    lining = Polygon(WINDOW).buffer(LINER, join_style=2)
    use = face.buffer(-ACC_MARGIN, join_style=2).difference(lining.buffer(ACC_MARGIN, join_style=2))
    use = use.difference(Point(AXLE).buffer(3.5 + ACC_MARGIN))
    u0, u1 = (np.array(p) for p in MID_UPPER)
    du = unit(u1 - u0)
    n = np.array([du[1], -du[0]])
    n = n if n[1] < 0 else -n                                    # into the strip
    shapes = [_trap(u0, du, n, *MID_LANE, 1.0, -1.0)]
    lean = unit(n + 0.8 * du)
    for k, s in enumerate(MID_SLASHES):
        a, b = u0 + s * du + 4.4 * n, u0 + (s + 2.4 + 0.3 * k) * du + 4.4 * n
        shapes.append(Polygon([tuple(a), tuple(b), tuple(b + 30 * lean), tuple(a + 30 * lean)]))
    shapes.append(_trap(u0, du, n, *MID_PLATE, 0.6, -0.6))
    return _fit(shapes, use)


def prism_poly(poly, z0, z1):
    """A shapely (Multi)Polygon, holes and all, as a prism z0..z1."""
    out = None
    for p in getattr(poly, "geoms", [poly]):
        if p.is_empty or p.area < 0.01:
            continue
        s = prism_z(list(p.exterior.coords)[:-1], z0, z1)
        for h in p.interiors:
            s = s - prism_z(list(h.coords)[:-1], z0 - 1, z1 + 1)
        out = s if out is None else out + s
    return out


def section_xy(solids, z):
    """The section of the solids at height z, as one shapely region (union of the
    section faces' triangles -- no 3D fuse: OCC fails fusing many touching bodies)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from render3d import tessellate
    cut = Pos(-135.0, -10.0, z) * Box(140.0, 100.0, 0.002)
    tris = []
    for s in solids:
        piece = s & cut
        if piece is None or not piece.solids():
            continue
        V, T, _ = tessellate(piece, 0.02)
        tris += [Polygon(V[t][:, :2]) for t in T if abs(V[t][:, 2] - (z + 0.001)).max() < 1e-4]
    return unary_union([t.buffer(1e-4) for t in tris if t.area > 1e-9]).buffer(-1e-4)


def seam_tool(sec):
    """The seam's cutter, from the part's section at |z| = SEAM_H: the slab |z| <= SEAM_H
    less the section shrunk by INLAY (the core) and less everything within INLAY of the
    top face (it sits on the floor, mated: stays white).  What a body loses to it is its
    skin in the slab."""
    slab = Pos(-135.0, -10.0, 0.0) * Box(140.0, 100.0, 2 * SEAM_H)
    core = prism_poly(sec.buffer(-INLAY, join_style=2), -SEAM_H - 1, SEAM_H + 1)
    # 0.2 past the core's own top (y 14.4): on it exactly, the two pinch an edge at the nose
    top = Pos(-135.0, Y_TOP - INLAY - 0.2 + 25.0, 0.0) * Box(140.0, 50.0, 2 * SEAM_H + 2)
    return slab - core - top


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


def colour_tools(body):
    """blue, inlaid INLAY: the back ridge, the window lining and the MID accents on both keel
    flanks.  (The band is graphite by z alone, its TOP accents pressed: band_bodies.)"""
    from shapely.ops import unary_union
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
    mid = mid_accents(body)
    for sz in (1, -1):                                          # flank z = +-16 is planar
        z0 = sz * KEEL_Z_TOP
        blue.append(prism_z(list(ring.exterior.coords)[:-1], min(z0 - sz * INLAY, z0 + sz * 2),
                            max(z0 - sz * INLAY, z0 + sz * 2)))
        blue.append(prism_poly(unary_union(mid), *sorted((z0 - sz * INLAY, z0 + sz * 2))))
    return blue


def body_shape():
    """The whole part, one solid, before colour."""
    body = K.union([pad(), keel()] + fangs() + teeth())
    for c in fork_cuts() + holes() + [prism_z(WINDOW, -40, 40)]:
        body = body - c
    return body


def build():
    body = body_shape()
    Bt = K.union(colour_tools(body))
    G, Ba = band_bodies()                 # the band + fangs: graphite through, by z alone; TOP accents
    rest = body - K.union(band_tool())
    Bl = rest & Bt
    W = rest - Bt
    sk = W & seam_tool(section_xy([body], SEAM_H - 0.01).intersection(section_xy([body], -SEAM_H + 0.01)))
    W = W - sk
    # the seam its own body: fused with the ridge line it meets them along an edge (non-manifold)
    return {"TailStrut": [("white", W, WHITE), ("graphite", G, DARK), ("blue", Bl, BLUE), ("accents", Ba, BLUE),
                          ("seam", sk, BLUE)]}, body


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
    worst = 1.0                                   # 1 mm of solid behind every TOP floor (the fang thins out)
    for p in top_accents(band_shape()):
        for sz in (1, -1):
            zf = sz * (PAD_HALF_W - TOP_PRESS - INLAY)
            probe = prism_poly(p.buffer(-0.05, join_style=2), *sorted((zf - sz * 0.02, zf - sz * 1.0)))
            ov = body & probe
            worst = min(worst, (0.0 if ov is None else ov.volume) / probe.volume)
    rows.append(("solid 1 mm behind the top floors", worst, ">= 0.98", worst >= 0.98))
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
    if "--3mf" in sys.argv:                       # print orientation: the Front plane on the bed (+z up)
        import box_facet_print as BF
        BF.PRINT_UP = {"TailStrut": (0.0, 0.0, 1.0)}
        BF.write_print_3mf(parts, sys.argv[sys.argv.index("--3mf") + 1])
