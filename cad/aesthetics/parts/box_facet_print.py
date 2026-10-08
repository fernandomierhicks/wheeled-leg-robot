"""Body box D FACET -- the PRINT parts (2026-10-04).  Concept D (box_concept_facet.py)
split into printable, bolt-together parts.  ROBOT coords, mm, with Distance1 = 75
(RobotMount inner faces at z = +-75, outer faces at +-80, the box symmetric about
z = 0 -- his call, so the v3 floor is reused without a reprint).

    C:/Users/ferna/cadenv/Scripts/python.exe cad/aesthetics/parts/box_facet_print.py <out_dir> [--3mf <print_dir>]

writes <Part>.step (colour bodies via lib/stepcolor), preview_facet_print.png and a
fastener list.  Parts (print orientation in brackets):

  FacetFront   plate (the v3 FrontPanel itself: screen window, M2 + bracket holes)
               + solid faceted relief to x 65, screen well with a dark lining  [plate down]
  FacetBack    plate (the v3 BackPanel itself: switch, LED, buzzer, 2 USB slots,
               bracket holes) + solid relief to x -164, a dark I/O bay (USB) and a
               dark control pod (switch, LED, buzzer), bolt wells           [plate down]
  BumperFront  grey TPU: two cheek posts + chin bar, 6 mm proud of the face,
  BumperBack   held by long M3 through the side bracket holes              [face down]
  FacetRing    graphite: flange on the box top, screwed to the 9 top corner
               brackets (the v3 Cap's holes), upright wall with 10 self-tap
               bosses for the hood                                          [flange down]
  FacetHood    skirt + tier 1 + the strip channel + tier 2 + armour; 10 M3
               countersunk through the skirt into the ring             [skirt down, tree
               supports inside for the roof only].  Tier 2's top is variant B, PLATES
               (hood_variants.py, his pick 2026-10-07); hood_details() below is the
               earlier top, kept as hood_variants' "today" reference
  NeopixelStrip  NOT printed: the 5.1 x 2.7 strip on its route, for fit + length

Fasteners (his rules, 2026-10-04): no heat-set inserts; M3 into the existing
corner-bracket nuts or into undersized holes (Ø2.8, as his v3 Cap/Cage) as
self-tapping threads; visible heads are welcome.

Every solid is solid on purpose: the slicer infills it, so no trapped supports.
Anything that needs a thin panel (switch nut, USB plugs, M2 screen screws, the
centre bracket screws) sits at the bottom of a well down to the 3 mm plate.
"""
import os
import sys
import math
import numpy as np
from build123d import Box, Pos, Plane, Polyline, make_face, extrude, Cylinder, Cone, Align, import_step
import shapely.geometry as sg
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)
import stepcolor
import box_concept_facet as K           # the concept's geometry kit (polyhedron, carve, pick, ...)

WHITE, GRAPHITE, DARK, BLUE = K.WHITE, K.GRAPHITE, K.DARK, K.BLUE
TPU_GREY = (0x6E, 0x73, 0x7A)
STRIP = (0xFF, 0xF4, 0xC8)
INPUT = os.path.join(HERE, "..", "input", "box")
BIG = K.BIG

# ------------------------------------------------------------------ the frame
ZW, ZO = 75.0, 80.0                 # RobotMount inner / outer face
Z_HOOD = 78.0                       # nothing above the box wider than this (femur plates >= 81.9)
Y0, Y1 = 15.0, 75.0                 # floor underside, box top
XF0, XF1 = 50.0, 53.0               # front plate (v3 FrontPanel)
XB0, XB1 = -150.0, -153.0           # back plate  (v3 BackPanel)
FACE_F, FACE_B = 65.0, -164.0       # concept D lower-body faces
TPU_T = 6.0                         # bumper stands this proud of the face
WALL = 2.5
SELF_TAP = 1.4                      # Ø2.8 pilot -- his v3 Cap/Cage value for M3
CLEAR = 1.7                         # Ø3.4 M3 clearance
CSK = 3.2                           # M3 flathead countersink radius at the surface
CBORE = 3.25                        # Ø6.5 counterbore for M3 socket heads in TPU

# hood stack (y)
Y_RING_T = 78.0                     # ring flange top = hood skirt bottom
Y_SKIRT = 86.0                      # skirt top = tier 1 base
Y_RING_W = Y_SKIRT - 0.5            # ring wall top: clear of tier 1's inner edge (no double seat)
FIT = 0.3                           # skirt-to-ring-wall gap (FDM, 700 mm perimeter)
Y_LEDGE = 94.0                      # tier 1 top
Y_CHAN = 100.0                      # strip channel top = tier 2 base
Y_TOP = 113.0                       # tier 2 top
STRIP_W, STRIP_T = 5.1, 2.7         # his strip incl. the silicone diffuser
STRIP_Y0 = Y_LEDGE + 0.45
CHAN_WALL = 4.0
LIP = 1.2                           # tier 2 overhangs the strip's back by this much

# front / back panel plans (x, z), half outlines, front to back
FRONT_HALF = [(FACE_F, 0), (FACE_F, 46), (59, 64), (XF1, ZO), (XF0, ZO), (XF0, 0)]
BACK_HALF = [(XB0, 0), (XB0, ZO), (XB1, ZO), (-158, 66), (FACE_B, 48), (FACE_B, 0)]
FRONT_SIDE = [(XF0, Y0), (55, Y0), (FACE_F, 29.5), (FACE_F, 73.8), (63.8, Y1), (XF0, Y1)]
BACK_SIDE = [(XB0, Y0), (-152, Y0), (-161, 27), (FACE_B, 34), (FACE_B, 73.8), (-162.8, Y1), (XB0, Y1)]
BODY_HALF = [(FACE_F, 0), (FACE_F, 46), (59, 64), (XF1, ZO), (XB1, ZO), (-158, 66), (FACE_B, 48), (FACE_B, 0)]


# ------------------------------------------------------------------ 2D / 3D helpers
def full(half):
    return K.mirror(half)


def spoly(pts):
    return sg.Polygon(pts)


def soff(p, d):
    """Mitred offset of a shapely polygon (d < 0 = in)."""
    return p.buffer(d, join_style=2, mitre_limit=10.0)


def pts(p):
    return [(float(x), float(z)) for x, z in list(p.exterior.coords)[:-1]]


def prism_xz(p, y0, y1):
    """Plan polygon (x, z) extruded from y0 up to y1."""
    P = pts(p) if hasattr(p, "exterior") else p
    pl = Plane(origin=(0, y1, 0), x_dir=(1, 0, 0), z_dir=(0, -1, 0))     # local (u, v) = (x, z)
    return extrude(pl * make_face(Polyline(*P, close=True)), amount=y1 - y0)


def prism_xy(P, z0, z1):
    """Side profile (x, y) extruded from z0 to z1."""
    pl = Plane(origin=(0, 0, z0), x_dir=(1, 0, 0), z_dir=(0, 0, 1))
    return extrude(pl * make_face(Polyline(*P, close=True)), amount=z1 - z0)


def prism_x(P_zy, x0, x1):
    """Elevation polygon (z, y) extruded from x0 to x1 (x1 > x0)."""
    pl = Plane(origin=(x0, 0, 0), x_dir=(0, 0, -1), z_dir=(1, 0, 0))     # local (u, v) = (-z, y)
    return extrude(pl * make_face(Polyline(*[(-z, y) for z, y in P_zy], close=True)), amount=x1 - x0)


def rect_zy(z0, z1, y0, y1, c=0.0):
    return [(z, y) for z, y in K.chamfer_rect(z0, z1, y0, y1, c)] if c else [(z0, y0), (z1, y0), (z1, y1), (z0, y1)]


def axis_solid(o, d, shape):
    """`shape` built along +Z from the origin, placed with its Z on direction d at o."""
    d = K.unit(d)
    x = np.cross(d, [0, 1, 0]) if abs(d[1]) < 0.9 else np.cross(d, [1, 0, 0])
    return Plane(origin=tuple(map(float, o)), x_dir=tuple(map(float, K.unit(x))), z_dir=tuple(map(float, d))) * shape


def bore(o, d, r, depth, back=1.0):
    """Cylinder of radius r from `back` mm before o to depth after it, along d."""
    o = np.asarray(o, float) - back * K.unit(d)
    return axis_solid(o, d, Cylinder(r, depth + back, align=(Align.CENTER, Align.CENTER, Align.MIN)))


def csk(o, d, r_top=CSK, r_hole=CLEAR, back=0.5):
    """90-degree countersink entering at o along d."""
    o = np.asarray(o, float) - back * K.unit(d)
    h = r_top - r_hole
    return axis_solid(o, d, Cone(r_top + back, r_hole, h + back, align=(Align.CENTER, Align.CENTER, Align.MIN)))


def union(parts):
    parts = [p for p in parts if p is not None]
    return K.union(parts) if parts else None


def solids_volume(s):
    return sum(x.volume for x in s.solids()) if s is not None else 0.0


def hole_axes(plate, r=CLEAR):
    """Centres (y, z) of every x-axis cylinder of radius r in a plate -- the
    inherited bracket-screw pattern, read off the v3 panel itself."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from build123d import GeomType
    out = []
    for f in plate.faces():
        if f.geom_type != GeomType.CYLINDER:
            continue
        cyl = BRepAdaptor_Surface(f.wrapped).Cylinder()
        if abs(cyl.Radius() - r) > 0.01:
            continue
        a = cyl.Axis()
        if abs(abs(a.Direction().X()) - 1) > 1e-6:
            continue
        p = (round(a.Location().Y(), 3), round(a.Location().Z(), 3))
        if not any(abs(p[0] - q[0]) < 0.01 and abs(p[1] - q[1]) < 0.01 for q in out):
            out.append(p)
    return sorted(out)


# ------------------------------------------------------------------ outlines
BODY = spoly(full(BODY_HALF))
H0 = soff(BODY, 2.0).intersection(sg.box(-BIG, -Z_HOOD, BIG, Z_HOOD))          # hood skirt outline
H0 = sg.Polygon(pts(H0)).simplify(0.01)

# the v3 Cap's 9 bracket screws (vertical, into the top corner brackets), d75 frame
CAP_SCREWS = [(-140.0, 0.0), (-145.0, -70.0), (-145.0, 70.0), (-83.5, -65.0), (-83.5, 65.0),
              (-9.0, -65.0), (-9.0, 65.0), (45.0, -70.0), (45.0, 70.0)]
# hood -> ring screws, on the skirt at y = 82: (x, z, outward normal)
HOOD_SCREWS = [(x, s * Z_HOOD, (0, 0, s)) for s in (1, -1) for x in (-120.0, -45.0, 25.0)] + \
              [(None, s * 32.0, (1, 0, 0)) for s in (1, -1)] + [(None, s * 32.0, (-1, 0, 0)) for s in (1, -1)]
Y_HOOD_SCREW = 82.0


def skirt_point(x, z, n):
    """Point on the skirt outline H0 where the screw goes in (x or z given)."""
    if n[2]:
        return np.array([x, Y_HOOD_SCREW, z])
    xs = [p[0] for p in pts(H0)]
    xe = max(xs) if n[0] > 0 else min(xs)
    return np.array([xe, Y_HOOD_SCREW, z])


# ------------------------------------------------------------------ front + back
def panel_envelope(half, side, z_extra=0.0):
    return prism_xz(spoly(full(half)), Y0, Y1) & prism_xy(side, -BIG / 4, BIG / 4)


def front_parts():
    plate = import_step(os.path.join(INPUT, "v3_FrontPanel_d75.step"))
    env = panel_envelope(FRONT_HALF, FRONT_SIDE)
    relief = env - prism_xz(sg.box(XF0 - 1, -ZO - 1, XF1, ZO + 1), Y0 - 1, Y1 + 1)   # x >= 53 only
    screws = hole_axes(plate)                                 # 8 bracket holes (y, z)
    # screen well: floor = the plate face (x 53), straight walls, dark lining
    WELL = (-41.0, 41.0, 29.5, 73.0, 6.0)
    well = prism_x(rect_zy(*WELL), XF1 - 0.01, FACE_F + 1)
    lining_zone = prism_x(rect_zy(WELL[0] - 2, WELL[1] + 2, WELL[2] - 2, min(WELL[3] + 2, Y1), WELL[4] + 1),
                          XF1, FACE_F + 1) - well
    relief = relief - well
    # every bracket hole continues through the relief (bumper screws or wells)
    for y, z in screws:
        relief = relief - bore((FACE_F + 5, y, z), (-1, 0, 0), CLEAR, 20)
    dark = relief & lining_zone
    white = (plate + relief) - lining_zone
    # blue accents: a vertical trace with a dogleg each side of the well
    accents = []
    for s in (1, -1):
        f = K.face_frame((FACE_F, 0, 0), (1, 0, 0), (0, 0, -1))              # u = -z, v = y
        band = K.polyline_band([(-s * 48, 34), (-s * 48, 52), (-s * 51, 56), (-s * 51, 69)], 1.4)
        accents.append(K.carve(f, band, 0.8))
    blue = union(accents) & white
    white = white - blue
    tpu = bumper(env, FACE_F, +1, screws, cheek=(56.0, 74.0), chin_top=lambda z: 29.5)
    return {"white": white, "dark": dark, "blue": blue}, tpu, screws


def back_parts():
    plate = import_step(os.path.join(INPUT, "v3_BackPanel_d75.step"))
    env = panel_envelope(BACK_HALF, BACK_SIDE)
    relief = env - prism_xz(sg.box(XB1, -ZO - 1, XB0 + 1, ZO + 1), Y0 - 1, Y1 + 1)   # x <= -153 only
    screws = hole_axes(plate)
    # I/O bay over both USB slots, control pod over switch + LED + buzzer: dark-lined
    # wells down to the plate face (x -153) so plugs and the switch nut see 3 mm
    BAY = (-40.0, 9.0, 42.0, 60.5, 4.0)
    POD = (18.0, 58.0, 29.0, 71.0, 3.0)                      # switch bezel is 23 x 25: small chamfers
    wells, lining = [], []
    for z0, z1, y0, y1, c in (BAY, POD):
        w = prism_x(rect_zy(z0, z1, y0, y1, c), FACE_B - 1, XB1 + 0.01)
        wells.append(w)
        lining.append(prism_x(rect_zy(z0 - 2, z1 + 2, y0 - 2, y1 + 2, c + 1), FACE_B - 1, XB1) - w)
    relief = relief - union(wells)
    lining_zone = union(lining)
    # centre screws outside the bays: (65, 0) gets a bolt well down to the plate
    # (the v3 M3x8 flathead stays); (30, 0) goes under the bumper's chin tab
    for y, z in screws:
        if abs(z) < 1 and y > 50:
            relief = relief - bore((FACE_B - 1, y, z), (1, 0, 0), 3.75, (XB1 - FACE_B) + 1 - 0.01)
        relief = relief - bore((FACE_B - 5, y, z), (1, 0, 0), CLEAR, 20)
    dark = relief & lining_zone
    white = (plate + relief) - lining_zone
    # three graphite vent slashes left of the bay (blind, 2 mm), a blue brow trace
    bf = K.face_frame((FACE_B, 48, 0), (-1, 0, 0), (0, 0, 1))                  # u = +z, v = y - 48
    vents = union([K.carve(bf, K.slash(uc, 0, 3.0, 22, 3), 2.0) for uc in (-55.0, -50.5, -46.0)]) & white
    trace = K.carve(bf, K.polyline_band([(-40, 19.5), (-10, 19.5), (-6, 23.5), (14, 23.5)], 1.4), 0.8) & white
    white = white - vents - trace

    def chin_top(z):
        return 34.5 if abs(z) < 14 else 28.0
    tpu = bumper(env, FACE_B, -1, screws, cheek=(60.0, 74.0), chin_top=chin_top)
    return {"white": white, "dark": dark, "graphite": vents, "blue": trace}, tpu, screws


def bumper(env, face, sx, screws, cheek, chin_top):
    """Grey TPU U-frame: cheek posts |z| in cheek, full height, and a chin bar
    between them, TPU_T proud of the face; back faces conform to the white part.
    Held by long M3 socket heads through the side bracket holes (counterbored,
    3 mm of TPU under each head)."""
    half = FRONT_HALF if sx > 0 else BACK_HALF
    grown = soff(spoly(full(half)), TPU_T).intersection(sg.box(-BIG, -ZO + 0.5, BIG, ZO - 0.5))
    side = FRONT_SIDE if sx > 0 else BACK_SIDE
    # side profile grown forward by TPU_T, bottom kept at y 15, top 0.5 under the ring
    xs = [p[0] for p in side]
    xf = max(xs) if sx > 0 else min(xs)
    plate_x = XF1 if sx > 0 else XB1
    prof = [(plate_x, Y0), (xf + sx * TPU_T, Y0), (xf + sx * TPU_T, Y1 - 0.5), (plate_x, Y1 - 0.5)]
    outer = prism_xz(grown, Y0, Y1 - 0.5) & prism_xy(prof, -BIG / 4, BIG / 4)
    c0, c1 = cheek
    cheeks = union([prism_xz(sg.box(-BIG, s * c0, BIG, s * c1) if s > 0 else sg.box(-BIG, -c1, BIG, -c0), Y0, Y1)
                    for s in (1, -1)])
    zs = np.linspace(-c0, c0, 57)
    # the chin's top edge, with 45-degree jogs where its height changes
    chin_pts = [(-c0, Y0 - 1), (c0, Y0 - 1)]
    tops = sorted({chin_top(z) for z in zs})
    if len(tops) == 1:
        chin_pts += [(c0, tops[0]), (-c0, tops[0])]
    else:
        lo, hi = tops
        zj = 14.0
        chin_pts += [(c0, lo), (zj + (hi - lo), lo), (zj, hi), (-zj, hi), (-zj - (hi - lo), lo), (-c0, lo)]
    chin = prism_x(chin_pts, -BIG, BIG)
    tpu = (outer & (cheeks + chin)) - env
    # 1.5 mm chamfer round the outer face: four planes tilted 45 deg off the face rectangle
    xface = xf + sx * TPU_T
    cham = []
    for n_zy, off in (((0, 1), Y1 - 0.5), ((0, -1), -Y0), ((1, 0), c1), ((-1, 0), c1)):
        mz, my = n_zy
        pt = (xface, off * my if my else 0.0, off * mz if mz else 0.0)
        pt = (pt[0] - sx * 1.5, pt[1], pt[2])
        cham.append((pt, (sx * 1.0, my * 1.0, mz * 1.0), "c"))
    tpu = tpu & K.polyhedron(cham)
    # screws: the six side bracket holes + the centre-bottom one under the chin
    for y, z in screws:
        if abs(z) > 1 or y < 35:
            entry = np.array([xf + sx * TPU_T, y, z])
            tpu = tpu - bore(entry, (-sx, 0, 0), CLEAR, 40)
            # TPU thickness along this axis: from the white surface to the face
            col = (env & bore(entry, (-sx, 0, 0), 0.2, 40, back=0.5))
            bb = col.bounding_box()
            white_x = bb.max.X if sx > 0 else bb.min.X
            depth = abs(entry[0] - white_x) - 3.0
            tpu = tpu - bore(entry, (-sx, 0, 0), CBORE, depth)
    return tpu


# ------------------------------------------------------------------ ring + hood
def t1_planes():
    P = pts(H0)
    w = K.frustum_walls(P, Y_SKIRT, Y_LEDGE, K.insets_by_direction(P, 6.0, 6.0, 6.0, 8.0), "t1")
    return w + [((0, Y_LEDGE, 0), (0, 1, 0), "ledge"), ((0, Y_SKIRT, 0), (0, -1, 0), "base")]


def t1_top():
    """Tier 1's top outline (the ledge's outer edge)."""
    s = K.polyhedron(t1_planes()) & K.slab_y(Y_LEDGE - 0.5, Y_LEDGE)
    f = [f for f in s.faces() if abs(f.center().Y - Y_LEDGE) < 1e-6][0]
    vs = [(v.X, v.Z) for v in f.outer_wire().vertices()]
    return sg.MultiPoint(vs).convex_hull


CAP2_SHIFT = 4.0
T2_HALF = [(60, 0), (50, 29), (38, 57), (-122, 57), (-142, 41), (-154, 17), (-154, 0)]


def strip_wall():
    """S: the strip's back wall -- concept D's irregular crystal outline of tier 2,
    kept inside tier 1's top so the strip sits wholly on the ledge."""
    C = sg.Polygon(K.crystal_profile(K.mirror(T2_HALF), 44, 0.14, seed=6.0))
    C = sg.Polygon([(x + CAP2_SHIFT, z) for x, z in pts(C)])
    S = C.intersection(soff(t1_top(), -(STRIP_T + 0.6)))
    return sg.Polygon(pts(S)).simplify(0.05)


def t2_planes(S):
    T2b = soff(S, LIP)
    P = pts(T2b)
    w = K.frustum_walls(P, Y_CHAN, Y_TOP, K.insets_by_direction(P, 12.0, 14.0, 12.0, 12.0), "t2")
    return w + [((0, Y_TOP, 0), (0, 1, 0), "top"), ((0, Y_CHAN, 0), (0, -1, 0), "base")]


def ring_part():
    flange_out = soff(H0, -0.5)
    inner = soff(H0, -18.0)
    tabs = [sg.Point(x, z).buffer(7.0) for x, z in CAP_SCREWS]
    inner = inner.difference(unary_union(tabs))
    # bridge every tab that sits inside the opening back to the flange
    bridges = []
    for x, z in CAP_SCREWS:
        if soff(H0, -18.0).contains(sg.Point(x, z)):
            bridges.append(sg.LineString([(x, z), (x - 40 if x < 0 else x + 40, z)]).buffer(6.0))
    inner = inner.difference(unary_union(bridges)) if bridges else inner
    inner = max(inner.geoms, key=lambda g: g.area) if inner.geom_type == "MultiPolygon" else inner
    flange = prism_xz(flange_out, Y1, Y_RING_T) - prism_xz(inner, Y1 - 1, Y_RING_T + 1)
    w_out, w_in = soff(H0, -WALL - FIT), soff(H0, -WALL - FIT - 3.0)
    wall = prism_xz(w_out, Y_RING_T - 0.01, Y_RING_W) - prism_xz(w_in, Y1, Y_RING_W + 1)
    bosses = []
    for x, z, n in HOOD_SCREWS:
        p = skirt_point(x, z, n)
        n = np.asarray(n, float)
        c = p - n * (WALL + FIT + 4.0)                        # boss centre, 8 deep from the wall face
        t = np.cross(n, [0, 1, 0])
        sq = [c + a * t * 4.5 + b * n * 4.01 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        bosses.append(prism_xz([(q[0], q[2]) for q in sq], Y_RING_T - 0.01, Y_RING_W))
    ring = flange + wall + union(bosses)
    for x, z, n in HOOD_SCREWS:
        p = skirt_point(x, z, n)
        ring = ring - bore(p, -np.asarray(n, float), SELF_TAP, WALL + FIT + 9.0, back=0.0)
    for x, z in CAP_SCREWS:
        top = np.array([x, Y_RING_T, z])
        ring = ring - bore(top, (0, -1, 0), CLEAR, 4.0) - csk(top, (0, -1, 0))
        ring = ring - (Pos(x, (Y_RING_T + Y_SKIRT) / 2 + 2, z) * Cylinder(4.6, Y_SKIRT - Y_RING_T + 4,
                                                                          rotation=(90, 0, 0)))
    return ring


def hood_part():
    S = strip_wall()
    p1, p2 = t1_planes(), t2_planes(S)
    skirt_o = prism_xz(H0, Y_RING_T, Y_SKIRT)
    skirt_i = prism_xz(soff(H0, -WALL), Y_RING_T - 1, Y_SKIRT + 0.01)
    o1 = K.polyhedron(p1)
    i1 = K.polyhedron(p1, WALL, skip=("base",)) & K.slab_y(Y_SKIRT, BIG)
    chan_o = prism_xz(S, Y_LEDGE - 0.01, Y_CHAN + 0.01)
    chan_i = prism_xz(soff(S, -CHAN_WALL), Y_LEDGE - 4, Y_CHAN + 0.5)
    o2 = K.polyhedron(p2)
    i2 = K.polyhedron(p2, WALL, skip=("base",)) & K.slab_y(Y_CHAN, BIG)
    outer = skirt_o + o1 + chan_o + o2
    shell = outer - skirt_i - i1 - chan_i - i2
    # data wire: through the channel's back wall at the back centre
    xb = min(p[0] for p in pts(S))
    shell = shell - bore((xb + 1, Y_LEDGE + 3.0, 0), (1, 0, 0), 2.5, CHAN_WALL + 2)
    # hood screws: countersunk through the skirt
    for x, z, n in HOOD_SCREWS:
        p = skirt_point(x, z, n)
        d = -np.asarray(n, float)
        shell = shell - bore(p, d, CLEAR, WALL + 1) - csk(p, d)
    return shell, outer, p1, p2, S


def hood_details(shell, p1, p2, S):
    """Concept D's carving, moved onto the print hood (tier 2 is 13 mm tall here,
    not 19; the strip channel replaces the jagged fence; real screws on the skirt
    replace the tier-1 rivet row)."""
    cuts, gfx, blue, pockets = [], [], [], []
    ym2 = (Y_CHAN + Y_TOP) / 2
    ym1 = (Y_SKIRT + Y_LEDGE) / 2
    f = K.pick(p2, "t2", (0, 0.8, 1), (-50, ym2, 45), (1, 0, 0))
    sg_ = np.sign(f[1][0])
    for k in range(8):
        cuts.append(K.carve(f, K.slash(sg_ * (-48 + k * 7.2), 0, 3.2, 12, 6 * sg_), 4.8))
    gfx.append(K.carve(f, [(sg_ * a, b) for a, b in [(-56, -6), (2, -6), (8, -1), (8, 6), (-48, 6), (-56, 1)]], 1.2))
    f = K.pick(p2, "t2", (0, 0.8, -1), (-50, ym2, -45), (1, 0, 0))
    sg_ = np.sign(f[1][0])
    for uc, w, h, sk in ((-46, 6.0, 13, 6), (-18, 6.5, 14, -5), (14, 6.0, 13, 6)):
        cuts.append(K.carve(f, K.slash(sg_ * uc, 0, w, h, sk * sg_), 5.2))
    gfx.append(K.carve(f, [(sg_ * a, b) for a, b in [(-56, -8), (18, -8), (27, -1), (27, 8), (-48, 8), (-56, 1)]], 1.3))
    for zs in (1, -1):
        f = K.pick(p2, "t2", (1, 0.93, 0.29 * zs), (47, ym2, 15 * zs), (0, 0, -zs))
        gfx.append(K.carve(f, K.polyline_band([(-9, -4.5), (5, -4.5), (9, -0.5), (9, 3.5)], 3.0), 1.2))
        blue.append(union([K.carve(f, K.slash(uc, 2.5, 1.3, 5.0, 2.4), 1.0) for uc in (-9, -5, -1)]))
    f = K.pick(p2, "t2", (-1, 1.0, 0), (-150, ym2, 0), (0, 0, 1))
    pockets.append(K.carve(f, K.chamfer_rect(-34, -14, -2.4, 2.4, 1.6), 1.5))
    pockets.append(K.carve(f, K.chamfer_rect(6, 34, -3.2, 3.2, 1.8), 1.5))
    for zs, u0s in ((1, (-112, -94, -78, 10, 30)), (-1, (-108, -86, -60, -30, 14, 32))):
        f = K.pick(p1, "t1", (0, 0.5, zs), (-50, ym1, zs * 74), (1, 0, 0))
        sg_ = np.sign(f[1][0])
        pockets.append(union([K.carve(f, K.chamfer_rect(sg_ * u0 - 6.2, sg_ * u0 + 6.2, -2.6, 2.6, 1.6), 1.6)
                              for u0 in u0s]))
    f = K.pick(p1, "t1", (-1, 0.5, 0), (-166, ym1, 0), (0, 0, 1))
    pockets.append(K.carve(f, K.chamfer_rect(-42, -30, -2.6, 2.6, 1.4), 1.5))
    pockets.append(K.carve(f, K.chamfer_rect(-10, 14, -3.2, 3.2, 1.6), 1.5))
    pockets.append(K.carve(f, K.chamfer_rect(26, 36, -2.2, 2.2, 1.2), 1.5))
    f = K.pick(p1, "t1", (1, 0.5, 0), (67, ym1, 0), (0, 0, -1))
    blue.append(K.carve(f, K.polyline_band([(-50, -1.0), (-18, -1.0), (-14, 1.2), (50, 1.2)], 1.4), 1.0))
    # armour on tier 2's top, recentred on tier 2
    ax = CAP2_SHIFT
    base_dense = K.densify(K.ARMOUR_BASE_FULL, 22.0)
    base_sh = [(x + ax, z * 0.92) for x, z in base_dense]
    ins = [K.jag(i, 1.0, 4.0, seed=0.0) for i in range(len(base_sh))]
    H1, H2 = K.ARMOUR_H1, K.ARMOUR_H2
    w = K.frustum_walls(base_sh, Y_TOP, Y_TOP + H1, ins)
    arm_base = K.polyhedron(w + [((0, Y_TOP + H1, 0), (0, 1, 0), "t"), ((0, Y_TOP - 0.2, 0), (0, -1, 0), "b")])
    top = K.face_frame((0, Y_TOP + H1, 0), (0, 1, 0), (1, 0, 0))              # u = x, v = -z
    jog = [(x + ax, -z * 0.92) for x, z in K.ARMOUR_FULL]
    arm_top = K.carve(top, jog, 0.2, lift=H2)
    t2 = K.face_frame((0, Y_TOP + H1 + H2, 0), (0, 1, 0), (1, 0, 0))
    o = ax
    bus = K.carve(t2, K.polyline_band([(-128 + o, 11), (-90 + o, 11), (-76 + o, -2), (-30 + o, -2), (-16 + o, 11),
                                       (18 + o, 11)], 8), 1.3) & arm_top
    traces = union([
        K.carve(t2, K.polyline_band([(-124 + o, -18), (-78 + o, -18), (-70 + o, -24), (-20 + o, -24), (-12 + o, -18),
                                     (12 + o, -18)], 1.6), 1.0),
        K.carve(t2, K.polyline_band([(-118 + o, 23), (-98 + o, 23), (-92 + o, 17)], 1.6), 1.0),
        K.carve(t2, K.polyline_band([(-40 + o, 21), (-6 + o, 21)], 1.6), 1.0),
        union([K.carve(t2, K.chamfer_rect(x + o - 2.2, x + o + 2.2, z - 2.2, z + 2.2, 0.9), 1.0)
               for x, z in ((-124, -18), (12, -18), (-118, 23), (-92, 17), (-40, 21), (-6, 21))]),
    ]) & arm_top
    tf = K.face_frame((0, Y_TOP, 0), (0, 1, 0), (1, 0, 0))
    frame_line = K.jagged_ring(tf, [(x + o, z * 0.92) for x, z in K.ARMOUR_BASE_FULL], 4.6, 2.4, 1.3, 0.8, seed=0.0)
    dense_rp = K.densify([(x + o, z * 0.92) for x, z in K.ARMOUR_BASE_FULL], 10.0)
    ring_pts = K.offset_plan_var(dense_rp, [K.jag(i, 7.0, 8.4, seed=5.0) for i in range(len(dense_rp))])
    gfx.append(union([K.carve(tf, K.chamfer_rect(x - 1.1, x + 1.1, -z - 1.1, -z + 1.1, 0.4), 0.6) for x, z in ring_pts]))
    win_top = union([K.carve(t2, K.chamfer_rect(x + o - 5.0, x + o + 5.0, 22.5, 25.5, 1.0), 1.4)
                     for x in (-84, -72)]) & arm_top

    capc = shell - union(cuts) - union(pockets) - frame_line
    g = union(gfx) & capc
    b = union(blue) & capc
    armour_w = arm_top - bus - traces - win_top
    white = (capc - g - b) + armour_w
    graphite = (g - b) + (bus - traces) + (arm_base - arm_top - capc)      # the base plate sinks 0.2 into the roof
    accent = b + (traces - bus)
    return {"white": white, "graphite": graphite, "blue": accent}


def strip_part(S):
    mid = soff(S, STRIP_T / 2)
    band = prism_xz(soff(S, STRIP_T), STRIP_Y0, STRIP_Y0 + STRIP_W) - prism_xz(S, STRIP_Y0 - 1, STRIP_Y0 + STRIP_W + 1)
    xb = min(p[0] for p in pts(S))
    band = band - Pos(xb, STRIP_Y0 + STRIP_W / 2, 0) * Box(12, 10, 3.0)              # the data-wire seam
    return band, mid.exterior.length


# ------------------------------------------------------------------ fasteners
STD = (6, 8, 10, 12, 16, 20, 25, 30, 35, 40)


def fastener_list(ft, bt, fscrews, bscrews):
    """Screw lengths: a bumper screw runs from its counterbore floor through the
    TPU, the relief and the plate, then 5 mm on into the corner bracket (its 2 mm
    wall + the 2.4 mm captive nut), the same reach as the v3 M3x8 flatheads."""
    rows = []
    for name, tpu, screws, plate_in, sx in (("front", ft, fscrews, XF0, 1), ("back", bt, bscrews, XB0, -1)):
        lens = []
        for y, z in screws:
            if abs(z) > 1 or y < 35:
                xface = (FACE_F + TPU_T) if sx > 0 else (FACE_B - TPU_T)
                rod = tpu & bore((xface, y, z), (-sx, 0, 0), CBORE - 0.2, 40, back=0.5)
                bb = rod.bounding_box()
                floor = bb.max.X if sx > 0 else bb.min.X          # the counterbore floor: TPU under the head starts here
                need = abs(floor - plate_in) + 5.0
                lens.append(min(L for L in STD if L >= need))
        rows.append(f"bumper {name}: {len(lens)} x M3 socket head, lengths {sorted(lens)} (counterbored, 3 mm TPU under the head)")
    rows.append("hood -> ring: 10 x M3x10 countersunk (skirt 2.5 + gap 0.3 + 7 into the Ø2.8 boss pilot)")
    rows.append("ring -> top corner brackets: 9 x M3x8 countersunk (the v3 Cap screws, same holes)")
    rows.append("back centre-top (65, 0): M3x8 countersunk at the bottom of its bolt well (the v3 screw)")
    rows.append("front centre-top (70, 0): no bracket behind it in v3 either -- hole kept, no screw")
    rows.append("screen: the v3 M2 screws, unchanged (heads in the well floor)")
    return rows


# ------------------------------------------------------------------ build
def build():
    fb, ft, fscrews = front_parts()
    bb, bt, bscrews = back_parts()
    ring = ring_part()
    # the hood is variant B, PLATES (his pick, 2026-10-07): hood_variants.py builds it on
    # hood_part()'s shell, everything below tier 2's base exactly as before.  Its two field
    # hoses go in through 24_pipes like every other pipe, so not into this STEP.
    import hood_variants as HV
    hb = HV.variant_B()
    hb.pipes = []
    hood, _ = hb.bodies()
    S = hb.S
    strip, strip_len = strip_part(S)
    parts = {
        "FacetFront": [("white", fb["white"], WHITE), ("dark", fb["dark"], DARK), ("blue", fb["blue"], BLUE)],
        "FacetBack": [("white", bb["white"], WHITE), ("dark", bb["dark"], DARK), ("graphite", bb["graphite"], GRAPHITE),
                      ("blue", bb["blue"], BLUE)],
        "BumperFront": [("tpu", ft, TPU_GREY)],
        "BumperBack": [("tpu", bt, TPU_GREY)],
        "FacetRing": [("graphite", ring, GRAPHITE)],
        "FacetHood": [("white", hood["white"], WHITE), ("graphite", hood["graphite"], GRAPHITE),
                      ("blue", hood["blue"], BLUE), ("dark", hood["dark"], DARK)],
        "NeopixelStrip": [("strip", strip, STRIP)],
    }
    info = dict(strip_len=strip_len, front_screws=fscrews, back_screws=bscrews, S=S,
                fasteners=fastener_list(ft, bt, fscrews, bscrews))
    return parts, info


# print orientation: which robot direction points UP off the bed (the face on the bed is the opposite one)
PRINT_UP = {"FacetFront": (1, 0, 0),       # plate (inner face, x 50) on the bed
            "FacetBack": (-1, 0, 0),       # plate (inner face, x -150) on the bed
            "BumperFront": (-1, 0, 0),     # outer face on the bed
            "BumperBack": (1, 0, 0),
            "FacetRing": (0, 1, 0),        # flange on the bed
            "FacetHood": (0, 1, 0)}        # skirt on the bed; tree supports inside only
FILAMENT = {"white": 1, "graphite": 2, "blue": 3, "dark": 4, "tpu": 1}


def write_print_3mf(parts, out_dir):
    """Bambu project 3MF per printed part, in print orientation, one object per
    colour body with its filament slot (1 white, 2 graphite, 3 blue, 4 dark; TPU
    parts are single-filament).  Same writer as the links' 3MFs (lib/export3mf)."""
    import export3mf
    from render3d import tessellate
    for name, up in PRINT_UP.items():
        u = np.asarray(up, float)
        a = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(a, u); e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        R = np.vstack([e1, e2, u])                        # robot -> print frame (rows)
        objs = []
        for n, b, rgb in parts[name]:
            if b is None or not b.solids():
                continue
            for k, so in enumerate(b.solids()):
                V, T, _ = tessellate(so, 0.05)
                objs.append((f"{name}_{n}_{k}", (R @ np.asarray(V, float).T).T, np.asarray(T), FILAMENT[n]))
        path = os.path.join(out_dir, f"{name}_bambu.3mf")
        export3mf.write_3mf(objs, path, name=f"{name} (box D FACET)")
        allV = np.vstack([o[1] for o in objs])
        ext = allV.max(0) - allV.min(0)
        print(f"  {name:12s} {len(objs):3d} bodies, print box {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm"
              f"{'  ** OVER 256 BED **' if max(ext[:2]) > 256 else ''} -> {os.path.basename(path)}")


def preview(parts, path, title=""):
    from render3d import tessellate
    from render_color import view, sheet
    tb, allV = [], []
    for name, bodies in parts.items():
        for n, b, rgb in bodies:
            if b is None or not b.solids():
                continue
            V, T, _ = tessellate(b, 0.15)
            V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)
            tb.append((V, T, rgb))
            allV.append(V)
    allV = np.vstack(allV)
    tiles = [("front 3/4", "", view(tb, allV, (760, 560), 24, -36, light="camera")),
             ("rear 3/4", "", view(tb, allV, (760, 560), 28, -142, light="camera")),
             ("side", "", view(tb, allV, (760, 560), 0.5, -90, light="camera")),
             ("top", "", view(tb, allV, (760, 560), 89.5, -90, light="camera"))]
    sheet(tiles, 2, path, header=title, size=(760, 560))


if __name__ == "__main__":
    out = sys.argv[1]                       # STEP + preview; add `--3mf <dir>` for the Bambu print files
    os.makedirs(out, exist_ok=True)
    parts, info = build()
    for name, bodies in parts.items():
        tot = 0.0
        for n, b, _ in bodies:
            v = solids_volume(b)
            tot += v
            bbx = b.bounding_box()
            print(f"  {name:13s} {n:9s} {v / 1000:8.2f} cm3  {len(b.solids())} solid(s)  "
                  f"x {bbx.min.X:7.1f}..{bbx.max.X:7.1f} y {bbx.min.Y:6.1f}..{bbx.max.Y:6.1f} "
                  f"z {bbx.min.Z:6.1f}..{bbx.max.Z:6.1f}", flush=True)
            # (not the hood: its white plates, gills and pads are islands on the graphite
            # field by design -- hood_variants.checks() proves none of them floats)
            if n in ("white", "tpu", "graphite") and name != "FacetHood" and len(b.solids()) > 1:
                for so in sorted(b.solids(), key=lambda q: -q.volume)[1:]:
                    c = so.center()
                    print(f"      stray solid {so.volume:9.3f} mm3 at ({c.X:.1f}, {c.Y:.1f}, {c.Z:.1f})")
        stepcolor.write([(n, b, rgb) for n, b, rgb in bodies if b is not None and b.solids()],
                        os.path.join(out, f"{name}.step"), part=name)
    print(f"  strip route {info['strip_len']:.0f} mm (his strip: 975 mm)")
    print("  fasteners:")
    for row in info["fasteners"]:
        print("    " + row)
    print(f"  front bracket holes {info['front_screws']}")
    print(f"  back bracket holes  {info['back_screws']}")
    preview(parts, os.path.join(out, "preview_facet_print.png"), "Box D FACET -- print parts")
    if "--3mf" in sys.argv:
        write_print_3mf(parts, sys.argv[sys.argv.index("--3mf") + 1])
    print("  wrote", flush=True)
