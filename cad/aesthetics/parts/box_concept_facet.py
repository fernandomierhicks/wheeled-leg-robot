"""Body-box concept D, FACET (2026-10-04, his round-2 brief: "jagged angled
surfaces like the links, a trapezoid with angled faces everywhere and smaller
features carved in").  ROBOT coords, mm.  Concept only, not print-checked.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/aesthetics/parts/box_concept_facet.py <out_dir>

writes BoxConcept_D_facet.step + preview_D_facet.png; put it on the robot with
    cad/solidworks_api/20_box_concepts.py import <out_dir> --names=D_facet   (then assemble, render)

TRAP: polyhedron() intersects half-spaces, so it only builds CONVEX outlines --
a jogged (concave) outline came out silently trimmed.  Concave shapes are
carve()n (extruded) instead, like the armour's jogged top plate.

Only planar faces.  Every solid is an intersection of half-spaces (so every
inner wall is an exact offset), and every detail is an extrusion or an extruded
cut on one of those faces:

  lower body   faceted walls between the RobotMounts: raked chin, a deep
               chamfered face well with the screen at its back, faceted back
  light band   a recessed channel all the way round under the cap's overhang
  cap          a trapezoid frustum (every face its own slope), chamfered corners
  armour       a raised plate on the cap, jogged 45-degree outline, chamfered
  carving      vent slashes through the side slopes, chamfered windows and a
               hatch comb on the front slope, a graphite bus with 45-degree bends
               and blue traces with ticks and pads on the armour, notches
"""
import os
import sys
import math
import numpy as np
from build123d import Box, Pos, Rot, Plane, Polyline, make_face, extrude, split, Keep, Cylinder, Vector
from shapely.geometry import MultiPoint, Polygon, LineString

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import stepcolor

WHITE = (0xF2, 0xF4, 0xF7)
GRAPHITE = (0x7E, 0x87, 0x95)
DARK = (0x2B, 0x2F, 0x36)
BLUE = (0x1E, 0x7B, 0xFF)

WALL = 2.5
ZW = 80.0
Y_FLOOR = 15.0
RM = dict(x0=-150.0, x1=50.0, y1=75.0)
OLED = dict(x=53.8, y0=29.0, y1=66.0, z0=-45.3, z1=36.0)
SWITCH = dict(y=45.0, z=50.0, r=10.0)
BIG = 600.0


# ------------------------------------------------------------------ geometry kit
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def boxp(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)


def slab_y(y0, y1):
    return boxp(-BIG, BIG, y0, y1, -BIG, BIG)


def polyhedron(planes, t=0.0, skip=()):
    """planes: [(point, outward normal, tag)]; every plane moved in by t except
    tags in `skip` (dropped entirely)."""
    p = boxp(-BIG, BIG, -BIG, BIG, -BIG, BIG)
    for pt, n, tag in planes:
        if tag in skip:
            continue
        n = unit(n)
        o = np.asarray(pt, float) - t * n
        p = split(p, bisect_by=Plane(origin=tuple(map(float, o)), z_dir=tuple(map(float, n))), keep=Keep.BOTTOM)
    return p


def ccw(P):
    P = np.asarray(P, float)
    if np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]) < 0:
        P = P[::-1]
    return P


def plan_walls(plan_xz, tag="wall"):
    """Vertical half-spaces from a closed plan polygon (x, z)."""
    P = ccw(plan_xz)
    out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        e = b - a
        m = np.array([e[1], -e[0]])                    # outward for CCW
        out.append(((a[0], 0.0, a[1]), (m[0], 0.0, m[1]), tag))
    return out


def frustum_walls(plan_xz, y0, y1, insets, tag="slope"):
    """Sloped half-spaces: each base edge at y0 rises to the same edge moved in by
    insets[i] at y1 -- a frustum whose faces are each their own slope."""
    P = ccw(plan_xz)
    h = y1 - y0
    out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        e = b - a
        m = unit([e[1], -e[0]])
        d = insets[i] if np.ndim(insets) else insets
        out.append(((a[0], y0, a[1]), (h * m[0], d, h * m[1]), tag))
    return out


def side_profile_walls(prof_xy, tag="side"):
    """Half-spaces from a side profile polygon (x, y), constant in z."""
    P = ccw(prof_xy)
    out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        e = b - a
        m = np.array([e[1], -e[0]])
        t = "floor" if abs(e[1]) < 1e-9 and a[1] <= Y_FLOOR + 1e-6 and m[1] < 0 else tag
        out.append(((a[0], a[1], 0.0), (m[0], m[1], 0.0), t))
    return out


def face_frame(point, normal, u_hint):
    n = unit(normal)
    u = np.asarray(u_hint, float)
    u = unit(u - (u @ n) * n)
    return np.asarray(point, float), u, np.cross(n, u), n


def carve(frame, pts_uv, depth, lift=0.6):
    """Prism under a planar face: polygon (u, v) on the face frame, from `lift`
    above the face to `depth` into it."""
    o, u, v, n = frame
    pl = Plane(origin=tuple(map(float, o + lift * n)), x_dir=tuple(map(float, u)), z_dir=tuple(map(float, n)))
    f = pl * make_face(Polyline(*[(float(a), float(b)) for a, b in pts_uv], close=True))
    return extrude(f, amount=depth + lift, dir=tuple(map(float, -n)))


def boss(frame, pts_uv, height):
    """Prism PROUD of a planar face: polygon (u, v) on the face frame, from the
    face up to `height` above it -- for UNIONing on, unlike carve (which is
    built for subtracting)."""
    o, u, v, n = frame
    pl = Plane(origin=tuple(map(float, o)), x_dir=tuple(map(float, u)), z_dir=tuple(map(float, n)))
    f = pl * make_face(Polyline(*[(float(a), float(b)) for a, b in pts_uv], close=True))
    return extrude(f, amount=height, dir=tuple(map(float, n)))


def chamfer_rect(u0, u1, v0, v1, c):
    return [(u0 + c, v0), (u1 - c, v0), (u1, v0 + c), (u1, v1 - c), (u1 - c, v1), (u0 + c, v1), (u0, v1 - c), (u0, v0 + c)]


def slash(uc, vc, w, h, skew):
    """Parallelogram vent: width w along u, height h along v, leaning by skew."""
    return [(uc - w / 2 - skew / 2, vc - h / 2), (uc + w / 2 - skew / 2, vc - h / 2),
            (uc + w / 2 + skew / 2, vc + h / 2), (uc - w / 2 + skew / 2, vc + h / 2)]


def polyline_band(pts, width):
    """A constant-width band along a polyline with mitred corners (2D)."""
    P = np.asarray(pts, float)
    L, R = [], []
    for i in range(len(P)):
        if i == 0:
            d = unit(P[1] - P[0]); nrm = np.array([-d[1], d[0]]); s = 1.0
        elif i == len(P) - 1:
            d = unit(P[-1] - P[-2]); nrm = np.array([-d[1], d[0]]); s = 1.0
        else:
            d1, d2 = unit(P[i] - P[i - 1]), unit(P[i + 1] - P[i])
            n1, n2 = np.array([-d1[1], d1[0]]), np.array([-d2[1], d2[0]])
            nrm = unit(n1 + n2); s = 1.0 / max(nrm @ n1, 0.2)
        L.append(P[i] + nrm * width / 2 * s)
        R.append(P[i] - nrm * width / 2 * s)
    return [tuple(p) for p in L] + [tuple(p) for p in R[::-1]]


def union(parts):
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


# ------------------------------------------------------------------ the shape
# lower body: side profile (x, y) and plan (x, z) -- polygons, so all planar
# corners bevelled (4 mm) so the hull itself reads as faceted, not just the cap
LOW_SIDE = [(54.5, Y_FLOOR), (63, 27), (65, 34), (65, 73), (61, 77), (-160, 77),
            (-164, 73), (-164, 34), (-161, 27), (-152, Y_FLOOR)]
LOW_PLAN_HALF = [(65, 0), (65, 50), (58, 68), (RM["x1"], ZW), (RM["x0"], ZW), (-158, 70), (-164, 52), (-164, 0)]
Y_BAND0, Y_BAND1 = 77.0, 81.0


def mirror(half):
    """Half outline (x, z >= 0), front to back -> closed outline; centreline points
    are not doubled (a zero-length edge has no normal)."""
    rev = [(x, -z) for x, z in reversed(half)]
    if abs(half[-1][1]) < 1e-9:
        rev = rev[1:]
    if abs(half[0][1]) < 1e-9:
        rev = rev[:-1]
    return half + rev


LOW_PLAN = mirror(LOW_PLAN_HALF)


def offset_plan(plan, d):
    """Plan polygon moved out by d (each edge parallel)."""
    P = ccw(plan)
    out = []
    n = len(P)
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        e1, e2 = unit(b - a), unit(c - b)
        n1, n2 = np.array([e1[1], -e1[0]]), np.array([e2[1], -e2[0]])
        out.append(b + d * (n1 + n2) / max(1 + n1 @ n2, 1e-3))
    return [tuple(p) for p in out]


def offset_plan_var(plan, d_list):
    """Like offset_plan, but a DIFFERENT offset per vertex -- the outline wanders
    in and out instead of staying a uniform parallel copy."""
    P = ccw(plan)
    n = len(P)
    out = []
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        e1, e2 = unit(b - a), unit(c - b)
        n1, n2 = np.array([e1[1], -e1[0]]), np.array([e2[1], -e2[0]])
        d = d_list[i % len(d_list)]
        out.append(b + d * (n1 + n2) / max(1 + n1 @ n2, 1e-3))
    return [tuple(p) for p in out]


def densify(poly, max_seg):
    """Subdivide every edge to at most max_seg long -- more vertices to wander
    at, without changing the outline (points stay on the original straight
    edges, so convexity is unaffected)."""
    P = ccw(poly)
    n = len(P)
    out = []
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / max_seg)))
        out += [tuple(a + (b - a) * j / k) for j in range(k)]
    return out


def jag(i, lo, hi, seed=0.0):
    """Deterministic value in [lo, hi] with NO correlation between neighbours --
    a hash, not a wave. A sine wave eases from one value to the next, which
    reads as a smooth, lofted wander; consecutive integers here can land at
    opposite ends of the range, so the straight segment between them is a hard
    angular cut instead of a gentle curve."""
    h = math.sin(i * 12.9898 + seed * 78.233) * 43758.5453
    frac = h - math.floor(h)
    return lo + frac * (hi - lo)


def crystal_profile(base_poly, n, amp, seed=0.0):
    """An irregular CONVEX outline that keeps base_poly's general proportions:
    sample n points at evenly spaced angles around its centroid, radius at each
    angle = base_poly's OWN TRUE boundary distance there (exact ray cast, not
    interpolated between vertex radii -- linear interpolation of a convex
    outline's vertex radii always overshoots the real edge, since a straight
    edge's distance-from-centre is a convex function of angle and lies BELOW
    its own chord; that overshoot alone was once +37% area at zero jitter)
    times a jittered factor, then take the convex hull. polyhedron() /
    frustum_walls() need convex, so the hull step isn't optional -- but because
    the jitter is a genuine per-angle factor (not a small wobble on a smooth
    curve), the surviving hull points land at irregular angles and distances,
    not a faceted circle."""
    poly = Polygon(base_poly)
    cx, cz = poly.centroid.x, poly.centroid.y
    boundary = poly.boundary
    far = 10 * max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1])

    def r_at(theta):
        ray = LineString([(cx, cz), (cx + far * math.cos(theta), cz + far * math.sin(theta))])
        inter = ray.intersection(boundary)
        if inter.is_empty:
            return 0.0
        pts = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
        p = min(pts, key=lambda q: (q.x - cx) ** 2 + (q.y - cz) ** 2)
        return math.hypot(p.x - cx, p.y - cz)

    pts = []
    for i in range(n):
        th = 2 * math.pi * i / n
        r = r_at(th) * jag(i, 1 - amp, 1 + amp, seed)
        pts.append((cx + r * math.cos(th), cz + r * math.sin(th)))
    hull = MultiPoint(pts).convex_hull
    return list(hull.exterior.coords)[:-1]


def lower_planes():
    return side_profile_walls(LOW_SIDE) + plan_walls(LOW_PLAN, "plan") + \
        [((0, 0, ZW), (0, 0, 1), "flank"), ((0, 0, -ZW), (0, 0, -1), "flank")]


def lower_body():
    pl = lower_planes()
    outer = polyhedron(pl)
    # inner: walls moved in; open at the floor, open at the top (into the cap),
    # open at the flanks where the RobotMounts are
    inner_closed = polyhedron([p for p in pl if not (p[2] == "side" and p[1][1] > 0.9)], WALL, skip=("floor",))
    inner_open = polyhedron([p for p in pl if p[2] != "flank" and not (p[2] == "side" and p[1][1] > 0.9)],
                            WALL, skip=("floor",)) & boxp(RM["x0"], RM["x1"], -BIG, RM["y1"], -BIG, BIG)
    shell = outer - inner_closed - inner_open
    return shell, outer


# each tier is pushed off-centre from the one below it -- toward the nose and
# toward the intake side, more with each tier up -- so the stack reads as
# cantilevered over whatever's inside, not as a lid centred for its own sake.
# This is the move that actually breaks "a box with a trapezoid on top": the
# massing itself, not the surface texture on it.
# z stays at 0: tier 2 already touches the |z| <= 82 femur-clearance limit
# symmetric, so any lateral shift would push one side past it. The cantilever
# is fore-aft (x) only -- no clearance rule constrains that axis the same way.
CAP1_SHIFT = (4.0, 0.0)
CAP2_SHIFT = (10.0, 0.0)
CAP_BASE = [(x + CAP1_SHIFT[0], z + CAP1_SHIFT[1]) for x, z in offset_plan(LOW_PLAN, 2.0)]
Y_T1, Y_T2, Y_TOP = 81.0, 93.0, 112.0
# tier 2: narrower, with a chevron nose and chamfered tail (plan half, x / z)
T2_HALF = [(60, 0), (50, 34), (38, 62), (-122, 62), (-142, 46), (-154, 22), (-154, 0)]
ARMOUR_HALF = [(20, 0), (14, 14), (6, 24), (-28, 24), (-34, 30), (-50, 30), (-50, 34), (-66, 34), (-66, 30),
               (-92, 30), (-100, 24), (-118, 24), (-124, 12), (-126, 0)]
# the -z lobe is NOT a mirror of the above -- one notch instead of two, a
# different reach, like it's actually routed around something that's only on
# one side. Keeps near-identical depth to ARMOUR_HALF under x -118..-6 (where
# the existing trace art already runs) so that art isn't clipped by the change.
ARMOUR_NEG = [(-126, 0), (-123, -14), (-116, -26), (-92, -28), (-60, -28), (-60, -32), (-44, -32), (-44, -27),
              (-20, -27), (-6, -24), (10, -10), (20, 0)]
ARMOUR_FULL = ARMOUR_HALF + ARMOUR_NEG[1:-1]          # closes back to ARMOUR_HALF[0]
# the base plate under it is shallower on the -z side too (asymmetric, still convex)
ARMOUR_BASE_NEG = [(-128, 0), (-110, -18), (14, -18), (24, 0)]


def offset_half(half, d):
    """Half outline grown by d (via the full mirrored outline), back to a half."""
    full = offset_plan(mirror(half), d)
    return [p for p in full if p[1] >= -1e-6]


def insets_by_direction(plan, front, side, back, corner):
    P = ccw(plan)
    out = []
    for i in range(len(P)):
        e = P[(i + 1) % len(P)] - P[i]
        m = unit([e[1], -e[0]])
        out.append(front if m[0] > 0.95 else back if m[0] < -0.95 else side if abs(m[1]) > 0.95 else corner)
    return out


def tier1_planes():
    w = frustum_walls(CAP_BASE, Y_T1, Y_T2, insets_by_direction(CAP_BASE, 6.0, 6.0, 6.0, 8.0), "t1")
    return w + [((0, Y_T2, 0), (0, 1, 0), "ledge"), ((0, Y_T1, 0), (0, -1, 0), "base")]


T2_PLAN = crystal_profile(mirror(T2_HALF), 44, 0.14, seed=6.0)   # irregular, but convex


def tier2_planes():
    plan = [(x + CAP2_SHIFT[0], z + CAP2_SHIFT[1]) for x, z in T2_PLAN]
    w = frustum_walls(plan, Y_T2, Y_TOP, insets_by_direction(plan, 17.0, 22.0, 17.0, 17.0), "t2")
    # the crystal hull can bulge past the original profile's own reach in
    # places -- clamp to the documented |z| <= 82 femur-clearance limit (a few
    # mm of margin) rather than trust the jitter to stay inside it on its own.
    return w + [((0, Y_TOP, 0), (0, 1, 0), "top"), ((0, Y_T2 - 0.01, 0), (0, -1, 0), "base"),
                ((0, 0, ZW), (0, 0, 1), "flank"), ((0, 0, -ZW), (0, 0, -1), "flank")]


def cap():
    p1, p2 = tier1_planes(), tier2_planes()
    o1 = polyhedron(p1)
    # the crystal profile's irregular bulges can reach further out at some
    # angles than tier 1 does directly below -- clip tier 2 to tier 1's own
    # sloped wall (extended past Y_T2 at the same slope), or it leaves an
    # unsupported sliver poking out past the real silhouette, same issue as
    # the fence trim earlier and the same fix.
    t1_prism = polyhedron([p for p in p1 if p[2] == "t1"])
    o2 = polyhedron(p2) & t1_prism
    i1 = polyhedron(p1, WALL, skip=("base",))
    i2 = polyhedron(p2, WALL, skip=("base",)) & slab_y(Y_T2, BIG)
    outer = o1 + o2
    return outer - i1 - i2, outer, p1, p2


def band():
    ring_o = polyhedron(plan_walls(offset_plan(LOW_PLAN, -3.0)) + [((0, Y_BAND1, 0), (0, 1, 0), "t"),
                                                                   ((0, Y_BAND0, 0), (0, -1, 0), "b")])
    ring_i = polyhedron(plan_walls(offset_plan(LOW_PLAN, -5.5)))
    return ring_o - ring_i


ARMOUR_BASE_HALF = [(24, 0), (14, 22), (-112, 22), (-128, 0)]          # convex: chamfered plate
ARMOUR_BASE_FULL = ARMOUR_BASE_HALF + ARMOUR_BASE_NEG[1:-1]            # asymmetric, still convex
ARMOUR_H1, ARMOUR_H2 = 2.2, 2.0


def armour():
    """Two stepped plates, the links' way: a convex plate with 45-degree chamfered
    walls, and on it the jogged outline extruded straight up (a jogged outline is
    concave, so it cannot be a half-space intersection)."""
    # armour rides on tier 2, so it's recentred on tier 2's own shifted centre.
    # The chamfer's RUN (not its rise) varies edge to edge -- densify first so
    # there's enough resolution along the long flats to actually wander, then
    # jitter the inset per edge instead of one constant width all the way round.
    base_dense = densify(ARMOUR_BASE_FULL, 22.0)
    base_shifted = [(x + CAP2_SHIFT[0], z + CAP2_SHIFT[1]) for x, z in base_dense]
    chamfer_insets = [jag(i, 1.0, 4.4, seed=0.0) for i in range(len(base_shifted))]
    w = frustum_walls(base_shifted, Y_TOP, Y_TOP + ARMOUR_H1, chamfer_insets)
    p1 = polyhedron(w + [((0, Y_TOP + ARMOUR_H1, 0), (0, 1, 0), "t"), ((0, Y_TOP - 0.2, 0), (0, -1, 0), "b")])
    top = face_frame((CAP2_SHIFT[0], Y_TOP + ARMOUR_H1, CAP2_SHIFT[1]), (0, 1, 0), (1, 0, 0))  # u = x, v = -z
    jog = [(x, -z) for x, z in ARMOUR_FULL]                                   # asymmetric, not a mirror
    p2 = carve(top, jog, 0.2, lift=ARMOUR_H2)                                  # from +H2 down to -0.2
    return p1, p2


def ring_groove(frame, full, d_out, d_in, depth):
    """A groove following a FULL (closed) outline: grown by d_out minus grown by d_in."""
    outer = [(x, -z) for x, z in offset_plan(full, d_out)]
    inner = [(x, -z) for x, z in offset_plan(full, d_in)]
    return carve(frame, outer, depth) - carve(frame, inner, depth + 1.0)


def jagged_ring(frame, full, d_mid, amp, width, depth, seed=0.0):
    """A groove of constant WIDTH whose centreline cuts between d_mid-amp and
    d_mid+amp in a handful of long straight jumps instead of tracing one
    smooth, constant-radius offset -- an angular picture-frame line, not a
    uniform parallel rectangle or a lofted wander."""
    dense = densify(full, 27.0)
    outer_pts = offset_plan_var(dense, [jag(i, d_mid - amp, d_mid + amp, seed) for i in range(len(dense))])
    inner_pts = offset_plan(outer_pts, -width)
    outer = [(x, -z) for x, z in outer_pts]
    inner = [(x, -z) for x, z in inner_pts]
    return carve(frame, outer, depth) - carve(frame, inner, depth + 1.0)


def jagged_fence(frame, full, d_mid, amp, width, height, seed=0.0):
    """Same cut, but PROUD instead of sunk: a trim band standing height above
    the face whose both edges zigzag sharply. A flat groove viewed from
    straight above has no silhouette to catch -- this renderer (and the eye)
    only picks up a real edge where a raised or sloped face meets a flat one,
    so height is what actually makes the jaggedness visible from the top."""
    dense = densify(full, 27.0)
    outer_pts = offset_plan_var(dense, [jag(i, d_mid - amp, d_mid + amp, seed) for i in range(len(dense))])
    inner_pts = offset_plan(outer_pts, -width)
    outer = [(x, -z) for x, z in outer_pts]
    inner = [(x, -z) for x, z in inner_pts]
    return boss(frame, outer, height) - boss(frame, inner, height + 0.5)


def pick(planes, tag, direction, target, u_hint):
    """Frame on the face (of `tag`) whose outward normal is closest to `direction`;
    origin = `target` projected onto it; u horizontal (from u_hint), v up-slope."""
    d = unit(direction)
    best = max((p for p in planes if p[2] == tag), key=lambda p: unit(p[1]) @ d)
    pt, nn = np.asarray(best[0], float), unit(best[1])
    T = np.asarray(target, float)
    o = T - ((T - pt) @ nn) * nn
    o, u, v, n = face_frame(o, nn, u_hint)
    if v[1] < 0:
        v = -v
        u = np.cross(v, n)
    return o, u, v, n


def concept_facet():
    low, low_outer = lower_body()
    cp, cap_outer, p1, p2 = cap()
    ring = band()
    arm_base, arm_top = armour()
    arm = arm_base + arm_top

    # ---- lower body: face well with the screen at its back, graphite chin, back panel
    win = boxp(40, 90, OLED["y0"], OLED["y1"], OLED["z0"], OLED["z1"])
    wf = face_frame((65, 47.5, -4.5), (1, 0, 0), (0, 0, -1))
    well = carve(wf, chamfer_rect(-47.5, 47.5, -22.5, 22.5, 7), 8.5)
    well_wall = carve(wf, chamfer_rect(-50, 50, -25, 25, 8.0), 11.0) - well
    lining = (well_wall & low_outer) - win
    # bezel rivets on the lining frame -- future-split hint: this is the screen
    # panel's own mounting ring if the front face becomes its own printed part
    bezel_riv = union([carve(wf, chamfer_rect(u - 1.3, u + 1.3, v - 1.3, v + 1.3, 0.5), 1.0)
                       for u, v in ((48.75, 23.75), (-48.75, 23.75), (48.75, -23.75), (-48.75, -23.75),
                                    (0, 23.75), (0, -23.75))]) & lining
    lining = lining - bezel_riv
    low = low - well - win - well_wall
    chin = low & slab_y(Y_FLOOR, 30.5)
    # raked chin: vent slashes down the front ramp, unevenly spaced and sized --
    # whatever is breathing through here isn't centred on the robot's midline
    pl = lower_planes()
    chin_f = pick(pl, "side", (0.8, -0.6, 0), (58.75, 21, 0), (0, 0, 1))
    chin_specs = [(-35, 4.5, 11, 3), (-19, 3.0, 7, -2), (-4, 5.5, 13, 4), (16, 3.2, 8, 2), (34, 4.5, 11, -3)]
    chin_vents = union([carve(chin_f, slash(uc, 0, w, h, sk), 4.2) for uc, w, h, sk in chin_specs])
    low = low - chin_vents
    chin = chin - chin_vents
    # the front taper corners (nose shoulders), one each side, but NOT the same
    # feature: right is an intake vent, left is a flush access hatch (the
    # service side) -- different jobs, so they don't match
    cf_r = pick(pl, "plan", (0.93, 0, 0.36), (61.5, 50, 59), (0, 1, 0))
    low = low - carve(cf_r, slash(50, 0, 6.5, 34, 5), 4.5)
    cf_l = pick(pl, "plan", (0.93, 0, -0.36), (61.5, 50, -59), (0, 1, 0))
    hatch_outer = carve(cf_l, chamfer_rect(40, 60, -17, 17, 2.5), 1.6)
    hatch_inner = carve(cf_l, chamfer_rect(42, 58, -14.5, 14.5, 2.0), 1.6)
    low = low - (hatch_outer - hatch_inner)
    low = low - union([carve(cf_l, chamfer_rect(38.8, 41.2, vy - 1.2, vy + 1.2, 0.5), 0.8) for vy in (-12, 12)])
    sw = Pos(-150, SWITCH["y"], SWITCH["z"]) * Rot(0, 90, 0) * Cylinder(SWITCH["r"], 60)
    back_panel = (low & boxp(-BIG, -162.0, 34, 62, -66, 66)) - sw
    low = low - sw
    bf = face_frame((-164, 48, 0), (-1, 0, 0), (0, 0, 1))
    back_vents = union([carve(bf, slash(uc, 0, 3.2, 18, 6), 4) for uc in (-56, -46, -36)])
    low = low - back_vents
    back_panel = back_panel - back_vents
    # back-panel corner rivets -- future-split hint: this is the switch panel's
    # own mounting ring if the back face becomes its own printed part
    back_riv = union([carve(bf, chamfer_rect(u - 1.0, u + 1.0, v - 1.0, v + 1.0, 0.4), 0.8)
                      for u, v in ((58, 11), (58, -11), (-5, 11))])
    low = low - back_riv
    back_panel = back_panel - back_riv

    cuts, gfx, blue, pockets = [], [], [], []
    # ---- tier 2 side faces: the two sides do different jobs, so they don't
    # match -- right is a dense intake grille, left is three big service vents
    # with more graphite field around them
    f = pick(p2, "t2", (0, 0.8, 1), (-50, 102.5, 51), (1, 0, 0))
    sg = np.sign(f[1][0])
    for k in range(8):
        cuts.append(carve(f, slash(sg * (-48 + k * 7.2), 0, 3.4, 15, 7 * sg), 4.8))
    poly_r = [(-56, -7), (2, -7), (9, -1), (9, 7), (-48, 7), (-56, 1)]
    gfx.append(carve(f, [(sg * a, b) for a, b in poly_r], 1.3))

    f = pick(p2, "t2", (0, 0.8, -1), (-50, 102.5, -51), (1, 0, 0))
    sg = np.sign(f[1][0])
    for uc, w, h, sk in ((-46, 6.5, 17, 7), (-18, 7.0, 18, -6), (14, 6.5, 17, 7)):
        cuts.append(carve(f, slash(sg * uc, 0, w, h, sk * sg), 5.2))
    poly_l = [(-56, -9), (18, -9), (28, -1), (28, 10), (-48, 10), (-56, 1)]
    gfx.append(carve(f, [(sg * a, b) for a, b in poly_l], 1.4))
    # ---- tier 2 nose faces: graphite chevron stripe + a blue hatch comb -- the
    # two sides don't match: +z keeps the round-2 chevron, -z runs a bigger one
    # with an extra hatch tick, like there's more to route past on that side
    f = pick(p2, "t2", (1, 0.93, 0.29), (47, 102.5, 15), (0, 0, -1))
    gfx.append(carve(f, polyline_band([(-9, -5.5), (5, -5.5), (9.5, -1.0), (9.5, 4.0)], 3.2), 1.2))
    blue.append(union([carve(f, slash(uc, 3.5, 1.3, 6.0, 2.8), 1.0) for uc in (-9, -5, -1)]))

    f = pick(p2, "t2", (1, 0.93, -0.29), (47, 102.5, -15), (0, 0, 1))
    gfx.append(carve(f, polyline_band([(-13, -6.5), (3, -6.5), (10, 2.0), (10, 8.5)], 3.6), 1.3))
    blue.append(union([carve(f, slash(uc, 4.0, 1.4, 7.0, 3.0), 1.1) for uc in (-11, -6, -1, 4)]))
    # ---- tier 2 back faces: two chamfered windows, uneven -- one bigger than the other
    f = pick(p2, "t2", (-1, 1.0, 0), (-150, 102.5, 0), (0, 0, 1))
    pockets.append(carve(f, chamfer_rect(-34, -14, -2.6, 2.6, 1.8), 1.6))
    pockets.append(carve(f, chamfer_rect(6, 34, -3.6, 3.6, 2.0), 1.6))
    # ---- tier 1 sides: a row of chamfered windows (the coupler's motif), plus a
    # rivet row near the base -- future-split hint: the screw line between the
    # cap assembly ("top parts") and the lower body, under the light band.
    # Bigger than round 2, and the two sides no longer match.
    for zs, u0s, riv0s in ((1, (-112, -94, -78, 10, 30), (-103, -66, -40, -8, 20)),
                           (-1, (-108, -86, -60, -30, 14, 32), (-98, -72, -46, -16, 24))):
        f = pick(p1, "t1", (0, 0.5, zs), (-50, 87, zs * 80), (1, 0, 0))
        sg = np.sign(f[1][0])
        pockets.append(union([carve(f, chamfer_rect(sg * u0 - 6.2, sg * u0 + 6.2, -3.0, 3.0, 1.8), 1.8)
                              for u0 in u0s]))
        gfx.append(union([carve(f, chamfer_rect(sg * u0 - 1.2, sg * u0 + 1.2, -6.2, -3.6, 0.5), 0.7)
                          for u0 in riv0s]))
    # ---- tier 1 back: three chamfered windows, uneven sizes, bare in round 2
    f = pick(p1, "t1", (-1, 0.5, 0), (-166, 87, 0), (0, 0, 1))
    pockets.append(carve(f, chamfer_rect(-42, -30, -3.0, 3.0, 1.6), 1.6))
    pockets.append(carve(f, chamfer_rect(-10, 14, -3.8, 3.8, 1.8), 1.6))
    pockets.append(carve(f, chamfer_rect(26, 36, -2.4, 2.4, 1.3), 1.6))
    # ---- tier 1 front: a blue brow line with a dogleg
    f = pick(p1, "t1", (1, 0.5, 0), (66, 87, 0), (0, 0, -1))
    blue.append(carve(f, polyline_band([(-56, -1.2), (-20, -1.2), (-16, 1.2), (56, 1.2)], 1.4), 1.0))
    # ---- armour: graphite bus with 45-degree bends, blue traces with a comb, pads,
    # windows -- a stray branch breaks off and runs further on the -z side than
    # anything mirrors on +z, like it's actually routed to something over there
    top = face_frame((CAP2_SHIFT[0], Y_TOP + ARMOUR_H1 + ARMOUR_H2, CAP2_SHIFT[1]), (0, 1, 0), (1, 0, 0))  # u = x, v = -z
    bus = carve(top, polyline_band([(-128, 12), (-90, 12), (-76, -2), (-30, -2), (-16, 12), (18, 12)], 9), 1.3) & arm_top
    traces = union([
        carve(top, polyline_band([(-124, -20), (-78, -20), (-70, -27), (-20, -27), (-12, -20), (12, -20)], 1.6), 1.0),
        carve(top, polyline_band([(-118, 25), (-98, 25), (-92, 19)], 1.6), 1.0),
        carve(top, polyline_band([(-40, 23), (-6, 23)], 1.6), 1.0),
        carve(top, polyline_band([(-30, 22), (-30, 26), (-50, 26)], 1.4), 1.0),
        union([carve(top, polyline_band([(xc, -32), (xc, -23)], 1.3), 1.0) for xc in (-52, -48, -44)]),
        union([carve(top, chamfer_rect(x - 2.3, x + 2.3, z - 2.3, z + 2.3, 0.9), 1.0)
               for x, z in ((-124, -20), (12, -20), (-118, 25), (-92, 19), (-40, 23), (-6, 23), (-50, 26))]),
    ]) & arm_top
    # both panel lines WANDER instead of tracing a smooth, constant-radius
    # offset -- a picture-frame line is exactly what reads as "designed", so
    # neither ring stays a fixed distance from the plate edge
    armour_frame = face_frame((CAP2_SHIFT[0], Y_TOP, CAP2_SHIFT[1]), (0, 1, 0), (1, 0, 0))
    frame_line = jagged_ring(armour_frame, ARMOUR_BASE_FULL, 4.6, 2.6, 1.3, 0.8, seed=0.0)
    # a second, wider panel line out on the bare tier-2 top field -- another
    # stepped-plate seam, echoing the armour's own frame line, cutting on its
    # own schedule (different seed) so the two never run parallel
    frame_line2 = jagged_ring(armour_frame, ARMOUR_BASE_FULL, 12.5, 4.0, 1.6, 0.6, seed=3.3)
    # rivets scattered between the two panel lines, also off an irregular path
    base_frame = armour_frame
    dense_rp = densify(ARMOUR_BASE_FULL, 10.0)
    ring_pts = offset_plan_var(dense_rp, [jag(i, 7.0, 8.6, seed=5.0) for i in range(len(dense_rp))])
    gfx.append(union([carve(base_frame, chamfer_rect(x - 1.1, x + 1.1, -z - 1.1, -z + 1.1, 0.4), 0.6)
                      for x, z in ring_pts]))
    win_top = union([carve(top, chamfer_rect(x - 5.5, x + 5.5, 25.0, 28.0, 1.0), 1.4) for x in (-84, -72)]) & arm_top
    # the tier-1/tier-2 ledge itself was the real offender in the top view: a
    # smooth, near-constant-width shoulder all the way round tier 2's footprint
    # reads as a lofted oval at a glance, even though it's built from straight
    # facets. tier 2's volume has to stay convex (frustum_walls/polyhedron), so
    # the fix isn't to that solid -- a flat groove cut into the ledge turned out
    # to be invisible from directly above (no silhouette on a flat top face).
    # A trim raised proud of it, zigzagging sharply (concave corners are fine
    # for an extrude), actually casts the edge the eye follows.
    # the swing is now a big fraction of the ledge's own size, not a small
    # wobble on top of a smooth offset -- sometimes it hugs tier 2 closely,
    # sometimes it reaches nearly to tier 1's real edge, with no steady
    # relationship to either boundary. The clip below is what makes this safe:
    # whatever reaches past real material just gets cut away cleanly.
    t2_plan = [(x + CAP2_SHIFT[0], z + CAP2_SHIFT[1]) for x, z in T2_PLAN]
    ledge_frame = face_frame((CAP2_SHIFT[0], Y_T2, CAP2_SHIFT[1]), (0, 1, 0), (1, 0, 0))
    fence = jagged_fence(ledge_frame, t2_plan, 9.0, 8.0, 2.4, 1.8, seed=9.0)
    # the fore-aft cantilever (CAP1/CAP2_SHIFT) narrows the ledge unevenly, so
    # clip the fence to tier 1's own sloped wall (extended past Y_T2 at the
    # same slope) rather than trust a fixed width -- it was poking past the
    # real silhouette at the nose before this.
    t1_prism = polyhedron([p for p in tier1_planes() if p[2] == "t1"])
    fence = fence & t1_prism

    vent_cuts = union(cuts)
    pk = union(pockets)
    capc = cp - vent_cuts - pk - frame_line - frame_line2
    g = union(gfx) & capc
    b = union(blue) & capc
    accent = ring + b + (traces - bus)
    graphite = chin + back_panel + (g - b) + (bus - traces) + (arm_base - arm_top) + bezel_riv + fence
    armour_w = arm_top - bus - traces - win_top
    white = (low - chin - back_panel) + (capc - g - b) + armour_w
    return [("white", white, WHITE), ("graphite", graphite, GRAPHITE), ("dark", lining, DARK),
            ("accent", accent, BLUE)]


def preview(bodies, path, title=""):
    from render3d import tessellate
    from render_color import view, sheet
    tb, allV = [], []
    for n, b, rgb in bodies:
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
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    bodies = concept_facet()
    for n, b, _ in bodies:
        sols = b.solids()
        bb = b.bounding_box()
        print(f"  {n:9s} {sum(s.volume for s in sols) / 1000:8.2f} cm3  {len(sols)} solid(s)  "
              f"{(round(bb.min.X), round(bb.min.Y), round(bb.min.Z), round(bb.max.X), round(bb.max.Y), round(bb.max.Z))}",
              flush=True)
    stepcolor.write(bodies, os.path.join(out, "BoxConcept_D_facet.step"), part="BoxConcept_D_facet")
    preview(bodies, os.path.join(out, "preview_D_facet.png"), "Box concept D  FACET")
    print("  wrote", flush=True)
