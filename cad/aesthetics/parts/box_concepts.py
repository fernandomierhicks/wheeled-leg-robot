"""Body-box concepts (2026-10-04, round 3 of 3): sculpted in three views.  ROBOT
coordinates, mm.  Concept only -- not styled parts, not checked for printing.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/aesthetics/parts/box_concepts.py <out_dir> [A_helm B_carapace C_shells]

writes BoxConcept_<name>.step (white / graphite / dark / blue bodies, via
lib/stepcolor) and preview_<name>.png per concept.  The STEPs are imported into
SolidWorks and put on the real robot by cad/solidworks_api/20_box_concepts.py.

Every shell is the intersection of three extrusions -- side silhouette (XY),
plan (XZ) and front section (YZ).  The plan is exactly |z| = 80 along the
RobotMounts (x -150..50) so the flanks sit on them, and TAPERS beyond them: the
nose narrows to the screen, the tail to a point.  The inner surface is the
same construction from every profile offset in by the wall.

Kept by every concept (measured on the v5 CAD): open at the RobotMount
footprint (x -150..50, y < 75) and at the bottom (floor underside y = 15);
never wider than |z| 80; the OLED (face x 53.8, y 29..66, z -45.3..36) in a
window; the power switch (back, y 45, z 50); an RGB light band all the way
round near the top.
"""
import os
import sys
import math
import numpy as np
from build123d import Box, Pos, Rot, Plane, Polyline, make_face, extrude, Cylinder

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
BIG = 400.0


# ---------------------------------------------------------------- 2D helpers
def clean(pts):
    out = []
    for p in pts:
        p = (float(p[0]), float(p[1]))
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-4:
            out.append(p)
    if len(out) > 2 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) < 1e-4:
        out.pop()
    return out


def smooth(points, n=20):
    """Catmull-Rom through points (open)."""
    P = np.array(points, float)
    out = []
    for i in range(len(P) - 1):
        p0 = P[i - 1] if i > 0 else P[i]
        p1, p2 = P[i], P[i + 1]
        p3 = P[i + 2] if i + 2 < len(P) else P[i + 1]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    return [tuple(p) for p in out]


def poly_offset(pts, d, floor=None, axis=1):
    """Mitred offset of a closed polygon (d < 0 = inward).  floor: vertices that
    land at/below floor + |d| on `axis` drop to -BIG (an open bottom)."""
    P = np.array(clean(pts), float)
    if np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]) < 0:
        P = P[::-1]
    out = []
    n = len(P)
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        e1, e2 = b - a, c - b
        n1 = np.array([-e1[1], e1[0]]) / np.linalg.norm(e1)
        n2 = np.array([-e2[1], e2[0]]) / np.linalg.norm(e2)
        out.append(b + (-d) * (n1 + n2) / max(1.0 + n1 @ n2, 1e-3))
    out = np.array(out)
    if floor is not None:
        out[out[:, axis] <= floor + abs(d) + 1e-3, axis] = -BIG
    return [tuple(q) for q in out]


def rounded_rect(y0, y1, zh, r, k=20):
    """Front section (y, z): flanks z = +-zh, top corners radius r."""
    pts = [(y0, -zh), (y0, zh), (y1 - r, zh)]
    pts += [(y1 - r + r * math.sin(t), zh - r + r * math.cos(t)) for t in np.linspace(0, math.pi / 2, k)[1:]]
    pts += [(y1 - r + r * math.sin(t), -zh + r + r * math.cos(t)) for t in np.linspace(math.pi / 2, math.pi, k)[1:]]
    return pts


def mirror_plan(half):
    """Plan half-outline (x, z>=0) from front to back -> closed (x, z) polygon."""
    return half + [(x, -z) for x, z in reversed(half)]


# ---------------------------------------------------------------- 3D helpers
def boxp(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)


def slab_y(y0, y1):
    return boxp(-BIG, BIG, y0, y1, -BIG, BIG)


def e_side(pts, half=BIG):          # XY profile along Z
    return Pos(0, 0, -half) * extrude(make_face(Polyline(*clean(pts), close=True)), amount=2 * half)


def e_plan(pts, y0=-BIG, y1=BIG):   # (x, z) profile along Y
    f = Plane.XZ * make_face(Polyline(*[(x, -z) for x, z in clean(pts)], close=True))
    return Pos(0, y1, 0) * extrude(f, amount=y1 - y0)


def e_front(pts, x0=-BIG, x1=BIG):  # (y, z) profile along X
    f = Plane.YZ * make_face(Polyline(*clean(pts), close=True))
    return Pos(x0, 0, 0) * extrude(f, amount=x1 - x0)


def window():
    return boxp(40, 90, OLED["y0"], OLED["y1"], OLED["z0"], OLED["z1"])


def switch_hole():
    return Pos(-150, SWITCH["y"], SWITCH["z"]) * Rot(0, 90, 0) * Cylinder(SWITCH["r"], 120)


class Sculpt:
    """outer = side x plan x front; inner(d) = the same from offset profiles."""

    def __init__(self, side, plan, front):
        self.side, self.plan, self.front = side, plan, front
        self.outer = e_side(side) & e_plan(plan) & e_front(front)

    def inner(self, d, open_rm=True):
        fl = Y_FLOOR if open_rm else None
        s = poly_offset(self.side, -d, fl)
        p = poly_offset(self.plan, -d)
        f = poly_offset(self.front, -d, fl, axis=0)
        closed = e_side(s) & e_plan(p) & e_front(f)
        if not open_rm:
            return closed
        return closed + (e_side(s) & boxp(RM["x0"], RM["x1"], -BIG, RM["y1"], -BIG, BIG))

    def shell(self):
        return self.outer - self.inner(WALL)

    def skin(self, d):
        return self.outer - self.inner(d, open_rm=False)

    def grown(self, d):
        return e_side(poly_offset(self.side, d)) & e_plan(poly_offset(self.plan, d)) & \
            e_front(poly_offset(self.front, d))


def trace_prism(pts_xz, width=1.8, y0=60, y1=200):
    parts = []
    for (x0, z0), (x1, z1) in zip(pts_xz[:-1], pts_xz[1:]):
        L = math.hypot(x1 - x0, z1 - z0)
        ang = math.degrees(math.atan2(z1 - z0, x1 - x0))
        parts.append(Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Rot(0, -ang, 0) * Box(L + width, y1 - y0, width))
    for x, z in (pts_xz[0], pts_xz[-1]):
        parts.append(Pos(x, (y0 + y1) / 2, z) * Box(width * 2.4, y1 - y0, width * 2.4))
    p = parts[0]
    for q in parts[1:]:
        p = p + q
    return p


def rm_plan_half(nose, tail):
    """Plan half outline: nose points (front), flat flank on the RobotMounts, tail points."""
    return nose + [(RM["x1"], ZW), (RM["x0"], ZW)] + tail


# --------------------------------------------------------------------------- A
def concept_helm():
    """HELM: a forward-leaning helmet.  Tapered nose holding a wrap-around dark
    visor, high crown, swept tail, a graphite crest down the spine, the light
    line round the shoulders."""
    side = [(-160, Y_FLOOR), (58, Y_FLOOR)] + smooth(
        [(58, Y_FLOOR), (61.5, 27), (62.5, 48), (60.5, 70), (54, 80), (34, 92), (-20, 108), (-75, 117),
         (-125, 112), (-160, 92), (-182, 62), (-184, 40), (-170, 22), (-160, Y_FLOOR)])
    nose = smooth([(63, 0), (62.5, 40), (60, 60), (55, 73), (RM["x1"], ZW)], 12)[:-1]
    tail = smooth([(RM["x0"], ZW), (-166, 68), (-180, 44), (-186, 0)], 12)[1:]
    plan = mirror_plan(rm_plan_half(nose, tail))
    front = rounded_rect(Y_FLOOR, 122, ZW, 50)
    S = Sculpt(side, plan, front)
    shell = S.shell() - window() - switch_hole()
    visor = (S.skin(WALL + 0.01) & boxp(38, 90, 25, 79.5, -BIG, BIG) & shell) - window()
    jaw = shell & slab_y(Y_FLOOR, 25)
    tail_b = (shell & boxp(-BIG, -162, 25, 79.5, -BIG, BIG)) - switch_hole()
    ring = shell & S.skin(1.6) & slab_y(80, 84)
    crest = (S.grown(2.5) - S.outer) & boxp(-176, 46, 84.5, BIG, -10, 10)
    tr = (trace_prism([(-128, 30), (-74, 30), (-66, 22), (-14, 22)]) +
          trace_prism([(-120, -30), (-64, -30), (-56, -22), (-4, -22)])) & S.skin(1.2) & slab_y(86, BIG) & shell
    white = shell - visor - jaw - tail_b - ring - tr
    return [("white", white, WHITE), ("graphite", jaw + tail_b + crest, GRAPHITE), ("dark", visor, DARK),
            ("accent", ring + tr, BLUE)]


# --------------------------------------------------------------------------- B
def superellipse_half(cx, cy, a, b, n, t0, t1, k=80):
    pts = []
    for t in np.linspace(t0, t1, k):
        c, s = math.cos(t), math.sin(t)
        pts.append((cx + a * np.sign(c) * abs(c) ** (2 / n), cy + b * np.sign(s) * abs(s) ** (2 / n)))
    return pts


def concept_carapace():
    """CARAPACE: a smooth superelliptic shell.  The front flattens into a face for
    the screen, a halo light ring circles the crown, a graphite spine with a
    trace runs nose to tail."""
    side = [(x, y) for x, y in superellipse_half(-55, Y_FLOOR, 117, 100, 3.0, 0, math.pi)]
    side = [(min(x, 58.0), y) for x, y in side]
    nose = superellipse_half(RM["x1"], 0, 12, ZW, 2.6, 0, math.pi / 2, 30)          # x 50 -> 62
    tail = superellipse_half(RM["x0"], 0, 30, ZW, 2.6, math.pi / 2, math.pi, 30)    # x -150 -> -180
    half = [(x, z) for x, z in nose] + [(RM["x0"], ZW)] + [(x, z) for x, z in tail[1:]]
    plan = mirror_plan(half)
    front = [(Y_FLOOR, -ZW), (Y_FLOOR, ZW)] + [(y, z) for z, y in superellipse_half(0, 0, ZW, 112, 2.6, 0, math.pi, 80)
                                               if y >= Y_FLOOR]
    front = [(max(y, Y_FLOOR), z) for y, z in front]
    S = Sculpt(side, plan, front)
    shell = S.shell() - window() - switch_hole()
    ann = e_plan(mirror_plan([(-55 + 62 * math.cos(t), 48 * math.sin(t)) for t in np.linspace(0, math.pi, 60)])) - \
        e_plan(mirror_plan([(-55 + 57 * math.cos(t), 43 * math.sin(t)) for t in np.linspace(0, math.pi, 60)]))
    ring = shell & ann & S.skin(1.6) & slab_y(90, BIG)
    spine = ((S.grown(2.0) - S.outer) & boxp(-200, 56, 70, BIG, -9, 9)) - ann
    face = (S.skin(WALL + 0.01) & boxp(52, 90, 21, 77, -62, 54) & shell) - window()
    belly = shell & slab_y(Y_FLOOR, 34)
    tr = trace_prism([(-140, 3.5), (-118, 3.5), (-110, -3.5), (-88, -3.5)], width=1.6) & spine
    white = shell - ring - face - belly
    return [("white", white, WHITE), ("graphite", (spine - tr) + belly, GRAPHITE), ("dark", face, DARK),
            ("accent", ring + tr, BLUE)]


# --------------------------------------------------------------------------- C
def concept_shells():
    """SHELLS: GLACIER's stepped, overlapping plates.  A faceted lower body that
    tapers to a chevron nose and a tail; a recessed light ring all the way
    round; a two-step cap floating 3 mm proud over it."""
    P = [(54.3, Y_FLOOR), (66, 32), (66, 60), (58, 78), (-156, 78), (-166, 56), (-160, 26), (-150, Y_FLOOR)]
    half = [(66, 0), (66, 40), (58, 64), (RM["x1"], ZW), (RM["x0"], ZW), (-160, 66), (-166, 40), (-166, 0)]
    plan = mirror_plan(half)
    front = [(Y_FLOOR, -ZW), (Y_FLOOR, ZW), (78, ZW), (78, -ZW)]
    S = Sculpt(P, plan, front)
    body = S.shell()
    well = boxp(56.3, 120, 25, 70, -52, 43)
    lining = (S.outer & boxp(53.8, 56.3, 25, 70, -52, 43)) - window()
    body = body - well - window() - switch_hole()
    chin = body & slab_y(Y_FLOOR, 30)
    ring = (e_plan(poly_offset(plan, -3.0)) - e_plan(poly_offset(plan, -5.5))) & slab_y(78, 83)
    cap_lo = e_plan(poly_offset(plan, 3.0)) & slab_y(83, 88)
    cap_hi = e_plan(poly_offset(plan, -2.0)) & slab_y(88, 92)
    cap = (cap_lo + cap_hi) - (e_plan(poly_offset(plan, 0.5)) & slab_y(70, 89.5))
    armour = e_plan(poly_offset(mirror_plan([(14, 0), (14, 30), (4, 44), (-112, 44), (-124, 30), (-124, 0)]), 0)) \
        & slab_y(92, 95)
    tr = (trace_prism([(-110, 30), (-52, 30), (-44, 22), (0, 22)]) +
          trace_prism([(-106, -12), (-62, -12), (-54, -20), (-14, -20)]) +
          trace_prism([(-114, 2), (-82, 2)])) & armour & slab_y(93.8, 96)
    white = (body - chin) + cap
    return [("white", white, WHITE), ("graphite", chin + (armour - tr), GRAPHITE), ("dark", lining, DARK),
            ("accent", ring + tr, BLUE)]


CONCEPTS = {"A_helm": concept_helm, "B_carapace": concept_carapace, "C_shells": concept_shells}


def preview(bodies, path, title=""):
    from render3d import tessellate
    from render_color import view, sheet
    tb, allV = [], []
    for n, b, rgb in bodies:
        V, T, _ = tessellate(b, 0.2)
        V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)
        tb.append((V, T, rgb))
        allV.append(V)
    allV = np.vstack(allV)
    tiles = [("front 3/4", "", view(tb, allV, (760, 560), 22, -38, light="camera")),
             ("rear 3/4", "", view(tb, allV, (760, 560), 26, -140, light="camera")),
             ("side", "", view(tb, allV, (760, 560), 0.5, -90, light="camera")),
             ("top", "", view(tb, allV, (760, 560), 89.5, -90, light="camera"))]
    sheet(tiles, 2, path, header=title, size=(760, 560))


if __name__ == "__main__":
    out = sys.argv[1]
    which = sys.argv[2:] or list(CONCEPTS)
    os.makedirs(out, exist_ok=True)
    for name in which:
        bodies = CONCEPTS[name]()
        for n, b, _ in bodies:
            sols = b.solids() if b is not None else []
            v = sum(s.volume for s in sols)
            bb = b.bounding_box() if sols else None
            print(f"  {name}: {n:9s} {v / 1000:8.2f} cm3  {len(sols)} solid(s)  "
                  f"{'' if bb is None else (round(bb.min.X), round(bb.min.Y), round(bb.min.Z), round(bb.max.X), round(bb.max.Y), round(bb.max.Z))}",
                  flush=True)
        stepcolor.write(bodies, os.path.join(out, f"BoxConcept_{name}.step"), part=f"BoxConcept_{name}")
        preview(bodies, os.path.join(out, f"preview_{name}.png"), f"Box concept {name}")
        print("  wrote", name, flush=True)
