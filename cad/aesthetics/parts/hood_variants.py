"""FacetHood variants (2026-10-07).  His brief: "make the facet hood more aggressive
looking, it has large area but little details; the links are more densely
populated with aesthetics."  His calls on the questions:

  * reshape the TOP (tier 2's roof + the armour) and fill the hood to link density;
    the skirt, tier 1, the strip channel, the ring fit and the screws stay as they are
  * graphite-heavy, like the links (graphite fields under white frames, blue
    traces with doglegs + hatch combs over the graphite)
  * NOTHING above today's armour top, y 117.2
  * 2-3 variants as offline renders; he picks one, and only that one goes into
    box_facet_print.py -> SolidWorks (21) -> pipes (24) -> 3MF

    C:/Users/ferna/cadenv/Scripts/python.exe cad/aesthetics/parts/hood_variants.py <out_dir> [now A B C] [--sheets] [--board]

builds each (default: today + all three), prints volumes and the checks; --sheets
writes a 4-view sheet per hood, --board one comparison board of them on the robot.

ROBOT d75 coordinates, mm (x forward, y up, z right), as box_facet_print.py.
The hood interior is empty above y 80 in the CAD (checked against all 226
components of the 24 sweep), so a RECESS is pressed into the cavity (`press`):
the outline grown by WALL is backed down to depth + WALL first, so every wall
stays 2.5 mm and nothing opens into the hood.
"""
import os
import sys
import time
import numpy as np
import shapely.geometry as sg
from build123d import Plane, Polyline, make_face, extrude, Pos

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)
import box_facet_print as B
import box_concept_facet as K

WHITE, GRAPHITE, BLUE, DARK = K.WHITE, K.GRAPHITE, K.BLUE, K.DARK
WALL, BIG = B.WALL, K.BIG
Y_MAX = 117.2                     # today's armour top: the height cap (his call)
Y_TOP = B.Y_TOP                   # 113, today's tier-2 roof
Y_MECH = 99.5                     # below this (skirt, ring fit, screws, tier 1, strip channel) nothing may change
Y_T2_MIN = B.Y_CHAN + 0.5         # no tier-2 feature (cut or backing) reaches below this


# ------------------------------------------------------------------ the hood body
class Hood:
    """The hood shell with a given tier-2 plane set, plus everything a variant
    needs to add to it: `adds` (unioned), `cuts` (subtracted), colour regions
    `gfx` / `blue` / `dark` (intersected with the final solid)."""

    def __init__(self, p2_mod=None):
        self.S = B.strip_wall()
        self.p1 = B.t1_planes()
        p2 = B.t2_planes(self.S)
        self.p2 = p2_mod(p2) if p2_mod else p2
        skirt_o = B.prism_xz(B.H0, B.Y_RING_T, B.Y_SKIRT)
        skirt_i = B.prism_xz(B.soff(B.H0, -WALL), B.Y_RING_T - 1, B.Y_SKIRT + 0.01)
        o1 = K.polyhedron(self.p1)
        i1 = K.polyhedron(self.p1, WALL, skip=("base",)) & K.slab_y(B.Y_SKIRT, BIG)
        chan_o = B.prism_xz(self.S, B.Y_LEDGE - 0.01, B.Y_CHAN + 0.01)
        chan_i = B.prism_xz(B.soff(self.S, -B.CHAN_WALL), B.Y_LEDGE - 4, B.Y_CHAN + 0.5)
        self.o2 = K.polyhedron(self.p2)
        i2 = K.polyhedron(self.p2, WALL, skip=("base",)) & K.slab_y(B.Y_CHAN, BIG)
        self.outer = skirt_o + o1 + chan_o + self.o2
        self.void = skirt_i + i1 + chan_i + i2
        shell = self.outer - self.void
        xb = min(p[0] for p in B.pts(self.S))
        shell = shell - B.bore((xb + 1, B.Y_LEDGE + 3.0, 0), (1, 0, 0), 2.5, B.CHAN_WALL + 2)
        for x, z, n in B.HOOD_SCREWS:
            p = B.skirt_point(x, z, n)
            d = -np.asarray(n, float)
            shell = shell - B.bore(p, d, B.CLEAR, WALL + 1) - B.csk(p, d)
        self.shell = shell
        self.ops = []                 # ("+", solid) / ("-", solid), applied IN ORDER
        self.gfx, self.blue, self.dark = [], [], []
        self.gfx_over = []            # graphite inlays ON raised white features (win over their white)
        self.pipes = []               # (24_pipes pipe dict, face height y): hoses on a horizontal face
        self.white_adds = []          # raised white features (added, and kept white)

    def add(self, s):
        self.ops.append(("+", s))

    def cut(self, s):
        self.ops.append(("-", s))

    # -- faces
    def t2(self, direction, target, u_hint):
        return K.pick(self.p2, "t2", direction, target, u_hint)

    def t1(self, direction, target, u_hint):
        return K.pick(self.p1, "t1", direction, target, u_hint)

    def roof_skin(self, t):
        """The layer within t (vertically) under every up-facing surface of tier 2."""
        o = self.o2 & K.slab_y(B.Y_CHAN + 0.5, BIG)
        return o - (Pos(0, -t, 0) * o)

    # -- operations
    def press(self, frame, uv, depth, colour=None, floor_t=1.2, y_min=None):
        """A recess `depth` deep under a face, backed so the wall stays WALL thick:
        the outline grown by WALL gets the layer depth..depth+WALL under the face's
        own inner surface (never above it, so it cannot refill a neighbour's recess).
        colour: the floor's inlay colour ("gfx" / "blue" / "dark"), floor_t thick."""
        grown = B.pts(B.soff(sg.Polygon(uv), WALL))
        keep = self.outer if y_min is None else self.outer & K.slab_y(y_min, BIG)
        self.add(K.carve(frame, grown, depth + WALL, lift=-WALL) & keep)
        c = K.carve(frame, uv, depth)
        self.cut(c if y_min is None else c & K.slab_y(y_min, BIG))
        if colour:
            o, u, v, n = frame
            fl = (o - depth * n, u, v, n)
            getattr(self, colour).append(K.carve(fl, uv, floor_t, lift=0.3))

    def press_plan(self, region_xz, depth, colour=None, floor_t=1.2):
        """A recess under the ROOF following its slope, outline given in plan (x, z)."""
        grown = B.soff(region_xz, WALL)
        self.add(B.prism_xz(grown, 0, BIG) & (self.roof_skin(depth + WALL) - self.roof_skin(WALL)))
        self.cut(B.prism_xz(region_xz, 0, BIG) & self.roof_skin(depth))
        if colour:
            fl = Pos(0, -depth, 0) * (B.prism_xz(region_xz, 0, BIG) & self.roof_skin(floor_t))
            getattr(self, colour).append(fl)

    def inlay_plan(self, region_xz, colour, t=1.2):
        """A flush colour region on the roof, outline in plan."""
        getattr(self, colour).append(B.prism_xz(region_xz, 0, BIG) & self.roof_skin(t))

    def raise_(self, frame, uv, h, colour=None, taper=0.0):
        """A plate standing h proud of a face (taper: wall draft in degrees)."""
        o, u, v, n = frame
        pl = Plane(origin=tuple(map(float, o - 0.2 * n)), x_dir=tuple(map(float, u)), z_dir=tuple(map(float, n)))
        f = pl * make_face(Polyline(*[(float(a), float(b)) for a, b in uv], close=True))
        for tp in dict.fromkeys((taper, taper / 2, 0.0)):        # thin clipped ends can't take the draft
            try:
                s = extrude(f, amount=h + 0.2, taper=tp)
                if s.is_valid and s.volume > 0:
                    break
            except Exception:
                pass
        if tp != taper:
            print(f"    raise_: draft {taper} -> {tp} on a {f.area:.1f} mm2 plate")
        self.add(s)
        if colour == "white":
            self.white_adds.append(s)
        elif colour:
            getattr(self, colour).append(s)
        return s

    # -- result
    def bodies(self):
        solid = self.shell
        i = 0
        while i < len(self.ops):                     # consecutive same-sign ops as one boolean
            j = i
            while j < len(self.ops) and self.ops[j][0] == self.ops[i][0]:
                j += 1
            grp = B.union([s for _, s in self.ops[i:j]])
            solid = solid + grp if self.ops[i][0] == "+" else solid - grp
            i = j
        out = {}
        taken = None
        for name in ("blue", "dark", "gfx"):
            regs = getattr(self, name)
            if not regs:
                continue
            r = B.union(regs) & solid
            if self.white_adds:
                r = r - B.union(self.white_adds) if name == "gfx" else r
            if name == "gfx" and self.gfx_over:
                r = r + (B.union(self.gfx_over) & solid)
            if taken is not None:
                r = r - taken
            out[name] = r
            taken = r if taken is None else taken + r
        white = solid - taken if taken is not None else solid
        res = {"white": white, "graphite": out.get("gfx"), "blue": out.get("blue"), "dark": out.get("dark")}
        for pd, y in self.pipes:
            hose, collars = field_pipe(pd, y, solid)
            res["blue"] = B.union([res["blue"], hose])
            res["graphite"] = B.union([res["graphite"]] + collars)
            solid = B.union([solid, hose] + collars)
        return res, solid


# ------------------------------------------------------------------ helpers
def deck(y=Y_TOP, x0=0.0):
    """Frame on a horizontal plane at y: u = x, v = -z (seen from above, front right)."""
    return K.face_frame((x0, y, 0), (0, 1, 0), (1, 0, 0))


def xz(points):
    """(x, z) plan points -> (u, v) on a deck() frame."""
    return [(x, -z) for x, z in points]


def P(points):
    return sg.Polygon(points)


def deck_outline(hood, y=Y_TOP):
    s = hood.o2 & K.slab_y(y - 0.3, y)
    f = max(s.faces(), key=lambda f: f.area if abs(f.normal_at().Y - 1) < 1e-6 else 0)
    return sg.MultiPoint([(v.X, v.Z) for v in f.outer_wire().vertices()]).convex_hull     # convex (polyhedron face)


def full(half):
    return K.mirror(half)


def prism_any(region, y0, y1):
    """B.prism_xz for a Polygon or a MultiPolygon (slivers under 0.5 mm2 dropped)."""
    parts = [B.prism_xz(g, y0, y1) for g in getattr(region, "geoms", [region]) if g.area > 0.5]
    return B.union(parts)


def zedge(poly, x, s):
    """The outline's z extent at x on side s (+1 / -1)."""
    seg = poly.intersection(sg.LineString([(x, -BIG), (x, BIG)]))
    zs = [c[1] for c in (seg.coords if hasattr(seg, "coords") else [q for g in seg.geoms for q in g.coords])]
    return max(zs) if s > 0 else min(zs)


def tab(x0, x1, z_in, s, reach=30.0):
    """A frame tab reaching into a field from side s: inner edge at z_in from x0 to
    x1, 45-degree flanks running out past the field edge."""
    return P([(x0 - reach, z_in + s * reach), (x0, z_in), (x1, z_in), (x1 + reach, z_in + s * reach)])


def para(x, z0, z1, w, rake):
    """A slat across z0 -> z1, w long in x, its far end raked back by `rake` (x)."""
    return P([(x, z0), (x - w, z0), (x - w - rake, z1), (x - rake, z1)])


def tilt(p2, sel, angle):
    """Re-slope the t2 faces picked by sel(normal) to `angle` (deg from horizontal),
    each turned about ITS OWN base edge at y = Y_CHAN -- the base outline (and with
    it the strip's lip) is unchanged."""
    out = []
    a = np.radians(angle)
    for pt, n, tag in p2:
        nn = K.unit(n)
        if tag == "t2" and sel(nn):
            hz = K.unit([nn[0], 0, nn[2]])
            out.append((pt, (hz[0] * np.sin(a), np.cos(a), hz[2] * np.sin(a)), tag))
        else:
            out.append((pt, n, tag))
    return out


# ------------------------------------------------------------------ reports
def report(name, bodies, solid, t0):
    vol = {k: B.solids_volume(v) / 1000 for k, v in bodies.items() if v is not None}
    tot = sum(vol.values())
    bb = solid.bounding_box()
    print(f"  {name}: {tot:.2f} cm3 (today 153.55)  " +
          "  ".join(f"{k} {v:.2f} ({100 * v / tot:.0f}%)" for k, v in vol.items()) +
          f"  white solids {len(bodies['white'].solids())}"
          f"  y {bb.min.Y:.2f}..{bb.max.Y:.3f}  |z| {max(-bb.min.Z, bb.max.Z):.2f}  ({time.time() - t0:.0f} s)",
          flush=True)
    if bb.max.Y > Y_MAX + 1e-3:
        print(f"  ** {name}: ABOVE THE CAP y {bb.max.Y:.3f} > {Y_MAX}")
    if max(-bb.min.Z, bb.max.Z) > B.Z_HOOD + 1e-3:
        print(f"  ** {name}: WIDER THAN |z| {B.Z_HOOD}")


def _bb_near(a, b, tol=0.01):
    return not (a.min.X > b.max.X + tol or b.min.X > a.max.X + tol or a.min.Y > b.max.Y + tol or
                b.min.Y > a.max.Y + tol or a.min.Z > b.max.Z + tol or b.min.Z > a.max.Z + tol)


def relief_zone():
    """Where the relief may change the hood below Y_MECH: the skirt's outer 1.7 mm away from
    the screws (SKIRT_KEEP - 0.4), and tier 1 between its base and 0.2 under the ledge
    (its backing goes into the empty cavity under it, never below Y_T1_MIN)."""
    skin = B.prism_xz(B.H0, B.Y_RING_T - 1.0, B.Y_SKIRT + 0.01) - B.prism_xz(B.soff(B.H0, -1.7), 0, BIG)
    for x, z, n in B.HOOD_SCREWS:
        p = B.skirt_point(x, z, n)
        skin = skin - B.bore(p + 2.0 * np.asarray(n, float), -np.asarray(n, float), SKIRT_KEEP - 0.4, 6.0)
    t1 = B.prism_xz(B.soff(B.H0, 0.5), Y_T1_MIN, B.Y_LEDGE - 0.2)
    return skin + t1


def checks(name, h, bodies, solid, ref_low):
    """Nothing floats (every colour solid touches another); nothing below Y_MECH
    differs from today's hood (skirt, ring fit, screws, tier 1, strip channel + lip);
    outside faces overhanging more than 50 deg in print orientation (skirt on the bed)."""
    from build123d import Vector
    sols = [(n, s, s.bounding_box()) for n, b in bodies.items() if b is not None for s in b.solids()]
    loose = []
    for i, (n, s, bb) in enumerate(sols):
        if not any(i != j and _bb_near(bb, tb) and s.distance_to(t) < 1e-4 for j, (_, t, tb) in enumerate(sols)):
            c = s.center()
            loose.append(f"{n} {s.volume:.2f} mm3 at ({c.X:.0f}, {c.Y:.0f}, {c.Z:.0f})")
    low = solid & K.slab_y(-BIG, Y_MECH)
    d_shape = (low - ref_low) + (ref_low - low)
    diff = B.solids_volume(d_shape)
    if diff > 1e-3:
        # the relief (2026-10-09) changes the skirt's outer skin and tier 1 ON PURPOSE; what
        # must not change is everything else down there: the skirt's inner 0.8 mm (the ring
        # fit), the screw countersinks, the ledge the strip sits on, the strip channel
        diff = B.solids_volume(d_shape - relief_zone())
    def base(o2):
        return sum(f.area for f in o2.faces() if abs(f.center().Y - B.Y_CHAN) < 1e-6)
    lip = base(h.o2) - base(Hood().o2)              # tier 2's base outline = the strip's lip
    voids = h.void.solids()
    parts = solid.solids()
    over, worst = 0.0, 0.0
    for s in solid.solids():
        for f in s.faces():
            c = f.center()
            if c.Y < 78.5:
                continue                                       # on the bed
            nv = f.normal_at(c)
            if s.is_inside(Vector(c.X + 0.05 * nv.X, c.Y + 0.05 * nv.Y, c.Z + 0.05 * nv.Z)):
                nv = -nv
            if nv.Y > -0.766:                                  # 50 deg past vertical
                continue
            p = Vector(c.X + 0.3 * nv.X, c.Y + 0.3 * nv.Y, c.Z + 0.3 * nv.Z)
            if any(v.is_inside(p) for v in voids):
                continue                                       # looks into the cavity: tree supports inside
            if any(q.is_inside(p) for q in parts):
                continue                                       # an interface between two colour bodies
            over += f.area
            worst = max(worst, np.degrees(np.arcsin(min(1.0, -nv.Y))))
    vol = B.solids_volume(solid) / 1000
    print(f"  {name} checks: loose solids {len(loose)}{' ' + '; '.join(loose) if loose else ''} | "
          f"below y {Y_MECH} vs today {diff:.4f} mm3, tier-2 base {lip:+.4f} mm2 | outside overhang > 50 deg {over:.1f} mm2"
          f"{f' (worst {worst:.0f} deg)' if over else ''} | ~{vol * 1.24:.0f} g PLA solid", flush=True)


# ------------------------------------------------------------------ render
def meshes(bodies, dev=0.12, frame=True):
    """(V, T, rgb) per colour body; frame=True: in the render frame (x, -z, y)."""
    from render3d import tessellate
    cols = {"white": WHITE, "graphite": GRAPHITE, "blue": BLUE, "dark": DARK}
    out = []
    for n, b in bodies.items():
        if b is None or not b.solids():
            continue
        V, T, _ = tessellate(b, dev)
        out.append((np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1) if frame else V, T, cols[n]))
    return out


SWAPI = os.path.normpath(os.path.join(HERE, "..", "..", "solidworks_api"))


_P24 = []


def p24():
    """cad/solidworks_api/24_pipes.py as a module (its name starts with a digit)."""
    if not _P24:
        import importlib
        sys.path.insert(0, SWAPI)
        _P24.append(importlib.import_module("24_pipes"))
    return _P24[0]


def field_pipe(pd, y, solid):
    """24_pipes' hose + collars on the horizontal face at y (its deck frame: u = x, v = -z),
    proud half only (clipped at the face) and minus the hood (raised features it meets)."""
    from build123d import Location, Compound
    P24 = p24()
    hose, collars, wire, run = P24.build_pipe(pd)
    loc = Location(Plane(origin=(0, y, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0)))
    keep = K.slab_y(y, BIG)

    def place(sh):
        g = (sh.moved(loc) & keep) - solid
        k = [x for x in g.solids() if x.volume > 1.0]
        return k[0] if len(k) == 1 else Compound(k)
    out = place(hose), [place(c) for c in collars]
    print(f"    pipe {pd['name']}: visible {run.length:.1f} mm (20..50), tightest bend {P24.min_radius(run):.1f} mm "
          f"(>= {P24.MIN_BEND}), hose {len(out[0].solids())} solid, collars {[len(c.solids()) for c in out[1]]}",
          flush=True)
    return out


def robot_scene(hip=20.0):
    """Everything but the hood at one hip angle (24_pipes' sweep meshes + styled-part
    exports + their pipes), ROBOT coords; the hood's tier-1 pipe only (its deck
    pipes get re-routed on the picked variant)."""
    from build123d import import_step
    P24 = p24()
    pj = __import__("json").load(open(os.path.join(P24.SWEEP, "poses.json")))
    k = int(np.argmin(np.abs(np.array(pj["hips"]) - hip)))
    own = {}
    for part in P24.PARTS:
        info = __import__("json").load(open(os.path.join(P24.SRC, f"{part}.json")))
        meshes_ = []
        if part != "FacetHood":
            meshes_ = [(V, T, np.array(rgb) * 255) for V, T, rgb in P24.Part(part).mesh()]
        pipe = os.path.join(P24.OUT, f"{part}_pipes.step")
        if os.path.exists(pipe):
            for s in import_step(pipe).solids():
                if part == "FacetHood" and s.bounding_box().max.Y > 100:
                    continue
                V, T = P24._tess(s, 0.04)
                meshes_.append((V, T, np.array(P24.BLUE if s.volume > 60 else P24.GRAPHITE) * 255))
        own[info["instances"][0]["name"]] = meshes_
    out, cache = [], {}
    hood_M = None
    for name, key in pj["mesh"].items():
        M = np.array(pj["placements"][name][k])
        if name == "Box-1/FacetHood-1":
            hood_M = M
        if name in own:
            out += [((V @ M[:, :3].T) + M[:, 3], T, col) for V, T, col in own[name]]
            continue
        if key not in cache:
            z = np.load(os.path.join(P24.SWEEP, key + ".npz"))
            cache[key] = (z["V"], z["T"])
        V, T = cache[key]
        out.append(((V @ M[:, :3].T) + M[:, 3], T, np.array([150, 155, 162.0])))
    return out, hood_M


def board(results, path):
    """Today + the variants on the robot: body front-right, body rear-left, hood top."""
    from render_color import view, sheet as rsheet
    F = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)          # ROBOT -> render frame
    scene, M = robot_scene()
    base = [(V @ F.T, T, col) for V, T, col in scene]
    tiles = []
    for name, bodies, sub in results:
        hood = [((V @ M[:, :3].T + M[:, 3]) @ F.T, T, col) for V, T, col in meshes(bodies, frame=False)]
        everything = base + hood
        allV = np.vstack([b[0] for b in everything])
        r = allV @ np.linalg.inv(F.T)                                  # back to ROBOT for the crop
        body = allV[(r[:, 0] > -178) & (r[:, 0] < 80) & (r[:, 1] > 8) & (r[:, 1] < 122) & (np.abs(r[:, 2]) < 95)]
        hv = np.vstack([b[0] for b in hood])
        t0 = time.time()
        tiles.append((f"{name}: front-right", sub, view(everything, body, (620, 480), 24, -50)))
        tiles.append((f"{name}: rear-left", "", view(everything, body, (620, 480), 28, 128)))
        tiles.append((f"{name}: hood from above", "", view(hood, hv, (620, 480), 89.5, -90, light="camera")))
        print(f"  board row {name} ({time.time() - t0:.0f} s)", flush=True)
    rsheet(tiles, 3, path, header=f"FacetHood -- {' / '.join(r[0] for r in results)} (hip +20)",
           sub="nothing above y 117.2; skirt, tier 1, strip channel, ring fit and screws unchanged; "
               "today's two deck pipes left off (a variant's own pipes are drawn)", size=(620, 480))


def sheet(name, bodies, path, sub=""):
    from render_color import view, sheet as rsheet
    tb = meshes(bodies)
    allV = np.vstack([t[0] for t in tb])
    tiles = [("front 3/4", "", view(tb, allV, (760, 560), 26, -36, light="camera")),
             ("rear 3/4", "", view(tb, allV, (760, 560), 30, -142, light="camera")),
             ("side", "", view(tb, allV, (760, 560), 6, -90, light="camera")),
             ("top", "", view(tb, allV, (760, 560), 89.5, -90, light="camera"))]
    rsheet(tiles, 2, path, header=f"FacetHood -- {name}", sub=sub, size=(760, 560))


# ------------------------------------------------------------------ variants
def current():
    shell, outer, p1, p2, S = B.hood_part()
    h = B.hood_details(shell, p1, p2, S)
    return {"white": h["white"], "graphite": h["graphite"], "blue": h["blue"], "dark": None},         B.union([h["white"], h["graphite"], h["blue"]])


K_A = 0.19                        # A's gable: y = Y_MAX - K_A |z|  (|z| 50 -> 107.7)


def massing_A(p2):
    """SPINE: the flat roof becomes a gable -- a ridge along z = 0 at the cap,
    one facet each side falling ~11 degrees to the side slopes."""
    p2 = [p for p in p2 if p[2] != "top"]
    return p2 + [((0, Y_MAX, 0), (0, 1, s * K_A), "roof") for s in (1, -1)]


C_TIPS = (30.0, -8.0, -46.0, -84.0)     # C's terrace step lines: chevron tips, front to back
C_STEP = 1.8                            # each terrace 1.8 below the one behind it
C_SWEEP = np.tan(np.radians(50))        # chevron arms swept back 50 deg off the centre line


def massing_C(p2):
    """TERRACES: the roof goes up to the cap (117.2) and the nose faces are raked
    back to 32 degrees about their own base edges; the terraces are cut in variant_C."""
    p2 = [p for p in p2 if p[2] != "top"]
    p2 = tilt(p2, lambda n: n[0] > 0.45, 32.0)
    return p2 + [((0, Y_MAX, 0), (0, 1, 0), "top")]


def ahead(tip, d=0.0):
    """Plan region ahead of (forward of) a chevron line with its tip at x = tip, moved forward by d."""
    t = tip + d
    return P([(t, 0), (t - 90 * C_SWEEP, 90), (300, 90), (300, -90), (t - 90 * C_SWEEP, -90)])


def plane_frame(point, normal):
    """Frame on a roof plane: u along +x (projected), v = n x u."""
    return K.face_frame(point, normal, (1, 0, 0))


def on_plane(frame, pts_xz):
    """Plan points (x, z) -> (u, v) of the points straight above/below them on the frame's plane."""
    o, u, v, n = frame
    out = []
    for x, z in pts_xz:
        y = o[1] - (n[0] * (x - o[0]) + n[2] * (z - o[2])) / n[1]
        p = np.array([x, y, z]) - o
        out.append((float(p @ u), float(p @ v)))
    return out


def roof_region(h, inset=0.0):
    """Plan outline of tier 2's up-facing roof faces (n_y > 0.9), unioned."""
    from shapely.ops import unary_union
    polys = []
    for f in h.o2.faces():
        if f.normal_at().Y > 0.9:
            polys.append(sg.MultiPoint([(v.X, v.Z) for v in f.outer_wire().vertices()]).convex_hull)   # faces are convex
    r = unary_union(polys).buffer(0.01, join_style=2).buffer(-0.01, join_style=2)
    return B.soff(r, -inset) if inset else r


Y_T1_MIN = B.Y_SKIRT + 0.2       # no tier-1 backing reaches below this (the ring wall tops out at 85.5)


def tier1_today(h, relief=False):
    """Tier 1 as today's hood_details: the chamfered pockets each side and at the back,
    the blue brow on the front.  relief (2026-10-09, his image 3 -- "the widest periphery
    is white space with no features; aggressive dark grey and blue"): the pockets get dark
    floors, blue hatch combs between them, the brow is pressed, the front facet and its
    two corner facets get dark raked vents and graphite pressed plates."""
    ym1 = (B.Y_SKIRT + B.Y_LEDGE) / 2
    for zs, u0s in ((1, (-112, -94, -78, 10, 30)), (-1, (-108, -86, -60, -30, 14, 32))):
        f = h.t1((0, 0.5, zs), (-50, ym1, zs * 74), (1, 0, 0))
        sg_ = np.sign(f[1][0])
        rects = [K.chamfer_rect(sg_ * u0 - 6.2, sg_ * u0 + 6.2, -2.6, 2.6, 1.6) for u0 in u0s]
        h.cut(B.union([K.carve(f, r, 1.6) for r in rects]))
        if relief:
            h.dark.append(B.union([K.carve(sunk(f, 1.6), r, FLOOR_T, lift=0.3) for r in rects]))
            us = sorted(u0s)
            for a, b in zip(us, us[1:]):
                if b - a < 16 or (zs < 0 and a <= -128 <= b):
                    continue
                m = (a + b) / 2
                for k in (-1, 0, 1):
                    h.press(f, K.slash(sg_ * (m + 2.0 * k), 0.0, 0.9, 4.6, 1.6 * sg_), PRESS_D, "blue",
                            floor_t=FLOOR_T, y_min=Y_T1_MIN)
    f = h.t1((-1, 0.5, 0), (-166, ym1, 0), (0, 0, 1))
    for r in (K.chamfer_rect(-42, -30, -2.6, 2.6, 1.4), K.chamfer_rect(-10, 14, -3.2, 3.2, 1.6),
              K.chamfer_rect(26, 36, -2.2, 2.2, 1.2)):
        h.cut(K.carve(f, r, 1.5))
        if relief:
            h.dark.append(K.carve(sunk(f, 1.5), r, FLOOR_T, lift=0.3))
    f = h.t1((1, 0.5, 0), (67, ym1, 0), (0, 0, -1))
    brow = K.polyline_band([(-50, -1.0), (-18, -1.0), (-14, 1.2), (50, 1.2)], 1.4)
    if not relief:
        h.blue.append(K.carve(f, brow, 1.0))
        return
    h.press(f, brow, 0.8, "blue", floor_t=FLOOR_T, y_min=Y_T1_MIN)
    # front facet: dark raked vents over the brow's left run, a graphite plate under its right run
    for uc in (-44.0, -40.6, -37.2):
        h.press(f, K.slash(uc, 2.6, 1.4, 3.4, 1.6), 0.8, "dark", floor_t=FLOOR_T, y_min=Y_T1_MIN)
    plate = [(14, -1.4), (40, -1.4), (43, -4.4), (17, -4.4)]
    h.press(f, plate, PRESS_D, "gfx", floor_t=FLOOR_T, y_min=Y_T1_MIN)
    for uc in (24.0, 27.0, 30.0):
        h.press(sunk(f, PRESS_D), K.slash(uc, -2.9, 1.0, 2.2, 1.2), 0.5, "dark", floor_t=FLOOR_T, y_min=Y_T1_MIN)
    # the front corner facets: a graphite plate with a dark vent row, raked back
    for zs in (1, -1):
        fc = h.t1((0.8, 0.5, 0.45 * zs), (63, ym1, 56 * zs), (0, 0, -zs))
        h.press(fc, [(-7.0, -3.0), (5.0, -3.0), (7.0, 2.6), (-5.0, 2.6)], PRESS_D, "gfx", floor_t=FLOOR_T, y_min=Y_T1_MIN)
        for uc in (-3.0, 0.0, 3.0):
            h.press(sunk(fc, PRESS_D), K.slash(uc, -0.2, 1.0, 3.6, 1.6), 0.5, "dark", floor_t=FLOOR_T, y_min=Y_T1_MIN)


SKIRT_KEEP = 5.0     # mm round a hood screw (countersink r 3.2 + 1.8) nothing is pressed
SKIRT_D = 0.6        # the skirt has the ring wall 0.3 behind it: no backing, so shallow


def skirt_features(h):
    """The skirt (y 78..86, a 2.5 wall with the ring wall 0.3 behind it -- nothing can be
    backed, so everything is 0.6-1.0 deep), his image 3: on every facet between the hood
    screws, a graphite armour plate pressed 0.6 with raked ends, a row of dark raked slots
    sunk 0.4 further into its middle, and a blue hatch comb at its forward end."""
    pts_ = B.pts(B.H0)
    cen = np.array(B.H0.centroid.coords[0])
    screws = [B.skirt_point(x, z, n) for x, z, n in B.HOOD_SCREWS]
    for i in range(len(pts_)):
        a, b = np.array(pts_[i]), np.array(pts_[(i + 1) % len(pts_)])
        L = float(np.linalg.norm(b - a))
        if L < 16.0:
            continue
        t = (b - a) / L
        nrm = np.array([t[1], -t[0]])
        mid = (a + b) / 2
        if nrm @ (mid - cen) < 0:
            nrm = -nrm
        n3 = np.array([nrm[0], 0.0, nrm[1]])
        o = np.array([mid[0], B.Y_HOOD_SCREW, mid[1]])
        f = K.face_frame(o, n3, np.cross([0, 1.0, 0], n3))
        if f[2][1] < 0:                                    # v up
            f = (f[0], -f[1], -f[2], f[3])
        uax = f[1]
        su = [float((s - o) @ uax) for s in screws if abs((s - o) @ n3) < 0.5 and abs((s - o) @ uax) < L / 2]
        cuts = sorted([(-L / 2 - 1.0, -L / 2 + 3.0), (L / 2 - 3.0, L / 2 + 1.0)] +
                      [(s_ - SKIRT_KEEP, s_ + SKIRT_KEEP) for s_ in su])
        spans, at = [], -L / 2
        for c0, c1 in cuts:
            if c0 > at:
                spans.append((at, c0))
            at = max(at, c1)
        fwd = 1.0 if uax[0] > 0.3 else (-1.0 if uax[0] < -0.3 else 0.0)   # +u towards the nose?
        for s0, s1 in spans:
            if s1 - s0 < 14.0:
                continue
            rake = 5.2 * (-fwd if fwd else 1.0)            # tops lean back (front/back facets: one way)
            comb_at = s1 if fwd >= 0 else s0               # the comb at the forward end
            if s1 - s0 >= 24.0:
                for k in range(3):
                    uc = comb_at - (1.4 + 2.2 * k) * (1 if fwd >= 0 else -1)
                    pol = K.slash(uc, 0.0, 1.0, 5.2, rake * 0.6)
                    h.cut(K.carve(f, pol, SKIRT_D))
                    h.blue.append(K.carve(sunk(f, SKIRT_D), pol, FLOOR_T, lift=0.3))
                if fwd >= 0:
                    s1 = s1 - 8.0
                else:
                    s0 = s0 + 8.0
            p0, p1 = s0 + 1.0 + abs(rake) / 2, s1 - 1.0 - abs(rake) / 2
            plate = K.slash((p0 + p1) / 2, 0.0, p1 - p0, 5.2, rake)
            h.cut(K.carve(f, plate, SKIRT_D))
            h.gfx.append(K.carve(sunk(f, SKIRT_D), plate, FLOOR_T, lift=0.3))
            Lp = p1 - p0
            if Lp >= 16.0:
                nsl = max(2, int(Lp * 0.45 / 3.0))
                for k in range(nsl):
                    uc = (p0 + p1) / 2 + (k - (nsl - 1) / 2) * 3.0
                    sl = K.slash(uc, 0.0, 1.3, 3.6, rake * 0.7)
                    h.cut(K.carve(sunk(f, SKIRT_D), sl, 0.4))
                    h.dark.append(K.carve(sunk(f, SKIRT_D + 0.4), sl, 0.6, lift=0.3))


def face_poly(h, frame):
    """The tier-2 face a frame sits on, as a polygon in the frame's (u, v)."""
    o, u, v, n = frame
    for f in h.o2.faces():
        fn = f.normal_at()
        c = f.center()
        if np.dot([fn.X, fn.Y, fn.Z], n) > 0.9999 and abs((np.array([c.X, c.Y, c.Z]) - o) @ n) < 0.01:
            q = [np.array([p.X, p.Y, p.Z]) - o for p in f.outer_wire().vertices()]
            return sg.MultiPoint([(float(p @ u), float(p @ v)) for p in q]).convex_hull
    raise SystemExit("face_poly: no tier-2 face under that frame")


def clip_uv(poly_uv, F, inset):
    """A (u, v) polygon clipped to face F shrunk by inset -> list of point lists."""
    g = sg.Polygon(poly_uv).intersection(B.soff(F, -inset))
    return [B.pts(x) for x in getattr(g, "geoms", [g]) if x.area > 0.5]


VENT_D, VENT_V0 = 3.0, 3.6       # vents: through the 2.5 wall + 0.5; start 3.6 up the slope, so the
                                 # slot's inner corner stays above y 100.5 (the strip channel's wall)


PRESS_D = 0.6      # relief (2026-10-09, his brief: "a shape of a different colour -> a recess"):
FLOOR_T = 0.8      # a colour shape sinks PRESS_D on a floor FLOOR_T thick, backed so the wall stays WALL


def sunk(frame, d):
    """The same face frame moved d into the part (the floor of a press there)."""
    o, u, v, n = frame
    return (o - d * n, u, v, n)


def side_slopes(h, vents=(-74.0, 9), ticks=(-38.0, -34.0, -30.0, -26.0), top_margin=1.5, avoid=None, y_top=None,
                relief=False):
    """Shared: every side facet of tier 2 graphite edge to edge, raked vents through
    (positions in ROBOT x, sized to each facet's real height, skipped inside `avoid`,
    a plan region), blue hatch ticks.  relief: the graphite panel (1 mm in from the facet
    edges) is pressed, leaving a white frame round it; the ticks sink further into it."""
    xs = [vents[0] + 7.0 * k for k in range(vents[1])]
    for pt, nn, tag in h.p2:
        n = K.unit(nn)
        if tag != "t2" or abs(n[2]) < 0.6:
            continue
        f = K.pick(h.p2, "t2", n, np.asarray(pt, float) + np.array([0, 6.0, 0]), (1, 0, 0))
        try:
            F = face_poly(h, f)
        except SystemExit:
            continue                                  # a plane that never became a face
        o, u, v, _ = f
        u0, v0, u1, v1 = F.bounds
        for pts_ in clip_uv(B.pts(F), F, 1.0):
            if relief:
                h.press(f, pts_, PRESS_D, "gfx", floor_t=FLOOR_T, y_min=Y_T2_MIN)
            else:
                h.gfx.append(K.carve(f, pts_, 1.2))
        vt = v1 - top_margin if y_top is None else min(v1 - top_margin, (y_top - o[1]) / v[1])
        L = vt - v0 - VENT_V0
        vc = v0 + VENT_V0 + L / 2
        Fv = B.soff(F, -1.2).intersection(sg.box(u0 - 1, v0 - 1, u1 + 1, vt))   # vents stay in here
        sg_ = np.sign(u[0])
        for x in xs:
            uc = (x - o[0]) / u[0]
            p = o + uc * u + vc * v
            if avoid is not None and avoid.buffer(3.0).contains(sg.Point(p[0], p[2])):
                continue
            for pts_ in clip_uv(K.slash(uc, vc, 3.0, L, -0.67 * L * sg_), Fv, 0.0):
                h.cut(K.carve(f, pts_, VENT_D) & K.slab_y(Y_T2_MIN, BIG))
        if n[2] > 0:
            for x in ticks:
                for pts_ in clip_uv(K.slash((x - o[0]) / u[0], vc, 1.2, L, -0.3 * L * sg_), F, 0.8):
                    if relief:
                        h.press(sunk(f, PRESS_D), pts_, 0.5, "blue", floor_t=FLOOR_T, y_min=Y_T2_MIN)
                    else:
                        h.blue.append(K.carve(f, pts_, 1.0))


def nose_fangs(h, half=26.0, n=8, brow=5.0, fang=6.5, y_top=None):
    """Shared: a graphite brow along the top of the nose face whose lower edge is a
    row of fangs pointing down at the screen, blue ticks at each end."""
    f = h.t2((1, 0.93, 0), (59, (B.Y_CHAN + Y_TOP) / 2, 0), (0, 0, -1))
    F = face_poly(h, f)
    v0, v1 = F.bounds[1], F.bounds[3]
    if y_top is not None:                           # the face is cut lower than its plane's own top
        v1 = min(v1, (y_top - f[0][1]) / f[2][1])
        F = F.intersection(sg.box(-BIG, -BIG, BIG, v1))
    top = v1 + 2.0                                   # clipped to the face, 1 mm in
    root = v1 - 1.0 - brow
    tip = max(root - fang, v0 + 1.5)
    w = 2 * half / n
    poly = [(half, top), (-half, top), (-half, root)]
    for i in range(n):
        poly += [(-half + (i + 0.5) * w, tip), (-half + (i + 1) * w, root)]
    for pts_ in clip_uv(poly, F, 1.0):
        h.gfx.append(K.carve(f, pts_, 1.2))
    for sg_ in (1, -1):
        for k in range(3):
            for pts_ in clip_uv(K.slash(sg_ * (half + 3.0 + 2.6 * k), (root + v1) / 2, 1.1, v1 - root, 0.0), F, 0.8):
                h.blue.append(K.carve(f, pts_, 1.0))


def nose_visor(h, half=27.0, band=11.0, slot=19.0, relief=False):
    """Hood B's nose (his call, 2026-10-08: "remove the sawtooth, put some other thing"):
    a graphite brow band across the top of the nose face, chamfered ends, straight lower
    edge; a long dark hexagonal visor pressed into it; a blue hatch comb at each end.
    Nothing raised on a 47-deg face, so no new overhang.  relief: the brow is pressed too,
    and the visor and the combs sink further into it."""
    f = h.t2((1, 0.93, 0), (59, (B.Y_CHAN + Y_TOP) / 2, 0), (0, 0, -1))
    F = face_poly(h, f)
    v1 = F.bounds[3]
    top, bot, c = v1 + 2.0, v1 - band, 3.0                   # top clipped to the face, 1 mm in
    for pts_ in clip_uv([(-half, top), (half, top), (half, bot + c), (half - c, bot), (-half + c, bot),
                         (-half, bot + c)], F, 1.0):
        if relief:
            h.press(f, pts_, PRESS_D, "gfx", floor_t=FLOOR_T, y_min=Y_T2_MIN)
        else:
            h.gfx.append(K.carve(f, pts_, 1.2))
    fi = sunk(f, PRESS_D) if relief else f
    vc, hh = (v1 - 1.0 + bot) / 2, 1.9
    h.press(fi, [(-slot, vc), (-slot + 3, vc + hh), (slot - 3, vc + hh), (slot, vc), (slot - 3, vc - hh),
                 (-slot + 3, vc - hh)], 1.0 if relief else 1.4, "dark", y_min=Y_T2_MIN)
    for sg_ in (1, -1):
        for u in (slot + 2.5, slot + 4.5, slot + 6.5):
            for pts_ in clip_uv(K.slash(sg_ * u, vc, 1.0, 2 * hh + 1.6, 0.0), F, 1.0):
                if relief:
                    h.press(fi, pts_, 0.5, "blue", floor_t=FLOOR_T, y_min=Y_T2_MIN)
                else:
                    h.blue.append(K.carve(f, pts_, 1.0))


def tail_exhaust(h, relief=False):
    """Shared: the two back facets of tier 2 -- a graphite band with dark pressed
    exhaust slots.  relief: the band is pressed too, the slots sink further into it."""
    ym = (B.Y_CHAN + Y_TOP) / 2
    for d, t in (((-1, 0.93, 0.0), (-158, ym, 14)), ((-0.95, 0.93, -0.3), (-156, ym, -15))):
        f = h.t2(d, t, (0, 0, 1))
        F = face_poly(h, f)
        v0, v1 = F.bounds[1], F.bounds[3]
        for pts_ in clip_uv([(-12, v0), (12, v0), (12, v1 - 2), (9, v1 + 1), (-9, v1 + 1), (-12, v1 - 2)], F, 1.0):
            if relief:
                h.press(f, pts_, PRESS_D, "gfx", floor_t=FLOOR_T, y_min=Y_T2_MIN)
            else:
                h.gfx.append(K.carve(f, pts_, 1.2))
        a, b = v0 + 4.0, v1 - 2.2                  # low end high enough that the backing clears y 100
        for uc in (-7.5, -2.5, 2.5, 7.5):
            h.press(sunk(f, PRESS_D) if relief else f, K.chamfer_rect(uc - 1.6, uc + 1.6, a, b, 0.8),
                    1.0 if relief else 1.4, "dark", y_min=Y_T2_MIN)


def variant_A():
    """SPINE -- a gable roof: a white spine along the ridge carrying a blue bus,
    graphite roof flanks, white armour slats raked back off the spine in pairs
    (forward-pointing chevrons seen from above), serrated roof edges."""
    h = Hood(massing_A)
    R = roof_region(h)
    ridge = sg.box(-BIG, -9.0, BIG, 9.0)
    for s in (1, -1):
        half = R.intersection(sg.box(-BIG, 0, BIG, BIG) if s > 0 else sg.box(-BIG, -BIG, BIG, 0))
        h.inlay_plan(half.difference(ridge), "gfx")
    fr = {s: plane_frame((0, Y_MAX, 0), (0, 1, s * K_A)) for s in (1, -1)}
    # serrations: triangular bites in both roof edges, between the slats
    for s in (1, -1):
        for xb in (36, 18, 0, -18, -36, -54, -72, -90, -108, -126):
            h.press_plan(P([(xb - 6, s * 70), (xb + 6, s * 70), (xb + 1, s * 43.5)]), 2.4, "gfx")
    # slats: white, 1.8 proud of the roof, raked back 45 deg from the spine
    for s in (1, -1):
        for xk in (40, 22, 4, -14, -32, -50, -68, -86, -104, -122):
            reg = para(xk, s * 12.5, s * 41.0, 6.5, 26.0).intersection(B.soff(R, -1.0))
            if reg.area > 20:
                h.raise_(fr[s], on_plane(fr[s], B.pts(reg)), 1.8, "white", taper=12)
    # the spine: blue bus with doglegs + combs, dark vertebra windows
    top = {s: fr[s] for s in (1, -1)}
    bus = [(38, 0), (8, 0), (2, 4), (-46, 4), (-52, 0), (-100, 0), (-106, -4), (-134, -4)]
    for s in (1, -1):
        band = sg.Polygon(K.polyline_band(bus, 2.0)).intersection(
            sg.box(-BIG, 0, BIG, BIG) if s > 0 else sg.box(-BIG, -BIG, BIG, 0))
        for g in getattr(band, "geoms", [band]):
            if g.area > 0.5:
                h.blue.append(K.carve(top[s], on_plane(top[s], B.pts(g)), 1.0))
    for x0 in (-14, -78):
        for i in range(4):
            x = x0 - 3.0 * i
            seg = sg.box(x - 0.55, 0.3, x + 0.55, 7.5)
            h.blue.append(K.carve(top[1], on_plane(top[1], B.pts(seg)), 1.0))
    for x in (-24, -64, -118):
        for s in (1, -1):
            w = sg.Polygon([(x + 3, s * 1.0), (x - 3, s * 1.0), (x - 6, s * 6.5), (x, s * 6.5)])
            h.press(top[s], on_plane(top[s], B.pts(w)), 1.0, "dark")
    side_slopes(h, vents=(-80.0, 8), top_margin=4.0)
    nose_fangs(h)
    tail_exhaust(h)
    tier1_today(h)
    return h


def variant_C():
    """TERRACES -- GLACIER's "few large stepped plates": the roof steps down to the
    nose in four chevron terraces pointing forward (1.8 mm risers), a graphite band
    with dark vent slots at the foot of each riser, a blue lip line along each edge,
    fangs, dark intakes on the nose corners, exhaust at the back."""
    h = Hood(massing_C)
    R = roof_region(h)
    ys = [Y_MAX - C_STEP * (i + 1) for i in range(len(C_TIPS))][::-1]   # tread heights, front to back
    for tip, y in sorted(zip(C_TIPS, ys), key=lambda q: -q[1]):        # highest (rearmost) first
        reg = ahead(tip)
        h.add(B.prism_xz(B.soff(reg, WALL), y - WALL, y) & h.outer)    # the new roof under the tread
        h.cut(B.prism_xz(reg, y, BIG))
    # each tread: graphite band at the foot of the riser behind it, dark slots in it,
    # a blue lip line on the tread above, 2.5 back from the riser
    R_in = B.soff(R, -1.0)
    for i, (tip, y) in enumerate(zip(C_TIPS, ys)):
        band = ahead(tip).difference(ahead(tip, 8.0)).intersection(R_in)
        h.gfx.append(prism_any(band, y - 1.2, y + 0.01))
        y_up = y + C_STEP
        lip = ahead(tip, -2.5).difference(ahead(tip, -1.3)).intersection(B.soff(R, -4.0))
        if not lip.is_empty:
            h.blue.append(prism_any(lip, y_up - 1.0, y_up + 0.01))
        for s in (1, -1):
            for zc in (14.0, 30.0):
                xc = tip + 4.0 - zc * C_SWEEP
                c = (xc, s * zc)
                if not R_in.contains(sg.Point(c).buffer(5)):
                    continue
                d = np.array([-C_SWEEP, s * 1.0]) / np.hypot(C_SWEEP, 1.0)     # along the arm
                q = np.array([d[1], -d[0]])
                pts_ = [tuple(np.array(c) + a * d * 4.5 + b * q * 1.4) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                h.press(deck(y), xz(pts_), 1.0, "dark")
    # each tread's centre: a blue seam with a hatch comb, graphite rivets either side
    edges = [50.0] + list(C_TIPS)
    for i, (tip, y) in enumerate(zip(C_TIPS, ys)):
        x0, x1 = tip + 11.0, edges[i] - (6.0 if i else 14.0)
        if x1 - x0 < 12:
            continue
        fr = deck(y)
        h.blue.append(K.carve(fr, K.polyline_band(xz([(x0, 0), (x1, 0)]), 1.4), 1.0))
        xm = (x0 + x1) / 2
        h.blue.append(B.union([K.carve(fr, K.polyline_band(xz([(xm + 2.6 * j, -3.2), (xm + 2.6 * j, 3.2)]), 1.0), 1.0)
                               for j in (-1, 0, 1)]))
        h.gfx.append(B.union([K.carve(fr, xz(K.chamfer_rect(x0 + 2 - 1.3, x0 + 2 + 1.3, s * 9 - 1.3, s * 9 + 1.3, 0.5)), 0.8)
                              for s in (1, -1)]))
    # the rear (top) terrace: a graphite chevron plate stack behind the last riser
    rear = P(full([(-92, 0), (-110, 22), (-132, 22), (-140, 8), (-140, 0)]))
    h.press_plan(rear.intersection(R_in), 1.4, "gfx")
    h.raise_(deck(Y_MAX - 1.4), xz(full([(-100, 0), (-112, 13), (-128, 13), (-133, 5), (-133, 0)])), 1.4, "white", taper=30)
    # intakes: dark pressed scoops on the two front corner facets
    ym = (B.Y_CHAN + 110.0) / 2
    for d, tg in (((0.7, 0.9, 0.5), (48, ym, 50)), ((0.7, 0.9, -0.5), (38, ym, -55))):
        f = h.t2(d, tg, (0, 0, 1))
        F = face_poly(h, f)
        for uc in (-5.0, 0.0, 5.0):
            for pts_ in clip_uv(K.slash(uc, F.bounds[1] + 8.0, 2.6, 7.0, 3.0), F, 1.2):
                h.press(f, pts_, 1.4, "dark", y_min=Y_T2_MIN)
    side_slopes(h, vents=(-74.0, 9), y_top=ys[0] - 1.5)
    nose_fangs(h, half=24.0, n=7, brow=4.0, fang=5.0, y_top=ys[0])
    tail_exhaust(h)
    tier1_today(h)
    return h


def variant_B(relief=True):
    """PLATES -- the links' own vocabulary on a flat deck: a white rim frame with
    45-degree jogs round a big pressed graphite field, an asymmetric armour stack
    (graphite chamfered base routed round the circuit, two white plates split by a
    diagonal gap), raked white gill
    slats each side, circuit traces with hatch combs and pads on the field, dark
    windows, bites out of the deck edge.  relief (2026-10-09, his call): every flush
    colour shape pressed (rivets, the plates' traces and pads, the field circuit, the
    side-slope panels and ticks, the visor brow, the exhaust bands, tier 1's brow), and
    new dark / blue / graphite features on the skirt and tier 1.  relief=False = "B0",
    the hood as it was."""
    h = Hood()
    D = deck_outline(h)
    FD = 1.6
    yF = Y_TOP - FD
    floor = deck(yF)
    # rim frame + field: inset 6, tabs of frame reaching into it at 45 degrees
    inner = B.soff(D, -6.0)
    tabs = []
    for x0, x1, s, d in ((-122, -108, 1, 6.0), (-66, -40, 1, 4.0), (-4, 8, 1, 6.0),
                         (-114, -94, -1, 4.0), (-50, -36, -1, 6.0), (-10, 18, -1, 4.0)):
        tabs.append(tab(x0, x1, zedge(inner, (x0 + x1) / 2, s) - s * d, s))
    field = inner.difference(sg.MultiPolygon(tabs).buffer(0))
    h.press_plan(field, FD, "gfx")
    # bites out of the deck edge (pressed 3, graphite floors)
    bites = [P([(-34, 47), (-16, 47), (-10, 70), (-40, 70)]),
             P([(-92, -47), (-78, -47), (-72, -70), (-98, -70)]),
             P([(-104, 47), (-96, 47), (-90, 70), (-110, 70)]),
             P([(36, 30), (60, 30), (60, 70), (20, 70)]).intersection(P([(14, 40), (60, -6), (80, 80)]).convex_hull),
             P([(36, -30), (60, -30), (60, -70), (20, -70)]).intersection(P([(14, -40), (60, 6), (80, -80)]).convex_hull)]
    for reg in bites:
        h.press_plan(reg, 3.0, "gfx")
    # graphite rivets along the rim, clear of the bites and the field
    keep = sg.MultiPolygon(bites).buffer(2.5).union(B.soff(field, 2.0))
    rim = sg.LinearRing(B.pts(B.soff(D, -3.0)))
    riv = []
    for k in range(int(rim.length // 13)):
        q = rim.interpolate(k * 13.0)
        sq = P(K.chamfer_rect(q.x - 1.2, q.x + 1.2, q.y - 1.2, q.y + 1.2, 0.4))
        if not sq.intersects(keep) and B.soff(D, -0.8).contains(sq):
            if relief:
                h.press(deck(), xz(B.pts(sq)), 0.5, "gfx", floor_t=FLOOR_T)
            else:
                riv.append(K.carve(deck(), xz(B.pts(sq)), 0.8))
    if riv:
        h.gfx.append(B.union(riv))
    # armour stack, deliberately NOT an arrow (his note, 2026-10-07: "too regular, like
    # an arrow -- more irregular or asymmetric"): the graphite base is routed round the
    # field's circuit, each side on its own schedule -- tip off the centre line, two jogs
    # on +z vs one notch on -z, a lopsided tail; on it two white plates with their own
    # outlines, split by a diagonal gap that shows the graphite between them
    rise = 115.0 - yF
    base = [(38, 6), (30, 18), (14, 24), (4, 24), (-2, 19), (-30, 19), (-36, 26), (-62, 26), (-66, 22),
            (-84, 22), (-92, 28), (-112, 28), (-122, 20), (-130, 20), (-134, 12), (-120, 4), (-116, -6),
            (-118, -14), (-104, -22), (-74, -22), (-68, -16), (-46, -16), (-40, -24), (-12, -24), (-4, -18),
            (18, -18), (30, -10)]
    h.raise_(floor, xz(base), rise, "gfx", taper=40)
    safe = B.soff(P(base), -(rise * np.tan(np.radians(40)) + 1.0))      # the base's top face, 1 mm in
    plates = []
    for outline in ([(34, 6), (24, 15), (8, 19), (-6, 14), (-22, 14), (-30, 21), (-44, 21), (-34, -9),
                     (-20, -9), (-14, -19), (4, -19), (20, -6)],                                  # front
                    [(-49, 22), (-58, 22), (-64, 15), (-96, 15), (-104, 22), (-124, 22), (-126, 12),
                     (-112, -2), (-100, -10), (-84, -17), (-60, -17), (-54, -10.5), (-40, -10.5)]):   # rear
        g = P(outline).intersection(safe)
        g = g.buffer(-0.8, join_style=2).buffer(0.8, join_style=2)       # opening: no slivers < 1.6 wide
        plates += [q for q in getattr(g, "geoms", [g]) if q.area > 30]
    for q in plates:
        h.raise_(deck(115.0), xz(B.pts(q)), Y_MAX - 115.0, "white", taper=25)
    a_top = deck(Y_MAX)
    top_ok = sg.MultiPolygon([B.soff(q, -1.5) for q in plates]).buffer(0)   # details stay on the plates

    def on_top(poly_xz, colour, depth=1.0):
        g = P(poly_xz).intersection(top_ok)
        for q in getattr(g, "geoms", [g]):
            if q.area > 0.5:
                lst = h.gfx_over if colour == "gfx" else getattr(h, colour)
                if relief:          # the white plate is solid 2.2 here: no backing needed
                    h.cut(K.carve(a_top, xz(B.pts(q)), PRESS_D))
                    lst.append(K.carve(sunk(a_top, PRESS_D), xz(B.pts(q)), FLOOR_T, lift=0.3))
                else:
                    lst.append(K.carve(a_top, xz(B.pts(q)), depth))

    def dark_well(poly_xz):                                             # 1.4 into the white, dark floor
        g = P(poly_xz).intersection(top_ok)
        for q in getattr(g, "geoms", [g]):
            if q.area > 2.0:
                h.cut(K.carve(a_top, xz(B.pts(q)), 1.4))
                h.dark.append(K.carve(deck(Y_MAX - 1.4), xz(B.pts(q)), 0.6, lift=0.2))

    # front plate: a blue trace with a dogleg, graphite pads, two dark raked slits
    on_top(K.polyline_band([(26, 5), (10, 5), (4, 10), (-18, 10)], 1.6), "blue")
    for x, z in ((26, 5), (-18, 10), (-24, -4)):
        on_top(K.chamfer_rect(x - 2.4, x + 2.4, z - 2.4, z + 2.4, 0.9), "gfx")
    for x in (-3.0, 4.0):
        dark_well(K.slash(x, -6.5, 2.4, 7.0, 4.0))                      # (x, z), leaning back
    # rear plate: the blue bus off-centre with a hatch comb, two dark windows
    on_top(K.polyline_band([(-56, 15), (-66, 9), (-92, 9), (-98, 5), (-112, 5)], 1.6), "blue")
    for x in (-74.0, -77.5, -81.0):
        on_top(K.polyline_band([(x, 5.5), (x, 12.5)], 1.1), "blue")
    for x, z in ((-56, 15), (-112, 5)):
        on_top(K.chamfer_rect(x - 2.4, x + 2.4, z - 2.4, z + 2.4, 0.9), "gfx")
    for x0, x1 in ((-94, -84), (-70, -62)):
        dark_well(K.chamfer_rect(x0, x1, -9.0, -5.0, 1.2))
    # gills: raked white slats on the field, each side
    for s in (1, -1):
        for xk in (-26, -44, -62, -80):
            h.raise_(floor, xz(B.pts(para(xk, s * 31.0, s * 42.5, 8.0, 11.0))), 2.8, "white", taper=12)
    # circuit on the field: traces with doglegs, hatch combs, pads
    tr = [[(-104, 33), (-118, 33), (-124, 26), (-134, 26)],
          [(14, -33), (-2, -33), (-8, -29.5), (-18, -29.5)]]
    for t in tr:
        if relief:
            h.press(floor, K.polyline_band(xz(t), 1.6), PRESS_D, "blue", floor_t=FLOOR_T)
        else:
            h.blue.append(K.carve(floor, K.polyline_band(xz(t), 1.6), 1.0, lift=0.3))
    for x0, z0, s in ((-112, 33, 1), (10, -33, -1)):
        combs = [K.polyline_band(xz([(x0 - 3.2 * i, z0 - 3.5), (x0 - 3.2 * i, z0 + 3.5)]), 1.1) for i in range(4)]
        if relief:
            for cb in combs:
                h.press(floor, cb, PRESS_D, "blue", floor_t=FLOOR_T)
        else:
            h.blue.append(B.union([K.carve(floor, cb, 1.0, lift=0.3) for cb in combs]))
    for x, z in ((-134, 26), (-18, -29.5), (14, -33)):
        h.raise_(floor, xz(K.chamfer_rect(x - 2.4, x + 2.4, z - 2.4, z + 2.4, 0.9)), 0.8, "white")
    # dark windows in the front corners of the field
    for x, z in ((27, 25.5), (24, -23.0)):
        h.press(floor, K.chamfer_rect(x - 4, x + 4, -z - 2.5, -z + 2.5, 1.2), 1.5, "dark")
    # corrugated half-pipes on the field where the two traces were (his marks, 2026-10-07):
    # 24_pipes' own hose, pts in 24's deck frame (u = x, v = -z) so they paste into PIPES
    # on a face at the field floor
    for pd in B_PIPES:
        h.pipes.append((pd, yF))
    side_slopes(h, avoid=sg.MultiPolygon(bites).buffer(0), relief=relief)
    nose_visor(h, relief=relief)
    tail_exhaust(h, relief=relief)
    tier1_today(h, relief=relief)
    if relief:
        skirt_features(h)
    return h


B_PIPES = [   # face: B's field floor, y = Y_TOP - 1.6
    dict(name="hook", bend="spline", ends=("collar", "dive"),         # rear -z: round the armour's tail
         pts=[(-102, 35), (-113, 34), (-121, 30.5), (-125.5, 24), (-127, 16), (-126, 9)]),
    dict(name="sweep", bend="spline", ends=("dive", "collar"),        # front +z: a shallow S
         pts=[(-18, -30.5), (-6, -31), (6, -32), (18, -34), (28, -35.5)]),
]


VARIANTS = {"A": variant_A, "B": variant_B, "B0": lambda: variant_B(relief=False), "C": variant_C}


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    which = [a for a in sys.argv[2:] if a in VARIANTS or a == "now"] or ["now"] + list(VARIANTS)
    now_bodies, now_solid = current()
    ref_low = now_solid & K.slab_y(-BIG, Y_MECH)
    results = []
    for name in which:
        t0 = time.time()
        if name == "now":
            h, (bodies, solid) = Hood(), (now_bodies, now_solid)
        else:
            h = VARIANTS[name]()
            bodies, solid = h.bodies()
        report(name, bodies, solid, t0)
        checks(name, h, bodies, solid, ref_low)
        doc = (VARIANTS[name].__doc__ or "").strip().split(chr(10))[0] if name != "now" else "today's hood"
        results.append(("today" if name == "now" else name, bodies, doc))
        if "--sheets" in sys.argv:
            sheet(name, bodies, os.path.join(out, f"hood_{name}_{time.strftime('%H%M%S')}.png"))
    if "--board" in sys.argv:
        board(results, os.path.join(out, f"board_{time.strftime('%H%M%S')}.png"))
