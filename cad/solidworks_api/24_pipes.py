"""Step 24: corrugated half-pipes -- blue accent hoses on the printed parts.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/24_pipes.py <command> [Part ...] [--trench]

His brief (2026-10-06, reference: a sci-fi wall panel with ribbed hoses):
a few corrugated pipes per part, 5 mm wide, 20-50 mm long, printed in the blue
accent; a TRUE HALF cylinder (centre line on the face, 2.5 mm proud) so it
reads as buried in the part, not a tube tangent to it; routes may curve, to
break the straight edges of everything else; ends either a clamp collar or a
dive back into the part, chosen per pipe.  His picks after the previews:
graphite collars; TRENCH pipes (sunk in a slot, crest 0.8 proud) where a link
sweeps ~2 mm over the part; skip the parts with no room.  The design is PIPES.

export   READ-ONLY on the parts: a multi-body STEP copy of each part as it is in
         SolidWorks now + a JSON of its bodies' colours / volumes and its placement
         (out/pipes/src/) -- the design runs offline on the CURRENT geometry.
sweep    every live component's placement at 86 hip angles + a mesh per part file
         (out/pipes/sweep/; in memory, hip put back, helper mate deleted; ~30 min).
map      per candidate face: flat ground, edge / seat margins, swept keep-out.
routes   candidate routes per face (--trench: at trench height, slot-wide corridor).
preview  builds every pipe in PIPES, checks it, writes out/pipes/<Part>_pipes.step
         (+ _trench.step, the cutter) and the board <Part>_pipes.png.  Offline.
robot    the whole robot with the pipes, offline (out/pipes/robot_pipes.png).
build    the previewed pipes into the v5 parts (PP_* Imported bodies, PT_* trench
         cuts, end of the tree), checked body by body, saved with 16 --chain.
print    Bambu 3MFs with the pipes (out/print/), filaments as the originals.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "pipes")
SRC = os.path.join(OUT, "src")

PARTS = {"Femur": r"Links\Femur.SLDPRT",
         "Coupler": r"Links\Coupler.SLDPRT",
         "Tibia": r"Links\Tibia.SLDPRT",
         "Side panel": r"Body\Side panel.SLDPRT",
         "RobotMount": r"Body\OldRobotBodyMount\RobotMount.SLDPRT",
         "FacetFront": r"Box\Facet\FacetFront.SLDPRT",
         "FacetBack": r"Box\Facet\FacetBack.SLDPRT",
         "FacetHood": r"Box\Facet\FacetHood.SLDPRT",
         "FacetRing": r"Box\Facet\FacetRing.SLDPRT",
         "TailStrut": r"Box\TailStrut\TailStrut.SLDPRT"}


# ============================================================ export (SolidWorks)
def _rgb(b, doc):
    """A body's colour: its own, else its first coloured face's, else the part's.
    Parts imported from STEP can carry colour on faces only."""
    from swlib import wrap, sld
    m = b.MaterialPropertyValues2
    if m is not None:
        return [round(float(x), 4) for x in m[:3]], "body"
    for f in b.GetFaces() or []:
        m = wrap(f, sld.IFace2).MaterialPropertyValues
        if m is not None:
            return [round(float(x), 4) for x in m[:3]], "face"
    m = doc.MaterialPropertyValues
    return ([round(float(x), 4) for x in m[:3]], "part") if m is not None else (None, "none")


def export(which):
    import swlib
    import swstyle as S
    from swlib import c, wrap, sld
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    comps = swlib.components(robot)
    os.makedirs(SRC, exist_ok=True)
    docs = {os.path.normcase(wrap(d, sld.IModelDoc2).GetPathName()): wrap(d, sld.IModelDoc2)
            for d in sw.GetDocuments() or []}
    for part in which:
        path = os.path.join(swlib.V5, PARTS[part])
        doc = docs.get(os.path.normcase(path))
        if doc is None:
            print(f"{part}: NOT LOADED ({path})")
            continue
        bodies = []
        for b in S.bodies(doc):
            rgb, src = _rgb(b, doc)
            box = [round(v * 1000.0, 4) for v in b.GetBodyBox()]
            bodies.append({"name": b.Name, "volume": round(S.volume(b), 4), "rgb": rgb,
                           "colour_from": src, "box": box,
                           "group": S.group_of(b) if b.MaterialPropertyValues2 is not None else None})
        step = os.path.join(SRC, f"{part}.step")
        ok, err, warn = doc.Extension.SaveAs3(step, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy,
                                              None, None, 0, 0)
        inst = []
        for n, cp in comps.items():
            if os.path.normcase(cp.GetPathName() or "") == os.path.normcase(path) and not cp.IsSuppressed():
                inst.append({"name": n, "placement": np.round(swlib.placement(cp), 6).tolist()})
        info = {"part": part, "file": PARTS[part], "dirty": bool(doc.GetSaveFlag()),
                "bodies": bodies, "instances": inst}
        with open(os.path.join(SRC, f"{part}.json"), "w") as fh:
            json.dump(info, fh, indent=1)
        vol = sum(x["volume"] for x in bodies)
        print(f"{part:11s} STEP {'ok' if ok else 'FAILED'}  {len(bodies)} bodies {vol:10.1f} mm3  "
              f"colours from {sorted({x['colour_from'] for x in bodies})}  "
              f"{len(inst)} instance(s)  {'DIRTY in memory' if info['dirty'] else 'saved'}")


SWEEP = os.path.join(OUT, "sweep")


def sweep(step=1.0):
    """Every live component's placement at every hip angle, plus one mesh per part
    file in its own frame -> out/pipes/sweep/.  In memory only: the hip goes back
    where it was and the AI_HipDrive helper is deleted; nothing is saved."""
    import time
    import importlib
    import swlib
    from swlib import c, wrap, sld
    body_mesh = importlib.import_module("11_check_and_export").body_mesh
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    comps = {n: cp for n, cp in swlib.components(robot).items()
             if not cp.IsSuppressed() and (cp.GetPathName() or "").lower().endswith(".sldprt")
             and n.split("/")[0] not in swlib.NOT_FOR_COLLISION}
    os.makedirs(SWEEP, exist_ok=True)
    t0 = time.time()
    files, nv = {}, 0
    for n, cp in comps.items():
        p = os.path.normcase(cp.GetPathName())
        key = f"{os.path.basename(p)[:-7]}__{cp.ReferencedConfiguration}"
        cached = os.path.join(SWEEP, key + ".npz")
        if key not in files and os.path.exists(cached) and os.path.getmtime(cached) > os.path.getmtime(p):
            files[key] = p                     # meshed already, and the part file is older
        if key not in files:
            Vs, Ts, k = [], [], 0
            for b in cp.GetBodies2(c.swSolidBody) or []:
                V, T = body_mesh(wrap(b, sld.IBody2))
                if V is not None:
                    Vs.append(V)
                    Ts.append(T + k)
                    k += len(V)
            if not Vs:
                continue
            V, T = np.vstack(Vs), np.vstack(Ts)
            np.savez_compressed(os.path.join(SWEEP, key + ".npz"), V=V, T=T)
            files[key] = p
            nv += len(V)
        comps[n] = (cp, key)
    comps = {n: v for n, v in comps.items() if isinstance(v, tuple) and v[1] in files}
    print(f"  meshed {len(files)} part files ({nv} vertices) for {len(comps)} components, "
          f"{time.time() - t0:.0f} s", flush=True)
    hip = swlib.HipDriver(robot)
    start = hip.hip()
    hips = list(np.round(np.arange(-28.0, 57.0 + 1e-9, step), 3))
    pl = {n: [] for n in comps}
    got = []
    t0 = time.time()
    try:
        for h in hips:
            got.append(round(hip.set(h), 4))
            for n, (cp, key) in comps.items():
                pl[n].append(np.round(swlib.placement(cp), 5).tolist())
    finally:
        hip.set(start)
        swlib.HipDriver.remove(robot)
    worst = max(abs(a - b) for a, b in zip(got, hips))
    with open(os.path.join(SWEEP, "poses.json"), "w") as fh:
        json.dump({"hips": got, "mesh": {n: key for n, (cp, key) in comps.items()},
                   "placements": pl}, fh)
    print(f"  {len(hips)} poses {hips[0]}..{hips[-1]} in {time.time() - t0:.0f} s, worst hip error "
          f"{worst:.4f} deg; hip back at {hip.hip():.3f} (was {start:.3f}), helper mate deleted")


# ============================================================ the design numbers
R = 2.5            # crest radius: 5 mm wide.  The centre line lies ON the face: a true half
CORE = 1.95        # tube radius between the ribs
RIB = R - CORE     # 0.55: each rib is a torus on the core, crest = R exactly
PITCH = 1.7        # rib spacing along the centre line
RC, LC, CH = 3.1, 3.0, 0.4               # clamp collar: radius, length, edge chamfer
DIVE_R, DIVE_A, DIVE_L = 9.0, 40.0, 7.0  # dive end: bend radius (>= MIN_BEND), angle into the part, run-out
MIN_BEND = 8.0     # centre-line bend radius: the ribs on the inside of a bend stay apart
EDGE = 1.0         # the footprint stays this far inside the flat ground
SEAT = 3.5 + 0.6   # a fastener opening's head/washer seat + the additions clearance (08's rule)
SEAT_MAX = 120.0   # mm2: an opening smaller than this is a fastener hole and gets the seat
CLR = 1.0          # clearance to anything that sweeps past: sideways, and above the collar crest
CLIP = 0.0         # pipes are clipped AT the face: the proud half only (the FacetBack relief
                   # is a skin thinner than 2 mm -- a deeper clip printed blue out of its back)
# TRENCH pipes (his pick for Coupler and RobotMount, where a link passes 1.9-2 mm over all the
# open ground): the hose lies in a slot cut into the part, crest only T_CREST proud
T_CREST = 0.8      # crest above the face: >= 1.1 mm to the femur plate (1.9) and the tibia (2.0)
T_AXIS = R - T_CREST   # 1.7: the centre line under the face; the slot is cut down to it
T_FLOOR = 2.5      # the hose is clipped flat this deep (the RobotMount field is a 5 mm plate)
T_GAP = 0.2        # slot half-width R + T_GAP: rib crests and the end clamps clear the walls
T_SOLID = 3.5      # the part must be solid this deep under the whole slot footprint
PX = 0.1           # mm per pixel, keep-out rasters
MAP = 4.0          # px per mm, the map drawings
BLUE = (0.13, 0.45, 0.95)        # swstyle.GROUPS blue -- the accent filament
GRAPHITE = (0.25, 0.27, 0.30)    # swstyle.GROUPS graphite

# Candidate faces per part: (part-local outward normal, face at n.p = d), approximate --
# snapped to the real planar face on load.  PRINT_UP: the print's up direction, part-local
# (None = not known; then the face is assumed to print facing up, like the raised styling).
FACES = {   # links: "out" = the show face; "floorN" = a pocket floor N mm below it (a hose in a trench)
    "Femur": {"out": ((0, 0, -1), 5.0), "floor3": ((0, 0, -1), 1.5), "floor7": ((0, 0, -1), -1.86)},
    "Coupler": {"out": ((0, 0, -1), 5.0), "floor3": ((0, 0, -1), 1.89), "floor6": ((0, 0, -1), -1.5)},
    "Tibia": {"out": ((0, 0, 1), 13.5), "floor7": ((0, 0, 1), 6.07), "floor8": ((0, 0, 1), 5.0)},
    "Side panel": {"out": ((0, 0, 1), 10.0)},
    "RobotMount": {"out": ((0, 0, 1), 5.0)},
    "FacetFront": {"well": ((1, 0, 0), 53.0), "top": ((1, 0, 0), 65.0),
                   "chin": ((0.82, -0.57, 0), 36.76),
                   "cheek_r": ((0.95, 0, 0.32), 76.21), "cheek_l": ((0.95, 0, -0.32), 76.21)},
    "FacetBack": {"bay": ((-1, 0, 0), 153.0), "top": ((-1, 0, 0), 164.0),
                  "chin": ((-0.8, -0.6, 0), 112.6),
                  "cheek_r": ((-0.95, 0, 0.32), 170.76), "cheek_l": ((-0.95, 0, -0.32), 170.76)},
    "FacetHood": {"deck": ((0, 1, 0), 113.0), "tier1": ((0, 1, 0), 94.0),
                  "field": ((0, 1, 0), 111.4),        # hood B: the pressed graphite field (deck - 1.6)
                  "slope_r": ((0.01, 0.73, 0.68), 116.09), "slope_l": ((-0.02, 0.73, -0.68), 118.14),
                  "nose": ((0.73, 0.68, 0), 111.11)},
    "FacetRing": {"side_r": ((0, 0, 1), 75.2), "front": ((1, 0, 0), 64.2), "back": ((-1, 0, 0), 163.2)},
    "TailStrut": {"flank_r": ((0, 0, 1), 16.0), "under": ((0.58, -0.82, 0), -65.88),
                  "pad": ((0, -1, 0), -5.0)},
}
PRINT_UP = {"FacetFront": (1, 0, 0), "FacetBack": (-1, 0, 0), "FacetHood": (0, 1, 0),
            "FacetRing": (0, 1, 0), "TailStrut": (0, -1, 0)}

# The pipes: per part, a list of
#   face  a key of FACES[part]
#   pts   centre-line waypoints (u, v) in mm on that face: u = right, v = up, seen from OUTSIDE
#         (the map draws exactly these axes)
#   bend  fillet radius at every inner waypoint (>= MIN_BEND), or "spline": one smooth
#         curve through the waypoints
#   ends  (start, end), each "collar" or "dive"
PIPES = {   # first set: the proposer's candidates (`routes`), ends chosen per pipe
    "Femur": [
        dict(name="strip", face="out", bend="spline", ends=('dive', 'collar'),
             pts=[(36.68, 50.81), (33.55, 47.10), (30.79, 43.25), (28.16, 39.33), (25.57, 35.40), (22.97, 31.48), (20.38, 27.56), (18.68, 24.26)]),
        dict(name="field", face="out", bend="spline", ends=('collar', 'dive'),
             pts=[(9.68, 8.81), (6.15, 5.71), (3.25, 2.04), (0.61, -1.85), (-2.14, -5.71), (-4.82, -9.60), (-6.32, -11.98)]),
    ],
    "Side panel": [
        dict(name="arc", face="out", bend="spline", ends=('collar', 'collar'),
             pts=[(8.33, 61.83), (12.56, 63.69), (16.74, 65.66), (21.35, 66.61), (26.18, 66.62), (30.75, 65.61), (34.79, 63.30), (38.50, 60.17), (39.91, 58.75)]),
    ],
    "FacetBack": [
        dict(name="cheek", face="cheek_l", bend="spline", ends=('dive', 'collar'),
             pts=[(1.79, 70.15), (1.15, 65.41), (0.93, 60.51), (0.81, 55.56), (0.79, 50.56), (0.79, 45.56), (0.78, 40.57), (0.38, 35.74)]),
    ],
    "FacetHood": [   # hood B (2026-10-07): wave + corner sat where B's gills / rear trace are now;
        # his marks put two hoses on the field instead, in place of two traces -- the same
        # routes as hood_variants.B_PIPES (its offline previews), keep the two in step
        dict(name="hook", face="field", bend="spline", ends=("collar", "dive"),
             pts=[(-102, 35), (-113, 34), (-121, 30.5), (-125.5, 24), (-127, 16), (-126, 9)]),
        dict(name="sweep", face="field", bend="spline", ends=("dive", "collar"),
             pts=[(-18, -30.5), (-6, -31), (6, -32), (18, -34), (28, -35.5)]),
        dict(name="tier", face="tier1", bend="spline", ends=('dive', 'collar'),
             pts=[(-150.75, -54.35), (-147.20, -57.86), (-143.53, -61.05), (-139.50, -63.41), (-135.13, -64.92), (-130.61, -66.07), (-126.13, -67.35), (-125.13, -67.35)]),
    ],
    "TailStrut": [
        dict(name="keel", face="under", bend="spline", ends=('dive', 'collar'),
             pts=[(-148.49, 11.65), (-144.87, 8.34), (-141.52, 4.82), (-138.97, 0.89), (-136.93, -3.27), (-134.34, -7.19), (-131.02, -10.82), (-130.24, -11.35)]),
    ],
    "RobotMount": [   # TRENCH (his pick): the inner femur plate passes 1.9 mm over this field
        dict(name="trench", face="out", kind="trench", bend="spline", ends=("clamp", "clamp"),
             pts=[(90.04, 27.24), (94.52, 25.99), (98.90, 24.50), (103.30, 23.04), (107.85, 22.20), (112.35, 22.66), (116.41, 24.80), (119.74, 28.16), (121.97, 32.22), (122.99, 36.80), (123.34, 40.66)]),
    ],
}


# ============================================================ geometry (offline)
def _aes():
    lib = os.path.normpath(os.path.join(HERE, "..", "aesthetics", "lib"))
    if lib not in sys.path:
        sys.path.insert(0, lib)


class Part:
    """A part's bodies (from `export`) and its colours, in part-local mm."""

    def __init__(self, part):
        from build123d import import_step
        self.name = part
        self.info = json.load(open(os.path.join(SRC, f"{part}.json")))
        self.solids = import_step(os.path.join(SRC, f"{part}.step")).solids()
        left = list(self.info["bodies"])
        self.rgb = []
        for s in self.solids:          # colour by volume: STEP body order is not promised
            b = min(left, key=lambda b: abs(b["volume"] - s.volume))
            if abs(b["volume"] - s.volume) > 1e-3 * max(1.0, b["volume"]) + 0.05:
                raise SystemExit(f"{part}: STEP body {s.volume:.2f} mm3 matches nothing in the JSON")
            left.remove(b)
            self.rgb.append(tuple(b["rgb"]) if b["rgb"] else (0.9, 0.9, 0.9))
        P = np.array(self.info["instances"][0]["placement"])
        self.R = P[:, :3]               # part-local -> ROBOT rotation (saved pose)
        self._mesh = None

    def mesh(self):
        """[(V, T, rgb)] per body, part-local."""
        if self._mesh is None:
            _aes()
            from render3d import tessellate
            self._mesh = []
            for s, rgb in zip(self.solids, self.rgb):
                V, T, _ = tessellate(s, 0.05)
                if len(T):
                    self._mesh.append((V, T, rgb))
        return self._mesh

    def snap(self, n, d):
        """The real planar face nearest (n, d): exact normal + offset."""
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Plane
        n = np.asarray(n, float) / np.linalg.norm(n)
        best = None
        for s in self.solids:
            for f in s.faces():
                if BRepAdaptor_Surface(f.wrapped).GetType() != GeomAbs_Plane:
                    continue
                m = f.normal_at()
                m = np.array([m.X, m.Y, m.Z])
                c = f.center()
                dd = float(m @ [c.X, c.Y, c.Z])
                err = np.degrees(np.arccos(np.clip(m @ n, -1, 1))) + abs(dd - d)
                if best is None or err < best[0]:
                    best = (err, m, dd)
        if best is None or best[0] > 2.0:
            raise SystemExit(f"{self.name}: no planar face near n={n.round(2)} d={d}")
        return best[1], best[2]


class Face:
    """A planar face as a frame: (u, v, w) with w the outward normal, w = 0 on the face,
    u = right and v = up AS SEEN FROM OUTSIDE (so drawings are never mirrored).
    'Right' is ROBOT +X projected on the face, or for a face looking along ROBOT X the
    viewer's right (ROBOT -Z from the front, +Z from the back)."""

    def __init__(self, part, key):
        self.part, self.key = part, key
        n, d = part.snap(*FACES[part.name][key])
        nR = part.R @ n
        right = np.array([1.0, 0, 0]) if abs(nR[0]) < 0.7 else np.array([0, 0, -1.0 if nR[0] > 0 else 1.0])
        r = part.R.T @ right
        u = r - (r @ n) * n
        u /= np.linalg.norm(u)
        self.A = np.array([u, np.cross(n, u), n])     # rows: u, v, w in part-local
        self.d = d
        up = PRINT_UP.get(part.name)
        self.up = None if up is None else self.A @ (np.asarray(up, float) / np.linalg.norm(up))

    def to(self, V):
        """part-local -> face frame"""
        return V @ self.A.T - [0, 0, self.d]

    def back(self, V):
        """face frame -> part-local"""
        return (V + [0, 0, self.d]) @ self.A

    def location(self):
        """build123d Location taking face-frame geometry to part-local"""
        from build123d import Location, Plane
        o = self.A[2] * self.d
        return Location(Plane(origin=tuple(o), x_dir=tuple(self.A[0]), z_dir=tuple(self.A[2])))


class Raster:
    """A (u, v) grid at PX mm/pixel covering a face's part outline + margin."""

    def __init__(self, umin, vmin, umax, vmax, pad=6.0):
        self.u0, self.v0 = umin - pad, vmin - pad
        self.W = int(np.ceil((umax - umin + 2 * pad) / PX))
        self.H = int(np.ceil((vmax - vmin + 2 * pad) / PX))

    def blank(self):
        return np.zeros((self.H, self.W), bool)

    def px(self, uv):
        uv = np.asarray(uv, float)
        return np.stack([(uv[..., 0] - self.u0) / PX, (self.H - 1) - (uv[..., 1] - self.v0) / PX], -1)

    def draw(self, tris_uv):
        """Fill triangles [(3, 2) in mm] into a new mask."""
        from PIL import Image, ImageDraw
        img = Image.new("1", (self.W, self.H), 0)
        dr = ImageDraw.Draw(img)
        for t in self.px(tris_uv):
            dr.polygon([tuple(p) for p in t], fill=1, outline=1)
        return np.array(img, bool)

    def poly(self, geom):
        """Fill a shapely (multi)polygon into a new mask."""
        from PIL import Image, ImageDraw
        img = Image.new("1", (self.W, self.H), 0)
        dr = ImageDraw.Draw(img)
        gs = getattr(geom, "geoms", [geom])
        for g in gs:
            if g.is_empty:
                continue
            dr.polygon([tuple(p) for p in self.px(np.array(g.exterior.coords))], fill=1, outline=1)
            for h in g.interiors:
                dr.polygon([tuple(p) for p in self.px(np.array(h.coords))], fill=0, outline=0)
        return np.array(img, bool)


def _disk(r):
    k = int(np.ceil(r / PX))
    y, x = np.mgrid[-k:k + 1, -k:k + 1]
    return (x * x + y * y) * PX * PX <= r * r


def _tri_normals(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-12
    n[ok] /= ln[ok, None]
    return n, ok


def ground(face):
    """Masks on the face's raster: where the part's top is exactly AT the face (flat),
    and where a pipe may go (allowed: flat, EDGE inside its boundary, fastener openings
    with their seats, pockets and windows kept clear)."""
    from scipy import ndimage
    meshes = [(face.to(V), T) for V, T, _ in face.part.mesh()]
    allV = np.vstack([V for V, _ in meshes])
    ras = Raster(allV[:, 0].min(), allV[:, 1].min(), allV[:, 0].max(), allV[:, 1].max())
    flat_t, above_t, floor_t, blue_t = [], [], [], []
    for (V, T), (_, _, rgb) in zip(meshes, face.part.mesh()):
        n, ok = _tri_normals(V, T)
        w = V[:, 2][T]
        on = ok & (np.abs(w).max(1) < 0.03) & (n[:, 2] > 0.99)
        flat_t.append(V[T[on]][:, :, :2])
        above_t.append(V[T[w.max(1) > 0.05]][:, :, :2])
        floor_t.append(V[T[ok & (w.max(1) < -0.03) & (n[:, 2] > 0.5)]][:, :, :2])
        if rgb[2] > 0.8 and rgb[0] < 0.3:          # an existing blue accent: blue on blue merges
            blue_t.append(V[T[on]][:, :, :2])
    flat = ras.draw(np.vstack(flat_t)) & ~ras.draw(np.vstack(above_t))
    floor = ras.draw(np.vstack(floor_t))
    blue = ras.draw(np.vstack(blue_t)) if blue_t else ras.blank()
    lab, nl = ndimage.label(~flat)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    seats = ras.blank()
    n_open = 0
    for i in range(1, nl + 1):
        if i in border:
            continue
        m = lab == i
        a = m.sum() * PX * PX
        if (floor & m).sum() < 0.95 * m.sum() and a < SEAT_MAX:   # a hole through, not a pocket
            ys, xs = np.nonzero(m)
            long_side = max(np.ptp(ys), np.ptp(xs)) * PX
            if long_side < 1.4 * 2 * np.sqrt(a / np.pi):           # ROUND: a fastener, gets the seat
                seats |= m                                          # (the Coupler's slots do not)
                n_open += 1
    allowed = (ndimage.binary_erosion(flat, _disk(EDGE)) & ~ndimage.binary_dilation(seats, _disk(SEAT))
               & ~ndimage.binary_dilation(blue, _disk(1.5)))
    return ras, flat, allowed, n_open


def swept(face, ras, top=RC + CLR):
    """Everything any OTHER component passes through, at any hip angle of the sweep, in
    the slab just outside the face (w -0.3 .. top), projected, dilated by CLR."""
    from scipy import ndimage
    pj = json.load(open(os.path.join(SWEEP, "poses.json")))
    me = face.part.info["instances"][0]["name"]
    if me not in pj["placements"]:
        raise SystemExit(f"{face.part.name}: {me} not in the sweep (re-run `sweep`)")
    meshes = {}
    tris = []
    PP = np.array(pj["placements"][me])                       # (poses, 3, 4)
    for name, key in pj["mesh"].items():
        if name == me:
            continue
        if key not in meshes:
            z = np.load(os.path.join(SWEEP, key + ".npz"))
            meshes[key] = (z["V"], z["T"])
        V, T = meshes[key]
        Q = np.array(pj["placements"][name])
        lo, hi = V.min(0), V.max(0)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        for k in range(len(PP)):
            RP, tP, RQ, tQ = PP[k][:, :3], PP[k][:, 3], Q[k][:, :3], Q[k][:, 3]
            M = face.A @ RP.T @ RQ                              # Q-local -> face frame
            off = face.A @ (RP.T @ (tQ - tP)) - [0, 0, face.d]
            c = corners @ M.T + off
            if (c[:, 2].max() < -0.3 or c[:, 2].min() > top or c[:, 0].max() < ras.u0 or
                    c[:, 0].min() > ras.u0 + ras.W * PX or c[:, 1].max() < ras.v0 or
                    c[:, 1].min() > ras.v0 + ras.H * PX):
                continue
            Vf = V @ M.T + off
            w = Vf[:, 2][T]
            sel = (w.max(1) >= -0.3) & (w.min(1) <= top)
            if sel.any():
                tris.append(Vf[T[sel]][:, :, :2])
    if not tris:
        return ras.blank()
    return ndimage.binary_dilation(ras.draw(np.vstack(tris)), _disk(CLR))


def _seg_edges(pts, bend):
    """In-plane centre line (w = 0) through pts: every corner filleted with `bend`, or
    with bend = "spline" one smooth curve through them (the auto routes)."""
    from build123d import FilletPolyline, Polyline, Spline
    P = [(float(u), float(v), 0.0) for u, v in pts]
    if bend == "spline":
        Q = [P[0]]                  # a waypoint 1 mm from the last makes the spline hook
        for q in P[1:-1]:
            if np.hypot(q[0] - Q[-1][0], q[1] - Q[-1][1]) >= 2.5:
                Q.append(q)
        if np.hypot(P[-1][0] - Q[-1][0], P[-1][1] - Q[-1][1]) < 2.5 and len(Q) > 1:
            Q.pop()
        Q.append(P[-1])
        return list(Spline(*Q).edges())
    line = FilletPolyline(*P, radius=bend) if len(P) > 2 else Polyline(*P)
    return list(line.edges())


def min_radius(run, n=400):
    """Smallest radius of curvature along the in-plane run (mm), numerically."""
    P = np.array([tuple(run.position_at(x))[:2] for x in np.linspace(0, 1, n)])
    a, b, c = P[:-2], P[1:-1], P[2:]
    ab, bc, ca = (np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                  np.linalg.norm(a - c, axis=1))
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    r = ab * bc * ca / np.maximum(2 * cross, 1e-12)
    return float(r.min())


def dive_reach():
    """How far past its start a dive still shows: the cross-section's top line leaves the
    face where -h + R cos(phi) = 0 on the arc, h = Rd (1 - cos phi) -> cos phi = Rd / (Rd + R),
    at x = (Rd - R) sin(phi).  (Past the arc, on the run-out, if DIVE_A is shallower.)"""
    phi = np.arccos(DIVE_R / (DIVE_R + R))
    if phi <= np.radians(DIVE_A):
        return (DIVE_R - R) * np.sin(phi)
    a = np.radians(DIVE_A)                     # on the straight run-out
    h0, x0 = DIVE_R * (1 - np.cos(a)), DIVE_R * np.sin(a)
    d = (R * np.cos(a) - h0) / np.sin(a)
    return x0 + d * np.cos(a) - R * np.sin(a)


def centreline(p):
    """The whole centre line: in-plane run + the dives at either end (3D, into the part).
    The waypoints give the VISIBLE pipe: at a dive end the run is shortened by the dive's
    reach, so the hose disappears into the face right at the end waypoint."""
    from build123d import Edge, PositionMode, Spline, Vector, Wire
    run = Wire(_seg_edges(p["pts"], p["bend"]))
    d0, d1 = (dive_reach() if e == "dive" else 0.0 for e in p["ends"])
    if d0 or d1:
        L = run.length
        S = np.linspace(d0, L - d1, max(6, int((L - d0 - d1) / 2.0)))
        run = Wire(Spline(*[run.position_at(x, position_mode=PositionMode.LENGTH) for x in S]).edges())
    edges = list(run.edges())
    a = np.radians(DIVE_A)
    for side, kind in zip((0, 1), p["ends"]):
        if kind != "dive":
            continue
        if side == 1:
            E, t = run.position_at(1.0), run.tangent_at(1.0)
        else:
            E, t = run.position_at(0.0), -run.tangent_at(0.0)
        t = Vector(t.X, t.Y, 0).normalized()
        down = Vector(0, 0, -1)
        A_end = E + t * (DIVE_R * np.sin(a)) + down * (DIVE_R * (1 - np.cos(a)))
        tA = t * np.cos(a) + down * np.sin(a)
        B = A_end + tA * DIVE_L
        arc = Edge.make_tangent_arc(E, t, A_end)
        out = [arc, Edge.make_line(A_end, B)]
        if side == 1:
            edges = edges + out
        else:
            edges = [Edge.make_line(B, A_end), Edge.make_tangent_arc(A_end, -tA, E)] + edges
    return Wire(edges), run


def build_pipe(p):
    """(hose, [collars]) as build123d solids in the FACE frame, before the part is cut away."""
    from build123d import Circle, Compound, Cylinder, Plane, Torus, sweep, chamfer, PositionMode, Transition
    wire, run = centreline(p)
    L = wire.length
    # where one edge of the centre line meets the next: a rib straddling such a seam does not
    # fuse (it came back as a loose torus, and the clip then failed round it) -- none there
    seams = np.cumsum([e.length for e in wire.edges()])[:-1]
    p0, t0 = wire.position_at(0.0), wire.tangent_at(0.0)
    try:
        hose = sweep(Plane(origin=p0, z_dir=t0) * Circle(CORE), path=wire, transition=Transition.ROUND)
    except Exception:                           # OCC's pipe shell can refuse a mixed wire
        parts = []
        for e in wire.edges():
            parts.append(sweep(Plane(origin=e.position_at(0), z_dir=e.tangent_at(0)) * Circle(CORE), path=e))
        hose = parts[0].fuse(*parts[1:]).clean() if len(parts) > 1 else parts[0]
    # where the run sits inside the full wire (a dive adds its arc + run-out before / after it)
    arc = DIVE_R * np.radians(DIVE_A)
    s_run0 = arc + DIVE_L if p["ends"][0] == "dive" else 0.0
    s_run1 = s_run0 + run.length
    # ribs: along the run, up to the collars.  NONE on an arc (a dive's, or a fillet's): there
    # the swept core is itself a torus and OCC does not fuse a rib torus onto it -- on the core
    # it came back loose, 0.05 inside it came back loose or inverted and 20x slower.  So a dive
    # ends ribbed, and the plain core sinks into the face over the dive's ~4 mm.
    s0 = s_run0 + LC + 0.4 if p["ends"][0] == "collar" else s_run0
    s1 = s_run1 - LC - 0.4 if p["ends"][1] == "collar" else s_run1
    ribs = []
    ends = np.cumsum([e.length for e in wire.edges()])
    kinds = [e.geom_type.name for e in wire.edges()]
    for s in np.arange(s0, s1 + 1e-6, PITCH):
        if len(seams) and np.abs(seams - s).min() < RIB + 0.3:
            continue
        if kinds[min(int(np.searchsorted(ends, s)), len(kinds) - 1)] == "CIRCLE":
            continue
        q, t = wire.position_at(s, position_mode=PositionMode.LENGTH), \
            wire.tangent_at(s, position_mode=PositionMode.LENGTH)
        ribs.append(Plane(origin=q, z_dir=t) * Torus(CORE, RIB))
    if ribs:
        hose = hose.fuse(*ribs).clean()
        if len(hose.solids()) > 1:              # boolean debris; more than one left fails a check
            hose = Compound([x for x in hose.solids() if x.volume > 1.0])
    collars = []
    for side, kind in zip((0, 1), p["ends"]):
        if kind != "collar":
            continue
        s = s_run0 + LC / 2 if side == 0 else s_run1 - LC / 2
        q, t = wire.position_at(s, position_mode=PositionMode.LENGTH), \
            wire.tangent_at(s, position_mode=PositionMode.LENGTH)
        cyl = Plane(origin=q, z_dir=t) * Cylinder(RC, LC)
        collars.append(chamfer(cyl.edges(), CH))
    if collars:
        hose = hose.cut(*collars)
    return hose, collars, wire, run


def build_trench(p):
    """A trench pipe in the FACE frame: (hose, [clamps], cutter, run, slot footprint).
    The centre line runs T_AXIS under the face; a slot R + T_GAP wide is cut down to it, and
    below it the part keeps hugging the hose's lower half (no undercuts).  Flush graphite
    clamps (radius R, so they sit in the slot) close both ends; the slot ends flat at their
    outer faces.  cutter = slot + the clipped hose and clamps: what the part loses."""
    from build123d import (Circle, Cylinder, Location, Plane, Torus, Wire, extrude, sweep, chamfer,
                           PositionMode, Pos, Box)
    from shapely.geometry import LineString
    run = Wire(_seg_edges(p["pts"], p["bend"]))
    wire = run.moved(Location((0, 0, -T_AXIS)))
    L = wire.length
    p0, t0 = wire.position_at(0.0), wire.tangent_at(0.0)
    hose = sweep(Plane(origin=p0, z_dir=t0) * Circle(CORE), path=wire)
    seams = np.cumsum([e.length for e in wire.edges()])[:-1]
    kinds = [e.geom_type.name for e in wire.edges()]
    ribs = []
    for s in np.arange(LC + 0.4, L - LC - 0.4 + 1e-6, PITCH):
        if (len(seams) and np.abs(seams - s).min() < RIB + 0.3) or \
                kinds[min(int(np.searchsorted(np.cumsum([e.length for e in wire.edges()]), s)), len(kinds) - 1)] == "CIRCLE":
            continue
        q, t = wire.position_at(s, position_mode=PositionMode.LENGTH), wire.tangent_at(s, position_mode=PositionMode.LENGTH)
        ribs.append(Plane(origin=q, z_dir=t) * Torus(CORE, RIB))
    if ribs:
        hose = hose.fuse(*ribs).clean()
    clamps = []
    for s in (LC / 2, L - LC / 2):
        q, t = wire.position_at(s, position_mode=PositionMode.LENGTH), wire.tangent_at(s, position_mode=PositionMode.LENGTH)
        clamps.append(chamfer((Plane(origin=q, z_dir=t) * Cylinder(R, LC)).edges(), CH))
    hose = hose.cut(*clamps)
    keep = Pos(0, 0, 50.0 - T_FLOOR) * Box(4000, 4000, 100)          # w >= -T_FLOOR
    def _solid(x):                                   # drop boolean slivers (< 1 mm3)
        from build123d import Compound
        k = [y for y in x.solids() if y.volume > 1.0]
        return k[0] if len(k) == 1 else Compound(k)
    hose = _solid(hose & keep)
    clamps = [_solid(c & keep) for c in clamps]
    S = np.linspace(0.0, run.length, max(20, int(run.length / 0.5)))
    line = LineString([tuple(run.position_at(x, position_mode=PositionMode.LENGTH))[:2] for x in S])
    foot = line.buffer(R + T_GAP, cap_style=2, join_style=1)
    _aes()
    from shputil import prism
    slot = prism(foot, -T_AXIS, 3.0)
    cutter = slot.fuse(hose, *clamps).clean()
    return hose, clamps, cutter, run, foot


def _cut_part(shape, face):
    """shape (face frame) minus every part body it overlaps.  First clipped at
    w = -CLIP (= the face): what is below it would be hidden in the material anyway,
    and on a thin wall it would come out of the back.  The cut then takes away any
    raised feature the pipe runs into."""
    from build123d import Box, Pos
    out = shape & (Pos(0, 0, 50.0 - CLIP) * Box(4000, 4000, 100))
    bb = out.bounding_box()
    for s in face.part.solids:
        g = s.moved(face.location().inverse())
        b = g.bounding_box()
        if (b.max.X < bb.min.X or b.min.X > bb.max.X or b.max.Y < bb.min.Y or b.min.Y > bb.max.Y or
                b.max.Z < bb.min.Z or b.min.Z > bb.max.Z):
            continue
        out = out.cut(g)
    keep = [x for x in out.solids() if x.volume > 1.0]   # boolean slivers on shared faces
    from build123d import Compound
    return keep[0] if len(keep) == 1 else Compound(keep)


def overhang(face, run):
    """Worst downward-facing slope on the proud half, along the in-plane run, as the
    downward component of the surface normal (-1 = a ceiling; > -0.707 is printable).
    For a half cylinder with axis t on a face whose normal makes A = up.w and side
    b = w x t with B = up.b: the normals are cos(a) w + sin(a) b, a in [-90, 90]."""
    if face.up is None:
        return None
    U, worst = face.up, 1.0
    for x in np.linspace(0.0, 1.0, 60):
        t = run.tangent_at(x)
        t = np.array([t.X, t.Y, 0.0])
        t /= np.linalg.norm(t)
        A, B = U[2], U @ np.cross([0, 0, 1.0], t)
        m = -np.hypot(A, B) if A < 0 else -abs(B)
        worst = min(worst, m)
    return worst


# ============================================================ route proposals
GRID = 0.5         # mm, route search grid


def _smooth(P, win=9.0, step=1.0, passes=3):
    """Resample a polyline every `step` mm and moving-average it (ends pinned)."""
    from shapely.geometry import LineString
    ls = LineString(P)
    Q = np.array([tuple(ls.interpolate(d).coords[0]) for d in np.arange(0, ls.length + 1e-9, step)])
    h = max(1, int(round(win / step / 2)))
    for _ in range(passes):
        if len(Q) <= 2 * h + 1:
            break
        R_ = Q.copy()
        for i in range(h, len(Q) - h):
            R_[i] = Q[i - h:i + h + 1].mean(0)
        Q = R_
    return Q


def _turning(Q):
    """Total heading change (rad), measured every ~5 mm so grid jitter does not count."""
    Q = Q[::5] if len(Q) > 10 else Q
    d = np.diff(Q, axis=0)
    a = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return np.abs(np.diff(a)).sum() if len(a) > 1 else 0.0


def propose(face, maps, lmax=45.0, lmin=20.0, half=RC + 0.05):
    """Candidate routes on a face.  In every region where a collar's centre line may run
    (the free ground eroded by RC), the longest geodesic path (farthest point twice,
    Dijkstra), weighted to keep to the middle of the region, smoothed.  A path longer
    than lmax gives its most curving lmax window whose bends stay >= MIN_BEND.
    Returns [{"length", "turn_deg", "pts"}], longest first."""
    from scipy import ndimage, sparse
    from scipy.sparse.csgraph import dijkstra
    ras, flat, allowed, sw, free, _ = maps
    C = ndimage.binary_erosion(free, _disk(half))
    k = int(round(GRID / PX))
    H2, W2 = C.shape[0] // k, C.shape[1] // k
    Cc = C[:H2 * k, :W2 * k].reshape(H2, k, W2, k).all(axis=(1, 3))
    D = ndimage.distance_transform_edt(Cc) * GRID
    lab, nl = ndimage.label(Cc, structure=np.ones((3, 3)))
    out = []
    for i in range(1, nl + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) * GRID * GRID < 2.0:
            continue
        idx = -np.ones(Cc.shape, int)
        idx[ys, xs] = np.arange(len(ys))
        rows, cols, wts = [], [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            y2, x2 = ys + dy, xs + dx
            ok = (y2 >= 0) & (y2 < H2) & (x2 >= 0) & (x2 < W2)
            ok[ok] &= idx[y2[ok], x2[ok]] >= 0
            a, b = np.arange(len(ys))[ok], idx[y2[ok], x2[ok]]
            dm = 0.5 * (D[ys[ok], xs[ok]] + D[y2[ok], x2[ok]])
            rows.append(a), cols.append(b), wts.append(GRID * np.hypot(dy, dx) * (1 + 1.5 / (dm + 0.25)))
        G = sparse.coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(len(ys), len(ys))).tocsr()
        d0 = dijkstra(G, directed=False, indices=0)
        A = int(np.argmax(np.where(np.isinf(d0), -1, d0)))
        dA, pred = dijkstra(G, directed=False, indices=A, return_predecessors=True)
        B = int(np.argmax(np.where(np.isinf(dA), -1, dA)))
        path = [B]
        while path[-1] != A and pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        u = ras.u0 + (xs[path] * k + k / 2) * PX
        v = ras.v0 + ((ras.H - 1) - (ys[path] * k + k / 2)) * PX
        P = np.stack([u, v], 1)
        if len(P) < 3 or np.linalg.norm(np.diff(P, axis=0), axis=1).sum() < lmin:
            continue
        Q = _smooth(P)
        seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        L = s[-1]
        if L < lmin:
            continue
        if L <= lmax + 1:
            picks = [(0, len(Q) - 1)]
        else:                # up to 3 windows, most curving first, 10 mm apart along the path
            wins = []
            for a in range(0, len(Q), 2):
                b = int(np.searchsorted(s, s[a] + lmax))
                if b >= len(Q):
                    break
                if _radius_pts(Q[a:b + 1]) >= MIN_BEND:
                    wins.append((_turning(Q[a:b + 1]), a, b))
            picks = []
            for t, a, b in sorted(wins, reverse=True):
                if all(s[a] > s[b2] + 10 or s[b] < s[a2] - 10 for a2, b2 in picks):
                    picks.append((a, b))
                if len(picks) == 3:
                    break
        for a, b in picks:
            W = Q[a:b + 1]
            keep = np.unique(np.concatenate([np.arange(0, len(W), 5), [len(W) - 1]]))
            out.append({"length": round(float(np.linalg.norm(np.diff(W, axis=0), axis=1).sum()), 1),
                        "turn_deg": round(float(np.degrees(_turning(W))), 0),
                        "pts": [(round(float(x), 2), round(float(y), 2)) for x, y in W[keep]]})
    return sorted(out, key=lambda c: -c["length"])


def _radius_pts(P):
    if len(P) < 3:
        return 1e9
    a, b, c = P[:-2], P[1:-1], P[2:]
    ab, bc, ca = (np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1))
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    return float((ab * bc * ca / np.maximum(2 * cross, 1e-12)).min())


def cmd_routes(which, trench=False):
    """Candidate routes for every face of every part -> out/pipes/routes/<Part>__<face>.png/.json
    (trench=True: the sweep band only up to a trench crest, the corridor as wide as the slot,
    files <Part>__<face>__trench.*)"""
    from PIL import ImageDraw, ImageFont
    os.makedirs(os.path.join(OUT, "routes"), exist_ok=True)
    try:
        f = ImageFont.truetype("arialbd.ttf", 16)
    except OSError:
        f = ImageFont.load_default()
    for part in which:
        P = Part(part)
        for key in FACES[part]:
            face = Face(P, key)
            maps = face_maps(face, T_CREST + CLR) if trench else face_maps(face)
            cands = propose(face, maps, half=R + T_GAP + 0.05) if trench else propose(face, maps)
            img = draw_map(face, maps)
            dr = ImageDraw.Draw(img)
            ras = maps[0]
            for j, cnd in enumerate(cands):
                q = ras.px(np.array(cnd["pts"])) * MAP * PX + [0, 52]
                dr.line([tuple(x) for x in q], fill=(40, 120, 255), width=int(2 * R * MAP))
                dr.line([tuple(x) for x in q], fill=(255, 255, 255), width=1)
                dr.text(tuple(q[len(q) // 2] + [6, -8]), f"{j}", font=f, fill=(255, 255, 0))
            base = os.path.join(OUT, "routes", f"{part}__{key}" + ("__trench" if trench else ""))
            img.save(base + ".png")
            with open(base + ".json", "w") as fh:
                json.dump(cands, fh, indent=1)
            print(f"{part:11s} {key:8s} {len(cands)} candidates: " +
                  ", ".join(f"#{j} {c['length']:.0f} mm / {c['turn_deg']:.0f} deg" for j, c in enumerate(cands)),
                  flush=True)


# ============================================================ drawings
def _base_map(face, ras):
    """The face seen from outside at MAP px/mm: every upward-facing triangle, painted
    lowest first, in its body's colour, shaded by height."""
    from PIL import Image, ImageDraw
    s = MAP * PX
    W, H = int(ras.W * s), int(ras.H * s)
    img = Image.new("RGB", (W, H), (23, 25, 28))
    dr = ImageDraw.Draw(img)
    items = []
    for V, T, rgb in face.part.mesh():
        V = face.to(V)
        n, ok = _tri_normals(V, T)
        sel = ok & (n[:, 2] > 0.05)
        for t, nz in zip(T[sel], n[sel, 2]):
            items.append((V[t, 2].mean(), V[t], rgb, nz))
    items.sort(key=lambda x: x[0])
    for wm, tri, rgb, nz in items:
        k = (0.55 + 0.45 * nz) * float(np.clip(1.0 + wm / 25.0, 0.45, 1.15))
        col = tuple(int(np.clip(255 * c * k, 0, 255)) for c in rgb)
        dr.polygon([tuple(p * s) for p in ras.px(tri[:, :2])], fill=col)
    return img


def _tint(img, mask, rgb, alpha):
    from PIL import Image
    m = Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size, Image.NEAREST)
    over = Image.new("RGB", img.size, rgb)
    return Image.composite(Image.blend(img, over, alpha), img, m)


def _grid(img, ras, title, sub):
    from PIL import ImageDraw, ImageFont
    dr = ImageDraw.Draw(img)
    try:
        f, fb = ImageFont.truetype("arial.ttf", 13), ImageFont.truetype("arialbd.ttf", 17)
    except OSError:
        f = fb = ImageFont.load_default()
    s = MAP * PX
    W, H = img.size
    for u in range(int(np.ceil(ras.u0 / 10)) * 10, int(ras.u0 + ras.W * PX), 10):
        x = (u - ras.u0) / PX * s
        dr.line([(x, 0), (x, H)], fill=(80, 90, 100) if u % 50 else (140, 150, 160), width=1)
        if u % 20 == 0:
            dr.text((x + 2, H - 16), str(u), font=f, fill=(200, 205, 210))
    for v in range(int(np.ceil(ras.v0 / 10)) * 10, int(ras.v0 + ras.H * PX), 10):
        y = ((ras.H - 1) - (v - ras.v0) / PX) * s
        dr.line([(0, y), (W, y)], fill=(80, 90, 100) if v % 50 else (140, 150, 160), width=1)
        if v % 20 == 0:
            dr.text((2, y + 1), str(v), font=f, fill=(200, 205, 210))
    from PIL import Image
    out = Image.new("RGB", (max(W, 760), H + 52), (23, 25, 28))
    out.paste(img, (0, 52))
    d2 = ImageDraw.Draw(out)
    d2.text((6, 4), title, font=fb, fill=(240, 242, 245))
    d2.text((6, 27), sub, font=f, fill=(150, 158, 168))
    return out


def face_maps(face, top=RC + CLR):
    """`top`: how high over the face a neighbour still blocks -- RC + CLR for a proud pipe,
    T_CREST + CLR for a trench pipe."""
    ras, flat, allowed, n_open = ground(face)
    sw = swept(face, ras, top) if os.path.exists(os.path.join(SWEEP, "poses.json")) else None
    free = allowed & ~sw if sw is not None else allowed
    return ras, flat, allowed, sw, free, n_open


def draw_map(face, maps, pipes_fp=None, bad=None):
    ras, flat, allowed, sw, free, n_open = maps
    img = _base_map(face, ras)
    img = _tint(img, flat & ~allowed, (255, 150, 40), 0.35)        # flat but edge / seat / blue
    if sw is not None:
        img = _tint(img, sw, (230, 40, 40), 0.45)                  # swept by a neighbour
    if pipes_fp is not None:
        img = _tint(img, pipes_fp, (40, 120, 255), 0.75)
    if bad is not None:
        img = _tint(img, bad, (255, 0, 255), 0.9)
    nR = face.part.R @ face.A[2]
    up = "print up: unknown (assumed face up)" if face.up is None else \
        f"print up = {np.round(face.up, 2)} in (u, v, w)"
    sub = (f"u right, v up, seen from outside; normal ROBOT {np.round(nR, 2)}; free {free.sum() * PX * PX:.0f} mm2 "
           f"(orange: edge/seat, red: swept by neighbours{'' if sw is not None else ' NOT COMPUTED'}); "
           f"{n_open} fastener openings; {up}")
    return _grid(img, ras, f"{face.part.name} / {face.key}", sub)


def cmd_map(which):
    os.makedirs(os.path.join(OUT, "maps"), exist_ok=True)
    for part in which:
        P = Part(part)
        for key in FACES[part]:
            face = Face(P, key)
            maps = face_maps(face)
            out = os.path.join(OUT, "maps", f"{part}__{key}.png")
            draw_map(face, maps).save(out)
            print(f"{part:11s} {key:8s} free {maps[4].sum() * PX * PX:7.0f} mm2 -> {out}", flush=True)


# ============================================================ preview: build + check + render
def _nsolids(s):
    return 0 if s is None else len(s.solids())


def _below(shape, face):
    """Volume of `shape` under the face (w < -0.05): a pipe hanging into a pocket or a
    window -- nothing of the cut pipe should be there."""
    from build123d import Box, Pos
    x = shape & (Pos(0, 0, -50.05) * Box(4000, 4000, 100))
    return 0.0 if x is None else sum(s.volume for s in x.solids())


def _tess(shape, dev):
    _aes()
    from render3d import tessellate
    V, T, _ = tessellate(shape, dev)
    return V, T


def build_part(part, render=True):
    """Every pipe of one part: built, cut, checked; STEP (part-local) + board."""
    _aes()
    import stepcolor
    P = Part(part)
    specs = PIPES.get(part, [])
    if not specs:
        print(f"{part}: no pipes designed")
        return None
    faces, maps, rows, bodies, report, cutters = {}, {}, [], [], [], []
    ok_all = True
    for i, p in enumerate(specs):
        trench = p.get("kind") == "trench"
        if p["face"] not in faces:
            faces[p["face"]] = Face(P, p["face"])
            maps[p["face"]] = face_maps(faces[p["face"]], T_CREST + CLR if trench else RC + CLR)
        face = faces[p["face"]]
        ras, flat, allowed, sw, free, _ = maps[p["face"]]
        extra = {}
        if trench:
            from build123d import Compound
            hose_c, coll_c, cutter, run, foot = build_trench(p)
            parts_c = [hose_c] + coll_c
            fp = ras.poly(foot)
            _aes()
            from shputil import prism
            local = [s_.moved(face.location().inverse()) for s_ in P.solids]
            under = prism(foot, -T_SOLID, 0.0)
            for g in local:
                under = under.cut(g)
            void = sum(x.volume for x in under.solids()) if under is not None else 0.0
            crest = max(x.bounding_box().max.Z for x in parts_c)
            removed = []
            for k, g in enumerate(local):
                x = g & cutter
                v = 0.0 if x is None else sum(y.volume for y in x.solids())
                if v > 0.01:
                    removed.append({"body": k, "volume": round(P.solids[k].volume, 4), "removed": round(v, 4)})
            cutters.append(cutter)
            extra = {"kind": "trench", "cutter_mm3": round(cutter.volume, 3), "removed": removed,
                     "void_under_mm3": round(void, 3), "crest_mm": round(crest, 3)}
            below = 0.0
        else:
            hose, collars, wire, run = build_pipe(p)
            hose_c = _cut_part(hose, face)
            coll_c = [_cut_part(c, face) for c in collars]
            parts_c = [hose_c] + coll_c
            fp = ras.blank()
            for s in parts_c:
                V, T = _tess(s, 0.05)
                fp |= ras.draw(V[T][:, :, :2])
            below = sum(_below(s, face) for s in parts_c)
        bad = fp & ~free
        bad_mm2 = bad.sum() * PX * PX
        oh = overhang(face, run)
        rmin = min_radius(run)
        from build123d import Wire
        vis = Wire(_seg_edges(p["pts"], p["bend"])).length      # what shows: waypoint to waypoint
        checks = {
            "one hose body": _nsolids(hose_c) == 1,
            "one body per collar": all(_nsolids(c) == 1 for c in coll_c),
            f"bend radius >= {MIN_BEND}": rmin >= MIN_BEND - 0.05,
            "visible 20..50 mm": 19.5 <= vis <= 50.5,
            "footprint on free ground (< 0.5 mm2 out)": bad_mm2 < 0.5,
            "nothing under the face (< 0.5 mm3)": below < 0.5,
            "printable overhang": oh is None or oh > -0.7072,
        }
        if trench:
            checks["solid under the slot (< 0.5 mm3 void)"] = extra["void_under_mm3"] < 0.5
            checks[f"crest <= {T_CREST}"] = extra["crest_mm"] <= T_CREST + 1e-3
        ideal = (1.0 if trench else 0.5) * np.pi * R * R * vis    # tube at crest radius: an upper bound
        checks["volume sane (< half tube)"] = 0.3 * ideal < hose_c.volume < ideal
        checks = {k: bool(v) for k, v in checks.items()}
        ok = all(checks.values())
        ok_all &= ok
        name = p.get("name", f"pipe{i + 1}")
        info = {"name": name, "face": p["face"], "visible_mm": round(vis, 1), "min_bend": round(rmin, 1),
                "ends": list(p["ends"]), "hose_mm3": round(hose_c.volume, 1),
                "collars_mm3": [round(c.volume, 1) for c in coll_c], "out_of_free_mm2": round(bad_mm2, 2),
                "under_face_mm3": round(below, 3), "overhang": None if oh is None else round(oh, 3),
                "checks": checks, "ok": ok, **extra}
        report.append(info)
        print(f"  {name:10s} {p['face']:8s} visible {vis:5.1f} mm  bend {rmin:5.1f}  hose {hose_c.volume:6.1f} mm3  "
              f"collars {[round(c.volume, 1) for c in coll_c]}  off-free {bad_mm2:5.2f} mm2  "
              f"under {below:5.2f} mm3  overhang {oh if oh is None else round(oh, 2)}  "
              f"{'trench: crest %.2f, void under %.2f, removes %s  ' % (extra['crest_mm'], extra['void_under_mm3'], [r['removed'] for r in extra['removed']]) if trench else ''}"
              f"{'OK' if ok else 'FAIL: ' + ', '.join(k for k, v in checks.items() if not v)}", flush=True)
        L = face.location()
        bodies.append((f"PP_{name}", hose_c.moved(L), tuple(int(255 * c) for c in BLUE)))
        for j, c in enumerate(coll_c):
            bodies.append((f"PP_{name}_collar{j + 1}", c.moved(L), tuple(int(255 * c_) for c_ in GRAPHITE)))
        rows.append((p, face, parts_c, fp, bad))
    step = os.path.join(OUT, f"{part}_pipes.step")
    stepcolor.write(bodies, step, part=f"{part}_pipes")
    if cutters:                     # what the part loses: one tool body per trench, part-local
        stepcolor.write([(f"PT_cut{k + 1}", c_.moved(faces[specs[0]["face"]].location()), (255, 0, 0))
                         for k, c_ in enumerate(cutters)], os.path.join(OUT, f"{part}_trench.step"),
                        part=f"{part}_trench")
    with open(os.path.join(OUT, f"{part}_pipes.json"), "w") as fh:
        json.dump({"part": part, "R": R, "pipes": report}, fh, indent=1)
    if render:
        board(part, P, faces, maps, rows, cutters)
    return ok_all


def board(part, P, faces, maps, rows, cutters=()):
    """One PNG: each face's map with the pipes, the part face-on and oblique, and a
    close-up of every pipe -- all from the same geometry that goes to SolidWorks."""
    _aes()
    import render_color as RCOL
    tiles = []
    for key, face in faces.items():
        fps = [r for r in rows if r[1] is face]
        fp = np.any([r[3] for r in fps], axis=0)
        bad = np.any([r[4] for r in fps], axis=0)
        from PIL import Image
        im = draw_map(face, maps[key], fp, bad)
        im.thumbnail((760, 560))
        tile = Image.new("RGB", (760, 560), RCOL.BG)
        tile.paste(im, ((760 - im.width) // 2, (560 - im.height) // 2))
        tiles.append((f"{key}: map", "blue = pipes, magenta = off free ground", tile))
        base = []
        for s, rgb in zip(P.solids, P.rgb):
            g = s.moved(face.location().inverse())
            for c_ in cutters:             # a trench: show the part with its slot cut
                g = g.cut(c_)
            for x in (g.solids() if g is not None else []):
                V, T = _tess(x, 0.12)
                base.append((V, T, np.array(rgb) * 255))
        pipe_meshes = []
        for p, f, parts_c, _, _ in fps:
            for k, s in enumerate(parts_c):
                V, T = _tess(s, 0.02)
                pipe_meshes.append((V, T, np.array(BLUE if k == 0 else GRAPHITE) * 255))
        allV = np.vstack([V for V, _, _ in base])
        for title, el, az in (("face-on", 89.9, -90), ("oblique", 38, -62)):
            tiles.append((f"{key}: {title}", "", RCOL.view(base + pipe_meshes, allV, (760, 560), el, az)))
        for p, f, parts_c, _, _ in fps:
            PV = np.vstack([_tess(s, 0.1)[0] for s in parts_c])
            lo, hi = PV.min(0) - [8, 8, 0], PV.max(0) + [8, 8, 0]
            near = []
            for V, T, col in base:
                c = V[T].mean(1)
                m = np.all((c[:, :2] >= lo[:2] - 4) & (c[:, :2] <= hi[:2] + 4), axis=1)
                if m.any():
                    near.append((V, T[m], col))
            box = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (-2, 4)])
            tiles.append((f"{p.get('name', '')}: close-up", f"{p['ends'][0]} -> {p['ends'][1]}",
                          RCOL.view(near + pipe_meshes, box, (760, 560), 34, -64)))
    RCOL.sheet(tiles, 3, os.path.join(OUT, f"{part}_pipes.png"), header=f"{part} -- corrugated half-pipes",
               sub=f"R {R} (5 mm wide, centre line on the face), ribs pitch {PITCH} / {RIB}, collars graphite "
                   f"r {RC} x {LC}, dives {DIVE_A:.0f} deg", size=(760, 560))


def cmd_robot(hips=(-28.0, 20.0, 57.0)):
    """The whole robot with every part's pipes, offline: the sweep's meshes at a few hip
    angles; the ten styled parts in their colours, everything else neutral grey."""
    _aes()
    import render_color as RCOL
    from build123d import import_step
    pj = json.load(open(os.path.join(SWEEP, "poses.json")))
    own = {}                                       # instance name -> (part meshes, pipe meshes)
    for part in PARTS:
        info = json.load(open(os.path.join(SRC, f"{part}.json")))
        P = Part(part)
        tr = os.path.join(OUT, f"{part}_trench.step")
        if os.path.exists(tr):                      # the part with its trench cut
            cut = import_step(tr).solids()
            meshes = []
            for s_, rgb in zip(P.solids, P.rgb):
                for c_ in cut:
                    s_ = s_.cut(c_)
                for x in s_.solids():
                    V, T = _tess(x, 0.05)
                    meshes.append((V, T, np.array(rgb) * 255))
        else:
            meshes = [(V, T, np.array(rgb) * 255) for V, T, rgb in P.mesh()]
        pipe = os.path.join(OUT, f"{part}_pipes.step")
        if os.path.exists(pipe):
            for s in import_step(pipe).solids():
                V, T = _tess(s, 0.04)
                col = BLUE if s.volume > 60 else GRAPHITE      # collars are ~45 mm3
                meshes.append((V, T, np.array(col) * 255))
        own[info["instances"][0]["name"]] = meshes
    # render frame: z up (ROBOT X forward, Y up, Z right -> x, z, -y)
    F = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    cache, tiles = {}, []
    for hip in hips:
        k = int(np.argmin(np.abs(np.array(pj["hips"]) - hip)))
        bodies = []
        for name, key in pj["mesh"].items():
            M = np.array(pj["placements"][name][k])
            if name in own:
                for V, T, col in own[name]:
                    bodies.append((((V @ M[:, :3].T) + M[:, 3]) @ F.T, T, col))
                continue
            if key not in cache:
                z = np.load(os.path.join(SWEEP, key + ".npz"))
                cache[key] = (z["V"], z["T"])
            V, T = cache[key]
            bodies.append((((V @ M[:, :3].T) + M[:, 3]) @ F.T, T, np.array([150, 155, 162.0])))
        allV = np.vstack([b[0] for b in bodies])
        for title, el, az in (("front-right", 18, -58), ("back-right", 22, -128)):
            tiles.append((f"hip {pj['hips'][k]:+.0f}: {title}", "",
                          RCOL.view(bodies, allV, (760, 760), el, az)), )
            print(f"  rendered hip {pj['hips'][k]:+.0f} {title}", flush=True)
    RCOL.sheet(tiles, 2, os.path.join(OUT, "robot_pipes.png"), header="Pipes on the robot",
               sub="offline render from the sweep meshes; styled parts in colour, the rest grey; "
                   "left leg suppressed as in ROBOT.SLDASM", size=(760, 760))


# ============================================================ build: into the SolidWorks parts
LINK_PAL = {"blue": (0.13, 0.45, 0.95), "graphite": (0.25, 0.27, 0.30)}          # swstyle.GROUPS
BOX_PAL = {"blue": (0x1E / 255, 0x7B / 255, 1.0),                                # box_concept_facet: BLUE,
           "graphite": (0x7E / 255, 0x87 / 255, 0x95 / 255)}                     # GRAPHITE = filament 2
# (the box's near-black 2B2F36 is DARK, filament 4 -- not the collars' graphite).  Except the
# TailStrut: tail_strut.py draws its graphite (filament 2) bodies in the DARK colour, so there
# DARK is filament 2, and its collars take that colour.
TAIL_GRAPHITE = (0x2B / 255, 0x2F / 255, 0x36 / 255)


def _palette(part):
    """The part's own accent colours, so a pipe groups with that part's filament."""
    rgbs = [tuple(b["rgb"] or ()) for b in json.load(open(os.path.join(SRC, f"{part}.json")))["bodies"]]
    if any(r and abs(r[0] - 0.1294) < 0.01 and abs(r[2] - 0.949) < 0.01 for r in rgbs):
        return LINK_PAL
    return dict(BOX_PAL, graphite=TAIL_GRAPHITE) if part == "TailStrut" else BOX_PAL


def _features(doc):
    from swlib import wrap, sld
    f = wrap(doc.FirstFeature(), sld.IFeature)
    while f is not None:
        yield f
        f = wrap(f.GetNextFeature(), sld.IFeature)


def _delete_pp(doc):
    """Remove an earlier run's PP_* / PT_* features (pipe bodies, trench tools and cuts)."""
    from swlib import c
    doc.ClearSelection2(True)
    k = 0
    for f in list(_features(doc)):
        if f.Name.startswith(("PP_", "PT_")) and f.Select2(k > 0, 0):
            k += 1
    if k:
        doc.Extension.DeleteSelection2(c.swDelete_Absorbed | c.swDelete_Children)
        doc.ClearSelection2(True)
        doc.EditRebuild3()
    return k


def sw_build(which):
    """Each part's pipes (out/pipes/<Part>_pipes.step, part-local) into the v5 part as
    separate Imported bodies at the END of the tree, coloured, named PP_*.  The part's own
    bodies are not touched (checked by volume, body by body), so its mates, openings and
    GL_ styling stay as they were.  Re-running replaces the PP_ features."""
    import swlib
    import swstyle as S
    from swlib import c, wrap, sld
    sw, _ = swlib.connect()
    swlib.open_v5(sw)
    done = []
    for part in which:
        rep = json.load(open(os.path.join(OUT, f"{part}_pipes.json")))
        if not all(x["ok"] for x in rep["pipes"]):
            raise SystemExit(f"{part}: the preview has failing pipes -- fix them first")
        path = os.path.join(swlib.V5, PARTS[part])
        d, err, warn = sw.OpenDoc6(path, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
        doc = wrap(d, sld.IModelDoc2)
        sw.ActivateDoc3(doc.GetTitle(), False, 0, 0)
        n_old = _delete_pp(doc)
        # the part must still be what `export` saw (the pipes were cut against that)
        have = sorted(round(S.volume(b), 1) for b in S.bodies(doc))
        want = sorted(round(b["volume"], 1) for b in json.load(open(os.path.join(SRC, f"{part}.json")))["bodies"])
        if have != want:
            raise SystemExit(f"{part}: bodies differ from the export ({len(have)} vs {len(want)}) -- "
                             f"re-run `export`, `preview` first")
        before = {b.Name: S.volume(b) for b in S.bodies(doc)}
        expect = list(before.values())
        # TRENCH: each affected body minus the trench's cutter (Imported PT_tool, Combine PT_trench)
        trench = [x for x in rep["pipes"] if x.get("kind") == "trench"]
        if trench:
            tstep = os.path.normpath(os.path.join(OUT, f"{part}_trench.step"))
            res = sw.LoadFile4(tstep, "r", None, 0)
            tdoc = wrap(res[0] if isinstance(res, tuple) else res, sld.IModelDoc2)
            tools = S.bodies(tdoc)
            if len(tools) != len(trench):
                raise SystemExit(f"{part}: {len(tools)} cutters in {tstep}, {len(trench)} trenches")
            jobs = [(r, wrap(tools[ti].Copy(), sld.IBody2)) for ti, x in enumerate(trench) for r in x["removed"]]
            sw.CloseDoc(tdoc.GetTitle())
            sw.ActivateDoc3(doc.GetTitle(), False, 0, 0)
            pdoc = wrap(doc, sld.IPartDoc)
            for k, (r, tool) in enumerate(jobs):
                # OCC's volume of the STEP vs SolidWorks': 0.30 mm3 (3 ppm) on the 92706 mm3 plate
                tgt = [min(S.bodies(doc), key=lambda b: abs(S.volume(b) - r["volume"]))]
                v_sw = S.volume(tgt[0])
                if abs(v_sw - r["volume"]) > 1e-4 * r["volume"]:
                    raise SystemExit(f"{part}: no body of {r['volume']} mm3 to cut (nearest {v_sw:.4f})")
                names0 = {b.Name for b in S.bodies(doc)}
                f = wrap(pdoc.CreateFeatureFromBody3(tool, False, c.swCreateFeatureBodyCheck), sld.IFeature)
                if f is None:
                    raise SystemExit(f"{part}: SolidWorks refused the trench cutter")
                f.Name = f"PT_tool{k + 1}"
                newb = [b for b in S.bodies(doc) if b.Name not in names0]
                cf, _ = S.combine(doc, "cut", tgt[0], newb)
                cf.Name = f"PT_trench{k + 1}"
                expect[min(range(len(expect)), key=lambda i: abs(expect[i] - v_sw))] = v_sw - r["removed"]
            doc.EditRebuild3()
            print(f"{part:11s} trench: {len(jobs)} bodies cut, removing {sum(r['removed'] for r, _ in jobs):.2f} mm3",
                  flush=True)
        # the pipe bodies, through a temporary import (closed unsaved)
        step = os.path.normpath(os.path.join(OUT, f"{part}_pipes.step"))
        res = sw.LoadFile4(step, "r", None, 0)
        tmp = wrap(res[0] if isinstance(res, tuple) else res, sld.IModelDoc2)
        if tmp is None:
            raise SystemExit(f"{part}: could not import {step}")
        src = [(S.volume(b), b.GetBodyBox(), wrap(b.Copy(), sld.IBody2)) for b in S.bodies(tmp)]
        sw.CloseDoc(tmp.GetTitle())
        sw.ActivateDoc3(doc.GetTitle(), False, 0, 0)
        pal = _palette(part)
        hoses = {x["name"]: x["hose_mm3"] for x in rep["pipes"]}
        centre = lambda bx: np.array(bx[:3]) * 500.0 + np.array(bx[3:]) * 500.0   # m box -> mm centre
        named, hose_c = [], {}
        for v, bx, body in sorted(src, key=lambda x: -x[0]):
            nm = min(hoses, key=lambda n: abs(hoses[n] - v))
            if abs(hoses[nm] - v) < 0.5 and nm not in hose_c:
                hose_c[nm] = centre(bx)
                named.append((f"PP_{nm}", "blue", body, v))
            else:
                near = min(hose_c, key=lambda n: np.linalg.norm(hose_c[n] - centre(bx)))
                k = sum(1 for x in named if x[0].startswith(f"PP_{near}_collar")) + 1
                named.append((f"PP_{near}_collar{k}", "graphite", body, v))
        pdoc = wrap(doc, sld.IPartDoc)
        for name, grp, body, v in named:
            names0 = {b.Name for b in S.bodies(doc)}
            f = wrap(pdoc.CreateFeatureFromBody3(body, False, c.swCreateFeatureBodyCheck), sld.IFeature)
            if f is None:
                raise SystemExit(f"{part}: SolidWorks refused body {name}")
            f.Name = name
            new = [b for b in S.bodies(doc) if b.Name not in names0]
            if len(new) != 1:
                raise SystemExit(f"{part}: {name} made {len(new)} bodies")
            S.colour(new[0], pal[grp], name)
        doc.EditRebuild3()
        # checks, body for body by volume: the originals exactly as before (a trenched one
        # exactly minus the predicted removal), plus exactly the STEP's pipe bodies
        got = sorted(S.volume(b) for b in S.bodies(doc))
        want = sorted(expect + [v for *_, v in named])
        worst = max(abs(a - b) / max(1.0, 2e-6 * b / 0.05) for a, b in zip(got, want))             if len(got) == len(want) else float("inf")         # 0.05 mm3, or 2 ppm on big bodies
        ok = worst < 0.05
        print(f"{part:11s} {len(named)} bodies in ({', '.join(n for n, *_ in named)}); "
              f"{'replaced ' + str(n_old) + ' old PP_/PT_ features; ' if n_old else ''}"
              f"{len(got)} bodies vs {len(want)} predicted, worst {worst:.4f} mm3  {'OK' if ok else 'FAIL'}",
              flush=True)
        if not ok:
            raise SystemExit(f"{part}: check failed -- nothing saved")
        done.append(PARTS[part])
    return sw, done


# ============================================================ print files (offline)
LINK_FIL = {(0.93, 0.94, 0.95): 1, (0.25, 0.27, 0.30): 2, (0.13, 0.45, 0.95): 3}
BOX_FIL = {(0xF2 / 255, 0xF4 / 255, 0xF7 / 255): 1, (0x7E / 255, 0x87 / 255, 0x95 / 255): 2,
           (0x1E / 255, 0x7B / 255, 1.0): 3, (0x2B / 255, 0x2F / 255, 0x36 / 255): 4}


def cmd_print(which):
    """Bambu project 3MFs with the pipes, from the same geometry as SolidWorks (the export
    + the pipe STEP; `build` proved the part = export + pipes by volume).  Links: part frame,
    filament 1 white / 2 graphite / 3 blue (as 11).  Box parts: print orientation and
    1 white / 2 graphite / 3 blue / 4 dark (as box_facet_print / tail_strut)."""
    _aes()
    import export3mf
    from build123d import import_step
    for part in which:
        P = Part(part)
        box = part in PRINT_UP
        fil = BOX_FIL if box else LINK_FIL
        if part == "TailStrut":
            fil = {k: (2 if k == TAIL_GRAPHITE else v) for k, v in BOX_FIL.items()}
        pal = _palette(part)
        bodies = [(s, rgb) for s, rgb in zip(P.solids, P.rgb)]
        tr = os.path.join(OUT, f"{part}_trench.step")
        if os.path.exists(tr):                      # a trench: the part with its slot cut
            cut = import_step(tr).solids()
            out_ = []
            for s, rgb in bodies:
                for c_ in cut:
                    s = s.cut(c_)
                out_ += [(x, rgb) for x in s.solids() if x.volume > 1.0]
            bodies = out_
        pipe = os.path.join(OUT, f"{part}_pipes.step")
        for s in import_step(pipe).solids():
            bodies.append((s, pal["blue"] if s.volume > 60 else pal["graphite"]))   # collars ~44 mm3
        if box:
            u = np.asarray(PRINT_UP[part], float)
            a = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            e1 = np.cross(a, u)
            e1 /= np.linalg.norm(e1)
            Rp = np.vstack([e1, np.cross(u, e1), u])          # part -> print frame (rows), as box_facet_print
        else:
            Rp = np.eye(3)
        objs, by = [], {}
        for k, (s, rgb) in enumerate(bodies):
            f = fil[min(fil, key=lambda c_: sum((a_ - b_) ** 2 for a_, b_ in zip(rgb, c_)))]
            V, T = _tess(s, 0.05)
            objs.append((f"{part}_{k}", V @ Rp.T, T, f))
            by[f] = by.get(f, 0) + 1
        name = f"{part}_bambu.3mf" if box else f"{part.replace(' ', ' ')}_glacier_native.3mf"
        out = os.path.join(HERE, "out", "print", name)
        export3mf.write_3mf(objs, out, name=f"{part} (with pipes)")
        allV = np.vstack([o[1] for o in objs])
        ext = allV.max(0) - allV.min(0)
        print(f"{part:11s} {len(objs)} bodies, filaments {dict(sorted(by.items()))}, "
              f"{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm -> out/print/{name}", flush=True)


def cmd_preview(which):
    ok = {}
    for part in which:
        print(f"\n=== {part}", flush=True)
        ok[part] = build_part(part)
    print("\nSUMMARY " + "  ".join(f"{k}: {'OK' if v else ('-' if v is None else 'FAIL')}" for k, v in ok.items()))


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    args = [a for a in sys.argv[1:]]
    cmd = args.pop(0) if args else "preview"
    if cmd == "sweep":
        sweep(float(args[0]) if args else 1.0)
        raise SystemExit
    if cmd == "robot":
        cmd_robot(tuple(float(a) for a in args) if args else (-28.0, 20.0, 57.0))
        raise SystemExit
    flags = {a for a in args if a.startswith("--")}
    args = [a for a in args if not a.startswith("--")]
    which = args or list(PARTS)
    bad = [p for p in which if p not in PARTS]
    if bad:
        raise SystemExit(f"unknown part(s) {bad}; known: {list(PARTS)}")
    if cmd == "export":
        export(which)
    elif cmd == "map":
        cmd_map(which)
    elif cmd == "routes":
        cmd_routes(which, trench="--trench" in flags)
    elif cmd == "build":
        from importlib import import_module
        sw, done = sw_build([p for p in which if p in PIPES] if not args else which)
        import_module("16_check_and_save").chain(sw, done)
    elif cmd == "print":
        cmd_print([p for p in which if p in PIPES] if not args else which)
    elif cmd == "preview":
        cmd_preview([p for p in which if p in PIPES] if not args else which)
    else:
        raise SystemExit(f"unknown command {cmd}")
