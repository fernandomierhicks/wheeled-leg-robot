"""Step 28: relief -- colour shapes pressed into the parts, grooves on the link sides.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/28_relief.py <command> [Part ...]

His brief (2026-10-09, four marked screenshots): "whenever there is a shape of a
different colour in a part, when possible make it a recess or a shallow extrusion --
we are 3D printed, complexity is free"; the white sides of the links get accents.
His calls: PRESSED by default (removals only, so nothing new can collide); the box
parts, the RobotMount's left field, the encoder parts and the wheel hub; recessed
grooves with coloured floors on the link sides; offline renders before SolidWorks.

The FacetHood is NOT done here: its source of truth is hood_variants.py, and its
mates are locks only, so it goes the documented way (hood_variants -> 21 reimport).

How a part changes (everything at the END of its feature tree, like 24's pipes, so
the faces its mates reference are never touched -- FacetBack / FacetFront are
mated to brackets and the switch, the TailStrut by its TS_* mates):

  press(face, region, d, colour, t)
      the region sinks d below the face; under it a floor t thick in `colour`.
      Every body loses what the recess covers; every body of another colour also
      loses the floor slab, which comes back as ONE new body of `colour`.
  raise(face, region, h, colour)
      a new body h proud of the face (the wheel hub: its floor is too thin to press).

  In SolidWorks: per body, its cutter (Imported RL_toolN, Combine-subtracted:
  RL_cutN), a body the recess swallows whole is deleted (RL_gone), then every new
  body (Imported RL_<name>, coloured).  Checked body for body by volume against
  the offline result.

export   READ-ONLY: a multi-body STEP of each part as it is in SolidWorks now + a JSON
         of its bodies' colours / volumes / placement (out/relief/src/).
faces    the planar faces of a part, grouped by plane, with area per colour.
preview  builds every op offline, checks it, writes out/relief/<Part>_relief.json,
         the cutters + new bodies (STEP) and a before/after board <Part>_relief.png.
build    the previewed relief into the v5 parts (RL_* features), 16 --chain.
print    Bambu 3MFs (out/print/), filaments as the originals.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "relief")
SRC = os.path.join(OUT, "src")

PARTS = {"RobotMount": r"Body\OldRobotBodyMount\RobotMount.SLDPRT",
         "FacetFront": r"Box\Facet\FacetFront.SLDPRT",
         "FacetBack": r"Box\Facet\FacetBack.SLDPRT",
         "TailStrut": r"Box\TailStrut\TailStrut.SLDPRT",
         "EncoderCarrier": r"Links\EncoderCarrier.SLDPRT",
         "EncoderCableClamp": r"Links\EncoderCableClamp.SLDPRT",
         "Wheel": r"Motor\Wheel Motor\Wheel.SLDPRT",
         "Femur": r"Links\Femur.SLDPRT",
         "Femur_inside": r"Links\Femur_inside.SLDPRT",
         "Coupler": r"Links\Coupler.SLDPRT",
         "Tibia": r"Links\Tibia.SLDPRT"}
# every other printed part: `export` + `print` only (no relief), so ALL the 3MFs can be
# regenerated from SolidWorks the same way (his call 2026-10-09: "regenerate all print files")
PRINT_ONLY = {"Side panel": r"Body\Side panel.SLDPRT",
              "FacetHood": r"Box\Facet\FacetHood.SLDPRT",
              "FacetRing": r"Box\Facet\FacetRing.SLDPRT",
              "BumperFront": r"Box\Facet\BumperFront.SLDPRT",
              "BumperBack": r"Box\Facet\BumperBack.SLDPRT",
              "BearingWasher": r"Links\BearingWasher.SLDPRT",
              "SmallBearingWahser": r"Body\SmallBearingWahser.SLDPRT",
              "InsideFemurShaft": r"Body\InsideFemurShaft.SLDPRT"}
ALL_PARTS = {**PARTS, **PRINT_ONLY}


def _aes():
    lib = os.path.normpath(os.path.join(HERE, "..", "aesthetics", "lib"))
    if lib not in sys.path:
        sys.path.insert(0, lib)


# ============================================================ export (SolidWorks, read-only)
def export(which):
    import importlib
    import swlib
    import swstyle as S
    from swlib import c, wrap, sld
    rgb_of = importlib.import_module("24_pipes")._rgb
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    comps = swlib.components(robot)
    os.makedirs(SRC, exist_ok=True)
    docs = {os.path.normcase(wrap(d, sld.IModelDoc2).GetPathName()): wrap(d, sld.IModelDoc2)
            for d in sw.GetDocuments() or []}
    for part in which:
        path = os.path.join(swlib.V5, ALL_PARTS[part])
        doc = docs.get(os.path.normcase(path))
        if doc is None:
            print(f"{part}: NOT LOADED ({path})")
            continue
        bodies = []
        for b in S.bodies(doc):
            rgb, src = rgb_of(b, doc)
            box = [round(v * 1000.0, 4) for v in b.GetBodyBox()]
            bodies.append({"name": b.Name, "volume": round(S.volume(b), 4), "rgb": rgb,
                           "colour_from": src, "box": box})
        step = os.path.join(SRC, f"{part}.step")
        ok, err, warn = doc.Extension.SaveAs3(step, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy,
                                              None, None, 0, 0)
        inst = []
        for n, cp in comps.items():
            if os.path.normcase(cp.GetPathName() or "") == os.path.normcase(path) and not cp.IsSuppressed():
                inst.append({"name": n, "placement": np.round(swlib.placement(cp), 6).tolist()})
        rl = [f.Name for f in S._iter_features(doc) if f.Name.startswith("RL_")]
        info = {"part": part, "file": ALL_PARTS[part], "dirty": bool(doc.GetSaveFlag()),
                "relief_features": rl, "bodies": bodies, "instances": inst}
        with open(os.path.join(SRC, f"{part}.json"), "w") as fh:
            json.dump(info, fh, indent=1)
        vol = sum(x["volume"] for x in bodies)
        print(f"{part:17s} STEP {'ok' if ok else 'FAILED'}  {len(bodies):3d} bodies {vol:10.1f} mm3  "
              f"colours from {sorted({x['colour_from'] for x in bodies})}  {len(inst)} instance(s)  "
              f"{'DIRTY in memory' if info['dirty'] else 'saved'}{'  ALREADY HAS RL_ features' if rl else ''}",
              flush=True)


# ============================================================ the part, offline
# colour kinds: the links' GLACIER (swstyle.GROUPS) and the box's (box_concept_facet)
KINDS = {"white": [(0.93, 0.94, 0.95), (0xF2 / 255, 0xF4 / 255, 0xF7 / 255)],
         "graphite": [(0.25, 0.27, 0.30), (0x7E / 255, 0x87 / 255, 0x95 / 255)],
         "blue": [(0.13, 0.45, 0.95), (0x1E / 255, 0x7B / 255, 1.0)],
         "dark": [(0x2B / 255, 0x2F / 255, 0x36 / 255)]}


def kind(rgb):
    return min(((k, sum((a - b) ** 2 for a, b in zip(rgb, c))) for k, cs in KINDS.items() for c in cs),
               key=lambda x: x[1])[0]


class Part:
    """A part's bodies (from `export`), each with its colour, part-local mm."""

    def __init__(self, part, built=False):
        from build123d import import_step
        self.name = part
        self.info = json.load(open(os.path.join(SRC, f"{part}.json")))
        if self.info.get("relief_features") and not built:
            raise SystemExit(f"{part}: the export already carries RL_ features -- `build` replaces "
                             f"them; re-export after deleting them, or design on top knowingly")
        solids = import_step(os.path.join(SRC, f"{part}.step")).solids()
        left = list(self.info["bodies"])
        self.bodies = []
        def ctr(bx):
            return np.array([(bx[0] + bx[3]) / 2, (bx[1] + bx[4]) / 2, (bx[2] + bx[5]) / 2])
        for s in solids:              # colour by volume: STEP body order is not promised
            b = min(left, key=lambda b: abs(b["volume"] - s.volume))
            if abs(b["volume"] - s.volume) > 1e-3 * max(1.0, b["volume"]) + 0.05:
                # SolidWorks' STEP of a small curved body can read up to ~1 mm3 off in OCC (the
                # wheel's raised inlays, 2026-10-09): match it by where it is, then by volume
                sb = s.bounding_box()
                c = np.array([(sb.min.X + sb.max.X) / 2, (sb.min.Y + sb.max.Y) / 2, (sb.min.Z + sb.max.Z) / 2])
                b = min(left, key=lambda b: np.linalg.norm(ctr(b["box"]) - c) + abs(b["volume"] - s.volume))
                if np.linalg.norm(ctr(b["box"]) - c) > 0.5 or abs(b["volume"] - s.volume) > 0.1 * b["volume"]:
                    raise SystemExit(f"{part}: STEP body {s.volume:.2f} mm3 matches nothing in the JSON")
            left.remove(b)
            rgb = tuple(b["rgb"]) if b["rgb"] else (0.9, 0.9, 0.9)
            self.bodies.append({"solid": s, "rgb": rgb, "kind": kind(rgb), "name": b["name"],
                                "volume": b["volume"]})
        P = np.array(self.info["instances"][0]["placement"])
        self.R = P[:, :3]               # part-local -> ROBOT rotation (saved pose)
        self.pal = {}
        for b in sorted(self.bodies, key=lambda b: -b["volume"]):
            self.pal.setdefault(b["kind"], b["rgb"])
        self._planes = None

    def rgb(self, k):
        """The part's own shade of a colour kind (box graphite is lighter than the links')."""
        if k in self.pal:
            return self.pal[k]
        box = any(abs(c[0] - 0xF2 / 255) < 0.01 for c in [self.pal.get("white", (0, 0, 0))])
        return KINDS[k][1 if box and len(KINDS[k]) > 1 else 0]

    def planes(self, min_area=20.0):
        """Planar faces grouped by plane: [(n, d, {kind: area}, [(face, body index)])], biggest first."""
        if self._planes is None:
            from OCP.BRepAdaptor import BRepAdaptor_Surface
            from OCP.GeomAbs import GeomAbs_Plane
            groups = {}
            for i, b in enumerate(self.bodies):
                for f in b["solid"].faces():
                    if BRepAdaptor_Surface(f.wrapped).GetType() != GeomAbs_Plane:
                        continue
                    m = f.normal_at()
                    n = np.array([m.X, m.Y, m.Z])
                    c = f.center()
                    d = float(n @ [c.X, c.Y, c.Z])
                    key = (*np.round(n, 3), round(d, 1))
                    g = groups.setdefault(key, [n, d, {}, []])
                    g[2][b["kind"]] = g[2].get(b["kind"], 0.0) + f.area
                    g[3].append((f, i))
            self._planes = sorted(groups.values(), key=lambda g: -sum(g[2].values()))
        return [g for g in self._planes if sum(g[2].values()) >= min_area]

    def union(self):
        return _fuse([b["solid"] for b in self.bodies])


# ============================================================ geometry (offline)
AIR = 3.0          # every cutter runs this far out of the face, into air: no tool face lies on a
                   # part face (coincident faces kill booleans, gotcha 19)
GROW = 0.05        # a recess over an EXISTING inlay is grown this much, so its walls sit in the
                   # body round the inlay, never on the inlay's own walls (gotcha 19 again)
MIN_W = 0.9        # a shape narrower than this stays flush: a 0.4 nozzle cannot print its recess
MIN_WALL = 1.0     # solid that must stay behind every floor


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _fuse(shapes):
    shapes = [s for s in shapes if s is not None]
    if not shapes:
        return None
    out = shapes[0]
    for s in shapes[1:]:
        out = out.fuse(s)
    return out.clean() if len(shapes) > 1 else out


def _vol(s):
    return 0.0 if s is None else sum(x.volume for x in s.solids())


def _near(a, b, tol=0.05):
    p, q = a.bounding_box(), b.bounding_box()
    return not (p.min.X > q.max.X + tol or q.min.X > p.max.X + tol or p.min.Y > q.max.Y + tol or
                q.min.Y > p.max.Y + tol or p.min.Z > q.max.Z + tol or q.min.Z > p.max.Z + tol)


def _solid_or_none(s, min_mm3=0.01):
    """Drop boolean slivers; None when nothing is left."""
    from build123d import Compound
    if s is None:
        return None
    k = [x for x in s.solids() if x.volume > min_mm3]
    return None if not k else (k[0] if len(k) == 1 else Compound(k))


class Frame:
    """A plane of the part, snapped to its real faces: origin on it, (u, v) in it, w = outward
    normal, all part-local.  u defaults to ROBOT +X projected (24's rule), v = w x u."""

    def __init__(self, P, n, d, u=None, name=""):
        n = unit(n)
        best = None
        for m, dd, areas, fs in P.planes(min_area=0.0):
            err = np.degrees(np.arccos(np.clip(m @ n, -1, 1))) + abs(dd - d)
            if best is None or err < best[0]:
                best = (err, m, dd, fs)
        if best is None or best[0] > 1.0:
            raise SystemExit(f"{P.name}: no planar face near n={n.round(3)} d={d}")
        self.P, self.name = P, name
        self.w, self.d, self.faces = unit(best[1]), best[2], best[3]
        if u is None:
            nR = P.R @ self.w
            right = np.array([1.0, 0, 0]) if abs(nR[0]) < 0.7 else np.array([0, 0, -1.0 if nR[0] > 0 else 1.0])
            u = P.R.T @ right
        u = np.asarray(u, float)
        self.u = unit(u - (u @ self.w) * self.w)
        self.v = np.cross(self.w, self.u)
        self.o = self.w * self.d
        self.A = np.array([self.u, self.v, self.w])

    def to(self, V):
        """part-local -> (u, v, w)"""
        return (np.asarray(V, float) - self.o) @ self.A.T

    def loc(self):
        from build123d import Location, Plane
        return Location(Plane(origin=tuple(self.o), x_dir=tuple(self.u), z_dir=tuple(self.w)))

    def prism(self, region, w0, w1):
        _aes()
        from shputil import prism
        s = prism(region, w0, w1, min_area=0.01)
        return None if s is None else s.moved(self.loc())

    def footprint(self, kinds=None, bodies=None):
        """The (u, v) outline of this plane's faces -- of bodies of the given colour kinds
        (or body indices) -- as one shapely geometry."""
        _aes()
        from render3d import tessellate
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        tris = []
        for f, i in self.faces:
            if (kinds and self.P.bodies[i]["kind"] not in kinds) or (bodies is not None and i not in bodies):
                continue
            V, T, _ = tessellate(f, 0.02)
            if not len(T):
                continue
            uv = self.to(V)[:, :2]
            tris += [Polygon(uv[t]) for t in T]
        if not tris:
            return Polygon()
        return unary_union([t.buffer(1e-4) for t in tris if t.area > 1e-8]).buffer(-1e-4)


def press(frame, region, d, colour, t=0.8, grow=0.0, name="press"):
    """The region sinks d below the face, on a floor t thick in `colour`."""
    return {"op": "press", "frame": frame, "region": region, "d": d, "t": t, "colour": colour,
            "grow": grow, "name": name}


def raise_(frame, region, h, colour, name="raise"):
    """A new body h proud of the face."""
    return {"op": "raise", "frame": frame, "region": region, "h": h, "colour": colour, "name": name}


def press_inlays(frame, kinds, d, t, name, keep=None, min_w=MIN_W, inside=0.2):
    """Press every flush inlay of the given colour kinds on this plane: one press per
    connected shape, each on a floor of its own colour.  A shape narrower than min_w
    anywhere, or touching the face's outer edge (it would notch the edge), stays flush.
    keep: a (u, v) region nothing is pressed in."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    face_all = frame.footprint()
    # the face's OUTER edge only: an inlay round a well or a hole (the dark linings) is pressed
    edge = unary_union([Polygon(g.exterior) for g in getattr(face_all, "geoms", [face_all])
                        if isinstance(g, Polygon)]).buffer(-inside, join_style=2)
    ops, skipped = [], []
    for k in kinds:
        g = frame.footprint(kinds=[k])
        for j, p in enumerate(getattr(g, "geoms", [g])):
            if not isinstance(p, Polygon) or p.area < 0.5:
                continue
            why = None
            opened = p.buffer(-min_w / 2, join_style=2).buffer(min_w / 2, join_style=2)
            if keep is not None and p.intersects(keep):
                why = "keep-out"
            elif not edge.contains(p):
                why = "reaches the face edge"
            elif p.area - opened.area > 0.15 * p.area:
                # mostly wide enough (a lining with one thin sliver): press the wide part only,
                # the sliver stays flush in the same colour
                big = [q for q in getattr(opened, "geoms", [opened]) if isinstance(q, Polygon) and q.area > 2.0]
                if big and sum(q.area for q in big) > 0.5 * p.area:
                    skipped.append((k, round(p.area - sum(q.area for q in big), 1), "its thin part stays flush"))
                    for q in big:
                        ops.append(press(frame, q, d, k, t=t, grow=GROW, name=f"{name}_{k}{j + 1}"))
                    continue
                why = f"narrower than {min_w} mm"
            if why:
                skipped.append((k, round(p.area, 1), why))
                continue
            ops.append(press(frame, p, d, k, t=t, grow=GROW, name=f"{name}_{k}{j + 1}"))
    return ops, skipped


def apply(P, ops):
    """Every op in order on the part's bodies.  Returns (state, cuts, log): state = the final
    bodies [{solid, kind, rgb, orig, name}] (orig = index of the original body, None for a
    new one); cuts[i] = the cutter prisms applied to original body i."""
    state = [dict(solid=b["solid"], kind=b["kind"], rgb=b["rgb"], orig=i, name=b["name"])
             for i, b in enumerate(P.bodies)]
    cuts = {i: [] for i in range(len(P.bodies))}
    log = []
    for op in ops:
        F = op["frame"]
        if op["op"] == "press":
            reg = op["region"]
            # pressing an EXISTING inlay: the recess grows GROW into the body round it and the
            # floor slab shrinks GROW inside it -- neither has a wall on the inlay's own walls.
            # (A floor slab on them, where the inlay runs deeper than the floor -- the
            # RobotMount's through-plate graphite -- wiped the whole white plate in OCC.)
            g = op["grow"]
            Rz = F.prism(reg.buffer(g, join_style=2) if g else reg, -op["d"], AIR)
            Fl = F.prism(reg.buffer(-g, join_style=2) if g else reg, -op["d"] - op["t"], -op["d"])
            floor, removed = [], 0.0
            for s in state:
                if s["solid"] is None or not _near(s["solid"], Rz):
                    continue
                tool = Rz
                if s["kind"] != op["colour"] and Fl is not None:
                    piece = _solid_or_none(s["solid"] & Fl, 0.05)
                    if piece is not None:
                        floor.append(piece)
                        tool = Rz.fuse(Fl).clean()
                before = _vol(s["solid"])
                new = _solid_or_none(s["solid"].cut(tool))
                dv = before - _vol(new)
                if dv > _vol(tool) + 0.05:
                    raise SystemExit(f"{P.name} {op['name']}: body {s['name']} lost {dv:.1f} mm3 to a "
                                     f"{_vol(tool):.1f} mm3 cutter -- a failed boolean, stopped")
                if dv < 1e-3:
                    continue
                s["solid"] = new
                if s["orig"] is not None:
                    cuts[s["orig"]].append(tool)
                removed += dv
            fl = _fuse(floor)
            if fl is not None:
                state.append(dict(solid=fl, kind=op["colour"], rgb=P.rgb(op["colour"]), orig=None,
                                  name=op["name"]))
            log.append(dict(name=op["name"], op="press", area=round(reg.area, 2), d=op["d"], t=op["t"],
                            colour=op["colour"], removed=round(removed, 3), floor=round(_vol(fl), 3)))
        elif op["op"] == "raise":
            body = F.prism(op["region"], 0.0, op["h"])
            # cut only by what stands ABOVE the face (a probe 0.02 up, 0.05 in): the flush inlay
            # under a raised shape touches the prism AT the face, and that coincident-face cut
            # gave junk (the wheel 2026-10-09: a chevron at half volume, one bigger than its prism)
            probe_ = F.prism(op["region"].buffer(-0.05, join_style=2), 0.02, op["h"])
            for s in state:
                if s["solid"] is not None and _near(s["solid"], probe_) and \
                        _vol(_solid_or_none(s["solid"] & probe_, 0.0)) > 1e-3:
                    body = body.cut(s["solid"])
            body = _solid_or_none(body, 0.5)
            if body is not None:
                state.append(dict(solid=body, kind=op["colour"], rgb=P.rgb(op["colour"]), orig=None,
                                  name=op["name"]))
            log.append(dict(name=op["name"], op="raise", area=round(op["region"].area, 2), h=op["h"],
                            colour=op["colour"], added=round(_vol(body), 3)))
    return state, cuts, log


def checks(P, ops, state):
    """Per press: solid behind its floor (MIN_WALL), how much of it is off the face, and how
    much of it is narrower than MIN_W.  Returns [(name, ok, text)]."""
    def inside(probe, solids):
        """Volume of probe inside the bodies -- summed body by body (the bodies partition the
        part; fusing 18-46 touching colour bodies into one is what OCC fails at)."""
        return sum(_vol(_solid_or_none(probe & s, 0.0)) for s in solids if s is not None and _near(s, probe))

    now = [s["solid"] for s in state]
    was = [b["solid"] for b in P.bodies]
    out = []
    for op in ops:
        F, reg = op["frame"], op["region"]
        if op["op"] == "press":
            # probes 0.05 INSIDE the region and off the face: a probe face lying on a body
            # face makes the boolean return nothing (gotcha 19, measured here: 0 % solid)
            z = op["d"] + op["t"]
            r_in = reg.buffer(-0.05, join_style=2)
            back = F.prism(r_in, -z - MIN_WALL, -z - 0.02)
            frac = 1.0 - inside(back, now) / max(_vol(back), 1e-9)
            skin = F.prism(r_in, -0.10, -0.02)
            on = inside(skin, was) / max(_vol(skin), 1e-9)
            thin = reg.area - reg.buffer(-MIN_W / 2, join_style=2).buffer(MIN_W / 2, join_style=2).area
            ok = frac < 0.02 and on > 0.95
            out.append((op["name"], ok, f"wall behind {100 * (1 - frac):5.1f}% solid, on the face {100 * on:5.1f}%, "
                                        f"narrower than {MIN_W}: {thin:5.2f} mm2"))
        else:
            out.append((op["name"], True, "raised"))
    return out


# ============================================================ the design, per part
def _proj(shape, frame, dev=0.05):
    """A solid's silhouette on a frame's (u, v)."""
    _aes()
    from render3d import tessellate
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    V, T, _ = tessellate(shape, dev)
    uv = frame.to(V)[:, :2]
    return unary_union([Polygon(uv[t]).buffer(1e-3) for t in T if Polygon(uv[t]).area > 1e-6])


def d_robotmount(P):
    """His image 1: the graphite step-bar of the left field (extras.py) and the recipe bar it
    carries on from, pressed 1.0 -- the white vent slashes in its head stay at the face as
    standing ribs.  Stops 2 mm short of the trench pipe's slot.  The plate is graphite
    through there, so no new floor body: the recess just lowers the graphite."""
    from build123d import import_step
    from shapely.ops import unary_union
    F = Frame(P, (0, 0, 1), 5.0, u=(1, 0, 0), name="show face")
    g = F.footprint(kinds=["graphite"])
    slot = _proj(import_step(os.path.join(HERE, "out", "pipes", "RobotMount_trench.step")), F)
    field = unary_union([p for p in getattr(g, "geoms", [g]) if p.bounds[0] < 100 and p.bounds[1] > 0])
    region = field.difference(slot.buffer(2.0, join_style=2))
    return [press(F, region, 1.0, "graphite", t=1.0, grow=GROW, name="stepbar")], []


def _slash(uc, vc, w, h, skew):
    """A parallelogram slot: w wide, h tall, top shifted `skew` along u."""
    from shapely.geometry import Polygon
    return Polygon([(uc - w / 2 - skew / 2, vc - h / 2), (uc + w / 2 - skew / 2, vc - h / 2),
                    (uc + w / 2 + skew / 2, vc + h / 2), (uc - w / 2 + skew / 2, vc + h / 2)])


def d_facetback(P):
    """His image 2: the blue brow trace (0.8 inlay) -> a 0.9 channel on a blue floor; the
    graphite vent slashes (2.0 inlay, they run round the corner onto the left cheek) and the
    dark linings round the I/O bay and the control pod -> pressed 1.0 into their own colour.
    New: three dark raked slots pressed under the brow trace, right of the centre screw."""
    from shapely.ops import unary_union
    F = Frame(P, (-1, 0, 0), 164.0, u=(0, 0, 1), name="back face")
    Fc = Frame(P, (-0.949, 0, -0.316), 170.76, name="left cheek")
    ops, skipped = press_inlays(F, ["blue"], 0.9, 0.8, "back")
    o2, s2 = press_inlays(F, ["graphite", "dark"], 1.0, 0.8, "back", inside=-1.0)
    o3, s3 = press_inlays(Fc, ["graphite"], 1.0, 0.8, "cheek", inside=-1.0)
    slots = unary_union([_slash(u, 63.6, 1.6, 4.4, 1.6) for u in (5.5, 8.5, 11.5)])
    o4 = [press(F, slots, 1.0, "dark", t=0.8, name="back_slots")]
    return ops + o2 + o3 + o4, skipped + s2 + s3


def d_facetfront(P):
    """The dark lining round the screen well -> pressed 1.0 (a stepped rim); the two blue
    traces on the cheek facets beside it -> 0.9 channels on blue floors."""
    F = Frame(P, (1, 0, 0), 65.0, u=(0, 0, -1), name="front face")
    ops, skipped = press_inlays(F, ["dark"], 1.0, 0.8, "front", inside=-1.0)
    for s, nm in ((1, "right cheek"), (-1, "left cheek")):
        Fc = Frame(P, (0.949, 0, 0.316 * s), 76.21, name=nm)
        o, sk = press_inlays(Fc, ["blue"], 0.9, 0.8, f"cheek{'R' if s > 0 else 'L'}")
        ops += o
        skipped += sk
    return ops, skipped


def _auto_inlays(P, planes, d, t, kinds=("blue", "graphite", "dark", "white"), inside=0.2, raise_h=None):
    """press_inlays (or, with raise_h, raise) every shape of a colour OTHER than the plane's
    main colour on each listed plane."""
    ops, skipped = [], []
    for i, (n, dd, nm) in enumerate(planes):
        F = Frame(P, n, dd, name=nm)
        areas = {}
        for f, j in F.faces:
            areas[P.bodies[j]["kind"]] = areas.get(P.bodies[j]["kind"], 0.0) + f.area
        base = max(areas, key=areas.get)
        ks = [k for k in kinds if k != base and k in areas]
        if raise_h is None:
            o, s = press_inlays(F, ks, d, t, f"p{i}", inside=inside)
        else:
            from shapely.geometry import Polygon
            o, s = [], []
            for k in ks:
                g = F.footprint(kinds=[k])
                for j, p in enumerate(getattr(g, "geoms", [g])):
                    if not isinstance(p, Polygon) or p.area < 0.5:
                        continue
                    if p.area - p.buffer(-MIN_W / 2, join_style=2).buffer(MIN_W / 2, join_style=2).area > 0.15 * p.area:
                        s.append((k, round(p.area, 1), f"narrower than {MIN_W} mm"))
                        continue
                    o.append(raise_(F, p, raise_h, k, name=f"r{i}_{k}{j + 1}"))
        ops += o
        skipped += s
    return ops, skipped


def d_tailstrut(P):
    """The strut's blue inlays (0.8 deep): the chevrons on the pad flanks, the lining round
    the V window on both keel flanks, the tail facet's line -> 0.9 channels on blue floors.
    NOT the dark crystal panels: they are most of their facets (pressing them just lowers
    the facet) and the keel behind them is thinner than floor + MIN_WALL (checked: 0-85 %)."""
    planes = [((0, -0.196, 0.981), 28.44, "pad flank R"), ((0, -0.196, -0.981), 28.44, "pad flank L"),
              ((0, 0, 1), 16.0, "keel flank R"), ((0, 0, -1), 16.0, "keel flank L"),
              ((-0.779, 0.628, 0), 121.02, "tail facet")]
    return _auto_inlays(P, planes, 0.9, 0.8, kinds=("blue",), inside=-1.0)


def d_encoder_carrier(P):
    """The encoder face (y -5), the arm strips (y 7.8), the side tabs (y 6): every 0.6 inlay
    at least MIN_W wide -> pressed 0.4 on a 0.6 floor of its own colour.  Narrower ticks,
    the bus and the chevrons stay flush (a 0.4 nozzle cannot print their recess)."""
    planes = [((0, -1, 0), 5.0, "encoder face"), ((0, -1, 0), -7.8, "arm"), ((0, -1, 0), -6.0, "tabs")]
    return _auto_inlays(P, planes, 0.4, 0.6, inside=0.1)


def d_encoder_clamp(P):
    """The clamp top (y 10): the graphite frame band, chevrons and pad -> pressed 0.4."""
    return _auto_inlays(P, [((0, 1, 0), 10.0, "clamp top")], 0.4, 0.6, inside=0.1)


def d_wheel(P):
    """The hub face's flush inlays (white chevron brackets and pads, blue traces) RAISED 0.6:
    the face is a 0.8 mm floor carrying the wheel on its 4 screws, so nothing is pressed into
    it; nothing in the robot is outboard of the hub (22's note)."""
    return _auto_inlays(P, [((0, 0, -1), 4.05, "hub face")], 0, 0, raise_h=0.6)


# ---- the link sides: grooves with coloured floors ----------------------------------
SIDE_D, SIDE_T = 0.8, 0.8      # groove depth, coloured floor
SIDE_MARGIN = 1.2              # from the side's edges, its pockets and every other colour
LINE_W = 1.2                   # the blue channel


def _contact_keep(part):
    """The contacts the robot is designed to make (17_contact_keepout.py), plan xy + 2 mm."""
    from shapely.geometry import shape
    from shapely.ops import unary_union
    p = os.path.join(HERE, "out", "contact_keepout.json")
    if not os.path.exists(p):
        return None
    gs = [shape(g) for g in json.load(open(p)).get(part, [])]
    return unary_union(gs).buffer(2.0) if gs else None


def _side_pattern(R, k):
    """One side run (a shapely region, u along the link, v across its thickness) ->
    {colour: geometry}.  Along the run, repeating units in the GLACIER vocabulary: a
    graphite plate with one 45-degree raked end, a raked blue hatch comb, a blue channel
    with a 45-degree dogleg and a diamond pad.  On a tall side (the Tibia, the femur's
    flange) a long blue lane runs beside the units as well.  k varies the order."""
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union
    u0, v0, u1, v1 = R.bounds
    L, H = u1 - u0, v1 - v0
    out = {}
    if H < 2.6 or L < 12:
        return out
    vc = (v0 + v1) / 2
    lane = H >= 12.0
    if lane:                                        # units in the lower part, the lane above
        hb = min(6.0, H - 5.2)
        vc_u, vc_l = v0 + 0.3 + hb / 2, v0 + 0.3 + hb + 1.8 + LINE_W / 2
    else:
        hb = min(H - 0.6, 6.0)
        vc_u = vc
    blue, gfx = [], []
    seq = ["plate", "comb", "line", "plate", "line", "comb"]
    seq = seq[k % len(seq):] + seq[:k % len(seq)]
    u, i = u0 + 1.5, 0
    while u < u1 - 5.0:
        kd = seq[i % len(seq)]
        room = u1 - 1.5 - u
        if kd == "plate" and room >= 7.0:
            Lp, c = min(16.0, room), min(hb * 0.7, 3.0)
            a, b, lo, hi = u, u + Lp, vc_u - hb / 2, vc_u + hb / 2
            gfx.append(Polygon([(a, lo), (b - c, lo), (b, lo + c), (b, hi), (a, hi)] if i % 2 else
                               [(a + c, lo), (b, lo), (b, hi), (a, hi), (a, lo + c)]))
            u += Lp + 2.4
        elif kd == "comb" and room >= 7.0:
            for j in range(3):
                x = u + 0.5 + j * 2.1
                blue.append(Polygon([(x - 0.5 - 0.6, vc_u - hb / 2), (x + 0.5 - 0.6, vc_u - hb / 2),
                                     (x + 0.5 + 0.6, vc_u + hb / 2), (x - 0.5 + 0.6, vc_u + hb / 2)]))
            u += 3 * 2.1 + 2.4
        elif kd == "line" and room >= 8.0:
            Ll = min(20.0, room)
            dv = min(hb * 0.25, 1.6) * (1 if (i + k) % 2 else -1)
            w = min(LINE_W, hb * 0.35)
            if w >= MIN_W:
                ud = u + Ll * 0.4
                path = [(u + 1.0, vc_u + dv), (ud, vc_u + dv), (ud + 2 * abs(dv), vc_u - dv), (u + Ll - 1.6, vc_u - dv)]
                blue.append(LineString(path).buffer(w / 2, cap_style=2, join_style=2, mitre_limit=2.0))
                x, y = path[-1]
                blue.append(Polygon([(x - 1.0, y), (x, y - 1.0), (x + 1.0, y), (x, y + 1.0)]))
            u += Ll + 2.4
        else:
            u += 1.0
        i += 1
    if lane:
        blue.append(LineString([(u0 + 2.0, vc_l), (u1 - 2.0, vc_l)]).buffer(LINE_W / 2, cap_style=2))
    if blue:
        out["blue"] = unary_union(blue).intersection(R)
    if gfx:
        g = unary_union(gfx)
        out["graphite"] = (g.difference(out["blue"].buffer(0.8)) if blue else g).intersection(R)
    return out


def side_grooves(P, planes, seed=0):
    """Grooves on the given side planes: each white run of the side (its white faces minus
    SIDE_MARGIN round edges, pockets and other colours), exposed (nothing of the part in
    front of it), clear of the designed contacts, gets _side_pattern.  A shape whose floor
    would not have MIN_WALL of solid behind it is dropped (printed, never silent)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    keep = _contact_keep(P.name)
    ops, skipped = [], []
    for pi, (n, dd, nm) in enumerate(planes):
        F = Frame(P, n, dd, u=(1, 0, 0), name=nm)
        white = F.footprint(kinds=["white"])
        other = F.footprint(kinds=[k for k in KINDS if k != "white"])
        ground = white.buffer(-SIDE_MARGIN, join_style=2)
        if not other.is_empty:
            ground = ground.difference(other.buffer(SIDE_MARGIN, join_style=2))
        runs = sorted([g for g in getattr(ground, "geoms", [ground]) if isinstance(g, Polygon) and g.area > 30],
                      key=lambda g: g.bounds[0])
        for ri, R in enumerate(runs):
            pat = _side_pattern(R, seed + pi * 7 + ri)
            for colour, g in pat.items():
                for si, p in enumerate(getattr(g, "geoms", [g])):
                    if not isinstance(p, Polygon) or p.area < 1.5:
                        continue
                    p = p.buffer(-0.01, join_style=2).buffer(0.01, join_style=2)
                    if p.area - p.buffer(-MIN_W / 2, join_style=2).buffer(MIN_W / 2, join_style=2).area > 0.15 * p.area:
                        skipped.append((colour, round(p.area, 1), f"{nm} run {ri}: narrower than {MIN_W}"))
                        continue
                    if keep is not None:
                        xy = [tuple((F.o + a * F.u + b * F.v)[:2]) for a, b in p.exterior.coords]
                        from shapely.geometry import LineString
                        if LineString(xy).intersects(keep):
                            skipped.append((colour, round(p.area, 1), f"{nm} run {ri}: designed contact"))
                            continue
                    # exposed: nothing of the part within 25 mm in front of it
                    probe = F.prism(p.buffer(-0.1, join_style=2), 0.3, 25.0)
                    if probe is not None and any(_near(b["solid"], probe) and _vol(_solid_or_none(b["solid"] & probe, 0.0)) > 0.05
                                                 for b in P.bodies):
                        skipped.append((colour, round(p.area, 1), f"{nm} run {ri}: hidden behind the part"))
                        continue
                    # solid behind the floor
                    back = F.prism(p.buffer(-0.05, join_style=2), -SIDE_D - SIDE_T - MIN_WALL, -SIDE_D - SIDE_T - 0.02)
                    got = sum(_vol(_solid_or_none(b["solid"] & back, 0.0)) for b in P.bodies if _near(b["solid"], back))
                    if got < 0.98 * _vol(back):
                        skipped.append((colour, round(p.area, 1), f"{nm} run {ri}: thin wall behind"))
                        continue
                    ops.append(press(F, p, SIDE_D, colour, t=SIDE_T, name=f"side{pi}_{ri}_{colour[0]}{si}"))
    return ops, skipped


# side planes per link: (outward normal, offset) of the long sides, from `faces`
SIDES = {
    "Femur": [((0.013, 1, 0), 18.2, "+Y side"), ((0.013, -1, 0), 18.2, "-Y side"),
              ((0, 1, 0), 22.6, "+Y flange side"), ((0, -1, 0), 23.53, "-Y flange side")],
    "Femur_inside": [((0.028, 1, 0), 19.62, "+Y side"), ((0.028, -1, 0), 19.63, "-Y side")],
    "Coupler": [((-0.03, 1, 0), 19.51, "+Y side"), ((-0.03, -1, 0), 19.51, "-Y side")],
    "Tibia": [((-0.026, 1, 0), 16.02, "+Y side"), ((0.034, -0.999, 0), 27.59, "-Y side"),
              ((0.034, -0.999, 0), 34.64, "-Y side, wheel end")],
}


def d_link(part):
    return lambda P: side_grooves(P, SIDES[part], seed={"Femur": 0, "Femur_inside": 3, "Coupler": 5, "Tibia": 2}[part])


DESIGN = {"RobotMount": d_robotmount, "FacetBack": d_facetback, "FacetFront": d_facetfront,
          "TailStrut": d_tailstrut, "EncoderCarrier": d_encoder_carrier, "EncoderCableClamp": d_encoder_clamp,
          "Wheel": d_wheel, **{p: d_link(p) for p in SIDES}}


# ============================================================ preview (offline)
def _rgb255(rgb):
    return tuple(float(x) * 255 for x in rgb)


def _meshes(items, frame, dev=0.04):
    _aes()
    from render3d import tessellate
    out = []
    for solid, rgb in items:
        if solid is None:
            continue
        V, T, _ = tessellate(solid, dev)
        if len(T):
            out.append((frame.to(V), T, _rgb255(rgb)))
    return out


def board(P, ops, state, path):
    _aes()
    from render_color import view, sheet
    before = [(b["solid"], b["rgb"]) for b in P.bodies]
    after = [(s["solid"], s["rgb"]) for s in state]
    tiles = []
    frames = {}
    for op in ops:
        frames.setdefault(op["frame"].name, (op["frame"], []))[1].append(op["region"])
    for fname, (F, regs) in frames.items():
        from shapely.ops import unary_union
        u0, v0, u1, v1 = unary_union(regs).bounds
        m = 10.0
        crop = np.array([[u, v, w] for u in (u0 - m, u1 + m) for v in (v0 - m, v1 + m) for w in (-2.0, 2.0)])
        for label, items in (("before", before), ("after", after)):
            ms = _meshes(items, F)
            # only what is near the crop: big parts render in seconds, not minutes
            keep = []
            for V, T, col in ms:
                c = V[T].mean(1)
                sel = (c[:, 0] > u0 - 3 * m) & (c[:, 0] < u1 + 3 * m) & (c[:, 1] > v0 - 3 * m) & (c[:, 1] < v1 + 3 * m)
                if sel.any():
                    keep.append((V, T[sel], col))
            tiles.append((f"{fname}: {label}", "face-on", view(keep, crop, (760, 520), 89.5, -90, light="camera")))
            tiles.append((f"{fname}: {label}", "raking", view(keep, crop, (760, 520), 38, -70, light=(0.3, -0.6, 0.75))))
    sheet(tiles, 4, path, header=f"{P.name} -- relief (pressed colour)", size=(760, 520))


def cmd_preview(which):
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for part in which:
        if part not in DESIGN:
            print(f"{part}: no design yet")
            continue
        print(f"\n=== {part}", flush=True)
        P = Part(part)
        ops, skipped = DESIGN[part](P)
        for k, a, why in skipped:
            print(f"  left flush: {k} {a} mm2 ({why})")
        state, cuts, log = apply(P, ops)
        ck = checks(P, ops, state)
        for row in log:
            print(f"  {row}")
        for name, ok, text in ck:
            print(f"  {'OK  ' if ok else 'FAIL'} {name:16s} {text}")
        # outputs: one cutter STEP per original body it changes, one STEP of the new bodies
        _aes()
        import stepcolor
        # stepcolor drops bodies under 1 mm3 as "debris" -- here a 0.49 mm3 floor is a real body
        # (EncoderCarrier 2026-10-09: three floors never reached SolidWorks, left 0.4 mm voids)
        stepcolor.MIN_BODY_MM3 = 0.0
        rep = {"part": part, "ops": log, "checks": [dict(name=n, ok=o, text=t) for n, o, t in ck],
               "changed": [], "new": []}
        for i, tl in cuts.items():
            if not tl:
                continue
            tool = _fuse(tl)
            fin = next(s for s in state if s["orig"] == i)["solid"]
            f = os.path.join(OUT, f"{part}_tool{i}.step")
            stepcolor.write([(f"tool{i}", tool, (128, 128, 128))], f, part=f"{part}_tool{i}", say=lambda *a: None)
            rep["changed"].append(dict(index=i, name=P.bodies[i]["name"], kind=P.bodies[i]["kind"],
                                       volume=P.bodies[i]["volume"], final=round(_vol(fin), 4),
                                       pieces=sorted(round(x.volume, 4) for x in (fin.solids() if fin else [])),
                                       gone=fin is None, tool=os.path.basename(f)))
        new = [s for s in state if s["orig"] is None and s["solid"] is not None]
        for s in new:
            rep["new"].append(dict(name=s["name"], kind=s["kind"], rgb=list(s["rgb"]),
                                   pieces=sorted(round(x.volume, 4) for x in s["solid"].solids())))
        if new:
            stepcolor.write([(s["name"], s["solid"], tuple(int(round(c * 255)) for c in s["rgb"])) for s in new],
                            os.path.join(OUT, f"{part}_new.step"), part=f"{part}_new", say=lambda *a: None)
        # the whole part after (renders, 3MF): every final body in its colour
        stepcolor.write([(f"b{k}", s["solid"], tuple(int(round(c * 255)) for c in s["rgb"]))
                         for k, s in enumerate(state) if s["solid"] is not None],
                        os.path.join(OUT, f"{part}_after.step"), part=f"{part}_after", say=lambda *a: None)
        rep["ok"] = all(o for _, o, _ in ck)
        with open(os.path.join(OUT, f"{part}_relief.json"), "w") as fh:
            json.dump(rep, fh, indent=1)
        v0 = sum(b["volume"] for b in P.bodies)
        v1 = sum(_vol(s["solid"]) for s in state)
        print(f"  {len(rep['changed'])} bodies cut ({sum(c['gone'] for c in rep['changed'])} gone), "
              f"{sum(len(n['pieces']) for n in rep['new'])} new; volume {v0:.1f} -> {v1:.1f} mm3 ({v1 - v0:+.1f})",
              flush=True)
        board(P, ops, state, os.path.join(OUT, f"{part}_relief.png"))
        summary[part] = rep["ok"]
    print("\nSUMMARY " + "  ".join(f"{k}: {'OK' if v else 'FAIL'}" for k, v in summary.items()))


# ============================================================ build: into the SolidWorks parts
def _delete_rl(doc):
    """Remove an earlier run's RL_* features (cutters, combines, deleted bodies, new bodies)."""
    from swlib import c
    import swstyle as S
    doc.ClearSelection2(True)
    k = 0
    for f in list(S._iter_features(doc)):
        if f.Name.startswith("RL_") and f.Select2(k > 0, 0):
            k += 1
    if k:
        doc.Extension.DeleteSelection2(c.swDelete_Absorbed | c.swDelete_Children)
        doc.ClearSelection2(True)
        doc.EditRebuild3()
    return k


def _import_bodies(sw, doc, step):
    """The solids of a STEP as body copies, through a temporary import closed unsaved."""
    import swstyle as S
    from swlib import wrap, sld
    res = sw.LoadFile4(os.path.normpath(step), "r", None, 0)
    tmp = wrap(res[0] if isinstance(res, tuple) else res, sld.IModelDoc2)
    if tmp is None:
        raise SystemExit(f"could not import {step}")
    out = [(S.volume(b), wrap(b.Copy(), sld.IBody2)) for b in S.bodies(tmp)]
    sw.CloseDoc(tmp.GetTitle())
    sw.ActivateDoc3(doc.GetTitle(), False, 0, 0)
    return out


def probe(part, step, n=40):
    """Point proof on the AS-BUILT part (a STEP exported from SolidWorks): inside every press
    region (0.12 in from its edge), a point half-way down the recess must be in NO body (it
    was cut) and a point 0.3 into the floor must be in one (the floor is there); every raised
    region must be solid half-way up.  Volumes alone cannot tell a missed cut from the two
    kernels disagreeing (FacetFront 2026-10-09: 3.5 mm3 of 57284, every point right).
    Returns [(op name, recess wrong, floor wrong)]."""
    from build123d import import_step, Vector
    from shapely.geometry import Point
    P = Part(part)
    ops, _ = DESIGN[part](P)
    built = import_step(step).solids()
    rng = np.random.default_rng(1)
    out = []
    for op in ops:
        F = op["frame"]
        reg = op["region"].buffer(-0.12, join_style=2)
        if reg.is_empty:
            continue
        x0, y0, x1, y1 = reg.bounds
        pts, tries = [], 0
        while len(pts) < n and tries < 50 * n:
            tries += 1
            q = (rng.uniform(x0, x1), rng.uniform(y0, y1))
            if reg.contains(Point(q)):
                pts.append(q)
        bad_a = bad_f = 0
        for u, v in pts:
            base = F.o + u * F.u + v * F.v
            if op["op"] == "press":
                pa, pf = base - 0.5 * op["d"] * F.w, base - (op["d"] + 0.3) * F.w
                bad_a += any(s.is_inside(Vector(*pa)) for s in built)
                bad_f += not any(s.is_inside(Vector(*pf)) for s in built)
            else:
                pa = base + 0.5 * op["h"] * F.w
                bad_a += not any(s.is_inside(Vector(*pa)) for s in built)
        out.append((op["name"], bad_a, bad_f))
    return out


def sw_build(which):
    """Each part's previewed relief into the v5 part, END of the tree: per changed body its
    cutter (Imported RL_toolN) Combine-subtracted (RL_cutN; the pieces get the body's colour
    back -- a Combine drops it, gotcha of 24's trench), a swallowed body deleted (RL_goneN),
    then the new bodies (RL_<name>, coloured).  Checked body for body by volume against the
    preview; nothing saved here (16 --chain after)."""
    import swlib
    import swstyle as S
    from swlib import c, wrap, sld
    sw, _ = swlib.connect()
    swlib.open_v5(sw)
    done = []
    for part in which:
        rep = json.load(open(os.path.join(OUT, f"{part}_relief.json")))
        if not rep["ok"]:
            raise SystemExit(f"{part}: the preview has failing checks -- fix them first")
        info = json.load(open(os.path.join(SRC, f"{part}.json")))
        path = os.path.join(swlib.V5, PARTS[part])
        d, err, warn = sw.OpenDoc6(path, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
        doc = wrap(d, sld.IModelDoc2)
        sw.ActivateDoc3(doc.GetTitle(), False, 0, 0)
        n_old = _delete_rl(doc)
        have = sorted(round(S.volume(b), 1) for b in S.bodies(doc))
        want = sorted(round(b["volume"], 1) for b in info["bodies"])
        if have != want:
            raise SystemExit(f"{part}: bodies differ from the export ({len(have)} vs {len(want)}) -- "
                             f"re-run `export` and `preview` first")
        pdoc = wrap(doc, sld.IPartDoc)
        expect = [b["volume"] for b in info["bodies"]]
        for k, ch in enumerate(rep["changed"], 1):
            tgt = min(S.bodies(doc), key=lambda b: abs(S.volume(b) - ch["volume"]))
            v_sw = S.volume(tgt)
            if abs(v_sw - ch["volume"]) > 1e-4 * ch["volume"] + 0.01:
                raise SystemExit(f"{part}: no body of {ch['volume']} mm3 to cut (nearest {v_sw:.4f})")
            # the pieces' colour: the body's own, else the export's (from its faces or the part
            # -- a Combine's new faces would otherwise show the part's default appearance)
            rgb = tgt.MaterialPropertyValues2
            rgb = list(rgb[:3]) if rgb is not None else \
                min(info["bodies"], key=lambda b: abs(b["volume"] - ch["volume"]))["rgb"]
            expect.remove(min(expect, key=lambda v: abs(v - ch["volume"])))
            if ch["gone"]:
                S.delete_bodies(doc, [tgt])
                S.last_feature(doc).Name = f"RL_gone{k}"
                continue
            names0 = {b.Name for b in S.bodies(doc)}
            tools = []
            for j, (v, body) in enumerate(_import_bodies(sw, doc, os.path.join(OUT, ch["tool"]))):
                f = wrap(pdoc.CreateFeatureFromBody3(body, False, c.swCreateFeatureBodyCheck), sld.IFeature)
                if f is None:
                    raise SystemExit(f"{part}: SolidWorks refused cutter {ch['tool']}")
                f.Name = f"RL_tool{k}_{j + 1}"
            tools = [b for b in S.bodies(doc) if b.Name not in names0]
            tgt = min((b for b in S.bodies(doc) if b.Name in names0), key=lambda b: abs(S.volume(b) - v_sw))
            others = {b.Name for b in S.bodies(doc)} - {tgt.Name}
            cf, _ = S.combine(doc, "cut", tgt, tools)
            cf.Name = f"RL_cut{k}"
            for b in S.bodies(doc):                    # the target's pieces: every body that is new
                if b.Name not in others and rgb is not None:
                    S.colour(b, rgb)
            expect += ch["pieces"]
        if rep["new"]:
            src = _import_bodies(sw, doc, os.path.join(OUT, f"{part}_new.step"))
            want_new = [(v, nb) for nb in rep["new"] for v in nb["pieces"]]
            if len(src) != len(want_new):
                raise SystemExit(f"{part}: {len(src)} bodies in {part}_new.step, the preview predicts "
                                 f"{len(want_new)} -- re-run `preview` (a writer dropped some?)")
            for i, (v, body) in enumerate(sorted(src, key=lambda x: -x[0])):
                vv, nb = min(want_new, key=lambda x: abs(x[0] - v))
                want_new.remove((vv, nb))
                names0 = {b.Name for b in S.bodies(doc)}
                f = wrap(pdoc.CreateFeatureFromBody3(body, False, c.swCreateFeatureBodyCheck), sld.IFeature)
                if f is None:
                    raise SystemExit(f"{part}: SolidWorks refused new body {nb['name']}")
                f.Name = f"RL_{nb['name']}_{i + 1}"
                new = [b for b in S.bodies(doc) if b.Name not in names0]
                if len(new) != 1:
                    raise SystemExit(f"{part}: {f.Name} made {len(new)} bodies")
                S.colour(new[0], nb["rgb"])
                # against the PREDICTION, not what was imported: a STEP round trip of a broken
                # solid reads differently in each kernel (the wheel's chevrons, 2026-10-09)
                if abs(S.volume(new[0]) - vv) > max(0.05, 1e-3 * vv):
                    raise SystemExit(f"{part}: {f.Name} came in at {S.volume(new[0]):.3f} mm3, the preview "
                                     f"has {vv:.3f} -- nothing saved")
                expect.append(vv)
        doc.EditRebuild3()
        # Combine debris (zero-volume slivers on shared faces, gotcha 21): ONLY bodies the
        # prediction does not have, and only true slivers.  (First version dropped every body
        # under 0.5 mm3 as GL_DropDebris does -- and took the encoder's 0.48 mm3 code-wheel
        # ticks, a wheel dot and two clamp chevron pieces with it: real colour bodies.)
        n_extra = len(S.bodies(doc)) - len(expect)
        if n_extra > 0:
            junk = sorted(S.bodies(doc), key=S.volume)[:n_extra]
            if all(S.volume(b) < 0.02 for b in junk):
                S.delete_bodies(doc, junk)
                S.last_feature(doc).Name = "RL_DropDebris"
        got = sorted(S.volume(b) for b in S.bodies(doc))
        want = sorted(expect)
        # per body: 0.15 mm3 or 250 ppm.  Parasolid and OCC disagree on the volume of the big
        # cut bodies (EncoderCableClamp's white +0.097 of 5895, FacetFront's +3.5 of 57284,
        # Coupler's +9.6 of 68229; every other body 0.0000) -- yet 3300 points sampled round and
        # through the Coupler's 11 grooves, edges included, agree everywhere: kernel noise, not
        # a missed cut.  The body count must match exactly, and `probe` below proves every
        # recess cut and every floor there, point by point
        worst = max(abs(a - b) / max(0.15, 2.5e-4 * b) for a, b in zip(got, want)) \
            if len(got) == len(want) else float("inf")
        asb = os.path.join(OUT, f"{part}_asbuilt.step")
        doc.Extension.SaveAs3(asb, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy, None, None, 0, 0)
        pr = probe(part, asb)
        bad_pr = [(nm, a, f) for nm, a, f in pr if a or f]
        print(f"{part:17s} probe: {len(pr)} ops, {sum(1 for _ in pr) * 40} points each way, "
              f"{'all right' if not bad_pr else 'WRONG ' + str(bad_pr)}", flush=True)
        # a body may carry its colour on its faces only (a STEP import: TailStrut, Facet*); every
        # cut piece got a body colour above, so there may be no more bare bodies than before
        bare = [b.Name for b in S.bodies(doc) if b.MaterialPropertyValues2 is None]
        was_bare = sum(1 for b in info["bodies"] if b["colour_from"] != "body")
        bare = bare if len(bare) > was_bare else []
        ok = worst < 1.0 and not bare and not bad_pr
        print(f"{part:17s} {len(rep['changed'])} bodies cut, {len(rep['new'])} new; "
              f"{'replaced ' + str(n_old) + ' old RL_ features; ' if n_old else ''}"
              f"{len(got)} bodies vs {len(want)} predicted, worst {worst:.3f} of tolerance"
              f"{', UNCOLOURED ' + str(bare) if bare else ''}  {'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            raise SystemExit(f"{part}: check failed -- nothing saved; `{part}` is left as built for a look "
                             f"(reload it from disk to undo)")
        done.append(PARTS[part])
    return sw, done


# ============================================================ print files (offline)
# the print's up direction, part-local (the face on the bed is the opposite one) -- as each
# part's own script printed it: 24 (box parts), 22 (hub face up), 26 (styled face up); links
# in their part frame (show face up, as 11 / 24)
PRINT_UP = {"FacetFront": (1, 0, 0), "FacetBack": (-1, 0, 0), "TailStrut": (0, -1, 0),
            "Wheel": (0, 0, -1), "EncoderCarrier": (0, -1, 0), "EncoderCableClamp": (0, 1, 0),
            "FacetHood": (0, 1, 0), "FacetRing": (0, 1, 0),           # skirt / flange down (box_facet_print)
            "BumperFront": (-1, 0, 0), "BumperBack": (1, 0, 0)}       # outer face down
BOX_NAMES = ("FacetFront", "FacetBack", "TailStrut", "FacetHood", "FacetRing", "BumperFront", "BumperBack")
TPU = ("BumperFront", "BumperBack")                                  # one filament, grey TPU
FILAMENT = {"white": 1, "graphite": 2, "blue": 3, "dark": 4}


def cmd_print(which):
    """Bambu project 3MFs of the parts AS BUILT: from `export` run after `build` (the STEP
    SolidWorks writes + each body's colour read from SolidWorks), refused unless that export
    carries the RL_ features: box parts / wheel / encoder parts in print orientation, the
    links in their part frame; filament 1 white / 2 graphite / 3 blue / 4 dark.  The
    TailStrut's graphite is drawn DARK and prints as filament 2 (24's rule).  Same file
    names as before, so the 3MFs he has are replaced."""
    _aes()
    import export3mf
    from render3d import tessellate
    names = {p: f"{p}_bambu.3mf" for p in BOX_NAMES}
    for part in which:
        info = json.load(open(os.path.join(SRC, f"{part}.json")))
        if part in PARTS and not info.get("relief_features"):
            print(f"{part}: the export has no RL_ features -- `build`, then `export {part}` first")
            continue
        try:
            bodies = [(b["solid"], b["kind"]) for b in Part(part, built=True).bodies]
            src = "SolidWorks export"
        except SystemExit as e:
            # SolidWorks' STEP of the wheel's raised chevrons reads 1.3 mm3 short in OCC while
            # SolidWorks holds them at exactly the predicted volume: the RL_ bodies are taken
            # from <Part>_new.step (the very bodies `build` imported, proven by volume), the
            # rest from the SolidWorks export -- each matched by volume, the count exact
            import stepcolor
            from build123d import Solid, import_step
            print(f"  {e} -> the RL_ bodies from {part}_new.step")
            sw_s = list(import_step(os.path.join(SRC, f"{part}.step")).solids())
            new_s = [(Solid(s), rgb) for s, rgb in stepcolor.read(os.path.join(OUT, f"{part}_new.step"))]
            bodies, ok_ = [], True
            for b in info["bodies"]:
                pool = new_s if b["name"].startswith("RL_") else sw_s
                vol = (lambda x: x[0].volume) if pool is new_s else (lambda x: x.volume)
                if not pool:
                    ok_ = False
                    break
                m = min(pool, key=lambda x: abs(vol(x) - b["volume"]))
                if abs(vol(m) - b["volume"]) > 1e-3 * max(1.0, b["volume"]) + 0.05:
                    ok_ = False
                    break
                pool.remove(m)
                bodies.append((m[0] if pool is new_s else m, kind(b["rgb"] or (1, 1, 1))))
            n_rl = sum(1 for b in info["bodies"] if b["name"].startswith("RL_"))
            if not ok_ or len(sw_s) != n_rl or new_s:    # left over: exactly the export's own RL_ copies
                print(f"{part}: could not account for every body -- not printed")
                continue
            src = "SolidWorks export + the imported RL_ bodies"
        up = PRINT_UP.get(part)
        if up is not None:
            u = unit(up)
            ax = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            e1 = unit(np.cross(ax, u))
            Rp = np.vstack([e1, np.cross(u, e1), u])
        else:
            Rp = np.eye(3)
        objs, by = [], {}
        for k, (solid, kd) in enumerate(bodies):
            f = 1 if part in TPU else (2 if (part == "TailStrut" and kd == "dark") else FILAMENT[kd])
            V, T, _ = tessellate(solid, 0.05)
            objs.append((f"{part}_{k}", np.asarray(V) @ Rp.T, np.asarray(T), f))
            by[f] = by.get(f, 0) + 1
        name = names.get(part, f"{part}_glacier_native.3mf")
        out = os.path.join(HERE, "out", "print", name)
        export3mf.write_3mf(objs, out, name=f"{part} (relief)")
        allV = np.vstack([o[1] for o in objs])
        ext = allV.max(0) - allV.min(0)
        print(f"{part:17s} {len(objs)} bodies ({src}), filaments {dict(sorted(by.items()))}, "
              f"{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm -> out/print/{name}", flush=True)


# ============================================================ the whole robot, before / after (offline)
VIEWS = [   # (title, elev, azim, crop in ROBOT mm (x0, x1, y0, y1, z0, z1) or None) -- his four screenshots
    ("right side: RobotMount field, Side panel (his image 1)", 12, -78, (-160, 60, 10, 125, -10, 120)),
    ("back: FacetBack (his image 2)", 14, 180, (-200, -100, 0, 125, -95, 95)),
    ("front-top: the hood's skirt + tier 1 (his image 3)", 30, -38, (-175, 80, 60, 122, -95, 95)),
    ("front, whole robot: the link sides (his image 4)", 4, 0, None),
    ("rear-right 3/4, whole robot", 18, -140, None),
]
LEG_ROOTS = ("Femur-1", "FEMUR_INSIDE-1", "COUPLER-1", "Tibia-1", "BODY-1")


def _hood_meshes(relief):
    """The FacetHood from its source (hood_variants B, or B0 = as it was), with its two field
    hoses; the tier-1 hose from 24's pipe STEP."""
    _aes()
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "aesthetics", "parts")))
    import hood_variants as HV
    from render3d import tessellate
    from build123d import import_step
    hb = HV.variant_B(relief=relief)
    bodies, _ = hb.bodies()
    cols = {"white": HV.WHITE, "graphite": HV.GRAPHITE, "blue": HV.BLUE, "dark": HV.DARK}
    out = []
    for k, b in bodies.items():
        if b is not None and b.solids():
            V, T, _ = tessellate(b, 0.08)
            out.append((V, T, np.array(cols[k], float)))
    pipe = os.path.join(HERE, "out", "pipes", "FacetHood_pipes.step")
    for s in import_step(pipe).solids():
        if s.bounding_box().max.Y < 100:
            V, T, _ = tessellate(s, 0.05)
            out.append((V, T, np.array([0x1E, 0x7B, 0xFF], float) if s.volume > 60 else np.array([0x7E, 0x87, 0x95], float)))
    return out


def cmd_robot(hip=20.0, only=None):
    """The robot at one hip angle, every view before | after -> out/relief/robot_relief_<n>.png.
    Parts in DESIGN from their export (before) and <Part>_after.step (after); the hood from
    hood_variants; the other styled parts from 24's exports + pipes; the rest grey.  The left
    leg and left body side are the right ones mirrored (they are exact mirrors since 27)."""
    _aes()
    import importlib
    import stepcolor
    from build123d import Solid
    from render3d import tessellate
    import render_color as RCOL
    P24 = importlib.import_module("24_pipes")
    pj = json.load(open(os.path.join(P24.SWEEP, "poses.json")))
    k = int(np.argmin(np.abs(np.array(pj["hips"]) - hip)))

    def tess_step(path):
        out = []
        for s, rgb in stepcolor.read(path):
            V, T, _ = tessellate(Solid(s), 0.05)
            out.append((V, T, np.array(rgb or (0.9, 0.9, 0.9)) * 255))
        return out

    base = {}                                        # instance -> meshes, same before and after
    for part in P24.PARTS:
        if part in DESIGN or part == "FacetHood":
            continue
        info = json.load(open(os.path.join(P24.SRC, f"{part}.json")))
        P = P24.Part(part)
        ms = [(V, T, np.array(rgb) * 255) for V, T, rgb in P.mesh()]
        pipe = os.path.join(P24.OUT, f"{part}_pipes.step")
        if os.path.exists(pipe):
            from build123d import import_step
            for s in import_step(pipe).solids():
                V, T, _ = tessellate(s, 0.05)
                ms.append((V, T, np.array(P24.BLUE if s.volume > 60 else P24.GRAPHITE) * 255))
        base[info["instances"][0]["name"]] = ms
    before, after = dict(base), dict(base)
    for part in DESIGN:
        info = json.load(open(os.path.join(SRC, f"{part}.json")))
        a = os.path.join(OUT, f"{part}_after.step")
        if not os.path.exists(a):
            continue
        b = [(V, T, np.array(bb["rgb"]) * 255) for bb in Part(part).bodies
             for V, T, _ in [tessellate(bb["solid"], 0.05)]]
        af = tess_step(a)
        for inst in info["instances"]:
            before[inst["name"]], after[inst["name"]] = b, af
    hood_name = "Box-1/FacetHood-1"
    if only is None or any(VIEWS[i - 1][3] is None or VIEWS[i - 1][3][3] > 75 for i in only):
        before[hood_name], after[hood_name] = _hood_meshes(False), _hood_meshes(True)
    print("  meshes ready", flush=True)
    F = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)      # ROBOT -> render frame (z up)
    S = np.diag([1.0, 1.0, -1.0])                                  # the robot's mid-plane mirror

    def scene(own):
        cache, out = {}, []
        for name, key in pj["mesh"].items():
            M = np.array(pj["placements"][name][k])
            if name in own:
                ms = own[name]
            else:
                if key not in cache:
                    z = np.load(os.path.join(P24.SWEEP, key + ".npz"))
                    cache[key] = [(z["V"], z["T"], np.array([150, 155, 162.0]))]
                ms = cache[key]
            for V, T, col in ms:
                W = V @ M[:, :3].T + M[:, 3]
                out.append((W, T, col))
                if name.split("/")[0] in LEG_ROOTS:
                    out.append((W @ S, T[:, ::-1], col))
        return out

    scenes = {"before": scene(before), "after": scene(after)}
    for i, (title, el, az, crop) in enumerate(VIEWS):
        if only is not None and i + 1 not in only:
            continue
        tiles = []
        for lab in ("before", "after"):
            sc = scenes[lab]
            if crop is not None:
                x0, x1, y0, y1, z0, z1 = crop
                lo, hi = np.array([x0, y0, z0]), np.array([x1, y1, z1])
                keep = []
                for W, T, col in sc:
                    c = W[T].mean(1)
                    m = np.all((c > lo - 40) & (c < hi + 40), axis=1)
                    if m.any():
                        keep.append((W @ F.T, T[m], col))
                corners = np.array([[a, b, c] for a in (x0, x1) for b in (y0, y1) for c in (z0, z1)]) @ F.T
            else:
                keep = [(W @ F.T, T, col) for W, T, col in sc]
                corners = np.vstack([b[0] for b in keep])
            # cull, vectorised, what view() would drop one triangle at a time in Python: the
            # back faces and (whole-robot views) the sub-pixel triangles -- minutes -> seconds
            e_, a_ = np.radians(el), np.radians(az)
            fwd = np.array([np.cos(e_) * np.cos(a_), np.cos(e_) * np.sin(a_), np.sin(e_)])
            amin = 0.0 if crop is not None else 0.02
            culled = []
            for V, T, col in keep:
                if not len(T):
                    continue
                nn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
                m = (nn @ fwd > 0) & (np.linalg.norm(nn, axis=1) > 2 * amin)
                if m.any():
                    culled.append((V, T[m], col))
            tiles.append((lab, "", RCOL.view(culled, corners, (1100, 820), el, az)))
            print(f"  view {i + 1} {lab}", flush=True)
        RCOL.sheet(tiles, 2, os.path.join(OUT, f"robot_relief_{i + 1}.png"), header=title,
                   sub=f"offline render, hip {pj['hips'][k]:+.0f}; left side = the right one mirrored",
                   size=(1100, 820))


def cmd_uvmap(part, n, d, u=None, step=10.0):
    """A face-on render of one plane of the part with a (u, v) grid in mm: the sheet to
    design a face on.  -> out/relief/maps/<Part>_<n>_<d>.png"""
    _aes()
    from render_color import view
    from PIL import ImageDraw
    P = Part(part)
    F = Frame(P, n, d, u=u)
    fp = F.footprint()
    u0, v0, u1, v1 = fp.bounds
    m = 4.0
    crop = np.array([[a, b, w] for a in (u0 - m, u1 + m) for b in (v0 - m, v1 + m) for w in (-2.0, 2.0)])
    ms = _meshes([(b["solid"], b["rgb"]) for b in P.bodies], F)
    W = 1600
    H = int(W * (v1 - v0 + 2 * m) / (u1 - u0 + 2 * m) * 1.04) + 40
    H = max(300, min(H, 1400))
    img = view(ms, crop, (W, H), 89.5, -90, light="camera")
    # the same mapping as view(): right = u, up ~ v
    e = np.radians(89.5)
    up = np.array([0, np.sin(e), np.cos(e)])
    Pc = np.stack([crop[:, 0], crop @ up], 1)
    mn, mx = Pc.min(0), Pc.max(0)
    ctr, ext = (mn + mx) / 2, mx - mn
    s = min(W / (ext[0] * 1.06), H / (ext[1] * 1.10))

    def px(a, b):
        return ((a - ctr[0]) * s + W / 2, H - ((b * up[1] - ctr[1]) * s + H / 2))
    dr = ImageDraw.Draw(img)
    for a in np.arange(np.floor(u0 / step) * step, u1 + m, step):
        dr.line([px(a, v0 - m), px(a, v1 + m)], fill=(255, 80, 80) if abs(a) < 1e-6 else (230, 120, 60), width=1)
        dr.text(px(a + 0.4, v0 - m + 1.5), f"{a:.0f}", fill=(255, 140, 60))
    for b in np.arange(np.floor(v0 / step) * step, v1 + m, step):
        dr.line([px(u0 - m, b), px(u1 + m, b)], fill=(255, 80, 80) if abs(b) < 1e-6 else (230, 120, 60), width=1)
        dr.text(px(u0 - m + 0.5, b + 2.5), f"{b:.0f}", fill=(255, 140, 60))
    os.makedirs(os.path.join(OUT, "maps"), exist_ok=True)
    out = os.path.join(OUT, "maps", f"{part}_{'_'.join(f'{x:+.2f}' for x in F.w)}_{F.d:.1f}.png")
    img.save(out)
    print(f"{part}: plane w {np.round(F.w, 3)} d {F.d:.2f}, u {np.round(F.u, 3)}, v {np.round(F.v, 3)}; "
          f"face u {u0:.1f}..{u1:.1f} v {v0:.1f}..{v1:.1f} -> {out}")


def cmd_faces(which):
    for part in which:
        P = Part(part)
        print(f"\n=== {part}: {len(P.bodies)} bodies "
              + ", ".join(f"{k} {sum(1 for b in P.bodies if b['kind'] == k)}" for k in KINDS))
        for n, d, areas, fs in P.planes()[:40]:
            pts = np.array([[v.X, v.Y, v.Z] for f, _ in fs for v in f.vertices()])
            lo, hi = pts.min(0), pts.max(0)
            print(f"  n ({n[0]:+.3f} {n[1]:+.3f} {n[2]:+.3f}) d {d:8.2f}  "
                  + " ".join(f"{k} {a:7.1f}" for k, a in areas.items())
                  + f"   box {np.round(lo, 1).tolist()}..{np.round(hi, 1).tolist()}")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    args = sys.argv[1:]
    cmd = args.pop(0) if args else "preview"
    if cmd == "uvmap":       # uvmap <Part> nx ny nz d [ux uy uz]
        nums = [float(a) for a in args[1:]]
        cmd_uvmap(args[0], nums[0:3], nums[3], nums[4:7] if len(nums) >= 7 else None)
        raise SystemExit
    if cmd == "robot":       # robot [hip] [--views=4,5]
        vs = [a for a in args if a.startswith("--views=")]
        nums = [a for a in args if not a.startswith("--")]
        cmd_robot(float(nums[0]) if nums else 20.0,
                  [int(x) for x in vs[0].split("=")[1].split(",")] if vs else None)
        raise SystemExit
    which = [a for a in args if not a.startswith("--")] or list(ALL_PARTS if "--all" in args else PARTS)
    known = ALL_PARTS if cmd in ("export", "print") else PARTS
    bad = [p for p in which if p not in known]
    if bad:
        raise SystemExit(f"unknown part(s) {bad}; known: {list(known)}")
    if cmd == "export":
        export(which)
    elif cmd == "faces":
        cmd_faces(which)
    elif cmd == "preview":
        cmd_preview(which)
    elif cmd == "build":
        from importlib import import_module
        sw, done = sw_build(which)
        import_module("16_check_and_save").chain(sw, done)
    elif cmd == "print":
        cmd_print(which)
    elif cmd == "verify":    # READ-ONLY: export each part as it is in SolidWorks, run `probe`
        import swlib
        from swlib import c, wrap, sld
        sw, _ = swlib.connect()
        swlib.open_v5(sw)
        for part in which:
            d = wrap(sw.OpenDoc6(os.path.join(swlib.V5, PARTS[part]), c.swDocPART, c.swOpenDocOptions_Silent,
                                 "", 0, 0)[0], sld.IModelDoc2)
            asb = os.path.join(OUT, f"{part}_asbuilt.step")
            d.Extension.SaveAs3(asb, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy, None, None, 0, 0)
            pr = probe(part, asb)
            bad = [(nm, a, f) for nm, a, f in pr if a or f]
            print(f"{part:17s} {len(pr)} ops x 40 points: {'all right' if not bad else 'WRONG ' + str(bad)}"
                  f"{'  (unsaved changes in memory)' if d.GetSaveFlag() else ''}", flush=True)
    else:
        raise SystemExit(f"unknown command {cmd}")
