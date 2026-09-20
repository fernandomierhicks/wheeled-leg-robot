r"""Where a part is NOT allowed to grow, derived from the assembly itself.

    C:/Users/ferna/cadenv/Scripts/python.exe lib/asmkeepout.py [Part ...]

Styling grows a part's silhouette, and the v4 leg has very little room: the
three links clear each other IN PLAN only, inside lateral bands they share
(Coupler 111-146, Femur 123-162, Tibia 123-150 global Z).  Left unchecked the
GLACIER outline growth drove 13 cm3 of the Femur into the Tibia.

So instead of growing and then checking, the recipe is told up front where the
metal cannot go.  For each part, in each pose, every other component is
transformed into THAT PART's local frame, clipped to the z band the styled part
actually occupies, projected into plan, and unioned.  The result is buffered by
the clearance and cached as WKT; `plan()` subtracts it from the grown outline.

Two decisions worth knowing:

* **Neighbours are taken STYLED where a styled export exists**, otherwise from
  source.  A neighbour that is itself being styled will grow too, so clipping
  against its source would leave both parts free to meet in the middle.  The
  styled exports on disk are the UNCLIPPED, maximal versions, and clipping
  cannot make a part bigger -- so clipping against those is conservative and
  converges in a single pass rather than needing iteration.

* **The z band is the source span padded by `PAD_Z`**, not the styled span,
  because the styled part does not exist yet when this runs.  Padding is
  conservative: it over-blocks slightly rather than under-blocking.

Subtracting the keep-out can never cut the source solid -- `plan()` unions the
original silhouette back in afterwards -- so `verify.py` is unaffected.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from build123d import import_step
from shapely.geometry import Polygon as ShPoly, box as shbox
from shapely.ops import unary_union
from shapely import wkt as shwkt
import numpy as np
from render3d import tessellate
from collide import leaves, POSES

CACHE = os.path.join(paths.INPUT, "keepout")
PAD_Z = 6.0        # mm; styled relief stands proud of the source by ~3.5
TESS = 1.2         # mm; plan footprints do not need a fine mesh
MIN_A = 0.5        # mm2; drop slivers


def _neighbour(name, shp):
    """The styled export for `name` if there is one, else the assembly's copy.

    A neighbour that is itself being styled will GROW, so clipping against its
    source would leave both parts free to meet in the middle.  The styled
    exports on disk are the unclipped, maximal versions and clipping only ever
    shrinks a part, so clipping against those converges in one pass.
    """
    import json
    sf = os.path.join(paths.SPECS, name.lower() + ".json")
    if os.path.exists(sf):
        tag = json.load(open(sf)).get("tag", "arctic")
        q = paths.styled_step(name, tag)
        if os.path.exists(q):
            return import_step(q)
    return shp


def _matrix(loc):
    """4x3 affine of a build123d Location, as numpy."""
    t = loc.wrapped.Transformation()
    return np.array([[t.Value(r + 1, c + 1) for c in range(4)] for r in range(3)])


def _tess_global(shape, loc):
    """Tessellate ONCE, in the assembly's frame.  Vertices are cheap to move
    into any part's local frame afterwards; re-tessellating per part is not --
    doing that made this pass take minutes on the AK45 stator alone."""
    V, T, _ = tessellate(shape.moved(loc), TESS)
    return np.asarray(V, dtype=float), T


def _footprint(V, T, zlo, zhi):
    """Plan outline of the triangles lying in [zlo, zhi].  V is already local.

    Tessellations of real CAD carry slivers, and unioning thousands of
    exactly-touching triangles makes GEOS throw "found non-noded intersection".
    Coordinates are snapped and degenerate triangles dropped to avoid it, and if
    the union still fails the caller falls back to the CONVEX HULL -- this
    output gates collisions, so the only acceptable failure is to over-block.
    """
    if len(T) == 0:
        return None
    idx = np.asarray(T, dtype=int)
    z = V[:, 2][idx]
    keep = (z.max(axis=1) >= zlo) & (z.min(axis=1) <= zhi)
    if not keep.any():
        return None
    xy = np.round(V[:, :2][idx[keep]], 3)
    # 2x signed area; a sliver below this cannot be noded reliably
    cross = ((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) -
             (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
    xy = xy[np.abs(cross) > 2e-3]
    if not len(xy):
        return None
    polys = [ShPoly(t) for t in xy]
    polys = [q for q in polys if q.is_valid]
    if not polys:
        return None
    for attempt in (polys, [q.buffer(0.01, join_style=2) for q in polys]):
        try:
            u = unary_union(attempt).buffer(0)
            if not u.is_empty:
                return u
        except Exception:
            continue
    # last resort: never return None here, that would UNDER-block
    try:
        return unary_union([q.convex_hull for q in polys]).convex_hull
    except Exception:
        return None


def _bands(bb, n):
    """`n` z bands spanning the part, each padded by PAD_Z so a neighbour just
    outside a band still blocks growth that would reach it."""
    z0, z1 = bb.min.Z, bb.max.Z
    step = (z1 - z0) / n
    return [(z0 + i * step - PAD_Z, z0 + (i + 1) * step + PAD_Z) for i in range(n)]


def compute_all(parts, poses=None, verbose=True, layers=1):
    """Forbidden plan region per part, unioned over every pose.

    Each pose STEP is read ONCE and each neighbour tessellated ONCE, then reused
    for every part -- the assembly is 68 MB and reading it is ~20 s, so doing it
    per part does not scale past a couple of parts.
    """
    srcs = {p: import_step(paths.part_step(p)).solids()[0] for p in parts}
    bbs = {p: srcs[p].bounding_box() for p in parts}
    blocked = {p: [[] for _ in range(layers)] for p in parts}
    for pose in (poses or POSES):
        step = os.path.join(paths.EXPORTS, pose + ".STEP")
        if not os.path.exists(step):
            continue
        if verbose:
            print(f"  reading {pose} ...", flush=True)
        inst = leaves(step)
        locs = {n: l for n, s, l in inst if n in parts}
        resolved, tess = {}, {}
        for part in parts:
            if part not in locs:
                if verbose:
                    print(f"    {part}: not in this assembly")
                continue
            LP = locs[part]
            M = _matrix(LP.inverse())
            bb = bbs[part]
            bands = _bands(bb, layers)
            gb = srcs[part].moved(LP).bounding_box()
            n_hit = 0
            for name, shp, loc in inst:
                if name == part:
                    continue
                if name not in resolved:
                    resolved[name] = _neighbour(name, shp)
                ob = resolved[name].moved(loc).bounding_box()
                if (ob.max.X < gb.min.X - 40 or ob.min.X > gb.max.X + 40 or
                    ob.max.Y < gb.min.Y - 40 or ob.min.Y > gb.max.Y + 40 or
                    ob.max.Z < gb.min.Z - 40 or ob.min.Z > gb.max.Z + 40):
                    continue
                key = (name, id(loc.wrapped))
                if key not in tess:
                    tess[key] = _tess_global(resolved[name], loc)
                Vg, T = tess[key]
                V = Vg @ M[:, :3].T + M[:, 3]
                got = False
                for li, (zlo, zhi) in enumerate(bands):
                    try:
                        f = _footprint(V, T, zlo, zhi)
                    except Exception as e:
                        # over-block rather than skip: a missing keep-out shows
                        # up later as a collision in a printed part
                        print(f"      {name}: footprint failed ({type(e).__name__}); "
                              f"falling back to its bounding box", flush=True)
                        sel = (V[:, 2] >= zlo) & (V[:, 2] <= zhi)
                        P2 = V[sel][:, :2] if sel.any() else V[:, :2]
                        f = shbox(P2[:, 0].min(), P2[:, 1].min(),
                                  P2[:, 0].max(), P2[:, 1].max())
                    if f is not None:
                        blocked[part][li].append(f)
                        got = True
                n_hit += 1 if got else 0
            if verbose:
                print(f"    {part}: {n_hit} neighbours in the z band", flush=True)
    out = {}
    for part in parts:
        per = []
        for li in range(layers):
            if not blocked[part][li]:
                per.append(None)
                continue
            u = unary_union(blocked[part][li]).buffer(0)
            gs = u.geoms if u.geom_type == "MultiPolygon" else [u]
            per.append(unary_union([g for g in gs if g.area > MIN_A]))
        out[part] = per
    return out


def save(part, polys):
    """One .wkt per z layer.  A single polygon is written as layer 0, so the
    one-layer cache and the layered cache share a format."""
    os.makedirs(CACHE, exist_ok=True)
    if not isinstance(polys, (list, tuple)):
        polys = [polys]
    for old in os.listdir(CACHE):
        if old.startswith(part + ".L") or old == part + ".wkt":
            os.remove(os.path.join(CACHE, old))
    out = []
    for i, g in enumerate(polys):
        q = os.path.join(CACHE, f"{part}.L{i}.wkt")
        open(q, "w").write(g.wkt if g is not None else "POLYGON EMPTY")
        out.append(q)
    return out


def n_layers(part):
    if not os.path.isdir(CACHE):
        return 0
    return len([f for f in os.listdir(CACHE)
                if f.startswith(part + ".L") and f.endswith(".wkt")])


def load(part, clearance=1.0, layer=None):
    """Cached keep-out for one layer, buffered by `clearance`.

    `layer=None` means the union of every layer -- the old single-envelope
    behaviour, and the conservative one: material blocked at ANY depth blocks a
    full-thickness prism.
    """
    n = n_layers(part)
    if n == 0:
        q = os.path.join(CACHE, part + ".wkt")          # pre-layer cache
        if not os.path.exists(q):
            return None
        g = shwkt.loads(open(q).read())
        return None if g.is_empty else g.buffer(float(clearance), join_style=2)
    idx = range(n) if layer is None else [layer]
    gs = []
    for i in idx:
        q = os.path.join(CACHE, f"{part}.L{i}.wkt")
        if not os.path.exists(q):
            continue
        g = shwkt.loads(open(q).read())
        if not g.is_empty:
            gs.append(g)
    if not gs:
        return None
    return unary_union(gs).buffer(float(clearance), join_style=2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*", default=["Femur", "Tibia", "Coupler"])
    ap.add_argument("--layers", type=int, default=1)
    a = ap.parse_args()
    want = a.parts or ["Femur", "Tibia", "Coupler"]
    res = compute_all(want, layers=a.layers)
    for part in want:
        per = res.get(part) or []
        save(part, per)
        areas = ", ".join("--" if g is None else f"{g.area:.0f}" for g in per)
        print(f"{part:<10} {a.layers} layer(s), blocked area by layer: {areas} mm2")
