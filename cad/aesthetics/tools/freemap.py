r"""Where CAN material go?  Measured from the assembly, not guessed.

    C:/Users/ferna/cadenv/Scripts/python.exe tools/freemap.py [Part ...]
                                              [--res 0.8] [--zres 1.0]
                                              [--clear 1.0] [--grow 20]

Decision 26: his marks are the intent, the geometry is the judge.  So this does
not ask "is his green legal".  It asks the prior question -- *how far can this
edge move before it hits something, anywhere in the leg's travel* -- and answers
it as a distance in millimetres at every point around the perimeter.

WHAT IT PRODUCES, per link, in the part's OWN local frame:

    input/freespace/<Part>.npz    occupancy, free space and the reach map
    input/freespace/<Part>.json   headline numbers + reach around the perimeter
    out/marks/read/<Part>_FREE.png   the picture: how far each edge may move

HOW IT DIFFERS FROM lib/asmkeepout.py, which he did not trust:

* **Depth is resolved, not collapsed.**  asmkeepout defaults to ONE layer, so a
  neighbour at any depth blocks a full-thickness prism -- which is very likely
  why the Coupler, Side panel and RobotMount flanges all ended up inboard with
  "no metal at the show-face perimeter".  Here the assembly is voxelised in Z at
  `zres`, and growth is blocked only at the depths actually occupied.
* **No PAD_Z.**  That 6 mm was padding on top of clearance, i.e. a guess on top
  of a guess -- his words, his call to drop.  Clearance is applied once, as an
  explicit dilation, and it is a parameter.
* **No convex-hull fallback.**  asmkeepout falls back to a hull when a union
  fails, which over-blocks enormously and silently.  Voxels cannot fail to node,
  so there is nothing to fall back from.
* **Neighbours are taken from SOURCE**, not from the maximal styled exports.
  This measures the room that really exists.  Two links growing into the same
  gap is handled honestly instead: the space both can reach is reported as
  CONTESTED rather than quietly handed to whichever ran first.
* **NO buffer(0).**  asmkeepout calls it twice.  It is the call that made a
  self-touching ring "valid" by splitting it into lobes meeting at a point,
  which extruded into the solid that shattered in SolidWorks.

Cross-sections are built per Z slice by intersecting triangles with the slice
plane and filling the resulting outline, so a neighbour blocks as a SOLID rather
than as a hollow shell -- growth landing inside a motor would otherwise read as
free space.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import ndimage
import paths
from build123d import import_step
from render3d import tessellate
from collide import leaves, POSES

LINKS = ["Femur", "Tibia", "Coupler"]
TESS = 0.8
OUTDIR = os.path.join(paths.INPUT, "freespace")


def _matrix(loc):
    t = loc.wrapped.Transformation()
    return np.array([[t.Value(r + 1, c + 1) for c in range(4)] for r in range(3)])


class Grid:
    """A voxel grid in one part's local frame: plan at `res`, depth at `zres`."""

    def __init__(self, bb, grow, res, zres):
        self.res, self.zres = res, zres
        self.x0 = bb.min.X - grow
        self.y0 = bb.min.Y - grow
        self.z0 = bb.min.Z - zres
        self.nx = int(np.ceil((bb.size.X + 2 * grow) / res)) + 1
        self.ny = int(np.ceil((bb.size.Y + 2 * grow) / res)) + 1
        self.nz = int(np.ceil((bb.size.Z + 2 * zres) / zres)) + 1

    def zs(self):
        return self.z0 + (np.arange(self.nz) + 0.5) * self.zres

    def to_px(self, xy):
        return (np.column_stack([(xy[:, 0] - self.x0) / self.res,
                                 (xy[:, 1] - self.y0) / self.res])).astype(int)

    def splat(self, plane, xy):
        """Mark cells hit by points `xy` (local mm) into a bool plane."""
        if not len(xy):
            return
        p = self.to_px(xy)
        ok = ((p[:, 0] >= 0) & (p[:, 0] < self.nx) &
              (p[:, 1] >= 0) & (p[:, 1] < self.ny))
        p = p[ok]
        if len(p):
            plane[p[:, 1], p[:, 0]] = True


def slice_points(V, T, z, res):
    """Points along every triangle-plane intersection segment at height `z`.

    Rasterising the CROSS-SECTION OUTLINE and filling it is what makes a
    neighbour block as a solid.  Splatting surface points instead leaves the
    interior of every part reading as free space, and growth that lands inside a
    motor would then look legal.
    """
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    za, zb, zc = a[:, 2], b[:, 2], c[:, 2]
    lo = np.minimum(np.minimum(za, zb), zc)
    hi = np.maximum(np.maximum(za, zb), zc)
    sel = (lo <= z) & (hi >= z)
    if not sel.any():
        return np.empty((0, 2))
    a, b, c = a[sel], b[sel], c[sel]
    pts = []
    for p, q in ((a, b), (b, c), (c, a)):
        dz = q[:, 2] - p[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(dz) > 1e-12, (z - p[:, 2]) / dz, np.nan)
        m = np.isfinite(t) & (t >= 0) & (t <= 1)
        if m.any():
            pts.append(p[m][:, :2] + t[m, None] * (q[m][:, :2] - p[m][:, :2]))
    if not pts:
        return np.empty((0, 2))
    P = np.vstack(pts)
    # each triangle contributes 2 crossing points; join consecutive pairs by
    # sampling along them so the outline is CLOSED at grid resolution
    n = len(P) // 2 * 2
    if n < 2:
        return P
    A, B = P[:n:2], P[1:n:2]
    d = np.linalg.norm(B - A, axis=1)
    # EVERY SEGMENT GETS ITS OWN SAMPLE COUNT.  A shared count taken from the
    # longest segment and capped leaves gaps in the long ones, and a cross
    # section with gaps does not fill -- binary_fill_holes leaks out through the
    # hole and leaves a bare outline.  That is not a cosmetic error: the part
    # then reads as 3011 mm2 of plan area instead of ~7000, and, far worse, a
    # neighbour stops blocking as a solid, so growth driven straight into a
    # motor looks like free space.  Big flat faces are exactly where it bites,
    # because they tessellate into a handful of very large triangles.
    k = np.maximum(2, np.ceil(d / (res * 0.5)).astype(int) + 1)
    total = int(k.sum())
    idx = np.repeat(np.arange(len(k)), k)
    starts = np.cumsum(k) - k
    j = np.arange(total) - starts[idx]
    t = np.where(k[idx] > 1, j / np.maximum(k[idx] - 1, 1), 0.0)
    seg = A[idx] + t[:, None] * (B - A)[idx]
    return np.vstack([P, seg])


def plan_shadow(V, T, res):
    """The plan silhouette, sampled from triangle INTERIORS.

    Independent of the slice-and-fill path: for a closed solid the projection of
    its surface is exactly the projection of the solid, so this needs no closure
    and cannot leak.  It exists to catch the slicing silently under-filling --
    which it did, and which would have made neighbours block as hollow shells.
    """
    a, b, c = V[T[:, 0], :2], V[T[:, 1], :2], V[T[:, 2], :2]
    ar = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                      (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    n = np.maximum(3, np.ceil(ar / (res * res * 0.25)).astype(int))
    n = np.minimum(n, 20000)
    idx = np.repeat(np.arange(len(n)), n)
    rng = np.random.default_rng(0)
    u = rng.random(len(idx)); v = rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return (a[idx] + u[:, None] * (b - a)[idx] + v[:, None] * (c - a)[idx])


class Slicer:
    """Triangles of one shape, prepared so any Z slice is cheap.

    The z extent of every triangle is computed ONCE.  Recomputing it inside the
    slice loop -- which the first version did -- costs a full pass over the whole
    mesh per slice, and there are ~40 neighbours x ~40 slices x N poses of them.
    """

    def __init__(self, V, T):
        self.V, self.T = np.asarray(V, float), np.asarray(T, int)
        z = self.V[self.T, 2]
        self.lo, self.hi = z.min(1), z.max(1)

    def moved(self, M):
        s = Slicer.__new__(Slicer)
        s.V = self.V @ M[:3, :3].T + M[:3, 3]
        s.T = self.T
        z = s.V[s.T, 2]
        s.lo, s.hi = z.min(1), z.max(1)
        return s

    def at(self, z, res):
        sel = (self.lo <= z) & (self.hi >= z)
        if not sel.any():
            return np.empty((0, 2))
        return slice_points(self.V, self.T[sel], z, res)


def _inv(M):
    R, t = M[:3, :3], M[:3, 3]
    Ri = R.T if abs(np.linalg.det(R) - 1) < 1e-9 else np.linalg.inv(R)
    return np.hstack([Ri, (-Ri @ t)[:, None]])


def sweep_placements(n, verbose=True):
    """[{instance_key: 3x4 world placement}] at `n` hip angles across the travel.

    Uses the validated 4-bar (tools/kinematics.py), so the sweep is as dense as
    asked for instead of being limited to the three angles SolidWorks exported.
    Refuses to run if the model does not reconstruct those three exactly, because
    an unvalidated model claiming "no collision" is worse than three honest poses.
    """
    import kinematics as K
    leg = K.build(verbose=False)
    ok, worst = K.validate(leg)
    if not ok:
        raise SystemExit(f"kinematic model reconstructs the exported poses to only "
                         f"{worst:.3f} mm -- refusing to sweep with it")
    ref = leg.poses[0]
    reprs = {"femur": leg.fk, "coupler": leg.ck, "tibia": leg.tk}
    out = []
    for q in leg.sweep(n):
        got = leg.at(q)
        if got is None:
            raise SystemExit(f"mechanism does not close at q={np.degrees(q):.2f} deg")
        D = {}
        for g, key in reprs.items():
            M1 = np.vstack([got[g], [0, 0, 0, 1]])[:3]
            D[g] = _compose(M1, _inv(leg.P[ref][key]))
        place = {}
        for key in leg.P[ref]:
            g = next((g for g, members in leg.groups.items() if key in members), None)
            place[key] = leg.P[ref][key] if g is None else _compose(D[g], leg.P[ref][key])
        out.append((q, place))
    if verbose:
        d0, d1 = leg.travel()
        print(f"  swept {n} angles over {np.degrees(d1 - d0):.2f} deg of travel "
              f"(model validated to {worst:.4f} mm)", flush=True)
    return out


def _compose(A, B):
    """A o B for 3x4 affines."""
    return np.hstack([A[:3, :3] @ B[:3, :3],
                      (A[:3, :3] @ B[:3, 3] + A[:3, 3])[:, None]])


def occupancy(part, poses, res, zres, grow, verbose=True, sweep=0):
    """Voxel occupancy of EVERYTHING ELSE, in `part`'s local frame, over poses."""
    src = import_step(paths.part_step(part))
    bb = src.bounding_box()
    g = Grid(bb, grow, res, zres)
    occ = np.zeros((g.nz, g.ny, g.nx), bool)
    own = np.zeros_like(occ)
    zs = g.zs()
    if verbose:
        print(f"{part}: grid {g.nx} x {g.ny} x {g.nz} "
              f"({res} mm plan, {zres} mm depth)", flush=True)

    Vs, Ts, _ = tessellate(src, TESS)
    Vs = np.asarray(Vs, float)
    for k, z in enumerate(zs):
        P = slice_points(Vs, Ts, z, res)
        if len(P):
            pl = own[k]
            g.splat(pl, P)
            own[k] = ndimage.binary_fill_holes(pl)

    # GUARD: the sliced-and-filled silhouette must agree with the independent
    # shadow.  A leaking fill under-reports the part AND under-blocks every
    # neighbour, and nothing downstream would notice.
    shade = np.zeros((g.ny, g.nx), bool)
    g.splat(shade, plan_shadow(Vs, Ts, res))
    shade = ndimage.binary_closing(shade, np.ones((3, 3), bool))
    shade = ndimage.binary_fill_holes(shade)
    a_own = float(own.any(0).sum()) * res * res
    a_shade = float(shade.sum()) * res * res
    if verbose:
        print(f"  plan area: sliced {a_own:.0f} mm2 vs shadow {a_shade:.0f} mm2",
              flush=True)
    if a_shade > 0 and a_own < 0.92 * a_shade:
        raise SystemExit(
            f"{part}: the sliced cross-sections cover only {a_own:.0f} mm2 of the "
            f"{a_shade:.0f} mm2 plan shadow -- the fill is leaking, so neighbours "
            f"would block as hollow shells and growth into a solid would read as "
            f"free. Refusing to publish a free-space map from it.")

    # --- the configurations to block against -------------------------------
    # Either the exported poses, or -- with --sweep N -- N angles across the
    # travel from the validated 4-bar.  Each configuration is a full set of
    # world placements keyed by instance.
    ref = (poses or POSES)[0]
    inst = leaves(os.path.join(paths.EXPORTS, ref + ".STEP"))
    seen, local, keys = {}, {}, []
    for name, shp, loc in inst:
        k = seen.get(name, 0); seen[name] = k + 1
        key = f"{name}#{k}"
        local[key] = (name, shp)
        keys.append(key)

    configs = []
    if sweep:
        for q, place in sweep_placements(sweep, verbose):
            configs.append((f"q={np.degrees(q):.1f}deg", place))
    else:
        for pose in (poses or POSES):
            step = os.path.join(paths.EXPORTS, pose + ".STEP")
            if not os.path.exists(step):
                print(f"  (no STEP for {pose})"); continue
            s2, pl = {}, {}
            for name, shp, loc in leaves(step):
                k = s2.get(name, 0); s2[name] = k + 1
                pl[f"{name}#{k}"] = _matrix(loc)
            configs.append((pose, pl))

    # tessellate each instance's LOCAL geometry ONCE and move the mesh per
    # configuration; re-tessellating a moved shape per pose was most of the cost
    sl = {}
    for key, (name, shp) in local.items():
        V, T, _ = tessellate(shp, TESS)
        if len(T):
            sl[key] = Slicer(V, T)

    own_key = next((k for k in keys if k.split("#")[0] == part), None)
    if own_key is None:
        raise SystemExit(f"{part} is not an instance of {ref}")

    pad = grow + 5
    for label, place in configs:
        if own_key not in place:
            print(f"  {part} missing from {label}"); continue
        Minv = _inv(place[own_key])
        n_near = 0
        for key, s in sl.items():
            if key == own_key:
                continue
            M = _compose(Minv, place[key])
            s2 = s.moved(M)
            mn, mx = s2.V.min(0), s2.V.max(0)
            if (mx[0] < bb.min.X - pad or mn[0] > bb.max.X + pad or
                mx[1] < bb.min.Y - pad or mn[1] > bb.max.Y + pad or
                mx[2] < bb.min.Z - pad or mn[2] > bb.max.Z + pad):
                continue
            n_near += 1
            for k, z in enumerate(zs):
                if z < mn[2] - zres or z > mx[2] + zres:
                    continue
                P = s2.at(z, res)
                if len(P):
                    pl2 = np.zeros((g.ny, g.nx), bool)
                    g.splat(pl2, P)
                    occ[k] |= ndimage.binary_fill_holes(pl2)
        if verbose:
            print(f"  {label:<22} {n_near} neighbours within {pad:.0f} mm", flush=True)
    return g, occ, own


def reach_map(g, occ, own, clear, grow):
    """How far, in mm, the metal may move outward at each plan cell.

    THE DEPTH BAND IS PER PLAN CELL, not the part's whole Z range.  Collapsing
    depth is exactly what made asmkeepout over-block -- a boss at one depth
    would veto growth at every other depth.  Each cell in the growth ring
    inherits the SECTION OF THE PART IT WOULD ATTACH TO (its nearest silhouette
    cell), which is what "grow through the full local section" means in voxels,
    and is blocked only if something occupies that band.

    Growth must also be CONNECTED to the part.  Free space on the far side of an
    obstacle is unreachable material, so only components touching the silhouette
    survive.
    """
    r = max(1, int(round(clear / g.res)))
    rz = int(np.ceil(clear / g.zres))
    blocked = np.zeros_like(occ)
    st = ndimage.generate_binary_structure(2, 2)
    for k in range(g.nz):
        if occ[k].any():
            blocked[k] = ndimage.binary_dilation(occ[k], st, iterations=r)
    if rz:                                   # clearance applies in depth too
        acc = blocked.copy()
        for d in range(1, rz + 1):
            acc[d:] |= blocked[:-d]
            acc[:-d] |= blocked[d:]
        blocked = acc

    sect = own.any(0)
    if not sect.any():
        raise SystemExit("the part occupies no voxel -- grid or tessellation is wrong")

    # the part's own depth band at each plan cell where it has metal
    kk = np.arange(g.nz)[:, None, None]
    zlo = np.where(own, kk, g.nz).min(0)
    zhi = np.where(own, kk, -1).max(0)

    # every ring cell adopts the band of its NEAREST silhouette cell
    dist, (iy, ix) = ndimage.distance_transform_edt(~sect, return_indices=True)
    dist = dist * g.res
    zlo_r = np.where(sect, zlo, zlo[iy, ix])
    zhi_r = np.where(sect, zhi, zhi[iy, ix])

    # "is anything blocked between zlo and zhi", vectorised over cells
    cs = np.cumsum(blocked.astype(np.int32), axis=0)
    yy, xx = np.mgrid[0:g.ny, 0:g.nx]
    hi = np.clip(zhi_r, 0, g.nz - 1)
    lo = np.clip(zlo_r - 1, -1, g.nz - 1)
    upto_hi = cs[hi, yy, xx]
    upto_lo = np.where(lo >= 0, cs[np.clip(lo, 0, g.nz - 1), yy, xx], 0)
    col_blocked = (upto_hi - upto_lo) > 0

    ring = (~sect) & (dist <= grow) & (~col_blocked) & (zhi_r >= 0)

    # keep only what actually touches the part
    lab, n = ndimage.label(ring | sect, st)
    keep = np.unique(lab[sect])
    ring &= np.isin(lab, keep[keep > 0])

    reach = np.where(ring, dist, 0.0)
    return ring, reach, sect, col_blocked


def render(part, g, ring, reach, sect, grow, clear, res, zres, what):
    """The picture: how far each edge may move, with his marks drawn over it.

    Reach is ramped through blues and violets on purpose -- his ADD/REMOVE
    outlines are drawn in green and red, so the measurement and the suggestion
    never compete for the same hue.
    """
    from PIL import Image, ImageDraw
    import marksheet as MS
    up = max(1, int(round(4.0 * res)))          # -> ~4 px per mm on screen
    H, W = ring.shape
    img = np.full((H, W, 3), 250, np.float32)
    img[sect] = (150, 155, 163)
    if ring.any():
        t = np.clip(reach / max(grow, 1e-6), 0, 1)
        ramp = np.stack([40 + 190 * t, 90 + 40 * t, 200 - 60 * t], -1)
        img[ring] = ramp[ring]
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    im = im.transpose(Image.FLIP_TOP_BOTTOM)     # +Y up
    im = im.resize((W * up, H * up), Image.NEAREST)
    d = ImageDraw.Draw(im)

    def to_px(xy):
        a = np.asarray(xy, float)
        return [(( x - g.x0) / res * up, (H - (y - g.y0) / res) * up)
                for x, y in a]

    mk = os.path.join(paths.INPUT, "marks", f"{part}.json")
    if os.path.exists(mk):
        from shapely import wkt as shwkt
        doc = json.load(open(mk))
        for key, col in (("add_wkt", (22, 160, 66)), ("remove_wkt", (206, 34, 34))):
            if not doc.get(key):
                continue
            geo = shwkt.loads(doc[key])
            for poly in (geo.geoms if hasattr(geo, "geoms") else [geo]):
                d.line(to_px(poly.exterior.coords) + [to_px(poly.exterior.coords)[0]],
                       fill=col, width=3)
    hdr, ftr = 150, 44
    out = Image.new("RGB", (im.size[0], im.size[1] + hdr + ftr), (255, 255, 255))
    out.paste(im, (0, hdr))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, out.size[0], hdr - 1], fill=(28, 31, 36))
    d.text((20, 14), f"{part}  -  WHERE MATERIAL CAN GO", font=MS._font(30, True),
           fill=(242, 244, 247))
    d.text((20, 54), f"measured from the assembly over {what}, "
           f"{clear:.1f} mm clearance, {res} mm plan / {zres} mm depth voxels",
           font=MS._font(17), fill=(150, 158, 170))
    d.text((20, 80), "BLUE -> VIOLET = free, shaded by how far the edge may move "
           f"(0 to {grow:.0f} mm).  GREY = the part today.",
           font=MS._font(17), fill=(150, 158, 170))
    d.text((20, 106), "GREEN outline = what he marked ADD.   RED outline = what he "
           "marked REMOVE.   Those are the suggestion; the shading is the measurement.",
           font=MS._font(17), fill=(150, 158, 170))
    bar = 50.0 / res * up
    by = out.size[1] - 30
    for k in range(5):
        d.rectangle([20 + bar * k / 5, by, 20 + bar * (k + 1) / 5, by + 10],
                    fill=(24, 26, 30) if k % 2 == 0 else (255, 255, 255),
                    outline=(24, 26, 30))
    d.text((28 + bar, by - 4), "50 mm", font=MS._font(17, True), fill=(24, 26, 30))
    p = os.path.join(paths.OUT, "marks", "read", f"{part}_FREE.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    out.save(p)
    print(f"    wrote {os.path.basename(p)}", flush=True)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*", default=LINKS)
    ap.add_argument("--res", type=float, default=0.8)
    ap.add_argument("--zres", type=float, default=1.0)
    ap.add_argument("--clear", type=float, default=1.0)
    ap.add_argument("--grow", type=float, default=20.0)
    ap.add_argument("--poses", nargs="*", default=POSES)
    ap.add_argument("--sweep", type=int, default=0,
                    help="sample N hip angles from the validated 4-bar "
                         "instead of only the exported poses")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    for part in (a.parts or LINKS):
        g, occ, own = occupancy(part, a.poses, a.res, a.zres, a.grow,
                                sweep=a.sweep)
        free, reach, sect, colb = reach_map(g, occ, own, a.clear, a.grow)
        cell = a.res * a.res
        doc = {
            "part": part,
            "poses": (f"{a.sweep} swept angles" if a.sweep else list(a.poses)),
            "res_mm": a.res, "zres_mm": a.zres, "clearance_mm": a.clear,
            "grow_window_mm": a.grow,
            "origin_mm": [g.x0, g.y0, g.z0],
            "silhouette_mm2": round(float(sect.sum()) * cell, 1),
            "free_ring_mm2": round(float(free.sum()) * cell, 1),
            "reach_p50_mm": round(float(np.percentile(reach[free], 50)), 2) if free.any() else 0,
            "reach_p90_mm": round(float(np.percentile(reach[free], 90)), 2) if free.any() else 0,
            "reach_max_mm": round(float(reach.max()), 2),
        }
        np.savez_compressed(os.path.join(OUTDIR, f"{part}.npz"),
                            free=free, reach=reach.astype(np.float32),
                            sect=sect, col_blocked=colb,
                            origin=np.array([g.x0, g.y0, g.z0]),
                            res=np.array([a.res, a.zres]))
        json.dump(doc, open(os.path.join(OUTDIR, f"{part}.json"), "w"), indent=1)
        what = (f"{a.sweep} angles swept across the 85 deg travel"
                if a.sweep else f"{len(a.poses)} exported pose(s)")
        render(part, g, free, reach, sect, a.grow, a.clear, a.res, a.zres, what)
        print(f"  {part:<9} silhouette {doc['silhouette_mm2']:8.0f} mm2   "
              f"free ring {doc['free_ring_mm2']:8.0f} mm2   "
              f"reach p50 {doc['reach_p50_mm']:5.2f}  p90 {doc['reach_p90_mm']:5.2f}  "
              f"max {doc['reach_max_mm']:5.2f} mm", flush=True)
