r"""Constraint 5 -- "no very thin walls", min_wall = 3.0 mm.

    C:/Users/ferna/cadenv/Scripts/python.exe lib/thinwall.py Femur [tag]
    C:/Users/ferna/cadenv/Scripts/python.exe lib/thinwall.py --selftest

This is the last of the six constraints to get a check behind it.  Until now
"no very thin walls" was an assertion; the pre-Phase-3 Femur turns out to have
**20% of its surface backed by less than 3 mm**, and 2389 mm2 of that at a hard
mode of exactly 0.50 mm.

HOW IT MEASURES.  From every sample point on the surface, cast a ray along the
INWARD normal and take the distance to the first surface it meets.  That is what
SolidWorks' Evaluate -> Thickness Analysis does, which matters here: Fernando
can check this number himself in the kernel that actually consumes the files,
and Phase 1 established that only a Parasolid consumer is an independent witness
(lib/manifold.py).  Renders from this pipeline come out of the same OCC kernel
that wrote the STEP and agree with it by construction.

WHY NOT THE OTHER OBVIOUS METHOD.  The inscribed-sphere / distance-transform
measure ("local thickness") is the textbook one, and it is wrong for this job:
its value goes to ZERO at every convex edge, because no ball that fits inside
the material can contain a corner point.  A greebled part is mostly edges, so it
would flag several cm3 of perfectly sound metal and train you to ignore it.  The
ray measure has the opposite bias -- on a wedge-shaped wall it reads the normal
distance, which is 1/cos(angle) too LARGE -- so it errs toward silence rather
than toward crying wolf.

VALIDATED BEFORE IT WAS BELIEVED, on solids whose answer is known.  `--selftest`
runs seven of them and checks each to 0.05 mm.  The two that matter are the last
two: a 3 mm chamfer on a 10 mm plate and a 0.5 mm-deep ledge on a thick block
must produce NO thin reading at all.  Both pass.  This is the same discipline
that made the topology gate believable -- a gate nobody has calibrated is worth
nothing, and this repo has shipped three of those.

    3 mm plate                        -> 3.000
    10 mm plate                       -> 10.000
    10 mm plate + 2 mm rib            -> 2.000
    10 mm plate, pocket to 1 mm floor -> 1.000
    10 mm plate + 0.5 mm fin          -> 0.500
    10 mm plate chamfered 3 mm        -> 10.000, no thin area      <- no false alarm
    thick block, 0.5 mm-deep ledge    -> no thin area              <- no false alarm

SCOPE: the FUSED part only (decision 29).  The three filament bodies are NOT
checked and this is a known, accepted gap rather than an oversight: the colour
split shaves skins off a solid that is itself thick, so the Femur's graphite
body reads 0.13 mm over a quarter of its surface while the fused part it came
from is sound there.  Gating on that would have forced a redesign of the locked
GLACIER accent.  A 0.13 mm graphite skin is still a real slicing problem and
nothing in the pipeline will catch it.

PASS/FAIL (decision 29): thin samples are clustered into connected patches, and
a single patch of PATCH_MM2 or more fails, as does TOTAL_MM2 of thin surface in
total.  Zero tolerance was rejected because a knife edge tessellates to
finite-but-small and would block builds that are genuinely fine.  Everything is
reported either way.

TWO THRESHOLDS, AND THE REASON (decision 36, his call).  `min_wall` 3.0 is his
number and it is what gets REPORTED; `GATE_WALL` 1.5 is what PASSES or FAILS.
They are not a fudge and they are not two measurements -- one ray cast, two
masks, so the printed number and the gated number cannot disagree.  The reason
for the split is in the numbers: two thirds of what the styling adds sits in
the 2-3 mm band, which is thinner than he asked for but is not a razor and is
not a print risk at four perimeters; and his OWN source parts fail 3.0, so a
3 mm gate is one that argues with you rather than one you fix.  1.5 mm asks the
question worth asking -- did the styling leave a KNIFE EDGE -- and it is a line
these parts can actually be brought to.

MEASURED AGAINST THE SOURCE, NOT IN ABSOLUTE TERMS.  `gate()` answers "is this
solid thin anywhere", which is the honest question to ask of a finished part.
`compare()` answers the one the rebuild is actually judged on -- "did the
STYLING make it thin" -- and that is what `verify.py` gates.  All three source
links fail the absolute test, and on the Femur every place they fail is the wall
of the hip boss seen through its own bore: a designed bearing seat the ground
rules forbid touching.  See the note above `compare()` for the measurements.
The two-measurement pattern is lib/collide.py's, for the same reason.

    CLI:  lib/thinwall.py Femur              absolute -- is it thin anywhere
          lib/thinwall.py Femur --vs-source  what the styling introduced
          lib/thinwall.py --step <path>      any STEP at all
          lib/thinwall.py --selftest         the seven known solids

TWO LIMITS, STATED RATHER THAN HIDDEN:
  * The sample point is a tessellation centroid, so on a CURVED face it sits up
    to `dev` off the true surface and the reading carries that error.  Almost
    all of the thin material found so far is planar; `dev` is a parameter.
  * A wall thin in a direction other than its own normal is under-reported.  A
    cone of rays would catch it, but taking the minimum over a cone reinstates
    exactly the convex-edge false alarm this method exists to avoid.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import paths
from render3d import tessellate
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.gp import gp_Pnt, gp_Lin, gp_Dir

MIN_WALL   = 3.0     # his number, decision 25
PATCH_MM2  = 10.0    # one connected thin patch this big or bigger fails
TOTAL_MM2  = 50.0    # ... or this much thin surface in total
GATE_WALL  = 1.5     # decision 36: REPORT at min_wall, PASS/FAIL here
TESS_DEV   = 0.20    # tessellation deflection; also the error on curved faces
SAMPLE_MM2 = 1.5     # target surface area per ray
EPS        = 0.05    # start the ray this far inside, clear of the start face
CLUSTER_MM = 2.5     # two thin samples closer than this are the same patch
MISS_MM2   = 1.0     # missed rays worth less area than this are sliver noise
MAX_SUB    = 16      # cap on subdivisions per edge of one triangle


# ------------------------------------------------------------------ sampling

def _bary(n):
    """The n^2 sub-triangle centroids of a triangle uniformly cut n per edge,
    as barycentric (u, v) against (B-A, C-A).

    n(n+1)/2 point one way and (n-1)n/2 the other, which is n^2 in total.
    """
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    up = (i + j) <= (n - 1)
    dn = (i + j) <= (n - 2)
    return np.vstack([
        np.stack([(i[up] + 1 / 3) / n, (j[up] + 1 / 3) / n], 1),
        np.stack([(i[dn] + 2 / 3) / n, (j[dn] + 2 / 3) / n], 1),
    ])


def sample_surface(shape, dev=TESS_DEV, per_mm2=SAMPLE_MM2):
    """Points, outward normals and carried area, at roughly one per `per_mm2`.

    Triangle centroids alone are not enough: a big flat face tessellates into a
    handful of very large triangles, so a 2000 mm2 plate would contribute two
    samples and a thin patch in the middle of it would be invisible.  Large
    triangles are subdivided; small ones are left alone, so the sample count is
    never below the triangle count.

    The normal is the facet normal of the PARENT triangle, so it is exact on a
    planar face and carries the tessellation's own error on a curved one.
    """
    V, T, _ = tessellate(shape, dev=dev)
    if len(T) == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0, int),
                V, T)
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    nrm = np.cross(B - A, C - A)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-9                       # drop degenerate slivers
    idx = np.where(ok)[0]
    A, B, C, nrm, ln = A[ok], B[ok], C[ok], nrm[ok], ln[ok]
    nrm = nrm / ln[:, None]
    area = 0.5 * ln

    n = np.clip(np.ceil(np.sqrt(area / per_mm2)), 1, MAX_SUB).astype(int)
    P, N, W, TI = [], [], [], []
    for k in np.unique(n):
        m = n == k
        uv = _bary(int(k))                                   # (k^2, 2)
        a, b, c = A[m][:, None, :], B[m][:, None, :], C[m][:, None, :]
        pts = a + uv[None, :, 0, None] * (b - a) + uv[None, :, 1, None] * (c - a)
        P.append(pts.reshape(-1, 3))
        N.append(np.repeat(nrm[m], uv.shape[0], axis=0))
        W.append(np.repeat(area[m] / uv.shape[0], uv.shape[0]))
        TI.append(np.repeat(idx[m], uv.shape[0]))
    return (np.vstack(P), np.vstack(N), np.concatenate(W),
            np.concatenate(TI), V, T)


# --------------------------------------------------------------- measurement

def measure(shape, dev=TESS_DEV, per_mm2=SAMPLE_MM2, eps=EPS):
    """Wall thickness at every sample.  Returns a dict; `t` is inf where the
    ray hit nothing, which on a closed solid should never happen.

    A miss is retried once from deeper in.  A degenerate sliver triangle has a
    centroid that can sit a hair OUTSIDE the true surface -- measured on the
    Femur, all 15 misses classified `TopAbs_OUT`, together carrying 0.08 mm2 of
    a 31 135 mm2 part -- so the ray starts in fresh air and leaves without ever
    entering material.  Starting deeper recovers those.  It does NOT paper over
    a real defect: an open shell or a flipped face normal loses a whole face,
    which is square millimetres, and `gate` thresholds on the missed AREA for
    exactly that reason.
    """
    P, N, W, TI, V, T = sample_surface(shape, dev, per_mm2)
    isec = IntCurvesFace_ShapeIntersector()
    isec.Load(shape.wrapped, 1e-7)
    t = np.full(len(P), np.inf)
    deep = max(5 * eps, 0.25)
    for i in range(len(P)):
        d = gp_Dir(*(-N[i]))
        for e in (eps, deep):
            isec.Perform(gp_Lin(gp_Pnt(*(P[i] - N[i] * e)), d), 0.0, 1e6)
            if isec.NbPnt() > 0:
                t[i] = isec.WParameter(1) + e
                break
    return dict(t=t, pts=P, nrm=N, area=W, tri=TI, V=V, T=T)


def percentiles(t, w, qs=(0.01, 0.05, 0.25, 0.50)):
    """Area-weighted percentiles.  Sample COUNT is meaningless here -- a curved
    face gets far more triangles per mm2 than a flat one, so counting samples
    over-weights bores and fillets."""
    f = np.isfinite(t)
    if not f.any():
        return {q: float("nan") for q in qs}
    o = np.argsort(t[f])
    cw = np.cumsum(w[f][o]) / w[f].sum()
    return {q: float(t[f][o][min(np.searchsorted(cw, q), len(o) - 1)]) for q in qs}


def patches(pts, area, mask, radius=CLUSTER_MM):
    """Connected thin regions, largest first.

    Clustered by PROXIMITY, not by mesh adjacency: `tessellate` emits its own
    vertex block per face, so shared-index adjacency stops dead at every face
    boundary and a thin wall spanning two faces would come back as two patches.

    Both faces of a thin wall land in the same patch, since they are by
    definition less than min_wall apart.  The area reported is therefore the
    SURFACE area of the patch and counts a wall twice; it is a size, not a
    footprint, and PATCH_MM2 is calibrated against it.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    i = np.where(mask)[0]
    if len(i) == 0:
        return []
    p = pts[i]
    pairs = cKDTree(p).query_pairs(radius, output_type="ndarray")
    if len(pairs):
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(p), len(p)))
    else:
        g = coo_matrix((len(p), len(p)))
    _, lab = connected_components(g, directed=False)
    return sorted((dict(area=float(area[i][lab == k].sum()),
                        n=int((lab == k).sum()),
                        lo=p[lab == k].min(0), hi=p[lab == k].max(0))
                   for k in range(lab.max() + 1)),
                  key=lambda d: -d["area"])


# --------------------------------------------------------------------- gate

def gate(shape, min_wall=MIN_WALL, res=None, **kw):
    """Returns (problems, res).  Empty problems means constraint 5 is met."""
    if res is None:
        res = measure(shape, **kw)
    t, w = res["t"], res["area"]
    thin = t < min_wall
    res["min_wall"] = min_wall
    res["thin_area"] = float(w[thin].sum())
    res["total_area"] = float(w.sum())
    res["pct"] = percentiles(t, w)
    res["patches"] = patches(res["pts"], w, thin)
    miss = ~np.isfinite(t)
    res["missed"] = int(miss.sum())
    res["missed_area"] = float(w[miss].sum())

    probs = []
    big = [p for p in res["patches"] if p["area"] >= PATCH_MM2]
    if big:
        probs.append(f"{len(big)} thin patch(es) at or over {PATCH_MM2:g} mm2 "
                     f"-- largest {big[0]['area']:.1f} mm2, thinner than "
                     f"{min_wall:g} mm")
    elif res["thin_area"] >= TOTAL_MM2:
        probs.append(f"{res['thin_area']:.1f} mm2 of surface under {min_wall:g} mm "
                     f"in scattered patches (limit {TOTAL_MM2:g} mm2)")
    if res["missed_area"] >= MISS_MM2:
        probs.append(f"{res['missed']} inward ray(s) covering "
                     f"{res['missed_area']:.2f} mm2 hit nothing -- the shell is "
                     f"open, or a face normal points the wrong way; this "
                     f"measurement cannot be trusted until that is explained")
    return probs, res


def gate_report(shape, name="", min_wall=MIN_WALL, res=None, say=print, **kw):
    """Run `gate` and print it.  True if the part has no very thin walls."""
    probs, res = gate(shape, min_wall, res, **kw)
    p, ta, tot = res["pct"], res["thin_area"], res["total_area"]
    say(f"  thin-wall {name:<20} min_wall {min_wall:g} mm   "
        f"{len(res['t'])} rays over {tot:.0f} mm2")
    say(f"      thickness  p1 {p[0.01]:6.2f}   p5 {p[0.05]:6.2f}   "
        f"p25 {p[0.25]:6.2f}   p50 {p[0.50]:6.2f} mm")
    say(f"      under {min_wall:g} mm: {ta:.1f} mm2 ({100*ta/max(tot,1e-9):.1f}%) "
        f"in {len(res['patches'])} patch(es)"
        + (f"   [{res['missed']} ray(s), {res['missed_area']:.2f} mm2, hit nothing]"
           if res["missed"] else ""))
    for i, q in enumerate(res["patches"][:8]):
        if q["area"] < 1.0:
            break
        say(f"        {i+1:2d} {q['area']:8.1f} mm2   "
            f"X {q['lo'][0]:7.1f}..{q['hi'][0]:7.1f}  "
            f"Y {q['lo'][1]:6.1f}..{q['hi'][1]:6.1f}  "
            f"Z {q['lo'][2]:6.1f}..{q['hi'][2]:6.1f}")
    if len(res["patches"]) > 8:
        say(f"        ... {len(res['patches'])-8} more")
    for x in probs:
        say(f"      - {x}")
    return len(probs) == 0, res


# ------------------------------------------- what the STYLING introduced

# THE SOURCE PARTS FAIL THE ABSOLUTE GATE, and they are right to.  Measured on
# the three links as exported from SolidWorks, before this pipeline touches
# them:
#
#     Femur    source  844.7 mm2 thin (3.5%), 17 patches, biggest  63.5
#     Tibia    source  719.9       (1.5%),     7 patches, biggest 142.0
#     Coupler  source 1602.1       (6.6%),    17 patches, biggest 1042.3
#
# The picture says what the numbers cannot: on the Femur every one of those
# patches is the wall of the HIP BOSS, seen through its own bore.  That is a
# designed bearing seat, every hole is untouchable by the ground rules, and no
# restyle is allowed to thicken it.  Holding the styled part to an absolute
# 3.0 mm would therefore fail a PERFECT rebuild, and a gate that can never pass
# is a gate nobody reads.
#
# So this measures the same way lib/collide.py does, for the same reason: twice,
# once on the source and once on the styled part, reporting only the INCREASE.
# Nominally-thin CAD -- a boss wall, a bearing seat -- does not read as a
# defect, and anything the styling introduces has nowhere to hide.  On the
# pre-Phase-3 Femur that is 845 mm2 inherited against 5826 styled.

CACHE = os.path.join(paths.OUT, "cache", "thinwall")


def _cached_thin(path, min_wall, dev, per_mm2):
    """The source's thin points, measured once and kept.

    The source STEP does not change between rebuilds but `verify.py` runs on
    every one, and the Tibia's source alone is 32 s.  Keyed on the file's
    mtime and size as well as the parameters, so editing or re-exporting the
    STEP invalidates it rather than silently comparing against stale geometry.
    """
    from build123d import import_step
    st = os.stat(path)
    key = f"{os.path.basename(path)}_{int(st.st_mtime)}_{st.st_size}_" \
          f"{min_wall:g}_{dev:g}_{per_mm2:g}".replace(" ", "_")
    f = os.path.join(CACHE, key + ".npz")
    if os.path.exists(f):
        try:
            return np.load(f)["pts"]
        except Exception:
            pass
    r = measure(import_step(path), dev=dev, per_mm2=per_mm2)
    pts = r["pts"][r["t"] < min_wall]
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(f, pts=pts)
    return pts


def compare(sty, src_pts, min_wall=MIN_WALL, near=1.5, res=None,
            gate_wall=GATE_WALL, **kw):
    """Split the styled part's thin material into INHERITED and INTRODUCED.

    `src_pts` is the source's thin sample cloud.  A styled thin sample counts as
    inherited when a source thin sample sits within `near` mm of it -- both are
    measured in the same frame, because `build()` mirrors the part back to the
    source orientation before export.  Returns (problems, res).
    """
    from scipy.spatial import cKDTree
    if res is None:
        res = measure(sty, **kw)
    t, w, P = res["t"], res["area"], res["pts"]
    thin = t < min_wall
    idx = np.where(thin)[0]
    if len(idx) and len(src_pts):
        d, _ = cKDTree(src_pts).query(P[idx])
        new = d > near
    else:
        new = np.ones(len(idx), bool)

    intro = np.zeros(len(t), bool)
    intro[idx[new]] = True
    res["min_wall"] = min_wall
    res["total_area"] = float(w.sum())
    res["thin_area"] = float(w[thin].sum())
    res["inherited_area"] = float(w[thin].sum() - w[intro].sum())
    res["introduced_area"] = float(w[intro].sum())
    res["pct"] = percentiles(t, w)
    res["patches"] = patches(P, w, intro)          # patches of NEW thin only
    miss = ~np.isfinite(t)
    res["missed"] = int(miss.sum())
    res["missed_area"] = float(w[miss].sum())

    # REPORT AT min_wall, DECIDE AT gate_wall -- decision 36, his call.
    #
    # min_wall 3.0 is his number and it stays the number that gets printed, so
    # nothing is hidden.  But it is not a number these parts can be held to: his
    # OWN source parts fail it (source Coupler 1729 mm2 under 3 mm, and every
    # thin patch on the source Femur is the hip bearing seat, which no restyle
    # may thicken), and two thirds of what the styling adds sits in the 2-3 mm
    # band -- thin against the number, but not a razor and not a print risk.
    # Gating there made the gate a thing to be argued with rather than fixed.
    #
    # So the gate is 1.5 mm, about four perimeters on the X2D, and it answers
    # the question worth answering: did the styling leave a KNIFE EDGE.  Both
    # thresholds come off the same measurement -- one ray cast, two masks -- so
    # the reported number and the gated number can never disagree.
    gw = min(gate_wall, min_wall)
    gate_thin = np.zeros(len(t), bool)
    gate_thin[idx[new]] = t[idx[new]] < gw
    res["gate_wall"] = gw
    res["gate_area"] = float(w[gate_thin].sum())
    res["gate_patches"] = patches(P, w, gate_thin)

    probs = []
    big = [p for p in res["gate_patches"] if p["area"] >= PATCH_MM2]
    if big:
        probs.append(f"the styling INTRODUCED {len(big)} thin patch(es) at or "
                     f"over {PATCH_MM2:g} mm2 -- largest {big[0]['area']:.1f} mm2, "
                     f"thinner than {gw:g} mm")
    elif res["gate_area"] >= TOTAL_MM2:
        probs.append(f"the styling INTRODUCED {res['gate_area']:.1f} mm2 of "
                     f"surface under {gw:g} mm in scattered patches "
                     f"(limit {TOTAL_MM2:g} mm2)")
    if res["missed_area"] >= MISS_MM2:
        probs.append(f"{res['missed']} inward ray(s) covering "
                     f"{res['missed_area']:.2f} mm2 hit nothing -- the shell is "
                     f"open, or a face normal points the wrong way")
    return probs, res


def compare_report(sty, src_step, name="", min_wall=MIN_WALL, say=print,
                   dev=TESS_DEV, per_mm2=SAMPLE_MM2,
                   gate_wall=GATE_WALL, **kw):
    """Measure the styled part against its own source and print the increase."""
    src_pts = _cached_thin(src_step, min_wall, dev, per_mm2)
    probs, res = compare(sty, src_pts, min_wall, dev=dev, per_mm2=per_mm2,
                         gate_wall=gate_wall, **kw)
    p = res["pct"]
    say(f"  thin-wall {name:<20} report {min_wall:g} mm / gate "
        f"{res['gate_wall']:g} mm   {len(res['t'])} rays over "
        f"{res['total_area']:.0f} mm2")
    say(f"      thickness  p1 {p[0.01]:6.2f}   p5 {p[0.05]:6.2f}   "
        f"p25 {p[0.25]:6.2f}   p50 {p[0.50]:6.2f} mm")
    say(f"      under {min_wall:g} mm: {res['thin_area']:.1f} mm2 total "
        f"= {res['inherited_area']:.1f} inherited from the source "
        f"+ {res['introduced_area']:.1f} INTRODUCED by the styling")
    say(f"      of which under {res['gate_wall']:g} mm (the GATE): "
        f"{res['gate_area']:.1f} mm2 introduced, "
        f"{len(res['gate_patches'])} patch(es)")
    if res["missed"]:
        say(f"      [{res['missed']} ray(s), {res['missed_area']:.2f} mm2, "
            f"hit nothing]")
    if res["patches"]:
        say(f"      new thin material, {len(res['patches'])} patch(es):")
    for i, q in enumerate(res["patches"][:8]):
        if q["area"] < 1.0:
            break
        say(f"        {i+1:2d} {q['area']:8.1f} mm2   "
            f"X {q['lo'][0]:7.1f}..{q['hi'][0]:7.1f}  "
            f"Y {q['lo'][1]:6.1f}..{q['hi'][1]:6.1f}  "
            f"Z {q['lo'][2]:6.1f}..{q['hi'][2]:6.1f}")
    if len(res["patches"]) > 8:
        say(f"        ... {len(res['patches'])-8} more")
    for x in probs:
        say(f"      - {x}")
    return len(probs) == 0, res


# ---------------------------------------------------------------- the picture

# A ramp, not a red/green split: "how thin" is the useful question once you know
# there is something to fix.  Below min_wall the colour is hot; above it the
# part goes quiet grey so the hot regions are the only thing the eye lands on.
RAMP = [(0.00, (214,  38,  38)),      # under a quarter of min_wall -- critical
        (0.35, (236, 104,  32)),
        (0.70, (233, 176,  54)),
        (1.00, (120, 196, 140)),      # just over min_wall
        (2.00, (150, 156, 166)),
        (6.00, (210, 215, 222))]      # thick


def _colour(frac):
    for (a, ca), (b, cb) in zip(RAMP, RAMP[1:]):
        if frac <= b:
            u = (frac - a) / max(b - a, 1e-9)
            return tuple(ca[k] + u * (cb[k] - ca[k]) for k in range(3))
    return RAMP[-1][1]


def map_png(res, out, part="", tag="", min_wall=MIN_WALL, views=None):
    """Shade the part by wall thickness onto one sheet.

    Reuses `render_color.view` untouched by handing it one pseudo-body per
    colour band -- it already z-buffers several bodies together, so bucketing
    the triangles costs nothing and no renderer has to learn about thickness.
    """
    import render_color as RC
    V, T, tri, t = res["V"], res["T"], res["tri"], res["t"]

    # a triangle is as thin as its thinnest sample
    per_tri = np.full(len(T), np.inf)
    np.minimum.at(per_tri, tri, t)
    seen = np.isfinite(per_tri)
    if not seen.any():
        return None
    frac = np.where(seen, per_tri / min_wall, 9.0)

    edges = [0, .12, .25, .40, .60, .85, 1.0, 1.5, 3.0, 1e9]
    bodies = []
    for lo, hi in zip(edges, edges[1:]):
        m = (frac >= lo) & (frac < hi)
        if m.any():
            bodies.append((V, T[m], _colour(min(lo + (hi - lo) / 2, 6.0))))
    if not bodies:
        return None

    views = views or [("end-on  (down X)", 0, 2), ("end-on  (down Y)", 0, 92),
                      ("show face", 88, -90), ("back face", -88, -90)]
    tiles = [(nm, "", RC.view(bodies, V, (820, 700), elev=e, azim=a))
             for nm, e, a in views]
    ta, tot = res["thin_area"], res["total_area"]
    RC.sheet(tiles, 2, out,
             header=f"{part} - wall thickness" + (f"  [{tag}]" if tag else ""),
             sub=f"red = under {min_wall:g} mm.  "
                 f"{ta:.0f} mm2 of {tot:.0f} ({100*ta/max(tot,1e-9):.1f}%) is thin, "
                 f"in {len(res['patches'])} patch(es); "
                 f"thinnest {np.nanmin(t[np.isfinite(t)]):.2f} mm",
             size=(820, 700))
    return out


# ------------------------------------------------------------------ selftest

def selftest(say=print):
    """Six solids whose thickness is known by construction.

    The last two are the point of the exercise: a chamfered edge and a shallow
    ledge are the two shapes a naive thickness measure gets wrong, and both must
    come back with NO thin area.  A checker that has not been calibrated against
    a known answer is an opinion.
    """
    from build123d import Box, Pos, chamfer
    b = Box(60, 40, 10)
    cases = [
        ("3 mm plate",                     Box(60, 40, 3),                        3.0,  None),
        ("10 mm plate",                    Box(60, 40, 10),                      10.0,  0.0),
        ("10 mm plate + 2 mm rib",         Box(60,40,10) + Pos(0,0,10)*Box(50,2,12),  2.0,  None),
        ("10 mm plate -> 1 mm floor",      Box(60,40,10) - Pos(0,0,0.5)*Box(30,20,9), 1.0,  None),
        ("10 mm plate + 0.5 mm fin",       Box(60,40,10) + Pos(0,0,9)*Box(50,0.5,8),  0.5,  None),
        ("10 mm plate, 3 mm chamfer",      chamfer(b.edges(), 3.0),              10.0,  0.0),
        ("thick block, 0.5 mm ledge",      Box(60,40,20) - Pos(0,20,5)*Box(70,1.0,10), 10.0, 0.0),
    ]
    ok = True
    say(f"{'case':<32} {'expect':>8} {'min':>8} {'thin mm2':>10}   verdict")
    for name, shp, want_min, want_thin in cases:
        r = measure(shp, dev=0.25, per_mm2=1.0)
        t, w = r["t"], r["area"]
        f = np.isfinite(t)
        mn = float(t[f].min())
        ta = float(w[t < MIN_WALL].sum())
        good = abs(mn - want_min) <= 0.05
        if want_thin is not None:
            good &= ta <= want_thin + 1e-6
        ok &= good
        say(f"{name:<32} {want_min:8.2f} {mn:8.3f} {ta:10.1f}   "
            f"{'ok' if good else 'WRONG'}"
            + ("" if want_thin is None else f"   (expect {want_thin:g} mm2 thin)"))
    say(f"\nselftest: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------- CLI

if __name__ == "__main__":
    import argparse, json
    from build123d import import_step
    ap = argparse.ArgumentParser(description="constraint 5 -- no very thin walls")
    ap.add_argument("part", nargs="?", default=None)
    ap.add_argument("tag", nargs="?", default=None)
    ap.add_argument("--step", help="measure any STEP instead of a styled part")
    ap.add_argument("--min-wall", type=float, default=MIN_WALL)
    ap.add_argument("--dev", type=float, default=TESS_DEV)
    ap.add_argument("--sample", type=float, default=SAMPLE_MM2)
    ap.add_argument("--vs-source", action="store_true",
                    help="report only the thinness the STYLING introduced "
                         "(what verify.py gates on); default is absolute")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)

    if a.step:
        path, part, tag = a.step, os.path.splitext(os.path.basename(a.step))[0], ""
    else:
        part = a.part or "Femur"
        tag = a.tag
        if tag is None:
            sf = os.path.join(paths.SPECS, f"{part.lower()}.json")
            tag = (json.load(open(sf)).get("tag", "glacier")
                   if os.path.exists(sf) else "glacier")
        path = paths.styled_step(part, tag)
        if not os.path.exists(path):
            raise SystemExit(f"no styled part at {path} -- "
                             f"run parts/{part.lower()}.py first")

    shp = import_step(path)
    print(f"=== {part} : wall thickness ===")
    print(f"{path}\n{shp.volume/1000:.2f} cm3, {len(shp.solids())} solid(s)\n")
    if a.vs_source and not a.step:
        ok, res = compare_report(shp, paths.part_step(part), part, a.min_wall,
                                 dev=a.dev, per_mm2=a.sample)
    else:
        ok, res = gate_report(shp, part, a.min_wall, dev=a.dev, per_mm2=a.sample)
    if not a.no_map:
        paths.ensure_out()
        out = os.path.join(paths.RENDERS,
                           f"thinwall_{part.replace(' ', '_')}"
                           + (f"_{tag}" if tag else "") + ".png")
        map_png(res, out, part, tag, a.min_wall)
    print(f"\nthin-wall gate: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
