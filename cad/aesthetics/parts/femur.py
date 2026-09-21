r"""Femur v4 -- same styling system as the Tibia, second part.

    plan(spec)   -> pure shapely/2D.  Cheap.  Feeds the board AND the build.
    build(spec)  -> OCC solids, per-filament bodies, 3MF.

Datums are DERIVED, not assumed.  The source has bolt-circle openings at
x = -93.79 and x = +93.79, which are 187.58 mm apart -- exactly |AC| from
CLAUDE.md -- so A (hip output shaft) and C (knee pivot) sit on those centres.

SHOW FACE: the recipe always decorates local +Z, but the Femur is mounted with
its local +Z pointing INBOARD -- at the side panel, four millimetres away.  Its
show face is local -Z.  `show_face: "-Z"` in the spec reflects the source about
the XY plane on the way in and reflects the finished bodies back on the way out,
so the features land on the right side while every hole returns to exactly where
it started.  See DECISIONS.md 14 for how the assembly settled this.

ONE FEMUR-SPECIFIC TRAP: `silhouette()` unions horizontal faces, and this part
has a ~7 mm sloped transition with no horizontal face in it, so the silhouette
comes back as TWO disjoint pieces.  Taking the larger one -- which is what the
Tibia recipe does -- would drop the far block from `grown`, and since the
envelope INTERSECTS the solid that would cut the block clean off the part.  The
pieces are bridged instead.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import numpy as np
from build123d import *
from shapely.geometry import Polygon as ShPoly, Point as ShPoint, box as shbox, LineString
from shapely.ops import unary_union
from keepout import openings, silhouette
from shputil import prism, raised, frustum, geoms, union
from feat import trap_xz, side_solid, wall_shell, trap_plan
import accents as ACC
import spec as spec_mod
import asmkeepout

PART = "Femur"
A, C = (-93.79, 0.0), (93.79, 0.0)      # derived; see module docstring
SHOW_FACE = "+Z"                        # set from the spec by _face()
EPS_Z = 0.05       # mm; keeps layer boundaries from being coincident faces
SIL_TOL = 1.0      # mm2 of source silhouette `grown` may lose to GEOS noise
_CACHE = {}


def _flip(shape):
    """Reflect about the XY plane.

    Applied twice it is the identity, which is what makes `show_face` safe:
    mirror the source in, style local +Z as always, mirror the result back.
    verify.py compares against the UNMIRRORED source, so the round trip has to
    be exact -- a Plane.XY reflection only negates z, so it is.  A reflection
    can still come back with reversed face orientation, so volume is checked:
    a sign flip or a collapse would otherwise ship an unprintable part.
    """
    if shape is None:
        return None
    out = mirror(shape, about=Plane.XY)
    if abs(out.volume - shape.volume) > max(1.0, 0.001 * shape.volume):
        raise RuntimeError(
            f"mirror changed volume {shape.volume/1000:.2f} -> {out.volume/1000:.2f} cm3")
    return out


def _split_z(sp, ZT, ZB):
    """Where white stops and graphite starts.

    `split_z` is an ABSOLUTE z tuned on the Tibia, and it does not transfer: it
    sits 18.5 mm below the Tibia's show face, the full 39 mm on the Femur and
    only 10 mm on the Coupler, which came out 29% white against the Femur's 61%.
    Same defect as trap 10, one axis over.  `white_depth` replaces it with a
    fraction of the part's OWN thickness below the show face; set it to
    (ZT - split_z) / (ZT - ZB) and a part reproduces bit-identically.
    """
    wd = sp.get("white_depth")
    if wd is None:
        return sp["split_z"]
    return ZT - float(wd) * (ZT - ZB)


def _side_band(sp, ZT, ZB, f_near, f_far, z0, z1):
    """A side-pocket band: fractions below the show face, or legacy absolute z."""
    if sp.get("white_depth") is None:
        return (z0, z1)
    T = ZT - ZB
    return (ZT - f_far * T, ZT - f_near * T)


def _face(sp):
    """Adopt the spec's show face, dropping the cache if it moved."""
    global SHOW_FACE
    f = sp.get("show_face", "+Z")
    if f not in ("+Z", "-Z"):
        raise ValueError(f"show_face must be '+Z' or '-Z', got {f!r}")
    if f != SHOW_FACE:
        SHOW_FACE = f
        _CACHE.clear()


def _bridge(poly, rmax=9.0):
    """Join disjoint silhouette pieces into one polygon.

    The pieces here OVERLAP in both X and Y yet do not touch -- they interlock
    across a sloped transition -- so a gap test on bounding boxes does not find
    the gap.  A morphological close does, and because closing only ever ADDS
    area the result is guaranteed to still contain the true silhouette.
    """
    gs = geoms(poly)
    if len(gs) < 2:
        return poly
    u = unary_union(gs)
    r = 1.0
    while r <= rmax:
        c = u.buffer(r, join_style=2).buffer(-r, join_style=2)
        got = geoms(c)
        if len(got) == 1 and got[0].contains(u.buffer(-0.01)):
            return got[0]
        r += 1.0
    return max(gs, key=lambda g: g.area)


def source():
    if "solid" not in _CACHE:
        s = import_step(paths.part_step(PART)).solids()[0]
        if SHOW_FACE == "-Z":
            s = _flip(s)
        bb = s.bounding_box()
        op = openings(s, bb.max.Z, bb.min.Z)
        out = _bridge(silhouette(s))
        out = max(geoms(out), key=lambda g: g.area)
        _CACHE.update(solid=s, ZT=bb.max.Z, ZB=bb.min.Z, OUT=out,
                      OPEN=op, KEEP=unary_union(op) if op else ShPoly())
    return _CACHE


def marks():
    """His add / remove regions, as part-local plan polygons.  Empty if absent.

    NO TRANSFORM IS NEEDED even though `source()` mirrors this part.  The marks
    are written in the part's own local XY by `tools/markplan.py`, and `_flip`
    reflects about Plane.XY, which negates Z and leaves X and Y untouched.  That
    is worth stating rather than leaving to be rediscovered: a frame mismatch
    here would put every pocket on the wrong side of the part and still build,
    verify and render perfectly.

    `confirmed: false` in the file is his own caveat (decision 26) -- the marks
    are intent, the geometry is the judge -- so these are used to say WHERE, and
    the free map and the collision sweep say HOW FAR.
    """
    if "marks" not in _CACHE:
        from shapely import wkt as _wkt
        f = os.path.join(paths.INPUT, "marks", f"{PART}.json")
        add = rem = ShPoly()
        if os.path.exists(f):
            d = json.load(open(f))
            if d.get("add_wkt"):
                add = _wkt.loads(d["add_wkt"]).buffer(0)
            if d.get("remove_wkt"):
                rem = _wkt.loads(d["remove_wkt"]).buffer(0)
        _CACHE["marks"] = (add, rem)
    return _CACHE["marks"]


def _drop_detached(body, say, what="body"):
    """Constraint 3, "no floating pieces": reduce `body` to ONE connected solid.

    THIS MUST RUN LAST.  It used to sit immediately after the growth fuse, but
    the raised frame/rail/pads union on afterwards and the flange later still,
    so anything stranded after that point was never looked at -- which is how
    the shipped Femur kept a 25.2 mm3 chunk and the Coupler four totalling
    1.15 cm3, both of them found by the topology gate rather than by the guard
    whose whole job they were.

    Keeping the LARGEST solid rather than everything that touches the source:
    once cuts are in, two pieces can both overlap the source and still be
    disconnected from each other, so "touches the source" does not mean "is one
    part".  A big drop is a design error, not debris -- a removal severed the
    part -- so it raises rather than quietly printing half a Femur.
    """
    sols = body.solids()
    if len(sols) < 2:
        return body
    sols = sorted(sols, key=lambda s: -s.volume)
    keep, drop = sols[0], sols[1:]
    lost = sum(s.volume for s in drop)
    say(f"  {what}: dropped {len(drop)} detached piece(s), {lost/1000:.3f} cm3")
    # WHERE it is, not just how much.  A guard that silently eats several cm3
    # every build is how the earlier defects stayed hidden; the bbox says which
    # feature is stranding material.
    for d in drop[:4]:
        b = d.bounding_box()
        say(f"      {d.volume/1000:7.3f} cm3 at "
            f"X {b.min.X:+7.1f}..{b.max.X:+7.1f}  "
            f"Y {b.min.Y:+7.1f}..{b.max.Y:+7.1f}  "
            f"Z {b.min.Z:+7.1f}..{b.max.Z:+7.1f}")
    if lost > max(500.0, 0.01 * body.volume):
        raise RuntimeError(
            f"{PART}: {lost/1000:.2f} cm3 came away as {len(drop)} separate "
            f"solid(s) -- a removal has SEVERED the part, which is a design "
            f"error rather than debris.  Largest kept piece is "
            f"{keep.volume/1000:.2f} cm3 of {body.volume/1000:.2f}.")
    return keep


def _band_faces(solid, bands, tol=0.8):
    """Plan footprint of the solid's own material inside each z band.

    A flange is extruded over a FIXED z range, but a part's perimeter is not a
    constant-section prism: where its surface is recessed there is no metal
    beside the band to fuse to, and the flange comes out detached.  The Coupler
    threw away 13.8 cm3 that way, its whole flange.  Growth is clipped to sit
    within reach of the material actually present in that band.
    """
    from render3d import tessellate
    V, T, _ = tessellate(solid, tol)
    V = np.asarray(V, dtype=float)
    idx = np.asarray(T, dtype=int)
    out = []
    if not len(idx):
        return [None] * len(bands)
    z = V[:, 2][idx]
    zmax, zmin = z.max(axis=1), z.min(axis=1)
    for z0, z1 in bands:
        keep = (zmax >= z0) & (zmin <= z1)
        if not keep.any():
            out.append(None); continue
        xy = np.round(V[:, :2][idx[keep]], 3)
        cross = ((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) -
                 (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
        xy = xy[np.abs(cross) > 2e-3]
        polys = [p for p in (ShPoly(t) for t in xy) if p.is_valid]
        got = None
        for attempt in (polys, [p.buffer(0.01, join_style=2) for p in polys]):
            if not attempt:
                break
            try:
                u = unary_union(attempt).buffer(0)
                if not u.is_empty:
                    got = u; break
            except Exception:
                continue
        out.append(got)
    return out


def _top_face(solid, ZT, depth=2.0):
    """Plan region where the part's OWN surface reaches the show face.

    A raised pad is extruded from a fixed z, so wherever the real surface sits
    lower than that the pad has nothing under it and floats free -- that is what
    put a detached plate with two prongs under the Coupler, and it is the honest
    version of "paint on a pig": material sitting off the part rather than on it.
    Clipping every raised feature to this region guarantees a pad lands on metal.
    """
    from render3d import tessellate
    V, T, _ = tessellate(solid, 0.8)
    V = np.asarray(V, dtype=float)
    idx = np.asarray(T, dtype=int)
    if not len(idx):
        return None
    keep = V[:, 2][idx].max(axis=1) >= ZT - depth
    if not keep.any():
        return None
    xy = np.round(V[:, :2][idx[keep]], 3)
    cross = ((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) -
             (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
    xy = xy[np.abs(cross) > 2e-3]
    polys = [p for p in (ShPoly(t) for t in xy) if p.is_valid]
    if not polys:
        return None
    for attempt in (polys, [p.buffer(0.01, join_style=2) for p in polys]):
        try:
            u = unary_union(attempt).buffer(0)
            if not u.is_empty:
                return u
        except Exception:
            continue
    return None


def _yspan(p, x):
    s = p.intersection(LineString([(x, -90), (x, 90)]))
    if s.is_empty: return None
    ys = [c[1] for g in (s.geoms if s.geom_type == 'MultiLineString' else [s]) for c in g.coords]
    return min(ys), max(ys)


def _slots(x0, x1, n, gap=9.0):
    if n <= 0: return []
    span = x1 - x0
    g = min(gap, span / (n * 2.5))
    w = (span - g * (n - 1)) / n
    if w <= 1.5: return []
    return [(x0 + i * (w + g), x0 + i * (w + g) + w) for i in range(n)]


def _trap_edge(poly, segs, side=1, ramp=14.0):
    """Trapezoidal growth band along one edge of `poly`.

    The band is closed back to the part's MID-HEIGHT.  That used to be a
    hard-coded y = 0.0, which assumes the part straddles the axis -- true for
    the Femur (-19.4..+19.4) and Coupler (-22.0..+22.0), false for the Side
    panel (-29.25..+75.00), where the closing edge crossed the band and GEOS
    threw "side location conflict".  Trap 12, fourth axis-assumption of its
    kind.  On a part centred on y = 0 this is bit-identical to the old code.
    """
    sgn = 1.0 if side > 0 else -1.0
    ymid = (poly.bounds[1] + poly.bounds[3]) / 2.0
    out = []
    for xa, xb, h in segs:
        if h <= 0.05 or (xb - xa) < 2 * ramp + 4: continue
        ys_a, ys_b = _yspan(poly, xa), _yspan(poly, xb)
        if ys_a is None or ys_b is None: continue
        ya = ys_a[1] if side > 0 else ys_a[0]
        yb = ys_b[1] if side > 0 else ys_b[0]
        base = None
        for x in np.linspace(xa, xb, 28):
            ys = _yspan(poly, x)
            if ys is None: continue
            v = ys[1] if side > 0 else ys[0]
            base = v if base is None else (max(base, v) if side > 0 else min(base, v))
        if base is None: continue
        top = base + sgn * h
        band = ShPoly([(xa, ya), (xa + ramp, top), (xb - ramp, top), (xb, yb),
                       (xb, ymid), (xa, ymid)])
        if band.is_valid and band.area > 1e-6:
            out.append(band)
    if not out:
        return ShPoly()
    try:
        return unary_union(out)
    except Exception:
        return unary_union([b.buffer(0) for b in out])


def plan(sp):
    _face(sp)
    src = source()
    OUT, KEEP = src["OUT"], src["KEEP"]
    ZT, ZB = src["ZT"], src["ZB"]
    kb = KEEP.buffer(2.2) if not KEEP.is_empty else ShPoly()
    X0, _, X1, _ = OUT.bounds
    facet = sp.get("facet", True)
    L = X1 - X0

    # circular bosses at the two pivots
    r_end = 20.0 + sp["knee_grow"]
    bosses = unary_union([ShPoint(*A).buffer(r_end, 96), ShPoint(*C).buffer(r_end, 96)])
    if facet:
        bosses = bosses.simplify(1.1)

    # trapezoidal edge growth, expressed as fractions of the part's own length
    oh, ramp = sp.get("out_h", 0.0), sp.get("out_ramp", 14.0)
    dorsal = ShPoly()
    if oh > 0.05:
        f = lambda t: X0 + t * L
        if sp.get("out_steps", 2) >= 2:
            top_segs = [(f(.10), f(.52), oh * 0.68), (f(.47), f(.90), oh)]
            bot_segs = [(f(.08), f(.55), oh * 0.85), (f(.50), f(.88), oh * 0.55)]
        else:
            top_segs = [(f(.10), f(.90), oh)]
            bot_segs = [(f(.08), f(.88), oh * 0.8)]
        bands = [_trap_edge(OUT, top_segs, 1, ramp)]
        if sp.get("out_sides", "both") == "both":
            bands.append(_trap_edge(OUT, bot_segs, -1, ramp))
        dorsal = unary_union([b for b in bands if not b.is_empty])

    g0 = unary_union([OUT, bosses, dorsal]).buffer(0)

    orr = sp.get("org_r", 0.0)
    if orr > 0.05:
        g1 = (g0.buffer(orr, join_style=1).buffer(-2 * orr, join_style=1)
                .buffer(orr, join_style=1))
        wd = sp.get("waist_d", 0.0)
        if wd > 0.05:
            R, xmid = 165.0, (X0 + X1) / 2
            ys = _yspan(g0, xmid)
            if ys is not None:
                lo, hi = ys
                g1 = (g1.difference(ShPoint(xmid, hi + R - wd).buffer(R, 200))
                        .difference(ShPoint(xmid, lo - R + wd).buffer(R, 200)))
        g0 = unary_union([g1, OUT]).buffer(0)      # never inside the source

    # `no_growth` clamps the silhouette to the SOURCE outline: the part may be
    # cut, but its footprint may not grow by so much as a millimetre.  This robot
    # is packed too tightly for additive styling -- the keep-out blocks the
    # bosses at every pivot, so whatever growth survived came through as a lone
    # sliver and read as a fin stuck on the part.  Even with outline and out_h at
    # zero the bosses still grew (RAD + 3.5 + knee_grow is always wider than the
    # lobe) and the taper turned that ring into a thin plate hanging off the
    # show face.  With this set, all styling is subtractive plus raised pads.
    if sp.get("no_growth"):
        g0 = ShPoly(OUT.exterior) if OUT.interiors else OUT

    tol = 0.4 if (facet and orr <= 0.05) else (0.5 if orr > 0.05 else 0.05)
    grown = max(geoms(g0.buffer(0).simplify(tol)), key=lambda g: g.area)
    grown = ShPoly(grown.exterior)
    grown = max(geoms(grown.buffer(-0.12, join_style=2).buffer(0.12, join_style=2)),
                key=lambda g: g.area)

    # ADDED MATERIAL THINNER THAN min_wall IS NOT MATERIAL, IT IS AN ARTEFACT.
    #
    # `simplify(tol)` above moves the design boundary by up to tol -- 0.5 mm in
    # organic mode -- so everywhere the design outline nearly coincides with the
    # source outline, `grown - OUT` is a hairline ribbon a few tenths wide.
    # `build()` extrudes exactly that difference, so the ribbon becomes real
    # metal: a half-millimetre skirt running most of the way round the part.
    #
    # It is the single biggest defect on the part.  Measured on the rebuild:
    # 2049 mm2 -- THIRTY-SEVEN PER CENT of all the thinness the styling
    # introduces -- lies outside the source silhouette, at a median thickness of
    # 0.51 mm.  That is the 0.50 mm mode in the thin-wall map, and it is not a
    # design feature anybody chose; it is `tol` leaking into the solid.
    #
    # Morphological opening is the exact tool: a band keeps a ball of radius w/2,
    # so opening the annulus by min_wall/2 preserves every band at least
    # min_wall wide and deletes every one that is not, without touching the
    # source silhouette.  Grown can only ever gain area over OUT, so the
    # "grown still contains OUT" guard below stays satisfied by construction.
    _MW = float(sp.get("min_wall", 3.0))
    _ann = grown.difference(OUT)
    if not _ann.is_empty and _MW > 0.1:
        _keep = _ann.buffer(-_MW / 2, join_style=2).buffer(_MW / 2, join_style=2)
        _shed = _ann.area - (0.0 if _keep.is_empty else _keep.area)
        if _shed > 1.0:
            print(f"  growth: shed {_shed:.0f} mm2 of sub-{_MW:g} mm skirt "
                  f"({_ann.area:.0f} -> {_ann.area - _shed:.0f} mm2 of real growth)")
        _u = unary_union([OUT, _keep]).buffer(0)
        grown = ShPoly(max(geoms(_u), key=lambda g: g.area).exterior)
    # `env` INTERSECTS the solid, so anything missing from `grown` is cut off the
    # part.  That mistake has been expensive here before (traps 2, 3, 9), and it
    # is silent -- verify.py only catches it once a hole is hit.  Check directly.
    # The assembly says where this part may NOT go.  This runs AFTER the
    # simplify/open above, not before: `simplify(tol)` moves a boundary by up to
    # `tol`, and where a keep-out cut runs along the source edge that was
    # shaving ~25 mm2 off the part -- 0.00 mm2 lost without the clip, 25 with
    # it.  Cutting last and restoring OUT last makes the invariant structural
    # rather than something the tolerances have to be trusted not to break.
    clear = sp.get("asm_clearance", 1.0)

    def _clip(poly, ko_):
        """Remove the keep-out from `poly`, then put the source silhouette
        straight back -- the clip governs GROWTH, never the part itself."""
        if ko_ is None:
            return poly
        g = (poly.difference(ko_)
                 .buffer(-0.12, join_style=2).buffer(0.12, join_style=2))
        g = unary_union([g, OUT]).buffer(0)
        return ShPoly(max(geoms(g), key=lambda x: x.area).exterior)

    # Growth is LAYERED through the thickness.  A single full-thickness prism
    # cannot grow at all where the links overlap in plan: they clear each other
    # by interleaving in DEPTH, not in plan, so one contested region blocks the
    # whole 39 mm -- the flat keep-out took 83% of the Femur's growth with it.
    # Clipping per z band puts material where the space actually is; outboard of
    # the Femur, global Z above 162, there is nothing at all.  The features all
    # live on the show face, so `grown` becomes the OUTERMOST layer.
    # COLLISION layers come from the keep-out cache; TAPER bands are finer.
    # Each taper band uses the keep-out of the collision layer it sits in, so
    # the flare can be as smooth as we like without recomputing the assembly
    # pass -- 3 collision layers gave a visible 3-step stair on the Femur's
    # edge, which reads as an artefact rather than as a drafted face.
    NKO = asmkeepout.n_layers(PART)
    NL = int(sp.get("grow_layers", 1))
    NT = max(NL, int(sp.get("taper_steps", NL)))
    if NL > 1 and NKO >= NL:
        # the design outline BEFORE any keep-out clipping -- the flange must
        # stay inside what the style asked for, not just outside the keep-outs
        _design = grown
        step = (ZT - ZB) / NT
        ko_h = (ZT - ZB) / NKO
        layers, _KOS = [], []
        for k in range(NT):
            z0, z1 = ZB + k * step, ZB + (k + 1) * step
            ki = min(NKO - 1, max(0, int(((z0 + z1) / 2 - ZB) / ko_h)))
            # the cache is indexed in the SOURCE frame; mirroring reverses it
            src_k = (NKO - 1 - ki) if SHOW_FACE == "-Z" else ki
            _ko = asmkeepout.load(PART, clear, layer=src_k)
            _KOS.append(_ko)
            layers.append((z0, z1, _clip(grown, _ko), None))
        grown = layers[-1][2]

        # Added material must read as STRUCTURE, not as a fin stuck on the
        # original.  out_h is 4.5 mm and the Femur is 39 mm deep, so a
        # full-depth growth band is a 1:9 wall -- "paint on a pig".  The band's
        # width instead ramps from the back face to the show face, giving a
        # drafted buttress, and `min_wall` is a FLOOR on that width rather than
        # a cutoff, so the thin end is a solid lip instead of vanishing.
        # Growth is ONE FLANGE, one width, one prism.
        #
        # Four shapes were tried and measured before this one:
        #   * full-depth band          -> 4.5 mm over 39 mm = a 1:9 FIN
        #   * per-layer tapered bands  -> each starts at the silhouette while
        #     the real section is smaller, so it fused to nothing: the Coupler
        #     threw away its whole 13.8 cm3 flange
        #   * per-layer bands grown from each band's own section -> the section
        #     changes with depth, so every band juts out somewhere different:
        #     a stack of SHELVES, 44 cm3 of trays
        #   * one prism pinned to the show face -> fine on the links, but a
        #     plate whose top face is recessed has its RIM lower down, so the
        #     ring had no metal to fuse to and was dropped whole
        #
        # So the prism is placed where it can actually attach: the show face
        # first, falling back to the band where the part's section is fullest.
        minw = float(sp.get("min_wall", 0.0))
        gdf = float(sp.get("grow_depth_frac", 1.0))
        GD = max(1e-6, gdf * (ZT - ZB))
        # The flange width is a SPEC value, not the part's maximum growth.
        # Max-growth made the Tibia's edge flange inherit the width of its wheel
        # boss -- 16 mm instead of 7 -- and drove 1929 mm3 into the Coupler in
        # every pose.  `Wmax` only decides whether anything grew at all.
        Wmax = max((ShPoint(*c).distance(OUT) for c in grown.exterior.coords),
                   default=0.0)
        FW = float(sp.get("flange_w", sp.get("out_h", 4.0)))

        def _flange_at(z0f, z1f):
            """The flange band for a prism spanning [z0f, z1f], or None."""
            # Ring shape from a DEEPER sample: on a part whose top face is
            # mostly recessed a shallow sample returns a few islands, and the
            # ring around those lands inland where the keep-out is dense (the
            # Side panel gave 516 mm2 on a plate worth ~2400).
            mf = _band_faces(src["solid"], [(max(ZB, z0f - 2.0 * GD), z1f)])[0]
            if mf is None:
                return None
            # Keep-outs subtracted DIRECTLY, not via the clipped layer outlines:
            # `_clip` ends with ShPoly(...exterior), which discards interior
            # holes, so a neighbour whose keep-out falls inside the growth
            # region had its hole filled straight back in.  That is how the
            # Tibia kept driving 1363 mm3 into the Coupler beside their shared
            # pivot in all three poses.
            ko_f = None
            for k, (z0, z1, gk, _m) in enumerate(layers):
                if z1 > z0f + 1e-6 and z0 < z1f - 1e-6 and k < len(_KOS)                         and _KOS[k] is not None:
                    ko_f = (_KOS[k] if ko_f is None
                            else unary_union([ko_f, _KOS[k]]))
            band = mf.buffer(FW, join_style=2).difference(mf)
            a0 = band.area
            band = band.intersection(_design)          # inside the design outline
            a1 = band.area
            if ko_f is not None:
                band = band.difference(ko_f)           # outside every keep-out
            if not kb.is_empty:
                band = band.difference(kb)             # clear of every hole
            a2 = band.area
            o = max(0.6, minw / 2.0)
            if not band.is_empty:
                band = band.buffer(-o, join_style=2).buffer(o, join_style=2)
            # Contact tested against metal inside the PRISM's OWN band.  An
            # island beside metal that exists only BELOW the prism passes a
            # deep-band test and still fuses to nothing -- that was the Side
            # panel's 4.79 cm3, sitting at Y +75..+81 past the plate edge.
            mfp = _band_faces(src["solid"], [(z0f, z1f)])[0]
            if not band.is_empty and mfp is not None:
                touch = mfp.buffer(0.2, join_style=2)
                keep = [g for g in geoms(band) if g.intersects(touch)]
                band = unary_union(keep) if keep else ShPoly()
            elif mfp is None:
                band = ShPoly()
            print(f"  flange @ z {z0f:+6.1f}..{z1f:+6.1f}: ring {a0:.0f} -> "
                  f"design {a1:.0f} -> clear {a2:.0f} -> attached {band.area:.0f} mm2")
            return None if band.is_empty else band

        flange = None
        if gdf < 0.999 and Wmax > 0.05 and FW > 0.05:
            places = [(ZT - GD, ZT)]
            # fallback: centred on the band holding the most metal, i.e. the rim
            _bs = [(ZB + i * (ZT - ZB) / NT, ZB + (i + 1) * (ZT - ZB) / NT)
                   for i in range(NT)]
            _fs = _band_faces(src["solid"], _bs)
            _ar = [(0.0 if f is None else f.area) for f in _fs]
            if _ar:
                _bi = max(range(len(_ar)), key=lambda i: _ar[i])
                _c = (_bs[_bi][0] + _bs[_bi][1]) / 2.0
                _z1 = min(ZT, _c + GD / 2.0)
                _z0 = max(ZB, _z1 - GD)
                if abs(_z0 - (ZT - GD)) > 0.5:
                    places.append((_z0, _z1))
            for _z0f, _z1f in places:
                _b = _flange_at(_z0f, _z1f)
                if _b is not None:
                    flange = (_z0f, _z1f, _b)
                    _u = unary_union([grown, _b]).buffer(0)
                    if _u.geom_type == "Polygon":
                        grown = ShPoly(_u.exterior)
                    break
        # every layer keeps the bare source outline; the flange is the only growth
        layers = [(z0, z1, ShPoly(OUT.exterior) if OUT.interiors else OUT, m)
                  for z0, z1, gk, m in layers]

    else:
        grown = _clip(grown, asmkeepout.load(PART, clear))
        layers = [(ZB, ZT, grown, None)]
        flange = None
    # `env` INTERSECTS the solid, so anything missing from `grown` is cut off the
    # part -- silently, until verify.py happens to hit a hole.  Check directly.
    # Area, not `contains`: GEOS leaves sub-mm2 residue on boundaries shared
    # between `grown` and OUT and `contains` fails on it even at 0.00 mm2 lost.
    # 1 mm2 is ~0.01% of these silhouettes; the regression this caught was 25.
    _lost = OUT.difference(grown.buffer(1e-6)).area
    if _lost > SIL_TOL:
        raise RuntimeError(
            f"{PART}: grown no longer contains the source silhouette "
            f"(missing {_lost:.2f} mm2 of {OUT.area:.0f}) -- the envelope would "
            f"cut the part.  Check the keep-out clip.")
    clip = lambda p, i=3.2: p.intersection(grown.buffer(-i, join_style=2)).difference(kb)
    gk = sp.get("gap_k", 1.0)
    fx = lambda t: X0 + t * L

    # WHERE MATERIAL MAY BE REMOVED.  Two conditions, and every deep removal
    # below is intersected with both.
    #
    # (1) min_wall of rim on the REAL edge.  `clip()` measures its margin from
    #     `grown`, which INCLUDES the flange -- and the flange is a prism over
    #     part of the depth, so at a z where it does not reach there is no metal
    #     out there at all.  Measured on the pre-Phase-3 build at x=+60: the
    #     source is solid from y -17.25 to +18.5, and the styled part had a
    #     0.5 mm-wide free-standing column at y -17.25 with TWELVE MILLIMETRES
    #     of air beside it.  `clip(cuts, 4.5)` had faithfully kept 4.5 mm clear
    #     of `grown` and left half a millimetre of real material.  That single
    #     razor is the 0.50 mm mode in the thin-wall map.  Measuring the margin
    #     from OUT instead makes the rim structural rather than incidental.
    #
    # (2) inside his RED region.  Constraint 1 is "add and remove only where
    #     identified", and his red is the CENTRAL WEB -- x -73.8..63.3,
    #     y -12.7..13.3 on a part spanning y +-19.4.  It stops well short of the
    #     rim and of both pivots on purpose: thicken the edge, hollow the middle,
    #     leave the bosses alone.  The old cuts ran to y -17, straight through
    #     the band he marked GREEN for growth -- the exact inverse of the intent.
    #
    # Surface relief is NOT bound by (2).  The accent engraving is 1.2 mm into a
    # 10 mm plate and the side-wall pockets are shallow: they are finish, not
    # removal, and confining them to the web would strip the locked GLACIER look
    # off the whole rim for no structural gain.  They are still bound by the
    # thin-wall gate, which measures what is actually left.
    # WHICH removals each condition binds is the part that needed thought.
    # Depth decides it, because that is what "removing material" means here:
    #
    #   cutouts   ZB-10 .. ZTOP+12   FULL DEPTH -- a real hole through the part
    #   pock      ZT-2.1 ..          2.1 mm into the show face
    #   wins      ftop-4.075 ..      through the RAISED frame, 0.6 mm into the part
    #   side      side_d into the side wall
    #   engraving 1.2 mm
    #
    # Only `cutouts` takes material out of the web.  The rest is surface relief
    # -- it is the GLACIER look, it is what he approved, and confining it to the
    # web would strip the finish off the whole rim for no structural gain.  A
    # first pass that bound every removal to his red zone deleted all four
    # windows outright, which is how that distinction got found.
    #
    # So: (1) binds everything, (2) binds the through-cut alone.
    MW = float(sp.get("min_wall", 3.0))
    _red = marks()[1]
    core = OUT.buffer(-MW, join_style=2)           # (1) min_wall of real rim
    web = core if _red.is_empty else core.intersection(_red)   # (2) his red
    if web.is_empty:
        web = core
    rclip = lambda p: (p.intersection(core).difference(kb)
                       if not p.is_empty else p)
    wclip = lambda p: (p.intersection(web).difference(kb)
                       if not p.is_empty else p)

    # The frame sits on the lower part of the plan and the rail on the upper.
    # Those bounds used to be absolute mm (-6.0 and +5.0), tuned on a 38.8 mm
    # tall link -- trap 10 a third time, now in Y: on a 104 mm plate they land
    # near mid-height and the frame swallows the whole lower half.  Expressed as
    # fractions of the part's own Y extent, taken so the Femur is unchanged.
    Y0, Y1 = grown.bounds[1], grown.bounds[3]
    fy = lambda t: Y0 + t * (Y1 - Y0)
    frame = (grown.buffer(-3.6, join_style=2)
             .intersection(shbox(fx(.14), Y0 - 50, fx(.86), fy(.345))).difference(kb)
             .difference(bosses.buffer(1.5)))
    frame = unary_union([g for g in geoms(frame) if g.area > 60]) if sp["frame_h"] > 0.3 else ShPoly()
    rail = (grown.buffer(-2.6, join_style=2).difference(grown.buffer(-7.2, join_style=2))
            .intersection(shbox(fx(.12), fy(.629), fx(.88), Y1 + 50)).difference(kb))
    rail = unary_union([g for g in geoms(rail) if g.area > 20]) if sp["rail_h"] > 0.3 else ShPoly()

    ps, pads = sp.get("pad_size", 1.0), []
    for i, (a_, b_) in enumerate(_slots(fx(.16), fx(.84), sp["n_pads"], 8.0 * gk)):
        ys = _yspan(grown, (a_ + b_) / 2)
        if ys is None: continue
        lo, hi = ys
        if i % 2 == 0: pads.append(trap_plan(a_, b_, lo + 1.0, lo + 1.0 + 6.0 * ps, 7.0, 1.4))
        else:          pads.append(trap_plan(a_, b_, hi - 7.5 - 5.0 * ps, hi - 7.5, 7.0, -0.9))
    pads = clip(unary_union(pads), 3.0) if pads else ShPoly()

    qs, pock = sp.get("pock_size", 1.0), []
    for i, (a_, b_) in enumerate(_slots(fx(.13), fx(.81), sp["n_pockets"], 7.0 * gk)):
        ys = _yspan(grown, (a_ + b_) / 2)
        if ys is None: continue
        lo, hi = ys
        if i % 2 == 0: pock.append(trap_plan(a_, b_, lo + 7.0, lo + 7.0 + 4.0 * qs, 6.0, 1.2))
        else:          pock.append(trap_plan(a_, b_, hi - 12.0 - 3.5 * qs, hi - 12.0, 6.0, -0.7))
    pock = rclip(clip(unary_union(pock))) if pock else ShPoly()

    cuts = []
    for a_, b_ in _slots(fx(.18), fx(.82), sp.get("n_cuts", 0), 11.0):
        ys = _yspan(grown, (a_ + b_) / 2)
        if ys is None: continue
        cuts.append(trap_plan(a_, b_, ys[0] + 4.0, ys[0] + 11.0, 6.0, 1.0))
    cutouts = wclip(clip(unary_union(cuts), 4.5)) if cuts else ShPoly()

    # THE THROUGH-CUT SIZE RULE (decision 29).  His words were "all the way
    # through would reduce part strength, so only do that in certain small
    # areas"; the operational form is a largest-inscribed-circle test.  A piece
    # that fits inside a 12 mm circle goes through; anything bigger becomes a
    # pocket that stops min_wall short of the far face, so it still reads as the
    # same feature from the show side while the back keeps a continuous skin.
    # Tested by erosion rather than by area: a long thin slot and a fat blob can
    # have the same area and are not remotely the same thing structurally.
    THRU_R = 0.5 * float(sp.get("thru_max_d", 12.0))
    cut_thru, cut_pocket = [], []
    for _g in geoms(cutouts):
        (cut_pocket if not _g.buffer(-THRU_R, join_style=2).is_empty
         else cut_thru).append(_g)
    cutouts = unary_union(cut_thru) if cut_thru else ShPoly()
    cut_pockets = unary_union(cut_pocket) if cut_pocket else ShPoly()

    wins = []
    for a_, b_ in _slots(fx(.22), fx(.78), sp["n_wins"], 8.0):
        ys = _yspan(grown, (a_ + b_) / 2)
        if ys is None: continue
        wins.append(trap_plan(a_, b_, ys[0] + 3.0, ys[0] + 9.0, 5.0, 1.0))
    wins = (rclip(unary_union(wins).intersection(frame.buffer(-3.4, join_style=2)))
            if wins and not frame.is_empty else ShPoly())

    # Side-wall pockets.  Absolute z, like split_z was, and just as untransferable:
    # on the Tibia the pair straddles mid-thickness, on the Coupler the SAME
    # numbers land against the inboard face.  When the part opts into
    # `white_depth` they become fractions of thickness below the SHOW face.  The
    # fractions are the Tibia's own, so the Tibia reproduces exactly.
    lo_pr = [trap_xz(a_, b_, *_side_band(sp, ZT, ZB, 0.407, 0.815, -8.5, 2.5), skew=4.5)
             for a_, b_ in _slots(X0 - 10, X1 + 10, sp["n_side_lo"], 14.0)]
    hi_pr = [trap_xz(a_, b_, *_side_band(sp, ZT, ZB, 0.185, 0.444, 1.5, 8.5),
                     skew=4.0, taper_end=False)
             for a_, b_ in _slots(X0 - 6, X1 + 6, sp["n_side_hi"], 16.0)]

    # Raised features may only sit where the part actually reaches the show
    # face; see _top_face().  Anything outside that region would float.
    _tf = src.get("TOPF")
    if _tf is None:
        _tf = _top_face(src["solid"], ZT)
        src["TOPF"] = _tf if _tf is not None else ShPoly()
        _tf = src["TOPF"]
    if not _tf.is_empty:
        frame = frame.intersection(_tf)
        rail = rail.intersection(_tf)
        pads = pads.intersection(_tf)
        pock = pock.intersection(_tf) if not pock.is_empty else pock

    strip, blocks, ring, channel = ACC.accents(
        sp, grown=grown, kb=kb, clip=clip, yspan=_yspan, slots=_slots, gk=gk,
        joints=[(A, 19.0), (C, 19.0)])

    ct = sp.get("collar_t", 3.4)
    collars = unary_union([ShPoint(*p).buffer(17.0 + ct, 96).difference(ShPoint(*p).buffer(17.0, 96))
                           for p in (A, C)])
    if not channel.is_empty:
        collars = unary_union([collars, channel])
    collars = collars.intersection(grown).difference(kb)

    # min_wall BETWEEN REMOVALS, not just against the silhouette.  Keeping every
    # cut min_wall clear of the outer edge is necessary and nowhere near
    # sufficient: the side-wall pockets eat `side_d` inward from +-Y while the
    # slots eat outward from the middle, and neither knows the other exists.
    # Measured at x=+40 on the first rebuild: the side pocket floor landed at
    # y -12.3 and the slot wall at y -11.6, leaving a SEVEN-TENTHS of a
    # millimetre web between two features that each satisfied their own rule.
    # That pair is 1540 mm2 of the thin map and the single largest patch on the
    # part.
    #
    # The rim wins the argument.  Two ways to break the sandwich: merge the two
    # removals into one wider opening, or pull one of them back.  Merging would
    # delete the rib -- and the rib is inside the band he marked GREEN to
    # THICKEN, so it is the last thing that should go.  The side pocket is the
    # one that yields, and only where it actually crowds something.
    side_guard = (unary_union([g for g in (pock, wins, cutouts, cut_pockets)
                               if not g.is_empty]).buffer(MW, join_style=2)
                  if any(not g.is_empty for g in (pock, wins, cutouts, cut_pockets))
                  else ShPoly())

    return dict(spec=sp, grown=grown, layers=layers, flange=flange, OUT=OUT, KEEP=KEEP, kb=kb, knee=bosses, wheel=bosses,
                frame=frame, rail=rail, pads=pads, pock=pock, wins=wins, cutouts=cutouts,
                cut_pockets=cut_pockets, side_guard=side_guard,
                lo_pr=lo_pr, hi_pr=hi_pr, ring=ring, strip=strip, blocks=blocks,
                collars=collars, ZT=ZT, ZB=ZB,
                ZTOP=ZT + max(sp["frame_h"], sp["rail_h"], sp["pad_h"]) + 1.0)


def build(sp, verbose=True):
    _face(sp)
    P = plan(sp)
    src = source(); solid = src["solid"]
    grown, KEEP = P["grown"], P["KEEP"]
    ZT, ZB, ZTOP = P["ZT"], P["ZB"], P["ZTOP"]
    CH = sp["chamfer"]
    say = print if verbose else (lambda *a, **k: None)
    X0, _, X1, _ = grown.bounds
    fx = lambda t: X0 + t * (X1 - X0)
    slab = lambda z0, z1: Pos(0, 0, (z0 + z1) / 2) * Box(900, 700, z1 - z0)

    # One envelope and one addition per z layer.  With a single layer this is
    # exactly what it always was.
    layers = P.get("layers") or [(ZB, ZT, grown, None)]
    env = add_solid = None
    for _i, (_z0, _z1, _gk, _mk) in enumerate(layers):
        _top, _bot = _i == len(layers) - 1, _i == 0
        _pz = None
        if CH > 0.05:
            try:
                _pz = prism(_gk, _z0 + (CH if _bot else 0.0),
                            _z1 - (CH if _top else 0.0))
                if _top:
                    _pz = union(_pz, Pos(0, 0, _z1 - CH) * frustum(_gk, CH, 45))
                if _bot:
                    _pz = union(_pz, Pos(0, 0, _z0 + CH)
                                * mirror(frustum(_gk, CH, 45), about=Plane.XY))
            except Exception as e:
                say(f"  chamfer taper failed on layer {_i} "
                    f"({type(e).__name__}); square-edged")
                _pz = None
        if _pz is None:
            _pz = prism(_gk, _z0, _z1)
        # Grow the envelope layer a hair past its own z range at INTERIOR
        # boundaries.  add_solid and env are built from the same layer z values,
        # so without this their horizontal faces are exactly coincident, and
        # `add_solid & env` across coincident faces is where OCC produces
        # degenerate geometry: RobotMount came out with a body whose booleans
        # all lied -- the filament bodies summed to 14 cm3 of a 127 cm3 part and
        # verify reported material inside holes that nothing can reach.
        # EPS is far below the 1.0 mm assembly clearance, so nothing escapes.
        _lo = _z0 - (0.0 if _bot else EPS_Z)
        _hi = _z1 + (0.0 if _top else EPS_Z)
        if (_lo, _hi) != (_z0, _z1):
            _pz = union(_pz, prism(_gk, _lo, _hi))
        env = _pz if env is None else union(env, _pz)

        # Layer-by-layer growth is OFF: every layer carries the bare source
        # outline, so this is empty and the flange below is the only addition.
        # Growing each layer from its own cross-section produced a stack of
        # shelves, because the section changes with depth.
        _a = _gk.difference(P["OUT"]).buffer(0.25)
        if not KEEP.is_empty:
            _a = _a.difference(KEEP)
        if not _a.is_empty:
            _ap = prism(_a, _z0, _z1)
            add_solid = _ap if add_solid is None else union(add_solid, _ap)
    # The flange: ONE prism spanning its own z range, the only added material.
    _fl = P.get("flange")
    if _fl is not None:
        _fz0, _fz1, _fband = _fl
        _fp = prism(_fband, _fz0, _fz1)
        if _fp is not None:
            env = _fp if env is None else union(env, _fp)
            add_solid = _fp if add_solid is None else union(add_solid, _fp)
            say(f"  flange {_fband.area:.0f} mm2 over z {_fz0:.1f}..{_fz1:.1f} "
                f"({_fz1 - _fz0:.1f} mm deep)")

    # The envelope bounds the ADDITION; it is never applied to the source.
    # Written the other way -- union(solid, add) & env -- the 4.4 mm chamfer
    # bevels the real part wherever growth is thinner than the chamfer, which
    # cost 14-21 cm3 per part.  That was invisible to a whole-part volume delta
    # (the Coupler was net BIGGER while losing a fifth of its original metal)
    # and invisible to an openings-only check, because no bore happened to sit
    # where the bevel ran.  This ordering makes "material is only ever ADDED
    # outside the original silhouette" true by construction rather than by
    # tolerance, and the assert below keeps it that way.
    _add = (add_solid & env) if add_solid is not None else None
    if _add is None or _add.volume < 1.0:
        body = solid
    else:
        # ONE fuse, not a chain.  shputil.union() fuses terms one at a time, and
        # on the Tibia that produced a body with the correct VOLUME (276 585 mm3)
        # whose topology then broke the NEXT boolean: `solid - body` came back as
        # the whole solid instead of ~0, which reads exactly like the envelope
        # having eaten the part.  A single fuse of the whole compound, then
        # clean(), behaves.  The volume guard stays because a fuse can only ever
        # ADD volume, so a drop is proof a term was dropped (trap 6).
        body = solid.fuse(_add).clean()
        if body.volume < solid.volume - 1.0:
            raise RuntimeError(
                f"{PART}: fusing the addition LOST volume "
                f"({solid.volume:.0f} -> {body.volume:.0f} mm3) -- a term was dropped.")
    _lost = solid - body
    _lv = 0.0 if _lost is None else _lost.volume
    if _lv > 1.0:
        raise RuntimeError(
            f"{PART}: the envelope removed {_lv:.1f} mm3 of the source solid "
            f"before a single feature was cut -- material must only be ADDED.")

    # A fuse can only ever ADD volume, so a drop is proof the union dropped a
    # term.  `shputil.union()` collapses a multi-solid first argument via
    # `solids()[0]` and can lose the body that way -- it reduced one Femur build
    # from ~130 cm3 to 2.05.  Losing the raised features is cosmetic; losing the
    # body ships an unprintable part, so fall back rather than trust it.
    _b0 = body
    body = union(body,
                 raised(P["frame"], ZT - CH - 1.0, sp["frame_h"] + 1.0, draft=12) if sp["frame_h"] > .3 else None,
                 raised(P["rail"],  ZT - CH - 0.8, sp["rail_h"]  + 0.8, draft=10) if sp["rail_h"]  > .3 else None,
                 raised(P["pads"],  ZT - CH - 0.8, sp["pad_h"]   + 0.8, draft=16) if sp["pad_h"]   > .3 else None)
    if body is None or body.volume < _b0.volume * 0.98:
        say(f"  raised-feature union collapsed the body "
            f"({_b0.volume/1000:.2f} -> {(body.volume/1000 if body else 0):.2f} cm3); "
            f"keeping it plain")
        body = _b0

    inset = None
    if not P["pock"].is_empty and sp["pocket_d"] > 0.1:
        body = body - prism(P["pock"], ZT - sp["pocket_d"], ZTOP + 10)
        inset = prism(P["pock"].buffer(0.25), ZB - 20, ZTOP + 20)
    if not P["wins"].is_empty and sp["win_d"] > 0.1:
        ftop = ZT - CH - 1.0 + sp["frame_h"] + 1.0
        body = body - prism(P["wins"], ftop - sp["win_d"], ZTOP + 12)
        w = prism(P["wins"].buffer(0.25), ZB - 20, ZTOP + 20)
        inset = w if inset is None else union(inset, w)
    if not P["cutouts"].is_empty:
        body = body - prism(P["cutouts"], ZB - 10, ZTOP + 12)
    # The oversized share of the same feature, stopped min_wall short of the far
    # face so it reads the same from the show side without holing the part
    # through.  Decision 29's size rule split them in plan(); see THRU_R there.
    if not P["cut_pockets"].is_empty:
        _mw = float(sp.get("min_wall", 3.0))
        body = body - prism(P["cut_pockets"], ZB + _mw, ZTOP + 12)
        say(f"  {P['cut_pockets'].area:.0f} mm2 of cutout was too big to go "
            f"through (> {sp.get('thru_max_d', 12.0):g} mm circle); pocketed to "
            f"leave a {_mw:g} mm floor")
    if sp["side_d"] > 0.1 and (P["lo_pr"] or P["hi_pr"]):
        # A SIDE CUT MUST BREAK THROUGH, OR STAY min_wall CLEAR.  Never between.
        #
        # `wall_shell(grown, t)` bounds the cut by `grown` on the OUTSIDE, and
        # `grown` is a design outline, not the material boundary: `simplify()`
        # moves it, the flange moves it again, and the source can locally stick
        # out past it.  Wherever it sits inside the real wall the cut stops
        # short and leaves a skin.  Measured at x=+44.59: a 3.4 mm groove into
        # the +Y wall ending 0.6 mm below the surface, over the full z band of
        # the pocket.  Trap 11 is the same defect in z -- "a cut that stops
        # short of the surface leaves a razor" -- and it was only ever fixed
        # there.
        #
        # Taking the outer bound WELL OUTSIDE the part removes the failure mode
        # rather than tuning it: the cut now always reaches open air, so no skin
        # can survive.  Depth is set on the inside instead, and measured from
        # OUT, which is the one outline that IS the material.
        shell = (prism(grown.buffer(4.0, join_style=2), ZB - 1, ZTOP + 1)
                 - prism(P["OUT"].buffer(-sp["side_d"], join_style=2),
                         ZB - 1, ZTOP + 1))
        cs = [side_solid(p, -1) for p in P["lo_pr"]] + [side_solid(p, +1) for p in P["hi_pr"]]
        sc = cs[0]
        for c in cs[1:]: sc = union(sc, c)
        sub = (shell & sc)
        if not P["kb"].is_empty:
            sub = sub - prism(P["kb"], ZB - 20, ZTOP + 20)
        # Keep the side pockets min_wall clear of every other removal, so the
        # two cannot sandwich the rim into a razor.  See `side_guard` in plan().
        if not P["side_guard"].is_empty:
            sub = sub - prism(P["side_guard"], ZB - 20, ZTOP + 20)
        body = body - sub
    say(f"styled body {body.volume/1000:.2f} cm3, {len(body.faces())} faces")

    # White is the cap at the SHOW face, down to the split plane.  The source is
    # mirrored so the show face is always ZT, which is what lets one rule cover
    # both orientations.  The bands straddle the split, 4.5 mm of white pushed
    # below it and 14.5 mm of graphite pushed above -- constants shared by both
    # locked parts, so measuring them from SZ reproduces each exactly.
    # Accent and collar geometry first: the colour split has to be MEASURED, and
    # it cannot be measured until everything that eats into white is in hand.
    acc_poly = unary_union([g for g in (P["ring"], P["strip"], P["blocks"]) if not g.is_empty])
    eng = unary_union([g for g in (P["strip"], P["blocks"]) if not g.is_empty])
    if not eng.is_empty:
        body = body - prism(eng, ZT - 1.2, ZTOP + 10)
    # Nothing below this point adds or removes geometry -- the colour split only
    # partitions what is here -- so this is where constraint 3 can finally be
    # enforced.  See _drop_detached().
    body = _drop_detached(body, say)
    groove = prism(acc_poly, ZB - 20, ZTOP + 20) if not acc_poly.is_empty else None
    collars = prism(P["collars"], ZB - 20, ZTOP + 20) if not P["collars"].is_empty else None
    full_y = lambda p: Pos(0, 200, 0) * side_solid(p, -1, reach=400)

    def _colour(SZ):
        """Split the body at plane SZ.  White is the cap at the show face; the
        trapezoid bands straddle the plane, white pushed 4.5 mm below it and
        graphite 14.5 mm above."""
        wz = slab(SZ, ZTOP + 20)
        for p in [trap_xz(a_, b_, SZ - 4.5, SZ, skew=6)
                  for a_, b_ in [(fx(.02), fx(.24)), (fx(.50), fx(.72))]]:
            wz = union(wz, full_y(p))
        gz = None
        for p in [trap_xz(a_, b_, SZ, SZ + 14.5, skew=6, taper_end=False)
                  for a_, b_ in [(fx(.26), fx(.48)), (fx(.74), fx(.96))]]:
            t = full_y(p)
            gz = t if gz is None else union(gz, t)
        if gz is not None:
            wz = wz - gz
        cap = body & wz
        w = cap
        if groove is not None:  w = w - groove
        if inset is not None:   w = w - inset
        if collars is not None: w = w - collars
        bl = (cap & groove) if groove is not None else None
        gr = body - w if bl is None else body - w - bl
        return w, bl, gr, w.volume + gr.volume + (bl.volume if bl else 0)

    target = sp.get("white_share")
    if target is None:
        SZ = _split_z(sp, ZT, ZB)
        white, blue, graph, tot = _colour(SZ)
    else:
        # SOLVE for the plane rather than guess it.  Depth does not map to
        # volume: white_depth 0.72 gave the Coupler 42% white and the Femur 74%,
        # because the share depends on how the cross-section varies through the
        # part.  White share falls monotonically as the plane rises, so bisect.
        # This is the `white_share` dial TOURNAMENT.md deferred until the
        # geometry settled -- it is what makes one locked look transfer to a
        # part whose proportions nobody tuned by hand.
        # Keep the BEST result, not the last.  The trapezoid bands move with the
        # split plane, so white share is not strictly monotonic in SZ -- the
        # Side panel went 50.1% then 47.6% as the plane dropped -- and a plain
        # bisection can walk away from a better answer it already had.
        lo, hi = ZB, ZT
        best = None
        for _ in range(int(sp.get("share_iters", 7))):
            SZ = (lo + hi) / 2
            white, blue, graph, tot = _colour(SZ)
            got = white.volume / tot if tot else 0.0
            say(f"  split {SZ:+7.2f} -> white {100*got:4.1f}%  (target {100*target:.0f}%)")
            if best is None or abs(got - target) < best[0]:
                best = (abs(got - target), white, blue, graph, tot, got, SZ)
            if abs(got - target) < 0.015:
                break
            if got > target:
                lo = SZ          # too white: raise the plane
            else:
                hi = SZ
        _, white, blue, graph, tot, got, SZ = best
        if abs(got - target) > 0.03:
            say(f"  note: {100*target:.0f}% white is not reachable on this part "
                f"-- best {100*got:.1f}% at split {SZ:+.2f}; the accent, collars "
                f"and pocket insets claim the remainder")
    say(f"white {white.volume/1000:6.2f} ({100*white.volume/tot:.0f}%) | "
        f"accent {(blue.volume/1000 if blue else 0):5.2f} | "
        f"graphite {graph.volume/1000:6.2f} ({100*graph.volume/tot:.0f}%)")
    # The three filament bodies MUST partition the styled body.  They are built
    # by boolean subtraction, and a subtraction that quietly fails leaves volume
    # in none of them -- RobotMount came out with 14.12 cm3 of filament against a
    # 126.87 cm3 body, which would have printed as fragments.  Nothing downstream
    # would have caught it: verify.py checks the fused body, and the 3MF is
    # whatever it is handed.
    if abs(tot - body.volume) > max(10.0, 0.01 * body.volume):
        raise RuntimeError(
            f"{PART}: the filament bodies do not add up to the part -- "
            f"{tot/1000:.2f} cm3 of white+graphite+accent against a "
            f"{body.volume/1000:.2f} cm3 body. A colour boolean failed; the 3MF "
            f"would be missing {100*(1-tot/body.volume):.0f}% of the part.")
    if SHOW_FACE == "-Z":
        # back to the exported orientation, so every hole is where it started
        # and the part still drops into the assembly.
        body, white, graph, blue = (_flip(body), _flip(white),
                                    _flip(graph), _flip(blue))
        say(f"  mirrored back to source orientation (show face local -Z)")

    # WELD HERE, not in the exporter.  lib/stepcolor.py welds on its way out, so
    # only the coloured STEP was ever sound; the per-filament print STEPs and
    # the 3MF still carried the pinch edges and still shattered in SolidWorks.
    # Welding the bodies themselves makes every downstream artifact clean and
    # leaves stepcolor's own pass a no-op.  A pinch is also a knife-edge
    # self-contact carrying no load, which is exactly what constraint 4 rules
    # out, so this is the mechanical answer as well as the topological one.
    # It runs AFTER the partition check above, because the 0.06 mm rods add a
    # few hundredths of a percent and that check is exact.
    # Wrapping the welded TopoDS shape back up needs the CONCRETE class.
    # `Shape.cast()` looks like the obvious call and returns None for the
    # compound the weld produces -- silently, so the failure surfaces three
    # frames later inside export_step as "'NoneType' has no attribute
    # 'wrapped'", which names nothing.  Pick the class from the shape type.
    import manifold as _mf
    from build123d import Compound as _B3DCompound, Solid as _B3DSolid
    from OCP.TopAbs import TopAbs_SOLID

    def _wrap(ts):
        return (_B3DSolid(ts) if ts.ShapeType() == TopAbs_SOLID
                else _B3DCompound(ts))

    _welded = {}
    for _nm, _b in (("body", body), ("white", white),
                    ("graphite", graph), ("accent", blue)):
        if _b is None or _b.volume < 1.0:
            _welded[_nm] = _b
            continue
        _w, _left, _dv = _mf.weld_nonmanifold(_b.wrapped, say=lambda *a: None)
        if _left:
            say(f"  {_nm}: {_left} non-manifold edge(s) SURVIVED the weld")
        if _dv:
            say(f"  {_nm}: welded pinches, {_dv:+.3f} mm3")
        _welded[_nm] = _wrap(_w) if _dv or _left else _b
    body, white, graph, blue = (_welded["body"], _welded["white"],
                                _welded["graphite"], _welded["accent"])
    return dict(body=body, white=white, graphite=graph, accent=blue, plan=P)


def export(res, tag="arctic"):
    from render3d import tessellate
    from export3mf import write_3mf
    import stepcolor
    import spec as _spec
    paths.ensure_out()
    pdir = paths.print_dir(PART, tag, make=True)
    paths.styled_dir(PART, tag, make=True)
    pal = _spec.PALETTES[res["plan"]["spec"].get("palette", "arctic_lt")]
    col = {"white": pal["white"], "graphite": pal["dark"], "accent": pal["accent"]}
    parts, coloured, fil = [], [], {"graphite": 1, "white": 2, "accent": 3}
    for n, f in fil.items():
        b = res[n]
        if b is None or b.volume < 1: continue
        export_step(b, os.path.join(pdir, f"{PART}_{tag}_{n}.step"))
        coloured.append((n, b, col[n]))
        V, T, _ = tessellate(b, 0.08); parts.append((n, V, T, f))
    export_step(res["body"], paths.styled_step(PART, tag))
    # A fused body is ONE solid and can only ever show one colour in CAD, which
    # is what it is for.  This is the file to open to look at the part.
    try:
        stepcolor.write(coloured, paths.colour_step(PART, tag))
    except Exception as e:
        print(f"  coloured STEP failed ({type(e).__name__}: {e})")
    write_3mf(parts, os.path.join(pdir, f"{PART}_{tag}.3mf"), name=f"{PART} v4 {tag}")
    print(f"wrote 3MF + STEPs to {pdir}")


if __name__ == "__main__":
    sf = os.path.join(paths.SPECS, "femur.json")
    sp = spec_mod.derive()
    if os.path.exists(sf):
        saved = json.load(open(sf))
        sp = spec_mod.derive(**{k: v for k, v in saved.items()
                                if k in ("aggression", "density", "relief", "scale",
                                         "accent", "outline", "organic", "waist",
                                         "language", "palette", "mix")})
        sp.update(saved)
    print(f"spec: {spec_mod.label(sp)}")
    # tag the output after the locked concept, not after a palette default --
    # otherwise the GLACIER build ships as "Tibia_arctic.3mf" and the next
    # person reads the filename instead of the spec
    export(build(sp), tag=sp.get("tag", "arctic"))
