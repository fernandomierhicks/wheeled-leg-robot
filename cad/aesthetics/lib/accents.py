"""Accent vocabulary, shared by every part recipe.

One contract: given a part's grown outline and its keep-out set, return the
accent polygons.  Every style is driven by the part's own geometry -- the
clearance search finds where a run fits, the bands follow the part's own
contour -- so nothing is hard-coded to one part and a new recipe gets the whole
vocabulary for free.

  spine smooth dual dashed jog        contour-following strokes
  traps hex chev hatch bracket ticks  discrete repeated elements
  stepbar circuit axial               routed and straight runs
  wrap jointring led plate            end panel, joint rings, light strip, panel
"""
import numpy as np
from shapely.geometry import Polygon as ShPoly, Point as ShPoint, box as shbox, LineString
from shapely.ops import unary_union
from shputil import geoms
from feat import trap_plan, band_along


# The last routed circuit path, so a board can draw alternative GREY traces on
# exactly the line the blue already follows instead of re-deriving the routing
# and drifting from it.  Diagnostic only -- nothing in the build reads it.
LAST = {}


def _route(xa, xb, lane, n_jog, amp, jw, rng):
    """A routed polyline: straight runs joined by diagonal doglegs.

    The same vocabulary as the blue -- straight, jog, straight, never a square
    corner -- but its own span, lane and jog positions.  `rng` is seeded from
    the spec so a chosen arrangement rebuilds exactly instead of re-rolling.
    """
    pts, y = [(xa, lane)], lane
    for xj in np.linspace(xa + 0.20 * (xb - xa), xb - 0.26 * (xb - xa), n_jog):
        dy = amp * float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.55, 1.45))
        pts.append((xj, y))
        pts.append((xj + jw, y + dy))
        y += dy
    pts.append((xb, y))
    return pts


def _trace(pts, w):
    """Width `w` along `pts`, flat caps and MITRE joins -- a rounded join reads
    as a drawn stroke, a mitred one reads as a routed track."""
    return LineString(pts).buffer(w / 2.0, cap_style=2, join_style=2)


def grey_trace(sp, *, grown, kb, path=None, avoid=None):
    """GRAPHITE AS A DRAWN SHAPE.  Decision 32, variant B.

    Until now graphite was never drawn at all: the colour split was a Z plane,
    white the cap above it, graphite whatever was left.  So it had no path, no
    width and no direction, it read as camouflage, and on the back it landed as
    whatever rectangle the plane sliced.  This draws it instead.

    THREE INDEPENDENT RUNS, each with its own span and lane.  They share the
    blue's dogleg vocabulary and overlap its territory, but none of them tracks
    it -- a constant offset was round 1, and his note was that it followed the
    blue "almost exactly ... it needs to be a little bit more random".  The
    rhythm being copied is the old feature bands', which start and stop at
    different places; the relationship to the accent is deliberately loose.

    `path` only supplies the accent's mean lane, so the runs sit in the same
    region of the part.  Nothing else is taken from it.
    """
    seed = int(sp.get("grey_seed", 7))
    tt = float(sp.get("trace_t", 3.0))
    gx0, gy0, gx1, gy1 = grown.bounds
    fx = lambda t: gx0 + t * (gx1 - gx0)
    lane0 = (float(np.mean([p[1] for p in path])) if path
             else (gy0 + gy1) / 2.0)
    amp, jw = float(sp.get("jog_amp", 4.5)), float(sp.get("jog_w", 7.0))
    r = np.random.default_rng(seed)
    # Lane offsets are FRACTIONS OF THE PART'S OWN HALF-WIDTH, never millimetres.
    # Trap 12 in TOURNAMENT.md, already paid for in all three axes: an absolute
    # millimetre does not transfer between parts.  Written as mm these read -11,
    # +7.5, -8 on the Femur's 23.4 half-width, and the Coupler is 25.9 -- the
    # runs sat too far inboard there and the trace came out 13% of the
    # silhouette against the Femur's 22%.  As fractions they land in the same
    # place on both.  x spans are already fractions of length.
    half = max(1e-6, (gy1 - gy0) / 2.0)
    runs = sp.get("grey_runs") or [[.12, .52, -0.470, 2.8],
                                   [.38, .86, +0.321, 2.2],
                                   [.60, .93, -0.342, 3.2]]
    segs = [_trace(_route(fx(a), fx(b), lane0 + dy * half, 2, amp, jw, r), tt * wf)
            for a, b, dy, wf in runs]
    g = unary_union(segs).intersection(grown.buffer(-2.2, join_style=2))
    if not kb.is_empty:
        g = g.difference(kb)
    # Keep the runs off the pivot bosses.  The back half of the inlay runs the
    # full depth so it always reaches the real back surface, which over a boss
    # TUBE would mean painting a stripe down 29 mm of cylinder; and the joints
    # already have their own accent ring, so the trace has no business there.
    if avoid is not None and not avoid.is_empty:
        g = g.difference(avoid)
    return g


def accents(sp, *, grown, kb, clip, yspan, slots, joints, gk=1.0):
    """-> (strip, blocks, ring, channel).

    `joints` is [((x, y), radius), ...]; the LAST one terminates the accent.
    `clip`, `yspan` and `slots` are the host recipe's helpers, passed in so this
    module stays purely geometric.
    """
    _yspan, _slots = yspan, slots
    # `wrap` and `plate` span a REGION of the part, so their defaults have to be
    # fractions of this part's own length -- absolute millimetres would be one
    # part's coordinates leaking into another's.
    GX0, _, GX1, _ = grown.bounds
    GL = GX1 - GX0
    fx = lambda t: GX0 + t * GL
    aw = sp.get("acc_spine", 1.8)
    live = sp["aggression"] > 1 and aw > 0.15
    style = sp.get("accent_style", "spine")
    channel, blocks, strip = ShPoly(), ShPoly(), ShPoly()
    ring = ShPoly()
    if live and sp.get("accent_ring", 1) and joints:
        (wx, wy), wr = joints[-1]           # the terminating joint, e.g. the wheel
        ring = (ShPoint(wx, wy).buffer(wr, 96)
                .difference(ShPoint(wx, wy).buffer(wr - aw, 96)).intersection(grown))

    # A band drawn along `grown` inherits every step of the trapezoid creases,
    # which is what made the line read as jagged.  Taking the steps out of the
    # path -- not out of the part -- is what organises it.
    gs = grown.simplify(3.4).buffer(3.4, join_style=1).buffer(-3.4, join_style=1)

    def upper(i0, i1, x0=None, x1=None):
        x0 = fx(0.293) if x0 is None else x0
        x1 = fx(0.848) if x1 is None else x1
        b = band_along(gs, i0, i1, x0, x1).intersection(shbox(x0, 1.5, x1, 90))
        b = b.intersection(grown.buffer(-2.6, join_style=2)).difference(kb)
        return unary_union([g for g in geoms(b) if g.area > 3.0])

    def straight(w0, w1, xa=None, xb=None, lo=-20.0, hi=18.0):
        xa = fx(0.307) if xa is None else xa
        xb = fx(0.837) if xb is None else xb
        """Straight tapered run, placed where it has the most clearance.  A
        straight line cannot weave around obstacles the way a per-station
        corridor search does, which is exactly why it reads as organised."""
        free = grown.buffer(-3.0, join_style=2).difference(kb)
        xs = np.linspace(xa, xb, 80)
        tap = [max(0.5, w0 * (1 - (x - xa) / (xb - xa)) + w1 * (x - xa) / (xb - xa)) for x in xs]
        xm, best = (xa + xb) / 2, None
        for off in np.linspace(lo, hi, 39):
            for slope in np.linspace(-0.14, 0.14, 15):
                t_, b_ = [], []
                for x, hw in zip(xs, tap):
                    ym = off + slope * (x - xm)
                    t_.append((x, ym + hw)); b_.append((x, ym - hw))
                poly = ShPoly(t_ + b_[::-1])
                if not poly.is_valid: poly = poly.buffer(0)
                if poly.is_empty: continue
                ins = poly.intersection(free).area
                sc = ins - 4.0 * (poly.area - ins)
                if best is None or sc > best[0]: best = (sc, poly)
        return best[1] if best else ShPoly()

    def biggest(g):
        return max(g.geoms, key=lambda q: q.area) if g.geom_type == "MultiPolygon" else g

    if live:
        if style == "spine":                    # as built: contour band + blocks
            band = band_along(grown, 4.2, 4.2 + aw, fx(0.293), fx(0.848))
            band = band.intersection(shbox(fx(0.293), 1.5, fx(0.848), 90)).difference(kb)
            strip = unary_union([g for g in geoms(band) if g.area > 3.0])
            bs = sp.get("acc_block", 7.0)
            if not strip.is_empty and bs > 0.4:
                bl = []
                for a_, b_ in _slots(fx(0.337), fx(0.815), sp.get("n_acc", 4), 15.0 * gk):
                    xc = (a_ + b_) / 2
                    ys = _yspan(strip, xc)
                    if ys is None: continue
                    yc = (ys[0] + ys[1]) / 2
                    w = min(b_ - a_, bs * 1.45)
                    bl.append(trap_plan(xc - w / 2, xc + w / 2, yc - bs / 2, yc + bs / 2, 2.0, 0.0))
                blocks = clip(unary_union(bl), 2.2) if bl else ShPoly()
        elif style == "smooth":                 # same path, steps taken out
            strip = upper(4.2, 4.2 + aw)
        elif style == "dual":                   # two thin parallel runs
            t = max(1.1, aw * 0.62)
            strip = unary_union([upper(4.0, 4.0 + t), upper(4.0 + t + 2.6, 4.0 + 2 * t + 2.6)])
        elif style == "dashed":                 # one path, cut into even dashes
            base = upper(4.2, 4.2 + aw)
            cuts = [shbox(x0, -90, x1, 90) for x0, x1 in _slots(fx(0.307), fx(0.830), 7, 9.0)]
            strip = unary_union([base.intersection(c) for c in cuts]) if cuts else base
            strip = unary_union([g for g in geoms(strip) if g.area > 2.0])
        elif style == "jog":                    # straight runs, one deliberate step
            hw = max(0.9, aw * 0.85)
            yA, yB = None, None
            base = upper(4.2, 4.2 + aw)
            for xq, key in ((fx(0.43), "A"), (fx(0.75), "B")):
                ys = _yspan(base, xq)
                if ys is None: continue
                if key == "A": yA = (ys[0] + ys[1]) / 2
                else:          yB = (ys[0] + ys[1]) / 2
            if yA is not None and yB is not None:
                ctr = [(fx(0.315), yA), (fx(0.565), yA), (fx(0.64), yB), (fx(0.83), yB)]
                up_ = [(x, y + hw) for x, y in ctr]
                dn_ = [(x, y - hw) for x, y in ctr]
                strip = ShPoly(up_ + dn_[::-1]).buffer(0)
        elif style == "traps":
            # DISCRETE elongated trapezoids, not a stroke.  A continuous line
            # "just looks like a line"; the same trapezoid vocabulary as the
            # cutouts reads as deliberate and futuristic.  They all share one
            # axis and one skew direction, which is what stops them reading as
            # the scattered slivers this started out as.
            th = sp.get("trap_h", 6.5)
            # Keep the row on the UPPER panel.  Unconstrained, the clearance
            # search puts it in the keel, where it tangles with the dark cutouts
            # and stops reading as a deliberate row.
            zl, zh = ((2.0, 20.0) if sp.get("trap_zone", "upper") == "upper"
                      else (-20.0, 18.0))
            band = straight(th, th, lo=zl, hi=zh)
            n = max(1, int(sp.get("n_traps", 4)))
            sk, grad = sp.get("trap_skew", 10.0), sp.get("trap_grad", 0.0)
            out_ = []
            for i, (a_, b_) in enumerate(_slots(fx(0.322), fx(0.823), n, 11.0)):
                xc = (a_ + b_) / 2
                ys = _yspan(band, xc)
                if ys is None: continue
                f = 1.0 - grad * (i / max(1, n - 1))
                half = (b_ - a_) / 2 * max(0.25, f)
                if half < 3.0: continue
                # clamp the skew to the trapezoid's own length -- a fixed skew
                # wider than the shape inverts it, and the guard that used to
                # catch that silently dropped the whole row instead
                ske = min(sk, half * 0.75)
                out_.append(trap_plan(xc - half, xc + half, ys[0], ys[1], ske, 0.0))
            strip = unary_union(out_) if out_ else ShPoly()
        elif style == "wrap":
            # The concept limbs carry blue as a PANEL wrapping the end of the
            # link, not as a line along it.  Chamfered leading edge so it reads
            # as a fitted cover rather than a paint mask.
            xw = sp.get("wrap_x", fx(0.74))
            ch_ = sp.get("wrap_ch", 16.0)
            ys = _yspan(grown, xw)
            if ys is not None:
                lo, hi = ys
                xe = GX1 + 40
                strip = clip(ShPoly([(xw + ch_, hi + 6), (xe, hi + 6), (xe, lo - 6),
                                     (xw + ch_ * 0.35, lo - 6), (xw, (lo + hi) / 2)]), 2.6)
        elif style == "jointring":
            # accent reserved for the joints, which is what their own palette
            # note says it is for: light strips, joint rings, small panels
            rs = []
            for (px, py), r0 in joints:
                for k in range(int(sp.get("n_rings_j", 2))):
                    rr = r0 + k * 3.4
                    rs.append(ShPoint(px, py).buffer(rr, 96)
                              .difference(ShPoint(px, py).buffer(rr - aw, 96)))
            strip = clip(unary_union(rs), 1.6)
        elif style == "led":
            # a light strip: thin bright core sunk in a wide dark channel
            tt = sp.get("trace_t", 2.0)
            band = straight(tt, tt, lo=2.0, hi=20.0)
            strip = clip(band, 2.6)
            cwd = sp.get("comp_w", 7.0)
            if cwd > 0.1 and not strip.is_empty:
                channel = clip(strip.buffer(cwd / 2, join_style=2).difference(strip), 2.2)
        elif style == "plate":                  # one large chamfered panel
            xa_, xb_ = sp.get("plate_x0", fx(0.34)), sp.get("plate_x1", fx(0.74))
            ys = _yspan(grown, (xa_ + xb_) / 2)
            if ys is not None:
                lo, hi = ys
                y1 = hi - sp.get("plate_inset", 5.0)
                y0 = y1 - sp.get("plate_h", 17.0)
                c = 7.0
                strip = clip(ShPoly([(xa_ + c, y1), (xb_ - c, y1), (xb_, y1 - c),
                                     (xb_, y0 + c), (xb_ - c, y0), (xa_ + c, y0),
                                     (xa_, y0 + c), (xa_, y1 - c)]), 2.8)
        elif style == "circuit":
            # ONE routed trace: long straight runs that jog diagonally to a new
            # level and carry on, like a circuit track.  Optionally shadowed by
            # a wider GREY trace running parallel over part of its length, so
            # the blue accompanies the grey instead of floating on its own.
            tt = sp.get("trace_t", 3.0)
            zl, zh = ((2.0, 20.0) if sp.get("trap_zone", "upper") == "upper"
                      else (-20.0, 18.0))
            band = straight(max(tt * 2.4, 9.0), max(tt * 2.4, 9.0), lo=zl, hi=zh)
            if not band.is_empty:
                bx0, _, bx1, _ = band.bounds

                def cy(x):
                    ys = _yspan(band, x)
                    return None if ys is None else (ys[0] + ys[1]) / 2

                nj = max(1, int(sp.get("n_jogs", 3)))
                amp, jw = sp.get("jog_amp", 4.5), sp.get("jog_w", 7.0)
                pts, lvl = [], 0.0
                x0_ = bx0 + 3.0
                c0 = cy(x0_)
                if c0 is not None:
                    pts.append((x0_, c0 + lvl))
                for k, xj in enumerate(np.linspace(bx0 + 26, bx1 - 26, nj)):
                    c1, c2 = cy(xj), cy(min(xj + jw, bx1 - 3))
                    if c1 is None or c2 is None: continue
                    nl = amp * (1 if k % 2 == 0 else -1)
                    pts.append((xj, c1 + lvl))
                    pts.append((min(xj + jw, bx1 - 3), c2 + nl))
                    lvl = nl
                ce = cy(bx1 - 3.0)
                if ce is not None:
                    pts.append((bx1 - 3.0, ce + lvl))
                LAST["path"] = list(pts)
                LAST["band"] = band
                if len(pts) > 2:
                    line = LineString(pts)
                    # flat caps + mitre joins: a trace has square ends and sharp
                    # corners, round ones would read as a drawn stroke
                    strip = clip(line.buffer(tt / 2, cap_style=2, join_style=2), 2.6)
                    cw_ = sp.get("comp_w", 0.0)
                    if cw_ > 0.1:
                        off = tt / 2 + sp.get("comp_gap", 1.7) + cw_ / 2
                        comp = LineString([(px, py - off) for px, py in pts])                             .buffer(cw_ / 2, cap_style=2, join_style=2)
                        # only over PART of the run -- it shadows the blue "at
                        # some points", it does not trail it the whole way
                        zones = unary_union([shbox(a_, -90, b_, 90)
                                             for a_, b_ in _slots(bx0 + 8, bx1 - 8, 2, 30.0)])
                        channel = clip(comp.intersection(zones), 2.2)
                    nh_ = int(sp.get("n_hatch", 0))
                    if nh_ > 0:
                        hs = []
                        for a_, b_ in _slots(bx0 + 14, bx1 - 14, nh_, 34.0):
                            yc_ = cy((a_ + b_) / 2)
                            if yc_ is None: continue
                            for k in range(3):
                                x = a_ + k * 4.2
                                hs.append(ShPoly([(x + 3.4, yc_ - 5.0), (x + 5.0, yc_ - 5.0),
                                                  (x + 1.6, yc_ + 5.0), (x, yc_ + 5.0)]))
                        if hs:
                            strip = unary_union([strip, clip(unary_union(hs), 2.4)])
        elif style in ("hex", "chev", "hatch", "bracket", "ticks", "stepbar"):
            # HUD vocabulary from the reference sheets.  All of these are OPEN
            # shapes -- no closed frames -- and all sit on the same straight
            # upper-panel axis as the trapezoid row, so whichever is chosen the
            # accent still reads as one system rather than applied decoration.
            th = sp.get("trap_h", 6.5)
            zl, zh = ((2.0, 20.0) if sp.get("trap_zone", "upper") == "upper"
                      else (-20.0, 18.0))
            band = straight(th, th, lo=zl, hi=zh)
            n = max(1, int(sp.get("n_traps", 4)))
            out_ = []
            for i, (a_, b_) in enumerate(_slots(fx(0.322), fx(0.823), n, 11.0)):
                xc = (a_ + b_) / 2
                ys = _yspan(band, xc)
                if ys is None: continue
                y0, y1 = ys
                ym = (y0 + y1) / 2
                L = b_ - a_
                if L < 8.0: continue
                if style == "hex":              # chamfered lozenge
                    c = min(L * 0.26, (y1 - y0) * 1.5)
                    out_.append(ShPoly([(a_, ym), (a_ + c, y1), (b_ - c, y1),
                                        (b_, ym), (b_ - c, y0), (a_ + c, y0)]))
                elif style == "chev":           # arrow, notched tail
                    t_ = L * 0.34
                    out_.append(ShPoly([(a_, y0), (a_ + t_, y0), (b_, ym),
                                        (a_ + t_, y1), (a_, y1), (a_ + t_ * 0.55, ym)]))
                elif style == "hatch":          # group of parallel slashes
                    nh, sk = 4, (y1 - y0) * 0.75
                    wdt = L / (nh * 2.1)
                    for k in range(nh):
                        x = a_ + k * (L - wdt) / max(1, nh - 1)
                        out_.append(ShPoly([(x + sk, y0), (x + sk + wdt, y0),
                                            (x + wdt, y1), (x, y1)]))
                elif style == "bracket":        # L brackets, open on one side
                    t_ = max(1.3, (y1 - y0) * 0.30)
                    arm = min(L * 0.42, 15.0)
                    out_.append(ShPoly([(a_, y1), (a_ + arm, y1), (a_ + arm, y1 - t_),
                                        (a_ + t_, y1 - t_), (a_ + t_, y0), (a_, y0)]))
                    out_.append(ShPoly([(b_, y0), (b_ - arm, y0), (b_ - arm, y0 + t_),
                                        (b_ - t_, y0 + t_), (b_ - t_, y1), (b_, y1)]))
                elif style == "ticks":          # thin rail with perpendicular ticks
                    t_ = max(1.1, (y1 - y0) * 0.26)
                    out_.append(ShPoly([(a_, ym - t_ / 2), (b_, ym - t_ / 2),
                                        (b_, ym + t_ / 2), (a_, ym + t_ / 2)]))
                    for k in range(4):
                        x = a_ + 3.0 + k * (L - 6.0) / 3.0
                        out_.append(ShPoly([(x - t_ / 2, y0), (x + t_ / 2, y0),
                                            (x + t_ / 2, y1), (x - t_ / 2, y1)]))
                else:                           # "stepbar": runs, steps once, runs
                    t_ = max(1.4, (y1 - y0) * 0.34)
                    xs_ = a_ + L * 0.52
                    out_.append(ShPoly([(a_, y1 - t_), (xs_, y1 - t_), (xs_ + t_ * 1.6, y0 + t_),
                                        (b_, y0 + t_), (b_, y0), (xs_ + t_ * 0.9, y0),
                                        (xs_ - t_ * 0.7, y1 - 2 * t_), (a_, y1 - 2 * t_)]))
            out_ = [q.buffer(0) for q in out_]
            strip = unary_union([q for q in out_ if not q.is_empty])
        elif style == "axial":                  # straight along the limb
            strip = straight(sp.get("axial_w0", 4.6), sp.get("axial_w1", 1.5))

        strip = clip(strip, 2.8) if not strip.is_empty else ShPoly()
        if style in ("spine", "smooth", "axial", "jog"):
            strip = biggest(strip)
        cw = sp.get("accent_channel", 0.0)
        if cw > 0.05 and not strip.is_empty:
            # the reference never puts blue straight onto white; the graphite
            # border is what makes it read as a lit channel rather than paint
            acc = unary_union([g for g in (strip, blocks) if not g.is_empty])
            channel = clip(acc.buffer(cw, join_style=2).difference(acc), 2.2)

    return strip, blocks, ring, channel
