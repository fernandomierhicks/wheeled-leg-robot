r"""Round: how should GRAPHITE be shaped?

    C:/Users/ferna/cadenv/Scripts/python.exe parts/grey_board.py [Femur]

His note, on the Coupler and Femur backs: the grey *"just doesn't go with the
aesthetic ... it seems like camouflage or just random triangles and lines here
and there. It needs to be more fluidic."*  And the direction: *"the gray needs
to follow it somehow, or simulate a similar trajectory, but with more area,
since it's not an accent color."*  Reference: `artistic concepts/high res
render.png`, the CIRCUIT-LIKE LINE PATTERNS row -- a wide grey band and a thin
coloured line sharing one routed path, 45 degree doglegs, no square corners.

WHY THE GREY IS A FIELD TODAY, which is the thing to change.  The colour split
is a PLANE: white is the cap above `SZ`, and graphite is simply everything that
is left.  Grey is therefore a by-product of a Z cut, not a drawn shape -- so it
has no path, no width and no direction, and on the back it lands as whatever
rectangle the plane happens to slice.  Every variant below instead DRAWS the
grey, on the same routed path the blue already follows (`accents.LAST["path"]`),
so the two read as one circuit.

This board is 2D on purpose.  It judges routing, width and rhythm, which is all
that is being chosen here; depth is settled separately (he chose a surface inlay
on both faces, a few mm deep, not a full-depth cut).  A plan board costs a few
seconds against minutes for a 3D build, and TOURNAMENT.md's rule is one png per
round with the current state included so a round cannot go backwards.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import paths
from PIL import Image, ImageDraw
from shapely.geometry import LineString, box as shbox
from shapely.ops import unary_union
import accents as ACC
import board as B

PAL = dict(white=(0xF2, 0xF4, 0xF7), grey=(0x7E, 0x87, 0x95),
           blue=(0x1E, 0x7B, 0xFF), bg=(0x17, 0x19, 0x1C),
           hole=(0x17, 0x19, 0x1C), edge=(0x3A, 0x40, 0x49))


def _mitre(pts, w):
    """A trace of width `w` along `pts`: flat caps, mitre joins.

    join_style=2 is what keeps the corners square-cut at the dogleg instead of
    rounding them off -- a rounded corner reads as a drawn stroke, a mitred one
    reads as a routed track.  The path's own 45 degree jogs supply the "no
    straight corners" he asked for; the buffer must not soften them.
    """
    return LineString(pts).buffer(w / 2.0, cap_style=2, join_style=2)


def _offset(pts, d):
    """Shift a path sideways by `d` (positive = -Y side)."""
    return [(x, y - d) for x, y in pts]


def route(xa, xb, lane, n_jog, amp, jw, rng):
    """A routed polyline: straight runs joined by diagonal doglegs.

    Same vocabulary as the blue -- straight, jog, straight, no square corners --
    but its own spans, lanes and jog positions, so it does NOT shadow the blue.
    `rng` is seeded and the seed is printed, so a chosen cell can be rebuilt
    exactly.
    """
    pts, y = [(xa, lane)], lane
    for xj in np.linspace(xa + 0.20 * (xb - xa), xb - 0.26 * (xb - xa), n_jog):
        dy = amp * float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.55, 1.45))
        pts.append((xj, y))
        pts.append((xj + jw, y + dy))
        y += dy
    pts.append((xb, y))
    return pts


def variants2(pts, sp, grown, kb, P, seed=7):
    """ROUND 2.  His note on round 1: *"the gray line was following the blue
    almost exactly ... it needs to be a little bit more random, but we still
    did the same dog-legging kind of thing"*, with an older board attached as
    the example that reads better *"because it's not as following one line
    after the other"*.

    Looking at that older board, its grey is not a trace at all: it is the
    FEATURE BANDS -- frame, rail, pads, the hip collar -- which already sit at
    different lanes, run different lengths and stop in different places.  That
    staggered rhythm is what he is responding to, not any relationship to the
    blue.  So round 2 keeps the dogleg language and throws away the constant
    offset: independent runs, their own spans and lanes, seeded so a pick is
    reproducible.
    """
    tt = float(sp.get("trace_t", 3.2))
    x0, x1 = pts[0][0], pts[-1][0]
    gx0, gy0, gx1, gy1 = grown.bounds
    blue_y = float(np.mean([p[1] for p in pts]))
    L = gx1 - gx0
    fx = lambda t: gx0 + t * L
    amp, jw = float(sp.get("jog_amp", 4.5)), float(sp.get("jog_w", 7.0))
    out = []

    # A. the old board's own scheme: grey IS the feature bands.
    feat = unary_union([g for g in (P["frame"], P["rail"], P["pads"], P["collars"])
                        if not g.is_empty])
    out.append(("A  feature bands", "grey = frame + rail + pads + collar,\n"
                                    "the scheme in the board he liked", feat))

    # B. three independent runs, staggered spans and lanes.
    r = np.random.default_rng(seed)
    segs = []
    for xa, xb, lane, w in ((fx(.12), fx(.52), blue_y - 11.0, tt * 2.8),
                            (fx(.38), fx(.86), blue_y + 7.5, tt * 2.2),
                            (fx(.60), fx(.93), blue_y - 8.0, tt * 3.2)):
        segs.append(_mitre(route(xa, xb, lane, 2, amp, jw, r), w))
    out.append(("B  three runs", "independent spans and lanes,\nthey overlap but never track",
                unary_union(segs)))

    # C. more of them, shorter -- the "here and there" reading.
    r = np.random.default_rng(seed + 1)
    segs = []
    for xa, xb, lane, w in ((fx(.10), fx(.34), blue_y - 12.0, tt * 2.4),
                            (fx(.26), fx(.49), blue_y + 9.0, tt * 1.9),
                            (fx(.45), fx(.70), blue_y - 9.5, tt * 2.9),
                            (fx(.62), fx(.80), blue_y + 6.0, tt * 1.8),
                            (fx(.74), fx(.94), blue_y - 6.5, tt * 2.6)):
        segs.append(_mitre(route(xa, xb, lane, 1, amp, jw, r), w))
    out.append(("C  five short runs", "shorter, more of them,\nnothing runs the whole length",
                unary_union(segs)))

    # D. one long run that CROSSES the blue's lane instead of pacing it.
    r = np.random.default_rng(seed + 2)
    main = _mitre(route(fx(.10), fx(.92), blue_y + 10.0, 4, amp * 1.9, jw, r), tt * 3.0)
    out.append(("D  lane-hopping run", "one long run with bigger jogs,\ncrossing the blue rather than pacing it",
                main))

    # E. long run plus short branch stubs to the edge: circuit-branch language.
    r = np.random.default_rng(seed + 3)
    spine = route(fx(.14), fx(.88), blue_y - 10.5, 3, amp, jw, r)
    segs = [_mitre(spine, tt * 2.6)]
    for t_ in (0.28, 0.52, 0.71):
        i = int(t_ * (len(spine) - 1))
        sx, sy = spine[i]
        segs.append(_mitre([(sx, sy), (sx + 5.0, sy - 9.0)], tt * 1.5))
    out.append(("E  spine + branches", "one run with short stubs breaking\naway toward the edge",
                unary_union(segs)))

    # F. the feature bands, re-cut with doglegs so they share the trace language.
    r = np.random.default_rng(seed + 4)
    segs = []
    for xa, xb, lane, w in ((fx(.11), fx(.46), gy0 + 7.0, tt * 3.4),
                            (fx(.30), fx(.90), gy1 - 6.5, tt * 2.4),
                            (fx(.52), fx(.84), blue_y - 12.0, tt * 2.8)):
        segs.append(_mitre(route(xa, xb, lane, 2, amp * 1.3, jw, r), w))
    out.append(("F  banded, edge-hugging", "runs pushed out to the rim,\nthe old band rhythm with doglegs",
                unary_union(segs)))

    keep = grown.buffer(-2.2, join_style=2)
    return [(t, s, g.intersection(keep).difference(kb)) for t, s, g in out]


def variants(pts, sp, grown, kb):
    """Six ways to shape the grey, all on the blue's own path."""
    tt = float(sp.get("trace_t", 3.2))          # the blue's width
    x0, x1 = pts[0][0], pts[-1][0]
    span = x1 - x0
    out = []

    # 1. what exists today: a timid 4.5 mm companion over two short zones, with
    #    the REAL grey being the plane leftover this board exists to replace.
    cw = float(sp.get("comp_w", 4.5))
    off = tt / 2 + float(sp.get("comp_gap", 1.7)) + cw / 2
    zones = unary_union([shbox(a, -90, b, 90) for a, b in
                         ((x0 + 8, x0 + 0.33 * span), (x0 + 0.55 * span, x1 - 8))])
    out.append(("1  today", f"{cw:g} mm companion, two zones\nplus a plane leftover",
                _mitre(_offset(pts, off), cw).intersection(zones)))

    # 2. the straight read of his note: one band, same path, ~3x the blue.
    w2 = tt * 3.2
    out.append(("2  single broad", f"{w2:.0f} mm band, full run, alongside",
                _mitre(_offset(pts, tt / 2 + 1.6 + w2 / 2), w2)))

    # 3. the reference's dominant motif: the blue runs INSIDE a grey channel.
    w3 = tt * 4.6
    out.append(("3  blue inside grey", f"{w3:.0f} mm channel, blue centred in it",
                _mitre(pts, w3)))

    # 4. a band that changes width along the run, the way the concept panels do.
    seg = []
    for i in range(len(pts) - 1):
        f = i / max(1, len(pts) - 2)
        wv = tt * (4.4 - 2.6 * f)
        seg.append(_mitre([pts[i], pts[i + 1]], wv))
    out.append(("4  tapered", "wide at the hip, narrowing to the knee",
                unary_union(seg)))

    # 5. two grey runs bracketing the blue -- the reference uses paired lines a lot.
    w5 = tt * 2.0
    out.append(("5  bracketing pair", f"two {w5:.0f} mm runs, blue between them",
                unary_union([_mitre(_offset(pts, tt / 2 + 1.5 + w5 / 2), w5),
                             _mitre(_offset(pts, -(tt / 2 + 1.5 + w5 / 2)), w5)])))

    # 6. trace plus pads at each jog: circuit-board vocabulary, and it puts the
    #    extra AREA he asked for where the path already changes direction.
    w6 = tt * 2.6
    base = _mitre(_offset(pts, tt / 2 + 1.5 + w6 / 2), w6)
    pads = []
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        pads.append(_mitre([(px - 7.0, py - (tt / 2 + 1.5 + w6 / 2)),
                            (px + 7.0, py - (tt / 2 + 1.5 + w6 / 2))], w6 * 2.3))
    out.append(("6  trace + pads", "band widens into a pad at every dogleg",
                unary_union([base] + pads)))

    # every variant stays on the part and clear of every hole
    keep = grown.buffer(-2.2, join_style=2)
    return [(t, s, g.intersection(keep).difference(kb)) for t, s, g in out]


def draw(P, grey, blue, title, sub, size=(1180, 300), ss=2):
    W, H = size[0] * ss, size[1] * ss
    img = Image.new("RGB", (W, H), PAL["bg"])
    gx0, gy0, gx1, gy1 = P["grown"].bounds
    # the caption is three lines deep, so the drawing starts below it -- with a
    # uniform margin the part rides up over "% of the silhouette"
    m, top, bot = 18 * ss, 84 * ss, 12 * ss
    s = min((W - 2 * m) / (gx1 - gx0), (H - top - bot) / (gy1 - gy0))
    xf = lambda x, y: ((x - gx0) * s + m, H - bot - (y - gy0) * s)
    B._paint(img, P["grown"], xf, PAL["white"], outline=PAL["edge"], width=ss)
    B._paint(img, grey, xf, PAL["grey"])
    B._paint(img, blue, xf, PAL["blue"])
    B._paint(img, P["KEEP"], xf, PAL["hole"])
    img = img.resize(size, Image.LANCZOS)
    d = ImageDraw.Draw(img)
    d.text((14, 8), title, font=B._font(19, True), fill=(0xE8, 0xEC, 0xF2))
    for i, ln in enumerate(sub.split("\n")):
        d.text((14, 30 + i * 15), ln, font=B._font(13), fill=(0x93, 0x9C, 0xA8))
    return img


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    rnd = 2 if "--r2" in sys.argv else 1
    seed = 7
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    part = (args[0] if args else "Femur")
    mod = __import__(part.lower().replace(" ", "_"))
    sp = json.load(open(os.path.join(paths.SPECS, f"{part.lower()}.json")))
    P = mod.plan(sp)
    pts = ACC.LAST.get("path")
    if not pts:
        raise SystemExit("no circuit path -- accent_style must be 'circuit'")
    print(f"{part}: routed path with {len(pts)} nodes, "
          f"x {pts[0][0]:.1f}..{pts[-1][0]:.1f}")

    vs = (variants2(pts, sp, P["grown"], P["kb"], P, seed) if rnd == 2
          else variants(pts, sp, P["grown"], P["kb"]))
    if rnd == 2:
        print(f"round 2, seed {seed} -- a chosen cell rebuilds exactly from it")
    tiles = []
    for t, s_, g in vs:
        pct = 100 * g.area / P["grown"].area
        tiles.append(draw(P, g, P["strip"], t, f"{s_}\n{g.area:.0f} mm2 in plan, "
                                                f"{pct:.0f}% of the silhouette"))
    W, Hc = 1180, 300
    sheet = Image.new("RGB", (W, 86 + len(tiles) * (Hc + 8)), PAL["bg"])
    d = ImageDraw.Draw(sheet)
    d.text((16, 16), f"{part} — how should the GREY be shaped?"
                     + ("  (round 2)" if rnd == 2 else ""),
           font=B._font(30, True), fill=(0xF2, 0xF4, 0xF7))
    d.text((16, 54),
           ("same dog-legging, but its OWN spans and lanes — grey no longer "
            f"tracks the blue.  seed {seed}.  reply with one letter."
            if rnd == 2 else
            "every variant draws grey on the SAME routed path the blue "
            "follows.  reply with one number."),
           font=B._font(15), fill=(0x93, 0x9C, 0xA8))
    for i, im in enumerate(tiles):
        sheet.paste(im, (0, 86 + i * (Hc + 8)))
    out = os.path.join(paths.BOARDS,
                       f"{part.lower()}_grey_trace{'_r2' if rnd == 2 else ''}.png")
    os.makedirs(paths.BOARDS, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}  ({sheet.width}x{sheet.height})")
