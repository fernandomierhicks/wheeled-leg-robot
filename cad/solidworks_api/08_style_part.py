"""Step 8: GLACIER as NATIVE SolidWorks features, on the v5 part itself.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/08_style_part.py Femur [--dry]

The look comes from the existing recipe's plan() (cad/aesthetics/parts/<part>.py,
the same plan the OCC build uses), and this file replays that recipe's build()
operation by operation as SolidWorks features on the part in
cad/v5 Ai designed -- so the result is a feature tree he can edit, not an
imported lump.  Order, heights and drafts follow femur.py build() exactly:

  1. flange            Extrude Boss, merged          (the only added growth)
  2. frame/rail/pads   drafted Extrude Boss, merged  (12 / 10 / 16 deg)
  3. pockets, windows, through-cutouts, deep pockets, accent engraving,
     back circuit grooves                       Extrude Cut
  4. colour            Combine: blue accent, then graphite trace, each inlaid on
                       BOTH faces as body ∩ slab; white is what is left.
                       Exact partition -- the bodies add up to the part.

ONE DELIBERATE CHANGE from the locked recipe (his request, 2026-10-02): every
removal and every raised or added feature is kept out of a MECHANICAL SEAT
around each opening -- screw heads and washers, and the bearing retaining
washers.  The recipe only kept 2.2 mm clear of a hole edge, which leaves the
retaining washer of a 6804 bearing sitting on cut-away material.  See seat().

The recipe builds in a frame mirrored to put the show face at +Z (show_face
-Z); every height here is mapped back: real z = SIGN * recipe z.
"""
import os
import sys
import json
import time
import shutil
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
AES = os.path.normpath(os.path.join(HERE, "..", "aesthetics"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AES, "lib"))
sys.path.insert(0, os.path.join(AES, "parts"))
import swlib
import swstyle as S
from swlib import c, wrap, sld
from extras import EXTRA
from shapely.geometry import Point
from shapely.ops import unary_union

PARTS = {   # part -> (recipe module, spec, v5 file)
    "Femur":      ("femur", "femur.json", r"Links\Femur.SLDPRT"),
    "Coupler":    ("coupler", "coupler.json", r"Links\Coupler.SLDPRT"),
    "Side panel": ("side_panel", "side panel.json", r"Body\Side panel.SLDPRT"),
    "RobotMount": ("robotmount", "robotmount.json", r"Body\OldRobotBodyMount\RobotMount.SLDPRT"),
}
ORIGINALS = os.path.join(swlib.V5, "_originals")

# GLACIER (palette arctic_lt): white body, graphite trace, blue accent
WHITE, GRAPHITE, BLUE = (0.93, 0.94, 0.95), (0.25, 0.27, 0.30), (0.13, 0.45, 0.95)


def seat(opening):
    """Mechanical keep-out around one opening (a plan polygon, mm).

    r is the equivalent radius.  Fastener holes keep the head / washer seat
    (+3.5 mm: an M3 roundhead is 5.5 mm across, its washer 7; M4 7.6 / 9).
    Bearing and shaft bores keep the retaining washer and its screw ring
    (+8 mm: the 6804's 32 mm bore, washer screws on a 38.05 mm circle).
    """
    import math
    r = math.sqrt(opening.area / math.pi)
    grow = 3.5 if r <= 3.2 else (4.0 if r <= 8.0 else 8.0)
    return opening.convex_hull.buffer(grow, 48)


def all_openings(s):
    """Every opening, from EVERY horizontal face -- not just the two outermost.

    keepout.openings() looks only at the faces at the bounding-box top and
    bottom.  The Side panel's bosses reach z 20.5 while its plate face is at
    z 10, so holes and counterbores that start on an intermediate face were
    invisible: the styling grew under four M3 screw heads there.  An inner loop
    of a horizontal face is an OPENING only if the material side of the face is
    empty inside it (a hole, a counterbore); a boss rising from the face makes
    an inner loop too, and is not an opening."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane
    from build123d import Vector
    from outline import outline_polygon
    out = []
    for f in s.faces():
        if BRepAdaptor_Surface(f.wrapped).GetType() != GeomAbs_Plane:
            continue
        n = f.normal_at(f.center())
        if abs(n.Z) < 0.99:
            continue
        z = f.center().Z
        for w in f.inner_wires():
            p = outline_polygon(w)
            if not p.is_valid or p.area < 0.5:
                continue
            q = p.representative_point()
            if not s.is_inside(Vector(q.x, q.y, z - 0.2 * (1 if n.Z > 0 else -1))):
                out.append(p)
    return out


def mech_keepout(src):
    """Union of every opening's seat, every horizontal face."""
    ops = all_openings(src["solid"])
    return unary_union([seat(o) for o in ops]), ops


def main(part, dry=False, fresh=False, restyle=False):
    mod_name, spec_name, rel = PARTS[part]
    R = importlib.import_module(mod_name)
    sp = json.load(open(os.path.join(AES, "specs", spec_name)))
    t0 = time.time()
    R._face(sp)
    P = R.plan(sp)
    src = R.source()
    SIGN = -1.0 if R.SHOW_FACE == "-Z" else 1.0
    print(f"{part}: plan() {time.time() - t0:.0f} s, show face {R.SHOW_FACE} -> SIGN {SIGN:+.0f}")

    M, ops = mech_keepout(src)
    if SIGN < 0:      # plan() polygons are in the recipe frame; XY is unchanged by a Z mirror
        pass
    print(f"  {len(ops)} openings -> mechanical keep-out {M.area:.0f} mm2 "
          f"(the recipe's own kb was {P['kb'].area:.0f} mm2)")
    if part in EXTRA:       # hand-laid features for ground the recipe leaves bare
        print(f"  extras.py: {EXTRA[part](P, M)}")

    def z(a, b):
        """Recipe-frame z interval -> real (lo, hi)."""
        a, b = SIGN * a, SIGN * b
        return (min(a, b), max(a, b))

    ZS, ZB, ZTOP, ZT = P["ZS"], P["ZB"], P["ZTOP"], P["ZT"]
    CH = sp["chamfer"]
    MW = float(sp.get("min_wall", 3.0))
    gd = float(sp.get("grey_d") or 3.5)
    bd = float(sp.get("accent_d") or gd)
    bed = float(sp.get("back_eng_d") or 1.2)
    ftop = ZS - CH - 1.0 + sp["frame_h"] + 1.0

    M_cut = M.buffer(0.3, 48)
    M_add = M.buffer(0.6, 48)   # additions clear the seat FURTHER than removals: see mech()
    # where the styling's ADDITIONS ran into a neighbour anywhere in the stroke
    # (13_collision_keepout.py, measured by SolidWorks): additions stay out
    _ck = os.path.join(HERE, "out", "collision_keepout.json")
    if os.path.exists(_ck):
        from shapely.geometry import shape as _shape
        _kc = [_shape(g) for g in json.load(open(_ck)).get(part, [])]
        if _kc:
            K_col = unary_union(_kc)
            M_add = unary_union([M_add, K_col])
            print(f"  collision keep-out for the additions: {K_col.area:.0f} mm2")

    def mech(g, what, removal=True):
        """Clip a feature out of the mechanical keep-out, and SAY what it cost.

        Additions clear the seat by +0.6 mm, removals by +0.3 mm.  The same
        circle for both gives them one shared edge, the cut then runs along a
        face coincident with the frame's, and SolidWorks refuses the boolean
        (the Coupler's pockets).  But additions INSIDE removals (+0.0 / +0.3)
        leave a 0.3 mm ridge of frame round every seat: Coupler frame slivers
        under 1.5 mm went 10.4 -> 18.5 mm2 in plan.  +0.6 / +0.3 gives 9.8,
        below the original recipe, with no shared edge.
        """
        if g is None or g.is_empty:
            return g
        K = M_cut if removal else M_add
        lost = g.intersection(K).area
        if lost > 0.5:
            print(f"    {what}: {lost:6.1f} mm2 of {g.area:6.1f} sat on a fastener/bearing seat or in a neighbour's path -> removed")
        return g.difference(K)

    # ---- the operation list, in build() order --------------------------------
    ops_list = []           # (kind, name, geometry, args)
    fl = P.get("flange")
    if fl is not None:
        ops_list.append(("boss", "GL_Flange", mech(fl[2], "flange", removal=False), z(fl[0], fl[1])))
    up = SIGN > 0           # recipe +Z ("up", away from the show face plate) in real terms
    for key, h, off, draft in (("frame", sp["frame_h"], 1.0, 12), ("rail", sp["rail_h"], 0.8, 10),
                               ("pads", sp["pad_h"], 0.8, 16)):
        proud = (ZS - CH - off) + (h + off) - ZS     # how far it stands off the plate
        if proud < 0.3:
            print(f"    {key}: top sits {proud:+.2f} mm off the plate (buried, as in the OCC build) -> skipped")
            continue
        if h > .3 and not P[key].is_empty:
            base = SIGN * (ZS - CH - off)
            ops_list.append(("raised", f"GL_{key.capitalize()}", mech(P[key], key, removal=False),
                             (base, h + off, up, draft)))
    if not P["pock"].is_empty and sp["pocket_d"] > 0.1:
        ops_list.append(("cut", "GL_Pockets", mech(P["pock"], "pockets"), z(ZS - sp["pocket_d"], ZTOP + 10)))
    if not P["wins"].is_empty and sp.get("win_d", 0) > 0.1:
        ops_list.append(("cut", "GL_Windows", mech(P["wins"], "windows"), z(ftop - sp["win_d"], ZTOP + 12)))
    if not P["cutouts"].is_empty:
        ops_list.append(("cut", "GL_Cutouts", mech(P["cutouts"], "through-cutouts"), z(ZB - 10, ZTOP + 12)))
    if not P["cut_pockets"].is_empty:
        ops_list.append(("cut", "GL_DeepPockets", mech(P["cut_pockets"], "deep pockets"), z(ZB + MW, ZTOP + 12)))
    if sp["side_d"] > 0.1 and (P["lo_pr"] or P["hi_pr"]):
        from shapely.geometry import box as shbox, Polygon as ShPoly
        grown = P["grown"]
        band = grown.buffer(4.0, join_style=2).difference(grown.buffer(-sp["side_d"], join_style=2))
        for g in (P["kb"], P["side_guard"]):
            if not g.is_empty:
                band = band.difference(g)
        band = mech(band, "side pockets")
        for side, prs, half in (("Lo", P["lo_pr"], shbox(-1e3, -1e3, 1e3, 0)),
                                ("Hi", P["hi_pr"], shbox(-1e3, 0, 1e3, 1e3))):
            if not prs:
                continue
            prs = [_open_slot(p, P, MW, sp) for p in prs]
            prs = [p for p in prs if p is not None]
            if not prs:
                continue
            xs = [q[0] for p in prs for q in p]
            zr = [SIGN * q[1] for p in prs for q in p]
            b = band.intersection(half).intersection(shbox(min(xs) - 1, -1e3, max(xs) + 1, 1e3))
            # Top Plane sketch coordinates are (X, -Z): measured, see swstyle.sketch
            prof = unary_union([ShPoly([(x, -SIGN * zm) for x, zm in p]) for p in prs])
            ops_list.append(("side", f"GL_Side{side}", b, (prof, min(zr), max(zr))))
    eng = unary_union([g for g in (P["strip"], P["blocks"]) if not g.is_empty])
    if not eng.is_empty:
        ops_list.append(("cut", "GL_AccentEngrave", mech(eng, "accent engraving"), z(ZS - 1.2, ZTOP + 10)))
    if not P["back_eng"].is_empty and bed > 0.05:
        zg = P["ZGROOVE"]
        ops_list.append(("cut", "GL_BackCircuit", mech(P["back_eng"], "back circuit grooves"), z(zg - 10, zg + bed)))
    acc = unary_union([g for g in (P["ring"], P["strip"], P["blocks"]) if not g.is_empty])
    zbk = P["ZBACK"]
    colours = [("blue", BLUE, acc, [z(P["ZSHOW"] - bd, ZTOP + 12), z(ZB - 12, zbk + bd)]),
               ("graphite", GRAPHITE, P["grey"], [z(P["ZSHOW"] - gd, ZTOP + 12), z(ZB - 12, zbk + gd)])]

    for kind, name, g, args in ops_list:
        shown = args if kind != "side" else f"profiles {args[0].area:.0f} mm2, z {args[1]:.2f}..{args[2]:.2f}"
        print(f"  {kind:6s} {name:18s} {g.area if g is not None else 0:7.1f} mm2  {shown}")
    for nm, _, g, slabs in colours:
        print(f"  colour {nm:18s} {g.area:7.1f} mm2  slabs {slabs}")
    if dry:
        return

    # ---- SolidWorks ----------------------------------------------------------
    path = os.path.join(swlib.V5, rel)
    bak = os.path.join(ORIGINALS, rel)
    if not os.path.exists(bak):
        os.makedirs(os.path.dirname(bak), exist_ok=True)
        shutil.copy2(path, bak)
        print(f"  backed up the original to {bak}")
    sw, _ = swlib.connect()
    model = None
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        if os.path.normcase(d.GetPathName()) == os.path.normcase(path):
            model = d
    if model is None:
        doc, err, warn = sw.OpenDoc6(path, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
        if doc is None:
            raise SystemExit(f"cannot open {path} (err {err})")
        model = wrap(doc, sld.IModelDoc2)
    sw.ActivateDoc3(model.GetTitle(), False, 0, 0)
    if fresh and any(f.Name.startswith("GL_") for f in _features(model)):
        if open(path, "rb").read() != open(bak, "rb").read():
            raise SystemExit(f"--fresh: {path} on disk is not the original any more; "
                             f"restore it from {bak} by hand first")
        model.ReloadOrReplace(False, path, True)      # discard the unsaved styling
        model = None
        for d in sw.GetDocuments() or []:
            d = wrap(d, sld.IModelDoc2)
            if os.path.normcase(d.GetPathName()) == os.path.normcase(path):
                model = d
        print("  --fresh: reloaded the unstyled part from disk")
    if restyle and any(f.Name.startswith("GL_") for f in S._iter_features(model)):
        k = S.strip_styling(model, src["solid"].volume)
        print(f"  --restyle: deleted {k} GL_ features -> the original part again")
    if any(f.Name.startswith("GL_") for f in _features(model)):
        raise SystemExit(f"{path} already carries GL_ features -- restore it from "
                         f"{bak} before re-running")

    bs = S.bodies(model)
    if len(bs) != 1:
        raise SystemExit(f"{part}: expected one source body, found {len(bs)}")
    bx = [v * 1000 for v in bs[0].GetBodyBox()]
    sb = src["solid"].bounding_box()
    want = (sb.min.X, sb.min.Y, sb.min.Z, sb.max.X, sb.max.Y, sb.max.Z)
    if SIGN < 0:          # source() is the mirrored recipe frame
        want = (want[0], want[1], -want[5], want[3], want[4], -want[2])
    dev = max(abs(a - b) for a, b in zip(bx, want))
    dv = abs(S.volume(bs[0]) - src["solid"].volume)
    print(f"  SolidWorks body vs STEP source: box worst {dev:.3f} mm, volume {dv:.1f} mm3")
    # GetBodyBox is LOOSE around curved faces (Tibia +Y: 21.224 vs the true
    # 21.000), so the box is checked loosely and the volume tightly.
    if dev > 0.5 or dv > 1e-4 * src["solid"].volume:
        raise SystemExit(f"{part}: the .SLDPRT is not in the STEP's frame "
                         f"({[round(v, 2) for v in bx]} vs {[round(v, 2) for v in want]})")
    v_src = S.total_volume(model)

    t1 = time.time()
    for kind, name, g, args in ops_list:
        if g is None or g.is_empty or g.area < 0.5:
            continue
        v0 = S.total_volume(model)
        if kind in ("boss", "raised", "cut"):
            sk = _with_retries(model, kind, name, g, args)
            if sk is None:
                continue
        elif kind == "side":
            sk = S.sketch(model, g, name=name + "_Sk")
            # band body -> keep only what lies inside the X-Z profiles (scoped,
            # flipped cut on the Top Plane) -> subtract from the part
            prof, zlo, zhi = args
            before = S.names(model)
            S.tool(model, sk, zlo, zhi, name=name + "_Band")
            psk = S.sketch(model, prof, name=name + "_ProfSk", plane="Top Plane")
            S.cut(model, psk, scope=S.new_since(model, before), outside=True, name=name + "_Trim")
            pieces = S.new_since(model, before)
            S.combine(model, "cut", _largest(model), pieces)
            S.last_feature(model).Name = name
        # an ADDITION that did not merge is not attached to anything: drop it
        # whatever its size (RobotMount: a 2.9 cm3 flange band at z 13.7..19
        # hovering over a plate edge that is 5 mm tall -- plan() checks
        # attachment in plan only).  A CUT keeps the 1 % scrap rule.
        _drop_detached(model, name, frac=1.0 if kind in ("boss", "raised") else 0.01)
        nb = len(S.bodies(model))
        print(f"    {name:18s} {sk.__dict__.get('n_lines', 0):4d} lines -> {S.total_volume(model):10.1f} mm3 "
              f"({S.total_volume(model) - v0:+8.1f}), {nb} body{'ies' if nb != 1 else ''}")
        if nb != 1:
            raise SystemExit(f"{name} left {nb} bodies -- something floated off")
    styled = S.total_volume(model)

    # COLOUR, per colour (accent first, so it wins where the two cross):
    #   W1 = copy(white); cut each inlay slab out of white (feature scope =
    #   white only); Wc = copy(what is left); Combine W1 - Wc = exactly the
    #   inlay.  Every input is a real body, so the partition is exact.
    made = {}
    for nm, rgb, g, slabs in colours:
        if g is None or g.is_empty:
            continue
        sk = S.sketch(model, g, name=f"GL_{nm}_Sk")
        done = _all(made)
        white = _largest_except(model, done)
        w1 = S.copy_body(model, white, name=f"GL_{nm}_Copy")
        cut_any = False
        for i, (lo, hi) in enumerate(slabs):
            white = _largest_except(model, done | {w1.Name})
            try:
                S.cut(model, sk, lo, hi, scope=[white], name=f"GL_{nm}_Out{i}")
                cut_any = True
            except S.FeatureFailed:
                print(f"    colour {nm}: slab {i} misses the part, skipped")
        if not cut_any:
            S.delete_bodies(model, [w1])
            continue
        rest = [b for b in S.bodies(model) if b.Name != w1.Name and b.Name not in done]
        keep = [S.copy_body(model, b, name=f"GL_{nm}_Keep{i}") for i, b in enumerate(rest)]
        before = S.names(model)
        try:
            S.combine(model, "cut", S.by_name(model, w1.Name), keep)
            S.last_feature(model).Name = f"GL_{nm}"
            made[nm] = [b.Name for b in S.new_since(model, before)]
        except S.FeatureFailed:
            # Parasolid occasionally refuses this subtraction outright (the
            # Coupler's blue, 3 bodies, nothing odd about them).  Same answer by
            # the other route: (copy of white) INTERSECT (each slab's tool body),
            # one tool per Combine.  w1 = white before the slab cuts, so the
            # intersections plus the cut white still partition w1 exactly.
            print(f"    colour {nm}: Combine subtract refused -> intersecting slab by slab")
            S.delete_bodies(model, [S.by_name(model, b.Name) for b in keep if S.by_name(model, b.Name)])
            got = []
            for i, (lo, hi) in enumerate(slabs):
                for j, piece in enumerate(S.tool(model, sk, lo, hi, name=f"GL_{nm}_Slab{i}")):
                    wc = S.copy_body(model, S.by_name(model, w1.Name), name=f"GL_{nm}_W{i}_{j}")
                    b0 = S.names(model)
                    try:
                        S.combine(model, "common", wc, [piece])
                        got += [b.Name for b in S.new_since(model, b0)]
                    except S.FeatureFailed:          # this slab misses the part here
                        S.delete_bodies(model, [b for b in (S.by_name(model, wc.Name),
                                                            S.by_name(model, piece.Name)) if b])
            S.delete_bodies(model, [S.by_name(model, w1.Name)])
            made[nm] = [n for n in got if S.by_name(model, n) is not None]
        print(f"    colour {nm:9s} -> {len(made[nm])} body(ies), "
              f"{sum(S.volume(b) for b in S.bodies(model) if b.Name in made[nm]):8.1f} mm3")

    dropped = drop_debris(model)
    # White need not be ONE body: an inlay can enclose an island of white
    # (RobotMount: 342 mm3 inside a blue loop).  The printed part is still one
    # piece -- the islands are fused to it through the inlay around them.
    whites = sorted([b for b in S.bodies(model) if b.Name not in _all(made)], key=lambda b: -S.volume(b))
    for i, w in enumerate(whites, 1):
        S.colour(w, WHITE, "white" if i == 1 else f"white_{i}")
    for nm, rgb, _, _ in colours:
        live = [b for b in S.bodies(model) if b.Name in made.get(nm, [])]
        for i, b in enumerate(live, 1):
            S.colour(b, rgb, f"{nm}_{i}")
    tot = S.total_volume(model)
    print(f"\n  source {v_src:10.1f} mm3 -> styled {styled:10.1f} mm3 "
          f"({styled - v_src:+.1f}); after colour split {tot:10.1f} (must equal styled)")
    for b in S.bodies(model):
        print(f"    {b.Name:12s} {S.volume(b):9.1f} mm3")
    if abs(tot + dropped - styled) > max(2.0, 1e-4 * styled):
        raise SystemExit("the colour bodies do not add up to the styled part")
    model.ShowNamedView2("*Isometric", c.swIsometricView)
    model.ViewZoomtofit2()
    ok, err, warn = model.Extension.SaveAs3(path, c.swSaveAsCurrentVersion,
                                            c.swSaveAsOptions_Silent, None, None, 0, 0)
    if not ok:
        raise SystemExit(f"save failed (err {err}, warn {warn})")
    print(f"  saved {path}  ({time.time() - t1:.0f} s of SolidWorks)")


def _features(model):
    f = wrap(model.FirstFeature(), sld.IFeature)
    while f is not None:
        yield f
        f = wrap(f.GetNextFeature(), sld.IFeature)


def _largest(model):
    return max(S.bodies(model), key=S.volume)


def drop_debris(model, min_mm3=1.0):
    """Delete colour bodies under `min_mm3`.  Combine leaves ZERO-volume sliver
    bodies wherever an inlay slab's face lies on an existing face -- the
    Coupler's graphite came out as 71 bodies, 40 of them 0.0 mm3.  They print
    as nothing and clutter the tree and the 3MF; the OCC build drops debris
    under 1 mm3 for the same reason (femur.py _debris).  Returns the volume
    removed, so the partition check can still be exact."""
    junk = [b for b in S.bodies(model) if S.volume(b) < min_mm3]
    if not junk:
        return 0.0
    v = sum(S.volume(b) for b in junk)
    S.delete_bodies(model, junk)
    S.last_feature(model).Name = "GL_DropDebris"
    print(f"    dropped {len(junk)} debris bodies under {min_mm3:g} mm3 ({v:.2f} mm3 total)")
    return v


def recolour(model):
    """Renumber and recolour bodies by their name prefix (white/blue/graphite)."""
    groups = {"white": WHITE, "blue": BLUE, "graphite": GRAPHITE}
    n = {}
    for i, b in enumerate(S.bodies(model)):           # unique temporaries first
        g = b.Name.split("_")[0]
        if g in groups:
            b.Name = f"{g}_tmp{i}"
    for b in sorted(S.bodies(model), key=lambda b: -S.volume(b)):
        g = b.Name.split("_")[0]
        if g not in groups:
            continue
        n[g] = n.get(g, 0) + 1
        S.colour(b, groups[g], g if (g == "white" and n[g] == 1) else f"{g}_{n[g]}")
    return n


def _one(model, kind, name, g, args):
    sk = S.sketch(model, g, name=name + "_Sk", grow=0.05 if kind in ("boss", "raised") else -0.005)
    if sk is None:
        return None
    try:
        if kind == "boss":
            S.boss(model, sk, *args, name=name)
        elif kind == "raised":
            base, h, upw, draft = args
            S.raised(model, sk, base, h, upw, draft, name=name)
        else:
            S.cut(model, sk, *args, name=name)
        return sk
    except S.FeatureFailed:
        _delete_feature(model, sk)          # leave no orphan sketch behind
        raise


def _with_retries(model, kind, name, g, args):
    """Whole sketch -> each region alone -> each region shrunk 0.05 mm -> skip.

    One region whose edge happens to coincide with existing geometry must not
    cost the whole feature, and must not stop the part.  Every fallback is
    printed, so a skipped region is never silent."""
    try:
        return _one(model, kind, name, g, args)
    except S.FeatureFailed:
        pass
    regions = S.polys(g)
    print(f"    {name}: whole sketch refused -> trying its {len(regions)} region(s) one by one")
    last = None
    for i, r in enumerate(regions, 1):
        for shrink in (0.0, 0.05, 0.2):
            rg = r if shrink == 0 else r.buffer(-shrink, join_style=2)
            if rg.is_empty:
                break
            try:
                last = _one(model, kind, f"{name}_{i}", rg, args)
                if shrink:
                    print(f"      region {i} ({r.area:.1f} mm2) went through shrunk {shrink} mm")
                break
            except S.FeatureFailed:
                continue
        else:
            print(f"      region {i} ({r.area:.1f} mm2) REFUSED at every shrink -> skipped")
    return last


def _delete_feature(model, f):
    model.ClearSelection2(True)
    if f.Select2(False, 0):
        model.Extension.DeleteSelection2(0)
    model.ClearSelection2(True)


SKIN_MIN = 2.0     # mm; a side pocket may not leave a skin thinner than this


def _open_slot(prof, P, mw, sp):
    """A SIDE CUT MUST BREAK THROUGH A FACE, OR LEAVE min_wall -- in Z.

    The profiles' heights are ABSOLUTE millimetres designed around the Femur's
    plate (z -5..+5, recipe frame), and an absolute millimetre does not
    transfer between parts.  On the Side panel (plate 0..10) the 'hi' profile
    z 1.5..8.5 cut a closed slot into the edge with 1.5 mm skins above and
    below -- 1478 mm2 of the plate that carries the hip motor under 1.5 mm.
    A skin under SKIN_MIN is opened through the show face, and a floor under
    SKIN_MIN is raised to min_wall.  On the Femur and Coupler every skin is
    2.5 mm or more, so the locked links do not move."""
    zs_ = P["ZS"]
    zb_ = P.get("ZGROOVE", P["ZBACK"])
    zmax, zmin = max(q[1] for q in prof), min(q[1] for q in prof)
    top, bot = zs_ - zmax, zmin - zb_
    nmax, nmin = zmax, zmin
    if 0 < top < SKIN_MIN:
        nmax = P["ZTOP"] + 5.0
    if 0 < bot < SKIN_MIN:
        nmin = zb_ + mw
    if nmax - nmin < 1.0:
        return None
    if (nmax, nmin) != (zmax, zmin):
        print(f"    side profile x {min(q[0] for q in prof):.0f}..{max(q[0] for q in prof):.0f}: skins "
              f"top {top:.1f} / floor {bot:.1f} mm -> cut z {nmin:.1f}..{nmax:.1f} (recipe frame)")
    return [(x, nmax if abs(z - zmax) < 1e-9 else nmin if abs(z - zmin) < 1e-9 else z) for x, z in prof]


def _drop_detached(model, what, frac=0.01):
    """Delete free-floating scraps a feature cut loose -- the native twin of
    femur.py's _drop_detached().  Anything under `frac` of the main body is a
    scrap (on the Femur: a 21 mm3 piece of flange that two side pockets
    isolate, x -78.8..-74.9 -- the same piece the OCC build drops).  Anything
    bigger is a real defect and is left for the body-count check to stop on."""
    bs = S.bodies(model)
    if len(bs) < 2:
        return
    # volumes read ONCE, and the main body excluded by identity: GetMassProperties
    # is not bit-identical between calls, so "smaller than the largest" was once
    # true of the largest body itself and the Side panel lost its whole part
    vol = [S.volume(b) for b in bs]
    imax = max(range(len(bs)), key=lambda i: vol[i])
    big = vol[imax]
    scraps = [b for i, b in enumerate(bs) if i != imax and vol[i] < frac * big]
    if not scraps:
        return
    for b in scraps:
        print(f"    {what}: dropped a detached {S.volume(b):.1f} mm3 piece at x "
              f"{b.GetBodyBox()[0] * 1000:.1f}..{b.GetBodyBox()[3] * 1000:.1f}")
    S.delete_bodies(model, scraps)
    S.last_feature(model).Name = f"{what}_DropScraps"


def _largest_except(model, skip):
    return max((b for b in S.bodies(model) if b.Name not in skip), key=S.volume)


def _all(made):
    return {n for v in made.values() for n in v}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(" ".join(args) or "Femur", dry="--dry" in sys.argv, fresh="--fresh" in sys.argv,
         restyle="--restyle" in sys.argv)
