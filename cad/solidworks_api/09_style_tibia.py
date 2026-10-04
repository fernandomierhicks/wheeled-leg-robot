"""Step 9: the Tibia's GLACIER as native SolidWorks features.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/09_style_tibia.py [--dry] [--fresh]

The Tibia is the one recipe NOT generated from femur.py (decision 05/14: its
band layout and its colour scheme were approved on their own), so it gets its
own replay of parts/tibia.py build():

  geometry  flange, drafted frame, pockets, windows, through-cutouts, accent
            engraving -- as for the Femur -- and SIDE POCKETS PER PROFILE:
            each X-Z profile is cut only inside its OWN plan trapezoid
            (lo_rk/hi_rk), which is what makes the openings diagonal.  The
            trapezoids overlap their neighbours' profiles, so they cannot be
            grouped per side.
  colour    NOT inlays.  White is the CAP above a split plane SZ with
            interlocking trapezoid fingers through the side, plus a white
            skin on the back with the graphite trace cut out of it; blue is
            the accent groove inside that; graphite is everything else.

SZ = -8.4375 is what tibia.py's white_share bisection lands on for the 70 %
target: recomputed with the recipe's own colour rule against the OCC styled
body it gives 68.8 % white -- the exported OCC white body to the cubic
millimetre (148036.7 / 215116.9).

Same mechanical keep-out as the Femur: no removal and no raised or added
feature on a fastener head / washer seat or a bearing retaining washer.
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
from shapely.geometry import Polygon as ShPoly, box as shbox
from shapely.ops import unary_union
from shapely.validation import make_valid      # NOT buffer(0): HANDOFF Phase 1

ST = importlib.import_module("08_style_part")
REL = r"Links\Tibia.SLDPRT"
SZ = -8.4375                      # see module docstring
SKIN = 3.5                        # tibia.py back_skin default


def trap_xz(x0, x1, z0, z1, skew=5.0, taper_end=True):
    """feat.trap_xz, copied so this file does not import build123d for it."""
    if taper_end:
        return [(x0 + skew, z0), (x0, z1), (x1 - skew, z1), (x1, z0)]
    return [(x0, z0), (x0 + skew, z1), (x1, z1), (x1 - skew, z0)]


def xz(poly_pts):
    """Recipe (x, z) -> Top Plane sketch coordinates (x, -z)."""
    return ShPoly([(x, -z) for x, z in poly_pts])


def main(dry=False, fresh=False, restyle=False):
    T = importlib.import_module("tibia")
    sp = json.load(open(os.path.join(AES, "specs", "tibia.json")))
    t0 = time.time()
    P = T.plan(sp)
    src = T.source()
    print(f"Tibia: plan() {time.time() - t0:.0f} s, show face +Z (no mirror)")
    M, ops = ST.mech_keepout(src)
    M_cut = M.buffer(0.3, 48)
    M_add = M.buffer(0.6, 48)   # additions clear the seat FURTHER than removals: see mech()
    # where the styling's ADDITIONS ran into a neighbour anywhere in the stroke
    # (13_collision_keepout.py, measured by SolidWorks): additions stay out
    _ck = os.path.join(HERE, "out", "collision_keepout.json")
    if os.path.exists(_ck):
        from shapely.geometry import shape as _shape
        _kc = [_shape(g) for g in json.load(open(_ck)).get("Tibia", [])]
        if _kc:
            K_col = unary_union(_kc)
            M_add = unary_union([M_add, K_col])
            print(f"  collision keep-out for the additions: {K_col.area:.0f} mm2")
    print(f"  {len(ops)} openings -> mechanical keep-out {M.area:.0f} mm2 (recipe kb {P['kb'].area:.0f})")

    def mech(g, what, removal=True):
        if g is None or g.is_empty:
            return g
        K = M_cut if removal else M_add
        lost = g.intersection(K).area
        if lost > 0.5:
            print(f"    {what}: {lost:6.1f} mm2 of {g.area:6.1f} sat on a fastener/bearing seat or in a neighbour's path -> removed")
        return g.difference(K)

    ZT, ZB, ZTOP = P["ZT"], P["ZB"], P["ZTOP"]
    CH = sp["chamfer"]
    ops_list = []
    fl = P["flange"]
    ops_list.append(("boss", "GL_Flange", mech(fl[2], "flange", False), (fl[0], fl[1])))
    for key, h, off, draft in (("frame", sp["frame_h"], 1.0, 12), ("rail", sp["rail_h"], 0.8, 10),
                               ("pads", sp["pad_h"], 0.8, 16)):
        base = ZT - CH - off
        if base + h + off - ZT < 0.3:
            print(f"    {key}: top sits {base + h + off - ZT:+.2f} mm off the plate (buried, as in the OCC build) -> skipped")
            continue
        if not P[key].is_empty:
            ops_list.append(("raised", f"GL_{key.capitalize()}", mech(P[key], key, False), (base, h + off, True, draft)))
    ops_list.append(("cut", "GL_Pockets", mech(P["pock"], "pockets"), (ZT - sp["pocket_d"], ZTOP + 10)))
    ftop = ZT - CH - 1.0 + sp["frame_h"] + 1.0
    ops_list.append(("cut", "GL_Windows", mech(P["wins"], "windows"), (ftop - sp["win_d"], ZTOP + 12)))
    ops_list.append(("cut", "GL_Cutouts", mech(P["cutouts"], "through-cutouts"), (ZB - 10, ZTOP + 12)))
    grown = P["grown"]
    band = grown.buffer(4.0, join_style=2).difference(grown.buffer(-sp["side_d"], join_style=2))
    band = make_valid(mech(make_valid(band).difference(make_valid(P["kb"])), "side pockets"))
    sides = []
    for side, prs, rks, half in (("Lo", P["lo_pr"], P["lo_rk"], shbox(-1e3, -1e3, 1e3, 0)),
                                 ("Hi", P["hi_pr"], P["hi_rk"], shbox(-1e3, 0, 1e3, 1e3))):
        for i, (pr, rk) in enumerate(zip(prs, rks)):
            # the recipe's plan trapezoids can self-touch (GEOS "side location
            # conflict"); make_valid repairs without buffer(0)'s pinched lobes
            b = band.intersection(half).intersection(make_valid(rk))
            if b.area > 1.0:
                zs = [q[1] for q in pr]
                sides.append((f"GL_Side{side}{i}", b, xz(pr), min(zs), max(zs)))
    eng = unary_union([g for g in (P["strip"], P["blocks"]) if not g.is_empty])
    ops_list.append(("cut", "GL_AccentEngrave", mech(eng, "accent engraving"), (ZT - 1.2, ZTOP + 10)))

    acc = unary_union([g for g in (P["ring"], P["strip"], P["blocks"]) if not g.is_empty])
    inset = unary_union([P["pock"].buffer(0.25), P["wins"].buffer(0.25)])
    zbk = P["ZBACK"]
    region = unary_union([shbox(-1e3, SZ, 1e3, 1e3)] +
                         [ShPoly(trap_xz(a, b, SZ - 4.5, SZ, skew=6)) for a, b in [(58, 102), (138, 182)]])
    region = region.difference(unary_union([ShPoly(trap_xz(a, b, SZ, SZ + 14.5, skew=6, taper_end=False))
                                            for a, b in [(104, 136), (184, 208)]]))
    region_sk = ShPoly([(x, -z) for x, z in region.exterior.coords]) if region.geom_type == "Polygon" else \
        unary_union([ShPoly([(x, -z) for x, z in p.exterior.coords]) for p in region.geoms])

    for kind, name, g, args in ops_list:
        print(f"  {kind:6s} {name:18s} {g.area:7.1f} mm2  {args}")
    print(f"  side   {len(sides)} profiles, each in its own plan trapezoid")
    print(f"  colour cap above SZ {SZ} with fingers; back skin z {zbk - 1:.2f}..{zbk + SKIN:.2f} minus the "
          f"trace ({P['grey'].area:.0f} mm2); accent {acc.area:.0f} mm2; inset {inset.area:.0f}; collars {P['collars'].area:.0f}")
    if dry:
        return

    path = os.path.join(swlib.V5, REL)
    bak = os.path.join(ST.ORIGINALS, REL)
    if not os.path.exists(bak):
        os.makedirs(os.path.dirname(bak), exist_ok=True)
        shutil.copy2(path, bak)
        print(f"  backed up the original to {bak}")
    sw, _ = swlib.connect()
    model = _doc(sw, path)
    if model is None:
        doc, err, warn = sw.OpenDoc6(path, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
        model = wrap(doc, sld.IModelDoc2)
    sw.ActivateDoc3(model.GetTitle(), False, 0, 0)
    if fresh and any(f.Name.startswith("GL_") for f in ST._features(model)):
        if open(path, "rb").read() != open(bak, "rb").read():
            raise SystemExit("--fresh: the file on disk is not the original any more")
        model.ReloadOrReplace(False, path, True)
        model = _doc(sw, path)
        print("  --fresh: reloaded the unstyled part from disk")
    if restyle and any(f.Name.startswith("GL_") for f in S._iter_features(model)):
        k = S.strip_styling(model, src["solid"].volume)
        print(f"  --restyle: deleted {k} GL_ features -> the original part again")
    if any(f.Name.startswith("GL_") for f in ST._features(model)):
        raise SystemExit("already styled -- use --fresh or restore from _originals")
    bs = S.bodies(model)
    sb = src["solid"].bounding_box()
    bx = [v * 1000 for v in bs[0].GetBodyBox()]
    dev = max(abs(a - b) for a, b in zip(bx, (sb.min.X, sb.min.Y, sb.min.Z, sb.max.X, sb.max.Y, sb.max.Z)))
    dv = abs(S.volume(bs[0]) - src["solid"].volume)
    print(f"  SolidWorks body vs STEP source: box worst {dev:.3f} mm, volume {dv:.1f} mm3")
    # GetBodyBox is LOOSE around curved faces (Tibia +Y: 21.224 vs the true
    # 21.000), so the box is checked loosely and the volume tightly.
    if len(bs) != 1 or dev > 0.5 or dv > 1e-4 * src["solid"].volume:
        raise SystemExit("the .SLDPRT is not the STEP's single body in the STEP's frame")
    v_src = S.total_volume(model)
    t1 = time.time()

    def report(name, v0):
        ST._drop_detached(model, name)
        nb = len(S.bodies(model))
        print(f"    {name:18s} -> {S.total_volume(model):10.1f} mm3 ({S.total_volume(model) - v0:+8.1f}), {nb} body")
        if nb != 1:
            raise SystemExit(f"{name} left {nb} bodies")

    for kind, name, g, args in ops_list[:-1]:          # all but the engraving
        v0 = S.total_volume(model)
        if ST._with_retries(model, kind, name, g, args) is not None:
            report(name, v0)
    for name, b, prof, zlo, zhi in sides:
        v0 = S.total_volume(model)
        before = S.names(model)
        sk = S.sketch(model, b, name=name + "_Sk")
        S.tool(model, sk, zlo - 0.5, zhi + 0.5, name=name + "_Band")
        psk = S.sketch(model, prof, name=name + "_ProfSk", plane="Top Plane")
        try:
            S.cut(model, psk, scope=S.new_since(model, before), outside=True, name=name + "_Trim")
            S.combine(model, "cut", ST._largest(model), S.new_since(model, before))
            S.last_feature(model).Name = name
        except S.FeatureFailed:
            S.delete_bodies(model, S.new_since(model, before))
            print(f"    {name}: profile misses its band -> skipped")
        report(name, v0)
    kind, name, g, args = ops_list[-1]
    v0 = S.total_volume(model)
    if ST._with_retries(model, kind, name, g, args) is not None:
        report(name, v0)
    styled = S.total_volume(model)
    body = S.bodies(model)[0]

    # ---- colour ---------------------------------------------------------------
    def scoped(sk, targets, **kw):
        """A feature-scoped cut.  If SolidWorks refuses it for the group -- it
        does when ANY one selected body misses the cut -- retry body by body,
        so a genuine "no overlap" on one body cannot silently cancel the cut on
        the others (the Tibia's insets kept 6433 mm3 of accent that way)."""
        if not targets:
            return False
        try:
            S.cut(model, sk, scope=targets, **kw)
            return True
        except S.FeatureFailed:
            if len(targets) == 1:
                return False
        names = [t.Name for t in targets]
        hit = False
        for i, nm in enumerate(names):
            b = S.by_name(model, nm)
            if b is None:
                continue
            k = dict(kw)
            if "name" in k:
                k["name"] = f"{k['name']}_{i}"
            try:
                S.cut(model, sk, scope=[b], **k)
                hit = True
            except S.FeatureFailed:
                pass
        return hit

    # THE ORIGINAL BODY BECOMES WHITE.  Mates in Tibia.SLDASM reference faces
    # of the original body (bearing bores and seats, the stator face, the M4
    # seat); a split that turned the original into the graphite remainder and
    # the cap into COPIES broke 9 of them (swSketchErrorExtRefFail, 51).  So:
    #   white    = original - band - trace-in-skin - (accent | insets | collars)
    #   blue     = copy     - band - trace-in-skin - (everything but accent)
    #   graphite = copy     - cap region - skin                          [band]
    #            + copy     - above the skin - (all but the trace)       [skin x trace]
    #            + copy     - band - trace-in-skin - (all but insets) - accent
    # (a white finger dipping into the skin under the trace goes graphite: the
    # one deviation from tibia.py, a few mm3, and it keeps the split exact)
    # where wz = cap region (above SZ, with fingers) + back skin minus trace.
    # NORMAL cuts only (gotcha 28), no Combine, every body from the same source
    # body, so the four lineages partition it exactly -- checked by volume below.
    # These lineages are cut with DIFFERENT sketches, each shrunk 5 um, so two
    # lineages can each own a ~10 um film along a shared colour boundary (387
    # mm3 = 0.18 % on the Tibia).  Exact (unshrunk) sketches were tried: the
    # "everything but accent" complement is then refused outright and blue
    # swallowed the whole cap.  Films a twentieth of a layer thick are below
    # anything the printer resolves, so the partition check allows 0.5 % here.
    big = shbox(-1e3, -1e3, 1e3, 1e3)
    band_xz = shbox(-1e3, zbk + SKIN, 1e3, 1e3).difference(region)          # recipe (x, z)
    flip = lambda g: unary_union([ShPoly([(x, -z) for x, z in q.exterior.coords],
                                         [[(x, -z) for x, z in r.coords] for r in q.interiors])
                                  for q in S.polys(g, 0)])
    ic = unary_union([inset, P["collars"]])
    sk_band = S.sketch(model, flip(band_xz), name="GL_C_BandSk", plane="Top Plane")
    sk_top = S.sketch(model, region_sk, name="GL_C_CapSk", plane="Top Plane")
    sk_grey = S.sketch(model, P["grey"], name="GL_C_TraceSk")
    sk_notgrey = S.sketch(model, big.difference(P["grey"]), name="GL_C_NotTraceSk")
    sk_holes = S.sketch(model, unary_union([acc, ic]), name="GL_C_HolesSk")
    sk_notacc = S.sketch(model, big.difference(acc), name="GL_C_NotAccentSk")
    sk_notic = S.sketch(model, big.difference(ic), name="GL_C_NotInsetSk")
    sk_acc = S.sketch(model, acc, name="GL_C_AccentSk")
    sk_big = S.sketch(model, big, name="GL_C_BigSk")

    copies = {}
    for tag in ("Blue", "GraphA1", "GraphA2", "GraphB"):
        b0 = S.names(model)
        S.copy_body(model, S.by_name(model, body.Name), name=f"GL_C_{tag}_Copy")
        copies[tag] = {S.new_since(model, b0)[0].Name}
    lineage = {"White": {body.Name}, **copies}

    def run(tag, steps):
        """Apply `steps` (sketch, z0, z1) as scoped normal cuts to one lineage,
        tracking its bodies as everything not owned by another lineage."""
        for i, (sk, z0, z1) in enumerate(steps):
            others = set().union(*[v for k, v in lineage.items() if k != tag])
            mine = [b for b in S.bodies(model) if b.Name not in others]
            if scoped(sk, mine, z0=z0, z1=z1, name=f"GL_C_{tag}_{i}") is False:
                print(f"    colour {tag} step {i}: no overlap")
            others = set().union(*[v for k, v in lineage.items() if k != tag])
            lineage[tag] = {b.Name for b in S.bodies(model) if b.Name not in others}

    skin = (-1e3, zbk + SKIN)
    run("White", [(sk_band, None, None), (sk_grey, *skin), (sk_holes, None, None)])
    run("Blue", [(sk_band, None, None), (sk_grey, *skin), (sk_notacc, None, None)])
    # body - wz, in two lineages: a blind cut of "everything but the trace" was
    # refused outright, so the trace part of the skin is its own copy
    run("GraphA1", [(sk_top, None, None), (sk_big, *skin)])              # band, fingers' gaps
    run("GraphB", [(sk_band, None, None), (sk_grey, *skin), (sk_notic, None, None), (sk_acc, None, None)])
    # skin x trace, BY REMAINDER: a cut with "everything but the trace" is
    # refused outright (that sketch has an island nested in a hole), so
    #   skin x trace = skin - (skin - trace), trace being a plain cut
    run("GraphA2", [(sk_big, zbk + SKIN, 1e3)])                            # the skin slab
    slab = sorted(lineage["GraphA2"], key=lambda n: -S.volume(S.by_name(model, n)))
    owned = set().union(*lineage.values())
    for n in slab:
        S.copy_body(model, S.by_name(model, n), name="GL_C_GraphA2_Rest")
    rest = [b for b in S.bodies(model) if b.Name not in owned]
    scoped(sk_grey, rest, name="GL_C_GraphA2_NoTrace")
    rest = [b for b in S.bodies(model) if b.Name not in owned]
    before = S.names(model)
    S.combine(model, "cut", S.by_name(model, slab[0]), rest)
    S.last_feature(model).Name = "GL_C_GraphA2"
    lineage["GraphA2"] = (set(slab[1:]) | {b.Name for b in S.new_since(model, before)}) - {
        b.Name for b in rest}
    blue_names = lineage["Blue"]
    graphite = lineage["GraphA1"] | lineage["GraphA2"] | lineage["GraphB"]
    for k, v in lineage.items():
        print(f"    {k:7s} {len(v):3d} piece(s) {sum(S.volume(S.by_name(model, n)) for n in v):10.1f} mm3")

    dropped = ST.drop_debris(model)
    gnames = graphite
    for i, b in enumerate(S.bodies(model)):
        # UNIQUE temporary names: giving every blue body the same name collides,
        # and SolidWorks keeps only one of them under it
        b.Name = ("blue" if b.Name in blue_names else "graphite" if b.Name in gnames
                  else "white") + f"_tmp{i}"
    n = ST.recolour(model)
    tot = S.total_volume(model)
    print(f"\n  source {v_src:10.1f} mm3 -> styled {styled:10.1f} mm3 ({styled - v_src:+.1f}); colour split {tot:10.1f} "
          f"(+{dropped:.2f} debris) {n}")
    for nm in ("white", "blue", "graphite"):
        v = sum(S.volume(b) for b in S.bodies(model) if b.Name.split("_")[0] == nm)
        print(f"    {nm:9s} {v:10.1f} mm3 ({100 * v / tot:.1f} %)")
    film = tot + dropped - styled
    print(f"  colour bodies vs styled part: {film:+.1f} mm3 ({100 * film / styled:+.3f} %) -- "
          f"overlap films along colour boundaries, allowed up to 0.5 %")
    if abs(film) > 0.005 * styled:
        raise SystemExit("the colour bodies do not add up to the styled part")
    model.ShowNamedView2("*Isometric", c.swIsometricView)
    model.ViewZoomtofit2()
    ok, err, warn = model.Extension.SaveAs3(path, c.swSaveAsCurrentVersion, c.swSaveAsOptions_Silent, None, None, 0, 0)
    if not ok:
        raise SystemExit(f"save failed ({err})")
    print(f"  saved {path} ({time.time() - t1:.0f} s of SolidWorks)")


def _doc(sw, path):
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        if os.path.normcase(d.GetPathName()) == os.path.normcase(path):
            return d
    return None


if __name__ == "__main__":
    main(dry="--dry" in sys.argv, fresh="--fresh" in sys.argv,
         restyle="--restyle" in sys.argv)
