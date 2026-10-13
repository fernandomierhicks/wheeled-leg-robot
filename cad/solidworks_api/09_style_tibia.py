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
  colour    flush 0.6 mm inlays on a white body (README standing rule: accents are
            skin, the inside is white): blue accent + graphite collars on the show
            face, graphite back trace, graphite pocket / window floors.  Pass is the
            same as 08_style_part's.

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
ACC_D = 0.6                       # every accent is 0.6 mm (README standing rule)


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

    acc = unary_union([g for g in (P["ring"], P["strip"], P["blocks"]) if not g.is_empty])
    zbk = P["ZBACK"]

    print(f"  side   {len(sides)} profiles, each in its own plan trapezoid")
    print(f"  colour inlays {ACC_D} mm: accent {acc.area:.0f} mm2, back trace {P['grey'].area:.0f}, "
          f"collars {P['collars'].area:.0f}, pocket floors {P['pock'].area:.0f}, window floors {P['wins'].area:.0f}")
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

    for kind, name, g, args in ops_list:
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
    styled = S.total_volume(model)
    body = S.bodies(model)[0]

    # ---- colour: his 0.6 mm rule (README standing rules) -----------------------
    # The body stays white; every accent is a flush 0.6 mm inlay (3 layers), so the
    # inside is white.  Same pass as 08_style_part: copy white, cut each slab out of
    # it, Combine the difference = the inlay (accent first, so it wins where they cross).
    ZS = ZT                                   # the real show-face plate (ZSHOW == ZT here)
    ZP = ZT - sp["pocket_d"]                  # pocket floor
    ZW = ftop - sp["win_d"]                   # window floor
    D = ACC_D
    cut_g = {name: g for kind, name, g, args in ops_list}      # what was really cut (after the seat keep-out)
    colours = [("blue", ST.BLUE, acc, [(ZS - D, ZTOP + 12)]),
               ("graphite", ST.GRAPHITE, P["grey"], [(ZB - 12, zbk + D)]),
               ("graphite_c", ST.GRAPHITE, P["collars"], [(ZS - D, ZTOP + 12)]),
               # his mark: a thick grey accent on the frame top where the wheel-end cutout gave a band back
               ("graphite_t", ST.GRAPHITE, P["cut_freed"].buffer(-0.8, join_style=2).intersection(cut_g["GL_Frame"]),
                [(ftop - D, ZTOP + 12)]),
               ("graphite_p", ST.GRAPHITE, cut_g["GL_Pockets"], [(ZP - D, ZP)]),
               ("graphite_w", ST.GRAPHITE, cut_g["GL_Windows"], [(ZW - D, ZW)])]
    made = {}
    for nm, rgb, g, slabs in colours:
        if g is None or g.is_empty:
            continue
        sk = S.sketch(model, g, name=f"GL_{nm}_Sk")
        done = ST._all(made)
        white = ST._largest_except(model, done)
        w1 = S.copy_body(model, white, name=f"GL_{nm}_Copy")
        cut_any = False
        for i, (lo, hi) in enumerate(slabs):
            white = ST._largest_except(model, done | {w1.Name})
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
                    except S.FeatureFailed:
                        S.delete_bodies(model, [b for b in (S.by_name(model, wc.Name),
                                                            S.by_name(model, piece.Name)) if b])
            S.delete_bodies(model, [S.by_name(model, w1.Name)])
            made[nm] = [n for n in got if S.by_name(model, n) is not None]
        print(f"    colour {nm:10s} -> {len(made[nm])} body(ies), "
              f"{sum(S.volume(b) for b in S.bodies(model) if b.Name in made[nm]):8.1f} mm3")

    dropped = ST.drop_debris(model)
    whites = sorted([b for b in S.bodies(model) if b.Name not in ST._all(made)], key=lambda b: -S.volume(b))
    for i, w in enumerate(whites, 1):
        S.colour(w, ST.WHITE, "white" if i == 1 else f"white_{i}")
    for nm, rgb, _, _ in colours:
        live = [b for b in S.bodies(model) if b.Name in made.get(nm, [])]
        for i, b in enumerate(live, 1):
            S.colour(b, rgb, f"{nm}_{i}")
    tot = S.total_volume(model)
    print(f"\n  source {v_src:10.1f} mm3 -> styled {styled:10.1f} mm3 ({styled - v_src:+.1f}); "
          f"after colour split {tot:10.1f} (+{dropped:.2f} debris, must equal styled)")
    if abs(tot + dropped - styled) > max(2.0, 1e-4 * styled):
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
