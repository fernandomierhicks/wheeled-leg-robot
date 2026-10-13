"""Step 32: Femur -- fill the voids round the hip (AK45) end's screw boss. ADDS material only.

    cadenv/python 32_femur_motorfill.py [--dry]

His call 2026-10-09: "the end attached to the motor has deep cutouts that make no sense
mechanically (a cut round the circumference from both faces that nearly meets in the middle);
OK to add material, don't remove any".  The disc (centre A = (-93.79, 0) in the Femur frame,
r 19.4, z -5..5; the AK45 rotor sits on the z = +5 face) has two big voids: 646 mm3 from the +z
face (z -2.5..5) and 509 mm3 from the -z face (z -5..-1.5), leaving a ~1 mm web.  Each is
filled flush: fill = cylinder(r 19.4, z -5..5) - every Femur body, pieces >= 100 mm3 only (the
smaller bits are the relief grooves + hole rings, left as they are).
Checked offline against the rotor / stator / wheel motor (0 mm3 overlap, the rotor starts at
z 5.0 and the M2.5 heads are below z -5) -- the fill is inside the disc's own swept cylinder.
In SolidWorks: two Imported features `MF_motorfill_1/2` (NOT `RL_`: 28 build would delete them),
white bodies, nothing merged, so no mate face changes.  Then 25 mirror-colours Femur + 16 chain.
Delete MF_motorfill_* before re-running 28 build on Femur (its body check).
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
R = importlib.import_module("28_relief")
R._aes()
from build123d import Cylinder, Pos

A = (-93.79, 0.0)
R_DISC, Z0, Z1 = 19.4, -5.0, 5.0
MIN_PIECE = 100.0


def pieces():
    P = R.Part("Femur", built=True)
    fill = Pos(A[0], A[1], (Z0 + Z1) / 2) * Cylinder(R_DISC, Z1 - Z0)
    for b in P.bodies:
        fill = fill - b["solid"]
    out = [p for p in fill.solids() if p.volume >= MIN_PIECE]
    return P, sorted(out, key=lambda p: -p.volume)


def main():
    dry = "--dry" in sys.argv
    import swlib
    import swstyle as S
    import stepcolor
    from swlib import c, wrap, sld
    P, pcs = pieces()
    white = max(P.bodies, key=lambda b: b["volume"])
    print(f"{len(pcs)} fill pieces: {[round(p.volume, 1) for p in pcs]} mm3")
    if len(pcs) != 2:
        raise SystemExit("expected the two big voids -- re-check the export")
    stepcolor.MIN_BODY_MM3 = 0.0
    step = os.path.join(R.OUT, "Femur_motorfill.step")
    stepcolor.write([(f"mf{k}", p, (237, 239, 242)) for k, p in enumerate(pcs)], step, part="Femur_motorfill",
                    say=lambda *a: None)
    if dry:
        return
    sw, _ = swlib.connect()
    swlib.open_v5(sw)
    path = os.path.join(swlib.V5, R.PARTS["Femur"])
    doc = wrap(sw.OpenDoc6(path, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)[0], sld.IModelDoc2)
    sw.ActivateDoc3(doc.GetTitle(), False, 0, 0)
    old = [f for f in S._iter_features(doc) if f.Name.startswith("MF_motorfill")]
    if old:
        doc.ClearSelection2(True)
        for k, f in enumerate(old):
            f.Select2(k > 0, 0)
        doc.Extension.DeleteSelection2(c.swDelete_Absorbed | c.swDelete_Children)
        doc.EditRebuild3()
    n0 = len(S.bodies(doc))
    v0 = sum(S.volume(b) for b in S.bodies(doc))
    pdoc = wrap(doc, sld.IPartDoc)
    wrgb = list(max(S.bodies(doc), key=S.volume).MaterialPropertyValues2[:3])
    src = R._import_bodies(sw, doc, step)
    if len(src) != len(pcs):
        raise SystemExit(f"{len(src)} bodies in the STEP, expected {len(pcs)}")
    for k, (v, body) in enumerate(sorted(src, key=lambda x: -x[0]), 1):
        want = pcs[k - 1].volume
        names0 = {b.Name for b in S.bodies(doc)}
        f = wrap(pdoc.CreateFeatureFromBody3(body, False, c.swCreateFeatureBodyCheck), sld.IFeature)
        if f is None:
            raise SystemExit("SolidWorks refused a fill body")
        f.Name = f"MF_motorfill_{k}"
        new = [b for b in S.bodies(doc) if b.Name not in names0]
        if len(new) != 1 or abs(S.volume(new[0]) - want) > max(0.2, 1e-3 * want):
            raise SystemExit(f"MF_motorfill_{k}: {len(new)} bodies, {S.volume(new[0]) if new else 0:.2f} vs {want:.2f} mm3")
        S.colour(new[0], wrgb)
        print(f"  MF_motorfill_{k}: {S.volume(new[0]):.1f} mm3 (predicted {want:.1f}) -> white", flush=True)
    doc.EditRebuild3()
    v1 = sum(S.volume(b) for b in S.bodies(doc))
    print(f"  bodies {n0} -> {len(S.bodies(doc))}; volume {v0:.1f} -> {v1:.1f} (+{v1 - v0:.1f}); nothing removed")
    importlib.import_module("25_fasteners").mirror_colours(only=["Femur"])
    importlib.import_module("16_check_and_save").chain(
        sw, [R.PARTS["Femur"], r"Links\MirrorFemur.SLDPRT"], quick="--quick" in sys.argv)


if __name__ == "__main__":
    main()
