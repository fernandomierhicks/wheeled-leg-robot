"""Step 11: per-part checks on the native styled parts, plus the Bambu 3MFs.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/11_check_and_export.py [Part ...]

For every styled v5 part:

  1. FUSED STEP   colour features suppressed -> one styled body -> STEP, then
                  the colour features are switched back on.  Written to
                  out/styled/<Part>_glacier_native.step.
  2. OPENINGS     every opening of the SOURCE part must hold exactly the
                  material it held before (cad/aesthetics/lib/verify.py's test,
                  run on the SolidWorks geometry): "0 changed" is the pass.
                  Plus every vertical bore of the source still present.
  3. THIN WALL    lib/thinwall.compare_report, styled vs source: how much thin
                  wall the STYLING introduced (report 3.0 mm, gate 1.5 mm).
  4. 3MF          every body tessellated by SolidWorks itself, grouped by its
                  colour name -> Bambu project, filament 1 white / 2 graphite /
                  3 blue (export3mf.write_3mf, the writer his Bambu test passed).
                  Written to out/print/<Part>_glacier_native.3mf.
"""
import os
import re
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AES = os.path.normpath(os.path.join(HERE, "..", "aesthetics"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AES, "lib"))
import swlib
import swstyle as S
from swlib import c, wrap, sld

PARTS = {"Femur": r"Links\Femur.SLDPRT", "Coupler": r"Links\Coupler.SLDPRT",
         "Tibia": r"Links\Tibia.SLDPRT", "Side panel": r"Body\Side panel.SLDPRT",
         "RobotMount": r"Body\OldRobotBodyMount\RobotMount.SLDPRT",
         "Femur_inside": r"Links\Femur_inside.SLDPRT"}
COLOUR_FEATURE = re.compile(r"^GL_(blue|graphite|Cap|Skin|Graphite|Blue|Inset|White|DropDebris|C_)")
FILAMENT = {"white": 1, "graphite": 2, "blue": 3}
OUT = os.path.join(HERE, "out")


def _doc(sw, rel):
    path = os.path.join(swlib.V5, rel)
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        if os.path.normcase(d.GetPathName()) == os.path.normcase(path):
            return d
    doc, err, warn = sw.OpenDoc6(path, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
    return wrap(doc, sld.IModelDoc2)


def reload(sw, rel):
    return swlib.reload_from_disk(sw, os.path.join(swlib.V5, rel))


def colour_features(model, on):
    f = wrap(model.FirstFeature(), sld.IFeature)
    n = 0
    while f is not None:
        if COLOUR_FEATURE.match(f.Name):
            f.SetSuppression2(c.swUnSuppressFeature if on else c.swSuppressFeature,
                              c.swThisConfiguration, None)
            n += 1
        f = wrap(f.GetNextFeature(), sld.IFeature)
    model.EditRebuild3()
    return n


def body_mesh(b):
    """Vertices / triangles of one body, mm, from SolidWorks' own tessellation."""
    V = []
    for f in b.GetFaces() or []:
        t = wrap(f, sld.IFace2).GetTessTriangles(True)
        if t:
            V.append(np.asarray(t, float).reshape(-1, 3) * 1000.0)
    if not V:
        return None, None
    V = np.vstack(V)
    T = np.arange(len(V)).reshape(-1, 3)
    # weld coincident vertices so the mesh is closed for the slicer
    key = np.round(V, 4)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    return uniq, inv.reshape(-1)[T]


def check(part, model):
    from build123d import import_step
    import paths
    from keepout import openings
    from shputil import prism
    import thinwall
    from verify import vertical_cylinders

    os.makedirs(os.path.join(OUT, "styled"), exist_ok=True)
    step = os.path.join(OUT, "styled", f"{part}_glacier_native.step")
    n = colour_features(model, False)
    nb = len(S.bodies(model))
    ok, err, warn = model.Extension.SaveAs3(step, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy,
                                            None, None, 0, 0)
    colour_features(model, True)
    print(f"  fused STEP: {n} colour features off -> {nb} body -> {os.path.basename(step)} ({ok})")

    src = import_step(paths.part_step(part))
    sty = import_step(step)
    bb = src.bounding_box()
    # EVERY horizontal face's openings, not just the two outermost faces' --
    # the outer-faces-only test missed 1/3 to 1/2 of them (counterbores and
    # holes starting on a recessed face); see 08_style_part.all_openings()
    from importlib import import_module
    ops = import_module("08_style_part").all_openings(src.solids()[0])
    worst, bad = 0.0, []
    for p in sorted(ops, key=lambda p: -p.area):
        core = p.buffer(-0.35)
        if core.is_empty:
            continue
        pr = prism(core, bb.min.Z - 12, bb.max.Z + 40)
        a, b = (src & pr), (sty & pr)
        d = abs((0.0 if b is None else b.volume) - (0.0 if a is None else a.volume))
        worst = max(worst, d)
        if d > 1.0:
            bad.append((p.centroid.x, p.centroid.y, d))
    miss = sorted(vertical_cylinders(src) - vertical_cylinders(sty))
    print(f"  openings: {len(ops)} checked, {len(bad)} changed, worst {worst:.3f} mm3"
          + "".join(f"\n     CHANGED at ({x:.1f},{y:.1f}) {d:.1f} mm3" for x, y, d in bad))
    print(f"  bores missing: {len(miss)}" + "".join(f"\n     D{d} at ({x},{y})" for d, x, y in miss))
    try:
        wall_ok, _ = thinwall.compare_report(sty, paths.part_step(part), f"{part} native", 3.0,
                                             gate_wall=thinwall.GATE_WALL)
        print(f"  thin-wall gate (1.5 mm, styling-introduced): {'PASS' if wall_ok else 'FAIL'}")
    except Exception as e:      # OCC's ray intersector can die on one degenerate face
        wall_ok = None
        print(f"  thin-wall: NOT MEASURED -- OCC ray intersector failed ({type(e).__name__})")
    return len(bad) == 0 and not miss, wall_ok


def export_3mf(part, model):
    import export3mf
    parts = []
    for b in S.bodies(model):
        g = S.group_of(b)
        V, T = body_mesh(b)
        if V is not None:
            parts.append((b.Name, V, T, FILAMENT[g]))
    os.makedirs(os.path.join(OUT, "print"), exist_ok=True)
    out = os.path.join(OUT, "print", f"{part}_glacier_native.3mf")
    export3mf.write_3mf(parts, out, name=f"{part} GLACIER (native SolidWorks)")
    by = {}
    for nm, V, T, fil in parts:
        by[fil] = by.get(fil, 0) + 1
    print(f"  3MF: {len(parts)} bodies (filament 1 white x{by.get(1, 0)}, 2 graphite x{by.get(2, 0)}, "
          f"3 blue x{by.get(3, 0)}) -> {out}")


def main(which):
    sw, _ = swlib.connect()
    summary = []
    for part in which:
        print(f"\n=== {part}")
        model = _doc(sw, PARTS[part])
        holes_ok, wall_ok = check(part, model)
        # Toggling the colour features rebuilt the bodies: names reset and on
        # the Tibia some bodies came back with NO colour.  The saved file is
        # intact, so reload it rather than patch the in-memory copy.
        model = reload(sw, PARTS[part])
        export_3mf(part, model)
        summary.append((part, holes_ok, wall_ok))
    print("\nSUMMARY")
    for part, h, w in summary:
        print(f"  {part:11s} openings+bores {'PASS' if h else 'FAIL'}   thin wall "
              f"{'not measured' if w is None else 'PASS' if w else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(PARTS))
