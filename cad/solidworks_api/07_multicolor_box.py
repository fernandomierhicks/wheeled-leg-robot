"""Step 7: can a native SolidWorks part print in several colours on the Bambu?

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/07_multicolor_box.py

Starts from the step-2 box (cad/v5 Ai designed/ai_components/hello_box.SLDPRT),
saves it as hello_box_multicolor.SLDPRT and adds the two GLACIER situations as
SEPARATE BODIES:

  * INLAY  (blue)     a 3 mm 45-degree circuit trace cut 1.2 mm into the top,
                      then filled by a second body from the same sketch
  * PADS   (graphite) two raised pads on the top face, 2 mm, 10 deg draft
  * BOX    (white)    what is left of the cube

Then exports it three ways, to find which one Bambu Studio takes best:
  hello_box_multicolor.STEP        SolidWorks STEP AP214 (colours per body)
  hello_box_multicolor_sw.3mf      SolidWorks' own 3MF export
  hello_box_multicolor_bambu.3mf   Bambu project, filaments pre-assigned
                                   1 = white box, 2 = graphite pads, 3 = blue inlay
                                   (cad/aesthetics/lib/export3mf.py, the writer
                                   the GLACIER prints already use)
"""
import os
import sys
import math
import zipfile
from shapely.geometry import LineString

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "aesthetics", "lib"))
import swlib
from swlib import c, wrap, sld

DIR = os.path.join(swlib.V5, "ai_components")
SRC = os.path.join(DIR, "hello_box.SLDPRT")
OUT = os.path.join(DIR, "hello_box_multicolor")

COLOURS = {                     # name: (rgb 0..1, Bambu filament slot)
    "white_box":     ((0.95, 0.95, 0.95), 1),
    "graphite_pad":  ((0.22, 0.22, 0.24), 2),
    "blue_inlay":    ((0.10, 0.40, 0.95), 3),
}
INLAY_D, PAD_H, PAD_DRAFT = 0.0012, 0.002, 10.0     # m, m, deg


def must(ok, what):
    if not ok:
        raise SystemExit(f"FAILED: {what}")


def bodies(model):
    part = wrap(model, sld.IPartDoc)
    return [wrap(b, sld.IBody2) for b in (part.GetBodies2(c.swSolidBody, False) or [])]


def box_of(b):
    return [x * 1000 for x in b.GetBodyBox()]          # mm: xmin ymin zmin xmax ymax zmax


def vol(b):
    return b.GetMassProperties(1.0)[3] * 1e9           # mm3


def sketch_on_face(model, x, y, z):
    """Open a sketch on the face under model point (x, y, z) (metres)."""
    model.ClearSelection2(True)
    must(model.Extension.SelectByID2("", "FACE", x, y, z, False, 0, None, 0),
         f"select face at {x, y, z}")
    model.SketchManager.InsertSketch(True)


def last_feature(model):
    f, last = wrap(model.FirstFeature(), sld.IFeature), None
    while f is not None:
        last, f = f, wrap(f.GetNextFeature(), sld.IFeature)
    return last


def close_and_select(model):
    """Close the sketch and select it.  Returns its IFeature, fetched from the
    TREE: the ActiveSketch object goes stale once the sketch is closed, and
    Select2 on it fails with "Invalid number of parameters"."""
    model.SketchManager.InsertSketch(True)
    sk = last_feature(model)
    must(sk.GetTypeName2() == "ProfileFeature", f"last feature is {sk.GetTypeName2()}, not a sketch")
    model.ClearSelection2(True)
    must(sk.Select2(False, 0), f"select {sk.Name}")
    return sk


def polygon(model, coords_mm):
    sm = model.SketchManager
    pts = [(x / 1000, y / 1000) for x, y in coords_mm]
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        must(sm.CreateLine(x1, y1, 0, x2, y2, 0), "sketch line")


sw, _ = swlib.connect()
for d in sw.GetDocuments() or []:                 # a half-built copy from a failed run
    d = wrap(d, sld.IModelDoc2)
    if os.path.normcase(d.GetPathName()) in (os.path.normcase(OUT + ".SLDPRT"),
                                             os.path.normcase(SRC)):
        print("closing leftover", d.GetTitle())
        sw.CloseDoc(d.GetTitle())
doc, err, warn = sw.OpenDoc6(SRC, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
must(doc, f"open {SRC} (err {err})")
model = wrap(doc, sld.IModelDoc2)
ok, err, warn = model.Extension.SaveAs3(OUT + ".SLDPRT", c.swSaveAsCurrentVersion,
                                        c.swSaveAsOptions_Silent, None, None, 0, 0)
must(ok, f"save as {OUT}.SLDPRT ({err})")
print("working on", model.GetTitle())

(b0,) = bodies(model)
bx = box_of(b0)
ztop = bx[5] / 1000
print(f"box {bx[3]-bx[0]:.1f} x {bx[4]-bx[1]:.1f} x {bx[5]-bx[2]:.1f} mm, top face at z={bx[5]:.2f}")
sm = model.SketchManager
sm.AddToDB = True                     # no snapping / auto-relations to existing edges

# --- the inlay: cut a groove, then fill it with its own body ---------------
trace = LineString([(-11, -3), (-3, -3), (3, 3), (11, 3)]).buffer(1.5, cap_style=2, join_style=2)
ring = list(trace.exterior.coords)[:-1]
sketch_on_face(model, 0.0, 0.008, ztop)
polygon(model, ring)
sk = close_and_select(model)
cut = model.FeatureManager.FeatureCut4(
    True, False, False, c.swEndCondBlind, 0, INLAY_D, 0,
    False, False, False, False, 0, 0, False, False, False, False,
    False, True, True, False, False, False, c.swStartSketchPlane, 0, False, False)
must(cut, "groove cut")
must(sk.Select2(False, 0), "reselect the groove sketch")
fill = model.FeatureManager.FeatureExtrusion3(
    True, False, True,                        # Dir=True: into the groove, not up
    c.swEndCondBlind, 0, INLAY_D, 0,
    False, False, False, False, 0, 0, False, False, False, False,
    False, True, True,                        # Merge=FALSE -> its own body
    c.swStartSketchPlane, 0, False)
must(fill, "inlay body")

# --- the pads: two raised, drafted, separate bodies ------------------------
sketch_on_face(model, 0.0155, 0.0, ztop)
for xc in (-0.0155, 0.0155):
    must(sm.CreateCenterRectangle(xc, 0, 0, xc + 0.0025, 0.005, 0), "pad rectangle")
close_and_select(model)
pads = model.FeatureManager.FeatureExtrusion3(
    True, False, False, c.swEndCondBlind, 0, PAD_H, 0,
    True, False, False, False, math.radians(PAD_DRAFT), 0,   # draft inward
    False, False, False, False,
    False, True, True,                        # Merge=FALSE
    c.swStartSketchPlane, 0, False)
must(pads, "pad bodies")
sm.AddToDB = False
model.ClearSelection2(True)

# --- name and colour every body by where it sits --------------------------
bs = bodies(model)
print(f"\n{len(bs)} bodies:")
for b in bs:
    x0, y0, z0, x1, y1, z1 = box_of(b)
    if z0 >= bx[5] - 1e-3:
        name = "graphite_pad"
    elif z0 >= bx[5] - INLAY_D * 1000 - 1e-3:
        name = "blue_inlay"
    else:
        name = "white_box"
    rgb, slot = COLOURS[name]
    b.Name = name if name != "graphite_pad" or x0 < 0 else name + "_2"
    b.MaterialPropertyValues2 = list(rgb) + [1.0, 1.0, 0.3, 0.3, 0.0, 0.0]
    print(f"  {b.Name:15s} {vol(b):9.2f} mm3   z {z0:6.2f}..{z1:6.2f}")
must(len(bs) == 4, "expected 4 bodies: box, inlay, 2 pads")
v = {b.Name: vol(b) for b in bs}
must(abs(v["white_box"] + v["blue_inlay"] - 8000.0) < 0.01,
     "box + inlay must add back to the original 8000 mm3 (no overlap, no gap)")
print(f"  box + inlay = {v['white_box'] + v['blue_inlay']:.3f} mm3 (the original cube) -- no overlap")

model.ShowNamedView2("*Isometric", c.swIsometricView)
model.ViewZoomtofit2()

# --- save + three exports ---------------------------------------------------
for ext, opt in ((".SLDPRT", c.swSaveAsOptions_Silent),
                 (".STEP", c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy),
                 ("_sw.3mf", c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy)):
    ok, err, warn = model.Extension.SaveAs3(OUT + ext, c.swSaveAsCurrentVersion, opt,
                                            None, None, 0, 0)
    must(ok and os.path.exists(OUT + ext), f"save {ext} (err {err}, warn {warn})")
    print(f"saved {os.path.basename(OUT + ext):34s} {os.path.getsize(OUT + ext) / 1024:7.1f} KB")

with zipfile.ZipFile(OUT + "_sw.3mf") as z:
    m = z.read("3D/3dmodel.model").decode("utf-8", "replace")
print(f"  SolidWorks 3MF: {m.count('<object ')} object(s), "
      f"{'has' if 'basematerials' in m or 'colorgroup' in m else 'NO'} colour data")

# --- the Bambu project 3MF, built from SolidWorks' own STEP ----------------
from build123d import import_step
from render3d import tessellate
import export3mf
step = import_step(OUT + ".STEP")
parts = []
for s in step.solids():
    bb = s.bounding_box()
    if bb.min.Z >= bx[5] - 1e-3:
        name = "graphite_pad"
    elif bb.min.Z >= bx[5] - INLAY_D * 1000 - 1e-3:
        name = "blue_inlay"
    else:
        name = "white_box"
    V, T, _ = tessellate(s, 0.02)
    parts.append((name, V, T, COLOURS[name][1]))
    print(f"  STEP solid -> {name:13s} filament {COLOURS[name][1]}  {s.volume:9.2f} mm3")
must(len(parts) == 4, "the STEP should hold 4 solids")
export3mf.write_3mf(parts, OUT + "_bambu.3mf", name="hello box multicolour")
print(f"saved {os.path.basename(OUT + '_bambu.3mf'):34s} "
      f"{os.path.getsize(OUT + '_bambu.3mf') / 1024:7.1f} KB")
