"""Step 2 of the SolidWorks automation ladder: build a box and save it.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/02_box.py

New part from the default template -> sketch a 40 x 20 mm centre rectangle on
the Front Plane -> extrude 10 mm -> check the volume -> save to
cad/v4 Larger Ball bearings/ai_components/hello_box.SLDPRT and leave it open.

Touches no existing document.  The SolidWorks API works in METRES.
"""
import os
import sys
import win32com.client
from win32com.client import gencache, constants as c

sys.stdout.reconfigure(encoding="utf-8")

# sldworks.tlb (interfaces) and swconst.tlb (enums), SolidWorks 2023 = 31.0
sldworks = gencache.EnsureModule("{83A33D31-27C5-11CE-BFD4-00400513BB57}", 0, 31, 0)
gencache.EnsureModule("{4687F359-55D0-4CD3-B6CF-2EB42C11F989}", 0, 31, 0)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "v4 Larger Ball bearings", "ai_components")
OUT = os.path.normpath(os.path.join(OUT_DIR, "hello_box.SLDPRT"))

W, H, D = 0.040, 0.020, 0.010          # m


def must(ok, what):
    if not ok:
        raise SystemExit(f"FAILED: {what}")


sw = sldworks.ISldWorks(win32com.client.GetActiveObject("SldWorks.Application")._oleobj_)
print("connected to SolidWorks", sw.RevisionNumber())

template = sw.GetUserPreferenceStringValue(c.swDefaultTemplatePart)
print("part template:", template)
doc = sw.NewDocument(template, 0, 0, 0)
must(doc, "new part document")
model = sldworks.IModelDoc2(doc._oleobj_)

must(model.Extension.SelectByID2("Front Plane", "PLANE", 0, 0, 0, False, 0, None, 0),
     "select Front Plane")
model.SketchManager.InsertSketch(True)
must(model.SketchManager.CreateCenterRectangle(0, 0, 0, W / 2, H / 2, 0),
     "sketch centre rectangle")
model.SketchManager.InsertSketch(True)          # close the sketch

must(model.Extension.SelectByID2("Sketch1", "SKETCH", 0, 0, 0, False, 0, None, 0),
     "select Sketch1")
feat = model.FeatureManager.FeatureExtrusion3(
    True, False, False,                         # one direction, no flip, no reverse
    c.swEndCondBlind, 0, D, 0,                  # blind, depth D
    False, False, False, False, 0, 0,           # no draft
    False, False, False, False,                 # no offsets / surface translate
    True, True, True,                           # merge, feature scope, auto-select
    c.swStartSketchPlane, 0, False)             # start at the sketch plane
must(feat, "extrude")

vol = model.Extension.CreateMassProperty().Volume * 1e9     # m3 -> mm3
want = W * H * D * 1e9
print(f"volume {vol:.1f} mm3 (expected {want:.1f})")
must(abs(vol - want) < 0.01, "volume check")

model.ShowNamedView2("*Isometric", c.swIsometricView)
model.ViewZoomtofit2()

os.makedirs(OUT_DIR, exist_ok=True)
ok, errors, warnings = model.Extension.SaveAs3(
    OUT, c.swSaveAsCurrentVersion, c.swSaveAsOptions_Silent, None, None, 0, 0)
must(ok and os.path.exists(OUT), f"save (errors={errors}, warnings={warnings})")
print("saved:", OUT)
