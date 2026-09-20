"""Pull named simple-shape components out of an assembly STEP via XCAF."""
import os, sys, time
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence
from OCP.TDataStd import TDataStd_Name
from OCP.TCollection import TCollection_ExtendedString
from OCP.IFSelect import IFSelect_RetDone

SRC  = r"c:/Dropbox/Personal Projects/Robotics/wheeled-leg-robot/cad/v4 Larger Ball bearings/STEP exports/Extended.STEP"
WANT = {"coupler", "side panel"}
DEST = r"c:/Dropbox/Personal Projects/Robotics/wheeled-leg-robot/cad/aesthetics/input/parts"

t0 = time.time()
doc = TDocStd_Document(TCollection_ExtendedString("d"))
rd  = STEPCAFControl_Reader()
rd.SetNameMode(True)
print("reading...", flush=True)
if rd.ReadFile(SRC) != IFSelect_RetDone:
    sys.exit("read failed")
print(f"  parsed in {time.time()-t0:.0f}s", flush=True)
rd.Transfer(doc)
print(f"  transferred in {time.time()-t0:.0f}s", flush=True)

st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
seq = TDF_LabelSequence()
st.GetShapes(seq)
print(f"  {seq.Length()} labels", flush=True)

from build123d import Compound, export_step
found = {}
for i in range(1, seq.Length() + 1):
    lab = seq.Value(i)
    if not st.IsSimpleShape_s(lab):
        continue
    nm = TDataStd_Name()
    if not lab.FindAttribute(TDataStd_Name.GetID_s(), nm):
        continue
    name = nm.Get().ToExtString()
    if name.lower().strip() in WANT and name.lower().strip() not in found:
        shp = Compound(st.GetShape_s(lab))
        bb = shp.bounding_box()
        vol = sum(s.volume for s in shp.solids())
        out = os.path.join(DEST, name.replace(" ", "_") + ".step")
        export_step(shp, out)
        found[name.lower().strip()] = True
        print(f"{name:<14} bbox {bb.size.X:7.1f} x {bb.size.Y:7.1f} x {bb.size.Z:7.1f} mm  "
              f"vol {vol/1000:7.1f} cm3  solids {len(shp.solids())}  -> {os.path.basename(out)}", flush=True)
print(f"done in {time.time()-t0:.0f}s; missing: {WANT - set(found)}", flush=True)
