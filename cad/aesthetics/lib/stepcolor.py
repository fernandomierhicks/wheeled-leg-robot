r"""Write ONE STEP carrying all three filament bodies, each with its colour.

`out/styled/<Part>_glacier_styled.step` is the FUSED body -- a single solid, so
SolidWorks can only ever show it in one colour.  That is what it is for: a
geometry reference.  This writes the other thing you actually want to look at,
a single file with white, graphite and accent as three coloured bodies, via
XCAF, which is the only part of STEP that carries colour at all.

Plain `export_step` cannot do this: colour lives in the XCAF document, not in
the shape, so the document has to be built and handed to STEPCAFControl_Writer.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorGen
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.Quantity import Quantity_Color, Quantity_TOC_sRGB
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.STEPControl import STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static


def write(bodies, path, schema="AP214IS"):
    """bodies = [(name, shape, (r, g, b) 0-255)] -> one coloured STEP."""
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    n = 0
    for name, shape, rgb in bodies:
        if shape is None or shape.volume < 1.0:
            continue
        col = Quantity_Color(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0,
                             Quantity_TOC_sRGB)
        # EVERY solid gets its own label.  Handing AddShape the whole compound
        # with makeAssembly=False leaves it undecomposed, and an importer that
        # takes one solid per product then shows ONE of them -- the Femur's
        # white body is 12 solids, so eleven twelfths of the part went missing
        # and it read as hollow.  That is trap 7 again, on the consumer side.
        solids = shape.solids()
        for i, sol in enumerate(solids, 1):
            lab = st.AddShape(sol.wrapped, False)
            TDataStd_Name.Set_s(lab, TCollection_ExtendedString(
                name if len(solids) == 1 else f"{name}_{i}"))
            ct.SetColor(lab, col, XCAFDoc_ColorGen)
            n += 1
    if not n:
        raise ValueError("no bodies with volume to write")
    Interface_Static.SetCVal_s("write.step.schema", schema)
    w = STEPCAFControl_Writer()
    w.SetColorMode(True)
    w.SetNameMode(True)
    w.Transfer(doc, STEPControl_AsIs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if w.Write(path) != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: {path}")
    return path
