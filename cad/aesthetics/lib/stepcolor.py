r"""Write ONE STEP carrying all three filament bodies, each with its colour.

`out/styled/<Part>_glacier_styled.step` is the FUSED body -- a single solid, so
SolidWorks can only ever show it in one colour.  That is what it is for: a
geometry reference.  This writes the other thing you actually want to look at,
a single file with white, graphite and accent as three coloured bodies, via
XCAF, which is the only part of STEP that carries colour at all.

Plain `export_step` cannot do this: colour lives in the XCAF document, not in
the shape, so the document has to be built and handed to STEPCAFControl_Writer.

THREE THINGS THIS FILE GETS RIGHT, EACH PAID FOR (see DECISIONS.md 23)
---------------------------------------------------------------------
1. EVERY BODY IS WELDED MANIFOLD FIRST.  The Femur's `graphite_1` had four
   non-manifold edges -- vertical lines where FOUR planar faces met, because a
   2D profile pinched to a point and was then extruded.  OCC calls such a solid
   valid; Parasolid (SolidWorks, NX, Solid Edge) CANNOT REPRESENT ONE AT ALL and
   splits the body at every self-contact on import.  Measured: one 17.68 cm3
   body became 30 solid bodies and 11 surface bodies totalling 0.00 cm3, WITH NO
   ERROR RAISED, because splitting is a legitimate repair from its side.  The
   part looked hollow and the cause was invisible to every check that existed.

2. ONE ROOT PRODUCT, NOT ONE PER SOLID.  This used to give every solid its own
   root -- RobotMount shipped as 42 unrelated products with no assembly
   structure.  That turned out NOT to be the import bug (both forms failed
   identically), but N unrelated roots is not what "a part" means in STEP and
   there is no reason to ship it.

3. THE DECLARED TOLERANCE IS HONEST.  OCC's default write.precision.mode 0
   averages hundreds of faces sitting at the 1e-7 floor against a few that are
   genuinely 500x looser, so a body whose worst face tolerance was 4e-4 mm
   shipped declaring 2e-6.  Mode 1 declares the shape's actual worst case.
   This did NOT fix the import on its own -- do not expect it to -- but a file
   should not lie about its own precision.

WHAT DOES NOT WORK, so it is not tried again: ShapeFix_Shape and
ShapeUpgrade_UnifySameDomain are both no-ops here (1404 -> 1404 faces on white,
723 -> 723 on graphite), and re-sewing with BRepBuilderAPI_Sewing at
non-manifold mode OFF preserves volume exactly but leaves all four bad edges.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifold
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorGen, XCAFDoc_ColorSurf
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.Quantity import Quantity_Color, Quantity_TOC_sRGB
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.STEPControl import STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.TopoDS import TopoDS_Compound, TopoDS_Builder
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID

MIN_BODY_MM3 = 1.0      # below this a "body" is boolean debris: a tenth of a
                        # nozzle width, unprintable, and pure noise in the tree


def _solids(shape):
    out = []
    ex = TopExp_Explorer(getattr(shape, "wrapped", shape), TopAbs_SOLID)
    while ex.More():
        out.append(ex.Current()); ex.Next()
    return out


def write(bodies, path, schema="AP214IS", part=None, weld=True, say=print):
    """bodies = [(name, shape, (r, g, b) 0-255)] -> one coloured STEP."""
    part = part or os.path.splitext(os.path.basename(path))[0]
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())

    keep, dropped, pinched, dv = [], 0, 0, 0.0
    for name, shape, rgb in bodies:
        if shape is None:
            continue
        sols = _solids(shape)
        for s in sols:
            v = manifold.volume(s)
            if v < MIN_BODY_MM3:
                dropped += 1
                continue
            bad = manifold.nonmanifold_edges(s)
            if bad and weld:
                s, left, d = manifold.weld_nonmanifold(s, say=say)
                pinched += len(bad); dv += d
                if left:
                    raise RuntimeError(
                        f"{part}: {name} still has {left} non-manifold edge(s) "
                        f"after welding -- SolidWorks would shatter this body.")
            elif bad:
                say(f"  WARNING {name}: {len(bad)} non-manifold edge(s), not welded")
            keep.append((name, s, rgb))
    if not keep:
        raise ValueError("no bodies with volume to write")
    if pinched:
        say(f"  welded {pinched} pinch edge(s); volume +{dv:.3f} mm3")
    if dropped:
        say(f"  dropped {dropped} body(s) under {MIN_BODY_MM3} mm3 (boolean debris)")

    # ONE product holding every solid; colour attached per SUB-SOLID, so the
    # writer emits a STYLED_ITEM per MANIFOLD_SOLID_BREP inside a single shape
    # representation -- the ordinary way a coloured multibody part travels.
    c = TopoDS_Compound(); b = TopoDS_Builder(); b.MakeCompound(c)
    for _, s, _ in keep:
        b.Add(c, s)
    top = st.AddShape(c, False)
    TDataStd_Name.Set_s(top, TCollection_ExtendedString(part))
    per = {}
    for name, s, rgb in keep:
        per[name] = per.get(name, 0) + 1
        col = Quantity_Color(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0,
                             Quantity_TOC_sRGB)
        sub = st.AddSubShape(top, s)
        tgt = sub if not sub.IsNull() else None
        if tgt is not None:
            TDataStd_Name.Set_s(tgt, TCollection_ExtendedString(f"{name}_{per[name]}"))
            ct.SetColor(tgt, col, XCAFDoc_ColorSurf)
            ct.SetColor(tgt, col, XCAFDoc_ColorGen)
        else:
            ct.SetColor(s, col, XCAFDoc_ColorSurf)

    Interface_Static.SetIVal_s("write.precision.mode", 1)    # declare the WORST
    Interface_Static.SetCVal_s("write.step.schema", schema)
    w = STEPCAFControl_Writer()
    w.SetColorMode(True)
    w.SetNameMode(True)
    w.Transfer(doc, STEPControl_AsIs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if w.Write(path) != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: {path}")
    return path
