r"""Write the SAME bodies as a STEP in three different structural FORMS.

`lib/stepcolor.py` writes one ROOT PRODUCT PER SOLID and no assembly structure
at all -- RobotMount ships as 42 unrelated roots in one file.  That was chosen
to dodge an importer that showed only one solid of an undecomposed compound,
and it is the leading suspect for SolidWorks showing an incomplete part.

Rather than argue about what SolidWorks does, write all three candidate forms of
one part and let `tools/swcheck.ps1` open each and report what SolidWorks built:

  roots      one root product per solid          (what ships today)
  multibody  ONE root product, one shape holding every solid as a compound,
             colour attached per sub-solid       -> should read as one part,
                                                    N coloured bodies
  assembly   ONE root assembly, one sub-assembly per filament colour, each
             solid a component under its colour

The geometry is byte-identical across the three; only the XCAF structure差.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import paths
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
from OCP.TopLoc import TopLoc_Location


def _qc(rgb):
    return Quantity_Color(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0,
                          Quantity_TOC_sRGB)


def _compound(shapes):
    c = TopoDS_Compound(); b = TopoDS_Builder(); b.MakeCompound(c)
    for s in shapes:
        b.Add(c, s)
    return c


def _name(lab, text):
    TDataStd_Name.Set_s(lab, TCollection_ExtendedString(text))


def _solids(bodies):
    """[(name, shape, rgb)] -> [(label_name, TopoDS_Solid, rgb)], skipping empties.

    Works on a build123d shape OR a raw TopoDS_Shape, and enumerates solids with
    TopExp rather than `.solids()`: a healed shape comes back from OCC as a bare
    TopoDS and the build123d wrapper around it has no `.volume`.
    """
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    out = []
    for name, shape, rgb in bodies:
        if shape is None:
            continue
        topo = getattr(shape, "wrapped", shape)
        sols = []
        ex = TopExp_Explorer(topo, TopAbs_SOLID)
        while ex.More():
            p = GProp_GProps(); BRepGProp.VolumeProperties_s(ex.Current(), p)
            if p.Mass() >= 1.0:
                sols.append(ex.Current())
            ex.Next()
        for i, s in enumerate(sols, 1):
            out.append((name if len(sols) == 1 else f"{name}_{i}", s, rgb, name))
    return out


def _save(doc, path, schema="AP214IS", precision=None):
    """`precision` sets the UNCERTAINTY_MEASURE_WITH_UNIT the file DECLARES.

    This is not cosmetic.  An importer sews faces into solids at the tolerance
    the file claims, and OCC's default (`write.precision.mode` 0, "average")
    averages hundreds of faces sitting at the 1e-7 floor against a handful that
    are genuinely 500x looser -- so a Femur body whose worst face tolerance is
    5.45e-5 mm ships declaring 2e-7 mm.  SolidWorks believes it, finds every
    real gap 270x too wide to close, and SILENTLY shatters one body into 30
    solids plus 11 loose surfaces.  No error is raised, because by the file's
    own declaration nothing is wrong.

        None    OCC default (average)            -- what shipped, and is broken
        "max"   mode 1: the shape's WORST tolerance
        <float> mode 2: that value, in mm
    """
    if precision == "max":
        Interface_Static.SetIVal_s("write.precision.mode", 1)
    elif precision is not None:
        Interface_Static.SetIVal_s("write.precision.mode", 2)
        Interface_Static.SetRVal_s("write.precision.val", float(precision))
    else:
        Interface_Static.SetIVal_s("write.precision.mode", 0)
    Interface_Static.SetCVal_s("write.step.schema", schema)
    w = STEPCAFControl_Writer()
    w.SetColorMode(True); w.SetNameMode(True)
    w.Transfer(doc, STEPControl_AsIs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if w.Write(path) != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: {path}")
    return path


def write(bodies, path, form="multibody", part="Part", schema="AP214IS",
          precision=None):
    """bodies = [(name, shape, (r,g,b) 0-255)] -> one STEP in the chosen form."""
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    sols = _solids(bodies)
    if not sols:
        raise ValueError("no bodies with volume to write")

    if form == "roots":
        for nm, sol, rgb, _ in sols:
            lab = st.AddShape(sol, False)
            _name(lab, nm)
            ct.SetColor(lab, _qc(rgb), XCAFDoc_ColorGen)

    elif form == "multibody":
        # ONE product.  The colour goes on each SUB-SOLID of that one shape, so
        # the writer emits a STYLED_ITEM per MANIFOLD_SOLID_BREP inside a single
        # shape representation -- the ordinary way a coloured multibody part is
        # carried in STEP.
        top = st.AddShape(_compound([s for _, s, _, _ in sols]), False)
        _name(top, part)
        for nm, sol, rgb, _ in sols:
            sub = st.AddSubShape(top, sol)
            if not sub.IsNull():
                _name(sub, nm)
                ct.SetColor(sub, _qc(rgb), XCAFDoc_ColorSurf)
                ct.SetColor(sub, _qc(rgb), XCAFDoc_ColorGen)
            else:
                ct.SetColor(sol, _qc(rgb), XCAFDoc_ColorSurf)

    elif form == "assembly":
        # A real assembly: top product -> one sub-assembly per filament colour
        # -> one component per solid.  Gives an importer an unambiguous single
        # root and a named tree it can hide and show by colour.
        groups = {}
        for nm, sol, rgb, grp in sols:
            groups.setdefault(grp, (rgb, []))[1].append((nm, sol))
        top = st.NewShape()
        _name(top, part)
        for grp, (rgb, items) in groups.items():
            sub = st.AddShape(_compound([s for _, s in items]), True)
            _name(sub, grp)
            ct.SetColor(sub, _qc(rgb), XCAFDoc_ColorGen)
            ct.SetColor(sub, _qc(rgb), XCAFDoc_ColorSurf)
            st.AddComponent(top, sub, TopLoc_Location())
        st.UpdateAssemblies()

    else:
        raise ValueError(f"unknown form {form!r}")

    return _save(doc, path, schema, precision)


if __name__ == "__main__":
    # Rebuild the three forms for an ALREADY BUILT part, straight from the
    # per-filament STEPs in out/print/<Part>/, so no 2-4 minute rebuild is
    # needed just to test file structure.
    from build123d import import_step
    import spec as _spec, json
    part = sys.argv[1] if len(sys.argv) > 1 else "Femur"
    tag = "glacier"
    sf = os.path.join(paths.SPECS, part.lower().replace(" ", "_") + ".json")
    palette = "arctic_lt"
    if os.path.exists(sf):
        palette = json.load(open(sf)).get("palette", palette)
    pal = _spec.PALETTES[palette]
    col = {"white": pal["white"], "graphite": pal["dark"], "accent": pal["accent"]}
    pdir = paths.print_dir(part, tag)
    bodies = []
    for n in ("white", "graphite", "accent"):
        f = os.path.join(pdir, f"{part}_{tag}_{n}.step")
        if os.path.exists(f):
            bodies.append((n, import_step(f), col[n]))
    out = os.path.join(paths.ROOT, "out", "stepforms", part)
    only = sys.argv[2] if len(sys.argv) > 2 else None
    if only:
        bodies = [b for b in bodies if b[0] == only]
        part_lbl = f"{part}_{only}"
    else:
        part_lbl = part
    import re
    for tag, prec in (("tolmax", "max"), ("tol1um", 1e-3), ("tol10um", 1e-2)):
        nm = f"{part_lbl}_{tag}.step"
        p = write(bodies, os.path.join(out, nm), form="multibody",
                  part=part_lbl, precision=prec)
        u = re.search(r"UNCERTAINTY_MEASURE_WITH_UNIT\(LENGTH_MEASURE\(([^)]*)\)",
                      open(p, encoding="utf-8", errors="ignore").read())
        print(f"  {tag:<9} declares {u.group(1) if u else '?':<12} "
              f"{os.path.getsize(p)/1e6:6.2f} MB  {nm}")
