r"""Find and repair NON-MANIFOLD solids before they are written to STEP.

An edge used by three or more faces means the body touches ITSELF along that
edge.  OCC represents that happily: `BRepCheck_Analyzer` calls it valid, the
shell reports Closed, and the volume integrates to a sensible number.  Parasolid
-- SolidWorks, NX, Solid Edge -- cannot represent a non-manifold solid AT ALL,
so on import it SPLITS the body at every such edge.  Splitting a 687-face body
at four self-touching edges does not yield four nice pieces; it yields debris.

Measured on the Femur's `graphite_1` (17.68 cm3, 4 non-manifold edges): SolidWorks
produced 30 solid bodies and 11 surface bodies totalling 0.00 cm3 of the 17.68,
and raised NO ERROR, because splitting is a legitimate repair from its side.

Nothing in the pipeline tested for this.  Every other check passed it:
  BRepCheck_Analyzer      valid
  BOPAlgo self-intersect  clean
  shell closed            True
  mean wall thickness     2.1 mm
  declared tolerance      irrelevant -- this is topology, not tolerance

Bambu Studio shows the same file correctly, but that is not evidence the file is
sound: Bambu imports STEP through OpenCascade, the same kernel that wrote it, so
it reproduces OCC's interpretation by construction.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Builder
from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCP.ShapeFix import ShapeFix_Solid, ShapeFix_Shape
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp


def volume(shape):
    p = GProp_GProps(); BRepGProp.VolumeProperties_s(shape, p); return p.Mass()


def nonmanifold_edges(shape):
    """Edges used by 3+ faces.  Degenerate (seam-pole) edges are not counted --
    a sphere pole is legitimately used oddly and is not what breaks Parasolid."""
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    bad = []
    for k in range(1, m.Extent() + 1):
        e = TopoDS.Edge_s(m.FindKey(k))
        if BRep_Tool.Degenerated_s(e):
            continue
        if m.FindFromIndex(k).Extent() > 2:
            bad.append(e)
    return bad


def free_edges(shape):
    """Edges used by exactly 1 face -- a hole in the shell."""
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    return [TopoDS.Edge_s(m.FindKey(k)) for k in range(1, m.Extent() + 1)
            if not BRep_Tool.Degenerated_s(TopoDS.Edge_s(m.FindKey(k)))
            and m.FindFromIndex(k).Extent() == 1]


def _faces(shape):
    out = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        out.append(TopoDS.Face_s(ex.Current())); ex.Next()
    return out


def _compound(shapes):
    c = TopoDS_Compound(); b = TopoDS_Builder(); b.MakeCompound(c)
    for s in shapes:
        b.Add(c, s)
    return c


def make_manifold(shape, tol=1e-6, say=print):
    """Re-sew the faces with non-manifold mode OFF, so the body is divided into
    manifold shells at its self-contacts, and rebuild a solid per shell.

    Returns the repaired shape, or the ORIGINAL if the repair lost volume --
    a body that imports cleanly is worth nothing if it is the wrong shape.
    """
    bad = nonmanifold_edges(shape)
    if not bad:
        return shape, 0
    v0 = volume(shape)
    sew = BRepBuilderAPI_Sewing(tol)
    sew.SetNonManifoldMode(False)
    for f in _faces(shape):
        sew.Add(f)
    sew.Perform()
    sewed = sew.SewedShape()

    solids = []
    ex = TopExp_Explorer(sewed, TopAbs_SHELL)
    while ex.More():
        sh = TopoDS.Shell_s(ex.Current())
        try:
            mk = BRepBuilderAPI_MakeSolid(sh)
            sol = mk.Solid()
            fx = ShapeFix_Solid(sol); fx.Perform()
            sol = fx.Solid()
            if volume(sol) > 1e-9:
                solids.append(sol)
        except Exception:
            pass
        ex.Next()
    if not solids:
        say("      re-sew produced no solid; kept the original")
        return shape, len(bad)
    out = solids[0] if len(solids) == 1 else _compound(solids)
    v1 = volume(out)
    if abs(v1 - v0) > max(1e-3, 1e-5 * abs(v0)):
        say(f"      re-sew moved volume {v0:.3f} -> {v1:.3f} mm3; kept the original")
        return shape, len(bad)
    left = len(nonmanifold_edges(out))
    say(f"      {len(bad)} non-manifold edge(s) -> {len(solids)} manifold solid(s), "
        f"{left} left, volume {v0:.3f} -> {v1:.3f} mm3")
    return out, len(bad)


def weld_nonmanifold(shape, r=0.06, say=print):
    """Fuse a small rod along every non-manifold edge so the pinch becomes a
    real join.

    These edges are where a 2D profile pinched to a POINT and was then
    extruded: two lobes of one polygon meet along a single vertical line, four
    planar faces to the line.  `buffer(0)` is what leaves them -- it makes a
    self-touching ring OGC-valid by splitting it into two lobes that still
    touch, and extruding that gives two prisms sharing one edge.

    Welding is the right answer mechanically as well as topologically: a
    knife-edge contact carries no load and is exactly the fragile sharp feature
    the brief rules out.  A 0.06 mm rod costs a few hundredths of a percent of
    volume and turns it into material that can actually be printed.
    """
    bad = nonmanifold_edges(shape)
    if not bad:
        return shape, 0, 0.0
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    v0 = volume(shape)
    out = shape
    welded = 0
    for e in bad:
        c = BRepAdaptor_Curve(e)
        p0, p1 = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
        d = gp_Pnt(p1.X() - p0.X(), p1.Y() - p0.Y(), p1.Z() - p0.Z())
        L = (d.X() ** 2 + d.Y() ** 2 + d.Z() ** 2) ** 0.5
        if L < 1e-6:
            continue
        # over-run both ends so the rod cannot itself end flush with a face and
        # create a fresh tangency where the old pinch was
        ax = gp_Ax2(gp_Pnt(p0.X() - d.X() / L * r, p0.Y() - d.Y() / L * r,
                           p0.Z() - d.Z() / L * r),
                    gp_Dir(d.X() / L, d.Y() / L, d.Z() / L))
        rod = BRepPrimAPI_MakeCylinder(ax, r, L + 2 * r).Shape()
        try:
            f = BRepAlgoAPI_Fuse(out, rod); f.SetFuzzyValue(1e-7); f.Build()
            if f.IsDone():
                out = f.Shape(); welded += 1
        except Exception as ex:
            say(f"      weld failed on one edge ({type(ex).__name__})")
    v1 = volume(out)
    left = len(nonmanifold_edges(out))
    say(f"      welded {welded}/{len(bad)} pinch edge(s) at r={r} mm; "
        f"{left} non-manifold left; volume {v0:.3f} -> {v1:.3f} mm3 "
        f"(+{100*(v1-v0)/v0:.4f}%)")
    return out, left, v1 - v0


def report(shape, name=""):
    return dict(name=name, vol=volume(shape),
                nonmanifold=len(nonmanifold_edges(shape)),
                free=len(free_edges(shape)))


# ---------------------------------------------------------------- the gate

NEEDLE_MM3 = 1.0        # a solid smaller than this is debris, not a feature
NEEDLE_MIN_MM = 0.5     # ... or thinner than this in its smallest dimension


def _solids(shape):
    out = []
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        out.append(TopoDS.Solid_s(ex.Current())); ex.Next()
    return out


def _shells(shape):
    out = []
    ex = TopExp_Explorer(shape, TopAbs_SHELL)
    while ex.More():
        out.append(TopoDS.Shell_s(ex.Current())); ex.Next()
    return out


def _bbox(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    b = Bnd_Box(); BRepBndLib.Add_s(shape, b)
    x0, y0, z0, x1, y1, z1 = b.Get()
    return (x1 - x0, y1 - y0, z1 - z0)


def gate(shape, name="", one_solid=True):
    """The four topology checks `verify.py` never had.  Returns a list of
    problems; empty means the body is sound.

    Every one of these would have caught a defect ON THE DAY IT WAS INTRODUCED,
    and each instead reached Fernando or a printer:

      non-manifold edges  the Phase 1 bug -- Parasolid shatters the body into
                          debris and raises NO ERROR, so it looks like a
                          modelling mistake rather than an export one
      free edges          an open shell; not a solid at all
      one solid per body  his "NO FLOATING BODIES" -- the Coupler fused to FIVE
                          disconnected solids, four of them ~21x5x7 mm chunks
                          hanging in space, 1.15 cm3 in total
      one shell per solid a second shell is a SEALED INTERNAL CAVITY.  The Tibia
                          had three, 1.41 cm3, two of them 36x7 mm bubbles with
                          no way out -- unprintable and invisible from outside

    `one_solid=False` for a per-filament body: the accent legitimately comes out
    as several separate traces.  The FUSED part must always be one solid.
    """
    probs = []
    nm = nonmanifold_edges(shape)
    if nm:
        probs.append(f"{len(nm)} non-manifold edge(s) -- Parasolid will split this "
                     f"body into debris on import, silently")
    fe = free_edges(shape)
    if fe:
        probs.append(f"{len(fe)} free edge(s) -- the shell is open, this is not a solid")

    solids = _solids(shape)
    if not solids:
        probs.append("no solids at all")
        return probs

    vols = sorted(((volume(s), s) for s in solids), key=lambda t: -t[0])
    if one_solid and len(solids) > 1:
        extra = vols[1:]
        detail = ", ".join(f"{v:.1f} mm3 at {tuple(round(d,1) for d in _bbox(s))}"
                           for v, s in extra[:4])
        probs.append(f"{len(solids)} disconnected solids -- {len(extra)} FLOATING "
                     f"PIECE(S) totalling {sum(v for v,_ in extra):.1f} mm3: {detail}"
                     + (" ..." if len(extra) > 4 else ""))

    needles = [(v, s) for v, s in vols
               if v < NEEDLE_MM3 or min(_bbox(s)) < NEEDLE_MIN_MM]
    if needles:
        probs.append(f"{len(needles)} needle/debris solid(s) (<{NEEDLE_MM3} mm3 or "
                     f"<{NEEDLE_MIN_MM} mm thick) -- fragile sharp features")

    for i, (v, s) in enumerate(vols):
        sh = _shells(s)
        if len(sh) > 1:
            probs.append(f"solid {i} ({v:.1f} mm3) has {len(sh)} shells -- "
                         f"{len(sh)-1} SEALED INTERNAL CAVITY/IES, unprintable")
    return probs


def gate_report(shape, name="", one_solid=True, say=print):
    """Run `gate` and print it.  True if the body is sound."""
    probs = gate(shape, name, one_solid)
    if not probs:
        say(f"  topology {name:<22} OK  "
            f"({len(_solids(shape))} solid(s), {volume(shape)/1000:.2f} cm3)")
        return True
    say(f"  topology {name:<22} {len(probs)} PROBLEM(S)")
    for p in probs:
        say(f"      - {p}")
    return False
