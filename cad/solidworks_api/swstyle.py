"""Native-SolidWorks styling primitives: shapely outlines -> sketches -> features.

Every styling feature in cad/aesthetics is "a plan outline, extruded between two
heights" (sometimes drafted).  This turns exactly that into SolidWorks features
on the part's Front Plane (part XY, normal +Z), so the result is a real feature
tree he can edit.

    sk = sketch(model, poly)                       # Front Plane sketch of a shapely (Multi)Polygon
    boss(model, sk, z0, z1, merge=True)            # extrude between part-Z heights z0 < z1
    raised(model, sk, base, h, up, draft)          # drafted pad from `base` going up(+Z) or down
    cut(model, sk, z0, z1)
    tool(model, sk, z0, z1)                        # a NEW body (Merge=False)
    split_off(model, part_body, [tools...], name)  # body ∩ tools as new bodies, body − tools

All heights are part-local millimetres.
"""
import math
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
import swlib
from swlib import c, wrap, sld
import pythoncom
import win32com.client

MM = 1e-3
MIN_SEG = 0.01        # mm; drop sketch segments shorter than this
SIMPLIFY = 0.01       # mm; outline simplification tolerance


class FeatureFailed(RuntimeError):
    pass


def polys(g, min_area=0.5):
    """Flatten any shapely geometry to a list of valid, non-trivial Polygons."""
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        out = [g]
    elif isinstance(g, (MultiPolygon, GeometryCollection)):
        out = [p for q in g.geoms for p in polys(q, 0)]
    else:
        return []
    return [p for p in out if isinstance(p, Polygon) and p.area >= min_area]


def _clean(poly, grow=-0.005):
    """Simplify and separate, so no two loops touch.

    Loops that touch at a point (what shapely calls valid) give SolidWorks a
    sketch contour it rejects or silently mis-reads, so a REMOVAL is shrunk by
    5 um -- invisible, and enough to pull touching loops apart.

    An ADDITION must not shrink: a flange band that only meets the part along
    a face (RobotMount: band from y=60.01, plate edge at y=60) is pulled 5 um
    clear of it and comes out as a separate floating body.  Additions are
    grown 0.05 mm instead, so they overlap and merge; touching loops of the
    same addition simply fuse, which is what the union means anyway.
    """
    p = poly.simplify(SIMPLIFY, preserve_topology=True)
    if grow > 0:
        p = p.buffer(grow, join_style=2)
        return polys(p, 0.25)
    return polys(p.buffer(grow, join_style=2), 0.25)


def last_feature(model):
    f, last = wrap(model.FirstFeature(), sld.IFeature), None
    while f is not None:
        last, f = f, wrap(f.GetNextFeature(), sld.IFeature)
    return last


def sketch(model, g, name=None, plane="Front Plane", grow=-0.005):
    """Sketch every ring of `g` (a shapely geometry, mm) on a datum plane.
    Returns the sketch's IFeature (fetched from the tree, not ActiveSketch).

    Front Plane: sketch (x, y) = part (X, Y), normal +Z.
    Top Plane:   sketch (x, y) = part (X, -Z), normal +Y   (measured)."""
    if grow > 0:
        from shapely.ops import unary_union as _uu
        ps = _clean(_uu(polys(g)), grow)
    else:
        ps = [q for p in polys(g) for q in _clean(p, grow)]
    if not ps:
        return None
    model.ClearSelection2(True)
    if not model.Extension.SelectByID2(plane, "PLANE", 0, 0, 0, False, 0, None, 0):
        raise FeatureFailed(f"cannot select {plane}")
    sm = model.SketchManager
    sm.InsertSketch(True)
    sm.AddToDB = True
    sm.DisplayWhenAdded = False
    n = 0
    try:
        for p in ps:
            for ring in [p.exterior] + list(p.interiors):
                pts = [(x, y) for x, y in ring.coords[:-1]]
                keep = [pts[0]]
                for q in pts[1:]:
                    if math.dist(q, keep[-1]) >= MIN_SEG:
                        keep.append(q)
                if len(keep) >= 2 and math.dist(keep[0], keep[-1]) < MIN_SEG:
                    keep.pop()
                if len(keep) < 3:
                    continue
                for (x1, y1), (x2, y2) in zip(keep, keep[1:] + keep[:1]):
                    if sm.CreateLine(x1 * MM, y1 * MM, 0, x2 * MM, y2 * MM, 0) is None:
                        raise FeatureFailed("CreateLine returned None")
                    n += 1
    finally:
        sm.DisplayWhenAdded = True
        sm.AddToDB = False
        sm.InsertSketch(True)
    sk = last_feature(model)
    if sk is None or sk.GetTypeName2() != "ProfileFeature":
        raise FeatureFailed("sketch did not land in the tree")
    if name:
        sk.Name = name
    sk.__dict__["n_lines"] = n          # makepy wrappers refuse new attributes
    return sk


def _select(model, sk):
    model.ClearSelection2(True)
    if not sk.Select2(False, 0):
        raise FeatureFailed(f"cannot select {sk.Name}")


def _start(z):
    """Start condition for an extrusion beginning at part height z (mm)."""
    if abs(z) < 1e-9:
        return c.swStartSketchPlane, 0.0, False
    return c.swStartOffset, abs(z) * MM, z < 0


def _extrude(model, sk, z_start, depth, up, merge, draft=0.0):
    _select(model, sk)
    t0, off, flip = _start(z_start)
    f = model.FeatureManager.FeatureExtrusion3(
        True, False, not up, c.swEndCondBlind, 0, depth * MM, 0,
        draft > 0, False, False, False, math.radians(draft), 0,
        False, False, False, False,
        merge, True, True, t0, off, flip)
    model.ClearSelection2(True)
    return f


def boss(model, sk, z0, z1, merge=True, name=None):
    """Extrude `sk` to fill part-Z z0..z1 (mm), merged into the body it touches."""
    f = _extrude(model, sk, z0, z1 - z0, True, merge)
    if f is None:
        raise FeatureFailed(f"boss {sk.Name} {z0:.2f}..{z1:.2f}")
    f = wrap(f, sld.IFeature)
    if name:
        f.Name = name
    return f


def tool(model, sk, z0, z1, name=None):
    """NEW bodies filling z0..z1 inside `sk` (for Combine).  Returns the bodies:
    a sketch with several separate regions makes several bodies, so they are
    found by diffing the body list, never by name (body names change with every
    feature that touches them)."""
    before = {id_(b) for b in bodies(model)}
    boss(model, sk, z0, z1, merge=False, name=name)
    new = [b for b in bodies(model) if id_(b) not in before]
    if not new:
        raise FeatureFailed(f"tool {sk.Name} made no body")
    return new


def raised(model, sk, base, h, up, draft, merge=True, name=None):
    """A drafted pad: full size at `base`, `h` tall towards +Z (up) or -Z,
    tapering inward by `draft` deg.  Falls back to no draft if SolidWorks
    refuses the taper (the OCC recipe does the same for narrow pads)."""
    f = _extrude(model, sk, base, h, up, merge, draft)
    if f is None and draft > 0:
        f = _extrude(model, sk, base, h, up, merge, 0.0)
    if f is None:
        raise FeatureFailed(f"raised {sk.Name}")
    f = wrap(f, sld.IFeature)
    if name:
        f.Name = name
    return f


SCOPE_MARK = 8      # measured: bodies for a feature's scope must be selected with mark 8


def _scope(model, scope):
    """Append `scope` bodies to the selection with the feature-scope mark."""
    sm = wrap(model.SelectionManager, sld.ISelectionMgr)
    for b in scope:
        d = wrap(sm.CreateSelectData(), sld.ISelectData)
        d.Mark = SCOPE_MARK
        if not b.Select2(True, d):
            raise FeatureFailed(f"cannot scope body {b.Name}")


def cut(model, sk, z0=None, z1=None, name=None, scope=None, outside=False):
    """Remove everything inside `sk` between part-Z z0 and z1 (mm).

    A CUT runs the opposite way to a boss by default: Dir=False from an offset
    start plane cut towards -Z (measured: "z 8..13" removed z 3..8).  Dir=True
    makes it run +Z like everything else here.

    z0 = z1 = None cuts THROUGH ALL in both directions (for side profiles on
    the Top Plane).  `scope` limits the cut to those bodies (feature scope,
    mark 8).  `outside=True` flips the side: keep what is inside the sketch.
    """
    _select(model, sk)
    if scope:
        _scope(model, scope)
    auto = not scope
    if z0 is None:
        f = model.FeatureManager.FeatureCut4(
            False, outside, False, c.swEndCondThroughAll, c.swEndCondThroughAll, 0, 0,
            False, False, False, False, 0, 0, False, False, False, False,
            False, True, auto, False, False, False, c.swStartSketchPlane, 0, False, False)
    else:
        t0, off, flip = _start(z0)
        f = model.FeatureManager.FeatureCut4(
            True, outside, True, c.swEndCondBlind, 0, (z1 - z0) * MM, 0,
            False, False, False, False, 0, 0, False, False, False, False,
            False, True, auto, False, False, False, t0, off, flip, False)
    model.ClearSelection2(True)
    if f is None:
        raise FeatureFailed(f"cut {sk.Name} {z0}..{z1}")
    f = wrap(f, sld.IFeature)
    if name:
        f.Name = name
    return f


# --- bodies -----------------------------------------------------------------
def bodies(model):
    part = wrap(model, sld.IPartDoc)
    return [wrap(b, sld.IBody2) for b in (part.GetBodies2(c.swSolidBody, False) or [])]


def volume(b):
    return b.GetMassProperties(1.0)[3] * 1e9


def total_volume(model):
    return sum(volume(b) for b in bodies(model))


def names(model):
    return {b.Name for b in bodies(model)}


def new_since(model, before):
    return [b for b in bodies(model) if b.Name not in before]


def id_(b):
    """A body's identity that survives renames: its name is NOT that."""
    return b.Name


def _select_bodies(model, bs, mark=0):
    """Select bodies by OBJECT with a selection mark.  SelectByID2 by name breaks
    on the names SolidWorks makes for split results, e.g. 'Combine1[2]'."""
    sm = wrap(model.SelectionManager, sld.ISelectionMgr)
    for i, b in enumerate(bs):
        data = wrap(sm.CreateSelectData(), sld.ISelectData)
        data.Mark = mark
        if not b.Select2(i > 0, data):
            raise FeatureFailed(f"cannot select body {b.Name}")


def copy_body(model, b, name=None):
    """Move/Copy Body with copy on and zero motion.  Returns the new body."""
    before = {x.Name for x in bodies(model)}
    model.ClearSelection2(True)
    _select_bodies(model, [b], mark=1)
    f = model.FeatureManager.InsertMoveCopyBody2(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, True, 1)
    model.ClearSelection2(True)
    if f is None:
        raise FeatureFailed(f"copy body {b.Name}")
    new = [x for x in bodies(model) if x.Name not in before]
    if len(new) != 1:
        raise FeatureFailed(f"copy of {b.Name} made {len(new)} bodies")
    if name:
        last_feature(model).Name = name
    return new[0]


def delete_bodies(model, bs):
    """Delete/Keep Body feature removing `bs` (cleanup after a failed combine)."""
    if not bs:
        return
    model.ClearSelection2(True)
    _select_bodies(model, bs, mark=0)
    model.FeatureManager.InsertDeleteBody2(False)
    model.ClearSelection2(True)


def _darray(objs):
    return win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_DISPATCH,
                                   [o._oleobj_ for o in objs])


def combine(model, op, main, tools_):
    """Combine: op = 'cut' (main - tools) or 'common' (main ∩ ONE tool).
    Returns (feature, bodies that are new after it)."""
    before = {x.Name for x in bodies(model)}
    model.ClearSelection2(True)
    if op == "cut":
        f = model.FeatureManager.InsertCombineFeature(c.SWBODYCUT, main._oleobj_,
                                                      _darray(tools_))
    else:
        f = model.FeatureManager.InsertCombineFeature(c.SWBODYINTERSECT, None,
                                                      _darray([main] + list(tools_)))
    model.ClearSelection2(True)
    if f is None:
        raise FeatureFailed(f"combine {op} {main.Name}")
    return wrap(f, sld.IFeature), [x for x in bodies(model) if x.Name not in before]


def colour(b, rgb, name=None):
    """Body colour.  MUST be a typed VT_R8 array: a plain Python list goes over
    as an array of VARIANTs, SolidWorks reads it as raw doubles, and the colour
    lands scrambled (red channel in green, blue in transparency -- the femur
    came out 95 % transparent green).  SolidWorks turns this into a
    color.p2m appearance on the body, which overrides the part's own."""
    if name:
        b.Name = name
    b.MaterialPropertyValues2 = win32com.client.VARIANT(
        pythoncom.VT_ARRAY | pythoncom.VT_R8, list(rgb) + [1.0, 1.0, 0.3, 0.3, 0.0, 0.0])


def by_name(model, name):
    for b in bodies(model):
        if b.Name == name:
            return b
    return None


# --- colour groups ------------------------------------------------------------
GROUPS = {"white": (0.93, 0.94, 0.95), "graphite": (0.25, 0.27, 0.30), "blue": (0.13, 0.45, 0.95)}


def group_of(b):
    """white / graphite / blue, from the body's COLOUR.  Not from its name:
    suppressing and unsuppressing features rebuilds the bodies and every name
    set through the API reverts to a feature-derived one ('GL_blue[3]'), while
    the colour survives."""
    mpv = b.MaterialPropertyValues2
    if mpv is None:
        raise FeatureFailed(f"body {b.Name} has no colour -- reload the part from disk")
    rgb = list(mpv[:3])
    return min(GROUPS, key=lambda g: sum((a - c_) ** 2 for a, c_ in zip(rgb, GROUPS[g])))


def name_by_colour(model):
    """Re-apply white / white_2 / graphite_1 / blue_1 ... names from colour."""
    bs = bodies(model)
    for i, b in enumerate(bs):
        b.Name = f"tmp_{i}"
    n = {}
    for b in sorted(bodies(model), key=lambda b: -volume(b)):
        g = group_of(b)
        n[g] = n.get(g, 0) + 1
        b.Name = g if (g == "white" and n[g] == 1) else f"{g}_{n[g]}"
    return n


def strip_styling(model, want_volume):
    """Delete every GL_* feature (and what they absorb) -> the original part.

    For re-styling a part whose STYLED file is already saved: the file on disk
    is locked while SolidWorks has it open, so the original cannot be copied
    back over it.  Proves the result is the original by volume (and one body)."""
    # A sketch shared by two features (a slab sketch cut twice, a profile used
    # by a trim) survives "delete absorbed": pass again until none are left.
    n = 0
    for _ in range(4):
        model.ClearSelection2(True)
        k = 0
        for f in list(_iter_features(model)):
            if f.Name.startswith("GL_") and f.Select2(k > 0, 0):
                k += 1
        if not k:
            break
        model.Extension.DeleteSelection2(c.swDelete_Absorbed | c.swDelete_Children)
        n += k
    model.ClearSelection2(True)
    model.EditRebuild3()
    left = [f for f in _iter_features(model) if f.Name.startswith("GL_")]
    bs = bodies(model)
    v = sum(volume(b) for b in bs)
    if left or len(bs) != 1 or abs(v - want_volume) > 1e-4 * want_volume:
        raise FeatureFailed(f"strip left {len(left)} GL_ features, {len(bs)} bodies, "
                            f"{v:.1f} mm3 vs the original {want_volume:.1f}")
    return n


def _iter_features(model):
    f = wrap(model.FirstFeature(), sld.IFeature)
    while f is not None:
        yield f
        f = wrap(f.GetNextFeature(), sld.IFeature)
