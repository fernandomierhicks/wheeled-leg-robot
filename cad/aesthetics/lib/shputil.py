"""shapely <-> build123d helpers, hardened against slivers and duplicate points."""
import numpy as np
from build123d import *

def _clean_ring(coords, tol=1e-4):
    pts=[]
    for x,y in coords:
        if not pts or abs(x-pts[-1][0])>tol or abs(y-pts[-1][1])>tol: pts.append((x,y))
    if len(pts)>1 and abs(pts[0][0]-pts[-1][0])<tol and abs(pts[0][1]-pts[-1][1])<tol:
        pts.pop()
    return pts

def geoms(poly, min_area=0.05):
    gs = list(poly.geoms) if poly.geom_type=='MultiPolygon' else [poly]
    out=[]
    for g in gs:
        if g.is_empty or g.geom_type!='Polygon' or g.area<min_area: continue
        g=g.simplify(0.008)
        if g.is_empty or g.area<min_area: continue
        if len(_clean_ring(g.exterior.coords))<3: continue
        out.append(g)
    return out

def to_face(poly, min_area=0.05):
    out=None
    for g in geoms(poly, min_area):
        ring=_clean_ring(g.exterior.coords)
        f=make_face(Polyline(*[(x,y,0) for x,y in ring], close=True))
        for r in g.interiors:
            rr=_clean_ring(r.coords)
            if len(rr)>=3:
                f=f-make_face(Polyline(*[(x,y,0) for x,y in rr], close=True))
        out=f if out is None else out+f
    return out

def prism(poly, z0, z1, min_area=0.05):
    f=to_face(poly, min_area)
    return None if f is None else Pos(0,0,z0)*extrude(f, amount=z1-z0)

def frustum(g, h, draft):
    """Solid spanning z 0..h, FULL size at z=0, inset by h*tan(draft) at z=h.

    build123d's tapered extrude returns z in [-h, 0] with the wide end at z=0,
    so it must be mirrored before use.
    """
    ring=_clean_ring(g.exterior.coords)
    f=make_face(Polyline(*[(x,y,0) for x,y in ring], close=True))
    return mirror(extrude(f, amount=h, taper=draft), about=Plane.XY)

def raised(poly, z0, h, draft=14.0, min_area=4.0):
    """Raised pad sitting ON the surface: base at z0, drafted sides, top at z0+h."""
    out=None
    for g in geoms(poly, min_area):
        ins=h*np.tan(np.radians(draft))
        if g.buffer(-ins*1.05, join_style=2).is_empty:      # too narrow to draft
            ins=0.0
        try:
            s=Pos(0,0,z0)*frustum(g, h, draft if ins else 0.0)
        except Exception:
            s=prism(g, z0, z0+h)
        out=s if out is None else union(out, s)
    return out

def union(*parts):
    """Robust N-way fuse.  build123d's chained `+` can silently drop terms."""
    ps=[p for p in parts if p is not None]
    if not ps: return None
    out=ps[0]
    if hasattr(out,'solids') and len(out.solids())>1:
        acc=out.solids()[0]
        for s in out.solids()[1:]: acc=acc.fuse(s)
        out=acc
    for p in ps[1:]:
        for s in (p.solids() if hasattr(p,'solids') and p.solids() else [p]):
            out=out.fuse(s)
    return out.clean()
