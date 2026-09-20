"""Every opening in the plate, from BOTH plan faces -> a shapely keep-out set."""
import numpy as np
from build123d import *
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane
from shapely.geometry import Polygon as ShPoly
from shapely.ops import unary_union
from outline import outline_polygon

def plan_faces(solid, z, nz):
    return [f for f in solid.faces()
            if BRepAdaptor_Surface(f.wrapped).GetType()==GeomAbs_Plane
            and abs(f.center().Z-z)<1e-6 and f.normal_at(f.center()).Z*nz>0.9]

def openings(solid, ZT, ZB):
    polys=[]
    for z,nz in ((ZT,1),(ZB,-1)):
        for f in plan_faces(solid,z,nz):
            for w in f.inner_wires():
                p=outline_polygon(w)
                if p.is_valid and p.area>0.5: polys.append(p)
    return polys

if __name__=="__main__":
    SRC=r"C:\Dropbox\Personal Projects\Robotics\wheeled-leg-robot\cad\v4 Larger Ball bearings\Tibia.STEP"
    s=import_step(SRC).solids()[0]; bb=s.bounding_box(); ZT,ZB=bb.max.Z,bb.min.Z
    ps=openings(s,ZT,ZB)
    print(f"{len(ps)} openings found")
    for p in sorted(ps,key=lambda p:-p.area)[:40]:
        c=p.centroid; xs,ys=p.exterior.xy
        w=max(xs)-min(xs); h=max(ys)-min(ys)
        print(f"   area={p.area:8.1f}  c=({c.x:7.2f},{c.y:7.2f})  bbox {w:6.2f} x {h:6.2f}")
    U=unary_union(ps)
    print(f"\nunion area={U.area:.0f} mm2  parts={len(U.geoms) if U.geom_type=='MultiPolygon' else 1}")
    for m in (2.0,3.0):
        B=U.buffer(m)
        print(f"  buffered +{m}: area={B.area:.0f} parts={len(B.geoms) if B.geom_type=='MultiPolygon' else 1}")

def silhouette(solid):
    """True plan silhouette: union of the outer wires of every horizontal face.

    The top plan face alone is NOT the silhouette -- on the Tibia it stops at
    x=204.7 because the wheel hub sits in a recess at a lower Z.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface as _S
    from OCP.GeomAbs import GeomAbs_Plane as _P
    polys=[]
    for f in solid.faces():
        if _S(f.wrapped).GetType()!=_P: continue
        n=f.normal_at(f.center())
        if abs(n.Z)<0.99: continue
        p=outline_polygon(f.outer_wire())
        if p.is_valid and p.area>1.0: polys.append(p)
    return unary_union(polys).buffer(0)
