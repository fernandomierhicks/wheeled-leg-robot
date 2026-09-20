"""Extract the true plan outline as a shapely polygon, per-edge discretised."""
import numpy as np
from build123d import *
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane
from shapely.geometry import Polygon as ShPoly

def top_plan_face(solid, ZT):
    return max((f for f in solid.faces()
                if BRepAdaptor_Surface(f.wrapped).GetType()==GeomAbs_Plane
                and abs(f.center().Z-ZT)<1e-6 and f.normal_at(f.center()).Z>0.9),
               key=lambda f:f.area)

def outline_polygon(wire, n_per_edge=64):
    pts=[]
    for e in wire.order_edges() if hasattr(wire,'order_edges') else wire.edges():
        ln = max(e.length, 1e-6)
        k  = max(6, min(n_per_edge, int(ln/0.35)+6))
        seg=[e@(i/(k-1)) for i in range(k)]
        if pts and (np.hypot(seg[0].X-pts[-1][0], seg[0].Y-pts[-1][1]) >
                    np.hypot(seg[-1].X-pts[-1][0], seg[-1].Y-pts[-1][1])):
            seg=seg[::-1]
        pts += [(p.X,p.Y) for p in seg]
    P=ShPoly(pts)
    if not P.is_valid: P=P.buffer(0)
    return P

if __name__=="__main__":
    SRC=r"C:\Dropbox\Personal Projects\Robotics\wheeled-leg-robot\cad\v4 Larger Ball bearings\Tibia.STEP"
    s=import_step(SRC).solids()[0]; ZT=s.bounding_box().max.Z
    f=top_plan_face(s,ZT); ow=f.outer_wire()
    P=outline_polygon(ow)
    print(f"outer wire: {len(ow.edges())} edges, length {ow.length:.1f} mm")
    print(f"polygon: valid={P.is_valid} area={P.area:.0f} mm2  bounds={[round(v,1) for v in P.bounds]}")
    for t in (1.6, 3.0, 6.0):
        Q=P.buffer(-t, join_style=2)
        print(f"  buffer(-{t}): type={Q.geom_type} area={Q.area:.0f} bounds={[round(v,1) for v in Q.bounds]}")
