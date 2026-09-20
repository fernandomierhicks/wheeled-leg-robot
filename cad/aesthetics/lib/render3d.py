"""Load a STEP, report real geometry, and render shaded orthographic views."""
import sys, numpy as np
from build123d import import_step
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                         GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface)
SURF = {GeomAbs_Plane:'plane', GeomAbs_Cylinder:'cylinder', GeomAbs_Cone:'cone',
        GeomAbs_Sphere:'sphere', GeomAbs_Torus:'torus', GeomAbs_BSplineSurface:'bspline'}

def tessellate(shape, dev=0.15):
    BRepMesh_IncrementalMesh(shape.wrapped, dev, False, 0.3, True)
    verts, tris, kinds = [], [], []
    exp = TopExp_Explorer(shape.wrapped, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        kinds.append(SURF.get(BRepAdaptor_Surface(face).GetType(), 'other'))
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            base = len(verts)
            for i in range(1, tri.NbNodes()+1):
                p = tri.Node(i).Transformed(trsf)
                verts.append((p.X(), p.Y(), p.Z()))
            rev = face.Orientation() == 1
            for i in range(1, tri.NbTriangles()+1):
                a,b,c = tri.Triangle(i).Get()
                if rev: a,c = c,a
                tris.append((base+a-1, base+b-1, base+c-1))
        exp.Next()
    return np.array(verts,float), np.array(tris,int), kinds

def render(V, T, out, elev=22, azim=-58, size=900, bg=(28,30,34)):
    e,a = np.radians(elev), np.radians(azim)
    fwd = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    up0 = np.array([0,0,1.0])
    right = np.cross(up0, fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    P = np.stack([V@right, V@up, V@fwd], 1)
    mn, mx = P[:,:2].min(0), P[:,:2].max(0)
    span = (mx-mn).max() * 1.12
    ctr = (mn+mx)/2
    s = size/span
    xy = (P[:,:2]-ctr)*s + size/2
    xy[:,1] = size - xy[:,1]
    img = np.full((size,size,3), bg, np.uint8)
    zbuf = np.full((size,size), -1e18)
    tv = P[T][:,:,2].mean(1)
    order = np.argsort(tv)                      # far -> near
    light = np.array([0.35,0.55,0.75]); light/=np.linalg.norm(light)
    for ti in order:
        i0,i1,i2 = T[ti]
        p0,p1,p2 = V[i0],V[i1],V[i2]
        n = np.cross(p1-p0, p2-p0)
        ln = np.linalg.norm(n)
        if ln < 1e-12: continue
        n /= ln
        if n@fwd <= 0: continue                 # backface
        lam = max(0.0, n@light)
        shade = 0.20 + 0.80*lam**0.85
        col = np.clip(np.array([196,201,208])*shade + 26*lam**8, 0, 255)
        q = xy[[i0,i1,i2]]
        x0,y0 = np.floor(q.min(0)).astype(int); x1,y1 = np.ceil(q.max(0)).astype(int)
        x0,y0 = max(x0,0), max(y0,0); x1,y1 = min(x1,size-1), min(y1,size-1)
        if x1<x0 or y1<y0: continue
        yy,xx = np.mgrid[y0:y1+1, x0:x1+1]
        d = (q[1,1]-q[2,1])*(q[0,0]-q[2,0]) + (q[2,0]-q[1,0])*(q[0,1]-q[2,1])
        if abs(d) < 1e-9: continue
        w0 = ((q[1,1]-q[2,1])*(xx-q[2,0]) + (q[2,0]-q[1,0])*(yy-q[2,1]))/d
        w1 = ((q[2,1]-q[0,1])*(xx-q[2,0]) + (q[0,0]-q[2,0])*(yy-q[2,1]))/d
        w2 = 1-w0-w1
        m = (w0>=-1e-6)&(w1>=-1e-6)&(w2>=-1e-6)
        if not m.any(): continue
        z = w0*P[i0,2]+w1*P[i1,2]+w2*P[i2,2]
        sub = zbuf[y0:y1+1, x0:x1+1]
        m &= z > sub
        sub[m] = z[m]
        img[y0:y1+1, x0:x1+1][m] = col
    try:
        from PIL import Image; Image.fromarray(img).save(out)
    except ImportError:
        import matplotlib.pyplot as plt; plt.imsave(out, img)
    return out

if __name__ == '__main__':
    path, stem = sys.argv[1], sys.argv[2]
    shp = import_step(path)
    solids = shp.solids()
    print(f"--- {stem} ---")
    print(f"solids={len(solids)}  faces={len(shp.faces())}  edges={len(shp.edges())}")
    bb = shp.bounding_box()
    print(f"bbox mm  X {bb.min.X:8.2f}..{bb.max.X:8.2f}  ({bb.size.X:7.2f})")
    print(f"         Y {bb.min.Y:8.2f}..{bb.max.Y:8.2f}  ({bb.size.Y:7.2f})")
    print(f"         Z {bb.min.Z:8.2f}..{bb.max.Z:8.2f}  ({bb.size.Z:7.2f})")
    print(f"volume = {shp.volume/1000:.1f} cm^3  -> PLA mass ~{shp.volume/1000*1.24:.0f} g solid")
    V,T,kinds = tessellate(shp)
    from collections import Counter
    print("face types:", dict(Counter(kinds)))
    print(f"mesh: {len(V)} verts, {len(T)} tris")
    for nm,(el,az) in {'iso':(24,-58),'top':(89.9,-90),'side':(0,-90),'front':(0,0)}.items():
        render(V,T,f"{stem}_{nm}.png", elev=el, azim=az)
        print("wrote", f"{stem}_{nm}.png")
