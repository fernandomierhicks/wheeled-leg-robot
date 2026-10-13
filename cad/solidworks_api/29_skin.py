"""Step 29: 3MF with colour only in a thin skin (offline, SolidWorks untouched).

    cadenv/python 29_skin.py <Part> [--depth 0.8]

Blue and graphite only need ~2 layers to show; deeper they cost colour changes and
are weaker PLA.  From `28 export` (out/relief/src/): every planar face of a blue /
graphite body that is exposed (nothing of the part just outside it) is pushed 0.8 mm
into the body; what no such prism reaches becomes WHITE.  White is filament 1,
graphite 2, blue 3; every filament is ONE object in the 3MF (all greys = one body).
Curved bodies (the PP_ hoses) are left whole.  Writes out/print/<Part>_skin.3mf --
the normal 3MF is not touched.  Prints volumes: kept + turned white = the original.
"""
import importlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = importlib.import_module("28_relief")
R._aes()
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane
from build123d import Solid, Vector

NOT_WHITE = ("blue", "graphite", "dark")
FIL = {"white": 1, "graphite": 2, "blue": 3, "dark": 2}     # one grey: dark prints as graphite


def exposed(face, bodies):
    """True if a point just outside the face (every sample) is in no body of the part."""
    n = face.normal_at()
    c = face.center()
    pts = [c]
    bb = face.bounding_box()
    for u in (0.25, 0.75):
        for v in (0.25, 0.75):
            p = Vector(bb.min.X + u * (bb.max.X - bb.min.X), bb.min.Y + u * (bb.max.Y - bb.min.Y) * 0 + v * (bb.max.Y - bb.min.Y),
                       bb.min.Z + u * (bb.max.Z - bb.min.Z))
            if face.is_inside(p, 1e-3):
                pts.append(p)
    out = 0
    for p in pts:
        q = p + n * 0.05
        if not any(b["solid"].is_inside(q) for b in bodies):
            out += 1
    return out == len(pts)


def thick(f, depth):
    """The solid `depth` mm behind face f: a prism if planar, a thickened shell otherwise."""
    if BRepAdaptor_Surface(f.wrapped).GetType() == GeomAbs_Plane:
        return Solid.extrude(f, -f.normal_at() * depth)
    return Solid.thicken(f, -depth)


def skin(P, depth):
    """[(solid, kind)] with every colour body cut to its skin; the rest as white.
    Each EXPOSED face (nothing of the part just outside it) is thickened `depth` into the
    body; colour stays only there.  A body with a free-form face (the PP_ hoses) stays whole."""
    from OCP.GeomAbs import GeomAbs_BSplineSurface
    res, report = [], []
    for b in P.bodies:
        if b["kind"] not in NOT_WHITE:
            res.append((b["solid"], b["kind"]))
            continue
        s = b["solid"]
        if any(BRepAdaptor_Surface(f.wrapped).GetType() == GeomAbs_BSplineSurface for f in s.faces()):
            res.append((s, b["kind"]))
            report.append(f"{b['name']}: free-form faces, left whole")
            continue
        deep, n_exp, bad = s, 0, 0
        for f in s.faces():
            if not exposed(f, P.bodies):
                continue
            n_exp += 1
            try:
                deep = deep - thick(f, depth)
            except Exception:                       # noqa: BLE001
                bad += 1
        keep = s - deep
        v0, vk, vd = s.volume, R._vol(keep), R._vol(deep)
        if abs(vk + vd - v0) > 0.05 or vd < 0 or bad:      # a thickened curved face broke the boolean
            res.append((s, b["kind"]))
            report.append(f"{b['name']:15s} {b['kind']:8s} skin boolean unreliable (err {vk + vd - v0:+.1f}, {bad} failed): left WHOLE")
            continue
        report.append(f"{b['name']:15s} {b['kind']:8s} {n_exp:3d} exposed faces{f' ({bad} failed)' if bad else ''}  "
                      f"{v0:9.1f} -> colour {vk:9.1f} + white {vd:9.1f}  (sum err {vk + vd - v0:+.3f})")
        for x in keep.solids():
            if x.volume > 0.01:
                res.append((x, b["kind"]))
        for x in deep.solids():
            if x.volume > 0.01:
                res.append((x, "white"))
    return res, report


def main():
    part = sys.argv[1]
    depth = float(sys.argv[sys.argv.index("--depth") + 1]) if "--depth" in sys.argv else 0.8
    import export3mf
    from render3d import tessellate
    P = R.Part(part, built=True)
    res, report = skin(P, depth)
    print("\n".join(report))
    by = {}
    for s, k in res:
        V, T, _ = tessellate(s, 0.05)
        by.setdefault(FIL[k], []).append((np.asarray(V), np.asarray(T)))
    objs = []
    for f, ms in sorted(by.items()):
        V = np.vstack([m[0] for m in ms])
        off = np.cumsum([0] + [len(m[0]) for m in ms[:-1]])
        T = np.vstack([m[1] + o for m, o in zip(ms, off)])
        objs.append((f"{part}_f{f}", V, T, f))
    out = os.path.join(HERE, "out", "print", f"{part}_skin.3mf")
    export3mf.write_3mf(objs, out, name=f"{part} (skin {depth})")
    print({o[3]: len(o[2]) for o in objs}, "triangles ->", out)


if __name__ == "__main__":
    main()
