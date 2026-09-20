r"""Heal bodies before writing a STEP, and prove the healing changed no volume.

BRepCheck_Analyzer calls every one of these bodies valid, and SolidWorks still
will not take one of them.  OCC-valid and SolidWorks-importable are different
bars: booleans leave behind coplanar face splits, sliver faces and seam edges
that OCC tolerates and a stricter kernel rejects.

`ShapeUpgrade_UnifySameDomain` merges faces that lie on the same surface -- it
is the standard cure for boolean debris -- and `ShapeFix_Shape` repairs
wire/face/shell defects.  Both can silently CHANGE geometry, so every body is
volume-checked and the original is kept if healing moved more than a tolerance.

    python tools/stepheal.py Femur            # whole part, healed, + one-body probes
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import paths
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.ShapeFix import ShapeFix_Shape
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE


def _vol(s):
    p = GProp_GProps(); BRepGProp.VolumeProperties_s(s, p); return p.Mass()


def _nfaces(s):
    n = 0; ex = TopExp_Explorer(s, TopAbs_FACE)
    while ex.More(): n += 1; ex.Next()
    return n


def heal(shape, tol=1e-4, say=print):
    """TopoDS_Shape -> healed TopoDS_Shape.  Falls back to the original if the
    healing moved volume, because a smaller file is worth nothing if the part
    changed shape on the way to it."""
    v0, f0 = _vol(shape), _nfaces(shape)
    out = shape
    try:
        fx = ShapeFix_Shape(out)
        fx.SetPrecision(tol); fx.SetMaxTolerance(1e-2)
        fx.Perform()
        cand = fx.Shape()
        if abs(_vol(cand) - v0) <= max(1e-3, 1e-6 * abs(v0)):
            out = cand
        else:
            say(f"      ShapeFix moved volume {v0:.3f} -> {_vol(cand):.3f}; kept original")
    except Exception as e:
        say(f"      ShapeFix failed ({type(e).__name__}); kept original")
    try:
        u = ShapeUpgrade_UnifySameDomain(out, True, True, True)
        u.Build()
        cand = u.Shape()
        if abs(_vol(cand) - v0) <= max(1e-3, 1e-6 * abs(v0)):
            out = cand
        else:
            say(f"      UnifySameDomain moved volume {v0:.3f} -> {_vol(cand):.3f}; kept original")
    except Exception as e:
        say(f"      UnifySameDomain failed ({type(e).__name__}); kept original")
    say(f"      faces {f0} -> {_nfaces(out)}   volume {v0:.3f} -> {_vol(out):.3f} mm3")
    return out


if __name__ == "__main__":
    # Two probes, because they answer different questions:
    #   <Part>_healed.step      does healing cure the whole part?
    #   <Part>_<body>_only.step is that ONE body importable on its own?
    # A body that opens alone but vanishes in company is an interaction; a body
    # that vanishes alone is the body.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import stepforms
    from build123d import Shape, import_step
    import spec as _spec, json

    part = sys.argv[1] if len(sys.argv) > 1 else "Femur"
    probe = sys.argv[2] if len(sys.argv) > 2 else None      # e.g. graphite
    tag = "glacier"
    sf = os.path.join(paths.SPECS, part.lower().replace(" ", "_") + ".json")
    palette = json.load(open(sf)).get("palette", "arctic_lt") if os.path.exists(sf) else "arctic_lt"
    pal = _spec.PALETTES[palette]
    col = {"white": pal["white"], "graphite": pal["dark"], "accent": pal["accent"]}
    pdir = paths.print_dir(part, tag)
    out = os.path.join(paths.ROOT, "out", "stepforms", part)

    raw, healed = [], []
    for n in ("white", "graphite", "accent"):
        f = os.path.join(pdir, f"{part}_{tag}_{n}.step")
        if not os.path.exists(f):
            continue
        shp = import_step(f)
        raw.append((n, shp, col[n]))
        print(f"  {n}:")
        healed.append((n, Shape(heal(shp.wrapped, say=lambda m: print(m))), col[n]))

    p1 = stepforms.write(healed, os.path.join(out, f"{part}_healed.step"),
                         form="multibody", part=part)
    print(f"  wrote {p1}  {os.path.getsize(p1)/1e6:.2f} MB")

    if probe:
        one = [b for b in raw if b[0] == probe]
        if one:
            p2 = stepforms.write(one, os.path.join(out, f"{part}_{probe}_only.step"),
                                 form="multibody", part=f"{part}_{probe}")
            print(f"  wrote {p2}  {os.path.getsize(p2)/1e6:.2f} MB")
