r"""Rewrite every part's coloured STEP from the per-filament STEPs already on
disk -- welded manifold, one root, honest tolerance -- WITHOUT a 2-4 minute
rebuild of the geometry.

The bodies in out/print/<Part>/<Part>_glacier_{white,graphite,accent}.step are
the build's own output and are not re-derived here; only the packaging changes.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import paths, manifold, stepcolor, spec as _spec
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopoDS import TopoDS_Compound, TopoDS_Builder

TAG = "glacier"
for part in (sys.argv[1:] or ["Femur", "Tibia", "Coupler", "Side panel", "RobotMount"]):
    sf = os.path.join(paths.SPECS, part.lower().replace(" ", "_") + ".json")
    pal = _spec.PALETTES[json.load(open(sf)).get("palette", "arctic_lt")
                         if os.path.exists(sf) else "arctic_lt"]
    col = {"white": pal["white"], "graphite": pal["dark"], "accent": pal["accent"]}
    pdir = paths.print_dir(part, TAG)
    bodies, before = [], 0.0
    for n in ("white", "graphite", "accent"):
        f = os.path.join(pdir, f"{part}_{TAG}_{n}.step")
        if not os.path.exists(f):
            continue
        r = STEPControl_Reader(); r.ReadFile(f); r.TransferRoots()
        c = TopoDS_Compound(); b = TopoDS_Builder(); b.MakeCompound(c)
        ex = TopExp_Explorer(r.OneShape(), TopAbs_SOLID)
        while ex.More():
            b.Add(c, ex.Current()); before += manifold.volume(ex.Current()); ex.Next()
        bodies.append((n, c, col[n]))
    if not bodies:
        print(f"{part}: no per-filament STEPs; skipped"); continue
    print(f"\n{part}:")
    p = stepcolor.write(bodies, paths.colour_step(part, TAG), part=part,
                        say=lambda m: print(m))
    r = STEPControl_Reader(); r.ReadFile(p); r.TransferRoots()
    ex = TopExp_Explorer(r.OneShape(), TopAbs_SOLID)
    n = 0; tot = 0.0; nm = 0
    while ex.More():
        n += 1; tot += manifold.volume(ex.Current())
        nm += len(manifold.nonmanifold_edges(ex.Current())); ex.Next()
    status = "OK" if nm == 0 else f"STILL {nm} NON-MANIFOLD"
    print(f"  -> {n} bodies, {tot:.3f} mm3 (was {before:.3f}), "
          f"non-manifold edges {nm}   {status}")
