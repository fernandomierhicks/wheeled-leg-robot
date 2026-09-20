"""How much room does each part actually have to grow, per z layer?

Answers the question I guessed at: is additive styling really blocked, or was
my growth just badly shaped?  Reports, per layer, the plan area available for a
band of each width outside the source silhouette, after the assembly keep-out.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "lib"))
import paths, asmkeepout
from build123d import import_step
from keepout import silhouette
from shputil import geoms

WIDTHS = (3.0, 6.0, 10.0)
for part in ("Femur", "Tibia", "Coupler", "Side panel", "RobotMount"):
    s = import_step(paths.part_step(part)).solids()[0]
    bb = s.bounding_box()
    OUT = max(geoms(silhouette(s)), key=lambda g: g.area)
    n = asmkeepout.n_layers(part)
    print(f"\n=== {part} ===  silhouette {OUT.area:8.0f} mm2   "
          f"Z {bb.min.Z:+.1f}..{bb.max.Z:+.1f}   {n} keep-out layers")
    for li in range(n):
        ko = asmkeepout.load(part, 1.0, layer=li)
        row = []
        for w in WIDTHS:
            band = OUT.buffer(w).difference(OUT)
            free = band.difference(ko) if ko is not None else band
            row.append(f"{w:.0f}mm: {free.area:7.0f} mm2 ({100*free.area/band.area:4.0f}%)")
        zlo = bb.min.Z + li * (bb.max.Z - bb.min.Z) / n
        zhi = bb.min.Z + (li + 1) * (bb.max.Z - bb.min.Z) / n
        print(f"  L{li} z {zlo:+6.1f}..{zhi:+6.1f}   " + "   ".join(row))
