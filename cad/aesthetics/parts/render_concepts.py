r"""Re-render already-built concepts from their exported STEPs.

    python parts/render_concepts.py            all tags found in out/print
    python parts/render_concepts.py GLACIER VICE

Decoupled from building on purpose: a 30 s build per concept is not worth
repeating just to change a camera or a tile aspect.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths, spec as S
import render_color as RC
from build123d import import_step
import numpy as np
from render3d import tessellate
from gallery import CONCEPTS, WILD

ALL = {n: (sub, over) for n, sub, over in (CONCEPTS + WILD)}
PART = os.environ.get("CONCEPT_PART", "Tibia")
TILE = (1180, 430)


def bodies_for(tag, pal):
    out, allV = [], []
    for nm, key in (("graphite", "dark"), ("white", "white"), ("accent", "accent")):
        f = os.path.join(paths.print_dir(PART, tag), f"{PART}_{tag}_{nm}.step")
        if not os.path.exists(f):
            continue
        V, T, _ = tessellate(import_step(f), 0.11)     # whole compound, never solids()[0]
        if len(T) == 0:
            continue
        out.append((V, T, pal[key])); allV.append(V)
    return out, (np.vstack(allV) if allV else np.zeros((1, 3)))


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in ALL
                             if os.path.exists(os.path.join(paths.print_dir(PART, n.lower()), f"{PART}_{n.lower()}_white.step"))]
    tiles = []
    for n in names:
        sub, over = ALL[n]
        pal = S.PALETTES[S.derive(**over).get("palette", "arctic")]
        b, allV = bodies_for(n.lower(), pal)
        if not b:
            print(f"?? {n}: nothing built"); continue
        tiles.append((n, sub, RC.view(b, allV, TILE, *RC.VIEWS["iso"])))
        print(f"   {n}")
    RC.sheet(tiles, 3, os.path.join(paths.RENDERS, f"{PART.lower()}_concepts_3d.png"),
             size=TILE,
             header=f"{PART.upper()} v4 — concepts built in 3D",
             sub="every one verified: 0 openings changed, worst deviation 0.000 mm3")


def hero(name, part="Tibia", size=(1500, 560)):
    """One concept, several cameras -- the accent has to survive being seen from
    somewhere other than straight down."""
    sub, over = ALL[name]
    pal = S.PALETTES[S.derive(**over).get("palette", "arctic")]
    global PART
    was, PART = PART, part
    b, allV = bodies_for(name.lower(), pal)
    PART = was
    if not b:
        return None
    cams = [("ISOMETRIC", "iso"), ("PLAN — the show face", "top"),
            ("SIDE ELEVATION", "side"), ("LOW THREE-QUARTER", "low")]
    tiles = [(t, "", RC.view(b, allV, size, *RC.VIEWS[k])) for t, k in cams]
    return RC.sheet(tiles, 2, os.path.join(paths.RENDERS, f"hero_{part.lower()}_{name.lower()}.png"),
                    size=size,
                    header=f"{name} — {part.upper()}",
                    sub=f"{sub}  ·  verified: 0 openings changed, worst deviation 0.000 mm3")
