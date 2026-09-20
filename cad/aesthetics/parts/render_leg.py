r"""Assemble femur + tibia into a leg pose and render the pair in colour.

    python parts/render_leg.py                 every concept built for both parts
    python parts/render_leg.py GLACIER VICE

The pose is ILLUSTRATIVE, not kinematically solved -- `input/assembly/` is still
empty, so the 4-bar has never been closed against the real assembly.  What IS
real is the link geometry, taken from CLAUDE.md:

    |AC| = 187.58 mm     femur, hip output shaft -> knee pivot
    |CW| = 185.91 mm     tibia, knee pivot -> wheel centre

and the datums in each part's own STEP frame:

    femur   A = (-93.79, 0)     C = (+93.79, 0)      -> |AC| = 187.58  exact
    tibia   E = (0, 0)  C = (38.65, -5.28)  W = (224.49, 0)
            |C->W| = 185.91  exact

Both parts are plates in their own XY plane, and the leg swings in the sagittal
plane, so assembling in XY and keeping Z gives a true side elevation of the leg.
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import paths, spec as S
import render_color as RC
from build123d import import_step
from render3d import tessellate
from gallery import CONCEPTS, WILD

ALL = {n: (sub, over) for n, sub, over in (CONCEPTS + WILD)}
A_F, C_F = np.array([-93.79, 0.0]), np.array([93.79, 0.0])
C_T, W_T = np.array([38.65, -5.28]), np.array([224.49, 0.0])
TH_FEMUR = math.radians(-30.0)          # hip angle, illustrative
TH_SHANK = math.radians(-100.0)         # knee->wheel direction, illustrative
TILE = (900, 1180)


def _rot(V, th, about=(0.0, 0.0)):
    c, s = math.cos(th), math.sin(th)
    out = V.copy()
    p = out[:, :2] - np.asarray(about)
    out[:, 0] = p[:, 0] * c - p[:, 1] * s + about[0]
    out[:, 1] = p[:, 0] * s + p[:, 1] * c + about[1]
    return out


def _load(part, tag, pal):
    out = []
    for nm, key in (("graphite", "dark"), ("white", "white"), ("accent", "accent")):
        f = os.path.join(paths.print_dir(part, tag), f"{part}_{tag}_{nm}.step")
        if not os.path.exists(f):
            continue
        V, T, _ = tessellate(import_step(f), 0.11)   # whole compound, never solids()[0]
        if len(T):
            out.append([V, T, pal[key]])
    return out


def leg(tag, pal):
    fem, tib = _load("Femur", tag, pal), _load("Tibia", tag, pal)
    if not fem or not tib:
        return None, None
    # femur: put A at the origin, then swing it about A
    knee = np.array([187.58 * math.cos(TH_FEMUR), 187.58 * math.sin(TH_FEMUR)])
    for b in fem:
        V = b[0].copy(); V[:, :2] -= A_F
        b[0] = _rot(V, TH_FEMUR)
    # tibia: land its C on the knee, then swing the shank about C
    cw = W_T - C_T
    th = TH_SHANK - math.atan2(cw[1], cw[0])
    for b in tib:
        V = b[0].copy(); V[:, :2] -= C_T
        V = _rot(V, th)
        V[:, :2] += knee
        b[0] = V
    bodies = fem + tib
    return bodies, np.vstack([b[0] for b in bodies])


if __name__ == "__main__":
    names = sys.argv[1:] or [
        n for n in ALL
        if os.path.exists(os.path.join(paths.print_dir("Femur", n.lower()), f"Femur_{n.lower()}_white.step"))
        and os.path.exists(os.path.join(paths.print_dir("Tibia", n.lower()), f"Tibia_{n.lower()}_white.step"))]
    tiles = []
    for n in names:
        sub, over = ALL[n]
        pal = S.PALETTES[S.derive(**over).get("palette", "arctic")]
        b, allV = leg(n.lower(), pal)
        if b is None:
            print(f"?? {n}: not built for both parts"); continue
        # The leg swings in the parts' own XY plane, so the sagittal view is a
        # near-PLAN camera.  A low oblique looks straight down the femur's edge.
        tiles.append((n, sub, RC.view(b, allV, TILE, 74, -90)))
        print(f"   {n}")
    if tiles:
        RC.sheet(tiles, 4, os.path.join(paths.RENDERS, "leg_concepts_3d.png"), size=TILE,
                 header="WHEELED-LEG ROBOT — the leg, built and rendered",
                 sub="femur + tibia, one concept each · pose is illustrative; "
                     "link lengths are real (|AC| 187.58, |CW| 185.91 from CLAUDE.md)")
