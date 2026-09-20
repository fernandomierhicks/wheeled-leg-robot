r"""Render the real assembly, in the real poses, with the styled parts in colour.

    C:/Users/ferna/cadenv/Scripts/python.exe parts/render_asm.py

parts/render_leg.py predates the assembly STEP and poses the leg illustratively.
This does not: every component sits at the transform SolidWorks exported, so
what comes out is the actual robot, not an approximation of it.

Styled parts are drawn from their three filament bodies in GLACIER colours.
Everything else -- motors, bearings, fasteners, parts not yet styled -- is drawn
in a flat neutral so the eye goes to the styling and so an unstyled part cannot
be mistaken for a finished one.

The camera looks from +Z, which the assembly walk showed is OUTBOARD: the wheel
sits at Z 169..203 and the RobotMount at 80..99, so +Z is the side of the robot
you actually see.  That is the same finding that moved the Femur's show face.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import paths, spec as S
import render_color as RC
from build123d import import_step
from render3d import tessellate
from collide import leaves, POSES

PAL = S.PALETTES["arctic_lt"]
NEUTRAL = (86, 92, 102)          # unstyled / COTS
DEV_STYLED, DEV_OTHER = 0.15, 0.6
_CACHE = {}


def _styled_bodies(part):
    """[(solid, rgb)] for a styled part, or None if it has not been built."""
    if part in _CACHE:
        return _CACHE[part]
    sf = os.path.join(paths.SPECS, part.lower() + ".json")
    out = None
    if os.path.exists(sf):
        tag = json.load(open(sf)).get("tag", "arctic")
        got = []
        for nm, col in (("white", PAL["white"]), ("graphite", PAL["dark"]),
                        ("accent", PAL["accent"])):
            p = os.path.join(paths.print_dir(part, tag), f"{part}_{tag}_{nm}.step")
            if os.path.exists(p):
                got.append((import_step(p), col))
        out = got or None
    _CACHE[part] = out
    return out


def pose_bodies(pose):
    step = os.path.join(paths.EXPORTS, pose + ".STEP")
    if not os.path.exists(step):
        return None, None, 0
    bodies, allV, n_styled = [], [], 0
    for name, shp, loc in leaves(step):
        sb = _styled_bodies(name)
        if sb:
            n_styled += 1
            for solid, col in sb:
                # a filament body is SEVERAL solids -- tessellate the whole
                # compound, never solids()[0] (trap 7)
                V, T, _ = tessellate(solid.moved(loc), DEV_STYLED)
                if len(T):
                    bodies.append((V, T, col)); allV.append(V)
        else:
            V, T, _ = tessellate(shp.moved(loc), DEV_OTHER)
            if len(T):
                bodies.append((V, T, NEUTRAL)); allV.append(V)
    return bodies, (np.vstack(allV) if allV else np.zeros((1, 3))), n_styled


if __name__ == "__main__":
    paths.ensure_out()
    poses = sys.argv[1:] or POSES
    tiles = []
    for pose in poses:
        print(f"  {pose} ...", flush=True)
        bodies, allV, n = pose_bodies(pose)
        if bodies is None:
            continue
        im = RC.view(bodies, allV, (900, 1000), elev=74, azim=-64)
        tiles.append((pose, f"{n} styled parts, {len(bodies)} bodies", im))
        print(f"    {n} styled, {len(bodies)} bodies, {sum(len(t) for _,t,_ in bodies)} tris",
              flush=True)
    out = os.path.join(paths.RENDERS, "assembly_glacier.png")
    RC.sheet(tiles, len(tiles), out,
             header="GLACIER on the v4 leg - real assembly, real poses",
             sub="styled parts in colour; motors, bearings, fasteners and "
                 "unstyled parts in neutral grey. Viewed from OUTBOARD (+Z).",
             size=(900, 1000))
    print(f"wrote {out}")
