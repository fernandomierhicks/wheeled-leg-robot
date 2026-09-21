r"""Render one styled part from several angles onto a single sheet.

The END-ON view is the important one and the reason this exists: looking down
the part's long axis is what exposed added material standing off the original
as a thin full-depth fin.  A three-quarter view flatters that; an end-on view
does not, so it is always drawn.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import paths, spec as S
import render_color as RC
from build123d import import_step
from render3d import tessellate

PAL = S.PALETTES["arctic_lt"]


def views_for(part, tag="glacier"):
    """The four views, with the show-face elevation taken from the SPEC.

    THIS WAS HARD-CODED TO elev=+88 AND WAS WRONG ON TWO OF THE THREE LINKS.
    +88 looks at the part's +Z side.  The Tibia's show face is local +Z so it
    was right there; the Femur's and the Coupler's are local -Z, so the tile
    labelled "show face" has been showing their BACK all along.

    It is not obvious from the picture either, because the light is fixed at
    roughly +Z: a -Z face renders at the 0.24 ambient floor, so the Femur's
    white show face comes out near-black and its graphite back comes out
    bright.  The two errors compound into a render that looks plausible and is
    inside out.  Measured on the current build -- probing 0.6 mm inside each
    face of the plate -- the -Z face is 89.9% white and the +Z face is 65.6%
    graphite, and classifying rendered pixels by hue agrees: elev -88 reads
    76.6% white, elev +88 reads 57.0% graphite.

    `light="camera"` below keeps a face-on view legible whichever way it faces.
    """
    sf = "+Z"
    f = os.path.join(paths.SPECS, f"{part.lower()}.json")
    if os.path.exists(f):
        sf = json.load(open(f)).get("show_face", "+Z")
    shw = 88 if sf == "+Z" else -88
    return [("end-on  (down X)", 0, 2), ("end-on  (down Y)", 0, 92),
            ("three-quarter", 62, -58),
            (f"SHOW face (local {sf})", shw, -90),
            (f"BACK face (local {'-Z' if sf == '+Z' else '+Z'})", -shw, -90)]


def bodies_for(part, tag="glacier"):
    pdir = paths.print_dir(part, tag)
    out, allV = [], []
    for nm, col in (("white", PAL["white"]), ("graphite", PAL["dark"]),
                    ("accent", PAL["accent"])):
        f = os.path.join(pdir, f"{part}_{tag}_{nm}.step")
        if not os.path.exists(f):
            continue
        shp = import_step(f)                    # whole compound (trap 7)
        V, T, _ = tessellate(shp, 0.12)
        if len(T):
            out.append((V, T, col)); allV.append(V)
    return out, (np.vstack(allV) if allV else np.zeros((1, 3)))


if __name__ == "__main__":
    TAG = "glacier"
    if "--tag" in sys.argv:
        i = sys.argv.index("--tag"); TAG = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    paths.ensure_out()
    parts = sys.argv[1:] or ["Femur", "Coupler", "Tibia", "Side panel"]
    for part in parts:
        b, allV = bodies_for(part, TAG)
        if not b:
            print(f"{part}: nothing built"); continue
        tiles = [(nm, "", RC.view(b, allV, (820, 700), elev=e, azim=a,
                                  light="camera" if abs(e) > 80 else None))
                 for nm, e, a in views_for(part, TAG)]
        out = os.path.join(paths.RENDERS, f"part_{part.replace(' ', '_')}_{TAG}.png")
        RC.sheet(tiles, 2, out, header=f"{part} - GLACIER",
                 sub="end-on views first: added material must read as structure, "
                     "not as a fin stuck on the original.  Show/back taken from "
                     "the spec, face-on views lit from the camera",
                 size=(820, 700))
        print(f"wrote {out}")
