"""Step 30: Side panel -- colour the bodies he marked green white (they are never seen).

    cadenv/python 30_sidepanel.py

On the inner face (z 10): the blue lining of the bearing bore (blue_6), the blue combs / S
trace (blue_2, blue_5), the two lower graphite plates (graphite_1, graphite_3).  Geometry is
untouched (no mate face changes); each body is only re-coloured to the part's white.  The
bodies are found by the volumes in out/relief/src/Side panel.json (the export before the
accents).  Then 16 --chain.
"""
import json
import os
import sys
from importlib import import_module

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib
import swstyle as S
from swlib import c, wrap, sld

GREEN = ["blue_6", "blue_2", "blue_5", "graphite_1", "graphite_3",      # inner face (z 10)
         "blue_4", "blue_3", "blue_1", "graphite_5", "graphite_6", "graphite_2"]   # back face (z 0), his call 2026-10-09: not visible


def whiten(sw, doc=None):
    """Colour GREEN white in the open Side panel (geometry untouched).  Idempotent."""
    info = json.load(open(os.path.join(HERE, "out", "relief", "src", "Side panel.json")))
    vol = {b["name"]: b["volume"] for b in info["bodies"]}
    if doc is None:
        path = os.path.join(swlib.V5, r"Body\Side panel.SLDPRT")
        doc = wrap(sw.OpenDoc6(path, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)[0], sld.IModelDoc2)
    sw.ActivateDoc3(doc.GetTitle(), False, 0, 0)
    bs = S.bodies(doc)
    white = max(bs, key=S.volume)
    wrgb = list(white.MaterialPropertyValues2[:3])
    for n in GREEN:
        b = min(bs, key=lambda x: abs(S.volume(x) - vol[n]))
        if abs(S.volume(b) - vol[n]) > 1e-4 * vol[n] + 0.05:
            raise SystemExit(f"{n}: no body of {vol[n]} mm3 (nearest {S.volume(b):.3f})")
        S.colour(b, wrgb)
        print(f"{n} ({vol[n]} mm3) -> white", flush=True)
    doc.EditRebuild3()


def main():
    sw, _ = swlib.connect()
    swlib.open_v5(sw)
    whiten(sw)
    import_module("16_check_and_save").chain(sw, [r"Body\Side panel.SLDPRT"])


if __name__ == "__main__":
    main()
