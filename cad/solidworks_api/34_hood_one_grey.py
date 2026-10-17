"""Step 34: FacetHood -- one grey, one blue, and the slots in its graphite plates kept readable.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/34_hood_one_grey.py [--dry] [--skirt blue|white]

His calls 2026-10-16: there is only one grey filament and one blue.  The hood had two greys:
the box graphite #7E8795 (filament 2, also its hose collars) and the near-black dark #2B2F36
(filament 4); both are now the links' graphite (0.25, 0.27, 0.30), and its blue #1E7BFF the
links' blue (0.13, 0.45, 0.95) -- the hood prints on 3 filaments.  Then (his "make sure the
accents still make sense"): 56 of the 76 former dark bodies are slot floors sunk INTO graphite
(the skirt plates' slots, tier 1's front / corner plate slots, the visor, the exhaust, the field
windows) and went graphite on graphite; they become blue (the skirt's 36: --skirt).  The other
20 (pocket floors, front vents, armour windows) are in white faces and stay graphite.
Body colour only -- geometry and mates untouched.  Idempotent.  --dry: count, change nothing.
Then 16 --chain (done here), 28 export + print FacetHood.
"""
import json
import os
import sys
from importlib import import_module

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib
import swstyle as S

HOOD = r"Box\Facet\FacetHood.SLDPRT"
GRAPHITE = (0.25, 0.27, 0.30)                                       # swstyle.GROUPS graphite
BLUE = (0.13, 0.45, 0.95)                                           # swstyle.GROUPS blue
WHITE = (0xF2 / 255, 0xF4 / 255, 0xF7 / 255)                        # the hood's white
GREYS = [(0x7E / 255, 0x87 / 255, 0x95 / 255), (0x2B / 255, 0x2F / 255, 0x36 / 255)]   # box graphite, dark
BOX_BLUE = (0x1E / 255, 0x7B / 255, 1.0)
# which bodies were dark: 28's export of the hood before the one-grey change (names = SolidWorks')
PRE = os.path.join(swlib.V5, r"_originals\pre_one_grey_2026-10-16\relief_export\FacetHood.json")


def _is(rgb, ref):
    return max(abs(a - b) for a, b in zip(rgb, ref)) < 0.01


def on_graphite(c):
    """A former dark floor sunk into graphite (not into white), by its place (hood frame: x
    forward, y up): the skirt slots (y 82), tier 1's front + corner plate slots (y 85-88, x > 55;
    the front vents at y 89.9 and the back pockets at x -162 are in white), and on tier 2 the
    exhaust (y 105-106), visor (107) and field windows (109) -- not the armour wells (115.5)."""
    x, y, z = c
    return y < 84.0 or (84.0 < y < 88.5 and x > 55.0) or (100.0 < y < 113.0)


def slot_floors():
    """{body name: (volume, "skirt" / "plate")} -- the former dark floors in graphite."""
    out = {}
    for b in json.load(open(PRE))["bodies"]:
        if _is(b["rgb"], (0x2A / 255, 0x2F / 255, 0x35 / 255)):
            bx = b["box"]
            c = [(bx[i] + bx[i + 3]) / 2 for i in range(3)]
            if on_graphite(c):
                out[b["name"]] = (b["volume"], "skirt" if c[1] < 84.0 else "plate")
    return out


def recolour(sw, skirt="blue", dry=False):
    doc = import_module("21_box_facet").find_doc(sw, os.path.join(swlib.V5, HOOD))
    if doc is None:
        raise SystemExit("FacetHood is not open")
    sw.ActivateDoc3(doc.GetTitle(), False, 0, 0)
    floors = slot_floors()
    want = {"skirt": BLUE if skirt == "blue" else WHITE, "plate": BLUE}
    bodies = {b.Name: b for b in S.bodies(doc)}
    missing = [n for n in floors if n not in bodies]
    if missing or len(floors) != 56:
        raise SystemExit(f"slot floors: {len(floors)} found (56 expected), not in the part: {missing}")
    seen, n = {}, {"grey": 0, "blue": 0, "skirt": 0, "plate": 0}
    for name, b in bodies.items():
        rgb = tuple(b.MaterialPropertyValues2[:3])
        key = "#%02X%02X%02X" % tuple(round(x * 255) for x in rgb)
        seen[key] = seen.get(key, 0) + 1
        new = None
        if name in floors:
            vol, cls = floors[name]
            if abs(S.volume(b) - vol) > 1e-4 * vol + 0.01:
                raise SystemExit(f"{name}: {S.volume(b):.3f} mm3, the export had {vol} -- not the same body")
            new, k = want[cls], cls
        elif any(_is(rgb, g) for g in GREYS):
            new, k = GRAPHITE, "grey"
        elif _is(rgb, BOX_BLUE):
            new, k = BLUE, "blue"
        if new is not None and not _is(rgb, new):
            n[k] += 1
            if not dry:
                S.colour(b, new)
    print(f"FacetHood {len(bodies)} bodies, colours {dict(sorted(seen.items()))}")
    print(f"  {'would change' if dry else 'changed'}: greys -> graphite {n['grey']}, box blue -> links' blue "
          f"{n['blue']}, skirt slot floors -> {skirt} {n['skirt']}, plate / visor / exhaust / window floors "
          f"-> blue {n['plate']}", flush=True)
    if not dry:
        doc.EditRebuild3()


def main():
    dry = "--dry" in sys.argv
    skirt = sys.argv[sys.argv.index("--skirt") + 1] if "--skirt" in sys.argv else "blue"
    if skirt not in ("blue", "white"):
        raise SystemExit("--skirt blue|white")
    sw, _ = swlib.connect()
    swlib.open_v5(sw)
    recolour(sw, skirt, dry)
    if not dry:
        import_module("16_check_and_save").chain(sw, [HOOD])


if __name__ == "__main__":
    main()
