r"""Concept gallery -- the solo exploration run.

    python parts/gallery.py palettes     16 palettes on one resolved design
    python parts/gallery.py accents      every accent style, one palette
    python parts/gallery.py silhouette   creased vs organic vs waisted
    python parts/gallery.py concepts     12 named, fully-composed concepts
    python parts/gallery.py wild         the bolder end of the range
    python parts/gallery.py all          all of the above

Boards go to out/boards/.  One png per board, as always.

The named concepts in CONCEPTS are the point of this file: each one is a
deliberate pairing of palette, silhouette treatment and accent vocabulary,
rather than a sweep of one axis.  A sweep shows what the knobs do; a named
concept shows what the knobs are FOR.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths, spec as S, board as B
import tibia, femur

# every board can be drawn for either part -- the styling system is shared, only
# the source solid and its datums differ
PARTS = {"tibia": tibia, "femur": femur}
PART = tibia

CELL = (1320, 548)

# the geometry settled through round 9: the crossbreed, the axes he chose, and
# the mild rounding he picked in round 8
BASE = dict(mix=("exposed", "armored", 0.5), aggression=7.0, density=1.0,
            relief=1.0, scale=1.5, accent=1.1, outline=0.8, out_steps=2,
            out_sides="both", collar_t=2.0, organic=0.8)

PAL_ORDER = ["arctic", "arctic_lt", "concept", "bone", "precision", "titanium",
             "mono", "amber", "industrial", "desert", "forest", "ocean",
             "nebula", "cyber", "stealth", "concept_d"]

PAL_NOTE = {
    "arctic": "01 · clean / modern / technical", "stealth": "02 · dark / futuristic",
    "industrial": "03 · rugged / durable", "precision": "04 · high contrast / sport",
    "ocean": "05 · calm / premium", "forest": "06 · natural / earthy",
    "nebula": "07 · space / mysterious", "titanium": "08 · premium / sophisticated",
    "desert": "09 · warm / adventurous", "cyber": "10 · cyberpunk / neon",
    "concept": "build-guide sheet palette", "concept_d": "build guide, lighter still",
    "arctic_lt": "01 Arctic, softer structure", "amber": "HUD yellow",
    "mono": "monochrome — accent by value alone", "bone": "warm body, cool structure",
}

ACCENTS = [
    ("spine",     "contour spine + blocks",      dict()),
    ("smooth",    "same path, organised",        dict(accent_channel=2.4)),
    ("axial",     "straight tapered run",        dict(accent_channel=2.4)),
    ("jog",       "straight, one jog",           dict(accent_channel=2.4)),
    ("dual",      "two parallel runs",           dict(accent_channel=2.0)),
    ("dashed",    "even breaks",                 dict(accent_channel=2.0)),
    ("traps",     "elongated trapezoids",        dict()),
    ("hex",       "chamfered lozenges",          dict()),
    ("chev",      "chevrons",                    dict()),
    ("hatch",     "hatch groups",                dict(n_traps=3)),
    ("bracket",   "corner brackets",             dict(n_traps=3)),
    ("ticks",     "rail with ticks",             dict(n_traps=3)),
    ("stepbar",   "stepped bars",                dict(n_traps=3)),
    ("circuit",   "routed trace",                dict(comp_w=4.5)),
    ("circuit",   "trace + hatch",               dict(comp_w=4.5, n_hatch=2)),
    ("led",       "light strip in a channel",    dict(trace_t=2.0, comp_w=7.0)),
    ("jointring", "joint rings only",            dict(n_rings_j=2)),
    ("wrap",      "panel wrapping the end",      dict()),
    ("plate",     "one large chamfered panel",   dict()),
]

# ---------------------------------------------------------------- concepts --
# name, one-line pitch, spec overrides
CONCEPTS = [
    ("GLACIER", "arctic body, softer structure, a routed trace with hatch marks",
     dict(palette="arctic_lt", accent_style="circuit", comp_w=4.5, n_hatch=2,
          trace_t=3.2)),
    ("NIGHTRUN", "stealth black, one cyan light strip sunk in a channel",
     dict(palette="stealth", accent_style="led", trace_t=2.2, comp_w=7.5)),
    ("FOUNDRY", "hard creases, warm grey, safety-orange trapezoids",
     dict(palette="industrial", organic=0.0, outline=1.1, accent_style="traps",
          n_traps=4, trap_skew=12.0)),
    ("APEX", "off-white and race red, chevrons running to the wheel",
     dict(palette="precision", accent_style="chev", n_traps=5, accent=1.3)),
    ("ABYSS", "deep navy and teal, waisted shaft, single light strip",
     dict(palette="ocean", organic=1.4, waist=1.0, accent_style="led",
          trace_t=2.4, comp_w=8.0)),
    ("RECON", "sage and sand, corner brackets, nothing decorative",
     dict(palette="forest", accent_style="bracket", n_traps=3, accent=0.9)),
    ("PULSAR", "violet on cool grey, accent reserved entirely for the joints",
     dict(palette="nebula", accent_style="jointring", n_rings_j=3, accent=1.4)),
    ("BULLION", "titanium and gold, one large chamfered panel",
     dict(palette="titanium", accent_style="plate", plate_h=15.0, accent=1.0)),
    ("DUNE", "desert tan, burnt orange wrapping the wheel end",
     dict(palette="desert", organic=0.0, outline=1.0, accent_style="wrap")),
    ("VICE", "cyber black and neon pink, dense circuit routing",
     dict(palette="cyber", accent_style="circuit", comp_w=5.5, n_jogs=5,
          jog_amp=3.4, jog_w=5.0, n_hatch=2, trace_t=3.4)),
    ("SIGNAL", "HUD amber, rail and ticks, instrument-panel logic",
     dict(palette="amber", accent_style="ticks", n_traps=3, accent=1.2)),
    ("GHOST", "monochrome — the accent carried by value alone",
     dict(palette="mono", accent_style="smooth", accent_channel=3.0, accent=1.4)),
]

WILD = [
    ("OVERDRIVE", "cyber, waisted, chevrons at full aggression",
     dict(palette="cyber", organic=1.6, waist=1.4, accent_style="chev",
          aggression=9.0, n_traps=6, accent=1.6)),
    ("BLACKOUT", "stealth, no outline growth at all — the source, restyled",
     dict(palette="stealth", organic=1.2, outline=0.0, accent_style="jointring",
          n_rings_j=3, density=0.5)),
    ("CIRCUITRY", "arctic, the trace language pushed to its limit",
     dict(palette="concept", accent_style="circuit", n_jogs=6, jog_amp=3.0,
          jog_w=4.5, comp_w=5.5, n_hatch=3, trace_t=2.6)),
    ("MONOLITH", "titanium, heaviest relief, one plate, nothing else",
     dict(palette="titanium", relief=1.7, scale=1.9, accent_style="plate",
          plate_h=20.0, density=0.5)),
    ("HAZARD", "industrial, hatch groups reading as warning stripes",
     dict(palette="industrial", organic=0.0, outline=1.2, accent_style="hatch",
          n_traps=4, accent=1.5)),
    ("SCALPEL", "precision, stripped to the minimum, one thin jog",
     dict(palette="precision", density=0.4, scale=1.2, accent_style="jog",
          accent_channel=2.6, accent=0.8)),
]


def _cells(items, note_of=None):
    out = []
    for title, sub, over in items:
        sp = S.derive(**{**BASE, **over})
        out.append(B.cell(PART.plan(sp), size=CELL, title=title, sub=sub,
                          note=note_of(sp) if note_of else S.label(sp)))
        print(f"   {title}")
    return out


def palettes(out=None):
    items = [(p.upper(), PAL_NOTE.get(p, ""), dict(palette=p,
              accent_style="circuit", comp_w=4.5, n_hatch=2)) for p in PAL_ORDER]
    return B.sheet(_cells(items), 4, os.path.join(paths.BOARDS, out or f"gal_{PART.PART.lower()}_palettes.png"), size=CELL,
                   header=f"{PART.PART.upper()} v4 — palette exploration",
                   sub="the ten from artistic concepts/ plus six of mine · "
                       "same geometry and accent throughout · reply with names or numbers")


def accents(out=None):
    items = [(f"{st.upper()}", sub, dict(accent_style=st, palette="concept", **kw))
             for st, sub, kw in ACCENTS]
    return B.sheet(_cells(items), 4, os.path.join(paths.BOARDS, out or f"gal_{PART.PART.lower()}_accents.png"), size=CELL,
                   header=f"{PART.PART.upper()} v4 — the whole accent vocabulary",
                   sub="every accent style built so far, one palette, one geometry")


def silhouette(out=None):
    items = [
        ("SOURCE", "untouched, for reference", dict(aggression=0.0, organic=0.0, outline=0.0)),
        ("CREASED", "hard trapezoid creases — round 4", dict(organic=0.0, outline=0.8)),
        ("CREASED, DEEPER", "same language, more of it", dict(organic=0.0, outline=1.4)),
        ("ORGANIC, MILD", "corners rounded — round 8 cell 2", dict(organic=0.8)),
        ("ORGANIC, STRONG", "every vertex a tangent curve", dict(organic=1.5)),
        ("WAISTED", "necked shaft, circular bosses", dict(organic=1.4, waist=1.0)),
        ("WAISTED, DEEP", "the concept-art link", dict(organic=1.8, waist=1.7)),
        ("SPARSE", "few, very large features", dict(organic=0.8, density=0.45, scale=1.9)),
        ("DENSE", "many small ones", dict(organic=0.8, density=1.9, scale=0.8)),
    ]
    items = [(a, b, dict(c, palette="concept", accent_style="circuit", comp_w=4.5))
             for a, b, c in items]
    return B.sheet(_cells(items), 3, os.path.join(paths.BOARDS, out or f"gal_{PART.PART.lower()}_silhouette.png"), size=CELL,
                   header=f"{PART.PART.upper()} v4 — silhouette treatments",
                   sub="what the shape axes actually do, held against the untouched source")


def concepts(out=None):
    return B.sheet(_cells(CONCEPTS), 3, os.path.join(paths.BOARDS, out or f"gal_{PART.PART.lower()}_concepts.png"), size=CELL,
                   header=f"{PART.PART.upper()} v4 — twelve resolved concepts",
                   sub="each one a deliberate pairing of palette, silhouette and accent · "
                       "name the ones worth building")


def wild(out=None):
    return B.sheet(_cells(WILD), 3, os.path.join(paths.BOARDS, out or f"gal_{PART.PART.lower()}_wild.png"), size=CELL,
                   header=f"{PART.PART.upper()} v4 — the bolder end",
                   sub="pushed further than anything shown so far")


def legset(out=None):
    """Both parts of the leg, one concept per cell.

    He circled the LEG, not a part.  A concept only really holds if the femur
    and the tibia read as the same design, so this stacks them: femur above,
    tibia below, identical spec."""
    from PIL import Image, ImageDraw
    W, H = CELL
    cells = []
    for title, sub, over in CONCEPTS:
        sp = S.derive(**{**BASE, **over})
        top = B.cell(femur.plan(sp), size=(W, H), title=f"{title}   ·   FEMUR", sub=sub,
                     note=S.label(sp))
        bot = B.cell(tibia.plan(sp), size=(W, H), title=f"{title}   ·   TIBIA", sub=sub,
                     note=S.label(sp))
        im = Image.new("RGB", (W, 2 * H), B.PAGE)
        im.paste(top, (0, 0)); im.paste(bot, (0, H))
        ImageDraw.Draw(im).rectangle([0, 0, W - 1, 2 * H - 1], outline=B.INK, width=2)
        cells.append(im)
        print(f"   {title}")
    return B.sheet(cells, 3, os.path.join(paths.BOARDS, "gal_legset.png"),
                   size=(W, 2 * H),
                   header="WHEELED-LEG ROBOT — twelve concepts, both leg parts",
                   sub="femur above, tibia below, identical spec · "
                       "a concept only holds if the pair reads as one design")


BOARDS = dict(legset=legset, palettes=palettes, accents=accents, silhouette=silhouette,
              concepts=concepts, wild=wild)

if __name__ == "__main__":
    paths.ensure_out()
    args = sys.argv[1:]
    if "--part" in args:
        i = args.index("--part")
        PART = PARTS[args[i + 1].lower()]
        globals()["PART"] = PART
        args = args[:i] + args[i + 2:]
    which = args or ["all"]
    todo = list(BOARDS) if which == ["all"] else which
    for name in todo:
        if name not in BOARDS:
            print(f"?? {name}"); continue
        print(f"== {name}")
        BOARDS[name]()
