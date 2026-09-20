r"""Tibia design boards -- all tournament rounds.

    python parts/tibia_board.py                  round 1: pick a LANGUAGE
    python parts/tibia_board.py A                round 2: axis sweep inside A
    python parts/tibia_board.py A 8 1.0 1.0      round 3: mutations around that point

Round 1 settles the categorical choice (which design idea).  Only then do the
axes mean anything -- "density 8" is a different thing in ARMORED than in
GREEBLED, so sweeping axes before the language is picked just produces nine
variants that all look alike.

Boards go to out/boards/ as ONE png per round.  Never emit N separate images:
the whole point is that the reviewer looks, and the model does not have to.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths, spec as S, board as B
import tibia

LANGS = ["angular", "exposed", "armored", "curved", "greebled"]
# explicit: "angular" and "armored" both start with A, so first-letter
# derivation silently maps A to the wrong language
LETTER = dict(zip("ABCDE", LANGS))                 # A..E -> language name
COLS = "ABC"


def round1(out="tibia_round1_languages.png"):
    cells = []
    for i, lang in enumerate(LANGS):
        sp = S.derive(language=lang, aggression=8.0)
        cells.append(B.cell(tibia.plan(sp), title=f"{chr(65+i)}  ·  {lang.upper()}",
                            sub=sp["blurb"], note=S.label(sp)))
        print(f"  {chr(65+i)}  {lang}")
    sp = S.derive(language="angular", aggression=0.0)
    cells.append(B.cell(tibia.plan(sp), title="F  ·  SOURCE",
                        sub="the part as it is today, untouched", note=S.label(sp)))
    return B.sheet(cells, 3, os.path.join(paths.BOARDS, out),
                   header="TIBIA v4 — round 1: pick a DESIGN LANGUAGE",
                   sub="all at aggression 8 so the differences are obvious · "
                       "F is the untouched source · reply with one letter")


def round2(lang, out=None):
    rows = [("AGGRESSION", "how far it departs from the source",
             [dict(aggression=a) for a in (4, 7, 10)], ["4 restrained", "7 strong", "10 maximum"]),
            ("DENSITY", "how MANY features",
             [dict(density=d) for d in (0.5, 1.0, 1.8)], ["sparse", "medium", "dense"]),
            ("RELIEF", "how TALL / DEEP those features are",
             [dict(relief=r) for r in (0.4, 1.0, 1.8)], ["shallow", "medium", "deep"])]
    base = dict(language=lang, aggression=7.0, density=1.0, relief=1.0)
    cells = []
    for r, (name, blurb, variants, capt) in enumerate(rows, start=1):
        for c, over in enumerate(variants):
            sp = S.derive(**{**base, **over})
            cells.append(B.cell(tibia.plan(sp), title=f"{r}{COLS[c]}  ·  {name}  ·  {capt[c]}",
                                sub=blurb, note=S.label(sp)))
            print(f"  {r}{COLS[c]}  {S.label(sp)}")
    out = out or f"tibia_round2_{lang}_axes.png"
    return B.sheet(cells, 3, os.path.join(paths.BOARDS, out),
                   header=f"TIBIA v4 — round 2: tune the axes inside {lang.upper()}",
                   sub="rows = axes, columns = settings, everything else held · "
                       "reply with ONE CELL PER ROW, e.g. 1C 2A 3B")


def round2_mix(la, lb, out=None):
    """Round 2 for a CROSSBREED.

    The stock round 2 sweeps aggression / density / relief.  When the note is
    "the count is right, the size should be bigger", sweeping density spends
    three cells on the one axis already known to be correct -- so this board
    sweeps the axes the feedback actually named: where between the two parent
    languages, how BIG each feature is, and how much blue."""
    base = dict(aggression=7.0, density=1.0, relief=1.0, scale=1.45, accent=1.4)
    rows = [("BLEND", f"how far between {la.upper()} and {lb.upper()}",
             [dict(mix=(la, lb, t)) for t in (0.3, 0.5, 0.7)],
             [f"toward {la.upper()}", "even split", f"toward {lb.upper()}"]),
            ("FEATURE SIZE", "how BIG each feature is -- the COUNT is held fixed",
             [dict(mix=(la, lb, 0.5), scale=v) for v in (1.0, 1.5, 2.0)],
             ["as shown in round 1", "bigger", "biggest"]),
            ("BLUE", "how much accent, and how big the blocks on the spine are",
             [dict(mix=(la, lb, 0.5), accent=v) for v in (0.8, 1.4, 2.0)],
             ["restrained", "more", "most"])]
    cells = []
    for r, (name, blurb, variants, capt) in enumerate(rows, start=1):
        for c, over in enumerate(variants):
            sp = S.derive(**{**base, **over})
            cells.append(B.cell(tibia.plan(sp), title=f"{r}{COLS[c]}  ·  {name}  ·  {capt[c]}",
                                sub=blurb, note=S.label(sp)))
            print(f"  {r}{COLS[c]}  {S.label(sp)}")
    out = out or f"tibia_round2_{la[:3]}_{lb[:3]}_mix.png"
    return B.sheet(cells, 3, os.path.join(paths.BOARDS, out),
                   header=f"TIBIA v4 — round 2: {la.upper()} × {lb.upper()} crossbreed",
                   sub="blue is now ONE spine following the part's contour, with blocks on it · "
                       "rows = axes, columns = settings · reply ONE CELL PER ROW, e.g. 1C 2A 3B")


def round3_outline(la, lb, out=None):
    """Round 3: how far the SILHOUETTE departs, all the way around.

    Cell 1 is the round-2 winner untouched, so the round cannot go backwards.
    Every other cell raises `outline`, which grows the upper edge and the ends
    as well as the keel.  Adding material outside the source silhouette never
    approaches a bore, so the circles stay exactly as they are."""
    base = dict(mix=(la, lb, 0.5), aggression=7.0, density=1.0, relief=1.0,
                scale=1.5, accent=1.1)
    muts = [("1  round-2 winner, unchanged",  dict(outline=0.0)),
            ("2  upper edge, subtle",         dict(outline=0.6)),
            ("3  upper edge + ends, strong",  dict(outline=1.2)),
            ("4  all round, maximum",         dict(outline=1.8)),
            ("5  strong outline + even larger features", dict(outline=1.2, scale=1.9)),
            ("6  maximum outline + more aggression",     dict(outline=1.8, aggression=9.0))]
    cells = []
    for title, over in muts:
        sp = S.derive(**{**base, **over})
        cells.append(B.cell(tibia.plan(sp), title=title,
                            sub="blend 0.50 · size 1.50 · blue 1.10 unless noted",
                            note=S.label(sp)))
        print(f"  {title:44s} {S.label(sp)}")
    out = out or f"tibia_round3_{la[:3]}_{lb[:3]}_outline.png"
    return B.sheet(cells, 3, os.path.join(paths.BOARDS, out),
                   header="TIBIA v4 — round 3: change the OUTLINE on every side",
                   sub="cell 1 is the round-2 winner, unchanged · growth is now upper edge "
                       "and ends, not just the keel · reply with one number, plus red marks")


def round4_creases(la, lb, out=None):
    """Round 4: trapezoidal creases, and non-circular features at the bores.

    Two independent notes came back together -- the outline growth was the right
    idea but too round, and the collars were fat circles -- so each gets its own
    row and is varied against the approved base with nothing else moving."""
    base = dict(mix=(la, lb, 0.5), aggression=7.0, density=1.0, relief=1.0,
                scale=1.5, accent=1.1, outline=0.0)
    rows = [("CREASED OUTLINE", "straight plateaus, sharp vertices -- no curves",
             [dict(outline=0.8, out_steps=1, out_sides="top"),
              dict(outline=0.8, out_steps=2, out_sides="both"),
              dict(outline=1.4, out_steps=2, out_sides="both")],
             ["one plateau, top only", "stepped, top + bottom", "stepped, deeper"]),
            ("CIRCLES", "the collars at the E / C / W bores",
             [dict(collar_t=2.0),
              dict(collar_t=2.0, collar_seg=2),
              dict(collar_t=2.0, collar_seg=2, collar_tabs=4)],
             ["slimmer, still round", "octagonal - straight edges", "octagonal + square tabs"])]
    cells = []
    for r, (name, blurb, variants, capt) in enumerate(rows, start=1):
        for c, over in enumerate(variants):
            sp = S.derive(**{**base, **over})
            cells.append(B.cell(tibia.plan(sp), title=f"{r}{COLS[c]}  ·  {name}  ·  {capt[c]}",
                                sub=blurb, note=S.label(sp)))
            print(f"  {r}{COLS[c]}  {capt[c]}")
    out = out or f"tibia_round4_{la[:3]}_{lb[:3]}_creases.png"
    return B.sheet(cells, 3, os.path.join(paths.BOARDS, out),
                   header="TIBIA v4 — round 4: trapezoidal creases, and the circles",
                   sub="row 1 varies the OUTLINE only · row 2 varies the COLLARS only · "
                       "everything else is your round-3 cell 1 · reply ONE CELL PER ROW, e.g. 1B 2C")


# everything settled through round 4: blend, axes, creases, slim round collars
WIN = dict(mix=("exposed", "armored", 0.5), aggression=7.0, density=1.0, relief=1.0,
           scale=1.5, accent=1.1, outline=0.8, out_steps=2, out_sides="both",
           collar_t=2.0)


def round5_rings(out=None):
    """Round 5: where else the thin-ring motif goes.

    Everything else is locked to the round-4 winner (1B creases + 2A slim round
    collars).  Cell 1 carries no extra rings, so "1" means the motif stays at
    the bores only and the part is done."""
    muts = [("1  no extra rings — bores only",   dict()),
            ("2  three large rings, mid-body",   dict(n_rings=3, ring_r=11.0)),
            ("3  five medium rings, mid-body",   dict(n_rings=5, ring_r=8.0)),
            ("4  seven small rings",             dict(n_rings=7, ring_r=6.0)),
            ("5  five rings along the upper edge", dict(n_rings=5, ring_r=7.5, ring_band="upper")),
            ("6  mixed sizes, staggered",        dict(n_rings=6, ring_r=7.0, ring_band="mixed"))]
    cells = []
    for title, over in muts:
        sp = S.derive(**{**WIN, **over})
        cells.append(B.cell(tibia.plan(sp), title=title,
                            sub="1B creases + 2A slim round collars, locked · rings vary only",
                            note=S.label(sp)))
        print(f"  {title}")
    out = out or "tibia_round5_rings.png"
    return B.sheet(cells, 3, os.path.join(paths.BOARDS, out),
                   header="TIBIA v4 — round 5: the thin-ring motif, reused",
                   sub="cell 1 keeps rings at the bores only · reply with one number, plus red marks")


SHELL = dict(language="shell", aggression=7.0, density=1.0, relief=1.0, scale=1.5,
             accent=1.1, outline=0.8, out_steps=2, out_sides="both", collar_t=2.0,
             accent_style="axial")


def round6_shell(out=None):
    """Round 6: match the concept art.

    Cell 1 is the built part as locked, for comparison.  The rest swap in the
    SHELL language (few, very large panels) and run the accent as ONE tapered
    stripe down the limb's midline instead of a broken band around the
    perimeter.  Cells 3-6 border that stripe in graphite, which is what stops
    the blue reading as paint dropped onto white."""
    muts = [("1  as built — where we are now", dict(WIN=True)),
            ("2  SHELL + one axial stripe",          dict()),
            ("3  + graphite channel around the stripe", dict(accent_channel=2.4)),
            ("4  thin line in a broad dark channel", dict(accent_channel=4.2, axial_w0=2.6, axial_w1=1.0)),
            ("5  + channel, bolder stripe",          dict(accent_channel=3.0, axial_w0=7.0, axial_w1=2.4)),
            ("6  + channel, panels stripped right back", dict(accent_channel=2.4, density=0.45))]
    cells = []
    for title, over in muts:
        if over.pop("WIN", False):
            sp = S.derive(**WIN)
            sub = "exp+arm 0.50 · perimeter spine + blocks · the locked spec"
        else:
            sp = S.derive(**{**SHELL, **over})
            sub = "SHELL · " + sp["blurb"]
        cells.append(B.cell(tibia.plan(sp), title=title, sub=sub, note=S.label(sp)))
        print(f"  {title}")
    out = out or "tibia_round6_shell.png"
    return B.sheet(cells, 3, os.path.join(paths.BOARDS, out),
                   header="TIBIA v4 — round 6: match the concept art",
                   sub="one large panel, one flowing stripe along the limb, blue bordered by "
                       "graphite · cell 1 is the part as built · reply with one number")


def round7_lines(out=None):
    """Round 7: different KINDS of accent line.

    Geometry is the locked build throughout -- only the line changes.  Cell 1 is
    the line as built, for comparison; every other cell is organised (the crease
    steps taken out of the path, not out of the part) and bordered by the
    graphite channel."""
    CH = 2.4
    muts = [("1  as built — contour spine + blocks", dict(accent_style="spine")),
            ("2  same path, organised + channel",    dict(accent_style="smooth", accent_channel=CH)),
            ("3  straight run + channel",            dict(accent_style="axial", accent_channel=CH)),
            ("4  two thin parallel lines + channel", dict(accent_style="dual", accent_channel=CH)),
            ("5  dashed — one path, even breaks",    dict(accent_style="dashed", accent_channel=CH)),
            ("6  straight with one deliberate jog",  dict(accent_style="jog", accent_channel=CH))]
    cells = []
    for title, over in muts:
        sp = S.derive(**{**WIN, **over})
        cells.append(B.cell(tibia.plan(sp), title=title,
                            sub="locked geometry · only the accent line changes",
                            note=S.label(sp)))
        print(f"  {title}")
    out = out or "tibia_round7_lines.png"
    return B.sheet(cells, 3, os.path.join(paths.BOARDS, out),
                   header="TIBIA v4 — round 7: what KIND of line",
                   sub="cell 1 is the line as built · 2-6 are organised and bordered in graphite · "
                       "reply with one number, plus red marks")


def round8_organic(out=None):
    """Round 8: the concept art's organic, waisted link.

    `artistic concepts/` shows the tibia's analogue ("Leg Link (Lower)") as a
    WAISTED bar -- circular boss at each end, necked-in shaft between, every
    transition a tangent curve.  Cell 1 is the locked hard-creased part so the
    two languages can be compared directly."""
    ST = dict(accent_style="axial", accent_channel=2.6, axial_w0=5.4, axial_w1=1.4)
    muts = [("1  locked — hard trapezoid creases",  dict()),
            ("2  organic — corners rounded off",    dict(organic=0.8, accent_style="smooth",
                                                         accent_channel=2.6)),
            ("3  organic, stronger",                dict(organic=1.4, accent_style="smooth",
                                                         accent_channel=2.6)),
            ("4  organic + waisted shaft",          dict(organic=1.4, waist=1.0,
                                                         accent_style="smooth", accent_channel=2.6)),
            ("5  waisted + tapered stripe (the reference)",
             dict(organic=1.4, waist=1.0, **ST)),
            ("6  fully organic, deepest waist",     dict(organic=2.0, waist=1.6, **ST))]
    cells = []
    for title, over in muts:
        sp = S.derive(**{**WIN, **over})
        cells.append(B.cell(tibia.plan(sp), title=title,
                            sub="circular bosses, necked shaft, tangent transitions",
                            note=S.label(sp)))
        print(f"  {title}")
    out = out or "tibia_round8_organic.png"
    return B.sheet(cells, 3, os.path.join(paths.BOARDS, out),
                   header="TIBIA v4 — round 8: the organic, waisted link",
                   sub="from artistic concepts/ — 'Leg Link (Lower)' · cell 1 is the locked "
                       "hard-creased part · reply with one number, plus red marks")


def round9_traps(out=None):
    """Round 9: the accent as elongated TRAPEZOIDS, not a stroke.

    Base is round 8 cell 2 -- organic 0.8, the mild rounding he picked, no
    waist -- and the lighter concept palette throughout, so the only variable
    across cells is the shape and arrangement of the accent."""
    BASE = dict(WIN, organic=0.8, palette="concept")
    muts = [("1  the line, for comparison",         dict(accent_style="smooth", accent_channel=2.6)),
            ("2  four elongated trapezoids",        dict(accent_style="traps")),
            ("3  six, shorter",                     dict(accent_style="traps", n_traps=6)),
            ("4  graduated — long at the knee, short at the wheel",
             dict(accent_style="traps", n_traps=5, trap_grad=0.55)),
            ("5  four, bordered in grey",           dict(accent_style="traps", accent_channel=2.4)),
            ("6  graduated + bordered, harder skew",
             dict(accent_style="traps", n_traps=5, trap_grad=0.55, trap_skew=15.0,
                  accent_channel=2.4))]
    cells = []
    for title, over in muts:
        sp = S.derive(**{**BASE, **over})
        cells.append(B.cell(tibia.plan(sp), title=title,
                            sub="organic 0.80 · concept palette — light grey, not near-black",
                            note=S.label(sp)))
        print(f"  {title}")
    out = out or "tibia_round9_traps.png"
    return B.sheet(cells, 3, os.path.join(paths.BOARDS, out),
                   header="TIBIA v4 — round 9: the accent as elongated trapezoids",
                   sub="same vocabulary as the cutouts · one shared axis and skew · "
                       "lighter palette throughout · reply with one number, plus red marks")


def round10_hud(out=None):
    """Round 10: HUD accent vocabulary from the reference sheets.

    Same base and same axis as round 9 -- only the SHAPE of the accent element
    changes, so the comparison is clean.  Cell 1 is the round-9 trapezoid row."""
    BASE = dict(WIN, organic=0.8, palette="concept")
    muts = [("1  trapezoids — round 9, for comparison", dict(accent_style="traps")),
            ("2  chamfered lozenges",                   dict(accent_style="hex")),
            ("3  chevrons",                             dict(accent_style="chev")),
            ("4  hatch groups",                         dict(accent_style="hatch", n_traps=3)),
            ("5  corner brackets",                      dict(accent_style="bracket", n_traps=3)),
            ("6  rail with ticks",                      dict(accent_style="ticks", n_traps=3))]
    cells = []
    for title, over in muts:
        sp = S.derive(**{**BASE, **over})
        cells.append(B.cell(tibia.plan(sp), title=title,
                            sub="open shapes, one shared axis · organic 0.80 · concept palette",
                            note=S.label(sp)))
        print(f"  {title}")
    out = out or "tibia_round10_hud.png"
    return B.sheet(cells, 3, os.path.join(paths.BOARDS, out),
                   header="TIBIA v4 — round 10: HUD accent shapes",
                   sub="from the reference sheets · no closed frames · "
                       "reply with one number, plus red marks")


def round11_circuit(out=None):
    """Round 11: the accent as a routed circuit trace."""
    BASE = dict(WIN, organic=0.8, palette="concept", accent_style="circuit")
    muts = [("1  trace, three trapezoidal jogs",      dict()),
            ("2  trace + grey trace alongside it",    dict(comp_w=4.5)),
            ("3  + hatch groups on the run",          dict(comp_w=4.5, n_hatch=2)),
            ("4  more jogs, tighter steps",           dict(comp_w=4.5, n_jogs=5,
                                                          jog_amp=3.4, jog_w=5.0)),
            ("5  heavier grey, thin blue",            dict(comp_w=7.0, trace_t=2.0,
                                                          n_hatch=2)),
            ("6  bold blue, long runs, two jogs",     dict(comp_w=4.0, trace_t=4.6,
                                                          n_jogs=2, jog_amp=6.0))]
    cells = []
    for title, over in muts:
        sp = S.derive(**{**BASE, **over})
        cells.append(B.cell(tibia.plan(sp), title=title,
                            sub="one routed path · straight runs, occasional level change",
                            note=S.label(sp)))
        print(f"  {title}")
    out = out or "tibia_round11_circuit.png"
    return B.sheet(cells, 3, os.path.join(paths.BOARDS, out),
                   header="TIBIA v4 — round 11: circuit-trace accent",
                   sub="blue runs alongside the grey trace at some points · "
                       "reply with one number, plus red marks")


def round3(lang, agg, dens, rel, out=None):
    # elite first: the current winner unchanged, so a round can never go backwards
    muts = [("1  current winner", dict()),
            ("2  bolder",         dict(aggression=min(10, agg + 1.8))),
            ("3  calmer",         dict(aggression=max(0, agg - 1.8))),
            ("4  busier",         dict(density=dens * 1.4)),
            ("5  quieter",        dict(density=dens * 0.65)),
            ("6  deeper relief",  dict(relief=rel * 1.45))]
    base = dict(language=lang, aggression=agg, density=dens, relief=rel)
    cells = []
    for title, over in muts:
        sp = S.derive(**{**base, **over})
        cells.append(B.cell(tibia.plan(sp), title=title,
                            sub=f"{lang.upper()} · {sp['blurb']}", note=S.label(sp)))
        print(f"  {title:20s} {S.label(sp)}")
    out = out or f"tibia_round3_{lang}.png"
    return B.sheet(cells, 3, os.path.join(paths.BOARDS, out),
                   header=f"TIBIA v4 — round 3: mutations around {lang.upper()} "
                          f"agg {agg:g} / dens {dens:g} / rel {rel:g}",
                   sub="cell 1 is the current winner, unchanged · "
                       "reply with one number, plus red marks on anything")


if __name__ == "__main__":
    paths.ensure_out()
    a = sys.argv[1:]
    if not a:
        round1()
    elif len(a) == 1:
        key = a[0].upper()
        if len(key) == 2 and all(k in LETTER for k in key):   # e.g. "BC" -> crossbreed
            round2_mix(LETTER[key[0]], LETTER[key[1]])
        elif key.endswith("-OUTLINE") and all(k in LETTER for k in key[:2]):
            round3_outline(LETTER[key[0]], LETTER[key[1]])
        elif key.endswith("-CREASES") and all(k in LETTER for k in key[:2]):
            round4_creases(LETTER[key[0]], LETTER[key[1]])
        elif key == "RINGS":
            round5_rings()
        elif key == "SHELL":
            round6_shell()
        elif key == "LINES":
            round7_lines()
        elif key == "ORGANIC":
            round8_organic()
        elif key == "TRAPS":
            round9_traps()
        elif key == "HUD":
            round10_hud()
        elif key == "CIRCUIT":
            round11_circuit()
        else:
            round2(LETTER.get(key, a[0].lower()))
    elif len(a) == 4:
        key = a[0].upper()
        round3(LETTER.get(key, a[0].lower()), float(a[1]), float(a[2]), float(a[3]))
    else:
        sys.exit(__doc__)
