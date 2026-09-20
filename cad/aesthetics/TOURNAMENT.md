# Tournament — protocol and current state

Read this first if you are picking up the aesthetics work in a new session.
The copy-paste restart prompt lives at the top of `README.md`.

---

## STATUS (2026-09-20) -- five parts styled, zero collisions

**GLACIER is locked** and now applied to five parts: **Femur, Tibia, Coupler,
Side panel, RobotMount**.  Every one verifies **0 openings changed, worst
deviation 0.000 mm3**, and `lib/collide.py` reports **no new collisions in any
pose**.

```
part         styled     source    grew      flange       white/graphite/accent
Femur        78.56 cm3   86.74    +8.0 Y     785 mm2      70 / 24 /  6 %
Tibia       211.35      228.12   +12.6 Y    1297          69 / 26 /  5
Coupler      65.19       74.27    +6.2 Y     718          62 / 35 /  3
Side panel   89.99       92.01    +7.2 Y     289          57 / 39 /  3
RobotMount  107.07      110.32    +6.0 Y    1023          54 / 44 /  2
```

### Added material must read as STRUCTURE

His verdict on the first build that grew: *"the part doesn't look functional...
it looks fake"* -- the original part with two thin full-depth walls stuck on.
`out_h` is 4.5 mm and the Femur is 39 mm deep, so growth extruded over the full
depth is a **1:9 fin**.  Confined to the outer third it is a **1:2.8 flange**.

Growth is now ONE PRISM: width from `flange_w` (defaults to `out_h`), depth
`grow_depth_frac` of the part's own depth, placed at the show face if it can
attach there and otherwise at the band holding the most metal.  Its plan shape
is measured from a DEEPER sample band than it spans, because a part whose top
face is recessed gives a tiny sample and the ring lands inland.

The build prints the whole funnel, so a failure explains itself:

```
flange @ z  +2.7.. +13.5: ring 5455 -> design 3356 -> clear 1486 -> attached 1297 mm2
```

**The pivots are what is blocked, not the part.**  `scratchpad/freespace.py`
measured it: a 6 mm band outside the silhouette is 85% clear on the Tibia,
41-74% on the others, 20-28% on the Femur.  `bosses` grew AT the pivots, which
is why the only growth that ever survived came through as one lone sliver.

**On three parts the flange lands INBOARD** -- the Coupler, Side panel and
RobotMount have no metal at their show-face perimeter, so it attaches on the
hidden face.  Material is added and it is structural, but it is not seen.
Opening that up means loosening the keep-out (PAD_Z 6 mm, 1.0 mm clearance,
neighbours at maximal styled size), which is margin he chose.

### Guards in build() -- every one caught a real defect

| guard | what it caught |
|---|---|
| envelope removes nothing from the source | 14-21 cm3 per part bevelled off by the chamfer |
| a fuse cannot lose volume | a body whose booleans all lied |
| filament bodies must sum to the part | a 3MF holding 20 cm3 of a 125 cm3 part |
| detached pieces reported WITH their bbox | the Side panel's 4.79 cm3 at Y +75..+81 |
| flange funnel printed | which clip was eating the flange |

---

## Adding a new part — the next phase

The styling system is part-agnostic. A new part needs a recipe, not a redesign:
`lib/spec.py`, `lib/accents.py` (18 accent styles) and `lib/board.py` are shared,
and every accent range is a FRACTION of the host part's own length.

**Copy `parts/femur.py`, not `parts/tibia.py`.** The femur recipe is the general
one: derived datums, a bridged silhouette, and feature bands expressed as
fractions. The tibia recipe still has hard-coded millimetre bands from before the
system was generalised.

1. Drop the STEP in `input/parts/` (`lib/paths.py` also searches the live CAD
   folder, so nothing has to be duplicated).
2. Copy `parts/femur.py` -> `parts/<part>.py`, set `PART`, and find the datums.
   **Derive them, do not assume.** The femur's fell out of its own openings:
   bolt circles at x = ±93.79, 187.58 mm apart, which is `|AC|` from CLAUDE.md.
3. Check the silhouette is ONE polygon. If it is not, `_bridge()` it — see trap 9.
4. Tune the fraction-based bands in `plan()` (`fx(.16)`, `fx(.84)`, ...).
5. `python parts/<part>.py` then `python lib/verify.py <Part>`. It must print
   **0 changed, worst deviation 0.000 mm3** before anything is shown or shipped.
6. Boards come free: add the module to `PARTS` in `parts/gallery.py` and
   `parts/build_concepts.py`.

**The spec is shared, so a new part inherits GLACIER by copying
`specs/tibia.json` to `specs/<part>.json`.** Only `tag` and `concept` need
changing if anything.

---

## Why it is structured this way

An earlier round 1 swept the three axes across nine cells and **he could not choose —
they all looked alike.** The cause: every cell used the same design language, so they
were parameter variations on one idea, with nothing categorically different to pick
between. Hence: language first, axes second.

---

## The rounds

```
python parts/tibia_board.py                 round 1: pick a LANGUAGE        -> one letter
python parts/tibia_board.py A               round 2: axis sweep inside A    -> one cell per row
python parts/tibia_board.py BC              round 2: CROSSBREED B x C       -> one cell per row
python parts/tibia_board.py A 7 1.0 1.0     round 3: mutations              -> one number
python parts/tibia_board.py BC-outline      round 3: silhouette, all sides  -> one number
python parts/tibia_board.py BC-creases      round 4: creases + collars      -> one cell per row
python parts/tibia_board.py rings           round 5: ring motif reuse       -> one number
```

**The axes grew during the Tibia run.** `lib/spec.py` now carries five, not three:

| Axis | Range | What it moves |
|---|---|---|
| `aggression` | 0–10 | how far the part departs from the source |
| `density` | 0–2 | how MANY features |
| `relief` | 0–2 | how TALL / DEEP they are |
| `scale` | 0–2 | how BIG each feature is in plan — **count held** |
| `outline` | 0–2 | trapezoidal growth of the SILHOUETTE, every side |
| `accent` | 0–2 | how much blue, and how big the blocks on the spine |

`scale` and `outline` both exist because the stock three could not express real
notes: *"the count is right, the size could be bigger"* is not density, and
*"you're doing it only in one direction"* is not aggression. `outline` defaults
to **0.0** so any spec locked before it existed still reproduces exactly.

`derive(mix=("exposed","armored",0.5))` crossbreeds two languages. Languages are
plain dicts of numbers, so a blend is a weighted average — "something between B
and C" is one argument, not a new language.

| Round | Board | He replies |
|---|---|---|
| 1 | 5 languages at aggression 8, plus the source | one letter, A–E |
| 2 | rows = aggression / density / relief, cols = settings | one cell per row, `1C 2A 3B` |
| 3 | 6 mutations around that point, cell 1 = winner unchanged | one number, plus red marks |
| 4 | narrower round 3, only if it has not settled | one number |

Letters map **A=angular, B=exposed, C=armored, D=curved, E=greebled** — explicitly,
because "angular" and "armored" both start with A.

Then: write `specs/tibia.json`, run `parts/tibia.py`, run `lib/verify.py Tibia`,
render, ship the 3MF.

## Log every answer — this is not optional

After he answers a board, record it before doing anything else:

```
python lib/logdec.py out/boards/tibia_round1_languages.png        "round 1 - language" "A" "liked the facets, wants the keel busier"
```

That archives the board into `decisions/` and appends to `DECISIONS.md`, so the
history survives. Boards in `out/boards/` get overwritten when a round is re-run —
the archived copy is the permanent record. He asked for this explicitly so he can
look back at what he chose and why.

## Rules that make this cheap

- **One png per round. Never N images.** The cost is not generating pictures, it is
  *reading* them. He is the fitness function; the model does not need to look.
  Spot-check a board yourself maybe every third round, not every round.
- A board is ~4 s and ~1–2k tokens. A 3D build is ~60 s and much more. **Only build
  3D after a spec is approved.**
- Always keep the current winner as an unchanged cell, so a round cannot go backwards.
- He can reply with two letters (crossbreed) or "B but with C's end blocks" —
  languages are just parameter dicts, so mixing is trivial.

## His feedback channel

**Red marks on a render are by far the highest-bandwidth input he gives.** He has
said plainly he lacks art vocabulary; annotated screenshots have moved the design
more than any text. Invite them every round. His main verbal lever is
"more aggressive", which is why the aggression axis is anchored 0–10 with 10 shown.

---

## Decisions already locked

| | |
|---|---|
| Palette | **01 Arctic** — white `#F2F4F7` / graphite `#2B2F36` / blue `#1E7BFF` |
| Colour balance | ~50/50 white/graphite, blue **under 1%** — he said there was too much blue |
| Printer | Bambu **X2D, dual nozzle**. Two colours swap by nozzle; a third purges, so blue stays in a narrow Z band |
| Deliverables | print-ready 3MF **and** a fused styled STEP as a SolidWorks reference |
| Scope | silhouette reshaping allowed; stiffness explicitly not a concern |
| Mechanical master | SolidWorks. This layer is downstream and re-runnable |
| Tibia | **Settled.** See STATUS for the numbers; `specs/tibia.json` is the record |
| Blue quantity | Raised on his instruction — "increase a little bit more on the blue accents". The built Tibia is **5.3% blue by volume**, over the old ≤1% rule. Flagged to him; he has not yet said whether to pull it back |
| Blue placement | One continuous spine following the part's own contour, blocks strung along it, terminating in the wheel ring. Replaces scattered slivers, which he read as "random blue lines here and there" |

## Deferred on purpose

**Colour balance as a tunable ("more blue than white") comes after the geometry is
settled** — it repaints, it does not relocate, and it can only be judged once the
geometry is fixed. Add a `white_share` dial next to `palette` then, as its own quick
board. Spec loading merges over defaults, so a spec locked today survives new fields.

More languages can be added anytime — a language is ~20 lines in `spec.LANGUAGES`.
Don't add more unless none of the five land; five was already at the edge of
comfortable choosing.

## Phase 4 -- DONE (2026-09-19)

The assembly STEPs arrived as three poses of the half robot, in
`cad/v4 Larger Ball bearings/STEP exports/`: **Retracted, Middle single part,
Extended**.  All three preserve structure (41 products, 98 solids) and read in
about 20 s.  `Middle multiple parts/` alongside them is one STEP per assembly
node -- that is where `Femur.STEP` and `Tibia.STEP` now resolve from.

**The show face is settled, and it is NOT uniformly +Z.**  Walking the
transforms: the hip axis is global Z, and increasing global Z is OUTBOARD (the
wheel is outermost at Z 169..203, RobotMount innermost at 80..99).

```
Tibia        local +Z -> global +Z     show face local +Z    (as built)
Side panel   local +Z -> global +Z     show face local +Z
RobotMount   local +Z -> global +Z     show face local +Z
Femur        local +Z -> global -Z     show face local -Z    MIRRORED
Coupler      local +Z -> global -Z     show face local -Z    MIRRORED
```

`show_face: "-Z"` reflects the source about XY on the way in and reflects the
finished bodies back on the way out, so features land on the right side while
every hole returns exactly where it started.

**Two STEPs were lost to a Windows case collision** and had to be recovered by
XCAF extraction from `Extended.STEP` into `input/parts/`: SolidWorks wrote the
ASSEMBLY nodes `COUPLER.STEP` and `SIDE PANEL.STEP` (0 faces, structure only),
and those filenames blocked the real `Coupler` and `Side panel` parts.
`SwicthMount`/`Switch_Mount` escaped only because the spelling differs.  If the
export is ever redone, rename the assembly nodes or export to separate folders.

---

## Collisions -- prevented, not just checked

`lib/collide.py` measures every pair TWICE, once with the source part and once
with the styled part, and reports only the increase, so nominally-touching CAD
(a bearing in its seat, a bolt in its counterbore) does not read as a false
positive.  It found **25 new collisions**, worst 12 969 mm3 Femur x Tibia.

`lib/asmkeepout.py` prevents them instead: for each part, in each pose, every
other component is transformed into that part's local frame, clipped to a z
band, projected into plan and unioned.  `plan()` subtracts the result.

**Growth is LAYERED through the thickness** (`grow_layers`, default 1).  A
single full-thickness prism cannot grow at all where the links overlap in plan,
because they clear each other by interleaving in DEPTH, not in plan -- one
contested region blocked the whole 39 mm and took 83% of the Femur's growth.
Per-layer clipping puts material where the space actually is; on the Femur the
show-face layer is the least blocked (32 069 mm2 against 39 016 on the far side).

Neighbours are taken from their STYLED exports where those exist, because a
neighbour that is also being styled will grow too.  **Re-run `asmkeepout` after
the parts change** -- the caches go stale, which is what left an 847 mm3
Tibia x Coupler residue after the envelope fix made parts locally bigger.

## Traps already paid for — do not rediscover these

1. `extrude(face, amount=h, taper=d)` returns z in **[-h, 0] with the wide end at
   z=0**. Mirror it before using it as a raised pad, or every boss comes out inverted
   and buried in the plate. `shputil.frustum()` handles it.
2. **The top plan face is not the silhouette.** On the Tibia it stops at x=204.7
   because the wheel hub is recessed. Use `keepout.silhouette()`, or added material
   fills that pocket and destroys the hub counterbores.
3. **Never re-cut the source solid.** Add material only *outside* the original
   silhouette. Cutting every opening through the full depth turns counterbored
   bearing seats into straight through-holes. `lib/verify.py` catches this: it must
   always print **0 openings changed, worst deviation 0.000 mm³**.
4. Accents and collars **must be clipped to the part**. Unclipped, they float outside
   the silhouette and the board stops being a contract.
5. PIL has no polygon-with-holes. `board._paint` masks; `board._shade` marks a blind
   recess. Painting a blind pocket in the ground colour claims it goes through.
6. Chained `+` on build123d Parts can silently drop terms. Use `shputil.union()`.
7. **A filament body is SEVERAL solids.** `import_step(f).solids()[0]` takes one
   fragment -- it once rendered a part with almost no accent while the build was
   perfectly fine. Tessellate the whole compound and cross-check the rendered
   volume against what `build()` printed.
8. **A tapered extrude can fail on a valid outline.** Organic mode's round-join
   buffers leave sub-millimetre edges; offsetting them by the 4.4 mm chamfer
   throws `BRepFill_OffsetWire::FixHoles(): Wrong wire`. Simplify to 0.5 mm
   (invisible at this size). The chamfer now falls back to a square edge rather
   than losing the whole build.
9. **`silhouette()` unions horizontal faces only.** A part with a sloped
   transition and no horizontal face in it comes back as SEVERAL disjoint
   pieces -- the Femur does. Taking the largest, as the Tibia recipe does, drops
   the rest, and because the envelope INTERSECTS the solid that silently cuts
   those regions off the part. `femur._bridge()` closes them first.
10. **Accent ranges must be fractions of the host part's length.** Nine absolute
   Tibia x-coordinates in the accent code put every femur accent in a heap
   against one end. `lib/accents.py` is now part-relative throughout.
11. **A cut that stops short of the surface leaves a razor.** The top notches used
   to stop 0.3 mm below the outline. Against a curved edge that was invisible;
   against the flat plateau `outline` introduced it became a **0.4° spike**, and
   OCC's 45° chamfer taper dies on it with a bare `Standard_TypeMismatch:
   TopoDS::Solid` that names nothing. Cuts now run 2.5 mm clear of the surface,
   and `grown` is morphologically opened and stripped of interior rings before
   it is extruded. If a taper ever fails again, check the minimum corner angle
   of `grown` first — it is almost always this.
8. **A filament body is usually SEVERAL solids.** The accent came out as 5. Any
   code doing `import_step(f).solids()[0]` silently renders or measures one
   fragment — for the accent that was the 0.00 cm3 piece, and the render showed
   a part with almost no blue on it while the build was perfectly fine. Pass the
   whole compound to `tessellate()`. Cross-check the render's reported volume
   against what `build()` printed; they must agree.

12. **An absolute millimetre in a spec does not transfer between parts.** Paid
    for in all three axes now. X: nine hard-coded Tibia x-coordinates put every
    Femur accent in a heap against one end (trap 10). Z: `split_z = -5.0` sits
    18.5 mm below the Tibia's show face, the full 39 mm on the Femur and only
    10 mm on the Coupler, which came out 29% white against the Femur's 61%.
    Y: `frame`/`rail` used bounds of -6.0 and +5.0, tuned on a 38.8 mm tall
    link, which land near mid-height on a 104 mm plate. **Express every band as
    a fraction of the host part's own extent.** When the target is a VOLUME
    rather than a position, state the target and SOLVE for the geometry --
    `white_share` bisects for the split plane, because depth through a part does
    not map to volume when the cross-section varies (`white_depth` 0.72 gave the
    Coupler 42% and the Femur 74%).
13. **The envelope must bound the ADDITION, not be intersected with the whole
    body.** `union(solid, add) & env` lets the 4.4 mm chamfer bevel the SOURCE
    wherever growth is thinner than the chamfer: 14-21 cm3 per part, and the
    Coupler was NET BIGGER while losing a fifth of its original metal, so a
    volume delta hides it completely. Write `union(solid, add & env)`. `build()`
    now asserts the envelope removes nothing from the source.
14. **`verify.py`'s openings test does not prove material was not removed.**
    Every one of those builds reported 0 openings changed, 0.000 mm3, because no
    bore happened to sit where the bevel ran. It now also reports total source
    material removed -- but read that number carefully: MOST of it is deliberate
    (pockets, windows, full-depth trapezoid cutouts, side-wall pockets and the
    accent engraving are the approved look). The envelope-only figure printed by
    `build()` is the one that must be zero.
15. **Apply the keep-out AFTER `simplify()`, not before.** `simplify(tol)` moves
    a boundary by up to `tol`, and where a keep-out cut runs along the source
    edge it shaves into it -- 0.00 mm2 lost without the clip, 25 mm2 with it.
    Cut last, union OUT back last. The guard that `grown` still contains OUT is
    area-based, not `contains`: GEOS leaves sub-mm2 residue on shared boundaries
    and `contains` fails even at 0.00 mm2 lost.
16. **Unioning tessellation triangles throws `TopologyException: found non-noded
    intersection`.** Real CAD meshes carry slivers. Snap coordinates, drop
    degenerate triangles, retry buffered -- and when that still fails, fall back
    to a convex hull or bounding box. A keep-out that silently returns empty does
    not throw; it just lets the part grow into its neighbour, and that surfaces
    as two printed parts that will not fit.
17. **A mirror transforms a physical region by REFLECTION, not translation.**
    Translating the absolute-z colour bands to follow `show_face` moved them to
    the far face and took 7.6 cm3 of extra material with them. Better still,
    measure everything from ZT so the mirror needs no fix-up at all.
18. **`asmkeepout` reads a 68 MB assembly per pose.** Read each pose ONCE and
    tessellate each neighbour ONCE, then transform vertices per part. Per-part
    reading does not scale past about two parts.
