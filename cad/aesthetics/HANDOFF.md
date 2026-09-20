# Handoff — copy the prompt below to continue in a new session

`README.md` and `TOURNAMENT.md` both point here. This file is the current
restart prompt plus everything a new session needs. `TOURNAMENT.md` holds the
locked look; `DECISIONS.md` is every decision Fernando made, newest at the
bottom.

**Phase 1 — outputs trustworthy in SolidWorks — DONE, 2026-09-20.**
**Phase 2 — the add/remove map and the free-space measurement — DONE, 2026-09-20.**
**Phase 3 — THE REBUILD — is the current work and has not been started.**

---

## The prompt

```
Continue the wheeled-leg-robot aesthetics work.  You are starting PHASE 3: the
rebuild of the three LINKS.

READ FIRST, in this order:
  1. cad/aesthetics/HANDOFF.md     — this file
  2. cad/aesthetics/TOURNAMENT.md  — the locked look, and the trap list
  3. cad/aesthetics/DECISIONS.md   — every decision, newest at the bottom

THE LOOK IS LOCKED. GLACIER, in specs/*.json. Do not restart the tournament,
do not offer new palettes or accent styles, do not change the spec unless asked.

SCOPE: Femur, Tibia, Coupler.  NOT Side panel, NOT RobotMount — he took the
plates out of scope explicitly ("let's not focus on the robot plates, either the
side mount or the robot plate, let's just focus on the links for now").
Femur_inside_InsideBox is the MIRRORED LEFT FEMUR and inherits the Femur
treatment; it is not a separate recipe.

REBUILD EACH LINK AGAINST SIX HARD CONSTRAINTS.  These are his words, and every
one of them has a check that must pass before anything is shown:

  1. ADD AND REMOVE ONLY WHERE IDENTIFIED.  His marks give the INTENT (which
     edges, which regions); tools/freemap.py gives the EXTENT.  See "The two
     inputs" below.  His marks are NOT gospel — he said so himself.
  2. COLLISION FREE THROUGH THE ENTIRE STROKE.  Not three poses.  The whole
     85 deg of travel, sampled from the validated 4-bar.
  3. NO FLOATING PIECES.  Exactly one connected solid per body.
  4. NO SHARP EDGES.  No knife edges, no zero-thickness contacts, no needles.
  5. NO VERY THIN WALLS.  min_wall = 3.0 mm, his number.
  6. NO BARE BLOCKS OF COLOUR ON THE BACK.  The ENTIRE surface carries the
     aesthetic — the back is not a leftover of the colour split.

BUILD THE FEMUR FIRST and show him before touching the other two: every other
recipe is generated from parts/femur.py, so a mistake there propagates.

RULES:
  - Interpreter is C:/Users/ferna/cadenv/Scripts/python.exe. Never the repo .venv.
  - Log decisions with lib/logdec.py BEFORE doing anything else.
  - Builds are ~2-4 min. Run long jobs in the background; ASK before launching.
  - Every hole is untouchable: diameters, positions, counterbores.
  - Ask, don't assume. Ask for marks on renders — his highest-bandwidth feedback.
```

---

## The six constraints, and how each one is enforced

He listed these as the definition of done. None is satisfied today.

| # | constraint | the check | status |
|---|---|---|---|
| 1 | add / remove only where identified | `input/marks/*.json` ∩ `input/freespace/*.npz`, see `tools/reconcile.py` | inputs ready, unused |
| 2 | collision free through the ENTIRE stroke | `lib/collide.py` driven by `tools/kinematics.py` over ≥21 angles | **collide.py still only does 3 poses — must be rewired** |
| 3 | no floating pieces | exactly 1 solid per body | **fails today: Coupler 4 chunks, Femur 1** |
| 4 | no sharp edges | `lib/manifold.py` non-manifold + free edges empty; no needle solids | welding exists; needles still produced |
| 5 | no very thin walls | `min_wall = 3.0` mm | never measured on a finished body |
| 6 | whole surface carries the aesthetic | the back must be designed, not left over | **fails today: bare grey squares** |

**Constraint 2 is the one with real work in it.** `lib/collide.py` reads three
pose STEPs. `tools/kinematics.py` can now place every instance at any hip angle,
validated to 0.0000 mm, so the sweep should be driven from it. Until that is
done, "collision free" means three samples, which is not what he asked for.

**Constraint 6 has a trap.** Some of the "bare grey on the back" may have been
the Phase 1 shattering rather than a styling failure. **Look at the backs again
in a trustworthy export before treating it as an aesthetic problem.** The
`*_colour.step` files are welded and correct now; the per-filament print STEPs
are not.

---

## The two inputs Phase 2 produced — intent and extent

### Intent — what he drew

He marked five sheets into `input/human feedback/`, then immediately demoted
them himself (decision 26):

> *"They're just suggestions or ideas of what I could see with my naked eye. But
> **the collision check is the ultimate judge** of all of this."*

So his marks say WHERE and the geometry says HOW FAR. Where they disagree the
geometry wins **in both directions** — growth he marked that fouls gets carved
back, and free space he did not mark is fair game.

**What he drew is one coherent idea: thicken the rim, hollow the web, leave the
bosses alone.** GREEN is a band following each link's outer perimeter along both
±Y edges for most of its length. RED is the central web, and in every case it
stops well short of the pivots — Femur x −73..63 of a part spanning −113..111,
Tibia x 63..158 of −21..245, Coupler x −45..60 of −102..107.

Persisted by `tools/markplan.py` as part-local polygons:

```
input/marks/<Part>.json     add_wkt / remove_wkt, part-local XY mm, confirmed:false
```

A bounding box cannot express a perimeter band — the Tibia's halo wraps the
whole part, so its bbox IS the whole part. If you find code consuming a bbox
from these marks, it is wrong.

**His three answers on how to apply them (decision 25):**

- **Growth runs through the FULL LOCAL SECTION.** The word that matters is
  *local*: the Femur's 39 mm is a bounding box, its section at the rim is far
  thinner. Growing through the actual section at each point is what stops the
  flange becoming the 1:9 fin of decision 20.
- **Removes are POCKETS by default.** *"It could go all the way through or
  partially; all the way through would reduce part strength, so only do that in
  certain small areas."* A size threshold for through-cuts is yours to propose.
- **`min_wall` stays 3.0 mm** — about 7 extrusion widths at a 0.4 mm nozzle.

### Extent — what actually fits

`tools/freemap.py` voxelises the assembly in each link's own frame across the
swept travel and answers *how far can this edge move*, in mm, at every point on
the perimeter. Over **21 angles**, 0.8 mm plan / 1.0 mm depth voxels, 1.0 mm
clearance:

| part | silhouette | free ring | reach p50 | p90 |
|---|---|---|---|---|
| Femur | 7868 mm² | 11212 mm² | 10.5 mm | 18.4 |
| Tibia | 11356 | 12739 | 10.9 | 18.4 |
| Coupler | 8004 | 7364 | 9.9 | 18.4 |

**p90 = 18.4 and max = 20.0 on all three because the 20 mm `--grow` window is
binding.** The real free space is larger and has not been measured to its edge.

### The two reconciled — `tools/reconcile.py`

| part | he marked | TAKE (free) | PULL BACK (blocked) | OFFER (free, unmarked) |
|---|---|---|---|---|
| Femur | 673 mm² | 670 — **100 %** | 3 | 10541 |
| Tibia | 2230 | 2070 — **93 %** | 159 | 10668 |
| Coupler | 200 | 191 — **96 %** | 9 | 7173 |

**His naked eye was almost exactly right**, and there is 15–50× more room than
he marked. The Tibia's 159 mm² of pull-back is the notch at the wheel end where
the hub sweeps through.

**Do not spend the whole allowance.** ~10 mm available everywhere does not mean
10 mm everywhere — that reads as a bloated part, not a designed one. The stated
plan he has not objected to: use his band as the SHAPE and take roughly
**4–6 mm**, varying it around the perimeter using the reach map, opening up
where a feature wants more.

Pictures: `out/marks/read/<Part>_FREE.png` (reach map) and `<Part>_RECONCILE.png`
(green take, red pull back, pale offer).

---

## The kinematic model — `tools/kinematics.py`

The leg is a 1-DOF 4-bar, now solved from the exported transforms alone. Nothing
is taken from documentation: pivots from `(R_i - R_j) u = t_j - t_i`, joints
from the two-frame closure.

```
|AC| 187.580   |FE| 169.540   |CE| 39.010 mm   <- CLAUDE.md: 187.58 / 169.54 / 39.01
travel 85.00 deg                                <- CLAUDE.md: 85 deg stop to stop
reconstructs all three exported poses to 0.0000 mm
```

Recovering the documented geometry independently **is** the validation.
`--validate` runs it and refuses to hand out placements from a model that does
not reconstruct. `leg.sweep(n)` gives n hip angles across the travel;
`leg.at(q)` gives every moving instance's placement.

**Three traps it cost, all silent:**

- `pivot()` solves in the part's **LOCAL** frame. Using that as a world point
  gave `|AC|` 113.6 instead of 187.58 and a 585 mm miss.
- The circle-circle **branch sign** was inverted. Link lengths stayed perfect and
  the femur still reconstructed exactly while the coupler and tibia went 400 mm
  out — only the per-pose reconstruction caught it.
- The exported hip angles are 152.00, −160.02, −123.00 deg, so **the travel
  straddles ±180**. `linspace(min, max)` walks the long way round through 0 and
  the mechanism cannot close over most of it. Use offsets from pose 0, wrapped;
  they land on 0 / 47.98 / 85.00 deg.

**A guard in `freemap.py` that must not be removed.** `slice_points` originally
gave every segment a sample count taken from the longest one, capped at 64. Long
segments got gaps, cross-sections failed to fill, `binary_fill_holes` leaked —
so neighbours blocked as **hollow shells** and growth driven into a motor read
as free space. The Femur's plan area reported 3011 mm² against a true 7868.
`plan_shadow()` now measures plan area a second, independent way (triangle
interiors, which need no closure) and the run **refuses to publish** if the two
disagree by more than 8 %. Big flat faces are where it bites, because they
tessellate into a handful of very large triangles.

---

## Defects the rebuild must fix — all measured, none fixed

| part | defect | detail |
|---|---|---|
| Coupler | **4 floating chunks, 1.15 cm³** | the fused body is 5 disconnected solids; chunks ~21 × 5 × 7 mm |
| Femur | 1 floating chunk, 0.025 cm³ | |
| Tibia | **3 sealed internal cavities, 1.41 cm³** | two are 36 × 7 mm bubbles with no way out — unprintable |
| Tibia | 4 bodies with non-manifold edges | `white_2` (1), `graphite_4` (2), `accent_6` (2); welded at export, cause not fixed |
| RobotMount | **31 needle solids** | 0.01–0.5 mm wide × 5 mm tall (out of scope now, but the cause is shared) |
| all | the 2D `buffer(0)` pinch | root cause of the non-manifold edges; welding is a repair, not a fix |
| all | bare grey on the back | constraint 6 — but re-check in a welded export first |

**The floating-chunk cause is known and has a line number.** The detached-piece
guard runs at `parts/femur.py:699`, but the raised frame/rail/pads are unioned
on at `:730` and the flange later still — **anything stranded after line 699 is
never checked**. Move the guard to the end of `build()`.

---

## The verification gate

Before anything is shown:

```
python lib/verify.py <Part>     must print "0 changed, worst deviation 0.000 mm3"
python lib/collide.py           must print "no new collisions"   <- REWIRE TO THE SWEEP
```

**`verify.py` passing has been insufficient THREE times.** It has never tested
topology. **Add the manifold gate** — four checks that would have caught the
Phase 1 bug, the Coupler's floating chunks and the Tibia's sealed voids on the
day each was introduced:

```python
import manifold
manifold.nonmanifold_edges(body)   # empty -> SolidWorks will not shatter it
manifold.free_edges(body)          # empty -> closed shell
# plus: exactly 1 solid per body   -> constraint 3, no floating pieces
# plus: exactly 1 shell per solid  -> no sealed voids
```

### Guards already in `build()` — every one caught a real defect

| guard | what it caught |
|---|---|
| envelope removes nothing from the source | 14–21 cm³/part bevelled off by the chamfer |
| a fuse cannot lose volume | a body whose booleans all lied |
| filament bodies must sum to the part | a 3MF holding 20 cm³ of a 125 cm³ part |
| detached pieces reported with their bbox | Side panel's 4.79 cm³ at Y +75..+81 |
| `grown` still contains the source silhouette | a 25 mm² clip regression |

---

## Phase 1 — SOLVED. Read this before touching any exporter.

**The symptom:** parts opened in SolidWorks looked incomplete. The Femur showed
a thin blade standing off the knee boss where the design has a cylinder, and its
grey cap was simply absent. Bambu Studio showed the same files correctly.

**The cause: `graphite_1` was a NON-MANIFOLD SOLID.** Four vertical edges, each
the full 9.875 mm feature depth, each with **four planar faces meeting along one
line** — the body touched itself. OCC represents that happily and calls it
valid. **Parasolid — SolidWorks, NX, Solid Edge — cannot represent a non-manifold
solid at all**, so on import it SPLITS the body at every self-contact. One
17.68 cm³ body became **30 solid bodies and 11 surface bodies totalling
0.00 cm³, with NO ERROR RAISED**, because splitting is a legitimate repair from
its side.

**Where it came from:** `buffer(0)`, used in about a dozen places in the 2D
layer. It makes a self-touching ring OGC-valid by splitting it into two lobes
**that still touch at a point**. Extrude that and you get two prisms sharing one
vertical edge. The bug is upstream of every boolean, in the plan geometry.
**`lib/asmkeepout.py` still calls it twice.** `tools/freemap.py` and
`tools/markplan.py` deliberately use `shapely.make_valid()` instead.

**The fix that shipped:** `lib/manifold.py` → `weld_nonmanifold()` fuses a
0.06 mm rod along each pinch edge. On the Femur: 4/4 welded, 0 non-manifold
left, volume +0.0025 %. It is also the right mechanical answer — a knife-edge
contact carries no load and is exactly the fragile sharp feature constraint 4
rules out. `lib/stepcolor.py` now welds every body before writing, writes ONE
root product instead of one per solid, and declares the shape's true worst-case
tolerance. Confirmed by Fernando in SolidWorks: **17 solid bodies, 0 surface
bodies, 78,556.63 mm³** — *"bingo that looks beautiful"*.

### Five hypotheses that were WRONG, each tested and killed

Recorded so none is paid for twice. On every metric below, `white_1` is *worse*
than `graphite_1` and imports fine — none of these discriminate.

| hypothesis | how it died |
|---|---|
| 21–42 unrelated root products, no assembly structure | he opened a `roots` and a `multibody` build of identical geometry: **identical failure** |
| dirty micro-geometry (needles, tiny faces) | `white_1`: 179 sub-0.01 mm² faces, tol 4e-4. `graphite_1`: 49, tol 5e-5. The Tibia is worse than both and imports |
| self-intersection | `graphite_1` CLEAN; `white_1` self-intersects **and imports fine** |
| thin walls / skin from the colour split | `graphite_1` mean wall 2.108 mm |
| the declared STEP tolerance was 270× too tight | true, and fixed, but three rewrites at 5e-5 / 1e-3 / 1e-2 mm **all failed identically** |

### Things that do NOT repair this, so do not try them again

- `ShapeFix_Shape` — no-op here (1404 → 1404 faces on white, 723 → 723 on graphite)
- `ShapeUpgrade_UnifySameDomain` — no-op, same counts
- `BRepBuilderAPI_Sewing` with non-manifold mode OFF — preserves volume exactly
  and leaves **all four** bad edges

### Bambu Studio is NOT an independent witness

It imports STEP through OpenCascade — the same kernel that wrote the file — so
it reproduces OCC's interpretation by construction and will always agree.
**Only a Parasolid consumer tests portability.** If an export looks right in
Bambu, that is evidence of nothing.

### Two dead ends on the tooling, for the record

- **Driving SolidWorks over COM was tried and scrapped at his request.** It
  works (`SldWorks.Application` is registered, SolidWorks 2023 SP5), but
  PowerShell's COM layer cannot resolve SolidWorks' members
  (`TYPE_E_ELEMENTNOTFOUND`) and the Python route needs `pywin32`, which is now
  installed in `C:/Users/ferna/cadenv` and otherwise unused. He checks files
  himself and sends screenshots. **Do not re-automate SolidWorks without asking.**
- Renders from the pipeline come from the same OCC kernel that writes the files,
  so they agree with the build by construction and **cannot** catch an export
  defect.

---

## Where things stand

Five parts styled and exporting manifold — but these are the **pre-Phase-3**
versions and carry every defect listed above.

```
part         styled     source    grew      flange     openings
Femur        78.56 cm3   86.74    +8.0 Y     785 mm2   0 changed, 0.000 mm3
Tibia       211.35      228.12   +12.6 Y    1297       0 changed, 0.000
Coupler      65.19       74.27    +6.2 Y     718       0 changed, 0.000
Side panel   89.99       92.01    +7.2 Y     289       0 changed, 0.000   (out of scope)
RobotMount  107.07      110.32    +6.0 Y    1023       0 changed, 0.000   (out of scope)
```

Deliverables per part:

| path | what |
|---|---|
| `out/print/<Part>/<Part>_glacier.3mf` | print-ready, three filament bodies |
| `out/print/<Part>/..._{white,graphite,accent}.step` | per filament — **NOT welded; these still shatter in SolidWorks** |
| `out/styled/<Part>/..._colour.step` | **open this to LOOK at it** — welded, one root, coloured |
| `out/styled/<Part>/..._styled.step` | fused single body, geometry reference |
| `out/renders/` | `assembly_glacier.png`, `part_*`, `added_*` |
| `out/marks/` | the Phase 2 markup sheets, sidecars and echo/free/reconcile renders |
| `out/stepforms/Femur/` | the Phase 1 experiment files — delete when done |

`out/print/other colors/` and `out/styled/other colors/` hold the 17 tournament
concepts. Not GLACIER; ignore unless comparing.

### Source STEPs, and a trap in them

```
Femur, Tibia, RobotMount   v4 Larger Ball bearings\STEP exports\Middle multiple parts\
Coupler, Side panel        cad\aesthetics\input\parts\
```

**The export never wrote `Coupler.step` or `Side panel.step`.** SolidWorks wrote
the ASSEMBLY nodes `COUPLER.STEP` and `SIDE PANEL.STEP` (0 faces, structure
only), and Windows filenames being case-insensitive, those names blocked the
real parts. Both were recovered with `tools/extract_from_assembly.py`. **If the
export is redone they vanish again** unless the assembly nodes are renamed or
written elsewhere. `SwicthMount`/`Switch_Mount` survived only because the
spelling differs.

### Link local frames — X is length, Y is width in plan, Z is lateral

```
Femur     X -113.19..110.79 (223.98)   Y -19.40..19.40 (38.80)   Z  -5.00..34.00 (39.00)   86.74 cm3
Tibia     X  -21.00..245.49 (266.49)   Y -26.28..21.00 (47.28)   Z -13.50..13.50 (27.00)  228.12 cm3
Coupler   X -101.77..106.80 (208.57)   Y -22.03..22.03 (44.05)   Z -30.00.. 5.00 (35.00)   74.27 cm3
```

Local Z maps to global Z for all three links, and increasing global Z is
OUTBOARD. Show faces: **Tibia +Z, Femur −Z, Coupler −Z** (the last two are
mirrored; `show_face` in the spec reflects the source in and the bodies back
out, so holes return exactly where they started).

---

## How it works

```
lib/paths.py        where STEPs and outputs live; per-part output folders
lib/spec.py         axes, languages, 16 palettes
lib/manifold.py     non-manifold + free-edge detection, pinch welding
lib/keepout.py      a part's OWN openings and silhouette
lib/asmkeepout.py   SUPERSEDED by tools/freemap.py -- collapses depth to one
                    layer, pads by PAD_Z on top of clearance, falls back to a
                    convex hull, and calls buffer(0) twice.  Do not trust it.
lib/collide.py      interference -- THREE POSES ONLY, rewire to the sweep
lib/verify.py       holes survived + material removed
lib/stepcolor.py    the coloured STEP: welds, one root, honest tolerance
lib/export3mf.py    Bambu/Orca project 3MF, one filament per body
parts/femur.py      THE GENERAL RECIPE — everything else is generated from it
tools/mkpart.py     generates coupler / side_panel / robotmount from femur.py

--- Phase 2, the inputs for the rebuild ---
tools/kinematics.py solves the 1-DOF 4-bar from the exported transforms; any
                    hip angle, validated to 0.0000 mm, refuses if it cannot
tools/freemap.py    WHERE MATERIAL CAN GO: voxelises the assembly in each link's
                    frame over the swept travel, reach in mm around the perimeter
tools/markplan.py   his marks as part-local POLYGONS -> input/marks/<Part>.json
tools/reconcile.py  his marks vs the measurement: TAKE / PULL BACK / OFFER
tools/marksheet.py  the orthographic markup sheets + .json/.npz sidecars that
                    make a marked pixel mean a millimetre
tools/readmarks.py  reads red/green off a marked sheet, resolves each blob to
                    (part, local mm), writes the ECHO png he confirms

--- Phase 1 forensics, kept as evidence ---
tools/reexport.py   repackage the coloured STEPs without a full rebuild
tools/stepforms.py  the same bodies in roots/multibody/assembly form
tools/stepheal.py   ShapeFix/UnifySameDomain probes (both no-ops)
tools/freespace.py  the older per-layer room survey, superseded by freemap

parts/render_part.py   one part, 4 views, END-ON FIRST
parts/render_added.py  RED = what the styling added. The most useful render.
parts/render_asm.py    the real assembly at the exported transforms
```

`parts/tibia.py` is deliberately **not** generated. It has its own feature-band
layout (frame x 64–192, pads 66–194, split fingers at (58,102) and (138,182))
that he approved in decision 05. Regenerating it would move every band onto the
femur's fractions and silently redesign an approved part.

### The markup sheets — if he marks again

Each sheet has a `.json` (camera basis, px_per_mm, per-part 4×4 transform) and a
`.npz` (per-pixel part id + depth). Together they turn a marked pixel into an
exact point in the part's own frame; the round trip is verified to 0.01–0.04 mm
by `tools/readmarks.py --selftest`, which paints marks inside every part on
every sheet and checks each one maps back inside the zone read from it.

**The legend is RED = MAY REMOVE, GREEN = MAY ADD** (decision 24). Earlier drafts
of this file had RED meaning DO NOT TOUCH — the exact opposite. Unmarked means
leave alone. `readmarks.py` identifies a sheet BY CONTENT when the filename
cannot say, and prints the agreement fraction, because a confident match onto
the wrong sheet would put every mark on the wrong part invisibly.

---

## What did NOT work — the expensive lessons

### Four flange shapes were built and measured before one worked

| shape | result |
|---|---|
| band over the full depth | `out_h` 4.5 mm over a 39 mm part = a **1:9 fin**. His verdict: *"paint on a pig... it looks fake."* |
| per-layer tapered bands | each starts at the SILHOUETTE while the real section at depth is smaller, so it fused to nothing — the Coupler threw away its whole **13.8 cm³** flange |
| per-layer bands from each band's OWN section | the section changes with depth, so every band juts out somewhere different: a stack of **SHELVES, 44 cm³ of trays** |
| one prism pinned to the show face | fine on the links; a plate whose top face is recessed has its rim lower, so the ring had no metal to attach to and was dropped whole |

His answer to all of this is now on record: **grow through the full LOCAL
section** — the actual section at each perimeter point, not the bbox depth.

### Two wrong conclusions that had to be reversed

**"This robot has no room for additive styling."** Wrong, and now quantified:
`tools/freemap.py` measures a free ring of 7–13 cm² per link with a ~10 mm
median reach. He corrected the original claim with *"the whole idea was for you
to be able to add material."*

**"The styling removes 20 % of the source, which violates the rules."** Also
wrong, and reported to him before checking. `build()` removes material in five
places BY DESIGN — pockets, windows, full-depth trapezoid cutouts, side-wall
pockets, accent engraving. Those are the approved look.

### Five defects that every numeric check passed

1. **`verify.py` contained trap 7.** `import_step(styled).solids()[0]` measured
   ONE fragment of a multi-solid body, and every part is multi-solid.
2. **The envelope ate the source.** `union(solid, add) & env` lets the 4.4 mm
   chamfer bevel the real part wherever growth is thinner than the chamfer:
   14–21 cm³ per part. Fix: `union(solid, add & env)`.
3. **Raised pads extruded from a fixed z** floated wherever the real surface sat
   lower — a detached plate with two prongs under the Coupler. Fix: `_top_face()`.
4. **`_clip` discards interior holes** (`ShPoly(...exterior)`), so a neighbour
   whose keep-out fell INSIDE the growth region had its hole filled back in. The
   Tibia drove 1363 mm³ into the Coupler in all three poses.
5. **The flange inherited the wheel boss's width** — `Wmax` as a uniform width
   made the Tibia's edge flange 16 mm instead of 7.

**The pattern:** renders caught 1 and 3, the collision sweep caught 4 and 5, the
partition guard caught the broken 3MF, and **Fernando's own screenshot caught
the Phase 1 bug**. None was caught by the check nominally responsible for it.

### Smaller things that bit

- `from build123d import *` **exports `M`** (metres) and silently overwrites a
  module alias.
- Unioning tessellation triangles throws `TopologyException: found non-noded
  intersection`. Snap coordinates, drop degenerate triangles, retry buffered,
  fall back to a convex hull.
- `shputil.union()` fuses terms ONE AT A TIME; on the Tibia that produced a body
  with the correct volume whose topology then broke the NEXT boolean.
- `src & sty` and `src - sty` are **unreliable on multi-solid compounds** —
  informational only; do not act on that number.
- An absolute millimetre in a spec does not transfer between parts. Paid for in
  **all three axes**. When the target is a VOLUME, state the target and solve for
  the geometry.
- `build123d.Shape(topods)` has no `.volume` / `.solids()`. Enumerate with
  `TopExp_Explorer` and measure with `BRepGProp` when handling raw OCC shapes.
- numpy 2 removed the 2-D `np.cross`. Write the scalar cross out by hand.
- `*.npz` is gitignored repo-wide; the markup sidecars are whitelisted in
  `.gitignore` because a sheet without its sidecar cannot be read back.

---

## Open

- **The 20 mm free-space window is binding.** p90 and max sit at 18.4/20.0 on
  all three links, so the real reach is larger and unmeasured. Re-run
  `tools/freemap.py --grow 30` if the rebuild wants more than 20 mm anywhere.
- **`lib/collide.py` is still three poses.** Constraint 2 is not satisfiable
  until it is driven from `tools/kinematics.py`.
- **Contested space is not allocated.** `freemap` measures each link against
  SOURCE neighbours, so two links can both be told the same gap is free. Growing
  all three to the limit could have them meet in the middle; the sweep will
  catch it, but an explicit split would be better.
- **Accent volume.** GLACIER runs 2–6 % blue by volume against the original
  "under 1 %" rule from decision 00. Flagged, never re-decided.
- **The fourth palette value.** The reference sheet names matte white, matte MID
  grey, matte DARK grey and gloss blue. GLACIER has three; he chose to keep three.
- **BLACKOUT on the Femur** still fails verification (13 openings changed,
  1167 mm³). Contained, but do not lock that spec without fixing it.
- **The per-filament print STEPs are still unwelded** and will shatter in
  SolidWorks. Only the `_colour.step` is fixed. Move the weld into `build()` so
  every downstream artifact is clean.
- **The plates** (Side panel, RobotMount) read as plain fields and are now out of
  scope. Their flanges land inboard, which the free map suggests was never
  necessary — worth revisiting when they come back into scope.
- **`A_Z = −23.5 mm`** is inherited from baseline-1 and has never been
  re-measured on the v4 box. If anything comes to depend on body-centre
  coordinates, get it measured first.
