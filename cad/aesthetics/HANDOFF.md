# Handoff — copy the prompt below to continue in a new session

> **2026-10-03: superseded as the starting point.** The styling is now built
> natively in SolidWorks (`cad/solidworks_api/`, parts in `cad/v5 Ai designed/`).
> Start from **`cad/v5 Ai designed/README.md`**. This file remains the record
> of the OCC pipeline that produced the locked look.

`README.md` and `TOURNAMENT.md` both point here. This file is the current
restart prompt plus everything a new session needs. `TOURNAMENT.md` holds the
locked look; `DECISIONS.md` is every decision Fernando made, newest at the
bottom.

**Phase 1 — outputs trustworthy in SolidWorks — DONE, 2026-09-20.**
**Phase 2 — the add/remove map and the free-space measurement — DONE, 2026-09-20.**
**Phase 3 — THE REBUILD — the three links are built; next is a GLOBAL CHECK.**

State as of 2026-09-23 (decision 39): GLOBAL CHECK RE-RUN from clean, and the
Femur/Coupler back engraving redrawn as a circuit board.  Every number below
was re-measured on disk that day.  See "Global check, 2026-09-23" below.

| # | constraint | state on Femur / Coupler / Tibia |
|---|---|---|
| 1 | add / remove only where identified | **PARTIAL** — see below; this row used to say "holds" and that was wrong |
| 2 | collision free through the stroke | **PASSES** — 21 poses, no new collisions (three links only) |
| 3 | no floating pieces | **PASSES** — one solid each; `_drop_detached` raises rather than ship |
| 4 | no sharp edges | **PASSES** — topology gate: no non-manifold, no free edges, no needles |
| 5 | no very thin walls | **FAILS** — gate at 1.5 mm: 395 / 386 / 927 mm² introduced (Coupler rose when its relief was finally built, dec. 40) |
| 6 | no bare colour blocks on the back | **PASSES** — drawn grey trace, both faces |

Every opening on all three: **0 changed, worst deviation 0.000 mm³.**

**Constraint 1, honestly.** Only ONE of his two inputs is wired, on two of the
three parts. `femur.py` (and the generated `coupler.py`) read his RED marks and
use them to confine the full-depth *through-cutouts* to his remove zone. His
GREEN (may-add) marks are loaded and never used, `tools/freemap.py` and
`tools/reconcile.py` are read by no part at all, and **`tibia.py` never reads
`input/marks/Tibia.json`**. Where material is added is decided by the growth
rules plus `lib/asmkeepout.py`, the assembly keep-out, not by his marks.

### What the global check should cover

The collision sweep, topology gate and thin-wall gate each passed or failed on
its own terms. Nobody has yet looked at the whole thing together, and there are
blind spots that no single gate covers:

1. **The left femur.** The assembly holds one leg plus `Femur_inside_InsideBox`,
   the mirrored left femur this file says "inherits the Femur treatment". It has
   **no recipe, no styled output, and the sweep checked it as SOURCE.** If it is
   printed styled, that geometry has never been collision-checked.
2. **The plates.** The sweep ran with `--parts Femur Tibia Coupler`, so Side
   panel and RobotMount were checked as source. Their styled exports on disk are
   pre-Phase-3 and out of scope, but a printed robot would carry them.
3. **The keep-out samples 3 poses; the gate samples 21.** `asmkeepout` unions
   the exported poses only. The sweep passes today, but the design input is
   coarser than its own judge. See "Constraint 2" below.
4. **Constraint 1 as described above**, especially the Tibia ignoring its marks.
5. **The pipeline relies on a module this file used to call untrustworthy.**
   "How it works" said `asmkeepout.py` is superseded by `freemap.py`. Both
   `femur.py` and `tibia.py` call `asmkeepout.load()`, and nothing calls freemap.
6. **Warnings that print on every build and are tolerated:** the Tibia's chamfer
   taper fails on layers 0 and 9 (`Standard_TypeMismatch`, built square-edged),
   and the Femur drops the same 0.021 cm³ detached piece at X −78.8..−74.9 every
   time. Both are under the guards' limits. Neither has been explained.
7. **Printability.** The Tibia is 266.5 mm against the X2D's 256 mm bed and must
   print flat at ~45°. The graphite and accent bodies are not thin-wall checked
   (decision 29), and the Femur's graphite reads 0.13 mm over a quarter of its
   surface.
8. **Does it look right in SolidWorks?** Every render comes out of the OCC
   kernel that wrote the files, so none of them can catch an export defect
   (Phase 1). Fernando opening the three `_colour.step` files is the only
   independent check.

### Global check, 2026-09-23 — what was re-run and what it found

Re-run from clean, not trusted: `thinwall.py --selftest` PASS; `verify.py` x3;
`collide.py --sweep 21` BEFORE and AFTER the change (both "no new collisions",
model validated 0.0000 mm).  The Tibia reproduces this file to the digit
(215.12 cm3, 2911.8 / 927.4 mm2).  Datums re-checked against CLAUDE.md:
Femur |AC| 187.58, Coupler |FE| 169.54, Tibia |CE| 39.01 / |CW| 185.91 /
|EW| 224.49 / C_offset 5.28 -- all agree.

**The "lines" he complained about were `back_eng`** on the Femur and Coupler:
four contour-following strips over 86% of the part.  (The short dashes on the
Coupler and around the Tibia's eyes are SOURCE openings -- untouchable.)  They
are now `accents.circuit_grooves()`: parallel buses turning together at true
45 deg, taps with teardrops, 45-deg jumpers, chamfered via pads, laid out per
part by `back_circuit` in the spec (fractions of length / half-width).
Engraved 1.2 mm from the plate's MEASURED back face (`ZGROOVE`), never proud.

Four rules the grooves needed, each found by measuring, not predicting:
  * **min_wall of land** to the edge, source holes, other grooves and the
    styling's own THROUGH cuts (a track tangent to a cutout sliver left 0.06 mm)
  * **teardrop at every 45-deg tap** -- the acute wedge is a knife edge
    (12.4 mm2 of sub-1.5 mm on the Coupler before, 0.2 after)
  * **clear of the side-wall pockets' plan reach** where their z band touches
    the groove floor (0.49 mm wall on the Coupler at x=-8.5)
  * **ZBACK is quantised** (-5.36 on the Femur; the plate face is -5.00), so
    the groove depth is taken from the plate's own horizontal faces

Net: thin surface introduced FELL on both parts (Femur 3329/467 -> 3170/395,
Coupler 1167/202 -> 1075/201) because the old strips were themselves making
thin land.  Less than 2 mm2 of sub-1.5 mm wall remains within 4 mm of any
groove.  **The Coupler keeps only one 45-deg jumper**: its back is almost all
through-cuts, side pockets and lightening holes, and nothing else fits at
min_wall.  Honest, but ask him whether he wants it or none.

Still true, re-confirmed: the Femur drops the same 0.021 cm3 piece every build;
the left femur has no recipe; the plates are unchecked; the keep-out samples 3
poses.  NEW: `asmkeepout.load()` returns None when its cache is missing, and
`_clip`/`_flange_at` then grow with NO keep-out, silently -- trap 16 again.
FIXED (dec. 40): `asmkeepout.load()` now raises on a missing cache.
FIXED (dec. 40): the Coupler's show face had NO relief -- frame/rail/pads,
pockets, windows and the accent engraving all hung off ZT, which on the Coupler
is the bearing-tube end.  They hang off ZS, the measured plate face, now.  The
Femur reproduces exactly; the Coupler's thin wall doubled (1075/201 ->
2185/386) because it now carries the raised frame + windows the Femur has.
The OCC renders cannot show a 1.2 mm groove well (camera light, no edges);
`*_colour.step` in SolidWorks is the real check.

---

## The prompt

```
Continue the wheeled-leg-robot aesthetics work.  The three LINKS (Femur,
Coupler, Tibia) have been rebuilt in Phase 3.  This session is a GLOBAL CHECK:
verify the state on disk against every claim in HANDOFF.md, then report what
holds and what does not.  Do not start fixing things until he has seen that
report.

READ FIRST, in this order:
  1. cad/aesthetics/HANDOFF.md     -- this file; start at "What the global
                                      check should cover"
  2. cad/aesthetics/DECISIONS.md   -- every decision, newest at the bottom
                                      (34-38 are the latest round)
  3. cad/aesthetics/TOURNAMENT.md  -- the locked look, and the trap list

THE LOOK IS LOCKED. GLACIER, in specs/*.json. Do not restart the tournament,
do not offer new palettes or accent styles, do not change the spec unless asked.

SCOPE: Femur, Tibia, Coupler.  The plates (Side panel, RobotMount) are out of
scope by his decision, but the global check should SAY what they would do to a
printed robot, not ignore them.

RE-RUN, DON'T TRUST.  Every gate, from clean, on what is on disk:
  lib/verify.py <Part> glacier          (x3)  openings + topology + thin wall
  lib/collide.py --sweep 21 --parts Femur Tibia Coupler      (~12 min)
  lib/thinwall.py --selftest                    the gate's own calibration
  parts/render_part.py <Part>           (x3)  and LOOK at the images
Numbers to expect are in "Where things stand".  A mismatch is a finding.

RULES:
  - Interpreter is C:/Users/ferna/cadenv/Scripts/python.exe. Never the repo .venv.
  - Log decisions with lib/logdec.py BEFORE doing anything else.
  - Builds are ~2-4 min, the sweep ~12. Run long jobs in the background;
    he has given standing permission for collision sweeps.
  - Every hole is untouchable: diameters, positions, counterbores.
  - Ask, don't assume.  Ask for marks on renders -- his highest-bandwidth
    feedback.
  - SendUserFile does not reach him.  Give file paths.
```

---

## The six constraints, and how each one is enforced

He listed these as the definition of done. Status is kept current in the table
at the top of this file; this section is about HOW each one is checked.

| # | constraint | the check | status |
|---|---|---|---|
| 1 | add / remove only where identified | red marks bound the through-cuts (`web` in `femur.py`); nothing checks the green marks or freemap | **PARTIAL** — Tibia reads no marks |
| 2 | collision free through the ENTIRE stroke | `python lib/collide.py --sweep 21` | **PASSES** — 21 poses, no new collisions |
| 3 | no floating pieces | `python lib/verify.py <Part>` → topology gate | **PASSES** on all three |
| 4 | no sharp edges | same gate: non-manifold + free edges + needle solids | **PASSES** on all three |
| 5 | no very thin walls | `python lib/thinwall.py <Part> --vs-source` | **FAILS** — 467 / 202 / 927 mm² under the 1.5 mm gate |
| 6 | whole surface carries the aesthetic | the back must be designed, not left over | **PASSES** — drawn trace, both faces |

**All six constraints now have a gate that runs.** Constraint 5 was the last
one and is `lib/thinwall.py`; see "The minimum-wall check" below for what it
measures, how it was calibrated, and the one thing about it that surprised.

`lib/collide.py --sweep N` places every instance at N hip angles across the whole
85 deg travel using `tools/kinematics.py`, and **refuses to run if the kinematic
model does not reconstruct the exported poses** — an unvalidated model claiming
"no collision" is worse than three honest poses. Without `--sweep` it still does
the three exported poses, which is NOT what "the entire stroke" means.

**It is slow, and that is expected — do not assume it has hung.** Every pair is
an OCC boolean at roughly a second and that is the entire runtime. Measured:

```
3 configurations, 5 parts, before   10m 03s
3 configurations, 3 parts, after     5m 32s     (identical findings)
21 configurations, 3 parts          ~12 min     (MEASURED, 2026-09-21)
```

The 30-35 min that used to sit on that last line was an extrapolation and it
was wrong by nearly 3x. It is cheap enough to run before every review.

Three things bought that, all worth keeping: source STEPs are imported ONCE
(they were being re-read from disk inside the inner pair loop), the styled
boolean runs first and the source boolean is skipped whenever `v_new <= TOL`
(since `d = v_new - v_old` and `v_old >= 0`, it cannot be a new collision), and
`--parts` narrows to the parts actually being rebuilt. Use `--sweep 5` while
iterating and `--sweep 21` before showing him anything.

`lib/verify.py` now ends with the topology gate and **exits non-zero** if it
fails, so it can be used as a real gate in a script. It checks the FUSED part
(must be exactly one solid) and each filament body (may legitimately be several).
Run against today's parts it reproduces every defect in the table below to the
number — 1153.4 mm³ of floating Coupler, the Femur's 25.2 mm³ chunk, the Tibia's
3 sealed cavities, and non-manifold counts of white 1 / graphite 2 / accent 2.
That agreement is the evidence the gate is correct.

**Constraint 6 is done.** Graphite was never a shape — it was the leftover of a
Z-plane split, which is exactly why the backs read as bare blocks. It is now a
DRAWN circuit-board trace inlaid on both faces (decision 32 B), so the failure
mode is gone by construction rather than by tuning.

---

## The minimum-wall check — `lib/thinwall.py`

Constraint 5 was the last one with nothing behind it. It measures the way
SolidWorks' **Evaluate → Thickness Analysis** does: from every point on the
surface, cast a ray along the INWARD normal and take the distance to the first
surface it meets. That choice is deliberate — it is a measure **he can check
himself** in the kernel that actually consumes the files, and Phase 1 settled
that only a Parasolid consumer is an independent witness.

**The obvious alternative is wrong here.** The inscribed-sphere / distance
transform measure ("local thickness") is the textbook one, and its value goes to
**zero at every convex edge**, because no ball that fits inside the material can
contain a corner point. A greebled part is mostly edges, so it would flag several
cm³ of sound metal. The ray measure errs the other way — on a wedge it reads
1/cos(angle) too LARGE — so it stays quiet rather than crying wolf.

**Calibrated before it was believed.** `--selftest` runs seven solids whose
answer is known by construction and checks each to 0.05 mm:

```
3 mm plate 3.000 | 10 mm plate 10.000 | 2 mm rib 2.000 | 1 mm pocket floor 1.000
0.5 mm fin 0.500 | 3 mm chamfer -> NO thin area | 0.5 mm ledge -> NO thin area
```

The last two are the point: a chamfered edge and a shallow ledge are exactly what
a naive thickness measure gets wrong, and both come back clean.

### The surprise: THE SOURCE PARTS FAIL AN ABSOLUTE 3 mm GATE

```
Femur    source  844.7 mm2 thin (3.5%), 17 patches, biggest   63.5
Tibia    source  719.9       (1.5%),     7 patches, biggest  142.0
Coupler  source 1602.1       (6.6%),    17 patches, biggest 1042.3
```

The picture says what the numbers cannot: on the Femur **every one of those
patches is the wall of the hip boss**, seen through its own bore. That is a
designed bearing seat, every hole is untouchable by the ground rules, and no
restyle may thicken it. An absolute gate would therefore fail a PERFECT rebuild,
and a gate that can never pass is a gate nobody reads.

So the gate measures **twice and reports the increase** — the same shape as
`lib/collide.py`, for the same reason. `verify.py` calls `compare_report()`;
`gate_report()` is still there for the absolute question. On the pre-Phase-3
Femur:

```
under 3 mm: 5825.7 mm2 total = 897.7 inherited + 4928.1 INTRODUCED by the styling
```

**4928 mm² in 26 patches over 10 mm² is the number the rebuild must drive to
zero.** The biggest are 1006 and 966 mm² at the two ends, and there is a hard
mode at exactly **0.50 mm** — a genuine half-millimetre fin in a 10 mm-tall band
along the ±Y perimeter, which the map places on the **flange**. `grow_depth_frac`
0.34 × 39 mm puts the flange prism at Z −5..+8.3, exactly where the thin material
sits.

The source measurement is cached under `out/cache/thinwall/`, keyed on the STEP's
mtime and size, because the Tibia's source alone is 32 s and `verify.py` runs
every build.

**Scope is the FUSED part only** (decision 29). The filament bodies are far
thinner — Femur graphite reads 0.13 mm over a quarter of its surface — because
the colour split shaves skins off a solid that is itself thick. Gating on that
would force a redesign of the locked GLACIER accent, so it is an accepted,
named gap rather than an oversight.

### The thin-wall residue, and three fixes that did NOT work

Measured 2026-09-21 on the Phase-3 rebuild. Source-relative, i.e. the increase
the styling is responsible for, in mm² of surface (a wall counts twice):

```
part      < 3.0 mm   < 1.5 mm  = THE GATE     patches (all)
Femur        3329        467.5                      52
Coupler      1167        201.8                      33
Tibia        2912        927.4                      42
```

**Decision 36: report at 3.0, gate at 1.5.** One ray cast, two masks, so the
printed number and the gated number cannot drift apart. `min_wall` still comes
from the spec; `GATE_WALL` does not, because it is not a style choice. The
reasoning is in `thinwall.py`'s docstring and in decision 36 — the short form
is that two thirds of the 3 mm failure is 2–3 mm material that is not a razor,
and his own source parts fail 3.0, so a 3 mm gate is one you argue with rather
than one you fix.

**The job is now 28 patches, not 7400 mm².** That is the number to work
against.

**Read the shape of that table before touching anything.** Two thirds of the
Femur's and three quarters of the Coupler's failure is material between 2 and
3 mm — thin against his number, but not a razor. The Tibia is the different
case: a quarter of its failure is under 1 mm. Chase the sub-1 mm patches; the
2–3 mm band is a question for Fernando, not a bug.

Four fixes were tried, each built and measured. **One worked. Do not re-try the
other three without new evidence.**

1. *Shed sub-min_wall fins at the end of `build()`*, using the added material's
   own plan footprint in the flange band. Sound in principle — measure rather
   than predict — but the footprint of a whole z band projects a drafted face
   at its widest, so it cannot see thinness that varies with z, which is where
   these razors are. Result: Femur shed nothing, Coupler rolled back, Tibia
   shed 383 mm³ and got **680 mm² WORSE**, because cutting a fin in half leaves
   two fresh thin faces. Reverted.
2. *Take the side cut through the flange.* `side_d` 5.33 against `flange_w` 5.5
   leaves a nominally 0.17 mm fin, which matches the 0.05–0.07 mm minima on the
   Femur's three largest patches so well that it looked certain. Snapping the
   depth out to `flange_w + min_wall` made **every part worse** — Femur 3803 →
   4311, Coupler 1253 → 1938 — because a deeper pocket thins the web between
   the two opposing pockets and between a pocket and the through cuts. The
   forbidden-band reasoning is right; the direction to escape it is inward, not
   outward, and inward is a visible change to a locked look. Reverted.
3. *Port `side_guard` to the Tibia.* It is the Femur's rule and the Tibia's hand
   port never got it. Measured A/B on this part: **worse at every threshold** —
   2912 → 3296 mm² under 3 mm, 731 → 1073 under 1 mm, and the worst single
   razor patch 330 → 807. The reason is that the guard implements only half of
   its own rule. "A removal must merge with its neighbour or stay min_wall
   clear": on the Femur the side pockets and through cuts never met, so forcing
   them apart was free; on the Tibia they MERGE into one clean opening with no
   wall at all, and the guard inserts a wall exactly where the two are nearly
   tangent — the thinnest wall it could make. Removed again, with the numbers
   written into `tibia.py` so it does not get re-ported.
4. **WORKED — `GUARD_MARGIN`.** Chasing (3) turned up that a guard buffered by
   exactly `min_wall` leaves a rim whose nominal width IS `min_wall`, and a ray
   crossing it where it is raked or curved reads a hair under, so the guard
   manufactures a large area of "2.99 mm" wall and the gate counts all of it.
   Visible on the Femur as 31.8 mm² reading exactly 3.00. Buffering by
   `min_wall + 0.6` took **442 mm² off the Femur and 84 off the Coupler** and
   cost nothing. Kept.

The honest summary: every remaining razor is in ADDED material that some
removal has chewed, and the removals are all sized in absolute millimetres
against outlines rather than against what is behind them. A real fix is a
sizing pass that knows the local wall, not another clip — and (3) is the
warning that a rule which helps one link can hurt the next, so measure per
part rather than porting on principle.

---

## Constraint 2 — PASSES, and the one line that did it

    C:/Users/ferna/cadenv/Scripts/python.exe lib/collide.py --sweep 21 --parts Femur Tibia Coupler

Roughly 12 minutes. The kinematic model reconstructs all three exported poses
to **0.0000 mm**, so the sweep refuses to run on a model it has not validated
and this one is validated.

```
no new collisions in any of the 21 configuration(s) checked
```

It used to report three pairs — Femur × AK45-10 Stator at +12.4 mm³ at EVERY
one of the 21 angles, and Tibia × Femur at +2.0 mm³ at one end of travel. All
three were the same defect, in `_flange_at`:

```python
band = band.difference(ko_f)      # cut clear of every neighbour ...
...                               # ... twenty lines later ...
band = unary_union([band, band.buffer(FL_OVER).intersection(mfp)])   # put back
if not kb.is_empty:
    band = band.difference(kb)    # holes re-subtracted -- the assembly NOT
```

**The lap grows the flange `FL_OVER` back inward over the source, straight
through the boundary that was just enforced, and nothing downstream looks
again.** At x=−75 the source's +Y edge is y=+19.4 and the lap reaches y=+17.4;
the AK45-10's body is a cylinder of r 26.5 about the hip axis and passes
through exactly there. The flange band is also 3.3 mm deeper than the plate it
laps onto, so the lap does not lie flat on the metal — it hangs past the
plate's own face, into the motor. Annular, hence constant across the sweep.

The fix is `band = band.difference(ko_f)` after the lap, and it costs 15 mm² of
Femur flange (1164 → 1149) and 19 of Coupler. A flange that has to lap into a
motor to stay attached was never attached.

**Two things tried on the way that are NOT in the tree, both worth knowing:**

* *Clipping the flange to the source's own section in z* — band the source
  footprint, dilate by the flange's reach, keep only flange within reach of
  real metal at its own height. Correct in principle and it is what the render
  caption asks for ("structure, not a fin stuck on the original"), but it
  removes **62 % of the Femur's flange**: the flange stands 3.3 mm proud along
  its whole length, and that proud step IS the visible edge rail. Far too blunt
  for a 12 mm³ problem, and a big change to a locked look. Reverted.
* *Making the keep-out sweep.* `asmkeepout.compute_all` unions over the THREE
  exported poses; `collide.py --sweep 21` judges twenty-one. **The design input
  is sampled more coarsely than the gate that judges it**, which is how
  Tibia × Femur at q = −123° could exist at all — no exported pose is near that
  angle. A working `--sweep N` for `asmkeepout` was written (tessellate each
  neighbour in its OWN frame once, then move vertices per configuration, so
  sweeping costs what one pose used to) and then reverted, because the lap fix
  cleared the gate without it and recomputing the keep-out rewrites every
  part's design input. **This blind spot is still there.** If a future
  collision appears at an angle between the exported poses, this is why.

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

**None of this is wired into the build yet.** The table above is the plan; the
rebuild did not consume it. No part reads `reconcile.json` or the freemap, and
the "4–6 mm varying around the perimeter" plan was never implemented. Growth is
a fixed `flange_w` band clipped by `asmkeepout`. See constraint 1 at the top.

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

## Defects from the pre-Phase-3 audit — all fixed

Kept because every one of them passed some numeric check before it was found.

| part | defect | fixed by |
|---|---|---|
| Coupler | 4 floating chunks, 1.15 cm³ | a plan touch is not a 3D join: flange pieces must OVERLAP metal by area and lap `FL_OVER` onto it (decision 31) |
| Femur | 1 floating chunk, 0.025 cm³ | `_drop_detached` moved to the END of `build()`, and run again after the weld |
| Tibia | 3 sealed cavities, 1.41 cm³ | side cuts break through to open air instead of stopping at a design outline (decision 33) |
| Tibia | non-manifold edges on 3 bodies | weld moved into `build()`, plus four fixes to a weld that could silently eat a part (decision 33) |
| RobotMount | 31 needle solids | out of scope, not re-checked |
| all | bare grey on the back | graphite is a DRAWN trace now, not the leftover of a plane (decision 32 B) |
| Coupler | one face plain white | `_plate_face` finds the real plate, not the bbox end of a boss tube (decision 33) |
| Femur | +12.4 mm³ into the AK45-10 at every hip angle | the flange lap re-entered the keep-out; subtracted again after it (decision 38) |

---

## The verification gate

Before anything is shown:

```
python lib/verify.py <Part>        "0 changed, worst deviation 0.000 mm3"
                                   AND "topology gate: PASS"
                                   AND "thin-wall gate: PASS"  (exits non-zero on fail)
python lib/collide.py --sweep 21   "no new collisions in any of the 21 configurations"
python lib/thinwall.py <Part>      the PICTURE -- out/renders/thinwall_<Part>_<tag>.png
```

**`verify.py` passing used to be insufficient, three times over**, because it
only ever compared VOLUMES through openings and never looked at topology. A body
can match the source hole for hole to 0.000 mm³ and still be non-manifold
(Parasolid shreds it), be five disconnected solids, or contain a sealed bubble.
All three shipped. `lib/manifold.gate()` now runs four checks:

```python
nonmanifold_edges(body)   # empty -> Parasolid will not shatter it
free_edges(body)          # empty -> closed shell
exactly 1 solid per body  # constraint 3, no floating pieces  (fused part only)
exactly 1 shell per solid # no sealed internal cavities
```

plus a needle/debris report (<1 mm³ or <0.5 mm thick) for constraint 4. The
fused part must be one solid; a per-filament body may legitimately be several,
since the accent comes out as separate traces — hence `one_solid=False` there.

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
  **SUPERSEDED 2026-10-02:** he asked for SolidWorks automation again. The
  Python route works (`EnsureModule` on the type library + QueryInterface; the
  `TYPE_E_ELEMENTNOTFOUND` above is gotcha 2 there), and GLACIER is now built as
  native SolidWorks features. Start from `cad/v5 Ai designed/README.md`.
- Renders from the pipeline come from the same OCC kernel that writes the files,
  so they agree with the build by construction and **cannot** catch an export
  defect.

---

## Where things stand

The three LINKS are rebuilt (2026-09-21) and **all three pass the topology
gate** — one connected solid, no sealed cavities, no non-manifold edges, no
needles, every opening untouched. The two plates are still pre-Phase-3 and out
of scope.

```
part         fused      source    grew      flange     openings      thin <3 / <1.5
Femur        85.28 cm3   86.74    +8.0 Y    1149 mm2   13, 0 changed   3170 / 395
Coupler      73.00       74.27    +6.2 Y    1038       21, 0 changed   2185 / 386
Tibia       215.12      228.12   +12.6 Y    1297       35, 0 changed   2912 / 927
Side panel   89.99       92.01    +7.2 Y     289        0 changed      (out of scope)
RobotMount  107.07      110.32    +6.0 Y    1023        0 changed      (out of scope)
```

Worst opening deviation across all three: **0.000 mm³**, and the 21-pose
collision sweep reports **no new collisions**. Thin wall is the only gate left.

Deliverables per part:

| path | what |
|---|---|
| `out/print/<Part>/<Part>_glacier.3mf` | print-ready, three filament bodies |
| `out/print/<Part>/..._{white,graphite,accent}.step` | per filament. Welded inside `build()` now, and these are the bodies the topology gate reads, so they pass it. Not yet opened in SolidWorks. |
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
lib/manifold.py     non-manifold + free-edge detection, pinch welding, and
                    gate()/gate_report() -- the four topology checks
lib/keepout.py      a part's OWN openings and silhouette
lib/asmkeepout.py   THE ASSEMBLY KEEP-OUT, AND WHAT plan() ACTUALLY USES.  This
                    line used to call it "superseded by freemap, do not trust
                    it" -- but femur.py and tibia.py both call load() and no part
                    reads freemap.  Its known weaknesses are real: it collapses
                    depth, pads by PAD_Z, can fall back to a hull, and unions
                    only the THREE exported poses.  The 21-pose sweep passes.
lib/collide.py      interference.  --sweep N places every instance at N hip
                    angles across the WHOLE travel via tools/kinematics.py, and
                    refuses to run on a kinematic model that does not reconstruct
                    the exported poses.  Without --sweep it does the old 3 poses.
lib/thinwall.py     CONSTRAINT 5, reported at min_wall 3.0, GATED at 1.5 (dec. 36).
                    Wall thickness by casting a ray along the
                    inward normal from every point on the surface -- the same
                    measure as SolidWorks' Thickness Analysis, so he can check
                    it himself.  --selftest calibrates it on 7 known solids.
                    --vs-source reports only what the STYLING made thin.
lib/verify.py       holes survived + material removed + THE TOPOLOGY GATE
                    (non-manifold, free edges, one solid, one shell, needles)
                    + THE THIN-WALL GATE, source-relative.
                    Exits non-zero on failure so it can gate a script.
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

**Constraint 5, the only failing gate.** Under the 1.5 mm gate the styling adds
467 / 202 / 927 mm² (Femur / Coupler / Tibia). Read "The thin-wall residue"
first; three fixes have been measured and rejected. He put this on hold until
the leg was collision-free (decision 37). It is now, so this is his call next.

**Found in the last round, not yet acted on:**

- **`side_guard` is deliberately absent from the Tibia**, with the A/B numbers in
  `tibia.py`. It only implements half its rule: it cannot let a nearly-tangent
  pocket and cut MERGE. Teach it that before porting it anywhere.
- **The Coupler's grey trace covers 14 % of its silhouette against the Femur's
  25 %.** Its five spine lightening holes eat the runs. Not raised with him.
- **The Femur's +Y flange stands 3.3 mm proud of the plate along its whole
  length.** That is the visible edge rail and it is intended, but it is also why
  the lap reached the motor. Clipping it flush removes 62 % of the flange.

**Older, still open:**

- **The 20 mm free-space window is binding.** p90 and max sit at 18.4/20.0 on
  all three links, so the real reach is larger and unmeasured. Re-run
  `tools/freemap.py --grow 30` if the rebuild wants more than 20 mm anywhere.
- **Contested space is not allocated.** `freemap` measures each link against
  SOURCE neighbours, so two links can both be told the same gap is free.
- **Accent volume.** GLACIER runs 2–6 % blue by volume against the original
  "under 1 %" rule from decision 00. Flagged, never re-decided.
- **The fourth palette value.** The reference sheet names matte white, matte MID
  grey, matte DARK grey and gloss blue. GLACIER has three; he chose to keep three.
- **BLACKOUT on the Femur** still fails verification (13 openings changed,
  1167 mm³). Contained, but do not lock that spec without fixing it.
- **The plates** (Side panel, RobotMount) are pre-Phase-3, out of scope, and have
  never been through the new gates.
- **`A_Z = −23.5 mm`** is inherited from baseline-1 and has never been
  re-measured on the v4 box. If anything comes to depend on body-centre
  coordinates, get it measured first.
