# Handoff — copy the prompt below to continue in a new session

`README.md` and `TOURNAMENT.md` both point here. This file is the current
restart prompt plus everything a new session needs: where things stand, how the
pipeline works, and what has already been tried and failed. `TOURNAMENT.md`
holds the locked look and the trap list; `DECISIONS.md` is every board Fernando
was shown and what he chose.

---

## The prompt

```
Continue the wheeled-leg-robot aesthetics work. The tournament is OVER — the
look is locked. This phase applies it to parts and keeps them printable.

READ FIRST, in this order:
  1. cad/aesthetics/HANDOFF.md     — this file: state, method, what failed
  2. cad/aesthetics/TOURNAMENT.md  — STATUS, and the traps already paid for
  3. cad/aesthetics/DECISIONS.md   — every decision, newest at the bottom

THE LOOK IS LOCKED. GLACIER, in specs/*.json. Do not restart the tournament,
do not offer new palettes or accent styles, do not change the spec unless asked.

FOR EACH NEW PART:
  1. STEP goes in cad/aesthetics/input/parts/ (lib/paths.py also searches the
     live CAD export folders, so usually nothing needs copying).
  2. Add it to tools/mkpart.py's PARTS dict and generate the recipe from
     parts/femur.py. Do NOT hand-copy the recipe — four are derived from it.
  3. DERIVE the datums from the part's own openings and cross-check against
     CLAUDE.md. Every part so far matched exactly; if yours does not, stop.
  4. Work out its show face from the assembly. It is NOT uniformly +Z.
  5. python tools/freespace.py                  is there room to grow?
     python lib/asmkeepout.py <Part> --layers 3
     python parts/<part>.py
     python lib/verify.py <Part>
     python lib/collide.py
  6. LOOK at parts/render_part.py and parts/render_added.py output yourself
     before showing him anything. Renders caught defects that every numeric
     check passed.

RULES:
  - Interpreter is C:/Users/ferna/cadenv/Scripts/python.exe. Never the repo .venv.
  - verify.py must print "0 changed, worst deviation 0.000 mm3" AND collide.py
    must print "no new collisions" before anything ships.
  - Every hole is untouchable: diameters, positions, counterbores.
  - Re-run lib/asmkeepout.py after ANY part changes shape. Stale caches let
    parts grow into each other.
  - Builds are ~2-4 min. Run long jobs in the background; ask before launching.
  - Log decisions with lib/logdec.py BEFORE doing anything else.
  - Ask for red marks on renders — his highest-bandwidth feedback by far.
```

---

## Where things stand (2026-09-20)

Five parts styled, verified, collision-clean.

```
part         styled     source    grew      flange     openings
Femur        78.56 cm3   86.74    +8.0 Y     785 mm2   0 changed, 0.000 mm3
Tibia       211.35      228.12   +12.6 Y    1297       0 changed, 0.000
Coupler      65.19       74.27    +6.2 Y     718       0 changed, 0.000
Side panel   89.99       92.01    +7.2 Y     289       0 changed, 0.000
RobotMount  107.07      110.32    +6.0 Y    1023       0 changed, 0.000

lib/collide.py: no new collisions in any pose checked
```

Deliverables per part:

| path | what |
|---|---|
| `out/print/<Part>/<Part>_glacier.3mf` | print-ready, three filament bodies |
| `out/print/<Part>/..._{white,graphite,accent}.step` | per filament |
| `out/styled/<Part>/..._colour.step` | **open this to LOOK at it** — 3 coloured bodies |
| `out/styled/<Part>/..._styled.step` | fused single body, geometry reference |
| `out/renders/` | `assembly_glacier.png`, `part_*`, `added_*` |

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
real parts. Both were recovered with `tools/extract_from_assembly.py`, which
pulls a named component out of `Extended.STEP` via XCAF. **If the export is
redone they vanish again** unless the assembly nodes are renamed or written to a
different folder. `SwicthMount`/`Switch_Mount` survived only because the
spelling differs.

### Show face — settled from the assembly, not assumed

The hip axis is global Z, and increasing global Z is OUTBOARD (wheel outermost
at Z 169..203, RobotMount innermost at 80..99).

```
Tibia        local +Z -> global +Z     show face local +Z
Side panel   local +Z -> global +Z     show face local +Z
RobotMount   local +Z -> global +Z     show face local +Z
Femur        local +Z -> global -Z     show face local -Z    MIRRORED
Coupler      local +Z -> global -Z     show face local -Z    MIRRORED
```

`show_face: "-Z"` reflects the source about XY on the way in and reflects the
finished bodies back on the way out, so features land on the right side while
every hole returns exactly where it started.

---

## How it works

```
lib/paths.py        where STEPs and outputs live; per-part output folders
lib/spec.py         axes, languages, 16 palettes
lib/keepout.py      a part's OWN openings and silhouette
lib/asmkeepout.py   where a part may NOT grow, derived from the assembly
lib/collide.py      interference across all three poses
lib/verify.py       holes survived + material removed
lib/stepcolor.py    one STEP carrying all three bodies with colour (XCAF)
parts/femur.py      THE GENERAL RECIPE — everything else is generated from it
tools/mkpart.py     generates coupler / side_panel / robotmount from femur.py
tools/freespace.py  how much room each part has to grow, per layer
tools/extract_from_assembly.py   pull a named part out of an assembly STEP
parts/render_part.py   one part, 4 views, END-ON FIRST
parts/render_added.py  RED = what the styling added. The most useful render.
parts/render_asm.py    the real assembly at the exported transforms
```

`parts/tibia.py` is deliberately **not** generated. It has its own feature-band
layout (frame x 64–192, pads 66–194, split fingers at (58,102) and (138,182))
that he approved in decision 05. Regenerating it would move every band onto the
femur's fractions and silently redesign an approved part. The shared
capabilities — `white_share`, keep-out clipping, the flange, the guards — were
backported into it instead.

### The flange — how added material is placed

Growth is ONE PRISM, never a per-layer band:

- **width** — `flange_w` (defaults to `out_h`). Never the part's max growth.
- **depth** — `grow_depth_frac` of the part's own depth.
- **placement** — show face first; if the ring cannot attach there, the band
  holding the most metal (a plate whose top face is recessed has its rim lower).
- **plan shape** — measured from a sample band DEEPER than the prism spans,
  because a recessed top face gives a tiny sample and its ring lands inland.
- **contact** — tested against metal inside the PRISM's own band.
- **clipping** — inside the design outline, outside every keep-out layer the
  prism spans, clear of the hole buffer, islands not touching metal removed.

The build prints the whole funnel, so a failure explains itself:

```
flange @ z  +2.7.. +13.5: ring 5455 -> design 3356 -> clear 1486 -> attached 1297 mm2
```

### Guards in build() — every one caught a real defect

| guard | what it caught |
|---|---|
| envelope removes nothing from the source | 14–21 cm3/part bevelled off by the chamfer |
| a fuse cannot lose volume | a body whose booleans all lied |
| filament bodies must sum to the part | a 3MF holding 20 cm3 of a 125 cm3 part |
| detached pieces reported with their bbox | Side panel's 4.79 cm3 at Y +75..+81 |
| `grown` still contains the source silhouette | a 25 mm2 clip regression |

---

## What did NOT work

Recorded because each cost real time and none is obvious in hindsight.

### Four flange shapes were built and measured before one worked

| shape | result |
|---|---|
| band over the full depth | `out_h` 4.5 mm over a 39 mm part = a **1:9 fin**. His verdict: *"paint on a pig... it looks fake."* |
| per-layer tapered bands | Each starts at the SILHOUETTE, but the real cross-section at depth is smaller, so it fused to nothing — the Coupler threw away its whole **13.8 cm3** flange |
| per-layer bands grown from each band's OWN section | The section changes with depth, so every band juts out somewhere different: a stack of **SHELVES, 44 cm3 of trays** |
| one prism pinned to the show face | Fine on the links; a plate whose top face is recessed has its rim lower, so the ring had no metal to attach to and was dropped whole |

### Two wrong conclusions I reached and had to reverse

**"This robot has no room for additive styling."** Wrong. `tools/freespace.py`
measured it: a 6 mm band outside the silhouette is **85 % clear on the Tibia**,
41–74 % on the others, 20–28 % on the Femur. What is blocked is the **pivots**,
which is exactly where `bosses` grew — so the only growth that ever survived
came through as one lone sliver, which is why it read as a fin. The shaft edges
were open the whole time. I went fully subtractive on this basis and he had to
correct me: *"The whole idea was for you to be able to add material."*

**"The styling removes 20 % of the source, which violates the rules."** Also
wrong, and reported to him before checking. `build()` removes material in five
places BY DESIGN — pockets, windows, full-depth trapezoid cutouts, side-wall
pockets, accent engraving. Those are the approved look. The metric summed all of
it. The removal that IS a defect is the envelope cutting the source, which
`build()` now measures separately.

### Five defects that every numeric check passed

1. **`verify.py` contained trap 7.** `import_step(styled).solids()[0]` measured
   ONE fragment of a multi-solid body, and every part is multi-solid. The Side
   panel read 0.39 cm3 against a 92.01 cm3 source and reported nine false
   "changed" openings — and a genuinely broken part could have passed.
2. **The envelope ate the source.** `union(solid, add) & env` lets the 4.4 mm
   chamfer bevel the real part wherever growth is thinner than the chamfer:
   14–21 cm3 per part. The Coupler was NET BIGGER while losing a fifth of its
   metal, so a volume delta hid it completely, and no bore sat where the bevel
   ran so the openings test saw nothing. Fix: `union(solid, add & env)`.
3. **Raised pads extruded from a fixed z** floated wherever the real surface sat
   lower — a detached plate with two prongs under the Coupler. Fix: `_top_face()`.
4. **`_clip` discards interior holes** (`ShPoly(...exterior)`), so a neighbour
   whose keep-out fell INSIDE the growth region had its hole filled back in.
   The Tibia drove 1363 mm3 into the Coupler beside their shared pivot in all
   three poses — identical in every pose, which is what gave it away.
5. **The flange inherited the wheel boss's width.** I used `Wmax`, the part's
   maximum growth anywhere, as the uniform flange width; the Tibia's edge flange
   came out 16 mm instead of 7 and buried 1929 mm3 in the Coupler.

**The pattern worth remembering:** renders caught 1 and 3, the collision sweep
caught 4 and 5, the partition guard caught the broken 3MF. **None was caught by
the check nominally responsible for it.** `verify.py` passing is necessary and
has twice been insufficient. Look at `render_added.py` output — red on grey —
before believing any of it.

### Smaller things that bit

- `from build123d import *` **exports `M`** (metres). It silently overwrote a
  module alias and produced `'int' object has no attribute 'plan'`.
- Unioning tessellation triangles throws `TopologyException: found non-noded
  intersection`. Snap coordinates, drop degenerate triangles, retry buffered,
  and fall back to a convex hull — a keep-out that silently returns empty does
  not throw, it just lets parts collide.
- `shputil.union()` fuses terms ONE AT A TIME. On the Tibia that produced a body
  with the correct volume whose topology then broke the NEXT boolean:
  `solid - body` returned the whole solid. Use one fuse of the whole compound.
- `src & sty` and `src - sty` are **unreliable on multi-solid compounds** — they
  still report "100 % removed" for some parts in `verify.py` while every opening
  verifies clean. Informational only; do not act on that number.
- An absolute millimetre in a spec does not transfer between parts. Paid for in
  **all three axes**: accent ranges in X, `split_z` in Z, `frame`/`rail` in Y.
  When the target is a VOLUME rather than a position, state the target and solve
  for the geometry — `white_share` bisects for the split plane, because depth
  through a part does not map to volume when the cross-section varies.

---

## Open

- **The plates read as plain fields.** Side panel and RobotMount are the biggest
  visible surfaces and the least designed. Higher feature counts helped but did
  not solve it; they may need a different treatment, not more of the same.
- **Three flanges land INBOARD.** The Coupler, Side panel and RobotMount have no
  metal at their show-face perimeter, so the flange attaches on the hidden face.
  Structural but not seen. Opening it up means loosening the keep-out (`PAD_Z`
  6 mm, 1.0 mm clearance, neighbours at maximal styled size) — margin he chose,
  so it is his call.
- **Accent volume.** GLACIER runs 2–6 % blue by volume against the original
  "under 1 %" rule from decision 00. Flagged, never re-decided. Blue is the
  purging third colour on the X2D, so it costs filament and time.
- **The fourth palette value.** The reference sheet (`artistic concepts/high res
  render.png`) names matte white, matte MID grey, matte DARK grey and gloss
  blue. GLACIER has three. He chose to keep three rather than reopen the look.
- **BLACKOUT on the Femur** still fails verification (13 openings changed,
  1167 mm3). Contained — `build_concepts.py` verifies before it renders — but do
  not lock that spec without fixing it.
- **Wheel rim, EncoderCarrier, EncoderCableClamp, Switch_Mount** are unstyled.
  The wheel is a disc; the linear band layout does not map and it would need a
  radial variant of `lib/accents.py`.
