# Concept run — solo session, 2026-09-19

Fernando asked for a wide creative sweep with full liberty, having pointed at
`artistic concepts/` and `artistic concepts/circuit/`. This is the index of what
came out of it. **Every board is one png.** Boards live in `out/boards/`,
3D renders in `out/renders/`.

---

## Start here

| Board | What it answers |
|---|---|
| **`out/renders/leg_concepts_3d.png`** | **The leg, assembled and rendered in 3D — 17 concepts.** Start here. Femur and tibia built, verified, posed at the hip and knee, rendered in colour. |
| **`gal_legset.png`** | The same idea in 2D plan, twelve concepts, both parts |
| `gal_tibia_concepts.png` | The same twelve, tibia only, larger |
| `gal_femur_concepts.png` | The same twelve, femur only |
| `tibia_concepts_3d.png` | 18 tibia concepts built in 3D, in colour |
| `hero_tibia_*.png` | One concept from four cameras — the accent judged off-plan |

## Then, by question

| Board | Question |
|---|---|
| `gal_tibia_palettes.png` / `gal_femur_palettes.png` | Which colourway? 16 of them |
| `gal_tibia_accents.png` / `gal_femur_accents.png` | Which accent language? 19 of them |
| `gal_tibia_silhouette.png` / `gal_femur_silhouette.png` | Which shape treatment? Creased vs organic vs waisted |
| `gal_tibia_wild.png` / `gal_femur_wild.png` | Six pushed harder than anything shown before |

---

## The twelve concepts

Each is a deliberate pairing of palette, silhouette and accent — not a sweep of
one axis. A sweep shows what the knobs do; these show what they are *for*.

| | Concept | Palette | Accent | Character |
|---|---|---|---|---|
| 1 | **GLACIER** | 01 Arctic, softer structure | routed trace + hatch | clean, technical, closest to the current direction |
| 2 | **NIGHTRUN** | 02 Stealth | cyan light strip in a channel | dark, one bright line, the most "product" |
| 3 | **FOUNDRY** | 03 Industrial | orange trapezoids | rugged, hard creases, no rounding |
| 4 | **APEX** | 04 Precision | red chevrons | sport; the chevrons march toward the wheel |
| 5 | **ABYSS** | 05 Ocean | teal light strip | premium, waisted shaft |
| 6 | **RECON** | 06 Forest | sand corner brackets | utility; nothing decorative |
| 7 | **PULSAR** | 07 Nebula | violet joint rings only | accent reserved entirely for the joints |
| 8 | **BULLION** | 08 Titanium | one gold chamfered panel | the most restrained composition |
| 9 | **DUNE** | 09 Desert | orange wrapping the end | bold, warm, panel-not-line |
| 10 | **VICE** | 10 Cyber | dense pink circuit routing | neon, the most graphic |
| 11 | **SIGNAL** | amber (mine) | rail and ticks | instrument-panel logic |
| 12 | **GHOST** | mono (mine) | value only, no hue | the accent carried by lightness alone |

**Wild six:** OVERDRIVE, BLACKOUT, CIRCUITRY, MONOLITH, HAZARD, SCALPEL.

---

## Where the palettes came from

`artistic concepts/` sheet **"5. COLOR PALETTE EXPLORATIONS"** names ten with
exact hex codes — that sheet is where the locked "01 Arctic" decision came from.
All ten are now in `lib/spec.py` verbatim, plus six of mine (`concept`,
`concept_d`, `arctic_lt`, `amber`, `mono`, `bone`).

Their own note, which drove the accent work: *"accent colours applied to light
strips, joint rings, small panels and functional highlights; the main body
remains neutral."* Outside sources agree — a two-tone body (light panels, dark
mechanism) plus **one** vivid accent reserved for light-emitting and functional
elements is the standard robot-design vocabulary.

---

## What changed in the code

- **`lib/accents.py` is new.** The whole accent vocabulary — 18 styles — was
  lifted out of `parts/tibia.py`. It is part-agnostic: every range is a fraction
  of the host part's own length, so a new part gets all 18 for free. Nine
  hard-coded Tibia x-coordinates were the reason the femur's first pass bunched
  every accent against one end.
- **`parts/femur.py` is new** — second part, same system.
- **`lib/render_color.py` is new** — colour-aware 3D rendering, promoted out of
  scratch so the multi-solid bug can't come back.
- **`parts/gallery.py`, `parts/build_concepts.py`** — the board and build drivers.
- `lib/spec.py` gained `organic`, `waist`, 16 palettes and the accent parameters.

## Two traps found this session

1. **A filament body is several solids.** `import_step(f).solids()[0]` renders one
   fragment. It once produced a render of a part with almost no accent while the
   build was perfectly fine. Cross-check the render's volume against `build()`'s.
2. **A tapered extrude can fail on a valid outline.** Organic mode's round-join
   buffers leave sub-millimetre edges, and offsetting them by the 4.4 mm chamfer
   throws `BRepFill_OffsetWire::FixHoles(): Wrong wire`. Fixed by simplifying to
   0.5 mm (invisible at this size) and by falling back to a square edge rather
   than losing the build.

## Build results

**36 builds: 35 verified clean** — 0 openings changed, worst deviation 0.000 mm3.
18 tibia concepts, 17 femur.

**One failure, contained: BLACKOUT on the femur.** It reports 13 openings changed
and a 1167 mm3 deviation. `build_concepts.py` verifies before it renders, so it
was dropped rather than shown, and its exported STEPs and 3MF are deleted so a
downstream renderer cannot pick them up by filename and present them as real.

I localised it but did not fix it. What is known:
- The 2D plan is clean — `grown` loses **0.000 mm2** of the source silhouette,
  and added material overlaps the bores by **0.000 mm2**.
- It is the raised features. With `frame_h`/`rail_h`/`pad_h` all zero the same
  spec verifies perfectly; disabling any *one* of them does not help.
- Disabling the side-wall cuts does not help.
- `shputil.union()` collapses a multi-solid first argument through
  `solids()[0]` and can drop the body — one run produced a styled body of
  **2.05 cm3** against a source of 86.74.

A volume guard is now in both recipes: a fuse can only ever ADD volume, so a
drop is proof a term was dropped, and the build falls back to the plain body
rather than shipping a collapsed one. **That guard did not fix BLACKOUT** — it
addresses the 2.05 cm3 failure mode, which is related but not the same thing.
BLACKOUT is the only one of 36 that fails, so this is a contained defect, not a
systemic one. Do not lock that spec without fixing it first.

## Outcome

**GLACIER was chosen and is locked** for both parts — `specs/tibia.json` and
`specs/femur.json`. *"Glacier looks really good — that's the aesthetics we're
going for."* Both build and verify 0 changed / 0.000 mm3. See `TOURNAMENT.md`
STATUS for the exact spec and `HANDOFF.md` for the next phase.

## Still open

- **Accent volume.** The old ≤1% rule is long gone. GLACIER runs **3.3% on the
  tibia, 5.6% on the femur** by volume. Never re-decided explicitly — worth a
  look before a production print, since blue is the purging third colour on the
  X2D.
- **The assembly STEP** still hasn't arrived, so Phase 4 and the show-face
  question are still blocked.
