# v5 Ai designed — AI sandbox + SolidWorks automation cheatsheet

**This folder is a full copy of `cad/v4 Larger Ball bearings/` that the AI is
free to break** (Fernando, 2026-10-02). v4 stays the master and is never
touched by automation. **It IS tracked by git** (committed in 59dd8db; this
line used to say otherwise), so a Save All -- or `16_check_and_save.py`, which
saves every rebuilt document -- shows dozens of binaries as modified with no
real change: `git restore` the ones you did not mean to change. The original
v4 design notes (bearings, 2 mm link-to-link clearance, ...) are in
`../v4 Larger Ball bearings/README.txt`.

The scripts that drive SolidWorks live in **`cad/solidworks_api/`** (tracked by
git). This file is the cheatsheet: read it before touching SolidWorks from code.

> **RULE (his, 2026-10-06): save the assemblies, and save them often.** Never
> leave an edited part saved while the assemblies above it are not -- that is a
> fragile assembly (mate references updated only in memory). After EVERY part
> edit, and at every working checkpoint in between, run
> `16_check_and_save.py --chain <v5-relative part> ...`: it saves the part(s)
> and every open v5 assembly containing them at any depth, sub-assemblies
> first, ROBOT.SLDASM last, refusing on a mate error or an unstyled styled part,
> and deleting AI_HipDrive first. Nothing else is saved (the ~30 parts that come
> up dirty on load stay out of git). A script that edits a part ends with
> `chain()` itself (22, 23); a new one must too. Don't end a session -- or start
> a long sweep -- with unsaved assemblies.

## ▶ Hood B "PLATES", 2026-10-08 -- the FacetHood's top redone, in the robot

His brief: "make the facet hood more aggressive looking -- large area, little detail; the
links are more densely populated".  His calls: reshape the TOP only and fill it to link
density (skirt, tier 1, strip channel, ring fit, screws unchanged); graphite-heavy like
the links; **nothing above today's armour top, y 117.2**; 2-3 variants as offline renders
first.  `cad/aesthetics/parts/hood_variants.py` built A SPINE (gable roof, raked slats),
B PLATES, C TERRACES (stepped chevrons); boards in `cad/solidworks_api/out/renders/hood_variants/`.
**He picked B**, then: the armour "too regular, like an arrow" -> made asymmetric; two of
its traces -> corrugated hoses (his marks).

* **What B is:** a white rim with 45-deg frame tabs round a graphite field pressed 1.6
  (y 111.4); bites out of the deck edge; graphite rim rivets; an asymmetric armour stack
  (graphite chamfered base routed round the circuit -- tip off-centre, different jogs each
  side, lopsided tail -- and two white plates split by a diagonal gap, slits / windows /
  pads / a bus with a comb on them); raked white gills; two traces with combs; dark
  windows; every side facet of tier 2 graphite with raked vents; fangs on the nose; dark
  exhaust slots on the tail; a visor on the nose (graphite brow, a dark hexagonal slot
  pressed in, blue combs -- it replaced a row of fangs, his call 2026-10-08:
  `nose_visor()`).  4 filaments now (dark = 4), was 3.  163.4 cm3 + pipes (was 153.6).
* **Recesses are pressed into the hood's cavity** (`Hood.press*`): the outline grown by the
  2.5 wall is backed under the face first, so no wall thins; the cavity is empty above
  y 80 (checked against all 226 components of the 24 sweep).
* **Checked offline (`hood_variants.py <dir> B`):** below y 99.5 identical to the previous
  hood (0.0000 mm3: skirt, ring fit, 10 screws, tier 1, strip channel) and tier 2's base
  outline unchanged (the strip's lip); no colour solid floats; outside overhangs > 50 deg
  identical to before (189 mm2, the tier-1 pockets, worst 53); y <= 117.2, |z| <= 78.
* **Pipes:** `wave` + `corner` (deck) are GONE -- B's gills and rear trace sit there.
  New, on face `field` (y 111.4): `hook` collar -> dive round the armour's tail (-z, rear),
  `sweep` dive -> collar, a shallow S (+z, front); `tier` unchanged.  `24 preview`: all
  OK, 0.00 mm2 off free ground.  The same routes are `hood_variants.B_PIPES` (its renders)
  -- keep the two in step.
* **Into SolidWorks without closing anything** (72 documents were open and dirty, so
  `21 import`'s delete-the-file route was out): `21 reimport FacetHood` deletes the old
  `MBimport` + PP_ features in the OPEN part and inserts the new STEP
  (`IPartDoc.InsertImportedFeature`), refusing unless every mate on the component is a
  Lock (the hood's two are: `FX_Hood_lock_Ring`, `FX_Strip_lock_Hood`); 66 bodies =
  the STEP to 0.01 mm3; Box 30 components fully defined, 0 mate errors.  Then
  `16 --chain`, `24 export / preview / build FacetHood` (72 bodies = 72 predicted,
  worst 0.0000 mm3), `24 print FacetHood`.
* **Gotcha: 3D Interconnect drops the STEP's colours** -- every body came in uncoloured
  (`24 export` read them all from the part).  `21 colour <Part>` (also run by
  `reimport`) colours each body from the STEP itself (`stepcolor.read`, matched by centre
  of mass + volume: 66/66, worst 0.0000 mm).
* **Print:** `out/print/FacetHood_bambu.3mf`, 72 bodies, filament 1 x10 / 2 x31 / 3 x16 /
  4 x15, 233 x 156 x 39 mm, skirt down.  No collision sweep: the hood's envelope did not
  grow (y <= 117.2, |z| <= 78) and nothing moves over its top.
* **Backups** of everything replaced: `_originals/pre_hoodB_2026-10-07/` (the part, its
  STEP, the 3MF, 24's FacetHood export + pipe files); `_originals/pre_visor_2026-10-08/`
  (B with the fangs).  The previous top is still
  `box_facet_print.hood_details()` (hood_variants' "today" reference).
* **To change the hood:** edit `variant_B()` in `hood_variants.py`, check with
  `hood_variants.py <dir> B --sheets`, then `box_facet_print.py <scratch dir>`, copy ONLY
  `FacetHood.step` into `Box/Facet/` (the other parts' 3MFs carry pipes -- never re-run it
  with `--3mf`), `21 reimport FacetHood`, `16 --chain Box/Facet/FacetHood.SLDPRT`,
  `24 export`, `preview`, `build`, `print FacetHood`.

## ▶ Corrugated pipes, 2026-10-07 -- blue ribbed hoses half-buried in the parts

His brief (reference: a sci-fi wall panel with ribbed hoses): a few corrugated
half-pipes per part, 5 mm wide, 20-50 mm long, in the blue accent, a TRUE HALF
cylinder (centre line on the face, 2.5 mm proud) so it reads as buried, routes
allowed to curve "to break the straight edges", ends a clamp collar or a dive
back into the part. His picks: graphite collars; trench pipes where a link
sweeps over the part; skip the parts with no room.

All of it is `cad/solidworks_api/24_pipes.py`; the design (routes, ends) is the
`PIPES` dict in it -- explicit (u, v) waypoints in mm on a named face, edit them.

| part | pipes | |
|---|---|---|
| Femur | `strip` 32 mm dive -> collar, `field` 26 mm collar -> dive | its free ground is straight strips between GLACIER frames |
| Side panel | `arc` 35 mm, collars both ends | clear of the coupler's swing |
| FacetBack | `cheek` 35 mm, left cheek beside the I/O bay | vertical |
| FacetHood | hood B (2026-10-08): `hook` 43 mm collar -> dive and `sweep` 46 mm dive -> collar on the pressed `field`, `tier` 30 mm on tier 1 | `wave` / `corner` (the first hood's deck) removed with that top |
| TailStrut | `keel` 30 mm on the down-facing keel facet | |
| RobotMount | `trench` 46 mm, TRENCH: a slot round the bearing boss, flush graphite clamps | the inner femur plate passes 1.9 mm over this field |
| skipped | Coupler (the tibia sweeps half its face, staggered slots fill the rest), Tibia (no 5 mm-wide flat run: pockets and rims), FacetFront (bumper posts over the cheeks, screen well), FacetRing (7.5-10 mm visible band), Femur_inside (inside the box), wheel, TPU bumpers | his call |

**The pipe.** Core r 1.95 + a torus rib (minor r 0.55) every 1.7 mm: crest r 2.5.
Proud pipes are clipped AT the face (the proud half only -- the FacetBack relief
is a skin thinner than 2 mm, a deeper clip printed blue out of its back) and cut by
the part (raised features it runs into). Collar r 3.1 x 3, 0.4 chamfer, graphite
(filament 2). A dive bends 40 deg into the face on r 9 and disappears 4.05 mm past
its waypoint (`dive_reach()`: cos phi = Rd/(Rd+R)) -- the run is shortened by that,
so the hose sinks into the face right at the end waypoint. Ribs stop where a dive
starts (see gotcha below), so the last ~4 mm is plain tube. TRENCH: centre line
1.7 under the face (crest 0.8 proud, >= 1.1 mm to the femur plate), slot R + 0.2 wide
cut down to the axis, the part hugs the hose's lower half (no undercuts), hose
clipped flat 2.5 deep (the RobotMount field is a 5 mm plate), clamps r R flush in
the slot ends.

**How a route is found (`24 routes [Part ...] [--trench]`).** Per face, offline:
* `ground()`: the part's face-level top triangles minus anything above (frames,
  bosses) = flat ground; EDGE 1 mm in from its boundary; ROUND openings < 120 mm2
  get the fastener seat (3.5 + 0.6), slots and windows only the edge margin;
  existing blue accents + 1.5 mm are out (blue on blue merges).
* `swept()`: every OTHER component at every hip angle of `24 sweep` (86 poses,
  placements from SolidWorks, meshes cached in `out/pipes/sweep/`), clipped to the
  slab just outside the face (w -0.3 .. collar crest + 1), projected, + 1 mm.
* `propose()`: on the free ground eroded by the collar radius, the longest
  geodesic path per region (Dijkstra, kept to the middle), smoothed, best 45 mm
  windows by curvature with every bend >= 8 mm.
Maps in `out/pipes/maps/`, candidates in `out/pipes/routes/<Part>__<face>.png/.json`
(orange = edge/seat, red = swept by a neighbour).

**The loop.** `24 export` (READ-ONLY: multi-body STEP + colours + placements of
the parts as they are in SolidWorks -> `out/pipes/src/`), `24 sweep` (once; ~30
min, in memory, hip put back, helper mate deleted), `24 routes`, edit `PIPES`,
`24 preview` (builds, checks, writes `out/pipes/<Part>_pipes.step` + board
`<Part>_pipes.png`; a trench also `<Part>_trench.step`, the cutter), `24 robot`
(offline whole-robot render), `24 build` (into SolidWorks + `16 --chain`), `24 print`
(Bambu 3MFs). `preview` refuses nothing silently: every pipe prints its checks --
one hose body, one body per collar, bend >= 8, visible 20..50 mm, footprint on
free ground, nothing under the face, printable overhang (box parts: their print
orientation; links: assumed show face up, as the raised styling), volume sane;
trench also: solid 3.5 mm under the whole slot, crest <= 0.8.

**In SolidWorks (`24 build`).** Each pipe body goes in as an `Imported` feature
(`CreateFeatureFromBody3` on a copy of the body of a temporary STEP import) at the
END of the tree, named `PP_<pipe>` / `PP_<pipe>_collarN`, coloured with the part's
own palette (links: swstyle GROUPS; box: box_concept_facet BLUE / GRAPHITE; the
TailStrut's graphite is drawn DARK, so its collars are too). A trench first adds
the cutter (`PT_toolN`) and Combine-subtracts it from each body it crosses
(`PT_trenchN`) -- the original bodies stay the ones mates hold. Checks, body for
body by volume: every original exactly as before (a trenched one exactly minus the
predicted removal) + exactly the STEP's pipe bodies; then `16 --chain`. Re-running
`build` deletes the PP_/PT_ features first and proves the part is the export again.
**After `08/09 --restyle` of a part, re-run `24 build <Part>`** (the PP_ bodies sit
after the GL_ features; a restyle rebuilds those).

**Checked:**
* `build`, 6 parts: every original body exactly as before (RobotMount: the white
  plate and one graphite inlay exactly minus the predicted 340.11 + 240.53 mm3),
  plus exactly the STEP's pipe bodies (worst 0.0007 mm3); `16 --chain` 0 mate
  errors, GL_ styling intact everywhere (0 suppressed). Saved: Femur, Side panel,
  RobotMount, FacetBack, FacetHood, TailStrut + Femur.SLDASM, SIDE PANEL, BODY,
  BottomPanelWithAvionics, Box, ROBOT.
* Collision sweep, 86 poses, SolidWorks (`06_sweep.py`, whole robot), first set
  (5 parts, 8 pipes) vs `out/sweep_wheel_1deg.csv`: **0 pairs new, gone or
  changed** at any pose (`out/sweep_pipes_1deg.csv`). The 4 hip-dependent pairs are
  the known ones (ODrive USB block, limit switch, styled Side panel x Coupler 0.49,
  retract hard stop).
* The same sweep after the RobotMount trench (the inner femur plate passes ~1.9 mm
  over that field, crest 0.8): again **0 new, gone or changed**
  (`out/sweep_trench_1deg.csv`; `sweep_diff.py`).
* Offline, before any of that: the swept keep-out (`24 sweep`, 226 components x 86
  poses) and the checks listed above; 3MFs per part = the original filament counts
  + one per hose (3) and per collar/clamp (2).

Backups of everything `build` saved: `_originals/pre_pipes_2026-10-07/` (Box/ is
not in git). Print files: `out/print/<Part>_glacier_native.3mf` (links) and
`<Part>_bambu.3mf` (box, print orientation), filament 1 white / 2 graphite /
3 blue (/ 4 dark); the pre-pipe 3MFs were overwritten.

**Gotchas paid for (OCC):**
* A rib torus whose spine circle lies ON a tube swept along an ARC is one of that
  tube-torus's own meridian circles: OCC returns NO intersection and the rib comes
  back as a loose torus (and the clip then fails round it). Moving the spine
  0.05 inside made it 20x slower and still loose or inverted. Ribs are not placed on
  arc edges (dives; fillet arcs of `bend` = number routes), nor within 0.85 mm of a
  seam between edges. Spline runs are fine.
* `Edge.make_spline_approx` of the whole centre line (to avoid seams) makes
  `MakePipeShell` fail outright (curvature jump between arc and line).
* A waypoint 1 mm from the next makes a spline hook and the sweep produced a
  1.6e9 mm3 inverted solid -- waypoints closer than 2.5 mm are dropped, and the
  "volume sane" check catches the rest.

## ▶ The tail strut, 2026-10-06 -- replaces the v3 BackWheelSupport (suppressed, kept)

`Box/TailStrut/TailStrut.SLDPRT` in `BottomPanelWithAvionics.SLDASM`, built by
`cad/aesthetics/parts/tail_strut.py` (build123d, ROBOT d75 frame; source of
truth -- change it, re-run, re-import) and put in by `cad/solidworks_api/23_tail_strut.py`
(`import / install / check / save / render`). His brief: "free to edit the shape
aggressively"; round 1 (a calm graphite keel) was sent back for "more aggressive
shape + colours".

* **Shape:** a 10 mm pad (chevron nose, chevron heel, drafted, |z| 32) and a
  swept keel to the caster fork: a sharp V ridge down its back (crystal facets
  that run out to nothing above the fork -- no ledges), facets on the under
  edge, three raked teeth, a V window, two graphite fangs splayed down-back
  under the pad's outer edges (a jaw from behind).
* **Colours:** white body; graphite facets, fangs, teeth; blue ridge line,
  window lining (both flanks), chevrons on the pad flanks. 66.4 cm3 (v3 65.1).
* **Kept exactly, so the v3 mates re-made on it by geometry:** top face y 15 on
  the floor; 4 M3 clearance holes (x -147 / -112, z +-20), heads on the pad
  underside at y 5 (10 mm grip, as v3), key clearance r 3.3 below each head;
  caster axle (-171.676, -32): r 1.7 through the +z arm, r 1.4 self-tap in the
  -z arm, fork inner faces z +-5, outer faces +-16 (the v3 axle screw fits).
* **Kept out of:** below y -35 (the v3 fork bottom; caster to -39.5), behind
  the tip-over line (caster back -> bumper heel, 0.4 mm clear), |z| > 40.
* **Print:** pad down, no supports by construction (every face that looks up
  in the robot is >= ~40 deg off horizontal; the window's floor is a V).
  `out/print/TailStrut_bambu.3mf`, 86 x 68 x 50 mm, filament 1 white /
  2 graphite / 3 blue.
* **Checked:** `install` re-made all 4 mates (`TS_*`: strut concentric-lock +
  coincident to BottomPanel, caster axle concentric + 1.000 gap), 0 refused;
  strut fully defined, caster UNDER (its spin) as before, moved 0.0004 mm;
  mate errors 0 in BottomPanelWithAvionics, Box, ROBOT; Interference Detection
  in Box: 0 involving the strut. No hip sweep: the strut stays inside |z| 34
  and the legs never come inside |z| 75. Saved with `16 --chain`: TailStrut,
  BottomPanelWithAvionics, Box, ROBOT.

**The bottom plate was left as is (measured, his question):** BottomPanel-1
coloured magenta in memory, pixels counted in the robot renders: 0.00-0.05 % of
the robot in every view from the side, front, back, top and the three isos (the
2 % from the left side is the open left wall -- the left leg and its side panel
are suppressed in CAD); 52 % only from straight below. Not worth restyling.

## ▶ The wheel hub, 2026-10-06 -- GLACIER "chip" on the rim, dark wheel

`Motor/Wheel Motor/Wheel.SLDPRT`, built by `cad/solidworks_api/22_style_wheel.py`
(design + 2D `preview` + build + `verify` + `render`). His pick: palette **B,
the graphite wheel, to contrast the white links**.

* **Outboard is the rim's part -Z** (wheel frame z = robot -Z; nothing in the
  robot is outboard of it). The face: flat disc z -4.0527, r <= 20.91, on a
  **0.8 mm floor**; 4 holes r 1.45 on r 17.00 at 45.681 + k*90 (screw heads on
  this face); a 1 x 1 mm centre nub (his call: buried).
* **Raised, inside the screw circle only:** white octagon chip (circumradius
  11, +1.6, 45-deg bevel, a NEW body) + graphite crown (circumradius 7.2,
  turned 22.5, +1.2, 45-deg bevel, merged into the rim through a column in the
  chip) with a blue die, legs to the screws, white pin-1 dot.
* **Flush 0.6 mm inlays on the flat:** a white chevron bracket round each
  screw head, a different blue trace on each lobe between screws, white pads,
  a white chevron panel under a three-line bus.
* **Why nothing raised goes round the screws:** their seats (r 11.45..22.55)
  fill the face, and every plate shaped round them came out four-lobed --
  flared lobes read as a cross pattee, hooked ones as a swastika (both built
  in preview and dropped). Keep raised motifs on this face non-four-armed.
* **Checked:** additions 608.87 mm3 = predicted (frustums - nub) to 0.001;
  inlays partition exactly; 23 bodies (graphite 1, white 11, blue 11); mates
  0 in error (WheelMotorASM, ROBOT); `22 verify` (OCC, styled STEP vs
  `out/styled/Wheel_source.step`): removed 0, everything inboard of the inlays
  identical (37909.406 mm3), 4 holes empty, 4 head seats (r 5.55) clear.
* **Printing:** the 3MF (`out/print/Wheel_glacier_native.3mf`) is in PRINT
  orientation, cup down / hub face up, so the chip and inlays are the last
  layers. Not yet printed; the orientation was his "not sure".
* `22 --restyle` rebuilds from the recipe (deletes the `GL_*` features, proves
  the original by volume), then saves the rim + WheelMotorASM, Tibia, ROBOT
  (`16 --chain`).

## ▶ The print box (D FACET), 2026-10-06 -- in the robot, old box suppressed

Concept D split into **six printable, bolt-together parts** + the strip, in
`Box/Facet/` (STEP + SLDPRT), built by `cad/aesthetics/parts/box_facet_print.py`
(build123d; source of truth -- change the script, re-run, re-import) and put
in by `cad/solidworks_api/21_box_facet.py`. Bambu 3MFs (print orientation,
filament 1 white / 2 graphite / 3 blue / 4 dark; TPU single) in
`cad/solidworks_api/out/print/<Part>_bambu.3mf`; renders in
`out/renders/box_facet_print/`.

| part | what | print |
|---|---|---|
| FacetFront | the v3 FrontPanel ITSELF as a 3 mm plate (window, M2, 8 bracket holes exact) + solid faceted relief to x 65; screen at the bottom of a dark-lined well | plate down |
| FacetBack | the v3 BackPanel itself (switch + notches, LED, buzzer, 2 USB slots, bracket holes) + relief to x -164; dark I/O bay (USB) and control pod (switch, LED, buzzer) down to plate depth, so plugs and the switch nut see 3 mm | plate down |
| BumperFront/Back | grey TPU U: cheek posts + chin, 6 mm proud of the face (4 proud of the hood); 7 long M3 socket heads each through the bracket holes (12 x M3x20 + 2 x M3x25), counterbored, 3 mm TPU under the head | face down |
| FacetRing | graphite; replaces Cap + NeopixelSupport + NeoPixelCage: the v3 Cap's 9 bracket screws (M3x8 csk), wall with 10 Ø2.8 self-tap bosses | flange down |
| FacetHood | skirt + tier 1 + strip channel + tier 2 + armour, 2.5 mm wall; 10 x M3x10 csk through the skirt into the ring; 233 x 156 x 39. Top = hood B PLATES since 2026-10-08 (see above) | skirt down, tree supports INSIDE only |
| NeopixelStrip | not printed: his 5.1 x 2.7 strip on its route -- round tier 2's base (the irregular crystal outline), above the legs; 623 of 975 mm, both ends + data wire (Ø5 hole) at the back centre | -- |

His rules it follows (2026-10-04): no heat-set inserts (captive nuts or Ø2.8
self-tap), more screws, visible heads OK; bumpers front AND back in grey TPU;
strip already diffused; old parts kept, suppressed.

**`Distance1` 80 -> 75 (his call: keep the v3 floor, no reprint).** RobotMount
inner faces at z +-75: the 150 mm box now meets BOTH side walls (gap 0.000 each
side, was 0 / 10). **The wheel track drops 340 -> 330 mm on the real robot** --
firmware track constant (encoder yaw (vR-vL)/0.340) and roll tuning to update
when it is built; nothing in firmware changed. Everything above the box top
stays |z| <= 78 (femur plates >= 81.9).

In Box.SLDASM (all `FX_*` mates, each refused if it moved anything):
* the 22 old components (panels, cap, neopixel stack, lid + its 12 screws, TPU
  protector, cushions) **suppressed, not deleted**; their mates suppress with them;
* **PLANE1** (mirror plane of the left side) was the mid-plane of FrontPanel-1's
  side faces -- suppressing the panel would have taken every left bracket with
  it. Re-pointed to FacetFront-1's identical side faces (same place, nothing
  moved). `ModifyDefinition` REFUSES a switch to a one-reference offset plane;
  `SetReference(1, None)` cannot be passed from Python;
* 26 old mates re-made on the new parts' identical faces (brackets, OLED,
  switch, ring); the panels' datum mates, the ring's x/z (v3 used the Cap's
  corner radii -- now a rotation-locked concentric on CornerBracket-26) and the
  hood (locked to the ring: its only contact is the skirt on the flange, every
  screw concentric repeats that translation, gotcha 45) by `21 fixup`.
* **30 components fully defined, 0 mate errors**; ROBOT: Box-1 fully defined,
  0 errors. `19_box_check`: right 10/10 holes 0.0000 mm; left as before.
* **Collision sweep, 86 poses, saved robot (`out/sweep_facet_1deg.csv`): identical
  to the 2026-10-04 box sweep** -- 0 pairs new, gone or changed; no new part
  touches a leg anywhere in the stroke. The 4 hip-dependent pairs are the known
  ones (ODrive USB block x Femur_inside, limit switch, styled Side panel x
  Coupler 0.49 mm3, retract hard stop). Offline OCC check (the parts against
  every box part and both RobotMounts): 0 overlaps besides the inherited switch
  tabs; all 9 ring holes on bracket holes 0.000 mm.

**Run `21` with Box.SLDASM ALONE** (`install` does): with ROBOT open too, the
rebuild loop ran the NVIDIA OpenGL driver out of memory and SolidWorks died
mid-run (2026-10-06; nothing was saved). Save with `21 save-box` / `save-robot`
(each refuses on mate errors / unstyled links), not Save All: after a load,
~30 untouched parts come up dirty and would only be git noise.

Open: the switch model's key tabs overlap the v3 BackPanel notches by 6.6 mm3
(inherited, unchanged); the bumper screws are longer than the M3x8 flatheads
modelled inside CornerBracket.SLDASM (shared by every bracket, left alone);
mass about +100-150 g, mostly high up (M_BODY / balance trim).

## ▶ The body box, 2026-10-04 -- imported from v3, re-mated, in the robot

`Box/` holds the v3 body box (`cad/v3 .../Box`), Pack-and-Go'd by hand
(file copy + `ReplaceReferencedDocument`, gotcha 46): printed parts in `Box/`,
the v2 electronics in `Box/Electronics/`. **The two RobotMounts ARE the box's
side walls**: RobotMount was drawn in the old `SinePanel`'s frame
(`OldRobotBodyMount/ROBOT_MOUNT.SLDASM` coordinate-mates both to one origin)
and carries its 10-hole M3 pattern, so `SinePanel`, `MirrorSinePanel` and the
v3 limit-switch mount (superseded by `Body/LimitSwitch`) were left out.
`Box-1` sits in `ROBOT.SLDASM` (backup of the file before:
`_originals/pre_box_2026-10-04/ROBOT.SLDASM`). Check it with
`19_box_check.py`.

**Hierarchy -- one rule: every part is mated to what it is bolted to, inside the
lowest assembly that contains both.** In Box.SLDASM the Box planes stand in for
the right side wall: Front Plane = its inner face, Right Plane = its back edge,
Top Plane = its bottom edge (the old SinePanel frame = the RobotMount frame).

| | |
|---|---|
| Box.SLDASM | 45 components, **all fully defined, 0 mate errors**. Floor -> the three Box planes; back/front panels -> Right/Top Plane + 5 mm side overhang; cap -> front panel top + screw hole (rotation locked); each right-side corner bracket -> side datum + panel face + its screw hole; lid screws concentric in their holes; neopixel strip, bumper, lasers on their real contacts. Left brackets, left cushion and half the lid screws stay the v3 mirror features (about PLANE1, the front panel's mid-plane). Power `Switch` moved in here from the v3 robot level (bezel on the back panel, body in its Ø20 hole). All new mates are named `BX_*` |
| BottomPanelWithAvionics | fully defined except `SupportWheel`'s spin (a real DOF). Its 2 floor brackets now bolt to the floor (were mated to each other and dimensioned off the ODrive); 4 M2 standoffs locked in their board holes; the resistor/ODrive lock conflict removed. `BatteryCap` moved in here from Box (it only touches the battery holder); `BackWheelSupport` + `SupportWheel` moved in from the v3 robot level |
| CornerBracket, CustomBoard | fully defined; screws rotation-locked; Teensy and CAN boards locked to the board (the ESP32 lock conflict removed) |
| in-context references | the box parts were designed in the context of v3's **ROBOT.SLDASM** (BackPanel, BottomPanel) and v3's Box -- **all broken** (geometry checked unchanged to 0.0000 mm3), so every part edits on its own. SolidWorks still lists them (status 0 = broken, gotcha 47) |
| Box-1 in ROBOT | fully defined: `BOX_1` CornerBracket-13's side face on the RobotMount inner face, `BOX_2` its side screw concentric with RobotMount hole (10, 32.5), `BOX_3` Box Top Plane parallel to RobotMount Top Plane |

**Holes: right side 10 of 10 RobotMount holes on the box's bracket screws, worst
0.0000 mm.** Open, for him to decide (not changed):

* **RESOLVED 2026-10-06: `Distance1` = 75** (see the print box above). Was:
  **The body is 10 mm wider in CAD than the box.** `Distance1` = 80.000 mm puts
  each RobotMount inner face 80 from the robot mid-plane (160 apart); the v3
  box is 150 between its side faces. Mated to the right RobotMount, the box's
  left face is at z = -70 against the left RobotMount at -80, and the box
  mid-plane is at z = +5. `Distance1` 80 -> 75 would centre it (and narrow the
  wheel track by 10 mm -- the control model uses the track).
* **Left-side brackets:** 0.23 mm off the mirrored holes (the v3 box is itself
  asymmetric: back panel left holes at y 9.77 / 32.73 / 54.77 vs 10 / 32.5 / 55),
  and CornerBracket-30/31 (top, left) 4.53 mm off: the v3 mirror feature
  ROTATES that asymmetric bracket instead of mirroring it. The left leg is
  suppressed, so nothing shows today.
* **ODrive USB block vs the femur plate.** An 11 x 6 x 30 mm block on the
  ODrive's edge (ROBOT x -67..-56, y 27.5..33.5, z 75..105 -- the USB plug)
  runs through the RobotMount and into the leg's Side panel (constant 301 /
  173 mm3), and `Femur_inside` sweeps through it from hip -28 to -7 deg (617 mm3
  at -26). Relevant to TODO "Odrive USB hole on body": that spot is in the
  femur plate's swing.
* The RobotMount's own M3 roundheads and the brackets' v3 flatheads sit in
  the same 10 holes (two models of one screw -- it also proves the holes
  align); the flathead cones overlap the RobotMount's plain holes by 13 mm3.

Sweep with the box, 86 poses (`out/sweep_box_1deg.csv`): besides the above,
only the known contacts -- limit switch (-28..-26), retract hard stop
(-28..-27) -- and `Side panel x Coupler` 0.49 mm3 at +42..+57, which is NOT
the box: it is in the 2026-10-03 styled sweep (`sweep_on.json`) and absent
with styling off (`sweep_off.json`), so the styling introduced it.

**Concepts for an aesthetic upgrade** (concept only, nothing printed or
checked for printing): `cad/aesthetics/parts/box_concepts.py` builds three
shells in ROBOT coordinates -- A HELM (forward-leaning helmet, wrap-around
dark visor, crest), B CARAPACE (superelliptic dome, halo light ring),
C SHELLS (GLACIER stepped plates, a cap floating over a glowing seam) -- all
keeping the electronics, floor, screen, switch and the RobotMounts as side
walls, with the RGB band all the way round near the top and |z| <= 80 (C's
cap lip reaches 83 above y = 83, clear of the femur plates which stay at
z >= 86.9). `20_box_concepts.py` imports them as
`Box/Concepts/BoxConcept_*.SLDPRT` and puts each on the real robot in
`Box/Concepts/Concept_*.SLDASM`; board in
`cad/solidworks_api/out/renders/box_concepts/board_concepts.png`.

**His verdict on A-C: right direction, wrong surfaces -- "round instead of
jagged angled".** Round 2 is **D FACET** (`cad/aesthetics/parts/box_concept_facet.py`,
`Concept_D_facet.SLDASM`): only planar faces, the links' vocabulary -- a
two-tier faceted cap (steep tier 1 with a row of chamfered windows; tier 2 on a
4 mm ledge with a chevron nose, vent slashes through the side slopes on graphite
fields, chevron brows + hatch combs on the nose faces), a stepped armour plate
(graphite chamfered base, white jogged top with the graphite bus and blue
traces), the RGB band in a recessed channel under the cap's overhang, and the
screen at the back of a dark chamfered face well. Build it with
`box_concept_facet.py <dir>`, then `20_box_concepts.py import <dir>
--names=D_facet`, `assemble --names=D_facet`, `render --names=D_facet`.
Trap paid for: building a CONCAVE (jogged) outline as a half-space
intersection silently trims it -- extrude concave outlines instead.

## ▶ Status, 2026-10-03 — the styled assembly

The six printed parts -- **Femur, Coupler, Tibia, Side panel, RobotMount,
Femur_inside** -- carry GLACIER as **native SolidWorks features** (`GL_*` in each feature tree),
multi-body white / graphite / blue, in this folder. So do the three green
retaining washers (colour only, `15`), and RobotMount's bare left field got
hand-laid features (`extras.py`, his markup). `ROBOT.SLDASM` is saved with
the mirrored left leg suppressed (his call) and the hip free to drag.

| check (all on the saved v5 files) | result |
|---|---|
| mates, every v5 assembly and sub-assembly | **0 in error** |
| contacts lost to the styling (coincident faces, off vs on) | **0** |
| collisions inside sub-assemblies, Middle pose, full detail | **0** new or grown |
| collision sweep, right leg + body, 86 poses -28..+57 deg | **0** pairs new or bigger (`out/sweep_on.json` vs `sweep_off.json`) |
| openings, every horizontal face (Femur 20, Coupler 40, Tibia 50, Side panel 75, RobotMount 43) | **0 changed, worst 0.000 mm3**, no bore missing |
| thin wall < 1.5 mm introduced (gate FAILS everywhere, as the locked OCC look did) | Femur 386, Coupler 480, Tibia 494, Side panel 160 mm2; RobotMount not measured (OCC ray tracer crashes on it) |
| re-check after the RobotMount `extras.py` field (`10_verify_styled.py 85`, `out/verify_85deg.txt`) | mates 0 in error, contacts lost 0, inside new/grown 0 (measured on the saved files, before suppression). Its 2-pose sweep is VOID: its "ON" half ran after the silent reload failure of gotcha 37, i.e. OFF vs OFF. The 86-pose sweep above predates the field, but the field is colour + 1.2 mm engraving only (removals cannot collide) |

Deliverables in `cad/solidworks_api/out/`: `print/<Part>_glacier_native.3mf`
(Bambu project, filament 1 white / 2 graphite / 3 blue -- the route his test
passed), `styled/<Part>_glacier_native.step` (fused geometry), `renders/`
(assembly at the three poses, each part front/back/trimetric, the washers
top/bottom/trimetric + joint close-ups, `robotmount_left_field_assembly_front.png`).

**Added 2026-10-03 (his request): the green washers** (`15_style_washers.py`).
BearingWasher, SmallBearingWahser (both instances) and InsideFemurShaft are
now graphite with the links' circuit motif in miniature, a different layout
on every face: blue 1 mm traces with 45-degree doglegs and tick marks, white
pads/vias, a blue "chip" with a pin-1 dot on each small washer, and one
angular white panel with a 3-line blue bus on the BearingWasher. The small
washer faces outboard with its -Y face at F and its +Y face at E (same part
file), so each face has its own pattern, 0.6 mm deep. Everything stays off
the M3 head footprints (+0.5 mm) and inside each face's flat (an inlay prism
over a 0.5 mm edge fillet would ADD material). Colour only: each region is
cut from the original body and the same sketch extruded back as new bodies,
so the bodies add up to the original to 0.0000 mm3, box unchanged, 0 mate errors.
No collision re-check needed (identical outer shape). Design in
`15_style_washers.py faces()`; `preview` draws every face without
SolidWorks (`out/renders/washer_patterns.png`). Close-ups:
`out/renders/washer_F_coupler_body_front.png`, `washer_E_coupler_tibia_front.png`.

**Added 2026-10-03 (his request): `Femur_inside`, the inner femur plate.**
The Femur's inboard twin (same datums A/C, 10 mm plate + knee tube to the
Femur's tube), show face local -Z (towards the robot's centre line). Recipe
`parts/femur_inside.py` generated by `mkpart.py`; spec `femur_inside.json` is
the Femur's GLACIER with its own layout (blue: 4 jogs, first one down, 3 hatch
groups; graphite seed 23 + own lanes; own back circuit) and **nothing proud**:
`no_growth` (no flange) and the frame flush (`frame_h` = chamfer), because the
plate swings inside the body box, which is NOT in the CAD -- a collision check
cannot vouch for added material there. Removals and colour only; windows off
(`n_wins` 0: flush, they were 1-2 mm slivers along the slots). Through-cuts
confined to his Femur web (`input/marks/Femur_inside.json`, the remove region
inherited from `Femur.json`). The limit-switch edge and the hard-stop tube are
kept clear by the new contact keep-out (`17`). Result: 86490.6 -> 75699.5 mm3,
11 bodies (white 1, graphite 6, blue 4), partition exact; openings 28 checked,
0 changed, no bore missing; thin wall < 1.5 mm introduced 111.6 mm2 (gate
FAILS, as on every part -- the lowest of them); `10 85`: mates 0 in error,
contact lost 0, inside new/grown 0; full 86-pose sweep (`14`, right leg):
0 new or bigger, **contacts shrunk or vanished 0** -- limit switch 4.702 /
2.931 / 0.978 mm3 at -28/-27/-26 and hard stop 0.226 / 0.001 mm3, ON = OFF.
`MirrorFemur_inside` is a derived mirror of it, so the (suppressed) left plate
follows. The part's second configuration, `AttachedToHipMotor`, was not
opened (switching configurations would cycle the colour features, gotcha 32).

**Tyre lock, 2026-10-03 (his request; mechanical, not styling): `18_tire_lock.py`.**
The TPU tyre (10 mm ring, bore r 46.00, z 0..27) was held on the rim band
(OD r 46.70, 0.85 mm wall, 36 S-spokes) by a 0.70 mm radial press fit alone,
and slipped. Now:
* **rim** `TL_Ribs`: 36 axial ribs, 3.0 wide x 1.2 tall (r 46.7 -> 47.9, top
  corners chamfered 0.3), one on every spoke root (2.38 + k*10 deg in the
  rim's frame, measured), z 0 (the bottom lip) .. 26.0; `TL_RibRamp` (revolve
  cut) turns the top 2.6 mm of each into a lead-in ramp: the tyre goes on from
  the top, over the wedge lip, and rides up onto the ribs. +3149.7 mm3.
* **tyre** `TL_Grooves`: 36 axial grooves through the bore, same angles,
  3.0 wide (line to line with the ribs), floor r 47.9 = level with the rib
  tops (`TOP_FIT` 0), 1.9 mm deep from the bore. -5530.8 mm3. First try had
  the floor at r 47.2 (the band's 0.7 mm press fit carried onto the rib
  tops): in CAD the rib then stood 0.7 mm into the TPU and he judged the
  grooves too shallow. Now the press fit is only on the band between the
  grooves, where it grips; stretched on, ~0.7 mm clears the rib tops; torque
  goes through the flanks either way. **Do not go deeper than ~r 47.9**: the
  tyre's end faces are flat only to r 48.0 before the round shoulder (r 48.7
  at 0.05 mm in, 49.7 at 0.2), so a deeper floor notches both corners.
* **WheelMotorASM** `TL_TyreLock` (Right Planes coincident: the tyre's rotation
  was FREE before, so nothing put grooves on ribs) and `TL_TyreSeat` (Front
  Planes coincident, both parts' seat is local z 0). TL_TyreSeat REPLACES his
  `Coincident8` (tyre bottom face to a rim edge), which the grooves broke
  (error 51: they cut through that face's inner boundary); it was deleted.
* **checked:** 1 body each; mates 0 in error everywhere; rim x tyre overlap
  (OCC boolean) 5506.92 -> 3489.64 mm3 = the band press fit between the
  grooves, ribs IN the grooves (15.2 mm3 left at the rib-top radius, 0.4 per
  rib); the same tyre turned half a pitch reads 6620.81, so the number does
  detect misalignment. Nothing new outside the tyre, so no collision sweep.
* **tune after a test print:** RIB_W (groove width) first -- FDM TPU slots
  print narrow and PLA ribs wide, so line-to-line may come out tight; FIT;
  RIB_H. `18 --redo` rebuilds with the new numbers. Originals in `_originals/`.

**Open, honestly:** the colour features do not survive suppress/unsuppress or
upstream edits cleanly (gotcha 32 -- re-run `--restyle`); the Tibia's colour
bodies overlap by 10 um films (0.18 %, gotcha on SKETCH shrink); one Coupler
pocket region (305 mm2) was refused by SolidWorks and is not cut; the left
leg is not checked (suppressed) and its Femur collides there (see "The robot
assembly"); thin wall is the one gate still failing, as before.

---

## Quick start

```
C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/01_hello.py
```

| Script | What it does | Status |
|---|---|---|
| `01_hello.py` | attach to running SolidWorks, list open docs (read-only) | works |
| `02_box.py` | new part, sketch, extrude, verify volume, save to `ai_components/` | works |
| `03_read_assembly.py` | list configs, components, every top-level mate | works (but see the Variation trap below) |
| `swlib.py` | shared plumbing for step 4+: `connect`, `open_v5` (with the v4-reference guard), `components`, `placement`, `hip_dimension`, **`HipDriver`** | works |
| `04_drive_hip.py` | drive the hip to the 3 exported poses, compare with the STEP exports; check what follows | works — **0.0000 mm** on femur, coupler, tibia |
| `05_interference.py [hip] [--subassemblies]` | one Interference Detection at one hip angle, timed | works — 10.2 s full, **3.0 s** with sub-assemblies as blocks |
| `06_sweep.py [step_deg]` | interference at every pose −28..+57, sorted into FITS vs COLLISIONS, CSV in `cad/solidworks_api/out/` | works — 86 poses in 470 s (5.5 s/pose), whole robot, both legs |
| `07_multicolor_box.py` | the step-2 box + a blue inlay and two graphite pads as SEPARATE coloured bodies; exports STEP, SolidWorks 3MF, and a Bambu project 3MF | works — **the `_bambu.3mf` opened cleanly in Bambu Studio** (his test) |
| `swstyle.py` | native styling primitives: shapely outline -> sketch (Front or Top plane), boss / drafted raised / cut (scoped, flipped, through-all), tool bodies, copy body, Combine, colour | works |
| `08_style_part.py <Part> [--dry] [--fresh] [--restyle]` | GLACIER as native features on the v5 part, replaying the recipe's `build()`: Femur, Coupler, Side panel, RobotMount. `--restyle` deletes the GL_ features first | works — all four styled and saved |
| `extras.py` | hand-laid additions to a recipe's `plan()` (graphite into `P["grey"]`, blue into `P["strip"]`) for ground the recipe leaves bare; `08` applies them after `plan()`, kept off the seats. RobotMount's left field (his markup, 2026-10-03) | works |
| `09_style_tibia.py [--dry] [--fresh] [--restyle]` | the same for the Tibia's own recipe (per-profile side pockets, split-plane colour, original body kept white for its mates) | works |
| `10_verify_styled.py [step]` | mates (every assembly), contacts lost, inside-sub-assembly collisions, sweep -- styling ON from disk vs OFF suppressed | works; use `14` for the sweep half |
| `11_check_and_export.py [Part ...]` | per part: fused STEP, openings + bores vs source, thin wall, Bambu 3MF | works |
| `12_render.py` | PNGs of the assembly at the 3 poses and of each part | works |
| `13_collision_keepout.py sweep / bodies [n]` | styling-caused collisions -> interference bodies -> keep-out for ADDITIONS (resumable) | works |
| `14_sweep_compare.py on / off / compare` | the collision sweep, one phase per run, results on disk | works |
| `16_check_and_save.py [--dry]` | **the way to save after automation** (not Save All): refuses if any styled part is unstyled in memory, deletes `AI_HipDrive`, refuses on mate errors, saves every dirty v5 doc (parts, sub-assemblies, `ROBOT.SLDASM` last) | works |
| `16_check_and_save.py --chain <part> ... [--dry]` | **after every part edit** (the rule at the top): saves the part(s) + every open v5 assembly containing them, bottom-up, ROBOT last, same guards; nothing else. `chain(sw, parts)` from Python | works -- first run 2026-10-06: Wheel + TailStrut + WheelMotorASM, BottomPanelWithAvionics, Tibia, Box, ROBOT, each verified clean after its save |
| `15_style_washers.py preview` / `[--restyle]` / `render` | the green retaining washers (BearingWasher, SmallBearingWahser, InsideFemurShaft) -> graphite + blue/white circuit inlays, COLOUR ONLY (shape unchanged); 2D preview without SolidWorks; mates; 3MFs; close-ups at the F and E joints | works -- partition exact, 0 mate errors |
| `18_tire_lock.py [--redo]` | NOT styling: locks the TPU tyre to the rim against slip -- 36 axial ribs on the rim band (one per spoke root, lead-in ramp at the top), 36 matching grooves in the tyre bore, and the `TL_TyreLock` / `TL_TyreSeat` plane mates in WheelMotorASM. Parameters at the top of the file | works -- see "Tyre lock" below |
| `swmate.py` | geometry-driven mating that never moves anything: faces found by geometry in the assembly frame, `plan()` picks a minimal mate set from real contacts (touching faces, coaxial holes, flush faces) under SolidWorks' redundancy rule, `Mater.mate()` rejects and rolls back any mate that moves a component | works -- built the box hierarchy (2026-10-04) |
| `19_box_check.py` | the body box: constraint status + mate errors in Box and every sub-assembly, live in-context references, Box-1 in ROBOT, the 10 side-panel holes both sides | works -- see "The body box" |
| `20_box_concepts.py import <dir>` / `assemble` / `render` | the box concepts (from `cad/aesthetics/parts/box_concepts.py`) as parts, each on the real robot in its own concept assembly; renders. Never edits or saves ROBOT or Box | works |
| `21_box_facet.py import / install / fixup / save-box / distance / save-robot / check` | the D FACET print box into Box.SLDASM: STEP -> SLDPRT, old box suppressed, PLANE1 re-pointed, old mates re-made on the new parts, Distance1 75, guarded saves. Opens Box ALONE for the edits (GPU out-of-memory crash otherwise) | works -- see "The print box" |
| `21_box_facet.py reimport <Part>` / `colour <Part>` | a changed STEP into the existing, OPEN Facet part in place (lock-mated parts only; nothing closed or saved), bodies checked against the STEP by volume and coloured from it (3D Interconnect drops STEP colour) | works -- see "Hood B" |
| `22_style_wheel.py preview / [--restyle] / verify / render` | the wheel rim's outboard hub face: raised octagon chip + crown inside the screw circle, flush circuit inlays, dark wheel; checks volumes, mates, then OCC verify vs the source STEP; STEP + print-oriented Bambu 3MF | works -- see "The wheel hub" |
| `23_tail_strut.py import / install / check / save / render` | the tail strut (from `cad/aesthetics/parts/tail_strut.py`) replaces the v3 BackWheelSupport in BottomPanelWithAvionics: insert, record + suppress + re-make the old mates by geometry, checks, guarded save | works -- see "The tail strut" |
| `24_pipes.py export / sweep / map / routes [--trench] / preview / robot / build / print` | corrugated half-pipes and trench pipes on the printed parts: offline route finding against the swept keep-out, previews, then Imported PP_ bodies (and PT_ trench cuts) in the v5 parts, checked body by body, `16 --chain`; Bambu 3MFs | works -- see "Corrugated pipes" |
| `17_contact_keepout.py` | the contacts the robot is DESIGNED to make (pairs of the OFF baseline whose volume changes with the hip, touching a styled part) -> their interference bodies with styling OFF, in each part's frame -> `out/contact_keepout.json`; `08` keeps every removal and addition off them | works -- limit switch + retract hard stop on Femur_inside (127 mm2), the hard stop on the Coupler (22 mm2) |

Interpreter: **`C:/Users/ferna/cadenv/Scripts/python.exe`** (has pywin32).
SolidWorks Premium 2023 SP5 = API version **31**.

---

## Tweaking the CAD — the loop a new session follows

**Before anything.** SolidWorks should be running (he usually has it open;
`swlib.connect()` starts one if not -- gotcha 9). Every script opens
`ROBOT.SLDASM` itself through `swlib.open_v5`, which refuses if any part
resolves outside v5. Run from the repo root with `PYTHONIOENCODING=utf-8`
(gotcha 3). Anything over ~5 min goes in the background; a single run over
~30 min gets killed, which is why `13` and `14` are split into resumable
phases. Never save `ROBOT.SLDASM` while the `AI_HipDrive` helper mate exists
(it locks the leg; `HipDriver` adds it in memory only).

**Where a change goes:**

| to change | edit | then run |
|---|---|---|
| Femur / Coupler / Side panel / RobotMount look | the recipe `cad/aesthetics/parts/<part>.py` + `specs/<part>.json` (the LOCKED GLACIER look; `side_panel.py` and `robotmount.py` are GENERATED from `femur.py` by `cad/aesthetics/tools/mkpart.py` -- edit the generator, not them) | `08 <Part> --dry` (lists every op and what seats / keep-outs took from it), then `08 <Part> --restyle` (RobotMount ~4 min) |
| local additions where the recipe leaves ground bare | `extras.py` (`EXTRA[part](P, M)`, part-local mm, show-face coordinates) | same as above -- `08` applies them |
| Tibia look | `cad/aesthetics/parts/tibia.py` | `09 --dry`, `09 --restyle` |
| tail strut (Box/TailStrut) | `cad/aesthetics/parts/tail_strut.py` (its checks + old-vs-new preview run on every build) | `tail_strut.py "cad/v5 Ai designed/Box/TailStrut" --3mf cad/solidworks_api/out/print`, delete TailStrut.SLDPRT, `23 import`. NOT automated past that: a re-imported part gets new face IDs, so check the 4 `TS_*` mates (`23 check`) and re-make any in error by hand (`install` only transfers mates from the active v3 part, which is now suppressed) |
| wheel hub (Wheel.SLDPRT outboard face) | `design()` in `22_style_wheel.py` | `22 preview` (2D, no SolidWorks; spokes from `out/styled/wheel_ctx.json`), `22 --restyle`, `22 verify`, `22 render` |
| FacetHood top (hood B) | `variant_B()` in `cad/aesthetics/parts/hood_variants.py` (its pipes: `B_PIPES` there AND `PIPES["FacetHood"]` in 24) | `hood_variants.py <dir> B --sheets` (checks + 4 views), `box_facet_print.py <scratch>`, copy only `FacetHood.step` to `Box/Facet/`, `21 reimport FacetHood`, `16 --chain`, `24 export / preview / build / print FacetHood` |
| corrugated pipes (any part) | `PIPES` in `24_pipes.py` (waypoints on a named face; `24 routes` proposes them) | `24 preview <Part>`, look at `out/pipes/<Part>_pipes.png`, `24 build <Part>`, `24 print <Part>`; a proud pipe on a moving part or a trench: `06_sweep.py`, copy `out/sweep_1deg.csv` to a named file, `sweep_diff.py <last> <new>` |
| washers (BearingWasher, SmallBearingWahser, InsideFemurShaft) | `faces()` in `15_style_washers.py` | `15 preview` (2D, 1 s, no SolidWorks), `15 --restyle`, `15 render` |
| a design rule (seat sizes, margins) | `08_style_part.seat()` / `mech()` | `--restyle` every affected part |
| Femur_inside look | `specs/femur_inside.json` (recipe `parts/femur_inside.py` is GENERATED by `mkpart.py`, like Side panel / RobotMount) | `08 Femur_inside --dry`, then `--restyle` |
| a NEW part to style | its recipe / spec, plus **add it to the part lists**: `08.PARTS` (recipe, spec, v5 file), `10.STYLED`, `11.PARTS`, `12.PARTS`, `13.STYLED_COMPONENTS`. A recipe also needs its source STEP in `aesthetics/input/parts/` and an assembly keep-out cache in `aesthetics/input/keepout/` (`asmkeepout.compute_all`, under the part's LEAF name in the pose STEPs -- `Femur_inside_InsideBox` for a configured part) | as above |

**Then verify, cheapest first:**

1. `11_check_and_export.py <Part>` -- openings 0 changed, no bore missing; writes the fused STEP and the Bambu 3MF (~1 min/part).
2. `10_verify_styled.py 85` -- mates in every assembly, contacts lost, collisions inside sub-assemblies, plus a 2-pose sweep (~5 min). `10 1.0` is the full 86-pose version.
3. Only if the change ADDED material (flange, frame, bosses): `14_sweep_compare.py on`, then `compare`. The OFF baseline `out/sweep_off.json` stays valid while the SOURCE parts are unchanged. If it reports collisions: `13_collision_keepout.py sweep`, then `bodies` (repeat until done), then `--restyle` again (08 / 09 load `out/collision_keepout.json` themselves). Colour-only changes and removals cannot collide, so skip this step for them.
4. `12_render.py` (assembly at 3 poses + each part) / `15 render` -- look at the PNGs in `out/renders/`.
5. `16_check_and_save.py --chain <part>` -- saves the part and every assembly
   above it, ONLY if every styled part is really styled in memory (gotcha 37)
   and those assemblies have 0 mate errors. Do it after every part edit, not
   once at the end (the rule at the top). Plain `16_check_and_save.py` saves
   every dirty v5 doc, git noise included -- only when that is wanted. Tell him
   not to use Save All in SolidWorks while a script that toggles styling
   (`10`, `11`, `13`, `14`) has run since the last load.

**His own edits in SolidWorks:** the `GL_*` features are ordinary, editable
features, but the colour features after them hold references to specific
bodies (gotcha 32). After he edits an upstream one, re-run `--restyle`. That
DISCARDS hand edits to `GL_*` features, so lasting changes belong in the
recipe, `extras.py` or `15 faces()`.

---

## Connecting from Python — the recipe that works

```python
import win32com.client
from win32com.client import gencache, constants as c

sld = gencache.EnsureModule("{83A33D31-27C5-11CE-BFD4-00400513BB57}", 0, 31, 0)  # sldworks.tlb
gencache.EnsureModule("{4687F359-55D0-4CD3-B6CF-2EB42C11F989}", 0, 31, 0)        # swconst.tlb -> c.swXxx

def wrap(obj, cls):                      # re-type every bare IDispatch SolidWorks hands back
    return None if obj is None else cls(obj._oleobj_)

sw    = sld.ISldWorks(win32com.client.GetActiveObject("SldWorks.Application")._oleobj_)
model = wrap(sw.ActiveDoc, sld.IModelDoc2)
```

* `GetActiveObject` attaches to a SolidWorks **the user started**. `Dispatch("SldWorks.Application")`
  starts one if none is running (5 s), then set `sw.Visible = True`. `swlib.connect()`
  tries the first and falls back to the second (gotcha 9).
* **Look up any signature in the generated wrapper**, it is the ground truth for
  this exact install: `python -c "import win32com; print(win32com.__gen_path__)"`,
  then grep `83A33D31-27C5-11CE-BFD4-00400513BB57x0x31x0.py` for `def MethodName(`.
  Constants: grep `4687F359-...x0x31x0.py` for the name.
* Other type libraries in `C:\Program Files\SOLIDWORKS Corp\SOLIDWORKS\`:
  `swmotionstudy.tlb` {45DB5211-F358-4B5E-A235-E792EB818BAA}, `swcommands.tlb`,
  `swdimxpert.tlb`, `swpublished.tlb` (all v31.0).

### Calling conventions

| Situation | What to do |
|---|---|
| object returned as plain `IDispatch` (`ActiveDoc`, `GetDocuments()`, `GetComponents()`, `GetSpecificFeature2()`, `GetDefinition()`, `Transform2` arrays...) | `wrap(x, sld.IThing)` — `swlib.wrap` does a real `QueryInterface` first (gotcha 13) |
| property declared with a type (`model.Extension`, `.FeatureManager`, `.SketchManager`) | already typed, use directly |
| ByRef out args (`Errors`, `Warnings`) | pass `0`; they come back in a tuple: `doc, err, warn = sw.OpenDoc6(path, type, opts, "", 0, 0)`; `ok, err, warn = model.Extension.SaveAs3(path, ver, opts, None, None, 0, 0)` |
| NULL object arg (`Callout`, `ExportData`) | pass `None` |
| units | **metres and radians**, always |
| component names | `Name2` is the path: `"Femur-1/Femur-1"` is the PART inside sub-assembly `"Femur-1"` — the two share a name, filter on `"/"` |
| feature by name | `FeatureByName` is on **`IAssemblyDoc` / `IPartDoc`**, not `IModelDoc2` |
| placements | `IComponent2.Transform2.ArrayData` = 9 rotation + 3 translation (m) + scale, rotation stored for ROW vectors. `swlib.placement()` returns column-vector `[R | t]` in mm. **Verified**: matches the STEP exports to 0.0000 mm (step 4) |
| plane names for `SelectByID2` | assembly's own: `"Right Plane"`; a sub-assembly's: `"Top Plane@Femur-1@ROBOT"` (`feature@instance@top-assembly-without-extension`), type `"PLANE"` |
| new angle mate | `asm.AddMate5(c.swMateANGLE, c.swMateAlignCLOSEST, False, 0,0,0, 0,0, angle_rad, 0,0, False, False, 0)` with two planes preselected (mark 1) -> returns `(mate, err)`, `err == c.swAddMateError_NoError` (1). It shows up as type **`MatePlanarAngleDim`**, named `Angle1` -- rename via the IFeature's `.Name` |
| interference | `wrap(asm.InterferenceDetectionManager, sld.IInterferenceDetectionMgr)`, set its options, `GetInterferences()` -> `IInterference` (`.Volume` m³, `.Components`, `.IsFastener`), then **`mgr.Done()`**. See `05_interference.py` |

---

## Gotchas — each one cost a failed run

1. **Late binding calls zero-arg methods on sight.** Plain `GetActiveObject` gives
   a late-bound object where `sw.RevisionNumber` already returns the string, so
   `sw.RevisionNumber()` fails with `'str' object is not callable`.
2. **`gencache.EnsureDispatch` fails with "Element not found"** (TYPE_E_ELEMENTNOTFOUND,
   -2147319765). SolidWorks will not give type info from the live object. The
   2026-09 PowerShell attempt died on the same error. Fix: `EnsureModule` on the
   typelib GUID + `wrap()`, as above.
3. **Console encoding.** A bearing in the assembly is named
   `Bearing 6804-2RS (20x32x7)_По умолчанию`. Printing it on a cp1252 console
   raises `UnicodeEncodeError`. Every script calls `sys.stdout.reconfigure(encoding="utf-8")`.
4. **A copied assembly can load its parts from the ORIGINAL folder.** `swlib.open_v5()`
   refuses to return the model unless every component's path is under v5.
   (Checked 2026-10-02: v5 resolves to v5.)
5. **`IMate2.MinimumVariation / MaximumVariation` are RELATIVE TO THE CURRENT VALUE,
   not the limits.** Step 3 printed "limits −39.87..45.13" in v4 and
   "−52.54..32.46" in v5 — both are just −28..+57 minus wherever the leg was
   parked. The real limits are in `IAngleMateFeatureData.MinimumAngle/MaximumAngle`
   via `feat.GetDefinition()`.
6. **The hip limit-angle mate cannot be driven across 0 by its value.**
   `dim.SetSystemValue3(v)` + `EditRebuild3()` moves the femur 1:1 for v in 0..+57,
   but for negative v SolidWorks places the femur at the MIRROR solution (−2.5
   lands where +2.5 does) and past about −27 it silently stops moving.
   `EditRebuild3()` returns True throughout. **Always read the placement back;
   never trust the rebuild flag.** The angle constraint is unsigned to the solver;
   near 0 (and 180) the two solutions coincide and it takes whichever is closer.
7. **`ModifyDefinition` on `IAngleMateFeatureData` destroys a limit mate.** Setting
   `.Angle` collapses `MinimumAngle` and `MaximumAngle` to the same value, so the
   limit mate becomes a fixed angle. Negative `.Angle` is rejected (returns False).
   `FlipDimension` only changes which mirror solution is preferred; near 0 the
   solver still takes the nearer one. **Do not edit the hip mate's definition.**
8. **CRASH: do not `CloseDoc` an assembly whose mates were modified through the API.**
   After (7), `sw.CloseDoc(title)` raised "The server threw an exception"
   (RPC_E_SERVERFAULT), left the doc half-closed (listed but not active), and the
   next `CloseAllDocuments(True)` + `OpenDoc6` killed SolidWorks — Windows logged an
   APPCRASH in SLDWORKS.exe, exception 0xc0000374 (heap corruption), 2026-10-02 22:26.
   Nothing was lost (sandbox, unsaved). If a document is in a bad state, quit and
   relaunch SolidWorks instead of closing the document.
9. **A SolidWorks launched from code is invisible to `GetActiveObject`.** It runs
   as `SLDWORKS.exe -Embedding` and never registers in the Running Object Table,
   so `GetActiveObject` fails with "Operation unavailable" (-2147221021).
   `Dispatch("SldWorks.Application")` reattaches to that same process (checked:
   still one SLDWORKS.exe).
10. **Two dead ends for moving the leg — do not retry without new evidence:**
    * setting the femur sub-assembly's `Transform2` and rebuilding: the solver
      snaps it straight back;
    * the drag operator (`asm.GetDragOperator()`, `UseAbsoluteTransform=True`):
      the first `Drag` reads back a garbage placement, every later one returns
      False.
11. **The fix for gotcha 6 is a helper mate, `AI_HipDrive`** (`swlib.HipDriver`).
    An unsigned angle constraint is only ambiguous where its two solutions meet,
    at 0° and 180°. Measuring the femur's Top Plane against the assembly's
    **Right** Plane instead of its Top Plane makes the mate read 90 + hip =
    62..147° over the stroke, far from both. Result: 86 one-degree steps through
    0, femur error **0.0000°**, 0.23 s per step. `LimitAngle1` stays active and
    still enforces the stops — silently: commanding −30 leaves the femur at −28.
    `HipDriver.set()` returns the angle actually reached; check it.
12. **Interference Detection reports every designed fit.** Over the whole robot
    at one pose: 155 interferences, nearly all screws modelled into their holes,
    bearings line-to-line in their seats (8.910 mm³ each), the TPU tire over its
    rim (4713 mm³). `TreatSubAssembliesAsComponents = True` drops everything
    inside a sub-assembly (24 left, all between parts that can move relative to
    each other) and is 3.4x faster. A sweep then separates FITS (constant
    volume at every pose) from COLLISIONS (come and go, or change).
13. **Re-typing is not converting, and `ActiveSketch` goes stale.** An object's
    IDispatch only speaks its own interface; `swlib.wrap` therefore does a real
    `QueryInterface(cls.CLSID, IID_IDispatch)` before wrapping. Separately, the
    object from `SketchManager.ActiveSketch` stops being usable once the sketch
    is closed: `Select2` on it fails with "Invalid number of parameters"
    (-2147352562). After `InsertSketch(True)` closes a sketch, fetch it as the
    LAST feature in the tree (`07_multicolor_box.py: last_feature`).

14. **Arrays of doubles must be TYPED.** `body.MaterialPropertyValues2 = [r, g, b, ...]`
    sends a Python list as an array of VARIANTs; SolidWorks reads it as raw
    doubles and the colour lands scrambled (red in green, blue in
    transparency -- the femur came out 95 % transparent green). Use
    `win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, values)`.
    Suspect the same for every API call that takes a double array.
15. **Feature scope: bodies are selected with MARK 8.** A cut limited to some
    bodies = `FeatureCut4(..., UseFeatScope=True, UseAutoSelect=False, ...)` with
    the sketch selected and each body appended with `ISelectData.Mark = 8`.
    Marks 0/1/2/4 all fail silently (measured).
16. **A CUT extrudes the opposite way to a BOSS by default.** From an offset start
    plane `FeatureCut4(Dir=False)` cut towards -Z ("z 8..13" removed z 3..8).
    `swstyle.cut` passes Dir=True so everything runs +Z.
17. **Top Plane sketch coordinates are (X, -Z)**, default extrude +Y; Front Plane
    is (X, Y), +Z (measured on a probe part).
18. **Appearances beat colours.** The v4 parts carry `red.p2m` etc. at part level.
    A body colour set through `MaterialPropertyValues2` becomes a `color.p2m`
    appearance on that body, which then wins over the part's. Deleting
    appearances with `DeleteDisplayStateSpecificRenderMaterial` returned False.
19. **Coincident faces kill booleans.** Clipping a raised frame and a pocket
    against the SAME circle gives them the same edge; the cut then runs along a
    face coincident with the frame's and SolidWorks refuses it (Coupler pockets).
    Removals are clipped 0.3 mm wider than additions. Any refused feature goes
    through a retry ladder (whole sketch -> each region -> shrunk 0.05 / 0.2 mm
    -> skip, printed).
20. **Shrink removals, grow additions.** Sketches are shrunk 5 um so touching
    loops separate -- but that pulls an ADDED band that only meets the part
    along a face 5 um clear of it, and it comes out as a separate body
    (RobotMount flange). Additions are grown 0.05 mm instead.
21. **Combine leaves zero-volume sliver bodies** where an inlay face lies on an
    existing face (Coupler: 40 bodies of 0.0 mm3). Bodies under 1 mm3 are
    deleted (`GL_DropDebris`), as the OCC build's `_debris` does.
22. **`GetBodyBox` is loose around curved faces** (Tibia +Y 21.224 vs 21.000).
    Compare volumes tightly, boxes loosely.
23. **Discard unsaved changes with `model.ReloadOrReplace(False, path, True)`**,
    not CloseDoc (gotcha 8). `08_style_part.py --fresh` does this.
24. **A feature-scoped cut on SEVERAL bodies is all-or-nothing.** If any one
    selected body misses the cut, SolidWorks refuses the whole feature. Retry
    body by body (`09_style_tibia.py: scoped()`), or a real "no overlap" on one
    body silently cancels the cut on all the others.
25. **Body names must be unique.** Renaming several bodies to the same name
    leaves only one of them under it. Rename through unique temporaries.
26. **A Combine with 56 tool bodies is refused.** Subtract while the pieces are
    still few (the Tibia's graphite is taken as body - cap while the cap is 4
    bodies, not as body - every white and blue piece at the end).
27. **Suppress + unsuppress rebuilds bodies: names reset, colours can vanish.**
    API-set body names revert to feature-derived ones ('GL_blue[3]'); on the
    Tibia some bodies came back with NO colour. Classify bodies by colour
    (`swstyle.group_of`), and after any toggling RELOAD the part from disk
    (`ReloadOrReplace`) instead of patching the in-memory copy.
28. **A flipped ("keep inside") cut that is refused looks exactly like "no
    overlap".** Treating it as such lost 729 mm3 of the Tibia. Build splits by
    REMAINDER with normal cuts only: inside = X - (X - region).
30. **Keep the ORIGINAL body as the one mates attach to.** A colour split that
    turned the Tibia's original body into the graphite remainder and built the
    white cap from COPIES broke 9 mates in Tibia.SLDASM (error 51,
    swSketchErrorExtRefFail): bearing bores and seats, the stator face, the M4
    seat all moved to new bodies. White must be the original with colour
    regions cut OUT of it, as the Femur family does by construction.
31. **A cut with a sketch that has an island nested inside a hole is refused**
    (the "everything but the trace" complement: one trace polygon has a hole,
    which becomes an island in the complement). SolidWorks reads the sketch
    fine (8 regions) and refuses every cut with it. Use a remainder instead:
    A ∩ trace = A - (A - trace).
32. **Never measure a part after un-suppressing its features.** The colour
    features hold references to specific bodies (scoped cuts, Combine, Delete
    Body). After one suppress/unsuppress cycle the Coupler came back with 42
    bodies and 571552 mm3 (copies and slab tools left standing) where the saved
    part has 36 and 77094 -- and every "collision" measured on it was that
    debris. Measure the styled state from the file on disk; suppress only for a
    baseline; reload afterwards (`10`, `13`).
    **Known limitation for editing:** the same fragility means an edit to an
    UPSTREAM styling feature (say the flange outline) may not rebuild the
    colour features cleanly. Re-run `08/09 <Part> --restyle` after such an edit;
    a cleaner long-term layout is a separate derived print part holding the
    colour split, with the assembly part single-body.
29. **SolidWorks locks the files it has open** (`Permission denied` on write).
    To re-style a saved part: `08/09 --restyle` deletes every GL_ feature in
    SolidWorks (two passes: a sketch shared by two features survives the first)
    and proves the result is the original by volume, then rebuilds.
33. **`ViewZoomTo2` frames the right spot only in `*Front`.** The same
    assembly-mm box in `*Trimetric` framed a body corner. Unverified why
    (possibly the box is read in view coordinates). Close-ups use `*Front`;
    whole-part shots use zoom-to-fit with sketches hidden (`12_render.shot`).
34. **An inlay must stay inside a face's FLAT.** The colour inlay is a prism;
    laid over a 0.5 mm edge fillet it fills the rounded corner and ADDS
    material. `15` keeps every region inside the flat, and the partition check
    (bodies must add up to the original exactly) would catch a slip.
35. **A bare region in a render does not mean nothing was planned.** On
    RobotMount the recipe's only features for the left half were rails and
    pads, and both are skipped as "buried" (below the plate face). Read the
    `skipped` lines of `08 --dry` before blaming the layout; fill such ground with `extras.py`.
36. **Reading a part's colour:** `model.Extension.GetRenderMaterials2(c.swThisDisplayState, None)`
    -> each `IRenderMaterial.PrimaryColor` is an int with r = col & 255,
    g = (col >> 8) & 255, b = (col >> 16) & 255 (checked: `green.p2m` -> 87, 214, 105).
    That is how the green washers were found among 83 parts.
37. **`ReloadOrReplace` silently does nothing to a part with no window of its
    own** (loaded only as an assembly component). On 2026-10-03 `10`'s final
    reload therefore left Femur, Coupler, Tibia and Side panel in memory with
    every `GL_*` feature SUPPRESSED (its OFF baseline). A Save All in
    SolidWorks at 19:14 then wrote all four to disk UNSTYLED, and they had to
    be rebuilt with `--restyle`. Every reload now goes through
    `swlib.reload_from_disk`, which opens the part's window first and refuses
    a non-zero result. **After any script that toggles styling, check the
    parts before anyone saves** (`GL_` features suppressed = 0).
38. **Never have v4 and v5 open in the same SolidWorks session.** SolidWorks
    holds ONE document per file name, so whichever assembly opens second
    borrows the first one's parts. On 2026-10-03 a probe had v5
    `Femur.SLDPRT` loaded when v4 `ROBOT.SLDASM` was opened: v4's right AND
    left femur both resolved to the v5 styled part, and a save of v4 would
    have repointed the master at v5. `open_v5` catches the reverse case (it
    refuses components outside v5); nothing guards v4. Close one fully
    (Don't Save) before opening the other.
39. **Interference cannot see a contact that went away.** A removal over the
    limit switch's actuating edge or the hard stop deletes the contact, and
    "new or bigger" compares stay silent. `14 compare` now also lists
    contacts of styled parts that SHRANK or VANISHED, and `17` keeps the
    styling off them in the first place.
40. **SolidWorks' interference VOLUME can come back 0.00 on a real overlap.**
    After the tyre lock, Interference Detection still flagged
    `TPU wheel-1 x Wheel-1` but with 0.00 mm3, before and after fixing the
    mates; an OCC boolean of the same two parts gives 5240.00 (and 5506.92
    before, where SolidWorks agreed to 0.01). A flagged pair with zero volume
    is a failed boolean, not "no overlap": measure it another way.
41. **The InterferenceDetectionManager of a document that is not ACTIVE is
    None.** `ActivateDoc3` the assembly first (a sub-assembly opened with
    `OpenDoc6` is not activated by it).
42. **Opening a part with live in-context references loads the assembly they
    point at -- and gotcha 38 then binds it to what is in memory.** Giving the
    copied box parts a window (to break their references) silently opened
    **v3's ROBOT.SLDASM**, visibly; its `Box-1` resolved to the *v5*
    Box.SLDASM already open. It was never saved (the save script refused:
    documents open outside the target folder). Before any save, list the open
    documents; break in-context references on a copy before mixing versions.
43. **A refused `AddMate5` can still leave a broken mate behind.**
    `swAddMateError_OverDefinedAssembly` (5) came back with a new
    `DistanceN` in the tree each time. Compare the mate list before/after and
    delete what appeared (`swmate.Mater.mate` does).
44. **`EditUndo2` does not undo an API `AddMate5`.** To roll back: delete the
    mate, then put the moved component back with `Transform2` -- that sticks
    along free DOF only, which is exactly where a rejected mate moved it.
45. **SolidWorks refuses TRANSLATIONAL redundancy, even when consistent.** A
    face coincidence plus a concentric whose axis lies in that face, or two
    parallel concentrics (two screws), are refused at `AddMate5` with
    OverDefinedAssembly. Rotational overlap (a face plus a hole perpendicular
    to it) is fine. Fix the last rotation with a lock-rotation concentric or a
    parallel mate. On a still-free component a concentric can also slide it
    (18 mm, once): mate a face first.
46. **Pack and Go cannot be driven from Python here.**
    `IModelDocExtension.GetPackAndGo` raises "Parameter not optional" with
    every calling convention (typed, late-bound, property-get). The same job:
    close everything, copy the files, then
    `ISldWorks.ReplaceReferencedDocument(copy, old, new)` for every reference of
    every copied assembly (works on closed files), and verify with
    `GetDocumentDependencies2(path, True, False, False)` with nothing loaded --
    with documents open it reports the in-memory resolution instead.
47. **Broken external references stay listed, with status 0.**
    `BreakAllExternalFileReferences2(False)` works (geometry unchanged), but
    `ListExternalFileReferencesCount` does not drop: `swExternalReferenceBroken`
    is 0, live in-context is 3, out-of-context 4, dangling 5.
48. **Face normals:** negate `ISurface.PlaneParams`' normal when
    `IFace2.FaceInSurfaceSense()` is True to get the outward normal (measured
    on every box part; the other way round reports every face inside out).

### Design rules added on top of the locked look (all in 08/09)

* **Seat keep-out** (his request): no removal, raised feature or flange on the
  seat around any opening. Additions clear it by +0.6 mm, removals by +0.3 mm:
  the same circle for both shares an edge and the boolean is refused (gotcha
  19); additions inside removals left a 0.3 mm frame ridge round every seat
  (Coupler frame slivers 10.4 -> 18.5 mm2; +0.6/+0.3 gives 9.8).
* **A side pocket must break through a face or leave min_wall, in Z too.** The
  profiles' heights are absolute and were designed round the Femur's plate;
  on the Side panel they cut closed slots with 1.5 mm skins above and below.
  A skin under 2 mm is opened through the show face, a floor under 2 mm is
  raised to 3 mm. Femur and Coupler unchanged (their skins are >= 2.5 mm).
* **Openings come from EVERY horizontal face** (`08_style_part.all_openings`).
  The pipeline's `keepout.openings()` looks only at the two outermost faces of
  the bounding box, and missed a third to a half of every part's openings --
  counterbores and holes starting on a recessed face (Femur 13 -> 20, Coupler
  21 -> 40, Tibia 35 -> 50, Side panel 37 -> 75, RobotMount 31 -> 43). The
  styling grew under four M3 screw heads on the Side panel that way. An inner
  loop is an opening only if the material side of its face is empty inside it
  (a boss makes an inner loop too). The same blind spot was in the opening
  integrity check; `11` now uses the full set.
* **Contact keep-out** (`17_contact_keepout.py`): the contacts the robot is
  designed to make -- the inner femur plate's -Y edge pressing the retract
  limit switch (x -26..-15, mid-thickness) and the coupler landing on its knee
  tube (the hard stop) -- are measured with the styling OFF, outlined + 2 mm
  in each part's frame, and joined to the seat keep-out: nothing is cut or
  added there. The recipe's lower side pockets ran straight through the
  switch edge before this.
* **Collision keep-out** (`13_collision_keepout.py`): every interference the
  styling adds anywhere in the stroke is read back from SolidWorks as an
  interference body, moved into the styled part's frame, and its outline + 1
  mm is kept free of ADDITIONS on the next `--restyle`. Removals cannot
  collide, so they are never clipped by it.
* **Unattached additions are dropped**, whatever their size (RobotMount's
  flange band hovered above a 5 mm plate edge); cut-loose scraps under 1 % are
  dropped (the Femur's 21 mm3 flange scrap, as in the OCC build).
* **Rail and pads are skipped**: in the locked recipe they never clear the
  plate face (top 0.02 and -1.38 mm off it), so they were never visible.
  On RobotMount these were the ONLY features planned for the left half (a flat
  5 mm field, x 0..85, that the Side panel does not cover), so it came out
  bare. `extras.py` fills it by hand: the graphite step-bar carried left with
  white vent slashes, two blue traces with doglegs, ticks and pads, and a blue
  pin header between the lower-left screws.

### Multi-colour parts — the recipe (`07_multicolor_box.py`)

* **A new body = an extrude with `Merge=False`** (FeatureExtrusion3 arg 18).
* **An inlay** (the GLACIER accent case): cut the groove with `FeatureCut4`, then
  select the SAME sketch again and extrude it back into the groove with
  `Dir=True` (arg 3) and `Merge=False`. Check: box + inlay volume = the original
  solid exactly (8000.000 mm³ on the test cube), so no overlap and no gap.
* **Body colour and name:** `body.MaterialPropertyValues2 = [r, g, b, ambient,
  diffuse, specular, shininess, transparency, emission]` (rgb 0..1), `body.Name = "blue_inlay"`.
  Bodies from `wrap(model, sld.IPartDoc).GetBodies2(c.swSolidBody, False)`.
* **Sketch lines exactly where you put them:** `SketchManager.AddToDB = True`
  while drawing (no snapping or auto-relations to existing edges), then False.
* **Face sketch:** `SelectByID2("", "FACE", x, y, z, ...)` with a model point on
  that face (metres), then `InsertSketch(True)`.
* **Exports** via `Extension.SaveAs3(path.ext, ...)` with `swSaveAsOptions_Copy`
  so the open document keeps its .SLDPRT name. `.STEP` keeps one solid per
  body. SolidWorks' own `.3mf` writes one object holding every body as a
  component, each with its own `basematerials` `displaycolor`, but with
  internal body names (`body7032329`), not ours. The Bambu project 3MF is built
  from the STEP by `cad/aesthetics/lib/export3mf.write_3mf` with the filament slot
  set per part.

---

## The robot assembly, as SolidWorks has it

`ROBOT.SLDASM`, one configuration (`Default`), opens in ~6 s via the API.

* **13 top-level components**, all sub-assemblies except `WheelHanger-1`:
  `BODY-1`, `Femur-1`, `FEMUR_INSIDE-1`, `COUPLER-1`, `Tibia-1`, their `Mirror*`
  twins (the left leg), plus `AK_SIM-1` and `WheelHanger-1`, both hidden, and
  (since 2026-10-04) `Box-1`, the body box ("The body box" above).
* **`AK_SIM-1` and `WheelHanger-1` do not count for collisions** (his call).
* **The left leg has no mates.** No top-level mate references a `Mirror*`
  component, so it is presumably placed by a Mirror Components feature.
  Unverified whether it follows the right leg when that moves.
* **The left FEMUR is `Femur.SLDPRT` itself, flipped, not a mirror.**
  `MirrorFemur.SLDASM` (byte-identical in v4 and v5) holds `Femur-1` ->
  `Femur.SLDPRT`, placed at R_right · diag(1, -1, -1): rotated 180 deg about
  its long axis (checked 2026-10-03). So the left leg carries the styled
  Femur, rotated. `Links\MirrorFemur.SLDPRT` is a derived Mirror Part of
  `Femur.SLDPRT` that nothing uses. Every other left part IS a derived mirror
  of its right twin (`MirrorFemur_inside`, `MirrorCoupler`, `MirrorTibia`,
  ...), so they follow the right parts' styling as true mirrors.
* **The rotated Femur's styling collides on the left leg.** The ON sweep of
  2026-10-03 13:43 (`out/sweep_on.json`, left leg still on) against
  `sweep_off.json`: `MirrorFemur-4/Femur-1 x MirrorTibia-2/MirrorTibia-1`
  0 -> **208.5 mm3 at hip -21**; the left knee bearings 8.91 -> 27.5 mm3 at
  hip 0; two inner-plate screws 3.55 -> 6.62 mm3. The right leg is clean;
  the Femur's flange and keep-outs were tuned against the RIGHT neighbours,
  and rotating (not mirroring) the part moves them. Not acted on (he left
  the Femur part as is, 2026-10-03); a left-hand femur part would fix it.
* **The hip is `LimitAngle1`** (`MateLimitPlanarAngleDim`, assembly Top Plane ↔
  `Femur-1` Top Plane), limits **−28° .. +57°** = firmware Q_RET +28 / Q_EXT −57
  with the sign flipped, 85.00° of travel. **Drive it with `swlib.HipDriver`, never
  through this mate's value** (gotchas 6, 7, 11).
* **SolidWorks hip (deg) → pose:** −28 = Retracted, +19.98 = Middle, +57 = Extended
  (the three STEP exports in `STEP exports/`, reproduced to 0.0000 mm). Femur
  angle in assembly XY = hip − 180°. The hip axis passes through the assembly
  origin, along assembly Z.
* **What moves, hip −28 → +57:** femur and inner femur plate +85.000°, coupler
  +97.13°, tibia −24.68°. The mirrored left leg follows the right one exactly.
* **The two legs are NOT symmetric at the hip motor.** At the Middle pose the LEFT
  AK45 rotor overlaps its stator by 145 mm³ (4 pieces) and the left femur its
  rotor by 8.2 mm³; on the right it is 0 and 0.38 mm³. The mirrored AK45 is
  probably not a true mirror of the right-hand one. Not investigated.

### Baseline sweep, SOURCE parts, 2026-10-02 (`06_sweep.py 1.0`)

25 interfering pairs over 86 poses. **20 are FITS** (constant at every pose):
the 6804 bearings in their seats (8.910 mm³ each, every joint, both legs) and
the screws across the femur ↔ inner-plate bolted joint (2.9–3.6 mm³).
**5 change with the hip:**

| max mm³ | where in the stroke | pair | reading |
|---|---|---|---|
| 147.6 | all 86 poses, worst at −28 | LEFT AK45 rotor × stator | the left-leg mirror asymmetry above; varies because the rotor turns |
| 4.70 | −28 .. −26 (both legs) | limit switch × inner femur plate | **INTENDED** (his answer): the plate actuating the retract limit switch over the last 2° |
| 0.226 | −28 .. −27 (both legs) | coupler × inner femur plate | **INTENDED** (his answer): this contact IS the retracted hard stop |

Nothing else touches anywhere in the stroke: no link hits another link or the
body except at the bearings. **So these two are part of the baseline: a styled
sweep must reproduce them, not flag them.**

**Since then he SUPPRESSED the mirrored left leg** in v5 (he does not want the
AK45 asymmetry analysed). Sweeps from here on are the right leg only.
* Joints: hip `Coincident2`+`Concentric3` (AK45 rotor ↔ stator); knee
  `Coincident1`+`Concentric2` (tibia bearing ↔ femur); coupler–tibia
  `Concentric4`+`Coincident11`; coupler–body `Concentric5`; inner femur plate
  `Coincident17`+`Concentric25` to the femur, `Concentric26` to the shaft.

---

## Where this is going

The goal is SolidWorks as the **collision judge for the styled parts** across the
whole hip stroke: sweep the hip in ~0.5° steps, run Interference Detection and
minimum clearance at each step, write a CSV. Real mates, and SolidWorks reads the
parts with a different CAD kernel from the one that built them.

Done: driving the hip across the whole stroke (`HipDriver`), one interference
check (`05`), the sweep on the SOURCE parts (`06`) — the baseline, results
above. For comparison, the OCC pipeline's `cad/aesthetics/lib/collide.py` takes
~12 min for 21 poses of three links; this does 86 poses of the whole robot in 8.

**Superseded (2026-10-02, his call): no STEP import.** The styling is now built
as NATIVE SolidWorks features directly in the v5 parts (`08`, `09`), every
feature named `GL_*`. That makes the A/B trivial: suppress the `GL_*` features
and the part is the original again, in the same assembly (`10_verify_styled.py`).

## Native GLACIER — how it is built (`08_style_part.py`, `09_style_tibia.py`)

The look is not re-designed: each part's existing recipe `plan()` in
`cad/aesthetics/parts/` gives the outlines (shapely, part-local mm), and the
emitter replays that recipe's `build()` operation by operation as features:

| recipe op | SolidWorks |
|---|---|
| flange band, prism z0..z1 | Extrude Boss from an offset start plane, merged |
| frame / rail / pads, drafted | Extrude Boss with draft; skipped if it never clears the plate (rail and pads are buried in the locked recipe too) |
| pockets, windows, through-cutouts, engraving, back circuit | Extrude Cut |
| side pockets (X-Z profile ∩ band) | band tool body -> Top Plane profile cut, feature-scoped to it, keep inside -> Combine subtract |
| colour inlays (Femur family) | copy body; scoped cuts take the inlay slabs out of white; copy the rest; Combine subtract = exactly the inlay |
| colour cap with fingers (Tibia) | copies + scoped cuts (Top Plane split region, back skin slabs, trace, accent, insets) |

Show face -Z parts (Femur, Coupler) are built mirrored in the recipe; every
height is mapped back with real z = SIGN * recipe z.

**One deliberate change from the locked look: the mechanical seat keep-out.**
The recipe kept only 2.2 mm clear of a hole edge. Every removal, raised feature
and flange is now kept off a seat around each opening -- fastener holes r+3.5
mm (head/washer), medium holes r+4, bearing and shaft bores r+8 (the 6804
retaining washer and its 38.05 mm screw circle) -- and removals another 0.3 mm
beyond that (gotcha 19). What it took back from the old recipe, per part, is
printed by `--dry`; the biggest was the Tibia (side pockets 1236 mm2, pockets
140, engraving 98, frame 104).

Side panel and RobotMount recipes were regenerated from femur.py with
`cad/aesthetics/tools/mkpart.py` (they were pre-Phase-3).
