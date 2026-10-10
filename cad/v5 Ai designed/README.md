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

## ▶ Relief: colour shapes pressed, grooves on the link sides -- 2026-10-09 (`28_relief.py`)

His brief (4 marked screenshots): "whenever there is a shape of different colour in a part, when
possible make it a recess or a shallow extrusion -- we are 3D printed, complexity is free" (his
examples: the RobotMount step-bar, the FacetBack brow trace, the FacetHood tier-1 brow); "the
widest periphery of the FacetHood is white space with no features -- add aggressive dark grey and
blue features"; "more accents on the sides of the links".  His calls: **PRESSED by default**
(removals only, so nothing new can collide; raised only where nothing passes); scope = the box
parts, the RobotMount field, the encoder parts + wheel hub (NOT the recipe graphite of every
GLACIER link); **recessed grooves with coloured floors** on the link sides; offline renders
first (he approved them as rendered).

**How a part changes (`28_relief.py`, everything at the END of the tree, like 24's pipes):**
`press(face, region, d, colour, t)` -- the region sinks d, on a floor t thick in `colour`; every
body loses what the recess covers, every body of another colour also the floor slab, which
comes back as one new body.  `raise_(face, region, h, colour)` -- a new body h proud.  In
SolidWorks: per changed body its cutter (Imported `RL_toolN_k`) Combine-subtracted (`RL_cutN`,
the pieces re-coloured: a Combine drops the colour), a swallowed body deleted (`RL_goneN`), the
new bodies (`RL_<op>_<n>`, coloured).  **No face a mate uses is touched** -- FacetBack /
FacetFront (brackets, switch) and the TailStrut (`TS_*`) keep their mates; 0 mate errors in
every chain.  Checked against the offline result: body count exact, each body within 0.15 mm3
or 250 ppm by volume, and **`probe`**: the as-built part exported from SolidWorks, 40 points
per op half-way down every recess (must be in no body) and 0.3 into every floor (must be
solid).  Volumes alone could not tell a missed cut from kernel noise: Parasolid vs OCC on the
big cut bodies came out +0.097 / +3.5 / +9.6 mm3 (clamp / FacetFront / Coupler, every other
body 0.0000), while 3300 points sampled round and through the Coupler's 11 grooves, edges
included, agree everywhere.

| part | what | depth |
|---|---|---|
| RobotMount (image 1) | the field's graphite step-bar + the recipe bar it carries on from; the white slashes in its head stay as standing ribs; stops 2 mm short of the trench slot | 1.0 into its own (through-plate) graphite |
| FacetBack (image 2) | blue brow trace -> channel on a blue floor; dark linings of the I/O bay + control pod stepped down; graphite vent slashes pressed on the back face AND the left cheek they wrap onto; NEW: 3 dark raked slots right of the centre screw | 0.9 / 1.0 |
| FacetFront | dark lining of the screen well stepped (its thin sliver stays flush); the two blue cheek traces -> channels | 1.0 / 0.9 |
| TailStrut | blue chevrons (pad flanks), blue window lining (both keel flanks), tail facet line -> channels.  NOT the dark crystal panels: most of their facet, and the keel behind is thinner than floor + 1 mm | 0.9 |
| EncoderCarrier / CableClamp | every inlay >= 0.9 mm wide (magnet ring, centre dot, arm pads; the clamp's frame band, chevrons, pad) | 0.4 on a 0.6 floor |
| Wheel hub | the flush inlays >= 0.9 mm wide **RAISED** 0.6 (white chevron brackets + pads, blue traces): the hub face is a 0.8 mm floor that carries the wheel on its 4 screws -- nothing pressed into it; nothing in the robot is outboard of it | +0.6 |
| Femur, Femur_inside, Coupler, Tibia (image 4) | grooves on the long side faces (`SIDES`): repeating units -- a graphite plate with a 45-deg raked end, a raked blue hatch comb, a blue dogleg channel with a diamond pad; on the tall sides (Tibia, femur flange) a long blue lane beside them | 0.8 on a 0.8 floor |

**Rules the generator keeps (printed when it drops something, never silent):** a shape narrower
than 0.9 mm anywhere stays flush (a 0.4 nozzle cannot print its recess -- the encoder ticks and
bus, the wheel's fine traces, the clamp's bus); a shape touching a face's outer edge is not
pressed unless meant to (`inside=-1` for the linings and vents); solid >= 1.0 mm behind every
floor; a press over an EXISTING inlay grows 0.05 into the body round it and its floor shrinks
0.05 inside it (no tool face on the inlay's own walls -- a floor slab on the RobotMount's
through-plate graphite walls wiped the whole white plate in OCC); every cutter runs 3 mm out
into air.  Link sides: 1.2 mm from every edge, pocket and other colour; exposed (nothing of the
part within 25 mm in front of it); clear of the DESIGNED contacts (`out/contact_keepout.json`
+ 2 mm: the limit switch and the hard stop on Femur_inside / Coupler).

**The FacetHood is not done by 28** -- its source is `hood_variants.variant_B(relief=True)` ("B";
"B0" = the hood as it was), into SolidWorks by the documented reimport -- now
**`21 reimport FacetHood --remate-seats`**: since the fasteners (10-08) the hood's 10 countersunk
screws hold it by coincident SEAT mates (`FS_hood-ring±_L147..151_seat`: screw head plane on the
skirt face), which a reimport breaks; the option records each one's plane (`IMate2.MateEntity`
-> `EntityParams`, Box frame), deletes them, reimports, and re-makes each on the new face in that
plane and the screw's face found by geometry, through 25's `Asm.mate` (kept only if nothing
moves).  2026-10-09: 190 bodies = the STEP (worst 0.0097 mm3), 10 of 10 re-made, 0 mate errors;
then `16 --chain`, `24 export / preview / build FacetHood` (the 3 hoses: 0.00 mm2 off free
ground on the new hood, 196 bodies = predicted).  What the relief is:
* every flush colour shape pressed (`PRESS_D` 0.6 on `FLOOR_T` 0.8, `press()` backs the wall in
  the empty cavity so it stays 2.5): rim rivets, the plates' traces and pads (no backing: the
  plate is solid), the field circuit + combs, the side-slope panels (now a white frame round
  each), their ticks sunk further, the visor brow (visor + combs sunk into it), the exhaust
  bands (slots sunk into them), tier 1's front brow;
* tier 1 (image 3): dark floors in its side / back pockets, blue raked combs between them, dark
  raked vents + a graphite plate with dark slots on the front facet, a graphite plate with dark
  vents on each front corner facet (backing never below `Y_T1_MIN` 86.2: the ring wall);
* the skirt (y 78..86): between the hood screws (`SKIRT_KEEP` 5) on every facet, a graphite
  plate with raked ends pressed 0.6, a row of dark raked slots 0.4 further into its middle, a
  blue comb at its forward end.  **No backing on the skirt** -- the ring wall is 0.3 behind it --
  so 1.0 deep at most (1.5 of the 2.5 wall left);
* `checks()`: below y 99.5 only the skirt's outer skin (away from the screws) and tier 1 may
  differ (`relief_zone()`): 0.0000 mm3 elsewhere -- ring fit, countersinks, ledge, strip
  channel unchanged; envelope unchanged (y <= 117.2, |z| <= 78); 0 loose solids; outside
  overhang 530 mm2 (was 189): the 0.6 mm ceilings of the skirt recesses, skirt-down in print.

**Commands:** `28 export` (read-only: multi-body STEP + colours + placement of each part as it
is in SolidWorks -> `out/relief/src/`), `28 faces <Part>` (its planes by area per colour),
`28 uvmap <Part> nx ny nz d [ux uy uz]` (a face-on render with a mm grid, to design on),
`28 preview` (offline: every op, checks, cutters + new bodies + `<Part>_after.step`, board
`out/relief/<Part>_relief.png`), `28 robot [hip] [--views=..]` (whole robot before | after from
his four camera angles, `out/relief/robot_relief_<n>.png`), `28 build <Part>` (into SolidWorks,
`probe`, `16 --chain`), `28 verify <Part>` (read-only: export as built + `probe`),
`28 export --all` + `28 print --all` (EVERY printed part's Bambu 3MF from SolidWorks as it is
now -- the relief parts refused until built; box parts / wheel / encoder parts in their print
orientation, links and washers in the part frame, bumpers one TPU filament).

**Order matters -- after any of these, re-run what comes after it:** `08/09 --restyle` of a link
-> `24 build <Part>` (pipes) -> `28 export / preview / build <Part>` -> `25 mirror-colours`.
`box_facet_print.py` / `tail_strut.py` + reimport of a box part -> `24` -> `28` the same.
The RL_ features sit after the GL_ and PP_ ones; a restyle deletes GL_ only, and its volume
proof will then fail on the RL_ bodies -- delete the RL_ features first (`28 build` re-adds).

**Built 2026-10-09 (all saved through `16 --chain`, 0 mate errors in every chain):** RobotMount
18 bodies, FacetFront 4, FacetBack 13, TailStrut 20, EncoderCarrier 24, EncoderCableClamp 19,
Wheel 31, Femur 41, Femur_inside 41, Coupler 47, Tibia 69 -- each = the prediction, `probe`
all right (`28 verify` for the first four, built before the probe existed); `25 mirror-colours`:
MirrorFemur 41 / MirrorTibia 69 / MirrorCoupler 47 / MirrorFemur_inside 41 / MirrorRobotMount 18
bodies matched and coloured, chain-saved.  Then (his call) a guarded save-all (`16`, plain) and
**every print file regenerated from SolidWorks** (`28 export --all`, `28 print --all`: the 11
relief parts + Side panel, FacetHood, FacetRing, both bumpers, the 3 washers; same names and
orientations as before -- 19 3MFs in `out/print/`).  Backups of every file the build changed +
the old 3MFs: `_originals/pre_relief_2026-10-09/`.

**Gotchas paid for:**
* **Never drop "debris" by size alone.**  The first build deleted every body under 0.5 mm3 (as
  GL_DropDebris does) and took real colour bodies with it: the encoder's 0.48 mm3 code-wheel
  ticks, a wheel dot, two clamp chevron pieces.  Now only bodies the prediction does not have,
  and only true slivers (< 0.02 mm3); the three parts were rebuilt.
* **`stepcolor.write` drops bodies under 1 mm3** (`MIN_BODY_MM3`, meant for boolean debris): three
  EncoderCarrier floors (0.49-0.89 mm3) never reached SolidWorks and left 0.4 mm voids.  28 sets
  it to 0 for its outputs, and `build` refuses unless the new-body STEP holds exactly the
  predicted count.
* **Check imported bodies against the PREDICTION, not against themselves.**  The wheel's raised
  chevrons were cut by the flush inlay touching them AT the face (a coincident-face boolean:
  one came out at half volume, one bigger than its prism), and each kernel then read the STEP
  differently -- the build passed because it compared SolidWorks with what it had just
  imported.  `raise_` now cuts only by what stands above the face (a probe 0.02 up), and every
  new body is compared with its predicted volume.
* SolidWorks' STEP export of a broken body reads differently in OCC (the same wheel bodies: 12.3
  vs 13.6 mm3) -- `print` falls back to the imported RL_ bodies if a body cannot be matched.
* A body imported from STEP can carry its colour on its FACES only (`colour_from` "face" / "part"
  in the export): a Combine's new faces then show the part's default appearance -- every cut
  piece is given a body colour from the export.
* An OCC boolean probe whose face lies ON a body face returns nothing (the checks read "0 %
  solid" behind a 2 mm sleeve): probes sit 0.05 inside the region and off the face.
* Fusing 18-46 touching colour bodies into one is what OCC fails at (Null shape): volumes
  are summed body by body.

## ▶ Mirror fix: the left leg, the brackets, the ring screws -- 2026-10-08 (`27_mirror_fix.py`)

His brief (4 marked screenshots): fix every wrong fastener; align all the corner brackets; the
left femur "not in the right orientation and has an extra part".  His calls: the left femur a
**true mirror part**; **flip the two front top corner brackets**; keep the left AK45 rotor
(mirrored); "ok to create a mirror part/assembly and mate it manually for the correct motion".

**How it was found:** `25 scan --all` (the left side too -> `out/fasteners/scan_all.json`) +
`25 audit` (OCC, every screw: head buried / not seated, tip in air, axis through solid, doubled)
and every left part against the exact mirror of its right twin (centroid + inertia, robot frame).

| what was wrong | cause | fix |
|---|---|---|
| left femur a TRANSLATED copy: AK45 rotor 17.5 mm outboard (his "extra part"), knee tube + M3x50s sticking out, M2.5s through the rotor | FemurMirror instances Femur.SLDPRT (no rotation of it is its mirror) and a coordinate mate inside MirrorFemur.SLDASM pinned it at identity | **`Links/LeftFemur.SLDASM`**: `Links/MirrorFemur.SLDPRT` (the opposite-hand Femur, 25 bodies, = Femur's z-mirror to 1.7e-5, coloured by `25 mirror-colours`), rotor, 3 M2.5, 4 M3 at the exact mirror, locked |
| left encoder: all 13 screws head-for-tip, the cable clamp turned 1.77 mm (image 1) | TibiaMirror instances them with orientation 0 (MirroredX_MirroredY: R' = S R diag(1,1,-1)) -- a screw is symmetric under a flip of x or y, never z | **`Links/LeftTibia.SLDASM`** (copy of MirrorTibia.SLDASM): clamp + 13 screws at the exact mirror, every part locked to the tibia |
| CornerBracket-30/31 upside down (one flathead through 11.7 mm of the ring, image 3/4); 6 left brackets 0.229 mm off | v3 mirror features make INSTANCES; this bracket is symmetric only under swapping its own x and y (OCC 1.00000), so no instance orientation is its mirror | features dissolved (`DissolveComponentPattern`, names + placements kept); every left bracket = S . P_right . swap(x,y) (configurations NoFH1 <-> NoFH2); locked to its panel (`MF_CB*_lock_*`) |
| 4 top-corner ring countersinks empty (image 4) | the corner hole's screws went with the v3 cap | M3x14 countersunk + an M3 nut in the bracket's hex trap (`MF_nut_*`) |
| the front top corners' trap faced UP, under the ring: a nut there clamps nothing | bracket orientation (v3) | CornerBracket-4 (and its mirror -12) flipped 180 deg about its diagonal: side holes on the same panels, trap down |
| 5 mid ring holes + CB-26's FacetBack hole: the bracket's own 12 mm "M3x8" flathead, 2.1 mm proud (image 4 circle 1) | it sits for a 5.1 mm wall; ring and FacetBack plate are 3 mm | those stand-ins dropped (CB-21 NoFH2, CB-28/30 NoFH12, CB-31 NoFH1, CB-26 NoFH12); M3x10 countersunk flush in each countersink (`FS_ring_L901..909`, `FS_facetback_L911`) |
| limit-switch M2 heads 0.5 mm inside the switch (both sides); left switch 2.25 mm off its mount, its M2s reversed | 25's seat ray landed on a recess round the hole; BodyMirror reflects the switch about its bounding-box middle (x 3.975), its body at the holes runs x 0..5.7 | right M2s raised onto the switch face; left switch reflected about x 2.85 (a bought part: turned over onto the mirrored mount), its M2s mirrored; locked (`MF_*`) |

**The left leg now** (ROBOT): `LeftFemur-1` locked to `MirrorFEMUR_INSIDE-2` (the inner plate it
is bolted to); `LeftTibia-1` pinned like the right tibia -- `MF_Knee_conc/coinc` (its bearing-2 on
the left femur) and `MF_E_conc/coinc` (bearing-4 on the left coupler), the mirrors of
`Concentric2/Coincident1/Concentric4/Coincident11`.  The feature outputs `MirrorFemur-4` and
`MirrorTibia-2` are **suppressed, kept**; the left coupler, inner plate and body are still feature
outputs (they were exact).  The left leg follows the hip: at -28 / 0 / 44.3 / 57 every left femur /
rotor / tibia / clamp / screw / coupler is the exact mirror (0.0000; only the wheel's free spin
angle differs).

**Why not inside the mirror features' sub-assemblies (tried, does not last):** MirrorFemur.SLDASM /
MirrorTibia.SLDASM are REGENERATED by FemurMirror / TibiaMirror on every hip move -- the encoder
screws snapped back head-for-tip, a directly inserted MirrorFemur.SLDPRT was deleted, locks inside
did not hold (measured: rebuilds keep them, `HipDriver.set` undoes them).  And `ModifyDefinition`
on those features deleted MirrorTibia-2 from ROBOT twice (orientations only; then with the instance
and opposite-hand lists re-assigned too) -- recovered by reloading ROBOT each time.  Both are
recorded in 27 (`features` refuses for real; `left` / `femur` are the superseded in-sub fixes).
BodyMirror does not regenerate on a hip move (the body does not move): the switch fix inside
MirrorSwicthMount.SLDASM holds.

**Checked:** mirror check -- every left part the exact mirror except three bought parts (MP30
connector 0.093 mm, AK45 stator 0.057 mm, limit switch 0.056 mm: no mirror image of them exists);
`25 audit`: no reversed, buried, unseated, through-solid or doubled screw, left = right; mate errors
0 in every assembly; Box: brackets, ring, hood, CB-26 fully defined.  Interference at 5 hip poses
(sub-assemblies as blocks), left vs right: limit switch x inner plate 4.702 / 4.702 at -28, hard
stop 0.226 / 0.226, coupler x side panel 0.490 / 0.490 at +57, femur x knee bearings 17.821 /
17.821, femur x inner-plate screws 7.103 / 7.103, every bearing fit equal; the old left femur x
tibia collision (208 mm3 at -21) and **the left AK45 rotor x stator overlap (145-148 mm3, every
pose) are gone** -- it was the feature's rotor, not the stator.  Left-only / right-only pairs left:
the battery (199, his call), the ODrive USB block (right side only), M3 x bracket nut 3.747 vs
4.971 (the nut's hex angle in the mirrored bracket).  BOM 280 -> 288 (`out/fasteners/BOM.md`).

**Gotchas paid for:**
* **A mate on a component that a configuration suppresses reports NO error and holds nothing**:
  `FX_CB26_1_conce_FacetBack` was on CornerBracket-26's flathead-1; with NoFH12 the bracket, the
  FacetRing (concentric-locked to it) and the hood went UNDER and a ring screw's concentric dragged
  them 0.585 mm.  27's `dropped()` re-makes such mates on the bracket's own bore
  (`25.remate_on_bracket`) -- run after any bracket configuration change.
* A new screw mated concentric FIRST can spin about or slide along its own axis: 25's `mate` now
  ignores the spin (`spin_ok`); a slide is still refused -- the seat goes on, a re-run adds the
  concentric.
* `DissolveComponentPattern` works on a `MirrorCompFeat` of instances; feature names can carry a
  trailing blank (`"Mirror side panel "`) -- find them by type.
* swlib's `stdout.reconfigure` defeats `python -u`; never wrap a long SolidWorks run in `timeout`
  (killed mid-run, output lost) -- 27 sets line buffering, run it in the background with a log.

**Open (his call):**
* The v3 panels' left holes (FacetFront, FacetBack, BottomPanel) are 0.229 mm off the exact mirror
  now (they were drawn round the old bracket offset) -- inside an M3 clearance hole; fix in
  `box_facet_print.py` / BottomPanel if wanted.
* Wheel motor M4x25 (both sides) reach 3.2 mm past the modelled stator thread -- check the real
  depth (M4x20?).  Pre-existing.
* Bolt tips 2-5 mm past their nuts inside the brackets (side panel M3x30/35, bumper M3x20/25, ring
  M3x10, the 12 mm stand-ins) -- standard, not changed.
* No full 1 deg sweep run on the new left leg (~60 min) -- 5 poses only (-28, -21, 0, +20, +57).

## ▶ Fasteners, the whole robot, the encoder parts -- 2026-10-08

His brief: (1) a fastener in every hole that needs one, so the BOM says how many
and how long; (2) mirror what needs mirroring -- a full robot on screen; (3) the
EncoderCarrier + EncoderCableClamp in the robot's look.  His calls: the AK45
joints are **M2.5**; **standoffs + screws** under the electronics; the bumper
screws **for real in CAD** ("two levels of fasteners OK"); the **wheel-hub screws
left out, flagged**; encoder palette **A** (dark carrier, white clamp).

**BOM: `cad/solidworks_api/out/fasteners/BOM.md`** (`25_fasteners.py bom`, counted
in ROBOT, both legs) -- 280 fasteners (M2 / M2.5 / M3 / M4 screws, M3 nuts and
washers, M2 / M3 standoffs) + the open items below.

### Fasteners -- `cad/solidworks_api/25_fasteners.py` (its docstring lists the commands)

`scan` -> `holes` -> `lines` -> `profile` -> `plan` -> `build` -> `standins` ->
`left` / `plan-left` / `build --left` -> `mirror-colours` -> `bom`.  `scan` exports one
STEP per part + every placement (COM costs ~10 ms/call in this session: a face walk
of the styled parts never finished); `holes` (OCC) finds every bore (material outside,
>= 300 deg); `lines` clusters coaxial bores of different parts into screw lines and
marks those that already hold a fastener; `profile` reads the material along a line
at several radii (grip, blind ends, counterbore floors, domes); `plan` (GROUPS /
STACKS: model, length, which face the head bears on, the lowest assembly holding
every part the screw joins -- his hierarchy rule) puts each screw on its OWN bore's
axis with the seat refined from the profile; `build` ray-picks the bore and seat
faces BEFORE inserting (a screw in the hole blocks the ray), inserts, sets the
configuration, places exactly, and adds concentric + coincident (or a distance
mate on a domed seat), each refused and rolled back if anything moves.  Idempotent:
a placed fastener is found by its placement.

Added, right leg + box (80), each mated, 0 mate errors:

| joint | fastener | n | in |
|---|---|---:|---|
| encoder carrier -> tibia (3.5 mm skin) | M3x10 button | 5 | Tibia |
| cable clamp -> carrier (4 on domed bosses: distance mate 0.1 to the flat top) | M3x14 | 6 | Tibia |
| encoder PCB -> carrier | M2x6 | 2 | Tibia |
| AK45 stator <- side panel (head in the 5.5 counterbore) | M2.5x12 | 5 | SIDE PANEL |
| femur -> AK45 rotor (head on the femur face) | M2.5x16 | 3 | Femur |
| limit switch -> mount | M2x8 | 2 | SwicthMount |
| tail strut -> floor | M3x14 | 4 | BottomPanelWithAvionics |
| caster axle | M3x30 | 1 | " |
| battery cap -> holder / holder -> floor | M3x30 / M3x6 countersunk | 2 / 2 | " |
| ODrive: standoff 12 + screw (2 of them: + standoff 5 + the brake resistor tab) | M3 MF 12 / 5, M3x6 | 7 + 2 + 7 | " |
| IMU: standoff 8 + screw | M3 MF 8, M3x6 | 2 + 2 | " |
| FacetHood -> FacetRing (as designed) | M3x10 countersunk | 10 | Box |
| bumpers (12 through the bracket nuts, front-centre into the floor, back-centre into CornerBracket-27's nut) | M3x20 x13, M3x25 x1 | 14 | Box |
| OLED -> FacetFront | M2x4 | 4 | Box |

New models in `Common/` (plain shank, no modelled thread; **origin at the head's
bearing face, +Z to the head**): `M2.5 Roundhead` (12, 16), `M2 Roundhead` (4, 6, 8),
`M3 standoff MF` (5, 8, 12; 6 mm male, female through).  v5's own M3 Roundhead /
Flathead have their origin at MID-LENGTH (head face / top at +L/2).

**Stand-ins (`standins`).**  CornerBracket.SLDASM models both of its screws as M3x8
flatheads.  Where a longer screw comes from outside -- the 14 bumper screws and the
side panel's 5 M3x35s into the bracket nuts -- that flathead DOUBLED it (screw inside
screw, 55-64 mm3).  New configurations `NoFH1` / `NoFH2` / `NoFH12` suppress them;
9 bracket instances switched, the mirrored left instances follow their seed (23
flatheads gone).  `FX_CB27_1_conce_FacetBack` held CornerBracket-27's flathead: re-made
on the bracket's own coaxial bore first (moved 0.00000 mm).  **`M3x8 flathead.SLDPRT`
is 12 mm long** (its Length dimension); the BOM says so.

**The left side.**  The four suppressed mirror instances (MirrorFemur-4, MirrorBODY-2,
MirrorCOUPLER-2, MirrorFEMUR_INSIDE-2) are live; Box's hidden left brackets
(CornerBracket-9..12, 17, 18) shown.  The mirror features carried the new Tibia
screws over by themselves and LIST the femur ones, but the left femur and body did
not get them: 10 left screws inserted directly (`plan-left`, `build --left`) -- a
rigid copy of the right one where the left part is an INSTANCE, the robot's mirror
where it is opposite-hand.  Left = right part counts in every leg sub-assembly.
The opposite-hand parts are derived mirrors that carry the bodies but NOT their
colours (left tibia orange, side panel magenta ...): `mirror-colours` gives each
body its source body's colour (matched by volume + area, worst 1e-4 relative; colour
split identical to the source for all six).  On the way: the right **RobotMount's
`PT_trench1/2`** (the white plate and the graphite inlay the 24 trench crossed) had
lost their body colour in the trench Combine -- coloured back (white / graphite).
`out/print/RobotMount_glacier_native.3mf` was written before that: re-run
`24_pipes.py print RobotMount` before printing it.

**The left body was 10 mm wrong (`fix-left-body`).**  With the left leg live, one
interference check gave 137 pairs (6479 mm3) between the box and MirrorBODY: the left
box wall stood 10 mm INTO the box.  Cause: MirrorSIDE PANEL (BodyMirror's copy) had
lost the coincident that holds InsideFemurShaft along the hip axis (right:
`Coincident6` to the side panel) -- the shaft sat 20 mm out and the wrong way round,
the RobotMount (coincident to it) and its M3 Roundhead-16 followed, and the left
Femur_inside bearing was off its shaft.  The three are now at the exact mirror of
their right twins (S @ P_right @ m, m a reflection the part is symmetric under -- the
shaft's local z, OCC: full volume), their 8 old mates deleted, each LOCKED to the left
side panel (`LF_*`); moved 0.00000 mm from target, nothing else moved, 0 mate errors.
After: box x left body 64 pairs, 239 mm3; the left bearing fit 8.91 mm3 back, every
left joint = its right twin.  What is left there is the battery (next) and the known
left-bracket mirror offsets (CornerBracket-30/31 4.5 mm; the others 0.23 mm -- FIXED
2026-10-08, "Mirror fix" above).

**The battery sticks out of the box on the left**: `Battery-1` (BottomPanelWithAvionics)
spans z -77.35..+47.65 while its holder spans -50..+58 -- ~27 mm off its holder,
2.35 mm past the box side into the left wall (199 mm3).  Hidden while the left wall was
suppressed.  Not changed (where the battery really sits is his call).

**Gotchas paid for:**
* **`AddComponent5` ignores ExistingConfigName**: every M3 Roundhead came in 6mm (the
  model's active configuration).  Set `IComponent2.ReferencedConfiguration` after
  inserting, and check it.
* **A dimension object stays bound to the configuration active when it was fetched**:
  fetch it after `ShowConfiguration2`, then `EditRebuild3` (ForceRebuild3 alone left the
  old geometry).
* **NEVER `ModifyDefinition` a mirror-components feature with opposite-hand
  sub-assemblies**: appending 7 screws to BodyMirror returned True and DELETED
  MirrorBODY-2 from ROBOT.  Recovered by `ReloadOrReplace` of ROBOT (nothing had been
  saved).  `mirror-fasteners` now refuses to run.
* Seat faces by ray: one direction at one radius lands on a fillet -- try 8 directions
  x 3 radii and require the plane AT the seat offset.
* An island inside a ring in one inlay sketch (the clamp's chevrons inside its frame
  band) is gotcha 31 again: `26._layers` splits a region by nesting depth.

**Checked:** 06 sweep, 86 poses, right leg (as the baseline), all fasteners in, vs
`out/sweep_trench_1deg.csv`: **0 new, 0 changed**; 9 gone = exactly the doubled
stand-ins (`out/sweep_fasteners_1deg.csv`).  Both legs live, after the left-body fix,
43 poses (2 deg; 1 deg is ~60 min): `out/sweep_bothlegs_2deg.csv` -- no pair involves a
new fastener; the left leg reproduces the right's designed contacts to the mm3 (limit
switch 4.702, hard stop 0.226, side panel x coupler 0.490, every bearing 8.910); the
only extra left pairs are the ones "The robot assembly" already lists for the rotated
left femur (x left tibia 208.2 at hip -20, x knee bearings 27.5, inner-plate screws
6.6) and the left AK45 rotor x stator (147.6, all poses); plus the battery (199) and the
left-bracket offsets as fits.

**Open (his call):**
* Wheel hub -> motor can, 4 per wheel: the hub's bosses pass through 7 mm holes in the
  can and nothing behind them is modelled to thread into -- not modelled, in the BOM
  as open.
* EncoderCarrier hole over the tibia at line 165: the carrier is solid on the head
  side (2.6 mm) -- no screw can go in (5 carrier screws, not 6).
* The femur <-> Femur_inside joint has 4 extra short screws per leg (2 x M3x20 in Femur,
  2 x M3x18 in FEMUR_INSIDE) entering the far end of self-tap bores the M3x50s are
  already in: they clamp nothing.  Left in (counted); remove if not meant.
* ~~Left side, existing mirror set-up: the left femur is still the RIGHT Femur ROTATED;
  the left AK45 rotor turned; the left LimitSwitch ~2.45 mm off its mount.~~ **FIXED
  2026-10-08 ("Mirror fix" above):** true left femur `Links/MirrorFemur.SLDPRT`, rotor and
  all left screws at the exact mirror, the switch on its mount.
* Box `MirrorComponent1` (the v3 lid's flatheads, all suppressed) is in warning 51
  (its mirror plane went with the v3 lid) -- shows as a warning on Box-1 in ROBOT.
  Before this work; untouched.
* ODrive + brake resistor: the resistor tab is 5.15 mm above the ODrive, the standoff
  5 mm (0.15 gap); the IMU sits 8.13 above the floor on an 8 mm standoff.

### Encoder carrier + clamp -- `cad/solidworks_api/26_style_encoder.py`

Both sit on the INBOARD face of the tibia at the wheel axis; what is seen of them is
their inboard faces (all planar).  Colour only: flush 0.6 mm inlays (the 07/15/22
recipe), nothing added, nothing that mates or fits changed -- no collision check.
Palette A (his pick): carrier graphite (the wheel hub it sits in front of) with the
housing face as the ENCODER -- a code-wheel ring of white ticks round a white
"magnet" ring and a blue centre, a blue index tick -- white-padded blue traces up the
arm, blue chevrons on the side tabs; clamp white with a graphite frame band and
chevrons pointing down the cable, a blue 3-line bus up the neck between the four
screws (the cable path), a blue bar at the wide end.  `preview A B` (offline, renders
`out/renders/encoder_style_*.png`), `build [--restyle]`: carrier 19 inlays, clamp 13;
volume unchanged to 0.0001 mm3, box unchanged, mates 0 in error (Tibia, MirrorTibia,
ROBOT), `16 --chain`; STEP + Bambu 3MF (styled face UP, flat back / feet on the bed)
`out/print/EncoderCarrier_glacier_native.3mf`, `EncoderCableClamp_glacier_native.3mf`.
Both are now guarded by `16_check_and_save` (`STYLED_LATER`).  The left side's
carrier and clamp are instances: they follow.

Renders of the whole robot: `cad/solidworks_api/out/renders/robot_full_*.png`.

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
  with `--3mf`), `21 reimport FacetHood --remate-seats` (the screw seat mates, since 10-08),
  `16 --chain Box/Facet/FacetHood.SLDPRT`, `24 export`, `preview`, `build`, `print FacetHood`
  (or `28 export FacetHood` + `28 print FacetHood`).

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
  ROTATES that asymmetric bracket instead of mirroring it. **FIXED 2026-10-08**
  ("Mirror fix"): every left bracket is the exact mirror; the v3 panels' left holes
  (FacetFront / FacetBack / BottomPanel) are now the ones 0.23 mm off.
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
| `21_box_facet.py reimport <Part> [--remate-seats]` / `colour <Part>` | a changed STEP into the existing, OPEN Facet part in place (lock-mated parts only -- or with `--remate-seats`, fastener seat mates recorded, deleted and re-made by geometry; nothing closed or saved), bodies checked against the STEP by volume and coloured from it (3D Interconnect drops STEP colour) | works -- see "Hood B", "Relief" |
| `22_style_wheel.py preview / [--restyle] / verify / render` | the wheel rim's outboard hub face: raised octagon chip + crown inside the screw circle, flush circuit inlays, dark wheel; checks volumes, mates, then OCC verify vs the source STEP; STEP + print-oriented Bambu 3MF | works -- see "The wheel hub" |
| `23_tail_strut.py import / install / check / save / render` | the tail strut (from `cad/aesthetics/parts/tail_strut.py`) replaces the v3 BackWheelSupport in BottomPanelWithAvionics: insert, record + suppress + re-make the old mates by geometry, checks, guarded save | works -- see "The tail strut" |
| `24_pipes.py export / sweep / map / routes [--trench] / preview / robot / build / print` | corrugated half-pipes and trench pipes on the printed parts: offline route finding against the swept keep-out, previews, then Imported PP_ bodies (and PT_ trench cuts) in the v5 parts, checked body by body, `16 --chain`; Bambu 3MFs | works -- see "Corrugated pipes" |
| `25_fasteners.py scan / holes / lines / profile / plan / models / build / standins / left / plan-left / mirror-colours / fix-left-body / bom` | every hole that needs a fastener gets one: OCC hole analysis offline, then insert + mate in the lowest common assembly; the bracket stand-ins; the left leg live + its missing screws + its colours; the BOM | works -- see "Fasteners, the whole robot" |
| `25_fasteners.py scan --all` / `audit` | the whole robot, left side too (`scan_all.json`); every screw checked offline: head buried / not seated, tip in air, axis through solid, doubled (`out/fasteners/audit.json`) | works -- see "Mirror fix" |
| `27_mirror_fix.py brackets / ring / leftleg / switch` (`features --dry`) | the left side as the exact mirror: box brackets (mirror features dissolved, exact mirrors, locked), the FacetRing's 9 screws + nuts, the left femur + tibia as ordinary assemblies (`LeftFemur`, `LeftTibia`) mated for the motion, the limit switches | works -- see "Mirror fix" |
| `26_style_encoder.py preview / build [--restyle]` | EncoderCarrier + EncoderCableClamp: flush colour inlays on their inboard faces (palette A), STEP + 3MF | works -- see "Fasteners, the whole robot" |
| `28_relief.py export / faces / uvmap / preview / robot / build / print` | colour shapes pressed into the parts (or raised: the wheel hub), grooves with coloured floors on the link sides; offline design + checks, then RL_ features at the end of each part's tree (no mate face touched), `16 --chain`; 3MFs | works -- see "Relief" |
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
| encoder carrier / cable clamp look | `design()` in `26_style_encoder.py` | `26 preview A`, `26 build --restyle` (saves via chain, writes the 3MFs) |
| a fastener (add / change length) | `GROUPS` / `STACKS` in `25_fasteners.py` (lines from `25 lines`, seats from `25 profile`) | `25 plan`, `25 build --dry`, `25 build <asm>`, `16 --chain <asm>`, `25 bom` |
| pressed colour / link-side grooves (any part but the hood) | the part's `d_<part>()` in `28_relief.py` (`SIDES` + `_side_pattern` for the links) | `28 export <Part>` (if the part changed in SolidWorks), `28 preview <Part>`, look at `out/relief/<Part>_relief.png`, `28 build <Part>`, `25 mirror-colours` for a part with a left twin, `28 print <Part>` |
| any styled part with a left twin (Femur, Tibia, Coupler, Femur_inside, Side panel, RobotMount) | -- | after its restyle: `25 mirror-colours`, `16 --chain` the Mirror* parts (the derived mirrors keep the bodies, not their colours) |
| anything added to / moved in the right Femur.SLDASM or Tibia.SLDASM | the left twins are NOT mirror-feature outputs any more (`LeftFemur.SLDASM`, `LeftTibia.SLDASM`, "Mirror fix") | mirror the change into LeftFemur / LeftTibia by hand or with 27's `_local_mirror` (S . P . D, D the part's own symmetry); never edit FemurMirror / TibiaMirror from the API |
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
49. **Never `ModifyDefinition` a mirror-components feature that has opposite-hand
    sub-assemblies -- not even orientations only.** TibiaMirror: deleted MirrorTibia-2
    from ROBOT twice (2026-10-08; also with every array re-assigned), as appending to
    BodyMirror deleted MirrorBODY-2.  Recover with `ReloadOrReplace` of ROBOT; save first.
50. **An opposite-hand sub-assembly a mirror feature made is REGENERATED from the
    feature on every move of its source** (the hip): placements reset to the feature's
    instance orientations, directly inserted components deleted, mates inside do not
    hold -- rebuilds alone leave it alone, so a fix looks fine until the leg moves.
    Fix such parts outside it (27 `leftleg`: ordinary copies, the outputs suppressed).

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
* **Since 2026-10-08 the left femur IS a true mirror** (`Links/MirrorFemur.SLDPRT`,
  "Mirror fix" at the top); the rotated `Femur-1` is suppressed in `MirrorFemur.SLDASM`.
  The two bullets below are the history.
* **(history) The left FEMUR is `Femur.SLDPRT` itself, flipped, not a mirror.**
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
**2026-10-08: the left leg is LIVE again** (his request: a full robot on screen --
"Fasteners, the whole robot" at the top).  A right-leg-only sweep now needs the four
Mirror* instances suppressed in memory first (and not saved that way).
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
