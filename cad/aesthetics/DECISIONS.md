# Decision log

Every board shown, what was chosen, and why. Newest at the bottom.
Appended by `lib/logdec.py` — append, never rewrite.

The board images here are archived copies. `out/boards/` is disposable and gets
overwritten when a round is re-run; these do not.

---

## 00 · Ground rules, set before the tournament

*2026-09-18 — from the sessions that built the pipeline*

No board. These came out of discussion and apply to every part.

**Chose:**

| | |
|---|---|
| Palette | **01 Arctic** — white `#F2F4F7` / graphite `#2B2F36` / blue `#1E7BFF` |
| Colour balance | ~50/50 white/graphite, blue **under 1%** |
| Printer | Bambu **X2D**, dual nozzle |
| Deliverables | print-ready 3MF **and** a fused styled STEP for SolidWorks |
| Scope | silhouette reshaping allowed; stiffness explicitly not a concern |
| Untouchable | every hole — diameters, positions, counterbores |

**Said:**

- *"make more aggressive changes to achieve a better aesthetics"* — the recurring
  note across three passes. Led to the aggression axis being anchored 0–10 with
  10 actually shown, so "more" stops being guesswork.
- *"maybe there's a little bit too much blue actually"* — blue cut from 3.6 cm³ to
  1.9 cm³, then held under 1% by volume.
- *"the part should look like it has a very complicated design/machinery inside to
  explain all the shapes"* — drove the trapezoid feature vocabulary and the
  GREEBLED language.
- *"put features on the sides of the shape"* — drove side-wall pockets, which until
  then had no features at all.
- *"the areas marked in red look very plain"* (annotated screenshot) — the single
  most useful piece of direction given. Red marks beat prose every time.
- *"I don't have the artistic way to describe that"* — why the axes are numeric
  dials and the languages are named from his own concept art rather than invented.

**Deferred:** colour balance as a tunable ("more blue than white"). It repaints
rather than relocates, and can only be judged once geometry is settled.

---

## 01 · round 1 - language
*2026-09-18*

![round 1 - language](decisions/01_round-1-language.png)

**Chose:** B+C crossbreed (exposed x armored)

**Said:** Likes B and C - they have large, readable details. Rejects A, D, E: details far too small, hard to see. For C: feature COUNT/density is right, but each feature should be bigger - bigger blue squares, bigger graphite squares. Wants MORE blue accent than currently, and the blue made cohesive - following a line or some logic, because right now the blue reads as random lines scattered here and there.

---

## 02 · round 2 - crossbreed axes
*2026-09-18*

![round 2 - crossbreed axes](decisions/02_round-2-crossbreed-axes.png)

**Chose:** 2B + 3A, blue split midway (accent ~1.1); blend held at even 0.5

**Said:** Likes 2B best - the larger cutouts/features are 'probably the style I'm going for'. Likes the blue QUANTITY and SUBTLETY of 3A; wants blue midway between 2B (1.4) and 3A (0.8), so ~1.1. Did not call a row-1 cell, so the even 0.5 exposed/armored split stands. NEW DIRECTION, the big note: 'you're doing it only in one direction' - all silhouette growth so far is the keel on -Y only. He wants the OUTLINE changed on ALL sides, adding larger features around the whole perimeter, so the part reads as a different shape from the original. Only hard constraints are the circles (the bores at E, C, W).

---

## 03 · round 3 - outline all sides
*2026-09-18*

![round 3 - outline all sides](decisions/03_round-3-outline-all-sides.png)

**Chose:** 1 (unchanged) - rejected all the round outline growth

**Said:** Cell 1 still the favourite; 2-6 'just look fat'. Rejects the SHAPE of the growth, not the idea: 'the outline that you added is too round - you need sharper creases or trapezoidal lines'. Annotated the board with a red polyline over BOTH the top and bottom edges: straight horizontal runs joined at sharp vertices, two plateaus per edge with a step between them, angled ramps at the ends. That is the target silhouette language. Second note: 'the fat black circles also don't look very good' - the graphite collars around the E/C/W bores are too thick. Wants NON-CIRCULAR features next to the bores: 'a little square-like thing, or one edge'. Fatness came from outline inflating the round knee/wheel lobes - that inflation is being dropped.

---

## 04 · round 4 - creases and circles
*2026-09-18*

![round 4 - creases and circles](decisions/04_round-4-creases-and-circles.png)

**Chose:** 1B + 2A

**Said:** 1B (stepped trapezoid creases, top AND bottom) is the best - 'the creases are OK, they're getting to the right shape'. So the absolute-Y straight-plateau approach is correct and the fatness complaint is resolved. For the circles he picked 2A - the SLIM ROUND collars - over the octagonal (2B) and the tabbed (2C). He called them 'the thin black circles' and likes them as-is, so the non-circular/square-tab idea from round 3 is dropped: slimming them was what he actually wanted. NEW IDEA: 'maybe we can adopt those somewhere else' - reuse the thin graphite ring motif as a surface feature elsewhere on the part, not only at the E/C/W bores.

---

## 05 · round 5 - ring motif reuse
*2026-09-18*

![round 5 - ring motif reuse](decisions/05_round-5-ring-motif-reuse.png)

**Chose:** 1 - no extra rings, bores only

**Said:** 'The thin circles only work on the bores, not anywhere else in the part.' The motif is location-specific: it reads as a bearing collar because it sits on a bearing, and loses that meaning as free-floating surface decoration. n_rings stays 0. TIBIA IS SETTLED at: exposed x armored 0.50 blend, aggression 7, density 1.0, relief 1.0, scale 1.5, accent 1.1, outline 0.8 (stepped trapezoid creases top and bottom), slim 2.0 mm round collars.

---

## 06 · built tibia - reference-art critique
*2026-09-18*

![built tibia - reference-art critique](decisions/06_built-tibia-reference-art-critique.png)

**Chose:** REOPENED - does not match the concept art

**Said:** Attached the concept-art robot photo with the whole LEG circled in red. 'The accent and details seem random and not fluid or cohesive like in the attached picture.' Readable differences: (1) reference limb is one or two LARGE unbroken light panels, ours scatters small pads/pockets evenly along the length; (2) reference blue is a SINGLE narrow tapered stripe running LENGTHWISE along the limb axis, ours runs around the perimeter contour and is broken into blocks; (3) reference blue always sits INSIDE a dark recessed channel, never directly on white - that border is what makes it read as a light strip rather than paint; (4) reference graphite reads as MECHANISM showing between covers at the joints, ours comes from an arbitrary Z-split plane. Colour in the reference follows function: white = outer shell, dark = structure underneath.

---

## 07 · round 6 - match the concept art
*2026-09-18*

![round 6 - match the concept art](decisions/07_round-6-match-the-concept-art.png)

**Chose:** 1's blue LINE, but organised + 4's dark channel

**Said:** Keeps the as-built geometry and the as-built blue LINE (cell 1, the upper perimeter spine) rather than the new keel-running axial stripe. Two changes wanted: make that line 'less jagged / more organised', and give it the graphite CHANNEL border from cell 4. Then asked to be shown MULTIPLE DIFFERENT TYPES of line / accent detail - so the next board is a variety board of accent treatments, not another amount sweep. Note the jaggedness source: the spine is built with band_along() off , so it inherits every step of the trapezoid crease outline.

---

## 08 · round 7 - kind of line
*2026-09-18*

![round 7 - kind of line](decisions/08_round-7-kind-of-line.png)

**Chose:** none - redirected to artistic concepts/

**Said:** Pointed at 'artistic concepts/' (two sheets added 2026-09-18 13:29-13:30, Ascento-style build guides). 'We are still missing something related to the desired aesthetics... tweak the ORGANIC lines and details to match that vibe.' The reference 'Leg Link (Lower)' is the tibia's analogue and it is WAISTED: wide circular boss at each end, necked-in slender shaft between, every boss-to-shaft transition a smooth TANGENT CURVE - no steps, no hard vertices. Long tapered stripe down the shaft, concentric rings at the joints. So the gap is the SILHOUETTE, not the accent line drawn on it. Note the tension: round 4 he chose hard trapezoid creases over round ones; the reference is the opposite, so the new board must show both. Also: the reference palette is white #E6E6E6 / secondary blue-grey #6B72B0 / accent #00A8FF - that secondary is FAR lighter than our near-black graphite #2B2F36, which is likely a second contributor to the 'vibe' gap.

---

## 09 · round 8 - organic waisted link
*2026-09-18*

![round 8 - organic waisted link](decisions/09_round-8-organic-waisted-link.png)

**Chose:** 2's rounding (organic 0.8, no waist); accent rejected again

**Said:** Likes 'the outside rounded corners of 2' - so organic=0.8, the MILD rounding, and he did NOT take the waist. ACCENT REJECTED for the 4th time, and this is the useful sentence: 'it just looks like a line. perhaps just elongated trapezoids? wanna give futuristic vibes.' So a continuous stroke is wrong in kind - the accent must be DISCRETE elongated trapezoids, the same vocabulary as the cutouts. 'The trapezoidal cutouts give the correct vibe' - the dark trapezoid pockets are right and should stay. Reconciles with the round-1 'random blue lines' note: discrete trapezoids are fine as long as they share one axis and one skew. COLOUR UNLOCKED: 'feel free to change colors, dark grey to light grey if that is what you need' - so the near-black graphite #2B2F36 can move toward the concept's lighter blue-grey.

---

## 10 · round 9 - trapezoid accent + light palette
*2026-09-18*

![round 9 - trapezoid accent + light palette](decisions/10_round-9-trapezoid-accent-light-palette.png)

**Chose:** direction approved, no cell picked - wants more shape options

**Said:** 'Much better' - the elongated-trapezoid accent and the lighter concept palette (graphite #6B7280, accent #00A8FF) are both approved in principle. Did NOT pick a cell; asked to see other accent SHAPES inspired by five HUD/sci-fi frame reference sheets he attached. Explicitly: 'doesn't have to be closed frames, just the inspiration.' Vocabulary readable in those refs: chamfered elongated hexagons/lozenges, chevrons and arrow stacks, groups of parallel diagonal hatch slashes, L-shaped corner brackets, thin bars with perpendicular tick marks, and lines that step once and continue. All OPEN shapes sharing a common axis - consistent with the trapezoid row that just worked.

---

## 11 · round 10 - HUD accent shapes
*2026-09-18*

![round 10 - HUD accent shapes](decisions/11_round-10-hud-accent-shapes.png)

**Chose:** none - redirected to circuit-trace language

**Said:** Three cyber/circuit-trace reference images. Wants: 'lines that take trapezoidal turns every once in a while, with perhaps the blue following the dark grey at some points, circuit board looking.' Readable vocabulary: (1) long MOSTLY-STRAIGHT runs that jog diagonally to a new level and continue - a circuit trace, not a row of discrete elements; (2) a thin accent trace running PARALLEL and adjacent to a wider dark-grey trace, so the blue accompanies the grey rather than sitting alone - and only over part of the length, not everywhere; (3) small groups of 3 parallel hatch slashes at intervals along the traces; (4) chamfered/45-degree trace ends. Distinct from round 9/10 which were discrete repeated elements; this is ONE continuous routed path with occasional level changes.

---

## 12 · solo run - 62 concepts
*2026-09-19*

![solo run - 62 concepts](decisions/12_solo-run-62-concepts.png)

**Chose:** pending - awaiting his pick

**Said:** Three hours of full creative liberty. Produced 62 concepts on 11 boards across BOTH leg parts, 18 of them built in 3D and verified (0 openings changed, 0.000 mm3 each). Implemented all ten palettes from the concept art's own sheet 5 plus six more; built 18 accent styles; added organic/waist silhouette axes; made the Femur a second part on the same system. Best single board: gal_legset.png - femur and tibia per concept, because a concept only holds if the pair reads as one design. Index in CONCEPTS.md. Nothing locked to specs/.

---

## 13 · solo run - final pick
*2026-09-19*

![solo run - final pick](decisions/13_solo-run-final-pick.png)

**Chose:** GLACIER

**Said:** 'Glacier looks really good - that's the aesthetics we're going for.' LOCKED for both parts. GLACIER = 01 Arctic body with the softer/lighter structural grey (arctic_lt: white #F2F4F7, dark #7E8795, accent #1E7BFF), exposed x armored 0.50 blend, aggression 7, density 1.0, relief 1.0, scale 1.5, accent 1.1, outline 0.8 (stepped trapezoid creases top and bottom), organic 0.8 (mild rounding), collar_t 2.0 slim round collars, accent_style circuit (routed trace with trapezoidal jogs, grey companion trace alongside over part of the run, two hatch groups). Moving to the next phase: he will share all remaining STEP files.

---

## 14 · phase 4 - assembly arrived, scope and show face
*2026-09-19*

**Chose:** Femur rebuilt MIRRORED; scope = Coupler, Side panel, EncoderCarrier + EncoderCableClamp + Switch_Mount; Femur_inside_InsideBox is the mirrored LEFT femur; Mount/WheelHanger/AK_SIM are DEAD

**Said:** No board - answered from the assembly analysis. (1) SHOW FACE SETTLED, and it is NOT uniformly +Z. Walking Extended.STEP transforms: the hip axis is global Z and increasing global Z is OUTBOARD (wheel outermost at Z 169-203, RobotMount innermost at 80-99). Tibia local +Z -> global +Z, so the tibia is correct as built. Femur and Coupler local +Z -> global -Z, so their show face is local -Z. He chose to rebuild the Femur mirrored rather than leave it: same GLACIER geometry, features land on the other side. Needs a show_face parameter in the spec, NOT a redesign. (2) SCOPE: Coupler, Side panel, and the three small brackets (EncoderCarrier, EncoderCableClamp, Switch_Mount). Wheel rim explicitly NOT taken - it is a disc and would need a radial variant of the linear band code. (3) Femur_inside_InsideBox is the MIRRORED LEFT-SIDE FEMUR, not a separate part - it inherits the Femur treatment mirrored, it does not get its own recipe. (4) Mount, WheelHanger and the whole AK_SIM family (AK_SIM, Ak sim rotor, ak sim body, Plug_Ak45sim) are DEAD - leftovers, referenced by none of the three pose assemblies. Do not style them.

---

## 15 · step export - case collision lost two parts
*2026-09-19*

**Chose:** recovered by XCAF extraction from Extended.STEP, no re-export requested

**Said:** Coupler.STEP and "Side panel.STEP" do not exist in "Middle multiple parts/". SolidWorks wrote the ASSEMBLY nodes COUPLER.STEP and "SIDE PANEL.STEP" (0 faces, structure only) and Windows being case-insensitive those filenames blocked the real parts. SwicthMount/Switch_Mount escaped only because the spelling differs. Both parts were pulled out of Extended.STEP via STEPCAFControl_Reader + XCAF in 23 s into input/parts/. DATUMS CROSS-CHECK EXACTLY against CLAUDE.md: Coupler has 4-bolt crosses at x=-84.770 and a Dia26 bearing bore at x=+84.770, 169.54 mm apart = |EF|; the -84.77 end carries a 7 mm-arm bolt cross identical to the Side panel's, so -84.77 is F (body pivot) and +84.77 is E (tibia end). Side panel origin IS A (five holes on the Dia45 AK45-10 stator circle) and its 7 mm cross sits at (-36.420, +37.540), giving |AF| = 52.30 mm - CLAUDE.md verbatim. Both silhouettes come back as ONE polygon, so trap 9 bridging is not needed for either. If the export is ever redone, rename the assembly nodes or export to separate folders or these two vanish again.

---

## 16 · colour split made part-relative, target 70 percent white
*2026-09-19*

**Chose:** white_share replaces split_z; target 0.70 on Tibia, Femur and Coupler; three-value palette kept

**Said:** Prompted by the high res render in artistic concepts/, which he said conveys the aesthetic exactly except for the green. THE DEFECT: split_z is an ABSOLUTE z and does not transfer between parts. It sits 18.5 mm below the Tibia show face, the full 39 mm on the Femur and only 10 mm on the Coupler, so the Coupler came out 28.6% white against the Femur 61.4% - nearly inverted, a dark part between two light ones. Same defect as trap 10, one axis over. FIRST FIX FAILED: white_depth, a fraction of thickness, gave the Coupler 42% and the Femur 74% at the same 0.72 - depth does not map to volume because the cross-section varies through the part. SHIPPED FIX: white_share states the TARGET SHARE and bisects for the split plane; this is the white_share dial TOURNAMENT.md deferred until geometry settled. Coupler tops out at 65%, not 70 - the accent, collars and pocket insets claim the rest even with the plane at the far face. That is a ceiling of the part, not a tuning failure. PALETTE: the reference names FOUR values (matte white, matte mid grey, matte dark grey, gloss blue) where GLACIER has three. He chose to KEEP THREE and not reopen the look. Noted that a fourth filament is not free on the X2D dual nozzle - the third colour already purges, a fourth purges again.

---

## 17 · collisions - clip growth to free space
*2026-09-19*

**Chose:** clip growth against an assembly-derived keep-out, 1.0 mm clearance; port the Tibia too

**Said:** lib/collide.py found 25 NEW collisions caused by styling, across all three poses. Worst: Femur x Tibia 12969 mm3 and Coupler x Tibia 11556 mm3 in Retracted, which is the Q_RET +28 deg hard stop, so the three poses do bracket the travel rather than missing the extreme between samples. Also Femur x Coupler up to 6307, Femur x AK45-10 Stator ~1950, Coupler x AK45-10 Stator 1339, Tibia x wheel Stator 792. CAUSE: the three links share lateral bands (Coupler global Z 111-146, Femur 123-162, Tibia 123-150) and clear each other IN PLAN only, with little margin. Styling grew each part 6-13.5 mm in plan and +3.5 mm in Z - the raised frame stands proud of the show face because frame_h 7.86 exceeds the 4.38 chamfer. The motor hits are different: the hip stator is a D53 cylinder at the origin and the end bosses grew out over it. CHOSE prevention over reduction: lib/asmkeepout.py derives, per part and per pose, where the metal may not go, and plan() subtracts it from the grown outline before the envelope is built. Rejected turning growth down globally (thins the look everywhere including where there is room) and growing outboard only (drops the trapezoid outline growth chosen in round 4). TIBIA: he asked for it to be ported. Done NARROWLY on purpose - backported white_share, the keep-out clip and the silhouette guard, but LEFT ITS BAND LAYOUT ALONE. Generating it from femur.py would move every feature band (frame x 64-192, pads 66-194, split fingers (58,102)/(138,182)) onto the femur fractions and silently redesign a part approved in decision 05. Flagged to him; the full regeneration is still open if he wants it.

---

## 18 · the styling was cutting into the source part - pre-existing
*2026-09-19*

**Chose:** clamp env to contain prism(OUT, ZB, ZT); verify.py now measures source material removed

**Said:** FOUND while checking why the collision-clipped Femur came out SMALLER than its source (74.73 vs 86.74 cm3). The 4.4 mm chamfer bevels the outer edge of the grown outline, and body = union(...) & env means that wherever growth is thinner than the chamfer the bevel lands on the SOURCE and cuts it away. MEASURED, clipped build: Femur 21594 mm3 / 24.90%, Coupler 15129 / 20.37%, Tibia 31973 / 14.02%. THE COUPLER WAS NET BIGGER (80.56 vs 74.27) while losing a fifth of its original material - net volume hides this completely. AUDIT OF THE APPROVED PARTS, unclipped, i.e. the configuration locked in decision 13: Femur 14320 mm3 / 16.51%, Tibia 29092 mm3 / 12.75%. So this is PRE-EXISTING; the collision clipping only made it worse and visible. It went undetected because verify.py tested OPENINGS ONLY: every one of these builds reports 0 openings changed, worst deviation 0.000 mm3, because no bore happened to sit where the bevel ran. The brief has always said material is only ever ADDED outside the original silhouette; nothing tested it. FIX: verify.py now prints 'source material REMOVED' on every run, and env is clamped with prism(OUT, ZB, ZT) so the bevel can only ever touch added material. Cost is a square original edge wherever growth is thinner than the chamfer. He chose clamp-now-then-z-layer; z-layered growth is the way to get the growth, and therefore the chamfer, back.

---

## 19 · CORRECTION to 18 - most of that removal is deliberate
*2026-09-19*

**Chose:** the clamp stands; the metric in 18 was measuring the approved look as if it were a defect

**Said:** Entry 18 reported 14-32 cm3 of 'source material REMOVED' per part and called it a violation of 'material is only ever ADDED'. That framing is WRONG and this entry supersedes it. build() removes source material in FIVE deliberate places - pockets, windows, full-depth trapezoid cutouts, side-wall pockets and the accent engraving - and those are the GLACIER look: he chose the trapezoid cutouts explicitly in decision 09 ('the trapezoidal cutouts give the correct vibe') and decision 00 records 'silhouette reshaping allowed; stiffness explicitly not a concern'. The metric added to verify.py summed ALL removal, so the 16.51% on the approved Femur and 12.75% on the approved Tibia are mostly his own approved features, not a hidden defect. WHAT SURVIVES from 18: the ENVELOPE had no business cutting the source - it exists to bound ADDED material, and wherever growth fell below the 4.4 mm chamfer it bevelled the real part. That is now measured on its own in build(), as solid - env, where env is actually in scope, instead of being inferred from a whole-part volume delta. verify.py's number is relabelled as a TOTAL with its components named so it cannot be misread the same way again. The env clamp - env = union(env, prism(OUT, ZB, ZT)) - stays: it costs a square original edge where growth is thin and makes the guarantee structural rather than a tolerance that holds while growth exceeds 4.4 mm.

---

## 20 · paint on a pig - added material must read as structure
*2026-09-20*

**Chose:** growth becomes ONE FLANGE, placed where it can attach; all five parts grow, zero collisions

**Said:** HIS CALL, on a render of the Femur seen end-on: the original part was there with two thin full-depth walls stuck to its sides. 'The aesthetics are really good. But the part doesn't look functional... it looks fake.' Then, when I over-corrected to fully subtractive: 'The whole idea was for you to be able to add material. Just add in a way that makes aesthetic and collision free sense.' ROOT CAUSE: out_h is 4.5 mm and the Femur is 39 mm deep, so growth extruded over the full depth is a 1:9 FIN. Confined to the outer third it is a 1:2.8 flange and reads as a thickened edge. I ALSO CONCLUDED WRONGLY, mid-session, that the robot had no room to grow at all -- the free-space survey (scratchpad/freespace.py) disproved it: a 6 mm band outside the silhouette is 85% clear on the Tibia, 41-74% on the others, 20-28% on the Femur. What is blocked is the PIVOTS, which is exactly where `bosses` tried to grow, so the only growth that ever survived came through as one lone sliver. The shaft edges were open the whole time. FOUR SHAPES WERE BUILT AND MEASURED BEFORE ONE WORKED: (1) full-depth band = the fin; (2) per-layer tapered bands -- each starts at the SILHOUETTE while the real cross-section at depth is smaller, so it fused to nothing and the Coupler threw away its whole 13.8 cm3 flange; (3) per-layer bands grown from each band's OWN section -- the section changes with depth so every band juts out somewhere different: a stack of SHELVES, 44 cm3 of trays; (4) one prism pinned to the show face -- fine on the links, but a plate whose top face is recessed has its RIM lower down, so the ring had no metal to attach to and was dropped whole. SHIPPED: one prism, width from the spec, placed at the show face if it attaches there and otherwise at the band holding the most metal. RESULT: every part grows (Femur +8.0 mm Y, Tibia +12.6, Coupler +6.2, Side panel +7.2, RobotMount +6.0), all five verify 0 openings changed / 0.000 mm3, and collide.py reports NO NEW COLLISIONS in any pose. CAVEAT HE SHOULD KNOW: on the Coupler, Side panel and RobotMount the flange lands on the INBOARD face, because those parts have no metal at their show-face perimeter. Material is added and it is structural, but it is not seen. Opening that up means loosening the keep-out (PAD_Z 6 mm, 1.0 mm clearance, neighbours taken at maximal styled size) -- margin he chose, so his call.

---

## 21 · bugs found by looking at renders, not at numbers
*2026-09-20*

**Chose:** five defects, each invisible to the checks that existed

**Said:** Recorded because every one of these passed the verification of its day. (1) verify.py had TRAP 7 IN IT -- import_step(styled).solids()[0] measured ONE fragment of a multi-solid body. Every part built that day is multi-solid (Femur 2, Coupler 2, Tibia 4, Side panel 3, RobotMount 2). The Side panel read 0.39 cm3 against a 92.01 cm3 source and reported nine false 'changed' openings, while a genuinely broken part could have passed. (2) THE ENVELOPE WAS EATING THE SOURCE: `union(solid, add) & env` lets the 4.4 mm chamfer bevel the real part wherever growth is thinner than the chamfer -- 14-21 cm3 per part. The Coupler was NET BIGGER while losing a fifth of its original metal, so a volume delta hid it completely, and an openings-only check saw nothing because no bore sat where the bevel ran. Fix: `union(solid, add & env)`. (3) RAISED PADS EXTRUDED FROM A FIXED Z floated wherever the real surface sat lower -- a detached plate with two prongs under the Coupler. Fix: clip to _top_face(). (4) `_clip` ends with ShPoly(...exterior), which DISCARDS INTERIOR HOLES, so a neighbour whose keep-out fell inside the growth region had its hole filled straight back in. That let the Tibia drive 1363 mm3 into the Coupler beside their shared pivot in all three poses -- identical in every pose, which is what gave it away. (5) THE FLANGE INHERITED THE WIDTH OF THE WHEEL BOSS: I used Wmax, the part's maximum growth anywhere, as the uniform flange width, so the Tibia's edge flange came out 16 mm instead of 7. THE LESSON: renders caught 1 and 3; the collision sweep caught 4 and 5; the 'filament bodies must sum to the part' guard caught a 3MF holding 20 cm3 of a 125 cm3 part. None of these were caught by the check that was supposed to cover them. Guards now in build(): envelope removes nothing, fuse loses no volume, filament bodies partition the part, detached pieces reported WITH THEIR BBOX, flange funnel printed as ring -> design -> clear -> attached.

---
