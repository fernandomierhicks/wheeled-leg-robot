# v5 Ai designed — AI sandbox + SolidWorks automation cheatsheet

**This folder is a full copy of `cad/v4 Larger Ball bearings/` that the AI is
free to break** (Fernando, 2026-10-02). v4 stays the master and is never
touched by automation. Not tracked by git (452 MB). The original v4 design
notes (bearings, 2 mm link-to-link clearance, ...) are in
`../v4 Larger Ball bearings/README.txt`.

The scripts that drive SolidWorks live in **`cad/solidworks_api/`** (tracked by
git). This file is the cheatsheet: read it before touching SolidWorks from code.

## ▶ Status, 2026-10-03 — the styled assembly

The five printed parts -- **Femur, Coupler, Tibia, Side panel, RobotMount** --
carry GLACIER as **native SolidWorks features** (`GL_*` in each feature tree),
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

**Open, honestly:** the colour features do not survive suppress/unsuppress or
upstream edits cleanly (gotcha 32 -- re-run `--restyle`); the Tibia's colour
bodies overlap by 10 um films (0.18 %, gotcha on SKETCH shrink); one Coupler
pocket region (305 mm2) was refused by SolidWorks and is not cut; the
inner femur plate (`Femur_inside`, inside the box) and the left leg are not
styled; thin wall is the one gate still failing, as before.

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
| `15_style_washers.py preview` / `[--restyle]` / `render` | the green retaining washers (BearingWasher, SmallBearingWahser, InsideFemurShaft) -> graphite + blue/white circuit inlays, COLOUR ONLY (shape unchanged); 2D preview without SolidWorks; mates; 3MFs; close-ups at the F and E joints | works -- partition exact, 0 mate errors |

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
| washers (BearingWasher, SmallBearingWahser, InsideFemurShaft) | `faces()` in `15_style_washers.py` | `15 preview` (2D, 1 s, no SolidWorks), `15 --restyle`, `15 render` |
| a design rule (seat sizes, margins) | `08_style_part.seat()` / `mech()` | `--restyle` every affected part |
| a NEW part to style | its recipe / spec, plus **add it to the part lists**: `08.PARTS` (recipe, spec, v5 file), `10.STYLED`, `11.PARTS`, `12.PARTS` | as above |

**Then verify, cheapest first:**

1. `11_check_and_export.py <Part>` -- openings 0 changed, no bore missing; writes the fused STEP and the Bambu 3MF (~1 min/part).
2. `10_verify_styled.py 85` -- mates in every assembly, contacts lost, collisions inside sub-assemblies, plus a 2-pose sweep (~5 min). `10 1.0` is the full 86-pose version.
3. Only if the change ADDED material (flange, frame, bosses): `14_sweep_compare.py on`, then `compare`. The OFF baseline `out/sweep_off.json` stays valid while the SOURCE parts are unchanged. If it reports collisions: `13_collision_keepout.py sweep`, then `bodies` (repeat until done), then `--restyle` again (08 / 09 load `out/collision_keepout.json` themselves). Colour-only changes and removals cannot collide, so skip this step for them.
4. `12_render.py` (assembly at 3 poses + each part) / `15 render` -- look at the PNGs in `out/renders/`.
5. `16_check_and_save.py` -- saves everything ONLY if every styled part is
   really styled in memory (gotcha 37). Tell him not to use Save All in
   SolidWorks while a script that toggles styling (`10`, `11`, `13`, `14`) has
   run since the last load.

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

* **12 top-level components**, all sub-assemblies except `WheelHanger-1`:
  `BODY-1`, `Femur-1`, `FEMUR_INSIDE-1`, `COUPLER-1`, `Tibia-1`, their `Mirror*`
  twins (the left leg), plus `AK_SIM-1` and `WheelHanger-1`, both hidden.
* **`AK_SIM-1` and `WheelHanger-1` do not count for collisions** (his call).
* **The left leg has no mates.** No top-level mate references a `Mirror*`
  component, so it is presumably placed by a Mirror Components feature.
  Unverified whether it follows the right leg when that moves.
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
