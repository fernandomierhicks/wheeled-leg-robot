# Aesthetics pipeline

## ▶ The tournament is OVER — the look is locked

**GLACIER**, in `specs/tibia.json` and `specs/femur.json`. See `TOURNAMENT.md`
STATUS for the exact spec and how it prints.

**To continue in a new session, copy the prompt at the top of `HANDOFF.md`.**
That is the current one. The prompt that used to live here drove the design
tournament; that is finished, and re-running it would reopen settled decisions.

The tournament protocol, the rounds and the reasoning are preserved in
`TOURNAMENT.md` and `DECISIONS.md` if a decision is ever questioned.


---

Restyles the robot's printed parts for looks, without touching anything mechanical.
**SolidWorks stays the mechanical master.** This is a downstream, re-runnable layer:
when the mechanics change you re-export the STEP and re-run, and the styling reapplies.

**Interpreter: `C:/Users/ferna/cadenv/Scripts/python.exe`**
(build123d + OCP + shapely + pillow + numpy). Not the repo `.venv` — no CAD kernel there.

```
C:/Users/ferna/cadenv/Scripts/python.exe parts/tibia_board.py  # 2D board, ~4 s
C:/Users/ferna/cadenv/Scripts/python.exe parts/tibia.py        # 3D build, ~60 s
C:/Users/ferna/cadenv/Scripts/python.exe lib/verify.py Tibia   # prove holes survived
```

`TOURNAMENT.md` holds the current state, the round protocol, and the traps already
paid for. `DECISIONS.md` is the running log of every board shown and what was chosen.

## How a part gets styled

1. **Pick a design language** from a board (`parts/<part>_board.py`). Language is the
   categorical choice: angular / exposed / armored / curved / greebled. It must be
   settled first -- tuning axes inside the wrong language just yields variants that
   all look alike.
2. **Tune the axes** within it: aggression 0-10, density 0-2, relief 0-2.
3. **Lock it** to `specs/<part>.json`, then run the part recipe for the 3D build.

`plan(spec)` is pure shapely and feeds the board; `build(spec)` runs the same spec
through OCC. One dict, so a board cell is a contract, not an illustration.

## Layout

| Path | What lives here |
|---|---|
| `lib/` | Reusable, part-agnostic machinery. |
| `parts/` | One recipe per part. Reads a spec, emits geometry. |
| `specs/` | The chosen style parameters — the contract between the 2D board and the 3D build. |
| `input/parts/` | STEP exports of individual parts. Falls back to the live CAD folder. |
| `input/assembly/` | Assembly STEP exports, used for placement and clearance. |
| `out/boards/` | 2D variant contact sheets. |
| `out/renders/` | 3D colour renders. |
| `out/styled/` | Fused styled STEPs — reference imports for SolidWorks. |
| `out/print/` | Per-filament STEPs and the Bambu 3MF. |
| `decisions/` | Archived copies of every board actually shown. Permanent; `out/` is not. |

`out/` is disposable; everything in it regenerates.

## lib/

| Module | Role |
|---|---|
| `paths.py` | Resolves part names to STEP files and owns all output dirs. Import first. |
| `spec.py` | The style spec: design languages + the aggression/density/relief axes. |
| `board.py` | 2D contact-sheet renderer. Light green ground, supersampled. |
| `outline.py` | Plan outlines off a STEP as shapely polygons. |
| `keepout.py` | Every opening in a part, plus its true plan silhouette. |
| `shputil.py` | shapely <-> build123d: faces, prisms, drafted pads, robust N-way fuse. |
| `feat.py` | The trapezoid vocabulary: plan pads, X-Z side-wall profiles, wall shells. |
| `render3d.py` | Tessellation + shaded z-buffer renderer. |
| `export3mf.py` | Bambu/Orca project 3MF with per-part filament assignment. |
| `verify.py` | Compares a styled part against its source, opening by opening. |
| `logdec.py` | Archives a board and appends to `DECISIONS.md`. Run after every answer. |

## The rule that keeps parts printable

Material is only ever **added outside the original silhouette**, and the source solid
is never re-cut. That is what preserves counterbores and bearing seats. `verify.py`
checks it every run and should always report **0 openings changed, worst deviation
0.000 mm³**.

## Two OCC traps

- `extrude(face, amount=h, taper=d)` returns z in **[-h, 0] with the wide end at z=0**.
  It must be mirrored before use as a raised pad, or every boss comes out inverted and
  buried in the plate. `shputil.frustum()` handles this.
- **The top plan face is not the silhouette.** On the Tibia it stops at x=204.7 because
  the wheel hub sits in a recess. Use `keepout.silhouette()`, or added material fills
  that pocket and destroys the hub counterbores.

The full list of six is in `TOURNAMENT.md`.
