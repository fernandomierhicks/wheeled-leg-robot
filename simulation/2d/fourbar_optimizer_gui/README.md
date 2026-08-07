# fourbar_optimizer_gui

2D design lab for the wheeled-leg 4-bar: live geometry editing, real link
shapes, collision-limited range of motion, and traced-point path analysis.

Status: **core + GUI complete and verified.** Optimizer not yet added.

## Why

The existing tooling split the problem and never joined it:

- `simulation/2d/step06_mechanism_comparison.py` optimizes wheel-path
  straightness but has zero awareness of bearings, link widths or collisions
  (`MOTOR_R` there is a drawing radius, never a constraint).
- `simulation/mujoco/archive/4bar_optimization_with_balancing/` ran 70,818
  candidates but scored **jump height alone**. Its two mechanical checks
  (`MIN_TIBIA_SPAN_MM`, `MIN_AF_CLEAR_MM`) are logged as warnings and
  **never fail a run** — see `eval_jump_balanced.py:236-248`. The module
  docstring at `eval_jump.py:4-7` claims they are enforced; it is stale.

So baseline-1 was selected without anyone checking whether the links clear each
other through the stroke. This package answers that.

## Run

```bash
python simulation/2d/fourbar_optimizer_gui/gui.py            # the app
python simulation/2d/fourbar_optimizer_gui/api.py eval       # metrics as JSON
python simulation/2d/fourbar_optimizer_gui/api.py sweep --csv out.csv
python simulation/2d/fourbar_optimizer_gui/test_baseline.py  # verification
```

Dependencies: numpy, scipy, matplotlib, PyQt6 — all already installed. No
physics engine: the kinematics are closed-form, which is what will make
optimization runs cheap.

## Layout

| File | Role |
|---|---|
| `model.py` | `LinkageSpec` — every dimension, collision matrix, trace points |
| `kinematics.py` | closed-form IK, assembly-branch continuity, link frames |
| `geometry.py` | disc-hull plate shapes, SAT, circle tests, broad phase |
| `evaluate.py` | collision-free range + travel/verticality metrics |
| `api.py` | headless facade and CLI |
| `gui.py` | PyQt6 app (the only file that may import Qt/matplotlib) |

**Layering rule:** `model/kinematics/geometry/evaluate/api` never import PyQt6
or matplotlib. Test 9 in `test_baseline.py` enforces this.

## Frame

`+X` forward, `+Z` up (per CLAUDE.md). **The hip motor A is at the origin.**

The mujoco archive expresses F relative to the *body centre* with the motor at
`A_Z = -0.0235`, so converting baseline-1 into this frame gives:

```
F_X = -58.87 mm      (unchanged; A was already at x = 0)
F_Z = -18.21 - (-23.50) = +5.29 mm      <-- NOT -18.21
```

That +5.29 mm is the true hip-to-coupler *height* offset. `LinkageSpec.
to_body_frame_fz()` converts back, and the GUI shows both.

## Model

Three moving bodies, each defined in its own local frame as a node list, so
shapes are rigid and precomputed once:

| Link | Origin | +x_local | Nodes |
|---|---|---|---|
| femur | A | toward C | A, C |
| tibia | C | toward E | E, C, W(-L_tibia, `w_perp`) |
| coupler | F | toward E | F, E |

The closed loop is A-C-E-F. W rides on the tibia and is not in the loop, so
`w_perp` (a dogleg tibia) costs nothing in solve time.

A link's collision body is the **convex hull of the discs at its nodes** — the
tapered slot plate shape. Radii are set by hand per node (bearing OD/2 + wall)
and are inputs, never optimized. The polygon that is tested is the polygon that
is drawn.

### Collision pairs

All 15 pairs are individually checkable in the GUI. Off by default:

| Pair | Why off |
|---|---|
| femur / tibia | share the knee pin C |
| tibia / coupler | share pin E |
| femur / motor | the femur rotates on the motor's own output shaft |
| tibia / wheel | the wheel is bolted to the tibia |
| femur / wheel | per the build, the tyre runs clear of the femur plate in Y |
| tibia / wheel_motor | the hub motor is bolted to the tibia at W as well |
| wheel / wheel_motor | concentric by construction, so always overlapping |

Left on: **femur/coupler**, **coupler/motor**, **tibia/motor**, **wheel/motor**,
**coupler/wheel**, and the hub motor against **femur**, **coupler** and the hip
**motor**. The Ø52 mm hub motor is wider in Y than the tyre, so unlike the wheel
it does foul the femur plate — that is why femur/wheel is off but
femur/wheel_motor is on.

### Range of motion

The **contiguous** interval containing the seed pose: a mechanism can only
reach a configuration by moving there continuously, so a collision-free island
beyond an interference is unreachable. March outward from the seed, stop at the
first singularity / collision / user limit, bisect to 0.05°.

Verticality is max |x - mean x| over that range (best-fit vertical).

### Hip torque limit

A pose can be perfectly reachable and still be one the motor cannot hold, so
the range is also trimmed by static hip torque. With the wheel on the ground
and this leg carrying `leg_load_kg` of the body:

```
tau(q) = leg_load_kg * g * |dz_W/dq|
```

Only the **vertical** Jacobian appears: the wheel rolls, so W's horizontal
motion does no work against a vertical ground reaction. The torque is therefore
the load times the leg's vertical gear ratio, and says nothing about how far
the leg travels.

`dz_W/dq` is closed form (`kinematics.wheel_jacobian` — the loop constraint
|E-F| = Lc differentiated directly), so there is no step size to tune and no
extra IK solve; the existing 1° march plus 0.05° bisection does the
discretisation. Poses over `torque_limit_nm` block exactly like a collision,
so the trim flows straight into `stroke_deg`/`travel_mm` and hence into the
optimizer's fitness.

Defaults: 1.0 kg/leg (2 kg body, motors included, on two hips) against the
AK45-10's 2.5 N·m **continuous** rating — a standing robot holds this torque
indefinitely, so continuous is the number that binds, not peak.

This constraint is not cosmetic. The previous optimizer winners in `presets/`
need **4.9-5.9 N·m**, over twice what the motor can hold, and lose more than
half their travel once it is enforced:

| preset | stroke off→on | travel off→on | peak tau |
|---|---|---|---|
| archive baseline_18mm | 62.34° → 60.13° | 219.4 → 209.3 mm | 2.61 |
| archive_baseline1 | 67.41° → 67.41° | 209.4 → 209.4 mm | 2.34 |
| optimized_5mm_dev | 51.41° → 28.09° | 235.5 → 98.4 mm | 4.93 |
| optimized_5mm_nocollision motor_hip | 51.97° → 29.06° | 237.0 → 97.5 mm | 5.29 |
| optimized_all_links_200 | 55.50° → 30.00° | 260.5 → 101.5 mm | 5.88 |

## Verified

`test_baseline.py` — 25/25 checks pass:

- **IK matches the archive `solve_ik` to 2.5e-16 m** across 129 poses, after the
  A_Z frame shift. This is the load-bearing test.
- Loop closure |E-F| = Lc holds to 5.6e-17 m.
- Branch continuity across the sweep.
- SAT and circle/polygon primitives, incl. vertex and near-miss cases.
- With collisions off, the range brackets the archive's 61.93° stroke (ours is
  wider because the archive trimmed 3° of margin per end and clamped `Q_ret`).
- Dogleg offset moves W by exactly `w_perp` without breaking closure.

## Current build

`presets/current_build.json` — the dimensions actually being built:

| | length | end radii |
|---|---|---|
| femur A→C | 174 mm | 20 mm at A, tapering to 15 mm at C |
| tibia | 130 mm C→W, 35 mm C→E | 20 mm throughout |
| coupler F→E | 150 mm | 20 mm at F, 15 mm at E |
| wheel | Ø112 mm at W | checked against the hip motor and the coupler |
| wheel_motor | Ø52 mm at W, concentric with the wheel | checked against every link except the tibia, plus the hip motor |
| motor | Ø53 mm at A | |

Radii are **per link per end**: two links meeting at one pin are separate parts,
so the femur can be 15 mm at C while the tibia is 20 mm there.

## Headline result

| | stroke | travel of W | max dev | limited by |
|---|---|---|---|---|
| no collisions | 78.66° | 251.1 mm | 52.9 mm | singularity both ends |
| all pairs on | 38.53° | 110.3 mm | 34.1 mm | **coupler/femur** both ends |
| femur/coupler masked | 68.41° | 212.3 mm | 47.3 mm | singularity / **wheel–motor** |

Two findings:

1. **femur/coupler is the only link-on-link pair that ever fires.** Mask it and
   nothing else between the plates binds — not coupler/motor, not tibia/motor.
   The two bars are the parallel sides of the 4-bar and run face-to-face with
   ~35 mm of combined width across a ~59 mm pivot gap, so as drawn they overlap
   through most of the sweep. **If they sit on different planes in CAD this
   constraint is fictional — untick it.** `presets/current_build_fc_masked.json`
   is that variant.
2. **The Ø112 mm wheel hits the hip motor at retraction**, costing 10.25° and
   38.8 mm of travel even with femur/coupler masked. That is a genuine limit
   nothing in the repo previously modelled.

## Optimizer

`optimize.py`. Objective is **vertical travel of the traced point**, with
straightness handled either way, chosen at run time:

| mode | fitness |
|---|---|
| `constrained` | `travel_mm`, minus `penalty` per mm that `max_dev` exceeds `tol_mm` |
| `weighted` | `w_travel*travel_mm - w_vert*max_dev_mm` |

Free: `L_femur, L_stub, L_tibia, Lc, F_X, F_Z, w_perp`. Fixed: motor at the
origin, every collision radius. Two algorithms — scipy `differential_evolution`,
and a `(1+lambda)` ES with 1/5-success sigma adaptation mirroring
`optimize_balanced.py:33-51` for continuity with the archive.

~1.7 ms per candidate, so ~10k evaluations in under 20 s.

**Auto-seeding.** A perturbed geometry often puts the nominal `q_seed` inside a
collision, which would reject an otherwise good mechanism (only 18% of random
candidates survived without this). `find_range(auto_seed=True)` relocates the
seed to the nearest usable angle, so a candidate is judged on its best assembly
rather than one arbitrary pose. The GUI's manual view keeps `auto_seed=False`
so what you see is the pose you asked for.

```bash
python api.py opt --preset presets/current_build.json \
    --algo de --budget 10000 --mode constrained --tol 5 \
    --out best.json --csv runs.csv --progress
```

`--out` writes a preset the GUI opens directly. In the GUI the Optimizer tab
runs the same code on a `QThread`, live-previews the incumbent on the working
panel, and snapshots your starting config to the reference panel so you can
watch the search pull away from it.

### Link length cap

No link may span more than `max_link_mm` (default **200 mm**),
radius-centre to radius-centre. For the tibia that is the full E→W span
(`L_stub + L_tibia`), which per-variable bounds cannot express — so it is also
enforced as a graded constraint in `score()`, checked before the sweep. Current
build spans: femur 174, tibia 165, coupler 150 mm.

### Convergence

Both algorithms are **stochastic and derivative-free** — there is no gradient
and no optimality proof. The objective is not differentiable anyway: collision
boundaries, singularities and the validity cliff make it piecewise with jumps,
so a gradient method would stall at the first discontinuity.

Two convergence signals are reported:

- **Stall counter** — evaluations since the last improvement, as a fraction of
  the run. `OptResult.convergence_note()` turns it into a verdict; a run still
  improving at the cutoff says so explicitly.
- **Restart spread** — the real evidence. `optimize_multi()` runs N independent
  seeds; `MultiResult.verdict()` calls it converged under 2% spread, not
  converged over 10%.

```bash
python api.py opt --preset presets/current_build.json --restarts 5 --budget 8000
```

### Watch out

- **Bounds bind.** Runs routinely pin `F_X` and `F_Z` at their limits. Set them
  to real body-geometry limits or the "optimum" is just the corner of the box.
- **Best-fit vertical has no opinion about *where* the path sits.** A 1500-eval
  run reached 4.57 mm deviation partly by sliding the whole leg 96 mm behind the
  motor (`mean_x_mm = -96`). Straight, but possibly unmountable. Constrain
  `F_X` tightly, or switch the verticality reference if this matters.

## Comparison view

Two canvases side by side on **one shared fixed viewport** (same scale, same
origin, computed once over the whole sweep so nothing shifts while scrubbing).
Left is a reference picked from a dropdown of everything in `presets/`; right is
the working config or the optimizer's incumbent. The slider is **% of each
config's own reachable range**, so mechanisms with different strokes stay
comparable, and it overshoots each limit by 1° so the blocking pose renders with
solid red fills. "★ Save as favorite" snapshots the working config into the
dropdown.
