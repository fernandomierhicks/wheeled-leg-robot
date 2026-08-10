# TODO — sim fidelity gaps found 2026-07-31

Raised after a hardware post-mortem on bench log
`data/logs/runs/20260728T053232_409343Z_HOST` found the hips holding **4.07 N·m**
against the AK45-10's **2.5 N·m rated continuous** torque — a limit neither
MuJoCo sim models, at a leg pose neither MuJoCo sim visits.

**Scope note:** items below span `master_sim_jump/` *and* `master_sim/`. This
file lives here because that's where it was requested; each item names the file
it actually applies to.

Reference numbers used throughout, from the AK45-10 datasheet (now recorded in
`components/COMPONENTS.md`): **2.5 N·m rated continuous / 7 N·m peak**,
**2.1 A / 5 A DC**, Kt 0.127 N·m/A motor-side = 1.27 N·m/A at the output.

---

## 1. No continuous torque rating exists anywhere in either sim

`master_sim_jump/params.py:319` and `master_sim/params.py` —
`HipMotorParams.torque_limit = 7.0` is the **peak** rating. It is emitted as
the MuJoCo actuator clamp (`physics.py:774` / `physics.py:460` →
`ctrlrange="-7.0 7.0"`). There is no rated/continuous field, so nothing in
either sim can ever flag sustained over-rating: every pose sits 3–4× under the
only ceiling that exists, and clips silently pass.

- [ ] Add `torque_rated_nm: float = 2.5` to `HipMotorParams` in both sims.
- [ ] Report **RMS** hip torque per scenario against it (peak-vs-continuous is
      the same split `simulation/2d/fourbar_optimizer_gui` already makes via
      `LinkageSpec.torque_limit_nm`).
- [ ] Keep 7.0 as the actuator clamp — this is a *reporting* gap, not a clamp
      that should be tightened.

## 2. `master_sim` retracted stroke stops 25° short of the design stroke

`master_sim/params.py:39`:

```python
Q_RET: float = -0.78705   # [rad] fully retracted (crouch), 25° closer to Q_EXT
```

`components/COMPONENTS.md` gives Q_retracted = **−0.351**. The comment
documents the shortfall but nothing reconciles it. Hip torque demand rises
monotonically toward retraction, so the balancing sim only ever sees the easy
end of the travel:

| range | τ demand @ 2 kg sprung |
|---|---|
| COMPONENTS.md design stroke (−0.351…−1.432) | 1.45 – **2.16** N·m |
| `master_sim` stroke (−0.787…−1.432) | 1.45 – **1.77** N·m |
| **actual hardware operating band** (−0.434…−0.652) | **1.87 – 2.07** N·m |

The robot ran that entire log at −24.9°…−37.4°; `master_sim`'s retracted limit
is −45.1°. **The balancing sim has never evaluated a pose the robot actually
operates in.**

- [ ] Set `master_sim` `Q_RET = -0.351` to match COMPONENTS.md, **or** document
      why a 25°-reduced stroke is deliberate. Right now the two disagree
      silently and the sim is the optimistic one.
- [ ] `master_sim_jump/params.py:402` has its own `Q_RET = -0.378958` (TEMP,
      see item 6) — much closer to design. Only `master_sim` has this gap.

## 3. Thermal model is defined but never called

`models/thermal.py` and the constants at `params.py:329-336` (`R_eff`,
`C_winding`, `C_case`, `R_th_wc`, `R_th_ca`, `T_max_C = 130`) are real, but
grepping `master_sim/` for any use outside `logs/params_backups/` returns
nothing — neither `sim_loop.py` nor any scenario imports it.

Even wired in, the time constants make it useless for the current scenarios:

- winding→case: `C_winding × R_th_wc` = 30 J/°C × 1.0 °C/W = **30 s**
- case→air: `C_case × R_th_ca` = 200 J/°C × 6.0 °C/W = **1200 s (20 min)**

Jump scenarios are ~3 s events. A continuous-duty thermal problem is
structurally invisible at that horizon regardless of wiring.

- [ ] Either wire the thermal model into `sim_loop.py` **and** add a
      long-duration standing/holding scenario (minutes, not seconds) that can
      actually exercise it, or delete it. A dead safety model is worse than
      none — it reads as coverage that does not exist.

## 4. Coulomb friction is absent or ~20× too small

Hardware evidence (see § Evidence) implies a **1–2 N·m** Coulomb friction band
at the hip — comparable to the gravity load itself.

- `master_sim/physics.py:405-438` — viscous `damping` only (0.001–0.05
  N·m·s/rad), **no `frictionloss` at all**. Cannot reproduce stiction.
- `master_sim_jump/physics.py:719,743` — `frictionloss="0.06"` on `hinge_F` and
  the knee. Right idea, but 0.06 N·m/pivot is more than an order of magnitude
  below what the hardware shows.

This is a build/assembly property, not a design property, so no sim was ever
going to *predict* it. The gap is that neither sim can **reproduce** it even
when told to.

- [ ] After the bench hysteresis test quantifies the real friction (see
      Evidence), set `frictionloss` on all four pivots from the measurement in
      both sims, so control tuning is done against a representative plant.

## 5. `m_box` disagrees between the two sims

- `master_sim/params.py:27` — `m_box = 0.477` kg ("body box + electronics")
- `master_sim_jump/params.py:391` — `m_box = 0.985` kg (itemised: box 210 +
  electronics 244 + battery 275 + mounts 90 + wiring 100 + fasteners 50 +
  connectors 16)

The jump sim's figure is itemised and reconciles with COMPONENTS.md's 2783 g
total; `master_sim`'s does not appear to include the battery. Sprung mass sets
hip holding torque **linearly**, so a 2× error here is a 2× error in every hip
torque the balancing sim reports.

- [ ] Reconcile to one source of truth, ideally imported from a shared module
      rather than duplicated in two `params.py` files.

## 6. TEMP geometry overrides are unreconciled

`master_sim_jump/params.py:386,402-403`:

```python
F_Z:   float = -0.0055     # TEMP: coupler pivot 18 mm above hip axis (A_Z + 0.018). Original: -0.01821
Q_RET: float = -0.378958   # TEMP: auto_stroke_angles for raised F_Z. Original: -0.733038
Q_EXT: float = -1.218537   # TEMP: auto_stroke_angles for raised F_Z. Original: -1.431694
```

This is a **live design study**, not a mistake: F_Z − A_Z = +18.00 mm, which
matches the `archive baseline_18mm.json` preset in the 2-D tool exactly. But
COMPONENTS.md still documents the +5.29 mm build, so the repo currently
describes two different robots with nothing stating which is authoritative.

Torque consequence of the raised-F_Z variant, over its own TEMP stroke, at the
jump sim's own sprung mass (m_box + 2 × hip motor = 1.505 kg → 0.7525 kg/leg):

| geometry / stroke | peak τ | vs 2.5 N·m continuous |
|---|---|---|
| TEMP raised-F_Z, TEMP stroke | 1.68 N·m | 67 % |
| original F_Z, original stroke | 1.52 N·m | 61 % |
| same TEMP stroke @ 1.0 kg/leg | 2.23 N·m | **89 %** |

- [ ] Decide whether +18 mm F_Z is the build. If yes, update COMPONENTS.md and
      drop the "TEMP/Original" comments. If no, revert.
- [ ] Note the margin is only comfortable because sprung mass is 1.5 kg. At
      1 kg/leg the raised-F_Z geometry is at 89 % of continuous before any
      friction — see `archive baseline_18mm` peaking at 2.61 N·m untrimmed over
      the *full* range, where the 2.5 N·m limit actively trims 2.22° of stroke.

## 7. `R_eff` winding resistance is 10× too low — thermal model is inert

`master_sim_jump/params.py:330` and `master_sim/params.py` —
`R_eff = 0.22  # [Ω] phase-to-phase winding resistance (CubeMars datasheet)`.

The AK45-10 datasheet gives motor constant **km = 0.0858 N·m/√W** and
**Kt = 0.127 N·m/A** (motor side). Since km = Kt/√R_pp, the phase-to-phase
resistance is **(0.127/0.0858)² = 2.19 Ω** — a factor of **10** above the value
in both `params.py` files. Looks like a decimal slip.

Confirmed by measurement (2026-08-02): holding 3 N·m draws **0.4 A at 24 V**
(9.6 W). At stall there is no mechanical output, so that is all copper loss.
Predicted with R = 2.19 Ω: `0.75 × (3/1.27)² × 2.19` = **9.2 W = 0.38 A**, a 5 %
match. With the current `R_eff = 0.22`, the model predicts 0.92 W = 0.04 A —
**10× under-reporting motor heating.**

Consequence: even if item 3 wires the thermal model in, it would report roughly
a tenth of the real temperature rise and never trip `T_max_C = 130`. With the
corrected resistance the numbers are actionable — 2.5 N·m rated continuous maps
to a ~45 °C rise (self-consistent with how the rating would be set), while the
4.07 N·m holds from the hardware log map to ~143 °C, past the Class B limit.

- [ ] Set `R_eff = 2.19` in both `params.py` files, with the km derivation in
      the comment so it does not get "corrected" back.
- [ ] Measure `R_th_wc` / `R_th_ca` rather than leaving them as estimates — the
      whole temperature column above rests on an unverified 7 °C/W, and it is
      the number that decides whether a pose is safe to hold.

---

## Evidence

From `data/logs/runs/20260728T053232_409343Z_HOST` (163 s RUNNING), three
steady holds where measured torque = `kp × position_error + tff` exactly, i.e.
genuine equilibria where motor torque equals load torque:

| hold | q_hip | predicted @ 2 kg | measured | ratio | apparent mass |
|---|---|---|---|---|---|
| A (t=0–82 s) | −0.434 | 2.07 N·m | 0.80 | 0.39× | 0.77 kg |
| B (t=87–98 s) | −0.529 | 1.98 N·m | **4.07** | **2.05×** | **4.11 kg** |
| C (t=101–160 s) | −0.652 | 1.87 N·m | 2.83 | 1.51× | 3.02 kg |

Prediction is flat (~1.9–2.1 N·m) across all three poses while measurement
swings 5×. That rules out a mass, CG, or geometry error — those scale every row
by the same factor. The prediction sitting near the **midpoint** of the
0.80…4.07 spread is the signature of a large Coulomb band (`τ_g ± τ_f`),
consistent with holds B and C being reached by extending (friction opposing)
and hold A settling by collapsing (friction assisting).

Predictions cross-checked two ways: finite-differencing `solve_pose`, and the
independent closed-form `kinematics.wheel_jacobian` in
`simulation/2d/fourbar_optimizer_gui` (agreement 1.4e-09 N·m over 240 samples).
Both sims share byte-identical linkage geometry with the 2-D
`archive_baseline1` preset, so the torque model is not in question — only which
poses and limits get evaluated.

**Next physical step:** put the robot on a stand (wheels off the ground) and jog
one hip slowly through its stroke in both directions, logging
`hip_l_torque_nm`. The hysteresis loop width is **2 × τ_f** directly, and will
also show where in the stroke the binding is worst. That measurement is what
item 4 needs.

## Related — already fixed in firmware (2026-07-31)

The reason this went unnoticed on hardware for so long: the AK45-10 MIT reply
field was decoded with the wrong full-scale (±20 instead of ±8) **and**
mislabelled as amps. Hip torque was reported 2.5× high as "A", so 4.07 N·m
displayed as "10 A" and never read as a continuous-rating violation. Fixed in
`firmware/robot_teensy/teensy/lib/HipMotors/hip_motors.cpp`; telemetry field is
now `hip_l/r_torque_nm` in true N·m (`TELEM_VERSION` 12).
