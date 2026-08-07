# Hip retracted/extended angles — as-built geometry, calibrated span, and roll authority

**Date:** 2026-08-03
**Source material:** GUI host log `data/logs/runs/20260802T174130_628413Z_HOST/host.jsonl`
(1613 TELEM @ ~48 Hz, 33.4 s, `RUNNING` throughout, zero faults) + CAD
measurements + 4-bar kinematics (`simulation/mujoco/master_sim/physics.py`).

**TL;DR.** The robot was built with an **18 mm** hip-to-coupler height offset,
not the 5.29 mm in every model in this repo — a transcription error that reached
CAD. Separately, the firmware is configured for **80°** of hip travel while the
mechanism has deliberate hard stops at **66°**, so `gain_sched_alpha` never
exceeds ~0.825 and every `*_ext` scheduled anchor is unreachable. The roll
controller's code is correct; it saturated because `roll_offset_max = 0.15`
buys only ~11° of authority against a 15–17° disturbance.

---

## Finding 1 — As-built geometry: F is 18 mm above A, not 5.29 mm

`−18.21 mm` is F's Z coordinate in the **body-centre** frame from the baseline-1
optimisation. In that design the hip-to-coupler **height** offset (A→F) was
`A_Z − F_Z = −23.5 − (−18.21) = −5.29 mm`. The two were conflated: the body-frame
coordinate was reused as if it were the A→F offset, and that went into CAD. The
robot was built to it. **18 mm is now physical truth, not a bug to correct.**

`simulation/2d/fourbar_optimizer_gui/model.py` had caught this once — its header
literally read `F_Z = -0.01821 - (-0.0235) = +0.00529  <-- NOT -0.01821` — but the
warning was written after the CAD was already wrong.

Corroborating evidence already in the repo before this session:
- `presets/archive baseline_18mm.json` carries `F_Z: 0.018`
- `master_sim_jump/params.py` already had `F_Z = -0.0055` marked `TEMP`

### Measured vs repo

| Dimension | Repo | Measured (CAD) | |
|---|---|---|---|
| femur A→C | 173.78 mm | 174 mm | OK |
| coupler F→E | 150.81 mm | 150 mm | OK |
| stub C→E | 35.13 mm | 35 mm | OK |
| tibia C→W | 129.39 mm | 129 mm | OK |
| F_X | 58.87 mm | 59 mm | OK |
| **A→F height** | **5.29 mm** | **18 mm** | **WRONG** |

Only one dimension was wrong. The repo's more precise values were kept for the
other five.

### Consequence

The 4-bar's non-singular hip range shrinks from **83.6°** (`q ∈ [−1.523, −0.063]`)
to **75.9°** (`q ∈ [−1.301, +0.023]`). Hard stops limit actual travel to 66°,
which fits — but only for `Q_RET ≥ −0.149` (see [Open question](#open-question)).

---

## Finding 2 — The calibrated hip span is configured, not measured, and is 14° too large

`define_limits()` in `firmware/robot_teensy/teensy/lib/Calibration/calibration.cpp`
only *finds* the retract switch. The extended end is simply a fixed offset away:

```c
const float retracted = away_dir * param_get(PARAM_CALIB_BACKOFF_RAD);
const float extended  = away_dir * range_rad;      // calib_range_l/r_rad
limits.min_rad = fminf(retracted, extended);
limits.max_rad = fmaxf(retracted, extended);
```

So the span is `calib_range − calib_backoff` = **85° − 5° = 80°**, never checked
against the hardware. The mechanism has **66°**.

Confirmed independently from the log: regressing `gain_sched_alpha` against the
two hip positions recovers `span_L = span_R = 1.396 rad = 80.0°` with
**R² = 1.000** (residual sd 3.4e-6) — an exact inversion of the firmware's own
`measured_hip_alpha()` formula.

### Consequences

- `t = 1` from `hip_cmd_to_setpoints()` commands the leg **14° past the extended
  hard stop**. The motor pushes into the stop and holds there.
- `gain_sched_alpha` saturates near **0.825**, never 1.0. The extended anchor of
  every α-scheduled pair — `lqr_k_pitch_ext`, `lqr_k_rate_ext`,
  `hip_running_tff_ext`, `lqr_pitch_trim_ext`, `lqr_barrier_th_ext` — is
  **unreachable**, and the whole schedule is compressed by ~21%.
- Every hip angle in telemetry is in this same 80°-normalised frame.

### Fix

Set `calib_range_l_rad` = `calib_range_r_rad` = **1.239 rad (71°)**
= 66° travel + 5° backoff. This rescales α, so the α-scheduled gains must be
re-checked afterwards — they were tuned against the compressed schedule.

---

## Finding 3 — Plant gain: K = 1.0, G ≈ 1.5–1.7

**K (firmware-rad → femur-rad) = 1.0.** The AK45 output shaft *is* the femur
pivot A; there is no reduction between them. **The AK45 MIT position scale is
fine** — unlike speed and torque, it needed no correction.

**G = d(roll)/d(commanded differential offset) ≈ 1.5–1.7 rad/rad** over the
usable α band (falls slowly with extension), from
`G = 2 · |dW_z/dq| / track`, track = 2 × `leg_y` = 0.286 m.

Only ~**0.756** of the commanded offset is realised: with `hip_roll_kp = 25` the
hips sag under load, giving **G_eff ≈ 1.3**.

The sag reconstructs the MIT impedance law exactly, which independently confirms
the corrected AK45 torque scale:

| | predicted `(τ − tff)/kp` | measured follow error |
|---|---|---|
| hip L | (3.06 − 0.268)/25 = 0.112 rad | 0.1125 rad |
| hip R | (2.15 − 0.268)/25 = 0.075 rad | 0.0746 rad |

Roll authority is *additionally* capped by travel headroom
`min(t, 1−t) × 1.396`, which binds on the retract side at low ride height.

---

## Finding 4 — The roll controller's code is correct; it ran out of authority

The control law reproduces to 0.1%. Least-squares on the unsaturated samples:

```
offset_cmd = 0.999·(sp − roll) − 0.0199·roll_rate,  sp ≈ −0.05°
residual sd = 0.00004 rad  (0.1% of signal, at telemetry quantisation)
```

So `roll_kp ≈ 1.0`, `roll_kd ≈ 0.02`, `roll_ki = 0` (a pure-PD fit that tight
proves the integral contributed nothing), `roll_offset_max = 0.15`. Sign
convention correct, clamping exact, hip-travel limits never engaged.

**What went wrong:** the offset hit +0.15 at t = 9.72 s and stayed pinned until
t = 28.78 s (19.1 s) while body roll sat at −14.6° mean, −16.9° peak.
`roll_offset_max = 0.15` buys only ~11° of authority — short of the disturbance.

`roll_kp = 1.0` also gives a loop gain of only `G_eff · kp ≈ 1.3`, so even
unsaturated the P-only law removes just over half of any disturbance, and
`roll_ki = 0` guarantees the rest persists as steady-state droop.

The roll watchdog never tripped: peak 16.9° against a ~20° default limit. Margin
was thin, and `roll_watchdog_en` was **not** in the log's param dump so its state
is unconfirmed.

---

## Proposed roll parameters

Target: fully null a sustained ~10° cross-slope, inside the 20° watchdog.

| Param | Now | Proposed | Rationale |
|---|---|---|---|
| ride height (CH3) | 0.149 | **≥ 0.25** | headroom 0.349 > clamp; keeps offset symmetric |
| `roll_offset_max` | 0.15 | **0.25** | ~11° → ~19° authority |
| `roll_kp` | 1.0 | **2.0** | loop gain 1.3 → 2.6; 3.0 is too hot for the 0.25 s hip lag |
| `roll_kd` | 0.02 | **0.04** | holds the kd/kp ratio |
| `roll_ki` | 0 | **1.0** | kills the residual droop |
| `roll_int_max` | 0.1 | **0.20** | `ki × int_max` must reach ~0.20 to fully null 15° |
| `hip_roll_kp` | 25 | **40** (optional) | sag 6.4° → 4.0°, ~+11% authority |
| `roll_watchdog_en` | ? | **verify = 1** | default is off, persistent |

`roll_int_max = 0.20` is a deliberate 2× departure from the schema's "keep it
small" guidance. At true steady state the error is zero, so P contributes nothing
and the **integral must supply the entire offset the slope demands**. The schema
warning was written for nulling a small standing bias on level ground; holding a
real cross-slope is a different job. `roll_offset_max` still bounds the total and
the integral clears on every `RUNNING` entry.

Bandwidth check: integral crossover ≈ `ki · G_eff` ≈ 1.3 rad/s against the ~0.25 s
hip actuator lag measured from the log's cross-correlation — adequate separation.

Tune `roll_kp` live on CH7 (live-tune group 2, CH5 down + CH6 down, is already
wired to `roll_kp`/`roll_kd`).

---

## What the geometry error invalidated

| Invalidated (model-derived) | Survives (empirically tuned) |
|---|---|
| `L_EFF_RET` / `L_EFF_EXT` in `control_loop.cpp` | LQR gains, `vel_pi_*`, `yaw_pi_*` |
| `Q_RET` / `Q_EXT` | `lqr_pitch_trim_ret` (measured on the robot) |
| The baseline-1 4-bar optimisation result | AK45 torque scale (validated against a physical standard) |
| Jump sim trajectories | Wheel/comms/state-machine work |

The α-scheduled gains sit in between: empirical, but tuned against a schedule
compressed 21% by Finding 2. Re-check them after the calibration fix; do not
assume they transfer.

---

## Open question

**`Q_RET` / `Q_EXT` absolute placement is unresolved** and is flagged as such in
both `params.py` files. The 66° stroke is the measured quantity; the two angles
are not. 66° does not fit under the current auto-computed anchor (−0.379 →
−1.531 is past the singularity at −1.301) — it requires `Q_RET ≥ −0.149`.

**Needed: the retracted hip angle from CAD.** One number closes this, and
everything downstream in the sim depends on it.

---

## Next steps

### 1. Blocking — needs a CAD readout
- [ ] Read the **retracted hip angle** off the CAD; set `Q_RET`/`Q_EXT` in
      `master_sim/params.py` and `master_sim_jump/params.py`.

### 2. Firmware — do before any bench tuning
- [ ] Set `calib_range_l_rad` = `calib_range_r_rad` = **1.239 rad (71°)**.
- [ ] Recalibrate; confirm `gain_sched_alpha` reaches ~1.0 at full extension
      instead of topping out at 0.825.

> This changes what `t` and α physically mean, so anything measured or tuned
> beforehand is invalidated by it. Do it first.

### 3. Bench session (~30 min)
- [ ] **Measure G directly.** Level ground, `t = 0.25`, `roll_ki = roll_kd = 0`,
      command a known CH1 roll setpoint, read steady-state `roll_rad` ÷
      `(hip_l_cmd_pos_rad − hip_r_cmd_pos_rad)/2`. That ratio *is* G. Replaces a
      ±20% model number with a measurement.
- [ ] Verify `roll_watchdog_en = 1`, and the `hip_l_enable` / `wheel_l_enable`
      states (none appeared in the log's param dump).
- [ ] Apply the roll parameters above; repeat the same cross-slope run.

### 4. Re-derive
- [ ] Recompute `L_EFF_RET` / `L_EFF_EXT` from the corrected IK.
- [ ] Re-check the α-scheduled gains after the α rescale.
- [ ] **Decide:** re-run the 4-bar optimiser against as-built geometry? The build
      won't change, so the value is quantifying what the 18 mm cost — information,
      not a fix.

### 5. Code follow-ups
- [ ] **Make calibration find the extended stop** instead of trusting a
      configured range. The torque-trip machinery (`calib_move_trq_lim`,
      `calib_trq_trip_ms`) already exists for the retract switch; reusing it on
      the extended end would have caught Finding 2 automatically. This is the
      root cause, not a patch.
- [ ] **Symmetric roll-offset clamp** — clamp `offset` to `min(t, 1−t) × span`
      *before* the differential apply in `control_loop.cpp`, so running out of
      travel can't silently make the offset asymmetric and shift ride height
      mid-correction. Required to make `roll_offset_max = 0.25` safe at low `t`.
- [ ] **Full param dump at log start** — several params needed for this analysis
      were never captured, forcing inference where measurement was available.

---

## Traps for future analysis

Two wrong turns were taken during this investigation. Both are easy to repeat.

1. **Do not infer the hip scale from `Q_RET`/`Q_EXT`.** Their 0.699 rad span is a
   *nominal stance stroke*, not the calibrated range (1.396 rad). Treating them
   as the α endpoints understates the hip scale by ~2× and produces a spurious
   "AK45 position scale is wrong" conclusion. **K = 1.0.**

2. **Roll is a supported DOF, not a free one.** The wheels are coaxial, so the
   two contact patches are separated *laterally*: roll is rigidly constrained by
   the track, and *pitch* is the free unstable DOF. A sustained 15° body roll on
   a cross-slope is entirely normal and does not imply the robot was on a stand.

---

## Files changed (2026-08-03)

| File | Change |
|---|---|
| `CLAUDE.md` | F_Z → −5.5 mm + "do not use −18.21" warning block |
| `components/COMPONENTS.md` | geometry table row + as-built note |
| `simulation/mujoco/master_sim/params.py` | `F_Z = -0.0055`, `Q_RET`/`Q_EXT` flagged |
| `simulation/mujoco/master_sim_jump/params.py` | promoted `TEMP` → permanent |
| `simulation/2d/fourbar_optimizer_gui/model.py` | `F_Z = +0.018` + rewrote frame-shift header |
| `software/gui/tabs/robot_visualizer_tab.py` | `_F_Z = -0.0055` |
| `firmware/robot_teensy/teensy/src/control_loop.cpp` | stale α ↔ `Q_RET`/`Q_EXT` annotation |
| `firmware/robot_teensy/README.md` | new section on the calibrated-span bug |

`logs/params_backups/` and `archive/` were deliberately left untouched — they are
historical snapshots.
