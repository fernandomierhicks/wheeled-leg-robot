# HANDOFF.md — Project Status & Next Steps

Read this before starting work. Check `CLAUDE.md` for full design context.

---

## Current Status (2026-03-17, updated)

### Phase 1 — 3-State LQR + Gain Scheduling ✅ COMPLETE (superseded by Phase 2 baseline)

Phase 1 v2 result (run_id=6288): Q=[0.0438, 0.0021, 0.0001], R=9.027, fitness=0.400
Superseded — Phase 2 uses a combined 5-scenario fitness; Phase 1 gains performed poorly on drive scenarios.

---

### Phase 2 — Velocity PI + Drive Mode ✅ COMPLETE & BASELINED

**Active folder:** `simulation/mujoco/LQR_Control_optimization/`

Optimized Q/R/KP_V/KI_V — **Phase 2** (2346 gen / 18768 evals, 90 min, 8 workers):

| Param | Value | Notes |
|-------|-------|-------|
| Q_PITCH | 0.8168 | ~19× higher than Phase 1 v2 — needs stronger pitch control when driving |
| Q_PITCH_RATE | 0.2553 | ~122× higher than Phase 1 v2 — more damping |
| Q_VEL | 0.0001 | unchanged — velocity still unpenalised by LQR |
| R | 48.565 | at ceiling (50.0 max) — optimizer wants even higher R |
| KP_V | 0.010 | at lower bound (0.01 min) — tiny lean command preferred |
| KI_V | 0.001 | at lower bound (0.001 min) |

**Phase 2 Performance (run_id=13493, combined 5-scenario fitness):**

| Scenario | Fitness | Notes |
|----------|---------|-------|
| Balance | 0.249 | |
| Disturbance | 0.424 | |
| Drive slow | 0.486 | vel_track_rms ≈ 0.286 m/s |
| Drive medium | **1.346** | dominant term — velocity error large |
| Obstacle | 0.526 | |
| **Combined** | **0.619** | weights: 10/35/20/20/15% |

Other metrics: RMS pitch 0.115°, max pitch 0.665°, wheel_travel 0.268 m, liftoff=0s, survived 5.0s.

**Key findings:**
- Optimizer converged fully — all sigmas hit minimum (0.01) by gen ~1637, no improvement after.
- R ceiling (50.0) was hit repeatedly → **expand R to 100–200 for next run**.
- KP_V/KI_V at lower bounds → robot prefers minimal lean commands; drive accuracy via LQR v_ref term only.
- `fitness_drive_med = 1.346` is the dominant weakness — 0.8 m/s tracking needs a stronger velocity loop.
- Possible next step: increase `W_VEL_ERR` weight or reduce `KP_V` floor to 0.001 to allow optimizer more freedom.

**Phase 2–6** not yet started. See `docs/Control.MD` for full implementation plan.

---

Suggested next optimizer run to improve drive medium tracking:

```bash
cd simulation/mujoco/LQR_Control_optimization
# In optimize_lqr.py, change PARAM_RANGES:
#   "R": (0.1, 100.0)      — was (0.1, 50.0), R kept hitting ceiling
#   "KP_V": (0.001, 0.5)   — was (0.01, 0.5), allow smaller lean commands
python optimize_lqr.py --hours 2 --workers 8
python replay.py --top 1
```

---

## What Was Done This Session (2026-03-17)

### Optimizer & Scenario Overhaul (pre-optimizer prep)

All changes are in `simulation/mujoco/LQR_Control_optimization/`.

#### 1. CSV schema fixed
- `run_log.py` `CSV_COLS` previously had PD columns (`KP, KD, KP_pos, KP_vel`).
  Replaced with LQR columns (`Q_PITCH, Q_PITCH_RATE, Q_VEL, R`).
- `results.csv` cleared — fresh start for new optimization run.
- Added `wheel_liftoff_s` metric column.

#### 2. Fitness metrics corrected & extended (`scenarios.py`)

**Pitch RMS now measured from equilibrium, not from vertical.**
- `pitch_err_deg = degrees(|pitch_true − pitch_ff|)` where `pitch_ff = get_equilibrium_pitch()`
- Previously was measuring from 0° — penalised the ~2° natural forward lean as error.
- The LQR already controlled to `pitch_ff`; now the fitness agrees.

**New metric: `wheel_liftoff_s`** — accumulated time either wheel is off the ground (Z > WHEEL_R + 5 mm).
- Penalises bouncing controllers: `W_LIFTOFF = 50.0` N·m per second airborne.
- Applied in both balance and disturbance scenarios.

**Fitness breakdown:**

| Scenario | Formula |
|----------|---------|
| Balance | `1.0×rms_pitch + 0.5×wheel_travel + 50×liftoff + 200×(fell)` |
| Disturbance | `1.0×rms_pitch + 1.0×rms_pitch_post_dist + 0.5×wheel_travel + 50×liftoff + 200×(fell)` |
| Combined (optimised) | `0.2×fitness_balance + 0.8×fitness_disturbance` |

#### 3. Hip backdrivability (`sim_config.py`, `scenarios.py`, `replay.py`)
- Added `HIP_IMPEDANCE_TORQUE_LIMIT = 2.0 N·m` — separate from physical `HIP_TORQUE_LIMIT = 7 N·m`.
- Impedance controller (position-hold) is clamped to 2 N·m. Any disturbance above that will backdrive the leg.
- Full 7 N·m remains available for jump/recovery phases later.
- **Tune this value in Phase 4** when adding proper leg impedance suspension.

#### 4. Arena & disturbance tuning
- **Walls removed** from `physics.py` — open arena, robot can roll freely.
- **Disturbance duration** reduced from 1.0s → **0.2s** (sharp impulse at t=2.5s, 1N horizontal).

#### 5. `replay.py` improvements
- Fixed impedance clamp to use `HIP_IMPEDANCE_TORQUE_LIMIT` (was using full `HIP_TORQUE_LIMIT`).
- Added `--baseline` flag: bypasses CSV, loads gains directly from `sim_config.py`.
- Added `--scenario` flag: override scenario for any replay.

```bash
python replay.py --baseline                                 # disturbance (default)
python replay.py --baseline --scenario balance              # still balance
python replay.py --baseline --scenario balance_disturbance  # with impulse
python replay.py --top 1                                    # best CSV run
```

---

## Ready to Run: Longer Optimizer Run

Seeds from best PASS row (`results.csv`) — starts at Phase 1 v2 weights.
Two suggested improvements before next run:

1. **Expand R range** in `optimize_lqr.py`: change `"R": (0.1, 10.0)` → `"R": (0.1, 50.0)` — optimizer hit the ceiling.
2. **Multi-start**: add `--restarts N` flag or run twice and keep best.

```bash
cd simulation/mujoco/LQR_Control_optimization
python optimize_lqr.py --hours 2 --workers 8
python replay.py --top 1
```

---

## Phase 2 — Next Up

After optimizer converges, move to **Phase 2: Velocity PI + Drive Mode**.
See `docs/Control.MD` Phase 2 steps.

Files to create/modify: `scenarios.py` (DRIVE scenario), `replay.py` (W/S key HUD).

---

## Path Forward: Obstacle Scenario

**Goal:** Robot traverses a 5 cm step without falling, without leg liftoff, with pitch excursion < 5°.

**Prerequisites:** Phase 4 (Leg Impedance) must be complete first — the leg must absorb the step passively.

**Implementation plan:**
1. Add a floor step geometry to `physics.py` (`build_xml`): 50 mm tall × 100 mm wide box, placed at x=0.5 m.
2. Add `run_obstacle_scenario()` to `scenarios.py`:
   - Drive at v_ref=0.3 m/s (needs Phase 2 Velocity PI)
   - Measure: max pitch during step traverse, liftoff_s, fell flag
   - Fitness: `rms_pitch + 100×liftoff_s + 200×fell`
3. Key LQR concern: current `Q_VEL ≈ 0` means velocity is uncontrolled — the step will cause a hard velocity shock. Phase 2 velocity PI is needed to regulate this.
4. Tune `HIP_IMPEDANCE_TORQUE_LIMIT` (currently 2 N·m) — may need to increase to 3–4 N·m for step absorption.

**Metric to beat:** pitch excursion ≤ 5° during step traverse, liftoff = 0 s.

---

## Path Forward: Turning Scenario

**Goal:** Robot executes a 360° in-place turn with < 10 cm lateral drift, pitch held < 3°.

**Prerequisites:** Phase 3 (Yaw PI) — already designed, not implemented.

**Implementation plan:**
1. Add `YawPI` class to `scenarios.py` (or new `yaw_controller.py`):
   - `τ_yaw = Kp_yaw × (ω_desired − ω_z) + Ki_yaw × integral`
   - Clamp ±0.5 N·m; differential mixing: `τ_L = τ_sym + τ_yaw`, `τ_R = τ_sym − τ_yaw`
   - Starting gains: `Kp_yaw=0.3`, `Ki_yaw=0.05`
2. Add `run_turn_scenario()` to `scenarios.py`:
   - Command ω=1 rad/s for 6.28 s (360°)
   - Measure: yaw achieved, lateral drift (X/Y displacement), RMS pitch, fell flag
   - Fitness: `|360° − yaw_achieved_deg| + 10×lateral_drift_m + rms_pitch + 200×fell`
3. Key cross-coupling concern: differential torque will create a net forward/backward pitch disturbance — check that LQR suppresses it. If pitch spikes > 3°, reduce `τ_yaw` clamp.
4. Tune Kp_yaw/Ki_yaw with 5×5 grid sweep (manual or scripted).

**Metric to beat:** 360° ± 10°, lateral drift < 10 cm, pitch RMS < 3° during turn.

---

## Key Files

| File | Purpose |
|------|---------|
| `simulation/mujoco/LQR_Control_optimization/sim_config.py` | All geometry + tunable params |
| `simulation/mujoco/LQR_Control_optimization/scenarios.py` | Headless scenarios + fitness |
| `simulation/mujoco/LQR_Control_optimization/optimize_lqr.py` | (1+8)-ES optimizer |
| `simulation/mujoco/LQR_Control_optimization/replay.py` | MuJoCo viewer + telemetry |
| `simulation/mujoco/LQR_Control_optimization/lqr_design.py` | LQR solver + gain scheduling |
| `simulation/mujoco/LQR_Control_optimization/physics.py` | XML builder + IK + CoM |
| `simulation/mujoco/LQR_Control_optimization/run_log.py` | CSV logging (schema: Q_PITCH, Q_PITCH_RATE, Q_VEL, R) |
| `simulation/mujoco/LQR_Control_optimization/results.csv` | Run history (cleared, ready for new run) |
| `docs/Control.MD` | Full 4-controller architecture + all phase plans |

---

## Hardware Tasks (Parallel, Not Blocking)

- [ ] Pick specific 5065 130KV motor SKU (Flipsky, T-Motor — needs Hall sensors, D-shaft, ≥40A, 24V)
- [ ] Finalise battery: 24V nominal, ≥4Ah, XT60, ≤750g (6S LiPo or 24V LiFePO4)
- [ ] Design wheel in CAD: PLA hub + TPU tread, 150mm OD, D-shaft mount, target 70g

---

## Future Tuning Notes

- `HIP_IMPEDANCE_TORQUE_LIMIT` (currently 2 N·m): tune properly in Phase 4 leg impedance work.
- `DISTURBANCE_FORCE` (1 N) and `DISTURBANCE_TIME` (2.5s): fixed for now, may want to randomise in Phase 6 compound scenario.
- `W_LIFTOFF = 50.0`: aggressive penalty; reduce if it over-constrains the optimizer.
