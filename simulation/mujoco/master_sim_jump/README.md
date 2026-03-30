# master_sim — Quick Reference

Working directory for all commands: `simulation/mujoco/`

## Run a single scenario (headless) and print fitness

```python
python -c "
from master_sim.defaults import DEFAULT_PARAMS
from master_sim.scenarios import evaluate

m = evaluate(DEFAULT_PARAMS, 's09_integrated')
print(f'Status:            {m[\"status\"]}')
print(f'Survived:          {m[\"survived_s\"]}s')
print(f'RMS Pitch:         {m[\"rms_pitch_deg\"]:.4f} deg')
print(f'RMS Pitch Rate:    {m[\"rms_pitch_rate_dps\"]:.4f} deg/s')
print(f'Vel Tracking RMS:  {m[\"vel_track_rms_ms\"]:.4f} m/s')
print(f'Yaw Tracking RMS:  {m[\"yaw_track_rms_rads\"]:.4f} rad/s')
print(f'Max Roll:          {m[\"max_roll_deg\"]:.4f} deg')
print(f'Fitness:           {m[\"fitness\"]:.4f}')
"
```

- Replace `'s09_integrated'` with any scenario key: `s01_lqr_pitch_step`, `s02_leg_height_gain_sched`, `s03_vel_pi_disturbance`, `s04_vel_pi_staircase`, `s05_vel_pi_leg_cycling`, `s06_yaw_pi_turn`, `s07_drive_turn`, `s08_terrain_compliance`, `s09_integrated`.
- `DEFAULT_PARAMS` contains the current baseline gains from `defaults.py`.
- `evaluate()` runs `sim_loop.run()` headlessly and appends the `fitness` key via the scenario's `fitness_fn`.
- Typical runtime: ~10-30s depending on scenario duration.

## Launch the GUI launcher (scenario picker + MuJoCo viewer)

```bash
python master_sim/launcher.py
```

## Run a single optimizer. From /simulation/mujoco/

The unified entry point is `optimize_integrated` — it auto-selects the search space based on the scenario's group (e.g. S1 → LQR 4D, S9 → all 12D). A `ProgressUI` window shows live best-fitness, gain values, and play/pause.

```bash
# S1 LQR only, 5 minutes, from baseline gains
python -m master_sim.optimizer.optimize_integrated --hours 0.0833 --patience 50 --scenario s01_lqr_pitch_step

# S1 LQR only, 5 minutes, from random starting point (escape local minima)
python -m master_sim.optimizer.optimize_integrated --hours 0.0833 --patience 50 --scenario s01_lqr_pitch_step --random-seed

# S9 integrated (all 12 gains), 2 hours
python -m master_sim.optimizer.optimize_integrated --hours 2 --patience 500 --scenario s09_integrated

# Explicit seed gains
python -m master_sim.optimizer.optimize_integrated --hours 1 --scenario s01_lqr_pitch_step --seed-gains "Q_PITCH=0.5,Q_PITCH_RATE=0.01,Q_VEL=1e-6,R=0.02"
```

Common flags:

| Flag | Default | Description |
|---|---|---|
| `--hours H` | 1.0 | Wall-clock time limit |
| `--patience N` | 300 | Early-stop after N gens without improvement |
| `--tol F` | 1e-4 | Relative improvement threshold |
| `--workers N` | auto | Parallel eval workers |
| `--seed-gains "K=v,..."` | baseline | Explicit starting gains |
| `--random-seed` | off | Start from random point in search space |
| `--no-baseline` | off | Skip writing best gains into params.py |

There are also single-controller optimizer scripts (`optimize_lqr`, `optimize_vel_pi`, `optimize_yaw_pi`, `optimize_suspension`) but `optimize_integrated` is preferred as it uses the richer GUI with all gain groups visible.

## Run the full S1→S9 pipeline with random gains for initial controllers. 2 hours per scneario. 

```bash
python -m master_sim.optimizer.pipeline --hours 2 --fresh --random --patience 500
```

The pipeline chains steps sequentially, seeding each from the previous step's best result. Baseline steps write to `logs/baseline_gains.json`. Use `--fresh` to delete existing CSVs and re-run from scratch.

## Baseline gains after an optimizer run

Baselining means copying the best gains from an optimizer run into the dataclass defaults in `master_sim/params.py`, so all future simulations use them.

1. **Read the best params** from the optimizer output or CSV log (`logs/S_<scenario>.csv`).

2. **Edit `master_sim/params.py`** — update the default field values on the relevant dataclasses:

   | Gain group | Dataclass | Fields |
   |---|---|---|
   | LQR (4) | `LQRGains` | `Q_pitch`, `Q_pitch_rate`, `Q_vel`, `R` |
   | VelocityPI (2) | `VelocityPIGains` | `Kp`, `Ki` |
   | YawPI (2) | `YawPIGains` | `Kp`, `Ki` |
   | Suspension (4) | `SuspensionGains` | `K_s`, `B_s`, `K_roll`, `D_roll` |

   `defaults.py` simply does `SimParams()`, so changing the dataclass defaults is all that's needed.

3. **Verify** by running the scenario headlessly (see "Run a single scenario" above) and confirming fitness is in the expected range. Stochastic noise means each eval varies — a ±10% spread is normal.

4. **Visually check** via the launcher (`python master_sim/launcher.py`) — click the relevant scenario and watch for tracking quality, oscillations, or falls.

## Active controllers per scenario

Each scenario enables only the controllers relevant to its tuning phase. The `active_controllers` field on `ScenarioConfig` is the single source of truth, consumed by both `sim_loop.run()` and the visualizer replay via `tick_flags`.

| Scenario | Controllers | Hip mode |
|---|---|---|
| **S1** — LQR Pitch Step | LQR | position |
| **S2** — Gain Sched | LQR | position |
| **S3** — VelPI Disturbance | LQR, VelPI | position |
| **S4** — VelPI Staircase | LQR, VelPI | position |
| **S5** — VelPI Leg Cycling | LQR, VelPI | position |
| **S6** — Yaw PI Turn | LQR, VelPI, YawPI | position |
| **S7** — Drive + Turn | LQR, VelPI, YawPI | position |
| **S8** — Terrain Compliance | LQR, VelPI | impedance |
| **S9** — Integrated | LQR, VelPI, YawPI | impedance |

When `hip_mode="impedance"` (S8, S9), suspension and roll-leveling are automatically enabled.

## Delay predictor — what it can and can't do

The matrix predictor (`sim_loop.py`, ZOH-discretised 3-state) compensates for the sensor delay buffer by propagating the delayed state forward using `x̂(t) = A_d^n · x(t−n) + Σ B_d · u(t−n+k)`.

**Pitch prediction works well** (2.4× MAE reduction at 2 ms delay) because the correction is **kinematic**: `Δθ ≈ ω · Δt`. This is model-free — it only needs the current pitch rate, which is large during any dynamic motion.

**Pitch rate and wheel velocity prediction are negligible** (~1.02× improvement at 2 ms). Their corrections are **dynamic** (acceleration-level): `Δω ≈ α·θ·Δt` and `Δv ≈ β·θ·Δt`. These scale with pitch angle (small during balancing) multiplied by a tiny Δt, producing corrections orders of magnitude smaller than the signal. Even at 20 ms delay, improvement is only ~1.08× because the linearised model diverges at the large angles reached during a fall.

**Bottom line**: the pitch predictor is the channel that matters for LQR stability — and it works. The rate/velocity channels are mathematically correct but practically negligible at real-world sensor delays (≤ 5 ms).

Run the diagnostic to verify: `python -m master_sim.diag_delay`

## Key entry points

| What | Where |
|---|---|
| Baseline gains | `master_sim/defaults.py` → `DEFAULT_PARAMS` |
| Scenario configs + fitness functions | `master_sim/scenarios/s01_*.py` ... `s09_*.py` |
| Headless sim runner | `master_sim/sim_loop.run(params, scenario_config)` → returns metrics dict |
| `evaluate()` wrapper | `master_sim/scenarios/__init__.py` — calls `sim_loop.run` + `fitness_fn` |
| Optimizer scripts | `master_sim/optimizer/` |
| Visualizer / replay | `master_sim/viz/visualizer.py` → `run_unified()` |

---

## 4-Bar Geometry Optimizer (`master_sim_jump` only)

Optimises 4-bar link lengths to maximise jump height. Separate from the gain
optimizers above — it modifies **geometry**, not controller gains.

**Working directory:** `simulation/mujoco/`

### Quick start

```bash
# 4-hour run with progress GUI (recommended)
python -m master_sim_jump.optimizer.optimize_geometry --hours 4

# Overnight run
python -m master_sim_jump.optimizer.optimize_geometry --hours 8 --patience 500

# Dry run — 20 iterations, no output file written
python -m master_sim_jump.optimizer.optimize_geometry --iters 20 --workers 4 --no-save
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--hours H` | 4.0 | Wall-clock time limit (ignored if `--iters` set) |
| `--iters N` | — | Fixed generation count (overrides `--hours`) |
| `--workers N` | auto | Parallel eval workers (defaults to min(8, cpu_count)) |
| `--patience N` | 300 | Early-stop after N generations without improvement |
| `--tol F` | 1e-4 | Relative improvement threshold for patience counter |
| `--no-save` | off | Skip writing `optimizer/best_jump_params.py` |

### What is optimised — 5D search space

| Parameter | Baseline | Range | What it controls |
|---|---|---|---|
| `L_FEMUR` | 173.8 mm | 140–280 mm | Femur length A→C (hip to knee pivot) |
| `L_TIBIA` | 129.4 mm | 100–220 mm | Tibia length C→W (knee pivot to wheel) |
| `LC` | 150.8 mm | 120–240 mm | Coupler link F→E |
| `L_STUB` | 35.1 mm | 25–50 mm | Tibia stub C→E (above knee to coupler) |
| `CROUCH_TIME` | 0.20 s | 0.05–1.00 s | Duration of pre-jump crouch |

`max_torque` is **not** optimised — it is fixed at 7.0 N·m (motor limit). More
torque always increases jump energy so there is no benefit in searching over it.

### Fitness function

```
fitness = 200 × fell  +  (1 − peak_body_z_m / 0.30)
```

- **Minimised** (lower = better).
- `fell` — True if the robot crashes during or after the jump (+200 penalty, eliminates unstable geometries).
- `peak_body_z_m` — peak body CoM height above ground during the jump. Goes negative (i.e. fitness < 0) when the robot exceeds 300 mm, which is already better than baseline. The dry-run target fitness of ~−1.0 corresponds to ~600 mm peak body Z.
- Settle time is **not** included — pure jump height only.
- `wheel_liftoff_s` (airtime) is **not** used — it is biased by leg length (longer legs have higher baseline body Z regardless of jump quality).

### How stroke angles are handled — `auto_stroke_angles()`

Each candidate geometry has a different valid hip angle range. Rather than using
the hardcoded `Q_RET`/`Q_EXT` from `params.py` (which are only correct for the
baseline geometry), every evaluation calls `physics.auto_stroke_angles()`:

1. Sweeps `q_hip` from −0.05 to −2.5 rad in 500 steps, calling `solve_ik()` at each step.
2. Rejects positions where the knee exceeds ±60° (physical joint limit).
3. Trims the top/bottom 5% to avoid singularities.
4. Returns `(Q_RET, Q_EXT)` — the angles where the wheel is highest (crouch) and lowest (full extension).
5. Returns `None` (infeasible) if: fewer than 50 valid IK solutions, stroke < ~17°, or no closure found. Infeasible candidates score fitness = 500 and are eliminated.

### LQR stability with new geometry

The LQR gains are **not** re-tuned per candidate. The existing gains from
`params.py` are used as-is. Geometries that destabilise the LQR controller will
cause the robot to fall, scoring `fell=True` (+200 penalty) and being eliminated.
This is a known limitation: a geometry that is genuinely better but requires
slightly different LQR gains may be incorrectly rejected. If the optimizer
converges to a geometry significantly different from baseline that keeps falling,
re-run the LQR optimizer (`optimize_integrated`) on that geometry before judging it.

### Output — `optimizer/best_jump_params.py`

On completion, the best geometry is written to
`master_sim_jump/optimizer/best_jump_params.py`. This file is **never**
auto-applied to `params.py` — copy the values manually when satisfied.

Example output file:

```python
# RobotGeometry fields:
L_femur    = 0.220000   # [m]
L_tibia    = 0.170000   # [m]
Lc         = 0.188200   # [m]
L_stub     = 0.025000   # [m]
Q_RET      = -0.412345  # [rad]  auto-computed by auto_stroke_angles()
Q_EXT      = -1.451234  # [rad]  auto-computed by auto_stroke_angles()

# JumpGains fields:
crouch_time = 0.132400  # [s]
max_torque  = 7.0       # [N·m]  fixed at motor limit
```

To apply: copy these values into the corresponding fields in
`master_sim_jump/params.py` (`RobotGeometry` and `JumpGains` dataclasses).
A backup of `params.py` is created automatically at
`logs/params_backups/params_<timestamp>.py` before any write.

### Progress GUI

A `ProgressUI` window opens automatically showing:
- Progress bar, elapsed/remaining time, eval count, generation number
- Best fitness (large green) and best params string
- **Geometry panel** — live Best / Trying columns for all 5 parameters
- Play / Pause button

### Verifying the infrastructure

```bash
# Step 1 — confirm auto_stroke_angles() reproduces baseline stroke
python -c "
from master_sim_jump.params import RobotGeometry
from master_sim_jump.physics import auto_stroke_angles
print(auto_stroke_angles(RobotGeometry()))
# Expected: Q_RET ~ -0.40, Q_EXT ~ -1.43 (mechanical limits; Q_RET in params.py
# is set 25 deg inside this range for spring engagement — that is intentional)
"

# Step 2 — confirm peak_body_z_m is returned by s10_jump
python -c "
from master_sim_jump.defaults import DEFAULT_PARAMS
from master_sim_jump.scenarios import evaluate
m = evaluate(DEFAULT_PARAMS, 's10_jump')
print('peak_body_z_m =', m.get('peak_body_z_m'))
# Expected: ~0.47 m for baseline geometry
"
```
