# LQR_Control_optimization — Handoff

## What this folder is

Standalone simulation + evolutionary optimization framework for tuning the wheeled-leg robot's balance controller. No imports from outside this folder.

---

## Current state (2026-03-17)

### Working
- MuJoCo two-leg robot simulation matching `baseline1_leg_analysis` physics exactly
- PD balance controller wired to optimizer (gains passed in, not hardcoded)
- Evolutionary optimizer (1+8)-ES finding good gains in < 1 minute
- CSV logging of every run with full gains + metrics
- Visual replay of any run (`replay.py --top 1`)
- Live matplotlib telemetry panel alongside MuJoCo viewer

### Rates
| Layer | Rate |
|---|---|
| MuJoCo physics | 2000 Hz (timestep = 0.5 ms) |
| Controller (IMU read + torque cmd) | 500 Hz (`CTRL_STEPS = 4`) |

### Best PD gains found (run_id 221, balance scenario, no disturbance)
| Gain | Value | Baseline (2000 Hz) |
|---|---|---|
| `KP` (pitch proportional) | 10.1 N·m/rad | 60.0 |
| `KD` (pitch derivative) | 0.893 N·m·s/rad | 5.0 |
| `KP_pos` (wheel position feedback) | 2.16 rad/m | 0.30 |
| `KP_vel` (wheel velocity feedback) | 0.497 rad/(m/s) | 0.30 |

**Fitness: 0.415 | RMS pitch: ~0.4° | Wheel travel: ~0.025 m over 5 s**

Lower KP/KD vs baseline because 500 Hz controller needs less aggressive gains. Higher KP_pos compensates by keeping wheel position tightly regulated.

---

## File map

| File | Purpose |
|---|---|
| `sim_config.py` | Single source of truth — geometry, motor limits, leg impedance, timing |
| `physics.py` | MJCF XML builder, 4-bar IK, equilibrium pitch, torus wheel mesh |
| `scenarios.py` | Headless balance scenario runner; `balance_torque()` helper used by replay |
| `run_log.py` | CSV read/write; schema: `KP, KD, KP_pos, KP_vel` + metrics |
| `optimize.py` | (1+8)-ES optimizer; search space = PD gains; parallel workers |
| `replay.py` | MuJoCo viewer + matplotlib telemetry for any logged run |
| `results.csv` | All runs logged so far |

---

## Fitness function

```
fitness = W_RMS * rms_pitch_deg + W_TRAVEL * wheel_travel_m + (W_FALL=200 if fell)
W_RMS = 1.0,  W_TRAVEL = 0.5
```

Penalises pitch error AND wheel drift. Less wheel movement = robot balances more efficiently.

---

## Key physics parameters (must match baseline1)

- **Leg impedance**: `LEG_K_S = 8.0 N·m/rad`, `LEG_B_S = 4.0 N·m·s/rad`
  (matches `baseline1` `HIP_KP_SUSP / HIP_KD_SUSP` — wrong values cause jerkiness)
- **Solver**: Newton, 200 iterations, tolerance 1e-10
- **4-bar closure**: `eq_data` must be set after model build:
  ```python
  model.eq_data[eq_id, 0:3] = [-p['Lc'], 0.0, 0.0]
  model.eq_data[eq_id, 3:6] = [0.0, 0.0, p['L_stub']]
  ```

---

## Next steps (priority order)

### 1. Disturbance robustness scenario
Add an impulse push (horizontal force for ~50 ms) mid-run to the balance scenario.
Gains tuned only on a still start are brittle — they need to recover from perturbations.
Fitness should combine still-balance + push-recovery in a single score.

### 2. Replace PD with LQR
The optimizer currently tunes 4 PD gains directly.
The next phase: compute K via `scipy.linalg.solve_discrete_are` from Q/R weights,
then optimise Q/R instead of K directly.
State vector: `[pitch_error, pitch_rate, wheel_vel]` (3-state).
This gives a theoretically grounded controller that ports cleanly to firmware.
See `docs/Control.MD` for the derivation.

### 3. Driving scenario
Add forward/backward velocity tracking scenario.
New gains to tune: velocity PI (`Ki_vel`, `Kp_vel_outer`).
Extend `balance_torque()` to accept a velocity setpoint.

### 4. Yaw / turning scenario
Differential wheel torque for yaw rate control.
New gains: `Kp_yaw`, `Ki_yaw`.

### 5. Port to firmware
Once LQR K is stable in simulation, port the control loop to Arduino UNO R4:
- BNO086 → pitch, pitch_rate at 500 Hz
- ODESC encoder → wheel_vel
- Compute `u = -K @ [pitch_err, pitch_rate, wheel_vel]`
- Send torque cmd via CAN to ODESC

---

## How to run

```bash
cd simulation/mujoco/LQR_Control_optimization

# Run optimizer (50 generations, 4 workers)
python optimize.py --iters 50 --workers 4

# Replay best result
python replay.py --top 1

# Replay specific run
python replay.py 42

# List all runs
python replay.py --list

# Re-simulate a run and overwrite CSV
python replay.py --resim 42
```
