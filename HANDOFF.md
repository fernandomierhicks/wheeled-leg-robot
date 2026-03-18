# HANDOFF.md — Project Status & Next Steps

Read this before starting work. Check `CLAUDE.md` for full design context.

---

## Current Status (2026-03-18)

### Phase 1 — 3-State LQR + Gain Scheduling ✅ COMPLETE

Best result (run_id=6288, 8496 evals):

| Param | Value |
|-------|-------|
| Q_PITCH | 0.654 |
| Q_PITCH_RATE | 0.134 |
| Q_VEL | 0.0269 |
| R | 9.275 |
| K_nominal | [-12.672, -1.754, -0.003] |

Balance-only fitness: **2.94 m** wheel travel (before VelocityPI).

---

### Phase 2a — VelocityPI Wired + Optimized ✅ COMPLETE

VelocityPI outer loop (v_desired=0 in BALANCE mode) is wired into all balance scenarios.

Best result (3032 evals, 2 min, balance-only):

| Param | Value | Notes |
|-------|-------|-------|
| KP_V | 0.299 | rad/(m/s) |
| KI_V | 0.500 | rad/m — **at upper bound** |
| fitness | **0.142 m** | wheel travel over 5s (was 2.94 m) |

20× improvement in position hold. KI_V hit the 0.5 upper bound — there may be more gain from widening to 1.0–2.0.

**Sign convention confirmed:** `vel_est_ms = wheel_vel * WHEEL_R` — positive wheel_vel = forward body motion.

**Simulator capability:** `replay.py` with v_desired slider (±1 m/s) feeding VelocityPI allows manual drive control. Robot balances and drives forward/backward stably.

---

## What Was Done This Session (2026-03-18, evening)

| Area | Change |
|------|--------|
| `scenarios.py` | VelocityPI wired with correct sign (`vel_est_ms = wheel_vel * WHEEL_R`) |
| `replay.py` | 4-panel telemetry (Pitch + commanded pitch overlay, Wheel Torque, Hip Angle, Robot Velocity m/s) |
| `replay.py` | v_desired slider (±1.0 m/s) → feeds VelocityPI directly; Drive ON/OFF toggle |
| `replay.py` | XYZ world frame axes in MuJoCo viewer (red=+X forward, green=+Y left, blue=+Z up) |
| `sim_config.py` | VELOCITY_PI_KP=0.299, VELOCITY_PI_KI=0.500 updated from optimization |
| `results.csv` | Cleared — fresh start after sign fix; 3032 new runs logged |

---

## Next Task: Re-run Optimizer with All Scenarios

### Before optimizing tomorrow:

1. **Check scenario metrics** — run each scenario individually and review output metrics (rms_pitch, wheel_travel, survived_s, fail_reason) to verify they're meaningful before combining:
   ```
   python -c "import scenarios; scenarios.run_balance_scenario({}) "
   python -c "import scenarios; scenarios.run_balance_with_disturbance_scenario({})"
   # etc. for drive_slow, drive_med, obstacle
   ```
   Confirm:
   - Balance: robot holds position <0.5m drift over 5s ✓
   - Disturbance: robot recovers from 1N push at t=2.5s
   - Drive slow (0.3 m/s): robot reaches and holds speed
   - Drive medium (0.8 m/s): same
   - Obstacle (3cm step): robot crosses without falling

2. **Widen KI_V range** in `optimize_lqr.py`:
   ```python
   PARAM_RANGES = {
       "KP_V": (0.001, 2.0),
       "KI_V": (0.001, 2.0),   # was 0.5 — KI hit ceiling last run
   }
   ```

3. **Enable all scenarios** in `optimize_lqr.py`:
   ```python
   ACTIVE_SCENARIOS = None  # runs all 5: balance, disturbance, drive_slow, drive_med, obstacle
   ```

4. **Run 1–2 hour optimization:**
   ```
   python optimize_lqr.py --hours 2 --workers 8
   ```

---

## Key Files

| File | Purpose |
|------|---------|
| `simulation/mujoco/LQR_Control_optimization/sim_config.py` | All geometry + tunable params (source of truth) |
| `simulation/mujoco/LQR_Control_optimization/scenarios.py` | Headless scenarios + fitness functions |
| `simulation/mujoco/LQR_Control_optimization/optimize_lqr.py` | (1+8)-ES optimizer; set ACTIVE_SCENARIOS and PARAM_RANGES here |
| `simulation/mujoco/LQR_Control_optimization/replay.py` | MuJoCo viewer (`--top`, `--baseline`, `--slowmo`, `--freefall`) |
| `simulation/mujoco/LQR_Control_optimization/lqr_design.py` | LQR solver + gain scheduling |
| `simulation/mujoco/LQR_Control_optimization/physics.py` | XML builder + IK + CoM |
| `simulation/mujoco/LQR_Control_optimization/run_log.py` | CSV logging |
| `simulation/mujoco/LQR_Control_optimization/results.csv` | Run history |
| `docs/Control.MD` | Full 4-controller architecture + phase plans |

---

## Hardware Tasks (Parallel, Not Blocking)

- [ ] Pick specific 5065 130KV motor SKU (Flipsky, T-Motor — needs Hall sensors, D-shaft, ≥40A, 24V)
- [ ] Finalise battery: 24V nominal, ≥4Ah, XT60, ≤750g (6S LiPo or 24V LiFePO4)
- [ ] Design wheel in CAD: PLA hub + TPU tread, 150mm OD, D-shaft mount, target 70g

---

## Future Phase Plans

See `docs/Control.MD` for Phases 3–5 (Yaw PI, Leg Suspension, Jump Recovery).

Key tuning notes:
- `HIP_IMPEDANCE_TORQUE_LIMIT` (2 N·m): tune properly in Phase 4.
- `DISTURBANCE_FORCE` (1 N) and `DISTURBANCE_TIME` (2.5s): fixed; may increase force in Phase 3.
