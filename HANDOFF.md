# HANDOFF.md — Project Status & Next Steps

Read this before starting work. Check `CLAUDE.md` for full design context and `docs/Control.MD` for controller architecture and full scenario specs.

---

## Current Status (2026-03-18)

### Phase 1 — Balance LQR ✅ COMPLETE & S4-BASELINED

LQR gains optimised on S4 (`4_leg_height_gain_sched`) — legs cycling through full stroke + disturbances.

**S4 baseline (current `sim_config.py`):**
| Param | Value |
|-------|-------|
| Q_PITCH | 0.014168 |
| Q_PITCH_RATE | 0.033720 |
| Q_VEL | 0.000250 |
| R | 28.734420 |
| fitness (S4) | 0.017938 |
| rms_pitch (S4) | 1.242° — survived 12 s (3 leg cycles) |

---

### Phase 2 — VelocityPI Outer Loop ✅ COMPLETE & S5-BASELINED

VelocityPI gains optimised on S5 (`5_VEL_PI_leg_cycling`) — staircase velocity tracking + ±1N kicks + legs cycling + 5 bump obstacles.

**S5 baseline (current `sim_config.py`):**
| Param | Value |
|-------|-------|
| KP_V | 0.251209 |
| KI_V | 0.011405 |
| fitness (S5) | 2.0635 |
| vel_rms | 0.502 m/s |
| rms_pitch | 5.59° — survived 13 s |

---

### Phase 3 — Yaw PI + Turn Mode ✅ COMPLETE & BASELINED

YawPI implemented and verified. Starting gains performed well; no optimizer run needed.

**Baselined gains (current `sim_config.py`):**
| Param | Value |
|-------|-------|
| KP_YAW | 0.3 |
| KI_YAW | 0.05 |
| YAW_PI_TORQUE_MAX | 0.5 N·m |

**Key implementation facts:**
- Sign: `τ_L = τ_sym − τ_yaw`, `τ_R = τ_sym + τ_yaw` (positive ω = CCW = left turn)
- Yaw rate: `data.qvel[dof_free + 5]` — world-frame, no pitch contamination
- S6 pure turn: yaw_rms = 0.045 rad/s at 1 rad/s target (4.5% error), pitch = 3.3°
- S7 drive+turn: v=0.3 + ω=0.5, pitch = 3.2°, no cross-coupling instability
- `replay.py` updated: Yaw Rate panel, ω_desired slider, Turn ON/OFF, differential torque
- `sandbox.py` added: interactive arena, 28 obstacles (1–8 cm), gamepad + sliders

---

## Scenario Reference

| # | Name | Controller | Legs | v_desired | Disturbance | Duration |
|---|------|------------|------|-----------|-------------|----------|
| 1 | `1_LQR_pitch_step` | LQR only | fixed Q_NOM | 0 | +5° pitch | 5s |
| 2 | `2_VEL_PI_disturbance` | VelocityPI+LQR | fixed Q_NOM | 0 (hold) | ±1N at t=2/3s | 12s |
| 3 | `3_VEL_PI_staircase` | VelocityPI+LQR | fixed Q_NOM | staircase | none | 13s |
| 4 | `4_leg_height_gain_sched` | LQR only | cycling | 0 | ±4N at t=2/3s | 12s |
| 5 | `5_VEL_PI_leg_cycling` | VelocityPI+LQR | cycling | staircase | ±1N + 5 bumps | 13s |
| 6 | `6_YAW_PI_turn` | YawPI+VelocityPI+LQR | fixed Q_NOM | 0 (hold) | none | 8s |
| 7 | `7_DRIVE_TURN` | YawPI+VelocityPI+LQR | fixed Q_NOM | 0.3 m/s | none | 8s |

Full specs in `docs/Control.MD → Optimizer Scenarios`.

---

## Next Task — Phase 4: Leg Impedance Suspension Tuning

**Goal:** Make the legs act as a passive suspension — absorbing terrain impacts and complying to disturbances rather than rigidly holding `Q_NOM`. The LQR balance loop is unaffected; only the hip impedance parameters change.

**Current state:**
- `LEG_K_S = 8.0 N·m/rad` (spring stiffness — holds leg at Q_NOM)
- `LEG_B_S = 4.0 N·m·s/rad` (damping)
- `HIP_IMPEDANCE_TORQUE_LIMIT = 2.0 N·m` (placeholder — never validated)
- Hip target is fixed at `Q_NOM` in all scenarios; no terrain compliance

**What to do:**

### Step 4.1 — Tune `HIP_IMPEDANCE_TORQUE_LIMIT`

The 2 N·m limit was a placeholder. It defines when the hip backdrive — any terrain force above this will deflect the leg.
- Too high: rigid leg, impacts transmitted to body, large pitch spikes
- Too low: leg flops around, poor balance on rough terrain
- Start: lower to **1.0 N·m** and observe S5 bump performance
- Pass criterion: pitch < 6° during 3 cm bump impacts (currently ~5.6° RMS)

```bash
# Edit sim_config.py: HIP_IMPEDANCE_TORQUE_LIMIT = 1.0
python replay.py --baseline --scenario 5_VEL_PI_leg_cycling
```

### Step 4.2 — Tune K_s / B_s

After torque limit is set, sweep stiffness:
- Softer spring (K_s ↓): more terrain compliance, more leg sag at speed
- More damping (B_s ↑): less oscillation after bump
- Suggested range: K_s ∈ [4, 12], B_s ∈ [2, 8]
- Pass criterion: S5 bump impacts don't cause fall, pitch spike < 4° per bump

### Step 4.3 — Add Scenario 8: terrain compliance test (optional)

A dedicated scenario with harder bumps (5 cm) at higher speed (1.0 m/s) would let the optimizer tune K_s/B_s/torque_limit simultaneously. Use S5 as template.

**Key files to edit:**
- `sim_config.py` — `LEG_K_S`, `LEG_B_S`, `HIP_IMPEDANCE_TORQUE_LIMIT`
- `scenarios.py` — already applies impedance in `_run_sim_loop`; no structural changes needed
- `replay.py` — hip angle panel removed (now Yaw Rate); hip behaviour visible via pitch response

---

## Quick Reference

```bash
cd simulation/mujoco/LQR_Control_optimization

# Interactive sandbox (drive + turn + hip control)
python sandbox.py

# Replay best S6 (yaw turn)
python replay.py --top 1 --csv results_6_YAW_PI_turn.csv

# Replay S5 with current gains (bump + leg cycling)
python replay.py --baseline --scenario 5_VEL_PI_leg_cycling

# Replay S7 (drive+turn cross-coupling)
python replay.py --baseline --scenario 7_DRIVE_TURN

# Re-run YawPI optimizer (if needed)
python optimize_yaw_pi.py --hours 0.5
```

---

## Key Files

| File | Purpose |
|------|---------|
| `sim_config.py` | Single source of truth — all gains, durations, disturbance forces, S5 bumps |
| `scenarios.py` | All scenario runners + `_run_sim_loop` + `YawPI` + `VelocityPI` classes |
| `optimize_lqr.py` | (1+8)-ES — Q/R weights — pointed at S4 |
| `optimize_vel_pi.py` | (1+8)-ES — KP_V, KI_V — pointed at S5 |
| `optimize_yaw_pi.py` | (1+8)-ES — KP_YAW, KI_YAW — pointed at S6 |
| `replay.py` | MuJoCo viewer + 4-panel telemetry (pitch, yaw rate, pitch rate, velocity) |
| `sandbox.py` | Interactive arena — 28 obstacles 1–8 cm, gamepad + sliders |
| `physics.py` | XML builder — `build_xml(sandbox_obstacles=..., floor_size=...)` |
| `lqr_design.py` | LQR solver + gain scheduling at Q_RET/Q_NOM/Q_EXT |
| `results_4_leg_height_gain_sched.csv` | S4 LQR run history |
| `results_5_VEL_PI_leg_cycling.csv` | S5 VelocityPI run history |
| `results_6_YAW_PI_turn.csv` | S6 YawPI run history |
| `docs/Control.MD` | Full architecture, scenario specs, phase plans |

---

## Hardware Tasks (Parallel, Not Blocking)

- [ ] Pick specific 5065 130KV motor SKU (Flipsky, T-Motor — needs Hall sensors, D-shaft, ≥40A, 24V)
- [ ] Finalise battery: 24V nominal, ≥4Ah, XT60, ≤750g (6S LiPo or 24V LiFePO4)
- [ ] Design wheel in CAD: PLA hub + TPU tread, 150mm OD, D-shaft mount, target 70g
