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

### Phase 2 — VelocityPI Outer Loop ✅ COMPLETE & SANDBOX-RETUNED

VelocityPI gains optimised on S5, then manually increased after sandbox testing showed optimizer was too conservative (sluggish acceleration on flat ground).

**Current baseline (current `sim_config.py`):**
| Param | Value | Notes |
|-------|-------|-------|
| KP_V | 0.502418 | 2× optimizer value — snappier feel in sandbox |
| KI_V | 0.011405 | unchanged from optimizer |
| THETA_REF_RATE_LIMIT | 5.0 rad/s | 2.5× optimizer value — lean ramp ~52 ms |

---

### Phase 3 — Yaw PI + Turn Mode ✅ COMPLETE & BASELINED

YawPI implemented and verified. Starting gains performed well; no optimizer run needed.

**Baselined gains (current `sim_config.py`):**
| Param | Value | Notes |
|-------|-------|-------|
| KP_YAW | 2.272 | 55752 evals / 75 min — was visual 0.3 (7.5× increase) |
| KI_YAW | 1.125 | was visual 0.05 (22.5× increase) |
| YAW_PI_TORQUE_MAX | 0.5 N·m | unchanged |
| fitness (S6) | 0.4102 | gen 3622 |

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
| 8 | `8_terrain_compliance` | All+Suspension+Roll | impedance | 1.0 m/s | 5 one-sided bumps | 12s |

Full specs in `docs/Control.MD → Optimizer Scenarios`.

### Phase 4 — Leg Impedance Suspension + Roll Leveling ✅ COMPLETE & BASELINED

Suspension + roll leveling controller implemented and optimised on combined S5+S8 scenario.

**Baselined gains (current `sim_config.py`):**
| Param | Value | Notes |
|-------|-------|-------|
| LEG_K_S | 16.000000 N·m/rad | Very stiff — absorbs impacts sharply |
| LEG_B_S | 0.798710 N·m·s/rad | Low damping — spring-dominant |
| LEG_K_ROLL | 4.000000 rad/rad | At upper bound — wants more |
| LEG_D_ROLL | 1.000000 rad·s/rad | At upper bound — wants more |
| HIP_IMPEDANCE_TORQUE_LIMIT | 1.0 N·m | Keeps hips backdrivable |
| fitness (S5+S8 combined) | 4.0918 | 3127 gens, 25016 evals, 90 min (2026-03-18) |

**Controller structure (per leg, 500 Hz):**
```python
roll_true = atan2(2*(w*x + y*z), 1 - 2*(x²+y²))   # from data.xquat[box_bid]
roll_rate = data.qvel[d_root + 3]                    # ωx world-frame
roll_meas = roll_true + N(0, ROLL_NOISE_STD_RAD)     # BNO086 noise model

δq = K_ROLL * roll_meas + D_ROLL * roll_rate
q_nom_L = clamp(Q_NOM + δq, HIP_SAFE_MIN, HIP_SAFE_MAX)
q_nom_R = clamp(Q_NOM - δq, HIP_SAFE_MIN, HIP_SAFE_MAX)

τ_hip = clamp(-(LEG_K_S*(q_hip - q_nom) + LEG_B_S*dq_hip), ±1.0 N·m)
```

**Key facts:**
- Roll leveling is orthogonal to pitch/yaw — δq has zero effect on average hip height
- HIP_SAFE_MIN = Q_EXT + 10° = −1.257 rad; HIP_SAFE_MAX = Q_RET − 10° = −0.526 rad
- S8 fitness metric: `max_roll_deg` (peak spike, not average) — rewards leveling each individual bump
- K_s hit upper bound (16.0) — optimizer wants maximum vertical stiffness within searched range
- K_ROLL hit upper bound (4.0) — very aggressive roll correction

**Files:**
- `sim_config.py` — all suspension/roll params (LEG_K_S, LEG_B_S, LEG_K_ROLL, LEG_D_ROLL, etc.)
- `scenarios.py` — differential impedance block in `_run_sim_loop`, `run_8_terrain_compliance()`
- `optimize_suspension.py` — (1+8)-ES over 4 params, combined S5+S8 fitness
- `replay.py` — 8-panel telemetry (4×2); Roll and Suspension Δq panels verify leveling

---

### Phase 5 — Realistic Wheel Motor Model (Back-EMF) ⬜ NEXT

**Motivation:** Wheel motors are currently ideal torque sources (±3.67 N·m regardless of speed).
Real 5065 130KV at 24V has torque that tapers linearly with speed:
`T(ω) = T_peak × (1 − ω / ω_noload)` where `ω_noload = KV × V = 130 × 24 × 2π/60 ≈ 327 rad/s`.

At operating speeds (0–3 m/s, ω_wheel = 0–40 rad/s) the taper is only 0–12% — barely felt.
But it matters for:
- **Top speed prediction** — currently unbounded in sim; real robot tops out ~24 m/s at rim
- **High-speed balance** — available torque for disturbance rejection drops at speed
- **Firmware accuracy** — controller expecting full torque at high speed will be surprised

**Implementation plan:**

**5.1 — Add back-EMF clamp to `physics.py`** (or apply in `scenarios.py` / `sandbox.py`)

Option A (MuJoCo actuator): Add `<motor>` with `forcelimited` + velocity-dependent gain via
a custom actuator model — complex in MuJoCo XML.

Option B (control-loop clamp, preferred): After computing `tau_wheel` in `_run_sim_loop` /
`sandbox.py`, clamp with taper before writing to `data.ctrl`:
```python
OMEGA_NOLOAD = 326.7   # rad/s (KV×V×2π/60, from motor_models.py in baseline1)
KT           = 0.0735  # N·m/A
T_PEAK       = 3.67    # N·m

def motor_taper(tau_cmd, omega_wheel):
    """Reduce available torque linearly as speed approaches no-load."""
    taper = max(0.0, 1.0 - abs(omega_wheel) / OMEGA_NOLOAD)
    t_max = T_PEAK * taper
    return float(np.clip(tau_cmd, -t_max, t_max))
```

Apply in `_run_sim_loop` (scenarios.py), `sandbox.py`, and `replay.py` — same pattern as
all other gain sharing (import constants from `sim_config.py`).

Add constants to `sim_config.py`:
```python
WHEEL_OMEGA_NOLOAD = 326.7   # [rad/s]  ω_noload = KV × V_batt × 2π/60
WHEEL_KT           = 0.0735  # [N·m/A]  torque constant (Kt = 1/KV in SI)
```

**5.2 — Re-run sandbox validation**
Check that robot still balances and top speed is physically capped (~24 m/s at rim —
effectively unreachable in sandbox but limit should be visible in telemetry).

**5.3 — Re-optimize if needed**
Taper is small at typical speeds — gains probably don't need re-tuning.
Run S5 replay first; if vel_rms degrades significantly, re-run `optimize_vel_pi.py`.

**5.4 — Add friction model (optional)**
`motor_models.py` in baseline1 has `B_friction = 0.02 N·m·s/rad` viscous drag.
Apply similarly: `tau_net = tau_cmd - B_friction * omega_wheel` before physics step.

**Why Phase 5 before firmware:**
The firmware will implement the same motor limits. Without back-EMF in sim, the tuned gains
could be overconfident at speed — better to discover this in simulation.

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
