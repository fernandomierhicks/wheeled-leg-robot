# HANDOFF.md — Project Status & Next Steps

Read this before starting work. Check `CLAUDE.md` for full design context and `docs/Control.MD` for controller architecture and full scenario specs.

---

## Current Status (2026-03-18)

### Phase 1 — Balance LQR ✅ COMPLETE & S4-BASELINED

LQR gains optimised on S4 (`4_leg_height_gain_sched`) — legs cycling through full stroke + disturbances. S4 is a strict superset of S1.

**S4 baseline (current `sim_config.py`):**
| Param | Value |
|-------|-------|
| Q_PITCH | 0.014168 |
| Q_PITCH_RATE | 0.033720 |
| Q_VEL | 0.000250 |
| R | 28.734420 |
| fitness (S4) | 0.017938 |
| rms_pitch (S4) | 1.242° — survived 12 s (3 leg cycles) |

**S1 reference (fixed-height only, kept for comparison):**
Q=[0.138282, 0.023379, 0.004591], R=9.998298, fitness=0.003267, rms=0.926°

---

### Phase 2 — VelocityPI Outer Loop ✅ COMPLETE & S5-BASELINED

VelocityPI gains optimised on S5 (`5_VEL_PI_leg_cycling`) — the hardest scenario: staircase velocity tracking + ±1N disturbance kicks + legs cycling + 5× bump obstacles, all simultaneously. S5 is a strict superset of S2+S3.

**S5 baseline (current `sim_config.py`):**
| Param | Value |
|-------|-------|
| KP_V | 0.251209 |
| KI_V | 0.011405 |
| fitness (S5) | 2.0635 |
| vel_rms | 0.502 m/s |
| rms_pitch | 5.59° — survived 13 s |
| THETA_REF_RATE_LIMIT | 2.0 rad/s |

**Prior combined_PI baseline (kept for reference):** KP_V=0.502932, KI_V=0.012678, fitness=0.61

---

## Scenario Reference

| # | Name | Controller | Legs | v_desired | Disturbance | Duration | Status |
|---|------|------------|------|-----------|-------------|----------|--------|
| 1 | `1_LQR_pitch_step` | LQR only | fixed Q_NOM | 0 | ±4N at t=2/3s | 5s | ✅ Superseded by S4 |
| 2 | `2_VEL_PI_disturbance` | VelocityPI+LQR | fixed Q_NOM | 0 (hold) | ±1N at t=2/3s | 12s | ✅ Superseded by S5 |
| 3 | `3_VEL_PI_staircase` | VelocityPI+LQR | fixed Q_NOM | staircase | none | 13s | ✅ Superseded by S5 |
| 4 | `4_leg_height_gain_sched` | LQR only | cycling | 0 | ±4N at t=2/3s | 12s | ✅ LQR optimizer target |
| 5 | `5_VEL_PI_leg_cycling` | VelocityPI+LQR | cycling | staircase | ±1N + 5 bumps | 13s | ✅ PI optimizer target |

Full specs in `docs/Control.MD → Optimizer Scenarios`.

---

## Next Task — Extend S5 Optimization (Longer Run)

**Why:** The 5-min S5 run converged quickly (best at gen 91, little improvement after). A longer run (30–60 min) will explore the gain landscape more thoroughly and may find better velocity tracking, especially around direction reversal and bump impacts.

```bash
cd simulation/mujoco/LQR_Control_optimization
python optimize_vel_pi.py --hours 0.5   # 30-min run
```

`optimize_vel_pi.py` is already configured:
- `ACTIVE_SCENARIO = "5_VEL_PI_leg_cycling"`
- Seeds from best result in `results_5_VEL_PI_leg_cycling.csv` (the 5-min run)
- No CSV clear needed — will continue from where the 5-min run left off

**After optimisation:**
1. Get precise best values: `python -c "from run_log import *; ..."`
2. Update `VELOCITY_PI_KP` / `VELOCITY_PI_KI` in `sim_config.py`
3. Verify: `python replay.py --top 1 --csv results_5_VEL_PI_leg_cycling.csv`
4. Document in `docs/Control.MD` and update this file

---

## Phase 3 — Yaw PI (Next Phase After PI Tuning Complete)

Once VelocityPI is fully converged, the next controller to implement is the **Yaw PI** (`docs/Control.MD → Phase 3`):

- Add `YawPI` class: `ω_yaw_error → τ_yaw`, ±0.5 N·m clamp
- Differential mixing: `τ_L = τ_sym + τ_yaw`, `τ_R = τ_sym − τ_yaw`
- New scenario: TURN (ω=1 rad/s for 6.28s, target < 10 cm lateral drift)
- Starting gains: Kp_yaw=0.3, Ki_yaw=0.05

---

## Replay Quick Reference

```bash
cd simulation/mujoco/LQR_Control_optimization

# S5 — current best PI gains (bumps + leg cycling + staircase)
python replay.py --top 1 --csv results_5_VEL_PI_leg_cycling.csv

# S5 — baseline gains (for comparison)
python replay.py --baseline --scenario 5_VEL_PI_leg_cycling

# S4 — LQR gain scheduler validation
python replay.py --top 1 --csv results_4_leg_height_gain_sched.csv

# Any scenario with current sim_config gains
python replay.py --baseline --scenario 4_leg_height_gain_sched
```

Camera follows robot. Telemetry: pitch + pitch cmd, hip angle, pitch rate, velocity + cmd overlay.

---

## Re-running Optimizers

```bash
# VelocityPI on S5 (continues from existing CSV — do NOT clear)
python optimize_vel_pi.py --hours 0.5

# LQR on S4 (continues from existing CSV — do NOT clear)
python optimize_lqr.py --hours 0.5
```

> **Always clear the relevant CSV before re-optimising after changing scenario durations or fitness weights.**

---

## Key Files

| File | Purpose |
|------|---------|
| `sim_config.py` | Single source of truth — all gains, durations, disturbance forces, S5 bump positions |
| `scenarios.py` | All scenario runners + `_run_sim_loop` + `_leg_cycle_profile` |
| `optimize_lqr.py` | (1+8)-ES — Q_PITCH, Q_PITCH_RATE, Q_VEL, R — pointed at S4 |
| `optimize_vel_pi.py` | (1+8)-ES — KP_V, KI_V — pointed at S5 |
| `replay.py` | MuJoCo viewer + 4-panel telemetry, camera follows robot |
| `physics.py` | XML builder — `build_xml(bumps=...)` for S5 bump obstacles |
| `lqr_design.py` | LQR solver + gain scheduling at Q_RET/Q_NOM/Q_EXT |
| `results_4_leg_height_gain_sched.csv` | S4 LQR run history |
| `results_5_VEL_PI_leg_cycling.csv` | S5 PI run history |
| `docs/Control.MD` | Full architecture, scenario specs, phase plans |

---

## Hardware Tasks (Parallel, Not Blocking)

- [ ] Pick specific 5065 130KV motor SKU (Flipsky, T-Motor — needs Hall sensors, D-shaft, ≥40A, 24V)
- [ ] Finalise battery: 24V nominal, ≥4Ah, XT60, ≤750g (6S LiPo or 24V LiFePO4)
- [ ] Design wheel in CAD: PLA hub + TPU tread, 150mm OD, D-shaft mount, target 70g

---

## Future Phase Plans

See `docs/Control.MD` for Phases 3–5 (Yaw PI, Leg Suspension, Jump Recovery).
