# CLAUDE.md — Wheeled-Leg Robot

This file gives Claude Code full context on the robot design so it can assist
with simulation, firmware, and control code without needing re-explanation each session.

---

## Working Rules (always follow these)

- **Ask when in doubt.** Never go deep into an approach if something is ambiguous — ask first.
- **3-minute rule.** Every interaction must complete in under 3 minutes of work. If a task will take longer, stop and propose smaller intermediate steps so the user can confirm progress at each stage before continuing.
- Always run the file you are working with after you are done editing it.

---

## Control Architecture

**See `docs/Control.MD` for:**
- Complete 4-controller cascade (Balance LQR + Leg Impedance + Velocity PI + Yaw PI)
- Optimizer scenarios S1–S8 and per-phase tuning strategies
- Latency model framework and gain re-tuning for real hardware delays

**Current Status (2026-03-20):**
- ✅ Phase 1: 3-State LQR + Gain Scheduling — COMPLETE & S4-BASELINED
  - Q=[0.014168, 0.033720, 0.000250], R=28.734 (1896 evals, fitness=0.018, RMS pitch 1.24°)
- ✅ Phase 2: Velocity PI — COMPLETE & SANDBOX-RETUNED
  - KP_V=0.502418, KI_V=0.011405, THETA_REF_RATE_LIMIT=5.0 rad/s (optimizer was too conservative)
- ✅ Phase 3: Yaw PI — COMPLETE & RE-BASELINED (optimizer)
  - KP_YAW=2.272, KI_YAW=1.125 (55752 evals / 75 min, fitness=0.4102)
- ✅ Phase 4: Leg Impedance + Roll Leveling — COMPLETE & RE-BASELINED (optimizer)
  - K_s=16.0, B_s=0.799, K_ROLL=4.0, D_ROLL=1.0 (25016 evals / 90 min, fitness=4.0918)
- ✅ Phase 5: Realistic Wheel Motor Model — COMPLETE
  - Back-EMF taper, 6S LiPo battery model, voltage-scaled no-load speed (in LQR_Control_optimization/)
- ✅ Phase 6: Latency Sensitivity — COMPLETE (all 4 steps, latency_sensitivity/)
  - Delayed baselines: LQR Q=[0.0634,0.000219,0.000011] R=1.98 · VPI Kp=0.376 · Yaw Kp=2.192 Ki=0.427 · Suspension K_s=2.04

---

## Coordinate System (canonical, applies everywhere)

| Axis | Direction |
|---|---|
| **+X** | Robot forward |
| **+Y** | Robot left |
| **+Z** | Robot up |

This matches MuJoCo's default world frame. All simulation files, firmware, and CAD must use this convention.

---

## Project Overview

Home-built two-wheeled bipedal balancing robot. Balances as an inverted pendulum on two wheels,
with 4-bar linkage legs for terrain clearance and jumping.

Reference: Impulse by Aaed Musa (aaedmusa.com/projects/impulse) — our design targets ~11× better
jump height (~283 mm vs ~26 mm) via AK45-10 hip motors replacing GIM6010-8 + ODrive Micro.

---

## Architecture — Baseline 1

Geometry from evolutionary optimisation (run_id 51167, jump = 282.65 mm).
See `components/COMPONENTS.md` for full geometry table, BOM, and mass breakdown.
Simulation source of truth: `simulation/mujoco/baseline1_leg_analysis/sim_config.py`

### 4-bar Leg Topology

```
  [  body box  ]
  F────────────A  ← AK45-10 hip motor
  |  coupler   |
  |  link      | femur (173.78 mm)
  |            |
  E────────────C  ← knee pivot
  (red pin)  (white pin)
               |
               | tibia (129.39 mm down to W, 35.13 mm stub up to E)
               |
               W  ← wheel centre (Ø150 mm, Maytech MTO5065-70-HA-C direct drive)
```

- A = hip motor output shaft (femur origin)
- F = fixed body pivot at (−58.87 mm X, −18.21 mm Z) from body origin
- C = knee pivot (femur tip)
- E = tibia stub end (35.13 mm above C), connects to coupler at F

---

## Components

Full BOM, geometry, and motor electrical specs: `components/COMPONENTS.md` (single source of truth).
Machine-readable BOM: `components/database/bom.yaml`

---

## Software Stack

- **Firmware:** PlatformIO, Arduino framework, UNO R4 WiFi
- **OTA:** ESP32-S3 handles WiFi OTA (`upload_protocol = espota`)
- **Simulation:** Python + MuJoCo (Windows native, `pip install mujoco`)
- **Balance algorithm:** LQR on linearised inverted pendulum, 500 Hz, 3-state: [pitch−θ_ref, pitch_rate, wheel_vel_avg−v_ref] — fully tuned in simulation
- **Wheel odometry:** ODESC encoder feedback via CAN

---

## Simulation — Current State

### Active: `simulation/mujoco/LQR_Control_optimization/`

Primary sim folder for control tuning (Phases 1–5). All 8 optimizer scenarios, replay, and sandbox.

| File | Purpose |
|---|---|
| `sim_config.py` | All control gains, scenario params, motor/battery constants |
| `scenarios.py` | All scenario runners, `_run_sim_loop`, `VelocityPI`, `YawPI` classes |
| `optimize_lqr.py` / `optimize_vel_pi.py` / `optimize_yaw_pi.py` / `optimize_suspension.py` | (1+8)-ES optimizers per controller |
| `replay.py` | MuJoCo viewer + 8-panel telemetry |
| `sandbox_fastchart.py` | Interactive arena — 28 obstacles, gamepad + sliders, pyqtgraph |
| `lqr_design.py` | LQR solver + gain scheduling |

### Active: `simulation/mujoco/latency_sensitivity/`

Full copy of LQR_Control_optimization with 10-step sensor delay + 5-step actuator delay (ring buffers). Phase 6 — all 4 latency re-tuning steps complete.

### Reference: `simulation/mujoco/baseline1_leg_analysis/`

Two-leg robot with jump controller, balance PD, and force analysis. Source of truth for geometry/mass.

| File | Purpose |
|---|---|
| `sim_config.py` | Geometry, motor, and structural params (run_id 51167) |
| `viewer.py` | MuJoCo GUI viewer with jump, telemetry, CG marker |
| `motor_models.py` | Realistic per-axis motor models |

### Reference: `simulation/mujoco/`

- `4bar_leg.xml` + `4bar_leg_sim.py` — original single-leg 4-bar kinematic prototype (keep, do not modify)
- `fourbar_ref/` — reference implementations that informed the topology

---

## Folder Structure

```
firmware/src/                        ← PlatformIO firmware
software/
  dashboard/                         ← Dear PyGui telemetry (planned)
  tools/                             ← odrivetool scripts, calibration
simulation/
  mujoco/
    LQR_Control_optimization/        ← ✅ Active — control tuning (Phases 1–5)
      sim_config.py                  ← all gains + scenario params
      scenarios.py                   ← all scenario runners + controller classes
      replay.py / sandbox_fastchart.py ← viewer + interactive arena
    latency_sensitivity/             ← ✅ Active — Phase 6 (delayed plant re-tuning)
      logs/                          ← CSV + log files (S{name}.csv naming)
    baseline1_leg_analysis/          ← reference — geometry + jump sim
      sim_config.py                  ← source of truth for geometry + mass
    4bar_leg.xml / 4bar_leg_sim.py   ← single-leg kinematic reference
    fourbar_ref/                     ← reference 4-bar implementations
  sil/                               ← C++ DLL bridge (future)
docs/
  math/                              ← LQR derivations, jump energy
  design_decisions/
  datasheets/
components/
  COMPONENTS.md                      ← Full BOM/MEL with subtotals
  database/bom.yaml                  ← Machine-readable BOM (source of truth)
params/robot_params.yaml             ← Physical params
.claude/settings.json                ← Claude Code permissions
```

---

## Key Design Decisions

- **Synthesised 4-bar over 2-DOF:** Single motor per leg, passive knee coupling. Baseline1 geometry confirmed in MuJoCo (run_id 51167, 282.65 mm jump).
- **AK45-10 over GIM6010-8:** GIM6010-8 limited to 3 N·m by ODrive Micro 7A limit; AK45-10 delivers 7 N·m with integrated driver.
- **Maytech MTO5065-70-HA-C (KV70) wheel motor:** Direct drive, Hall sensors for ODESC, 6.82 N·m peak @ 50A; no gearbox needed — FOC gives full torque at zero RPM.
- **ODESC 3.6 over ODrive Pro:** Saves $189, identical firmware/API, 50A vs 40A continuous.
- **24V battery:** Compatible with all motors (AK45-10 ≤36V, 5065 ≤unrestricted, ODESC ≤56V).
- **UNO R4 over Teensy 4.1:** Native CAN + WiFi/OTA built-in; 32KB RAM adequate for 500Hz LQR.
- **HIL via USB not WiFi:** WiFi ~20ms jitter is fatal for 500Hz loop; WiFi used only for telemetry.

---

## Open Tasks

### Simulation
- [x] 4-bar single-leg kinematics + viewer
- [x] Two-leg model with ground contact + balance controller
- [x] Jump controller + force/bearing analysis (baseline1_leg_analysis)
- [x] Tune LQR + VelocityPI + YawPI + Suspension gains (Phases 1–4, LQR_Control_optimization)
- [x] Realistic motor/battery model + latency sensitivity (Phases 5–6)
- [ ] Port controller to C++ for SIL (`simulation/sil/`)

### Firmware
- [ ] Scaffold balance loop (500 Hz, BNO086 → pitch → wheel torque cmd)
- [ ] CAN driver for AK45-10 MIT protocol
- [ ] ODESC wheel velocity/torque interface
- [ ] HIL bridge via USB serial

### Hardware
- [ ] Finalise 24V battery selection
- [ ] Design PLA hub + TPU tread wheel (150mm OD, D-shaft mount)
- [ ] CAD: body box with AK45-10 mounts and ODESC/MCU tray
