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
- Phase 1 findings: 3-state LQR re-baselined (run_id=6288, 8496 evals, fitness=0.400)
- Implementation phases 2–6: Drive mode, turning, leg suspension, jump recovery
- Per-phase tuning strategies and success criteria

**Current Status (2026-03-18):**
- ✅ Phase 1: 3-State LQR + Gain Scheduling — COMPLETE & S4-BASELINED
  - Q=[0.014168, 0.033720, 0.000250], R=28.734 (1896 evals, fitness=0.018, RMS pitch 1.24°)
- ✅ Phase 2: Velocity PI — COMPLETE & S5-BASELINED
  - KP_V=0.251209, KI_V=0.011405 (1776 evals, vel_rms=0.502 m/s)
- ✅ Phase 3: Yaw PI — COMPLETE & RE-BASELINED (optimizer)
  - KP_YAW=2.272, KI_YAW=1.125 (55752 evals / 75 min, fitness=0.4102) — was visual 0.3/0.05
- ✅ Phase 4: Leg Impedance + Roll Leveling — COMPLETE & RE-BASELINED (optimizer)
  - K_s=16.0, B_s=0.799, K_ROLL=4.0, D_ROLL=1.0 (25016 evals / 90 min, fitness=4.0918)

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
Source of truth: `simulation/mujoco/baseline1_leg_analysis/sim_config.py`

| Parameter | Value |
|---|---|
| Leg type | 1-DOF synthesised 4-bar linkage, single AK45-10 per leg |
| Femur (A→C) | 173.78 mm |
| Tibia (C→W) | 129.39 mm |
| Tibia stub (C→E, upward) | 35.13 mm |
| Coupler (F→E) | 150.81 mm |
| Coupler pivot F | X = −58.87 mm, Z = −18.21 mm from body origin |
| Hip motor Z offset (A) | −23.5 mm from body centre |
| Wheel diameter | 150 mm |
| Stroke | 61.93° (Q_ret = −0.351 rad → Q_ext = −1.432 rad) |
| Total mass (base) | ~2.8 kg (2798 g); ~3.1 kg with 10% contingency |

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
               W  ← wheel centre (Ø150 mm, 5065 130KV direct drive)
```

- A = hip motor output shaft (femur origin)
- F = fixed body pivot at (−58.87 mm X, −18.21 mm Z) from body origin
- C = knee pivot (femur tip)
- E = tibia stub end (35.13 mm above C), connects to coupler at F

---

## Components

Full BOM/MEL: `components/COMPONENTS.md` and `components/database/bom.yaml`

Key facts for coding:

| ID | Part | Key spec |
|---|---|---|
| HIP_MOTOR | CubeMars AK45-10 KV75 | 7 N·m peak, 10:1 planetary, MIT CAN, L=1 R=2 |
| WHEEL_MOTOR | 5065 130KV outrunner | 3.67 N·m peak (50A), direct drive, Hall sensors |
| WHEEL_TYRE | PLA hub + TPU tread | 150 mm OD, 70 g each |
| WHEEL_CTRL | ODESC 3.6 Dual Drive | ODrive v0.5.x, axis0=L axis1=R, CAN id=3 |
| MCU | Arduino UNO R4 WiFi | RA4M1 48 MHz + ESP32-S3, native CAN, 24V VIN max |
| IMU | BNO086 | Game Rotation Vector at 500 Hz, I2C |
| BATTERY | 24V (TBD) | All motors rated ≥24V; 5V rail via DC-DC buck |

**Wheel motor electrical model** (see `motor_models.py`):
- ω_noload = 326.7 rad/s (24.5 m/s at rim)
- Kt = 0.0735 N·m/A → T_peak = 3.67 N·m at ODESC 50A limit
- τ_elec ≈ 2 ms, B_friction = 0.02 N·m·s/rad

---

## Software Stack

- **Firmware:** PlatformIO, Arduino framework, UNO R4 WiFi
- **OTA:** ESP32-S3 handles WiFi OTA (`upload_protocol = espota`)
- **Simulation:** Python + MuJoCo (Windows native, `pip install mujoco`)
- **Balance algorithm (planned):** LQR on linearised inverted pendulum, 500 Hz, state = [pitch, pitch_rate, wheel_pos, wheel_vel, leg_length, leg_vel]
- **Wheel odometry:** ODESC encoder feedback via CAN

---

## Simulation — Current State

### Working: `simulation/mujoco/baseline1_leg_analysis/`

Full two-leg robot with jump controller, balance PD, and force analysis.

| File | Purpose |
|---|---|
| `sim_config.py` | Single source of truth — all geometry, motor, and controller params |
| `viewer.py` | MuJoCo GUI viewer; buttons: Drive Fwd/Bwd, Restart, Neutral, Crouch, Jump |
| `motor_models.py` | Realistic per-axis motor models (back-EMF taper, FOC lag, viscous drag) |
| `force_log.csv` | Force/telemetry log written each run |

Viewer features: slow-motion during jump, matplotlib telemetry (pitch, forces, bearing loads, jump height), CG marker overlay.

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
    baseline1_leg_analysis/          ← ✅ Active two-leg sim (jump + balance)
      sim_config.py                  ← source of truth for geometry + params
      viewer.py                      ← MuJoCo GUI
      motor_models.py                ← motor electrical models
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
- **5065 130KV over Maytech MTO7052HBM:** Saves 360 g total, cheaper; 3.67 N·m direct drive is sufficient for balance (2× gravity margin at 10° lean). No gearbox needed — FOC gives full torque at zero RPM.
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
- [ ] Tune LQR gains in simulation (currently PD placeholder)
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
