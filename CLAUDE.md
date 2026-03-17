# CLAUDE.md — Two-Wheeled Legged Robot Project

This file gives Claude Code full context on the robot design so it can assist
with simulation, firmware, and control code without needing re-explanation each session.

---

## Working Rules (always follow these)

- **Ask when in doubt.** Never go deep into an approach if something is ambiguous — ask first.
- **3-minute rule.** Every interaction must complete in under 3 minutes of work. If a task will take longer, stop and propose smaller intermediate steps so the user can confirm progress at each stage before continuing.
- Always run file you are working with after you are done editing it. 

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

Home-built two-wheeled bipedal balancing robot inspired by the Ascento and
Impulse (Aaed Musa) designs. The robot balances as an inverted pendulum on two
wheels, with telescoping legs for terrain clearance and jumping.

**Reference build:** Impulse robot by Aaed Musa — aaedmusa.com/projects/impulse
- 5.35 kg, 140 mm links, GIM6010-8 × 4 hip motors, DBS2.0 hub motors
- Teensy 4.1 at 100 Hz PID, 24V Kobalt battery, ODrive Micro
- Jump height only ~26 mm (torque-starved by ODrive Micro's 7A limit)
- Our design targets 11× better jump height (~28.6 cm) by using AK45-10 motors

---

## Architecture

| Parameter | Value |
|---|---|
| Leg type | 1-DOF, synthesised 4-bar linkage, single AK45-10 per leg |
| Legs | 2 (left and right) |
| DOF per leg | 1 (hip motor drives full leg extension) |
| Femur length | 100 mm (A→C) |
| Tibia length | 115 mm (C→W) + 15 mm stub (C→E, upward) |
| Coupler length | 110 mm (F→E) |
| Wheel diameter | 150 mm (3D printed TPU) |
| Total mass | ~3.4 kg |

### Leg Topology — Confirmed 4-bar Linkage

Single AK45-10 hip motor at pivot **A** (mounted on +Y face of body box).
A synthesised 4-bar linkage enforces the knee angle passively:

```
  [  body box  ]
  F────────────A  ← AK45-10 hip motor (Φ53 mm)
  |  ground    |
  |  link      | femur (100 mm, yellow in sim)
  |            |
  E────────────C  ← knee pivot (white pin in sim)
  (red pin)
               |
               | tibia (115 mm down to W, 15 mm stub up to E, purple in sim)
               |
               W  ← wheel centre (Ø150 mm, Maytech hub motor)
```

**4-bar pivot geometry (confirmed in MuJoCo sim):**
- A = hip motor output shaft (origin of femur body)
- F = fixed body pivot, offset from A by (−15 mm X, +26 mm Z)
- C = knee pivot (femur tip, 100 mm along femur from A)
- E = tibia stub end (15 mm above C in tibia frame), connected to coupler at F

**Impulse note:** Impulse has separate hip + knee motors (3-DOF per leg). Our design
is fundamentally different — single motor, passive 4-bar coupling.

**Vertical wheel constraint:** Wheel centre W stays approximately below body centre
during the jump power stroke (mid-range hip angles). Exact vertical not required.

---

## Bill of Materials (Final)

| Component | Part | Qty | Mass | Cost |
|---|---|---|---|---|
| Hip motor | CubeMars AK45-10 KV75 | 2 | 260 g ea | ~$149 ea |
| Wheel motor | Maytech MTO7052HBM 60KV | 2 | 380 g ea | ~$45 ea |
| Wheel | 3D printed TPU 150 mm dia | 2 | ~30 g ea | filament |
| Wheel controller | ODESC 3.6 Dual Drive | 1 | 160 g | ~$60 |
| MCU | Arduino UNO R4 WiFi | 1 | 45 g | ~$28 |
| IMU | BNO086 | 1 | 3 g | ~$20 |
| CAN transceiver | SN65HVD230 | 1 | <1 g | ~$4 |
| Battery | 24V battery (TBD) | 1 | 720 g | ~$35 |
| Battery adapter | Power Wheels → XT60 | 1 | 45 g | ~$12 |
| Body structure | PLA printed | — | 210 g | filament |
| Leg links (×4) | PLA printed | 4 | 44 g ea | filament |
| Motor mounts | PLA printed | — | ~90 g | filament |
| Bearings 608 | Standard skateboard | 12 | 12 g ea | ~$8 |
| Wiring + fasteners | — | — | ~150 g | ~$20 |
| 5V regulator | DC-DC buck | 1 | ~20 g | ~$8 |
| **TOTAL** | | | **~3,010 g + 10% = ~3.4 kg** | **~$628** |

### Key Component Notes

**AK45-10 (hip motor)**
- 260 g, 10:1 planetary, 7 N·m peak torque, 75 KV
- Integrated FOC driver, MIT CAN protocol on CAN bus
- Rated 12–36V; runs fine at 24V
- CAN node IDs: Left = 1, Right = 2

**Maytech MTO7052HBM (wheel motor)**
- 380 g, 70×52 mm outrunner hub motor, 60KV, Hall sensors
- 22A max continuous, rated 12–42V
- At 24V no-load: ~1440 RPM → 11.3 m/s theoretical wheel speed
- Controlled via ODESC 3.6 (ODrive firmware, axis0 = left, axis1 = right)

**ODESC 3.6 Dual Drive**
- Runs ODrive v0.5.x firmware — use odrivetool for calibration
- 8–56V, 50A continuous / 120A peak per axis
- Identical API to ODrive Pro but $189 cheaper

**Arduino UNO R4 WiFi**
- RA4M1 Cortex-M4 48 MHz main MCU + ESP32-S3 240 MHz coprocessor
- 32 KB SRAM, 256 KB flash
- Native CAN controller (needs SN65HVD230 transceiver for PHY)
- ESP32-S3 handles WiFi; use PlatformIO OTA via `upload_protocol = espota`
- 12×8 LED matrix onboard (use for pitch angle / status display)
- 6–24V VIN range — 24V battery connects directly (24V is the rated maximum)

**BNO086 (IMU)**
- 9-axis, 32-bit ARM Cortex-M0+ onboard
- Outputs fused orientation + calibrated accel/gyro
- Use Game Rotation Vector report at 500 Hz for balance loop

**24V battery (TBD)**
- Powers everything: AK45-10 (rated 12–36V), Maytech (12–42V), ODESC (8–56V)
- 5V rail for MCU/IMU via DC-DC buck from 24V

---

## Software Stack

### Development Environment
- **IDE:** VSCode with Claude Code extension (sidebar chat)
- **Firmware:** PlatformIO (Arduino framework for UNO R4)
- **OTA flashing:** ESP32-S3 on UNO R4 handles WiFi OTA transparently
- **Claude Code permissions:** `.claude/settings.json` — `acceptEdits` mode, `Bash(*)` fully allowed
- **`code` CLI fix:** `~/.bashrc` shell function bypasses broken Git Bash wrapper (Windows)

### Simulation Pipeline (Model → Hardware)

| Stage | Tool | Status |
|---|---|---|
| 4-bar kinematics | Python + MuJoCo | ✅ Working — `4bar_leg_sim.py` |
| MIL balance | Python + MuJoCo | ⬜ Not started |
| SIL | C++ DLL + MuJoCo | ⬜ Not started |
| HIL | Real UNO R4 via USB serial | ⬜ Not started |
| Real robot | UNO R4 + CAN + motors | ⬜ Not started |

**MuJoCo install:** `pip install mujoco` (Windows native, no WSL needed)

### Balance Controller (planned)
- **Algorithm:** LQR (Linear Quadratic Regulator) on linearized inverted pendulum
- **State vector:** [body_pitch, pitch_rate, wheel_pos, wheel_vel, leg_length, leg_vel]
- **Loop rate:** 500 Hz on RA4M1 core
- **Sensor fusion:** BNO086 Game Rotation Vector at 500 Hz → pitch + pitch rate
- **Wheel odometry:** ODESC encoder feedback via CAN

---

## MuJoCo Simulation — Current State

**Working model:** `simulation/mujoco/4bar_leg.xml` + `4bar_leg_sim.py`

Single-leg 4-bar visualisation with sinusoidal hip animation and matplotlib telemetry.
See `simulation/mujoco/fourbar_ref/` for the reference implementation that informed the topology.

Key model parameters:
```
body_mass     = 1.260 kg (box only; single-leg sim)
femur         = 100 mm, tibia = 115 mm + 15 mm stub, coupler = 110 mm
wheel_radius  = 75 mm, wheel_width = 30 mm
hip_ctrlrange = −1.388 to −0.244 rad (safe 4-bar range, mirrored geometry)
hip kp        = 10000 N·m/rad (stiff position servo)
joint damping = 5 N·m·s/rad (mechanism joints), 0.05 (wheel)
connect constraint: solref="0.002 1" solimp="0.9999 0.9999 0.001"
```

---

## Project Folder Structure

```
firmware/src/            ← PlatformIO firmware (platformio.ini scaffolded)
software/
  dashboard/             ← Dear PyGui telemetry (planned)
  ota/                   ← OTA push tools
  tools/                 ← odrivetool scripts, calibration
simulation/
  mujoco/
    4bar_leg.xml         ← ✅ Working single-leg 4-bar MJCF model
    4bar_leg_sim.py      ← ✅ MuJoCo viewer + matplotlib telemetry
    fourbar_ref/         ← Reference 4-bar implementations (keep)
  2d/                    ← 2D linkage visualiser (matplotlib, step01–step08)
  sil/                   ← C++ DLL bridge (future)
docs/
  math/                  ← Derivations (LQR, jump energy, kinematics)
  design_decisions/      ← Rationale docs
  datasheets/            ← Local PDF copies
data/logs/ test_results/ tuning/
params/robot_params.yaml     ← Physical params (source of truth)
components/database/bom.yaml ← BOM with status tracking
.claude/settings.json        ← Claude Code permissions config
```

---

## Key Design Decisions Log

- **Synthesised 4-bar over 2-DOF or timing belt:** Single motor per leg, passive coupling,
  now confirmed working in MuJoCo. Coupler pivot F at (−15 mm X, +26 mm Z) from hip A.
- **AK45-10 over GIM6010-8:** GIM6010-8 only delivers 3 N·m with ODrive Micro
  (7A limit); AK45-10 delivers full 7 N·m with integrated driver
- **ODESC 3.6 over ODrive Pro:** Saves $189, identical ODrive firmware/API,
  50A vs 40A continuous; adequate for home build
- **150mm TPU wheel over 6.5" hoverboard wheel:** Saves 1010 g total
  (2×570g → 2×30g), faster, better clearance
- **24V battery:** compatible voltage with all motors (AK45-10 max 36V, Maytech max 42V, ODESC max 56V)
- **UNO R4 over Teensy 4.1:** Native CAN, WiFi/OTA built-in, adequate 32KB RAM
  for 500Hz LQR loop; Impulse used Teensy 4.1 at only 100Hz PID
- **HIL via USB not WiFi:** WiFi has ~20ms jitter, fatal for 500Hz loop;
  WiFi used only for telemetry

---

## Open Tasks

### Simulation — MuJoCo
- [x] 4-bar single-leg kinematics + viewer (`4bar_leg_sim.py`)
- [ ] Add second leg (mirror in Y), full two-leg model
- [ ] Add ground plane + wheel contact, free-floating body
- [ ] Implement LQR balance controller in Python — MIL
- [ ] Port LQR to C++ for SIL (`simulation/sil/`)
- [ ] Tune LQR gains in simulation before hardware
