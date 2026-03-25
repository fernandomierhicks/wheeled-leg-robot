# CLAUDE.md — Wheeled-Leg Robot




## Working Rules (always follow these)

- **Ask when in doubt.** Never go deep into an approach if something is ambiguous — ask first.
- **3-minute rule.** Every interaction must complete in under 3 minutes of work. If a task will take longer, stop and propose smaller intermediate steps so the user can confirm progress at each stage before continuing.
- Always run the file you are working with after you are done editing it.

---

## Control Architecture

**See `docs/Control.MD` for:**

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

Shopping list and best-estimate specs: `components/COMPONENTS.md`.
Each simulation folder owns its own parameters — sims may deviate from COMPONENTS.md intentionally.

---

## Software Stack

- **Firmware:** PlatformIO, Arduino framework, UNO R4 WiFi
- **OTA:** ESP32-S3 handles WiFi OTA (`upload_protocol = espota`)
- **Simulation:** Python + MuJoCo (Windows native, `pip install mujoco`)
- **Balance algorithm:** LQR on linearised inverted pendulum, 500 Hz, 3-state: [pitch−θ_ref, pitch_rate, wheel_vel_avg−v_ref] — fully tuned in simulation
- **Wheel odometry:** ODESC encoder feedback via CAN

---

## Simulation — Current State

### Active: `simulation\mujoco\master_sim_jump\`

Each sim folder owns all its parameters in `params.py`. New sims fork from the
previous one and note lineage in a comment at the top of `params.py`.

## Folder Structure

```
firmware/src/                        ← PlatformIO firmware
software/
  dashboard/                         ← Dear PyGui telemetry (planned)
  tools/                             ← odrivetool scripts, calibration
simulation/
  mujoco/
    master_sim_jump/                 ← ✅ Active — balance + jump (S1–S10)
      params.py                      ← all parameters (geometry, gains, timing)
      sim_loop.py                    ← main simulation loop
      scenarios/                     ← per-scenario configs & profiles
      controllers/                   ← LQR, VelocityPI, YawPI, hip, jump
    master_sim/                      ← prior iteration (balance only, no jump)
    archive/                         ← old sims (baseline1, LQR_opt, latency)
  sil/                               ← C++ DLL bridge (future)
docs/
  math/                              ← LQR derivations, jump energy
  design_decisions/
  datasheets/
components/
  COMPONENTS.md                      ← Shopping list / best-estimate BOM
.claude/settings.json                ← Claude Code permissions
```

---
