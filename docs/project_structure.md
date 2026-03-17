# Project Folder Structure

Two-wheeled legged balancing robot — folder guide.

---

## Tree

```
Wheeled leg robot Claude/
│
├── CLAUDE.md                        ← Full project context for Claude Code (always keep updated)
│
├── firmware/                        ← Arduino UNO R4 WiFi firmware (PlatformIO)
│   ├── platformio.ini
│   └── src/
│       ├── main.cpp                 ← setup(), loop(), 500 Hz timer ISR
│       ├── balance.cpp / .h         ← LQR controller
│       ├── imu.cpp / .h             ← BNO086 driver (I2C, 500 Hz)
│       ├── can_bus.cpp / .h         ← CAN init, ISR, TX/RX queues
│       ├── ak45.cpp / .h            ← AK45-10 MIT CAN protocol
│       ├── odesc.cpp / .h           ← ODESC/ODrive CAN protocol
│       ├── leg.cpp / .h             ← Pantograph kinematics (IK + FK)
│       └── telemetry.cpp / .h       ← WiFi WebSocket telemetry (ESP32-S3)
│
├── software/                        ← PC-side Python applications
│   ├── dashboard/                   ← Dear PyGui live telemetry dashboard
│   ├── ota/                         ← OTA firmware push tools
│   └── tools/                       ← Calibration, odrivetool scripts, etc.
│
├── simulation/                      ← All simulation work
│   ├── mujoco/                      ← MuJoCo MJCF model + Python sim scripts
│   │   ├── robot.xml                ← MJCF robot model
│   │   ├── balance_sim.py           ← MIL: LQR + MuJoCo in Python
│   │   └── sil_runner.py            ← SIL: real C++ controller DLL + MuJoCo
│   ├── 2d/                          ← 2D linkage sim (4-bar, IK, optimizer)
│   │   └── linkage_sim.py           ← Matplotlib pantograph visualizer + IK solver
│   └── sil/                         ← Software-in-the-loop glue code
│
├── cad/                             ← CadQuery 3D models (Python scripts)
│   ├── wheel.py                     ← 150 mm TPU wheel, Maytech bore pattern
│   ├── femur.py / tibia.py          ← 140 mm links, 608 bearing seats
│   ├── body_frame.py                ← Main body enclosure
│   ├── hip_mount.py                 ← AK45-10 mount + cable routing
│   ├── battery_dock.py              ← Ryobi P108 snap-in dock
│   ├── links/                       ← Rendered previews
│   ├── body/
│   ├── wheel/
│   └── exports/
│       ├── stl/                     ← Print-ready STL files
│       └── step/                    ← STEP for sharing / import to other CAD
│
├── docs/                            ← Engineering notes, math, explanations
│   ├── project_structure.md         ← This file
│   ├── math/                        ← Derivations, equations
│   │   ├── lqr_derivation.md
│   │   ├── jump_energy_calc.md
│   │   └── pantograph_kinematics.md
│   ├── design_decisions/            ← Rationale for key choices
│   └── datasheets/                  ← PDF datasheets (local copies)
│
├── data/                            ← Runtime data
│   ├── logs/                        ← Timestamped CSV/binary telemetry logs
│   ├── test_results/                ← Bench test results, photos
│   └── tuning/                      ← LQR/PID gain sweeps, Bode plots
│
├── params/                          ← Configuration files
│   └── robot_params.yaml            ← Physical parameters (source of truth)
│
├── components/                      ← Component reference
│   ├── datasheets/                  ← PDF datasheets for all parts
│   └── database/
│       └── bom.yaml                 ← Bill of materials with status tracking
│
└── misc/                            ← Non-code files
    └── spreadsheets/                ← Excel/Calc files, budget, mass budget
```

---

## Key Conventions

| Convention | Rule |
|---|---|
| Units | SI throughout (meters, kg, N·m, rad/s) — convert at I/O boundaries only |
| Angles | Radians in code/sim; degrees in docs/comments |
| Params | `params/robot_params.yaml` is the single source of truth — load from there |
| Logs | Filename format: `YYYYMMDD_HHMMSS_<description>.csv` |
| CAD exports | Always export STL + STEP together before printing |
| Docs | Any math-heavy explanation → write a `.md` file in `docs/math/` |

---

## Simulation Pipeline

```
Python LQR design  →  MuJoCo MIL  →  C++ SIL  →  HIL (USB)  →  Real robot
(balance_sim.py)     (robot.xml)    (sil_runner)  (UNO R4)    (CAN + motors)
```

Each stage uses the same `robot_params.yaml` for physical constants.
