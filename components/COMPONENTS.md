# Components — Wheeled-Leg Robot
**Shopping list / best estimate — not yet built.** Geometry and masses may differ
from simulation params, which are tuned per-sim (see each sim folder's `params.py`).
Original geometry source: `simulation/mujoco/archive/baseline1_leg_analysis/sim_config.py` (run_id 51167, jump = 282.65 mm)
Structural sizing source: `simulation/mujoco/archive/baseline1_leg_analysis/size_report.txt` (SF = 2.0×, 6061-T6)

---

## Electronics & Controls

| ID | Part | Qty | Mass ea (g) | Total (g) | Cost ea ($) | Total ($) | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| MCU | Arduino UNO R4 WiFi | 1 | 45 | 45 | 28 | 28 | designed | RA4M1 + ESP32-S3, native CAN, WiFi OTA |
| IMU | BNO086 | 1 | 3 | 3 | 20 | 20 | designed | 500 Hz Game Rotation Vector, I2C |
| WHEEL_CTRL | ODESC 3.6 Dual Drive | 1 | 160 | 160 | 41 | 41 | purchased | ODrive v0.5.x, axis0=L axis1=R, CAN id=3 |
| CAN_XCVR | SN65HVD230 | 1 | 1 | 1 | 4 | 4 | designed | 3.3V CAN transceiver |
| BUCK_5V | DC-DC buck 24V→5V | 1 | 20 | 20 | 8 | 8 | designed | Powers MCU + IMU |
| RECEIVER | FlySky FS-iA6B | 1 | 15 | 15 | 10 | 10 | purchased | AFHDS 2A, iBUS to Arduino Serial1, Telemetry |

**Subtotal electronics:** 244 g / $111

---

## Motors

| ID | Part | Qty | Mass ea (g) | Total (g) | Cost ea ($) | Total ($) | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| HIP_MOTOR | CubeMars AK45-10 KV75 | 2 | 260 | 520 | 149 | 298 | designed | Φ53×43 mm, 10:1, 7 N·m peak, MIT CAN, CAN id L=1 R=2 |
| WHEEL_MOTOR | Maytech MTO5065-70-HA-C | 2 | 450 | 900 | 90 | 180 | purchased | KV70, direct drive, Hall sensors req. for ODESC; Kt=0.1364 Nm/A, T_peak=6.82 Nm @ 50A, ω_noload=175.9 rad/s @ 24V; https://michobby.com/products/maytech-5065-220kv-brushless-outrunner-motor-for-electric-skateboards-e-bike (70KV variant) |

**Subtotal motors:** 1420 g / $478

---

## Structural — Links (6061-T6 Aluminium Tube)

Dimensions from winning optimisation; load cases at 2× simulation peak. All SF verified ≥ 2.0 (yield) and ≥ 5.0 (buckling).

| ID | Link | OD × wall | Length | Qty | Mass ea (g) | Total (g) | Cost ea ($) | Total ($) | SF yield | SF buck |
|---|---|---|---|---|---|---|---|---|---|---|
| FEMUR_TUBE | Femur (A → C) | 14 × 1.0 mm | 174 mm | 2 | 19.2 | 38.4 | 5 | 10 | 2.29 | 21 |
| TIBIA_TUBE | Tibia (C → W + stub C → E) | 16 × 1.0 mm | 144 mm | 2 | 18.3 | 36.6 | 5 | 10 | 2.36 | 35 |
| COUPLER_TUBE | Coupler (F → E) | 10 × 0.8 mm | 151 mm | 2 | 9.4 | 18.8 | 4 | 8 | 5.79 | 7 |

Peak axial loads (design case = 2× sim peak): femur 920 N, tibia 1234 N, coupler 1102 N.

**Subtotal links:** 93.8 g / $28

---

## Structural — Bearings

One bearing per pivot per leg (2 legs total). Double up if shaft loads require it.

| ID | Pivot | Series | Bore | OD | C₀ (N) | F_peak (N) | s₀ | Qty | Mass ea (g) | Total (g) | Cost ea ($) | Total ($) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BRG_608_A | A — hip pivot | 608 | 8 mm | 22 mm | 1370 | 471 | 2.91 | 2 | 12 | 24 | 1 | 2 |
| BRG_608_C | C — knee pivot | 608 | 8 mm | 22 mm | 1370 | 472 | 2.91 | 2 | 12 | 24 | 1 | 2 |
| BRG_6001_E | E — 4-bar closure (stub tip) | 6001 | 12 mm | 28 mm | 2850 | 1045 | 2.73 | 2 | 17 | 34 | 3 | 6 |
| BRG_6001_F | F — coupler body pivot | 6001 | 12 mm | 28 mm | 2850 | 1045 | 2.73 | 2 | 17 | 34 | 3 | 6 |
| BRG_608_W | W — wheel axle | 608 | 8 mm | 22 mm | 1370 | 126 | 10.87 | 2 | 12 | 24 | 1 | 2 |

Note: 608 bearings total = 6 (replaced original estimate of 12 — E and F now use 6001).

**Subtotal bearings:** 140 g / $18

---

## Printed Parts

| ID | Part | Material | Qty | Mass ea (g) | Total (g) | Notes |
|---|---|---|---|---|---|---|
| BODY | Body box + electronics tray | PLA | 1 set | 210 | 210 | Houses MCU, IMU, ODESC, battery |
| MTR_MNT | Motor mounts | PLA | 2 | 45 | 90 | AK45-10 to body interface |
| WHEEL | Wheel (150 mm OD) | PLA hub + TPU tread | 2 | 70 | 140 | PLA spoked hub ~45g + TPU tread band ~25g; D-shaft mount to 5065 motor |

**Subtotal printed:** 440 g / filament cost only

---

## Power

| ID | Part | Qty | Mass ea (g) | Total (g) | Cost ea ($) | Total ($) | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| BATTERY | 24V battery (TBD) | 1 | 720 | 720 | 35 | 35 | designed | TBD model; 24V nominal, all motors rated ≥ 24V |
| BATT_ADAPTER | XT60 connector + pigtail | 1 | 10 | 10 | 5 | 5 | designed | |

**Subtotal power:** 730 g / $40

---

## Wiring & Fasteners

| ID | Part | Qty | Mass (g) | Cost ($) | Status | Notes |
|---|---|---|---|---|---|---|
| MOTOR_CONN | Amass MR30 | 2 sets | 6 | 12 | purchased | 3-pin, 30A cont / 60A pulse |
| WIRING | Wiring harness | 1 lot | 100 | 15 | planned | CAN bus, power, signal |
| FASTENERS | M3/M4 hardware | 1 lot | 50 | 5 | planned | Bolts, standoffs, inserts |

**Subtotal wiring/fasteners:** 156 g / $32

---

## Mass & Cost Summary

### Detailed component breakdown

| # | Component | Qty | Each (g) | Total (g) | Category |
|---|---|---|---|---|---|
| 1 | Battery (24V) | 1 | 720 | 720 | Power |
| 2 | AK45-10 hip motor | 2 | 260 | 520 | Motors |
| 3 | Maytech MTO5065 wheel motor | 2 | 450 | 900 | Motors |
| 4 | Body box + electronics tray (PLA) | 1 | 210 | 210 | Printed |
| 5 | ODESC 3.6 Dual Drive | 1 | 160 | 160 | Electronics |
| 6 | Wheel (PLA hub + TPU tread) | 2 | 70 | 140 | Printed |
| 7 | Wiring harness | 1 | 100 | 100 | Wiring |
| 8 | Motor mounts (PLA) | 2 | 45 | 90 | Printed |
| 9 | Fasteners (M3/M4 hardware) | 1 | 50 | 50 | Wiring |
| 10 | Arduino UNO R4 WiFi | 1 | 45 | 45 | Electronics |
| 11 | XT60 connector + pigtail | 1 | 10 | 10 | Power |
| 12 | Femur tube (14×1.0 mm Al) | 2 | 19.2 | 38.4 | Links |
| 13 | Tibia tube (16×1.0 mm Al) | 2 | 18.3 | 36.6 | Links |
| 14 | 6001 bearing (E + F pivots) | 4 | 17 | 68 | Bearings |
| 15 | 608 bearing (A, C, W pivots) | 6 | 12 | 72 | Bearings |
| 16 | DC-DC buck 24V→5V | 1 | 20 | 20 | Electronics |
| 17 | Coupler tube (10×0.8 mm Al) | 2 | 9.4 | 18.8 | Links |
| 18 | FlySky FS-iA6B receiver | 1 | 15 | 15 | Electronics |
| 19 | Motor connectors (MR30) | 2 | 3 | 6 | Wiring |
| 20 | BNO086 IMU | 1 | 3 | 3 | Electronics |
| 21 | SN65HVD230 CAN transceiver | 1 | 1 | 1 | Electronics |
| | | | | | |
| | **TOTAL** | | | **3224** | |
| | **+10% contingency** | | | **~3546 g = 3.5 kg** | |

### By category

| Category | Mass (g) | Cost ($) |
|---|---|---|
| Power | 730 | 40 |
| Motors | 1420 | 478 |
| Printed parts | 440 | ~35 |
| Electronics & Controls | 244 | 111 |
| Wiring & Fasteners | 156 | 32 |
| Bearings | 140 | 18 |
| Links (Al tube) | 94 | 28 |
| **TOTAL** | **3224** | **742** |
| **+10% contingency** | **~3546 g = 3.5 kg** | **~$816** |

---

## Winning Geometry (baseline1 optimisation)

| Parameter | Value | Notes |
|---|---|---|
| L_femur | 173.78 mm | A → C |
| L_tibia | 129.39 mm | C → W |
| L_stub | 35.13 mm | C → E (upward) |
| L_coupler | 150.81 mm | F → E |
| F_X offset | −58.87 mm | Coupler pivot X from body origin |
| F_Z offset | −18.21 mm | Coupler pivot Z from body origin |
| A_Z offset | −23.5 mm | Hip motor Z from body centre |
| Q_retracted | −0.351 rad | Full retraction |
| Q_extended | −1.432 rad | Full extension |
| Stroke | 61.93 ° | |
| Jump height | 282.65 mm | run_id 51167 |
