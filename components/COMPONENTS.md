# Components — Wheeled-Leg Robot
**Single source of truth: BOM + MEL**
Geometry source: `simulation/mujoco/baseline1_leg_analysis/sim_config.py` (run_id 51167, jump = 282.65 mm)
Structural sizing source: `simulation/mujoco/baseline1_leg_analysis/size_report.txt` (SF = 2.0×, 6061-T6)

---

## Electronics & Controls

| ID | Part | Qty | Mass ea (g) | Total (g) | Cost ea ($) | Total ($) | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| MCU | Arduino UNO R4 WiFi | 1 | 45 | 45 | 28 | 28 | designed | RA4M1 + ESP32-S3, native CAN, WiFi OTA |
| IMU | BNO086 | 1 | 3 | 3 | 20 | 20 | designed | 500 Hz Game Rotation Vector, I2C |
| WHEEL_CTRL | ODESC 3.6 Dual Drive | 1 | 160 | 160 | 60 | 60 | designed | ODrive v0.5.x, axis0=L axis1=R, CAN id=3 |
| CAN_XCVR | SN65HVD230 | 1 | 1 | 1 | 4 | 4 | designed | 3.3V CAN transceiver |
| BUCK_5V | DC-DC buck 24V→5V | 1 | 20 | 20 | 8 | 8 | designed | Powers MCU + IMU |

**Subtotal electronics:** 229 g / $120

---

## Motors

| ID | Part | Qty | Mass ea (g) | Total (g) | Cost ea ($) | Total ($) | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| HIP_MOTOR | CubeMars AK45-10 KV75 | 2 | 260 | 520 | 149 | 298 | designed | Φ53×43 mm, 10:1, 7 N·m peak, MIT CAN, CAN id L=1 R=2 |
| WHEEL_MOTOR | Maytech MTO5065-70-HA-C | 2 | 200 | 400 | 30 | 60 | designed | KV70, direct drive, Hall sensors req. for ODESC; Kt=0.1364 Nm/A, T_peak=6.82 Nm @ 50A, ω_noload=175.9 rad/s @ 24V; https://maytech.cn/products/mto5065-170-ha-c?variant=29503884492894 |

**Subtotal motors:** 920 g / $358

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
| BATT_ADAPTER | Power Wheels → XT60 | 1 | 45 | 45 | 12 | 12 | designed | |

**Subtotal power:** 765 g / $47

---

## Wiring & Fasteners

| ID | Part | Qty | Mass (g) | Cost ($) | Notes |
|---|---|---|---|---|---|
| WIRING | Wiring harness | 1 lot | 100 | 15 | CAN bus, power, signal |
| FASTENERS | M3/M4 hardware | 1 lot | 50 | 5 | Bolts, standoffs, inserts |

**Subtotal wiring/fasteners:** 150 g / $20

---

## Mass & Cost Summary

| Category | Mass (g) | Cost ($) |
|---|---|---|
| Electronics & Controls | 229 | 120 |
| Motors | 920 | 358 |
| Links (Al tube) | 94 | 28 |
| Bearings | 140 | 18 |
| Printed parts | 440 | ~35 (filament) |
| Power | 765 | 47 |
| Wiring & Fasteners | 150 | 20 |
| **TOTAL** | **2738** | **626** |
| **+10% contingency** | **~3012 g = 3.0 kg** | **~$689** |

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
