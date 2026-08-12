# Components — Wheeled-Leg Robot

The mass inventory below is the authoritative as-built update from 2026-08-09.
Costs, electrical ratings, and structural sizing farther down remain useful,
but their old pre-build mass estimates are superseded by this section.
Original geometry source: `simulation/mujoco/archive/baseline1_leg_analysis/sim_config.py` (run_id 51167, jump = 282.65 mm)
Structural sizing source: `simulation/mujoco/archive/baseline1_leg_analysis/size_report.txt` (SF = 2.0×, 6061-T6)

---

## As-built mass inventory — 2026-08-09

All measured values are scale readings supplied in grams. The two AK45 hip
motors were not reweighed, so their existing 260 g catalog value is retained.

| Component | Qty | Mass each (g) | Accounted mass (g) | Status / placement |
|---|---:|---:|---:|---|
| 6804 bearing | 16 | 18 | 288 | Measured reference; 8 per leg |
| Wheel motor | 2 | 418 | 836 | Measured |
| Electronics/body box, no battery | 1 | 505 | 505 | Measured |
| Battery | 1 | 276 | 276 | Measured; mounted toward body front |
| TPU wheel | 2 | 29 | 58 | Measured |
| PLA rim | 2 | 31 | 62 | Measured |
| PLA coupler | 2 | 35 | 70 | Measured print only |
| PLA femur | 2 | 40 | 80 | Measured print only |
| PLA tibia/dogleg | 2 | 85 | 170 | Measured print only |
| AK45-10 hip motor | 2 | 260 | 520 | Catalog value retained |

Bearing count follows the supplied assembly description: each coupler carries
two bearings at each of its two joints (4 per side), and each tibia carries two
at the coupler joint plus two at the femur joint (4 per side), for 16 total.

Known measured parts plus the two catalog hip motors account for **2,589 g
without the battery**. The measured complete robot is **3,242 g without the
battery**, leaving **653 g** for unweighed fasteners, shafts, mounts, wiring,
and any catalog/component discrepancy. The driving mass with the 276 g battery
is therefore **3,518 g**.

For MuJoCo mass closure, the 653 g remainder is an explicitly labelled
effective mass, not a claim that the printed femurs weigh more. The trim/log fit
currently assigns it to the two femur bodies (326.5 g per side), making each
effective femur body 366.5 g while preserving the measured 40 g print value in
the parameter ledger. The battery is a separate 276 g rigid mass at the front
and bottom packaging limits. T0.2 CG measurements should replace these two
identification choices.

| Mass check | Total (g) |
|---|---:|
| Known measured + retained hip-motor catalog masses, no battery | 2,589 |
| Unweighed residual distributed in the twin | 653 |
| **Complete robot, no battery (measured)** | **3,242** |
| Battery | 276 |
| **Complete driving mass** | **3,518** |

---

## Historical BOM, cost, and design estimates

Masses in the tables below are the pre-build estimates and are retained only
for BOM provenance. Do not use them for simulation mass accounting.

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
| HIP_MOTOR | CubeMars AK45-10 KV75 | 2 | 260 | 520 | 149 | 298 | purchased | Φ53×43 mm, 10:1, **2.5 N·m rated / 7 N·m peak** (2.1 A / 5 A DC), Kt 0.127 N·m/A motor-side = 1.27 at output, 180 rpm no-load, MIT CAN, CAN id L=1 R=2 |
| WHEEL_MOTOR | Maytech MTO5065-70-HA-C | 2 | 450 | 900 | 90 | 180 | purchased | KV70, direct drive, Hall sensors req. for ODESC; Kt=0.1364 Nm/A, T_peak=6.82 Nm @ 50A, ω_noload=175.9 rad/s @ 24V; https://michobby.com/products/maytech-5065-220kv-brushless-outrunner-motor-for-electric-skateboards-e-bike (70KV variant) |

**Subtotal motors:** 1420 g / $478

---

## Structural — Links (6061-T6 Aluminium Tube)

Dimensions from winning optimisation; load cases at 2× simulation peak. All SF verified ≥ 2.0 (yield) and ≥ 5.0 (buckling).

| ID | Link | OD × wall | Length | Qty | Mass ea (g) | Total (g) | Cost ea ($) | Total ($) | SF yield | SF buck |
|---|---|---|---|---|---|---|---|---|---|---|
| FEMUR_TUBE | Femur (A → C) | 14 × 1.0 mm | 174 mm | 2 | 19.2 | 38.4 | 5 | 10 | 2.29 | 21 |
| TIBIA_TUBE | Tibia (C → W + stub C → E) | 16 × 1.0 mm | 144 mm | 2 | 18.3 | 36.6 | 5 | 10 | 2.36 | 35 |
| COUPLER_TUBE | Coupler (F → E) | 10 × 1.0 mm | 151 mm | 2 | 11.5 | 23.1 | 4 | 8 | 5.79 | 7 |

Peak axial loads (design case = 2× sim peak): femur 920 N, tibia 1234 N, coupler 1102 N.

**Subtotal links:** 98.1 g / $28

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
| WHEEL | Wheel (**112 mm OD**, v4) | PLA hub + TPU tread | 2 | 70 | 140 | Was 150 mm OD before v4 (2026-08-07) — `WHEEL_R` = 0.056 m in `control_loop.cpp`. PLA spoked hub ~45g + TPU tread band ~25g; D-shaft mount to 5065 motor. Mass not re-estimated for the smaller wheel. |

**Subtotal printed:** 440 g / filament cost only

---

## Power

| ID | Part | Qty | Mass ea (g) | Total (g) | Cost ea ($) | Total ($) | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| BATTERY | 6S LiPo 1800 mAh | 1 | 275 | 275 | 35 | 35 | designed | 6S, 22.2V nom / 25.2V full; 50C continuous (90A), 100C burst (180A) |
| BATT_ADAPTER | XT60 connector + pigtail | 1 | 10 | 10 | 5 | 5 | designed | |

**Subtotal power:** 285 g / $40

---

## Wiring & Fasteners

| ID | Part | Qty | Mass (g) | Cost ($) | Status | Notes |
|---|---|---|---|---|---|---|
| MOTOR_CONN | Amass MR30 | 2 sets | 6 | 12 | purchased | 3-pin, 30A cont / 60A pulse |
| WIRING | Wiring harness | 1 lot | 100 | 15 | planned | CAN bus, power, signal |
| FASTENERS | M3/M4 hardware | 1 lot | 50 | 5 | planned | Bolts, standoffs, inserts |

**Subtotal wiring/fasteners:** 156 g / $32

---

## Historical pre-build Mass & Cost Summary (superseded for mass)

### Detailed component breakdown

| # | Component | Qty | Each (g) | Total (g) | Category |
|---|---|---|---|---|---|
| 1 | Battery (6S LiPo 1800 mAh) | 1 | 275 | 275 | Power |
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
| 17 | Coupler tube (10×1.0 mm Al) | 2 | 11.5 | 23.1 | Links |
| 18 | FlySky FS-iA6B receiver | 1 | 15 | 15 | Electronics |
| 19 | Motor connectors (MR30) | 2 | 3 | 6 | Wiring |
| 20 | BNO086 IMU | 1 | 3 | 3 | Electronics |
| 21 | SN65HVD230 CAN transceiver | 1 | 1 | 1 | Electronics |
| | | | | | |
| | **TOTAL** | | | **2783** | |
| | **+10% contingency** | | | **~3061 g = 3.1 kg** | |

### By category

| Category | Mass (g) | Cost ($) |
|---|---|---|
| Power | 285 | 40 |
| Motors | 1420 | 478 |
| Printed parts | 440 | ~35 |
| Electronics & Controls | 244 | 111 |
| Wiring & Fasteners | 156 | 32 |
| Bearings | 140 | 18 |
| Links (Al tube) | 98 | 28 |
| **TOTAL** | **2783** | **742** |
| **+10% contingency** | **~3061 g = 3.1 kg** | **~$816** |

---

## Leg Geometry — v4 (2026-08-07)

Mechanical baseline drawing: **`components/2N_10mm_279mm.pdf`**. Machine-readable
form of the same geometry (agrees to 0.01 mm):
`simulation/2d/fourbar_optimizer_gui/presets/2N_10mm_279mm.json`.
**All coordinates are relative to A** (hip motor output shaft), +X forward, +Z up.

| Parameter | v4 | Previous (18 mm as-built) | Notes |
|---|---|---|---|
| L_femur | **187.58 mm** | 173.78 mm | A → C |
| L_coupler | **169.54 mm** | 150.81 mm | F → E |
| L_stub (EC) | **39.01 mm** | 35.13 mm | C → E (upward) |
| Tibia \|C→W\| | **185.91 mm** | 129.39 mm | Knee to wheel |
| E → W | **224.49 mm** | — | |
| C_offset | **5.28 mm** | 0 (straight) | Knee C's perpendicular offset from the E–W line, **away from the hip motor**. Equivalent preset form: W at 183.41 mm along the C→E axis + 30.35 mm perpendicular (9.4° bend at C) |
| F_X from A | **−36.42 mm** | −58.87 mm | |
| F_Z from A | **+37.54 mm** | +18.00 mm | \|AF\| = 52.30 mm; motor clearance +25.80 mm |
| A_Z offset | −23.5 mm | −23.5 mm | Hip axis below body centre — **inherited from baseline-1, NOT re-measured on the v4 box** |
| Q_retracted | **+0.4887 rad (+28°)** | — | Retract hard stop; 0° = femur horizontal, positive retracts |
| Q_extended | **−0.9948 rad (−57°)** | — | Extended hard stop, essentially on the 4-bar singularity |
| Stroke | **85.0°** | 66° | Stop to stop |
| Vertical wheel travel | **276.62 mm** over the stops; **279.95 mm** on the drawing | 209 mm | The drawing's 279.95 is over the optimizer's evaluated band (−57.64°…+28.65° = 86.29°), which runs slightly past both stops. 276.62 is what the firmware can actually command. |
| Ride height (A above ground) | **119.3 → 395.9 mm** | — | Retracted → extended |
| Peak static hip torque | **2.00 N·m** @ −12.4° | 2.5 N·m | At 1.0 kg/leg |

> **Express F relative to A.** The −18.21 mm figure that used to sit in this
> table is F's Z in the *body-centre* frame; it was mistaken for the A→F offset
> and a robot was built to it (`AngleRetractedExt.md`). The two differ by `A_Z`,
> which is itself unverified for v4 — so the A-relative numbers above are the
> measured ones and any body-centre F_Z for v4 is provisional.
>
> Everything derived from the pre-v4 geometry is invalid: the baseline-1
> optimisation result, the old stroke angles, and the old jump-height figure
> (282.65 mm, run_id 51167). `L_EFF_RET`/`L_EFF_EXT` in `control_loop.cpp` have
> been recomputed for v4; `M_BODY` has not.
