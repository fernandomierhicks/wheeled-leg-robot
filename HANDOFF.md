# HANDOFF.md — Project Status & Next Steps

Read this before starting work. Check `CLAUDE.md` for full design context.

---

## What Was Done This Session

- **Rebaselined wheel motor**: Maytech MTO7052HBM 60KV (380g, $45) → generic 5065 130KV outrunner (200g, $30). Saves 360g and $30 total.
- **Updated wheel assembly mass**: Added explicit `WHEEL_TYRE` BOM entry (PLA hub ~45g + TPU tread ~25g = 70g each). Old placeholder was 30g TPU.
- **Propagated changes** to `bom.yaml`, `sim_config.py`, `motor_models.py`, `COMPONENTS.md`, and `CLAUDE.md`.
- **Trimmed CLAUDE.md**: Removed duplicate BOM table, updated to baseline1 geometry, refreshed open tasks.

**Robot total mass is now ~3.0 kg** (2738g + 10% contingency), down from ~3.3 kg.

---

## Current Simulation State

| File | What it does |
|---|---|
| `simulation/mujoco/baseline1_leg_analysis/viewer.py` | Full two-leg MuJoCo sim — balance + jump, matplotlib telemetry, arena |
| `simulation/mujoco/baseline1_leg_analysis/sim_config.py` | All geometry, motor, and controller params — source of truth |
| `simulation/mujoco/baseline1_leg_analysis/motor_models.py` | Per-axis motor model (back-EMF, FOC lag, viscous drag) |

**Balance controller is currently a PD loop** (`PITCH_KP=60`, `PITCH_KD=5`) — not LQR yet. It works well enough to hold stance and execute jumps but has not been formally tuned.

**Jump sim is functional**: crouch → extend → air → land, with force/bearing logs per run.

---

## Hardware Status

| Item | Status |
|---|---|
| Hip motors (AK45-10) | Designed, not ordered |
| Wheel motors (5065 130KV) | Designed, not ordered — **no specific part selected yet** |
| ODESC 3.6 | Designed, not ordered |
| MCU (UNO R4 WiFi) | Designed, not ordered |
| IMU (BNO086) | Designed, not ordered |
| 24V battery | **TBD — no part selected** |
| Wheel (PLA hub + TPU tread) | **Not designed — no CAD** |
| Body / frame | Not designed |
| Leg links | Sized (Al tube, SF≥2), not designed in CAD |

---

## Suggested Next Steps (Priority Order)

### 1. Pick a specific 5065 130KV motor part
The BOM says "generic 5065 130KV outrunner" — this needs to be a real part before ordering. Requirements: Hall sensors (ODESC needs them), D-shaft, ≥40A continuous, 24V rated. Popular options: Flipsky 5065, T-Motor AT5065.

### 2. Finalise battery
Nothing selected yet. Requirements: 24V nominal, ≥4Ah for reasonable runtime, XT60 output, ≤750g. A 6S LiPo or 24V LiFePO4 pack both work.

### 3. Replace PD balance controller with LQR in simulation
The current PD controller is a placeholder. The planned algorithm is LQR on the linearised inverted pendulum:
- State: `[pitch, pitch_rate, wheel_pos, wheel_vel, leg_length, leg_vel]`
- Linearise the model around upright stance at neutral leg length
- Compute LQR gain matrix K offline (Python `scipy.linalg.solve_continuous_are`)
- Replace the PD block in `viewer.py` with `u = -K @ state`
- Tune Q/R matrices in simulation

This is the highest-value sim task — everything downstream (SIL, firmware) builds on it.

### 4. Design the wheel in CAD
PLA hub + TPU tread, 150mm OD. Needs to:
- Mount to 5065 D-shaft (confirm shaft diameter when part is selected — typically 8mm)
- TPU tread pressed/glued over PLA hub OD
- Target 70g total (45g hub + 25g tread)

### 5. Start firmware scaffold
Once LQR gains are known from simulation, scaffold the balance loop in PlatformIO:
- 500Hz timer interrupt on RA4M1
- BNO086 → pitch + pitch_rate via I2C
- LQR control law → wheel torque command
- ODESC torque command via CAN

---

## Key Files to Read First

```
CLAUDE.md                                          ← Full design context
components/COMPONENTS.md                           ← Full BOM with subtotals
components/database/bom.yaml                       ← Machine-readable BOM
simulation/mujoco/baseline1_leg_analysis/sim_config.py  ← All geometry + params
```
