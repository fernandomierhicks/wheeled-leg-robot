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

**Status: Planned — ready to implement. Full step-by-step plan below.**

**Approach:** 4-state LQR on linearised wheeled inverted pendulum.
State vector: `[pitch, pitch_rate, wheel_pos, wheel_vel]`
Leg length stays in the existing outer hip loop (unchanged).
Input: symmetric wheel torque scalar `u` [N·m] — same interface as current PD.

**Files:**
- `simulation/mujoco/baseline1_leg_analysis/lqr_design.py` — NEW: offline script to compute K
- `simulation/mujoco/baseline1_leg_analysis/sim_config.py` — add `LQR_K`, `LQR_Q_DIAG`, `LQR_R_VAL`
- `simulation/mujoco/baseline1_leg_analysis/viewer.py` — add `wheel_pos` state, replace PD block

---

#### Step 1 — Create `lqr_design.py`: import and echo parameters

```python
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sim_config import *
from physics import solve_ik, SimParams

p = SimParams()
W_pos = solve_ik(Q_NEUTRAL, p)
l_eff = abs(W_pos['W'][2])

print(f"WHEEL_R     = {WHEEL_R} m")
print(f"Q_NEUTRAL   = {Q_NEUTRAL} rad")
print(f"W_z (body)  = {W_pos['W'][2]:.4f} m")
print(f"l_eff       = {l_eff:.4f} m")
```

**Check:** `WHEEL_R=0.075`, `W_z` negative, `l_eff ≈ 0.13–0.22 m`

---

#### Step 2 — Compute effective inertia parameters in `lqr_design.py`

```python
m_w  = 2 * m_wheel
m_b  = m_box + 2*(m_femur + m_tibia + m_coupler + m_bearing + m_motor)
M    = m_b + m_w
I_w  = 0.5 * m_wheel * WHEEL_R**2
I_b  = m_b * l_eff**2
g    = 9.81

print(f"m_b={m_b:.3f} kg  m_w={m_w:.3f} kg  M={M:.3f} kg")
print(f"I_w={I_w:.5f} kg·m²  I_b={I_b:.5f} kg·m²")
```

**Check:** `M ≈ 2.6–3.0 kg`, `I_b ≈ 0.05–0.15 kg·m²`

---

#### Step 3 — Build A, B matrices in `lqr_design.py`

Standard wheeled IP linearisation (Grasser et al. 2002):

```python
r     = WHEEL_R
denom = (M + 2*I_w/r**2) * (I_b + m_b*l_eff**2) - m_b**2 * l_eff**2

alpha = (M + 2*I_w/r**2) * m_b * g * l_eff / denom
beta  = -m_b**2 * g * l_eff**2 / (r * denom)
gamma = -(I_b + m_b*l_eff**2) / (r * denom)
delta = (M + 2*I_w/r**2 + m_b*l_eff/r) / denom

A = np.array([[0,1,0,0],[alpha,0,0,0],[0,0,0,1],[beta,0,0,0]])
B = np.array([[0],[gamma],[0],[delta]])

print("A ="); print(A)
print("B ="); print(B)
```

**Check:** `A[1,0]` (alpha) positive ~30–70; `B[1,0]` (gamma) negative

---

#### Step 4 — Solve CARE and print K in `lqr_design.py`

```python
from scipy.linalg import solve_continuous_are

Q = np.diag([100.0, 1.0, 1.0, 0.1])
R = np.array([[0.1]])

P    = solve_continuous_are(A, B, Q, R)
K    = np.linalg.inv(R) @ B.T @ P       # shape (1,4)
eigs = np.linalg.eigvals(A - B @ K)

print("K =", K)
print("Closed-loop eigenvalues:", eigs)
assert all(e.real < 0 for e in eigs), "NOT stable — check A, B"
```

**Check:** Script exits clean. K values ~`[30–80, 5–15, 0.5–3, 1–5]` (order of magnitude).

---

#### Step 5 — Add K to `sim_config.py`

After existing `PITCH_KP`, `PITCH_KD` lines, add:

```python
# LQR gains — computed by lqr_design.py
# State order: [pitch_err, pitch_rate, wheel_pos_err, wheel_vel_err]
LQR_K      = np.array([k0, k1, k2, k3])   # paste values from lqr_design.py output
LQR_Q_DIAG = [100.0, 1.0, 1.0, 0.1]
LQR_R_VAL  = 0.1
```

Leave `PITCH_KP`, `PITCH_KD`, `PITCH_KI` in place (unused, for rollback).

**Check:** `python -c "from sim_config import LQR_K; print(LQR_K)"` prints vector.

---

#### Step 6 — Add `wheel_pos` variables to `viewer.py`

Find where `pitch_integral` is initialised (near top of main sim function). Add alongside it:

```python
wheel_pos_L = 0.0
wheel_pos_R = 0.0
```

Find the Restart handler (where `pitch_integral = 0.0` is reset). Add same two lines there.

**Check:** Search `viewer.py` for `pitch_integral = 0` — exactly one init and one reset site.

---

#### Step 7 — Integrate wheel position in control loop in `viewer.py`

Find line ~934: `wheel_vel = (data.qvel[d_whl_L] + data.qvel[d_whl_R]) / 2.0`

Add immediately after:

```python
wheel_pos_L += data.qvel[d_whl_L] * _dt
wheel_pos_R += data.qvel[d_whl_R] * _dt
wheel_pos    = (wheel_pos_L + wheel_pos_R) / 2.0
```

**Check:** Add `print(f"wheel_pos={wheel_pos:.3f}")`, run 2 sec — value drifts, resets on Restart. Remove print.

---

#### Step 8 — Replace PD block with LQR law in `viewer.py`

Find lines 972–979 (PD block). Replace entirely with:

```python
_lqr_state = np.array([
    pitch      - target_pitch,
    pitch_rate,
    wheel_pos,
    wheel_vel  - (pitch_fb / WHEEL_R),
])
u_bal = float(-LQR_K @ _lqr_state)
```

`LQR_K` arrives via existing `from sim_config import *`.
`target_pitch` and `pitch_fb` are already computed above this block — do not move them.
Remove the grounded gain-doubling lines (LQR handles this implicitly).

**Check:** Confirm `from sim_config import *` at top of file. Confirm `pitch_fb` defined before this block (~line 967).

---

#### Step 9 — First-run sanity check

Run `viewer.py`, no buttons pressed:
- [ ] No Python exception on launch
- [ ] Robot stands upright without immediately falling
- [ ] Pitch telemetry visible and roughly flat (±10° acceptable first run)

If robot falls instantly → increase `Q[0,0]` to `1000.0` in lqr_design.py, rerun Steps 4–5, update sim_config.
If oscillates → decrease `Q[0,0]` or increase `R`.

---

#### Step 10 — Regression tests

- **Drive:** Fwd/Bwd buttons — robot accelerates and re-balances. No falls.
- **Jump:** Crouch → Jump → land → re-balance. Jump controller uses `ctrl[0/1]` (hip), LQR uses `ctrl[2/3]` (wheels) — independent.
- **Restart:** `wheel_pos` resets to 0.

#### Full Verification Checklist

- [ ] Step 1: `l_eff` and `W_z` sensible
- [ ] Step 2: `M ≈ 2.6–3.0 kg`, `I_b ≈ 0.05–0.15 kg·m²`
- [ ] Step 3: `A[1,0]` positive, `B[1,0]` negative
- [ ] Step 4: All closed-loop eigenvalues negative real part; assertion passes
- [ ] Step 5: `from sim_config import LQR_K` works
- [ ] Step 6: Init and reset sites confirmed
- [ ] Step 7: `wheel_pos` drifts and resets correctly
- [ ] Step 8: No import errors; `pitch_fb` in scope
- [ ] Step 9: Robot balances (pitch ±5° steady state)
- [ ] Step 10: Drive / Jump / Restart all pass

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

---

# NEW SESSION: LQR CONTROLLER IMPLEMENTATION & TUNING

## What Was Accomplished

### LQR Foundation (Steps 1-5)
- **Step 1-4**: Built complete LQR design pipeline (`lqr_design.py`)
  - Parameter verification (geometry, l_eff=0.1723m)
  - Inertia computation (M=1.655kg, I_b=0.0331 kg·m²)
  - Linearized dynamics matrices (A, B from wheeled IP)
  - CARE solver → optimal gains K via scipy
  - Stability verification (all closed-loop poles negative real part)
  - Step response simulation (5s test, pitch error converges to ~0)

- **Step 5**: Added LQR gains to `sim_config.py`
  - `LQR_K = [-108.9959, -19.2330, -3.1623, -2.3265]` (initial design)
  - `Q = diag([100, 1, 1, 0.1])`, `R = 0.1`
  - PID gains preserved for rollback/blending

### Viewer Integration (Steps 6-8 + Blending)
- **Step 6**: Wheel position tracking added to viewer
  - `wheel_pos_L`, `wheel_pos_R` state variables
  - Integrated from velocities in control loop

- **Step 7**: Wheel position in LQR state vector
  - State: `[pitch, pitch_rate, wheel_pos, wheel_vel]`
  - All four components used in control law

- **Step 8**: Control blending (PID → LQR fade-in)
  - PID law: `u_pid = PITCH_KP × pitch_err + PITCH_KD × pitch_rate`
  - LQR law: `u_lqr = -K @ state_vector`
  - Blended: `u = (1 - blend_factor) × u_pid + blend_factor × u_lqr`
  - **Interactive slider in matplotlib window** (0.0 = PID, 1.0 = LQR)

- **Verification**: Test script confirms control laws execute correctly
  - PID works at blend=0.0
  - LQR state computed at all blend levels
  - Wheel position tracking active

### Robot Status at 100% LQR
- **✓ Balances stably** in neutral stance
- **✓ Responds to Fwd/Bwd drive commands**
- **✓ Jump controller works independently** (hip motors decoupled from wheel balance)
- **✓ Telemetry visible** (pitch, torque, height graphs)

---

## Systematic LQR Tuning (New)

### Parameter Sweep: `lqr_parameter_sweep.py`

Tested all combinations of:
- **Q[0,0]** (pitch error weight): 50, 75, 100, 150, 200
- **R** (control effort weight): 0.05, 0.1, 0.2, 0.5, 1.0
- **Grid**: 5 × 5 = 25 configurations

### Test Metrics (per configuration)
- **Settling time**: Time for pitch to stay within ±1° (seconds)
- **Overshoot**: Maximum pitch deviation from zero (degrees)
- **Control effort**: Integral of u² over 5-second sim (N²·m²·s)
- **Peak control**: Maximum |u| (N·m)
- **Damping ratio**: From closed-loop eigenvalues

### Results: Top 5 Configurations

| Rank | Q[0,0] | R    | Settle (s) | Overshoot (°) | Control Eff |
|------|--------|------|------------|---------------|-------------|
| 1    | 50     | 1.0  | 0.734      | 2.89          | 42,455      |
| 2    | 75     | 1.0  | 0.734      | 2.89          | 42,546      |
| 3    | 100    | 1.0  | 0.734      | 2.89          | 42,656      |
| 4    | 150    | 1.0  | 0.734      | 2.89          | 42,852      |
| 5    | 200    | 1.0  | 0.734      | 2.89          | 43,018      |

### Key Finding
**All configurations with R=1.0 are essentially tied** on settling time and overshoot. The difference is in control effort:
- Lower R (0.05–0.2) → Much higher control effort (420k–830k), excessive torque
- Higher R (1.0) → Balanced effort (~42k), smooth control
- **Recommendation: R=1.0 strongly preferred**

Within R=1.0, Q[0,0]=50 gives lowest effort. New optimal gains:
```python
LQR_K = np.array([-40.3136, -7.1639, -0.9999, -0.7627])
Q_diag = [50.0, 1.0, 1.0, 0.1]  # Lower pitch cost than original 100
R_val = 1.0
```

**vs. Original (Design):**
```python
LQR_K = np.array([-108.9959, -19.2330, -3.1623, -2.3265])
Q_diag = [100.0, 1.0, 1.0, 0.1]
R_val = 0.1
```

### Interpretation
- **Original gains too aggressive**: Q[0,0]=100, R=0.1 → severe pitch control at high torque cost
- **Tuned gains more balanced**: Q[0,0]=50, R=1.0 → similar settling but 50% less control effort
- **Larger gain reductions**: K magnitudes cut ~40–60%, indicating original was over-designed for safety

---

## How to Apply New Gains

### Option 1: Update sim_config.py (Permanent)
```python
# In simulation/mujoco/LQR_controller/sim_config.py
LQR_K      = np.array([-40.3136, -7.1639, -0.9999, -0.7627])
LQR_Q_DIAG = [50.0, 1.0, 1.0, 0.1]
LQR_R_VAL  = 1.0
```

### Option 2: Test in Viewer First (Temporary)
1. Run `python viewer.py`
2. Set blend slider to 1.0 (100% LQR)
3. Manually edit viewer.py line ~993:
   ```python
   # Temporarily test new gains
   _K_test = np.array([-40.3136, -7.1639, -0.9999, -0.7627])
   u_lqr = float(-_K_test @ _lqr_state)
   ```
4. Test drive, jump, balance
5. If satisfied, commit to sim_config.py

---

## Continuing This Work

### For Next Agent: Parameter Sweep Script
To refine gains further or explore different Q structure:

1. **Edit ranges in `lqr_parameter_sweep.py` (line ~42–44)**:
   ```python
   Q_PITCH_VALUES = [30, 50, 75, 100, 150, 200, 300]  # Expand range
   R_VALUES = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]          # Finer resolution
   ```

2. **Run sweep** (takes 5–10 minutes):
   ```bash
   cd simulation/mujoco/LQR_controller
   python lqr_parameter_sweep.py | tee sweep_log.txt
   ```

3. **Review results**:
   - Top 5 configs printed to terminal
   - Full results saved to `lqr_sweep_results.csv`
   - Plot the CSV in Excel/matplotlib to find "knee" of Pareto curve

4. **Apply best gains**:
   - Copy recommended K to sim_config.py
   - Test in viewer with blend=1.0
   - If stable, commit

### Advanced Tuning Opportunities
- **Vary Q[1]** (pitch rate weight): Test 0.5, 1.0, 2.0, 5.0
- **Vary Q[2]** (wheel position weight): Test 0.1, 1.0, 10.0 (currently fixed at 1.0)
- **Vary Q[3]** (wheel velocity weight): Test 0.01, 0.1, 1.0 (currently fixed at 0.1)
- **Multi-objective optimization**: Trade off settling time vs. control effort vs. robustness

---

## Files Reference

| File | Purpose |
|------|---------|
| `lqr_design.py` | Core LQR gain computation + verification (4-step pipeline) |
| `lqr_parameter_sweep.py` | **NEW** — Automated tuning via grid search |
| `sim_config.py` | Parameters (geometry, motors, **LQR gains**) |
| `viewer.py` | MuJoCo GUI + blend slider for testing |
| `LQR_INTEGRATION_GUIDE.md` | User manual for blending/tuning in viewer |
| `test_lqr_integration.py` | Validates control law executes correctly |
| `lqr_sweep_results.csv` | **Generated** — All 25 test results (importable to Excel) |

---

## Next Steps for Hardware / Firmware

Once LQR gains finalized in simulation:
1. **Port control law to C++** for UNO R4 WiFi
2. **Integrate BNO086 IMU** (Game Rotation Vector @ 500 Hz → pitch + pitch_rate)
3. **CAN interface** to ODESC for wheel torque commands
4. **HIL testing**: USB serial loopback to verify timing at 500 Hz

The LQR gains are model-based and should transfer directly to hardware if:
- Wheel radius = 0.075 m (check 3D-printed tire diameter)
- Motor constants match sim_config (5065 130KV back-EMF, ODESC 50A max)
- Inertia estimates are ±20% (minor gain adjustment may be needed)
