# HANDOFF.md — Project Status & Next Steps

Read this before starting work. Check `CLAUDE.md` for full design context.

---

## What Was Done (Sessions 1-3)

### Session 1: Hardware Rebaselining
- **Rebaselined wheel motor**: Maytech MTO7052HBM 60KV (380g, $45) → generic 5065 130KV outrunner (200g, $30). Saves 360g and $30 total.
- **Updated wheel assembly mass**: Added explicit `WHEEL_TYRE` BOM entry (PLA hub ~45g + TPU tread ~25g = 70g each).
- **Propagated changes** to `bom.yaml`, `sim_config.py`, `motor_models.py`, `COMPONENTS.md`, and `CLAUDE.md`.
- **Robot total mass is now ~3.0 kg** (2738g + 10% contingency), down from ~3.3 kg.

### Sessions 2-3: LQR Controller Implementation
- **Built LQR design pipeline** (`lqr_design.py`): Parameter verification → inertia computation → linearized dynamics → CARE solver → stability verification
- **Added to sim_config.py**: LQR gains `K = [-108.99, -19.23, -3.16, -2.32]` with `Q = diag([100, 1, 1, 0.1])`, `R = 0.1`
- **Integrated wheel position tracking** in viewer: Added `wheel_pos_L`, `wheel_pos_R` state variables; integrated from velocities
- **Implemented control blending**: Interactive slider (0.0 = PD, 1.0 = LQR) in matplotlib window
- **Parameter sweep (lqr_parameter_sweep.py)**: Tested 25 configurations (Q[0,0]: 50–200, R: 0.05–1.0)
  - **Key finding**: R=1.0 strongly preferred (40% lower control effort vs R=0.1)
  - **Recommended gains**: Q[0,0]=50, R=1.0 → `K = [-40.31, -7.16, -1.00, -0.76]` (50% less control effort)
- **Robot status at 100% LQR**: ✓ Balances stably, ✓ Responds to drive commands, ✓ Jumps independently, ✓ Telemetry visible

### Current Simulation State
- **viewer.py**: Full two-leg MuJoCo sim with LQR balance + jump, matplotlib telemetry, blend slider
- **lqr_design.py**: Offline CARE solver (compute K from physics)
- **lqr_parameter_sweep.py**: Automated 5×5 grid search (Q[0,0] vs R)
- **Arena**: 5 ramps distributed across space, ready for obstacle navigation testing

---

## Hardware Status

| Item | Status |
|---|---|
| Hip motors (AK45-10) | Designed, not ordered |
| Wheel motors (5065 130KV) | Designed, **no specific part selected yet** |
| ODESC 3.6 | Designed, not ordered |
| MCU (UNO R4 WiFi) | Designed, not ordered |
| IMU (BNO086) | Designed, not ordered |
| 24V battery | **TBD — no part selected** |
| Wheel (PLA hub + TPU tread) | **Not designed — no CAD** |
| Body / frame | Not designed |

---

## Suggested Next Steps (Priority Order)

### 1. Systematic LQR Tuning with Behavioral Testing (New — 3-Phase Plan)

**Goal**: Validate LQR gains across realistic robot behaviors (not just settling time), then replay/analyze results.

#### Phase 1: Behavioral Test Harness (`test_behavioral.py`)

**Create new file**: `simulation/mujoco/LQR_controller/test_behavioral.py`

Run 5 non-interactive scenarios, logging metrics per scenario:

| Scenario | Duration | Objective | Key Metrics |
|----------|----------|-----------|------------|
| **1. Self-balance** | 30s | Minimal drift, stable upright | RMS pitch <2°, drift <0.1m, no oscillations |
| **2. Drive responsiveness** | 20s | Fwd/Bwd acceleration then re-balance | Wheel tracking error <0.05m, settle <3s |
| **3. Turning stability** | 25s | In-place rotation & forward+turn | Bearing loads <8× GRF, pitch <15° |
| **4. Obstacle navigation** | 20s | Traverse ramps without falling | Jump height <50mm, landing stable |
| **5. Jump recovery** | 15s | Land then immediately re-stabilize | Landing g-shock <5g, settle <2s |

**Output**: Each scenario logs ~15 metrics to dict:
- Pitch control: `rms_pitch_deg`, `max_pitch_deviation_deg`, `pitch_settle_time_s`, `pitch_oscillations_count`
- Wheel control: `max_wheel_pos_drift_m`, `rms_wheel_tracking_error_m`, `wheel_velocity_rms_m_s`
- Control effort: `control_effort_integral`, `peak_torque_nm`, `avg_torque_nm`
- Stability: `max_bearing_load_n`, `max_femur_lateral_n`, `max_impact_g`
- Result: `pass_fail`, `notes`

**Usage**:
```bash
python test_behavioral.py --q 100 --r 0.1 --blend 1.0
```

---

#### Phase 2: Extended Parameter Sweep (`lqr_parameter_sweep.py` — Updated)

**Modify existing**: Add `--behavioral` flag to run all 5 scenarios per Q/R configuration.

**Workflow**:
1. For each Q/R pair, call Phase 1 test harness internally (or reuse functions)
2. Compute **weighted fitness score** across all 5 scenarios:
   ```
   fitness = 0.30×(self_balance) + 0.20×(drive) + 0.15×(turn) + 0.20×(obstacles) + 0.15×(jump)
   ```
3. Rank configurations by fitness instead of settling_time alone
4. Output new CSV: `behavioral_test_results.csv` with columns:
   - `Q_pitch, R, fitness, scenario_1_score, scenario_2_score, ... scenario_5_score, notes`

**Usage**:
```bash
# Full sweep with behavioral testing (~1–2 hours)
python lqr_parameter_sweep.py --behavioral
```

**Expected result**: Top 3–5 gain sets ranked by composite fitness, ready for Phase 3 analysis.

---

#### Phase 3: Replay & Analysis Tools (New)

**Three new CLI tools** for deterministic replay, visualization, and unified logging.

##### 3a. Unified `experiment_log.csv`

**Schema**: Single source of truth for all tests + gains + metrics

```
test_id, timestamp, scenario, Q_pitch, R, k0, k1, k2, k3,
settle_time, overshoot, control_effort, peak_control, damp_ratio,
notes, status
```

Each row = one complete test (LQR gains + all metrics). Auto-populated by Phase 2 sweep.

##### 3b. `replay_experiment.py` — Deterministic Test Re-runner

**Purpose**: Load a test row from `experiment_log.csv`, re-run exact scenario with logged gains, optionally save video/plots.

**CLI**:
```bash
python replay_experiment.py 42              # Replay test ID 42
python replay_experiment.py --best          # Replay best config (highest fitness)
python replay_experiment.py --top 5         # Batch replay top 5
python replay_experiment.py 42 --video out.mp4  # Save motion video
python replay_experiment.py --list          # List all tests with rankings
```

**Output**:
- Metrics table (logged vs. replay, should match ±0.1%)
- Optional MP4 video with motion overlay
- Useful for: validating sweep results, choosing gains for hardware deployment

##### 3c. `visualize_gains.py` — Interactive Gain Viewer

**Purpose**: Non-GUI script to pick a gain set, watch sim behavior, export metrics.

**CLI**:
```bash
python visualize_gains.py                    # Show top configs, pick one
python visualize_gains.py --test 42          # Print metrics for test 42
python visualize_gains.py --compare 10 25    # Side-by-side metrics table
python visualize_gains.py --best --output metrics.json  # Export to JSON
python visualize_gains.py --sweep-q 0.1 50:200:25  # Sweep Q[0,0] for fixed R
```

**Output**:
- Formatted metrics table (terminal)
- JSON export for Excel/matplotlib
- Fast execution (~30 sec per test, no video)

---

### 1b. File Structure (After Phase 1-3)

```
simulation/mujoco/LQR_controller/
├── experiment_log.csv               ← Unified log (all tests + gains)
├── test_behavioral.py               ← Phase 1: 5-scenario test harness
├── replay_experiment.py             ← Phase 3: Deterministic replay CLI
├── visualize_gains.py               ← Phase 3: Interactive gain viewer
├── lqr_parameter_sweep.py           ← Phase 2: Extended sweep (behavioral)
├── [existing files...]
├── sim_config.py
├── lqr_design.py
├── physics.py
├── motor_models.py
└── viewer.py
```

---

### 1c. Workflow Examples

**Example A: Full Tuning Pipeline**
```bash
# Phase 2: Grid search with behavioral metrics (~1–2 hours)
python lqr_parameter_sweep.py --behavioral
# → experiment_log.csv created with 25 tests

# Phase 3a: Show top 5 configurations
python visualize_gains.py --list

# Phase 3b: Replay best config with video
python replay_experiment.py --best --video best_config.mp4

# Phase 3c: Compare top 2
python visualize_gains.py --compare 1 2
```

**Example B: Iterative Refinement**
```bash
# Initial sweep finds R=1.0 is better
python lqr_parameter_sweep.py --behavioral

# Decision: R=1.0 is preferred, now refine Q[0,0] and other Q components
# Edit test_behavioral.py or lqr_parameter_sweep.py to vary Q[1], Q[2], Q[3]
python lqr_parameter_sweep.py --behavioral --narrow-q 30:100:10

# Analyze new results
python visualize_gains.py --best
python replay_experiment.py --best --video refined_best.mp4
```

---

### 2. Pick a specific 5065 130KV motor part

The BOM says "generic 5065 130KV outrunner" — this needs to be a real part before ordering.

**Requirements**: Hall sensors (ODESC needs them), D-shaft, ≥40A continuous, 24V rated.

**Popular options**: Flipsky 5065, T-Motor AT5065.

---

### 3. Finalise battery

Nothing selected yet.

**Requirements**: 24V nominal, ≥4Ah for reasonable runtime, XT60 output, ≤750g. A 6S LiPo or 24V LiFePO4 pack both work.

---

### 4. Design the wheel in CAD

PLA hub + TPU tread, 150mm OD.

**Requirements**:
- Mount to 5065 D-shaft (confirm shaft diameter when part is selected — typically 8mm)
- TPU tread pressed/glued over PLA hub OD
- Target 70g total (45g hub + 25g tread)

---

### 5. Start firmware scaffold

Once LQR gains are finalized in simulation (via Phase 1-3 above):

1. **Port control law to C++** for UNO R4 WiFi
2. **Integrate BNO086 IMU** (Game Rotation Vector @ 500 Hz → pitch + pitch_rate)
3. **CAN interface** to ODESC for wheel torque commands
4. **HIL testing**: USB serial loopback to verify timing at 500 Hz

The LQR gains are model-based and should transfer directly to hardware if:
- Wheel radius = 0.075 m (check 3D-printed tire diameter)
- Motor constants match sim_config (5065 130KV back-EMF, ODESC 50A max)
- Inertia estimates are ±20% (minor gain adjustment may be needed)

---

## Key Files to Read First

```
CLAUDE.md                                       ← Full design context
components/COMPONENTS.md                        ← Full BOM with subtotals
components/database/bom.yaml                    ← Machine-readable BOM
simulation/mujoco/LQR_controller/sim_config.py  ← All geometry + params
```

---

## Implementation Notes

### Phase 1-3 Estimation

| Phase | Task | Duration | Output |
|-------|------|----------|--------|
| **1** | Build 5-scenario test harness | 1 hour | `test_behavioral.py` + working tests |
| **2** | Extend parameter sweep | 30 min | `lqr_parameter_sweep.py --behavioral` flag + CSV output |
| **3** | Build replay + viz tools | 1–2 hours | `replay_experiment.py` + `visualize_gains.py` |
| **Full execution** | Run complete sweep + analysis | 1–2 hours sim time | `experiment_log.csv` with 25 tests, top gains identified |

**Total implementation + first run**: ~3 hours coding + 1–2 hours sim execution

---

## Testing & Validation

After Phase 1-3 complete:

1. ✓ `test_behavioral.py` runs all 5 scenarios without errors
2. ✓ `lqr_parameter_sweep.py --behavioral` populates `experiment_log.csv`
3. ✓ `replay_experiment.py 1` replays first test, metrics match logged values ±0.1%
4. ✓ `visualize_gains.py --best` shows top-ranked config
5. ✓ `replay_experiment.py --best --video` executes and outputs MP4
6. ✓ All tools have `--help` and exit cleanly on errors

---

## Hardware Deployment Checklist

Before uploading K gains to firmware:

- [ ] Phase 3 replay confirms test matches logged metrics
- [ ] All 5 behavioral scenarios pass with chosen gains
- [ ] Video review shows smooth, responsive balance
- [ ] No overshoot or oscillation visible in telemetry
- [ ] Wheel position drift minimal over 30s test
- [ ] Confirm actual wheel radius = 0.075 m (measure 3D-printed tire)
- [ ] Document gain values + Q/R/K parameters for commit message
- [ ] Unit test firmware LQR law against sim K values

---

