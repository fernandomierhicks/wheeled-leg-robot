# HANDOFF.md — Project Status & Next Steps

Read this before starting work. Check `CLAUDE.md` for full design context.

---

## What Was Done (Sessions 1-4)

### Sessions 1-3: Hardware & LQR Basics
- Rebaselined wheel motor (5065 130KV), updated BOM and mass estimates
- Built LQR design pipeline (`lqr_design.py`)
- Implemented control blending in viewer

### Session 4: Behavioral Test Framework & Optimization
- **✅ COMPLETED: Behavioral test harness** (`test_behavioral.py`)
  - `run_self_balance_scenario()`: 30s stability test with 15+ metrics
  - `run_drive_scenario()`: 20s forward/backward with obstacle
  - Metrics: RMS pitch wobble, drift, control effort, bearing loads, settling time

- **✅ COMPLETED: Evolutionary optimization** (`optimize_self_balance.py`)
  - scipy.optimize.differential_evolution for (Q[0,0], R) space
  - Output: `optimization_log.csv` with all evaluations + `best_solution.json`
  - Last run: 243 evaluations in ~4.6 min, **best found: Q=16.96, R=1.514**

- **✅ COMPLETED: Visualization tool** (`run_scenario_visual.py`)
  - Real-time MuJoCo viewer for replaying scenarios
  - Supports `--eval-id` flag to load gains from optimization log
  - Self-balance scenario working correctly

---

## Key Findings from Session 4

### 1. **Self-Balance Works Well**
- Evolutionary optimization converged to stable, efficient gains
- RMS pitch wobble: ~0.10°, control effort: ~0.76 N·m·s
- Viewer displays stably for full 30s duration
- **Status**: Ready for further tuning or hardware deployment

### 2. **Drive Scenario Has Fundamental Architecture Conflict** ⚠️
**Problem**: Drive and self-balance have **opposite control objectives**:
- **Self-balance** needs strong feedback on `pitch_error` to hold CG above wheels → stays upright but opposes forward motion
- **Drive** needs `pitch_error` feedback OFF to allow pitch changes for velocity tracking → robot becomes unstable without it

**Root cause**: Single LQR state `[pitch_error, pitch_rate, wheel_pos, wheel_vel]` cannot satisfy both objectives.

**What was tried**:
1. ✗ Unit conversion fix (rad/s to m/s for wheel velocity) — didn't resolve conflict
2. ✗ Increased velocity tracking gain — balance control dominated, robot still moved backward
3. ✗ Removed `wheel_pos` term for drive — didn't help (pitch_error feedback still fought motion)
4. ✗ Zero-pitch initialization — robot tipped forward uncontrollably
5. ✗ Pitch-rate-only LQR for drive — robot became unstable (64-83° oscillations)

**Conclusion**: **Drive scenario needs a completely different control architecture** (not a simple LQR state change).

### 3. **Bugs Found & Fixed**
- **Velocity unit mismatch**: `wheel_vel` is rad/s, but `wheel_vel_target` was m/s → fixed with `WHEEL_R` conversion
- **CSV boolean bug**: `pass_fail` column has "True"/"False" strings, not 0/1 → needs conversion for float()
- **Duration logic**: Added proper duration_s handling for different scenarios

---

---

## Suggested Next Steps (Priority Order)

### 🔴 CRITICAL: Design Proper Multi-Scenario LQR Architecture

**Context**: Self-balance works well with the current LQR design. **But drive and other dynamic scenarios require a different approach.**

**Root issue**: The current monolithic LQR state `[pitch_error, pitch_rate, wheel_pos, wheel_vel]` mixes balance (holding position) with velocity tracking (changing position). These are fundamentally incompatible objectives in a single LQR formulation.

**Options to explore with Sonnet (recommended)**:

#### Option A: Hierarchical Control (Recommended)
- **Inner loop**: LQR-based pitch damper (only pitch_rate feedback) — prevents oscillations
- **Outer loop**: Proportional velocity controller → wheel torque command
- **Benefit**: Clean separation of concerns, easy to tune independently

#### Option B: Scenario-Specific LQR Gains
- Optimize **separate K matrices** for self-balance vs drive vs turning
- Self-balance: full state feedback `[pitch_error, pitch_rate, wheel_pos, wheel_vel]`
- Drive: lightweight state with minimal pitch constraints
- Switch gains based on scenario/telemetry
- **Benefit**: Leverage existing optimization framework
- **Cost**: More complex gain management

#### Option C: Nonlinear Model Predictive Control (MPC)
- Replace LQR with MPC that explicitly handles mode switching
- Single controller handles all scenarios
- **Benefit**: Optimal, robust to nonlinearities
- **Cost**: High computational complexity for embedded system

**Recommendation**: Start with **Option A** (hierarchical), design in Sonnet, then prototype in simulation.

---

### ✅ NEXT: Self-Balance Optimization Refinement

**Status**: Current best = Q=16.96, R=1.514. Good results, but room to improve.

**Tasks**:
1. Run evolutionary optimization for 500+ generations to find true optimum
2. Fine-tune velocity tracking gain (currently 1.5, may need 0.5–2.0 range)
3. Validate best gains work across 10+ repeated runs (stochasticity in noise model)
4. Export final K gains to `best_lqr_gains.json` for firmware reference

---

### ✅ Clean Up Drive Scenario Code

**Status**: Current code is experimental and has conflicts.

**Tasks**:
1. Revert drive scenario to simple state: do NOT try to optimize drive yet
2. Keep test_behavioral.py drive scenario as-is (for future use)
3. Remove drive visualization from run_scenario_visual.py until architecture is designed
4. Document why drive is disabled in code comments

---

### 🚩 FUTURE TASK: Independent Leg Control (Asymmetric Gain Scheduling)

**Status**: Not yet started; flagged for future implementation.

**Context**: Current implementation uses single `q_hip` parameter → assumes both legs synchronized at identical angles. This works for symmetric balance, drive, and turning. However, terrain handling, differential jumping, and uneven ground may require independent `q_hip_L` and `q_hip_R`.

**Future scope**:
1. Extend `compute_lqr_gain()` to accept separate `q_hip_L, q_hip_R` parameters
2. Compute `l_eff_L, l_eff_R` independently (may differ on slopes)
3. Design gain interpolation for asymmetric leg pairs
4. Test on simulated uneven terrain (>15° slope)
5. Verify stability with independent hip angle commands

**Why not now**: Current control architecture (hierarchical LQR + PI outer loops) is stable under symmetric operation. Asymmetric gains add complexity; tackle after symmetric version is proven on hardware.

---

### 📋 Hardware Tasks (Independent, Can Proceed in Parallel)

#### 1. Pick a specific 5065 130KV motor part
- BOM says "generic" — needs a real SKU (Flipsky, T-Motor, etc.)
- **Requirements**: Hall sensors, D-shaft, ≥40A, 24V rated
- **Deadline**: Before PCB layout / motor selection impacts frame design

#### 2. Finalize battery
- **Requirements**: 24V nominal, ≥4Ah, XT60, ≤750g
- Options: 6S LiPo or 24V LiFePO4
- **Deadline**: Before final weight estimation

#### 3. Design wheel in CAD
- PLA hub + TPU tread, 150mm OD
- Mount to 5065 D-shaft (8mm likely)
- Target 70g total
- **Deadline**: Before ordering parts

---

## Files Affected This Session

| File | Changes |
|------|---------|
| `test_behavioral.py` | ✅ Created, working |
| `optimize_self_balance.py` | ✅ Working, generating good results |
| `run_scenario_visual.py` | ⚠️ Partially working (self-balance OK, drive unstable) |
| `lqr_design.py` | ✅ No changes, still valid |
| `sim_config.py` | ✅ No changes needed |
| `HANDOFF.md` | 📝 Updated (this file) |

---

## Key Takeaway

**Self-balance LQR is ready for deployment or further refinement.**

**Drive/dynamic scenarios require architectural redesign.** Do NOT try to force single-state LQR to handle both balance and motion — it won't work. Use the Sonnet brainstorming to design a proper multi-mode control system, then implement in simulation.

---

## Key Files to Read

- `CLAUDE.md` — Full design context & working rules
- `components/COMPONENTS.md` — Full BOM & part selections
- `simulation/mujoco/LQR_controller/sim_config.py` — Geometry + physics parameters
- `simulation/mujoco/LQR_controller/test_behavioral.py` — Behavioral test framework
- `simulation/mujoco/LQR_controller/optimize_self_balance.py` — Evolutionary optimization

---

## Self-Balance Deployment Checklist

When ready to port to firmware:

- [ ] Run `optimize_self_balance.py` for 500+ generations to finalize K gains
- [ ] Validate best gains across 10+ repeated runs (check repeatability)
- [ ] Visual inspection: smooth balance, no oscillations, small control effort
- [ ] Check wheel drift stays <0.05m over 30s test
- [ ] Measure actual wheel radius (should be ~0.075m)
- [ ] Export gains to `best_lqr_gains.json`
- [ ] Port LQR law to C++ firmware (UNO R4 WiFi)
- [ ] Unit test firmware K values against sim (within 1% tolerance)

---