# 4-Bar Geometry Optimizer for Jump Height

## Why This Is Feasible

The infrastructure already supports this cleanly:
- `physics.py::build_xml()` generates MuJoCo XML dynamically from `RobotGeometry` — changing link lengths just means passing different params, no XML editing
- ES optimizer + multiprocessing already exists in `optimizer/optimize_integrated.py`
- `physics.py::solve_ik()` can sweep hip angles to find valid 4-bar closure for any geometry

## Key Challenge: Hardcoded Stroke Angles

`Q_RET = -0.733038 rad` and `Q_EXT = -1.431694 rad` are empirical constants (verified against run_id 51167). Any new geometry has a different valid stroke range — these must be auto-computed per evaluation.

**Solution:** Add `auto_stroke_angles(robot: RobotGeometry) → (q_ret, q_ext)` to `physics.py` that sweeps hip angles through `solve_ik()` and finds the feasible extremes.

## Search Space

| Variables | Dimensions | Notes |
|-----------|-----------|-------|
| `L_femur`, `L_tibia`, `Lc`, `L_stub` | 4D | geometry |
| + `max_torque`, `crouch_time` | 6D total | recommended |
| + 4 LQR weights | 10D | not needed |

**Recommended: 6D.** LQR re-tuning is not required because:
1. Jump phase uses bang-bang torque, not LQR
2. Fitness penalizes falling — bad geometry just scores poorly and ES steers away
3. They already run 12D in `optimize_integrated.py`

### Bounds

| Param | Baseline | Min | Max |
|-------|----------|-----|-----|
| `L_femur` | 173.8 mm | 140 mm | 220 mm |
| `L_tibia` | 129.4 mm | 100 mm | 170 mm |
| `Lc` (coupler) | 150.8 mm | 120 mm | 190 mm |
| `L_stub` | 35.1 mm | 25 mm | 50 mm |
| `max_torque` | 7.0 N·m | 4.0 N·m | 7.0 N·m |
| `crouch_time` | 0.20 s | 0.10 s | 0.35 s |

## Jump Height Metric

`wheel_liftoff_s` (current airtime proxy) is **wrong for geometry comparison** — longer legs have higher baseline Z so airtime is biased. Must use **peak body CoM Z height** above ground (`peak_body_z_m`) instead.

## Implementation Steps

### Step 1 — `auto_stroke_angles()` in `physics.py`
Sweeps hip angles, calls `solve_ik()`, finds feasible closure range, returns `(q_ret, q_ext)`. Returns `None` if geometry is invalid (infeasible 4-bar).

### Step 2 — `peak_body_z_m` metric in `sim_loop.py`
Track peak body CoM Z during flight phase. Add to metrics dict returned by `run()`.

### Step 3 — `optimizer/optimize_geometry.py` (new file)
Per evaluation:
1. Build candidate `RobotGeometry` from 6-vector
2. Call `auto_stroke_angles()` — discard if infeasible
3. Verify wheel reaches ground at Q_NOM (W_z ≈ 0)
4. Run `sim_loop.run()` with s10_jump scenario
5. Fitness: `200*fell + (1 - peak_body_z_m/0.3) + 0.5*(settle_time/6.0)`

Use existing `ESOptimizer` + multiprocessing from `optimize_integrated.py`. Auto-baseline best geometry to `params.py` on completion.

## Constraints to Enforce
- 4-bar closure must be valid over full stroke
- Wheel must reach ground from nominal standing height (W_z ≈ 0 at Q_NOM)
- Max spring torque < 5.5 N·m (from `optimize_spring.py`)
- Total leg length within body clearance from ground

## Critical Files

| File | Change |
|------|--------|
| [physics.py](simulation/mujoco/master_sim_jump/physics.py) | Add `auto_stroke_angles()` |
| [sim_loop.py](simulation/mujoco/master_sim_jump/sim_loop.py) | Add `peak_body_z_m` metric |
| [optimizer/optimize_geometry.py](simulation/mujoco/master_sim_jump/optimizer/optimize_geometry.py) | New optimizer script |
| [params.py](simulation/mujoco/master_sim_jump/params.py) | Auto-baselined with best geometry on completion |

## Verification
1. Single eval with baseline geometry → peak_body_z_m should match current jump height
2. Verify `auto_stroke_angles()` reproduces Q_RET/Q_EXT for baseline geometry
3. Run ~100 iters → confirm geometry varies meaningfully and fitness improves
4. Replay best candidate in visualizer
