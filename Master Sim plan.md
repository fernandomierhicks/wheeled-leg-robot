# Master Sim/Optimizer Framework — Consolidation Plan

## Context

The robot project has 4 simulation folders with heavily duplicated code:
- `LQR_Control_optimization/` (Phases 1-5, 12+ files)
- `latency_sensitivity/` (Phase 6, near-complete copy with ring buffer delays)
- `4bar_optimization_with_balancing/` (geometry optimizer, standalone)
- `baseline1_leg_analysis/` (reference geometry, force analysis)

Key problems: `sim_config.py` duplicated 3x, `physics.py` 3x, `scenarios.py` 2x, `battery_model.py` 2x, 4 near-identical optimizer scripts duplicated across 2 folders. Changing a parameter in one place doesn't propagate. Replay and sandbox duplicate sim logic.

**Goal:** Single `simulation/mujoco/master_sim/` Python package — one source of truth for all parameters, physics, controllers, scenarios, optimization, and visualization.

---

## Design Decisions Resolved

| Question | Decision |
|---|---|
| Package path | `simulation/mujoco/master_sim/` |
| Charts | **pyqtgraph only** — no matplotlib anywhere |
| Replay vs sandbox | **Combined** into `viz/visualizer.py` — sandbox IS Scenario 0 |
| Hardware limits | **Every tick** — O(1) cost, negligible; early-exit saves optimizer time |
| Jump controller | **Defer** — add when ready, no stub now |
| Play/pause | **GUI button** in tkinter progress_ui |
| Early verification | **Phase 0** includes model load + 4-bar sweep with body fixed in air |
| Latency disable | Set `sensor_delay_steps=0, actuator_delay_steps=0` → pass-through buffers |
| Controller toggles | Visualizer has per-controller enable/disable in GUI |

---

## Target Folder Structure

```
simulation/mujoco/master_sim/           # Python package
    __init__.py
    params.py                           # Frozen dataclass hierarchy (ALL parameters)
    defaults.py                         # DEFAULT_PARAMS instance (Phase 6 values)
    physics.py                          # 4-bar IK, equilibrium pitch, MJCF builder
    sim_loop.py                         # THE single simulation core (_run_sim_loop)
    models/
        __init__.py
        battery.py                      # BatteryModel class
        motor.py                        # motor_taper, motor_currents, back-EMF
        latency.py                      # LatencyBuffer — n_steps=0 is pass-through
        thermal.py                      # 2-node motor thermal model
    controllers/
        __init__.py
        lqr.py                          # LQR gain solver + scheduling + lqr_torque()
        velocity_pi.py                  # VelocityPI class
        yaw_pi.py                       # YawPI class
        hip.py                          # Position servo + impedance + roll leveling
    scenarios/
        __init__.py                     # ScenarioRegistry, evaluate()
        base.py                         # ScenarioConfig dataclass, FitnessMetric
        profiles.py                     # Velocity staircase, leg cycle, disturbance fns
        s01_lqr_pitch_step.py           # Each scenario: config + fitness
        s02_leg_height_gain_sched.py
        s03_vel_pi_disturbance.py
        s04_vel_pi_staircase.py
        s05_vel_pi_leg_cycling.py
        s06_yaw_pi_turn.py
        s07_drive_turn.py
        s08_terrain_compliance.py
    optimizer/
        __init__.py
        search_space.py                 # SearchSpace class (bounds, log/linear)
        es_engine.py                    # Generic (1+lambda)-ES
        pipeline.py                     # S1->S8 chain with gain propagation
        progress_ui.py                  # tkinter window: play/pause GUI button
        run_log.py                      # CSV logging, run ID tracking
    viz/
        __init__.py
        visualizer.py                   # SINGLE file: MuJoCo viewer + pyqtgraph panels
                                        # --mode sandbox (S0) | --mode replay --scenario sXX
    logs/                               # All CSV results + pipeline logs
    README.md                           # Framework docs
```

Old folders archived to `simulation/mujoco/_archived/` with README noting frozen state.

---

## Key Design Decisions

### 1. Parameter Registry: Frozen dataclasses with `.replace()`
- `@dataclass(frozen=True)` hierarchy: `RobotGeometry`, `GainSet`, `BatteryParams`, `LatencyParams`, `MotorParams`, `NoiseParams`, `SimTiming`, `HardwareLimits`, and top-level `SimParams`
- Immutable by default, optimizer creates modified copies via `dataclasses.replace()`
- No file I/O during optimization — params are Python objects passed to subprocess workers
- Zero external dependencies (no Pydantic/YAML needed)

### 2. No More Module-Level Mutable Globals
- Currently: optimizer workers override `VELOCITY_PI_KP`, `LQR_K_TABLE` etc. as module globals
- New: `sim_loop.run(params, scenario)` takes explicit `SimParams` — each worker gets its own copy
- Eliminates the entire class of "forgot to set the global" bugs

### 3. Unified Sim Loop for All Modes
- `sim_loop.run(params, scenario, callbacks=None, command_queue=None) -> dict`
- **Headless optimizer:** `metrics = sim_loop.run(params, scenario)` — returns dict
- **Replay:** `sim_loop.run(params, scenario, callbacks=[telemetry_recorder])` + MuJoCo viewer
- **Sandbox:** `sim_loop.run(params, scenario, callbacks=[live_chart_push], command_queue=cmd_q)` — real-time control
- Latency model always present — `LatencyBuffer(n_steps=0)` = transparent pass-through

### 4. Hardware Limits Checked Every Tick
- `SimParams` includes a `HardwareLimits` sub-dataclass: `max_bearing_force`, `max_motor_current`, `max_link_stress`
- Checked every tick inside `_run_sim_loop` — 4–6 float comparisons, O(1), negligible overhead vs MuJoCo solver
- Violations logged with timestamps to metrics dict
- In optimizer mode: early-exit on hard violation (saves wall time for bad candidates)

### 5. Scenario as Data (Composition, Not Inheritance)
```python
@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    display_name: str
    duration: float
    active_controllers: frozenset[str]   # {"lqr"}, {"lqr","velocity_pi","yaw_pi"}
    hip_mode: str                        # "position" | "impedance"
    v_profile: Callable | None
    omega_profile: Callable | None
    hip_profile: Callable | None
    dist_fn: Callable | None
    world: WorldConfig
    fitness_fn: Callable[[dict], float]
    group: str                           # "lqr" | "velocity_pi" | "yaw_pi" | "suspension"
    order: float                         # 1.0, 2.0, 2.1 (supports sub-scenarios)
    init_fn: Callable | None
```

### 6. Unified Visualizer — Sandbox = Scenario 0
- `viz/visualizer.py` replaces both `replay.py` and `sandbox_fastchart.py`
- **Sandbox mode** (`--mode sandbox`, default): Scenario 0 — free-run, no fitness, interactive
  - 28 obstacles, gamepad/sliders, real-time pyqtgraph panels
  - **Controller enable/disable toggles** in GUI (LQR, VelocityPI, YawPI, Suspension independently)
- **Replay mode** (`--mode replay --scenario s01`): run named scenario, show telemetry
  - pyqtgraph 5×2 panel (pitch, wheel vel, hip angle, motor current, battery, yaw, roll, etc.)
  - MuJoCo passive viewer alongside
- No matplotlib anywhere — pyqtgraph only

### 7. Latency Model Always Present, Trivially Disabled
- `LatencyBuffer(n_steps=0)` = pass-through (push and oldest both return the current value)
- Disable: `dataclasses.replace(DEFAULT_PARAMS, latency=LatencyParams(sensor_delay_steps=0, actuator_delay_steps=0))`
- No conditional branches in sim_loop — buffer handles the logic

### 8. Generic Optimizer
- `SearchSpace` defines param names, bounds, log/linear flags
- `ESOptimizer` takes SearchSpace + `eval_fn(candidate_dict) -> metrics_dict`
- Each optimizer script ~40 lines: define space, define eval_fn, run
- play/pause via tkinter GUI button in `progress_ui.py`

### 9. Documentation Strategy
- `master_sim/README.md`: Framework usage
- `docs/Control.MD`: Algorithm theory — unchanged
- `components/COMPONENTS.MD`: BOM/hardware specs — unchanged

---

## Implementation Steps

Each step is small, independently verifiable, and builds on the previous.

### PHASE 0: Package Scaffolding + Early Visual Verification ✅ COMPLETE

**Step 0.1 — Create directory structure**
- Create all folders + empty `__init__.py` files
- Verify: `python -c "import master_sim"` succeeds

**Step 0.2 — Create `params.py`**
- All nested frozen dataclasses: `RobotGeometry`, `MotorParams`, `BatteryParams`, `GainSet`, `LatencyParams`, `NoiseParams`, `SimTiming`, `HardwareLimits`, top-level `SimParams`
- Verify: `python -c "from master_sim.params import SimParams; print(list(SimParams.__dataclass_fields__))"`

**Step 0.3 — Create `defaults.py` with DEFAULT_PARAMS**
- Instantiate `SimParams` with all Phase 6 values from `latency_sensitivity/sim_config.py`
- Verify: `python -c "from master_sim.defaults import DEFAULT_PARAMS as P; print(P.gains.lqr.Q_pitch)"`

**Step 0.4 — Early visualization: load MJCF model in MuJoCo viewer** ← VERIFY EARLY
- Port just enough of `physics.py` to call `build_xml(DEFAULT_PARAMS.robot, WorldConfig())`
- **Robot body welded to world frame** — body fixed in air, no balance needed
- Open in `mujoco.viewer.launch_passive()`, run 1 second
- **Visual gate:** user confirms 3D model loads, 4-bar geometry looks correct

**Step 0.5 — Verify 4-bar kinematics visually** ← VERIFY EARLY
- Body still fixed in air; sweep hip joint Q_RET → Q_NOM → Q_EXT over 3 seconds
- **Visual gate:** user confirms pantograph motion is correct
- Body weld removed in Phase 3 when sim_loop adds freejoint for balance

---

### PHASE 1: Core Physics ✅ COMPLETE

**Step 1.1 — Port `physics.py`**
- All functions take `RobotGeometry` param explicitly
- `solve_ik(q_hip, robot)`, `get_equilibrium_pitch(robot, q_hip)`, `build_xml(robot, world_config)`, `build_assets()`
- Verify: self-test IK at Q_RET, Q_NOM, Q_EXT, print results

**Step 1.2 — Port `battery_model.py` → `models/battery.py`**
- `BatteryModel.__init__(self, params: BatteryParams)`
- Verify: step 100 times at 10A, print voltage/SoC

**Step 1.3 — Port motor functions → `models/motor.py`**
- `motor_taper(tau_cmd, omega, v_batt, motor: MotorParams)`
- `motor_currents(tau_whl_L, tau_whl_R, tau_hip_L, tau_hip_R, params)`
- Verify: call motor_taper at omega=0, 50, 150 rad/s

**Step 1.4 — Port latency ring buffers → `models/latency.py`**
- `LatencyBuffer(n_steps, init_value)` — `n_steps=0` is transparent pass-through
- Verify: push 10 values into size-5 buffer, assert oldest is value[4]; push into size-0, assert oldest == last pushed

**Step 1.5 — Port thermal model → `models/thermal.py`**
- Takes `MotorThermalParams` dataclass
- Verify: heating at constant 20A for 60s

---

### PHASE 2: Controllers ✅ COMPLETE

**Step 2.1 — `controllers/lqr.py`**
- `compute_gain_table(robot, lqr_gains)`, `interpolate_gains(K_table, q_hip, robot)`, `lqr_torque(...)`
- Verify: compute K_table with Phase 6 gains, compare to current values

**Step 2.2 — `controllers/velocity_pi.py` + `yaw_pi.py`**
- Classes take params from `GainSet`
- Verify: step with known inputs, compare output

**Step 2.3 — `controllers/hip.py`**
- `hip_position_torque(...)`, `hip_impedance_torque(...)`
- Verify: unit test with known inputs

---

### PHASE 3: Simulation Loop ✅ COMPLETE (not yet regression-verified)

> **Note — latency disabled for now:** `LatencyParams` defaults are both `0.0` (pass-through).
> Before final regression (Phase 9), re-enable realistic latency defaults:
> `sensor_delay_s ≈ 0.005` (BNO086 + I2C), `actuator_delay_s ≈ 0.0025` (ODESC FOC + τ_elec).
> This will also require re-baselining gains to match the delayed plant.

**Step 3.1 — `scenarios/base.py`** — ScenarioConfig + WorldConfig dataclasses

**Step 3.2 — `scenarios/profiles.py`** — velocity staircase, leg cycle, disturbance fns

**Step 3.3 — Port `_run_sim_loop` → `sim_loop.py`** ← Critical
- `run(params, scenario, callbacks=None, command_queue=None, rng_seed=None) -> dict`
- Port from `latency_sensitivity/scenarios.py` (has latency model)
- **Hardware limit checks every tick** — violations logged, early-exit in optimizer mode:
  ```python
  if bearing_force > params.limits.max_bearing_force:
      metrics["hw_violation"] = "bearing"; break
  ```
- **Robot body uses freejoint** (body weld removed from Phase 0)
- Verify: **Critical regression** — run S1 headlessly, fitness ±1% of latency_sensitivity result
- Also **visual gate**: run S1 with MuJoCo passive viewer, confirm robot balances visually

**Step 3.4 — Port helper functions**
- `get_pitch_and_rate()`, sensor noise injection
- Verify: end-to-end S1 produces PASS

---

### PHASE 4: Scenario Definitions ✅ COMPLETE

**Step 4.1 — Define S1-S8 scenario configs**
- `s01_lqr_pitch_step.py` through `s08_terrain_compliance.py`
- Each: `CONFIG = ScenarioConfig(...)` + `def fitness(metrics: dict) -> float`
- Verify: `from master_sim.scenarios.s01_lqr_pitch_step import CONFIG; print(CONFIG.name)`

**Step 4.2 — Scenario registry + `evaluate()`**
- `scenarios/__init__.py`: SCENARIOS dict, `evaluate(params, scenario_name) -> dict`
- Verify: `evaluate(DEFAULT_PARAMS, "s01_lqr_pitch_step")` matches current values

**Step 4.3 — Regression test all 8 scenarios** ← Hard gate
- All S1-S8 headlessly with DEFAULT_PARAMS within 1% of Phase 6 baselines

---

### PHASE 5: Optimizer ✅ COMPLETE

**Step 5.1 — `optimizer/search_space.py`**
- `SearchSpace`, `sample_offspring(parent, sigma)`, `clamp()`, `random_init()`
- Verify: 8 offspring all within bounds

**Step 5.2 — `optimizer/es_engine.py`**
- Generic `ESOptimizer(search_space, eval_fn, csv_path, lambda_=8, patience=200)`
- Adaptive sigma, patience-based early stopping, multiprocessing pool
- Verify: 2 generations on S1, CSV written

**Step 5.3 — `optimizer/progress_ui.py`**
- tkinter window: gen count, best fitness, patience bar, convergence plot
- **Play/pause GUI button** (suspends worker pool)
- Verify: window appears, button toggles

**Step 5.4 — Thin optimizer entry points**
- `optimize_lqr.py`, `optimize_vel_pi.py`, `optimize_yaw_pi.py`, `optimize_suspension.py` (~40 lines each)
- Verify: each runs 1 generation

**Step 5.5 — `optimizer/pipeline.py`**
- Chain S1→S8 with gain propagation; writes `logs/baseline_gains.json`
- `defaults.py` reads this file if present, falls back to hardcoded Phase 6 values
- Verify: smoke test `--hours 0.02`

**Step 5.6 — `optimizer/run_log.py`**
- CSV I/O, run ID tracking
- Verify: can read existing CSV files from old folders

---

### PHASE 6: Unified Visualizer

**Step 6.1 — Replay mode** ✅ COMPLETE

Chart mode (CSV viewer) + replay mode (`--mode replay --scenario s01`):
- Run named scenario headlessly, collect telemetry via `TelemetryRecorder` callback
- pyqtgraph 5×2 panel (pitch, velocity, yaw rate, hip angle, roll, pitch rate, wheel torques, hip torques, battery V, position X)
- Optional `--viewer` flag launches MuJoCo passive viewer alongside (real-time rerun in background thread)
- `--list` prints available scenarios
- No matplotlib

Verify:
- `python -m master_sim.viz --mode replay --scenario s01` → replay runs, panels show data ✅
- `python -m master_sim.viz logs/S1_LQR_pitch_step.csv` → CSV chart viewer ✅

---

**Step 6.2 — Sandbox mode** (`--mode sandbox`, default — Scenario 0)

Ports `latency_sensitivity/sandbox_fastchart.py` into the unified visualizer.
Dual-process architecture: main process = MuJoCo sim + gamepad, child process = pyqtgraph telemetry.
Both share `master_sim` modules (controllers, models, physics) — no duplicated sim logic.

Sub-steps:

**Step 6.2a — Obstacle arena + MJCF builder integration** ✅ COMPLETE
- Ported `SANDBOX_OBSTACLES` (18 obstacles) and `SANDBOX_PROPS` (6 movable objects) into `viz/visualizer.py`
- Added `sandbox_world()` → `WorldConfig` with obstacles + props + 25m floor
- Added `prop_bodies` field to `WorldConfig` in `scenarios/base.py`
- Wired `prop_bodies` + `floor_size` through `build_model_and_data` in `sim_loop.py`
- Full `sandbox()` function: MuJoCo viewer + all controllers (LQR, VelPI, YawPI, impedance + roll leveling) + latency buffers + battery + auto-reset on fall
- CLI `--mode sandbox` wired; sandbox is default when no args given
- **Visual gate:** ✅ arena loads, obstacles visible, robot balances at standstill, props present

**Step 6.2b — Dual-process skeleton + 10-panel plot window** ✅ COMPLETE
- Plot process (`_plot_process`): independent Qt app with pyqtgraph `GraphicsLayoutWidget`
- 10 panels in 5×2 grid (matching original): Pitch, Pitch Rate, Velocity, Yaw Rate, Hip Joints (4 lines: L/R actual + L/R cmd), Roll, Wheel Torque (±limit lines), Hip Torque (±limit lines), Battery (dual-Y: voltage + current), Motor Currents (5 lines)
- Communication: `data_q` (main→plot, telemetry 25-tuples at 60 Hz) + `cmd_q` (plot→main, UI commands)
- 15-second rolling window via ring buffers (`deque(maxlen=1100)`)
- 60 Hz `QTimer` drains queue and updates all plot lines
- Status bar: SoC%, Batt Temp, I_bat, I_bat max (updated at 3 Hz throttle)
- Mouse hover Y-value display
- kill-on-exit daemon pattern (`os._exit(0)`)
- Verify: plot window opens, panels render (no data yet — flat lines)

**Step 6.2c — Control loop in main process** ✅ COMPLETE
- MuJoCo passive viewer on right half of screen, physics at model Hz, control at `CTRL_STEPS` sub-rate
- Full controller cascade using `master_sim` modules:
  - `controllers/lqr.py` → `lqr_torque()`
  - `controllers/velocity_pi.py` → `VelocityPI.update()`
  - `controllers/yaw_pi.py` → `YawPI.update()`
  - `controllers/hip.py` → `hip_impedance_torque()` / `hip_position_torque()` + `roll_leveling_offsets()`
  - `models/latency.py` → `LatencyBuffer` for sensor + actuator delay
  - `models/battery.py` → `BatteryModel.step()`
  - `models/motor.py` → `motor_taper()`, `motor_currents()`
- Sensor noise injection via `params.noise`
- Fall detection (|pitch| > 45°)
- Push 25-value telemetry tuple to `data_q` at 60 Hz
- Camera follow (track robot XY)
- Verify: robot balances at standstill, telemetry flows to plot panels, battery drains

**Step 6.2d — Plot process UI controls** ✅ COMPLETE
- Controller enable/disable checkboxes: LQR, VelPI, YawPI, Suspension, RollLev → `cmd_q.put(("CTRL_EN", key, bool))`
- Hip mode toggle button: Impedance ↔ Position PD → `cmd_q.put(("HIP_MODE", mode_str))`
- Restart button → `cmd_q.put("RESTART")` (resets sim state, clears ring buffers)
- Main process drains `cmd_q` each frame, applies enable flags + hip mode
- Verify: toggle LQR off → robot falls; toggle back on → balances after restart; hip mode toggles work

**Step 6.2e — Gamepad + keyboard input** ✅ COMPLETE
- Pygame joystick: left stick Y → velocity (±3 m/s), right stick X → yaw (±5 rad/s), right trigger → hip height (0–100%)
- Deadzone = 0.08
- Button 0 (A) = reset
- Fallback: sliders in plot UI when no joystick detected
- Verify: gamepad drives robot forward/back, turns, raises/lowers legs
- **Visual gate:** drive through obstacle arena, controller toggles + gamepad all working together

Final Phase 6 verification:
- `python -m master_sim.viz` → sandbox opens, pyqtgraph panels update live
- `python -m master_sim.viz --mode replay --scenario s01` → replay runs, panels show data
- Controller toggles work in sandbox
- Gamepad controls robot in sandbox
- `grep -r "import matplotlib" master_sim/` → zero results
- No `import matplotlib` anywhere in the package

---

### PHASE 7: Geometry Optimizer (4-bar)

**Step 7.1 — Port `eval_jump_balanced.py` to use shared physics**
- Replace inline IK/builder with `from master_sim.physics import ...`
- Create jump-specific `ScenarioConfig`
- Verify: same jump height for run_id 51167 geometry

**Step 7.2 — Geometry optimizer using generic ESOptimizer**
- SearchSpace over 6 geometry params
- Verify: 2 generations produce valid results

*(Jump controller: deferred — add in a future phase when ready)*

---

### PHASE 8: Log Migration + Documentation

**Step 8.1 — Copy CSV results to `master_sim/logs/`**
- Naming: `S01_lqr_pitch_step.csv` etc.
- Verify: `run_log.load_best("s02")` returns correct row

**Step 8.2 — Create `master_sim/README.md`**
- Architecture overview, quick-start (sandbox, optimizer, replay)
- How to add a scenario, define a search space
- Links to `docs/Control.MD` and `components/COMPONENTS.MD`

**Step 8.3 — Update `CLAUDE.md` and `docs/Control.MD`**
- Point all references to `simulation/mujoco/master_sim/`

---

### PHASE 9: Archive + Final Regression

**Step 9.1 — Archive old folders**
- Move to `simulation/mujoco/_archived/`
- Add README in each: "Frozen as of 2026-03-20. Replaced by master_sim/"
- Do NOT delete

**Step 9.2 — Full regression**
- All 8 scenarios ±1% of Phase 6 baselines
- Pipeline smoke test
- Visualizer: both modes, controller toggles
- `grep -r "import matplotlib" master_sim/` → zero results

---

## Verification Gates

| Phase | Gate |
|---|---|
| **0.4** | **Visual:** MuJoCo opens, robot model visible, body fixed in air |
| **0.5** | **Visual:** 4-bar sweeps hip joint correctly |
| 1–2 | Model + controller self-tests pass |
| **3.3** | **Visual + numeric:** S1 balances in viewer, fitness ±1% of Phase 6 |
| 4 | All 8 scenarios ±1% of Phase 6 baselines **(hard gate)** |
| 5 | 2 gen optimizer run, CSV written, play/pause button works |
| **6.1** | Replay mode: pyqtgraph telemetry panels + MuJoCo viewer; CSV chart viewer |
| **6.2a** | **Visual:** obstacle arena loads in MuJoCo viewer ✅ |
| **6.2b** | Plot window opens with 10 panels, ring buffers, status bar ✅ |
| **6.2c** | **Visual + numeric:** robot balances at standstill, telemetry flows to panels |
| **6.2d** | Controller toggles disable/enable controllers; restart resets sim ✅ |
| **6.2e** | **Visual:** gamepad drives robot through arena; no matplotlib anywhere ✅ |
| 7 | Jump height matches run_id 51167 |
| 9 | Full regression green |

---

## Critical Source Files (port from)

| New Module | Port From | Lines |
|---|---|---|
| `params.py` + `defaults.py` | `latency_sensitivity/sim_config.py` | 250 |
| `physics.py` | `latency_sensitivity/physics.py` | ~400 |
| `sim_loop.py` | `latency_sensitivity/scenarios.py` (_run_sim_loop + helpers) | ~600 |
| `scenarios/*.py` | `latency_sensitivity/scenarios.py` (8 runners) | ~800 |
| `controllers/lqr.py` | `latency_sensitivity/lqr_design.py` + scenarios.py | ~200 |
| `controllers/velocity_pi.py` | `latency_sensitivity/scenarios.py` (VelocityPI) | ~60 |
| `controllers/yaw_pi.py` | `latency_sensitivity/scenarios.py` (YawPI) | ~60 |
| `models/battery.py` | `latency_sensitivity/battery_model.py` | ~150 |
| `models/thermal.py` | `latency_sensitivity/thermal_model.py` | ~100 |
| `optimizer/es_engine.py` | `latency_sensitivity/optimize_lqr.py` (ES pattern) | ~300 |
| `optimizer/pipeline.py` | `latency_sensitivity/pipeline.py` | ~400 |
| `viz/visualizer.py` | `latency_sensitivity/sandbox_fastchart.py` + `replay.py` | ~1300 |
| `scenarios/s_jump.py` | `4bar_optimization_with_balancing/eval_jump_balanced.py` | ~300 |
