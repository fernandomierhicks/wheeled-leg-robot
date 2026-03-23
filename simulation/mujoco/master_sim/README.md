# master_sim — Quick Reference

Working directory for all commands: `simulation/mujoco/`

## Run a single scenario (headless) and print fitness

```python
python -c "
from master_sim.defaults import DEFAULT_PARAMS
from master_sim.scenarios import evaluate

m = evaluate(DEFAULT_PARAMS, 's09_integrated')
print(f'Status:            {m[\"status\"]}')
print(f'Survived:          {m[\"survived_s\"]}s')
print(f'RMS Pitch:         {m[\"rms_pitch_deg\"]:.4f} deg')
print(f'RMS Pitch Rate:    {m[\"rms_pitch_rate_dps\"]:.4f} deg/s')
print(f'Vel Tracking RMS:  {m[\"vel_track_rms_ms\"]:.4f} m/s')
print(f'Yaw Tracking RMS:  {m[\"yaw_track_rms_rads\"]:.4f} rad/s')
print(f'Max Roll:          {m[\"max_roll_deg\"]:.4f} deg')
print(f'Fitness:           {m[\"fitness\"]:.4f}')
"
```

- Replace `'s09_integrated'` with any scenario key: `s01_lqr_pitch_step`, `s02_leg_height_gain_sched`, `s03_vel_pi_disturbance`, `s04_vel_pi_staircase`, `s05_vel_pi_leg_cycling`, `s06_yaw_pi_turn`, `s07_drive_turn`, `s08_terrain_compliance`, `s09_integrated`.
- `DEFAULT_PARAMS` contains the current baseline gains from `defaults.py`.
- `evaluate()` runs `sim_loop.run()` headlessly and appends the `fitness` key via the scenario's `fitness_fn`.
- Typical runtime: ~10-30s depending on scenario duration.

## Launch the GUI launcher (scenario picker + MuJoCo viewer)

```bash
python master_sim/launcher.py
```

## Run a single optimizer. From /simulation/mujoco/

The unified entry point is `optimize_integrated` — it auto-selects the search space based on the scenario's group (e.g. S1 → LQR 4D, S9 → all 12D). A `ProgressUI` window shows live best-fitness, gain values, and play/pause.

```bash
# S1 LQR only, 5 minutes, from baseline gains
python -m master_sim.optimizer.optimize_integrated --hours 0.0833 --patience 50 --scenario s01_lqr_pitch_step

# S1 LQR only, 5 minutes, from random starting point (escape local minima)
python -m master_sim.optimizer.optimize_integrated --hours 0.0833 --patience 50 --scenario s01_lqr_pitch_step --random-seed

# S9 integrated (all 12 gains), 2 hours
python -m master_sim.optimizer.optimize_integrated --hours 2 --patience 500 --scenario s09_integrated

# Explicit seed gains
python -m master_sim.optimizer.optimize_integrated --hours 1 --scenario s01_lqr_pitch_step --seed-gains "Q_PITCH=0.5,Q_PITCH_RATE=0.01,Q_VEL=1e-6,R=0.02"
```

Common flags:

| Flag | Default | Description |
|---|---|---|
| `--hours H` | 1.0 | Wall-clock time limit |
| `--patience N` | 300 | Early-stop after N gens without improvement |
| `--tol F` | 1e-4 | Relative improvement threshold |
| `--workers N` | auto | Parallel eval workers |
| `--seed-gains "K=v,..."` | baseline | Explicit starting gains |
| `--random-seed` | off | Start from random point in search space |
| `--no-baseline` | off | Skip writing best gains into params.py |

There are also single-controller optimizer scripts (`optimize_lqr`, `optimize_vel_pi`, `optimize_yaw_pi`, `optimize_suspension`) but `optimize_integrated` is preferred as it uses the richer GUI with all gain groups visible.

## Run the full S1→S9 pipeline with random gains for initial controllers. 2 hours per scneario. 

```bash
python -m master_sim.optimizer.pipeline --hours 2 --fresh --random --patience 500
```

The pipeline chains steps sequentially, seeding each from the previous step's best result. Baseline steps write to `logs/baseline_gains.json`. Use `--fresh` to delete existing CSVs and re-run from scratch.

## Baseline gains after an optimizer run

Baselining means copying the best gains from an optimizer run into the dataclass defaults in `master_sim/params.py`, so all future simulations use them.

1. **Read the best params** from the optimizer output or CSV log (`logs/S_<scenario>.csv`).

2. **Edit `master_sim/params.py`** — update the default field values on the relevant dataclasses:

   | Gain group | Dataclass | Fields |
   |---|---|---|
   | LQR (4) | `LQRGains` | `Q_pitch`, `Q_pitch_rate`, `Q_vel`, `R` |
   | VelocityPI (2) | `VelocityPIGains` | `Kp`, `Ki` |
   | YawPI (2) | `YawPIGains` | `Kp`, `Ki` |
   | Suspension (4) | `SuspensionGains` | `K_s`, `B_s`, `K_roll`, `D_roll` |

   `defaults.py` simply does `SimParams()`, so changing the dataclass defaults is all that's needed.

3. **Verify** by running the scenario headlessly (see "Run a single scenario" above) and confirming fitness is in the expected range. Stochastic noise means each eval varies — a ±10% spread is normal.

4. **Visually check** via the launcher (`python master_sim/launcher.py`) — click the relevant scenario and watch for tracking quality, oscillations, or falls.

## Active controllers per scenario

Each scenario enables only the controllers relevant to its tuning phase. The `active_controllers` field on `ScenarioConfig` is the single source of truth, consumed by both `sim_loop.run()` and the visualizer replay via `tick_flags`.

| Scenario | Controllers | Hip mode |
|---|---|---|
| **S1** — LQR Pitch Step | LQR | position |
| **S2** — Gain Sched | LQR | position |
| **S3** — VelPI Disturbance | LQR, VelPI | position |
| **S4** — VelPI Staircase | LQR, VelPI | position |
| **S5** — VelPI Leg Cycling | LQR, VelPI | position |
| **S6** — Yaw PI Turn | LQR, VelPI, YawPI | position |
| **S7** — Drive + Turn | LQR, VelPI, YawPI | position |
| **S8** — Terrain Compliance | LQR, VelPI | impedance |
| **S9** — Integrated | LQR, VelPI, YawPI | impedance |

When `hip_mode="impedance"` (S8, S9), suspension and roll-leveling are automatically enabled.

## Key entry points

| What | Where |
|---|---|
| Baseline gains | `master_sim/defaults.py` → `DEFAULT_PARAMS` |
| Scenario configs + fitness functions | `master_sim/scenarios/s01_*.py` ... `s09_*.py` |
| Headless sim runner | `master_sim/sim_loop.run(params, scenario_config)` → returns metrics dict |
| `evaluate()` wrapper | `master_sim/scenarios/__init__.py` — calls `sim_loop.run` + `fitness_fn` |
| Optimizer scripts | `master_sim/optimizer/` |
| Visualizer / replay | `master_sim/viz/visualizer.py` → `run_unified()` |
