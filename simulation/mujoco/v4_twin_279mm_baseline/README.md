# v4 digital twin — 279 mm baseline

This package is the v4 digital-twin workspace described by `DigitalTwin.md`.
It is isolated from every older simulation. Its normal MuJoCo balance/drive/yaw
path and its low-order headless plant both use the schema-driven Python port of
the Teensy control loop. The copied legacy jump phase remains isolated pending
jump plant identification.

All commands below run from `simulation/mujoco/`.

## What is authoritative

- Control names/defaults/bounds: `firmware/robot_teensy/protocol/schema.json`.
- Generated twin view: `twin/params_control.py`; never edit it directly.
- Plant constants: `twin/params_plant.py`. Every unmeasured value is listed in
  `PROVISIONAL_FIELDS` with the test that replaces it.
- Mass prior: catalog motor/bearing masses plus analytical 6061 v4 tube masses
  give 1.769488 kg excluding wheels and 2.809488 kg total. These are accounting
  priors, not measurements; T0.1 remains authoritative.
- Geometry: `params.py` and `physics.py`, using the v4 dogleg dimensions and the
  calibrated alpha endpoints (+23° to -57° with the default 5° backoff).
- Log decoding and fitness: the existing GUI modules under
  `software/gui/analysis/`; the twin does not reimplement them.

The source plan's printed dogleg transform and its signed `w_perp=-30.354 mm`
are handedness-inconsistent with the stated acceptance anchors. This package
keeps the signed CAD parameter and uses the equivalent opposite local basis in
`physics.py`; the acceptance anchors are regression-tested exactly.

## Quick checks

```powershell
python -m pytest v4_twin_279mm_baseline/tests ../sil -q
python v4_twin_279mm_baseline/twin/generate_params.py --check
python -m v4_twin_279mm_baseline.twin.runtime `
  v4_twin_279mm_baseline/scenarios_json/pitch_step.json
```

The normal MuJoCo launcher remains available:

```powershell
python v4_twin_279mm_baseline/launcher.py
```

## Shared scenarios and genuine WLOG output

`scenarios_json/` contains pitch, velocity, and yaw timelines. The offline
runner writes genuine 500 Hz WLOG v1 records plus a matching `.PARAMS` sidecar:

```powershell
python -m v4_twin_279mm_baseline.twin.runtime `
  v4_twin_279mm_baseline/scenarios_json/velocity_step.json `
  --output ../../data/twin/velocity_step.WLOG
```

The output opens directly in the existing GUI Log Analyzer and works with the
existing `software/gui/tools/wlog_to_csv.py` and `analysis/wlog_metrics.py`.

The hardware runner consumes the same JSON. It is dry-run-only unless both
execution flags are present, then additionally requires a live, fault-free
RUNNING robot:

```powershell
python -m v4_twin_279mm_baseline.twin.tools.run_hardware_scenario `
  v4_twin_279mm_baseline/scenarios_json/velocity_step.json

# Hands-on only — review the printed plan first:
python -m v4_twin_279mm_baseline.twin.tools.run_hardware_scenario `
  v4_twin_279mm_baseline/scenarios_json/velocity_step.json `
  --execute --acknowledge-motion-risk
```

## Parameter flow

The pull/push tools communicate only through the GUI's existing local
`robot_ctl.py` server. Push is a diff-only dry run by default, is restricted to
`GAIN_ALLOWLIST`, validates current schema bounds, and refuses changed values
outside that allowlist.

```powershell
python -m v4_twin_279mm_baseline.twin.tools.pull_params params/live_robot.json
python -m v4_twin_279mm_baseline.twin.runtime `
  v4_twin_279mm_baseline/scenarios_json/pitch_step.json `
  --params params/live_robot.json
python -m v4_twin_279mm_baseline.twin.tools.push_params params/candidate.json

# Hands-on only, after reviewing the diff:
python -m v4_twin_279mm_baseline.twin.tools.push_params `
  params/candidate.json --apply
```

## Replay and fidelity score

Both modes load the matching sidecar and output per-channel NRMSE. Open-loop
uses recorded wheel torques to isolate plant error; closed-loop runs the Python
firmware controller from recorded commands.

```powershell
python -m v4_twin_279mm_baseline.twin.tools.replay_wlog RUN.WLOG --mode open
python -m v4_twin_279mm_baseline.twin.tools.replay_wlog RUN.WLOG --mode closed `
  --output-csv ../../data/twin/RUN_overlay.csv
```

These scores are not acceptance numbers until the provisional plant is replaced
with measurements and reference hardware logs exist.

## Robust firmware-gain optimization

The existing `(1+lambda)` ES engine is reused with signed/linear firmware-gain
spaces. Each candidate is scored by its worst result over the deterministic
plant ensemble (body inertia ±20%, delays ±1 ms, wheel torque ±15%, CG offset
±10%). Output is directly consumable by the guarded push tool.

```powershell
python -m v4_twin_279mm_baseline.optimizer.robust_firmware `
  v4_twin_279mm_baseline/scenarios_json/pitch_step.json `
  --stage lqr --generations 100 --ensemble-size 5 `
  --output ../../data/twin/lqr_candidate.json
```

Do not transfer optimized gains to hardware until the identification tests in
`HARDWARE_TEST_HANDOFF.md` have replaced the corresponding provisional plant
constants and the candidate passes the same shared scenario on the robot.

## SIL

`simulation/sil/test_sil_equivalence.py` compiles the production Teensy
`control_loop.cpp` with desktop hardware stubs, executes
`golden_vectors.csv`, and compares the result to `FirmwareController`. This is
the control-port drift gate; it deliberately compiles the real source instead
of a second C++ transcription.

## Current boundary

Implemented offline: v4 geometry, schema params, firmware control port,
MuJoCo balance route, provisional analytical plant, genuine WLOG/sidecars,
GUI-compatible analysis, replay, shared scenarios, guarded parameter flow,
robust optimizer, plant-ID firmware hook, SIL, tests, and both MCU builds.

Not claimed: identified mass/CG/inertia/friction/delay/noise, hardware fidelity,
hardware safety acceptance, jump fidelity, or GUI-as-live-wire-protocol client.
Those require the hands-on sequence beginning in `HARDWARE_TEST_HANDOFF.md`.
