# Real robot versus MuJoCo — 2026-08-10 evidence match

## Inputs

- Firmware parameters: `software/gui/parameter_exports/Default gains.json`
  (171 schema-validated values, locked in `robot_match.json` by SHA-256
  `d4a963aa3ba747d6e317b5df08947dd6b4ad1511430bf620defec87437b3f3fe`).
  Sixteen jump-only parameters were appended after the balance fit and were
  inactive in this evidence.
- Robot run: `data/logs/runs/20260810T045619_693178Z_SD_LOG0001/LOG0001.WLOG`
  and its parameter sidecar, decoded with the GUI's shared WLOG modules as
  required by `firmware/robot_teensy/AnalyzeLogClaude.md`.
- Wheel drive: latest saved ODrive preset,
  `components/characterization/odrive/USB GUI/savedPresets/Both axis working CS 3 and 4 .json`.
- Mechanical inputs: the v4 CAD geometry plus the 2026-08-09 as-built mass
  inventory recorded in `params.py` and `components/COMPONENTS.md`.

No powered robot experiment was performed; scale measurements were supplied by
the operator and incorporated offline.

All 138 values present in the run's initial sidecar were reconciled against the
171-value export; 136 names are common to both. Beyond float text rounding, the
meaningful differences are `jump_ext_timeout` (0.15/0.2), `jump_torque_max`
(0/7), `roll_ki` (0/0.1), and `roll_offset_max` (0.15/0.3), shown as
run/export. Jump was inactive in the reference run, and roll control was
disabled in both snapshots, so these differences do not alter the balance
evidence.
RUNNING telemetry shows active profile index 2 (profile 3), whose 0.4 N·m limit
matches the export.

## As-built mass closure

The measured robot is 3.242 kg without its 0.276 kg battery, hence 3.518 kg in
driving configuration. Each wheel assembly is 0.478 kg: 418 g motor, 29 g TPU,
and 31 g PLA rim. The electronics/body box is 0.505 kg without the battery.
Measured prints are 40 g femur, 85 g tibia, and 35 g coupler per side. Sixteen
6804 bearings at 18 g account for 288 g. The existing 260 g catalog value is
retained for each unweighed AK45 hip motor.

Those known parts account for 2.589 kg without the battery, leaving 0.653 kg of
unweighed fasteners, shafts, mounts, wiring, and possible catalog discrepancy.
MuJoCo carries that residual explicitly rather than losing it: a constrained
fit assigns it to the femur bodies as an equivalent inertial placement (326.5 g
per side). This makes each simulated femur body 366.5 g, but the parameter
ledger still records the actual printed femur as 40 g. The battery is a separate
276 g fixed body at `x=+10 mm, z=−34 mm`, the forward/lower limits allowed by
the current chassis and battery geometry.

## What the real log proves

The robot was in uninterrupted RUNNING for 215.48 s at 500 Hz with no fault.
The trim-relative pitch error was 2.7774° RMS over the entire, commanded
leg-height run. Only 0.5636° RMS was in the 1.5–4 Hz inner-LQR band; most of the
whole-run error was slow velocity-loop/operator motion. Symmetric torque was at
98% of the 0.4 N·m command limit for only 0.00835% of RUNNING samples. This is
evidence of a stable real inner balance loop, not a robot surviving by sitting
on its torque clamp.

The run contains only two plateaus that satisfy the strict static-equilibrium
gate (near-zero mean symmetric torque, near-zero wheel drift, and no motion
command):

| α | duration | measured balance pitch | mean τsym | wheel drift |
|---:|---:|---:|---:|---:|
| 0.72947 | 29.45 s | −4.394° | 0.00360 N·m | −0.0336 turn/s |
| 0.92041 | 13.31 s | −4.093° | 0.00396 N·m | 0.00394 turn/s |

Other plateaus are useful dynamic data, but are not valid CG/balance-point
measurements because the robot was commanded, drifting, or still carrying
nonzero wheel torque.

## Why the previous simulation fell

Four mismatches were present.

1. The MuJoCo firmware-controller path constructed `FirmwareController` with
   schema defaults rather than the current GUI export. The simulator now loads
   the complete export, and scenario commands override only their intended
   command fields.
2. The simulator interpreted one firmware torque-command unit as one physical
   N·m. The saved ODrive preset has `torque_constant=0.04 N·m/A` and a 10 A
   current limit on both axes. If the installed motor is the documented 70 KV
   Maytech (`Kt=9.55/70=0.13643 N·m/A`), ODrive requests
   `command/0.04` amps and the motor produces approximately 3.4107 times the
   commanded torque, capped at approximately 1.364 N·m. This is the dominant
   stability correction. It is an inference from a saved configuration, not a
   measured torque calibration.
3. The former catalog/CAD mass model was 2.809 kg total and collapsed the body
   into one fitted point. The new model is 3.518 kg, models the 276 g battery
   separately, and closes every MuJoCo body to the measured whole-robot total.
   A constrained placement fit leaves about 1.61° weighted trim residual. Its
   battery and 653 g residual placements are useful identification choices, not
   substitutes for a direct CG measurement.
4. WLOG hip positions use the calibrated encoder frame whose zero is the retract
   switch, while MuJoCo uses the CAD frame whose retract stop is +28 degrees.
   Replay now applies that +28-degree offset explicitly.

The simulator also used to enter RUNNING with the one-second hip feedforward
ramp at zero. Real logs reach RUNNING through STANDING_UP, after that transient.
Direct simulation initialization now represents the post-stand-up state.

## Controlled MuJoCo comparison

At α=0.72947, zero commanded velocity, using the same exported controller:

| Physical model | result | trim-relative RMS error |
|---|---:|---:|
| neutral mass placement, unit torque conversion | fell at 1.71 s | 12.47° |
| fitted mass placement only | fell at 3.07 s | 10.35° |
| ODrive conversion, neutral mass placement | survived 10 s | 0.64° |
| fitted mass placement and ODrive conversion | survived 10 s | 0.21° |

The combined robot-match profile survived ten 10 s regressions: α = 0, 0.5,
0.72947, 0.92041, and 1.0, each both quiet and initialized with a +5° pitch
perturbation. There were no simulated firmware faults. This answers the narrow
question “does this control law balance this current MuJoCo model?” with yes;
it does not yet establish sim-to-real fidelity.

## Direct WLOG-to-MuJoCo replay

`twin/tools/replay_wlog.py` now feeds every RUNNING command from the reference
capture into MuJoCo. It uses consecutive reset windows: 0.1 s open-loop because
an inverted pendulum cannot tolerate replay-state error without feedback, and
2.0 s closed-loop to measure local controlled response. Controller parameters
come from the matching sidecar and are never optimization variables.

| Replay | coverage | pitch RMSE | wheel-speed RMSE | yaw-rate RMSE | local dead-reckoned XY RMSE |
|---|---:|---:|---:|---:|---:|
| open-loop torque/setpoint, 0.1 s windows | 100% | 1.28° | 0.185 m/s | 0.271 rad/s | 0.0038 m |
| closed-loop commands, 2.0 s windows | 100% | 2.11° | 0.136 m/s | 0.221 rad/s | 0.102 m |

Closed-loop torque RMSE is 0.0227 N·m symmetric and 0.0109 N·m differential,
but hip reported-torque RMSE remains 0.71/0.64 N·m. The path reference is only
wheel-speed/yaw dead reckoning; WLOG has no independent global X/Y measurement.
These results support “roughly similar local commanded motion”, not accurate
long-distance path prediction.

## Hip sag/load identification

The controller remains fixed at exported `hip_running_kp/kd/tff`. MuJoCo now
separates the MIT-reported torque used by firmware/FF1 from plant-side joint
torque. A provisional linear effective transmission scale was fitted only to
the two equilibrium anchors: 1.25 at retraction and 0.5927 at extension. At the
actual fitted anchors, mean sag residual is −0.00242/+0.00207 rad and reported
load-torque residual is −0.0623/+0.0481 N·m.

This schedule is not claimed as gearbox efficiency. Its retracted end hits the
fit bound, and the excluded low-height transient still misses sag by 0.042 rad
and torque by 1.05 N·m. It is evidence that the remaining mismatch contains
linkage/mass/actuator effects which T2.3/T2.4 must separate.

## Additional physical information extracted without experiments

- Current calibration span: the export's 1° retract backoff and 85° ranges put
  the scheduled hip span at approximately +27° to −57° rather than the old
  +23° to −57° model.
- Logged measured hip position is exceptionally linear in schedule α:
  left `q=-0.01476−1.46537α`, right `q=-0.02015−1.46678α` rad, with about
  0.00243 rad RMS residual on each side. This provides encoder registration and
  calibrated-span anchors.
- Mean reported hip motor torque rises with extension: approximately −1.405
  N·m at α=0.0349, −2.029 N·m at α=0.7295, and −2.251 N·m at α=0.9204. Together
  with commanded position and sag, these are now explicit plant-fit anchors;
  only the two equilibrium rows are fitted.
- The WLOG timing is a clean 500 Hz over 224.899 s. It can constrain loop jitter
  and closed-loop frequency response, but not wheel current or true shaft
  torque because neither is present in telemetry.

The total/component scale inventory is now known, but the 653 g residual's true
distribution, vertical CG, body pitch inertia, wheel inertia, rolling
resistance, contact friction, and absolute wheel torque scale cannot be uniquely
recovered from this closed-loop log.

## Highest-value checks when hardware is available

1. Read both live ODrive `motor.config.torque_constant` and `current_lim` values.
2. Perform T2.1 blocked-wheel lever/scale calibration; this resolves the main
   provisional 3.4107× factor.
3. Perform T0.2/T0.3 CG and ride-height checks; T0.1 mass inventory is complete.
4. Perform the locked-wheel pendulum test for body pitch inertia.
5. Only after those, use new robot logs to fit wheel damping, rolling resistance,
   actuator lag, and contact parameters.

Machine-readable fit, stability, and replay details are in `robot_match.json`,
`robot_match_validation.json`, and `robot_match_replay.json`. Rebuild them with:

```powershell
python -m v4_twin_279mm_baseline.twin.tools.fit_robot_match
python -m v4_twin_279mm_baseline.twin.tools.validate_robot_match `
  --output v4_twin_279mm_baseline/robot_match_validation.json
python -m v4_twin_279mm_baseline.twin.tools.replay_wlog `
  ../../data/logs/runs/20260810T045619_693178Z_SD_LOG0001/LOG0001.WLOG `
  --mode both --output-json v4_twin_279mm_baseline/robot_match_replay.json
```
