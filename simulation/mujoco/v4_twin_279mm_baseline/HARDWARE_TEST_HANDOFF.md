# Morning hardware handoff

The offline framework and firmware builds are ready. No robot command, flash,
motor motion, or live parameter write was performed overnight.

Work through the tests one at a time. Do not start a dynamic test until its
setup, physical restraints, ESTOP access, expected range, and abort condition
have been reviewed together.

## T0.1 mass inventory — complete from 2026-08-09 scale data

The twin now closes to 3.242 kg without the 0.276 kg battery and 3.518 kg in
driving configuration. Each wheel assembly is 0.478 kg. Body mass excluding
wheels is 2.562 kg. Individual measured values and the explicit 0.653 kg
unweighed residual are recorded in `components/COMPONENTS.md` and
`robot_match.json`.

The residual is currently assigned to the femur bodies as an equivalent
inertial placement because that best fits the available static log plateaus.
This closes total mass but does not identify true mass distribution.

## Next test: T0.2 CG measurement

Use the measured 3.518 kg driving mass and the actual available scales to plan
the two-support weight-split measurement. The highest-value result is CG versus
leg-height alpha; it will replace the fitted battery/residual placement without
requiring any powered balance test.

## Controller lock and offline replay

Plant fitting did not change controller gains. The current twin is locked to
all 171 values in `software/gui/parameter_exports/Default gains.json` with
SHA-256 `d4a963aa3ba747d6e317b5df08947dd6b4ad1511430bf620defec87437b3f3fe`.
The lock was extended after fitting with the 16 jump nudge/landing/handoff
parameters; those parameters were inactive in the balance evidence.
The direct replay separately uses the exact parameter sidecar captured with the
run (hash `e5a5dc827750d38dd07eea0e8bbf4ef505623e07c0f03e26adcf58f0209d990d`).
The meaningful historical-sidecar/current-export differences are
`jump_ext_timeout=0.15/0.2`, `jump_torque_max=0/7`, `roll_ki=0/0.1`, and
`roll_offset_max=0.15/0.30` (run/export). Jump was inactive in the reference
balance run, and roll control was disabled in both snapshots.

The reference WLOG now replays directly through MuJoCo with 100% RUNNING-command
coverage. In 2 s closed-loop reset windows, pitch RMSE is 2.11 degrees,
wheel-speed RMSE is 0.136 m/s, yaw-rate RMSE is 0.221 rad/s, and local
wheel/yaw-dead-reckoned XY RMSE is 0.102 m. This supports similar short-horizon
response, not calibrated long-distance path prediction; WLOG contains no
independent global position.

Hip sag/load fitting is plant-side only. The provisional reported-to-physical
joint-torque scale is 1.25 at retraction and 0.5927 at extension while the
logged hip gains/feedforward remain fixed. The retraction value hits its fit
bound, so this curve must not be treated as measured gearbox efficiency.

## Simulation-only robust gain candidate

The offline MuJoCo optimizer tuned 27 parameters across the LQR, velocity,
yaw, running hip, roll/differential hip, and FF1/FF2 loops. It did not change
the plant, trims, watchdogs, torque limits, command limits, or jump controller;
the export now carries the later jump parameters at their schema defaults.
The final candidate passed all 49 combinations of seven scenarios and seven
plant variations with zero falls or policy failures. Robust fitness improved
from 2.6854 to 1.3070, and worst pitch RMS improved from 7.82 to 2.94 degrees.

The GUI-compatible candidate is
`software/gui/parameter_exports/Robust_balance_candidate_2026-08-11.json`
(171 parameters, SHA-256
`bc75f8ec9621624779c6e0a011b099ac71f3af275c38a137986ed29d4bbc5a6d`).
It has not been sent to the robot. Do not apply it as a batch while the fitted
CG, pitch inertia, wheel torque scale/delay, friction, and hip transmission are
still provisional.

The simulated high-gain boundary agrees qualitatively with the real report:
the candidate passes its LQR scale sweep at 0.5x and 1.0x, but 1.5x already
exceeds the 35 degree/s pitch-rate RMS guard in 11 cases and exceeds 50% wheel
torque saturation in one case. Larger factors fail more cases. Do not increase
the candidate LQR gains further.

The apparent workaround is coordinated tuning, not simply raising LQR gains:
the optimizer increased hip damping, reduced velocity proportional action,
added velocity feedforward, and reduced static hip feedforward while increasing
pitch feedback. The hip changes are the least transferable part because the
hip plant is fitted from only two loaded equilibrium anchors. Complete T0.2,
T1.1, T2.1/T2.2, and T2.3/T2.4 before treating this set as a hardware candidate.
When hardware validation is eventually authorized, first review a dry-run diff
and use restraint/ESTOP with one controller group at a time.

## Build and offline status

- Protocol generation/check: pass.
- Generated twin params check: pass.
- Twin + SIL + shared log-analysis tests: 50 pass.
- Complete GUI suite: 84 tests and 26 subtests pass.
- Teensy 4.1 release build: pass.
- ESP32 release build: pass.
- GUI WLOG decoder and metrics on twin output: pass.
- Hardware flash/run: deliberately not done.

## Plant-ID firmware hook for later tests

The non-persistent params are `plant_id_en`, `plant_id_amp`, `plant_id_f0`,
`plant_id_f1`, and `plant_id_dur`. The chirp is symmetric wheel torque, enters
after the normal control law/barrier, and remains behind the normal torque
clamps and wheel governors. It auto-disarms at duration and whenever a
controlled state is exited. Do not arm it until a specific restrained test and
amplitude have been agreed.
