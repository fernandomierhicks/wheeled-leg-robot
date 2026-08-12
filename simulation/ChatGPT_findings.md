# Digital twin and robust controller findings

**Project:** v4 wheeled-leg robot, 279 mm linkage baseline  
**Report date:** 2026-08-11  
**Simulation package:** `simulation/mujoco/v4_twin_279mm_baseline/`  
**Status:** offline framework, robot matching, log replay, and robust balance-gain optimization complete; hardware validation still required

## Executive summary

The current MuJoCo twin balances with the firmware-equivalent control law and
the current robot parameters. The original instability was not primarily a
controller-code mismatch after the firmware port was completed. It came from a
combination of incorrect controller loading, insufficient modeled wheel torque
authority, stale mass representation, encoder-frame mismatch, and incorrect
RUNNING-state initialization.

After correcting those issues, the current default controller survives ten
10-second regression cases over five leg heights, both quiet and with a +5
degree initial pitch perturbation. A more demanding robust suite also shows the
default controller is stable, although its worst cases have high pitch error at
extended height and under forward-CG/low-friction uncertainty.

A staged optimizer was then run against the real matched MuJoCo model, not the
older analytical surrogate. It tuned 27 parameters across the LQR, velocity,
yaw, running-hip, roll/differential-hip, and FF1/FF2 controllers while keeping
all physical parameters, trims, safety thresholds, torque limits, and command
limits fixed. The final candidate:

- passed all 49 combinations of seven scenarios and seven plant variations;
- had zero simulated falls, firmware faults, or hard-limit failures;
- improved robust fitness from 2.6854 to 1.3070, a 51.3% improvement;
- reduced worst pitch RMS from 7.82 degrees to 2.94 degrees;
- reduced worst commanded-motion velocity RMS from 0.307 to 0.227 m/s;
- reduced worst yaw-rate RMS from 0.145 to 0.104 rad/s; and
- reduced worst hip tracking RMS from 0.205 to 0.145 rad.

The simulation also reproduces the operator's qualitative high-gain finding.
The optimized LQR set passes at its selected 1.0x scale but fails at 1.5x and
above because pitch-rate RMS rises beyond 35 degrees/s and wheel torque begins
to saturate. The simulated workaround is not simply higher LQR gain. It is
coordinated loop shaping: stronger pitch feedback combined with more hip
damping, lower velocity proportional gain, added velocity feedforward, and
lower static hip feedforward.

The candidate must still be treated as simulation-only. The largest transfer
risks are pitch inertia, absolute wheel torque scale and bandwidth, delay,
friction, true CG, and the effective hip transmission model. These cannot be
uniquely identified from the available closed-loop log.

No robot was commanded, moved, flashed, or written during this work.

## 1. Scope and governing principle

The objective was to make the simulation use the same controller values and
control law as the real robot, then adjust only physical/model parameters when
matching reality. Controller gains were locked during physical identification.
Only after producing a fixed matched plant was a separate controller
optimization performed.

This separation is important:

1. `fit_robot_match.py` fits plant-side quantities and never changes gains.
2. Historical WLOG replay uses the parameter sidecar captured with the run.
3. New simulation scenarios use the current GUI parameter export.
4. `robust_mujoco.py` freezes both the plant and its controller baseline before
   starting gain optimization.
5. The optimized candidate is written to a separate file and never overwrites
   `Default gains.json`.

## 2. Inputs and provenance

### 2.1 Current controller snapshot

The current matched twin loads all 155 parameters from:

`software/gui/parameter_exports/Default gains.json`

Current export SHA-256:

`d5dcda03206f705eb5442e2dc90d2004474293d48015b2586681b063825a427e`

The twin refuses to load if the export count or hash changes without rebuilding
`robot_match.json`. This prevents an unnoticed GUI or firmware schema change
from invalidating a previously fitted plant.

The optimizer began before the last jump-only schema addition and froze a
154-parameter controller input with SHA-256:

`8241cc00c9c583efdffa390b9a93512341bfd17919df7fcf31e6714302722b07`

While it ran, `jump_retract_angle` was added to the live export. The delivered
candidate was rebased onto the new 155-parameter export while preserving the
new jump value and every optimized balance value. It then repeated the final
seven-plant validation with identical results. Delivered candidate SHA-256:

`195f3d6b2ca125e00d22b56c4b9d64ae6d974c68ad2f6a057111d119d347cdd1`

### 2.2 Real robot log

Reference capture:

`data/logs/runs/20260810T045619_693178Z_SD_LOG0001/LOG0001.WLOG`

The log and its parameter sidecar were decoded with the shared GUI analysis
path described by `firmware/robot_teensy/AnalyzeLogClaude.md`. Important facts:

- telemetry version: 12;
- sample rate: 500 Hz;
- total samples: 112,451;
- total duration: 224.899 s;
- RUNNING duration: 215.48 s;
- RUNNING faults: zero;
- trim-relative pitch error: 2.777 degrees RMS over RUNNING;
- pitch rate: 12.038 degrees/s RMS over RUNNING;
- 1.5–4 Hz inner-LQR pitch content: 0.564 degrees RMS; and
- symmetric torque at 98% of the 0.4 N.m command limit for only 0.00835% of
  RUNNING samples.

This is strong evidence that the real inner balance loop was stable and was not
surviving mainly by sitting on its torque clamp.

Only two sections met the strict stationary-equilibrium requirements used for
CG/trim and hip-sag inference:

| Gain-schedule alpha | Duration | Measured balance pitch | Mean symmetric torque | Wheel drift |
|---:|---:|---:|---:|---:|
| 0.72947 | 29.45 s | -4.394 deg | 0.00360 N.m | -0.0336 turn/s |
| 0.92041 | 13.31 s | -4.093 deg | 0.00396 N.m | 0.00394 turn/s |

Other plateaus contain useful dynamic data but are not clean equilibrium
measurements because the robot was commanded, drifting, or carrying nonzero
wheel torque.

### 2.3 Historical versus current parameters

The log sidecar contains 138 values, with 136 names in common with the current
155-value export. Beyond float representation, the meaningful historical/current
differences are:

| Parameter | Log sidecar | Current export | Relevance to reference balance run |
|---|---:|---:|---|
| `jump_ext_timeout` | 0.15 | 0.20 | jump inactive |
| `jump_torque_max` | 0.0 | 7.0 | jump inactive |
| `roll_ki` | 0.0 | 0.1 | roll controller disabled |
| `roll_offset_max` | 0.15 | 0.30 | roll controller disabled |

Historical replay uses the log-era sidecar. New scenarios and the default
matched twin use the current export. Neither snapshot is treated as a plant-fit
variable.

## 3. As-built mass and geometry model

### 3.1 Supplied and measured masses

| Component | Quantity | Mass each |
|---|---:|---:|
| 6804 bearing | 16 | 18 g |
| wheel motor | 2 | 418 g |
| electronics box without battery | 1 | 505 g |
| battery | 1 | 276 g |
| TPU wheel | 2 | 29 g |
| PLA rim | 2 | 31 g |
| PLA coupler | 2 | 35 g |
| PLA femur | 2 | 40 g |
| PLA tibia | 2 | 85 g |
| AK45 hip motor, retained catalog prior | 2 | 260 g |

Measured whole-robot totals:

- without battery: 3.242 kg;
- battery: 0.276 kg;
- driving configuration: 3.518 kg;
- each wheel assembly: 0.478 kg; and
- body excluding wheels: 2.562 kg.

Known parts plus the retained AK45 catalog masses total 2.589 kg without the
battery. This leaves 0.653 kg of unweighed fasteners, shafts, mounts, wiring,
and possible catalog error.

MuJoCo carries this residual explicitly so total mass closes exactly. The
current constrained fit assigns the residual to the femur bodies as an
equivalent inertial lump, producing an effective simulated femur body mass of
366.5 g per side. This does not claim the printed femur weighs 366.5 g; the
print remains documented as 40 g.

### 3.2 Current fitted placement

- battery CG: x = +10 mm, z = -34 mm relative to the body frame;
- box CG offset: x = 0, z = 0;
- effective femur body: 0.3665 kg each;
- tibia: 0.085 kg each;
- coupler: 0.035 kg each; and
- total model mass: 3.518 kg with battery, 3.242 kg without.

The battery is at the forward/lower bounds allowed by the fit, and all
unresolved mass landed on the femurs. That boundary-hitting result is useful as
an identification hypothesis, not a physical measurement. The weighted trim
fit residual is 1.61 degrees. Direct CG and ride-height measurements should
replace this inferred placement.

### 3.3 Hip encoder and geometry alignment

The WLOG hip positions use the calibrated encoder frame, while MuJoCo uses the
CAD joint frame. Replay now applies the required +28 degree offset. The logged
hip positions are highly linear in schedule alpha:

- left: `q = -0.01476 - 1.46537 alpha` rad;
- right: `q = -0.02015 - 1.46678 alpha` rad; and
- RMS residual: approximately 0.00243 rad per side.

The current 1 degree calibration backoff places the scheduled span near +27 to
-57 degrees in the CAD frame. This replaces the stale +23 to -57 degree
assumption in older simulation paths.

## 4. Actuator and hip-load findings

### 4.1 Wheel torque conversion

The latest saved ODrive configuration uses:

- configured torque constant: approximately 0.04 N.m/A;
- current limit: 10 A; and
- documented installed motor: approximately 70 KV.

The physical motor torque constant inferred from KV is:

`Kt = 9.55 / 70 = 0.13643 N.m/A`

If ODrive interprets a requested torque using 0.04 N.m/A, one command N.m asks
for 25 A, while the physical motor produces about 0.13643 N.m per amp. The
resulting provisional command-to-physical torque scale is:

`0.13643 / 0.04 = 3.4107`

The 10 A current limit gives a provisional physical ceiling of 1.364 N.m per
wheel. This 3.4107x conversion is the dominant correction that made the matched
simulator balance. It is inferred from a saved preset and motor KV, not a live
configuration read or blocked-wheel torque measurement.

### 4.2 Hip sag and effective transmission

The real log shows mean reported hip torque increasing with extension:

| Alpha | Mean reported hip torque | Mean measured sag |
|---:|---:|---:|
| 0.0349 | -1.405 N.m | 0.0396 rad |
| 0.7295 | -2.029 N.m | 0.0147 rad |
| 0.9204 | -2.251 N.m | 0.00579 rad |

Only the two stationary equilibrium anchors were fitted. The resulting
effective reported-to-physical joint-torque schedule is:

- retracted: 1.25; and
- extended: 0.5927.

At the fitted anchors, mean sag residual is approximately -0.00242/+0.00207 rad
and reported-load-torque residual is -0.062/+0.048 N.m.

This schedule is not identified gearbox efficiency. The retracted factor hits
its fit bound, and the excluded low-height transient misses sag by about 0.042
rad and torque by about 1.05 N.m. The effective schedule may be absorbing link
mass placement, linkage leverage, gearbox loss, torque telemetry scale,
backlash, stiction, and unmodeled inner-loop dynamics.

## 5. Why the old simulation was unstable

Five practical mismatches were corrected.

1. **Wrong controller source.** The MuJoCo controller path constructed the
   controller from schema defaults rather than the current GUI export. It now
   loads and hashes the complete export.
2. **Wrong wheel authority.** One firmware torque-command unit was treated as
   one physical N.m. The saved ODrive/motor evidence implies approximately
   3.4107 times that physical torque, current-limited to 1.364 N.m.
3. **Stale mass model.** The former model was approximately 2.809 kg and did not
   close to the 3.518 kg driving mass. The new model accounts for the battery
   separately and explicitly carries the 0.653 kg residual.
4. **Hip frame mismatch.** Recorded calibrated hip positions were used as if
   they were CAD joint positions. Replay now applies the +28 degree offset.
5. **Incorrect controller state at RUNNING entry.** The simulator began with
   the hip feedforward ramp at zero. The real robot reaches RUNNING after
   STANDING_UP, so normal direct simulation now starts in an equivalent
   post-stand-up controller state.

Controlled ablation at alpha = 0.72947 demonstrates the relative importance:

| Physical correction | Result | Trim-relative pitch RMS |
|---|---:|---:|
| neither mass placement nor ODrive scale | fall at 1.71 s | 12.47 deg |
| mass placement only | fall at 3.07 s | 10.35 deg |
| ODrive scale only | survives 10 s | 0.64 deg |
| mass placement and ODrive scale | survives 10 s | 0.21 deg |

The torque conversion is the dominant stability correction. Mass/CG placement
substantially improves the steady balance point once enough wheel authority is
present.

## 6. How closely simulation matches reality

### 6.1 Static and short-horizon balance

The default matched twin passes ten 10-second cases at alpha 0, 0.5, 0.72947,
0.92041, and 1.0, each quiet and with a +5 degree pitch perturbation. There are
no simulated firmware faults.

This answers the narrow question "does this firmware control law balance the
current MuJoCo model?" with yes.

### 6.2 Command replay and path similarity

Direct WLOG replay feeds all recorded RUNNING commands to MuJoCo. It uses short
consecutive reset windows so inevitable state-estimation and plant error do not
compound over the full 215-second capture.

| Replay mode | Coverage | Pitch RMSE | Wheel-speed RMSE | Yaw-rate RMSE | Local XY RMSE |
|---|---:|---:|---:|---:|---:|
| open-loop, 0.1 s windows | 100% | 1.28 deg | 0.185 m/s | 0.271 rad/s | 0.0038 m |
| closed-loop, 2.0 s windows | 100% | 2.11 deg | 0.136 m/s | 0.221 rad/s | 0.102 m |

Closed-loop symmetric/differential wheel-torque RMSE is approximately
0.0227/0.0109 N.m. Hip reported-torque RMSE remains much larger at roughly
0.71/0.64 N.m.

The robot should follow a somewhat similar short-horizon commanded motion in
simulation, including direction changes and yaw. Accurate long-distance path
prediction is not yet established. WLOG has no independent global position;
the path reference is wheel/yaw dead reckoning, and small velocity/yaw errors
accumulate quickly.

### 6.3 What is modeled and commanded

The current balance simulation includes:

- the firmware-equivalent wheel LQR;
- alpha-scheduled pitch/rate gains and pitch trims;
- backward barrier and velocity-term guard;
- velocity PI, feedforward, rate limit, integral limit, and anti-windup;
- yaw PI and differential wheel torque;
- running hip position, damping, and feedforward commands;
- commanded hip-height trajectories and gain scheduling through the full span;
- hip sag under modeled load;
- roll PI-D and differential hip commands;
- FF1 and FF2;
- forward/backward velocity commands;
- yaw-rate commands;
- stateful sensor/actuator delays and sensor noise; and
- model perturbations, external pushes, friction changes, and battery/authority
  variations used for robust testing.

Hip sag is modeled through MuJoCo load, joint control, linkage geometry, and an
effective alpha-dependent hip torque scale. Flexibility, backlash, stiction,
and high-frequency structural modes are not yet independently identified.

Jump controller optimization was deliberately excluded. Jump contact, flight,
landing, impact transmission, and the newest jump state-machine changes have
not been matched to hardware, so optimizing jump gains now would reward model
error.

## 7. Robust gain optimization method

### 7.1 Controllers tuned

The optimizer tuned 27 values in five stages:

1. six LQR/barrier parameters;
2. five velocity PI/feedforward parameters;
3. four yaw PI/limit parameters;
4. twelve hip/roll/feedforward parameters; and
5. a joint 27-dimensional polish.

All values remained inside conservative engineering ranges narrower than the
firmware schema's storage/bench limits.

### 7.2 Parameters held fixed

The optimizer was prohibited from changing:

- masses, CG, inertias, friction, damping, delays, contact, noise, battery, and
  actuator conversions;
- retracted and extended pitch trims;
- pitch watchdog thresholds;
- the LQR torque limit;
- profile velocity/yaw command limits;
- the roll offset limit; and
- all jump parameters.

This ensures the gain result cannot hide a controller problem by moving the
plant or a safety boundary.

### 7.3 Plant ensemble

Seven deterministic plant variants were used for final validation:

| Variant | Principal perturbations |
|---|---|
| nominal | current matched plant |
| low authority / high delay | wheel 0.85x, hip 0.90x, sensor 6 ms, actuator 4 ms, pitch inertia 1.20x |
| high authority / low inertia | wheel 1.15x, hip 1.10x, sensor 4 ms, actuator 2 ms, pitch inertia 0.80x |
| forward CG / low friction | CG +6 mm, wheel 0.90x, contact friction 0.65x, damping 0.70x |
| backward CG / high friction | CG -6 mm, wheel 1.10x, friction 1.20x, damping 1.30x |
| noisy / low battery | sensor noise 2.5x, battery 0.85x, sensor 4 ms, actuator 3 ms |
| combined conservative | low wheel/hip authority, +5 mm CG, high inertia/delay, low friction/damping, 2x noise, 0.90x battery |

These are robustness stress tests, not seven competing physical fits.

### 7.4 Scenario suite

Each final plant ran:

1. retracted-height +7 degree recovery with alternating pushes;
2. extended-height -7 degree recovery with alternating pushes;
3. full leg-height cycling with pushes;
4. reversible, rate-shaped drive up to 0.65 m/s;
5. drive while cycling leg height;
6. drive with yaw reversals up to 1.2 rad/s; and
7. integrated drive, yaw, roll, height, and push disturbances.

Velocity and yaw commands stay within the current profile-3 limits of 0.75 m/s
and 1.5 rad/s. This avoids the legacy scenario problem where unrealistic command
steps exceeded active robot limits and immediately triggered velocity-error
abort logic.

### 7.5 Hard acceptance constraints

A candidate fails if any case has:

- a fall or firmware fault;
- more than 20 degrees trim-relative pitch;
- more than 35 degrees/s RMS pitch rate;
- more than 12 degrees roll;
- more than 7.5 N.m hip torque;
- wheel liftoff longer than 0.1 s; or
- wheel torque saturation for more than 50% of the case.

The soft objective prioritizes pitch and pitch rate, followed by velocity, yaw,
roll, hip tracking, actuator effort, and a small regularizer against unearned
large gain changes.

## 8. Optimization results

### 8.1 Stage results

| Stage | Seed fitness | Final fitness | Evaluations | Result |
|---|---:|---:|---:|---|
| LQR | 2.6361 | 2.1945 | 193 | PASS |
| velocity | 1.5634 | 1.3902 | 193 | PASS |
| yaw | 1.1538 | 1.1531 | 193 | PASS |
| hip/roll/feedforward | 1.3503 | 1.1009 | 193 | PASS |
| integrated polish | 1.2985 | 1.1799 | 97 | PASS |

The independent final validation used all seven plant variants and different
deterministic seeds. Its candidate fitness is 1.3070 rather than the four-plant
search score of 1.1799.

### 8.2 Final aggregate validation

| Metric over all 49 cases | Default controller | Candidate |
|---|---:|---:|
| status | PASS | PASS |
| failures | 0 | 0 |
| robust fitness | 2.6854 | 1.3070 |
| worst-case score | 2.4199 | 1.1045 |
| worst pitch RMS | 7.8199 deg | 2.9412 deg |
| worst peak pitch | 13.613 deg | 9.856 deg |
| worst pitch-rate RMS | 15.320 deg/s | 16.837 deg/s |
| worst roll tracking RMS | 1.633 deg | 1.830 deg |
| worst commanded-motion velocity RMS | 0.3073 m/s | 0.2267 m/s |
| worst yaw tracking RMS | 0.1446 rad/s | 0.1038 rad/s |
| worst hip tracking RMS | 0.2048 rad | 0.1447 rad |
| peak hip torque | 3.811 N.m | 3.632 N.m |
| maximum wheel saturation fraction | 0% | 0.489% |

The candidate accepts a small trade in worst pitch-rate and roll-tracking RMS,
remaining well below hard limits, in exchange for much better pitch, drive,
yaw, and hip tracking. The pure height-sweep mean score also increases slightly
from 0.6202 to 0.6608, while the difficult extended-height, drive-height, and
integrated cases improve substantially.

### 8.3 Candidate values

| Controller | Parameter | Default | Candidate |
|---|---|---:|---:|
| LQR | `lqr_k_pitch_ret` | -0.5000 | -1.2476 |
| LQR | `lqr_k_pitch_ext` | -0.5000 | -0.9892 |
| LQR | `lqr_k_rate_ret` | -0.1300 | -0.1854 |
| LQR | `lqr_k_rate_ext` | -0.1300 | -0.4538 |
| LQR | `lqr_k_vel` | -0.0500 | -0.1804 |
| barrier | `lqr_barrier_k` | 0.2000 | 0.9620 |
| velocity | `vel_pi_kp` | 0.2500 | 0.07095 |
| velocity | `vel_pi_ki` | 0.0800 | 0.18855 |
| velocity | `vel_pi_kff` | 0.0000 | 0.10698 |
| velocity | `vel_pi_rate_lim` | 0.43633 | 0.56465 |
| velocity | `vel_pi_int_max` | 1.0000 | 1.15731 |
| yaw | `yaw_pi_kp` | 0.0500 | 0.07259 |
| yaw | `yaw_pi_ki` | 0.0250 | 0.02870 |
| yaw | `yaw_pi_torque_max` | 0.0500 | 0.05780 |
| yaw | `yaw_pi_int_max` | 0.0500 | 0.05093 |
| running hip | `hip_running_kp` | 25.000 | 41.723 |
| running hip | `hip_running_kd` | 0.5000 | 1.0841 |
| running hip | `hip_running_tff_ret` | -2.4000 | -0.7300 |
| running hip | `hip_running_tff_ext` | -2.4000 | -0.5197 |
| hip roll | `hip_roll_kp` | 25.000 | 45.000 |
| hip roll | `hip_roll_kd` | 0.5000 | 0.8024 |
| roll | `roll_kp` | 1.0000 | 0.7263 |
| roll | `roll_kd` | 0.0100 | 0.01833 |
| roll | `roll_ki` | 0.1000 | 0.12327 |
| roll | `roll_int_max` | 0.1000 | 0.14804 |
| feedforward | `ff1_alpha` | 0.0000 | 0.02511 |
| feedforward | `ff2_alpha` | 0.0000 | 0.00444 |

The live default export was not modified. The complete candidate is stored at:

`software/gui/parameter_exports/Robust_balance_candidate_2026-08-11.json`

### 8.4 High-gain boundary

The five signed LQR state-feedback gains were scaled together after tuning:

| LQR factor | Default set | Optimized set |
|---:|---|---|
| 0.5x | FAIL: too weak; three falls | PASS |
| 1.0x | PASS | PASS |
| 1.5x | PASS | FAIL: 11 pitch-rate violations and one prolonged saturation case |
| 2.0x | PASS | FAIL: pitch-rate and saturation violations |
| 3.0x | FAIL | FAIL |
| 4.0x | FAIL | FAIL |

The candidate has a useful low-side margin but is already near its simulated
upper dynamic boundary. Increasing its LQR gains further is not recommended.

## 9. Interpretation of the controller workaround

The real robot reportedly became unstable when LQR gains were raised. The
simulation finds a qualitatively similar upper limit once the whole controller
is tuned and stressed across uncertain plants. The failure first appears as
excess pitch-rate activity and then increasing torque saturation, rather than a
simple inability to recover static pitch.

The selected solution suggests the limit is a multi-loop interaction:

- pitch feedback becomes stronger, especially at retracted height;
- extended-height rate damping becomes much stronger;
- velocity `Kp` becomes about 72% smaller;
- velocity integral/feedforward take over more of the steady and commanded
  response;
- running-hip and differential-hip damping increase substantially;
- static hip feedforward magnitude becomes much smaller; and
- roll proportional action becomes softer while roll damping/integral action
  increases.

This is consistent with a system where a fast wheel LQR interacts with leg
position stiffness, hip lag/compliance, velocity-loop pitch bias, and finite
wheel torque. Raising only the LQR makes that interaction worse. Coordinated
damping and outer-loop shaping let the inner loop become stronger without
immediately exciting the same mode.

This remains a hypothesis until hardware data confirms the oscillation
frequency, phase lag, hip motion, and torque saturation at the real gain
boundary.

## 10. Important limitations and risks

### 10.1 Physical parameters still provisional

The following cannot be uniquely recovered from the current closed-loop log:

- true horizontal and vertical CG versus leg height;
- distribution of the 0.653 kg residual mass;
- body pitch inertia versus leg height;
- wheel and reflected rotor inertia;
- rolling resistance, Coulomb friction, and viscous damping;
- contact friction under real tire loading;
- absolute wheel torque scale;
- torque-loop bandwidth and actuator delay;
- assembled-robot IMU noise and vibration;
- hip inner-loop bandwidth, backlash, stiction, and compliance; and
- real hip output torque versus reported torque.

### 10.2 Candidate values near search limits

`hip_roll_kp` reached the conservative search upper bound of 45. This does not
prove the real robot wants more stiffness. It indicates the current simulation
objective continued to reward that direction within the allowed region. Since
the hip plant is the least well identified part of the balance model, this
value should receive the greatest skepticism during transfer.

Several LQR and velocity gains moved substantially from the real defaults. The
candidate's clean simulation pass does not justify applying all 27 changes to
hardware as one batch.

### 10.3 Unmodeled structural dynamics

MuJoCo currently represents joints and links largely as rigid bodies with
effective damping/friction. The real instability may include:

- hip gearbox compliance or backlash;
- flex in femur/tibia/coupler parts;
- bearing or shaft compliance;
- tire deformation and contact patch dynamics;
- ODrive torque/current-loop dynamics;
- encoder/IMU filtering phase lag; or
- timing jitter and asynchronous motor feedback.

If the real high-gain failure is caused by one of these modes, the present
simulation can reproduce a gain boundary for the wrong physical reason.
Frequency and phase matching are therefore more important than matching only a
single trajectory.

### 10.4 Path prediction

The available WLOG has no independent motion-capture, optical, or external
odometry reference. Wheel/yaw dead reckoning cannot distinguish tire slip,
wheel-radius error, and state-estimation error. The current twin is suitable
for short-horizon response comparison, not accurate room-scale path prediction.

### 10.5 Jump

No conclusion about jump stability or optimized jump gains should be drawn from
this report. The newest jump controller/schema work was preserved but was not
optimized or validated in MuJoCo.

## 11. Proposed steps forward

The fastest path to transferable gains is to reduce uncertainty in the
parameters that set phase and gain margin. The recommended order below
maximizes information while minimizing robot risk.

### Phase A — configuration and static measurements

#### A1. Capture live ODrive configuration

Read and record both axes' live:

- `motor.config.torque_constant`;
- current limits;
- velocity/current controller configuration;
- input and torque ramp settings; and
- any active filters.

**Why first:** the current 3.4107x wheel torque scale is inferred from a saved
preset. If the live controller differs, nearly every gain-margin result moves.

**Gate:** saved configuration and live read agree, or the twin is updated and
all validation/optimization is rerun.

#### A2. Measure CG versus leg height (T0.2)

Use a restrained two-support weight split at alpha = 0, 0.25, 0.5, 0.75, and 1.
Measure horizontal CG; repeat in a safe orientation to obtain vertical CG.

**Gate:** replace the fitted battery/residual placement and explain the trim
curve without forcing mass to packaging bounds. If measured CG differs by more
than about 5–6 mm from the present ensemble, expand and rerun the ensemble.

#### A3. Measure ride height and A_Z (T0.3)

At the same alpha points, measure hip-shaft height above the floor and validate
the linkage curve.

**Gate:** simulation within 3 mm at every point. Correct geometry before using
dynamic results if this gate fails.

#### A4. Confirm calibration span (T0.4)

Measure the switch-to-hard-stop offset independently on both legs and confirm
the alpha mapping.

**Gate:** simulated and firmware schedule endpoints agree with both real legs.

### Phase B — passive dynamics

#### B1. Locked-wheel axle pendulum (T1.1)

Measure free-swing period and decay at alpha = 0, 0.5, and 1 with hips held and
wheels locked.

**Why high priority:** pitch inertia and damping directly determine the LQR
natural frequency and upper gain boundary.

**Gate:** simulated period within 5% and decay envelope acceptably matched at
all heights.

#### B2. Free-wheel pendulum (T1.2)

Repeat with wheels free to isolate reflected wheel/rotor inertia.

**Gate:** simulated locked/free period difference within 10%.

#### B3. Wheel free spin-up and coast-down (T1.3)

Use restrained, low torque steps and record acceleration and coast-down.

**Gate:** spin-up slope within 8% and coast-down time constant within 15% after
fitting wheel inertia, viscous damping, and Coulomb friction.

### Phase C — actuator calibration

#### C1. Blocked-wheel torque calibration (T2.1)

Use the existing lever and scale at several low commands to measure end-to-end
wheel torque.

**Gate:** fit command-to-physical torque slope and nonlinearity; replace the
provisional 3.4107x factor. The result should be repeatable on both axes.

#### C2. Wheel torque bandwidth/delay (T2.2)

Measure response to low-amplitude restrained torque modulation at increasing
frequency. Estimate delay and first-order/second-order bandwidth from measured
wheel acceleration and the commanded torque.

**Gate:** simulated gain and phase agree over the balance-relevant band,
especially near the real high-gain oscillation frequency.

#### C3. Hip impedance response (T2.3)

With the leg unloaded and restrained, use small position steps at known `kp` and
`kd`. Identify rise time, overshoot, settling, reversal dead zone, and
rate/torque limits.

**Gate:** simulated response and reversal behavior within 15% before accepting
the optimized hip gains.

#### C4. Hip holding torque versus alpha (T2.4)

Record supported static plateaus at alpha = 0, 0.25, 0.5, 0.75, and 1.

**Gate:** replace the current two-anchor effective scale with a validated curve
and match sag/holding torque within about 10% across the span.

### Phase D — update and re-optimize

After Phases A–C:

1. update `robot_match.json` and plant parameters only;
2. rerun standard ten-case validation;
3. rerun historical open/closed WLOG replay;
4. compare fidelity metrics against this report;
5. add any measured compliance/backlash/torque lag to MuJoCo;
6. rerun the robust optimizer from the latest real default gains;
7. validate with at least seven plant variants and multiple random/noise seeds;
8. run finer LQR scale sweeps near the failure boundary, for example 1.0 to
   1.6 in 0.05 increments; and
9. compare simulated oscillation frequency and saturation sequence with the
   real robot's eventual boundary test.

Do not preserve the current candidate merely because it scores well. The
candidate should move when newly measured physics moves; that is the purpose of
the digital twin.

### Phase E — staged hardware gain validation

Only begin after the relevant plant-ID gates pass. Do not push all 27 values at
once.

Recommended group order:

1. **Hip damping/hold group**, using a restrained supported pose and current
   limits appropriate to the test. Confirm no chatter, impacts, or unexpected
   torque steps.
2. **Wheel LQR group at conservative interpolation**, starting between the real
   default and optimized candidate rather than jumping directly to the final
   values. Keep the existing torque limit and watchdogs.
3. **Velocity group**, initially with low command speed and rate-shaped
   reversals. Verify that lower `Kp` plus added feedforward does not introduce
   unacceptable bias or windup on real flooring.
4. **Yaw group**, at low forward velocity first.
5. **Roll/differential hip group**, with external support because
   `hip_roll_kp=45` is the least trusted optimized boundary value.
6. **FF1/FF2**, only after hip torque telemetry and mass/CG are credible.

For each group:

- review the dry-run parameter diff;
- preserve a known-good rollback snapshot;
- use restraint and immediate ESTOP access;
- begin with small setpoint/disturbance amplitude;
- abort on unexpected hip oscillation, repeated torque clamp, worsening pitch
  rate, encoder discontinuity, loss of foot/wheel contact, or any fault;
- save the exact parameter sidecar with every log; and
- compare the hardware log with the identical simulation scenario before
  advancing.

The optimized LQR should not be scaled above 1.0. The simulation already shows
failure at 1.5x.

### Phase F — validate transfer and path fidelity

Run shared, rate-shaped scenarios on simulation and hardware:

1. stationary recovery at retracted and extended height;
2. slow height sweep while balancing;
3. forward/reverse velocity ramps;
4. drive plus height cycling;
5. drive plus yaw reversal; and
6. the integrated scenario only after the individual cases pass.

Compare:

- pitch and pitch-rate RMS/peaks;
- oscillation frequency and damping;
- wheel velocity and yaw-rate tracking;
- hip position, rate, and torque;
- torque saturation duration;
- controller integrals and commanded pitch bias;
- delay/cross-correlation between command and response; and
- fault/watchdog margins.

For true path validation, add an independent position reference such as optical
tracking, external camera markers, UWB, or carefully surveyed floor markers.
Do not score path transfer using wheel dead reckoning alone.

### Phase G — jump model, later

Only after balance/hip/contact parameters are identified:

1. reproduce crouch and retract trajectories unloaded;
2. measure hip response and torque during supported crouch/extension;
3. validate ground contact and tire compliance;
4. validate liftoff timing and body pitch without optimizing height;
5. identify landing compliance and impact limits; and
6. then create a separate robust jump optimizer with explicit safety and
   landing objectives.

Jump optimization should never share an objective with ordinary balance tuning
until the contact/flight transition is validated.

## 12. Recommended software improvements

The current framework is usable, but the following would increase confidence:

1. **Frequency-domain validation.** Add swept-sine/chirp comparison plots for
   pitch, wheel speed, and hip response. Gain margin is fundamentally a phase
   and bandwidth problem.
2. **Finer boundary search.** Replace the coarse 0.5/1/1.5/2/3/4 LQR sweep with
   bisection or adaptive search once measured delay and inertia are available.
3. **Compliance/backlash model.** Add measured hip/link compliance and dead zone
   if T2.3 shows them. Avoid adding arbitrary compliance without data.
4. **Torque-loop dynamics.** Model the measured ODrive torque/current response,
   not only a pure delay and clamp.
5. **Multi-seed validation.** Retain deterministic reproducibility but validate
   the final set against more noise/contact seeds than used during search.
6. **Held-out scenarios.** Keep several disturbance and command profiles hidden
   from optimization and use them only for acceptance.
7. **Uncertainty derived from measurements.** Replace convenient +/- factors
   with confidence intervals from repeated tests.
8. **Candidate interpolation tool.** Generate reviewed 10%, 25%, 50%, 75%, and
   100% controller deltas from the current real gains, with group-specific
   snapshots and rollback files.
9. **Automated sim/hardware report.** Run the same WLOG metrics and produce a
   per-channel comparison report after every shared scenario.
10. **Independent path reference support.** Add import/alignment for external
    X/Y/yaw measurements.

## 13. Immediate recommended sequence for the next session

If only one short hardware session is available, use this order:

1. inspect and save the live ODrive configuration;
2. perform the unpowered/static CG and ride-height measurements;
3. verify both hip calibration endpoints;
4. do not push the optimized gain file yet;
5. update the twin and rerun validation from the new measurements;
6. prepare the locked-wheel pendulum test;
7. after inertia is measured, run the restrained wheel torque calibration; and
8. only then decide whether a small, grouped interpolation toward the candidate
   is justified.

The highest-value first powered measurement is absolute wheel torque. The
highest-value dynamic measurement is locked-wheel pitch inertia and damping.
Together with CG, these determine whether the current simulated gain boundary
has the same physical cause as the real one.

## 14. Artifacts

Primary outputs:

- matched package: `simulation/mujoco/v4_twin_279mm_baseline/`;
- current physical/controller fit: `robot_match.json`;
- ten-case default validation: `robot_match_validation.json`;
- direct hardware-log replay: `robot_match_replay.json`;
- optimization implementation: `optimizer/robust_mujoco.py`;
- frozen optimizer baseline: `gain_optimization_baseline.json`;
- full optimizer output: `gain_optimization_report.json`;
- focused gain report: `GAIN_OPTIMIZATION_FINDINGS.md`;
- robot-match report: `ROBOT_MATCH_FINDINGS.md`;
- hardware test checklist: `HARDWARE_TEST_HANDOFF.md`;
- delivered GUI-compatible candidate:
  `software/gui/parameter_exports/Robust_balance_candidate_2026-08-11.json`;
- optimizer console output:
  `data/twin/robust_mujoco_optimizer.stdout.log`; and
- optimizer error log:
  `data/twin/robust_mujoco_optimizer.stderr.log`.

## 15. Verification completed

- standard ten-case robot-match matrix: 10/10 stable;
- final robust candidate validation: 49/49 PASS;
- combined twin, SIL, and GUI regression suite: 118 tests PASS;
- GUI parameter/protocol subtests: 26 PASS;
- generated twin parameter drift check: PASS;
- firmware protocol generated-file drift check: PASS;
- candidate schema load/count/hash check: PASS;
- guarded candidate diff contains exactly the 27 optimized parameters; and
- `git diff --check`: PASS.

The project virtual environment was missing its already-declared `pyserial`
dependency. `pyserial==3.5` was installed so the complete GUI suite could run.

## Final conclusion

The digital twin is now useful for controller development and robustness
screening. It balances with the same firmware-equivalent law, uses the current
parameter export, models the supplied mass inventory exactly, replays all real
RUNNING commands, includes hip-height and sag behavior, and reproduces a
qualitative upper LQR gain boundary.

It is not yet calibrated well enough to claim that the optimized numerical
gains will transfer unchanged. The candidate is best viewed as a technically
plausible direction and a set of hypotheses:

- stronger inner pitch feedback is beneficial;
- outer velocity proportional action should be softer;
- more hip damping is valuable;
- static hip feedforward is likely too strong in the present model/controller
  combination; and
- the real gain ceiling is probably a coupled wheel/hip/delay/saturation mode.

The next work should measure CG, pitch inertia, wheel torque scale/bandwidth,
and hip response. Once those are incorporated, rerun the same optimizer and use
the same scenario definitions to validate a staged hardware transfer.
