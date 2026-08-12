# Robust balance gain optimization — 2026-08-11

## Outcome

The matched MuJoCo twin produced a simulation-only candidate for every
identified balancing controller: wheel LQR, velocity PI/feedforward, yaw PI,
running hip hold, roll/differential hip control, and FF1/FF2. The jump
controller was deliberately excluded because flight/contact fidelity has not
been identified and jump is outside the reliable-balancing objective.

The candidate passed all 49 final scenario/plant combinations with no fall,
firmware fault, or hard-limit failure. Robust fitness improved 51.3%, from
2.6854 to 1.3070. Worst-case score improved 54.4%, from 2.4199 to 1.1045.

This is evidence that the candidate is robust to the uncertainties represented
by the current ensemble. It is not hardware validation: pitch inertia, CG,
wheel torque scale/bandwidth, friction, delays, and the hip transmission model
remain provisional.

## What was held fixed

Only controller gains and controller-loop limits were search variables. The
optimizer did not change:

- any mass, CG, inertia, friction, delay, contact, noise, battery, or actuator
  plant parameter;
- retracted/extended pitch trims;
- pitch watchdog thresholds;
- the LQR torque limit;
- radio/profile velocity or yaw command limits;
- the roll offset limit; or
- any jump parameter.

The 154-parameter controller input was frozen at optimizer launch with hash
`8241cc00c9c583efdffa390b9a93512341bfd17919df7fcf31e6714302722b07`.
A concurrent jump-only schema change added `jump_retract_angle`; the delivered
candidate was rebased onto the new 155-parameter export without changing any
optimized value. The rebased candidate re-ran the seven-plant validation with
identical metrics and has hash
`195f3d6b2ca125e00d22b56c4b9d64ae6d974c68ad2f6a057111d119d347cdd1`.

## Search and validation suite

The optimizer used four deterministic plant variants during search and all
seven for final validation:

1. nominal matched plant;
2. low wheel/hip authority, high delay, and high pitch inertia;
3. high authority and low pitch inertia;
4. forward CG with low friction/damping;
5. backward CG with high friction/damping;
6. high sensor noise and low battery voltage; and
7. a combined conservative low-authority/high-delay/forward-CG case.

Each plant ran seven scenarios: retracted +7 degree recovery and pushes,
extended -7 degree recovery and pushes, full height cycling, reversible
rate-shaped drive, drive during height cycling, drive plus reversing yaw, and
an integrated drive/yaw/roll/height/push case. Velocity and yaw requests remain
inside the current profile-3 limits.

Hard failures include a fall or firmware fault, more than 20 degrees trim-
relative pitch, more than 35 degrees/s RMS pitch rate, more than 12 degrees
roll, more than 7.5 N.m hip torque, wheel liftoff over 0.1 s, or wheel torque
saturation for more than half a case.

## Staged optimizer results

| Stage | Seed fitness | Final fitness | Search evaluations | Status |
|---|---:|---:|---:|---|
| LQR | 2.6361 | 2.1945 | 193 | PASS |
| velocity | 1.5634 | 1.3902 | 193 | PASS |
| yaw | 1.1538 | 1.1531 | 193 | PASS |
| hip/roll/feedforward | 1.3503 | 1.1009 | 193 | PASS |
| integrated 27-D polish | 1.2985 | 1.1799 | 97 | PASS |

The final independent validation uses different deterministic seeds and all
seven plant variants; its fitness is 1.3070 rather than the four-plant
integrated-search score of 1.1799.

## Final parameter changes

| Group | Parameter | Default | Candidate |
|---|---|---:|---:|
| LQR | `lqr_k_pitch_ret` | -0.5000 | -1.2476 |
| LQR | `lqr_k_pitch_ext` | -0.5000 | -0.9892 |
| LQR | `lqr_k_rate_ret` | -0.1300 | -0.1854 |
| LQR | `lqr_k_rate_ext` | -0.1300 | -0.4538 |
| LQR | `lqr_k_vel` | -0.0500 | -0.1804 |
| LQR | `lqr_barrier_k` | 0.2000 | 0.9620 |
| velocity | `vel_pi_kp` | 0.2500 | 0.07095 |
| velocity | `vel_pi_ki` | 0.0800 | 0.18855 |
| velocity | `vel_pi_kff` | 0.0000 | 0.10698 |
| velocity | `vel_pi_rate_lim` | 0.43633 | 0.56465 |
| velocity | `vel_pi_int_max` | 1.0000 | 1.15731 |
| yaw | `yaw_pi_kp` | 0.0500 | 0.07259 |
| yaw | `yaw_pi_ki` | 0.0250 | 0.02870 |
| yaw | `yaw_pi_torque_max` | 0.0500 | 0.05780 |
| yaw | `yaw_pi_int_max` | 0.0500 | 0.05093 |
| hip | `hip_running_kp` | 25.000 | 41.723 |
| hip | `hip_running_kd` | 0.5000 | 1.0841 |
| hip | `hip_running_tff_ret` | -2.4000 | -0.7300 |
| hip | `hip_running_tff_ext` | -2.4000 | -0.5197 |
| hip/roll | `hip_roll_kp` | 25.000 | 45.000 |
| hip/roll | `hip_roll_kd` | 0.5000 | 0.8024 |
| roll | `roll_kp` | 1.0000 | 0.7263 |
| roll | `roll_kd` | 0.0100 | 0.01833 |
| roll | `roll_ki` | 0.1000 | 0.12327 |
| roll | `roll_int_max` | 0.1000 | 0.14804 |
| feedforward | `ff1_alpha` | 0.0000 | 0.02511 |
| feedforward | `ff2_alpha` | 0.0000 | 0.00444 |

## Final validation metrics

| Metric across all 49 cases | Default | Candidate |
|---|---:|---:|
| robust fitness | 2.6854 | 1.3070 |
| worst pitch RMS | 7.8199 deg | 2.9412 deg |
| worst peak pitch | 13.613 deg | 9.856 deg |
| worst pitch-rate RMS | 15.320 deg/s | 16.837 deg/s |
| worst commanded-motion velocity RMS | 0.3073 m/s | 0.2267 m/s |
| worst yaw-rate RMS | 0.1446 rad/s | 0.1038 rad/s |
| worst hip tracking RMS | 0.2048 rad | 0.1447 rad |
| peak hip torque | 3.811 N.m | 3.632 N.m |
| maximum wheel saturation fraction | 0% | 0.489% |

The candidate makes a modest trade: its pure height-sweep mean score increases
from 0.6202 to 0.6608, and worst pitch-rate/roll tracking are slightly higher.
The large improvements are in extended-height recovery, drive plus height, and
the integrated cases. Those tradeoffs remain below all hard limits.

## Does MuJoCo reproduce the real high-gain problem?

Qualitatively, yes. Scaling the five signed LQR feedback gains together gives:

| Gain factor | Default set | Optimized set |
|---:|---|---|
| 0.5x | FAIL: too weak; three falls | PASS |
| 1.0x | PASS | PASS |
| 1.5x | PASS | FAIL: pitch-rate limit; one saturation case |
| 2.0x | PASS | FAIL: pitch-rate and saturation limits |
| 3.0x | FAIL | FAIL |
| 4.0x | FAIL | FAIL |

The optimized set is already closer to its upper dynamic boundary. At 1.5x,
11 cases exceed 35 degrees/s RMS pitch rate and one case spends over half its
time on the wheel torque clamp. This agrees with the operator's observation
that simply raising LQR gains excites unstable robot/leg behavior.

The simulated workaround is coordinated loop shaping: stronger pitch feedback
is paired with softer velocity proportional action, explicit velocity
feedforward, more hip damping, and different static hip feedforward. It is not
evidence that the real robot should accept the full numerical gains. In
particular, `hip_roll_kp` landed at its conservative search bound and the hip
plant is inferred from only two equilibrium anchors. This is the candidate's
largest transfer risk.

## Artifacts and reproduction

- Candidate: `software/gui/parameter_exports/Robust_balance_candidate_2026-08-11.json`
- Full report: `gain_optimization_report.json`
- Frozen historical input: `gain_optimization_baseline.json`
- Optimizer: `optimizer/robust_mujoco.py`
- Console logs: `data/twin/robust_mujoco_optimizer.stdout.log` and `.stderr.log`

From `simulation/mujoco/`:

```powershell
python -m v4_twin_279mm_baseline.optimizer.robust_mujoco `
  --generations 24 --final-generations 12 --lambda 8 --workers 8 `
  --ensemble-size 4 --validation-ensemble-size 7 --seed 20260811 `
  --output "../../software/gui/parameter_exports/Robust_balance_candidate_2026-08-11.json" `
  --report v4_twin_279mm_baseline/gain_optimization_report.json
```

No robot was commanded, flashed, or written during this work.
