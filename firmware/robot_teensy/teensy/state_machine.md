# Teensy safety state machine

The numeric IDs and ordered transition list are frozen by
`test/state_machine_contract.json`; `software/gui/tests/test_state_machine_contract.py`
fails if this document's source implementation changes without an intentional
contract update.

Safety priority within every active state is: explicit ESTOP, motor feedback
fault, IMU fault, normal disarm/abort, then sequence completion or operator
requests.

| State | Normal exits | Safety exits |
| --- | --- | --- |
| STARTUP | readiness → STANDBY | ESTOP or startup failure → ESTOP |
| STANDBY | manual, calibration, arm → RUNNING/STANDING_UP | ESTOP or motor fault → ESTOP |
| MANUAL | operator exit or GUI timeout → STANDBY | ESTOP or motor fault → ESTOP |
| CALIBRATION | complete → STANDBY; radio stick-combo cancellation → DISARMING; GUI operator exit → STANDBY | ESTOP, motor fault, calibration failure → ESTOP |
| RUNNING | disarm → DISARMING; jump → JUMPING | ESTOP, motor fault, or stale/non-nominal IMU → ESTOP |
| JUMPING | disarm → DISARMING; complete → RUNNING | ESTOP, motor fault, or IMU fault → ESTOP |
| STANDING_UP | disarm → DISARMING; capture → RUNNING | ESTOP, motor fault, or IMU fault → ESTOP |
| DISARMING | running or calibration hip ramp complete → STANDBY | ESTOP, motor fault, or IMU fault → ESTOP |
| CMD_REJECT | one-second indication complete → STANDBY | ESTOP or motor fault → ESTOP |
| ESTOP | soft clear → STANDBY; reset → STARTUP | outputs remain in the ESTOP policy |

## JUMPING phases

JUMPING is entered only from RUNNING, through either a GUI/API
`SET_MODE(JUMPING)` or the CH6 radio switch's rising edge (SIMPLE live-tune mode
only), and only with `jump_enable` set. Its internal phase is reported as
`jump_state` in telemetry.

`stateMachine_request_jump()` also refuses to arm while the robot is already
moving hard: measured forward speed above `jmp_arm_fwd_ms`, backward speed
above `jmp_arm_bwd_ms`, `|imu_yaw_rate()|` above `jmp_arm_yaw_rate`, or
`|imu_roll()|` above `jmp_arm_roll` all reject the request (logged, no
CMD_REJECT transient). Forward speed gets more headroom than the other three
since some helps the launch. This is a one-time gate at request time, not
re-checked once JUMPING is entered — the pitch watchdog remains the live
in-flight guard.

| Phase | `jump_state` | Hip command |
| --- | --- | --- |
| `JP_CROUCH` | 0 | minimum-jerk position ramp to `jump_crouch_angle` at peak speed `jump_crouch_speed`, at `jump_kp`/`jump_kd` |
| `JP_EXTEND` | 1 | pure torque, `kp = 0`: `jump_torque_max × jump_effort`, onset at `jump_torque_rate`, tapered approaching `jump_extend_angle` (over `jump_ramp_down`) and by hip speed (`jump_omega_max`), hard-cut inside `jump_hs_margin` of the calibrated extended limit |
| `JP_RETRACT` | 2 | minimum-jerk ramp to the landing pose (`jump_retract_angle`, or the entry pose when it is negative) at peak speed `jump_retract_speed`, then hold there; gyro landing detection runs throughout |
| `JP_LANDING` | 3 | one-tick telemetry marker at detected contact; the normal RUNNING hip command takes over with handoff authority active |
| `JP_HANDOFF` | 4 | RUNNING controller with the temporary `jmp_handoff_*` LQR-gain, torque and wheel-speed overrides until capture or timeout |

Landing detection (`jump_landing.h`) starts at RETRACT entry and is gyro-only:
at least two fresh gyro-vector changes summing above `jump_land_gyro_imp` within
12 ms, blanked for `jump_land_min_air` after RETRACT starts. At contact the hip
rate limiter is seeded from the measured pose and its gain ramp completed, so
CH3 slews in from wherever the jump left the legs. If `jump_land_timeout`
expires with no detection the sequence enters HANDOFF anyway — uncaptured, so
the handoff wheel-speed limit still applies; this is logged, not a fault.

HANDOFF captures when trim-relative pitch is inside `jmp_handoff_pitch`, pitch
rate inside `jmp_handoff_rate`, and both wheels inside `wm_vel_limit`,
continuously for `jmp_handoff_hold_s`. If `jmp_handoff_timeout` expires first,
the overrides are dropped and it releases to RUNNING anyway — also not a fault.
Controller state is carried across HANDOFF→RUNNING.

The wheel side of the balance loop runs as in RUNNING except between detected
liftoff (a wheel departs its RETRACT-entry speed by > 5 turns/s) and detected
landing, when each wheel is held at its RETRACT-entry speed with gain
`jmp_air_whl_kp`. The velocity PI is frozen (integral and `theta_ref` held)
from RETRACT entry until HANDOFF captures or times out.
`controlLoop_wheel_vel_limit()` is state-scoped: `jmp_air_vel_lim` through
CROUCH/EXTEND/RETRACT, `jmp_handoff_vel_lim` through LANDING/HANDOFF. Both feed
the soft governor and the 2x runaway trip.

Phase *durations* are derived, not configured: `1.875 × travel / peak_speed`,
using the shared quintic minimum-jerk helpers in `standup_safety.h`. Angles are
hip extension measured from the retract switch (0 = at the switch, positive =
extended), so the same jump is the same motion from any ride height — which the
former `jump_crouch_time` could not be, since it specified a duration for a
distance that varied with wherever CH3 had left the legs.

A single exit: `jump_done()` requires a captured (or timed-out) HANDOFF. There
is no overall overrun fault — every phase timeout falls through to a normal
exit, and `FAULT_JUMP_TIMEOUT` (0x0F) is defined but never set. With
`jump_enable=0` or invalid calibration limits the sequence retires straight to
a captured handoff — an unarmed jump is a no-op, not a fault. The pitch
watchdog and the motor-feedback/IMU/ESTOP transitions in the table above are
what can stop a jump in progress.

Arming is admitted only from STANDBY, only with an IMU that is NOMINAL and
no more than 50 ms old, and only after the configured calibration/motor gates
pass. `CMD_PAYLOAD_V2` callers receive a correlated rejection result when a
guard fails. Repeated ESTOP while already in ESTOP is idempotent and does not
leave a stale event that can fire after reset.
