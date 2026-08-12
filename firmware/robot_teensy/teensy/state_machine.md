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

| Phase | `jump_state` | Hip command |
| --- | --- | --- |
| `JP_CROUCH` | 0 | minimum-jerk position ramp to `jump_crouch_angle` at peak speed `jump_crouch_speed`, at `jump_kp`/`jump_kd` |
| `JP_EXTEND` | 1 | pure torque, `kp = 0`: `jump_torque_max × jump_effort`, onset at `jump_torque_rate`, tapered approaching `jump_extend_angle` (over `jump_ramp_down`) and by hip speed (`jump_omega_max`), hard-cut inside `jump_hs_margin` of the calibrated extended limit |
| `JP_RETRACT` | 2 | minimum-jerk ramp to the landing pose (`jump_retract_angle`, or the entry pose when it is negative) at peak speed `jump_retract_speed` |
| `JP_DONE` | 3 | stiff hold at the landing pose until `JUMP_SETTLE_MS` (300 ms) elapses |

On a real hop the robot is still airborne through RETRACT and touches down
partway into the `JP_DONE` hold, so `jump_retract_angle` is the pose it lands
on. The RUNNING handoff re-seeds the hip rate limiter from the measured pose and
forces the gain ramp complete, so CH3 slews in from wherever the jump left the
legs instead of stepping them to the stick position.

The wheel side of the balance loop runs unchanged in every phase, exactly as in
RUNNING. There is no flight or landing detection.

Phase *durations* are derived, not configured: `1.875 × travel / peak_speed`,
using the shared quintic minimum-jerk helpers in `standup_safety.h`. Angles are
hip extension measured from the retract switch (0 = at the switch, positive =
extended), so the same jump is the same motion from any ride height — which the
former `jump_crouch_time` could not be, since it specified a duration for a
distance that varied with wherever CH3 had left the legs.

Two independent exits. `jump_done()` requires `JP_DONE` *and* the settle hold;
`jump_overrun()` raises `FAULT_JUMP_TIMEOUT` if the phase machine has not reached
`JP_DONE` by a deadline derived at entry from the live phase params. The overrun
transition is registered first, so it always wins the race. With `jump_enable=0`
or invalid calibration limits the sequence retires straight to `JP_DONE` and
exits normally — an unarmed jump is a no-op, not a fault.

Arming is admitted only from STANDBY, only with an IMU that is NOMINAL and
no more than 50 ms old, and only after the configured calibration/motor gates
pass. `CMD_PAYLOAD_V2` callers receive a correlated rejection result when a
guard fails. Repeated ESTOP while already in ESTOP is idempotent and does not
leave a stale event that can fire after reset.
