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
since some helps the launch, the same reasoning as `jump_nudge_fwd_vel` below.
This is a one-time gate at request time, not re-checked once JUMPING is
entered — the pitch watchdog (`pitch_watchdog_en`, unmasked through every
JUMPING phase, unlike STANDING_UP) remains the live in-flight guard.

| Phase | `jump_state` | Hip command |
| --- | --- | --- |
| `JP_CROUCH` | 0 | minimum-jerk position ramp to `jump_crouch_angle` at peak speed `jump_crouch_speed`, at `jump_kp`/`jump_kd` |
| `JP_EXTEND` | 1 | pure torque, `kp = 0`: `jump_torque_max × jump_effort`, onset at `jump_torque_rate`, tapered approaching `jump_extend_angle` (over `jump_ramp_down`) and by hip speed (`jump_omega_max`), hard-cut inside `jump_hs_margin` of the calibrated extended limit |
| `JP_RETRACT` | 2 | measured position/velocity-continuous 15 ms braking blend, then a minimum-jerk ramp to the landing pose (`jump_retract_angle`, or the entry pose when it is negative) at peak speed `jump_retract_speed`; predicted PD feedback is capped by `jump_retract_torque` |
| `JP_LANDING` | 3 | one explicit telemetry phase at detected contact; normal RUNNING hip/wheel control is already active with jump handoff authority |
| `JP_HANDOFF` | 4 | RUNNING controller with temporary `jmp_handoff_*` LQR gain, torque and wheel-speed overrides until capture |

During the final `jump_nudge_fwd_dur` seconds of CROUCH,
`jump_nudge_fwd_vel` is added to the pilot's live velocity request. It is an
offset, not a replacement: `effective velocity = sticks + nudge`.

Landing detection begins at RETRACT entry and is gyro-only. At least two fresh
gyro-vector changes whose magnitudes sum above `jump_land_gyro_imp` in 12 ms
declare impact. The detector keys on the IMU report timestamp, so a 400 Hz
sample held across two 500 Hz control ticks cannot be counted twice. Decisions
are blanked for `jump_land_min_air` to reject the launch impulse. The retired
`jump_air_accel_z` and `jump_land_accel_z` IDs remain protocol-compatible but
have no effect. If `jump_land_timeout` expires without gyro contact evidence,
that is not a fault: the ground-tuned wheel/balance loop has been
running unbroken since CROUCH regardless of jump phase, so a missed
*detection* is a bookkeeping gap, not a loss of control. The hip ramp is
reseeded from the legs' actual position and the sequence retires straight to
a captured HANDOFF, the same as an unarmed/invalid-limits jump. (Until
2026-08-12 an unconfirmed landing still raised
`FAULT_JUMP_TIMEOUT` here; retired once real jumps with large RETRACT-entry
hip velocities started tripping it, because the gentler braking-blend RETRACT
that fixed the wheel-runaway/false-gyro-landing problems also produces a
touchdown signature too soft for the tuned gyro threshold to reliably
catch — not an actual loss of control.)

At contact the hip rate limiter is seeded from measured pose and its gain ramp
is completed. LANDING/HANDOFF then use the same controller as RUNNING, with
temporary `jmp_handoff_kp_mul`, `jmp_handoff_kr_mul`,
`jmp_handoff_kv_mul`, `jmp_handoff_torque`, and `jmp_handoff_vel_lim`.
Capture requires trim-relative pitch inside `jmp_handoff_pitch`, pitch rate
inside `jmp_handoff_rate`, and both wheels inside the normal `wm_vel_limit`
continuously for `jmp_handoff_hold_s`. If `jmp_handoff_timeout` expires
without capture, that is not a fault either (as of 2026-08-12, same reasoning
as the landing-timeout case above): RUNNING with handoff authority has
already been driving pitch/wheels the whole time, so a slow convergence is
not a loss of control. The elevated handoff authority is dropped and plain
RUNNING limits take over regardless of whether the capture band was ever hit.
The controller state is preserved across HANDOFF→RUNNING either way.

The ground-tuned wheel loop still runs unchanged through CROUCH, EXTEND and
RETRACT. The detector creates the timing needed for a future dedicated
airborne reaction-wheel law, but the current firmware does not switch to one.

`controlLoop_wheel_vel_limit()` is state-scoped through all of this:
`STANDING_UP` gets `standup_vel_limit`, `LANDING`/`HANDOFF` get
`jmp_handoff_vel_lim` (tight — the wheel is back on the ground by then), and
`CROUCH`/`EXTEND`/`RETRACT` get `jmp_air_vel_lim` (loose — the wheels are
genuinely unloaded in the air). All three feed both the per-wheel soft torque
governor and the wheel-runaway watchdog's 2x trip point, so a real jump's
airborne wheel spin-up doesn't need to fit inside the plain RUNNING governor
the way it did before 2026-08-13 — that mismatch cost a real jump a
mid-RETRACT `FAULT_WHEEL_RUNAWAY` before landing could even be evaluated.

Phase *durations* are derived, not configured: `1.875 × travel / peak_speed`,
using the shared quintic minimum-jerk helpers in `standup_safety.h`. Angles are
hip extension measured from the retract switch (0 = at the switch, positive =
extended), so the same jump is the same motion from any ride height — which the
former `jump_crouch_time` could not be, since it specified a duration for a
distance that varied with wherever CH3 had left the legs.

A single exit: `jump_done()` requires a captured HANDOFF, and every phase
timeout (EXTEND's, RETRACT's landing detection, HANDOFF's capture) now falls
through to a normal, non-faulting path toward that rather than an ESTOP —
there is no jump-specific timeout fault left (`jump_overrun()` and its
derived overall deadline were removed 2026-08-12). `jump_enable=0` or invalid
calibration limits retire the same way, straight to a captured handoff — an
unarmed jump is a no-op, not a fault. `FAULT_JUMP_TIMEOUT` (0x0F) remains
defined in the protocol but nothing sets it anymore.
The pitch watchdog (unmasked throughout JUMPING, per above) and the hard
motor-feedback/IMU-fault/explicit-ESTOP transitions in the table at the top
are the only things that can still ESTOP a jump in progress.

Arming is admitted only from STANDBY, only with an IMU that is NOMINAL and
no more than 50 ms old, and only after the configured calibration/motor gates
pass. `CMD_PAYLOAD_V2` callers receive a correlated rejection result when a
guard fails. Repeated ESTOP while already in ESTOP is idempotent and does not
leave a stale event that can fire after reset.
