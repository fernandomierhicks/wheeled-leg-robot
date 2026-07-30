# Standing-up recovery (`STATE_STANDING_UP`)

Arm-time recovery sequence for arming while fallen, instead of requiring the
robot to already be balanced. Entered from STANDBY only when `standup_enable=1`
(default 0 — with it off, arming goes straight to RUNNING, byte-identical to
before this state existed). See `state_machine.md` for how STANDING_UP fits
into the overall FSM (transitions, safety-exit priority).

Never calls the RUNNING-mode LQR (`controlLoop_run()`): pitch is expected to
be large and outside the LQR's small-angle linearization at entry, so this
state codes its own minimal wheel-torque law inline (`on_standing_up()`,
`src/state_machine.cpp`), the same way `on_jumping()` codes its own hip logic
inline instead of reusing RUNNING's.

## Arm-time gate vs. in-progress divergence limit

Two independent pitch ranges, deliberately not the same params:

| | Checked | Params | Purpose |
|---|---|---|---|
| Entry gate | Once, at the moment STANDING_UP is requested (`stateMachine_request_running()`) | `standup_pitch_fwd/bwd` (default 0.6 rad, ~34°) | Refuse to even attempt recovery from too extreme a starting lean |
| Divergence limit | Every tick, all three phases, 50 ms debounced | `standup_div_fwd/bwd` (default 1.0 rad) | Abort a catch that's diverging *during* the attempt |

The divergence limit is deliberately looser than the entry gate: a saturated
catch can transiently swing past the angle it started from, so bounding it to
the same value as the entry gate would leave zero overshoot budget and make a
hard catch self-trip immediately. Divergence fires `FAULT_STANDUP_FAILED`
with no retry (distinct from an attempt simply timing out — see below).

Both are separate from `pitch_wd_fwd/bwd_ret/ext` (the normal RUNNING pitch
watchdog), which is not checked in this state at all — see "Bypassed
watchdogs" below.

## Phases (`StandupPhase`, `g_state.standup_state`)

**CROUCH** (`SU_CROUCH = 0`) — ramps both hips from their pose at entry to the
calibrated retracted target over `standup_crouch_time` seconds, at
`hip_running_kp/kd` — the same gains RUNNING uses, not a separate
standup-only stiffness, so the eventual RUNNING handoff is a hip *position*
step only, never also a stiffness step. Wheels commanded to zero throughout:
no wheel motion until the legs are in a known-good pose. Advances to RECOVER
unconditionally once the ramp timer elapses.

**RECOVER** (`SU_RECOVER = 1`) — hips held rigidly at the retracted target.
Wheel torque:

```
x0  = pitch - lqr_pitch_trim_ret
tau = standup_k_pitch * x0 + standup_k_rate * pitch_rate
```

clamped to `standup_torque_lim` then `MOTOR_TRQ_MAX`, sent symmetrically to
both wheels (no yaw/differential — straight push only). Uses
`lqr_pitch_trim_ret` (not zero) as the pitch target because legs are pinned
retracted for the whole sequence, so the retracted-anchor balance point is the
correct equilibrium throughout — the same one RUNNING's LQR regulates to
immediately after handoff. A law that targeted pitch=0 instead would have the
robot continuously accelerating to hold a lean the mechanism doesn't actually
sit at (the true zero-velocity equilibrium is off from vertical whenever the
CG isn't over the axle), and would still be rolling at the moment of capture.

**Capture**: `|x0| < standup_cap_pitch` and `|pitch_rate| < standup_cap_rate`,
continuously for `standup_cap_hold` seconds, then `s_su_captured` latches true
and the FSM transitions to RUNNING on the next tick. Centering the band on
`x0` rather than raw pitch is what makes "in-band" mean "actually settled",
not "passing through the trim angle while still rolling."

**Attempt timeout**: if `standup_timeout` seconds elapse in RECOVER without
capturing, and fewer than `standup_max_retries` retries remain, moves to
PAUSE. If retries are exhausted, faults `FAULT_STANDUP_FAILED` (failed to
converge — distinct from the divergence case above, which is a
wrong-direction response, not just a slow one).

**Dedicated wheel-runaway backup**: `|wheel vel| > 2 × standup_vel_limit`
faults `FAULT_WHEEL_RUNAWAY` immediately, independent of `wm_vel_limit` so
tuning one doesn't loosen the other's margin.

**PAUSE** (`SU_PAUSE = 2`) — wheels zeroed, hips still held retracted, for
`standup_retry_pause` seconds, then increments the attempt counter and
returns to RECOVER.

## Bypassed watchdogs

`on_standing_up()` never calls `controlLoop_run()`, so the pitch watchdog and
roll watchdog (both implemented inside it) do not run during STANDING_UP. The
only pitch bound in effect is the divergence limit above; there is no roll
bound at all. The dedicated wheel-runaway backup above is standup's own
equivalent of `controlLoop_run()`'s wheel-runaway watchdog, kept independent
so the two don't share a mistuned limit.

## Guards

- **Invalid hip calibration**: if `hm_limits_L/R.valid` is false on any tick,
  aborts immediately with `FAULT_STANDUP_FAILED` rather than commanding hips
  (`hip_cmd_to_setpoints(0.0)` silently returns pos=0, not "retracted", when
  the calibrated span is zero). Unlike `on_jumping()`'s equivalent guard
  (which just skips hip commands for the tick and keeps its LQR wheel-balance
  running regardless), there is no fallback control loop active here, so an
  invalid-limits abort has to be a full ESTOP-bound abort, not a skip.
- Motor feedback fault, IMU fault, and explicit ESTOP all take priority over
  everything above, per the same safety ordering as every other state (see
  `state_machine.md`).

## Handoff to RUNNING

`on_running()` skips the hip stiffness ramp-in when arriving from a captured
standup (`from_standing_up`) — the hips are already stiffly holding the
retracted position at `hip_running_kp/kd`, so re-ramping from 0 would loosen
them right when they need to hold. Combined with CROUCH/RECOVER/PAUSE using
those same gains throughout (rather than a separate standup-only kp/kd), the
hip command is continuous in gain across the handoff; only the *position*
target changes, to wherever `radio_hip_cmd` (CH3) is currently commanding.

## Telemetry

`g_state.standup_state` (`TelemetryPayload` V11) mirrors the current
`StandupPhase` (0=CROUCH, 1=RECOVER, 2=PAUSE) for GUI/log visibility.
