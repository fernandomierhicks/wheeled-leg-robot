# Standing-up recovery (`STATE_STANDING_UP`)

Arm-time settling window, instead of requiring the robot to already be balanced
at the instant CH10 goes high. Entered from STANDBY only when `standup_enable=1`
(default 0 — with it off, arming goes straight to RUNNING, byte-identical to
before this state existed). See `state_machine.md` for how STANDING_UP fits into
the overall FSM (transitions, safety-exit priority).

## What it actually is

**STANDING_UP runs the ordinary RUNNING control stack with the pitch watchdog
masked.** It is not a separate controller. `on_standing_up()` calls
`controlLoop_run()` — the same balance LQR, velocity PI, yaw PI, feedforward,
per-wheel soft velocity governor and wheel-runaway watchdog that RUNNING uses.

This replaced an earlier design that coded its own minimal wheel-torque law
inline (the way `on_jumping()` codes its own hip logic). Reusing the real
control loop is both simpler and safer:

- The catch and the steady state are the *same* controller, so the handoff to
  RUNNING is not a controller switch — nothing is rescheduled, retuned, or
  re-initialised at the moment the watchdog goes live.
- The inline law had no soft velocity governor, only a hard `2×` runaway fault.
  Under a near-constant catch torque the wheels would accelerate freely to the
  hard limit and ESTOP. `controlLoop_run()`'s per-wheel governor removes that
  failure mode for free.
- There is one set of gains to tune, not two.

The cost is scope: this recovers from a *lean*, not from flat on its back. The
entry gate below is what keeps the request inside the LQR's usable range.

## Phases (`StandupPhase`, `g_state.standup_state`)

The sequence is **CROUCH → STIFFEN → RECOVER**, with PAUSE only on a retry.
`SU_STIFFEN` is numbered `3`, out of sequence, so the telemetry codes already in
logs keep their meaning.

**CROUCH** (`SU_CROUCH = 0`) — the excursion. Ramps both hips from their pose at
entry to the calibrated retracted endpoint over `standup_crouch_time` seconds.
Wheels remain at zero: no wheel motion until the legs are in a measured
known-good pose.

The position ramp is a quintic minimum-jerk S-curve rather than a linear ramp.
Its commanded velocity and acceleration are zero at both endpoints, avoiding
the former instantaneous `0 -> dq -> 0` velocity steps while preserving the
configured total time.

Hip stiffness during the excursion is `standup_crouch_kpf` × `hip_running_kp`,
**faded in from zero on the same minimum-jerk profile as the position**. The hip
feedforward is scaled by the same factor, exactly as `control_loop.cpp`'s arm-in
`ramp_alpha` scales the pair; `hip_running_kd` is at full value throughout, also
matching that ramp — damping is what stops a softly-held leg oscillating, so it
is never the thing that gets faded. Because the commanded motion and the
stiffness that drives it both start at exactly zero, arming applies no torque
step whatever pose the legs are in.

> **The ramp must start from the *clamped* pose.** Every hip command is clamped
> to the calibrated span on the way out (`hip_motors_set_setpoint_*`), so a ramp
> that starts outside the span is not a ramp at all: its first command is
> already the clamped endpoint and the whole excursion collapses into a step.
> This is the normal case, not a corner case — calibration parks the leg *on*
> the retract switch, one `calib_backoff_rad` beyond the retracted end of the
> span. With a 15° backoff and `hip_running_kp = 25`, arming from the parked
> pose used to snap the hips through 15° at 6.5 N·m on the first tick, whatever
> `standup_crouch_time` said, followed by a long do-nothing wait for the ramp
> timer to expire. `on_standing_up()` now snapshots
> `hip_motors_clamp_to_limits(hm_*.pos_rad, ...)`.

The target is normalized `t=0`, which `define_limits()` (`calibration.cpp`)
defines at exactly `calib_backoff_rad` away from each retract switch. Therefore
a 15° backoff makes the stand-up target 15° from each switch — the leg does not
crouch to the hard stop, it crouches to one backoff short of it. There is no
separate stand-up hip-height parameter. Set `calib_backoff_rad` before running
calibration; changing it afterward requires recalibration because the calibrated
hip limits contain the endpoint used by `hip_cmd_to_setpoints(0)`.

The ramp time is only a minimum. CROUCH ends when the ramp has run its full
length **and** both hips have been moving no faster than 0.2 rad/s continuously
for 100 ms. Position is deliberately *not* checked here: a proportional hold at
`standup_crouch_kpf` of the running stiffness sags by
`(hold torque − tff)/kp`, which is far outside the 2° band, so "the legs got
there" is only a meaningful question once the gains are full. If the hips are
still moving 2 seconds after the ramp, stand-up faults `FAULT_STANDUP_FAILED`.

**STIFFEN** (`SU_STIFFEN = 3`) — hips hold the calibrated retracted endpoint
while `kp` and the hip feedforward ramp linearly from `standup_crouch_kpf` to
their full `hip_running_*` values over `standup_stiff_time` seconds. This is
also what pulls out the sag the softer crouch hold left.

STIFFEN ends — and the LQR is engaged — only when the ramp has completed **and**
both measured hips are, at full stiffness:

- within `standup_hip_pos_tol()` of their respective targets, and
- moving no faster than 0.2 rad/s,

continuously for 100 ms. If that gate is not met within 2 seconds after the
ramp, stand-up faults `FAULT_STANDUP_FAILED` (with the per-axis position and
velocity errors, and the tolerance in force, logged) instead of balancing from
an unknown leg pose. At the
successful gate the normal arm-in hip-gain ramp is explicitly marked complete,
so RECOVER and the RUNNING handoff continue from full stiffness rather than
re-loosening a verified pose the moment the wheel LQR comes on. These limits are
fixed safety criteria, not additional parameters.

> **The position tolerance is not a fixed 2°, and can't be.** The hip hold is
> proportional plus a constant feedforward, so it rests where
> `kp·(target − pos) + tff` balances the external load — offset from the target
> by `(tff − load)/kp`. Both ends of that range are legitimate during STIFFEN:
> off the ground the legs are near unloaded and the offset is the full `tff/kp`,
> while standing on the wheels the load is the one `hip_running_tff_ret` was
> tuned to cancel and the offset is ~0. At `tff_ret = −2.5 N·m` and
> `hip_running_kp = 25` that spread is **0.1 rad, three times the 2° band**, so
> a fixed 2° gate faults every bench arm at exactly `err ≈ −0.091 rad` no matter
> how long it waits. `standup_hip_pos_tol()` therefore allows `|tff|/kp` on top
> of the 2°, capped at `STANDUP_HIP_POS_TOL_MAX_RAD` (0.2 rad) so a badly scaled
> gain pair fails the gate rather than disabling it. A leg jammed part-way
> through the excursion is tens of degrees out and still fails.

**RECOVER** (`SU_RECOVER = 1`) — `controlLoop_run()` every tick. Hips are pinned
at the calibrated retracted endpoint by the control-loop hip override, so CH3
cannot walk the legs off the crouch height mid-catch.

**Capture**: `|pitch − applied_pitch_trim| < standup_cap_pitch`,
`|pitch_rate| < standup_cap_rate`, **and both wheels inside `wm_vel_limit`**,
continuously for `standup_cap_hold` seconds, then `s_su_captured` latches and
the FSM transitions to RUNNING on the next tick. The wheel-speed condition
exists because the catch may be running under the higher `standup_vel_limit`:
capturing above RUNNING's own governor hands off a wheel whose torque gets
zeroed on the first RUNNING tick, and above `2×` it, one that trips the runaway
watchdog ~50 ms later. Centering the band on the trim rather than raw pitch is what makes
"in-band" mean "actually settled" rather than "passing through the trim angle
while still rolling."

The trim used is `g_state.applied_pitch_trim` — whatever `controlLoop_run()`
applied on this same tick — not `lqr_pitch_trim_ret` read directly. With
`standup_ret_gains=1` the two are identical; with it 0 they are not, and
reading the applied value keeps the capture band centred on the equilibrium the
LQR is actually regulating to.

> `standup_cap_pitch` must stay well inside
> `pitch_wd_bwd_ret − |lqr_pitch_trim_ret|`. At its former default of 0.12 rad,
> with `trim_ret = −0.14` and `pitch_wd_bwd_ret = 0.262`, a capture at the
> backward edge of the band handed off **0.002 rad** from the backward
> watchdog trip angle — a successful catch immediately followed by
> `FAULT_PITCH_WATCHDOG` in RUNNING. Default is now 0.05.

**Attempt timeout**: if `standup_timeout` seconds elapse in RECOVER without
capturing, and fewer than `standup_max_retries` retries remain, moves to PAUSE.
If retries are exhausted, faults `FAULT_STANDUP_FAILED`.

**PAUSE** (`SU_PAUSE = 2`) — wheels zeroed, hips still held at the calibrated
retracted endpoint for `standup_retry_pause` seconds at full stiffness and full
feedforward, then increments the attempt counter and returns to RECOVER. Both
have to stay on: the hips were rigid entering the pause, and dropping either one
would sag the legs and hand RECOVER a different pose than the one it left.

> The retry/PAUSE structure is inherited from the ballistic-catch design, where
> cutting torque and re-trying from rest made sense. With the LQR running
> continuously it is questionable — zeroing the wheels for 0.3 s mid-balance is
> a disturbance, not a reset. **`standup_max_retries = 0` is the recommended
> setting**, which makes a timeout fault directly and never enters PAUSE. The
> path is kept working rather than deleted.

## Gain scheduling: `standup_ret_gains`

With `standup_ret_gains=1` (default), the gain-schedule alpha is pinned to 0 —
the retracted anchor that matches the fixed hip target — for the whole of
STANDING_UP. That means `lqr_k_pitch_ret`, `lqr_k_rate_ret`,
`lqr_pitch_trim_ret`, `lqr_barrier_th_ret` and `pitch_wd_*_ret` all govern the
catch.

This is the same mechanism as `alpha_force_ret_en`, scoped to STANDING_UP
instead of applied globally.

> **Handoff caveat:** pinning means alpha steps `0 → measured` at the RUNNING
> handoff, which steps the applied pitch trim with it. For measured alpha `h`,
> the step is `h·(trim_ext − trim_ret) + lqr_trim_curve·h·(1−h)`. If that proves
> to matter on the bench, set `standup_ret_gains = 0` and let standup schedule
> on measured leg height like RUNNING does.

## Pitch bounds — three different ones, deliberately

| | Checked | Params | Purpose |
|---|---|---|---|
| Entry gate | Once, at the moment STANDING_UP is requested (`stateMachine_request_running()`) | `standup_pitch_min/max` (-0.6/+0.6 rad, ~-34°/+34°) | Refuse to attempt recovery from a lean outside the LQR's usable range |
| Divergence limit | Every tick, all phases, 50 ms debounced | `standup_div_fwd` (1.0 rad), `standup_div_bwd` (0.5 rad) | Abort a catch that's diverging *during* the attempt |
| Pitch watchdog | **Not checked in this state** | `pitch_wd_*_ret/ext` | Re-arms on the RUNNING handoff |

The forward divergence limit is looser than `standup_pitch_max`, giving a catch
overshoot budget instead of self-tripping the instant it starts. The divergence
range is **asymmetric**: `standup_div_bwd` is intentionally tighter than the
magnitude of `standup_pitch_min` because the leg linkage reaches the ground on a
backward lean well before the forward limit, and because it is the only pitch
bound in effect while the watchdog is masked.

## Watchdogs

`controlLoop_run()` gates the pitch watchdog on
`g_state.state != STATE_STANDING_UP` — the state itself is the condition, so
the suppression cannot be left switched on by accident the way a param could.
The `else` branch also holds `s_pitch_fault_start_ms` at 0 throughout, so the
RUNNING handoff always starts a fresh 200 ms window rather than inheriting a
partly-elapsed one.

Everything else in `controlLoop_run()` stays live, including the **roll
watchdog** and the **wheel-runaway watchdog** — both of which the old inline
design bypassed entirely.

## Catch authority: `standup_torque_lim` / `standup_vel_limit`

The catch needs more wheel torque and more wheel *speed* than the balance tune
wants in steady state: it has to drive the wheels out from under the CG hard
enough to swing the body up, which is a one-off manoeuvre, not a regulation
problem. Both params replace their RUNNING counterparts for the duration of the
state (`standup_scoped()`, `control_loop.cpp`); **`0` means "use the RUNNING
value"**, so both are inert until deliberately set.

| | Replaces | Applies to |
|---|---|---|
| `standup_torque_lim` | `lqr_torque_limit` | `tau_sym` clamp and the mixed per-wheel clamp. `MOTOR_TRQ_MAX` (7 N·m) is still the hard ceiling |
| `standup_vel_limit` | `wm_vel_limit` | the per-wheel soft governor **and** the runaway watchdog's `2×` trip |

> `wm_vel_limit` is usually what ends the catch early, and not gently: the soft
> governor sets that wheel's torque to **exactly zero** the moment it passes the
> limit in the driven direction. At the 3.0 turns/s default that is 1.06 m/s —
> reached almost immediately under a hard backward drive, after which the tilt-up
> simply stalls. Raising `standup_vel_limit` is the single biggest lever on how
> energetic the catch is.
>
> The runaway trip follows whichever limit is in force, deliberately: tripping at
> `2 ×` a RUNNING governor the wheels are not being held to would ESTOP every
> energetic catch.

These are the params originally reserved for the removed inline catch law. There
is still no standup-specific *gain* set — `standup_k_pitch` and `standup_k_rate`
remain unread, and `standup_ret_gains` selects which end of the existing gain
schedule applies, not a separate tune.

## Guards

- **Invalid hip calibration**: if `hm_limits_L/R.valid` is false on any tick,
  aborts immediately with `FAULT_STANDUP_FAILED` rather than commanding hips
  (`hip_cmd_to_setpoints()` silently returns pos=0, not the requested height,
  when the calibrated span is zero).
- **Stale backoff calibration**: if the calibrated retracted endpoints no
  longer match the current `calib_backoff_rad`, aborts with
  `FAULT_STANDUP_FAILED` and asks for recalibration instead of silently moving
  to the old backoff distance.
- Motor feedback fault, IMU fault, and explicit ESTOP all take priority over
  everything above, per the same safety ordering as every other state.

## Handoff to RUNNING

`on_running()` treats arrival from a captured standup as a *continuation*, not
an arm:

- **`controlLoop_reset()` is skipped.** The loop has been running and settling
  for the last few hundred ms; resetting would discard a converged velocity
  integral and re-seed the hip rate-limiter from CH3 — a torque and leg-position
  step at the exact moment the watchdog goes live.
- **The hip override is released instead.** Because the override writes the
  rate-limiter shadow (rather than bypassing it), CH3 slews in from the pinned
  height at the normal `hip_cmd_rate_lim`, with no position step.
- The hip stiffness ramp-in is skipped (`from_standing_up`), as before — the
  hips are already stiffly holding at `hip_running_kp/kd`.

## Params no longer read

`standup_k_pitch` and `standup_k_rate` belonged to the inline torque law and are
**not read** by the current design — the scheduled `lqr_k_pitch/k_rate` anchors
govern the catch. They are retained in the schema (not deleted) so a future
standup-specific control law can pick them back up. This is recorded in each
one's schema description.

`standup_torque_lim` and `standup_vel_limit` were in this list until they were
revived as the STANDING_UP-scoped authority overrides documented above.

## Telemetry

`g_state.standup_state` (`TelemetryPayload` V11) mirrors the current
`StandupPhase` (0=CROUCH, 1=RECOVER, 2=PAUSE, 3=STIFFEN) for GUI/log
visibility.
