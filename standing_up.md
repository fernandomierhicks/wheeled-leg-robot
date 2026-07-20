# New state: STANDING_UP (between STANDBY and RUNNING)

## Context

Arming today (`STANDBY` → `RUNNING`) assumes the robot is already upright near
pitch = 0 and hands straight off to the small-angle balance LQR
(`control_loop.cpp`). If the robot is lying fallen over (large pitch, forward
or backward) when the user arms it, there's no recovery path today — it just
arms into a controller that isn't valid at that angle.

We're adding a new state, `STANDING_UP`, that sits between `STANDBY` and
`RUNNING`. On arm, it first retracts the legs to a known pose, then drives an
energetic, high-torque, saturated wheel push to catch/right the robot and
bring pitch back within a band the tuned small-angle LQR can take over from,
then hands off to `RUNNING`. If the robot's starting pitch is beyond what's
physically recoverable (or it can't converge after retrying), it's rejected
or faulted rather than attempting something unsafe.

**Physics basis** (per the user, confirmed against the codebase's own sign
conventions): the robot's CG stays above the wheel axle in all fall positions
— it physically can't flip past ~90° before legs/body hit the ground. That
means the direction to push the wheels is the same sign throughout the whole
fallen range (monotonic with `sin(pitch)`) — no swing-up/pumping needed, just
a saturated, much-larger-gain version of the same "push the wheel base under
the falling CG" principle the LQR already uses at small angles. This is
verified directly from the existing code: `control_loop.cpp`'s LQR term and
the FF2 gravity-compensation term (`tau_ff2 = ff2_alpha*M_BODY*GRAVITY*l_eff*
sin(pitch)`, added to wheel torque) both apply **positive** wheel torque for
**positive** (forward) pitch — i.e. push the wheels forward, same direction as
the lean, to move the contact patch back under the CG. The recovery law
reuses this exact, already-validated sign convention rather than inventing a
new one (lower risk of a sign-flip bug that would fling the robot over
harder), just with independently-tunable, much larger gains and torque limit.

**Decisions already made (do not re-litigate):**
1. Legs fully retract to the stowed position **first**, before any wheel
   motion — gives a fixed, known effective pendulum length (`L_EFF_RET`)
   during the energetic push, and a repeatable known starting pose.
2. A new dedicated fault code `FAULT_STANDUP_FAILED` is added (not reusing
   `FAULT_PITCH_WATCHDOG`) for standup denial/failure, with its own GUI/ESP32/
   README mirror updates — same propagation discipline the codebase already
   uses for every other fault code.

---

## State machine mechanics (for context on the design below)

The state machine uses a small Arduino `StateMachine`/`State` library
(`state_machine.cpp`). Each tick: the current state's `on_X()` runs
unconditionally, then its registered transition conditions are evaluated in
registration order — first one returning `true` wins, updating `currentState`
for the next tick. There's no separate onEnter/onExit — each `on_X()`
self-detects entry via `bool entering = (g_state.state != STATE_X)`, and
"exit" side effects are coded in the *next* state by checking
`bool from_X = (g_state.state == STATE_X)` before overwriting `g_state.state`.

`STATE_STANDING_UP = 8` (append, don't renumber — every existing precedent,
e.g. `JUMPING=7`, appends at the next free integer regardless of graph
position; renumbering would break every hardcoded mirror across ESP32/GUI/
`.wlog` history for no benefit).

The closest existing precedent for a self-contained, time-phased motion
sequence living entirely inside one state's `on_X()` (not delegated to
`control_loop.cpp`) is `on_jumping()` (`state_machine.cpp:253-376`) — its own
local phase enum, a phase-start timestamp, hip position ramps via
`hip_motors_set_setpoint_L/R(pos, vel, kp, kd, tff)`, gated by a master-enable
param defaulting to 0. **Difference from JUMPING**: `on_jumping()` calls the
full `controlLoop_run()` every tick because jumping happens from small pitch
where the linear LQR is still valid. `STANDING_UP` must **not** call
`controlLoop_run()` during its energetic phase — pitch is large/outside the
linearization region — so it needs its own minimal wheel-torque law, coded
inline in `state_machine.cpp` the same way `on_jumping()`'s hip logic is
(control_loop.cpp stays untouched except one constant move, see below).

---

## 1. `firmware/robot_teensy/teensy/src/robot_state.h`

- Add `STATE_STANDING_UP = 8,` to `RobotStateEnum` (after `STATE_JUMPING = 7`).
  (This enum lives here, not in `comm_protocol.h`.)
- Add `uint8_t standup_state;  // Standup FSM phase — 0 until then` to
  `RobotState`, next to `jump_state`.
- Update the file's top `// MIRROR:` comment to list the new GUI dicts below.

## 2. `firmware/robot_teensy/teensy/src/control_loop.h` / `control_loop.cpp`

- Move `constexpr float MOTOR_TRQ_MAX = 7.0f;` out of `control_loop.cpp`
  (currently a `static constexpr` at line 37) into `control_loop.h` as a
  plain header `constexpr`, so `state_machine.cpp` (which already includes
  `control_loop.h`) can reuse the same hard per-wheel clamp for the standup
  recovery torque instead of duplicating the literal `7.0f` a second time.
  `control_loop.cpp` uses the header's copy instead of its own. No other
  changes to this file — the recovery law needs none of `M_BODY`/`GRAVITY`/
  `WHEEL_R`/`L_EFF_RET` as literal inputs (those were only physics
  justification for retracting legs first).

## 3. New params — `param_ids.h` + `param_registry.cpp`

`GROUP_CONTROL` (0x04), IDs `0x042C`–`0x043B` (16 params; current group max in
use is `0x042B`, so this range is free). Add each as a row in
`param_registry.cpp`'s `g_params[]` right after the existing
`PARAM_GUI_MOTION_CTRL_EN` entry, same designated-initializer style as the
`PARAM_JUMP_*` block. Physical-torque-output gains start at **0** (must be
tuned up on the bench), matching the `PARAM_JUMP_ENABLE`/`PARAM_JUMP_TORQUE_MAX`
"off/inert until tested" convention.

| ID | Name | Default | Min | Max | Persist | Purpose |
|---|---|---|---|---|---|---|
| 0x042C | `standup_enable` | 0 | 0 | 1 | Y | Master gate. 0 = arm goes STANDBY→RUNNING directly (today's exact behavior, byte-identical, see §4). |
| 0x042D | `standup_max_pitch_fwd_rad` | 0.6 | 0.0 | 1.4 | Y | Arm-time gate: max forward pitch considered recoverable (~34° to start; raise once bench-validated). |
| 0x042E | `standup_max_pitch_bwd_rad` | 0.6 | 0.0 | 1.4 | Y | Same, backward. |
| 0x042F | `standup_crouch_kp` | 80.0 | 0.0 | 500.0 | Y | Hip MIT position gain during CROUCH ramp + held through RECOVER/PAUSE (mirrors `PARAM_JUMP_KP`). |
| 0x0430 | `standup_crouch_kd` | 1.0 | 0.0 | 5.0 | Y | Hip MIT damping, same phases (mirrors `PARAM_JUMP_KD`). |
| 0x0431 | `standup_crouch_time_s` | 0.30 | 0.05 | 2.0 | Y | Fixed-time CROUCH ramp duration (same convention as `PARAM_JUMP_CROUCH_TIME_S`). |
| 0x0432 | `standup_k_pitch` | 0.0 | 0.0 | 60.0 | Y | Recovery P gain on pitch [N·m/rad], same positive-sign convention as the existing LQR/FF2 terms. Starts inert. |
| 0x0433 | `standup_k_rate` | 0.0 | 0.0 | 15.0 | Y | Recovery D gain on pitch rate. Starts inert. |
| 0x0434 | `standup_torque_limit` | 0.0 | 0.0 | 7.0 | Y | `\|tau_recover\|` clamp, applied before the hard `MOTOR_TRQ_MAX` clamp. Starts inert; max = `MOTOR_TRQ_MAX`. |
| 0x0435 | `standup_wheel_vel_limit_turns_s` | 3.0 | 1.0 | 20.0 | Y | Dedicated runaway-backup baseline (trip at ×2.0, same pattern/constant as the existing `PARAM_WHEEL_VEL_LIMIT_TURNS_S` watchdog) — kept independent so tuning one doesn't loosen the other's safety margin. |
| 0x0436 | `standup_capture_pitch_rad` | 0.12 | 0.02 | 0.4 | Y | \|pitch\| below this (+ rate below next) for the hold time ⇒ "good enough for LQR". Must stay well under `PITCH_WATCHDOG_RAD` (50°) or a successful catch could immediately re-trip that watchdog on RUNNING entry. |
| 0x0437 | `standup_capture_rate_rads` | 1.0 | 0.1 | 5.0 | Y | \|pitch_rate\| capture threshold. |
| 0x0438 | `standup_capture_hold_s` | 0.15 | 0.02 | 1.0 | Y | Continuous in-band duration required before handoff (filters a single noisy in/out crossing). |
| 0x0439 | `standup_attempt_timeout_s` | 1.5 | 0.2 | 5.0 | Y | Max time in one RECOVER attempt before declaring it failed-to-converge (not diverged — see below). |
| 0x043A | `standup_max_retries` | 2 | 0 | 10 | Y | Retry attempts after the first (total attempts = this + 1); int-as-float, mirrors `PARAM_CALIB_STALL_TICKS` convention. |
| 0x043B | `standup_retry_pause_s` | 0.3 | 0.0 | 3.0 | Y | Wheels-off settle time between retry attempts. |

Companion GUI edit (required by the header comment in both files):
`software/gui/tabs/params_tab.py` — add
`(range(0x042C, 0x043C), 0x04, "Standing Up")` to `_SUBGROUPS`, a color entry
in `_SUBGROUP_COLORS`, and all 16 `(name, description)` pairs to
`_PARAM_DEFS` (names verbatim from the registry, descriptions from the
`param_ids.h` comments) — same pattern as the existing `"Jump"` subgroup.

## 4. `state_machine.cpp` — the core change

**New module statics** (mirrors the existing `s_jp_*` naming/style):

```cpp
typedef enum : uint8_t { SU_CROUCH = 0, SU_RECOVER = 1, SU_PAUSE = 2 } StandupPhase;
static StandupPhase s_su_phase             = SU_CROUCH;
static uint32_t     s_su_phase_ms          = 0;
static float        s_su_nom_L, s_su_nom_R;       // hip pos snapshot at entry
static float        s_su_ret_L, s_su_ret_R;       // calibrated retracted target
static uint8_t      s_su_attempt           = 0;   // 1-based RECOVER attempt counter
static uint32_t     s_su_capture_since_ms  = 0;   // 0 = not currently in-band
static bool         s_su_captured          = false;
static State*       S_STANDING_UP;                // alongside the other State* ptrs
```

**`on_standing_up()`** — styled after `on_jumping()`, entirely self-contained,
never calls `controlLoop_run()`:

- *Entry* (`entering = g_state.state != STATE_STANDING_UP`): log
  `-> STANDING_UP`; reset `s_su_phase=SU_CROUCH`, `s_su_phase_ms=millis()`,
  `s_su_attempt=0`, `s_su_captured=false`, `s_su_capture_since_ms=0`; snapshot
  `s_su_nom_L/R = hm_L/R.pos_rad`; `hip_cmd_to_setpoints(0.0f, &s_su_ret_L,
  &s_su_ret_R)`; `wheel_motors_set_mode(WheelMode::TORQUE)`.
- Every tick: `g_state.standup_state = (uint8_t)s_su_phase;`
- **`SU_CROUCH`**: linear-interpolate hips from `s_su_nom_*` to `s_su_ret_*`
  over `PARAM_STANDUP_CROUCH_TIME_S` using `PARAM_STANDUP_CROUCH_KP/KD`
  (identical structure to `JP_CROUCH`, `state_machine.cpp:291-303`);
  `wheel_motors_send(0.0f, 0.0f)` — no wheel motion until legs are known-good.
  On `elapsed >= crouch_time`: → `SU_RECOVER`, reset phase timer,
  `s_su_attempt = 1`.
- **`SU_RECOVER`**:
  1. Hold hips at `s_su_ret_L/R` with `PARAM_STANDUP_CROUCH_KP/KD`.
  2. Runaway hard-backup, same pattern as `control_loop.cpp:197-203` but using
     `PARAM_STANDUP_WHEEL_VEL_LIMIT_TURNS_S * 2.0f`: on trip,
     `g_state.fault_code = FAULT_WHEEL_RUNAWAY` (reuse — it's literally the
     same condition/meaning), `stateMachine_request_estop(); return;`.
  3. `tau = clamp(PARAM_STANDUP_K_PITCH*pitch + PARAM_STANDUP_K_RATE*pitch_rate,
     ±PARAM_STANDUP_TORQUE_LIMIT)`, then hard-clamp to `±MOTOR_TRQ_MAX`.
  4. `wheel_motors_send(tau, tau)` — symmetric, no yaw/differential (straight
     push only). Mirror into telemetry-visible fields for bench-tuning
     continuity: `g_state.whl_tau_l = g_state.whl_tau_r = g_state.tau_sym = tau;`
     (`tau_sym` etc. would otherwise sit frozen at stale values the whole time
     since `controlLoop_run()` never executes here, which would look like a
     bug on the Controllers/Raw Data tabs while wheels visibly spin).
  5. **Divergence check** (no retry — see rationale below): if
     `pitch > PARAM_STANDUP_MAX_PITCH_FWD_RAD || pitch < -PARAM_STANDUP_MAX_PITCH_BWD_RAD`
     → `g_state.fault_code = FAULT_STANDUP_FAILED; stateMachine_request_estop(); return;`
  6. **Capture check**: `in_band = |pitch| < PARAM_STANDUP_CAPTURE_PITCH_RAD &&
     |pitch_rate| < PARAM_STANDUP_CAPTURE_RATE_RADS`; track
     `s_su_capture_since_ms` the same way the existing pitch watchdog tracks
     its debounce (`control_loop.cpp:184-195`); set `s_su_captured = true`
     once held continuously for `PARAM_STANDUP_CAPTURE_HOLD_S`.
  7. **Attempt timeout**: if `elapsed_in_phase >= PARAM_STANDUP_ATTEMPT_TIMEOUT_S`
     and not captured: if `s_su_attempt >= PARAM_STANDUP_MAX_RETRIES + 1` →
     retries exhausted, `g_state.fault_code = FAULT_STANDUP_FAILED;
     stateMachine_request_estop(); return;`; else → `SU_PAUSE`, reset phase
     timer.
- **`SU_PAUSE`**: hold hips as above; `wheel_motors_send(0,0)`; still run the
  divergence check (step 5) — the robot can still be falling with wheels off.
  On `elapsed >= PARAM_STANDUP_RETRY_PAUSE_S`: `s_su_attempt++` → back to
  `SU_RECOVER`, reset phase timer.

  *Why divergence never retries*: if pitch grows **past** the recoverable
  range gate mid-attempt (not just fails to shrink), that's a wrong-direction
  or unstable response — retrying with the same gains is unlikely to help and
  could compound. Only "stayed within range but didn't converge in time"
  retries; "went outside the range" faults immediately.

`standup_captured()` transition condition is a thin wrapper:
`static bool standup_captured() { return s_su_captured; }` (same style as
`calibration_done_fn()`). No separate `standup_failed()` condition function is
needed — every failure path above sets `fault_code` and calls
`stateMachine_request_estop()` directly, which the *already-registered*
`req_estop` boilerplate transition (first in every state's list, evaluated in
the same `execute()` call) picks up on the same tick — exactly the idiom the
existing pitch watchdog / wheel runaway / `motor_feedback_fault` already use.

**`req_running()` split** — this is what keeps the disabled path byte-for-byte
identical to today with zero added tick latency:

```cpp
static bool req_running_checks_common() {
    // exact body of today's req_running(), minus the s_req_running clear
    // and the final `return true`
}
static bool req_running_to_standing_up() {
    if (!s_req_running) return false;
    if (param_get(PARAM_STANDUP_ENABLE) < 0.5f) return false;  // let _direct handle it
    s_req_running = false;
    if (!req_running_checks_common()) return false;
    float pitch = g_state.pitch_rad;
    if (pitch > param_get(PARAM_STANDUP_MAX_PITCH_FWD_RAD) ||
        pitch < -param_get(PARAM_STANDUP_MAX_PITCH_BWD_RAD)) {
        comm_log(LOG_LEVEL_WARN, "Standing-up denied: pitch %.3f rad outside recoverable range", pitch);
        stateMachine_request_cmd_reject();   // NOT a fault_code — see below
        return false;
    }
    return true;
}
static bool req_running_direct() {
    if (!s_req_running) return false;
    if (param_get(PARAM_STANDUP_ENABLE) >= 0.5f) return false;  // handled above
    s_req_running = false;
    return req_running_checks_common();
}
```

*Why the arm-time denial doesn't set `fault_code`*: `comm_protocol.h`
documents `fault_code` as "non-zero only when `robot_state == STATE_ESTOP`".
Every existing `req_running()` denial (IMU disabled, uncalibrated, motor
disabled) just logs a warning and routes through `CMD_REJECT`→`STANDBY` with
no fault code — the pitch-range denial follows the same pattern for
consistency. `FAULT_STANDUP_FAILED` is reserved for the ESTOP path: an
attempt that was actually in progress and failed/diverged/exhausted retries.

**Transition registration**, `stateMachine_init()`:

```cpp
S_STANDING_UP = sm.addState(on_standing_up);   // alongside the other addState calls
...
S_STANDBY->addTransition(req_estop,                  S_ESTOP);
S_STANDBY->addTransition(motor_feedback_fault,       S_ESTOP);
S_STANDBY->addTransition(req_manual,                 S_MANUAL);
S_STANDBY->addTransition(req_calibration,            S_CALIBRATION);
S_STANDBY->addTransition(req_running_to_standing_up, S_STANDING_UP);  // NEW
S_STANDBY->addTransition(req_running_direct,         S_RUNNING);      // replaces old req_running line
S_STANDBY->addTransition(req_cmd_reject,             S_CMD_REJECT);

S_STANDING_UP->addTransition(req_estop,            S_ESTOP);          // boilerplate, every state gets this
S_STANDING_UP->addTransition(motor_feedback_fault, S_ESTOP);          // boilerplate
S_STANDING_UP->addTransition(running_imu_fault,    S_ESTOP);          // live pitch is load-bearing here
S_STANDING_UP->addTransition(standup_captured,     S_RUNNING);
```

## 5. `on_running()` — hip-ramp skip fix

`on_running()` already skips `controlLoop_reset_hip_ramp()` when arriving
`from_jumping`, with the comment: "hips are already stiffly holding the
post-jump position; re-ramping kp from 0 here would loosen them right when
they need to hold." **The same applies arriving from `STANDING_UP`** — hips
are already stiffly holding the retracted position from `SU_RECOVER`'s hold
step. Without this fix, arming would momentarily loosen the hips at the exact
moment balance is most fragile (right after a catch). Add:

```cpp
bool from_standing_up = (g_state.state == STATE_STANDING_UP);
...
if (!from_jumping && !from_standing_up) controlLoop_reset_hip_ramp();
```

## 6. `firmware/robot_teensy/shared/comm_protocol.h`

- `#define FAULT_STANDUP_FAILED 0x0D  // standup denied (pitch out of recoverable range) or exhausted retries/diverged`
- `fault_severity()`: add `case FAULT_STANDUP_FAILED: return FAULT_SEVERITY_REPOSITION;` (matches `FAULT_PITCH_WATCHDOG` — "robot fell, reposition then reset").
- Bump `TELEM_VERSION` 10 → 11.
- Add `uint8_t standup_state;` at the end of `TelemetryPayload` (after
  `gain_sched_alpha`), new offset `[246]`, new `sizeof = 247`.
- `TELEM_B_LEN`: `128u` → `129u`; update both `static_assert`s
  (`sizeof(TelemetryPayload)==247`, `TELEM_A_LEN+TELEM_B_LEN==247`) and the
  byte-offset-map comment.
- `LogRecord` embeds `TelemetryPayload` directly → size auto-grows 250→251;
  update its `static_assert(sizeof(LogRecord) == 251, ...)` too (separate
  assert, easy to miss).
- Drive-by fix: the propagation-checklist comment at line 81 still points
  fault-name dicts at `software/gui/flash_monitor.py` — they've since moved to
  `software/gui/tabs/telem_format.py`. Fix the pointer while touching this
  block.

## 7. `firmware/robot_teensy/teensy/src/main.cpp`

- `fill_telemetry()`: add `t.standup_state = g_state.standup_state;` next to
  the existing `t.jump_state = ...` line.
- `update_led()`: this `switch(cur)` has no `default`, so the new state needs
  an explicit case — add
  `case STATE_STANDING_UP: g_led.blink(255, 60, 0, 60, 60); break;`
  (fast red-orange strobe — distinct from STANDBY's amber pulse, RUNNING's
  green blink, CMD_REJECT's slower orange blink, JUMPING's magenta; reads as
  "urgent, wheels may be moving hard"). Use whatever RGB is chosen in the
  README canonical color table (§10) so it's consistent everywhere.
- No changes needed to `radio_update()`, `on_command()`, or `cmd_allowed()`:
  `STATE_STANDING_UP` is never a directly-requestable `SET_MODE` target (same
  as `STATE_JUMPING` isn't — a stray packet just matches no branch and
  no-ops); a CH10 drop mid-`STANDING_UP` is already absorbed the same way a
  CH10 drop mid-`JUMPING` is today (disarm check is `STATE_RUNNING`-only and
  level-based, by existing design); ESTOP requests already work from any
  state unconditionally.

## 8. `firmware/robot_teensy/esp32/src/main.cpp`

Required (cheap, matches the propagation checklist's spirit — both switches
below currently have safe `default:` fallbacks, but STANDING_UP is exactly
the dramatic moment where a wrong/blank display matters most):
- `enum : uint8_t { ... RS_JUMPING=7, RS_STANDING_UP=8 };`
- `fault_description()`: add
  `case FAULT_STANDUP_FAILED: return "Standup denied or failed — pitch out of recoverable range";`
  (ASCII only, per the existing comment mandate).
- `mode_color()`: add `case RS_STANDING_UP: return tft.color565(255, 60, 0);`
  (match the README color table).
- `mode_name()`: add `case RS_STANDING_UP: return "STANDUP";` (space-constrained
  banner, matches `JUMPING`'s shortened `"JUMP!"`).

Deferred/optional (already degrade safely via existing `default:` branches —
bespoke animation code, not a required mirror): the NeoPixel per-state
animation switch (~line 324) and the TFT mode-banner animation switch
(~line 1060+).

## 9. GUI mirror updates

- **`software/gui/tabs/telem_format.py`** (actual single source of truth for
  the wire layout, per its own module docstring):
  - `_STATE_NAMES`: add `8: "STANDING_UP"`.
  - `_FAULT_NAMES`: add `0x0D: "STANDUP_FAILED"`.
  - `_FAULT_DESCRIPTIONS`: add `0x0D: "Standup denied or failed — pitch out of recoverable range"`.
  - Telemetry-B struct format: append one more `uint8` field; bump the
    `struct.calcsize` asserts (128→129 for TELEM_B, 246→247 for TELEM_FULL).
  - Decode functions: add `standup_state` to the unpack tuple/dict; extend the
    `payload[118:246]` slice to `payload[118:247]`.
  - `TELEM_VERSION` constant → 11.
- **`software/gui/tabs/flash_monitor.py`**: `_TELEM_VERSION = 11`.
- **`software/gui/tabs/params_tab.py`**: see §3 above.
- **`software/gui/tabs/hip_motors.py`** and **`software/gui/tabs/wheel_motors.py`**:
  add `8: ("STANDING_UP", "#ff3c00")` to each `_STATE_LABELS` dict. (These
  dicts are already missing `7: JUMPING` — pre-existing drift, not touched
  here per "don't touch unrelated code.")
- **`software/gui/tabs/raw_data_tab.py`**: add a `standup_state` row next to
  the existing `jump_state` row (both in the State section's `_add_row()`
  calls and in `_on_packet()`'s `_set()` calls).
- **`software/gui/main.py`** `StatusBar.set_mode()`: add `"STANDING_UP":
  <color>` to the inline `{"RUNNING": GREEN, ...}` dict (line ~978). (Same
  dict is already missing MANUAL/CMD_REJECT/JUMPING — pre-existing drift,
  not backfilled here.)

## 10. `firmware/robot_teensy/README.md`

- "Canonical state colour table" (~line 148): add a `STATE_STANDING_UP` row —
  pick one RGB (suggest red-orange, `255,60,0`, fast strobe) and reuse it in
  the Teensy LED, ESP32 `mode_color()`, and GUI status color above — this
  table is documented as the one source of truth for state→color across all
  four surfaces.
- Fault code table (~line 163): add the `0x0D | FAULT_STANDUP_FAILED | ... |
  REPOSITION` row.
- Fix the stale `flash_monitor.py` mirror-location note (~line 185) to point
  at `telem_format.py` instead.

---

## Verification

**Compile/read-only (no hardware):**
- Build the `teensy` PlatformIO env — catches param ID collisions, the
  `MOTOR_TRQ_MAX` header move, and any `static_assert` failure on
  `TelemetryPayload`/`LogRecord` size if the byte math is off.
- Build the `esp32` env — catches missing `RS_STANDING_UP` switch arms,
  `fault_description()` compiling.
- Launch the GUI with no device connected: confirm `params_tab.py` renders
  the new "Standing Up" subgroup with all 16 defs, and that
  `telem_format.py`'s `struct.calcsize` asserts pass at import (they throw
  immediately on a format-string/size mismatch — self-checking).
- Grep for any other hardcoded `246`/`FMT_TELEM_B` assumptions
  (`wlog_to_csv.py`, `log_playback.py`) to confirm nothing outside
  `telem_format.py` hardcodes the old size.
- With `PARAM_STANDUP_ENABLE=0` (the default), run the existing arm/state-
  machine stress-test scripts referenced in the README and confirm arm
  timing/behavior is unchanged from before this change — the best available
  regression check for "identical when disabled" without new hardware risk.

**Must be bench-tested on real hardware** (robot propped safely, e.g. on
blocks so wheels spin free without driving into anything; start every gain at
its default-0/conservative value and increase in small steps):

1. `PARAM_STANDUP_ENABLE=1`, gains still 0 (inert). Manually tip the robot
   past `PARAM_STANDUP_MAX_PITCH_FWD/BWD_RAD` and arm — confirm a clean
   `CMD_REJECT` denial (buzzer, red blink, back to STANDBY, no fault code).
   Tip it within range and arm — confirm it enters `STANDING_UP`
   (`standup_state` telemetry shows `SU_CROUCH` then `SU_RECOVER`, legs
   retract and hold, wheels output ~0 torque since gains are still 0).
2. Ramp `PARAM_STANDUP_K_PITCH`/`K_RATE`/`TORQUE_LIMIT` up from 0 in small
   steps, watching `whl_tau_l/r`, `pitch`, `pitch_rate`, `standup_state` live
   (Raw Data / Controllers tabs) — confirm the wheels push the **same**
   direction as the lean (toward upright) at low gain before increasing
   further. A wrong sign here would actively worsen the fall — verify at
   the lowest possible torque first.
3. Verify capture handoff: `standup_state` should hand off to
   `robot_state == RUNNING` with no visible hip "flinch" (tests the
   `from_standing_up` skip-ramp fix in §5 — reverting that one line and
   comparing is a good regression demo).
4. Force retry exhaustion (very low `PARAM_STANDUP_ATTEMPT_TIMEOUT_S` or
   under-tuned gains) — confirm `SU_RECOVER`/`SU_PAUSE` cycles
   `PARAM_STANDUP_MAX_RETRIES + 1` times, then lands in `STATE_ESTOP` with
   `fault_code = 0x0D`, GUI shows "STANDUP_FAILED" with REPOSITION severity
   (requires a full reset, not a soft-clear — confirm the Reset button
   behaves like it does for `FAULT_PITCH_WATCHDOG` today).
5. Force divergence (e.g. via `PARAM_ENABLE_SIM_PITCH_RAD` injection, or
   physically resist the catch) — confirm immediate ESTOP with
   `FAULT_STANDUP_FAILED`, no retry attempted.
6. Confirm the dedicated runaway backup trips independently: lower
   `PARAM_STANDUP_WHEEL_VEL_LIMIT_TURNS_S` and/or raise gains until wheels
   legitimately exceed 2× it — confirm `FAULT_WHEEL_RUNAWAY` (not
   `FAULT_STANDUP_FAILED`).
7. Confirm CH10 drop mid-`STANDING_UP` has no disarm effect until the state
   resolves to `RUNNING` or `ESTOP` (matches existing `JUMPING` behavior).

### Critical files
- `firmware/robot_teensy/teensy/src/state_machine.cpp` (core logic)
- `firmware/robot_teensy/teensy/src/robot_state.h` (state enum + `standup_state` field)
- `firmware/robot_teensy/teensy/src/control_loop.h` (`MOTOR_TRQ_MAX` move)
- `firmware/robot_teensy/teensy/src/main.cpp` (`fill_telemetry()`, `update_led()`)
- `firmware/robot_teensy/teensy/lib/ParamRegistry/param_ids.h` + `param_registry.cpp`
- `firmware/robot_teensy/shared/comm_protocol.h` (fault code, telemetry V11)
- `firmware/robot_teensy/esp32/src/main.cpp` (state/fault mirrors)
- `software/gui/tabs/telem_format.py`, `params_tab.py`, `hip_motors.py`, `wheel_motors.py`, `raw_data_tab.py`, `flash_monitor.py`, `main.py`
- `firmware/robot_teensy/README.md` (color table, fault table)
