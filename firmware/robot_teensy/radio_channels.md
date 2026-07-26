# Radio Channel Map

**TX Profile:** `ROBOT_03`  
**Reversals:** none

---

## Channel Assignments

| Ch  | TX Control         | Function             | Details                                                     |
|-----|--------------------|----------------------|-------------------------------------------------------------|
| C1  | Roll stick         | Roll setpoint        | 1000–2000 → −1…+1 × `RADIO_ROLL_MAX` → `PARAM_ROLL_CMD_RAD` (active-suspension roll controller; only acts when `roll_ctrl_en=1` in RUNNING) |
| C2  | Pitch stick        | Forward velocity     | 1000–2000 → −1…+1 × `RADIO_VEL_MAX` → `PARAM_V_CMD_MS`   |
| C3  | Throttle stick     | Hip height / angle   | 1000–2000 → 0…1 → `PARAM_RADIO_HIP_CMD`                   |
| C4  | Rudder stick       | Yaw rate             | 1000–2000 → −1…+1 × `RADIO_YAW_MAX` → `PARAM_OMEGA_CMD_RDS` |
| C5  | SWA (left switch)  | Calibration / live-tune gain-group select | In **STANDBY**: rising edge > 1990 starts calibration; falling edge gracefully cancels a radio-triggered calibration through DISARMING. In **RUNNING**: combines with C6 to select a live-tune gain group for C7/C8 (see "Live parameter tuning" below). Arming is refused while C5 is up. |
| C6  | SWB                 | Live-tune gain-group select | In **RUNNING**, combines with C5 to select a live-tune gain group for C7/C8. No effect outside RUNNING. |
| C7  | First knob         | Live tune, slot 0 of active group | 1000–2000 → active group's slot-0 range → its mapped param. See "Live parameter tuning" below. |
| C8  | Second knob        | Live tune, slot 1 of active group | 1000–2000 → active group's slot-1 range → its mapped param. See "Live parameter tuning" below. |
| C9  | 3-pos switch       | Speed profile        | < 1333 = profile 1, 1333–1667 = profile 2, > 1667 = profile 3. Selects the active `vel_max`, `yaw_max`, `torque_lim`, and `roll_max`. |
| C10 | Right switch (ARM) | Arm / disarm         | > 1990 → RUNNING (requires calibration); drop → STANDBY    |

## Arm authority

- With no live radio link, the GUI/API may arm and owns that running session.
- While the radio is live, GUI/API arm requests are rejected; C10 owns arming.
- If a radio link appears during a GUI-owned running session, the robot enters
  DISARMING regardless of C10 level. Re-arm from STANDBY using C10.
- During a radio-owned session, live C10-low or loss of the radio link enters
  DISARMING. C10 is level-based, so its initial low state cannot be missed.
- Motion authority is never handed directly between GUI and radio while armed;
  every takeover passes through STANDBY.
- A calibration started by C5 remains radio-owned until it completes or faults.
  Returning C5 low while calibration is active enters DISARMING, tapers each
  hip's current calibration gains to zero over `calib_rampdown_s`, then returns
  to STANDBY. Loss of the radio link is neutral and does not imitate this
  operator-requested cancellation.

## Live parameter tuning

Generic mechanism (`teensy/src/live_tune.h`, `LIVE_TUNE_SLOTS` in `main.cpp`)
for feeling out a gain/limit live on the bench with a knob instead of editing
the Params tab blind and re-arming to see the effect. C7/C8 each drive one
slot of whichever *gain group* is currently selected by the C5/C6 switch
combination (RUNNING only):

| C5 | C6 | Group | Slot 0 (C7) | Slot 1 (C8) |
|---|---|---|---|---|
| up | up | *(none — tuning inactive)* | — | — |
| down | up | 0 | `lqr_k_pitch_ret`, -0.1 … -0.5 | `lqr_k_rate_ret`, -0.01 … -0.5 |
| up | down | 1 | `vel_pi_kp`, 0.05 … 1.0 | `vel_pi_ki`, 0.02 … 0.5 |
| down | down | 2 | `roll_kp`, 0.3 … 4.0 | `roll_kd`, 0.02 … 0.5 |

Ranges are `(value at knob-zero, value at knob-max)`. Both slots are mirrored
live to telemetry regardless of group: `live_tune_ch7_val` / `live_tune_ch8_val`.

**Knob direction convention.** A slot's range is `(value at knob-zero, value at
knob-max)`, not a sorted `(min, max)` pair. Group 0's gains are negative, so
turning the knob up always moves the value *more* negative — i.e. "more knob"
always means "more gain" (larger magnitude / stronger action), regardless of
the sign of the underlying param. Never flip a slot's range so that the raw
signed value increases with the knob instead.

**Repointing a slot at a different param** (e.g. a future tuning session) is a
one-line edit to `LIVE_TUNE_SLOTS` + reflash — no other code changes needed as
long as the target's read site in `control_loop.cpp` goes through
`live_tune_value(PARAM_X)` instead of a bare `param_get(PARAM_X)`.

**Safety — pickup, no step change.** A knob is inert on entering live-tune mode
(or switching to a different group) until it's swept through the target
param's *current* value; only then does it "pick up" and start tracking 1:1.
This avoids the gain jumping instantly to wherever the knob happened to be
sitting when the group was selected. Pickup resets every time you leave
live-tune mode or change groups (C5/C6 combination changes, or leaving
RUNNING) — re-entering always requires re-sweeping.

**Nothing is written to the real, persistent param until you say so.** While a
slot is picked up, its live shadow value is what the control loop actually
uses (real-time effect, so you can feel the change) — but the underlying
persistent param is untouched, so exiting live-tune mode (or switching groups)
without latching leaves it exactly as it was.

1. Arm normally (C5 **down**, raise C10) → RUNNING.
2. Set C5/C6 to the desired group (see table above). Sweep C7/C8 through the
   group's current values to pick each slot up (watch `live_tune_ch7_val`/
   `live_tune_ch8_val` in telemetry — they update immediately; the *effect* on
   balance only kicks in once picked up, independently per slot).
3. Write `live_tune_latch` = 1 (Params tab) to persist every currently
   picked-up slot's shadow value into its real param. One-shot: firmware
   latches and resets the flag. A slot that never picked up this session is
   skipped, not latched at a stale/arbitrary value.
4. Set C5/C6 to another group to tune it (repeat steps 2-3), or return both to
   **up** to leave live-tune mode; the latched (or, if you didn't latch,
   unchanged) persistent values take over.
