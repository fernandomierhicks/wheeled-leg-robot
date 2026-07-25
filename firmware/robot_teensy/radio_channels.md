# Radio Channel Map

**TX Profile:** `ROBOT_03`  
**Reversals:** none

---

## Channel Assignments

| Ch  | TX Control         | Function             | Details                                                     |
|-----|--------------------|----------------------|-------------------------------------------------------------|
| C1  | Roll stick         | *(unused)*           | —                                                           |
| C2  | Pitch stick        | Forward velocity     | 1000–2000 → −1…+1 × `RADIO_VEL_MAX` → `PARAM_V_CMD_MS`   |
| C3  | Throttle stick     | Hip height / angle   | 1000–2000 → 0…1 → `PARAM_RADIO_HIP_CMD`                   |
| C4  | Rudder stick       | Yaw rate             | 1000–2000 → −1…+1 × `RADIO_YAW_MAX` → `PARAM_OMEGA_CMD_RDS` |
| C5  | SWA (left switch)  | Calibration / live parameter tuning | In **STANDBY**: rising edge > 1990 starts calibration; falling edge gracefully cancels a radio-triggered calibration through DISARMING. In **RUNNING**: > 1990 enables live-tune mode (C7/C8 knobs drive whatever params `LIVE_TUNE_SLOTS` currently maps them to). Arming is refused while C5 is up. |
| C6  | SWB / joystick btn | Launch / jump        | Rising edge > 1990, only from RUNNING                      |
| C7  | First knob         | Live tune, slot 0     | 1000–2000 → slot 0's range → its mapped param. Currently `vel_pi_kp`, range 0..0.5. See "Live parameter tuning" below. |
| C8  | Second knob        | Live tune, slot 1     | 1000–2000 → slot 1's range → its mapped param. Currently `vel_pi_ki`, range 0..0.5. |
| C9  | 3-pos switch       | Speed profile        | < 1333 = profile 1, 1333–1667 = profile 2, > 1667 = profile 3 |
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
the Params tab blind and re-arming to see the effect. Two slots, one per knob
(C7, C8), each independently mapped to a param + range. Currently:

| Slot | Channel | Param | Range | Live shadow (telemetry) |
|---|---|---|---|---|
| 0 | C7 | `vel_pi_kp` | 0..0.5 | `live_tune_ch7_val` |
| 1 | C8 | `vel_pi_ki` | 0..0.5 | `live_tune_ch8_val` |

**Repointing a slot at a different param** (e.g. a future tuning session) is a
one-line edit to `LIVE_TUNE_SLOTS` + reflash — no other code changes needed as
long as the target's read site in `control_loop.cpp` goes through
`live_tune_value(PARAM_X)` instead of a bare `param_get(PARAM_X)`.

**Safety — pickup, no step change.** A knob is inert on entering live-tune mode
until it's swept through the target param's *current* value; only then does it
"pick up" and start tracking 1:1. This avoids the gain jumping instantly to
wherever the knob happened to be sitting the moment C5 goes up. Pickup resets
every time you leave live-tune mode (C5 low, or leaving RUNNING) — re-entering
always requires re-sweeping.

**Nothing is written to the real, persistent param until you say so.** While a
slot is picked up, its live shadow value is what the control loop actually
uses (real-time effect, so you can feel the change) — but the underlying
persistent param (`vel_pi_kp`, etc.) is untouched, so exiting live-tune mode
without latching leaves it exactly as it was.

1. Arm normally (C5 **down**, raise C10) → RUNNING.
2. Raise C5 → live-tune mode. Sweep C7/C8 through each slot's current value to
   pick it up (watch `live_tune_ch7_val`/`live_tune_ch8_val` in telemetry —
   they update immediately; the *effect* on balance only kicks in once picked
   up). **Tune one knob at a time** — with both live, you can't tell which
   knob caused an observed change; freeze one, sweep the other, then swap.
3. Write `live_tune_latch` = 1 (Params tab) to persist every currently
   picked-up slot's shadow value into its real param. One-shot: firmware
   latches and resets the flag. A slot that never picked up this session is
   skipped, not latched at a stale/arbitrary value.
4. Lower C5 to leave live-tune mode; the latched (or, if you didn't latch,
   unchanged) persistent values take over.
