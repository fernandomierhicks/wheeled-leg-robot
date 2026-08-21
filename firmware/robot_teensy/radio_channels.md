# Radio Channel Map

**Link:** ExpressLRS / CRSF, 420000 baud, `Serial4` (RX pin 16, **TX pin 17**).
Bidirectional -- telemetry rides the return path. See
`teensy/src/crsf_protocol.h` for the wire format and, more importantly, its
FAILSAFE CONTRACT.

**TX Profile:** `WLR ROBOT` on the RadioMaster TX15 (`radio/sdcard/MODELS/`).
**Reversals:** none

> **"Up" is ambiguous, so read this once.** Below, "C5 up" means the *channel*
> is high (~2000 us). On the TX15 that is produced by pushing the switch
> **down** -- the model maps every switch so that up = channel low = safe, and
> the power-on switch warning enforces it. See `radio/CHANNELS.md`.

### Signal loss

`channel()` returns **0**, not the last value, whenever the link is not alive,
and `alive()` goes false on any of: no frame within 100 ms, uplink link
quality at 0 while frames still arrive, or fewer than 5 good frames since
boot. Set the receiver's own failsafe to **no pulses**, never "hold".

---

## Channel Assignments

| Ch  | TX Control         | Function             | Details                                                     |
|-----|--------------------|----------------------|-------------------------------------------------------------|
| C1  | Roll stick         | Roll setpoint        | 1000–2000 → −1…+1 × `RADIO_ROLL_MAX` → `PARAM_ROLL_CMD_RAD` (active-suspension roll controller; only acts when `roll_ctrl_en=1` in RUNNING) |
| C2  | Pitch stick        | Forward velocity     | 1000–2000 → −1…+1 × `RADIO_VEL_MAX` → `PARAM_V_CMD_MS`   |
| C3  | Throttle stick     | Hip height / angle   | 1000–2000 → 0…1 → `PARAM_RADIO_HIP_CMD`                   |
| C4  | Rudder stick       | Yaw rate             | 1000–2000 → −1…+1 × `RADIO_YAW_MAX` → `PARAM_OMEGA_CMD_RDS` |
| C5  | SWA (left switch)  | **SD logging** | **Up = start recording, down = stop.** Edge-triggered. A start is refused outside STANDBY/ESTOP (`"CH5 up ignored -- start the log before arming"`) because opening a log preallocates and blocks ~96 ms — but once started, **recording continues through RUNNING**, which is the whole point. Confirmed by a single G5 chirp, so check `buzzer_volume` — there is no LED cue. C5's level no longer gates arming. *(LEGACY mode: gain-group select with C6, and no SD-log function.)* |
| C6  | SWB                 | **Jump** | Rising edge > 1990 requests `STATE_JUMPING`. **One jump per edge** — drop C6 and raise it again for another; holding it up does not hop repeatedly. Refused unless the robot is in RUNNING *and* `jump_enable = 1` (default **0**, so this is inert until you deliberately enable it). *(LEGACY mode: gain-group select with C5, and no jump function.)* |
| C7  | First knob         | Live tune, slot 0 of active group | 1000–2000 → active group's slot-0 range → its mapped param. See "Live parameter tuning" below. |
| C8  | Second knob        | Live tune, slot 1 of active group | 1000–2000 → active group's slot-1 range → its mapped param. See "Live parameter tuning" below. |
| C9  | 3-pos switch       | Speed profile        | < 1333 = profile 1, 1333–1667 = profile 2, > 1667 = profile 3. Selects the active `vel_max`, `yaw_max`, `torque_lim`, and `roll_max`. |
| C10 | Right switch (ARM) | Arm / disarm         | > 1990 → RUNNING (requires calibration); drop → STANDBY    |
| C11 | SWC                | **Hard ESTOP**       | **Level-triggered.** > 1990 raises `FAULT_HUMAN_ESTOP` from any non-ESTOP state, and blocks arming while held. The only single-motion panic input: the rescue combo is armed solely in STANDBY/ESTOP, and C10 low merely starts DISARMING (a controlled ramp-down, not a stop). Debounced 3 ticks. Guarded by `alive()`, so a dead link cannot assert it — link loss has its own DISARMING path and should not latch a fault. Recovery: release C11, then toggle C10 to soft-clear (HUMAN_ESTOP is SOFT severity) back to STANDBY. |
| C12 | SWD                | *unassigned*         | Earmarked for `roll_ctrl_en`, but that param is **persistent** — driving it from a switch would rewrite the stored value, including at boot from whatever position the switch is in. Needs a non-persistent runtime gate first. |

## Calibration combo — STANDBY → CALIBRATION

Both sticks jammed into opposite corners, the **exact mirror of the rescue combo
below**: **C1 and C4 full up (> 1990), C2 and C3 full down (< 1010)**. Debounced
3 ticks (~6 ms @ 500 Hz). One-shot on the rising edge — release and re-enter for
the next action. Replaced the old C5 switch when C5 became the SD-log switch.

| Event | Effect |
|---|---|
| Enter combo, in STANDBY | Requests CALIBRATION (still subject to `stateMachine_request_calibration()`'s own check that at least one hip motor is enabled) |
| Enter combo again, during a radio-started CALIBRATION | Cancels it through DISARMING — tapers each hip's calibration gains to zero over `calib_rampdown_s`, then returns to STANDBY |
| Enter combo, any other state | Ignored, logged |

**Armed only in STANDBY and CALIBRATION — deliberately not in ESTOP**, so in a
fault state the rescue combo is the only live stick gesture and the two mirrored
gestures can't be confused. The two combos are mutually exclusive stick
positions and can never be satisfied on the same tick.

After any accepted action there is a **1 s lockout** before another combo edge is
honoured, so a stick glitch part-way through a deliberate hold cannot start a
calibration and immediately cancel it again. Powering up or reconnecting with the
sticks already in the combo position does nothing: a debounced *release* must be
seen before the first rising edge counts. Loss of the radio link reads as
"released", which is inert — only rising edges act, so a dead radio can never
imitate an operator request. There is no other radio-side abort; re-entering the
combo is it.

## Rescue combo — clear ESTOP / reboot

Both sticks jammed into opposite corners, held: **C3 and C2 full up (> 1990),
C1 and C4 full down (< 1010)**. Debounced 3 ticks (~6 ms @ 500 Hz). Only armed
in **STANDBY** or **ESTOP** — never with torque live. Intended as a
transmitter-only escape hatch when the GUI isn't connected; no normal driving
input produces this stick position.

| Event | Effect | Cue |
|---|---|---|
| Enter combo, in ESTOP | Full reset → STARTUP: clears the fault regardless of severity and re-runs the startup checks | A5⇄E6 siren + white LED flash |
| Enter combo, in STANDBY | Nothing to clear — beep only, and the 3 s countdown starts | same siren |
| Hold 3 s | Full MCU reset, identical to the GUI's Reboot command | E6→A5→D5→G4 descending fall |

If the fault can't actually be cleared (a REBOOT-severity hardware dropout),
STARTUP re-faults straight back to ESTOP — keep holding and the 3 s reboot
fires anyway. Releasing the sticks re-arms the combo, so a second attempt
starts a fresh countdown. Loss of the radio link cannot satisfy the combo.

## Arm authority

- With no live radio link, the GUI/API may arm and owns that running session.
- While the radio is live, GUI/API arm requests are rejected; C10 owns arming.
- If a radio link appears during a GUI-owned running session, the robot enters
  DISARMING regardless of C10 level. Re-arm from STANDBY using C10.
- During a radio-owned session, live C10-low or loss of the radio link enters
  DISARMING. C10 is level-based, so its initial low state cannot be missed.
- Motion authority is never handed directly between GUI and radio while armed;
  every takeover passes through STANDBY.
- A calibration started by the calibration combo remains radio-owned until it
  completes or faults. Re-entering the combo while it is active enters
  DISARMING, tapers each hip's current calibration gains to zero over
  `calib_rampdown_s`, then returns to STANDBY. Loss of the radio link is neutral
  and does not imitate this operator-requested cancellation.

## Live parameter tuning

Generic mechanism (`teensy/src/live_tune.h`, `LIVE_TUNE_SLOTS` in `main.cpp`)
for feeling out a gain/limit live on the bench with a knob instead of editing
the Params tab blind and re-arming to see the effect. C7/C8 each drive one
slot of the active *gain group*, RUNNING only.

**Whether live tuning is available at all is set by `live_tune_multi_en`.**

### `live_tune_multi_en = 0` — SIMPLE (default)

**Live tuning is off.** C5 is the SD-log switch and C6 is the jump trigger, so
there is no switch left to select a group with and the C7/C8 knobs are inert.
The groups are not deleted, just unreachable — set `live_tune_multi_en = 1`.

### `live_tune_multi_en = 1` — LEGACY (three groups)

The original combination scheme, and the only mode in which C5/C6 mean *gain
group*. In this mode C5 does **not** drive SD logging and C6 does **not** jump —
a tuning session trades both away for the knobs, so it is bench-only by nature.

| C5 | C6 | Group | Slot 0 (C7) | Slot 1 (C8) |
|---|---|---|---|---|
| up | up | *(none — tuning inactive)* | — | — |
| down | up | 0 | `lqr_k_pitch_ret`, -0.1 … -2.0 | `lqr_k_rate_ret`, -0.01 … -1.0 |
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
live-tune mode or change groups (any C5/C6 change, or leaving RUNNING) —
re-entering always requires re-sweeping.

**Nothing is written to the real, persistent param until you say so.** While a
slot is picked up, its live shadow value is what the control loop actually
uses (real-time effect, so you can feel the change) — but the underlying
persistent param is untouched, so exiting live-tune mode (or switching groups)
without latching leaves it exactly as it was.

*(All of the below requires `live_tune_multi_en = 1`.)*

1. Arm normally (raise C10) → RUNNING.
2. Select the group with the C5/C6 combination (see table above). Sweep C7/C8
   through the group's current values to pick each slot up (watch
   `live_tune_ch7_val`/`live_tune_ch8_val` in telemetry — they update
   immediately; the *effect* on balance only kicks in once picked up,
   independently per slot).
3. Write `live_tune_latch` = 1 (Params tab) to persist every currently
   picked-up slot's shadow value into its real param. One-shot: firmware
   latches and resets the flag. A slot that never picked up this session is
   skipped, not latched at a stale/arbitrary value.
4. To leave live-tune mode: return both C5 and C6 to **up**, or move to another
   group and repeat steps 2-3. Either way the latched (or, if you didn't latch,
   unchanged) persistent values take over.
