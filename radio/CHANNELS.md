# TX15 control map

Staged in `sdcard/MODELS/model90.yml`, model name **WLR ROBOT**. Derived from
`firmware/robot_teensy/radio_channels.md` and `tx15-robot-integration-plan.md`
§1. Verified against the radio's own hardware definition
(`radio/src/targets/tx15/hal.h`) and its own `MODELS/model1.yml`.

## The polarity rule, once

| Switch position | Mix output | Pulse width | Firmware sees | Meaning |
|---|---|---|---|---|
| **up** | −100% | ~1000 µs | channel LOW | safe / inert |
| middle (3-pos) | 0% | ~1500 µs | channel MID | — |
| **down** | +100% | ~2000 µs | channel HIGH | active |

Every switch function follows it, so **"all switches up" is always the inert
configuration** — which is what the model's power-on switch warning enforces.

> `radio_channels.md` says things like *"C5 up = start recording"*. That "up"
> means the **channel** is high, which on this transmitter is the switch's
> **down** position. Same signal, opposite word: the firmware talks in
> microseconds, the radio talks in switch positions. Don't reconcile them by
> flipping a mix weight — you'd break the "up = safe" invariant everywhere else.

## Channels

| CH | Control | Position | Function |
|---|---|---|---|
| 1 | Right stick ↔ | gimbal | Roll / lean setpoint |
| 2 | Right stick ↕ | gimbal | Forward velocity |
| 3 | Left stick ↕ | gimbal | Hip height (30% expo) |
| 4 | Left stick ↔ | gimbal | Yaw rate |
| 5 | **SB** | top row, 2nd from left | SD logging (robot + radio) |
| 6 | **SF** | momentary shoulder | Jump (rising edge) |
| 7 | **S1** | left dial | Live tune A |
| 8 | **S2** | right dial | Live tune B |
| 9 | **SC** | top row, right of dials | Speed profile 1–3 |
| 10 | **SD** | top row, rightmost | ARM |
| 11 | **SA** | top row, leftmost | Calibration request |
| 12 | **SE** | latching shoulder | Reset fault / clear ESTOP |
| 13–16 | — | — | *Reserved for the six RGB buttons (SG–SL) — TODO* |

Inputs are declared in the radio's native RETA order (I0=Rud, I1=Ele, I2=Thr,
I3=Ail) so the Inputs screen looks like every other model on this transmitter.

## Shoulder switches — no hardware work needed

The TX15 ships with **SF as the momentary button** and **SE as the latching
button** (RadioMaster's own manual, back view). That is exactly the pairing
this model wants, so nothing has to be opened or swapped:

- **SF (momentary) → CH6 jump.** Jump triggers on a rising edge, and a
  latching control here could sit latched where you cannot see it.
- **SE (latching) → CH10 ARM.** The firmware arm test is level-based
  (`armed = alive && ch10 > 1990`), so a momentary button would disarm the
  instant you let go.

The integration plan proposed the opposite lettering. That was arbitrary — what
matters is momentary-on-jump and latching-on-ARM, and following the factory fit
gets there with no disassembly and no chance of fitting the wrong one.

Alternative switch panels are in the box if you ever want toggles instead of
buttons; the case opens with four hex screws. If you do swap them, re-check the
model's switch warning afterwards.

## Trims do nothing, deliberately

All four trim rockers are disabled on this model (`carryTrim: TRIM_OFF` on
every stick mix). The robot already has its own trim authority —
`pitch_trim_rad` and the whole balance-point story in `CLAUDE.md` — and a
second one on the transmitter is the "two sources of truth" problem in
miniature.

The specific hazard: a bumped trim would put a permanent creep on velocity, a
standing lean on roll, a slow turn on yaw, or an offset on leg height. And
because the custom screens turn EdgeTX's trim display off, it would be doing
that invisibly. Radios get bumped in bags.

Re-enable per channel in `build_model.py` if you ever genuinely want it.

## Stopping the robot

**Drop SD.** There is deliberately no separate hard-ESTOP channel: disarming
covers it. Be aware of what that means — disarm goes through `DISARMING`, a
controlled taper of the hip gains, not an immediate torque cut. An immediate
version was implemented on CH11 and is in git history if the difference ever
matters to you.

## The two edge-triggered switches

CH11 (calibration) and CH12 (reset) are **edge**-triggered off latching
switches, so both require a debounced *release* before another edge counts.
That is what stops powering up — or reconnecting — with the switch already
down from firing the action immediately.

Neither has a hold-to-reboot. The rescue stick combo can afford one because
holding two sticks in opposite corners is unmistakably deliberate; a latching
switch left down is just the resting state of a switch somebody flicked and
forgot.

Both stick combos remain as fallbacks for when the transmitter is not the
arming authority.

## What the radio does *not* control

Arm, disarm, ESTOP and the rescue combo are physical channels and firmware
interlocks. The Lua HUD is an instrument: it reads and it speaks, it never
writes. A script can be exited, starved, or crash, and none of those may be
able to disarm a robot.

TX-side logical switches and special functions here do **annunciation and
logging only**. Firmware stays authoritative — TX-side logic that tried to
auto-disarm would fight the interlocks and produce behaviour nobody can
reconstruct afterwards.

## Combos still work

Both stick combos survive this mapping unchanged (mode 2):

- **Rescue** (clear ESTOP / reboot): CH3 + CH2 full up, CH1 + CH4 full down.
- **Calibration** (STANDBY → CALIBRATION): the exact mirror — CH1 + CH4 full
  up, CH2 + CH3 full down.

Verify after the CRSF swap that full deflection actually reaches the µs
thresholds. ELRS endpoints occasionally land a few ticks short and the combos
then silently stop working.

## The six RGB function switches

`hal.h` shows the TX15 also has **SG–SL**, six function switches with RGB LEDs
(`FUNCTION_SWITCHES_WITH_RGB`). The integration plan predates this and doesn't
account for them. Nothing is assigned to them yet — they're a genuinely good
fit for mirroring robot state in peripheral vision (EdgeTX has an `RGB_LED`
special function), or for latched bench toggles. Left free deliberately.
