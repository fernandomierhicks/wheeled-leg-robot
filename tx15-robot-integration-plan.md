# TX15 as the robot's field head — integration plan

Target: `fernandomierhicks/wheeled-leg-robot`, `firmware/robot_teensy`.
Radio: RadioMaster TX15 / TX15 MAX (STM32H7, EdgeTX 2.12+, 3.5" IPS touchscreen,
LR1121 ELRS 2.4 / 900, ICM-42607-C gyro, no sliders, 4× 3-position top switches,
2× swappable shoulder switches, 2 detented pots).

Scope of the win: today, anything beyond stick input requires the Python GUI over
WiFi/USB. The TX15 can absorb the *field-relevant* half of that — state, faults,
live tuning, jump ladder, presets — while the WiFi GUI keeps the deep work (log
analyzer, plots, firmware flashing, automation harness).

---

## 0. The blocking change: iBUS → CRSF

This is the whole precondition. Everything else in this document depends on it.

Current: `Serial4 RX → FlySky iBUS`. One wire, one direction. The TX15 has no
FlySky protocol — it is ELRS/CRSF. So the link swap is not optional, and the
payoff is that **CRSF is bidirectional**, which is what makes telemetry and an
on-radio GUI possible at all.

### Hardware
- ELRS receiver matched to the band you bind on (the LR1121 does 2.4 GHz *or*
  900 MHz — pick the RX to match; 900 is the better choice for a ground robot
  behind furniture/legs, 2.4 for lower latency).
- Wire to a Teensy UART using **both** TX and RX. `Serial4` already has RX
  committed; its TX pin is the natural pairing. Confirm nothing else claims it in
  `config.h`.
- CRSF runs 420000 baud by default. Teensy 4.1 UARTs handle this fine; the ESP32
  link is already at 4 Mbaud so this is not the tight constraint.

### Firmware
- Replace the `IBus` object with a CRSF parser. Frame: `0xC8` sync, length, type,
  payload, CRC-8 (poly 0xD5). RC channels arrive as type `0x16`, 16 channels
  packed 11 bits each.
- **Preserve the failsafe semantics exactly.** The current safety design leans on
  `IBus::channel()` returning `0` on signal loss:
  - the rescue combo is gated on `g_ibus.alive()` specifically because otherwise a
    dead radio satisfies both "stick low" tests for free;
  - `radio_update()`'s disarm interlock computes `armed = alive && (ch10 > 1990)`.

  A CRSF parser must reproduce both properties: an `alive()` that goes false on
  link timeout, and a `channel()` that returns 0 (not "last value") when not
  alive. Write the unit test for this before wiring a motor.
- Scale mapping: iBUS channels are already 1000–2000 µs. CRSF ticks are
  172–1811 for 1000–2000 µs. Every literal threshold in `main.cpp` — `> 1990`,
  `< 1010` — has to be converted or the parser has to normalise to µs at the
  boundary. **Normalise at the boundary**; leave the thresholds alone. Fewer
  places to get it wrong.

### Bring-up gate
Do the swap entirely with torque off, using the tooling that already exists:
`hip_l/r_enable = 0`, `wheel_l/r_enable = 0`, `calib_bypass_en = 1`,
`run_wheel_bypass_en = 1` (the GUI's **No Motors** preset), and verify each
channel lands on the intended function via the read-only `PARAM_IBUS_CH*` mirrors
before any re-arm. `stress_test_arm.py` already checks that gate — reuse it.

---

## 1. Switch / knob / stick assignment

Existing channel contract (from `radio_channels.md` and `main.cpp`):

| CH | Function |
|----|----------|
| 1 | roll setpoint (× `profileN_roll_max`) |
| 2 | forward velocity command |
| 3 | hip height (`radio_hip_cmd`, normalized `t∈[0,1]`, slewed by `hip_cmd_rate_lim`) |
| 4 | yaw rate |
| 5 | SD log (SIMPLE) / live-tune group select bit (LEGACY) |
| 6 | jump trigger, rising edge (SIMPLE) / group select bit (LEGACY) |
| 7 | live-tune knob A |
| 8 | live-tune knob B |
| 9 | profile select |
| 10 | ARM (> 1990) |

### Proposed mapping

| Control | CH | Function | Why this control |
|---|---|---|---|
| Right stick H | 1 | roll / lean setpoint | self-centring is correct — release returns to level |
| Right stick V | 2 | forward velocity | self-centring = release means stop |
| **Left stick V** | 3 | hip height | **must be the non-centring (throttle) stick** — leg height is a held pose, not a transient |
| Left stick H | 4 | yaw rate | self-centring |
| **SA** (3-pos) | 9 | profile select | three profiles, three detents, exact fit |
| **SB** (3-pos) | 5 | SD log (use 2 of 3 positions, or 3-state: off / log / log+marker) | |
| **SC** (3-pos) | 11 *(new)* | hard ESTOP — see §6 | top-row, deliberate reach |
| **SD** (3-pos) | 12 *(new)* | spare / roll-controller enable | |
| **SE** shoulder — install the **momentary** switch | 6 | jump trigger | rising-edge trigger on a momentary cannot be left latched; a latching switch here is a foot-gun |
| **SF** shoulder — install the **2-position latching** switch | 10 | ARM | index-finger reachable for an instant kill, latching so it holds |
| **S1** pot | 7 | live-tune A | centre detent gives you a repeatable "midpoint" reference during pickup sweeps |
| **S2** pot | 8 | live-tune B | |

The TX15 ships with alternative momentary and 2-position shoulder switches plus
matching panels, and the case opens with four hex screws — installing SE
momentary / SF latching is a 20-minute job and is the single highest-value
hardware change on the radio.

### Rescue combo compatibility
The combo is CH3+CH2 full up, CH1+CH4 full down — both sticks jammed into
opposite corners. That works unchanged with the mapping above (mode 2), and stays
physically distinct from any normal input. Verify after the CRSF swap that full
deflection actually reaches the µs thresholds; ELRS endpoints occasionally land a
few ticks short and the combo silently stops arming.

### EdgeTX-side guards to configure
- **Switch warning** on SF (ARM) — the radio refuses to finish booting with arm
  up. Free protection the current stack doesn't have.
- **Throttle warning** on the hip-height stick — refuses to boot with legs
  commanded away from the retracted end.
- **Channel endpoints/limits per model** — a "BENCH" model can clamp CH2/CH4
  travel to ±30% so a fumbled stick can't command full velocity on a stand.
- **Expo on CH3 (hip height)** — the useful resolution is near the crouch end;
  expo buys you fine control there without touching `hip_cmd_rate_lim`.

---

## 2. On-radio GUI (Lua + LVGL)

### Transport
EdgeTX Lua exposes `crossfireTelemetryPush(command, data)` and
`crossfireTelemetryPop()` → arbitrary CRSF frames in both directions. This is the
same mechanism the ELRS config script uses, and it means the radio can speak
**your existing command protocol** — `CMD_ID_PARAM_GET` / `PARAM_SET` /
`SET_MODE`, `PARAM_REPORT`, `COMMAND_RESULT` — tunnelled inside CRSF, through the
ELRS link, out the RX's CRSF UART, into the Teensy.

Architecturally that means the Teensy needs **no new command surface**. It needs a
CRSF-side adapter that unwraps a tunnelled frame and hands it to the same
`on_command()` path the ESP32 already feeds.

### Generate the Lua, don't hand-write it
`protocol/schema.json` is already the single source for the C++ tables, the
Python GUI module, the docs, the frozen vectors, and the twin's
`params_control.py`. **Add a Lua emitter to `generate_protocol.py`.** Then the
radio's param page inherits the same property the GUI Params tab has: new
parameters appear without a hand-written widget, and `--check` catches staleness
in CI.

### Pages

1. **Status** (read-only, the one you'll actually leave open)
   - State name + the canonical state colour, straight from the colour table
   - Fault name and description when in ESTOP, plus severity tier so you know
     whether you're reaching for the robot or the power switch
   - `gain_sched_alpha`, pitch, roll, `hip_l/r_torque_nm`, `wheel_vel_avg_ms`
   - Link: LQ / RSSI / TPWR (free from CRSF link stats)
   - `esp32_link_ok`, `uart_rx_drops`, `uart_seq_gaps` — the link-supervision
     fields, visible without a laptop

2. **Live tune** — the biggest single quality-of-life win
   The current pickup mechanism is invisible: you sweep a knob and have no
   feedback about whether it picked up. Render it:
   - two rows, one per slot, showing param name, current value, knob position,
     target value
   - a bar with a marker at the pickup point and a **PICKED UP** badge
   - `live_tune_ch7_val` / `live_tune_ch8_val` shadow values, which are already
     telemetry-visible
   - a **LATCH** button (`lvgl.button` with a confirm long-press) writing
     `live_tune_latch = 1`, and a read-back confirming which slots committed
   - group indicator reflecting the CH5/CH6 combination, including "no group"

3. **Params** — scrollable list generated from `PARAM_REPORT` (min/max/flags/name
   already come down the wire). `lvgl` number-edit / toggle / choice controls.
   Read-only params render greyed rather than hidden.

4. **Jump** — `jump_enable` toggle, `jump_effort` slider with the ladder stepped
   0.5 → 0.6 → 0.7 → 0.8 → 0.9 → 1.0, and a guard that refuses to raise effort
   until it has seen `jump_state == 1` produce an *increasing* `gain_sched_alpha`
   at least once this session. That encodes the README's sign-check discipline
   into the tool instead of into your memory.

5. **Presets** — **No Motors** / **Full Robot** as two buttons, writing and
   read-verifying the same set the GUI does (four motor enables, both arm
   bypasses, `standup_enable`, the three watchdog enables).

### Constraints to design around
- **Bandwidth.** CRSF telemetry is a small slice of the link. A full param dump
  will take seconds, not milliseconds. Fetch on demand per page; never poll the
  whole table.
- **Screen size.** The TX15 is not 480x272. Use `lvgl.LCD_SCALE` and
  `lvgl.PERCENT_SIZE` everywhere, or the layout will be wrong on this specific
  radio.
- **`.luac` nesting.** Very large or deeply nested `lvgl.build` tables can work
  from `.lua` and fail compiled. Split into several smaller `build` calls.
- **Never put safety on the Lua path.** A script can be exited, crash, or be
  starved. Arm, disarm, ESTOP and the rescue combo stay on physical channels and
  firmware interlocks. The Lua GUI is an *instrument*, not a control.

---

## 3. Native telemetry (works with zero Lua)

EdgeTX decodes standard CRSF frames into named sensors automatically. Emitting
these from the Teensy gets you a working telemetry screen, logical switches,
voice callouts and SD logging before any script exists. Verify exact field
scaling against `radio/src/telemetry/crossfire.cpp` when you implement.

| CRSF frame | EdgeTX sensors | Robot mapping |
|---|---|---|
| `ATTITUDE` (0x1E) | Ptch / Roll / Yaw | direct — pitch and roll are *the* state variables here |
| `FLIGHT_MODE` (0x21, text) | FM | state name; swap to the fault name while in ESTOP |
| `BATTERY_SENSOR` (0x08) | RxBt / Curr / Capa / Bat% | 24 V pack — and the README's thermal analysis makes bus current worth watching live |
| `LINK_STATISTICS` (0x14) | RQly / RSSI / TPWR / SNR | free |

Everything past that — `gain_sched_alpha`, `hip_l/r_torque_nm`,
`wheel_vel_avg_ms`, `vel_glitch_count`, `fault_code` as a number — goes in a
custom frame decoded by the Lua widget. Resist the temptation to smuggle them
through GPS/vario fields; it works, and it makes the logs lie.

**Do not try to mirror the full 247-byte payload at 50 Hz.** Pick ~10 fields at
5–10 Hz. The `.wlog` on the robot remains the authoritative record.

---

## 4. Sounds, alarms, haptics

This is where the radio earns its place, because when the robot is balancing you
are looking at *it*, not at a screen.

Format: `/SOUNDS/en/`, filenames ≤ 6 chars + `.wav`, 32 kHz (or 16/8), 16-bit,
mono, PCM. Model-specific sounds go in `/SOUNDS/en/<Model_Name>/`.

### State callouts
One Play Track special function per state transition, driven by logical switches
on the FM sensor: *"standby" / "calibrating" / "standing up" / "running" /
"jumping" / "disarming" / "estop"*. Matches the LED/Neopixel colour table so
audio and light agree.

### Fault announcements — the highest-value item here
Sixteen fault codes currently mean walking back to the laptop to find out which
one fired. Send `fault_code` as a numeric sensor, then one logical switch +
Play Track per code: *"pitch watchdog" / "wheel runaway" / "hip feedback lost" /
"standup failed" / "jump timeout"* …

Tier the alert by severity so you know the recovery before you reach the robot:
- **SOFT** → single beep
- **REPOSITION** → two-tone + spoken fault
- **GUI_FIX** → spoken fault, no alarm
- **REBOOT** → siren + haptic (if the unit has a vibration motor) + spoken fault

### Predictive warnings the current stack lacks
- **Pitch approaching the watchdog.** A rising-frequency tone (`playTone` with
  `freqIncr`) as pitch crosses the barrier threshold and climbs toward
  `pitch_wd_bwd/fwd` — a stall-warning analogue. You get an audible cue *before*
  the ESTOP, which is exactly the window where you can catch it.
- **Hip thermal.** Per the measured table, 3.0 N·m ≈ 9.2 W and ~64 °C rise, and
  4.07 N·m holds settle near 143 °C — past the Class B limit if sustained.
  Add a **hip-hold timer**: an EdgeTX timer that runs only while
  `max(hip_l_torque_nm, hip_r_torque_nm) > 3.0`, with a voice callout at 20 s
  against the ~30 s winding time constant. That directly operationalises the
  README's warning about crouch-hold poses.
- **Wheel glitch rate.** Callout when `vel_glitch_count` climbs — the bench log
  showed it rising to 14.6% of samples before a spurious runaway trip. Hearing it
  build gives you a chance to stop before the fall.
- **ESP32 link degraded** — announce on `esp32_link_ok` going false, so "GUI went
  quiet" and "robot went quiet" are distinguishable in the field.

### Model checklist
`/MODELS/<ModelName>.txt` displays on model select with **Display checklist**
enabled. Put the bring-up ladder in it: the suspended sign check at
`jump_effort` 0.05, "confirm `gain_sched_alpha` increases during
`jump_state == 1`", the ODrive heartbeat confirmation, "flash Teensy and ESP32
together". The radio then refuses to let you skip reading it.

---

## 5. Other capabilities this opens up

- **Radio-side SD logging.** EdgeTX logs every telemetry sensor to `/LOGS/*.csv`.
  An independent second record to cross-check the robot's own `.wlog` — different
  clock, different path, catches whole classes of "the log lied" bugs. Trigger
  from the same CH5 switch so both logs start together.
- **Multiple models = multiple safety envelopes.** `BENCH-NOMOTORS`,
  `BENCH-1LEG` (right-leg-only calibration), `GROUND-FULL`, `JUMP-LADDER` — each
  with its own channel limits, switch warnings, checklist and sound set.
  Switching robot configuration becomes selecting a model.
- **Gyro (ICM-42607-C) TiltX/TiltY.** Two genuine uses: tilt-to-command the roll
  setpoint as an active-suspension demo, and a dead-man — if the radio is laid
  flat on a bench, force CH10 low.
- **RGB gimbal LEDs (TX15 MAX), Lua-controlled.** Mirror the canonical state
  colour on the radio itself. Peripheral-vision state indication while your eyes
  are on the robot.
- **Trainer / buddy-box.** A second radio as a spotter's kill switch is a real
  safety upgrade for jump testing. Wired trainer port works; confirm what the
  TX15 supports wirelessly given its ELRS-only internal module.
- **Global variables + flight modes** to mirror the three profiles, so
  `profileN_vel_max` / `profileN_roll_max` have a visible TX-side counterpart.

---

## 6. New firmware surface this asks for

Small, but be explicit about it:

1. **CRSF driver** replacing `IBus`, with `alive()`/`channel()` failsafe parity.
2. **CRSF command tunnel** — unwrap tunnelled frames, dispatch to the existing
   `on_command()`. No new command semantics.
3. **CRSF telemetry emitter** — the four standard frames + one custom frame.
4. **CH11 hard ESTOP** (optional but recommended) — a level-triggered channel
   raising `FAULT_HUMAN_ESTOP` from any energetic state, independent of CH10's
   disarm path. Today a radio ESTOP requires the two-stick rescue combo, which
   is armed only in STANDBY/ESTOP; there is no single-motion panic input while
   running.
5. **Lua emitter in `generate_protocol.py`.**
6. `radio_channels.md` and this README updated in the same change — per the
   repo's own AI-maintenance note.

---

## 7. Build order

| Phase | Work | Gate before proceeding |
|---|---|---|
| **A** | CRSF driver, channel map, failsafe parity | No Motors preset; `stress_test_arm.py` green; rescue combo verified at real endpoints |
| **B** | Native telemetry frames + sounds + checklist | Sensors discovered; fault callouts fire for a deliberately injected fault |
| **C** | Lua status page (read-only) | Runs 10 min without starving the Lua VM; no link degradation |
| **D** | CRSF command tunnel + params + live-tune page | Param write round-trips with `COMMAND_RESULT`; latch confirmed against a known param |
| **E** | Jump page + presets | Suspended sign check passes through the radio UI |

---

## 8. Risks, honestly

- **The command tunnel is the main unknown.** ELRS relays *some* extended-header
  CRSF frames (the addressed 0x28–0x2D range is how the ELRS Lua config script
  works) but not arbitrary types. **Prototype this first**, before designing any
  UI around it — a Lua script that pushes one frame and confirms the Teensy saw
  it. If it doesn't relay, the fallback is a custom frame type the RX passes
  through, or MAVLink-over-CRSF, and the whole GUI plan changes shape.
- **Bandwidth reality check.** Don't design a UI that assumes the WiFi GUI's data
  rate. Everything on the radio should degrade gracefully to "stale".
- **Two sources of truth.** TX-side logical switches that auto-disarm can fight
  firmware interlocks and produce confusing behaviour. Keep firmware
  authoritative; use TX-side logic for *annunciation*, not control.
- **Failsafe regression risk is the serious one.** The iBUS→CRSF swap touches the
  exact code path that keeps a dead radio from looking like a valid arm command.
  Unit-test it, then bench-test it by pulling the RX power with the robot on a
  stand and motors disabled.
