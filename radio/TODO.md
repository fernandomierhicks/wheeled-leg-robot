# TX15 TODO

Ordered by what you can actually do next. Everything in "Now" needs no
firmware, no robot power, and no CRSF. Phases map to
`tx15-robot-integration-plan.md` §7.

Package installed on the card and verified byte-for-byte: 68 files, 0
overwrites. See [INSTALL.md](INSTALL.md) for the copy procedure and
[CHANNELS.md](CHANNELS.md) for the control map.

---

## Now — on the radio, nothing else powered

### Finish the install
- [ ] Eject USB, exit USB mode on the radio.
- [ ] **Radio settings → Themes → RoboBlue → Set as default.**
      Theme choice lives in `radio.yml`, which the installer deliberately
      never writes.
- [ ] **Model select → WLR ROBOT.** Confirm the bring-up checklist appears.
      If it doesn't, the `.txt` filename no longer matches the model *name* —
      EdgeTX builds that path from the name, not the filename.
- [ ] Confirm the model actually appears at all. EdgeTX auto-discovers
      `model<digits>.yml` files that aren't in `labels.yml`; that logic was
      read from the 2.12 source and 3.0 isn't public, so this is the one
      unverified assumption in the package. If it doesn't show: create an
      empty model on the radio, note the slot it makes, and copy
      `sdcard/MODELS/model90.yml` over that file instead.
- [ ] Add the HUD to a screen if it isn't already on screen 1, and confirm it
      renders (it will say **NO ROBOT TELEMETRY** — that's correct).

### Shoulder switches — the highest-value hardware change on the radio
The TX15 ships with alternative momentary and 2-position switches plus matching
panels; the case opens with four hex screws. ~20 minutes.

- [ ] Install **momentary at SE** (jump). Jump is edge-triggered; a latching
      switch here can sit latched where you cannot see it.
- [ ] Install **2-position latching at SF** (ARM). Must hold while you drive
      and be index-finger reachable for an instant kill.
- [ ] **Radio settings → Hardware**: set each switch's type to match.
- [ ] If your physical shoulder switches turn out to be different letters,
      change the two `srcRaw` values in the model — nothing else moves.

### Validate the channel map with nothing powered
- [ ] **Screen 2 (Outputs).** Move every stick, switch and pot. Confirm each
      drives the channel [CHANNELS.md](CHANNELS.md) says it should.
- [ ] Confirm the polarity rule holds: **switch up = channel low = safe**, on
      all of SA–SF.
- [ ] Power-cycle with a switch left down → the **switch warning** must fire
      and block boot.
- [ ] Power-cycle with the hip-height stick away from the retracted end → the
      **throttle warning** must fire.
- [ ] Flick SF → hear "armed"; flick back → "disarmed". These run off logical
      switches on CH10 and need no telemetry.
- [ ] Flick SC → siren + haptic. (CH11 is inert on the robot side; this is
      only checking the annunciation path.)
- [ ] Flick SB down → hear "logging", and confirm a file appears in `/LOGS`.
- [ ] Walk both stick combos and confirm they reach real endpoints:
      **rescue** = CH3+CH2 full up, CH1+CH4 full down; **calibration** = the
      exact mirror. Read the microseconds, not the stick position.

---

## Phase A — the blocking change: iBUS → CRSF

Everything downstream depends on this. Plan §0.

- [ ] Buy an ELRS receiver matched to the band you bind on. The LR1121 does
      2.4 GHz *or* 900 MHz — 900 is the better choice for a ground robot
      behind furniture and legs; 2.4 for lower latency.
- [ ] Wire to a Teensy UART using **both** TX and RX. `Serial4` RX is already
      committed; its TX pin is the natural pairing. Confirm nothing else
      claims it in `config.h`.
- [ ] Replace the `IBus` object with a CRSF parser (`0xC8` sync, CRC-8 poly
      `0xD5`, type `0x16` for 16×11-bit channels).
- [ ] **Preserve the failsafe semantics exactly.** `alive()` must go false on
      link timeout, and `channel()` must return **0, not the last value**,
      when not alive. Both the rescue-combo gate and `radio_update()`'s
      disarm interlock depend on it. Write the unit test before wiring a
      motor.
- [ ] Normalise CRSF ticks (172–1811) to microseconds **at the boundary**.
      Leave every `> 1990` / `< 1010` literal in `main.cpp` alone.
- [ ] Bind, then verify each channel lands on the intended function via the
      read-only `PARAM_IBUS_CH*` mirrors — with the GUI **No Motors** preset
      set (`hip_l/r_enable=0`, `wheel_l/r_enable=0`, `calib_bypass_en=1`,
      `run_wheel_bypass_en=1`).
- [ ] `stress_test_arm.py` green.
- [ ] Bench-test the regression that matters: pull RX power with the robot on
      a stand, motors disabled, and confirm it disarms. A dead radio must
      never look like a valid arm command.

---

## Phase B — native telemetry, then the HUD lights up

- [ ] Emit the four standard CRSF frames: `ATTITUDE` (0x1E), `FLIGHT_MODE`
      (0x21), `BATTERY_SENSOR` (0x08), `LINK_STATISTICS` (0x14). Verify
      scaling against `radio/src/telemetry/crossfire.cpp`.
- [ ] Confirm EdgeTX discovers `Ptch`, `Roll`, `Yaw`, `RxBt`, `Curr`, `Bat%`,
      `RQly`, `1RSS`, `TPWR`. The HUD picks them up with no changes.
- [ ] Emit one custom frame carrying `Stat`, `Flt`, `Alph`, `HipL`, `HipR`,
      `WVel`, `Jump`, `SUp`, `Hlth`, `Glch`, `E32`, `Prof`. **~10 fields at
      5–10 Hz — do not mirror the 247-byte payload at 50 Hz.** The `.wlog` on
      the robot stays the authoritative record.
- [ ] Names must match the `SENSOR` table at the top of
      `sdcard/WIDGETS/WLRHUD/telem.lua`, or change that table.
- [ ] Inject a deliberate fault and confirm the radio speaks the right one at
      the right tier.
- [ ] Confirm the HUD stops saying NO ROBOT TELEMETRY and every reading shows
      LIVE rather than amber/grey.
- [ ] Resist smuggling extra fields through GPS/vario sensors. It works, and
      it makes the logs lie.

---

## Phase C — soak the HUD

- [ ] Leave it running 10 minutes. No Lua VM starvation, no link degradation.
- [ ] Confirm the predictive warnings fire sensibly and aren't chatty: pitch
      tone toward the watchdog, hip thermal after 20 s over 3.0 N·m, wheel
      glitch rate. Tune the thresholds at the top of `annunc.lua` if they
      nag.
- [ ] Confirm annunciation still works when the HUD is *not* the visible
      screen (that's what `background()` is for).

---

## Phase D — command tunnel

**Prototype this before designing any UI around it** (plan §8). ELRS relays
*some* extended-header CRSF frames — the addressed `0x28`–`0x2D` range is how
the ELRS Lua config script works — but not arbitrary types.

- [ ] Write a throwaway Lua script that pushes one frame with
      `crossfireTelemetryPush()` and confirm the Teensy saw it. If it doesn't
      relay, the fallbacks are a custom pass-through frame type or
      MAVLink-over-CRSF, and the whole GUI plan changes shape.
- [ ] CRSF-side adapter that unwraps a tunnelled frame and hands it to the
      existing `on_command()`. **No new command semantics.**
- [ ] Add a Lua emitter to `protocol/generate_protocol.py` so the radio's
      param page inherits the same property the GUI Params tab has — new
      params appear without a hand-written widget, and `--check` catches
      staleness.
- [ ] Params page, generated. Read-only params greyed, not hidden.
- [ ] Live-tune page: two slots, knob position vs target, pickup marker and a
      **PICKED UP** badge, group indicator, and a long-press LATCH button with
      read-back. The current mechanism is invisible and this is the biggest
      single quality-of-life win.
- [ ] Param write round-trips with `COMMAND_RESULT`; latch confirmed against
      a known param.

---

## Phase E — jump and presets

- [ ] Jump page: `jump_enable` toggle and a `jump_effort` slider stepped
      0.5 → 0.6 → 0.7 → 0.8 → 0.9 → 1.0.
- [ ] Guard that refuses to raise effort until it has seen `jump_state == 1`
      produce an *increasing* `gain_sched_alpha` at least once this session.
      That puts the sign-check discipline in the tool instead of in your
      memory.
- [ ] Presets page: **No Motors** / **Full Robot**, writing and read-verifying
      the same set the GUI does.
- [ ] Suspended sign check passes through the radio UI.

---

## Firmware surface this asks for

Small, but be explicit (plan §6):

- [ ] CRSF driver replacing `IBus`, with `alive()`/`channel()` failsafe parity.
- [ ] CRSF command tunnel → existing `on_command()`.
- [ ] CRSF telemetry emitter — four standard frames + one custom.
- [ ] **CH11 hard ESTOP** — level-triggered, raising `FAULT_HUMAN_ESTOP` from
      any energetic state, independent of CH10's disarm path. Today there is
      no single-motion panic input while running; the rescue combo is armed
      only in STANDBY/ESTOP.
- [ ] Lua emitter in `generate_protocol.py`.
- [ ] Update `radio_channels.md` and `firmware/robot_teensy/README.md` in the
      same change, per the repo's own AI-maintenance note. Worth fixing the
      "up" wording while you're there — see the polarity note in
      [CHANNELS.md](CHANNELS.md).

---

## Loose ends, no rush

- [ ] **Six RGB function switches SG–SL** are unassigned. `hal.h` shows the
      TX15 has them with RGB LEDs (`FUNCTION_SWITCHES_WITH_RGB`); the
      integration plan predates this and doesn't account for them. Good fit
      for mirroring robot state in peripheral vision via the `RGB_LED`
      special function, or as latched bench toggles.
- [ ] **Crouch-biased curve on CH3.** There's 30% expo instead. Draw a 5-point
      curve on the touchscreen (Model → Curves) and point the Hip input's
      expo at it — safer than hand-editing the curve point pool in YAML.
- [ ] **CH12 spare** — earmarked for `roll_ctrl_en`.
- [ ] **Extra models as safety envelopes** (plan §5): `BENCH-NOMOTORS`,
      `BENCH-1LEG`, `JUMP-LADDER`, each with its own channel limits, switch
      warnings, checklist and sound set. Clone WLR ROBOT on the radio.
- [ ] **Gyro TiltX/TiltY** — tilt-to-command the roll setpoint as an
      active-suspension demo, and a dead-man that forces CH10 low if the radio
      is laid flat on the bench.
- [ ] **Trainer / buddy-box** — a second radio as a spotter's kill switch is a
      real safety upgrade for jump testing. Confirm what the TX15 supports
      wirelessly given its ELRS-only internal module.
- [ ] Wire `o_pickup`, `o_latch`, `o_nomtr`, `o_full` — built and on the card,
      unused until Phases D/E give them something to announce.

---

## Keep the package honest

Run before any commit that touches `radio/` or `protocol/schema.json`:

```
python radio/tools/gen_robotdef_lua.py --check
python radio/tools/make_sounds.py --check
python radio/tools/check_lua.py
python radio/tools/check_sdcard.py
```

`gen_robotdef_lua.py` fails on purpose if `fault_severity()` in
`comm_protocol.h` or `mode_color()` in the ESP32 source has drifted from the
radio's copy. A fault the radio names wrongly is worse than one it can't name
at all — consider wiring these into CI alongside the existing
`generate_protocol.py --check`.
