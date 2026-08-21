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

### Shoulder switches — nothing to do
- [x] The factory fit is **SF momentary, SE latching**, which is exactly the
      pairing this model wants: jump on SF (rising edge), ARM on SE (level).
      The model follows the hardware rather than asking you to change it, so
      no disassembly and no chance of fitting the wrong switch.
- [ ] Only if you swap the included alternative panels: re-check the model's
      switch warning afterwards.

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
- [ ] Flick SC → siren + haptic on the radio. CH11 is now live in firmware
      too, so once bound this really does stop the robot.
- [ ] Flick SB down → hear "logging", and confirm a file appears in `/LOGS`.
- [ ] Walk both stick combos and confirm they reach real endpoints:
      **rescue** = CH3+CH2 full up, CH1+CH4 full down; **calibration** = the
      exact mirror. Read the microseconds, not the stick position.

---

## Phase A — the blocking change: iBUS → CRSF

**Firmware done.** `teensy/src/crsf_protocol.h`, `Crsf.h`, and the 24 call
sites in `main.cpp` are ported and building; 26 native tests cover the
failsafe rules. What remains needs hardware.

- [x] CRSF parser replacing `IBus`, interface-compatible so every safety call
      site kept its shape.
- [x] Failsafe parity: `alive()` false on timeout, on LQ 0, and before warm-up;
      `channel()` returns 0, not the last value. Tested, including that a dead
      link cannot satisfy either stick combo or arm.
- [x] Ticks normalised to microseconds at the boundary; every `> 1990` /
      `< 1010` literal in `main.cpp` untouched, with a test asserting ~30 ticks
      of margin above the arm threshold.
- [ ] Buy an ELRS receiver matched to the band you bind on. The LR1121 does
      2.4 GHz *or* 900 MHz — 900 is the better choice for a ground robot
      behind furniture and legs; 2.4 for lower latency.
- [ ] Wire to a Teensy UART using **both** TX and RX. `Serial4` RX is already
      committed; its TX pin is the natural pairing. Confirm nothing else
      claims it in `config.h`.
- [ ] Set the receiver's own failsafe to **no pulses**, never "hold". The
      firmware's whole safety story assumes a dead link looks dead.
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

**Firmware and HUD done.** Scaling was read out of EdgeTX's own
`crossfire.cpp` rather than assumed, and the HUD decodes both paths.

- [x] Emit `ATTITUDE` (0x1E), `BATTERY_SENSOR` (0x08) and `FLIGHT_MODE`
      (0x21). `LINK_STATISTICS` (0x14) is *consumed*, not emitted — the
      receiver sends it, and uplink LQ feeds the liveness rule.
- [x] Private frame 0x24 for the robot numerics, ~350 B/s total.
- [x] HUD decodes 0x24 via `crossfireTelemetryPop()`, and falls back to the
      `FLIGHT_MODE` text for state and fault when it is absent.
- [ ] **Verify ExpressLRS actually relays frame type 0x24.** This is the one
      unknown in Phase B. If it does not, state, fault, attitude and pack all
      still work off the standard frames, and only the extra numerics are
      lost. Fallbacks: a standard frame type, or MAVLink-over-CRSF.
- [ ] Confirm EdgeTX discovers `Ptch`, `Roll`, `Yaw`, `RxBt`, `Bat%`, `FM`,
      `RQly`, `1RSS`, `TPWR`.
- [ ] **Leave the `Ptch`/`Roll`/`Yaw` sensor unit at RAD.** The wire carries
      radians and the HUD converts; switching the sensor to degrees makes
      EdgeTX convert too and the HUD double-counts.
- [x] Wheel-velocity glitch count wired through. It already existed in
      `lib/WheelMotors` (`wm_L/R.vel_glitch_count`); both axes are summed and
      sent 16-bit saturating, so `GLCH` and the "glitches rising" callout are
      live.
- [x] Bus current: sent as CRSF "no data" rather than a fake zero, so EdgeTX
      creates no `Curr` sensor at all.
- [ ] *Optionally* instrument bus current for real. ODrive `Get_Iq` (CAN
      `0x014`) gives phase current, not bus current, and adds periodic traffic
      to the control-critical CAN3 bus — a shunt on the pack would be the
      honest answer. Only worth it if you want live power/thermal numbers.
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
- [x] **CH11 hard ESTOP** — level-triggered, raises `FAULT_HUMAN_ESTOP` from
      any non-ESTOP state and blocks arming while held. Guarded by `alive()`
      so a dead link cannot assert it. Bench-verify it once bound.
- [ ] CH12 is still unassigned. `roll_ctrl_en` is persistent, so wiring a
      switch to it would rewrite the stored value at boot from whatever
      position the switch is in. Give it a non-persistent runtime gate first
      if you want the switch.
- [ ] Lua emitter in `generate_protocol.py`.
- [ ] Update `radio_channels.md` and `firmware/robot_teensy/README.md` in the
      same change, per the repo's own AI-maintenance note. Worth fixing the
      "up" wording while you're there — see the polarity note in
      [CHANNELS.md](CHANNELS.md).

---

## RGB push buttons

- [x] **SG/SH/SI → live-tune group 0/1/2** (CH13–15), mutually exclusive
      switch group, amber when lit. The lit button is the group indicator.
- [x] **SJ → commit tuned gains** (CH16), momentary, green.
- [ ] Confirm on the radio that the exclusive group behaves as expected: press
      one, the others release; press the lit one again, all off.
- [ ] SK/SL are free but **all 16 channels are used**. They'd need a channel
      freed or the command tunnel. Candidates if one comes free: `jump_enable`
      as a latched magenta "JUMP ARMED" lamp, or `roll_ctrl_en` (needs a
      non-persistent gate first).
- [ ] Mirroring robot *state* on the LEDs is still unproven — `FUNC_RGB_LED`
      exists and `onColorLuaOverride` in `customSwitches` hints Lua can drive
      them, but the binding has not been checked on the radio.

## Loose ends, no rush
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
