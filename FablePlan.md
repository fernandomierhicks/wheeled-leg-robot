# Robot Software Audit — Findings & Action Plan

## Context

Full review of the three software deployments of the self-balancing wheeled-leg robot, requested to (a) flag outdated docs, (b) find safety issues, (c) assess state-machine and comms robustness, (d) sanity-check the control cascade, (e) find refactor/single-source-of-truth opportunities, and (f) produce an overview MD that future Claude sessions can use as an entry point. Second pass added: GUI performance analysis, a consolidated double-command audit, a GUI-side 50 Hz recording feature, and UI redesign proposals.

**What was reviewed (read line-by-line):**
- **Teensy 4.1** (`firmware/robot_teensy/teensy`): main.cpp, state_machine.cpp/h, control_loop.cpp, config.h, robot_state.h, IBus.h; libs HipMotors, WheelMotors, Calibration, ParamRegistry, IMU, SDLogger, Esp32Link; the jrullan StateMachine library internals.
- **Shared**: comm_protocol.h, CommLink.cpp/h, udp_stream.h.
- **ESP32** (`firmware/robot_teensy/esp32`): main.cpp (comm/WiFi/task logic in full; display drawing skimmed), config.h, wifi_config.h, platformio.ini.
- **GUI** (`software/gui`): comm_commands, flash_monitor (PacketDecoder), telem_format, wifi_transport, source_manager, port_manager, telemetry_bus, hip_motors, wheel_motors, raw_data_tab, imu_tab, robot_visualizer_tab (update paths), log_playback, main.py.
- **Docs**: README.md, state_machine.md, pinout.MD, radio_channels.md, gui/CLAUDE.md, root CLAUDE.md, docs/other/Control.MD, lib READMEs, TODO.txt.

**Overall verdict:** well-engineered codebase — the FSM request-latch guards, propagation checklist, split-telemetry design, per-motor enable flags, and SD logger are all thoughtfully done. Findings are mostly edge cases and drift, but four are genuine safety bugs (F1–F4).

---

## 1. Safety Findings (can motors move outside known states?)

Ordered by severity. Each includes the failure scenario and proposed fix.

### F1 — IMU silence is never detected (dead watchdog) — **CRITICAL** — ✅ FIXED 2026-07-13
[IMU.cpp:222](firmware/robot_teensy/teensy/lib/IMU/IMU.cpp#L222)
```cpp
if (now - _last_update_ms > TIMEOUT_MS && now - _last_poll_ms >= TIMEOUT_MS) {
```
`_last_poll_ms = now` is set every tick at line 179, so `now - _last_poll_ms` is always **0**, and `0 >= 100` is always false → the ERROR-on-silence branch is unreachable. If the IMU cable breaks or the sensor hangs mid-RUNNING, `imu_state()` stays NOMINAL and **the LQR balances on frozen pitch** — the robot drives off/falls, and the pitch watchdog can't fire either (pitch frozen below 50°). Only the wheel-runaway watchdog remains as a backstop.
**Fix:** the second condition inverts its own intent; remove it or replace with a "we are actively polling" check (`now - _last_poll_ms < 10`). Additionally: add a `RUNNING/JUMPING → ESTOP` transition when `imu_state() != NOMINAL` (new fault, e.g. `FAULT_IMU_LOST`) — today no state except STARTUP reacts to IMU health at all.

### F2 — Stale hip command executes on MANUAL entry — **HIGH** — ✅ FIXED (via `cmd_allowed()` permission matrix + pending-clear on MANUAL entry)
[main.cpp:127-140](firmware/robot_teensy/teensy/src/main.cpp#L127-L140) accepts `CMD_ID_HIP` in **any** state and latches `g_hip_cmd.pending = true`. It is only consumed in [`on_manual()`](firmware/robot_teensy/teensy/src/state_machine.cpp#L151-L176). A hip MIT command sent while in STANDBY/RUNNING/ESTOP sits latched indefinitely and **fires the instant MANUAL is next entered** — unexpected motor motion. (The GUI disables its buttons outside MANUAL, but firmware must not rely on that; WiFi TCP is open to any LAN client.)
**Fix:** in `on_command()`, ignore `CMD_ID_HIP` unless `g_state.state == STATE_MANUAL` (log + reject otherwise), and clear `g_hip_cmd.pending` on MANUAL entry — matching how `CMD_ID_WHEEL` is already gated. See also §6 "command permission matrix" for the systematic version.

### F3 — `s_req_jump` survives ESTOP → unintended jump on next arm — **HIGH** — ✅ FIXED (all 9 request flags flushed in `on_estop()`)
[`on_estop()`](firmware/robot_teensy/teensy/src/state_machine.cpp#L329-L347) flushes `s_req_cmd_reject/manual/running/calibration` but **not `s_req_jump`, `s_req_disarm_running`, `s_req_standby`**. Scenario: in RUNNING, CH6 jump request latches; same tick a fault fires ESTOP (`req_estop` is evaluated first, so `req_jump` is never cleared). After reset → re-arm, the stale `req_jump()` fires immediately → **robot jumps the moment it is armed** (if `PARAM_JUMP_ENABLE=1`). Stale `s_req_disarm_running` causes instant disarm-on-arm; stale `s_req_standby` kicks you out of the next MANUAL/CALIBRATION session.
**Fix:** flush **all** request flags in `on_estop()` (and arguably in `on_standby()` too).

### F4 — Radio disarm is missed if CH10 drops during JUMPING — **HIGH** — ✅ FIXED (level-based disarm)
[main.cpp:547](firmware/robot_teensy/teensy/src/main.cpp#L547): disarm is **edge-triggered** and gated on `state == STATE_RUNNING`. If the operator drops CH10 (or the radio dies) while in JUMPING, the edge is consumed without effect; the jump completes → back to RUNNING → **robot stays armed with the arm switch down** until the operator re-toggles CH10.
**Fix:** make disarm level-based: every tick, `if (state == RUNNING && !(alive && ch10 > 1990)) disarm;` — the jump still finishes, then this catches it on RUNNING re-entry.

### F5 — Wheel feedback loss / ODrive error in RUNNING doesn't ESTOP — **MEDIUM** — ✅ FIXED (`running_wheel_fault()` → `FAULT_WHEEL_FEEDBACK_LOST` 0x0B, + `ever_heard` guard)
[wheel_motors.cpp:96-115](firmware/robot_teensy/teensy/lib/WheelMotors/wheel_motors.cpp#L96-L115): on encoder timeout or axis error, the driver silently forces `WheelMode::IDLE` and only prints to raw USB `Serial`. The state machine stays in RUNNING, `wheel_motors_send()` becomes a no-op, and the robot **falls with no fault code** until the pitch watchdog fires ~200 ms later. Contrast: hip feedback loss correctly ESTOPs via `standby_hip_fault()`.
**Fix:** add a `wheel_motors_fault()` predicate mirroring `standby_hip_fault()`, wired as a RUNNING/JUMPING → ESTOP transition with a new `FAULT_WHEEL_FEEDBACK_LOST`. Also: `wm_*.ok` is spuriously true for the first `enc_timeout` ms after boot (`last_fb_ms == 0`, no `ever_heard` guard like hips have).

### F6 — `CMD_ID_REBOOT` accepted while RUNNING — **MEDIUM** — ✅ FIXED (matrix-gated to STARTUP/STANDBY/ESTOP + MIT-exit/wheel-IDLE before reset)
[main.cpp:117-124](firmware/robot_teensy/teensy/src/main.cpp#L117-L124): a reboot mid-balance resets the MCU with hips still in MIT holding the last setpoint and ODrives in closed-loop TORQUE with the last torque. Whether the wheels stop depends on the (unverified) ODrive watchdog config.
**Fix:** reject REBOOT unless in STANDBY/ESTOP/STARTUP; before resetting, call `hip_motors_exit_mit()` + `wheel_motors_set_mode(IDLE)`.

### F7 — MANUAL GUI watchdog is 5 s, docs say 500 ms — **MEDIUM** — ✅ FIXED (500 ms; new `CMD_ID_PING` 0x02 sent by GUI at 10 Hz — Teensy and GUI must be updated together)
[state_machine.cpp:107](firmware/robot_teensy/teensy/src/state_machine.cpp#L107) `MANUAL_GUI_TIMEOUT_MS = 5000`. In MANUAL + wheel VELOCITY mode, a GUI crash leaves wheels spinning at the last setpoint for up to 5 s. README.md and state_machine.md promise 500 ms.
**Fix:** decide the intended value (500–1000 ms is reasonable) and align code + docs. Consider also the **ODrive onboard watchdog** so wheels stop even if the Teensy itself hangs — see §3.

### F8 — Param out-of-bounds ESTOP path is dead code — **LOW (docs/expectations)** — ✅ FIXED (flag + fault code deleted, 0x07 reserved; writes always clamp; docs updated)
`PARAM_FLAG_FAULT_ON_BOUNDS` is defined ([param_registry.h:9](firmware/robot_teensy/teensy/lib/ParamRegistry/param_registry.h#L9)) but **no param sets it** → `param_set()` never returns `FAULT`, `FAULT_PARAM_OUT_OF_BOUNDS` is unreachable, and out-of-range writes silently clamp. state_machine.md's "hidden ESTOP path" describes behavior that cannot happen.
**Fix:** either set the flag on safety-critical params or delete flag + fault code and fix docs. Recommend the latter — clamping is the safer, simpler behavior.

### F9 — Open TCP command port / UDP broadcast on WLAN — **LOW (accepted risk)** — ✅ DONE (doc note in firmware README; hardening deferred)
Any device on the home network can connect to `<esp32>:5006` and issue mode changes, MIT hip commands, or reboots; telemetry broadcasts to 255.255.255.255. Fine for a home lab; worth a doc note, and cheap to harden (accept only first client, or require a magic token as the first bytes of a TCP session).

### F10 — `hip_motors_send()` both-motor variant gates only on L — **LOW** — ✅ FIXED (per-motor gating; HipMotors README updated)
[hip_motors.cpp:239-245](firmware/robot_teensy/teensy/lib/HipMotors/hip_motors.cpp#L239-L245): `if (!hm_L.mit_active) return;` then sends **both** motors. In today's bench config (left leg disabled) it no-ops entirely even though R is active. Only used by `test/test_hip_motor` — fix the gating or delete it and update the test + HipMotors README.

**Direct answer to "can we move motors outside of known states?":** Yes, three ways today — F2 (stale hip cmd on MANUAL entry), F3 (stale jump on arm), and the F7 window (wheels coast on last command up to 5 s after GUI death). *(All three closed as of 2026-07-13 — pending bench verification, §15.)* Everything else is properly gated: torque paths only run inside RUNNING/JUMPING state actions, wheel commands are MANUAL-gated, hip position setpoints clamp to calibrated limits, MIT frames clamp to protocol limits, ESTOP is reachable from every state and evaluated first.

---

## 2. State Machine Review

**Verdict: fundamentally solid.** Verified in the library source: `run()` executes state logic **then** evaluates transitions in registration order, first-true wins, exactly one transition per tick. So: ESTOP transitions registered first always win the tick ✓; a fault raised inside `controlLoop_run()` takes effect the same tick ✓; no state skipping ✓.

**Strengths worth keeping:** every request setter is state-guarded against stale latching (with excellent comments); double command delivery is absorbed; `req_estop()` preserves pre-set fault codes; the `on_estop` flush concept is right (just incomplete — F3).

**Race conditions:** none of consequence. The `volatile bool` request flags are only written from the main loop (CommLink callbacks run from `update()` in-loop, not ISRs). The only true ISR/mainline shared state is `hm_*`/`wm_*` written from FlexCAN RX interrupts — fields are 32-bit (atomic on Cortex-M7) but not `volatile`; low practical risk, worth marking `volatile`.

**Gaps (beyond §1):**
- `startup_fail()` uses absolute `millis() > 2000` and `ever_heard` is sticky — the 2 s "init window" only exists on first boot; after ESTOP→reset with a hip powered off, the fault is caught one state later by `standby_hip_fault()`. Works; consider tracking time-of-STARTUP-entry.
- **CALIBRATION double MIT frames — ✅ FIXED 2026-07-13 (D1):** each motor got **two** MIT frames per tick — the zero-torque ping from `hip_motors_poll()` (setpoint cache inactive) *plus* the direct `hip_motor_send_L/R()` ramp from `calibration_update()`, alternating zero-gain/seek-gain every ~500 µs. **Fixed:** calibration now routes through `hip_motors_set_setpoint_L/R()` — the poll loop sends exactly one frame per motor per tick. The cache is also refreshed immediately after each encoder `zero()` so the stale pre-zero-frame target is never re-sent in the new frame. **Bench re-verify required (§15):** effective holding torque *increases* now that the zero-torque ping gaps are gone — stall current thresholds (5 A / 4 A bench values) may need retuning.

---

## 3. Teensy → ESP32 → GUI Chain Review

**Protocol verdict: appropriate and near-solid.** Framed, versioned, length-guarded, parse-timeout-protected, drop-counted, static asserts on both C++ ends, `struct.calcsize` asserts in Python. The split TELEM_A/B design with the FIFO-threshold rationale is exactly right for this hardware. Weaknesses:

- **W1 — GUI does not enforce the checksum. — ✅ FIXED 2026-07-13** PacketDecoder now drops bad-checksum frames and counts them; counters (`CRC drops` / `Seq gaps` / `Pair drops`) surfaced in the Raw Data tab's Frame page and attached to every packet as `link_*` keys.
- **W2 — XOR checksum is weak. — ✅ FIXED 2026-07-13 (FLAG DAY)** All three ends now use table-driven **CRC-8** (poly 0x07, init 0x00, MSB-first — CRC-8/SMBus, check value 0xF4): CommLink.cpp (both firmwares), telem_format.py `crc8()` (GUI build + decode), trigger_log_test.py (inlined). **Old firmware and new GUI (or any mixed pair) reject 100% of each other's frames — reflash Teensy + ESP32 and restart the GUI together.**
- **W3 — `seq` transmitted but never checked. — ✅ FIXED 2026-07-13** CommLink now counts per-link seq discontinuities (`rx_seq_gaps()`); ESP32's TFT GAP counter reads it (replacing the loop_count heuristic — now covers every packet type), and the GUI counts per-transport gaps in PacketDecoder.
- **W4 — A/B pairing doesn't verify adjacency. — ✅ FIXED 2026-07-13** ESP32 and GUI both require `B.seq == A.seq + 1` before merging a telemetry pair; non-adjacent halves are discarded (GUI counts them as `Pair drops`).
- **W5 — No flow control on the SD-log GET path. — ✅ FIXED 2026-07-13** `LOG_SUB_GET` is refused outside STANDBY/ESTOP (WARN + STATUS(1) reply). Pacing is per requesting transport (`source` byte distinguishes direct-USB vs ESP32-relayed): Teensy-USB unthrottled (as before), ESP32 path 1 chunk / 8 ms ≈ 61 kB/s (CP2102 is ~92 kB/s minus 13 kB/s telemetry). Bulk LOG_DATA chunks now go only to the requesting link instead of both.
- **W6 — comm_log truncation — ✅ FIXED 2026-07-13:** buffer raised to 120 B, clamp corrected to `sizeof(msg)-1` (the old clamp sent the NUL); calibration.cpp's limit-dodging comments updated.
- **W7 — Driver messages bypass comm_log — ✅ FIXED 2026-07-13:** WheelMotors init/fault/clear_errors messages now go through `comm_log()` (visible over WiFi + USB + SD log).

**"Are we using the hardware to help us?"** Mostly yes, and well: ESP32 UART RX FIFO threshold tuning + 4096 B ring buffer, Teensy `addMemoryForWrite` TX buffer, FlexCAN FIFO + interrupts, SDIO + RingBuf in DMAMEM, display/LED/ToF pinned to core 0 so core 1 owns UART parsing, DTR handling in the GUI. Unused hardware that would genuinely help:
- **Teensy hardware watchdog** (`Watchdog_t4`) — today a hung control loop leaves AK45s holding torque indefinitely. A ~100 ms HW watchdog is the single biggest robustness win available (AK45s then drop out of MIT ~4 s later; ODrive watchdog idles wheels).
- **ODrive onboard watchdog** (`axis.config.watchdog_timeout`, fed by any CAN cmd) — protects against Teensy hang/harness cut independently. Needs one ODrive USB session (per memory: re-probe hw first) + `save_configuration()`.
- **ESP32 task watchdog** on `loop()` — cheap insurance the bridge never wedges silently.
- Hardware RTS/CTS exists on both UARTs but isn't wired — bandwidth math is fine as documented; **not** recommended.
- **FlySky receiver failsafe**: configure CH10 to output <1000 µs on TX loss (belt-and-suspenders over the CH7/8 sentinel logic).

---

## 4. Control Logic Review

**Verdict: the cascade is wired correctly.** Verified signal-by-signal in [control_loop.cpp](firmware/robot_teensy/teensy/src/control_loop.cpp):
- Velocity PI → `theta_ref` (clamped, rate-limited, anti-windup, reset-on-reversal, reset-when-disabled) ✓
- Yaw PI on `omega_cmd − imu_yaw_rate()` → `tau_yaw` (clamped, anti-windup) ✓
- LQR state `[pitch − theta_ref, pitch_rate, vel_avg − v_ref]`, gains scheduled on calibrated hip span (coordinate-agnostic — nice), `tau_sym = −K·x` ✓
- Mix `tau_L = sym + yaw + FF`, `tau_R = sym − yaw + FF`; right-motor mirroring handled exactly once in the wheel driver (TX and RX) ✓
- FF1 (hip-reaction from measured currents × kt × r/l_eff) and FF2 (gravity comp) match the sim doc's formulas ✓
- Unconditional `wheel_motors_send()` every tick as implicit watchdog pet + stale-torque flush ✓ (good design)

**Issues:**
- **C1 — Controller state not reset on RUNNING entry.** `s_vel_integral`, `s_theta_ref_rlt`, `s_yaw_integral` persist across disarm→arm cycles (reset only when the PI is *disabled*). Reset in `on_running()` when `entering`.
- **C2 — FF terms can exceed `PARAM_LQR_TORQUE_LIMIT`** (added after the tau_sym clamp, re-clamped only at hard 7 N·m). If intentional, document; if not, clamp the sum at the test limit. → open question.
- **C3 — Pitch trim plumbed (CH7 → param → telemetry) but not applied** — TODO at control_loop.cpp:190 is accurate; tracked.
- **C5 — Hip gains in RUNNING hardcoded** (`RUNNING_KP/KD` 5.0/0.5) — the param_ids.h TODO for `PARAM_HIP_RUNNING_KP/KD` stands, and its comment points at the wrong file ("state_machine.cpp:117-119" → control_loop.cpp:16-18).
- The JUMPING phase logic (4-layer torque protection: softstart, distance ramp-out, speed taper, hard margin cutoff) is genuinely well done.

---

## 5. Real-Time Findings (Teensy)

- **R1 — Flash writes inside the 500 Hz loop — worst latent bug outside safety. — ✅ FIXED 2026-07-13** `param_set()` on any PERSISTENT param called `save_to_flash()` **synchronously** — a full LittleFS remove+rewrite. The CH9 profile-switch torque slew calls `param_set(PARAM_LQR_TORQUE_LIMIT, …)` **every tick** while ramping — a 4→1 N·m ramp ≈ 300 consecutive flash rewrites in 0.6 s, *while balancing at the moment of a speed-profile change*. Plus flash wear.
  **Fixed as planned:** `param_set()` now only marks a dirty flag; new `param_flush_service(allow_flush)` writes once changes are quiet for 1 s, called from `loop()` with `allow_flush = state ∉ {RUNNING, JUMPING}`. `param_save_all()` still forces an immediate flush.
- **R2 — No loop-overrun detection. — ✅ FIXED 2026-07-13** `loop()` now counts ticks whose work time exceeds the 2000 µs budget → new `HEALTH_LOOP_OVERRUN` bit (1<<11) in `health_flags` (lit for 1 s after an overrun) + a rate-limited (1 Hz) WARN log with the µs and running count. This is also how you'd *see* R1 regressing.
- **R3 —** SD transfer pacing (W5).

---

## 6. Double-Command / Double-Sequence Audit (consolidated)

Requested sweep for every place two writers command the same actuator or the same sequence runs twice:

| # | What | Where | Verdict / Fix |
|---|---|---|---|
| D1 | **Calibration: ping + direct send, 2 MIT frames/tick/motor** | `hip_motors_poll()` + `calibration_update()` | ✅ **FIXED 2026-07-13** — calibration routed through the setpoint cache (§2); bench re-verify stall thresholds (§15). |
| D2 | **GUI MIT-reinforce timer (4 s) duplicates firmware MIT re-enter (3 s)** | [hip_motors.py:676](software/gui/tabs/hip_motors.py#L676) vs [hip_motors.cpp:157](firmware/robot_teensy/teensy/lib/HipMotors/hip_motors.cpp#L157) | Firmware wins; **delete the GUI timer** (one less command stream; memory note `feedback_ak45_periodic_enter_mit` is satisfied by firmware). |
| D3 | **Wheel cleanup after MANUAL done by both sides**: firmware `on_standby()` (send 0 → IDLE → clear_errors) *and* GUI wheel tab auto-sequence on entering MANUAL (clear_errors → VELOCITY mode → zero, with a fire-and-forget 150 ms `QTimer.singleShot`) | [state_machine.cpp:135-139](firmware/robot_teensy/teensy/src/state_machine.cpp#L135-L139), [wheel_motors.py:735-738](software/gui/tabs/wheel_motors.py#L735-L738) | **Firmware owns motor state.** Demote the GUI auto-sequence to an explicit "Init wheels" button (the singleShot can interleave with a state change mid-sequence). |
| D4 | **Standing CAN traffic**: `wheel_motors_pet_watchdog()` sends zero-vel to both ODrives at 500 Hz whenever mode==IDLE (~1000 frames/s in STANDBY, ~13% of CAN3 @ 1 Mbps); meanwhile nothing pets in MANUAL VELOCITY/POSITION if the GUI is quiet | [main.cpp:499](firmware/robot_teensy/teensy/src/main.cpp#L499) | ✅ **FIXED 2026-07-13** — 50 Hz divider inside `pet_watchdog()` (CAN traffic ÷10). VELOCITY/POSITION are covered by the existing 50 Hz vbus poll (any axis-addressed frame feeds the ODrive watchdog; a zero-vel keepalive there would stomp the live setpoint), TORQUE by the control loop's unconditional send. |
| D5 | Hip zero-torque ping at 500 Hz per motor in MIT | by design (AK45 only replies to commands — feedback source) | Keep; document in HipMotors README (already is). |
| D6 | Telemetry double-send (UART + USB) | intentional, different consumers | Keep. |
| D7 | Command double-delivery GUI→robot | already solved: `send_frame()` sends on exactly one transport (SourceManager-active) | Keep; the FSM latch guards absorb any residual dupes. |
| D8 | Radio stick triple-representation (iBUS params via `param_force_set`, `ibus_ch[]` telemetry, derived `PARAM_V_CMD_MS` etc.) | main.cpp read_sensors/radio_update | Acceptable (RAM-only); note as SSOT observation, no action. |
| D9 | GUI hip wave (50 Hz MIT stream) + firmware 500 Hz re-send of cached setpoint | Setpoint cache makes this correct — GUI updates the target, firmware refreshes the wire | Keep — this *is* the right commanding pattern. |

**Recommendation — a better way to command the robot (unifying fix for F2/F6/D-class bugs):** keep the request-flag + telemetry-echo pattern (it is good — commands are requests, state is truth, telemetry is the ACK), but centralize acceptance in **one command-permission matrix** in main.cpp:

```cpp
static bool cmd_allowed(uint8_t cmd_id, RobotStateEnum s);  // single table
// on_command(): if (!cmd_allowed(...)) { comm_log(WARN, ...); optionally CMD_REJECT; return; }
```
Today gating is ad-hoc per command (WHEEL gated, HIP not, REBOOT not, LOG not). One table = one place to audit, one place for docs to point at, and rejected commands become *visible* (log + optional reject beep) instead of silently latching. This is the "better way" — no new protocol needed; the ACK question is already solved by param echo + state-in-telemetry.

---

## 7. GUI Performance Review (responsiveness wins)

The GUI's architecture (TelemetryBus fan-out, redraw timers in hip/wheel tabs) is right; the cost problem is that **every tab processes every packet at 50 Hz whether visible or not**, and several do heavy work per packet:

- **P1 — No visibility gating (biggest win).** All ~10 subscribers run `_on_packet` at 50 Hz. The Visualizer does numpy 3-D transforms + ~15 `setData()` calls *per packet* ([robot_visualizer_tab.py:1278](software/gui/tabs/robot_visualizer_tab.py#L1278)); imu_tab redraws 6+ curves + a 3-D cube per packet; raw_data_tab writes ~40 QLabels per packet.
  **Fix (uniform pattern):** `_on_packet` only stores `self._latest = info` (cheap dict ref); a per-tab QTimer at 10–20 Hz does all drawing, gated on `self.isVisible()`. Hip/wheel tabs already have the redraw timer — extend the pattern to Visualizer, IMU, Raw Data, Dashboard widgets, and stop doing *any* rendering in `_on_packet`.
- **P2 — Stylesheet churn.** Several widgets call `setStyleSheet()` per packet (e.g. TestValMiniWidget label, raw-data value colors) — each forces a re-polish. Only set style on **change** (cache last style key).
- **P3 — Serial decoding runs on the UI thread.** `SerialReader` (QThread) emits raw bytes → `PacketDecoder.feed()` executes on the main thread. Fine at 13 kB/s telemetry; during LOG_DATA downloads it hitches the UI. WifiTransport already decodes on its own thread — do the same for serial (decode in the reader thread, emit decoded dicts via queued signal).
- **P4 — Per-packet allocations.** `curve.setData(list(deque))` re-lists per curve per packet; with P1's 10–20 Hz timer this mostly disappears; optionally move to preallocated numpy ring buffers for the chart-heavy tabs.
- **P5 — Emit less.** `PacketDecoder` emits `packet_decoded` to the PacketInspector *and* TelemetryBus per frame; non-active sources still decode+emit inspector updates at 50 Hz per connected transport. Inspector should sample (e.g. update at 5 Hz).
- **P6 — Dead UI bug found:** TofMiniWidget reads `tof_stale` / `tof_age_ms` — keys that no longer exist since TELEM V6 removed `tof_age_ms` → permanently shows "TOF age 65535 ms". Fix to derive staleness GUI-side (age = now − last packet with changed ToF values) or drop the label.

Expected result: idle CPU of the GUI drops dramatically (only the visible tab renders), input latency (sliders, spinboxes) stops competing with 50 Hz repaints, and WiFi + USB + playback all get smoother.

---

## 8. GUI-Side 50 Hz Logging & Export (new feature)

**Goal:** record what the GUI is seeing (50 Hz live telemetry) to a file, distinct from the 500 Hz SD logs, with **identical replay + CSV export**.

**Design — reuse the `.wlog` format end-to-end** (no new format, no new replay code):
1. `PacketDecoder` keeps the raw TELEM_A payload bytes alongside the decoded dict; when TELEM_B completes a pair, attach `info["raw_telem"] = a_bytes + b_bytes` (the exact 235-byte TelemetryPayload).
2. New `tabs/gui_recorder.py` — `GuiLogRecorder` singleton subscribing to `TelemetryBus.packet`:
   - On start: write a `WlogHeader` (`format_version=1`, `telem_version=8`, `record_size=239`, **`sample_rate_hz=50`**, `start_millis=`first packet's `timestamp_ms`).
   - Per packet with `raw_telem`: write a `LogRecord` with `t_micros = timestamp_ms * 1000` (robot-clock timing → replay pacing immune to WiFi jitter/drops).
   - Skips packets while `playback_active` (never record a replay).
   - File naming: `gui/logs/GUI_YYYYMMDD_HHMMSS.wlog`.
3. **Replay + CSV work unchanged:** `WlogReader` already reads `record_size` and `sample_rate_hz` from the header, and `LogPlaybackController` paces from `sample_rate_hz` — a 50 Hz GUI log replays at true speed through every tab; `tools/wlog_to_csv.py` exports it as-is.
4. UI: in the Logs tab playback panel, add a "Record live" group — ● Record / ■ Stop button, elapsed time + record count, dropped-pair count; recorded files appear in the same file picker as downloaded SD logs (they're the same format).
5. Optional QoL: auto-start recording on arm (RUNNING entry) toggle — cheap flight recorder for every test session.

Bonus: `wlog_to_csv.py` and the header's `telem_version` field already guard against decoding a stale-version recording.

---

## 9. UI Redesign Proposals

Requested principles applied throughout: **no inline text boxes — every input field gets its own row** (QFormLayout: label left, field right), controls grouped in titled boxes, readouts big and passive, actions separated from data.

### 9.1 Dashboard (main.py `DashboardTab`) — currently 3 small widgets + empty space
Rebuild as the "glance = full robot health" screen, read-only (no controls), card grid:

```
┌─ STATE banner (full width): state name + color, fault name/desc + severity,
│  active source (USB/WiFi), battery bar (vbus), RC link, telemetry rate ─────┐
├───────────────┬──────────────────────────────┬──────────────────────────────┤
│ ATTITUDE      │ KEY NUMBERS (stat tiles)     │ HEALTH MATRIX                │
│ horizon or 3D │ pitch°   | vel m/s | v_ref   │ LED grid from health_flags:  │
│ cube + pitch/ │ τ_L/τ_R bar pair | θ_ref     │ HIP L/R · WM L/R · IMU ·     │
│ rate sparkline│ hip L/R pos w/ range gauges  │ LIMITS · LQR · VelSat·YawSat │
│               │ (big Consolas values, unit   │ + ToF 4-bar strip            │
│               │  small, one metric per tile) │ + loop_count gaps / CRC drops│
├───────────────┴──────────────────────────────┴──────────────────────────────┤
│ Recent robot log lines (last 5, colored by level)                           │
└──────────────────────────────────────────────────────────────────────────────┘
```
Rules: one metric per tile; value ≥ 20 pt Consolas, label 9 pt dim; colors only from `theme.py` state palette; tiles never move/resize with data (fixed grid → no layout thrash). The Visualizer tab already proves out most components (horizon, gauges) — the dashboard is its *passive, dense* sibling. Reuse `BatteryStatusWidget`, `RadioSignalWidget`, `RobotLogWidget` that already exist in main.py's status bar/bottom panel.

### 9.2 Hip Motors tab
- Left column per motor: readouts + chart (keep). Controls below as **QFormLayout, one row each**: `p [°]`, `kp`, `kd`, `τff [N·m]` (currently share inline rows).
- Presets → vertical `QGroupBox("Gain presets")`; wave controls (`amp`, `freq`) each their own form row inside `QGroupBox("Test wave")` with the Sine/Square toggles beneath.
- Jog slider full-width at panel bottom with min/max labels at the ends (calibrated range).
- Add JUMPING to `_STATE_LABELS` (currently shows "STATE 7").

### 9.3 Wheel Motors tab
- Mode selector → one segmented row of 4 exclusive toggle buttons (IDLE/VEL/POS/TRQ) with the active mode echoed from telemetry (highlight follows `wm_mode`, not the click).
- Setpoint, wave amp, wave freq → own form rows in `QGroupBox("Setpoint")` / `QGroupBox("Test wave")`.
- Diff-drive slider gets its own titled group with a live L/R split readout.
- Error display: decoded ODrive error flags as a vertical list (one flag per row, red only when set) instead of hex-in-a-label.

### 9.4 Parameters tab
- Add a filter/search box (top, its own row) — the param count is already ~70 and growing.
- One param per row: name | value field | unit/range hint | persist icon; group boxes per GROUP_* (already grouped — keep collapsible sections).
- Show CLAMPED feedback visibly (flash the row orange when the echo differs from the request).

### 9.5 Logs tab
- Two panels stay; add the "Record live" group (§8). Duration field gets its own row. Progress: chunks + kB/s + ETA on separate rows, not one packed string.

### 9.6 Raw Data / IMU / Controllers / Radio tabs
- Raw Data: already the right grid style; just decimate to 10 Hz (P1) and add the CRC-drop counter (W1).
- IMU: keep; gate on visibility (it's the most GPU-heavy after Visualizer).
- Controllers: surface `HEALTH_VEL_PI_SAT` / `HEALTH_YAW_PI_SAT` as LEDs next to their charts (data already in telemetry); gains stay read-only until C5 exposes them as params.
- Radio: add a channel→function legend column sourced from radio_channels.md content (static strings).

Implementation note: add tiny shared helpers to `theme.py` (`form_row()`, `stat_tile()`, `led_indicator()`) so all tabs converge on the same look instead of five hand-rolled variants (§10 SSOT).

---

## 10. Refactoring / Single Source of Truth

**Dead code to delete** (verified unused):
- `teensy/lib/Esp32Link/` — references `TELEM_PAYLOAD_V2` which no longer exists; main.cpp uses `CommLink` directly. Delete lib + README.
- Empty scaffolding dirs: `teensy/lib/CommLink/`, `teensy/lib/RC/`, `teensy/lib/ToF/`, `esp32/lib/Display/`, `esp32/lib/TeensyLink/`, `esp32/lib/WiFiLink/`.
- `shared/telemetry_protocol.h` (compat shim), `COMM_TYPE_TELEMETRY` + `COMM_START` legacy alias (after checking unit tests, per its own comment).
- GUI: `_CMD_HIP_MOTOR` alias in hip_motors.py.

**Duplication → consolidate:**
- **State→color table ×4** (README, Teensy `update_led()`, ESP32 `mode_color()`, GUI). Add `shared/state_colors.h` X-macro `STATE_TABLE(X)` consumed by both firmwares (also replaces ESP32's hand-copied `RS_*` enum); GUI mirror shrinks to one dict in `telem_format.py`/`theme.py`.
- **Fault names/descriptions ×3** (comm_protocol.h comments, ESP32 `fault_description()`, telem_format.py). Same X-macro treatment.
- **GUI state-label maps ×3** (telem_format has JUMPING; hip_motors & wheel_motors don't). One map, imported.
- **Sync-check script** `software/gui/tools/check_protocol_sync.py`: regex-parse comm_protocol.h / robot_state.h / param_ids.h and diff against Python constants (CMD IDs, fault codes, state names, TELEM_VERSION, struct sizes). Converts every "MIRROR:" comment from hope into verification.
- comm_commands.py `send_frame()` reaches into `SerialPortManager._lock/_open` — add a public `write(device, frame)`.
- D2/D3 from §6 (GUI-side duplicate command streams).

---

## 11. Documentation Audit

| File | Verdict | Issues |
|---|---|---|
| [firmware/robot_teensy/README.md](firmware/robot_teensy/README.md) | **Good, minor drift** | MANUAL watchdog "500 ms" → 5000 ms (or fix code per F7); "CAN timeout 20 ms" → `HIP_CAN_TIMEOUT_MS` is 50 ms; references `esp32/screen redo.md` which **no longer exists**; lists dead `Esp32Link` lib; comm_protocol.h:33's "120 bytes / frame 130" for TELEM_B is the stale one (117 is right) |
| [teensy/state_machine.md](firmware/robot_teensy/teensy/state_machine.md) | **Most outdated doc** | Missing STATE_JUMPING entirely (mermaid + tables); fault table missing 0x04/0x08/0x09; missing ESTOP→STANDBY soft-clear transition + radio CH10 soft-clear; missing RUNNING-denial guards (IMU enable, 4-motor enable); watchdog 500 ms → 5 s; calibration described "sequential per hip" → actually concurrent; stall defaults stale (0.5 A/15 ticks → registry 0.75 A/60; bench-tuned 5 A/4 A per TODO.txt); radio table missing CH2/CH4/CH6/CH7/CH9 |
| [shared/comm_protocol.h](firmware/robot_teensy/shared/comm_protocol.h) comments | Minor drift | Line 33 TELEM_B size; `ibus_alive` "within 500 ms" → IBus default 100 ms; CommLink.h "V5 payload = 244 bytes" → V8/235 |
| [software/gui/CLAUDE.md](software/gui/CLAUDE.md) | **Current** ✓ | — |
| Root [CLAUDE.md](CLAUDE.md) | **Broken refs** | Points at `docs/Control.MD` (moved to `docs/other/Control.MD`, which describes the **simulation**); "See docs/Control.MD for:" sentence truncated; says nothing about the three deployments — should link the new ROBOT.md (§12) |
| [docs/other/Control.MD](docs/other/Control.MD) | Misleading as firmware ref | Describes master_sim_jump (hip impedance, FF4, 6-state jump FSM, latency prediction) — none in firmware. Add banner: "SIM architecture — firmware truth is control_loop.cpp; implemented subset: LQR + vel PI + yaw PI + FF1/FF2 + 4-phase jump" |
| [pinout.MD](firmware/robot_teensy/pinout.MD) | Mostly good | Summary table IMU "INT=2, RST=3" → **INT=9, RST=6** (detail table correct; IMU.cpp:5 header comment has same stale pins); ESP32 link "1.2 Mbaud" → **4 Mbaud** (also stale in Esp32Link README + two CommLink/main.cpp comments) |
| [radio_channels.md](firmware/robot_teensy/radio_channels.md) | **Current** ✓ | Matches radio_update() exactly |
| lib READMEs | Mixed | HipMotors ✓ (see zeroing question §13); **Esp32Link fully stale/dead**; others get a sweep in the docs pass |
| param_ids.h / robot_state.h comments | Minor | GROUP_HIP TODO points at wrong file; param_ids says extend default 0, registry says 1 |

**Doc updates:** (1) regenerate state_machine.md from code (add JUMPING, soft-clear, full fault+severity table, real timeouts, concurrent calibration, link radio_channels.md instead of duplicating); (2) fix README's four drift items + add "hardware truths" section (MIT 4 s dropout, CAN gap, CP2102 budget); (3) fix stale numeric comments (comm_protocol.h / CommLink.h / IMU.cpp / pinout.MD); (4) repair root CLAUDE.md + add software map; (5) **new `esp32/README.md`** (the ESP32 has zero docs — tasks/cores, ports, personalities, wifi_config note); (6) delete Esp32Link README with the lib.

---

## 12. New Overview Document — `ROBOT.md` (repo root)

The "point future Claudes here" file. Content rule: **pointers and invariants only** — no numbers that live in code (they rot; link instead).

```
# ROBOT.md — Wheeled-Leg Robot: System Map
1. What this robot is (3 lines) + data-flow diagram (Teensy ⇄ ESP32 ⇄ GUI, radio, CAN)
2. The three deployments — role, entry point, build/flash command, doc links:
   • Teensy 4.1 — firmware/robot_teensy/teensy   (500 Hz control, FSM, CAN, SD log)
   • ESP32      — firmware/robot_teensy/esp32    (TFT, Neopixel, ToF, WiFi/USB bridge)
   • PC GUI     — software/gui                   (PyQt6; python main.py)
3. Single sources of truth (table): wire protocol → shared/comm_protocol.h |
   states → robot_state.h | params → param_ids.h + param_registry.cpp |
   telemetry layout (Python) → tabs/telem_format.py | pins → the two config.h |
   radio → radio_channels.md | FSM → teensy/state_machine.md
4. Key invariants: coordinate system; 500 Hz / 50 Hz / TELEM_VERSION; all-4-motors
   to arm; calibration before RUNNING; fault severity tiers; command permission matrix
5. Hardware map (condensed bus/pin table → pinout.MD)
6. How-to index: flash each board, run GUI, calibrate, arm, record/download/replay logs,
   add a telemetry field (→ PROPAGATION CHECKLIST), add a param, add a fault code
7. Known hazards & safety notes (§1 of this audit, post-fix)
8. Doc index with one-liners (sim docs marked as sim)
```

---

## 13. Other Suggestions & Open Questions

- **Hip encoder zeroing every calibration vs motor-flash wear:** HipMotors README warns `hip_motors_zero()` "writes the zero to flash inside the motor — call only once", but calibration.cpp zeroes on **every** SEEK_BOTTOM. Verify against CubeMars docs; fix whichever is wrong.
- **GUI ESTOP hotkey:** bind `Esc`/`Space` globally in MainWindow — ESTOP shouldn't require a mouse hunt.
- WifiTransport calls `SourceManager._on_opened/_released` directly from its thread — works (GIL + queued signals) but converting to a signal removes the wart.
- Comm unit tests exist (`test_comm_usb` etc.) — extend for CRC-8 when W2 lands.

---

## 14. Proposed Execution Plan

Ordered so each phase is independently flashable/testable. **Bench note:** left hip/wheel currently disabled, robot clamped — ideal for validating P0/P1 safely.

### Phase 0 — Safety + real-time fixes (Teensy, ~1 session) — ✅ EXECUTED + BENCH-VERIFIED 2026-07-13
1. ✅ F1 IMU silence fix + `FAULT_IMU_LOST` → ESTOP from RUNNING/JUMPING.
2. ✅ F2 gate `CMD_ID_HIP` to MANUAL + clear pending on entry → generalize into the **command permission matrix** (§6).
3. ✅ F3 flush all request flags in `on_estop()`.
4. ✅ F4 level-based disarm.
5. ✅ F5 wheel-feedback fault → ESTOP (+ `ever_heard` guard).
6. ✅ F6 state-gate REBOOT + motor-safe pre-reset (covered by the matrix).
7. ✅ F7 MANUAL watchdog → 500 ms + new `CMD_ID_PING` GUI heartbeat at 10 Hz.
8. ✅ **D1 calibration through setpoint cache** (single MIT frame/tick; re-verify stall thresholds on bench after).
9. ✅ R1 deferred param flash save (`param_flush_service`); R2 loop-overrun counter → `HEALTH_LOOP_OVERRUN` + WARN log.
10. ✅ New fault codes propagated per checklist (comm_protocol.h, ESP32, telem_format.py, main.py severity map, README, state_machine.md).
- **Verify:** `pio run -e teensy41`; bench: arm→CH6+fault→reset→re-arm (no jump); unplug IMU mid-RUNNING → ESTOP; kill GUI in MANUAL wheel-velocity → stop ≤500 ms; profile switch during RUNNING → zero overruns; calibration completes with correct limits.

### Phase 1 — Comms robustness (all three ends) — ✅ EXECUTED + BENCH-VERIFIED 2026-07-13
1. ✅ W1 GUI drops bad-checksum frames + counter; 2. ✅ W2 CRC-8 flag-day; 3. ✅ W3/W4 seq gap counters + seq-adjacent A/B pairing; 4. ✅ W5 SD GET gated to STANDBY/ESTOP + per-transport pacing + single-link chunk routing; 5. ✅ W6/W7 comm_log 120 B + WheelMotors via comm_log; 6. ✅ D4 watchdog pet at 50 Hz (see §6 D4 for the non-TORQUE coverage rationale).
- Comm unit tests needed **no changes** — they round-trip through CommLink itself (encode and decode share the CRC), and `test_bad_checksum_dropped` still corrupts a byte and asserts the drop.
- **Verify (bench, §15 rows 20-24):** telemetry soak over USB and WiFi with drop counters ≈ 0; log download over both transports, CRC32 pass; unit tests pass on hardware.

### Phase 2 — GUI: performance + recording + command cleanup
1. P1 visibility-gated 10–20 Hz render timers everywhere; `_on_packet` stores only.
2. P2 style-on-change; P5 inspector sampling; P6 ToF stale-age fix.
3. P3 decode in reader thread (serial path).
4. **§8 GuiLogRecorder** (`raw_telem` passthrough, .wlog @ 50 Hz, Logs-tab Record group, optional record-on-arm).
5. D2 remove GUI MIT-reinforce timer; D3 demote wheel auto-sequence to a button.
- **Verify:** GUI CPU while idle on each tab (Task Manager before/after); record a bench session → replay through all tabs → CSV export; wave tests still stream at 50 Hz.

### Phase 3 — UI redesign (§9)
1. `theme.py` helpers (`form_row`, `stat_tile`, `led_indicator`).
2. Dashboard rebuild (9.1).
3. Hip/Wheel form-layout conversion + segmented mode selector + JUMPING labels (9.2/9.3).
4. Params search + clamp feedback (9.4); Logs record group polish (9.5); small items (9.6).
- **Verify:** visual pass over every tab live on bench; all controls reachable at 1280×800.

### Phase 4 — Refactors / dead code (§10)
Dead code deletion; `shared/state_colors.h` + fault X-macro; GUI single state map; SerialPortManager public write; `check_protocol_sync.py`; C1 controller-state reset on arm; C2 decision; C5 hip RUNNING gains as params.
- **Verify:** both firmwares compile; GUI runs; sync-check passes; calibration single-frame/tick confirmed (already from P0).

### Phase 5 — Documentation (§11 + §12)
state_machine.md rewrite; README/pinout/comment fixes; esp32/README.md; **ROBOT.md**; root CLAUDE.md; lib README sweep.

### Explicitly deferred (your call later)
Teensy/ODrive/ESP32 watchdogs (needs ODrive USB session — re-probe hw first); F9 TCP token; pitch-trim wiring into LQR (C3); FlySky failsafe config.

## Verification (overall)
After each firmware phase: clean `pio run -e teensy41` / `-e esp32dev`; flash bench robot; walk the FSM end-to-end (STARTUP→STANDBY→CALIBRATION→STANDBY→MANUAL hip sine + wheel vel→arm→RUNNING w/ sim-pitch + LQR torque-limited→CH6 JUMPING w/ jump_enable=0→disarm→all ESTOP paths→soft-clear/reset). GUI: telemetry on all tabs over Teensy-USB, ESP32-USB, WiFi; param round-trip; SD log + GUI log record/download/replay/CSV. Log each session in docs/testing_log.md.

## Open questions
1. ~~F7: what MANUAL GUI-watchdog timeout do you want (500 ms doc'd vs 5 s coded)?~~ **Resolved:** 500 ms, made viable by the new 10 Hz `CMD_ID_PING` GUI heartbeat. If WiFi shows spurious MANUAL exits on the bench, raise to 1000 ms (one constant in state_machine.cpp).
2. C2: should FF1+FF2 respect `PARAM_LQR_TORQUE_LIMIT` or only the 7 N·m hard clamp?
3. ~~F9: open TCP command port acceptable for now (doc-only)?~~ **Resolved:** accepted risk, documented in firmware README; hardening options noted there.
4. AK45 zeroing flash-wear (§13) — do you have the CubeMars doc handy?
5. Dashboard redesign (9.1): keep the test_val sine widget anywhere, or retire it now that CRC/gap counters exist?

---

## 15. Hardware verification — ✅ COMPLETED on bench 2026-07-13

All of Phase 0 (F1–F10 + D1 + R1 + R2) and Phase 1 (W1–W7 + D4) are implemented and **bench-verified 2026-07-13** with a full reflash of Teensy + ESP32 and a GUI restart from the same checkout (CRC-8 flag day honored). The table below is retained as the regression checklist for future firmware changes.

**Before testing:**
- **FLAG DAY (W2):** the frame checksum changed from XOR to CRC-8 — an old firmware and a new GUI (or old ESP32 + new Teensy, any mix) reject 100% of each other's frames. **Reflash Teensy AND ESP32, and restart the GUI, all from this checkout, before anything else.** Symptom of a mixed pair: zero telemetry, rx_drops/CRC counters climbing on both ends.
- The 500 ms MANUAL watchdog requires the GUI's new `CMD_ID_PING` heartbeat; an old GUI gets kicked out of MANUAL after 500 ms.
- Tests marked **[RUNNING]** need all four `*_enable` params set (arm is denied in single-leg bench mode) and a prior calibration. Keep the robot clamped, `PARAM_LQR_TORQUE_LIMIT` low, `PARAM_JUMP_ENABLE = 0`.

| # | Fix | Procedure | Expected |
|---|---|---|---|
| 1 | F7 heartbeat | Enter MANUAL, hands off for >10 s (USB, then WiFi) | Stays in MANUAL — heartbeat feeds the watchdog |
| 2 | F7 crash-stop | MANUAL + wheel VELOCITY spinning → kill the GUI process (Task Manager, not graceful close) | Wheels stop and state → STANDBY within ~500 ms. Repeat over WiFi; if WiFi ever shows spurious MANUAL exits in normal use, raise timeout to 1000 ms |
| 3 | F1 idle detect | In STANDBY, unplug the IMU | Log "IMU: error — retrying in 1 s" within ~100 ms (previously never fired); `HEALTH_IMU_NOMINAL` drops; replug → recovers |
| 4 | F1 armed detect **[RUNNING]** | Clamped in RUNNING, unplug the IMU | Immediate ESTOP, fault `0x0A IMU_LOST`, wheels idle — no frozen-pitch balancing |
| 5 | F5 wheel loss **[RUNNING]** | Clamped in RUNNING, disconnect ODrive CAN (or power) | Immediate ESTOP, fault `0x0B WHEEL_FEEDBACK_LOST` (previously: silent fall, pitch watchdog ~200 ms later) |
| 6 | F5 ever_heard | Boot with ODrives powered off | WM L/R health flags not-OK immediately (previously spuriously OK for the first `enc_timeout` ms) |
| 7 | F3 stale jump **[RUNNING]** | Arm → flip CH6 and hit ESTOP in the same moment → reset → re-arm (CH6 still up) | No JUMPING entry on re-arm; `jump_state` stays 0 |
| 8 | F4 disarm-in-jump **[RUNNING]** | Arm → CH6 jump (`jump_enable=0`) → drop CH10 mid-JUMPING | Jump sequence completes → RUNNING re-entered → immediate disarm to STANDBY (disarm melody) |
| 9 | F2 hip gating | MANUAL: start hip sine wave → exit to STANDBY mid-wave → re-enter MANUAL | Any frames still in flight rejected with WARN "CMD 0x05 rejected in state 2"; zero hip motion outside MANUAL; on re-entry hips stay still until a new command is sent |
| 10 | F6 reboot gate | Send Reboot from MANUAL (and from RUNNING if convenient) | Rejected + WARN log; robot unaffected |
| 11 | F6 safe reboot | Send Reboot from STANDBY | Hips exit MIT and wheels IDLE before reset (no torque jerk), then a normal boot |
| 12 | F8 clamp | Write an out-of-range param (e.g. `lqr_torque_limit = 999`) | Echo shows value clamped to max; no ESTOP, no fault |
| 13 | F10 single-leg send | With only hip R enabled, exercise `hip_motors_send()` (`test/test_hip_motor`) | Right hip moves (previously the both-motor variant no-op'd entirely when L was disabled) |
| 14 | **D1 calibration** | Run a full calibration (retract + extend) on the bench leg | Completes with sane limits. **Watch the stall detection**: with the zero-torque ping gaps gone, effective stiffness during seek is higher — stall current (5 A / 4 A bench values) may trigger differently; retune `calib_stall_cur_*` if a hardstop is declared too early/late. Check the log's trigger diagnostics (current / pos error / distance traveled) against previous runs |
| 15 | D1 zero transient | During calibration, observe the bottom-hardstop zeroing moment | No torque kick into the hardstop at the instant of zeroing (setpoint cache is refreshed to the new frame the same tick) |
| 16 | R1 flush | Change a persistent param in STANDBY; power-cycle ≥2 s later | Value survives the power cycle (flush happens ~1 s after the last write). Also: change a param, power-cycle within <1 s → value may be lost (expected; note in docs if it bothers you) |
| 17 | R1 no-stall **[RUNNING]** | Flip CH9 speed profiles while balancing (clamped) | Zero `HEALTH_LOOP_OVERRUN` / no "Loop overrun" WARN logs during the torque slew (previously ~300 synchronous flash writes); after disarm to STANDBY, the profile's torque limit flushes to flash ~1 s later |
| 18 | R2 counter | Trigger a known-slow op outside RUNNING (e.g. start an SD log GET during STANDBY, or just watch the param flush) | "Loop overrun: N us (count M)" WARN appears, `health_flags` bit 11 (0x0800) lights in Raw Data for ~1 s |
| 19 | Regression | Full FSM walk: STARTUP→STANDBY→CALIBRATION→STANDBY→MANUAL (hip sine + wheel vel)→ESTOP paths→soft-clear/reset, over Teensy-USB, ESP32-USB, and WiFi | All behave as before |
| 20 | W2 CRC-8 soak | After reflashing everything: 10+ min telemetry soak over Teensy-USB, ESP32-USB, and WiFi; watch Raw Data → Frame → link health | Telemetry flows on all three transports; `CRC drops` ≈ 0 on clean links; TFT footer CRC/GAP counters ≈ 0 |
| 21 | W3/W4 loss visibility | During the WiFi soak, walk the robot to the edge of WiFi range (or wrap the ESP32 in foil briefly) | `Seq gaps` climbs during degradation, telemetry stays sane (no garbage values / impossible jumps — corrupted and mispaired frames are dropped, not decoded) |
| 22 | W5 GET gate | Request an SD-log download while in MANUAL or RUNNING | Denied: WARN "log get denied" in robot log, download does not start; works normally from STANDBY/ESTOP |
| 23 | W5 paced download | Download the same .wlog over Teensy-USB, then over ESP32-USB, then WiFi (from STANDBY) | All three complete with CRC32 pass; ESP32-path download is slower (~61 kB/s) but drop-free; telemetry stays live during the download |
| 24 | Comm unit tests | `pio test -e teensy41 -f test_comm_usb -f test_telemetry` on the bench Teensy | All pass (loopback round-trip + bad-checksum drop, now exercising CRC-8) |

F9 needs no bench test (doc note only). **Bench pass completed 2026-07-13** (fresh flash on all boards + GUI restart) — Phases 0 and 1 are implemented and verified; nothing deferred except the watchdog items explicitly parked in §14 "Explicitly deferred".
