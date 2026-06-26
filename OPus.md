# Review: firmware/robot_teensy + software/gui

## Context
Fernando asked for an audit of the three working deployments (Teensy firmware, ESP32
firmware, Python GUI), prioritized as: (1) safety, (2) bugs, (3) param/mode/profile
architecture, (4) GUI architecture + screen-redo + color consistency, (5) docs, (6)
AI-flow / cross-file propagation. This file records findings (verified against source)
and a prioritized fix list. No code has been changed yet.

---

## 1. SAFETY (highest priority)

### 1.1 [REAL BUG] Stale ODrive torque can be applied on arm — unexpected wheel motion
- On entering RUNNING, `on_running()` calls `wheel_motors_set_mode(WheelMode::TORQUE)`
  (state_machine.cpp:192). This puts both ODrives in CLOSED_LOOP torque control but does
  **not** send an `input_torque`.
- `controlLoop_run()` only calls `wheel_motors_send(tau_L, tau_R)` when
  `PARAM_LQR_ENABLE >= 0.5` (control_loop.cpp:229-250). When LQR is disabled, no torque is
  ever written.
- ODrive latches the last `input_torque`. After any prior RUNNING session where LQR ran
  (nonzero torque), then disarm→IDLE, then **re-arm with LQR off**, the ODrive immediately
  re-applies the stale nonzero torque → wheels lurch with no command.
- Boot is safe (input_torque defaults 0); the hazard is re-arm within a power cycle.
- **Fix:** in control_loop.cpp, call `wheel_motors_send(tau_L, tau_R)` unconditionally
  (tau stays 0 when LQR off) instead of skipping the call. This guarantees an explicit 0
  every tick and doubles as an ODrive watchdog pet. Alternatively send `wheel_motors_send(0,0)`
  once on entering TORQUE mode.

### 1.2 No ODrive watchdog pet while in TORQUE mode
- `wheel_motors_pet_watchdog()` (wheel_motors.cpp:180-187) only sends zeros in IDLE. In
  CLOSED_LOOP TORQUE there is no periodic pet from firmware. If the Teensy loop stalls while
  a nonzero torque is latched, the ODrive holds that torque indefinitely unless the ODrive's
  own `axis.config.watchdog_timeout` is configured.
- **Fix:** enable ODrive hardware watchdog (defense-in-depth) AND adopt 1.1 (send every tick)
  so the control loop itself pets it.

### 1.3 Radio HIP command left stale on signal loss (already self-noted in code)
- On radio loss, V_CMD and OMEGA are forced to 0 (main.cpp:533-534) but `PARAM_RADIO_HIP_CMD`
  is left at its last value (comment at main.cpp:434). Mitigated because radio loss disarms
  RUNNING→STANDBY, and STANDBY clears hip setpoints (state_machine.cpp:130). Low residual risk;
  zero it in the `else` branch for consistency/defense-in-depth.

### 1.4 Arming does not require sticks near neutral
- `armed = alive && (ch10 > 1990)` with rising-edge detection (main.cpp:454). If the ARM
  switch is already high at power-on, the first received packet counts as the rising edge.
  RUNNING is blocked unless calibration is valid, so motion is gated — but once calibrated,
  arming with throttle/yaw sticks off-center immediately applies those commands.
- **Suggested improvement:** gate arming on velocity/yaw stick within a neutral band (and/or
  require an explicit low→high CH10 transition seen after boot), so arming is always from a
  known-zero command state.

### 1.5 Profile switch applies torque limit live with no ramp
- CH9 profile change calls `param_set(PARAM_LQR_TORQUE_LIMIT, ...)` immediately
  (main.cpp:508), even mid-balance in RUNNING. A large downward step in torque limit during
  active balancing could destabilize. Low priority; consider slewing the limit.

### Good safety properties already present (keep)
- Pitch watchdog (50°/200 ms) and wheel-runaway watchdog (2× soft limit) → ESTOP
  (control_loop.cpp:79-96).
- Hip per-command 90° jump guard drops the frame and ESTOPs (hip_motors pack_and_send).
- Hip MIT setpoint 5 s timeout → safe zero-torque ping.
- Wheel auto-IDLE on encoder timeout / ODrive error (wheel_motors.cpp:103-112).
- Integrators reset on disable and on direction reversal (control_loop.cpp:131,157,185).
- RUNNING blocked until calibration limits valid.

---

## 2. BUGS / ROBUSTNESS

- **Telemetry version is hand-maintained (`_TELEM_VERSION = 8`) with no size guard.** A field
  add that shifts struct offsets but forgets a version bump silently corrupts every downstream
  field. Add `static_assert(sizeof(TelemetryPayload) == N, ...)` in C and a
  `struct.calcsize(_FMT_TELEM_A) == 118` assert at Python startup.
- **Right-wheel mirroring (`-R`) is duplicated** in encoder RX, torque send, velocity send, and
  position send (wheel_motors.cpp:63,150,160,171). Forgetting one path yields asymmetric
  behavior. Consider centralizing the sign convention.
- **XOR checksum is single-byte** (comm_protocol.h). Fine at current baud/short USB, but weak;
  CRC-8 is a cheap upgrade if corruption ever appears.
- `loop_count` uint32 wraps ~24 days @500 Hz — diagnostic only, not a safety issue.

---

## 3. PARAMETERS / MODES / PROFILES — architecture

- **Sound core design:** firmware is the runtime single source of truth — it streams
  `min/max/flags/name` per param via PARAM_REPORT, GUI renders from that. Keep this.
- **Weak spot:** GUI `_SUBGROUPS` ID ranges (params_tab.py:52-79) are hardcoded and must track
  `param_ids.h`. Adding a param at the end of a range silently drops it from its subgroup.
  **Fix:** include a group/subgroup id byte in PARAM_REPORT so the GUI never infers from ID
  ranges.
- Possible missing feature: per-profile persisted presets beyond the 3 speed profiles, and a
  "needs reboot" flag on params that aren't hot-applied. Optional.

---

## 4. GUI / VISUAL

### 4.1 GUI architecture
- Clean PyQt6 with a `TelemetryBus` singleton; good tab separation. No major architectural issue.
- **Repeated info:** state names, fault names/severity, and wheel-mode names are duplicated
  across flash_monitor.py, comm_commands.py, main.py, raw_data_tab.py. Consolidate into one
  generated/constants module mirroring the C enums.
- Robot geometry is hardcoded in robot_visualizer_tab.py (not synced to firmware/params).
- Two telemetry parse paths (flash_monitor + wifi_transport) can drift; share one decoder.

### 4.2 screen redo.md
- **Coordinate mismatch:** plan uses dividers at x=160/240; current code draws at x=170/250
  (esp32 main.cpp). Reconcile before implementing or panels misalign.
- Plan specifies animation *types* but not exact RGB565 values — reference existing `COL_*`
  constants so colors stay consistent.
- Convert all four widgets (banner/hip/wheel/footer) to sprites in **one** pass; mixing direct
  draws and sprites mid-migration causes tearing.
- **Benchmark first** with the existing `esp32/src/tft_benchmark/` suite: ~154 KB sprite budget
  plus per-frame animation may drop below the ~30 Hz target. Verify before committing.

### 4.3 Color consistency across Teensy LED / Neopixel / TFT / GUI
- Consistent: RUNNING = green everywhere; ESTOP = red everywhere. 
- **Inconsistent (worth fixing):**
  - **JUMPING:** Teensy LED orange (255,100,0) vs Neopixel rainbow vs TFT magenta (200,0,255)
    — three different identities for one state.
  - **CMD_REJECT:** Teensy LED red vs Neopixel orange vs TFT orange.
  - **STANDBY:** Teensy LED/Neopixel amber (255,200,0) vs TFT yellow (255,255,0) vs GUI
    yellow (#ffe57f) — close but not identical.
- **Fix:** define ONE canonical `state → RGB` table shared by all four surfaces (Teensy LED,
  Neopixel base hue, TFT banner, GUI), letting each layer apply its own animation but the same
  hue.

---

## 5. DOCUMENTATION

- **firmware/robot_teensy/README.md is out of date:**
  - State table lists 6 states; code has 8 (missing JUMPING, CMD_REJECT).
  - Fault list missing FAULT_PITCH_WATCHDOG and FAULT_WHEEL_RUNAWAY.
  - No description of the ESP32 role (WiFi/TFT/Neopixel/ToF), no pointer to control_loop.cpp,
    no explicit link to state_machine.md.
- For "fresh-Claude onboarding," README should become the canonical entry point: add an
  architecture diagram, the full state table, the canonical color table (§4.3), and the
  telemetry/param pipeline. TestWheels.MD was deleted (git status); screen redo.md is the
  active plan.

---

## 6. AI-FLOW / CROSS-FILE PROPAGATION (meta)

- The "cascading checkboxes when changing a telemetry packet" is the multi-file-edit problem:
  a packet change today touches ~5 files (comm_protocol.h, teensy send, esp32 reassembly,
  flash_monitor decode, raw_data_tab display).
- **Best structural answer: single source of truth + codegen.** Define the telemetry/param
  schema once (YAML/JSON, or treat the C header as authority) and generate: the Python struct
  format string, the field/display lists, and the version number. A packet change becomes one
  edit + regenerate, instead of N hand-synced edits.
- **Guardrails:** `static_assert` on struct size (C) + `struct.calcsize` assert (Python) so any
  desync fails loudly instead of silently corrupting fields.
- **HTML docs:** modest value *if auto-generated* from the schema (state diagram, color table,
  param/telemetry reference). Hand-maintained HTML will rot exactly like the README did. Higher
  ROI: living README + a generated schema page. Doxygen is overkill here.

---

## Prioritized fix list (proposed)
P0 (safety): 1.1 send torque every tick (also fixes 1.2 pet) — small, high value.
P0 (safety): enable ODrive hardware watchdog_timeout (1.2).
P1: telemetry static_assert + Python size assert (2).
P1: README update — states, faults, ESP32 role, color table (5).
P2: canonical state→color table shared across LED/Neopixel/TFT/GUI (4.3).
P2: arming neutral-stick gate (1.4); zero RADIO_HIP_CMD on radio loss (1.3).
P3: schema-driven codegen for telemetry/params (6); PARAM_REPORT group id (3).
P3: screen redo coordinate + color reconciliation before implementing screen redo.md (4.2).

## Verification
Each implemented fix verified by building the relevant firmware (PlatformIO) and, for
GUI/telemetry changes, launching the GUI and confirming live telemetry + arming behavior on
the flatsat. The torque-on-arm fix specifically verified by: arm with LQR off after a prior
LQR session and confirm wheels stay at zero torque.




GUI stuff:

Display Redesign Plan
Phase 0 — Shared state palette (foundation, do first)
Goal: one source of truth for state → {name, RGB}, read by TFT banner, Teensy LED, Neopixel base hue, and GUI header. Fixes the STANDBY/CMD_REJECT/JUMPING color mismatches.

New file firmware/robot_teensy/shared/robot_states.h:

ROBOT_STATE_COUNT and a struct { const char* name; uint8_t r, g, b; } table indexed by RobotStateEnum.
Canonical colors: STARTUP 255/255/255 · CALIBRATION 0/120/255 · STANDBY 255/180/0 · RUNNING 0/230/80 · ESTOP 255/40/40 · MANUAL 0/200/255 · CMD_REJECT 255/120/0 · JUMPING 200/0/255.
static_assert(ROBOT_STATE_COUNT == 8, ...) guard.
Wire-in (replace existing hardcoded color switches):

ESP32 esp32/src/main.cpp → mode_color() (main.cpp:661) reads the table; Neopixel base hue seeds from it.
Teensy LED state colors in teensy/src/main.cpp (the RGB LED switch ~line 353) read the table.
New software/gui/robot_states.py mirrors the table + a guard: assert len(STATES) == 8. theme.py keeps the hex theme colors; the state colors come from here.
Verify: flash both MCUs, cycle all 8 states, confirm LED/Neopixel/TFT show the same hue; launch GUI, confirm header matches.

Phase 1 — GUI visualizer (low risk, no hardware)
File: software/gui/robot_visualizer_tab.py (+ reuse from main.py).

Full-width status header across the top of the tab (above the 3-column QHBoxLayout at robot_visualizer_tab.py:655):

Big MODE label (color from robot_states.py), fault name, battery, RC●/TEL● dots, profile.
Reuse the computation already in StatusBar (main.py) — extract its mode/fault/battery/link logic into a small shared widget so both the main status bar and this header use it (no duplication).
Removes reliance on the buried size-10 state label at robot_visualizer_tab.py:908.
Pitch as hero in _build_orientation_group() (:679): give the pitch AngleArcWidget a larger fixed height + bigger value font; demote roll/yaw to a smaller side-by-side row. AngleArcWidget already supports setpoint + rate, so no new widget.

Shrink 3-D ~20%: drop the GL setMinimumSize(300,260) (:582) stretch weight / widen the side panels (currently fixed 212/238 px).

Optional emphasis selector (Balance/Drive/Jump): a small QButtonGroup at the top that bolds the relevant group (_make_group border/title) and dims others. Pure stylesheet toggle.

Verify: run the GUI against live or replayed telemetry; confirm header tracks mode/fault, pitch arc is dominant, layout still fits at the default window size.

Phase 2 — TFT paginated (do last; needs on-robot iteration)
File: firmware/robot_teensy/esp32/src/main.cpp. Apply the sprite/double-buffer approach from screen redo.md, but to fewer/larger regions. Minimum font size-3; hero elements size-4/5.

Open item before starting: confirm CH8 is free on the radio for the page selector (fallbacks: ESP32 GPIO button, or slow auto-cycle).

Page state: add g_tft_page (0=FLIGHT,1=DRIVE,2=COMMS), set from CH8 (or button). update_display() dispatches per page.

Page A · FLIGHT (default): status strip (26px) → MODE banner size-5, color from table, fault name size-3 in ESTOP (104px) → existing artificial horizon as hero with size-4 pitch number + lean tick (92px) → footer (44px): battery bar + size-3 voltage, RC●/TEL●, wheel_vel_avg_ms.
Page B · DRIVE: full-screen — big hip L/R arc gauges (actual cyan + commanded yellow needle, reuse the arc concept from screen redo.md) + wheel L/R velocity & torque bars + ODrive state dots. All ≥ size-3.
Page C · COMMS: link health big — RC, telemetry age, CRC drops (g_uart_crc_drops), seq gaps, WiFi/UART dots, profile.
Sprites: one per page region (banner / horizon / footer for A; full-frame for B and C), pushed atomically to kill the current tearing.

Verify: flash ESP32; toggle pages; confirm each page is legible at ~1 m, no flicker, ~30 Hz maintained (benchmark with the existing esp32/src/tft_benchmark/ first); confirm face personality path still works.

Suggested order & risk
Phase 0 — small, unblocks consistent color everywhere.
Phase 1 — no hardware, immediately useful for testing.
Phase 2 — biggest effort, needs the CH8 confirmation + bench iteration.
Want me to start on Phase 0, or wait until you've confirmed CH8 and want the whole thing?