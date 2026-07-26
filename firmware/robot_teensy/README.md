# robot_teensy firmware

Two-microcontroller architecture for a wheeled-leg balancing robot.

> **AI maintenance note:** If you find anything here that is stale while
> working in this tree, update this README in the same change.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Teensy 4.1  (500 Hz control loop)                              │
│                                                                  │
│  control_loop.cpp  ← LQR + vel/yaw PI + feedforward            │
│  state_machine.cpp ← 8-state FSM                                │
│  hip_motors.cpp    ← AK45-10 MIT Cheetah (CAN2)                │
│  wheel_motors.cpp  ← ODrive/ODESC (CAN3)                       │
│  IMU.cpp           ← BNO086 (SPI)                               │
│  main.cpp          ← scheduler, radio, telemetry, LED, buzzer   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ UART 4 Mbaud (CommLink framed)
                               │ telemetry @ 50 Hz (split TELEM_A + TELEM_B)
                               │ commands ← GUI / ESP32
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  ESP32  (display, telemetry bridge, obstacle sensing)           │
│                                                                  │
│  WiFi UDP leased unicast → Python GUI (software/gui/)            │
│  WiFi TCP server        ← GUI commands/results                   │
│  TFT display (SPI)   ← paginated flight/drive/comms pages       │
│  Neopixel strip      ← state colour + animations                │
│  VL53L1X ToF ×4      → obstacle distances forwarded to Teensy   │
│  USB serial (CP2102) → PC passthrough when WiFi unavailable     │
└─────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             Python GUI              Browser / raw serial
         software/gui/main.py
```

**Network exposure (accepted risk — home lab only):** a GUI first broadcasts
`WLR_CLAIM_V1 <token>` on UDP `:5007`. The ESP32 grants one 3.5-second session
lease and sends combined 247-byte telemetry datagrams by unicast to that
client on UDP `:5005`; renewals arrive every 0.8 seconds. A competing token is
reported busy and a second TCP `:5006` connection is closed without evicting
the owner. TCP carries commands, results, parameters, and log transfers but
does not duplicate telemetry. The token prevents accidental client collision,
not hostile access or spoofing: there is no authentication or encryption, so
do not operate the robot on an untrusted network.

**Key buses on Teensy:**
- CAN2 @ 1 Mbps → AK45-10 hip motors (MIT Cheetah protocol)
- CAN3 @ 1 Mbps → ODrive wheel motors
- SPI0 → BNO086 IMU (CS=D10, INT=D9, RST=D6)
- Serial2/3 → AK45 UART encoder readback
- Serial4 RX → FlySky iBUS RC receiver
- Serial5 ↔ ESP32 (CommLink UART)

## Directory layout

```
firmware/robot_teensy/
├── shared/
│   ├── comm_protocol.h   ← packet types, TelemetryPayload, fault codes (single source of truth)
│   └── CommLink/         ← framed protocol implementation used by both MCUs
├── teensy/               ← Teensy 4.1 PlatformIO project
│   ├── src/
│   │   ├── config.h          ← all pin and bus constants
│   │   ├── robot_state.h     ← RobotState struct + RobotStateEnum
│   │   ├── state_machine.cpp ← 8-state FSM (see state_machine.md)
│   │   ├── control_loop.cpp  ← LQR + vel PI + yaw PI + feedforward
│   │   ├── live_tune.h       ← generic radio-knob live param tuning (slot table in main.cpp)
│   │   └── main.cpp          ← loop(), radio, telemetry, LED, buzzer
│   └── lib/
│       ├── AK45Uart/         ← AK45-10 UART encoder readback
│       ├── HipMotors/        ← AK45-10 MIT Cheetah CAN driver (CAN2)
│       ├── WheelMotors/      ← ODrive CAN driver (CAN3)
│       ├── IMU/              ← BNO086 SPI driver
│       ├── LED/              ← Non-blocking RGB LED
│       ├── Buzzer/           ← Non-blocking passive buzzer
│       ├── ParamRegistry/    ← Runtime param table (GUI-tunable, 500 Hz safe)
│       └── Calibration/      ← Hip retract-switch calibration FSM
└── esp32/                ← ESP32 PlatformIO project
    └── src/main.cpp      ← all ESP32 logic (display, WiFi, Neopixel, ToF)
```

## GUI control API (preferred for agents and headless inspection)

The running Python GUI exposes a local-only command server on
`127.0.0.1:8765`. It starts automatically with the GUI and is controlled with:

```text
python software/gui/tools/robot_ctl.py <command> [args...]
```

Use this API whenever a task asks to inspect or control something "in the
GUI." Prefer it over Windows mouse/keyboard automation and over writing a
standalone client that reimplements the robot protocol. Start with
`capabilities`, `health`, or `service_status`; use `service_start` if the GUI
is not running.

The API includes dedicated robot commands (`telem`, `param_get`, `param_set`,
`set_mode`, logging, connection control, and firmware flashing) plus a generic
Qt operator bridge:

- `tab_select <title>` selects a GUI tab.
- `ui_manifest` enumerates widget IDs and supported actions.
- `ui_snapshot [query]` returns widget values and text without screen scraping.
- `ui_invoke <id> <action> [json_value]` invokes a real Qt widget action.
- `ui_screenshot` captures the GUI only when visual plot inspection is useful.

For an already-loaded Log Analyzer run, use
`ui_snapshot tab/log-analyzer` to read its source, duration, metrics, limits,
warnings, selected view, and available view choices. Change the analyzer view
through the combo box returned by that snapshot using
`ui_invoke <combo-id> select_text '"Vel-PI"'`, then snapshot the same group
again. To load a specific `.wlog` or host `jsonl` first, use
`analyzer_load <path>`. Full command and safety details are in
`software/gui/CLAUDE.md` under **Remote control / automation** and in
`software/gui/tools/robot_ctl.py`.

## Flashing (PlatformIO / GUI)

```
cd teensy && pio run -e teensy41 -t upload    # or esp32/ -e esp32dev
```

Or remotely, via the running GUI's command server (see `software/gui/CLAUDE.md`
→ "Remote control / automation"): `python software/gui/tools/robot_ctl.py
firmware_flash teensy`. Either way, PlatformIO builds first, then needs the
Teensy in its HalfKay bootloader to actually upload — normally triggered by a
soft-reset touch over the serial port, but that can fail to land (e.g. the
GUI already holds the port open in monitor mode) and `teensy_loader_cli` then
sits waiting for the bootloader indefinitely instead of erroring out.

**Expect to flash twice.** Symptom of the stuck first attempt: the build
succeeds but the command returns `ok: false` with a truncated/empty output
tail, and a `teensy_loader_cli` process is left running (check with
`Get-Process teensy_loader_cli` on Windows). A second `firmware_flash` call
while that's still alive fails fast with `"teensy flash is already
running"` — kill the stuck process first (`Stop-Process -Id <pid> -Force`),
then retry. The retry upload usually succeeds outright; if not, a physical
tap of the Teensy's PROGRAM button while the loader is waiting forces it into
bootloader mode. After a successful flash, confirm the new firmware is
actually running (not just that the loader exited 0) via `telem`'s
`timestamp_ms`/`loop_count` resetting to a small value — a stale-but-still-
responding board can look deceptively like a success.

## Generated protocol and durable parameters

The only human-edited source for shared state, fault, command, group, and
parameter definitions is `protocol/schema.json`. Run
`python protocol/generate_protocol.py` after editing it, and use `--check` in
tests/CI to detect stale generated artifacts. The generator writes the C++
IDs/tables, the pure-Python GUI module, documentation, and frozen vectors.
Generated files are checked in so embedded builds do not require Python.

Persistent Teensy parameters use two generation-numbered, CRC32-protected
LittleFS slots. A save is verified before it becomes current, the previous
valid slot remains available after interruption or corruption, and legacy
`params.bin` data is migrated on first boot. Recovery and migration are
reported in the robot log.

## Key constants (`teensy/src/config.h`)

| Symbol | Value | Meaning |
|---|---|---|
| `CONTROL_HZ` | 500 | Teensy main loop rate |
| `ESP32_BAUD` | 4 000 000 | Teensy↔ESP32 UART baud |
| `CAN_BAUD` | 1 000 000 | Both CAN buses |
| `CAN_INTER_FRAME_US` | 500 | Gap inserted between back-to-back CAN TX frames |
| `ODESC_NODE_L/R` | 0 / 1 | Wheel motor ODrive node IDs |
| `AK45_ID_L/R` | 11 / 12 | Hip motor CAN IDs |

## Coordinate system

+X forward, +Y left, +Z up (matches MuJoCo world frame).

## Control algorithm (`teensy/src/control_loop.cpp`)

LQR on 3-state linearised inverted pendulum: `[pitch−θ_ref−trim, pitch_rate, wheel_vel_avg−v_ref]`.  
Gains are scheduled with leg height (α ∈ [0,1], retracted→extended).  
Balance-point trim (`lqr_pitch_trim_ret/ext`, same α schedule) offsets the pitch
target so zero velocity holds at the true balance lean when the CG isn't over the
axle. Edit via the Params tab, or repoint a live-tune slot at it for a future
bench session (see below).  
Outer loops: velocity PI (sets θ_ref), yaw PI (differential torque). The
velocity PI uses directional conditional-integration anti-windup: its integral
freezes when an update would push the requested lean farther past the active
asymmetric clamp, but remains free to unwind back out of saturation.
Feedforward: FF1 cancels hip reaction torque; FF2 adds gravity compensation.

**Roll controller (active suspension)** — off by default (`roll_ctrl_en`). In
RUNNING only, a PD loop on roll angle/rate (`roll_kp`, `roll_kd`) produces a
differential hip position offset (`+offset` one leg, `−offset` the other,
clamped to `roll_offset_max` and to calibrated hip travel) that levels/leans the
body about +X. The setpoint comes from radio **CH1**, scaled by the active
profile's `radio_roll_max` (per-profile `profileN_roll_max`, selected by CH9) and
slew-limited (`roll_rate_lim`) so a snapped stick doesn't step-perturb pitch.
While active the hips are held with a soft, backdrivable impedance
(`hip_roll_kp`/`hip_roll_kd` replace `hip_running_kp`/`kd`; `hip_running_tff`
still carries the static leg load). A steady roll shifts the CG laterally toward
the low wheel and wheels make no lateral force, so the setpoint is hard-clamped
and a roll watchdog (`roll_watchdog_en`, `roll_watchdog_limit`) ESTOPs with
`FAULT_ROLL_WATCHDOG` if `|roll|` exceeds the limit for > 200 ms. FF1 uses the
hip-current *sum*, so pure differential roll motion cancels in it.

## Live parameter tuning (`teensy/src/live_tune.h`)

Lets an operator feel a gain/limit's effect live on the bench, via two radio
knobs (CH7, CH8), instead of editing the Params tab blind and re-arming to see
it. CH7/CH8 each drive one slot of whichever *gain group* is currently
selected by the CH5/CH6 switch combination — 3 groups of 2 slots, `LIVE_TUNE_SLOTS`
in `main.cpp`:

```cpp
static const LiveTuneSlot LIVE_TUNE_SLOTS[] = {
    // Group 0: CH5 down, CH6 up -- LQR pitch/rate (retracted)
    { 0, 7, PARAM_LIVE_TUNE_CH7_VAL, PARAM_LQR_K_PITCH_RET, -0.1f,  -0.5f },
    { 0, 8, PARAM_LIVE_TUNE_CH8_VAL, PARAM_LQR_K_RATE_RET,  -0.01f, -0.5f },
    // Group 1: CH5 up, CH6 down -- vel_pi KP/KI
    { 1, 7, PARAM_LIVE_TUNE_CH7_VAL, PARAM_VEL_PI_KP, 0.05f, 1.0f  },
    { 1, 8, PARAM_LIVE_TUNE_CH8_VAL, PARAM_VEL_PI_KI, 0.02f, 0.5f  },
    // Group 2: CH5 down, CH6 down -- roll KP/KD
    { 2, 7, PARAM_LIVE_TUNE_CH7_VAL, PARAM_ROLL_KP, 0.3f,  4.0f  },
    { 2, 8, PARAM_LIVE_TUNE_CH8_VAL, PARAM_ROLL_KD, 0.02f, 0.5f  },
};
```

Active while `RUNNING` with CH5+CH6 selecting a group (CH5 down + CH6 up ->
group 0, CH5 up + CH6 down -> group 1, both down -> group 2, both up -> no
group / tuning inactive). Two safety properties, independent of which params
are currently wired:

- **Pickup, not snap.** A knob does nothing on entering a group until it's
  swept through that slot's target *current* value; only then does it "pick
  up" and start tracking 1:1. Prevents a gain jumping instantly to wherever
  the knob physically happens to be sitting when the group is selected.
  Resets every time live-tune mode is exited or the group changes (CH5/CH6
  combination changes, or leaving `RUNNING`) — re-entering always requires
  re-sweeping.
- **Explicit latch, nothing persists by accident.** While picked up, a slot's
  live shadow value (`live_tune_ch7_val`/`live_tune_ch8_val`, telemetry-visible)
  is what the control loop actually uses — real, felt effect on balance — but
  the real underlying param is untouched until `PARAM_LIVE_TUNE_LATCH`
  (`live_tune_latch`) is written `1`. One-shot: firmware commits every
  currently-picked-up slot and resets the flag. Un-picked-up slots are
  skipped, not latched at a stale value.

**Repointing a knob at a different param** for a future bench session is a
one-line edit to `LIVE_TUNE_SLOTS` + reflash. The only other requirement: the
target's read site in `control_loop.cpp` must go through
`live_tune_value(PARAM_X)` instead of a bare `param_get(PARAM_X)` — that's
what makes the override actually take effect.

Full step-by-step operator procedure and the CH5/CH6/CH7/CH8 radio table: see
"Live parameter tuning" in `radio_channels.md`.

## Motor direction / sign conventions

Every place a sign flip or direction constant is applied between a "positive means X physically" intent and the raw motor command. Reference for bench-testing motor direction (`ControllersTest.md` Phase 1) and for anyone chasing a motor spinning the wrong way.

### All sign-affecting entities

| # | Name | Type | Location | Current value |
|---|---|---|---|---|
| 1 | Wheel L TX (no flip — reference side) | hardcoded | `teensy/lib/WheelMotors/wheel_motors.cpp:144` (`L_hw = L`) | identity (+1) |
| 2 | Wheel R TX flip | hardcoded | `wheel_motors.cpp:145` (`R_hw = -R`) | −1, unconditional |
| 3 | Wheel R RX flip (encoder feedback) | hardcoded | `wheel_motors.cpp:64` (`pos = -pos; vel = -vel;`) | −1, unconditional |
| 4 | Yaw→per-wheel torque split | hardcoded (structural) | `teensy/src/control_loop.cpp:242-243` | `tau_L = tau_sym + tau_yaw`, `tau_R = tau_sym − tau_yaw` |
| 5 | Hip L TX flip (commanded pos/vel/torque) | hardcoded | `teensy/lib/HipMotors/hip_motors.cpp:95-100` (`pack_and_send`) | −1, unconditional, applied when `id == AK45_ID_L` |
| 6 | Hip L RX flip (pos/vel/current feedback) | hardcoded | `hip_motors.cpp:137-140` (`rx_callback`) | −1, unconditional, applied when `msg.id == AK45_ID_L` |
| 7 | Hip L/R retract-switch seek direction | runtime param, `PARAM_FLAG_READONLY` | generated parameter table | L `-1.0`, R `+1.0`; each points toward its retract switch |
| 8 | Hip normalized-command mapping | hardcoded (structural) | `teensy/lib/HipMotors/hip_motors.cpp` (`hip_cmd_to_setpoints`) | `t∈[0,1]`: 0 = switch-zero backoff limit, 1 = configured extended limit; endpoint selection follows each axis's seek direction |
| 9 | GUI hip jog slider → raw degrees | GUI-side, not firmware | `software/gui/tabs/hip_motors.py:493-501` | slider low end → `lo_deg` (≈ `min_rad`), slider high end → `hi_deg` (≈ `max_rad`) — **raw degrees, not the normalized `t`**; now the same physical sense on both sides since #5/#6 unify the frame |

### Per-motor: what positive should do

| Motor | Applicable entities | Positive command means |
|---|---|---|
| **wheel_left** | #1 (no flip), #4 (`+tau_yaw` term) | Positive torque/velocity → wheel drives robot **forward (+X)**. This is the reference side; should stay unflipped. |
| **wheel_right** | #2, #3 (−1 flip both ways), #4 (`−tau_yaw` term) | Positive torque/velocity, in the *firmware-frame* value (same value the control loop and GUI use, before the internal CAN flip) → wheel also drives robot **forward (+X)** — same convention as left, because the −1 flip compensates for the physically mirrored mounting. Flip should stay in place; do not "fix" it by changing sign elsewhere. |
| **hip_right** | #7 (`dir_R=+1`), #8 | Reference side, no CAN-level flip. Raw MIT position: **more negative = leg extends**, near `max_rad` (≈0) = **retracted**. Increasing position retracts the leg. Via GUI jog slider (#9): dragging toward the **low** end extends, toward the **high** end retracts. |
| **hip_left** | #5, #6 (−1 flip both ways), #7 (`dir_L=+1`, same as R), #8 | Positive command/position, in the *firmware-frame* value (same value calibration, jump FSM, GUI, and radio all use, before the internal CAN flip) → behaves identically to hip_right: more negative = extend, increasing = retract. The −1 flip at the CAN boundary compensates for the physically mirrored mounting, so nothing above `hip_motors.cpp` needs to know left and right are wired differently. **Requires recalibration** after this fix — the previous calibration's zero point and limits were established in the old (unflipped, backwards) frame. |



Full FSM diagram: `teensy/state_machine.md`.

| Value | Name | Description |
|---|---|---|
| 0 | `STATE_STARTUP` | Boot checks: waits for IMU NOMINAL + hip CAN heartbeats |
| 1 | `STATE_CALIBRATION` | Hip retract-switch homing — only from STANDBY; lowering CH5 after a radio start cancels through DISARMING |
| 2 | `STATE_STANDBY` | Idle, motors energised but zero torque |
| 3 | `STATE_RUNNING` | Active balancing — LQR + vel/yaw PI — requires calibration valid |
| 4 | `STATE_ESTOP` | Fault latch — see fault table below |
| 5 | `STATE_MANUAL` | GUI direct control (MIT frames); watchdog 500 ms |
| 6 | `STATE_CMD_REJECT` | ~1 s transient: buzzer + red blink, auto-returns to prior state |
| 7 | `STATE_JUMPING` | ~3 s jump sequence from RUNNING; auto-returns to RUNNING |
| 8 | `STATE_STANDING_UP` | Arm-time recovery from a fallen pose — retract legs, energetic wheel push, then RUNNING |
| 9 | `STATE_DISARMING` | Normal active-state or radio-calibration exit: wheel IDLE immediately, hip torque ramps safely to zero, then STANDBY |

**Standing-up mode (`STATE_STANDING_UP`)**: entered on arm only when `standup_enable=1` and pitch is within the recoverable range; with `standup_enable=0` (default) arming goes straight to `RUNNING`, byte-identical to before this state existed. Two phases: **CROUCH** ramps the hips to the retracted pose and holds them there rigidly for the rest of the sequence — hips move once, to a fixed pose, and never actively right the robot. **RECOVER** does the actual catch entirely with wheel torque: a saturated P/D law on pitch and pitch-rate (`tau = K_pitch*pitch + K_rate*pitch_rate`, same sign convention as the small-angle LQR) pushes the wheelbase back under the CG until pitch settles in-band, then hands off to `RUNNING` for the tuned LQR to take over. Full spec: `standing_up.md`.

### Canonical state colour table

One source of truth for `state → RGB`; used by Teensy LED, ESP32 Neopixel base hue, TFT banner, and GUI header.

| State | R | G | B | Appearance |
|---|---|---|---|---|
| `STATE_STARTUP` | 255 | 255 | 255 | white |
| `STATE_CALIBRATION` | 0 | 120 | 255 | blue |
| `STATE_STANDBY` | 255 | 180 | 0 | amber |
| `STATE_RUNNING` | 0 | 230 | 80 | green |
| `STATE_ESTOP` | 255 | 40 | 40 | red |
| `STATE_MANUAL` | 0 | 200 | 255 | cyan |
| `STATE_CMD_REJECT` | 255 | 120 | 0 | orange |
| `STATE_JUMPING` | 200 | 0 | 255 | magenta |
| `STATE_STANDING_UP` | 255 | 60 | 0 | red-orange, fast strobe |
| `STATE_DISARMING` | 255 | 180 | 0 | amber blink during normal torque ramp-down |

## Fault codes (`shared/comm_protocol.h`)

Set in `g_state.fault_code` before entering `STATE_ESTOP`. Non-zero only while in ESTOP.

| Value | Name | Cause | Severity |
|---|---|---|---|
| `0x00` | `FAULT_NONE` | No fault | — |
| `0x01` | `FAULT_IMU_ERROR` | IMU ERROR during startup | REBOOT |
| `0x02` | `FAULT_HIP_INIT_TIMEOUT` | No CAN reply from hip motors within 2 s of boot | REBOOT |
| `0x03` | `FAULT_HIP_FEEDBACK_LOST` | Hip CAN feedback timed out (> 20 ms) during operation | REBOOT |
| `0x04` | `FAULT_HIP_LARGE_POS_CMD` | Hip position jump exceeded `MAX_HIP_DELTA_RAD` | GUI_FIX |
| `0x05` | `FAULT_CALIBRATION_TIMEOUT` | Retract-switch homing safety check failed | REPOSITION |
| `0x06` | `FAULT_HUMAN_ESTOP` | ESTOP requested by GUI or radio | SOFT |
| `0x07` | *(reserved)* | Was `FAULT_PARAM_OUT_OF_BOUNDS` — removed; out-of-range param writes always clamp | — |
| `0x08` | `FAULT_PITCH_WATCHDOG` | pitch outside `[-pitch_wd_bwd, +pitch_wd_fwd]` (asymmetric, gain-scheduled ret/ext by leg height) for > 200 ms | REPOSITION |
| `0x09` | `FAULT_WHEEL_RUNAWAY` | Wheel velocity exceeded 2× soft governor limit | SOFT |
| `0x0A` | `FAULT_IMU_LOST` | IMU left NOMINAL while RUNNING/JUMPING (silence or heavy packet loss) | REBOOT |
| `0x0B` | `FAULT_WHEEL_FEEDBACK_LOST` | Wheel encoder timeout or ODrive error during operation | REBOOT |
| `0x0C` | `FAULT_WHEEL_INIT_TIMEOUT` | No CAN reply from wheel motors within 2 s of boot | REBOOT |
| `0x0D` | `FAULT_STANDUP_FAILED` | Standup denied (pitch out of recoverable range) or exhausted retries/diverged | REPOSITION |
| `0x0E` | `FAULT_ROLL_WATCHDOG` | `|roll| > roll_watchdog_limit` for > 200 ms (lateral tip guard) | REPOSITION |

**Severity tiers:** SOFT → ESTOP→STANDBY directly; REPOSITION → reposition robot then reset; GUI_FIX → fix param in GUI then reset; REBOOT → power-cycle required.

Mirror `_FAULT_NAMES` / `_FAULT_DESCRIPTIONS` in `software/gui/tabs/telem_format.py` and `fault_description()` in `esp32/src/main.cpp` when adding/changing codes.

## Telemetry and param pipeline

```
Teensy main.cpp  send_telemetry()
    │  packs RobotState + sensor data into TelemetryPayload (242 bytes, TELEM_VERSION 9,
    │  see comm_protocol.h) — includes ESP32<->Teensy link-supervision fields (esp32_link_ok,
    │  esp32_status_age_ms, uart_rx_drops, uart_seq_gaps)
    │  splits into TELEM_A (118 bytes, offset 0) + TELEM_B (124 bytes, offset 118)
    │  sends both framed packets via CommLink at 50 Hz
    ▼
ESP32 on_teensy_packet()  (core 1, inside g_teensy.update()'s parse loop)
    │  version-checks TELEM_VERSION (mismatch → logged, packet dropped)
    │  copies fields into volatile g_telem_* globals
    │  enqueues the raw frame for uplink_task — never sends inline here (a blocking
    │  USB/TCP/UDP send in this parse loop was the root cause of intermittent
    │  NO TEENSY under WiFi load; see git log for "ESP32 Phase 1")
    ▼
ESP32 uplink_task  (core 0 — the only writer to Serial/TCP/UDP)
    │  drains control/result, ACK-paced bulk-log, and lossy telemetry queues
    │  in that order; every queue operation is bounded
    │  uses one CommLink sequence generator per physical output
    │  uses non-blocking TCP sends and disconnects a stalled peer after 250 ms
    forwards over USB serial (CP2102) and WiFi UDP/TCP as appropriate

Each ESP32 parser pass is byte-budgeted (`UART_PARSE_BUDGET_BYTES` and
`HOST_PARSE_BUDGET_BYTES`). Serial2 has one core-1 reader/writer, USB has one
core-1 reader and the sole core-0 writer, and each network stream has one
reader and one writer. No UART callback performs network or USB I/O.

For WiFi, the ESP32 reassembles each adjacent `TELEM_A`/`TELEM_B` pair and
sends one `COMM_TYPE_TELEM_FULL_WIFI` datagram at 50 Hz. UDP liveness and the
TCP command connection are supervised independently: losing telemetry marks
only the WiFi source stale, while a still-readable TCP connection remains
available for recovery. The GUI renews discovery/session ownership separately
and will rediscover an address after restart or DHCP change.

Independently, the ESP32 sends its own COMM_TYPE_ESP32_STATUS heartbeat to the
Teensy at 5 Hz (ESP32<->Teensy link supervision, telemetry-only), and its own
WIFI_DIAG (WifiDiagPayload V2, 38 bytes) to the GUI at 5 Hz — both keep flowing
even if the other side of the link goes quiet, so the GUI can tell "ESP32
alive, Teensy silent" apart from "everything down".

Python GUI  flash_monitor.py  PacketDecoder._parse()
    │  decodes with _FMT_TELEM_A / _FMT_TELEM_B (struct.calcsize asserted at import)
    │  emits dict via TelemetryBus.instance().packet signal
    ▼
GUI tabs  (imu_tab, raw_data_tab, robot_visualizer_tab, …)
    subscribe to TelemetryBus and render

Params (GUI → Teensy):
    GUI sends CMD_ID_PARAM_SET frames → CommLink → param_registry.cpp
    Teensy replies with PARAM_REPORT packets (min/max/flags/name per param)
    GUI renders controls from PARAM_REPORT — no hardcoded layout needed
    Normal GUI commands use CMD_PAYLOAD_V2: uint32 request_id + the existing
    command bytes. Teensy replies with COMMAND_RESULT (accepted/applied or a
    structured rejection reason), and ReliableCommand also verifies the
    operation-specific telemetry effect. V1 remains accepted during migration.
```

**Propagation checklist** when adding/removing telemetry fields: see the `PROPAGATION CHECKLIST` comment in `shared/comm_protocol.h`.

## Stress-test & fault-injection instrumentation

Built during the Phase 7-9 WiFi reboot-loop / reliability investigation (see
`UARTplat.md` at the repo root for the full narrative — root causes found,
fixes applied, and verification history). Kept in permanently, not stripped
from builds: gated by explicit commands, inert by default (zero behavior
change unless a test harness deliberately arms them). Use this instead of
writing new ad-hoc test scripts — it already speaks the real wire protocol
through the real Teensy/ESP32/GUI stack.

### Deliberate frame corruption (`CMD_ID_TEST_INJECT_CORRUPT`, `0x14`)

Asks the Teensy or ESP32 to deliberately damage its own next N outgoing
`TELEM_A` frames, to verify a receiver's `CommLink` parser actually detects
and drops a bad frame rather than accepting garbage. Payload:
`uint8_t count, uint8_t target, uint8_t mode` (`target`/`mode` default to
0/1 for backward compat with older callers).

| Field | Values | Meaning |
|---|---|---|
| `count` | 1-255 | how many upcoming `TELEM_A` sends to damage |
| `target` | 0 = UART | Teensy's next `count` sends to the ESP32 over Serial5 — forwarded through by the ESP32 like any command |
| | 1 = WiFi | ESP32's next `count` UDP datagrams to the GUI — intercepted by the ESP32 itself (`forward_to_teensy()`), never reaches the Teensy |
| `mode` | 1 = CRC | flips the CRC-8 byte (checksum-compare path) |
| | 2 = END | flips the END byte (`PS_END` bad-byte path) |
| | 3 = length | claims an oversized on-wire length (length-guard/resync path, `CommLink.cpp` Fix 2) |

Implemented via `CommLink::send()`'s `corrupt_mode_for_test` parameter
(`shared/CommLink/CommLink.{h,cpp}`) — default `0` = no corruption, so every
real call site is unaffected. Armed/consumed by `g_test_corrupt_remaining` /
`g_test_corrupt_mode` (Teensy, `teensy/src/main.cpp`) and
`g_test_corrupt_wifi_remaining` / `g_test_corrupt_wifi_mode` (ESP32,
`esp32/src/main.cpp`).

GUI sender: `comm_commands.send_test_inject_corrupt(count, target, mode)`
(`software/gui/tabs/comm_commands.py`). Ground truth for whether it worked:
watch `WIFI_DIAG.wifi_uart_crc_drops` for `target=0`, or this GUI's own
`link_crc_drops` (`PacketDecoder`, "wifi" transport) for `target=1` — note
mode 2 (END byte) frames are correctly rejected but don't increment either
counter (a GUI-side observability gap, not a detection failure — see
`command_corruption_probes` below for a mode-agnostic way to confirm this).

### Command-frame corruption, receive side (`build_frame_corrupted()`)

The corruption above only exercises the *send* side (Teensy/ESP32 sending a
deliberately bad `TELEM_A`). `comm_commands.build_frame_corrupted(payload,
mode)` builds a COMMAND frame with the same 3 corruption modes, for testing
the *receive* side of the identical `CommLink` parser instead — i.e.
confirming a malformed frame from the GUI can't get misdispatched as a
command. Send with `comm_commands.send_frame(...)` like any other frame.

### GUI automation harness (`main.py --automation <scenario.json>`)

```
python software/gui/main.py --automation path/to/scenario.json
```

Forces WiFi as the active transport, drives load through the real senders
(`comm_commands.py`, `WifiTransport`) for a fixed duration, and writes a
report to `software/gui/logs/<name>_report.json` before exiting — no GUI
interaction needed. Implemented in `software/gui/tabs/automation_runner.py`
(`AutomationRunner`); full field docs are in its module docstring.

| Scenario field | Type | Meaning |
|---|---|---|
| `name` | str | used in the default report filename |
| `duration_s` | float | timed run length, starting once WiFi telemetry is first seen |
| `param_dump_period_s` | float | interval between `PARAM_GET(0xFFFF)` dumps (load generator; default 5) |
| `tcp_churn_period_s` | float | if set, force-close/reopen the GUI's own WiFi TCP command socket on this interval |
| `bootstrap_timeout_s` | float | max wait for first WiFi telemetry packet before starting anyway (default 30) |
| `snapshot_period_s` | float | interval for periodic counter snapshots (default 30) |
| `corrupt_injections` | list | `{t_s, target: 0\|1, count, mode: 1\|2\|3}` — see corruption command above |
| `rogue_tcp_connects` | list | `{t_s, hold_ms}` — opens a second, independent raw TCP connection to the ESP32's command port alongside the GUI's own, to verify the accept-swap logic (`loop()`, `esp32/src/main.cpp`) survives an interloper client cleanly and the GUI's own connection self-heals afterward |
| `command_corruption_probes` | list | `{t_s, mode: 1\|2\|3, param_id}` — sends a corrupted `PARAM_GET`, confirms no reply leaked, then sends a valid follow-up and confirms the parser resynced (receive-side guard + recovery, mode-agnostic) |
| `report_path` | str | output path, relative to `software/gui/`; default `logs/<name>_report.json` |

Report highlights: `pass`/`fail_reasons` (note — **expect `pass: false`
whenever `corrupt_injections` is used**; the injected CRC/seq-gap counters
are exactly what the pass/fail check watches, so a "failing" report from a
corruption-injection scenario is normal, not a regression), `reboot_count`/
`reboot_events` (from `wifi_esp_uptime_ms` decreasing between packets),
`telemetry.actual_hz`/`inter_arrival_jitter`, per-direction UART counters,
`link_down_events`, `wifi_link_quality`, `injection_results`,
`rogue_connect_results`, `command_probe_results`.

Past example scenarios (kept for reference) and their reports:
`software/gui/logs/wifi_*_report.json`.

### Arm / state-machine stress test (`trigger_running_test.py`, `stress_test_arm.py`)

Built to bench-verify the arm state machine (`state_machine.cpp`) without
needing hands on the physical RC transmitter — e.g. after touching
`req_running()`/`req_calibration()` or the radio arm path in
`main.cpp radio_update()`. Standalone (pyserial only, no Qt — same pattern as
`tools/trigger_log_test.py`), in `software/gui/tools/`.

**`CMD_ID_SET_MODE(STATE_RUNNING)` is now a real command**, not just a radio
trigger — previously `on_command()`'s `SET_MODE` handler had no `RUNNING`
branch at all, so arming was only reachable via the physical CH10 switch
(`radio_update()`). It's now routed through the identical `req_running()`
gate the radio path uses (same IMU/calibration/motor-enable checks), so
there's no separate/weaker software arm path.

**`PARAM_RUNNING_WHEEL_BYPASS_EN`** (`0x0429`, `run_wheel_bypass_en`) — lets
`req_running()` arm with `wheel_l/r_enable` off, for a pure
command/state-machine smoke test with zero real torque anywhere when
combined with `hip_l/r_enable` also off (bypassed via the existing
`PARAM_CALIB_BYPASS_EN`). Independent of `PARAM_CALIB_BYPASS_EN`, which only
ever covered the hip check — that asymmetry was the bug this param fixes.
**Not persisted** — always boots to 0 (bypass off), unlike
`PARAM_CALIB_BYPASS_EN`, so it can't be left silently armed across a power
cycle.

**Software disarm is explicit.** `SET_MODE(STANDBY)` from `RUNNING`,
`JUMPING`, or `STANDING_UP` enters `STATE_DISARMING`. Fault/ESTOP guards have
higher priority, wheel output is idled immediately, jump/stand-up sequencing
is cancelled, and the hip command ramps to zero before STANDBY is reported.
Re-arm/manual/calibration requests are blocked during this interval.

**Radio disarm interlock fires regardless of arm source.**
`radio_update()`'s disarm check is level-based and covers every energetic state:
```cpp
bool armed = alive && (ch10 > 1990);
if (!armed && (g_state.state == STATE_RUNNING ||
               g_state.state == STATE_JUMPING ||
               g_state.state == STATE_STANDING_UP)) {
    stateMachine_disarm_running();
}
```
`alive` (iBus signal) is false whenever no RC receiver is connected, so
`armed` is always false — meaning a software-triggered `RUNNING` gets
sent to `DISARMING` on the next ~2 ms tick unless a live receiver also has
CH10 physically held up. This is intentional (RUNNING should only *persist*
with a live radio link corroborating "armed", regardless of entry path) —
confirmed and left as-is rather than "fixed". Both tools below account for
this: they check for the `-> RUNNING (armed)` log line (the `on_running()`
entry action — a guaranteed one-shot event) rather than expecting the
`TELEM_A` `robot_state` field to still read `RUNNING` by the time they check,
since the armed state can be shorter than one telemetry period.

#### `trigger_running_test.py` — single arm + confirm

```
python trigger_running_test.py [port]
```

Sends `CMD_ID_SET_MODE(STATE_RUNNING)` once and reports whether the state
machine actually armed, via `comm_log` lines and live `TELEM_A` decoding
(`tabs.telem_format.decode_telem_a`, Qt-free, imported directly). Prints the
pre-arm state first and warns if it isn't `STANDBY` (the request will be a
silent no-op — `stateMachine_request_running()`'s own `STANDBY`-only latch).

#### `stress_test_arm.py` — multi-round stress test

```
python stress_test_arm.py [--port COM12] [--rounds 5]
```

Per round: a fuzzed out-of-range `SET_MODE` target (expect safe no-op, no
crash), a `CALIBRATION` request (expect denial while hips are disabled), an
arm into `RUNNING` (expect the `-> RUNNING (armed)` log line, then recovery
to `STANDBY`), and explicit recovery before the next round. Prints a
pass/fail summary table at the end and exits non-zero on any failure.

**Safety gate, checked live before anything else runs:** reads
`hip_l/r_enable`, `wheel_l/r_enable`, `imu_enable`, `calib_bypass_en`, and
`run_wheel_bypass_en` via `CMD_ID_PARAM_GET` and aborts if any motor-enable
param is actually on (RUNNING would command real torque), or if
`imu_enable`/`calib_bypass_en` are off (test wouldn't be meaningful/would be
denied outright). `run_wheel_bypass_en` is the one param it will flip on
itself if needed — safe because it's non-persistent — and it flips it back
off during cleanup regardless of pass/fail.

**Out of scope — no software RC channel injection.** Neither tool can
simulate physical transmitter input. `PARAM_IBUS_CH*` and the radio-derived
params (`radio_vel_max`, `radio_yaw_max`, `active_profile`, ...) are all
`PARAM_FLAG_READONLY` — firmware-written mirrors of the real iBus receiver,
with no command-side injection point. Profile switching (CH9) in particular
has no non-radio trigger at all right now. "Radio commands" in these tools
means the same *state transitions* the radio triggers (`CALIBRATION`,
`RUNNING`), issued instead through `CMD_ID_SET_MODE`.

### ESP-IDF task watchdog (diagnostic, not invoked manually)

`esp_task_wdt_init()`/`esp_task_wdt_add(uplink_task_handle)` in
`esp32/src/main.cpp` — kept permanently as a safety net, not a test you run.
This originally caught `uplink_task` hanging inside a blocking
`WiFiClient::write()` call. Production TCP output now uses `MSG_DONTWAIT` and
disconnects a non-reading client after 250 ms; the watchdog remains a final
safety net for any unrelated future task stall.

## Each driver has its own README

See `teensy/lib/<DriverName>/README.md` for wiring, API, and gotchas.
