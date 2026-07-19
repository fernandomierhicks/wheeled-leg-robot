# robot_teensy firmware

Two-microcontroller architecture for a wheeled-leg balancing robot.

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
│  WiFi UDP broadcast  → Python GUI (software/gui/)               │
│  WiFi TCP server     ← GUI commands                             │
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

**Network exposure (accepted risk — home lab only):** the ESP32 TCP command
port (`:5006`) accepts a connection from **any device on the WLAN** and will
forward mode changes, hip MIT commands, and reboots to the Teensy; telemetry
is UDP-broadcast to `255.255.255.255:5005`. There is no authentication. Do
not operate the robot on an untrusted network; if this ever matters, cheap
hardening options are accept-first-client-only or a magic token as the first
bytes of a TCP session.

**Key buses on Teensy:**
- CAN2 @ 1 Mbps → AK45-10 hip motors (MIT Cheetah protocol)
- CAN3 @ 1 Mbps → ODrive wheel motors
- SPI0 → BNO086 IMU (CS=D10, INT=D2, RST=D3)
- Serial2/3 → AK45 UART encoder readback
- Serial4 RX → FlySky iBUS RC receiver
- Serial5 ↔ ESP32 (CommLink UART)

## Directory layout

```
firmware/robot_teensy/
├── shared/
│   └── comm_protocol.h   ← packet types, TelemetryPayload, fault codes (single source of truth)
├── teensy/               ← Teensy 4.1 PlatformIO project
│   ├── src/
│   │   ├── config.h          ← all pin and bus constants
│   │   ├── robot_state.h     ← RobotState struct + RobotStateEnum
│   │   ├── state_machine.cpp ← 8-state FSM (see state_machine.md)
│   │   ├── control_loop.cpp  ← LQR + vel PI + yaw PI + feedforward
│   │   └── main.cpp          ← loop(), radio, telemetry, LED, buzzer
│   └── lib/
│       ├── AK45Uart/         ← AK45-10 UART encoder readback
│       ├── HipMotors/        ← AK45-10 MIT Cheetah CAN driver (CAN2)
│       ├── WheelMotors/      ← ODrive CAN driver (CAN3)
│       ├── IMU/              ← BNO086 SPI driver
│       ├── LED/              ← Non-blocking RGB LED
│       ├── Buzzer/           ← Non-blocking passive buzzer
│       ├── CommLink/         ← Framed UART protocol (shared with ESP32)
│       ├── ParamRegistry/    ← Runtime param table (GUI-tunable, 500 Hz safe)
│       └── Calibration/      ← Hip hardstop calibration FSM
└── esp32/                ← ESP32 PlatformIO project
    └── src/main.cpp      ← all ESP32 logic (display, WiFi, Neopixel, ToF)
```

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

LQR on 3-state linearised inverted pendulum: `[pitch−θ_ref, pitch_rate, wheel_vel_avg−v_ref]`.  
Gains are scheduled with leg height (α ∈ [0,1], retracted→extended).  
Outer loops: velocity PI (sets θ_ref), yaw PI (differential torque).  
Feedforward: FF1 cancels hip reaction torque; FF2 adds gravity compensation.

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
| 7 | Hip L/R hardstop seek direction | runtime param, `PARAM_FLAG_READONLY` | `teensy/lib/ParamRegistry/param_registry.cpp:82,85` | `+1.0` for both — identical, because #5/#6 already handle the mirroring, exactly like wheels (`hip_motors.h:1-4` explains the frame) |
| 8 | Hip normalized-command mapping | hardcoded (structural) | `teensy/lib/HipMotors/hip_motors.cpp:277-282` (`hip_cmd_to_setpoints`) | `t∈[0,1]` (0=retract, 1=extend) → `pos = max_rad − t·span` when `dir>0` — same formula, same sign, both sides |
| 9 | GUI hip jog slider → raw degrees | GUI-side, not firmware | `software/gui/tabs/hip_motors.py:493-501` | slider low end → `lo_deg` (≈ `min_rad`), slider high end → `hi_deg` (≈ `max_rad`) — **raw degrees, not the normalized `t`**; now the same physical sense on both sides since #5/#6 unify the frame |

Note: `teensy/src/config.h:56` has a stale comment claiming "R seek dir −1" — that's out of date; #7 above (`param_registry.cpp`, both `+1.0`) is ground truth.

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
| 1 | `STATE_CALIBRATION` | Hip hardstop homing — only from STANDBY via CH5 |
| 2 | `STATE_STANDBY` | Idle, motors energised but zero torque |
| 3 | `STATE_RUNNING` | Active balancing — LQR + vel/yaw PI — requires calibration valid |
| 4 | `STATE_ESTOP` | Fault latch — see fault table below |
| 5 | `STATE_MANUAL` | GUI direct control (MIT frames); watchdog 500 ms |
| 6 | `STATE_CMD_REJECT` | ~1 s transient: buzzer + red blink, auto-returns to prior state |
| 7 | `STATE_JUMPING` | ~3 s jump sequence from RUNNING; auto-returns to RUNNING |

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

## Fault codes (`shared/comm_protocol.h`)

Set in `g_state.fault_code` before entering `STATE_ESTOP`. Non-zero only while in ESTOP.

| Value | Name | Cause | Severity |
|---|---|---|---|
| `0x00` | `FAULT_NONE` | No fault | — |
| `0x01` | `FAULT_IMU_ERROR` | IMU ERROR during startup | REBOOT |
| `0x02` | `FAULT_HIP_INIT_TIMEOUT` | No CAN reply from hip motors within 2 s of boot | REBOOT |
| `0x03` | `FAULT_HIP_FEEDBACK_LOST` | Hip CAN feedback timed out (> 20 ms) during operation | REBOOT |
| `0x04` | `FAULT_HIP_LARGE_POS_CMD` | Hip position jump exceeded `MAX_HIP_DELTA_RAD` | GUI_FIX |
| `0x05` | `FAULT_CALIBRATION_TIMEOUT` | Hardstop not found within `CALIB_SAFETY_BOUND_RAD` | REPOSITION |
| `0x06` | `FAULT_HUMAN_ESTOP` | ESTOP requested by GUI or radio | SOFT |
| `0x07` | *(reserved)* | Was `FAULT_PARAM_OUT_OF_BOUNDS` — removed; out-of-range param writes always clamp | — |
| `0x08` | `FAULT_PITCH_WATCHDOG` | `|pitch| > 50°` for > 200 ms | REPOSITION |
| `0x09` | `FAULT_WHEEL_RUNAWAY` | Wheel velocity exceeded 2× soft governor limit | SOFT |
| `0x0A` | `FAULT_IMU_LOST` | IMU left NOMINAL while RUNNING/JUMPING (silence or heavy packet loss) | REBOOT |
| `0x0B` | `FAULT_WHEEL_FEEDBACK_LOST` | Wheel encoder timeout or ODrive error while RUNNING/JUMPING | REBOOT |

**Severity tiers:** SOFT → ESTOP→STANDBY directly; REPOSITION → reposition robot then reset; GUI_FIX → fix param in GUI then reset; REBOOT → power-cycle required.

Mirror `_FAULT_NAMES` / `_FAULT_DESCRIPTIONS` in `software/gui/flash_monitor.py` and `fault_description()` in `esp32/src/main.cpp` when adding/changing codes.

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
    forwards over USB serial (CP2102) and WiFi UDP/TCP as appropriate

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
    Mode/param/reboot commands are retried by the GUI (ReliableCommand,
    tabs/comm_commands.py) against an observed telemetry effect — no protocol
    change, no firmware-side ACK.
```

**Propagation checklist** when adding/removing telemetry fields: see the `PROPAGATION CHECKLIST` comment in `shared/comm_protocol.h`.

## Each driver has its own README

See `teensy/lib/<DriverName>/README.md` for wiring, API, and gotchas.  
Active display redesign plan: `esp32/screen redo.md`.
