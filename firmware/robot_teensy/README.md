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
│       ├── Esp32Link/        ← UART link to ESP32 (CommLink)
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

## Robot states (`teensy/src/robot_state.h` + `state_machine.cpp`)

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
    │  packs RobotState + sensor data into TelemetryPayload (235 bytes, see comm_protocol.h)
    │  splits into TELEM_A (118 bytes, offset 0) + TELEM_B (117 bytes, offset 118)
    │  sends both framed packets via CommLink at 50 Hz
    ▼
ESP32 on_teensy_packet()
    │  version-checks TELEM_VERSION (mismatch → logged, packet dropped)
    │  copies fields into volatile g_telem_* globals
    │  forwards raw frame over USB serial (CP2102) and WiFi UDP
    ▼
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
```

**Propagation checklist** when adding/removing telemetry fields: see the `PROPAGATION CHECKLIST` comment in `shared/comm_protocol.h`.

## Each driver has its own README

See `teensy/lib/<DriverName>/README.md` for wiring, API, and gotchas.  
Active display redesign plan: `esp32/screen redo.md`.
