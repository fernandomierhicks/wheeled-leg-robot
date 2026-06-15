# robot_teensy firmware

Two-microcontroller architecture for a wheeled-leg balancing robot.

## MCU split

| MCU | Role | Toolchain |
|---|---|---|
| **Teensy 4.1** | Real-time control loop (500 Hz) | PlatformIO, Arduino framework |
| **ESP32** | Telemetry, WiFi, display | PlatformIO, Arduino framework |

The Teensy owns every safety-critical bus: CAN for motors, SPI for IMU, UART for RC receiver.
The ESP32 receives telemetry over UART (`Serial5` on Teensy ↔ UART on ESP32) and pushes it to a browser dashboard and/or a small TFT display. It sends commands back the same way (Teensy calls `g_esp32.update()` every tick and registers a callback for inbound packets).

```
[Teensy 4.1]  ←UART 1.2 Mbps→  [ESP32]  →WiFi→  browser dashboard
     │                                   →SPI→   TFT display
     ├─CAN2 1 Mbps─→ AK45-10 L/R (hip motors, MIT Cheetah)
     ├─CAN3 1 Mbps─→ ODrive/ODESC L/R (wheel motors)
     ├─SPI0──────── BNO086 IMU
     ├─Serial2/3──  AK45-10 UART encoder readback
     └─Serial4 RX── FlySky iBUS RC receiver
```

## Directory layout

```
firmware/robot_teensy/
├── teensy/          ← Teensy 4.1 PlatformIO project
│   ├── src/
│   │   └── config.h          ← all pin and bus constants
│   └── lib/
│       ├── AK45Uart/         ← AK45-10 UART encoder readback
│       ├── HipMotors/        ← AK45-10 MIT Cheetah CAN driver (CAN2)
│       ├── WheelMotors/      ← ODrive CAN driver (CAN3)
│       ├── IMU/              ← BNO086 SPI driver
│       ├── Esp32Link/        ← UART link to ESP32
│       ├── LED/              ← Non-blocking RGB LED
│       ├── Buzzer/           ← Non-blocking passive buzzer
│       ├── CommLink/         ← Framed UART protocol (used by Esp32Link)
│       ├── RC/               ← iBUS RC receiver
│       ├── SDLogger/         ← SD card logger stub
│       └── ToF/              ← VL53L1X time-of-flight stub
└── esp32/           ← ESP32 PlatformIO project
    └── lib/
        ├── TeensyLink/       ← UART link from ESP32 side
        ├── Display/          ← TFT display
        └── WiFiLink/         ← WiFi telemetry server
```

## Key constants (teensy/src/config.h)

| Symbol | Value | Meaning |
|---|---|---|
| `CONTROL_HZ` | 500 | Teensy main loop rate |
| `ESP32_BAUD` | 1 200 000 | Teensy↔ESP32 UART baud |
| `CAN_BAUD` | 1 000 000 | Both CAN buses |
| `CAN_INTER_FRAME_US` | 500 | Gap inserted between back-to-back CAN TX frames |
| `CAN_TIMEOUT_MS` | 20 | Motor feedback watchdog window |
| `AK45_ID_L/R` | 11 / 12 | Hip motor CAN IDs |
| `ODESC_NODE_L/R` | 0 / 1 | Wheel motor ODrive node IDs |
| `AK45_UART_BAUD` | 921 600 | AK45 encoder UART baud |

## Coordinate system

+X forward, +Y left, +Z up (matches MuJoCo world frame).

## Control algorithm

LQR on 3-state linearised inverted pendulum: `[pitch−θ_ref, pitch_rate, wheel_vel_avg−v_ref]`.
Hip motors hold leg height; wheel motors balance and drive.

## Robot states (`teensy/src/robot_state.h` + `state_machine.cpp`)

| Value | Name | LED | GUI colour |
|---|---|---|---|
| 0 | `STATE_STARTUP` | white breathe | grey |
| 1 | `STATE_CALIBRATION` | blue breathe | blue — hip hardstop homing, only from STANDBY |
| 2 | `STATE_STANDBY` | amber breathe | yellow |
| 3 | `STATE_RUNNING` | green breathe | green *(stub — no transitions yet)* |
| 4 | `STATE_ESTOP` | red blink | red |
| 5 | `STATE_MANUAL` | cyan breathe | cyan |

State diagram: `teensy/state_machine.md`.

## Fault codes (`shared/comm_protocol.h`)

Set in `g_state.fault_code` before any transition to `STATE_ESTOP`. Transmitted in every telemetry packet; non-zero only while in ESTOP.

| Value | Name | Cause |
|---|---|---|
| `0x00` | `FAULT_NONE` | No fault |
| `0x01` | `FAULT_IMU_ERROR` | IMU reported ERROR during startup |
| `0x02` | `FAULT_HIP_INIT_TIMEOUT` | No CAN reply from hip motors within 2 s of boot |
| `0x03` | `FAULT_HIP_FEEDBACK_LOST` | Hip CAN feedback timed out during operation (> 20 ms) |
| `0x04` | `FAULT_HIP_LARGE_POS_CMD` | Commanded hip position jump exceeded `MAX_HIP_DELTA_RAD` |
| `0x05` | `FAULT_CALIBRATION_TIMEOUT` | Hardstop not found within `CALIB_SAFETY_BOUND_RAD` during calibration |
| `0x06` | `FAULT_HUMAN_ESTOP` | ESTOP requested by user via GUI button |

Defined in `shared/comm_protocol.h` (lines ~38-45). Mirror `_FAULT_NAMES` in `software/gui/flash_monitor.py` when adding/changing codes.

## Each driver has its own README

See `teensy/lib/<DriverName>/README.md` for wiring, API, and gotchas.
