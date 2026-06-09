# WheelMotors

ODrive / ODESC wheel motor driver over CAN3 (FlexCAN_T4).

## Wiring

| Signal | Teensy 4.1 pin |
|---|---|
| CAN3 TX | 31 |
| CAN3 RX | 30 |

CAN bus: 1 Mbps. Left axis = ODrive node 0, right = node 1 (set in `config.h`).

## API

```cpp
wheel_motors_init();                        // call once in setup()
wheel_motors_set_mode(WheelMode::VELOCITY); // arm motors
// control loop:
wheel_motors_poll();                        // refresh wm_L/wm_R.ok, auto-IDLE on fault
wheel_motors_send(vel_L_rad_s, vel_R_rad_s);
wheel_motors_pet_watchdog();                // keep ODrive watchdog alive every tick
// on fault recovery:
wheel_motors_clear_errors();
wheel_motors_set_mode(WheelMode::VELOCITY);
```

Feedback (`pos_turns`, `vel_turns_s`) and heartbeat (`error`, `axis_state`) arrive via interrupt. `wm_L` / `wm_R` globals hold the latest state.

## Modes

| `WheelMode` | `send()` unit |
|---|---|
| `IDLE` | — |
| `VELOCITY` | rad/s (converted → turns/s internally) |
| `POSITION` | rad (converted → turns internally) |
| `TORQUE` | N·m |

## Gotchas

**Auto-IDLE on fault** — `wheel_motors_poll()` calls `wheel_motors_set_mode(IDLE)` automatically whenever any axis reports a timeout or non-zero error word. After clearing errors you must explicitly re-arm with `set_mode()`.

**Watchdog keepalive** — call `wheel_motors_pet_watchdog()` every tick unconditionally. In IDLE it sends a zero-velocity frame to prevent the ODrive from entering error state due to watchdog timeout. In active modes, the regular `wheel_motors_send()` already serves as the keepalive.

**Inter-frame delay** — a 500 µs gap between each back-to-back CAN frame pair is required; see `CAN_INTER_FRAME_US` in `config.h`.

**VBUS is async** — call `wheel_motors_request_vbus()`, then read `wm_L.vbus` / `wm_R.vbus` a few ms later after the ODrive reply arrives via interrupt.
