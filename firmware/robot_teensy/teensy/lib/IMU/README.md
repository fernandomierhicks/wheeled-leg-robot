# IMU

BNO086 driver — SPI, Game Rotation Vector + Rotation Vector + Gyro at 400 Hz.

## Wiring (SPI0)

| Signal | Teensy 4.1 pin |
|---|---|
| CS | 10 |
| INT | 9 |
| RST | 6 |
| SCK | 13 |
| MOSI | 11 |
| MISO | 12 |

PS0 and PS1 must be bridged on the BNO086 module to select SPI mode.

## API

```cpp
imu_init();    // call once in setup(); sets state to INITIALIZING
// control loop:
imu_update();  // non-blocking; drives state machine + fills data
if (imu_state() == ImuState::NOMINAL) {
    float pitch = imu_pitch();       // rad
    float rate  = imu_pitch_rate();  // rad/s
}
```

Full accessor list in `IMU.h`. Two fusion paths:
- **Game Rotation Vector** (`imu_pitch/roll/yaw`) — no magnetometer, immune to motor field disturbance. **Use this for balance control.**
- **Rotation Vector** (`imu_pitch_mag/roll_mag/yaw_mag`) — magnetometer fused, absolute heading.

## State machine

`NOT_READY → INITIALIZING → NOMINAL ↔ DEGRADED → ERROR → INITIALIZING …`

Do not enable motion until `imu_state() == NOMINAL`. On error, the driver retries every 1 s automatically.

## Gotchas

**Gate motion on NOMINAL** — `imu_update()` transitions to `ERROR` if the sensor is silent for >100 ms. The caller must gate the control output on `imu_state() == NOMINAL` (or `DEGRADED`); stale pitch/rate values are not zeroed on error.

**`begin_SPI` blocks ~100 ms** — the BNO086 reset handshake takes ~100 ms on first connect. This is acceptable at startup but the same delay applies on every auto-retry from `ERROR` state; keep motion gated during that window.

**Post-reset drain** — after `wasReset()` the sensor queues stale events. The driver drains 10 dummy events and then re-enables reports; do not read data until the next `imu_update()` returns cleanly.

**Packet loss tracking** — `imu_packet_loss()` returns a 0–1 rolling 1-second fraction based on SH2 sequence number gaps. `DEGRADED` fires at ≥ 10% loss. Gaps > 63 are treated as sensor resets and not counted.
