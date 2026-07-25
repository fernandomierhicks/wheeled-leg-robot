# IMU

> **AI maintenance note:** If you find anything here that is stale while
> working in this tree, update this README in the same change.

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
imu_init();  // call once, then poll imu_state() until it leaves INITIALIZING
// setup() blocks here on purpose — see "Startup vs runtime" below:
while (imu_state() == ImuState::INITIALIZING) imu_update();
// control loop:
imu_update();  // fills data; only blocks if state is still INITIALIZING/ERROR (see below)
if (imu_state() == ImuState::NOMINAL) {
    float pitch = imu_pitch();       // rad
    float rate  = imu_pitch_rate();  // rad/s
}
```

Full accessor list in `IMU.h`. Two fusion paths:
- **Game Rotation Vector** (`imu_pitch/roll/yaw`) — no magnetometer, immune to motor field disturbance. **Use this for balance control.**
- **Rotation Vector** (`imu_pitch_mag/roll_mag/yaw_mag`) — magnetometer fused, absolute heading.

## State machine

`NOT_READY → INITIALIZING → NOMINAL ↔ DEGRADED → ERROR (terminal)`

Do not enable motion until `imu_state() == NOMINAL`. Unlike a lot of retry logic, **`ERROR` does not auto-recover** — see below.

## Startup vs runtime: two very different failure paths

- **Boot (`imu_init()` from `setup()`):** `main.cpp` blocks in a tight loop calling `imu_update()` until the state leaves `INITIALIZING` (up to `MAX_INIT_ATTEMPTS` tries in `IMU.cpp`, each ~1 s — see below). This is deliberate: `STARTUP` doesn't need to hold the 500 Hz tick budget (no torque is commanded yet), so it's fine to take however long it takes to get a real answer instead of smearing a blocking SPI call across loop() ticks.
- **Runtime (sensor drops out mid-operation, i.e. after `STARTUP`):** `imu_update()` goes straight to terminal `ERROR` on a >100 ms silence timeout — it does **not** retry. A retry would mean calling the ~1 s blocking `attempt_init()` from inside a live 500 Hz loop() tick, which is never acceptable outside STARTUP. Recovery requires a reboot (`imu_init()` resets the attempt budget). Whatever consumes `imu_state()` (state machine, telemetry, etc.) is expected to react immediately to leaving `NOMINAL`, not wait around for a reconnect that will never come.

## Gotchas

**Gate motion on NOMINAL** — `imu_update()` transitions to `ERROR` if the sensor is silent for >100 ms. The caller must gate the control output on `imu_state() == NOMINAL` (or `DEGRADED`); stale pitch/rate values are not zeroed on error.

**`attempt_init()`/`begin_SPI` blocks ~0.7-1.0 s, not ~100 ms** — the BNO086 reset handshake (`hardwareReset()` + `sh2_open()` + `sh2_getProdIds()`) is synchronous with no timeout of its own; measured on real hardware via `test_imu`'s SH2 debug trace. An underlying SH2 library timeout used to be set to 100 ms (see `lib/Adafruit_BNO08x/`'s patched `sh2.c`), which caused every real connection attempt to spuriously fail before the sensor finished responding — fixed by raising it to 3 s. Only ever called during the boot-time blocking wait now (see above); a runtime dropout never reaches this call.

**Post-reset drain** — after `wasReset()` the sensor queues stale events. The driver drains 10 dummy events and then re-enables reports; do not read data until the next `imu_update()` returns cleanly.

**Packet loss tracking** — `imu_packet_loss()` returns a 0–1 rolling 1-second fraction based on SH2 sequence number gaps. `DEGRADED` fires at ≥ 10% loss. Gaps > 63 are treated as sensor resets and not counted.
