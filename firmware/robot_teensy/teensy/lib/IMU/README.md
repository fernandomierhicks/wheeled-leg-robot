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

### Report rates — not all four are equal

| Report | Rate | Consumer |
|---|---|---|
| Game Rotation Vector | 400 Hz | attitude → LQR — **required** |
| Gyro (calibrated) | 400 Hz | body rates → LQR — **required** |
| Rotation Vector (mag) | 10 Hz | absolute heading; no consumer in `src/` today |
| Linear acceleration | 50 Hz | telemetry display only |

Every enabled report costs SHTP events out of one shared FIFO, drained against a single per-tick budget (`MAX_DRAIN_EVENTS = 8` at the 500 Hz tick ⇒ 4000 events/s ceiling). All four at 400 Hz asked for 1600/s — under 3× headroom, and a backlog starves whichever report is queued behind the others. That is the leading suspect for the 2026-08-09 gyro stall. The current split costs ~860/s.

**If you add a report, spend from this budget deliberately** and put required streams ahead of aux ones in `enable_reports()`.

### Euler convention: intrinsic Z-Y-X

`quat_to_euler()` uses **yaw → pitch → roll (ZYX)**, the aerospace standard. The property that matters: **pitch is the nose-down lean of body +X out of horizontal, independent of heading** — exactly what the balance controller regulates. Singularity at pitch = ±90° (nose vertical), where roll and yaw go degenerate; the pitch watchdog ESTOPs long before that, and pitch itself stays exact regardless.

**Do not "fix" this to Y-X-Z.** Putting roll in the middle looks appealing for a machine that falls fore/aft (singularity moves to |roll| = 90°, on its side), but YXZ pitch is *heading-dependent*: it reports a true 10° lean as anywhere from −9° to +54° depending on which way the robot is facing. Verified numerically 2026-08-09.

## State machine

`NOT_READY → INITIALIZING → NOMINAL ↔ DEGRADED → ERROR (terminal)`

Do not enable motion until `imu_state() == NOMINAL`. Unlike a lot of retry logic, **`ERROR` does not auto-recover** — see below.

## Startup vs runtime: two very different failure paths

- **Boot (`imu_init()` from `setup()`):** `main.cpp` blocks in a tight loop calling `imu_update()` until the state leaves `INITIALIZING` (up to `MAX_INIT_ATTEMPTS` tries in `IMU.cpp`, each ~1 s — see below). This is deliberate: `STARTUP` doesn't need to hold the 500 Hz tick budget (no torque is commanded yet), so it's fine to take however long it takes to get a real answer instead of smearing a blocking SPI call across loop() ticks.
- **Runtime (sensor drops out mid-operation, i.e. after `STARTUP`):** `imu_update()` goes straight to terminal `ERROR` on a >100 ms silence timeout — it does **not** retry. A retry would mean calling the ~1 s blocking `attempt_init()` from inside a live 500 Hz loop() tick, which is never acceptable outside STARTUP. Recovery requires a reboot (`imu_init()` resets the attempt budget). Whatever consumes `imu_state()` (state machine, telemetry, etc.) is expected to react immediately to leaving `NOMINAL`, not wait around for a reconnect that will never come.

## Gotchas

**Gate motion on NOMINAL** — `imu_update()` transitions to `ERROR` if the sensor is silent for >100 ms. The caller must gate the control output on `imu_state() == NOMINAL` (or `DEGRADED`); stale pitch/rate values are not zeroed on error.

**Silence is tracked per stream, not per sensor** — the Game Rotation Vector and the gyro are separate reports and stall *independently*. On 2026-08-09 the gyro stream stopped while GRV kept flowing: `imu_pitch_rate()` froze at 0.000 for 3.66 s while `imu_pitch()` kept moving, `imu_state()` stayed `NOMINAL` throughout, and the balance controller's whole rate-damping term went silently dead. The watchdog now stamps and checks both, and `imu_last_update_ms()` returns the *older* of the two so callers doing their own freshness check cover the gyro for free. The other two reports (RV/mag, linear accel) are **not** watched — nothing in the control path reads them.

Note this catches a stream that stops *arriving*, not one that keeps arriving with a stuck value. The 2026-08-09 failure was the former (the value froze because nothing overwrote it).

**`attempt_init()`/`begin_SPI` blocks ~0.7-1.0 s, not ~100 ms** — the BNO086 reset handshake (`hardwareReset()` + `sh2_open()` + `sh2_getProdIds()`) is synchronous with no timeout of its own; measured on real hardware via `test_imu`'s SH2 debug trace. An underlying SH2 library timeout used to be set to 100 ms (see `lib/Adafruit_BNO08x/`'s patched `sh2.c`), which caused every real connection attempt to spuriously fail before the sensor finished responding — fixed by raising it to 3 s. Only ever called during the boot-time blocking wait now (see above); a runtime dropout never reaches this call.

**Post-reset drain** — after `wasReset()` the sensor queues stale events. The driver drains 10 dummy events and then re-enables reports; do not read data until the next `imu_update()` returns cleanly.

**A mid-run `wasReset()` re-enable failure is fatal** — `enable_reports()` bails on its first failure, so a partial re-enable (GRV back, gyro not) would otherwise leave the robot balancing on a frozen rate with `NOMINAL` still set. `imu_update()` now checks that return value and goes to `ERROR`. This is the most likely cause of the 2026-08-09 stall.

**Packet loss tracking** — `imu_packet_loss()` returns a 0–1 rolling 1-second fraction based on SH2 sequence number gaps. `DEGRADED` fires at ≥ 10% loss. Gaps > 63 are treated as sensor resets and not counted.
