# BNO086 Characterization — Adafruit Library (SPI)

## Hardware
- **IMU:** BNO086 (SparkFun breakout)
- **MCU:** Arduino UNO R4 WiFi (RA4M1, 48 MHz)
- **Interface:** SPI, CS=D10, INT=D2, RST=D3
- **Wiring:** PS0+PS1 bridged for SPI mode

## Test Setup
Three 10-second burst tests, each requesting the fastest possible report interval (1 ms):
1. **Game Rotation Vector (GRV)** — fused quaternion orientation
2. **Raw Gyroscope** — uncalibrated ADC gyro output
3. **Calibrated Gyro (ARVR Gyro)** — calibrated gyroscope

Timestamps buffered in RAM (no serial prints during capture) to eliminate I/O overhead.

## Results

### Stock Adafruit library (SPI @ 1 MHz, `delay(1)` INT poll)

| Report Type        | Effective Hz | Notes                        |
|--------------------|-------------|------------------------------|
| Game Rotation Vec  | 326 Hz      | SPI bus-limited              |
| Raw Gyroscope      | 402 Hz      | BNO086 internal limit        |
| ARVR Calibrated Gyro | 0 Hz     | Not supported by Adafruit lib |

### Patched library (SPI @ 3 MHz, tight-spin INT poll)

| Report Type        | Effective Hz | Notes                        |
|--------------------|-------------|------------------------------|
| Game Rotation Vec  | 400 Hz      | +23% vs stock                |
| Raw Gyroscope      | 402 Hz      | Unchanged — BNO086 limit     |
| ARVR Calibrated Gyro | 0 Hz     | Not supported by Adafruit lib |

## Patches Applied (local copy in `lib/Adafruit_BNO08x/`)

Two changes in `Adafruit_BNO08x.cpp`:

1. **SPI clock: 1 MHz -> 3 MHz** (line ~167)
   - BNO086 datasheet max is 3 MHz
   - Eliminates SPI as bottleneck for GRV reports

2. **INT wait: `delay(1)` loop -> `micros()` tight spin** (line ~549)
   - Stock code sleeps 1 ms per poll iteration in `spihal_wait_for_int()`
   - Replaced with busy-wait on `digitalRead()`, 500 ms timeout
   - Removes up to 2 ms dead time per SHTP read (header + payload each wait for INT)

## Conclusions

- **~400 Hz is the measured ceiling for these standard-channel reports on this
  UNO R4 test setup.** It is not a ceiling for the BNO086's separate
  Gyro-Integrated Rotation Vector channel; the production Teensy follow-up
  below measured 490 Hz from that channel.
- **SPI clock was the bottleneck for GRV** (326 -> 400 Hz), but not for raw gyro (already at the 400 Hz cap at 1 MHz).
- **`SH2_ARVR_STABILIZED_GRV` is an AR/VR-stabilized game rotation vector, not
  a calibrated-gyro report.** The historical test label was incorrect; its
  zero-event result says nothing about `SH2_GYROSCOPE_CALIBRATED` support.

## Production Teensy follow-up (2026-08-13)

The robot test polls SPI at its real 500 Hz control cadence and enables every
stream together. Removing the unused magnetometer Rotation Vector raised the
proven production rates to 191.0 Hz Game Rotation Vector, 191.0 Hz calibrated
gyro, and 12.3 Hz linear acceleration. It passed the 150 ms no-poll test, forced
hardware-reset recovery, and a 10-second soak without spontaneous recovery or
transport/decode/queue errors.

`SH2_GYRO_INTEGRATED_RV` was then tested as a possible lower-latency replacement
for the separate quaternion and gyro reports:

| Integrated request | Measured quaternion + gyro | Measured linear accel | Outcome |
|---:|---:|---:|---|
| 500 Hz | 490.0 Hz | 0 Hz | rejected; a repeated run also stalled/reset |
| 400 Hz | 399.3 Hz | 0.7 Hz | stable, but starves jump acceleration |
| 250 Hz | 400.0 Hz | 0.7 Hz | request was not honored as 250 Hz |
| 400 Hz, acceleration disabled | 399.7 Hz | 0 Hz | passed 60-second robustness test |

The dedicated report is fast: its host-transfer-to-driver age averaged
0.18-0.26 ms, versus 5.42 ms for production attitude and 3.42 ms for production
gyro. It was also noisier at rest and carries no per-report sequence field;
the Teensy driver now checks its dedicated SHTP channel sequence instead.
CEVA documents that enabling it can lengthen other fusion-output periods, and
the physical result made linear-acceleration landing unusable. With acceleration
explicitly disabled, however, it passed initialization, a 150 ms polling pause,
forced-reset recovery, and a 60-second soak with 25,188 paired reports and zero
sequence gaps or spontaneous recovery. On 2026-08-13 the robot intentionally
adopted gyro-only landing and promoted this 400 Hz integrated-only configuration
to production for the dynamic closed-loop trial. The 191 Hz three-stream mode
remains available as a legacy A/B test environment.
