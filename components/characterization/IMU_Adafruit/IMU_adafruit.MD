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

- **~400 Hz is the BNO086's hard ceiling** for any single report type via the Hillcrest SH2 report scheduler. Both GRV and raw gyro converge to this limit once SPI overhead is removed.
- **SPI clock was the bottleneck for GRV** (326 -> 400 Hz), but not for raw gyro (already at the 400 Hz cap at 1 MHz).
- **Cannot reach 500 Hz** with the Adafruit/SH2 report path. Exceeding 400 Hz would require bypassing the SH2 report scheduler or using a different IMU.
- **ARVR Calibrated Gyro** (SH2_ARVR_STABILIZED_GRV) returns zero events — the Adafruit library does not appear to support this report ID.
