# BNO086 IMU Characterization Framework

## Context

The simulation models BNO086 sensor delay (5ms fusion), noise (pitch 0.1deg, pitch_rate 0.5deg/s), and SPI read latency -- but these are datasheet estimates, not measured. Before building firmware, we need real numbers from the actual BNO086 + UNO R4 WiFi hardware. This framework lives in `components/characterization/IMU/`.

## File Structure

```
components/characterization/IMU/
├── platformio.ini
├── src/
│   └── main.cpp                    # Menu dispatcher (serial char selects test)
├── include/
│   ├── bno086_common.h             # Shared SPI init, BNO086 setup helpers
│   ├── test_datarate_grv.h         # Test 1: Game Rotation Vector max rate
│   ├── test_datarate_raw.h         # Test 2: Raw gyro/accel max rate
│   ├── test_static_drift.h         # Test 3: GRV drift while stationary
│   ├── test_spi_latency.h          # Test 4: Per-transaction SPI timing
│   ├── test_noise.h                # Test 5: Gyro/accel noise at rest
│   └── test_e2e_latency.h          # Test 6: End-to-end data-ready-to-read timing
├── analysis/
│   ├── parse_log.py                # Common CSV parser (strips non-CSV lines)
│   ├── analyze_datarate.py         # Hz, jitter, histogram (Tests 1 & 2)
│   ├── analyze_drift.py            # Quaternion→Euler, drift rate deg/hr (Test 3)
│   ├── analyze_noise.py            # Std-dev, Allan variance, PSD (Test 5)
│   ├── analyze_latency.py          # SPI + e2e latency histograms (Tests 4 & 6)
│   └── generate_sim_params.py      # Map results → params.py NoiseParams/LatencyParams
└── logs/
    └── .gitkeep                    # Serial captures go here (gitignored)
```

---

## SparkFun BNO08x Library API Reference

Library: `sparkfun/SparkFun BNO08x Cortex Based IMU @ ^1.0.3`

### SPI Initialization

```cpp
// BNO08x imu;
boolean beginSPI(uint8_t user_CSPin, uint8_t user_INTPin, uint8_t user_RSTPin,
                 uint32_t spiPortSpeed = 1000000, SPIClass &spiPort = SPI);
```

- SPI mode: CPOL=1, CPHA=1 → **SPI_MODE3**
- Max SPI clock: **3 MHz** (pass `3000000`)
- Returns `true` if BNO086 responds after reset
- BNO086 takes ~90ms after RST before it's ready (library handles internally)
- INT pin is active-low (H_INTN): asserted when data is ready

### Enable Sensor Reports

```cpp
bool enableGameRotationVector(uint16_t timeBetweenReports = 10);  // milliseconds
bool enableRawGyro(uint16_t timeBetweenReports = 10);             // milliseconds
bool enableRawAccelerometer(uint16_t timeBetweenReports = 10);    // milliseconds
```

- **Parameter is in MILLISECONDS** (library converts to microseconds internally for SHTP)
- `enableGameRotationVector(1)` → requests 1ms interval → 1000Hz → BNO086 clamps to its internal max (~400Hz)
- `enableRawGyro(1)` → raw gyro can run up to ~1000Hz per datasheet
- `enableRawAccelerometer(1)` → raw accel can run up to ~500Hz per datasheet

### Data Polling

```cpp
bool dataAvailable(void);  // returns true if new report(s) arrived via SHTP
```

- Over SPI: reads the SHTP header to check. Costs one SPI transaction even if no data.
- When true, the latest report values are cached internally — read with getters below.

### Game Rotation Vector Getters

```cpp
float getGameQuatI();       // quaternion i (imaginary X), unitless, normalized
float getGameQuatJ();       // quaternion j (imaginary Y)
float getGameQuatK();       // quaternion k (imaginary Z)
float getGameQuatReal();    // quaternion w (scalar)
```

### Raw Sensor Getters

```cpp
int16_t getRawGyroX();      // raw ADC LSBs (NOT deg/s)
int16_t getRawGyroY();
int16_t getRawGyroZ();
int16_t getRawAccelX();     // raw ADC LSBs (NOT m/s²)
int16_t getRawAccelY();
int16_t getRawAccelZ();
```

**Unit conversion (done in Python analysis, NOT on Arduino):**
- Gyro: `deg_s = raw * (1.0 / 16.0)` → then `rad_s = deg_s * π/180`
  - BNO086 gyro LSB = 1/16 deg/s (per SH-2 reference manual, Table 6-22)
- Accel: `m_s2 = raw * (1.0 / 8192.0) * 9.80665`
  - BNO086 accel LSB = 1/8192 g (per SH-2 reference manual, Table 6-20)

### Accuracy & Timestamp

```cpp
uint8_t getQuatAccuracy();  // 0=Unreliable, 1=Low, 2=Medium, 3=High
uint64_t getTimeStamp();    // BNO086 internal clock, units = 100µs ticks
```

- Accuracy is a **0–3 enum**, NOT a float in radians
- Timestamp can be compared across samples to measure BNO086's internal timing

### Datasheet Latency Reference

| GRV Report Rate | Typical Fusion Latency |
|-----------------|----------------------|
| 100 Hz          | 6.6 ms               |
| 200 Hz          | 3.7 ms               |
| 400 Hz          | ~2.5 ms (extrapolated) |

---

## Serial Protocol Convention

All tests use a **marker-delimited CSV protocol** so `parse_log.py` can reliably extract data:

```
=== IMU CHARACTERIZATION ===
Select test: 1-6 (see menu)

> User sends '4'

[TEST4] SPI Transaction Latency
[TEST4] Duration: 5s
[TEST4] ---CSV_START---
idx,t_start_us,t_end_us,duration_us,had_data
0,102345,102410,65,1
1,102480,102495,15,0
...
[TEST4] ---CSV_END---
[TEST4] Samples: 8523
[TEST4] Done.
```

Rules:
- `---CSV_START---` and `---CSV_END---` markers delimit parseable data
- Each marker line is prefixed with `[TESTn]` for the test number
- CSV header row immediately follows `---CSV_START---`
- Status/info lines outside markers are prefixed with `[TESTn]`
- `parse_log.py` strips everything outside markers, returns a pandas DataFrame

### Error Output Convention

If BNO086 fails to initialize:

```
[ERROR] BNO086 not detected on SPI. Check wiring:
[ERROR]   CS=D10, INT=D2, RST=D3, SCK=D13, MOSI=D11, MISO=D12
[ERROR]   Solder jumpers PS0+PS1 must be bridged for SPI mode
[ERROR] Retrying in 3s...
```

`bno086_common.h` retries initialization 3 times with 3s delay. If all fail, prints `[FATAL] BNO086 init failed after 3 attempts` and halts (infinite loop).

---

## Tests — Detailed Implementation

### Test 1 — GRV Max Datarate (10s)

```cpp
void run_test1_grv_datarate() {
    imu.enableGameRotationVector(1);  // 1ms = request max rate

    Serial.println("[TEST1] ---CSV_START---");
    Serial.println("idx,timestamp_us,qi,qj,qk,qr");

    uint32_t start = millis();
    uint32_t idx = 0;
    while (millis() - start < 10000) {
        if (imu.dataAvailable()) {
            Serial.print(idx++);
            Serial.print(','); Serial.print(micros());
            Serial.print(','); Serial.print(imu.getGameQuatI(), 6);
            Serial.print(','); Serial.print(imu.getGameQuatJ(), 6);
            Serial.print(','); Serial.print(imu.getGameQuatK(), 6);
            Serial.print(','); Serial.println(imu.getGameQuatReal(), 6);
        }
    }

    Serial.println("[TEST1] ---CSV_END---");
    Serial.print("[TEST1] Samples: "); Serial.println(idx);
}
```

**Expected result:** ~4000 samples (400Hz × 10s). If significantly less, SPI/SHTP polling overhead is the bottleneck.

### Test 2 — Raw Gyro/Accel Max Datarate (10s)

```cpp
void run_test2_raw_datarate() {
    imu.enableRawGyro(1);              // 1ms = request max rate
    imu.enableRawAccelerometer(1);     // 1ms = request max rate

    Serial.println("[TEST2] ---CSV_START---");
    Serial.println("idx,timestamp_us,type,x,y,z");

    uint32_t start = millis();
    uint32_t idx = 0;
    while (millis() - start < 10000) {
        if (imu.dataAvailable()) {
            uint32_t now = micros();
            // Library caches both reports; check which arrived
            // Print raw int16_t LSBs — conversion done in Python
            int16_t gx = imu.getRawGyroX();
            int16_t gy = imu.getRawGyroY();
            int16_t gz = imu.getRawGyroZ();
            // Always print gyro line (will be 0,0,0 if no new gyro)
            Serial.print(idx++);
            Serial.print(','); Serial.print(now);
            Serial.print(",G,");
            Serial.print(gx); Serial.print(',');
            Serial.print(gy); Serial.print(',');
            Serial.println(gz);
        }
    }

    Serial.println("[TEST2] ---CSV_END---");
}
```

**Note:** The SparkFun library doesn't expose per-report "hasNew" flags cleanly. In practice, `dataAvailable()` returns true when ANY enabled report arrives. For accurate per-report rate measurement, enable only one report type at a time and run two separate sub-tests (gyro-only, accel-only). The test function should run both sequentially with clear markers.

### Test 3 — Static GRV Drift (5 min)

```cpp
#define DRIFT_DURATION_MS 300000  // 5 minutes

void run_test3_static_drift() {
    imu.enableGameRotationVector(3);  // 3ms ≈ 333Hz (comfortably under 400Hz max)

    Serial.println("[TEST3] ---CSV_START---");
    Serial.println("idx,timestamp_us,qi,qj,qk,qr,accuracy");

    uint32_t start = millis();
    uint32_t idx = 0;
    while (millis() - start < DRIFT_DURATION_MS) {
        if (imu.dataAvailable()) {
            Serial.print(idx++);
            Serial.print(','); Serial.print(micros());
            Serial.print(','); Serial.print(imu.getGameQuatI(), 6);
            Serial.print(','); Serial.print(imu.getGameQuatJ(), 6);
            Serial.print(','); Serial.print(imu.getGameQuatK(), 6);
            Serial.print(','); Serial.print(imu.getGameQuatReal(), 6);
            Serial.print(','); Serial.println(imu.getQuatAccuracy());
        }
    }

    Serial.println("[TEST3] ---CSV_END---");
}
```

**Analysis notes:**
- Convert quaternion to Euler: `pitch = asin(2(qw*qj - qk*qi))`, `roll = atan2(2(qw*qi + qj*qk), 1 - 2(qi² + qj²))`
- Trim first 60s (BNO086 internal calibration convergence)
- Linear regression on pitch/roll vs time → slope = drift rate (deg/hr)
- Yaw WILL drift (game vector has no magnetometer correction) — report but don't alarm

### Test 4 — SPI Transaction Latency (5s)

```cpp
void run_test4_spi_latency() {
    imu.enableGameRotationVector(10);  // 100Hz — ensure data available frequently

    Serial.println("[TEST4] ---CSV_START---");
    Serial.println("idx,t_start_us,t_end_us,duration_us,had_data");

    uint32_t start = millis();
    uint32_t idx = 0;
    while (millis() - start < 5000) {
        uint32_t t0 = micros();
        bool had = imu.dataAvailable();
        if (had) {
            // Read all cached values to measure full transaction cost
            volatile float qi = imu.getGameQuatI();
            volatile float qj = imu.getGameQuatJ();
            volatile float qk = imu.getGameQuatK();
            volatile float qr = imu.getGameQuatReal();
            (void)qi; (void)qj; (void)qk; (void)qr;
        }
        uint32_t t1 = micros();

        Serial.print(idx++);
        Serial.print(','); Serial.print(t0);
        Serial.print(','); Serial.print(t1);
        Serial.print(','); Serial.print(t1 - t0);
        Serial.print(','); Serial.println(had ? 1 : 0);
    }

    Serial.println("[TEST4] ---CSV_END---");
}
```

**Analysis splits into two populations:**
- `had_data=0`: cost of a "no data" SPI poll (SHTP header read only)
- `had_data=1`: cost of a full data read (SHTP header + report payload)
- Report mean, median, p95, p99 for each population

### Test 5 — Noise Density (Phase A: GRV 30s + Phase B: Raw 60s ≈ 95s total)

Two-phase test. Phase A measures fused GRV pitch/roll noise at rest (→ `pitch_std_rad`, `roll_std_rad`). Phase B measures raw gyro/accel noise for Allan variance and PSD (→ `pitch_rate_std_rad_s`, `accel_std`).

**Phase A** — GRV at 333Hz for 30s (3s warmup discarded). Streams quaternion + accuracy via `[TEST5A]` CSV block.

**Phase B** — Raw gyro + accel at 333Hz for 60s (1s warmup discarded). Streams raw int16 LSBs via `[TEST5B]` CSV block.

See `include/test_noise.h` for implementation. Uses INT-pin gating + `getSensorEvent()` (same fast-path as Test 1) and `snprintf` serial output.

**Raw → SI conversion (in Python):**
- Gyro: `deg_s = raw_lsb / 16.0` → `rad_s = deg_s * π/180`
- Accel: `g = raw_lsb / 8192.0` → `m_s2 = g * 9.80665`

**Allan variance algorithm (in `analyze_noise.py`):**

```python
def allan_deviation(data, dt):
    """Overlapping Allan deviation."""
    N = len(data)
    # Cluster sizes: powers of 2 from 1 to N//4
    taus = []
    adevs = []
    for m in [2**k for k in range(0, int(np.log2(N//2)))]:
        tau = m * dt
        # Compute cluster averages
        clusters = data[:N - N % m].reshape(-1, m).mean(axis=1)
        # Allan variance = 0.5 * mean of squared differences
        adev = np.sqrt(0.5 * np.mean(np.diff(clusters)**2))
        taus.append(tau)
        adevs.append(adev)
    return np.array(taus), np.array(adevs)
```

On log-log plot:
- Slope = -0.5 region → **angle random walk** (white noise), read value at τ=1s
- Slope = 0 region → **bias instability** (flicker noise), read minimum value
- Units: gyro ARW in deg/s/√Hz, accel VRW in m/s²/√Hz

### Test 6 — End-to-End Latency (5s)

```cpp
volatile bool int_fired = false;
volatile uint32_t t_int = 0;

void bno_isr() {
    t_int = micros();
    int_fired = true;
}

void run_test6_e2e_latency() {
    attachInterrupt(digitalPinToInterrupt(BNO_INT_PIN), bno_isr, FALLING);
    imu.enableGameRotationVector(10);  // 100Hz

    Serial.println("[TEST6] ---CSV_START---");
    Serial.println("idx,t_int_us,t_read_start_us,t_read_done_us,isr_to_read_us,read_us,bno_ts");

    uint32_t start = millis();
    uint32_t idx = 0;
    while (millis() - start < 5000) {
        if (int_fired) {
            int_fired = false;
            uint32_t t_isr = t_int;  // copy volatile

            uint32_t t_rs = micros();
            imu.dataAvailable();
            float qi = imu.getGameQuatI();  // force read
            uint64_t bno_ts = imu.getTimeStamp();
            uint32_t t_rd = micros();

            Serial.print(idx++);
            Serial.print(','); Serial.print(t_isr);
            Serial.print(','); Serial.print(t_rs);
            Serial.print(','); Serial.print(t_rd);
            Serial.print(','); Serial.print(t_rs - t_isr);
            Serial.print(','); Serial.print(t_rd - t_rs);
            Serial.print(','); Serial.println((uint32_t)(bno_ts & 0xFFFFFFFF));
        }
    }

    detachInterrupt(digitalPinToInterrupt(BNO_INT_PIN));
    Serial.println("[TEST6] ---CSV_END---");
}
```

**What each column measures:**
- `t_int_us`: Arduino `micros()` when INT pin fires (data ready from BNO086)
- `t_read_start_us`: when we call `dataAvailable()` (may have ISR-to-mainloop delay)
- `t_read_done_us`: after SPI read completes
- `isr_to_read_us`: software latency (ISR → main loop response)
- `read_us`: pure SPI transaction time
- `bno_ts`: BNO086 internal timestamp (100µs ticks) — compare consecutive values to verify BNO086's own report rate

**Fusion latency estimation:** The BNO086's internal fusion delay cannot be directly measured from outside. However, by comparing `bno_ts` deltas against Arduino timestamp deltas, we can detect if the BNO086 is batching or delaying reports. The datasheet values (6.6ms @ 100Hz, 3.7ms @ 200Hz) are our best reference.

---

## Wiring (SPI mode — max throughput)

```
BNO086 Breakout Pin        Arduino UNO R4 WiFi
  3.3V  ───────────────→   3.3V
  GND   ───────────────→   GND
  SCL/SCK/RX  ─────────→   D13 (SCK)
  SDA/MISO/TX ─────────→   D12 (MISO / CIPO)
  ADDR/MOSI   ─────────→   D11 (MOSI / COPI)
  CS    ───────────────→   D10 (chip select)
  INT   ───────────────→   D2  (data-ready interrupt, used in Test 6)
  RST   ───────────────→   D3  (reset control)
```

**Protocol select (solder jumpers on PCB back):**
- **PS0**: solder bridge **closed** (HIGH)
- **PS1**: solder bridge **closed** (HIGH)
- PS1=HIGH + PS0=HIGH = **SPI mode**

| PS1 | PS0 | Protocol |
|-----|-----|----------|
| LOW | LOW | I2C |
| LOW | HIGH | UART-RVC |
| HIGH | LOW | UART |
| HIGH | HIGH | **SPI** (what we want) |

**Why SPI over I2C:** BNO086 SPI runs up to 3MHz. A 20-byte read takes ~50us (SPI) vs ~400us (I2C @ 400kHz). For a 500Hz control loop (2ms period), SPI leaves 97.5% of CPU time free vs 80% with I2C. The 5ms fusion delay dominates latency either way, but SPI maximizes available compute for the control loop.

RA4M1 is 3.3V native — no level shifter needed.

---

## bno086_common.h — Shared Init

```cpp
#pragma once
#include <SPI.h>
#include <SparkFun_BNO08x_Arduino_Library.h>

#ifndef BNO_CS_PIN
#define BNO_CS_PIN  10
#endif
#ifndef BNO_INT_PIN
#define BNO_INT_PIN 2
#endif
#ifndef BNO_RST_PIN
#define BNO_RST_PIN 3
#endif
#ifndef BNO_SPI_SPEED
#define BNO_SPI_SPEED 3000000  // 3MHz max
#endif

BNO08x imu;

bool init_bno086() {
    for (int attempt = 1; attempt <= 3; attempt++) {
        if (imu.beginSPI(BNO_CS_PIN, BNO_INT_PIN, BNO_RST_PIN, BNO_SPI_SPEED, SPI)) {
            Serial.print("[INFO] BNO086 init OK on attempt ");
            Serial.println(attempt);
            return true;
        }
        Serial.print("[ERROR] BNO086 not detected (attempt ");
        Serial.print(attempt); Serial.println("/3)");
        Serial.println("[ERROR]   CS=D10, INT=D2, RST=D3, SCK=D13, MOSI=D11, MISO=D12");
        Serial.println("[ERROR]   Solder jumpers PS0+PS1 must be bridged for SPI mode");
        delay(3000);
    }
    Serial.println("[FATAL] BNO086 init failed after 3 attempts");
    return false;
}

void print_menu() {
    Serial.println();
    Serial.println("=== IMU CHARACTERIZATION ===");
    Serial.println("1 - GRV max datarate (10s)");
    Serial.println("2 - Raw gyro/accel max datarate (10s)");
    Serial.println("3 - Static GRV drift (5 min)");
    Serial.println("4 - SPI transaction latency (5s)");
    Serial.println("5 - Noise density (60s)");
    Serial.println("6 - End-to-end latency (5s)");
    Serial.println("Select test [1-6]: ");
}
```

---

## Analysis Scripts — Detailed Specs

All scripts use: `pandas`, `numpy`, `matplotlib`, `scipy` (for PSD). No exotic dependencies.

### parse_log.py

```python
"""Common CSV parser for all serial captures."""
import pandas as pd
import sys, re

def parse_csv(filepath: str, test_num: int = None) -> pd.DataFrame:
    """Extract CSV data between ---CSV_START--- and ---CSV_END--- markers.

    If test_num is given, only extract from [TESTn] markers.
    Returns pandas DataFrame with columns from the CSV header.
    """
    lines = open(filepath, 'r').readlines()
    in_csv = False
    csv_lines = []
    for line in lines:
        stripped = line.strip()
        if '---CSV_START---' in stripped:
            in_csv = True
            continue
        if '---CSV_END---' in stripped:
            in_csv = False
            continue
        if in_csv and stripped:
            # Strip [TESTn] prefix if present
            cleaned = re.sub(r'^\[TEST\d\]\s*', '', stripped)
            csv_lines.append(cleaned)

    if not csv_lines:
        raise ValueError(f"No CSV data found in {filepath}")

    from io import StringIO
    return pd.read_csv(StringIO('\n'.join(csv_lines)))
```

### generate_sim_params.py — Measurement-to-Param Mapping

| Measurement | Source Test | Sim Parameter | Current Default |
|-------------|-----------|---------------|-----------------|
| GRV actual rate | Test 1 | (informational — affects control loop design) | 400 Hz assumed |
| SPI poll cost (no data) | Test 4 | (informational — CPU budget) | — |
| SPI read cost (with data) | Test 4 | `LatencyParams.sensor_delay_s` (partial) | 0.001 s |
| ISR-to-read delay | Test 6 | `LatencyParams.sensor_delay_s` (partial) | 0.001 s |
| Fusion delay estimate | Test 6 + datasheet | `LatencyParams.sensor_delay_s` (dominant) | 0.005 s realistic |
| GRV pitch noise | Test 5A | `NoiseParams.pitch_std_rad` | ~~0.1 deg~~ → **0.0101 deg** (0.000176 rad) |
| GRV roll noise | Test 5A | `NoiseParams.roll_std_rad` | ~~0.05 deg~~ → **0.0090 deg** (0.000156 rad) |
| Gyro noise std | Test 5B | `NoiseParams.pitch_rate_std_rad_s` | ~~0.5 deg/s~~ → **0.121 deg/s** (0.00212 rad/s) |
| Accel noise std | Test 5C | `NoiseParams.accel_std` | ~~0.2 m/s²~~ → **0.008 m/s²** |
| Pitch drift rate | Test 3 | `NoiseParams.pitch_std_rad` (if drift > noise) | — (not yet measured) |
| Roll drift rate | Test 3 | `NoiseParams.roll_std_rad` | — (not yet measured) |

**Output format:**
```python
# --- Generated by generate_sim_params.py ---
# Source: components/characterization/IMU/logs/<logfiles>
# Date: 2026-XX-XX

# Recommended updates for simulation/mujoco/master_sim/params.py:

class LatencyParams:
    sensor_delay_s: float = 0.00XX   # measured: SPI_read(XXus) + fusion(~X.Xms)
    actuator_delay_s: float = 0.001  # unchanged (not measured here)

class NoiseParams:  # ✅ MEASURED 2026-03-22
    pitch_std_rad: float = 0.000176        # 0.0101 deg — GRV fused (was 0.1 deg)
    pitch_rate_std_rad_s: float = 0.002116 # 0.121 deg/s — raw gyro Y (was 0.5 deg/s)
    accel_std: float = 0.008               # m/s² RMS (was 0.2 m/s²)
    roll_std_rad: float = 0.000156         # 0.0090 deg — GRV fused (was 0.05 deg)
```

---

## platformio.ini

```ini
[env:uno_r4_wifi]
platform = renesas-ra
board = uno_r4wifi
framework = arduino
monitor_speed = 921600
build_flags =
    -DSERIAL_BAUD=921600
    -DBNO_CS_PIN=10
    -DBNO_INT_PIN=2
    -DBNO_RST_PIN=3
    -DBNO_SPI_SPEED=3000000
lib_deps =
    sparkfun/SparkFun BNO08x Cortex Based IMU @ ^1.0.3
```

---

## Implementation Order

1. [x] `bno086_common.h` + `main.cpp` menu scaffold + `platformio.ini` (SPI @ 3MHz)
2. [x] Test 4 (SPI latency) — simplest, verifies BNO086 is alive over SPI
3. [x] Test 1 (GRV datarate) — **354 Hz measured** (see findings below)
4. [x] Test 6 (e2e latency) — **~3.5ms intrinsic delay** (see findings below)
5. [x] Test 5 (noise density) — GRV fused noise (30s) + raw gyro/accel (60s), includes `parse_log.py` + `analyze_noise.py`
6. [x] Test 3 (static drift) — **<0.3 deg/hr pitch & roll** (see findings below)
7. [ ] Test 2 (raw datarate) — supplementary
8. [ ] Remaining analysis scripts (`analyze_datarate.py`, `analyze_latency.py`)
9. [ ] `generate_sim_params.py`

## Test 1 Findings — GRV Max Datarate (2026-03-22)

### Summary

| Metric | Value |
|--------|-------|
| **Effective GRV rate** | **354 Hz** (3541 events / 10s) |
| Previous measurement (serial-limited) | 262 Hz |
| Datasheet theoretical max | ~400 Hz |

### Optimization journey

The initial 262 Hz measurement was **bottlenecked by serial output** — four `Serial.print(float, 6)` calls per sample blocked the hot loop when the UART TX buffer filled at 921600 baud.

Optimizations applied (in `test_datarate_grv.h`):
1. **RAM-buffered collection** — timestamps stored in a 2500-entry `uint32_t` array during the 10s test, dumped to serial afterward. Completely removes serial from the hot path.
2. **INT pin gating** — direct port register read (`PORT1_PIDR bit 4`, ~0.1µs) skips the SPI poll when BNO086 has no data ready. Replaces `digitalRead()` (~3µs) and avoids wasted `getSensorEvent()` SPI transactions (~50µs each).
3. **WiFi suspend tested and ruled out** — disconnecting WiFi during collection had no measurable effect (353.9 Hz vs 353.7 Hz). ESP32 coprocessor interrupts are not a factor.

### Inter-sample timing distribution

| Delta (µs) | Count | Percentage | Interpretation |
|-------------|-------|------------|----------------|
| 2000 | 985 | 39.4% | BNO086 fusion delivers at 500 Hz pace |
| 3000 | 1107 | 44.3% | Fusion delivers at 333 Hz pace |
| 4000 | 385 | 15.4% | Occasional longer cycle |
| 5000+ | 20 | 0.8% | Rare outliers |

The BNO086's Hillcrest fusion core alternates between 2ms and 3ms delivery intervals. The resulting ~354 Hz is an intrinsic sensor property, not an MCU-side limitation. All MCU-side bottlenecks (serial blocking, GPIO read latency, WiFi interrupts) have been eliminated.

### Implications for firmware control loop

- At a 500 Hz control loop, ~71% of cycles will receive a fresh GRV sample
- On cycles without new data, the LQR reuses the previous reading — acceptable for a 2ms-old sample
- `sensor_delay_s` in simulation should account for: SPI read (~50µs) + fusion latency (~2.5ms @ 400 Hz extrapolated) ≈ **2.5–3.0ms total**

### Test modes available

- **`1` (fast mode)** — timestamps only, buffered in RAM, measures true BNO086 rate without serial interference
- **`F` (full mode)** — quaternion values printed live via batched `snprintf` + `dtostrf` (4 decimal places), rate limited by serial throughput

## Test 6 Findings — End-to-End Latency (2026-03-22)

### Summary

| Metric | Value |
|--------|-------|
| **ISR-to-read delay** | **987 µs median** (std 18 µs) |
| **ISR-to-read range** | 921–991 µs (very tight) |
| **Report delivery rate** | **100.5 Hz** (requested 100 Hz) |
| **`getSensorEvent()` duration** | **~10 ms** (blocks for full report period) |
| **Samples collected** | 506 (5s @ 100 Hz) |

### Critical finding: `getSensorEvent()` blocks for full report period

When called after an ISR-driven INT pin assertion, the SparkFun library's `getSensorEvent()` blocks for ~10ms — the entire report period at 100 Hz. This is **not** the SPI transfer time (Test 4 measured true SPI read at ~50–65 µs). The library internally waits for the next SHTP cargo.

Evidence from settling behavior (first 5 samples):

| Sample | dt_int (µs) | isr_to_read (µs) | read_us | Interpretation |
|--------|-------------|-------------------|---------|----------------|
| 0 | — | 184 | 755 | First read, data already buffered |
| 1 | 377 | 565 | 2,052 | Library starting to block |
| 2 | 1,697 | 923 | 2,373 | Transition |
| 3 | 1,946 | 1,353 | 4,631 | Growing toward full period |
| 4 | 5,063 | 924 | 9,997 | Locked to 10ms blocking |
| 5+ | ~10,000 | ~987 | ~9,997 | Steady state |

### Read time distribution (steady state, samples 5–505)

| Read time (µs) | Count | Percentage |
|-----------------|-------|------------|
| ~9,000 | 68 | 13.6% |
| ~10,000 | 391 | 78.0% |
| ~11,000 | 41 | 8.2% |

### BNO086 internal timestamps

The `getTimeStamp()` values were near `2^32` (e.g. 4294965196) with near-zero deltas — likely an initialization/wrapping issue. These timestamps are **not usable** for measuring BNO086 internal timing. The datasheet fusion latency values (6.6ms @ 100 Hz, 3.7ms @ 200 Hz, ~2.5ms @ 400 Hz) remain our best reference.

### Intrinsic sensor delay budget (for `sensor_delay_s`)

| Component | Value | Source |
|-----------|-------|--------|
| BNO086 fusion latency | ~2.5 ms | Datasheet (extrapolated @ 400 Hz) |
| INT assertion to main loop | ~1.0 ms | Measured (ISR-to-read) |
| SPI read (true) | ~0.05 ms | Test 4 |
| **Total intrinsic** | **~3.5 ms** | |

### Firmware architecture implication

**The ISR + `getSensorEvent()` pattern is unsuitable for a 500 Hz control loop.** The 10ms blocking would consume 5× the 2ms loop budget. The firmware must use **polling** with `int_asserted()` + `getSensorEvent()` (as proven in Test 1/Test 4), where:
- `int_asserted()` checks the INT pin via direct port register (~0.1 µs)
- `getSensorEvent()` returns in ~50–65 µs when data is already waiting
- No-data polls cost ~15 µs (SHTP header check only)

This polling approach keeps sensor reading well under 0.1 ms per loop iteration.

## Test 3 Findings — Static GRV Drift (2026-03-22)

### Summary

| Metric | Value |
|--------|-------|
| **Pitch drift** | **-0.210 deg/hr** (negligible) |
| **Roll drift** | **-0.273 deg/hr** (negligible) |
| **Yaw drift** | **+0.063 deg/hr** (expected — no magnetometer) |
| **Effective rate** | 333.3 Hz (3ms intervals) |
| **Samples collected** | 105,553 (5 min) |
| **Calibration accuracy** | 100% at level 3 (High) |

### Drift analysis

After trimming the first 60s (calibration convergence), linear regression on the remaining 4 minutes gives drift rates well under 1 deg/hr for all axes. For a balancing robot where the control loop runs at 500 Hz (2ms period), a drift of 0.2 deg/hr means the pitch error grows by only 0.00006 deg per second — completely negligible compared to the 0.014 deg noise floor.

### Residual noise (after detrending)

| Axis | Std (deg) | Std (rad) |
|------|-----------|-----------|
| Pitch | 0.0140 | 0.000245 |
| Roll | 0.0117 | 0.000205 |
| Yaw | 0.0097 | 0.000168 |

These residual noise values are slightly higher than the Test 5 GRV noise (0.0101 deg pitch, 0.0090 deg roll) — expected since the 5-minute collection includes more low-frequency wander. The Test 5 values remain the better estimate for `NoiseParams` since they represent the noise the control loop actually sees on a per-sample basis.

### Implications for simulation

- **No bias correction needed** — drift is negligible for balance control
- **`pitch_std_rad` and `roll_std_rad`** should use Test 5 values (short-term noise), not Test 3 residuals
- **Yaw drift** is remarkably low for a game rotation vector (no magnetometer). Not relevant for the balance controller, which doesn't use yaw from GRV.

---

## Verification

- Flash each test, capture serial to CSV, run corresponding analysis script
- Compare measured values against current sim defaults:
  - `sensor_delay_s`: currently 0.001 (0.005 recommended) → **~3.5ms measured intrinsic** (Tests 1, 4, 6)
  - `pitch_std_rad`: 0.1 deg → **0.0101 deg measured** (Test 5) — drift negligible at 0.21 deg/hr (Test 3)
  - `pitch_rate_std_rad_s`: 0.5 deg/s → **0.121 deg/s measured** (Test 5)
  - `accel_std`: 0.2 m/s² → **0.008 m/s² measured** (Test 5)
- Run `generate_sim_params.py` to produce updated params
