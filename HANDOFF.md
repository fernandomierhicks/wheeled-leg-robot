# Hybrid IMU Sensor Model for master_sim

## Context

The BNO086 IMU provides two data paths with different latency characteristics:
- **Raw gyro**: ~1.05ms delay, ~1000Hz, but drifts when integrated
- **Game Rotation Vector (GRV)**: ~3.5ms delay, ~354Hz, drift-free fused pitch

Currently the simulation uses a single delay path (2ms) for all sensor readings. The real firmware can do better: use the fast gyro for low-latency pitch estimation, and periodically correct drift using the slower GRV. This reduces effective sensor delay from ~3.5ms to ~1ms — a significant win for the ZOH predictor.

## Approach: Complementary Filter with Dual Delay Buffers

### New file: `models/hybrid_imu.py`

`HybridIMUSensor` class with:
- **Gyro path** (every tick): `LatencyBuffer(1 tick)` → integrate pitch_rate for pitch angle, add simulated bias drift (random walk, default 0)
- **GRV path** (~every 3 ticks): `LatencyBuffer(~4 ticks)` → drift-free fused pitch, delivered at ~354Hz using a fractional accumulator (reproduces real 2ms/3ms alternating delivery)
- **Complementary filter**: On each GRV arrival: `pitch = alpha * pitch_gyro + (1-alpha) * pitch_grv` then reset integrator. Alpha ~0.98 (high-pass gyro, low-pass GRV).
- Single `update()` method returns `(pitch_fused, pitch_rate_delayed)` per tick

### New dataclass in `params.py`: `HybridIMUParams`

| Field | Default | Description |
|---|---|---|
| `enabled` | `False` | `False` = legacy single-delay mode (backward compat) |
| `gyro_delay_s` | `0.00105` | Raw gyro total delay |
| `grv_delay_s` | `0.0035` | GRV total delay |
| `grv_rate_hz` | `354.0` | GRV delivery rate |
| `alpha` | `0.98` | Complementary filter coeff (1.0=pure gyro) |
| `gyro_drift_rate` | `0.0` | Gyro bias random walk sigma [rad/s/sqrt(s)], opt-in |

Add `hybrid_imu: HybridIMUParams` field to `SimParams` (after `noise`).

### Changes to `sim_loop.py`

**`_init_controllers()`** (~line 284):
- If `hybrid_imu.enabled`: create `HybridIMUSensor`, set `n_sens` from `gyro_delay_s` (1 tick), set `sens_buf` to pass-through (0 steps) since hybrid sensor handles its own delays
- If not enabled: unchanged legacy path

**`tick()`** (~line 366):
- If `self.hybrid_sensor`: call `hybrid_sensor.update(pitch_true, pitch_rate_true, ...)` which returns fused pitch + delayed pitch_rate. Skip manual noise addition (hybrid sensor does it internally). Push result through pass-through `sens_buf` into predictor.
- If not: unchanged legacy path

Everything downstream (predictor, LQR, telemetry) stays the same — it just sees lower effective delay.

### Data flow

```
Legacy:  truth → +noise(GRV) → delay(2 ticks) → predictor(2 steps) → LQR

Hybrid:
         ┌─ gyro: +noise +drift → delay(1 tick) → integrate ──┐
truth ───┤                                                      ├─ comp.filter → predictor(1 step) → LQR
         └─ GRV:  +noise        → delay(4 ticks) → @354Hz ────┘
```

## Files to modify

1. **`simulation/mujoco/master_sim/params.py`** — Add `HybridIMUParams` dataclass + field on `SimParams`
2. **`simulation/mujoco/master_sim/models/hybrid_imu.py`** — New file: `HybridIMUSensor` class
3. **`simulation/mujoco/master_sim/sim_loop.py`** — Branch in `_init_controllers()` and `tick()`

Reuse: `models/latency.py` `LatencyBuffer` (no changes needed, just import).

## Verification

1. Run `python -m master_sim.sim_loop` with default params (hybrid disabled) — must match current behavior exactly
2. Enable hybrid (`HybridIMUParams(enabled=True)`) and run S01/S09 — should balance with similar or better performance
3. Test drift rejection: set `gyro_drift_rate=0.01`, verify complementary filter keeps pitch stable
4. Run `diag_delay.py` in both modes to compare prediction MAE

---

## HANDOFF.md Content (to be written to `\HANDOFF.md`)

# Handoff: Hybrid IMU Sensor Model

## What
Add a dual-path IMU sensor model to `simulation/mujoco/master_sim/` that mimics real BNO086 firmware behavior: fast raw gyro (~1ms delay, 1kHz) for low-latency pitch estimation, with periodic Game Rotation Vector (~3.5ms delay, 354Hz) corrections via complementary filter.

## Why
The BNO086 characterization (see `components/characterization/IMU/IMU_characterization_plan.MD`) measured:
- **Raw gyro total delay**: ~1.05ms (SPI 0.05ms + ISR 1.0ms)
- **GRV total delay**: ~3.5ms (fusion 2.5ms + ISR 1.0ms + SPI 0.05ms)
- **GRV rate**: ~354Hz (alternating 2ms/3ms delivery)
- **Gyro noise**: 0.121 deg/s RMS; **GRV pitch noise**: 0.0101 deg RMS
- **GRV drift**: negligible (0.2 deg/hr)

Using gyro for the fast path reduces effective sensor delay from ~3.5ms to ~1ms, improving ZOH predictor accuracy and LQR stability margin.

## Current state of simulation sensor model
- Single delay path: `ground_truth → +noise → LatencyBuffer(2ms) → ZOH predictor → LQR`
- Noise params from real BNO086 measurements (Test 5)
- Matrix ZOH predictor compensates for delay using discretized plant dynamics + torque history
- Code: `sim_loop.py` lines 284-313 (init), 366-401 (tick sensor section)
- Params: `params.py` `LatencyParams` (line 222), `NoiseParams` (line 236)

## Implementation plan
See `C:\Users\ferna\.claude\plans\mossy-jumping-reddy.md` for full details. Summary:

1. **`params.py`**: Add `HybridIMUParams` dataclass (`enabled`, `gyro_delay_s`, `grv_delay_s`, `grv_rate_hz`, `alpha`, `gyro_drift_rate`) + field on `SimParams`
2. **`models/hybrid_imu.py`** (new): `HybridIMUSensor` class with dual `LatencyBuffer`s, gyro integration, fractional GRV delivery accumulator, complementary filter
3. **`sim_loop.py`**: Branch in `_init_controllers()` and `tick()` — when hybrid enabled, sensor reads go through `HybridIMUSensor.update()` instead of the single-delay path; predictor uses gyro delay (1 tick) instead of sensor_delay_s

Backward compatible: `HybridIMUParams(enabled=False)` is default, legacy path unchanged.

## Key design decisions
- Complementary filter (not Kalman) — simpler, matches what UNO R4 WiFi can realistically run
- Alpha=0.98: high-pass gyro + low-pass GRV, -3dB crossover ~1.1Hz (well below balance bandwidth)
- Gyro drift modeled as Wiener process (random walk bias), default 0 (opt-in)
- GRV delivery uses fractional accumulator to reproduce real 2ms/3ms alternating pattern

## Verification
1. Default params (hybrid off) → identical to current behavior
2. Hybrid on, drift=0 → equal or better balance performance
3. Hybrid on, drift>0 → complementary filter rejects drift
4. `diag_delay.py` comparison of prediction MAE in both modes
