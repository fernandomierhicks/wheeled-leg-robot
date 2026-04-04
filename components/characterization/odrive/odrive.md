# ODrive 3.6 — Hardware & Firmware Reference

> Last probed: 2026-04-03 via `probe_firmware.py` (full recursive tree, depth 8).
> Raw results: `probe_results.txt`
>
> **Firmware updated 2026-04-03 to ODrive v0.5.6 (official release build).**
> Everything below this line was captured on the old dev snapshot (0.0.0 unreleased).
> Re-probe after any work begins on the 0.5.6 firmware to update all values.

---

## Hardware

| Attribute | Value |
|---|---|
| Board | ODrive hw v3.6 variant 56 |
| Serial number | 61985812394293 |
| vbus_voltage (at probe time) | 23.93 V |
| user_config_loaded | True |
| brake_resistor_armed | True |
| brake_resistor_saturated | False |

---

## Firmware

`fw_version_major/minor/revision` all report `0.0.0` — **cannot be trusted**.
`fw_version_unreleased = 1` confirms this is an official dev/snapshot build, not a release.

### Feature fingerprinting

| Feature | Present | Introduced |
|---|---|---|
| `get_adc_voltage()` | YES | 0.5.6 |
| `can` object | YES | ~0.5.4 |
| `anticogging` sub-object | YES | ~0.5.4 |
| `startup_homing` | YES | ~0.5.4 |
| `enable_phase_interpolation` | YES | ~0.5.4 |
| `move_incremental()` | YES | ~0.5.2 |
| `dc_bus_overvoltage/undervoltage_trip_level` | YES | ~0.5.3 |
| `inertia`, `input_filter_bandwidth` | YES | ~0.5.1 |
| `current_control_bandwidth` | YES | ~0.5.1 |
| `general_lockin`, `sensorless_estimator` | YES | ~0.5.0 |
| `trap_traj.config` | YES | ~0.5.0 |
| `use_index_offset` | **NO** | 0.5.5 |
| `gpio_modes` (new-style) | **NO** | 0.5.5 |
| `enable_brake_resistor` | **NO** | standard 0.5.x |
| `acim_estimator` | **NO** | stripped |
| `move_to_pos()` (old-style) | **NO** | pre-0.5.2 only |

### Conclusion

**Custom dev snapshot between 0.5.4 and 0.5.6.**
Has `get_adc_voltage` (cherry-picked from 0.5.6) but lacks 0.5.5 attrs.
Do not assume full compatibility with any released ODrive version.
Always probe attributes at runtime with `try/except` or `hasattr`.

---

## odrv0 — Top-level Config

```
odrv0.config.brake_resistance                 = 2.0 Ω
odrv0.config.dc_bus_overvoltage_trip_level    = 59.92 V
odrv0.config.dc_bus_overvoltage_ramp_start    = 59.92 V
odrv0.config.dc_bus_overvoltage_ramp_end      = 59.92 V
odrv0.config.enable_dc_bus_overvoltage_ramp   = False
odrv0.config.dc_bus_undervoltage_trip_level   = 8.0 V
odrv0.config.dc_max_positive_current          = inf
odrv0.config.dc_max_negative_current          = ~0 A  (regen effectively disabled)
odrv0.config.max_regen_current                = 0.0 A
odrv0.config.enable_uart                      = True
odrv0.config.uart_baudrate                    = 115200
odrv0.config.enable_ascii_protocol_on_usb     = True
odrv0.config.enable_i2c_instead_of_can        = False
```

Note: `enable_brake_resistor` does not exist on this firmware.
Brake resistor is controlled by `brake_resistance > 0` (currently 2.0 Ω, armed).

---

## GPIO / ADC

ADC-capable pins: **GPIO 3 and GPIO 4** (no mode config needed — analog is implicit).

```
odrv0.config.gpio3_analog_mapping  { endpoint=None, min=0.0, max=0.0 }
odrv0.config.gpio4_analog_mapping  { endpoint=None, min=0.0, max=0.0 }
odrv0.config.gpio1_pwm_mapping     { endpoint=None, min=0.0, max=0.0 }
odrv0.config.gpio2_pwm_mapping     { endpoint=None, min=0.0, max=0.0 }
odrv0.config.gpio3_pwm_mapping     { endpoint=None, min=0.0, max=0.0 }
odrv0.config.gpio4_pwm_mapping     { endpoint=None, min=0.0, max=0.0 }
```

- `get_adc_voltage(3)` / `get_adc_voltage(4)` — returns float volts (0–3.3 V). Non-ADC pins return NaN.
- `analog_mapping` routes ADC → a control endpoint (e.g. velocity setpoint). Separate from raw reads.
- `gpio_modes[]` array (0.5.5+ style) is **not present**.
- Do not exceed 3.3 V on GPIO pins.

---

## CAN

```
odrv0.can.config.baud_rate  = 250000
odrv0.can.config.protocol   = 0
odrv0.can.error             = 0
```

Callable: `odrv0.can.set_baud_rate()`

axis0 CAN node ID: **0** — axis1 CAN node ID: **1**

---

## Per-axis Summary

Both axes are structurally identical. Differences noted below.

### axis0

| Property | Value |
|---|---|
| current_state | 1 (IDLE) |
| error | 0 |
| is_homed | False |
| lockin_state | 0 |
| step_dir_active | False |
| FET temperature | 27.84 °C |
| Motor thermistor | disabled |

### axis1

| Property | Value |
|---|---|
| current_state | 1 (IDLE) |
| error | 0 |
| is_homed | False |
| lockin_state | 0 |
| step_dir_active | False |
| FET temperature | 25.82 °C |
| Motor thermistor | disabled |

---

## Encoder Config

### axis0 — SPI Absolute (AMS)

```
encoder.config.mode               = 257  (ENCODER_MODE_SPI_ABS_AMS — AS5047/AS5048 family)
encoder.config.cpr                = 16384  (14-bit, 2^14)
encoder.config.abs_spi_cs_gpio_pin = 3
encoder.config.bandwidth          = 1000.0
encoder.config.enable_phase_interpolation = True
encoder.config.pre_calibrated     = False
encoder.config.use_index          = False
encoder.config.offset             = 0
encoder.config.offset_float       = 0.0
encoder.is_ready                  = False
encoder.pos_abs                   = 9517  (raw counts at probe time)
encoder.spi_error_rate            = 0.0
```

### axis1 — Incremental

```
encoder.config.mode               = 0  (ENCODER_MODE_INCREMENTAL)
encoder.config.cpr                = 8192
encoder.config.abs_spi_cs_gpio_pin = 1  (unused for incremental)
encoder.config.bandwidth          = 1000.0
encoder.config.enable_phase_interpolation = True
encoder.config.pre_calibrated     = False
encoder.config.use_index          = False
encoder.is_ready                  = False
encoder.pos_abs                   = 0
```

---

## Motor Config (both axes identical)

```
motor.config.motor_type                = 0  (MOTOR_TYPE_HIGH_CURRENT / BLDC)
motor.config.pole_pairs                = 7
motor.config.calibration_current       = 10.0 A
motor.config.current_lim               = 10.0 A
motor.config.current_lim_margin        = 8.0 A
motor.config.requested_current_range   = 60.0 A
motor.config.resistance_calib_max_voltage = 2.0 V
motor.config.torque_constant           = 0.04 Nm/A
motor.config.torque_lim                = inf
motor.config.phase_inductance          = 0.0  (not yet calibrated)
motor.config.phase_resistance          = 0.0  (not yet calibrated)
motor.config.current_control_bandwidth = 1000.0
motor.config.direction                 = 0
motor.config.pre_calibrated            = False
motor.config.inverter_temp_limit_lower = 100.0 °C
motor.config.inverter_temp_limit_upper = 120.0 °C
motor.is_calibrated                    = False
motor.current_control.max_allowed_current  = 60.75 A
motor.current_control.overcurrent_trip_level = 67.5 A
motor.current_control.i_gain           = nan  (not calibrated)
motor.current_control.p_gain           = 0.0  (not calibrated)
```

Note: `phase_inductance` and `phase_resistance` are both 0 — motor calibration has not been saved/pre-calibrated.

---

## Controller Config (both axes identical unless noted)

```
controller.config.control_mode            = 3   (CONTROL_MODE_TORQUE_CONTROL)
controller.config.input_mode              = 1   (INPUT_MODE_PASSTHROUGH)
controller.config.pos_gain                = 20.0
controller.config.vel_gain                = 0.1667
controller.config.vel_integrator_gain     = 0.3333
controller.config.vel_limit               = 2.0  turns/s
controller.config.vel_limit_tolerance     = 1.2
controller.config.vel_ramp_rate           = 1.0
controller.config.torque_ramp_rate        = 0.01
controller.config.inertia                 = 0.0
controller.config.input_filter_bandwidth  = 2.0
controller.config.enable_vel_limit        = True
controller.config.enable_current_mode_vel_limit = True
controller.config.enable_overspeed_error  = True
controller.config.enable_gain_scheduling  = False
controller.config.gain_scheduling_width   = 10.0
controller.config.circular_setpoints      = False
controller.config.circular_setpoint_range = 1.0
controller.config.homing_speed            = 0.25
controller.config.load_encoder_axis       = 0  (axis0) / 1 (axis1)
controller.config.axis_to_mirror          = 255  (disabled)
controller.config.mirror_ratio            = 1.0
```

### Anticogging sub-object

```
controller.config.anticogging.anticogging_enabled = True
controller.config.anticogging.pre_calibrated      = False
controller.config.anticogging.calib_pos_threshold = 1.0
controller.config.anticogging.calib_vel_threshold = 1.0
controller.config.anticogging.cogging_ratio        = 1.0
controller.config.anticogging.index               = 0
controller.config.anticogging.calib_anticogging   = False
```

`anticogging_valid = False` on both axes — calibration not yet run or saved.

---

## Axis Config

```
axis.config.startup_motor_calibration          = False
axis.config.startup_encoder_index_search       = False
axis.config.startup_encoder_offset_calibration = False
axis.config.startup_closed_loop_control        = False
axis.config.startup_sensorless_control         = False
axis.config.startup_homing                     = False
axis.config.enable_step_dir                    = False
axis.config.step_dir_always_on                 = False
axis.config.enable_watchdog                    = False
axis.config.watchdog_timeout                   = 0.0 s
axis.config.turns_per_step                     = 0.000977 (= 1/1024)
axis.config.can_node_id                        = 0 (axis0) / 1 (axis1)
axis.config.can_heartbeat_rate_ms              = 100
axis.config.can_node_id_extended               = False
axis.config.step_gpio_pin                      = 1 (axis0) / 7 (axis1)
axis.config.dir_gpio_pin                       = 2 (axis0) / 8 (axis1)
```

### calibration_lockin defaults

```
accel=20.0, current=10.0, vel=40.0
ramp_time=0.4 s, ramp_distance=π rad
```

### general_lockin defaults

```
accel=20.0, current=10.0, vel=40.0, finish_distance=100.0
finish_on_distance=False, finish_on_vel=False, finish_on_enc_idx=False
```

### sensorless_ramp defaults

```
accel=200.0, current=10.0, vel=400.0, finish_distance=100.0
finish_on_vel=True
```

---

## Trap Trajectory Config

```
trap_traj.config.vel_limit   = 2.0
trap_traj.config.accel_limit = 0.5
trap_traj.config.decel_limit = 0.5
```

---

## Sensorless Estimator Config

```
sensorless_estimator.config.observer_gain   = 1000.0
sensorless_estimator.config.pll_bandwidth   = 1000.0
sensorless_estimator.config.pm_flux_linkage = 0.00158 Wb
```

---

## Thermistors

### FET (built-in, always enabled)

```
fet_thermistor.config.enabled           = True
fet_thermistor.config.temp_limit_lower  = 100 °C
fet_thermistor.config.temp_limit_upper  = 120 °C
```

### Motor thermistor (external, disabled)

```
motor_thermistor.config.enabled   = False
motor_thermistor.config.gpio_pin  = 4
motor_thermistor.config.poly_coefficient_0..3 = 0.0  (not configured)
```

---

## Endstops

Both min/max endstops on both axes are disabled:

```
endstop.config.enabled       = False
endstop.config.gpio_num      = 0
endstop.config.is_active_high = False
endstop.config.pullup        = True
endstop.config.debounce_ms   = 50
endstop.config.offset        = 0.0
```

---

## Callable Commands (complete list)

### odrv0 level

| Command | Notes |
|---|---|
| `odrv0.reboot()` | Soft reboot |
| `odrv0.save_configuration()` | Persist config to flash |
| `odrv0.erase_configuration()` | Factory reset config |
| `odrv0.enter_dfu_mode()` | Enter DFU for firmware flashing |
| `odrv0.get_adc_voltage(pin)` | Read ADC voltage; GPIO 3/4 only |
| `odrv0.get_oscilloscope_val(index)` | Read internal oscilloscope buffer |
| `odrv0.test_function()` | Internal test hook |
| `odrv0.can.set_baud_rate(baud)` | Change CAN baud rate |

### Per-axis

| Command | Notes |
|---|---|
| `axis.clear_errors()` | Clear all error flags |
| `axis.watchdog_feed()` | Feed watchdog timer |
| `axis.controller.move_incremental(displacement, from_input_pos)` | Relative move |
| `axis.controller.start_anticogging_calibration()` | Run anticogging cal |
| `axis.encoder.set_linear_count(count)` | Override encoder count |

### from_json() (internal)

Every sub-object exposes `from_json()` — this is an internal ODrive RPC method used for
config loading. Do not call manually.

---

## API Quirks vs. Documented 0.5.x

| Feature | Documented 0.5.x | This build |
|---|---|---|
| Firmware version | `fw_version_major/minor/revision` | Always 0.0.0 — unreliable |
| Unreleased flag | `fw_version_unreleased` | 1 — confirms dev snapshot |
| Brake resistor enable | `config.enable_brake_resistor` | **Not present** — use `brake_resistance > 0` |
| GPIO mode config | `config.gpio4_mode = 2` | **Not present** — analog implicit on GPIO 3/4 |
| GPIO mode array | `config.gpio_modes[n]` | **Not present** |
| ADC read | `get_adc_voltage(pin)` | Works on GPIO 3/4, NaN elsewhere |
| Encoder mode 257 | Not in standard docs | SPI absolute AMS (AS5047/AS5048) |
| `use_index_offset` | 0.5.5+ | **Not present** |
| ACIM estimator | some 0.5.x builds | **Not present** (stripped) |
| Old-style `move_to_pos()` | pre-0.5.2 | **Not present** |

---

## Connecting & Scripting

```python
import odrive
odrv0 = odrive.find_any()

# Hardware version (reliable)
print(odrv0.hw_version_major, odrv0.hw_version_minor, odrv0.hw_version_variant)
# → 3 6 56

# Firmware version (unreliable — always 0.0.0 on this board)
print(odrv0.fw_version_major, odrv0.fw_version_minor, odrv0.fw_version_revision)
# → 0 0 0

# Dev build flag
print(odrv0.fw_version_unreleased)
# → 1

# ADC read (GPIO 3 or 4 only, no config needed)
print(odrv0.get_adc_voltage(4))

# Torque control — set input torque directly (current control_mode=3, input_mode=1)
odrv0.axis0.controller.input_torque = 0.5  # Nm

# Safe attribute access pattern (needed for any attr that may not exist)
val = getattr(odrv0.config, 'enable_brake_resistor', None)
```

---

## Tools

| File | Purpose |
|---|---|
| `probe_firmware.py` | Full recursive attribute dump + fingerprints. Run after any firmware change. |
| `probe_results.txt` | Raw output from last probe run. |
| `odrive_gui.py` | PySide6 GUI for config, motor control, live plots, diagnostics. |
