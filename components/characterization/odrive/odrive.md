# ODrive 3.6 — Hardware & Firmware Reference (v0.5.6)

> Probed: 2026-04-03 via `probe_firmware.py` (full recursive tree, depth 8).
> Firmware: **ODrive v0.5.6 official release** (`fw_version_unreleased = 0`).
> Raw results: `probe_results.txt`
>
> **WARNING: `user_config_loaded = 0`** — firmware flash wiped all configuration.
> Axis0 encoder (SPI ABS AMS, mode=257, cpr=16384) must be re-configured before use.

---

## Hardware

| Attribute | Value |
|---|---|
| Board | ODrive hw v3.6 variant 56 |
| Serial number | 61985812394293 |
| vbus_voltage (at probe) | 23.94 V |
| user_config_loaded | **0 — config was erased by firmware flash** |
| brake_resistor_armed | False (was True — needs re-enable) |
| brake_resistor_saturated | False |
| brake_resistor_current | 0.0 A |
| misconfigured | False |
| otp_valid | False |

---

## Firmware

| Attribute | Value |
|---|---|
| fw_version_major | 0 |
| fw_version_minor | 5 |
| fw_version_revision | 6 |
| fw_version_unreleased | **0** (official release build) |

### Feature fingerprinting (all OK on 0.5.6)

| Feature | Present |
|---|---|
| `get_adc_voltage()` | YES |
| `can` object | YES |
| `anticogging` sub-object | YES |
| `startup_homing` | YES |
| `enable_phase_interpolation` | YES |
| `move_incremental()` | YES |
| `dc_bus_overvoltage/undervoltage_trip_level` | YES |
| `inertia`, `input_filter_bandwidth` | YES |
| `current_control_bandwidth` | YES |
| `general_lockin`, `sensorless_estimator` | YES |
| `trap_traj.config` | YES |
| `use_index_offset` | YES *(was absent on old snapshot)* |
| `enable_brake_resistor` | YES *(was absent on old snapshot)* |
| `acim_estimator` | YES *(was stripped from old snapshot)* |
| `gpio_modes[]` array | **NO** — per-pin `gpio{n}_mode` attrs used instead |
| `move_to_pos()` (pre-0.5.2) | **NO** — use `move_incremental()` |

---

## What Changed vs. Old Dev Snapshot

Major structural changes in 0.5.6 that break old scripting:

| Area | Old snapshot | 0.5.6 |
|---|---|---|
| Per-axis CAN config | flat: `axis.config.can_node_id` etc. | sub-object: `axis.config.can.node_id` etc. |
| GPIO mode config | not present | per-pin: `config.gpio{n}_mode` (int) |
| UART config | `config.enable_uart`, `uart_baudrate` | `config.enable_uart_a/b/c`, `uart_a/b/c_baudrate` |
| CAN enable | `config.enable_i2c_instead_of_can` | `config.enable_can_a`, `config.enable_i2c_a` |
| Brake resistor | `enable_brake_resistor` absent | `config.enable_brake_resistor` present |
| Thermistors | `axis.fet_thermistor`, `axis.motor_thermistor` | moved to `axis.motor.fet_thermistor`, `axis.motor.motor_thermistor` |
| Motor armed state | `motor.armed_state` (int) | `motor.is_armed` (bool) |
| Motor gate driver | `motor.gate_driver.drv_fault` | `axis.last_drv_fault` |
| Motor timing | `motor.timing_log.*` | `axis.task_times.*` (much more detailed) |
| Oscilloscope | `odrv0.get_oscilloscope_val()` | `odrv0.oscilloscope.get_val()` |
| Encoder offset fields | `config.offset`, `config.offset_float` | `config.phase_offset`, `config.phase_offset_float` |
| Controller vel limit | `enable_current_mode_vel_limit` | `enable_torque_mode_vel_limit` |
| ASCII USB protocol | `config.enable_ascii_protocol_on_usb` | `config.usb_cdc_protocol` (int) |
| `acim_estimator` | not present | present |
| `odrv0.clear_errors()` | not present | present |

---

## odrv0 — Top-level Config

```
odrv0.config.brake_resistance                 = 2.0 Ω
odrv0.config.enable_brake_resistor            = False  ← must set True to arm brake resistor
odrv0.config.dc_bus_overvoltage_trip_level    = 59.92 V
odrv0.config.dc_bus_overvoltage_ramp_start    = 59.92 V
odrv0.config.dc_bus_overvoltage_ramp_end      = 59.92 V
odrv0.config.enable_dc_bus_overvoltage_ramp   = False
odrv0.config.dc_bus_undervoltage_trip_level   = 8.0 V
odrv0.config.dc_max_positive_current          = inf
odrv0.config.dc_max_negative_current          = -0.01 A  (regen nearly disabled)
odrv0.config.max_regen_current                = 0.0 A
odrv0.config.enable_uart_a                    = True
odrv0.config.enable_uart_b                    = False
odrv0.config.enable_uart_c                    = False
odrv0.config.uart_a_baudrate                  = 115200
odrv0.config.uart_b_baudrate                  = 115200
odrv0.config.uart_c_baudrate                  = 115200
odrv0.config.uart0_protocol                   = 3
odrv0.config.uart1_protocol                   = 3
odrv0.config.uart2_protocol                   = 3
odrv0.config.usb_cdc_protocol                 = 3
odrv0.config.enable_can_a                     = True
odrv0.config.enable_i2c_a                     = False
odrv0.config.error_gpio_pin                   = 0  (disabled)
```

---

## GPIO Config

0.5.6 uses **per-pin mode integers** (`gpio{n}_mode`), not a `gpio_modes[]` array.

```
gpio1_mode  = 4    gpio2_mode  = 4    gpio3_mode  = 3    gpio4_mode  = 3
gpio5_mode  = 3    gpio6_mode  = 0    gpio7_mode  = 0    gpio8_mode  = 0
gpio9_mode  = 11   gpio10_mode = 11   gpio11_mode = 2    gpio12_mode = 12
gpio13_mode = 12   gpio14_mode = 2    gpio15_mode = 7    gpio16_mode = 7
```

Mode values (from ODrive 0.5.6 source):
`0=digital, 1=digital_pull_up, 2=digital_pull_down, 3=analog_in, 4=uart_a, 7=uart_b, 11=can_a, 12=i2c_a`

ADC-capable: GPIO 3 and 4 (mode=3 = analog_in).

```
odrv0.config.gpio3_analog_mapping  { endpoint=None, min=0.0, max=0.0 }
odrv0.config.gpio4_analog_mapping  { endpoint=None, min=0.0, max=0.0 }
odrv0.config.gpio1_pwm_mapping     { endpoint=None, min=0.0, max=0.0 }
odrv0.config.gpio2_pwm_mapping     { endpoint=None, min=0.0, max=0.0 }
odrv0.config.gpio3_pwm_mapping     { endpoint=None, min=0.0, max=0.0 }
odrv0.config.gpio4_pwm_mapping     { endpoint=None, min=0.0, max=0.0 }
```

Read raw ADC: `odrv0.get_adc_voltage(3)` / `odrv0.get_adc_voltage(4)` — returns volts (0–3.3 V).

---

## CAN

```
odrv0.can.config.baud_rate  = 250000
odrv0.can.config.protocol   = 1  (was 0 on old snapshot)
odrv0.can.error             = 0
```

Note: `odrv0.can.set_baud_rate()` is **gone** in 0.5.6 — change baud rate via `odrv0.can.config.baud_rate` + `save_configuration()`.

Per-axis CAN is now configured under `axis.config.can` (see Axis Config below).

---

## Per-axis Summary

### axis0

| Property | Value |
|---|---|
| current_state | 1 (IDLE) |
| error | 0 |
| is_homed | False |
| last_drv_fault | 0 |
| step_dir_active | False |
| steps | 0 |
| FET temperature | 29.2 °C |
| Motor thermistor | disabled |

### axis1

| Property | Value |
|---|---|
| current_state | 1 (IDLE) |
| error | 0 |
| is_homed | False |
| last_drv_fault | 0 |
| step_dir_active | False |
| steps | 0 |
| FET temperature | 26.7 °C |
| Motor thermistor | disabled |

---

## Axis Config

```
axis.config.startup_motor_calibration          = False
axis.config.startup_encoder_index_search       = False
axis.config.startup_encoder_offset_calibration = False
axis.config.startup_closed_loop_control        = False
axis.config.startup_homing                     = False
axis.config.enable_sensorless_mode             = False  ← NEW in 0.5.6
axis.config.enable_step_dir                    = False
axis.config.step_dir_always_on                 = False
axis.config.enable_watchdog                    = False
axis.config.watchdog_timeout                   = 0.0 s
axis.config.step_gpio_pin                      = 1 (axis0) / 7 (axis1)
axis.config.dir_gpio_pin                       = 2 (axis0) / 8 (axis1)
```

Note: `startup_sensorless_control` and `turns_per_step` from old snapshot are **gone**.

### Per-axis CAN config (new sub-object in 0.5.6)

```
axis.config.can.node_id               = 0 (axis0) / 1 (axis1)
axis.config.can.is_extended           = False
axis.config.can.heartbeat_rate_ms     = 100
axis.config.can.encoder_rate_ms       = 10
axis.config.can.encoder_count_rate_ms = 0
axis.config.can.iq_rate_ms            = 0
axis.config.can.bus_vi_rate_ms        = 0
axis.config.can.encoder_error_rate_ms = 0
axis.config.can.motor_error_rate_ms   = 0
axis.config.can.controller_error_rate_ms = 0
axis.config.can.sensorless_error_rate_ms = 0
axis.config.can.sensorless_rate_ms    = 0
```

### calibration_lockin / general_lockin / sensorless_ramp

Unchanged from old snapshot — defaults as before.

---

## Encoder Config

**Both axes reset to defaults after firmware flash.** Must re-configure axis0 for SPI ABS encoder.

### axis0 — NEEDS RECONFIGURATION

Factory reset state (post-flash):
```
encoder.config.mode               = 0  (incremental — WRONG, needs 257 for AMS SPI abs)
encoder.config.cpr                = 8192  (WRONG, needs 16384 for 14-bit AS5047/AS5048)
encoder.config.abs_spi_cs_gpio_pin = 1  (needs to be 3)
encoder.config.pre_calibrated     = False
encoder.config.use_index          = False
encoder.config.use_index_offset   = True   ← NEW in 0.5.6
encoder.config.index_offset       = 0.0    ← NEW in 0.5.6
encoder.config.direction          = 0      ← NEW in 0.5.6
encoder.config.hall_polarity      = 0      ← NEW in 0.5.6
encoder.config.hall_polarity_calibrated = False  ← NEW in 0.5.6
encoder.config.phase_offset       = 0     (renamed from 'offset')
encoder.config.phase_offset_float = 0.0   (renamed from 'offset_float')
encoder.is_ready                  = False
```

**To restore axis0 SPI abs encoder config:**
```python
odrv0.axis0.encoder.config.mode = 257               # ENCODER_MODE_SPI_ABS_AMS
odrv0.axis0.encoder.config.cpr = 16384              # 14-bit
odrv0.axis0.encoder.config.abs_spi_cs_gpio_pin = 3
odrv0.save_configuration()
odrv0.reboot()
```

### axis1 — incremental (correct as-is)

```
encoder.config.mode = 0   (incremental)
encoder.config.cpr  = 8192
encoder.config.abs_spi_cs_gpio_pin = 1  (unused)
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
motor.config.phase_inductance          = 0.0  (not calibrated)
motor.config.phase_resistance          = 0.0  (not calibrated)
motor.config.current_control_bandwidth = 1000.0
motor.config.direction                 = 0
motor.config.pre_calibrated            = False
motor.config.inverter_temp_limit_lower = 100.0 °C
motor.config.inverter_temp_limit_upper = 120.0 °C
motor.config.dc_calib_tau              = 0.2 s   ← NEW in 0.5.6
motor.config.I_bus_hard_max            = inf     ← NEW in 0.5.6
motor.config.I_bus_hard_min            = -inf    ← NEW in 0.5.6
motor.config.I_leak_max                = 0.1 A   ← NEW in 0.5.6
motor.config.R_wL_FF_enable            = False   ← NEW in 0.5.6
motor.config.bEMF_FF_enable            = False   ← NEW in 0.5.6
motor.is_calibrated                    = False
motor.is_armed                         = False   (replaces 'armed_state' int)
motor.max_allowed_current              = 60.75 A (moved from current_control)
motor.max_dc_calib                     = 6.075 A ← NEW in 0.5.6
motor.current_control.i_gain           = nan     (not calibrated)
motor.current_control.p_gain           = 0.0
```

---

## Controller Config (both axes identical unless noted)

```
controller.config.control_mode                    = 3  (TORQUE_CONTROL)
controller.config.input_mode                      = 1  (PASSTHROUGH)
controller.config.pos_gain                        = 20.0
controller.config.vel_gain                        = 0.1667
controller.config.vel_integrator_gain             = 0.3333
controller.config.vel_integrator_limit            = inf   ← NEW in 0.5.6
controller.config.vel_limit                       = 2.0
controller.config.vel_limit_tolerance             = 1.2
controller.config.vel_ramp_rate                   = 1.0
controller.config.torque_ramp_rate                = 0.01
controller.config.inertia                         = 0.0
controller.config.input_filter_bandwidth          = 2.0
controller.config.enable_vel_limit                = True
controller.config.enable_torque_mode_vel_limit    = True  (renamed from enable_current_mode_vel_limit)
controller.config.enable_overspeed_error          = True
controller.config.enable_gain_scheduling          = False
controller.config.gain_scheduling_width           = 10.0
controller.config.circular_setpoints              = False
controller.config.circular_setpoint_range         = 1.0
controller.config.steps_per_circular_range        = 1024  ← NEW in 0.5.6
controller.config.homing_speed                    = 0.25
controller.config.load_encoder_axis               = 0 / 1
controller.config.axis_to_mirror                  = 255  (disabled)
controller.config.mirror_ratio                    = 1.0
controller.config.torque_mirror_ratio             = 0.0  ← NEW in 0.5.6
controller.config.electrical_power_bandwidth      = 20.0  ← NEW in 0.5.6
controller.config.mechanical_power_bandwidth      = 20.0  ← NEW in 0.5.6
controller.config.spinout_electrical_power_threshold = 10.0   ← NEW in 0.5.6
controller.config.spinout_mechanical_power_threshold = -10.0  ← NEW in 0.5.6
```

### Live controller readings (new in 0.5.6)

```
controller.electrical_power = 0.0
controller.mechanical_power = 0.0
controller.last_error_time  = 0.0
controller.autotuning.frequency       = 0.0
controller.autotuning.pos_amplitude   = 0.0
controller.autotuning.vel_amplitude   = 0.0
controller.autotuning.torque_amplitude = 0.0
controller.autotuning_phase           = 0.0
```

### Anticogging

```
controller.config.anticogging.anticogging_enabled = True
controller.config.anticogging.pre_calibrated      = False
controller.config.anticogging.calib_pos_threshold = 1.0
controller.config.anticogging.calib_vel_threshold = 1.0
controller.config.anticogging.cogging_ratio        = 1.0
controller.config.anticogging.index               = 0
```

---

## Trap Trajectory Config

```
trap_traj.config.vel_limit   = 2.0
trap_traj.config.accel_limit = 0.5
trap_traj.config.decel_limit = 0.5
```

---

## Thermistors (moved inside motor in 0.5.6)

```
axis.motor.fet_thermistor.config.enabled           = True
axis.motor.fet_thermistor.config.temp_limit_lower  = 100 °C
axis.motor.fet_thermistor.config.temp_limit_upper  = 120 °C

axis.motor.motor_thermistor.config.enabled   = False
axis.motor.motor_thermistor.config.gpio_pin  = 4
```

Note: in the old snapshot these were at `axis.fet_thermistor` / `axis.motor_thermistor`.
All scripts must update the path.

---

## Mechanical Brake (new in 0.5.6)

```
axis.mechanical_brake.config.gpio_num    = 0  (disabled)
axis.mechanical_brake.config.is_active_low = True
```

Callables: `axis.mechanical_brake.engage()`, `axis.mechanical_brake.release()`

---

## Callable Commands (complete list)

### odrv0 level

| Command | Notes |
|---|---|
| `odrv0.reboot()` | Soft reboot |
| `odrv0.save_configuration()` | Persist config to flash |
| `odrv0.erase_configuration()` | Factory reset |
| `odrv0.clear_errors()` | Clear top-level errors ← NEW in 0.5.6 |
| `odrv0.enter_dfu_mode()` | Enter DFU for firmware flashing |
| `odrv0.get_adc_voltage(pin)` | ADC read — GPIO 3/4 only |
| `odrv0.get_gpio_states()` | Read all GPIO states ← NEW in 0.5.6 |
| `odrv0.get_drv_fault()` | Read DRV gate driver fault ← NEW in 0.5.6 |
| `odrv0.get_interrupt_status()` | Read interrupt status ← NEW in 0.5.6 |
| `odrv0.get_dma_status()` | Read DMA status ← NEW in 0.5.6 |
| `odrv0.oscilloscope.get_val()` | Read oscilloscope buffer (replaces `get_oscilloscope_val()`) |
| `odrv0.test_function()` | Internal test hook |

### Per-axis

| Command | Notes |
|---|---|
| `axis.clear_errors()` | Clear axis error flags |
| `axis.watchdog_feed()` | Feed watchdog |
| `axis.controller.move_incremental(d, from_input_pos)` | Relative move |
| `axis.controller.start_anticogging_calibration()` | Run anticogging cal |
| `axis.controller.get_anticogging_value()` | Read current anticogging value ← NEW |
| `axis.controller.remove_anticogging_bias()` | Remove anticogging bias ← NEW |
| `axis.encoder.set_linear_count(count)` | Override encoder count |
| `axis.mechanical_brake.engage()` | Engage mechanical brake ← NEW |
| `axis.mechanical_brake.release()` | Release mechanical brake ← NEW |

---

## Connecting & Scripting

```python
import odrive
odrv0 = odrive.find_any()

# Verify firmware
print(odrv0.fw_version_major, odrv0.fw_version_minor, odrv0.fw_version_revision)
# → 0 5 6

# Restore axis0 abs SPI encoder after config wipe
odrv0.axis0.encoder.config.mode = 257               # ENCODER_MODE_SPI_ABS_AMS
odrv0.axis0.encoder.config.cpr = 16384
odrv0.axis0.encoder.config.abs_spi_cs_gpio_pin = 3
odrv0.config.enable_brake_resistor = True
odrv0.save_configuration()
odrv0.reboot()

# Torque control (control_mode=3, input_mode=1)
odrv0.axis0.controller.input_torque = 0.5  # Nm

# Per-axis CAN node ID (new path in 0.5.6 — NOT axis.config.can_node_id)
odrv0.axis0.config.can.node_id = 0
odrv0.axis1.config.can.node_id = 1

# Thermistor path changed in 0.5.6
temp = odrv0.axis0.motor.fet_thermistor.temperature  # was axis0.fet_thermistor
```

---

## Tools

| File | Purpose |
|---|---|
| `probe_firmware.py` | Full recursive attribute dump + fingerprints. Re-run after any firmware change. |
| `probe_results.txt` | Raw output from last probe (0.5.6, post-flash, config wiped). |
| `odrive_gui.py` | PySide6 GUI for config, motor control, live plots, diagnostics. |
| `updateFirmware.md` | Step-by-step DFU → flash → Zadig driver procedure. |
