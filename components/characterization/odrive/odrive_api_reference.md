# ODrive 3.6 API Reference — Firmware v0.5.6

> **Source of truth** for the unified ODrive GUI.
> Generated from `probe_results.txt` (live device dump), `ODriveEnums.h`, ODrive 0.5.6 local docs, and `odrive_gui_v2.py` error hint tables.
>
> Device: HW v3.6 variant 56 &bull; FW v0.5.6 &bull; Serial: 61985812394293

---

## Table of Contents

1. [Device-Level](#device-level)
2. [Per-Axis (axis0 / axis1)](#per-axis-axis0--axis1)
3. [Enums](#enums)
4. [Error Bitfields](#error-bitfields)

---

## Device-Level

### Config (read/write) &mdash; `odrv0.config.*`

| Parameter | Type | Default | Range / Notes | Writable |
|---|---|---|---|---|
| `brake_resistance` | float | 2.0 | Ohms. Must match physical resistor | Yes |
| `enable_brake_resistor` | bool | False | **Must be True before closed-loop if brake resistor attached** | Yes |
| `dc_bus_overvoltage_trip_level` | float | 59.92 | V. Trips if Vbus exceeds this | Yes |
| `dc_bus_overvoltage_ramp_start` | float | 59.92 | V. Ramp starts dumping to brake resistor | Yes |
| `dc_bus_overvoltage_ramp_end` | float | 59.92 | V. Full brake duty at this level | Yes |
| `enable_dc_bus_overvoltage_ramp` | bool | False | Enable ramp-based overvoltage protection | Yes |
| `dc_bus_undervoltage_trip_level` | float | 8.0 | V. Trips if Vbus drops below | Yes |
| `dc_max_negative_current` | float | -0.01 | A. Max regen current (negative = into supply) | Yes |
| `dc_max_positive_current` | float | inf | A. Max draw from supply | Yes |
| `max_regen_current` | float | 0.0 | A. Legacy regen limit | Yes |
| `enable_uart_a` | bool | True | UART A (GPIO 1,2) | Yes |
| `enable_uart_b` | bool | False | UART B | Yes |
| `enable_uart_c` | bool | False | UART C | Yes |
| `uart_a_baudrate` | int | 115200 | baud | Yes |
| `uart_b_baudrate` | int | 115200 | baud | Yes |
| `uart_c_baudrate` | int | 115200 | baud | Yes |
| `uart0_protocol` | int | 3 | 0=Fibre, 1=ASCII, 2=stdout, 3=ASCII+stdout | Yes |
| `uart1_protocol` | int | 3 | Same as above | Yes |
| `uart2_protocol` | int | 3 | Same as above | Yes |
| `usb_cdc_protocol` | int | 3 | Same as above | Yes |
| `enable_can_a` | bool | True | CAN bus | Yes |
| `enable_i2c_a` | bool | False | I2C bus | Yes |
| `error_gpio_pin` | int | 0 | GPIO pin for error status output | Yes |
| `gpio1_mode` | int | 4 (UART_A) | See GpioMode enum | Yes |
| `gpio2_mode` | int | 4 (UART_A) | See GpioMode enum | Yes |
| `gpio3_mode` | int | 3 (ANALOG_IN) | See GpioMode enum | Yes |
| `gpio4_mode` | int | 3 (ANALOG_IN) | See GpioMode enum | Yes |
| `gpio5_mode` | int | 3 (ANALOG_IN) | See GpioMode enum | Yes |
| `gpio6_mode` | int | 0 (DIGITAL) | See GpioMode enum | Yes |
| `gpio7_mode` | int | 0 (DIGITAL) | See GpioMode enum | Yes |
| `gpio8_mode` | int | 0 (DIGITAL) | See GpioMode enum | Yes |
| `gpio9_mode` | int | 11 (ENC0) | See GpioMode enum | Yes |
| `gpio10_mode` | int | 11 (ENC0) | See GpioMode enum | Yes |
| `gpio11_mode` | int | 2 (DIGITAL_PULL_DOWN) | See GpioMode enum | Yes |
| `gpio12_mode` | int | 12 (ENC1) | See GpioMode enum | Yes |
| `gpio13_mode` | int | 12 (ENC1) | See GpioMode enum | Yes |
| `gpio14_mode` | int | 2 (DIGITAL_PULL_DOWN) | See GpioMode enum | Yes |
| `gpio15_mode` | int | 7 (CAN_A) | See GpioMode enum | Yes |
| `gpio16_mode` | int | 7 (CAN_A) | See GpioMode enum | Yes |

**PWM Mappings** (`odrv0.config.gpioN_pwm_mapping`): GPIO 1-4 have `.endpoint`, `.min`, `.max` fields. Default: endpoint=None, min=0.0, max=0.0.

**Analog Mappings** (`odrv0.config.gpioN_analog_mapping`): GPIO 3-4 have `.endpoint`, `.min`, `.max` fields. Default: endpoint=None, min=0.0, max=0.0.

### CAN Config &mdash; `odrv0.can.config.*`

| Parameter | Type | Default | Notes | Writable |
|---|---|---|---|---|
| `baud_rate` | int | 250000 | CAN bus baud rate | Yes |
| `protocol` | int | 1 (SIMPLE) | See Protocol enum | Yes |

### Status (read-only) &mdash; `odrv0.*`

| Parameter | Type | Probed Value | Notes |
|---|---|---|---|
| `vbus_voltage` | float | 23.94 | V. DC bus voltage |
| `ibus` | float | 0.0 | A. DC bus current |
| `ibus_report_filter_k` | float | 1.0 | Filter coefficient for ibus reporting |
| `serial_number` | int | 61985812394293 | Unique device ID |
| `hw_version_major` | int | 3 | Hardware version |
| `hw_version_minor` | int | 6 | |
| `hw_version_variant` | int | 56 | |
| `fw_version_major` | int | 0 | Firmware version |
| `fw_version_minor` | int | 5 | |
| `fw_version_revision` | int | 6 | |
| `fw_version_unreleased` | int | 0 | 0 = official release |
| `user_config_loaded` | int | 0 | 0 = factory defaults loaded |
| `error` | int | 0 | Device-level error bitfield (ODriveError) |
| `can.error` | int | 0 | CAN error bitfield (CanError) |
| `brake_resistor_armed` | bool | False | Brake resistor active |
| `brake_resistor_current` | float | 0.0 | A. Current through brake resistor |
| `brake_resistor_saturated` | bool | False | Brake resistor at duty limit |
| `misconfigured` | bool | False | Config validation failed |
| `otp_valid` | bool | False | One-time programmable memory valid |
| `task_timers_armed` | bool | False | Task timing measurement active |
| `n_evt_control_loop` | int | 2087130 | Control loop iteration count |
| `n_evt_sampling` | int | 2087133 | ADC sampling event count |
| `test_property` | int | 0 | Debug test property |

### System Stats &mdash; `odrv0.system_stats.*` (read-only)

| Parameter | Type | Probed Value | Notes |
|---|---|---|---|
| `uptime` | int | 261007 | ms since boot |
| `min_heap_space` | int | 47432 | bytes. Minimum free heap observed |
| `max_stack_usage_axis` | int | 564 | bytes |
| `max_stack_usage_usb` | int | 468 | bytes |
| `max_stack_usage_uart` | int | 304 | bytes |
| `max_stack_usage_can` | int | 408 | bytes |
| `max_stack_usage_analog` | int | 316 | bytes |
| `max_stack_usage_startup` | int | 532 | bytes |
| `stack_size_axis` | int | 2048 | bytes |
| `stack_size_usb` | int | 4096 | bytes |
| `stack_size_uart` | int | 4096 | bytes |
| `stack_size_can` | int | 1024 | bytes |
| `stack_size_analog` | int | 1024 | bytes |
| `stack_size_startup` | int | 2048 | bytes |
| `usb.rx_cnt` | int | 6938 | USB packets received |
| `usb.tx_cnt` | int | 0 | USB packets sent |
| `usb.tx_overrun_cnt` | int | 0 | USB TX overruns |
| `i2c.addr` | int | 0 | I2C address |
| `i2c.addr_match_cnt` | int | 0 | I2C address matches |
| `i2c.rx_cnt` | int | 0 | I2C packets received |
| `i2c.error_cnt` | int | 0 | I2C errors |

### Oscilloscope &mdash; `odrv0.oscilloscope.*` (read-only)

| Parameter | Type | Value | Notes |
|---|---|---|---|
| `size` | int | 4096 | Oscilloscope buffer size |
| `get_val()` | callable | — | Read oscilloscope sample |

### Functions &mdash; `odrv0.*`

| Function | Description |
|---|---|
| `save_configuration()` | Save all `.config` to NVM. Persists across power cycles |
| `erase_configuration()` | Reset to factory defaults. **Triggers reboot** |
| `reboot()` | Reboot the ODrive |
| `clear_errors()` | Clear all error flags on all axes |
| `get_adc_voltage(gpio)` | Read ADC voltage on specified GPIO pin (new in 0.5.6) |
| `get_dma_status()` | DMA controller status (debug) |
| `get_drv_fault()` | Read DRV8301 gate driver fault register |
| `get_gpio_states()` | Read all GPIO pin states |
| `get_interrupt_status()` | Read interrupt controller status (debug) |
| `enter_dfu_mode()` | Enter Device Firmware Update mode |
| `test_function()` | Debug test function |

---

## Per-Axis (axis0 / axis1)

Both axes have identical structure. All paths below use `<axis>` as placeholder for `odrv0.axis0` or `odrv0.axis1`.

### Axis Config &mdash; `<axis>.config.*`

| Parameter | Type | Default (axis0 / axis1) | Notes | Writable |
|---|---|---|---|---|
| `startup_motor_calibration` | bool | False | Auto-calibrate motor on startup | Yes |
| `startup_encoder_index_search` | bool | False | Auto-search encoder index on startup | Yes |
| `startup_encoder_offset_calibration` | bool | False | Auto-calibrate encoder offset on startup | Yes |
| `startup_closed_loop_control` | bool | False | Auto-enter closed loop on startup | Yes |
| `startup_homing` | bool | False | Auto-home on startup (0.5.4+) | Yes |
| `enable_step_dir` | bool | False | Enable step/dir input | Yes |
| `step_dir_always_on` | bool | False | Keep step/dir active even when not in closed loop | Yes |
| `step_gpio_pin` | int | 1 / 7 | GPIO pin for step input | Yes |
| `dir_gpio_pin` | int | 2 / 8 | GPIO pin for dir input | Yes |
| `enable_sensorless_mode` | bool | False | Allow sensorless operation | Yes |
| `enable_watchdog` | bool | False | Enable axis watchdog timer | Yes |
| `watchdog_timeout` | float | 0.0 | seconds. 0 = disabled | Yes |

#### Calibration Lockin &mdash; `<axis>.config.calibration_lockin.*`

| Parameter | Type | Default | Notes | Writable |
|---|---|---|---|---|
| `current` | float | 10.0 | A. Lockin drive current | Yes |
| `ramp_time` | float | 0.4 | s. Ramp-up time | Yes |
| `ramp_distance` | float | 3.14159 | rad. Ramp distance | Yes |
| `vel` | float | 40.0 | rad/s. Target velocity | Yes |
| `accel` | float | 20.0 | rad/s^2. Acceleration | Yes |

#### General Lockin &mdash; `<axis>.config.general_lockin.*`

| Parameter | Type | Default | Notes | Writable |
|---|---|---|---|---|
| `current` | float | 10.0 | A | Yes |
| `ramp_time` | float | 0.4 | s | Yes |
| `ramp_distance` | float | 3.14159 | rad | Yes |
| `vel` | float | 40.0 | rad/s | Yes |
| `accel` | float | 20.0 | rad/s^2 | Yes |
| `finish_distance` | float | 100.0 | rad. Distance to finish | Yes |
| `finish_on_distance` | bool | False | Stop after distance reached | Yes |
| `finish_on_vel` | bool | False | Stop after velocity reached | Yes |
| `finish_on_enc_idx` | bool | False | Stop after encoder index found | Yes |

#### Sensorless Ramp &mdash; `<axis>.config.sensorless_ramp.*`

| Parameter | Type | Default | Notes | Writable |
|---|---|---|---|---|
| `current` | float | 10.0 | A | Yes |
| `ramp_time` | float | 0.4 | s | Yes |
| `ramp_distance` | float | 3.14159 | rad | Yes |
| `vel` | float | 400.0 | rad/s. Target sensorless velocity | Yes |
| `accel` | float | 200.0 | rad/s^2 | Yes |
| `finish_distance` | float | 100.0 | rad | Yes |
| `finish_on_distance` | bool | False | | Yes |
| `finish_on_vel` | bool | True | Switch to closed loop when vel reached | Yes |
| `finish_on_enc_idx` | bool | False | | Yes |

#### CAN Config &mdash; `<axis>.config.can.*`

| Parameter | Type | Default (axis0 / axis1) | Notes | Writable |
|---|---|---|---|---|
| `node_id` | int | 0 / 1 | CAN node ID for this axis | Yes |
| `is_extended` | bool | False | Use extended CAN IDs | Yes |
| `heartbeat_rate_ms` | int | 100 | ms. Heartbeat broadcast interval. 0=disabled | Yes |
| `encoder_rate_ms` | int | 10 | ms. Encoder feedback broadcast interval | Yes |
| `encoder_count_rate_ms` | int | 0 | ms. Encoder count broadcast | Yes |
| `iq_rate_ms` | int | 0 | ms. Current measurement broadcast | Yes |
| `bus_vi_rate_ms` | int | 0 | ms. Bus voltage/current broadcast | Yes |
| `motor_error_rate_ms` | int | 0 | ms. Motor error broadcast | Yes |
| `encoder_error_rate_ms` | int | 0 | ms. Encoder error broadcast | Yes |
| `controller_error_rate_ms` | int | 0 | ms. Controller error broadcast | Yes |
| `sensorless_error_rate_ms` | int | 0 | ms. Sensorless error broadcast | Yes |
| `sensorless_rate_ms` | int | 0 | ms. Sensorless estimate broadcast | Yes |

### Axis Status (read-only) &mdash; `<axis>.*`

| Parameter | Type | Notes |
|---|---|---|
| `current_state` | int | Current AxisState (see enum). 1 = IDLE |
| `requested_state` | int | Last requested state. **Write to change state** |
| `error` | int | Axis error bitfield (AxisError) |
| `is_homed` | bool | Homing completed |
| `last_drv_fault` | int | Last DRV8301 gate driver fault code |
| `step_dir_active` | bool | Step/dir interface active |
| `steps` | int | Step count (step/dir mode) |

### Axis Functions

| Function | Description |
|---|---|
| `watchdog_feed()` | Feed the watchdog timer to prevent timeout |

### Endstops &mdash; `<axis>.min_endstop.*` / `<axis>.max_endstop.*`

| Parameter | Type | Default | Notes | Writable |
|---|---|---|---|---|
| `config.enabled` | bool | False | Enable this endstop | Yes |
| `config.gpio_num` | int | 0 | GPIO pin number | Yes |
| `config.is_active_high` | bool | False | Active high (True) or active low (False) | Yes |
| `config.offset` | float | 0.0 | Position offset when endstop triggers | Yes |
| `config.debounce_ms` | int | 50 | ms. Debounce time | Yes |
| `endstop_state` | bool | — | Current endstop state (read-only) | No |

### Mechanical Brake &mdash; `<axis>.mechanical_brake.*`

| Parameter | Type | Default | Notes | Writable |
|---|---|---|---|---|
| `config.gpio_num` | int | 0 | GPIO pin for brake control | Yes |
| `config.is_active_low` | bool | True | Brake logic level | Yes |
| `engage()` | callable | — | Engage mechanical brake | — |
| `release()` | callable | — | Release mechanical brake | — |

---

### Motor &mdash; `<axis>.motor.*`

#### Motor Config &mdash; `<axis>.motor.config.*`

| Parameter | Type | Default | Range / Notes | Writable |
|---|---|---|---|---|
| `motor_type` | int | 0 (HIGH_CURRENT) | See MotorType enum | Yes |
| `pole_pairs` | int | 7 | Must match motor. Common: 7 (5065), 11 (6374) | Yes |
| `torque_constant` | float | 0.04 | Nm/A. = 8.27 / KV(rpm/V) | Yes |
| `current_lim` | float | 10.0 | A. Max motor current | Yes |
| `current_lim_margin` | float | 8.0 | A. Margin above current_lim before hard fault | Yes |
| `calibration_current` | float | 10.0 | A. Current used during motor calibration | Yes |
| `resistance_calib_max_voltage` | float | 2.0 | V. Max voltage for resistance measurement. Increase to 4-8 for stubborn motors | Yes |
| `current_control_bandwidth` | float | 1000.0 | Hz. Current controller bandwidth | Yes |
| `requested_current_range` | float | 60.0 | A. Determines ADC gain. Set > max expected current | Yes |
| `phase_inductance` | float | 0.0 | H. Set by calibration. 0 = not calibrated | Yes |
| `phase_resistance` | float | 0.0 | Ohms. Set by calibration. 0 = not calibrated | Yes |
| `pre_calibrated` | bool | False | Skip motor cal on startup if True. Set after successful cal + save | Yes |
| `torque_lim` | float | inf | Nm. Torque limit | Yes |
| `inverter_temp_limit_lower` | float | 100.0 | C. FET temp where current derating begins | Yes |
| `inverter_temp_limit_upper` | float | 120.0 | C. FET temp where motor disarms | Yes |
| `I_bus_hard_max` | float | inf | A. Hard max bus current | Yes |
| `I_bus_hard_min` | float | -inf | A. Hard min bus current (regen) | Yes |
| `I_leak_max` | float | 0.1 | A. Max leakage current | Yes |
| `dc_calib_tau` | float | 0.2 | s. DC calibration time constant | Yes |
| `R_wL_FF_enable` | bool | False | Resistance-weighted inductance feedforward | Yes |
| `bEMF_FF_enable` | bool | False | Back-EMF feedforward | Yes |
| `acim_autoflux_enable` | bool | False | ACIM auto flux control | Yes |
| `acim_autoflux_min_Id` | float | 10.0 | A. ACIM minimum Id | Yes |
| `acim_autoflux_attack_gain` | float | 10.0 | ACIM flux attack gain | Yes |
| `acim_autoflux_decay_gain` | float | 1.0 | ACIM flux decay gain | Yes |
| `acim_gain_min_flux` | float | 10.0 | ACIM minimum flux for gain | Yes |

#### Motor Status (read-only) &mdash; `<axis>.motor.*`

| Parameter | Type | Notes |
|---|---|---|
| `error` | int | Motor error bitfield (MotorError) |
| `last_error_time` | float | Timestamp of last error |
| `is_armed` | bool | Motor FETs actively switching |
| `is_calibrated` | bool | Motor calibration completed this session |
| `effective_current_lim` | float | A. Current limit after thermal derating |
| `max_allowed_current` | float | A. Absolute max current (from requested_current_range) |
| `max_dc_calib` | float | A. Max DC calibration current |
| `I_bus` | float | A. DC bus current drawn by this motor |
| `DC_calib_phA` | float | A. Phase A DC calibration offset |
| `DC_calib_phB` | float | A. Phase B DC calibration offset |
| `DC_calib_phC` | float | A. Phase C DC calibration offset |
| `current_meas_phA` | float | A. Phase A current measurement |
| `current_meas_phB` | float | A. Phase B current measurement |
| `current_meas_phC` | float | A. Phase C current measurement |
| `n_evt_current_measurement` | int | Current measurement event count |
| `n_evt_pwm_update` | int | PWM update event count |
| `phase_current_rev_gain` | float | ADC-to-current conversion gain |

#### Current Control (read-only) &mdash; `<axis>.motor.current_control.*`

| Parameter | Type | Notes |
|---|---|---|
| `Iq_setpoint` | float | A. Commanded q-axis current (torque-producing) |
| `Iq_measured` | float | A. Measured q-axis current |
| `Id_setpoint` | float | A. Commanded d-axis current (flux) |
| `Id_measured` | float | A. Measured d-axis current |
| `Vd_setpoint` | float | V. d-axis voltage command |
| `Vq_setpoint` | float | V. q-axis voltage command |
| `Ialpha_measured` | float | A. Alpha-axis current (Clarke transform) |
| `Ibeta_measured` | float | A. Beta-axis current (Clarke transform) |
| `final_v_alpha` | float | V. Final alpha voltage output |
| `final_v_beta` | float | V. Final beta voltage output |
| `p_gain` | float | Current controller P gain (auto-computed from bandwidth) |
| `i_gain` | float | Current controller I gain (nan until calibrated) |
| `phase` | float | rad. Electrical phase angle |
| `phase_vel` | float | rad/s. Electrical phase velocity |
| `power` | float | W. Electrical power |
| `v_current_control_integral_d` | float | D-axis integrator state |
| `v_current_control_integral_q` | float | Q-axis integrator state |
| `I_measured_report_filter_k` | float | Filter coefficient for current reporting |

#### FET Thermistor &mdash; `<axis>.motor.fet_thermistor.*`

| Parameter | Type | Default | Notes | Writable |
|---|---|---|---|---|
| `config.enabled` | bool | True | Enable FET temperature monitoring | Yes |
| `config.temp_limit_lower` | float | 100.0 | C. Start derating | Yes |
| `config.temp_limit_upper` | float | 120.0 | C. Disarm motor | Yes |
| `temperature` | float | — | C. Current FET temperature (read-only) | No |

#### Motor Thermistor &mdash; `<axis>.motor.motor_thermistor.*`

| Parameter | Type | Default | Notes | Writable |
|---|---|---|---|---|
| `config.enabled` | bool | False | Enable motor temperature monitoring | Yes |
| `config.gpio_pin` | int | 4 | GPIO for external NTC thermistor | Yes |
| `config.temp_limit_lower` | float | 100.0 | C. Start derating | Yes |
| `config.temp_limit_upper` | float | 120.0 | C. Disarm motor | Yes |
| `config.poly_coefficient_0` | float | 0.0 | Steinhart-Hart polynomial coeff 0 | Yes |
| `config.poly_coefficient_1` | float | 0.0 | Steinhart-Hart polynomial coeff 1 | Yes |
| `config.poly_coefficient_2` | float | 0.0 | Steinhart-Hart polynomial coeff 2 | Yes |
| `config.poly_coefficient_3` | float | 0.0 | Steinhart-Hart polynomial coeff 3 | Yes |
| `temperature` | float | — | C. Current motor temperature (read-only) | No |

---

### Encoder &mdash; `<axis>.encoder.*`

#### Encoder Config &mdash; `<axis>.encoder.config.*`

| Parameter | Type | Default | Range / Notes | Writable |
|---|---|---|---|---|
| `mode` | int | 0 (INCREMENTAL) | See EncoderMode enum. 257 = SPI_ABS_AMS (AS5047) | Yes |
| `cpr` | int | 8192 | Counts per revolution. AS5047 = 16384 (14-bit) | Yes |
| `abs_spi_cs_gpio_pin` | int | 1 | GPIO pin for SPI chip select | Yes |
| `bandwidth` | float | 1000.0 | Hz. Encoder estimation bandwidth | Yes |
| `pre_calibrated` | bool | False | Skip encoder cal on startup if True | Yes |
| `use_index` | bool | False | Use encoder index pulse (incremental only) | Yes |
| `use_index_offset` | bool | True | Apply index offset correction | Yes |
| `find_idx_on_lockin_only` | bool | False | Only search for index during lockin | Yes |
| `index_offset` | float | 0.0 | turns. Offset from index pulse | Yes |
| `enable_phase_interpolation` | bool | True | Interpolate phase between encoder counts | Yes |
| `calib_range` | float | 0.02 | rad. Acceptable calibration error range | Yes |
| `calib_scan_distance` | float | 50.27 | rad. Distance to scan during calibration | Yes |
| `calib_scan_omega` | float | 12.57 | rad/s. Speed during calibration scan | Yes |
| `direction` | int | 0 | Encoder direction. Set by calibration | Yes |
| `phase_offset` | int | 0 | Encoder-to-motor phase offset. Set by calibration | Yes |
| `phase_offset_float` | float | 0.0 | Fine phase offset. Set by calibration | Yes |
| `hall_polarity` | int | 0 | Hall sensor polarity | Yes |
| `hall_polarity_calibrated` | bool | False | Hall polarity calibration done | Yes |
| `ignore_illegal_hall_state` | bool | False | Ignore illegal hall state errors | Yes |
| `sincos_gpio_pin_sin` | int | 3 | GPIO for sin/cos encoder sin input | Yes |
| `sincos_gpio_pin_cos` | int | 4 | GPIO for sin/cos encoder cos input | Yes |

#### Encoder Status (read-only) &mdash; `<axis>.encoder.*`

| Parameter | Type | Notes |
|---|---|---|
| `error` | int | Encoder error bitfield (EncoderError) |
| `is_ready` | bool | Encoder initialized and providing valid data |
| `index_found` | bool | Index pulse has been found |
| `pos_estimate` | float | turns. Estimated position |
| `pos_estimate_counts` | float | counts. Estimated position in encoder counts |
| `pos_circular` | float | turns. Position wrapped to [0, circular_setpoint_range) |
| `pos_cpr_counts` | float | counts. Position within one revolution |
| `pos_abs` | int | Absolute encoder raw position |
| `vel_estimate` | float | turn/s. Estimated velocity |
| `vel_estimate_counts` | float | count/s. Estimated velocity in encoder counts |
| `count_in_cpr` | int | Current count within one CPR |
| `shadow_count` | int | Unfiltered encoder count |
| `delta_pos_cpr_counts` | float | Change in position per cycle |
| `interpolation` | float | Phase interpolation value |
| `phase` | float | rad. Encoder electrical phase |
| `hall_state` | int | Current hall sensor state (0-7) |
| `spi_error_rate` | float | SPI communication error rate |
| `calib_scan_response` | float | Calibration scan response magnitude |

#### Encoder Functions

| Function | Description |
|---|---|
| `set_linear_count(count)` | Set the encoder linear position count |

---

### Controller &mdash; `<axis>.controller.*`

#### Controller Config &mdash; `<axis>.controller.config.*`

| Parameter | Type | Default | Range / Notes | Writable |
|---|---|---|---|---|
| `control_mode` | int | 3 (POSITION) | See ControlMode enum | Yes |
| `input_mode` | int | 1 (PASSTHROUGH) | See InputMode enum | Yes |
| `pos_gain` | float | 20.0 | (turn/s) / turn. Position P gain | Yes |
| `vel_gain` | float | 0.1667 | Nm/(turn/s). Velocity P gain | Yes |
| `vel_integrator_gain` | float | 0.3333 | Nm/(turn/s * s). Velocity I gain | Yes |
| `vel_integrator_limit` | float | inf | Nm. Integrator windup limit | Yes |
| `vel_limit` | float | 2.0 | turn/s. Velocity safety limit | Yes |
| `vel_limit_tolerance` | float | 1.2 | Multiplier. Trip at vel_limit * vel_limit_tolerance | Yes |
| `vel_ramp_rate` | float | 1.0 | turn/s^2. Velocity ramp rate (VEL_RAMP mode) | Yes |
| `torque_ramp_rate` | float | 0.01 | Nm/s. Torque ramp rate (TORQUE_RAMP mode) | Yes |
| `inertia` | float | 0.0 | Nm/(turn/s^2). Feedforward inertia compensation | Yes |
| `input_filter_bandwidth` | float | 2.0 | Hz. POS_FILTER input filter bandwidth | Yes |
| `homing_speed` | float | 0.25 | turn/s. Speed during homing | Yes |
| `circular_setpoints` | bool | False | Enable circular position mode | Yes |
| `circular_setpoint_range` | float | 1.0 | turns. Circular mode wrap range | Yes |
| `steps_per_circular_range` | int | 1024 | Steps per circular range (step/dir) | Yes |
| `enable_vel_limit` | bool | True | Enable velocity limiting | Yes |
| `enable_overspeed_error` | bool | True | Trip on overspeed | Yes |
| `enable_torque_mode_vel_limit` | bool | True | Limit velocity in torque mode for safety | Yes |
| `enable_gain_scheduling` | bool | False | Enable position-dependent gain scheduling | Yes |
| `gain_scheduling_width` | float | 10.0 | Gain scheduling transition width | Yes |
| `electrical_power_bandwidth` | float | 20.0 | Hz. Electrical power filter bandwidth | Yes |
| `mechanical_power_bandwidth` | float | 20.0 | Hz. Mechanical power filter bandwidth | Yes |
| `spinout_electrical_power_threshold` | float | 10.0 | W. Spinout detection electrical threshold | Yes |
| `spinout_mechanical_power_threshold` | float | -10.0 | W. Spinout detection mechanical threshold | Yes |
| `axis_to_mirror` | int | 255 | Axis index for mirror mode (255 = none) | Yes |
| `mirror_ratio` | float | 1.0 | Position mirror ratio | Yes |
| `torque_mirror_ratio` | float | 0.0 | Torque mirror ratio | Yes |
| `load_encoder_axis` | int | 0 / 1 | Axis to use for load encoder feedback | Yes |

#### Anticogging Config &mdash; `<axis>.controller.config.anticogging.*`

| Parameter | Type | Default | Notes | Writable |
|---|---|---|---|---|
| `anticogging_enabled` | bool | True | Apply anticogging correction when calibrated | Yes |
| `pre_calibrated` | bool | False | Anticogging map saved to NVM | Yes |
| `calib_anticogging` | bool | False | Calibration in progress (set by firmware) | Yes |
| `calib_pos_threshold` | float | 1.0 | turns. Position error threshold for calibration | Yes |
| `calib_vel_threshold` | float | 1.0 | turn/s. Velocity threshold for calibration | Yes |
| `cogging_ratio` | float | 1.0 | Anticogging torque scaling factor | Yes |
| `index` | int | 0 | Current anticogging map index (debug) | Yes |

#### Controller Status (read-only) &mdash; `<axis>.controller.*`

| Parameter | Type | Notes |
|---|---|---|
| `error` | int | Controller error bitfield (ControllerError) |
| `last_error_time` | float | Timestamp of last error |
| `input_pos` | float | turns. **Write to command position** |
| `input_vel` | float | turn/s. **Write to command velocity** |
| `input_torque` | float | Nm. **Write to command torque** |
| `pos_setpoint` | float | turns. Internal position setpoint |
| `vel_setpoint` | float | turn/s. Internal velocity setpoint |
| `torque_setpoint` | float | Nm. Internal torque setpoint |
| `vel_integrator_torque` | float | Nm. Velocity integrator accumulator |
| `electrical_power` | float | W. Estimated electrical power |
| `mechanical_power` | float | W. Estimated mechanical power |
| `trajectory_done` | bool | Trajectory planner finished (TRAP_TRAJ mode) |
| `anticogging_valid` | bool | Anticogging map loaded and valid |
| `autotuning_phase` | float | Autotuning phase (debug) |

#### Controller Autotuning &mdash; `<axis>.controller.autotuning.*`

| Parameter | Type | Default | Notes | Writable |
|---|---|---|---|---|
| `frequency` | float | 0.0 | Hz. Autotuning excitation frequency | Yes |
| `pos_amplitude` | float | 0.0 | turns. Position excitation amplitude | Yes |
| `vel_amplitude` | float | 0.0 | turn/s. Velocity excitation amplitude | Yes |
| `torque_amplitude` | float | 0.0 | Nm. Torque excitation amplitude | Yes |

#### Controller Functions

| Function | Description |
|---|---|
| `move_incremental(displacement, from_goal_point)` | Move by relative amount. `from_goal_point=True` = relative to goal, `False` = relative to current |
| `start_anticogging_calibration()` | Begin anticogging torque map calibration |
| `remove_anticogging_bias()` | Remove DC bias from anticogging map |
| `get_anticogging_value(index)` | Read anticogging map value at index |

---

### Sensorless Estimator &mdash; `<axis>.sensorless_estimator.*`

#### Config &mdash; `<axis>.sensorless_estimator.config.*`

| Parameter | Type | Default | Notes | Writable |
|---|---|---|---|---|
| `pm_flux_linkage` | float | 0.00158 | Wb. = 5.51328895422 / (pole_pairs * KV_rpm) | Yes |
| `observer_gain` | float | 1000.0 | Sensorless observer gain | Yes |
| `pll_bandwidth` | float | 1000.0 | Hz. Phase-locked loop bandwidth | Yes |

#### Status (read-only)

| Parameter | Type | Notes |
|---|---|---|
| `error` | int | SensorlessEstimatorError bitfield |
| `phase` | float | rad. Estimated electrical phase |
| `phase_vel` | float | rad/s. Estimated electrical phase velocity |
| `pll_pos` | float | rad. PLL position estimate |
| `vel_estimate` | float | turn/s. Sensorless velocity estimate |

---

### ACIM Estimator &mdash; `<axis>.acim_estimator.*`

(AC Induction Motor mode &mdash; not used for BLDC/PMSM motors)

#### Config

| Parameter | Type | Default | Notes | Writable |
|---|---|---|---|---|
| `config.slip_velocity` | float | 14.706 | rad/s. ACIM slip velocity | Yes |

#### Status (read-only)

| Parameter | Type | Notes |
|---|---|---|
| `rotor_flux` | float | Rotor flux estimate |
| `phase_offset` | float | Phase offset estimate |
| `slip_vel` | float | Slip velocity estimate |
| `stator_phase` | float | Stator phase angle |
| `stator_phase_vel` | float | Stator phase velocity |

---

### Trap Trajectory &mdash; `<axis>.trap_traj.config.*`

| Parameter | Type | Default | Notes | Writable |
|---|---|---|---|---|
| `vel_limit` | float | 2.0 | turn/s. Max trajectory coasting speed | Yes |
| `accel_limit` | float | 0.5 | turn/s^2. Max trajectory acceleration | Yes |
| `decel_limit` | float | 0.5 | turn/s^2. Max trajectory deceleration | Yes |

---

## Enums

### AxisState

| Value | Name | Description |
|---|---|---|
| 0 | `UNDEFINED` | Unknown/invalid state |
| 1 | `IDLE` | Motor disarmed, no action |
| 2 | `STARTUP_SEQUENCE` | Running configured startup sequence |
| 3 | `FULL_CALIBRATION_SEQUENCE` | Motor cal + encoder offset cal |
| 4 | `MOTOR_CALIBRATION` | Motor resistance/inductance measurement only |
| 6 | `ENCODER_INDEX_SEARCH` | Searching for encoder index pulse |
| 7 | `ENCODER_OFFSET_CALIBRATION` | Measuring encoder-to-motor phase offset |
| 8 | `CLOSED_LOOP_CONTROL` | Active motor control |
| 9 | `LOCKIN_SPIN` | Open-loop spinning at fixed current |
| 10 | `ENCODER_DIR_FIND` | Finding encoder direction |
| 11 | `HOMING` | Running homing sequence |
| 12 | `ENCODER_HALL_POLARITY_CALIBRATION` | Calibrating Hall sensor polarity |
| 13 | `ENCODER_HALL_PHASE_CALIBRATION` | Calibrating Hall sensor phase |

### ControlMode

| Value | Name | Description |
|---|---|---|
| 0 | `VOLTAGE_CONTROL` | Direct voltage control (open loop) |
| 1 | `TORQUE_CONTROL` | Torque (current) control. Set `input_torque` in Nm |
| 2 | `VELOCITY_CONTROL` | Velocity control. Set `input_vel` in turn/s |
| 3 | `POSITION_CONTROL` | Position control (default). Set `input_pos` in turns |

### InputMode

| Value | Name | Description |
|---|---|---|
| 0 | `INACTIVE` | Inputs ignored |
| 1 | `PASSTHROUGH` | Direct passthrough (default) |
| 2 | `VEL_RAMP` | Velocity ramping. Uses `vel_ramp_rate` |
| 3 | `POS_FILTER` | 2nd-order position filter. Uses `input_filter_bandwidth` |
| 4 | `MIX_CHANNELS` | Mix multiple input channels |
| 5 | `TRAP_TRAJ` | Trapezoidal trajectory planner. Uses `trap_traj.config` |
| 6 | `TORQUE_RAMP` | Torque ramping. Uses `torque_ramp_rate` |
| 7 | `MIRROR` | Mirror another axis. Uses `axis_to_mirror`, `mirror_ratio` |
| 8 | `TUNING` | Autotuning mode |

### MotorType

| Value | Name | Description |
|---|---|---|
| 0 | `HIGH_CURRENT` | Standard BLDC/PMSM (default). FOC with current sensing |
| 2 | `GIMBAL` | Gimbal motor. Open-loop voltage control, no current sensing |
| 3 | `ACIM` | AC induction motor |

### EncoderMode

| Value | Name | Description |
|---|---|---|
| 0 | `INCREMENTAL` | Quadrature incremental encoder (default) |
| 1 | `HALL` | Hall effect sensors (3-phase) |
| 2 | `SINCOS` | Sin/cos analog encoder |
| 256 | `SPI_ABS_CUI` | CUI absolute encoder via SPI |
| 257 | `SPI_ABS_AMS` | AMS absolute encoder via SPI (AS5047, AS5048) |
| 258 | `SPI_ABS_AEAT` | Broadcom AEAT absolute encoder via SPI |
| 259 | `SPI_ABS_RLS` | RLS absolute encoder via SPI |
| 260 | `SPI_ABS_MA732` | MPS MA732 absolute encoder via SPI |

### GpioMode

| Value | Name | Description |
|---|---|---|
| 0 | `DIGITAL` | Digital input (floating) |
| 1 | `DIGITAL_PULL_UP` | Digital input with pull-up |
| 2 | `DIGITAL_PULL_DOWN` | Digital input with pull-down |
| 3 | `ANALOG_IN` | Analog input (ADC) |
| 4 | `UART_A` | UART A function |
| 5 | `UART_B` | UART B function |
| 6 | `UART_C` | UART C function |
| 7 | `CAN_A` | CAN bus function |
| 8 | `I2C_A` | I2C function |
| 9 | `SPI_A` | SPI function |
| 10 | `PWM` | PWM input |
| 11 | `ENC0` | Encoder 0 (axis0) |
| 12 | `ENC1` | Encoder 1 (axis1) |
| 13 | `ENC2` | Encoder 2 |
| 14 | `MECH_BRAKE` | Mechanical brake output |
| 15 | `STATUS` | Status output |

### StreamProtocolType

| Value | Name | Description |
|---|---|---|
| 0 | `FIBRE` | Fibre protocol (native USB) |
| 1 | `ASCII` | ASCII protocol (human-readable) |
| 2 | `STDOUT` | Standard output only |
| 3 | `ASCII_AND_STDOUT` | ASCII protocol + stdout (default) |

---

## Error Bitfields

### ODriveError &mdash; `odrv0.error`

| Bit | Hex | Name | Description | Fix |
|---|---|---|---|---|
| 0 | `0x01` | `CONTROL_ITERATION_MISSED` | Control loop took too long | Reboot. If persistent, may indicate FW/HW issue |
| 1 | `0x02` | `DC_BUS_UNDER_VOLTAGE` | Vbus below undervoltage trip level | Check power supply. Increase dc_bus_undervoltage_trip_level if supply sags under load |
| 2 | `0x04` | `DC_BUS_OVER_VOLTAGE` | Vbus above overvoltage trip level | Enable brake resistor. Reduce decel rate. Check dc_bus_overvoltage_trip_level |
| 3 | `0x08` | `DC_BUS_OVER_REGEN_CURRENT` | Regen current exceeded limit | Enable brake resistor. Increase dc_max_negative_current (more negative). Reduce decel rate |
| 4 | `0x10` | `DC_BUS_OVER_CURRENT` | Bus current exceeded limit | Reduce load. Check for shorts. Reduce current_lim |
| 5 | `0x20` | `BRAKE_DEADTIME_VIOLATION` | Brake resistor timing error | Check brake_resistance value. Reboot |
| 6 | `0x40` | `BRAKE_DUTY_CYCLE_NAN` | Brake duty cycle computed as NaN | Reboot. Check config values for NaN |
| 7 | `0x80` | `INVALID_BRAKE_RESISTANCE` | Brake resistance value invalid | Set odrv0.config.brake_resistance to match physical resistor (e.g. 2.0 for 2 ohm) |

### CanError &mdash; `odrv0.can.error`

| Bit | Hex | Name | Description | Fix |
|---|---|---|---|---|
| 0 | `0x01` | `DUPLICATE_CAN_IDS` | Two axes have the same CAN node_id | Set unique node_id for each axis |

### AxisError &mdash; `<axis>.error`

| Bit | Hex | Name | Description | Fix |
|---|---|---|---|---|
| 0 | `0x000001` | `INVALID_STATE` | Requested invalid state transition | Clear errors, ensure axis is IDLE before commanding new state |
| 6 | `0x000040` | `MOTOR_FAILED` | Motor subsystem error (check motor.error) | Read motor.error for root cause |
| 7 | `0x000080` | `SENSORLESS_ESTIMATOR_FAILED` | Sensorless estimator error | Check sensorless_estimator.error |
| 8 | `0x000100` | `ENCODER_FAILED` | Encoder subsystem error (check encoder.error) | Read encoder.error for root cause |
| 9 | `0x000200` | `CONTROLLER_FAILED` | Controller subsystem error | Read controller.error for root cause |
| 11 | `0x000800` | `WATCHDOG_TIMER_EXPIRED` | No watchdog feed received in time | Feed watchdog or disable watchdog |
| 12 | `0x001000` | `MIN_ENDSTOP_PRESSED` | Min endstop triggered | Release endstop or disable endstop |
| 13 | `0x002000` | `MAX_ENDSTOP_PRESSED` | Max endstop triggered | Release endstop or disable endstop |
| 14 | `0x004000` | `ESTOP_REQUESTED` | Emergency stop requested | Clear errors to resume |
| 17 | `0x020000` | `HOMING_WITHOUT_ENDSTOP` | Homing attempted without endstop configured | Configure an endstop before homing |
| 18 | `0x040000` | `OVER_TEMP` | Axis over-temperature | Let ODrive cool. Check airflow and FET thermistor limits |
| 19 | `0x080000` | `UNKNOWN_POSITION` | Position not known (abs encoder not ready) | Check encoder is_ready. Ensure encoder configured and calibrated |

### MotorError &mdash; `<axis>.motor.error`

| Bit | Hex | Name | Description | Fix |
|---|---|---|---|---|
| 0 | `0x00000001` | `PHASE_RESISTANCE_OUT_OF_RANGE` | Resistance measurement failed | Check motor wiring for shorts/opens. Increase resistance_calib_max_voltage (try 4-8 V). Verify pole_pairs |
| 1 | `0x00000002` | `PHASE_INDUCTANCE_OUT_OF_RANGE` | Inductance measurement failed | Check motor wiring. May need to increase calibration_current for high-inductance motors |
| 3 | `0x00000008` | `DRV_FAULT` | DRV8301/DRV8323 gate driver fault | Power off immediately. Check for phase-to-phase shorts. Let cool. Reduce current_lim |
| 4 | `0x00000010` | `CONTROL_DEADLINE_MISSED` | Control loop overran | Reboot. If persistent, firmware/hardware issue |
| 7 | `0x00000080` | `MODULATION_MAGNITUDE` | PWM modulation too high (back-EMF vs Vbus) | Reduce vel_limit. Increase bus voltage |
| 10 | `0x00000400` | `CURRENT_SENSE_SATURATION` | ADC current sense saturated | Reduce load. Check for wiring shorts. Reduce current_lim |
| 12 | `0x00001000` | `CURRENT_LIMIT_VIOLATION` | Motor current exceeded current_lim | Reduce velocity/load. Increase current_lim if motor can handle it |
| 16 | `0x00010000` | `MODULATION_IS_NAN` | Numerical instability | Reduce gains (vel_gain, vel_integrator_gain). Clear errors and recalibrate |
| 17 | `0x00020000` | `MOTOR_THERMISTOR_OVER_TEMP` | Motor thermistor over temp | Let motor cool. Check thermistor wiring. Reduce current_lim |
| 18 | `0x00040000` | `FET_THERMISTOR_OVER_TEMP` | ODrive FETs over temp | Let ODrive cool. Improve airflow. Reduce current_lim |
| 19 | `0x00080000` | `TIMER_UPDATE_MISSED` | Timer slip | Reboot |
| 20 | `0x00100000` | `CURRENT_MEASUREMENT_UNAVAILABLE` | ADC data not ready | Reboot. Check for power supply noise |
| 21 | `0x00200000` | `CONTROLLER_FAILED` | Motor controller subsystem failed | Check controller.error for root cause (OVERSPEED, SPINOUT_DETECTED, etc.) |
| 22 | `0x00400000` | `I_BUS_OUT_OF_RANGE` | Bus current out of range | Check bus current draw. Reduce load or current_lim |
| 23 | `0x00800000` | `BRAKE_RESISTOR_DISARMED` | Brake resistor not enabled | Set enable_brake_resistor = True, save_configuration() |
| 24 | `0x01000000` | `SYSTEM_LEVEL` | System-level fault (cascading) | Most common: brake resistor not enabled. Enable brake resistor, clear errors, recalibrate |
| 25 | `0x02000000` | `BAD_TIMING` | Phase estimates at wrong time | Reboot |
| 26 | `0x04000000` | `UNKNOWN_PHASE_ESTIMATE` | Phase angle unavailable | Ensure encoder calibrated and is_ready = True before closed loop |
| 27 | `0x08000000` | `UNKNOWN_PHASE_VEL` | Phase velocity unavailable | Check encoder is_ready. Ensure pre_calibrated = True if saved |
| 28 | `0x10000000` | `UNKNOWN_TORQUE` | Torque estimate unavailable | Run full calibration. Ensure torque_constant and pole_pairs set |
| 29 | `0x20000000` | `UNKNOWN_CURRENT_COMMAND` | Current setpoint unavailable | Check controller mode and input_mode are set correctly |
| 30 | `0x40000000` | `UNKNOWN_CURRENT_MEASUREMENT` | Current measurement unavailable | Reboot. Check for power supply noise |
| 31 | `0x80000000` | `UNKNOWN_VBUS_VOLTAGE` | Vbus measurement unavailable | Check power supply. Enable brake resistor to prevent vbus spikes |
| 32 | `0x100000000` | `UNKNOWN_VOLTAGE_COMMAND` | Voltage command unavailable | Ensure motor is calibrated and controller is properly configured |
| 33 | `0x200000000` | `UNKNOWN_GAINS` | Control gains not set | Run motor calibration to auto-compute gains |
| 34 | `0x400000000` | `CONTROLLER_INITIALIZING` | Controller still starting up | Wait for initialization to complete. Clear errors if stuck |
| 35 | `0x800000000` | `UNBALANCED_PHASES` | Phase currents not balanced | Check motor wiring for open/shorted phase. Verify motor is healthy |

### ControllerError &mdash; `<axis>.controller.error`

| Bit | Hex | Name | Description | Fix |
|---|---|---|---|---|
| 0 | `0x01` | `OVERSPEED` | Motor exceeded vel_limit | Reduce velocity setpoint or increase vel_limit |
| 1 | `0x02` | `INVALID_INPUT_MODE` | Invalid input_mode for current control_mode | Check input_mode and control_mode compatibility |
| 2 | `0x04` | `UNSTABLE_GAIN` | Gains are unstable | Reduce vel_gain or vel_integrator_gain |
| 3 | `0x08` | `INVALID_MIRROR_AXIS` | Invalid mirror axis configuration | Check axis_to_mirror setting |
| 4 | `0x10` | `INVALID_LOAD_ENCODER` | Invalid load encoder axis | Check load_encoder_axis setting |
| 5 | `0x20` | `INVALID_ESTIMATE` | Position/velocity estimate invalid | Check encoder is_ready and calibration |
| 6 | `0x40` | `INVALID_CIRCULAR_RANGE` | Invalid circular range config | Check circular_setpoint_range and steps_per_circular_range |
| 7 | `0x80` | `SPINOUT_DETECTED` | Electrical and mechanical power disagree | Check motor wiring direction, encoder direction, or reduce spinout thresholds |

### EncoderError &mdash; `<axis>.encoder.error`

| Bit | Hex | Name | Description | Fix |
|---|---|---|---|---|
| 0 | `0x001` | `UNSTABLE_GAIN` | Position estimate diverging | Check motor wiring. Verify CPR. Check encoder power supply |
| 1 | `0x002` | `CPR_POLEPAIRS_MISMATCH` | CPR and pole_pairs inconsistent | Verify CPR = encoder resolution (e.g. 8192 for AS5047 in 13-bit, 16384 for 14-bit). Verify pole_pairs |
| 2 | `0x004` | `NO_RESPONSE` | SPI/communication failure | Check encoder wiring (SPI: SCK, MISO, MOSI, CS). Check CS GPIO pin. Verify 3.3V power |
| 3 | `0x008` | `UNSUPPORTED_ENCODER_MODE` | Encoder mode not supported | Use a supported encoder mode for this firmware |
| 4 | `0x010` | `ILLEGAL_HALL_STATE` | Illegal Hall sensor state | Check Hall sensor wiring and power. Set ignore_illegal_hall_state if intermittent |
| 5 | `0x020` | `INDEX_NOT_FOUND_YET` | Index pulse not found | Run encoder index search first, or use absolute encoder |
| 6 | `0x040` | `ABS_SPI_TIMEOUT` | SPI timed out | Check SPI wiring. Verify CS GPIO pin. Check encoder power |
| 7 | `0x080` | `ABS_SPI_COM_FAIL` | SPI communication failed (bad data/CRC) | Check SPI wiring for noise. Shorten cables. Verify encoder part |
| 8 | `0x100` | `ABS_SPI_NOT_READY` | Absolute encoder still initializing | Wait after power-on. Check encoder power supply ramp time |
| 9 | `0x200` | `HALL_NOT_CALIBRATED_YET` | Hall sensors not calibrated | Run encoder offset calibration. Not relevant for SPI ABS |

### SensorlessEstimatorError &mdash; `<axis>.sensorless_estimator.error`

| Bit | Hex | Name | Description | Fix |
|---|---|---|---|---|
| 0 | `0x01` | `UNSTABLE_GAIN` | Sensorless observer gain unstable | Reduce observer_gain. Check pm_flux_linkage value |
| 1 | `0x02` | `UNKNOWN_CURRENT_MEASUREMENT` | Current measurement not available | Ensure motor calibration complete. Check ADC/power supply |

---

## Quick Reference: Common Workflows

### State Transitions

```
IDLE (1) --> FULL_CALIBRATION_SEQUENCE (3)  -- motor cal + encoder offset cal
IDLE (1) --> MOTOR_CALIBRATION (4)          -- motor cal only
IDLE (1) --> ENCODER_OFFSET_CALIBRATION (7) -- encoder cal only (motor must be pre_calibrated)
IDLE (1) --> CLOSED_LOOP_CONTROL (8)        -- active control (both cals must be done)
any state --> IDLE (1)                      -- safe stop
```

**Usage:** `odrv0.axis0.requested_state = <value>`

### Control Modes

```
VELOCITY_CONTROL (2):  input_vel  [turn/s]
POSITION_CONTROL (3):  input_pos  [turns]
TORQUE_CONTROL   (1):  input_torque [Nm]
```

### Units

| Quantity | Unit | Notes |
|---|---|---|
| Position | turns | 1 turn = 360 degrees |
| Velocity | turn/s | |
| Torque | Nm | = torque_constant * Iq |
| Current | A | |
| Voltage | V | |
| Temperature | C | |
| Resistance | Ohms | |
| Inductance | H | |
| Bandwidth | Hz | |
| Flux linkage | Wb | |

### Torque Constant

```
torque_constant = 8.27 / KV
```

Where KV is in rpm/V. For Maytech MTO5065-70-HA-C (KV=270): `torque_constant = 8.27 / 270 = 0.0306`

---

*Generated 2026-04-12 from probe_results.txt + ODriveEnums.h + ODrive 0.5.6 docs + odrive_gui_v2.py error hint tables.*
