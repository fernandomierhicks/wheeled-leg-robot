# ODESC V4.2 — Hardware & Bring-up Reference

> **Board:** MKS ODESC V4.2 (ODrive 3.6 clone)
> **Status:** Not yet probed — run `probe_firmware.py` before first config.

---

## DO NOT FLASH FIRMWARE

The ODESC V4.2 (and related MKS boards like the XDRIVE Mini) ships with a **modified v0.5.1 firmware**.

- **Do NOT run `odrivetool dfu`**, `odrv0.enter_dfu_mode()`, or the Web GUI firmware updater.
- Standard ODrive firmware upgrades **brick the device** (recovery requires an ST-Link V2 programmer).
- Updating to v0.5.6 reportedly causes **motor inactivity** due to different DRV8301 pin assignments in the stock ODrive firmware vs the MKS hardware.
- **Stick to v0.5.1** — it is the only stable firmware for this board.

Source: [justlovescience/MKS-XDRIVE-MINI](https://github.com/justlovescience/MKS-XDRIVE-MINI)

---

## MKS XDRIVE Mini / ODESC — Hardware Summary

(From the `justlovescience/MKS-XDRIVE-MINI` repo — applies to the ODESC V4.2 which uses the same MKS platform)

| Attribute | Value |
|---|---|
| Board family | MKS FOC driver (ODrive 3.6 clone) |
| Stock firmware | Modified ODrive v0.5.1 |
| Gate driver | DRV8301 (pin mapping differs from stock ODrive 3.6) |
| Dual axis | Yes — but axis1 is a "ghost" on single-motor setups (see CAN note below) |
| CAN bus | Supported; CAN node arbitration issues exist with ghost axis1 |
| Recovery method | ST-Link V2 programmer + dumped original firmware from the repo |

### CAN Bus — Ghost Axis1 Issue

The board exposes dual axes by default. When only one motor is connected, axis1 still participates on the CAN bus and causes conflicts. Fix:

```
odrv0.axis1.config.can_node_id = 63   # move ghost axis to listen-only ID
odrv0.save_configuration()
```

### Compatibility

| Platform | Status |
|---|---|
| Windows (odrivetool + Web GUI) | Full support |
| macOS (odrivetool only) | Works; GUI not supported |
| ESP32 (CAN/TWAI) | Requires `ODriveEsp32Twai.hpp` adapter in Arduino libs |

---

## ODrive 3.6 Web GUI

**Repo:** [MoonLighTingPY/odrive3.6_web_gui](https://github.com/MoonLighTingPY/odrive3.6_web_gui)

A modern web-based interface for ODrive v3.6 boards running firmware **v0.5.1 through v0.5.6**. It fills the gap left by the official ODrive GUI which only supports newer firmware.

### Features

- **6-Step Configuration Wizard:** Power → Motor → Encoder → Control → Interfaces → Apply
- **Live command preview** — shows exact ODrive commands before execution
- **Inspector** — browse and edit all ODrive parameters in real time
- **Live Charts** — plot any property (position, velocity, current, voltage) in real time
- **Dashboard** — quick motor/encoder calibration with step-by-step guidance
- **Command Console** — native `odrivetool`-style command interface with history
- **Presets** — save, load, and share motor configurations
- **Auto-Discovery** — automatic USB device scanning
- **Multi-device support** (experimental, not fully tested)

### Installation (Windows — Standalone, Recommended)

No Python or Node.js required:

1. Download latest release from GitHub Releases
2. Run `ODrive_GUI_Tray.exe`
3. Browser opens automatically at `http://localhost:3000`
4. Connect ODESC via USB — auto-discovers

### Installation (Development / Other OS)

Prerequisites: Python 3.8+, Node.js 16+

```bash
# Backend
pip install -r requirements.txt

# Frontend
npm install
npm run dev

# Access at http://localhost:3000
```

### 6-Step Wizard Workflow

| Step | What it configures |
|---|---|
| **1. Power** | `brake_resistance`, `enable_brake_resistor`, `dc_bus_overvoltage_trip_level`, `dc_bus_undervoltage_trip_level` |
| **2. Motor** | `motor_type`, `pole_pairs`, `calibration_current`, `current_lim`, `resistance_calib_max_voltage`, `torque_constant` |
| **3. Encoder** | `mode`, `cpr`, `abs_spi_cs_gpio_pin`, `use_index`, `pre_calibrated` |
| **4. Control** | `control_mode`, `input_mode`, `vel_limit`, `pos_gain`, `vel_gain`, `vel_integrator_gain` |
| **5. Interfaces** | CAN, UART, GPIO, step/dir configuration |
| **6. Apply** | Write config to NVM via `save_configuration()`, optional reboot |

---

## Config Schema — v0.5.1 vs v0.5.6 Differences

Since the ODESC V4.2 runs v0.5.1 while our existing ODrive 3.6 runs v0.5.6, key schema differences apply. Full table is in [`../odrive/odrive.md`](../odrive/odrive.md) § "What Changed vs. Old Dev Snapshot". The most important ones:

| Area | v0.5.1 (ODESC) | v0.5.6 (ODrive 3.6) |
|---|---|---|
| Per-axis CAN | `axis.config.can_node_id` (flat) | `axis.config.can.node_id` (sub-object) |
| Thermistors | `axis.fet_thermistor` | `axis.motor.fet_thermistor` |
| Encoder offset | `encoder.config.offset` | `encoder.config.phase_offset` |
| Motor armed | `motor.armed_state` (int) | `motor.is_armed` (bool) |
| Gate driver fault | `motor.gate_driver.drv_fault` | `axis.last_drv_fault` |
| Vel limit rename | `enable_current_mode_vel_limit` | `enable_torque_mode_vel_limit` |
| ASCII protocol | `config.enable_ascii_protocol_on_usb` | `config.usb_cdc_protocol` (int) |
| CAN enable | `config.enable_i2c_instead_of_can` | `config.enable_can_a`, `config.enable_i2c_a` |
| `clear_errors()` | Not present | Present |
| `move_to_pos()` | May be present | Gone — use `move_incremental()` |
| Brake resistor | `enable_brake_resistor` absent | Present |

**Always run `probe_firmware.py` first** — the actual schema on this modified firmware may differ from both stock 0.5.1 and 0.5.6.

---

## Planned Bring-up Procedure

### Hardware Setup

- **PSU:** 24 V bench supply, current-limited to **2 A** for first power-on (raise to 5 A after USB enumerates)
- **Motor:** Maytech MTO5065-70-HA-C (same wheel motor as ODrive 3.6 setup)
- **Encoder:** AS5047/AS5048 SPI absolute (14-bit, cpr=16384), CS on GPIO3 — see `AbsoluteEncoder.md` for pinout
- **Brake resistor:** 2 Ω (same as ODrive 3.6 setup)

### Step-by-step

1. **Power on, probe:** Connect USB, run `probe_firmware.py`, save to `probe_results_stock.txt`
2. **Web GUI wizard:** Launch Web GUI, run 6-step wizard with conservative values:
   - `calibration_current = 5 A`, `current_lim = 5 A`
   - `encoder.mode = 257` (SPI ABS AMS), `cpr = 16384`, `abs_spi_cs_gpio_pin = 3`
   - `control_mode = VELOCITY_CONTROL`, `vel_limit = 2.0 turns/s`
3. **Calibrate:** Run `AXIS_STATE_FULL_CALIBRATION_SEQUENCE` via GUI dashboard
4. **First spin:** Enter closed-loop, command `input_vel = 0.5`, verify smooth rotation
5. **Save:** `save_configuration()`, power cycle, verify config persists
6. **Snapshot:** Re-probe → `probe_results_configured.txt`, commit

---

## YouTube Reference

- [ODESC V4.2 setup video](https://www.youtube.com/watch?v=yRx7dsJmNvU) — covers board overview, wiring, and basic motor calibration workflow

---

## Tools

| File | Purpose |
|---|---|
| `probe_firmware.py` | Full recursive attribute dump (copy from `../odrive/`). Run before any config. |
| `probe_results_stock.txt` | Raw output from first probe (stock firmware, before config) |
| `probe_results_configured.txt` | Raw output after calibration and save |
