# Unified ODrive GUI — Consolidation & Rewrite Plan

## Context

The project has **9 standalone Python scripts** and **2 GUIs** (a PySide6 desktop app and a Flask+React web app) for controlling an ODrive 3.6 motor controller (fw v0.5.6). The desktop GUI (v2, ~3100 lines) became buggy. The web GUI is feature-rich but over-engineered for a single-user local tool (4 processes: Flask + React + Vite + Node.js). Goal: consolidate everything into one clean, incrementally-built GUI that Claude can also interact with programmatically.

---

## Technology Choice: Pure Python Desktop (PySide6 + pyqtgraph)

**Why not web-based:**
- Single user on Windows — no need for Flask/React/Node.js overhead
- Claude automation needs to read output and send commands — a file-based inbox + log is simpler than HTTP
- pyqtgraph gives 60+ FPS real-time plots (proven in [fastchart.py](misc/fastchart.py))
- The existing v2 GUI is 3100 lines of working PySide6 — patterns are established

**What to port from the web GUI (architecture, not code):**
- `ODriveManager` as a central thread-safe device controller class
- Command normalization (`odrv0` -> device reference)
- USB error recovery with retry logic
- Factory presets concept (JSON config snapshots)
- Property tree inspector idea

---

## Architecture

```
                    odrive_gui.py (main window, poll timer, tabs)
                           |
          +----------------+----------------+
          |                |                |
    tabs/*.py        core/*.py         cmd/cmd_interface.py
    (UI only)        (logic only)      (Claude automation)
```

**Core principle: no code duplication.** Every ODrive operation (clear errors, enter idle, calibrate, etc.) is defined exactly once in `core/`, and all tabs + Claude commands call those shared functions.

### File Structure

```
components/characterization/odrive_unified/
    odrive_gui.py                   -- Entry point + MainWindow
    core/
        odrive_manager.py           -- Connect, disconnect, safe get/set, execute, reconnect
        odrive_errors.py            -- Error tables + decode + hint text
        odrive_operations.py        -- High-level workflows (calibrate, vel test, anticogging, etc.)
        odrive_presets.py           -- Factory presets, config import/export (JSON)
        constants.py                -- Enums, axis states, default values
    cmd/
        cmd_interface.py            -- File-based inbox/outbox for Claude automation
    tabs/
        tab_setup.py                -- Motor/encoder config, calibration
        tab_control.py              -- Velocity/position/torque, gains, enable/disable
        tab_anticogging.py          -- Full anticogging workflow (consolidated from 3 scripts)
        tab_inspector.py            -- Property tree browser + firmware probe
        tab_terminal.py             -- REPL + Claude command inbox
        tab_charts.py               -- Live pyqtgraph telemetry plots
    ui/
        widgets.py                  -- Shared widgets (spinbox, status label, etc.)
        theme.py                    -- Dark theme stylesheet
```

---

## Script Consolidation Map

| Standalone Script | Where It Goes | How |
|---|---|---|
| `test_connect.py` | Eliminated | Connect button replaces it |
| `probe_firmware.py` | `tab_inspector.py` | Property tree walk becomes "Probe" button |
| `vel_test.py` | `core/odrive_operations.py` | `velocity_smoke_test()` function, called from Control tab |
| `probe_anticogging.py` | `core/odrive_operations.py` | `anticogging_diagnostic()` function |
| `run_anticogging.py` | `tab_anticogging.py` + `core/odrive_operations.py` | Full workflow with step buttons |
| `check_anticog_nvm.py` | `core/odrive_operations.py` | `anticogging_quality_test()` function |
| `fix_and_spin.py` | `core/odrive_operations.py` | `fix_and_spin()` one-click function |
| `odesk/calibrate.py` | Subsumed by Setup tab | ODESC-specific v0.5.1 workarounds not needed |
| `odesk/read_calib_voltage.py` | Subsumed by Setup tab + Inspector | Already a form field |

---

## Incremental Build Steps

Each step produces a **runnable, testable GUI** (except Step 1.1 which produces a reference doc). We verify before moving to the next.

### Step 1.1: Firmware Probe + API Reference Document [COMPLETE]

**Output:** [odrive_api_reference.md](components/characterization/odrive/odrive_api_reference.md) — full device/axis/motor/encoder/controller parameter tables, all enums, and all error bitfields with fixes. Generated from probe_results.txt + ODriveEnums.h + v0.5.6 docs + v2 GUI hint tables.


**Goal:** Before writing any GUI code, create a definitive reference of every parameter, function, and enum available on our ODrive 3.6 fw 0.5.6. This becomes the source of truth for what the GUI can do.

**Process:**
1. Re-run `probe_firmware.py` against the connected ODrive to get a fresh attribute tree dump
2. Parse the 1023-line probe output + cross-reference with:
   - [ODriveEnums.h](components/characterization/odesk/webGUIfork/odrive_docs_local/odrive_local/docs.odriverobotics.com/v/0.5.6/_downloads/e6a0fd181b662855247fa773acf66f5e/ODriveEnums.h) — official enum definitions (error codes, states, modes)
   - [Local 0.5.6 docs](components/characterization/odesk/webGUIfork/odrive_docs_local/) — commands, control modes, encoder docs, etc.
   - ODrive 0.5.6 firmware source on GitHub (tag `fw-v0.5.6`, file `Firmware/odrive-interface.yaml`) for the formal interface spec
3. Produce `odrive_unified/odrive_api_reference.md` organized as:

```
## Device-Level
  ### Config (read/write)         — brake resistor, voltage limits, CAN, GPIO modes
  ### Status (read-only)          — vbus_voltage, serial_number, hw/fw version
  ### Functions                   — save_configuration(), erase_configuration(), reboot(), get_adc_voltage()

## Per-Axis (axis0/axis1)
  ### Config                      — startup flags, CAN node, watchdog, lockin, sensorless ramp
  ### Status                      — current_state, error, is_homed, last_drv_fault
  ### Functions                   — requested_state assignment

  ### Motor
    #### Config                   — motor_type, pole_pairs, current_lim, calibration_current, torque_constant, phase_R/L, etc.
    #### Status                   — is_armed, error, effective_current_lim, phase currents, I_bus
    #### Current Control (r/o)    — Iq_measured, Id_measured, Vd/Vq_setpoint, power
    #### Thermistors              — FET temp, motor temp (if present)

  ### Encoder
    #### Config                   — mode, cpr, abs_spi_cs_gpio_pin, pre_calibrated, bandwidth, use_index, etc.
    #### Status                   — pos_estimate, vel_estimate, is_ready, index_found, spi_error_rate, hall_state
    #### Functions                — set_linear_count()

  ### Controller
    #### Config                   — control_mode, input_mode, pos_gain, vel_gain, vel_integrator_gain, vel_limit, torque_lim, etc.
    #### Anticogging Config       — anticogging_enabled, pre_calibrated, calib_vel/pos_threshold, cogging_ratio, index
    #### Status                   — input_pos/vel/torque, pos/vel/torque_setpoint, error, electrical/mechanical_power
    #### Functions                — start_anticogging_calibration(), remove_anticogging_bias(), move_incremental(), get_anticogging_value()

  ### Trap Trajectory Config      — vel_limit, accel_limit, decel_limit

## Enums (from ODriveEnums.h)
  AxisState, ControlMode, InputMode, MotorType, EncoderMode, GpioMode
  Error bitfields: ODriveError, AxisError, MotorError, EncoderError, ControllerError, SensorlessEstimatorError
```

4. For each config parameter, note: **type**, **default value** (from probe), **range** (from docs), **writable?**
5. For each error bit, note: **hex value**, **description**, **common fix** (from v2 GUI hint tables)

**Source files to cross-reference:**
- [probe_results.txt](components/characterization/odrive_old/probe_results.txt) — ground truth from device
- [ODriveEnums.h](components/characterization/odesk/webGUIfork/odrive_docs_local/odrive_local/docs.odriverobotics.com/v/0.5.6/_downloads/e6a0fd181b662855247fa773acf66f5e/ODriveEnums.h) — official enums
- [commands.rst](components/characterization/odesk/webGUIfork/odrive_docs_local/odrive_local/docs.odriverobotics.com/v/0.5.6/_sources/commands.rst.txt) — command reference
- [control-modes.rst](components/characterization/odesk/webGUIfork/odrive_docs_local/odrive_local/docs.odriverobotics.com/v/0.5.6/_sources/control-modes.rst.txt) — control mode docs
- [Fibre type ref](components/characterization/odesk/webGUIfork/odrive_docs_local/odrive_local/docs.odriverobotics.com/v/0.5.6/_sources/fibre_types/com_odriverobotics_ODrive.rst.txt) — formal API spec
- [odrive_gui_v2.py](components/characterization/odrive_old/odrive_gui_v2.py) lines 73-236 — error hint tables
- ODrive GitHub `fw-v0.5.6` tag `Firmware/odrive-interface.yaml` (if accessible)

**Test:** The reference doc is complete enough that for any "what parameter controls X?" question, we can look it up without connecting to the ODrive.

### Step 1: Core + Connect + Status Bar
- `ODriveManager` class: connect, disconnect, safe_get/set, execute_command, thread lock
- `odrive_errors.py`: merged error tables from v2 GUI + vel_test.py
- `MainWindow`: axis selector, connect/disconnect button, status label, empty tab widget, 100ms poll timer
- Dark theme
- **Test:** Run GUI -> click Connect -> see hw/fw version + Vbus in status bar

### Step 2: Terminal Tab + Claude Command Interface
- Terminal with eval namespace (`odrv`, `ax0`, `ax1`)
- `cmd_interface.py`: polls `odrive_cmd_inbox.txt` every 500ms, writes results to `odrive_cmd_log.txt` + `odrive_cmd_output.json`
- Continuous `odrive_gui.log` captures all GUI events
- Meta-commands: `__CLEAR_ERRORS__`, `__IDLE__`, `__STOP__`, `__STATUS__`
- **Test:** Type `odrv.vbus_voltage` in terminal -> see result. Write command to inbox file -> see result in log

### Step 3: Motor Setup Tab
- Motor/encoder config form (from v2 `MotorSetupTab`)
- Flash config, reboot, reconnect, verify cycle
- Full calibration with progress polling
- Post-cal auto-tuning (compute vel_gain from phase_resistance)
- Startup flag safety (force all to False on connect)
- **Test:** Read config -> modify -> Flash -> reboot -> Verify. Run calibration -> motor beeps -> gains auto-set

### Step 4: Motor Control Tab
- Control mode selector (velocity/position/torque), setpoint, enable/disable
- Gains panel (pos_gain, vel_gain, vel_integrator_gain, current_bandwidth)
- Error display with decode + hint popups
- Live readout (state, pos, vel, Iq, Vbus)
- Quick buttons: Clear Errors, Stop Motor, Quick Spin Test
- **Test:** Enable closed loop -> command velocity -> motor spins -> live readouts update

### Step 5: Anticogging Tab
- Consolidates `run_anticogging.py` + `check_anticog_nvm.py` + `probe_anticogging.py`
- Step buttons: Prerequisites Check -> Velocity Smoke Test -> Start Calibration -> Quality Test -> Save
- Progress bar tracking sweep index
- Threshold presets (Good=5, Quick=100)
- **Test:** Full anticogging workflow from GUI with progress tracking

### Step 6: Inspector Tab
- Recursive property tree walk (from `probe_firmware.py` algorithm)
- QTreeWidget with search/filter
- Inline editing of writable properties
- Firmware version display
- **Test:** Browse full property tree, search for a property, edit it

### Step 7: Live Charts Tab
- pyqtgraph ring-buffer plots (based on [fastchart.py](misc/fastchart.py))
- Signals: Vbus, Iq_measured, vel_estimate, pos_estimate
- Toggle checkboxes, clear button, statistics (mean, stdev)
- **Test:** Connect ODrive -> see 4 live traces updating smoothly

### Step 8: Presets + Config Import/Export
- Save/load full ODrive config as JSON
- Factory preset for Maytech MTO5065-70-HA-C + AS5047
- "Restore factory defaults" button
- **Test:** Save config -> modify something -> load saved config -> values match

### Step 9 (optional): Characterization Tab
- Port KV test and friction sweep workers from v2 GUI
- Add as `tabs/tab_characterization.py`
- Only after Steps 1-8 are solid
- **Test:** Run KV test -> see live chart -> get KV/Kt values

---

## Claude Automation Loop

```
Claude writes to:    odrive_cmd_inbox.txt    (one command per line)
GUI reads, executes, writes to:
    odrive_cmd_log.txt       (human-readable, append-only)
    odrive_cmd_output.json   (structured JSON, last 100 results)
    odrive_gui.log           (full GUI log from all tabs, append-only)
```

**Auto-tuning example flow:**
1. Claude reads `odrive_gui.log`, sees calibration results (phase_R, phase_L)
2. Claude writes gain commands to inbox
3. GUI executes, logs results
4. Claude reads results, sends velocity test
5. If error -> Claude reads error decode from log -> adjusts -> retries

---

## Key Design Decisions

**Initialization safety (fixing old GUI bugs):**
- On connect: read all errors, display them (do NOT auto-clear)
- On connect: check startup flags, warn if any True, offer "Lock" button
- Never auto-move motor — all motion requires explicit user action
- After calibration: auto-set pre_calibrated, auto-tune gains, auto-save

**Thread model:**
- All ODrive access through `ODriveManager` with `threading.Lock`
- Long operations (calibration, vel test, anticogging) run in `QThread` with Signal callbacks
- 100ms poll timer reads telemetry on main thread (fast, try/except protected)

**Reconnection after reboot:**
- Set `_expecting_reboot` flag, wait 4s, retry connect 3x with 2s between (from web GUI pattern)

---

## Verification Plan

After each step:
1. Run `python odrive_gui.py` — GUI launches without errors
2. Connect to ODrive — status shows connected
3. Test the specific feature added in that step
4. Claude reads `odrive_gui.log` to verify events are logged
5. Claude sends a command via inbox and reads the result

Final integration test:
- Full cycle: Connect -> Setup -> Calibrate -> Control (spin motor) -> Anticogging -> Save -> Disconnect
- Claude automation: send 5 commands via inbox, verify all results in log
