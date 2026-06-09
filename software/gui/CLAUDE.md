# GUI — Claude Context

PyQt6 desktop app. Entry point: `main.py` → `MainWindow` → `QTabWidget`.

## Module map

| File | Role |
|---|---|
| `main.py` | Window, tab assembly, `StatusBar`. Kills duplicate instances on launch. |
| `theme.py` | All colors (`BG`, `TEXT`, `GREEN`, …) and `APP_STYLE` QSS. Import from here, never hardcode colors. |
| `telemetry_bus.py` | `TelemetryBus` singleton — `pyqtSignal(dict)`. Any decoded telemetry packet is emitted here; tabs subscribe to `TelemetryBus.instance().packet`. |
| `port_manager.py` | `SerialPortManager` singleton — owns all open serial ports. VID/PID auto-detect in `SIGNATURES`. Use `acquire`/`release` so Flash and Monitor don't collide. |
| `flash_monitor.py` | "Flash & Monitor" tab. Runs `pio` via `QProcess`. Paths: `firmware/robot_teensy/teensy` and `.../esp32`. Serial monitor streams to `logs/teensy.log` / `logs/esp32.log`. |
| `imu_tab.py` | Live pyqtgraph charts + 3-D orientation cube (pyqtgraph.opengl). Subscribes to `TelemetryBus`. |
| `hip_motors.py` | AK45-10 MIT CAN mode controls + live torque/position charts. Send path: GUI → `SerialPortManager` → Teensy → CAN. cmd_id schema is **TODO** in firmware. |
| `raw_data_tab.py` | Dumps every `TelemetryBus` packet as coloured text + packet rate counter. |

## Data flow

```
Teensy serial → (flash_monitor reader thread) → TelemetryBus.packet signal
                                                        ↓
                                           imu_tab / hip_motors / raw_data_tab
```

Telemetry fields are defined in `firmware/robot_teensy/shared/comm_protocol.h` (`TelemetryPayload`).

The firmware this GUI talks to lives in `firmware/robot_teensy/esp32`. The Teensy side (`firmware/robot_teensy/teensy`) runs the control loop; the ESP32 is the WiFi/serial bridge the GUI connects to.

## Adding a new tab

1. Create `my_tab.py`, subclass `QWidget`.
2. Subscribe to `TelemetryBus.instance().packet` if you need telemetry.
3. Use `SerialPortManager.instance().acquire("device", port, baud)` for serial writes.
4. Register in `main.py` `MainWindow.__init__`.

## Placeholder tabs

`DashboardTab` and `WheelMotorsTab` in `main.py` are stubs — just centered labels.
