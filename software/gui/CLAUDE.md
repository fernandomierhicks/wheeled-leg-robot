# GUI — Claude Context

PyQt6 desktop app. Entry point: `main.py` → `MainWindow` → `QTabWidget`. All tab/module implementations live under `tabs/`; only `main.py` stays at the `gui/` root, plus standalone offline scripts under `tools/`.

Import convention: `main.py` imports modules as `from tabs.module import X`. Modules inside `tabs/` import each other with **relative** imports (`from .theme import BG`), including lazy in-function imports — don't forget the leading dot there too (grep for `from \w+ import` if adding cross-module imports to catch ones that would silently break).

## Module map

Core singletons / infra (no UI):

| File | Role |
|---|---|
| `tabs/theme.py` | All colors (`BG`, `TEXT`, `GREEN`, …) and `APP_STYLE` QSS. Import from here, never hardcode colors. |
| `tabs/telemetry_bus.py` | `TelemetryBus` singleton — `pyqtSignal(dict)`. Every decoded TELEM packet (live or replayed) is emitted here; tabs subscribe to `TelemetryBus.instance().packet`. |
| `tabs/port_manager.py` | `SerialPortManager` singleton — owns all open serial ports. VID/PID auto-detect in `SIGNATURES`. Use `acquire`/`release` so Flash and Monitor don't collide. |
| `tabs/source_manager.py` | `SourceManager` singleton — tracks which devices (`teensy`/`esp32`/`wifi`) are connected and arbitrates which one is the "active" telemetry source (priority order in `PRIORITY`, or a user override from the status-bar source picker). |
| `tabs/wifi_transport.py` | `WifiTransport` — UDP telemetry receiver (port 5005) + TCP command sender for ESP32 WiFi mode. Feeds the same `PacketDecoder`/`TelemetryBus`/`LogPacketBus` path as USB serial. |
| `tabs/comm_commands.py` | CommLink COMMAND frame builders + senders (`shared/comm_protocol.h`). Sends over WiFi TCP and/or USB serial. Used by any tab that issues commands (mode changes, hip/wheel motor commands, param set, log control, ...). |
| `tabs/telem_format.py` | Single source of truth for the `TelemetryPayload` wire layout (currently `TELEM_VERSION` 8). Shared, non-Qt — used by both the live decoder (`flash_monitor.py`) and offline `.wlog` decoding (`log_playback.py`, `tools/wlog_to_csv.py`). Keep in sync with the PROPAGATION CHECKLIST in `shared/comm_protocol.h`. |
| `tabs/log_bus.py` | `LogPacketBus` singleton — every decoded LOG_INFO/LOG_DATA packet (ptype 0x12/0x13), from whichever transport, is emitted here so `LogTransferManager` doesn't need to know about individual `PacketDecoder`s. |
| `tabs/log_transfer.py` | `LogTransferManager` — SD-log directory listing + reliable chunked download, transport-agnostic (USB serial or WiFi). `LOG_DIR` (default download folder) lives at `gui/logs/`. |

Tabs (registered in `main.py` `MainWindow.__init__`, in display order):

| File | Tab | Role |
|---|---|---|
| `tabs/robot_visualizer_tab.py` | Visualizer | Cockpit instrument panels + live 3-D scene (pyqtgraph.opengl) built from `TelemetryBus`. |
| `main.py` (`DashboardTab`) | Dashboard | Compact overview combining `ImuMiniWidget` + `TestValMiniWidget` + `TofMiniWidget` (all defined in `main.py`). |
| `tabs/imu_tab.py` | IMU | Live pyqtgraph charts + 3-D orientation cube. |
| `tabs/raw_data_tab.py` | Raw Data | Dumps every `TelemetryBus` packet as coloured text + packet rate counter. |
| `tabs/hip_motors.py` | Hip Motors | AK45-10 MIT CAN mode controls + live torque/position charts. Send path: GUI → `SerialPortManager`/WiFi → Teensy → CAN. |
| `tabs/params_tab.py` | Parameters | Firmware `ParamRegistry` browser/editor — `CMD_ID_PARAM_GET`/`PARAM_REPORT` round trip, grouped by subsystem, collapsible sections. |
| `tabs/wheel_motors.py` | Wheel Motors | ODrive wheel control (node 0 = L / node 1 = R): mode (IDLE/VELOCITY/POSITION/TORQUE), setpoints, test waveforms, error clearing, differential-drive spin slider. |
| `tabs/controllers_tab.py` | Controllers | LQR/balance controller live state: gains (currently hardcoded from `control_loop.cpp`, not sent in telemetry), health/saturation flags, rolling charts. |
| `tabs/radio_tab.py` | Radio | RC channel values vs. time. |
| `tabs/log_playback.py` | Logs | Two panels: retrieve (start/stop/timed SD logging, list/download/delete via `LogTransferManager`) and playback (`LogPlaybackController` singleton replays a `.wlog` onto `TelemetryBus.packet`, so every other tab animates unchanged; CSV export shells out to `tools/wlog_to_csv.py`). |
| `tabs/flash_monitor.py` | Flash & Monitor | Runs `pio` via `QProcess` for `firmware/robot_teensy/teensy` and `.../esp32`. Owns `PacketDecoder` (the actual TELEM/LOG frame decode+dispatch used by every transport). Serial monitor streams to `logs/teensy.log` / `logs/esp32.log`. |

Other:

| File | Role |
|---|---|
| `tabs/robot_log_widget.py` | `RobotLogWidget` — firmware log line viewer (comm_log messages), used in `main.py`'s bottom panel. |
| `tools/wlog_to_csv.py` | Standalone (no Qt) `.wlog` → CSV decoder, reused by the Logs tab's CSV export. |
| `tools/trigger_log_test.py` | Standalone (pyserial only, no GUI) helper to start a timed SD log directly over USB serial. |

## Data flow

```
Teensy/ESP32 USB serial ──┐
ESP32 UDP (WifiTransport) ─┼─→ PacketDecoder (flash_monitor.py) ──┬─→ TelemetryBus.packet   (TELEM frames)
                           │                                       └─→ LogPacketBus.packet   (LOG_INFO/LOG_DATA frames)
                           └─→ SourceManager (tracks connected/active device)
```

Only the `SourceManager`-active device's packets are emitted onto `TelemetryBus`/`LogPacketBus` — with Teensy USB, ESP32 USB, and WiFi all potentially connected simultaneously, every transport independently decodes, so duplicate suppression happens at the emit point, not the decode point.

`LogPlaybackController` (in `log_playback.py`) replays a downloaded `.wlog` onto `TelemetryBus.packet` exactly like a live TELEM packet — every telemetry-driven tab (Visualizer, IMU, Raw Data, Controllers, ...) works unchanged during playback; `TelemetryBus.playback_active` suppresses live packets while replaying.

Telemetry fields are defined in `firmware/robot_teensy/shared/comm_protocol.h` (`TelemetryPayload`).

The firmware this GUI talks to lives in `firmware/robot_teensy/esp32`. The Teensy side (`firmware/robot_teensy/teensy`) runs the control loop; the ESP32 is the WiFi/serial bridge the GUI connects to.

## Adding a new tab

1. Create `tabs/my_tab.py`, subclass `QWidget`.
2. Subscribe to `TelemetryBus.instance().packet` if you need telemetry.
3. Use `SerialPortManager.instance().acquire("device", port, baud)` for serial writes, or `comm_commands.py` senders (which already handle WiFi + serial) for commands.
4. Register in `main.py` `MainWindow.__init__` via `from tabs.my_tab import MyTab` + `tabs.addTab(MyTab(), "My Tab")`.
