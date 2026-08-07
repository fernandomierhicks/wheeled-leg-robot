# GUI — Claude Context

PyQt6 desktop app. Entry point: `main.py` → `MainWindow` → `QTabWidget`. All tab/module implementations live under `tabs/`; only `main.py` stays at the `gui/` root, plus standalone offline scripts under `tools/`.

Import convention: `main.py` imports modules as `from tabs.module import X`. Modules inside `tabs/` import each other with **relative** imports (`from .theme import BG`), including lazy in-function imports — don't forget the leading dot there too (grep for `from \w+ import` if adding cross-module imports to catch ones that would silently break).

## Module map

Core singletons / infra (no UI):

| File | Role |
|---|---|
| `tabs/theme.py` | All colors (`BG`, `TEXT`, `GREEN`, …) and `APP_STYLE` QSS. Import from here, never hardcode colors. |
| `tabs/telemetry_bus.py` | `TelemetryBus` singleton — `pyqtSignal(dict)`. Every decoded TELEM packet (live or replayed) is emitted here; tabs subscribe to `TelemetryBus.instance().packet`. **`packet` is coalesced to 20 Hz** (latest-sample-wins, `_flush_live`) so UI work doesn't back up the event queue — it drops ~2 of every 3 packets from the 50 Hz stream. Anything *measuring* the stream (rates, inter-packet timing, logging) must use the uncoalesced `live_packet` instead; `packet` will report the coalescer's period, not the robot's. `live_packet` is live-only — playback bypasses it. |
| `tabs/port_manager.py` | `SerialPortManager` singleton — owns all open serial ports. VID/PID auto-detect in `SIGNATURES`. Use `acquire`/`release` so Flash and Monitor don't collide. |
| `tabs/source_manager.py` | `SourceManager` singleton — tracks which devices (`teensy`/`esp32`/`wifi`) are connected and arbitrates which one is the "active" telemetry source (priority order in `PRIORITY`, or a user override from the status-bar source picker). |
| `tabs/wifi_transport.py` | `WifiTransport` — UDP telemetry receiver (port 5005) + TCP command sender for ESP32 WiFi mode. Feeds the same `PacketDecoder`/`TelemetryBus`/`LogPacketBus` path as USB serial. |
| `tabs/comm_commands.py` | CommLink COMMAND frame builders + senders (`shared/comm_protocol.h`). Sends over WiFi TCP and/or USB serial. Used by any tab that issues commands (mode changes, hip/wheel motor commands, param set, log control, ...). |
| `tabs/telem_format.py` | Single source of truth for the `TelemetryPayload` wire layout (currently `TELEM_VERSION` 12). Shared, non-Qt — used by both the live decoder (`flash_monitor.py`) and offline `.wlog` decoding (`log_playback.py`, `tools/wlog_to_csv.py`). Keep in sync with the PROPAGATION CHECKLIST in `shared/comm_protocol.h`. |
| `tabs/log_bus.py` | `LogPacketBus` singleton — every decoded LOG_INFO/LOG_DATA packet (ptype 0x12/0x13), from whichever transport, is emitted here so `LogTransferManager` doesn't need to know about individual `PacketDecoder`s. |
| `tabs/log_transfer.py` | `LogTransferManager` — SD-log directory listing + reliable chunked download, transport-agnostic (USB serial or WiFi). Run bundles live under the ignored `data/logs/runs/` tree. |
| `tabs/host_logger.py` | `HostLogger` singleton — captures every `TelemetryBus` packet (TELEM, robot LOG lines, CALIB, WIFI_DIAG) to a timestamped `.jsonl`, independent of SD logging. Manual start/stop (Logs tab, or `host_log_start`/`host_log_stop` below). The only way to see firmware `comm_log()` lines and TELEM-arrival timing from outside the GUI process — see "Remote control" below. |
| `tabs/remote_control.py` | `RemoteControlServer` — local-only `QTcpServer` on `127.0.0.1:8765`, one JSON command per connection. Starts automatically with the GUI; this is what `tools/robot_ctl.py` talks to. See "Remote control / automation" below. |
| `tabs/operator_bridge.py` | `GuiOperatorBridge` — backs `ui_manifest`/`ui_snapshot`/`ui_invoke`/`ui_screenshot`: lets a remote client enumerate and drive real Qt widgets (buttons, tabs) by id, not just protocol commands. |

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
| `tools/trigger_log_test.py` | Standalone (pyserial only, no GUI) helper to start a timed SD log directly over USB serial — bypasses the GUI entirely. Prefer `robot_ctl.py` (below) instead when the GUI is already running: it drives the real connection instead of mimicking the wire protocol standalone. |
| `tools/robot_ctl.py` | **The primary way to drive a running GUI headlessly/remotely** (from a shell, or an agent). CLI client for `remote_control.py`'s TCP server. See "Remote control / automation" below. |

## Remote control / automation (driving a live GUI headlessly)

The GUI exposes a local command server (`tabs/remote_control.py`, `127.0.0.1:8765`, starts automatically whenever the GUI process is up) so a shell or an agent can drive the *real* connection — real serial ports, real WiFi session — instead of reimplementing the wire protocol. Always prefer this over writing a standalone script that mimics the protocol (`trigger_log_test.py` is a historical exception, kept for when the GUI isn't running at all).

```
python software/gui/tools/robot_ctl.py <cmd> [args...]
```

Full command list is in the script's module docstring; the frequently-useful ones:

| Command | Notes |
|---|---|
| `health` / `service_status` | Connection state, detected COM ports, WiFi status, `robot.state`/`robot.fault`/`telemetry_age_ms`. Start here. |
| `telem` | One decoded telemetry snapshot (`timestamp_ms`, `loop_count`, `robot_state`, ...). `timestamp_ms` resetting to a small value confirms a reboot actually happened. |
| `param_get` / `param_set <id_or_name> <value>` | `param_set` needs a lease (auto-acquired). Persistent params (`PARAM_FLAG_PERSISTENT`) survive reboot; check the param table before assuming a value sticks. |
| `set_mode <STATE>` | `STARTUP` from `ESTOP` is how you request the firmware's soft reset/re-init (`stateMachine_request_reset()`) — most fault latches need this, not a physical reboot, to clear. |
| `log_start [duration_ms]` / `log_stop` / `log_list` / `log_download <idx>` | SD logging on the robot. `duration_ms=0` logs until `log_stop`. |
| `host_log_start` / `host_log_stop` / `host_log_status` | GUI-side capture (`HostLogger`) of *everything* the GUI receives — including firmware `comm_log()` LOG frames — to `data/logs/runs/<ts>_HOST/host.jsonl`, independent of SD logging. **This is how you see firmware log lines and measure real inter-packet timing from a headless shell** — parse the `.jsonl`, filter `type_name == "LOG"` for `log_msg` (e.g. `Loop overrun: ...` warnings), or diff consecutive `TELEM` records' `host_monotonic_ns` to detect a stalled control loop. |
| `firmware_flash <teensy\|esp32>` | Builds + uploads via the GUI's own PlatformIO panel. **Often needs to be invoked twice** — see the firmware README's Flashing section for why and for the stuck-process recovery step. |
| `ui_manifest` / `ui_snapshot [query]` / `ui_invoke <id> <action> [json_value]` / `ui_screenshot` | Drive arbitrary Qt widgets by id (buttons, tabs) when there's no dedicated protocol command — e.g. the status-bar Reboot button has no `robot_ctl.py` verb of its own. `ui_manifest` lists every widget id; grep its JSON for the widget you need. |
| `service_start` / `service_restart` | Launches the GUI itself (headless-friendly, `pythonw.exe`) if it isn't already running, and waits for the command server to come up. Use this instead of assuming a session is live. |

**Gotcha — the Reboot button opens a native confirmation dialog that blocks the command server.** `ui_invoke window/qpushbutton-6-reboot click` triggers `main.py`'s `_on_reboot_clicked()`, which calls `QMessageBox.question(...)` — a nested Qt event loop that doesn't return (and so doesn't send the TCP response) until the dialog is dismissed. The default button is **No**, so blindly sending Enter cancels the reboot silently, not confirms it. `robot_ctl.py` will itself report a connect timeout while the dialog is open (the server is mid-handler, single connection at a time by contract). To actually confirm it from outside the GUI: activate the `Reboot Teensy` window (e.g. `wscript.shell`'s `AppActivate` in PowerShell) and send the literal **Yes** action — not a bare `{ENTER}`, which lands on No. Verify the reboot actually happened afterward via `telem`'s `timestamp_ms` resetting to a small value, not just that the click "succeeded" — a swallowed No looks identical to a successful no-op from the caller's side.

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
