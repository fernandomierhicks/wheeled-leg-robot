"""remote_control.py — local command server for Claude-driven hardware
tuning sessions (tuning.md Sec2). A QTcpServer bound to 127.0.0.1 only (no
new network exposure): one command per TCP connection — client connects,
writes one newline-terminated JSON line, server writes one newline-terminated
JSON response, closes. Simple, stateless, no session/reconnect logic.

Every command below calls existing, already-tested GUI code (comm_commands.py
senders, LogTransferManager, TelemetryBus) rather than reimplementing any
wire-protocol behavior — motion_set is the only genuinely new server-side
logic, and it's still just two param_set calls sequenced together.

Waiting for an async confirmation inside a synchronous-looking command
handler uses a nested QEventLoop (loop.exec() inside the readyRead-driven
handler, released by a QTimer timeout or the awaited signal) — the same safe
Qt pattern QDialog.exec() uses internally: the outer event loop keeps
servicing real serial I/O and UI while a handler "blocks". Every wait
disconnects its temporary signal listener in a finally block even on
timeout, mirroring ReliableCommand._finish()'s cleanup discipline.

Known limitation, accepted: this server handles one connection at a time by
contract, not by an enforced lock — fine since Claude is the only caller and
won't fire overlapping commands.
"""

import json
import math
import time
import uuid

from PyQt6.QtCore import QEventLoop, QObject, QTimer
from PyQt6.QtNetwork import QHostAddress, QTcpServer, QTcpSocket
from PyQt6.QtWidgets import QApplication

from .comm_commands import CommandResultBus, send_param_get, send_param_get_all, send_param_set
from .log_transfer import LogTransferManager
from .operator_bridge import GuiOperatorBridge
from .port_manager import SerialPortManager
from .source_manager import PRIORITY, SourceManager
from .telem_format import _STATE_NAMES
from .telemetry_bus import TelemetryBus

PORT = 8765

# GROUP_CONTROL params referenced directly by motion_set/motion_release
# (param_ids.h — tuning.md Sec1 d).
PARAM_GUI_MOTION_CTRL_EN = 0x042B
PARAM_V_CMD_MS           = 0x040B
PARAM_OMEGA_CMD_RDS      = 0x0411

_STATE_NAME_TO_ID = {name: pid for pid, name in _STATE_NAMES.items()}
_ACTIVE_STATE_IDS = {
    _STATE_NAME_TO_ID[name]
    for name in ("RUNNING", "MANUAL", "JUMPING", "STANDING_UP")
    if name in _STATE_NAME_TO_ID
}


class RemoteControlServer(QObject):
    """Owns the local QTcpServer. One instance held by MainWindow (main.py)
    for the app's lifetime — see win._remote_control there."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buffers: dict[QTcpSocket, bytearray] = {}
        self._param_cache: dict[int, dict] = {}
        self._latest_telem: dict = {}
        self._latest_telem_monotonic: float | None = None
        self._latest_command_result: dict = {}
        self._dispatching = False
        self._dispatch_socket: QTcpSocket | None = None
        self._operator = GuiOperatorBridge(parent)
        self._lease_token: str | None = None
        self._lease_deadline = 0.0
        self._lease_actor = ""
        self._lease_motion_authorized = False
        self._lease_timer = QTimer(self)
        self._lease_timer.setInterval(250)
        self._lease_timer.timeout.connect(self._lease_tick)
        self._lease_timer.start()

        TelemetryBus.instance().packet.connect(self._on_packet)
        CommandResultBus.instance().result.connect(self._on_command_result)
        send_param_get_all()  # populate _param_cache so name->id resolution works immediately

        self._server = QTcpServer(self)
        self._server.newConnection.connect(self._on_new_connection)
        if not self._server.listen(QHostAddress("127.0.0.1"), PORT):
            print(f"RemoteControlServer: failed to bind 127.0.0.1:{PORT} "
                  f"({self._server.errorString()}) — command server disabled "
                  "(another GUI instance already running?)")

    def close(self):
        """Close the listener and any client sockets during GUI shutdown."""
        self._lease_timer.stop()
        self._server.close()
        for sock in list(self._buffers):
            sock.abort()
        self._buffers.clear()

    # ── TelemetryBus feed ───────────────────────────────────────────────────

    def _on_packet(self, info: dict):
        if info.get("ptype") == 0x06:  # PARAM_REPORT
            pid = info.get("param_id")
            if pid is not None:
                self._param_cache[pid] = {
                    "value": info.get("param_value"),
                    "min":   info.get("param_min"),
                    "max":   info.get("param_max"),
                    "flags": info.get("param_flags"),
                    "name":  info.get("param_name"),
                }
        if info.get("type_name") == "TELEM":
            self._latest_telem = info
            self._latest_telem_monotonic = time.monotonic()

    def _on_command_result(self, info: dict):
        if info.get("command_id") == 0x02:  # heartbeat PING is not an operator transaction
            return
        self._latest_command_result = {
            key: info.get(key)
            for key in (
                "request_id", "command_id", "command_status", "command_status_name",
                "command_reason", "command_reason_name", "command_state", "command_accepted",
            )
        }
        self._latest_command_result["received_monotonic"] = time.monotonic()

    # ── TCP connection handling ─────────────────────────────────────────────

    def _on_new_connection(self):
        while self._server.hasPendingConnections():
            sock = self._server.nextPendingConnection()
            self._buffers[sock] = bytearray()
            sock.readyRead.connect(lambda s=sock: self._on_ready_read(s))
            sock.disconnected.connect(lambda s=sock: self._buffers.pop(s, None))

    def _on_ready_read(self, sock: QTcpSocket):
        buf = self._buffers.get(sock)
        if buf is None:
            return
        buf.extend(bytes(sock.readAll()))
        if b"\n" not in buf:
            return
        line = bytes(buf).split(b"\n", 1)[0]
        self._buffers.pop(sock, None)  # one command per connection — ignore anything after

        if self._dispatching and sock is not self._dispatch_socket:
            resp = {"ok": False, "error": "operator server busy; retry this request"}
        else:
            self._dispatching = True
            self._dispatch_socket = sock
            try:
                try:
                    cmd = json.loads(line.decode("utf-8"))
                except Exception as e:
                    resp = {"ok": False, "error": f"bad JSON: {e}"}
                else:
                    resp = self._dispatch(cmd)
                    resp.setdefault("request_id", cmd.get("request_id"))
            finally:
                self._dispatch_socket = None
                self._dispatching = False

        sock.write((json.dumps(resp) + "\n").encode("utf-8"))
        sock.disconnectFromHost()

    # ── Blocking waits (nested QEventLoop — see module docstring) ──────────

    def _wait_for_packet(self, predicate, timeout_ms: int, trigger=None) -> dict | None:
        loop = QEventLoop()
        result = {}

        def on_packet(info):
            if predicate(info):
                result["packet"] = info
                loop.quit()

        bus = TelemetryBus.instance()
        bus.packet.connect(on_packet)
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(timeout_ms)
        try:
            if trigger is not None:
                trigger()
            loop.exec()
        finally:
            bus.packet.disconnect(on_packet)
            timer.stop()
        return result.get("packet")

    def _wait_for_signal(self, signal, timeout_ms: int, trigger=None) -> tuple | None:
        loop = QEventLoop()
        result = {}

        def on_signal(*args):
            result["args"] = args
            loop.quit()

        signal.connect(on_signal)
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(timeout_ms)
        try:
            if trigger is not None:
                trigger()
            loop.exec()
        finally:
            signal.disconnect(on_signal)
            timer.stop()
        return result.get("args")

    # ── Confirmed param get/set (echo-back predicate, matches params_tab.py) ─

    def _param_get_confirmed(self, param_id: int, retries: int = 3, timeout_ms: int = 250) -> dict | None:
        if SourceManager.instance().active == "wifi":
            timeout_ms = max(timeout_ms, 750)
        for _ in range(retries + 1):
            pkt = self._wait_for_packet(
                lambda info: info.get("ptype") == 0x06 and info.get("param_id") == param_id,
                timeout_ms,
                trigger=lambda: send_param_get(param_id))
            if pkt is not None:
                return pkt
        return None

    def _param_set_confirmed(self, param_id: int, value: float, retries: int = 3, timeout_ms: int = 250):
        if SourceManager.instance().active == "wifi":
            timeout_ms = max(timeout_ms, 750)
        for _ in range(retries + 1):
            pkt = self._wait_for_packet(
                lambda info: (
                    info.get("ptype") == 0x06
                    and info.get("param_id") == param_id
                    and isinstance(info.get("param_value"), (int, float))
                    and math.isclose(float(info["param_value"]), value, rel_tol=1e-5, abs_tol=1e-6)
                ),
                timeout_ms,
                trigger=lambda: send_param_set(param_id, value))
            if pkt is not None:
                return pkt.get("param_value")
        return None

    def _set_mode_confirmed(self, target: int, timeout_ms: int = 1500) -> tuple[bool, str | None]:
        from main import _send_set_mode_reliable  # deferred: main.py imports this module

        loop = QEventLoop()
        result = {"ok": False, "msg": None}

        def on_success(_info):
            result["ok"] = True
            loop.quit()

        def on_fail(msg):
            result["ok"] = False
            result["msg"] = msg
            loop.quit()

        _send_set_mode_reliable(target, f"remote set_mode {target}",
                                 on_success=on_success, on_fail=on_fail)

        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(timeout_ms)
        loop.exec()
        timer.stop()
        return result["ok"], result["msg"]

    # ── Name/id resolution ──────────────────────────────────────────────────

    def _resolve_param_id(self, ident) -> int | None:
        if isinstance(ident, int):
            return ident
        if isinstance(ident, str):
            s = ident.strip()
            try:
                return int(s, 16) if s.lower().startswith("0x") else int(s)
            except ValueError:
                pass
            for pid, info in self._param_cache.items():
                if info.get("name") == s:
                    return pid
        return None

    def _resolve_state(self, ident) -> int | None:
        if isinstance(ident, int):
            return ident
        if isinstance(ident, str):
            return _STATE_NAME_TO_ID.get(ident.strip().upper())
        return None

    # ── Control lease ───────────────────────────────────────────────────────

    def _expire_lease(self):
        motion_was_authorized = self._lease_motion_authorized
        self._lease_token = None
        self._lease_deadline = 0.0
        self._lease_actor = ""
        self._lease_motion_authorized = False
        if motion_was_authorized:
            # Best available v1 dead-man behavior. Phase 2 replaces this with
            # an acknowledged DISARM event rather than the ESTOP fallback.
            from .comm_commands import send_set_mode

            send_param_set(PARAM_GUI_MOTION_CTRL_EN, 0.0)
            estop = _STATE_NAME_TO_ID.get("ESTOP")
            if estop is not None:
                send_set_mode(estop)

    def _lease_status(self) -> dict:
        now = time.monotonic()
        active = self._lease_token is not None and now < self._lease_deadline
        if not active and self._lease_token is not None:
            self._expire_lease()
        return {
            "active": active,
            "actor": self._lease_actor if active else "",
            "motion_authorized": bool(active and self._lease_motion_authorized),
            "expires_in_ms": max(0, int((self._lease_deadline - now) * 1000)) if active else 0,
        }

    def _lease_tick(self):
        if self._lease_token is None or time.monotonic() < self._lease_deadline:
            return
        self._expire_lease()

    def _require_lease(self, cmd: dict, *, motion=False) -> dict | None:
        status = self._lease_status()
        if not status["active"] or cmd.get("lease_token") != self._lease_token:
            return {"ok": False, "error": "a valid control lease is required", "lease": status}
        if motion and not status["motion_authorized"]:
            return {"ok": False, "error": "lease does not authorize motion", "lease": status}
        return None

    def _device_panel(self, device: str):
        flash_tab = getattr(self.parent(), "_tab_widgets", {}).get("Flash & Monitor")
        if flash_tab is None:
            return None
        return next((panel for panel in flash_tab._panels if panel._device == device), None)

    # ── Command dispatch ─────────────────────────────────────────────────────

    def _dispatch(self, cmd: dict) -> dict:
        action = cmd.get("cmd")
        try:
            handler = getattr(self, f"_cmd_{action}", None)
            if handler is None:
                return {"ok": False, "error": f"unknown cmd {action!r}"}
            return handler(cmd)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def _cmd_capabilities(self, cmd: dict) -> dict:
        commands = sorted(
            name.removeprefix("_cmd_")
            for name in dir(self)
            if name.startswith("_cmd_") and callable(getattr(self, name))
        )
        return {
            "ok": True,
            "api_version": 1,
            "commands": commands,
            "states": dict(sorted(_STATE_NAME_TO_ID.items())),
            "sources": list(PRIORITY),
            "gui": self._operator.manifest(),
        }

    def _cmd_health(self, cmd: dict) -> dict:
        sm = SourceManager.instance()
        pm = SerialPortManager.instance()
        from .wifi_transport import WifiTransport

        wifi = WifiTransport.instance()
        age_ms = None
        if self._latest_telem_monotonic is not None:
            age_ms = int((time.monotonic() - self._latest_telem_monotonic) * 1000)
        return {
            "ok": True,
            "service": {"listening": self._server.isListening(), "port": PORT, "api_version": 1},
            "source": {
                "active": sm.active,
                "override": sm.override,
                "connected": sorted(sm.connected),
            },
            "serial": {"detected": pm.scan(), "ports": pm.all_ports()},
            "wifi": {
                "thread_running": wifi.isRunning(),
                "telemetry_connected": wifi._connected,
                "esp_ip": wifi.esp_ip,
                "tcp_connected": wifi._tcp_sock is not None,
            },
            "robot": {
                "telemetry_age_ms": age_ms,
                "telemetry_fresh": age_ms is not None and age_ms < 3000,
                "state": self._latest_telem.get("state_name"),
                "fault": self._latest_telem.get("fault_name"),
                "protocol_version": self._latest_telem.get("version"),
                "timestamp_ms": self._latest_telem.get("timestamp_ms"),
                "last_command_result": {
                    key: value for key, value in self._latest_command_result.items()
                    if key != "received_monotonic"
                } or None,
            },
            "parameters_cached": len(self._param_cache),
            "lease": self._lease_status(),
        }

    def _cmd_service_status(self, cmd: dict) -> dict:
        return self._cmd_health(cmd)

    def _cmd_service_stop(self, cmd: dict) -> dict:
        QTimer.singleShot(100, QApplication.instance().quit)
        return {"ok": True, "stopping": True}

    def _cmd_lease_acquire(self, cmd: dict) -> dict:
        current = self._lease_status()
        actor = str(cmd.get("actor") or "local-agent")[:80]
        if current["active"] and actor != self._lease_actor and not cmd.get("takeover"):
            return {"ok": False, "error": "control lease is held by another actor", "lease": current}
        ttl_ms = min(60000, max(1000, int(cmd.get("ttl_ms", 15000))))
        self._lease_token = uuid.uuid4().hex
        self._lease_deadline = time.monotonic() + ttl_ms / 1000.0
        self._lease_actor = actor
        self._lease_motion_authorized = bool(
            cmd.get("authorize_motion", False)
            or (current["active"] and current["motion_authorized"] and actor == self._lease_actor)
        )
        return {"ok": True, "lease_token": self._lease_token, "lease": self._lease_status()}

    def _cmd_lease_renew(self, cmd: dict) -> dict:
        error = self._require_lease(cmd)
        if error:
            return error
        ttl_ms = min(60000, max(1000, int(cmd.get("ttl_ms", 15000))))
        self._lease_deadline = time.monotonic() + ttl_ms / 1000.0
        return {"ok": True, "lease_token": self._lease_token, "lease": self._lease_status()}

    def _cmd_lease_release(self, cmd: dict) -> dict:
        error = self._require_lease(cmd)
        if error:
            return error
        self._lease_token = None
        self._lease_deadline = 0.0
        self._lease_actor = ""
        self._lease_motion_authorized = False
        return {"ok": True, "lease": self._lease_status()}

    def _cmd_ui_manifest(self, cmd: dict) -> dict:
        return {"ok": True, "manifest": self._operator.manifest()}

    def _cmd_ui_parity_report(self, cmd: dict) -> dict:
        commands = [
            name.removeprefix("_cmd_")
            for name in dir(self)
            if name.startswith("_cmd_") and callable(getattr(self, name))
        ]
        return {"ok": True, "report": self._operator.parity_report(commands)}

    def _cmd_ui_snapshot(self, cmd: dict) -> dict:
        widgets = self._operator.snapshot(cmd.get("query"))
        return {"ok": True, "count": len(widgets), "widgets": widgets}

    def _cmd_ui_invoke(self, cmd: dict) -> dict:
        return self._operator.invoke(
            str(cmd.get("id", "")),
            str(cmd.get("action", "")),
            cmd.get("value"),
            acknowledge_risk=bool(cmd.get("acknowledge_risk", False)),
        )

    def _cmd_ui_screenshot(self, cmd: dict) -> dict:
        return self._operator.screenshot()

    def _cmd_tab_select(self, cmd: dict) -> dict:
        return self._operator.select_tab(str(cmd.get("title", "")))

    def _cmd_source_list(self, cmd: dict) -> dict:
        sm = SourceManager.instance()
        return {
            "ok": True,
            "active": sm.active,
            "override": sm.override,
            "connected": sorted(sm.connected),
            "priority": list(PRIORITY),
            "detected_ports": SerialPortManager.instance().scan(),
        }

    def _cmd_source_select(self, cmd: dict) -> dict:
        source = cmd.get("source")
        sm = SourceManager.instance()
        if source in (None, "", "auto"):
            sm.set_override(None)
        elif source not in PRIORITY:
            return {"ok": False, "error": f"unknown source {source!r}"}
        elif source not in sm.connected:
            return {"ok": False, "error": f"source {source!r} is not connected"}
        else:
            sm.set_override(source)
        return self._cmd_source_list({})

    def _cmd_connection_connect(self, cmd: dict) -> dict:
        device = str(cmd.get("device", ""))
        if device not in ("teensy", "esp32"):
            return {"ok": False, "error": "device must be 'teensy' or 'esp32'"}
        panel = self._device_panel(device)
        if panel is None:
            return {"ok": False, "error": "Flash & Monitor device panel is unavailable"}
        requested_port = cmd.get("port")
        if requested_port:
            panel._refresh_ports()
            for index in range(panel._port_combo.count()):
                if panel._port_combo.itemText(index).split(" ")[0].upper() == str(requested_port).upper():
                    panel._port_combo.setCurrentIndex(index)
                    break
            else:
                return {"ok": False, "error": f"port {requested_port!r} is not listed"}
        panel._user_disconnected = False
        panel._open_port()
        return {
            "ok": panel._reader is not None,
            "device": device,
            "port": panel._port_path(),
            "error": None if panel._reader is not None else "failed to open serial port",
        }

    def _cmd_connection_disconnect(self, cmd: dict) -> dict:
        device = str(cmd.get("device", ""))
        panel = self._device_panel(device)
        if panel is None:
            return {"ok": False, "error": f"unknown device {device!r}"}
        panel._user_disconnected = True
        panel._close_port("Operator disconnect", external=False)
        return {"ok": True, "device": device}

    def _cmd_param_list(self, cmd: dict) -> dict:
        params = [dict(id=pid, **info) for pid, info in sorted(self._param_cache.items())]
        return {"ok": True, "count": len(params), "parameters": params}

    def _cmd_param_get(self, cmd: dict) -> dict:
        pid = self._resolve_param_id(cmd.get("id"))
        if pid is None:
            return {"ok": False, "error": f"unknown param {cmd.get('id')!r}"}
        pkt = self._param_get_confirmed(pid)
        if pkt is None:
            return {"ok": False, "error": "timeout waiting for PARAM_REPORT"}
        return {"ok": True, "id": pid, "name": pkt.get("param_name"),
                "value": pkt.get("param_value"), "min": pkt.get("param_min"),
                "max": pkt.get("param_max"), "flags": pkt.get("param_flags")}

    def _cmd_param_set(self, cmd: dict) -> dict:
        lease_error = self._require_lease(cmd)
        if lease_error:
            return lease_error
        pid = self._resolve_param_id(cmd.get("id"))
        if pid is None:
            return {"ok": False, "error": f"unknown param {cmd.get('id')!r}"}
        value = float(cmd.get("value"))
        if not math.isfinite(value):
            return {"ok": False, "error": "parameter value must be finite"}
        got = self._param_set_confirmed(pid, value)
        if got is None:
            return {"ok": False, "error": "timeout waiting for PARAM_SET echo"}
        return {"ok": True, "id": pid, "value": got}

    def _cmd_set_mode(self, cmd: dict) -> dict:
        target = self._resolve_state(cmd.get("target"))
        if target is None:
            return {"ok": False, "error": f"unknown state {cmd.get('target')!r}"}
        if target in _ACTIVE_STATE_IDS:
            lease_error = self._require_lease(cmd, motion=True)
            if lease_error:
                return lease_error
        ok, msg = self._set_mode_confirmed(target)
        if not ok:
            return {"ok": False, "error": msg or "set_mode not confirmed (timeout)"}
        return {"ok": True, "robot_state": target}

    def _cmd_log_start(self, cmd: dict) -> dict:
        LogTransferManager.instance().start_logging(int(cmd.get("duration_ms", 0)))
        return {"ok": True}

    def _cmd_log_stop(self, cmd: dict) -> dict:
        LogTransferManager.instance().stop_logging()
        return {"ok": True}

    def _cmd_log_list(self, cmd: dict) -> dict:
        mgr = LogTransferManager.instance()
        args = self._wait_for_signal(mgr.directory_updated, 3000, trigger=mgr.refresh)
        if args is None:
            return {"ok": False, "error": "timeout waiting for directory listing"}
        return {"ok": True, "files": args[0]}

    def _cmd_log_download(self, cmd: dict) -> dict:
        idx = int(cmd.get("file_index"))
        mgr = LogTransferManager.instance()
        args = self._wait_for_signal(
            mgr.transfer_complete,
            90000,
            trigger=lambda: mgr.download(idx),
        )
        if args is None:
            return {"ok": False, "error": "timeout waiting for download"}
        file_index, local_path, crc_ok = args
        if not crc_ok or not local_path:
            return {"ok": False, "error": "download failed (CRC mismatch or transfer error)"}
        return {"ok": True, "file_index": file_index, "path": local_path}

    def _cmd_telem(self, cmd: dict) -> dict:
        age_ms = None
        if self._latest_telem_monotonic is not None:
            age_ms = int((time.monotonic() - self._latest_telem_monotonic) * 1000)
        return {
            "ok": True,
            "source": SourceManager.instance().active,
            "age_ms": age_ms,
            "fresh": age_ms is not None and age_ms < 3000,
            "telem": self._latest_telem,
        }

    def _cmd_motion_set(self, cmd: dict) -> dict:
        lease_error = self._require_lease(cmd, motion=True)
        if lease_error:
            return lease_error
        v     = float(cmd.get("v", 0.0))
        omega = float(cmd.get("omega", 0.0))
        if not math.isfinite(v) or not math.isfinite(omega):
            return {"ok": False, "error": "motion values must be finite"}
        already_on = self._param_cache.get(PARAM_GUI_MOTION_CTRL_EN, {}).get("value", 0.0) >= 0.5
        if not already_on:
            got_en = self._param_set_confirmed(PARAM_GUI_MOTION_CTRL_EN, 1.0)
            if got_en is None:
                return {"ok": False, "error": "timeout enabling gui_motion_ctrl_en"}
        got_v     = self._param_set_confirmed(PARAM_V_CMD_MS, v)
        got_omega = self._param_set_confirmed(PARAM_OMEGA_CMD_RDS, omega)
        if got_v is None or got_omega is None:
            return {"ok": False, "error": "timeout confirming v_cmd_ms/omega_cmd_rds"}
        return {"ok": True, "v_cmd_ms": got_v, "omega_cmd_rds": got_omega}

    def _cmd_motion_release(self, cmd: dict) -> dict:
        got = self._param_set_confirmed(PARAM_GUI_MOTION_CTRL_EN, 0.0)
        if got is None:
            return {"ok": False, "error": "timeout releasing gui_motion_ctrl_en"}
        return {"ok": True}

    def _cmd_firmware_flash(self, cmd: dict) -> dict:
        lease_error = self._require_lease(cmd)
        if lease_error:
            return lease_error
        device = str(cmd.get("device", ""))
        panel = self._device_panel(device)
        if panel is None:
            return {"ok": False, "error": f"unknown device {device!r}"}
        if panel._pio.running:
            return {"ok": False, "error": f"{device} flash is already running"}
        from .flash_monitor import FLASH_ACTIONS

        args = list(FLASH_ACTIONS[device]["main"][0][1])
        result = self._wait_for_signal(
            panel._pio.finished,
            int(cmd.get("timeout_ms", 240000)),
            trigger=lambda: panel._flash(args),
        )
        if result is None:
            return {"ok": False, "error": f"timeout flashing {device}"}
        ok = bool(result[0])
        return {
            "ok": ok,
            "device": device,
            "port": panel._port_path(),
            "output_tail": panel._output.toPlainText()[-12000:],
            "error": None if ok else f"{device} PlatformIO upload failed",
        }
