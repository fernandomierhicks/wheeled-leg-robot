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

from PyQt6.QtCore import QEventLoop, QObject, QTimer
from PyQt6.QtNetwork import QHostAddress, QTcpServer, QTcpSocket

from .comm_commands import send_param_get, send_param_get_all, send_param_set
from .log_transfer import LogTransferManager
from .telem_format import _STATE_NAMES
from .telemetry_bus import TelemetryBus

PORT = 8765

# GROUP_CONTROL params referenced directly by motion_set/motion_release
# (param_ids.h — tuning.md Sec1 d).
PARAM_GUI_MOTION_CTRL_EN = 0x042B
PARAM_V_CMD_MS           = 0x040B
PARAM_OMEGA_CMD_RDS      = 0x0411

_STATE_NAME_TO_ID = {name: pid for pid, name in _STATE_NAMES.items()}


class RemoteControlServer(QObject):
    """Owns the local QTcpServer. One instance held by MainWindow (main.py)
    for the app's lifetime — see win._remote_control there."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buffers: dict[QTcpSocket, bytearray] = {}
        self._param_cache: dict[int, dict] = {}
        self._latest_telem: dict = {}

        TelemetryBus.instance().packet.connect(self._on_packet)
        send_param_get_all()  # populate _param_cache so name->id resolution works immediately

        self._server = QTcpServer(self)
        self._server.newConnection.connect(self._on_new_connection)
        if not self._server.listen(QHostAddress("127.0.0.1"), PORT):
            print(f"RemoteControlServer: failed to bind 127.0.0.1:{PORT} "
                  f"({self._server.errorString()}) — command server disabled "
                  "(another GUI instance already running?)")

    def close(self):
        """Close the listener and any client sockets during GUI shutdown."""
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

        try:
            cmd = json.loads(line.decode("utf-8"))
        except Exception as e:
            resp = {"ok": False, "error": f"bad JSON: {e}"}
        else:
            resp = self._dispatch(cmd)

        sock.write((json.dumps(resp) + "\n").encode("utf-8"))
        sock.disconnectFromHost()

    # ── Blocking waits (nested QEventLoop — see module docstring) ──────────

    def _wait_for_packet(self, predicate, timeout_ms: int) -> dict | None:
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
            loop.exec()
        finally:
            bus.packet.disconnect(on_packet)
            timer.stop()
        return result.get("packet")

    def _wait_for_signal(self, signal, timeout_ms: int) -> tuple | None:
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
            loop.exec()
        finally:
            signal.disconnect(on_signal)
            timer.stop()
        return result.get("args")

    # ── Confirmed param get/set (echo-back predicate, matches params_tab.py) ─

    def _param_get_confirmed(self, param_id: int, retries: int = 3, timeout_ms: int = 250) -> dict | None:
        for _ in range(retries + 1):
            send_param_get(param_id)
            pkt = self._wait_for_packet(
                lambda info: info.get("ptype") == 0x06 and info.get("param_id") == param_id,
                timeout_ms)
            if pkt is not None:
                return pkt
        return None

    def _param_set_confirmed(self, param_id: int, value: float, retries: int = 3, timeout_ms: int = 250):
        for _ in range(retries + 1):
            send_param_set(param_id, value)
            pkt = self._wait_for_packet(
                lambda info: info.get("ptype") == 0x06 and info.get("param_id") == param_id,
                timeout_ms)
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
        pid = self._resolve_param_id(cmd.get("id"))
        if pid is None:
            return {"ok": False, "error": f"unknown param {cmd.get('id')!r}"}
        value = float(cmd.get("value"))
        got = self._param_set_confirmed(pid, value)
        if got is None:
            return {"ok": False, "error": "timeout waiting for PARAM_SET echo"}
        return {"ok": True, "id": pid, "value": got}

    def _cmd_set_mode(self, cmd: dict) -> dict:
        target = self._resolve_state(cmd.get("target"))
        if target is None:
            return {"ok": False, "error": f"unknown state {cmd.get('target')!r}"}
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
        mgr.refresh()
        args = self._wait_for_signal(mgr.directory_updated, 3000)
        if args is None:
            return {"ok": False, "error": "timeout waiting for directory listing"}
        return {"ok": True, "files": args[0]}

    def _cmd_log_download(self, cmd: dict) -> dict:
        idx = int(cmd.get("file_index"))
        mgr = LogTransferManager.instance()
        mgr.download(idx)
        args = self._wait_for_signal(mgr.transfer_complete, 90000)
        if args is None:
            return {"ok": False, "error": "timeout waiting for download"}
        file_index, local_path, crc_ok = args
        if not crc_ok or not local_path:
            return {"ok": False, "error": "download failed (CRC mismatch or transfer error)"}
        return {"ok": True, "file_index": file_index, "path": local_path}

    def _cmd_telem(self, cmd: dict) -> dict:
        return {"ok": True, "telem": self._latest_telem}

    def _cmd_motion_set(self, cmd: dict) -> dict:
        v     = float(cmd.get("v", 0.0))
        omega = float(cmd.get("omega", 0.0))
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
