"""cmd_interface.py — File-based command inbox/outbox for Claude automation.

Polls odrive_cmd_inbox.txt every 500 ms.  Executes each line against the
connected ODrive and writes results to:
    odrive_cmd_log.txt      (human-readable, append-only)
    odrive_cmd_output.json  (structured JSON, last 100 results)
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QTimer, Signal, QObject

log = logging.getLogger("odrive_gui")

# File paths — live next to the GUI entry point
_DIR = Path(__file__).resolve().parent.parent
INBOX_PATH = _DIR / "odrive_cmd_inbox.txt"
LOG_PATH = _DIR / "odrive_cmd_log.txt"
JSON_PATH = _DIR / "odrive_cmd_output.json"

MAX_JSON_ENTRIES = 100
POLL_MS = 500


class CmdInterface(QObject):
    """Polls an inbox text file and executes commands against ODriveManager."""

    # Emitted after every command so the Terminal tab can show the result
    result_ready = Signal(str, str, bool)  # (command, result_text, is_error)
    # Emitted when __CONNECT__ finishes (success: bool, message: str)
    connect_finished = Signal(bool, str)

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self._mgr = manager
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll_inbox)
        self._json_history: list[dict] = []

        # Load existing JSON history if present
        if JSON_PATH.exists():
            try:
                self._json_history = json.loads(JSON_PATH.read_text(encoding="utf-8"))
            except Exception:
                self._json_history = []

    # ── Start / stop ──────────────────────────────────────────────────────────

    def start(self):
        self._timer.start()
        log.info("CmdInterface: polling %s every %d ms", INBOX_PATH, POLL_MS)

    def stop(self):
        self._timer.stop()

    # ── Public: execute a single command string ───────────────────────────────

    def execute(self, cmd: str) -> tuple[str, bool]:
        """Run *cmd* and return (result_text, is_error).

        Called by the Terminal tab for interactive use as well as by the
        inbox poller.
        """
        cmd = cmd.strip()
        if not cmd:
            return ("", False)

        # Meta-commands
        meta = cmd.upper()
        if meta == "__CLEAR_ERRORS__":
            return self._meta_clear_errors()
        if meta == "__IDLE__":
            return self._meta_idle()
        if meta == "__STOP__":
            return self._meta_stop()
        if meta == "__STATUS__":
            return self._meta_status()
        if meta == "__CONNECT__":
            return self._meta_connect()
        if meta == "__DISCONNECT__":
            return self._meta_disconnect()

        # Regular eval/exec against ODrive namespace
        return self._eval_cmd(cmd)

    # ── Meta-commands ─────────────────────────────────────────────────────────

    def _meta_clear_errors(self) -> tuple[str, bool]:
        if not self._mgr.connected:
            return ("Not connected", True)
        try:
            self._mgr.execute("clear_errors")
            return ("Errors cleared", False)
        except Exception as e:
            return (str(e), True)

    def _meta_idle(self) -> tuple[str, bool]:
        if not self._mgr.connected:
            return ("Not connected", True)
        try:
            self._mgr.safe_set("axis0.requested_state", 1)
            self._mgr.safe_set("axis1.requested_state", 1)
            return ("Both axes set to IDLE", False)
        except Exception as e:
            return (str(e), True)

    def _meta_stop(self) -> tuple[str, bool]:
        """Same as __IDLE__ — safe stop."""
        return self._meta_idle()

    def _meta_connect(self) -> tuple[str, bool]:
        if self._mgr.connected:
            return ("Already connected", False)
        # Kick off blocking connect in a background thread; result logged async
        log.info("CMD __CONNECT__: searching for ODrive...")
        threading.Thread(target=self._connect_thread, daemon=True).start()
        return ("Connecting... (async, check log for result)", False)

    def _connect_thread(self):
        try:
            self._mgr.connect()
            info = self._mgr.device_info() or {}
            msg = f"Connected — hw={info.get('hw','')} fw={info.get('fw','')} serial={info.get('serial','')}"
            log.info("CMD __CONNECT__: %s", msg)
            self._log_result("__CONNECT__", msg, False)
            self.connect_finished.emit(True, msg)
            self.result_ready.emit("__CONNECT__", msg, False)
        except Exception as e:
            msg = f"Connect failed: {e}"
            log.error("CMD __CONNECT__: %s", msg)
            self._log_result("__CONNECT__", msg, True)
            self.connect_finished.emit(False, msg)
            self.result_ready.emit("__CONNECT__", msg, True)

    def _meta_disconnect(self) -> tuple[str, bool]:
        if not self._mgr.connected:
            return ("Not connected", True)
        self._mgr.disconnect()
        return ("Disconnected", False)

    def _meta_status(self) -> tuple[str, bool]:
        if not self._mgr.connected:
            return ("Not connected", True)
        lines = []
        info = self._mgr.device_info()
        if info:
            lines.append(f"hw={info['hw']}  fw={info['fw']}  serial={info['serial']}")
        vbus = self._mgr.vbus_voltage()
        if vbus is not None:
            lines.append(f"vbus={vbus:.2f} V")
        for i in range(2):
            state = self._mgr.axis_state_str(i)
            errs = self._mgr.axis_errors(i)
            err_parts = [f"{k}=0x{v:X}" for k, v in errs.items() if v != 0]
            err_str = ", ".join(err_parts) if err_parts else "none"
            lines.append(f"axis{i}: state={state}  errors={err_str}")
        return ("\n".join(lines), False)

    # ── Eval / exec ───────────────────────────────────────────────────────────

    def _build_namespace(self) -> dict:
        ns = {}
        odrv = self._mgr.odrv
        if odrv is not None:
            ns["odrv"] = odrv
            ns["odrv0"] = odrv
            ns["ax0"] = odrv.axis0
            ns["ax1"] = odrv.axis1
        return ns

    def _eval_cmd(self, cmd: str) -> tuple[str, bool]:
        if not self._mgr.connected:
            return ("Not connected", True)
        ns = self._build_namespace()
        try:
            # Try eval first (expressions)
            result = eval(cmd, {"__builtins__": {}}, ns)
            return (repr(result), False)
        except SyntaxError:
            pass
        except Exception as e:
            return (f"Error: {e}", True)
        # Fall back to exec (statements like assignments)
        try:
            exec(cmd, {"__builtins__": {}}, ns)
            return ("OK", False)
        except Exception as e:
            return (f"Error: {e}", True)

    # ── Inbox polling ─────────────────────────────────────────────────────────

    def _poll_inbox(self):
        if not INBOX_PATH.exists():
            return
        try:
            text = INBOX_PATH.read_text(encoding="utf-8").strip()
        except Exception:
            return
        if not text:
            return

        # Clear inbox immediately so commands aren't re-run
        try:
            INBOX_PATH.write_text("", encoding="utf-8")
        except Exception:
            return

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            log.info("CMD_INBOX: %s", line)
            result_text, is_error = self.execute(line)
            self._log_result(line, result_text, is_error)
            self.result_ready.emit(line, result_text, is_error)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_result(self, cmd: str, result: str, is_error: bool):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "ERR" if is_error else "OK"

        # Append to text log
        entry = f"[{ts}] {status} | {cmd} | {result}\n"
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            log.error("CmdInterface: failed to write log: %s", e)

        # Append to JSON history (keep last 100)
        record = {
            "timestamp": ts,
            "command": cmd,
            "result": result,
            "error": is_error,
        }
        self._json_history.append(record)
        self._json_history = self._json_history[-MAX_JSON_ENTRIES:]
        try:
            JSON_PATH.write_text(
                json.dumps(self._json_history, indent=2), encoding="utf-8"
            )
        except Exception as e:
            log.error("CmdInterface: failed to write JSON: %s", e)
