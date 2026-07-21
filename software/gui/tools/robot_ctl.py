"""robot_ctl.py — CLI client for the GUI's local remote-control TCP server
(tabs/remote_control.py, tuning.md Sec2). Pure stdlib (socket + json, no
pyserial) — never touches the robot directly, only the GUI's already-open
connection via its local command server. One-shot: connect, send one
command, print the JSON response, exit.

Usage:
    python robot_ctl.py param_get <id_or_name>
    python robot_ctl.py param_set <id_or_name> <value>
    python robot_ctl.py set_mode <STARTUP|CALIBRATION|STANDBY|RUNNING|ESTOP|MANUAL|CMD_REJECT|JUMPING>
    python robot_ctl.py log_start [duration_ms]
    python robot_ctl.py log_stop
    python robot_ctl.py log_list
    python robot_ctl.py log_download <file_index>
    python robot_ctl.py telem
    python robot_ctl.py motion_set <v> <omega>
    python robot_ctl.py motion_release
    python robot_ctl.py health
    python robot_ctl.py capabilities
    python robot_ctl.py source_list
    python robot_ctl.py source_select <auto|teensy|esp32|wifi>
    python robot_ctl.py connection_connect <teensy|esp32> [COM_port]
    python robot_ctl.py connection_disconnect <teensy|esp32>
    python robot_ctl.py param_list
    python robot_ctl.py tab_select <title>
    python robot_ctl.py ui_manifest
    python robot_ctl.py ui_parity_report
    python robot_ctl.py ui_snapshot [query]
    python robot_ctl.py ui_invoke <widget_id> <action> [json_value]
    python robot_ctl.py ui_screenshot
    python robot_ctl.py firmware_flash <teensy|esp32>
    python robot_ctl.py service_status
    python robot_ctl.py service_start
    python robot_ctl.py service_stop
    python robot_ctl.py service_restart
    python robot_ctl.py request '<json object>'

Exit code 0 on {"ok": true}, 1 otherwise (including connect/timeout failures).
"""

import json
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8765
TIMEOUT_S = 95.0  # covers remote_control.py's log_download wait (~90s)
GUI_DIR = Path(__file__).resolve().parent.parent


def _build(argv: list[str]) -> dict:
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]

    if cmd == "param_get":
        if len(rest) != 1:
            raise SystemExit("usage: param_get <id_or_name>")
        return {"cmd": "param_get", "id": rest[0]}
    if cmd == "param_set":
        if len(rest) != 2:
            raise SystemExit("usage: param_set <id_or_name> <value>")
        return {"cmd": "param_set", "id": rest[0], "value": float(rest[1])}
    if cmd == "set_mode":
        if len(rest) != 1:
            raise SystemExit("usage: set_mode <STATE_NAME>")
        return {"cmd": "set_mode", "target": rest[0]}
    if cmd == "log_start":
        if len(rest) > 1:
            raise SystemExit("usage: log_start [duration_ms]")
        return {"cmd": "log_start", "duration_ms": int(rest[0]) if rest else 0}
    if cmd == "log_stop":
        return {"cmd": "log_stop"}
    if cmd == "log_list":
        return {"cmd": "log_list"}
    if cmd == "log_download":
        if len(rest) != 1:
            raise SystemExit("usage: log_download <file_index>")
        return {"cmd": "log_download", "file_index": int(rest[0])}
    if cmd == "telem":
        return {"cmd": "telem"}
    if cmd == "motion_set":
        if len(rest) != 2:
            raise SystemExit("usage: motion_set <v> <omega>")
        return {"cmd": "motion_set", "v": float(rest[0]), "omega": float(rest[1])}
    if cmd == "motion_release":
        return {"cmd": "motion_release"}
    if cmd in (
        "health", "capabilities", "source_list", "param_list", "ui_manifest", "ui_parity_report", "ui_screenshot",
        "service_status", "service_stop",
    ):
        if rest:
            raise SystemExit(f"usage: {cmd}")
        return {"cmd": cmd}
    if cmd == "source_select":
        if len(rest) != 1:
            raise SystemExit("usage: source_select <auto|teensy|esp32|wifi>")
        return {"cmd": cmd, "source": rest[0]}
    if cmd in ("connection_connect", "connection_disconnect"):
        if len(rest) < 1 or len(rest) > (2 if cmd == "connection_connect" else 1):
            raise SystemExit(f"usage: {cmd} <teensy|esp32>" + (" [COM_port]" if cmd == "connection_connect" else ""))
        request = {"cmd": cmd, "device": rest[0]}
        if len(rest) == 2:
            request["port"] = rest[1]
        return request
    if cmd == "tab_select":
        if len(rest) != 1:
            raise SystemExit("usage: tab_select <title>")
        return {"cmd": cmd, "title": rest[0]}
    if cmd == "ui_snapshot":
        if len(rest) > 1:
            raise SystemExit("usage: ui_snapshot [query]")
        return {"cmd": cmd, "query": rest[0] if rest else ""}
    if cmd == "ui_invoke":
        if len(rest) not in (2, 3):
            raise SystemExit("usage: ui_invoke <widget_id> <action> [json_value]")
        request = {"cmd": cmd, "id": rest[0], "action": rest[1], "acknowledge_risk": True}
        if len(rest) == 3:
            request["value"] = json.loads(rest[2])
        return request
    if cmd == "firmware_flash":
        if len(rest) != 1:
            raise SystemExit("usage: firmware_flash <teensy|esp32>")
        return {"cmd": cmd, "device": rest[0], "timeout_ms": 240000}
    if cmd == "request":
        if len(rest) != 1:
            raise SystemExit("usage: request '<json object>'")
        request = json.loads(rest[0])
        if not isinstance(request, dict):
            raise SystemExit("request JSON must be an object")
        return request

    raise SystemExit(f"unknown command {cmd!r}\n\n{__doc__}")


def _request(request: dict) -> dict:
    request.setdefault("request_id", uuid.uuid4().hex)
    timeout_s = max(TIMEOUT_S, float(request.get("timeout_ms", 0)) / 1000.0 + 10.0)
    try:
        with socket.create_connection((HOST, PORT), timeout=timeout_s) as s:
            s.settimeout(timeout_s)
            s.sendall((json.dumps(request) + "\n").encode("utf-8"))
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
    except OSError as e:
        return {"ok": False, "error": f"connect failed: {e} "
                "(is the GUI running with the remote-control server up?)"}

    try:
        response = json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"bad response: {e}"}

    return response


def _needs_lease(request: dict) -> bool:
    if request.get("cmd") in ("param_set", "motion_set", "firmware_flash"):
        return True
    return request.get("cmd") == "set_mode" and str(request.get("target", "")).upper() in {
        "RUNNING", "MANUAL", "JUMPING", "STANDING_UP"
    }


def _start_gui() -> dict:
    existing = _request({"cmd": "service_status"})
    if existing.get("ok"):
        return {"ok": True, "started": False, "already_running": True, "health": existing}

    pythonw = Path(sys.executable).with_name("pythonw.exe")
    executable = str(pythonw if pythonw.exists() else Path(sys.executable))
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    subprocess.Popen(
        [executable, str(GUI_DIR / "main.py")],
        cwd=str(GUI_DIR),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
        close_fds=True,
    )
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        health = _request({"cmd": "service_status"})
        if health.get("ok"):
            return {"ok": True, "started": True, "already_running": False, "health": health}
        time.sleep(0.25)
    return {"ok": False, "error": "GUI did not start its operator server within 15 seconds"}


def main():
    if len(sys.argv) == 2 and sys.argv[1] in ("service_start", "service_restart"):
        if sys.argv[1] == "service_restart":
            stopped = _request({"cmd": "service_stop"})
            if stopped.get("ok"):
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    if not _request({"cmd": "service_status"}).get("ok"):
                        break
                    time.sleep(0.2)
        response = _start_gui()
        print(json.dumps(response, indent=2))
        sys.exit(0 if response.get("ok") else 1)

    request = _build(sys.argv[1:])
    if _needs_lease(request) and not request.get("lease_token"):
        lease = _request({
            "cmd": "lease_acquire",
            "actor": "robot_ctl",
            "ttl_ms": 60000 if request.get("cmd") == "firmware_flash" else 15000,
            "authorize_motion": request.get("cmd") in ("motion_set", "set_mode"),
            "takeover": False,
        })
        if not lease.get("ok"):
            print(json.dumps(lease, indent=2))
            sys.exit(1)
        request["lease_token"] = lease["lease_token"]

    response = _request(request)
    print(json.dumps(response, indent=2))
    sys.exit(0 if response.get("ok") else 1)


if __name__ == "__main__":
    main()
