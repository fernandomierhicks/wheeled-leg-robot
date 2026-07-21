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

Exit code 0 on {"ok": true}, 1 otherwise (including connect/timeout failures).
"""

import json
import socket
import sys

HOST = "127.0.0.1"
PORT = 8765
TIMEOUT_S = 95.0  # covers remote_control.py's log_download wait (~90s)


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

    raise SystemExit(f"unknown command {cmd!r}\n\n{__doc__}")


def main():
    request = _build(sys.argv[1:])

    try:
        with socket.create_connection((HOST, PORT), timeout=TIMEOUT_S) as s:
            s.sendall((json.dumps(request) + "\n").encode("utf-8"))
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
    except OSError as e:
        print(json.dumps({"ok": False, "error": f"connect failed: {e} "
                           "(is the GUI running with the remote-control server up?)"}))
        sys.exit(1)

    try:
        response = json.loads(data.decode("utf-8"))
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"bad response: {e}"}))
        sys.exit(1)

    print(json.dumps(response, indent=2))
    sys.exit(0 if response.get("ok") else 1)


if __name__ == "__main__":
    main()
