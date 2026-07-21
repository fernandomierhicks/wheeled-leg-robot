"""trigger_running_test.py — send CMD_ID_SET_MODE(STATE_RUNNING) over USB
serial and confirm whether the state machine actually entered RUNNING, via
comm_log lines and the live TELEM_A robot_state field.

Standalone (pyserial only) — mirrors tools/trigger_log_test.py's frame
handling; does not import the Qt GUI or its port manager, so it can run
without the app open. Imports tabs.telem_format (Qt-free, also used by
wlog_to_csv.py) to decode TELEM_A so the state/enum layout can't drift out
of sync with shared/comm_protocol.h.

SAFETY: entering RUNNING starts the live LQR balance loop and sends real MIT
torque commands to any hip motor with hip_l/r_enable=1. This script is meant
for a pure command/state-machine smoke test with NO real torque output
anywhere:
    hip_l_enable = 0, hip_r_enable = 0, wheel_l_enable = 0, wheel_r_enable = 0
    calib_bypass_en = 1          (bypasses the hip-enable + hip-limits check)
    run_wheel_bypass_en = 1      (bypasses the wheel-enable check)
Set those via the Params tab (or CMD_ID_PARAM_SET) before running this. If
any motor is actually enabled, req_running() may command real torque the
instant RUNNING is entered — do not run this against a robot that isn't
secured for that.

Usage:
    python trigger_running_test.py [port]
"""

import struct
import sys
import time
from pathlib import Path

import serial
from serial.tools import list_ports

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # software/gui/
from tabs.telem_format import TELEM_VERSION, decode_telem_a  # noqa: E402
from tabs.generated_protocol import (  # noqa: E402
    CMD_ID_SET_MODE, STATE_CMD_REJECT, STATE_NAMES, STATE_RUNNING, STATE_STANDBY,
)

TEENSY_VID_PID = (0x16C0, 0x0483)
BAUD = 115200

# ── CommLink frame constants (shared/comm_protocol.h) ─────────────────────────
START_A, START_B, END = 0xAA, 0x55, 0xEF
SRC_PC = 0x03
TYPE_COMMAND, TYPE_LOG, TYPE_TELEM_A = 0x02, 0x04, 0x10
LOG_LEVEL_NAMES = {1: "INFO", 2: "WARN", 3: "ERROR"}

# RobotStateEnum (robot_state.h)


# CRC-8 poly 0x07, init 0x00 — MIRROR of crc8() in tabs/telem_format.py and
# shared/CommLink/CommLink.cpp (inlined here to keep this script standalone).
_CRC8_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = ((_c << 1) ^ 0x07) & 0xFF if _c & 0x80 else (_c << 1) & 0xFF
    _CRC8_TABLE.append(_c)


def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = _CRC8_TABLE[crc ^ b]
    return crc


def build_frame(ftype: int, version: int, payload: bytes, seq: int) -> bytes:
    plen = len(payload)
    header = bytes([ftype, version, SRC_PC, seq & 0xFF, plen & 0xFF, (plen >> 8) & 0xFF])
    return bytes([START_A, START_B]) + header + payload + bytes([_crc8(header + payload), END])


def find_teensy_port() -> str:
    for p in list_ports.comports():
        if (p.vid, p.pid) == TEENSY_VID_PID:
            return p.device
    raise SystemExit(f"No Teensy found (VID:PID {TEENSY_VID_PID[0]:04X}:{TEENSY_VID_PID[1]:04X}). "
                      "Pass the COM port explicitly, e.g. `python trigger_running_test.py COM5`.")


def parse_frames(buf: bytearray):
    """Yield (type, version, payload) for each complete frame found in buf,
    consuming bytes as it goes. Best-effort — drops anything before a sync."""
    while True:
        i = buf.find(bytes([START_A, START_B]))
        if i < 0:
            buf.clear()
            return
        if i > 0:
            del buf[:i]
        if len(buf) < 8:
            return
        ftype, ver, _src, _seq, len_lo, len_hi = buf[2:8]
        plen = len_lo | (len_hi << 8)
        total = 10 + plen
        if len(buf) < total:
            return
        payload = bytes(buf[8:8 + plen])
        crc = buf[8 + plen]
        end = buf[9 + plen]
        del buf[:total]
        if end != END:
            continue  # corrupt frame, already dropped
        if crc != _crc8(bytes([ftype, ver, _src, _seq, len_lo, len_hi]) + payload):
            continue
        yield ftype, ver, payload


def _drain(ser: serial.Serial, buf: bytearray, deadline: float, on_frame):
    """Read until deadline, decoding LOG + TELEM_A frames and calling on_frame(ftype, payload)."""
    while time.time() < deadline:
        chunk = ser.read(4096)
        if not chunk:
            time.sleep(0.02)
            continue
        buf.extend(chunk)
        for ftype, ver, payload in parse_frames(buf):
            if ftype == TYPE_LOG and len(payload) >= 1:
                level = LOG_LEVEL_NAMES.get(payload[0], str(payload[0]))
                msg = payload[1:].decode("utf-8", errors="replace")
                print(f"  [{level}] {msg}")
            elif ftype == TYPE_TELEM_A and ver == TELEM_VERSION:
                on_frame(decode_telem_a(payload)["robot_state"])


def main():
    # comm_log messages are UTF-8 (some embed em-dashes); a console codepage
    # that can't represent a given character shouldn't crash the run.
    sys.stdout.reconfigure(errors="replace")

    port = sys.argv[1] if len(sys.argv) >= 2 else find_teensy_port()

    print(f"Opening {port} @ {BAUD}...")
    with serial.Serial(port, BAUD, timeout=0.2) as ser:
        time.sleep(0.3)
        ser.reset_input_buffer()
        buf = bytearray()

        print("Reading current state from telemetry (up to 2s)...")
        pre_state = {"value": None}
        _drain(ser, buf, time.time() + 2.0,
               lambda s: pre_state.__setitem__("value", s) if pre_state["value"] is None else None)
        cur = pre_state["value"]
        print(f"  current state: {STATE_NAMES.get(cur, cur)}")
        if cur != STATE_STANDBY:
            print("  WARNING: not in STANDBY — stateMachine_request_running() only latches "
                  "from STANDBY (state_machine.cpp), so this request will likely be a no-op.")

        frame = build_frame(TYPE_COMMAND, 1, struct.pack("<BB", CMD_ID_SET_MODE, STATE_RUNNING), seq=0)
        print("Sending CMD_ID_SET_MODE(STATE_RUNNING)...")
        ser.write(frame)

        print("Waiting up to 3s for confirmation (comm_log lines + TELEM_A state)...")
        result = {"running": False, "reject": False}

        def _watch(s):
            if s == STATE_RUNNING:
                result["running"] = True
            elif s == STATE_CMD_REJECT:
                result["reject"] = True

        _drain(ser, buf, time.time() + 3.0, _watch)

        if result["running"]:
            print("CONFIRMED: robot entered RUNNING.")
        elif result["reject"]:
            print("DENIED: state machine rejected the arm request — see the WARN log line "
                  "above for the reason (state_machine.cpp req_running()).")
        else:
            print("NOT CONFIRMED: never saw state=RUNNING in telemetry. Check the log lines "
                  "above, and that calib_bypass_en / run_wheel_bypass_en are set to 1 if the "
                  "corresponding motors are disabled.")


if __name__ == "__main__":
    main()
