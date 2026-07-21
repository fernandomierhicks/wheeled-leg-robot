"""trigger_log_test.py — one-off CHECKPOINT 3 helper: start a timed SD log
over the Teensy's direct USB serial port, no GUI required.

Standalone (pyserial only) — does not import the Qt GUI or its port manager,
so it can run without the app open. Auto-detects the Teensy by VID:PID
(matches software/gui/port_manager.py SIGNATURES).

Usage:
    python trigger_log_test.py [duration_s] [port]

    duration_s  seconds to log (default 30). Log auto-stops after this —
                no separate STOP command needed.
    port        serial port to use, e.g. COM5. Auto-detected if omitted.
"""

import struct
import sys
import time
from pathlib import Path

import serial
from serial.tools import list_ports

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # software/gui/
from tabs.generated_protocol import CMD_ID_LOG  # noqa: E402

TEENSY_VID_PID = (0x16C0, 0x0483)
BAUD = 115200

# ── CommLink frame constants (shared/comm_protocol.h) ─────────────────────────
START_A, START_B, END = 0xAA, 0x55, 0xEF
SRC_PC = 0x03
TYPE_COMMAND, TYPE_LOG, TYPE_LOG_INFO = 0x02, 0x04, 0x12
LOG_SUB_START = 0x01
LOG_LEVEL_NAMES = {1: "INFO", 2: "WARN", 3: "ERROR"}
LOG_INFO_TYPE_NAMES = {1: "ENTRY", 2: "LIST_END", 3: "XFER_BEGIN", 4: "XFER_END", 5: "STATUS"}


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
                      "Pass the COM port explicitly, e.g. `python trigger_log_test.py 30 COM5`.")


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


def main():
    duration_s = float(sys.argv[1]) if len(sys.argv) >= 2 else 30.0
    port = sys.argv[2] if len(sys.argv) >= 3 else find_teensy_port()

    print(f"Opening {port} @ {BAUD}...")
    with serial.Serial(port, BAUD, timeout=0.2) as ser:
        time.sleep(0.3)
        ser.reset_input_buffer()

        dur_ms = int(duration_s * 1000)
        frame = build_frame(TYPE_COMMAND, 1, struct.pack("<BBI", CMD_ID_LOG, LOG_SUB_START, dur_ms), seq=0)
        print(f"Sending CMD_ID_LOG START duration_ms={dur_ms}...")
        ser.write(frame)

        print(f"Waiting {duration_s:.0f}s for the log to run and auto-stop "
              "(watching for comm_log / LOG_INFO replies)...")
        buf = bytearray()
        deadline = time.time() + duration_s + 3.0  # small margin past auto-stop
        saw_status_ok = False
        while time.time() < deadline:
            chunk = ser.read(4096)
            if chunk:
                buf.extend(chunk)
                for ftype, _ver, payload in parse_frames(buf):
                    if ftype == TYPE_LOG and len(payload) >= 1:
                        level = LOG_LEVEL_NAMES.get(payload[0], str(payload[0]))
                        msg = payload[1:].decode("ascii", errors="replace")
                        print(f"  [{level}] {msg}")
                    elif ftype == TYPE_LOG_INFO and len(payload) >= 16:
                        info_type, file_index, file_size, total_chunks, crc32, status = \
                            struct.unpack("<BHIIIB", payload[:16])
                        name = LOG_INFO_TYPE_NAMES.get(info_type, str(info_type))
                        print(f"  [LOG_INFO {name}] file_index={file_index} "
                              f"file_size={file_size} status={status}")
                        if name == "STATUS" and status == 0:
                            saw_status_ok = True
            else:
                time.sleep(0.05)

        if saw_status_ok:
            print("Got a LOG_INFO STATUS ok ack — logging started/stopped successfully.")
        else:
            print("No STATUS ack seen — check the comm_log lines above for errors "
                  "(no SD card? sd_logger_begin() failed at boot?).")

    print("Done. Power down, pull the microSD card, and run:")
    print("    python wlog_to_csv.py LOG0001.WLOG")
    print("(adjust the filename to whatever index this run created).")


if __name__ == "__main__":
    main()
