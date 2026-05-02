#!/usr/bin/env python3
"""
ping_ak45.py — AK45-10 servo mode: read state, command positions, end.

Sequence:
  1. Read current position from 500 Hz broadcast
  2. Step through TARGET_POSITIONS_DEG with DWELL_S between each
  3. Return to starting position

Motor must already be in servo mode + position loop enabled via CubeMars PC.

Usage:
    python ping_ak45.py COM5
Requires: pyserial  (pip install pyserial)
"""

import sys
import time
import struct
import serial

# ── Config ───────────────────────────────────────────────────────────────────
MOTOR_ID = 69          # 0x41

# Positions to step through (degrees, absolute).  Edit as needed.
TARGET_POSITIONS_DEG = [120.0, 100.0, 160.0, 140.0]
DWELL_S = 2.0          # seconds to hold each position before moving to next

BAUD = 1_000_000

# ── CAN IDs ──────────────────────────────────────────────────────────────────
# Servo mode broadcast  (func 0x29, motor 0x41) — extended
FUNC_STATUS    = 0x29
EXT_ID_STATUS  = (FUNC_STATUS << 8) | MOTOR_ID   # 0x2941

# Position command      (func 0x04, motor 0x41) — extended
FUNC_POS       = 0x04
EXT_ID_POS_CMD = (FUNC_POS << 8) | MOTOR_ID      # 0x0441

ERROR_NAMES = {
    0: "OK", 1: "motor over-temp", 2: "over-current",
    3: "over-voltage", 4: "under-voltage", 5: "encoder fault",
    6: "MOSFET over-temp", 7: "motor stall",
}


# ── SLCAN helpers ─────────────────────────────────────────────────────────────
def slcan_send_ext(ser: serial.Serial, ext_id: int, data: bytes) -> None:
    frame = f"T{ext_id:08X}{len(data):1X}{data.hex().upper()}\r"
    ser.write(frame.encode())


def slcan_recv_frame(ser: serial.Serial, timeout: float) -> str | None:
    deadline = time.monotonic() + timeout
    buf = b""
    while time.monotonic() < deadline:
        c = ser.read(1)
        if not c:
            continue
        if c == b"\r":
            line = buf.decode(errors="replace")
            buf = b""
            if line.startswith("t") or line.startswith("T"):
                return line
        else:
            buf += c
    return None


def parse_ext_frame(frame: str) -> tuple[int, bytes] | None:
    if not frame.startswith("T") or len(frame) < 10:
        return None
    try:
        can_id = int(frame[1:9], 16) & 0x1FFFFFFF
        dlc    = int(frame[9], 16)
        data   = bytes.fromhex(frame[10:10 + dlc * 2])
        return can_id, data
    except (ValueError, IndexError):
        return None


# ── Motor data decode ─────────────────────────────────────────────────────────
def decode_status(data: bytes) -> dict:
    """Servo mode 0x29 broadcast — manual §5.2.1 p.44-45."""
    pos = int.from_bytes(data[0:2], "big", signed=True) * 0.1   # degrees
    spd = int.from_bytes(data[2:4], "big", signed=True) * 10.0  # ERPM
    cur = int.from_bytes(data[4:6], "big", signed=True) * 0.01  # amps
    tmp = int.from_bytes(data[6:7], "big", signed=True)          # °C
    err = data[7]
    return {"pos": pos, "spd": spd, "cur": cur, "tmp": tmp,
            "err": err, "err_str": ERROR_NAMES.get(err, f"#{err}")}


def send_position(ser: serial.Serial, deg: float) -> None:
    """Send position command — manual §5.1.5, int32 = deg × 10000."""
    raw = struct.pack(">i", int(deg * 10000.0))
    slcan_send_ext(ser, EXT_ID_POS_CMD, raw)


def read_current_pos(ser: serial.Serial, window: float = 0.5) -> float | None:
    """Drain broadcast frames for `window` seconds, return last position."""
    pos = None
    deadline = time.monotonic() + window
    while time.monotonic() < deadline:
        frame = slcan_recv_frame(ser, timeout=0.05)
        if not frame:
            continue
        parsed = parse_ext_frame(frame)
        if parsed and parsed[0] == EXT_ID_STATUS and len(parsed[1]) >= 8:
            pos = decode_status(parsed[1])["pos"]
    return pos


def monitor_position(ser: serial.Serial, target: float, duration: float) -> float:
    """Print live feedback for `duration` s, return final position."""
    deadline = time.monotonic() + duration
    pos = target
    tick = 0
    while time.monotonic() < deadline:
        frame = slcan_recv_frame(ser, timeout=0.05)
        if not frame:
            continue
        parsed = parse_ext_frame(frame)
        if not parsed or parsed[0] != EXT_ID_STATUS or len(parsed[1]) < 8:
            continue
        s = decode_status(parsed[1])
        pos = s["pos"]
        tick += 1
        if tick % 50 == 1:   # print ~every 100ms at 500Hz
            err = s['err'] if s['err'] else "OK"
            print(f"  pos={s['pos']:+8.1f}°  spd={s['spd']:+7.0f} ERPM  "
                  f"cur={s['cur']:+5.2f}A  tmp={s['tmp']}°C  err={err}")
    return pos


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    port = sys.argv[1] if len(sys.argv) > 1 else "COM5"

    with serial.Serial(port, BAUD, timeout=0.02) as ser:
        time.sleep(0.1)
        ser.reset_input_buffer()
        ser.write(b"O\r")
        time.sleep(0.05)

        # ── 1. Read starting position ─────────────────────────────────────
        print("Reading current position ...")
        start_pos = read_current_pos(ser, window=0.5)
        if start_pos is None:
            print("No broadcast received. Check motor power and servo mode.")
            ser.write(b"C\r")
            return
        print(f"Start position: {start_pos:+.1f}°\n")

        # ── 2. Step through target positions ─────────────────────────────
        targets = TARGET_POSITIONS_DEG + [start_pos]   # return home at end
        for i, deg in enumerate(targets):
            label = f"target {i+1}/{len(targets)}"
            if i == len(targets) - 1:
                label = "return to start"
            print(f"-- {label}: commanding {deg:+.1f} deg --")
            send_position(ser, deg)
            monitor_position(ser, target=deg, duration=DWELL_S)
            print()

        # ── 3. Done ───────────────────────────────────────────────────────
        ser.write(b"C\r")
        print("Done.")


if __name__ == "__main__":
    main()
