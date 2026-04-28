#!/usr/bin/env python3
"""
test_servo_to_mit.py — Test whether an AK45-10 in servo mode can be switched
to MIT mode by sending the MIT enter command over CAN.

Sequence:
  1. Confirm servo mode broadcast is alive (extended frame 0x2941)
  2. Send MIT enter command (standard 11-bit frame to motor ID, 8x 0xFF...0xFC)
  3. Listen for 2s and classify every frame received:
       - continued servo broadcasts → mode did NOT switch
       - standard frame from motor  → MIT mode reply, mode DID switch
       - silence                    → motor stopped responding
  4. If MIT mode detected, send MIT exit to leave motor safe

Usage:
    python test_servo_to_mit.py COM5

Requires: pyserial  (pip install pyserial)
"""

import sys
import time
import serial

MOTOR_ID = 65          # 0x41
BAUD     = 1_000_000

EXT_ID_STATUS  = (0x29 << 8) | MOTOR_ID   # 0x2941 — servo broadcast
MIT_ENTER      = bytes([0xFF]*7 + [0xFC])
MIT_EXIT       = bytes([0xFF]*7 + [0xFD])


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


def slcan_send_std(ser: serial.Serial, can_id: int, data: bytes) -> None:
    frame = f"t{can_id:03X}{len(data):1X}{data.hex().upper()}\r"
    ser.write(frame.encode())


def classify(frame: str) -> tuple[str, int, bytes]:
    """Returns (kind, can_id, data).  kind = 'std' | 'ext'."""
    if frame.startswith("T"):
        can_id = int(frame[1:9], 16) & 0x1FFFFFFF
        dlc    = int(frame[9], 16)
        data   = bytes.fromhex(frame[10:10 + dlc * 2])
        return "ext", can_id, data
    else:
        can_id = int(frame[1:4], 16)
        dlc    = int(frame[4], 16)
        data   = bytes.fromhex(frame[5:5 + dlc * 2])
        return "std", can_id, data


def drain(ser: serial.Serial, window: float, label: str) -> dict:
    """Collect frames for `window` seconds, return counts by type."""
    counts = {"servo_broadcast": 0, "mit_reply": 0, "other": 0}
    deadline = time.monotonic() + window
    while time.monotonic() < deadline:
        frame = slcan_recv_frame(ser, timeout=0.05)
        if not frame:
            continue
        kind, can_id, data = classify(frame)
        if kind == "ext" and can_id == EXT_ID_STATUS:
            counts["servo_broadcast"] += 1
        elif kind == "std" and can_id == MOTOR_ID and len(data) == 8:
            counts["mit_reply"] += 1
            if counts["mit_reply"] == 1:
                print(f"  [MIT reply] raw: {data.hex()}  "
                      f"(motor_id={data[0]} temp_raw={data[6]} err={data[7]})")
        else:
            counts["other"] += 1
            if counts["other"] <= 3:
                print(f"  [other] kind={kind} id=0x{can_id:X} data={data.hex()}")
    print(f"  {label}: servo={counts['servo_broadcast']}  "
          f"mit={counts['mit_reply']}  other={counts['other']}")
    return counts


def main() -> None:
    port = sys.argv[1] if len(sys.argv) > 1 else "COM5"

    with serial.Serial(port, BAUD, timeout=0.02) as ser:
        time.sleep(0.1)
        ser.reset_input_buffer()
        ser.write(b"O\r")
        time.sleep(0.05)

        # 1. Confirm servo broadcast is live
        print("Step 1: checking servo broadcast (1s) ...")
        before = drain(ser, 1.0, "before MIT enter")
        if before["servo_broadcast"] == 0:
            print("No servo broadcast — is motor powered and in servo mode?")
            ser.write(b"C\r")
            return

        # 2. Send MIT enter command
        print("\nStep 2: sending MIT enter command (standard 11-bit, ID=0x41) ...")
        slcan_send_std(ser, MOTOR_ID, MIT_ENTER)

        # 3. Listen and classify
        print("\nStep 3: listening for 2s after MIT enter ...")
        after = drain(ser, 2.0, "after MIT enter")

        # 4. Verdict
        print()
        if after["mit_reply"] > 0 and after["servo_broadcast"] == 0:
            print("RESULT: mode switched to MIT — servo broadcast stopped, MIT replies received.")
            print("Sending MIT exit to leave motor safe ...")
            slcan_send_std(ser, MOTOR_ID, MIT_EXIT)
            time.sleep(0.1)
        elif after["mit_reply"] > 0 and after["servo_broadcast"] > 0:
            print("RESULT: both MIT replies and servo broadcasts observed — dual-mode or partial switch.")
            slcan_send_std(ser, MOTOR_ID, MIT_EXIT)
            time.sleep(0.1)
        elif after["servo_broadcast"] > 0:
            print("RESULT: servo broadcast continued unchanged — MIT enter had no effect in servo mode.")
            print("Mode switching likely requires the CubeMars PC tool.")
        else:
            print("RESULT: no frames at all after MIT enter — motor may have stopped responding.")
            print("Check motor state; may need power cycle.")

        ser.write(b"C\r")


if __name__ == "__main__":
    main()
