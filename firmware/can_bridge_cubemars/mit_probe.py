#!/usr/bin/env python3
"""
mit_probe.py — Systematic MIT mode probe. Tries permutations of CAN ID,
frame type, and command, printing every raw frame received after each attempt.

Usage:
    python mit_probe.py COM5
"""

import sys, time, serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM5"
BAUD = 1_000_000

MIT_ENTER = bytes([0xFF] * 7 + [0xFC])
MIT_EXIT  = bytes([0xFF] * 7 + [0xFD])
ZERO_CMD  = bytes([0x7F, 0xFF, 0x7F, 0xF0, 0x00, 0x00, 0x07, 0xFF])  # p=0,v=0,kp=0,kd=0,t=0

def send_std(ser, can_id, data):
    ser.write(f"t{can_id:03X}{len(data):1X}{data.hex().upper()}\r".encode())

def send_ext(ser, can_id, data):
    ser.write(f"T{can_id:08X}{len(data):1X}{data.hex().upper()}\r".encode())

def drain(ser, label, window=0.5):
    print(f"\n--- {label} ---")
    buf = b""
    deadline = time.monotonic() + window
    got_any = False
    while time.monotonic() < deadline:
        c = ser.read(1)
        if not c:
            continue
        if c == b"\r":
            line = buf.decode(errors="replace").strip()
            buf = b""
            if line and not line.startswith("z") and not line.startswith("Z"):
                print(f"  RX: {line}")
                got_any = True
        else:
            buf += c
    if not got_any:
        print("  (no frames)")

def attempt(ser, label, fn, window=0.5):
    fn()
    drain(ser, label, window)
    time.sleep(0.05)

with serial.Serial(PORT, BAUD, timeout=0.02) as ser:
    time.sleep(0.1)
    ser.reset_input_buffer()
    ser.write(b"O\r")
    time.sleep(0.05)

    # 1. Passive listen — what's already on the bus?
    drain(ser, "PASSIVE LISTEN (no TX)", window=1.5)

    # 2. MIT ENTER + zero poll to common motor IDs (std 11-bit)
    for mid in [65, 64, 1, 2, 0x01, 0x00]:
        attempt(ser, f"STD ENTER+POLL to ID {mid} (0x{mid:02X})",
                lambda m=mid: [send_std(ser, m, MIT_ENTER), time.sleep(0.05), send_std(ser, m, ZERO_CMD)])

    # 3. Zero poll only (no ENTER) to most likely IDs
    for mid in [65, 64]:
        attempt(ser, f"STD ZERO POLL only, ID {mid}",
                lambda m=mid: send_std(ser, m, ZERO_CMD))

    # 4. Extended frame ENTER (some firmwares use ext instead of std)
    for mid in [65, 64]:
        attempt(ser, f"EXT ENTER to ID {mid}",
                lambda m=mid: send_ext(ser, m, MIT_ENTER))

    # 5. Longer listen after last ext attempt
    drain(ser, "FINAL LISTEN", window=1.0)

    ser.write(b"C\r")
    print("\nDone.")
