#!/usr/bin/env python3
"""
mit_control.py — AK45-10 MIT mode velocity sweep (CAN ID = 65).

Sends a sequence of velocity commands with benign stiffness/torque limits,
prints motor feedback each step, then disables.

Usage:
    python mit_control.py COM5

Requires: pyserial  (pip install pyserial)
"""

import sys
import time
import serial

MOTOR_ID  = 65         # 0x41 — motor listens on this ID
MASTER_ID = 64         # 0x40 — motor replies using this ID
BAUD      = 1_000_000
PORT     = sys.argv[1] if len(sys.argv) > 1 else "COM5"

# Velocity steps: (v_des rad/s, duration_s)
# kp=0 (no position stiffness), kd=0.5 (gentle speed tracking), t_ff=0
STEPS = [
    ( 1.0, 2.0),
    ( 2.0, 2.0),
    ( 0.0, 1.0),
    (-1.0, 2.0),
    (-2.0, 2.0),
    ( 0.0, 1.0),
]
KP  = 0.0   # N·m/rad  — no position stiffness
KD  = 0.5   # N·m·s/rad — velocity damping / speed gain
P_DES = 0.0 # rad      — ignored when kp=0

# ── AK45-10 physical limits ───────────────────────────────────────────────────
P_MIN, P_MAX   = -12.5,  12.5
V_MIN, V_MAX   = -20.0,  20.0
KP_MIN, KP_MAX =   0.0, 500.0
KD_MIN, KD_MAX =   0.0,   5.0
T_MIN, T_MAX   =  -8.0,   8.0

ERROR_NAMES = {
    0: "OK", 1: "motor over-temp", 2: "over-current",
    3: "over-voltage", 4: "under-voltage", 5: "encoder fault",
    6: "MOSFET over-temp", 7: "motor stall",
}


def _f2u(x, xmin, xmax, bits):
    x = max(xmin, min(xmax, x))
    return int((x - xmin) / (xmax - xmin) * ((1 << bits) - 1))

def _u2f(xi, xmin, xmax, bits):
    return xi / ((1 << bits) - 1) * (xmax - xmin) + xmin

def encode_cmd(p, v, kp, kd, t):
    pi = _f2u(p, P_MIN, P_MAX, 16)
    vi = _f2u(v, V_MIN, V_MAX, 12)
    ki = _f2u(kp, KP_MIN, KP_MAX, 12)
    di = _f2u(kd, KD_MIN, KD_MAX, 12)
    ti = _f2u(t, T_MIN, T_MAX, 12)
    return bytes([
        (pi >> 8) & 0xFF, pi & 0xFF,
        (vi >> 4) & 0xFF,
        ((vi & 0xF) << 4) | ((ki >> 8) & 0xF),
        ki & 0xFF,
        (di >> 4) & 0xFF,
        ((di & 0xF) << 4) | ((ti >> 8) & 0xF),
        ti & 0xFF,
    ])

def decode_rx(data):
    if len(data) < 8:
        return None
    p_int = (data[1] << 8) | data[2]
    v_int = (data[3] << 4) | (data[4] >> 4)
    return {
        "pos": _u2f(p_int, P_MIN, P_MAX, 16),
        "vel": _u2f(v_int, V_MIN, V_MAX, 12),
        "temp": data[6] - 40,
        "err": ERROR_NAMES.get(data[7], f"#{data[7]}"),
    }

def send_std(ser, can_id, data):
    ser.write(f"t{can_id:03X}{len(data):1X}{data.hex().upper()}\r".encode())

def recv_latest(ser, window):
    latest = None
    deadline = time.monotonic() + window
    buf = b""
    while time.monotonic() < deadline:
        c = ser.read(1)
        if not c:
            continue
        if c == b"\r":
            line = buf.decode(errors="replace").strip()
            buf = b""
            if line.startswith("t"):
                try:
                    can_id = int(line[1:4], 16)
                    dlc    = int(line[4], 16)
                    data   = bytes.fromhex(line[5:5 + dlc * 2])
                    if can_id == MASTER_ID and dlc == 8:
                        latest = decode_rx(data)
                except Exception:
                    pass
        else:
            buf += c
    return latest


def main():
    print(f"Opening {PORT} ...")
    with serial.Serial(PORT, BAUD, timeout=0.02) as ser:
        time.sleep(0.1)
        ser.reset_input_buffer()
        ser.write(b"O\r")
        time.sleep(0.05)

        # Enable
        send_std(ser, MOTOR_ID, bytes([0xFF]*7 + [0xFC]))
        print("MIT enable sent. Starting velocity sweep ...\n")
        time.sleep(0.1)

        for v_des, duration in STEPS:
            print(f"v_des={v_des:+.1f} rad/s  ({duration:.0f}s)")
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                send_std(ser, MOTOR_ID, encode_cmd(P_DES, v_des, KP, KD, 0.0))
                state = recv_latest(ser, 0.02)
                if state:
                    print(f"  pos={state['pos']:+7.3f} rad  vel={state['vel']:+6.2f} rad/s  "
                          f"temp={state['temp']}°C  err={state['err']}")
                else:
                    print("  (no reply)")
                time.sleep(0.05)

        # Disable
        send_std(ser, MOTOR_ID, bytes([0xFF]*7 + [0xFD]))
        print("\nMIT disable sent.")
        time.sleep(0.05)
        ser.write(b"C\r")
        print("Done.")


if __name__ == "__main__":
    main()
