#!/usr/bin/env python3
"""
mit_listen.py — Send one MIT velocity command and print every raw CAN frame received.
Motor must already be in MIT mode. Does NOT send MIT ENTER.

Usage:
    python mit_listen.py COM5
"""

import sys, time, serial

PORT     = sys.argv[1] if len(sys.argv) > 1 else "COM5"
BAUD     = 1_000_000
MOTOR_ID = 65   # 0x41 — motor listens here

# Command: p=0 rad, v=1 rad/s, kp=0, kd=0.5, t=0
# Limits:  p ±12.5 rad, v ±20 rad/s, kp 0–500, kd 0–5, t ±8 Nm
def encode_cmd(p, v, kp, kd, t):
    def f2u(x, lo, hi, bits): return int((x - lo) / (hi - lo) * ((1 << bits) - 1))
    pi, vi, ki, di, ti = f2u(p,-12.5,12.5,16), f2u(v,-20,20,12), f2u(kp,0,500,12), f2u(kd,0,5,12), f2u(t,-8,8,12)
    return bytes([pi>>8, pi&0xFF, vi>>4, (vi&0xF)<<4|(ki>>8), ki&0xFF, di>>4, (di&0xF)<<4|(ti>>8), ti&0xFF])

CMD = encode_cmd(p=0, v=1.0, kp=0, kd=0.5, t=0)

def main():
    with serial.Serial(PORT, BAUD, timeout=0.05) as ser:
        time.sleep(0.1)
        ser.write(b"O\r")
        time.sleep(0.05)

        frame = f"t{MOTOR_ID:03X}8{CMD.hex().upper()}\r"
        ser.write(frame.encode())
        print(f"Sent MIT CMD → {frame.strip()}")
        print("Listening for any CAN frames (Ctrl+C to stop)...\n")

        buf = b""
        while True:
            c = ser.read(1)
            if not c:
                continue
            if c == b"\r":
                line = buf.decode(errors="replace").strip()
                buf = b""
                if line:
                    print(line)
            else:
                buf += c

if __name__ == "__main__":
    main()
