"""spin_both.py - Spin M0 (node 0) and M1 (node 1) simultaneously via CAN bridge.

Usage:
    python spin_both.py [COM_PORT] [VEL_TURNS_S]  (defaults: COM5, 1.0)
"""
import serial, time, struct, sys

PORT = sys.argv[1] if len(sys.argv) > 1 else 'COM5'
VEL  = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
BAUD = 2000000
RUN  = 6.0

CMD_HEARTBEAT      = 0x001
CMD_SET_AXIS_STATE = 0x007
CMD_ENCODER_EST    = 0x009
CMD_SET_CTRL_MODES = 0x00B
CMD_SET_INPUT_VEL  = 0x00D
CMD_CLEAR_ERRORS   = 0x018

CTRL_VELOCITY     = 2
INPUT_PASSTHROUGH = 1
STATE_CLOSED_LOOP = 8
STATE_IDLE        = 1

def fid(node, cmd): return (node << 5) | cmd

def send(s, frame_id, data=b''):
    line = "t%03X%X" % (frame_id, len(data)) + data.hex() + "\r"
    s.write(line.encode('ascii'))
    time.sleep(0.008)

def read_frames(s, seconds):
    buf = b''
    deadline = time.time() + seconds
    frames = []
    while time.time() < deadline:
        chunk = s.read(256)
        if chunk: buf += chunk
        while b'\r' in buf:
            line, buf = buf.split(b'\r', 1)
            ln = line.decode('ascii', errors='ignore').strip()
            if not ln or len(ln) < 5 or ln[0] != 't': continue
            try:
                fid_ = int(ln[1:4], 16)
                dlc  = int(ln[4], 16)
                d    = bytes.fromhex(ln[5:5+dlc*2]) if dlc else b''
                frames.append((fid_, d))
            except Exception: pass
    return frames

def find_hb(frames, node):
    result = None
    for fid_, d in frames:
        if fid_ == fid(node, CMD_HEARTBEAT) and len(d) >= 5:
            result = d
    return result

def hb_state(hb): return hb[4] if hb else None
def hb_error(hb): return struct.unpack_from('<I', hb, 0)[0] if hb else None

def enable_axis(s, node, initial_hb):
    label = "M%d" % node
    st0  = hb_state(initial_hb)
    err0 = hb_error(initial_hb)

    # If already in CLOSED_LOOP just set mode and go
    if st0 == STATE_CLOSED_LOOP:
        print("  %s: already CLOSED_LOOP — setting velocity mode" % label)
        send(s, fid(node, CMD_SET_CTRL_MODES), struct.pack('<II', CTRL_VELOCITY, INPUT_PASSTHROUGH))
        return True

    # Try to clear errors then enable
    send(s, fid(node, CMD_CLEAR_ERRORS))
    time.sleep(0.2)
    send(s, fid(node, CMD_SET_CTRL_MODES), struct.pack('<II', CTRL_VELOCITY, INPUT_PASSTHROUGH))
    time.sleep(0.05)
    send(s, fid(node, CMD_SET_AXIS_STATE), struct.pack('<I', STATE_CLOSED_LOOP))

    # Wait up to 3 s for CLOSED_LOOP transition (M1 may need encoder offset calib first)
    deadline = time.time() + 3.0
    hb = initial_hb
    while time.time() < deadline:
        frames = read_frames(s, 0.3)
        h = find_hb(frames, node)
        if h: hb = h
        if hb_state(hb) == STATE_CLOSED_LOOP:
            break

    st  = hb_state(hb)
    err = hb_error(hb)
    ok  = (st == STATE_CLOSED_LOOP)
    print("  %s: state=%d  error=0x%08X  %s" % (label, st or 0, err or 0, 'OK' if ok else 'FAILED - use USB GUI to enable M%d first' % node))
    return ok

print("=== Dual-Motor Spin Test  port=%s  vel=%.2f t/s ===" % (PORT, VEL))
print()

with serial.Serial(PORT, BAUD, timeout=0.05) as s:
    time.sleep(0.3)
    s.reset_input_buffer()
    s.write(b'O\r'); time.sleep(0.2); s.read(20)
    print("CAN bridge open.")

    # Check initial state
    print()
    print("Step 1: Reading heartbeats...")
    frames = read_frames(s, 2.0)
    for node in (0, 1):
        hb = find_hb(frames, node)
        if hb:
            print("  M%d: state=%d  error=0x%08X" % (node, hb_state(hb), hb_error(hb)))
        else:
            print("  M%d: NO HEARTBEAT" % node)

    # Enable both axes
    print()
    hb0_init = find_hb(frames, 0)
    hb1_init = find_hb(frames, 1)

    print("Step 2: Enabling M0 (node 0)...")
    ok0 = enable_axis(s, 0, hb0_init)

    print("Step 3: Enabling M1 (node 1)...")
    ok1 = enable_axis(s, 1, hb1_init)

    if not ok0 or not ok1:
        print()
        print("One or both axes failed to enable. Check ODrive errors via USB.")
        s.write(b'C\r')
        sys.exit(1)

    # Spin both
    print()
    print("Step 4: Commanding both axes %.2f t/s for %.0f s..." % (VEL, RUN))
    send(s, fid(0, CMD_SET_INPUT_VEL), struct.pack('<ff', VEL, 0.0))
    send(s, fid(1, CMD_SET_INPUT_VEL), struct.pack('<ff', VEL, 0.0))

    print()
    print("  %5s  %10s  %9s  %10s  %9s" % ('time', 'M0 pos(t)', 'M0 v(t/s)', 'M1 pos(t)', 'M1 v(t/s)'))
    print("  " + "-"*52)

    deadline = time.time() + RUN
    buf = b''
    last_print = 0.0
    pos = {0: 0.0, 1: 0.0}
    vel = {0: 0.0, 1: 0.0}

    while time.time() < deadline:
        chunk = s.read(256)
        if chunk: buf += chunk
        while b'\r' in buf:
            line, buf = buf.split(b'\r', 1)
            ln = line.decode('ascii', errors='ignore').strip()
            if not ln or ln[0] != 't' or len(ln) < 5: continue
            try:
                fid_ = int(ln[1:4], 16)
                dlc  = int(ln[4], 16)
                d    = bytes.fromhex(ln[5:5+dlc*2]) if dlc else b''
                for node in (0, 1):
                    if fid_ == fid(node, CMD_ENCODER_EST) and dlc >= 8:
                        pos[node], vel[node] = struct.unpack_from('<ff', d)
                    elif fid_ == fid(node, CMD_HEARTBEAT) and dlc >= 5:
                        err = struct.unpack_from('<I', d, 0)[0]
                        if err:
                            print("  *** M%d ERROR 0x%08X" % (node, err))
            except Exception: pass

        now = time.time()
        if now - last_print >= 0.25:
            print("  %5.1f  %+10.4f  %+9.4f  %+10.4f  %+9.4f" % (
                now % 100, pos[0], vel[0], pos[1], vel[1]))
            last_print = now

    # Stop both
    print()
    print("Stopping both axes...")
    send(s, fid(0, CMD_SET_INPUT_VEL), struct.pack('<ff', 0.0, 0.0))
    send(s, fid(1, CMD_SET_INPUT_VEL), struct.pack('<ff', 0.0, 0.0))
    time.sleep(0.3)
    send(s, fid(0, CMD_SET_AXIS_STATE), struct.pack('<I', STATE_IDLE))
    send(s, fid(1, CMD_SET_AXIS_STATE), struct.pack('<I', STATE_IDLE))

    s.write(b'C\r')
    print("Done.")
