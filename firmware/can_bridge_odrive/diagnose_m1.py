"""diagnose_m1.py - CAN diagnostic for ODrive M1 (axis 1, node 1).

Usage:
    python diagnose_m1.py [COM_PORT]   (default: COM5)
"""

import serial, time, struct, sys

PORT = sys.argv[1] if len(sys.argv) > 1 else 'COM5'
BAUD = 2000000

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

AXIS_STATES = {
    0:'UNDEFINED', 1:'IDLE', 2:'STARTUP_SEQUENCE', 3:'FULL_CALIBRATION',
    4:'MOTOR_CALIBRATION', 7:'ENCODER_INDEX_SEARCH', 8:'ENCODER_OFFSET_CALIBRATION',
    9:'CLOSED_LOOP_CONTROL', 10:'LOCKIN_SPIN', 11:'ENCODER_DIR_FIND',
}

def fid(node, cmd): return (node << 5) | cmd

def send(s, frame_id, data=b''):
    line = "t%03X%X" % (frame_id, len(data)) + data.hex() + "\r"
    s.write(line.encode('ascii'))
    time.sleep(0.01)

def read_frames(s, seconds):
    buf = b''
    deadline = time.time() + seconds
    frames = []
    while time.time() < deadline:
        chunk = s.read(128)
        if chunk:
            buf += chunk
        while b'\r' in buf:
            line, buf = buf.split(b'\r', 1)
            ln = line.decode('ascii', errors='ignore').strip()
            if not ln or len(ln) < 5 or ln[0] != 't':
                continue
            try:
                fid_ = int(ln[1:4], 16)
                dlc  = int(ln[4], 16)
                data = bytes.fromhex(ln[5:5+dlc*2]) if dlc else b''
                frames.append((fid_, data))
            except Exception:
                pass
    return frames

def find_latest_hb(frames, node):
    result = None
    for fid_, data in frames:
        if fid_ == fid(node, CMD_HEARTBEAT) and len(data) >= 5:
            result = data
    return result

def hb_state(hb): return hb[4] if hb else None
def hb_error(hb): return struct.unpack_from('<I', hb, 0)[0] if hb else None

def decode_hb(hb):
    if not hb: return 'NO HEARTBEAT'
    err   = hb_error(hb)
    state = hb_state(hb)
    sname = AXIS_STATES.get(state, "STATE_%d" % state)
    return "%s  error=0x%08X" % (sname, err)

def decode_enc(data):
    if len(data) < 8: return "(short)"
    pos, vel = struct.unpack_from('<ff', data)
    return "pos=%+.4f t  vel=%+.4f t/s" % (pos, vel)

print("=== ODrive M1 CAN Diagnostic  (port=%s) ===" % PORT)
print()

with serial.Serial(PORT, BAUD, timeout=0.05) as s:
    time.sleep(0.3)
    s.reset_input_buffer()
    s.write(b'O\r')
    time.sleep(0.2)
    s.read(20)
    print("CAN bridge open.")
    print()

    # Step 1: passive listen
    print("Step 1: Listening 3 s for heartbeats from node 0 and node 1...")
    frames = read_frames(s, 3.0)

    hb0 = find_latest_hb(frames, 0)
    hb1 = find_latest_hb(frames, 1)
    enc1 = [(f_, d) for f_, d in frames if f_ == fid(1, CMD_ENCODER_EST)]

    print("  Node 0 (M0): " + decode_hb(hb0))
    print("  Node 1 (M1): " + decode_hb(hb1))
    print("  Node 0 encoder frames seen: %d" % len([f for f,d in frames if f==fid(0,CMD_ENCODER_EST)]))
    print("  Node 1 encoder frames seen: %d" % len(enc1))
    if enc1:
        print("  Node 1 last encoder: " + decode_enc(enc1[-1][1]))

    if not hb1:
        print()
        print("  *** M1 (node 1) not sending heartbeats.")
        print("  Check odrv0.axis1.config.can.node_id == 1 and baud rate.")
        s.write(b'C\r')
        sys.exit(1)

    st1  = hb_state(hb1)
    err1 = hb_error(hb1)
    print()
    print("  M1 axis_state=%d (%s)  axis_error=0x%08X" % (st1, AXIS_STATES.get(st1,'?'), err1))

    # Step 2: clear errors
    print()
    print("Step 2: Sending CLEAR_ERRORS to M1...")
    send(s, fid(1, CMD_CLEAR_ERRORS))
    time.sleep(0.3)
    frames = read_frames(s, 0.5)
    hb1 = find_latest_hb(frames, 1) or hb1
    err1 = hb_error(hb1)
    print("  M1 error after clear: 0x%08X  (%s)" % (err1, 'CLEARED' if err1==0 else 'STILL HAS ERRORS'))

    # Step 3: set velocity mode
    print()
    print("Step 3: Setting M1 -> VELOCITY / PASSTHROUGH mode...")
    send(s, fid(1, CMD_SET_CTRL_MODES), struct.pack('<II', CTRL_VELOCITY, INPUT_PASSTHROUGH))
    time.sleep(0.1)

    # Step 4: request closed-loop
    print("Step 4: Requesting M1 CLOSED_LOOP_CONTROL (state 8)...")
    send(s, fid(1, CMD_SET_AXIS_STATE), struct.pack('<I', STATE_CLOSED_LOOP))
    time.sleep(0.8)

    frames = read_frames(s, 1.0)
    hb1 = find_latest_hb(frames, 1) or hb1
    st1  = hb_state(hb1)
    err1 = hb_error(hb1)
    print("  M1 state after enable: %d (%s)" % (st1, AXIS_STATES.get(st1, '?')))
    print("  M1 error after enable: 0x%08X  (%s)" % (err1, 'NONE' if err1==0 else 'ERROR - check odrivetool'))

    if st1 != STATE_CLOSED_LOOP:
        print()
        print("  *** M1 did NOT reach CLOSED_LOOP (stuck in state %d)." % st1)
        print("  ROOT CAUSE: axis1 not calibrated. Fix via USB:")
        print("    odrv0.axis1.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE")
        print("    # wait for it to finish...")
        print("    odrv0.axis1.motor.config.pre_calibrated = True")
        print("    odrv0.axis1.encoder.config.pre_calibrated = True")
        print("    odrv0.save_configuration()")
        print("  Check specific encoder error: odrv0.axis1.encoder.error")
        s.write(b'C\r')
        sys.exit(1)

    # Step 5: spin
    VEL = 1.0
    RUN = 4.0
    print()
    print("Step 5: Commanding M1 %.1f t/s for %.0f s..." % (VEL, RUN))
    send(s, fid(1, CMD_SET_INPUT_VEL), struct.pack('<ff', VEL, 0.0))

    print()
    print("  %5s  %12s  %10s  state" % ('time', 'pos (turns)', 'vel (t/s)'))
    print("  " + "-"*46)
    deadline = time.time() + RUN
    buf = b''
    last_print = 0.0
    last_pos, last_vel, last_st = 0.0, 0.0, st1
    while time.time() < deadline:
        chunk = s.read(128)
        if chunk: buf += chunk
        while b'\r' in buf:
            line, buf = buf.split(b'\r', 1)
            ln = line.decode('ascii', errors='ignore').strip()
            if not ln or ln[0] != 't' or len(ln) < 5: continue
            try:
                fid_ = int(ln[1:4], 16)
                dlc  = int(ln[4], 16)
                d    = bytes.fromhex(ln[5:5+dlc*2]) if dlc else b''
                if fid_ == fid(1, CMD_ENCODER_EST) and dlc >= 8:
                    last_pos, last_vel = struct.unpack_from('<ff', d)
                elif fid_ == fid(1, CMD_HEARTBEAT) and dlc >= 5:
                    last_st = d[4]
                    last_err = struct.unpack_from('<I', d, 0)[0]
                    if last_err:
                        print("  *** M1 ERROR during spin: 0x%08X" % last_err)
            except Exception:
                pass
        now = time.time()
        if now - last_print >= 0.25:
            print("  %5.1f  %+12.4f  %+10.4f  %s" % (
                now % 100, last_pos, last_vel, AXIS_STATES.get(last_st, str(last_st))))
            last_print = now

    print()
    print("Stopping M1...")
    send(s, fid(1, CMD_SET_INPUT_VEL), struct.pack('<ff', 0.0, 0.0))
    time.sleep(0.3)
    send(s, fid(1, CMD_SET_AXIS_STATE), struct.pack('<I', STATE_IDLE))

    s.write(b'C\r')
    print()
    print("Done.")
    print("If M1 moved above: CAN comms fine, issue is in robot firmware enable sequence.")
    print("If M1 did NOT move: check odrv0.axis1.error and odrv0.axis1.motor.error via USB.")
