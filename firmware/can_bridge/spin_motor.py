"""Spin ODrive axis 0 at 5 t/s via CAN bridge, then stop."""
import serial, time, struct

PORT  = 'COM5'
BAUD  = 2000000
NODE  = 0
RUN_S = 5.0

CMD_HEARTBEAT        = 0x001
CMD_SET_AXIS_STATE   = 0x007
CMD_SET_CTRL_MODES   = 0x00B
CMD_SET_INPUT_VEL    = 0x00D

CTRL_VELOCITY    = 2
INPUT_PASSTHROUGH = 1
STATE_CLOSED_LOOP = 8

def frame_id(node, cmd): return (node << 5) | cmd

def send(s, fid, data: bytes):
    line = f"t{fid:03X}{len(data):X}" + data.hex() + "\r"
    s.write(line.encode())
    time.sleep(0.005)

def read_frames(s, seconds):
    buf = b''
    deadline = time.time() + seconds
    frames = []
    while time.time() < deadline:
        buf += s.read(64)
        while b'\r' in buf:
            line, buf = buf.split(b'\r', 1)
            ln = line.decode('ascii', errors='ignore').strip()
            if ln and ln[0] == 't' and len(ln) >= 5:
                try:
                    fid = int(ln[1:4], 16)
                    dlc = int(ln[4], 16)
                    data = bytes.fromhex(ln[5:5+dlc*2]) if dlc else b''
                    frames.append((fid, data))
                except Exception:
                    pass
    return frames

with serial.Serial(PORT, BAUD, timeout=0.05) as s:
    time.sleep(0.3); s.reset_input_buffer()
    s.write(b'O\r'); time.sleep(0.15); s.read(20)

    # --- Check axis state from heartbeat ---
    print("Checking axis state...")
    frames = read_frames(s, 0.5)
    hb = next((d for fid, d in frames if fid == frame_id(NODE, CMD_HEARTBEAT) and len(d) >= 5), None)
    if hb:
        ax_err   = struct.unpack_from('<I', hb)[0]
        ax_state = hb[4]
        print(f"  axis_error=0x{ax_err:08X}  axis_state={ax_state}")
        if ax_err != 0:
            print("  ERROR: axis has errors — clear them via USB first")
            s.write(b'C\r'); exit(1)
        if ax_state != STATE_CLOSED_LOOP:
            print(f"  Axis not in closed loop (state={ax_state}) — requesting state 8...")
            data = struct.pack('<I', STATE_CLOSED_LOOP)
            send(s, frame_id(NODE, CMD_SET_AXIS_STATE), data)
            time.sleep(0.5)
    else:
        print("  No heartbeat received — check wiring")
        s.write(b'C\r'); exit(1)

    # --- Set velocity control mode ---
    print("Setting VELOCITY control mode (passthrough)...")
    send(s, frame_id(NODE, CMD_SET_CTRL_MODES),
         struct.pack('<II', CTRL_VELOCITY, INPUT_PASSTHROUGH))

    # --- Spin at 5 t/s ---
    VEL = 5.0
    print(f"Commanding {VEL} t/s... (running {RUN_S} s)")
    send(s, frame_id(NODE, CMD_SET_INPUT_VEL), struct.pack('<ff', VEL, 0.0))

    # --- Live readout ---
    print(f"\n{'t':>5}  {'pos (turns)':>12}  {'vel (t/s)':>10}")
    print("-" * 34)
    buf = b''
    deadline = time.time() + RUN_S
    last_print = 0.0
    while time.time() < deadline:
        buf += s.read(64)
        while b'\r' in buf:
            line, buf = buf.split(b'\r', 1)
            ln = line.decode('ascii', errors='ignore').strip()
            if not ln or ln[0] != 't' or len(ln) < 5:
                continue
            try:
                fid = int(ln[1:4], 16)
                dlc = int(ln[4], 16)
                if fid == frame_id(NODE, 0x09) and dlc >= 8:
                    pos, vel = struct.unpack_from('<ff', bytes.fromhex(ln[5:5+16]))
                    now = time.time()
                    if now - last_print >= 0.2:
                        print(f"{now % 100:5.1f}  {pos:>+12.4f}  {vel:>+10.4f}")
                        last_print = now
            except Exception:
                pass

    # --- Stop ---
    print("\nStopping (vel = 0)...")
    send(s, frame_id(NODE, CMD_SET_INPUT_VEL), struct.pack('<ff', 0.0, 0.0))
    time.sleep(0.3)

    s.write(b'C\r')
    print("Done.")
