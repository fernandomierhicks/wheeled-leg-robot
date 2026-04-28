import serial, time, struct

PORT = 'COM5'
BAUD = 2000000

def parse_frame(line):
    line = line.strip()
    if not line:
        return None
    t = line[0]
    if t == 't' and len(line) >= 5:
        can_id = int(line[1:4], 16)
        dlc    = int(line[4], 16)
        data   = bytes.fromhex(line[5:5+dlc*2]) if dlc else b''
        return ('STD', can_id, dlc, data)
    if t == 'T' and len(line) >= 10:
        can_id = int(line[1:9], 16)
        dlc    = int(line[9], 16)
        data   = bytes.fromhex(line[10:10+dlc*2]) if dlc else b''
        return ('EXT', can_id, dlc, data)
    return None

CMD_HEARTBEAT       = 0x001
CMD_GET_ENCODER_EST = 0x009
CMD_GET_BUS_VOLTAGE = 0x017

def decode_heartbeat(data):
    if len(data) < 8: return f"short ({data.hex()})"
    err, state, proc, traj = struct.unpack_from('<IBBB', data)
    states = {0:'UNDEFINED',1:'IDLE',2:'STARTUP',3:'FULL_CAL',4:'MOTOR_CAL',
              7:'ENCODER_INDEX',8:'ENCODER_OFFSET',9:'CLOSED_LOOP',10:'LOCKIN',
              11:'ENCODER_DIR',12:'HOMING',13:'ENCODER_HALL',14:'ENCODER_HALL_PHASE'}
    sname = states.get(state, str(state))
    return f"axis_state={sname}({state})  axis_error=0x{err:08X}  proc={proc}  traj_done={traj}"

def decode_encoder_est(data):
    if len(data) < 8: return f"short ({data.hex()})"
    pos, vel = struct.unpack_from('<ff', data)
    return f"pos={pos:.4f} turns   vel={vel:.4f} turns/s"

def decode_bus_voltage(data):
    if len(data) < 8: return f"short ({data.hex()})"
    v, i = struct.unpack_from('<ff', data)
    return f"Vbus={v:.2f} V   Ibus={i:.3f} A"

def drain(s, t=0.2):
    deadline = time.time() + t
    buf = b''
    while time.time() < deadline:
        buf += s.read(64)
    return buf

def listen(s, seconds):
    buf = b''
    deadline = time.time() + seconds
    count = 0
    while time.time() < deadline:
        chunk = s.read(64)
        if not chunk:
            continue
        buf += chunk
        while b'\r' in buf:
            line_b, buf = buf.split(b'\r', 1)
            line = line_b.decode('ascii', errors='ignore').strip()
            frame = parse_frame(line)
            if not frame:
                if line:
                    print(f"  [raw] {line!r}")
                continue
            ftype, fid, dlc, data = frame
            cmd  = fid & 0x1F
            node = fid >> 5
            if cmd == CMD_HEARTBEAT:
                detail = decode_heartbeat(data)
            elif cmd == CMD_GET_ENCODER_EST:
                detail = decode_encoder_est(data)
            elif cmd == CMD_GET_BUS_VOLTAGE:
                detail = decode_bus_voltage(data)
            else:
                detail = data.hex()
            print(f"  [{ftype} 0x{fid:03X}] node={node} cmd=0x{cmd:02X}  {detail}")
            count += 1
    return count

with serial.Serial(PORT, BAUD, timeout=0.1) as s:
    time.sleep(0.3)
    s.reset_input_buffer()

    # --- Open ---
    s.write(b'O\r')
    time.sleep(0.15)
    resp = drain(s, 0.2)
    ok = b'\r' in resp
    print(f"Open: {'OK' if ok else 'NO ACK'}  raw={resp!r}")

    # --- Try transmitting a heartbeat-request frame ---
    # ODrive node 0, get_version cmd 0x000: send 0 data bytes
    print("\n[1] Sending t0000 (node=0 get_version request)...")
    s.write(b't0000\r')
    time.sleep(0.1)
    ack = drain(s, 0.2)
    print(f"    TX ack: {ack!r}  ({'OK - bridge can transmit' if b'z' in ack else 'FAIL - CAN bus error (check wiring/termination)'})")

    # --- Listen passively for 4 s ---
    print("\n[2] Passive listen 4 s (ODrive heartbeats if CAN is enabled)...")
    n = listen(s, 4.0)
    if n == 0:
        print("  No frames received.")
        print("\n  Possible causes:")
        print("  1. ODrive CAN not configured (odrv0.can.config.baud_rate = 1000000, save_configuration)")
        print("  2. Missing 120-ohm termination at both ends of the CAN bus")
        print("  3. ODrive not powered")
        print("  4. TX/RX pins swapped (try swapping D4/D5 connections)")
    else:
        print(f"\n  {n} frame(s) received.")

    s.write(b'C\r')
    print("\nChannel closed.")
