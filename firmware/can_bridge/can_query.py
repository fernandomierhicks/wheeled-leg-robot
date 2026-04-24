"""Send Get_Motor_Error request to ODrive node 0 and print the raw response.
ODrive CAN v0.5: cmd 0x003 = Get_Motor_Error.
Send an empty frame to trigger the response, then print the raw 8 bytes.
Also decodes the auto-sent heartbeat for comparison.
"""
import serial, time, struct

PORT = 'COM5'
BAUD = 2000000

# CAN ID = (node_id << 5) | cmd_id
NODE = 0
CMD_HEARTBEAT    = 0x001
CMD_MOTOR_ERROR  = 0x003
CMD_ENCODER_ERR  = 0x004

ID_HB  = (NODE << 5) | CMD_HEARTBEAT
ID_ME  = (NODE << 5) | CMD_MOTOR_ERROR
ID_EE  = (NODE << 5) | CMD_ENCODER_ERR

def can_id_str(fid):
    node = fid >> 5
    cmd  = fid & 0x1F
    return f"0x{fid:03X} (node={node} cmd=0x{cmd:02X})"

with serial.Serial(PORT, BAUD, timeout=0.2) as s:
    time.sleep(0.3)
    s.reset_input_buffer()

    # Open channel
    s.write(b'O\r')
    time.sleep(0.15)
    s.read(20)
    print("CAN channel open.\n")

    # Step 1: read one heartbeat first (auto-broadcast, no request needed)
    print("Waiting for heartbeat from node 0...")
    buf = b''
    deadline = time.time() + 3.0
    hb_raw = None
    while time.time() < deadline and hb_raw is None:
        buf += s.read(64)
        while b'\r' in buf:
            line, buf = buf.split(b'\r', 1)
            ln = line.decode('ascii', errors='ignore').strip()
            if not ln or ln[0] != 't' or len(ln) < 5:
                continue
            fid = int(ln[1:4], 16)
            dlc = int(ln[4], 16)
            if fid == ID_HB and dlc >= 5:
                hb_raw = bytes.fromhex(ln[5:5+dlc*2])
                break

    if hb_raw:
        axis_err   = struct.unpack_from('<I', hb_raw, 0)[0]
        axis_state = hb_raw[4]
        print(f"  Heartbeat raw  : {hb_raw.hex()}")
        print(f"  axis_error     : 0x{axis_err:08X}")
        print(f"  axis_state     : {axis_state}")
    else:
        print("  No heartbeat received (check wiring)")

    # Step 2: request Get_Motor_Error — send empty frame to cmd 0x003
    print(f"\nSending Get_Motor_Error request (ID={can_id_str(ID_ME)})...")
    s.reset_input_buffer()
    request = f"t{ID_ME:03X}0\r".encode()   # DLC=0, no data
    s.write(request)
    time.sleep(0.05)

    # Collect responses for 1 second, print everything non-heartbeat
    print("Responses received:")
    buf = b''
    deadline = time.time() + 1.0
    got_response = False
    while time.time() < deadline:
        buf += s.read(64)
        while b'\r' in buf:
            line, buf = buf.split(b'\r', 1)
            ln = line.decode('ascii', errors='ignore').strip()
            if not ln:
                continue
            if ln[0] == 'z':
                print(f"  TX ack: z  (frame sent OK)")
                continue
            if ln[0] != 't' or len(ln) < 5:
                continue
            fid = int(ln[1:4], 16)
            dlc = int(ln[4], 16)
            data = bytes.fromhex(ln[5:5+dlc*2]) if dlc else b''
            label = can_id_str(fid)
            if fid == ID_ME and dlc >= 4:
                motor_err = struct.unpack_from('<I', data, 0)[0]
                print(f"  Motor_Error reply  {label}: 0x{motor_err:08X}  raw={data.hex()}")
                got_response = True
            elif fid == ID_HB:
                pass  # skip heartbeats
            else:
                print(f"  {label}  dlc={dlc}  raw={data.hex()}")

    if not got_response:
        print("  (no reply to Get_Motor_Error request)")

    s.write(b'C\r')
    print("\nDone.")
