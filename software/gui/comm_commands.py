"""comm_commands.py — CommLink command framing + senders (shared/comm_protocol.h).

Shared helpers for building CommLink COMMAND frames and writing them to the
Teensy serial port. Used by any tab that needs to send commands (mode
changes, hip motor commands, ...).
"""

from port_manager import SerialPortManager

# ── CommLink frame constants (shared/comm_protocol.h) ─────────────────────────
COMM_START    = 0xFF
COMM_END      = 0xFE
COMM_SRC_PC   = 0x03
COMM_TYPE_CMD = 0x02
CMD_PAYLOAD_V = 1

# Command IDs (comm_protocol.h CMD_ID_*)
CMD_ID_SET_MODE  = 0x01
CMD_ID_HIP       = 0x05
CMD_ID_REBOOT    = 0x06
CMD_ID_WHEEL     = 0x07
CMD_ID_PARAM_SET = 0x10
CMD_ID_PARAM_GET = 0x11

# Wheel sub-commands (comm_protocol.h WHEEL_SUB_*)
WHEEL_SUB_SET_MODE     = 0x01
WHEEL_SUB_SEND         = 0x02
WHEEL_SUB_CLEAR_ERRORS = 0x03

# Wheel modes (WheelMode enum)
WHEEL_MODE_IDLE     = 0
WHEEL_MODE_VELOCITY = 1
WHEEL_MODE_POSITION = 2
WHEEL_MODE_TORQUE   = 3

# Robot state IDs (robot_state.h RobotStateEnum)
STATE_STARTUP = 0
STATE_STANDBY = 2
STATE_ESTOP   = 4
STATE_MANUAL  = 5

_seq = [0]  # rolling Tx sequence counter


def build_frame(payload: bytes) -> bytes:
    """Wrap payload in a CommLink COMMAND frame."""
    seq = _seq[0] & 0xFF
    _seq[0] += 1
    plen = len(payload)
    header = bytes([COMM_TYPE_CMD, CMD_PAYLOAD_V, COMM_SRC_PC,
                    seq, plen & 0xFF, (plen >> 8) & 0xFF])
    crc = 0
    for b in header + payload:
        crc ^= b
    return bytes([COMM_START]) + header + payload + bytes([crc, COMM_END])


def send_frame(frame: bytes):
    """Write a frame to the Teensy serial port (no-op if port is closed)."""
    pm = SerialPortManager.instance()
    with pm._lock:
        s = pm._open.get("teensy")
    if s and s.is_open:
        try:
            s.write(frame)
        except Exception:
            pass


def send_set_mode(target: int):
    """Send CMD_ID_SET_MODE with the given target RobotStateEnum value."""
    import struct
    send_frame(build_frame(struct.pack("<BB", CMD_ID_SET_MODE, target)))


def send_soft_clear():
    """Send soft-clear: ESTOP → STANDBY directly for SOFT severity faults.
    Firmware ignores this if the current fault is not SOFT severity."""
    send_set_mode(STATE_STANDBY)


def send_reboot():
    """Send CMD_ID_REBOOT — triggers a full Teensy MCU reset (reruns setup())."""
    import struct
    send_frame(build_frame(struct.pack("<B", CMD_ID_REBOOT)))


def send_param_set(param_id: int, value: float):
    """Send CMD_ID_PARAM_SET for one parameter. Firmware clamps and echoes back."""
    import struct
    send_frame(build_frame(struct.pack("<BHf", CMD_ID_PARAM_SET, param_id, value)))


def send_param_get_all():
    """Send CMD_ID_PARAM_GET 0xFFFF — firmware replies with one PARAM_REPORT per param."""
    import struct
    send_frame(build_frame(struct.pack("<BH", CMD_ID_PARAM_GET, 0xFFFF)))


def send_wheel_set_mode(mode: int):
    """Send CMD_ID_WHEEL / WHEEL_SUB_SET_MODE with WheelMode value."""
    import struct
    send_frame(build_frame(struct.pack("<BBB", CMD_ID_WHEEL, WHEEL_SUB_SET_MODE, mode)))


def send_wheel_setpoint(L: float, R: float):
    """Send CMD_ID_WHEEL / WHEEL_SUB_SEND with left and right setpoints."""
    import struct
    send_frame(build_frame(struct.pack("<BBff", CMD_ID_WHEEL, WHEEL_SUB_SEND, L, R)))


def send_wheel_clear_errors():
    """Send CMD_ID_WHEEL / WHEEL_SUB_CLEAR_ERRORS to both ODrive axes."""
    import struct
    send_frame(build_frame(struct.pack("<BB", CMD_ID_WHEEL, WHEEL_SUB_CLEAR_ERRORS)))
