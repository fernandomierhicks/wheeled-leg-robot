"""comm_commands.py — CommLink command framing + senders (shared/comm_protocol.h).

Shared helpers for building CommLink COMMAND frames and writing them to the
Teensy serial port. Used by any tab that needs to send commands (mode
changes, hip motor commands, ...).
"""

from .port_manager import SerialPortManager
from .telem_format import crc8

# ── CommLink frame constants (shared/comm_protocol.h) ─────────────────────────
COMM_START_A  = 0xAA
COMM_START_B  = 0x55
COMM_END      = 0xEF
COMM_SRC_PC   = 0x03
COMM_TYPE_CMD = 0x02
CMD_PAYLOAD_V = 1

# Command IDs (comm_protocol.h CMD_ID_*)
CMD_ID_SET_MODE  = 0x01
CMD_ID_PING      = 0x02
CMD_ID_HIP       = 0x05
CMD_ID_REBOOT    = 0x06
CMD_ID_WHEEL     = 0x07
CMD_ID_PARAM_SET = 0x10
CMD_ID_PARAM_GET = 0x11
CMD_ID_LOG       = 0x12

# Log sub-commands (comm_protocol.h LOG_SUB_*)
LOG_SUB_START  = 0x01
LOG_SUB_STOP   = 0x02
LOG_SUB_LIST   = 0x03
LOG_SUB_GET    = 0x04
LOG_SUB_DELETE = 0x05

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
    """Wrap payload in a CommLink COMMAND frame (CRC-8 checksum)."""
    seq = _seq[0] & 0xFF
    _seq[0] += 1
    plen = len(payload)
    header = bytes([COMM_TYPE_CMD, CMD_PAYLOAD_V, COMM_SRC_PC,
                    seq, plen & 0xFF, (plen >> 8) & 0xFF])
    crc = crc8(header + payload)
    return bytes([COMM_START_A, COMM_START_B]) + header + payload + bytes([crc, COMM_END])


def send_frame(frame: bytes):
    """Send a frame over the single active transport (matches whichever
    device SourceManager has picked for telemetry: USB serial to esp32/teensy,
    or WiFi). Commands must go out exactly once — sending over every connected
    transport simultaneously double-delivers every command to the Teensy,
    which firmware request flags aren't designed to absorb (see
    stateMachine_request_calibration's STANDBY-only latch)."""
    from .source_manager import SourceManager
    active = SourceManager.instance().active

    if active == "wifi":
        from .wifi_transport import WifiTransport
        WifiTransport.instance().send(frame)
        return

    pm = SerialPortManager.instance()
    with pm._lock:
        s = pm._open.get(active)
    if s and s.is_open:
        try:
            s.write(frame)
        except Exception:
            pass


def send_set_mode(target: int):
    """Send CMD_ID_SET_MODE with the given target RobotStateEnum value."""
    import struct
    send_frame(build_frame(struct.pack("<BB", CMD_ID_SET_MODE, target)))


def send_ping():
    """Send CMD_ID_PING — GUI heartbeat. Feeds the firmware MANUAL-mode GUI
    watchdog (500 ms): if pings stop (GUI crash/disconnect), the robot exits
    MANUAL and idles the wheels. Sent at 10 Hz by MainWindow."""
    import struct
    send_frame(build_frame(struct.pack("<B", CMD_ID_PING)))


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


def send_log_start(duration_ms: int = 0):
    """Send CMD_ID_LOG / LOG_SUB_START. duration_ms=0 logs until STOP."""
    import struct
    send_frame(build_frame(struct.pack("<BBI", CMD_ID_LOG, LOG_SUB_START, duration_ms)))


def send_log_stop():
    """Send CMD_ID_LOG / LOG_SUB_STOP — closes the active log file."""
    import struct
    send_frame(build_frame(struct.pack("<BB", CMD_ID_LOG, LOG_SUB_STOP)))


def send_log_list():
    """Send CMD_ID_LOG / LOG_SUB_LIST — firmware replies with one LOG_INFO ENTRY per file."""
    import struct
    send_frame(build_frame(struct.pack("<BB", CMD_ID_LOG, LOG_SUB_LIST)))


def send_log_get(file_index: int, start_chunk: int = 0):
    """Send CMD_ID_LOG / LOG_SUB_GET — streams LOG_DATA chunks for one file."""
    import struct
    send_frame(build_frame(struct.pack("<BBHI", CMD_ID_LOG, LOG_SUB_GET, file_index, start_chunk)))


def send_log_delete(file_index: int):
    """Send CMD_ID_LOG / LOG_SUB_DELETE — erases one .wlog file."""
    import struct
    send_frame(build_frame(struct.pack("<BBH", CMD_ID_LOG, LOG_SUB_DELETE, file_index)))
