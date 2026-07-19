"""comm_commands.py — CommLink command framing + senders (shared/comm_protocol.h).

Shared helpers for building CommLink COMMAND frames and writing them to the
Teensy serial port. Used by any tab that needs to send commands (mode
changes, hip motor commands, ...).
"""

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

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
CMD_ID_SET_TELEM_TRANSPORT = 0x08
CMD_ID_PARAM_SET = 0x10
CMD_ID_PARAM_GET = 0x11
CMD_ID_LOG       = 0x12
CMD_ID_PARAM_RESET_DEFAULTS = 0x13
CMD_ID_TEST_INJECT_CORRUPT  = 0x14

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
STATE_STARTUP    = 0
STATE_STANDBY    = 2
STATE_ESTOP      = 4
STATE_MANUAL     = 5
STATE_CMD_REJECT = 6

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


def build_frame_corrupted(payload: bytes, mode: int = 1) -> bytes:
    """TEST ONLY: build a COMMAND frame with one field deliberately damaged, to verify
    the receiving side's CommLink parser (Teensy/ESP32, shared/CommLink/CommLink.cpp)
    rejects it cleanly rather than misdispatching a garbled command. Mirrors
    CommLink::send()'s corrupt_mode_for_test on the firmware send side, applied here on
    the GUI's send side to test the *receive* path for COMMAND frames specifically —
    the same shared parser class that already guards TELEM frames also handles every
    incoming COMMAND frame (g_comm/g_comm_usb.onPacket(on_command) on the Teensy,
    g_comm_tcp/g_usb.onPacket(forward_to_teensy) on the ESP32), so this is checking that
    architectural claim empirically rather than adding a new guard.

    mode 1 = flipped CRC-8 byte (checksum-compare path)
    mode 2 = flipped END byte (PS_END bad-byte path)
    mode 3 = oversized on-wire length claim (length-guard/resync path)
    None of the three modes touch payload bytes — cmd_id and any command args are
    always intact; only whether the *frame* is accepted changes.
    """
    seq = _seq[0] & 0xFF
    _seq[0] += 1
    plen = len(payload)
    header = bytes([COMM_TYPE_CMD, CMD_PAYLOAD_V, COMM_SRC_PC,
                    seq, plen & 0xFF, (plen >> 8) & 0xFF])
    crc = crc8(header + payload)
    frame = bytearray(bytes([COMM_START_A, COMM_START_B]) + header + payload + bytes([crc, COMM_END]))
    if mode == 1:
        frame[-2] ^= 0xFF  # flip CRC byte
    elif mode == 2:
        frame[-1] ^= 0xFF  # flip END byte
    elif mode == 3:
        bogus_len = 512 + 50  # matches firmware COMM_MAX_PAYLOAD + margin
        frame[6] = bogus_len & 0xFF
        frame[7] = (bogus_len >> 8) & 0xFF
    return bytes(frame)


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


def send_reboot():
    """Send CMD_ID_REBOOT — triggers a full Teensy MCU reset (reruns setup())."""
    import struct
    send_frame(build_frame(struct.pack("<B", CMD_ID_REBOOT)))


def send_param_set(param_id: int, value: float):
    """Send CMD_ID_PARAM_SET for one parameter. Firmware clamps and echoes back."""
    import struct
    send_frame(build_frame(struct.pack("<BHf", CMD_ID_PARAM_SET, param_id, value)))


def send_param_get(param_id: int):
    """Send CMD_ID_PARAM_GET for one parameter. Firmware replies with a single PARAM_REPORT."""
    import struct
    send_frame(build_frame(struct.pack("<BH", CMD_ID_PARAM_GET, param_id)))


def send_param_get_all():
    """Send CMD_ID_PARAM_GET 0xFFFF — firmware replies with one PARAM_REPORT per param."""
    import struct
    send_frame(build_frame(struct.pack("<BH", CMD_ID_PARAM_GET, 0xFFFF)))


def send_param_reset_defaults():
    """Send CMD_ID_PARAM_RESET_DEFAULTS — firmware reverts every writable param to
    its compile-time default, persists it, and replies with a full PARAM_REPORT dump."""
    import struct
    send_frame(build_frame(struct.pack("<B", CMD_ID_PARAM_RESET_DEFAULTS)))


def send_test_inject_corrupt(count: int = 1, target: int = 0, mode: int = 1):
    """TEST ONLY (Phase 9, UARTplat.md stress testing): deliberately damage the
    next `count` outgoing TELEM_A frames per `mode`, to verify the receiving
    side's parser defenses actually detect and drop bad frames rather than
    just "hasn't seen corruption yet".

    target=0 (UART): Teensy's next `count` sends to the ESP32 over Serial5 —
        forwarded through by the ESP32 like any command. Watch the ESP32's
        wifi_uart_crc_drops (WIFI_DIAG field) for the count to land.
    target=1 (WiFi): ESP32's next `count` UDP datagrams to the GUI — this
        command is intercepted by the ESP32 itself (forward_to_teensy()),
        never reaches the Teensy. Watch this GUI's own link_crc_drops
        (PacketDecoder, TelemetryBus "wifi" source) for the count to land.

    mode=1 (default): flip the CRC-8 byte (checksum-compare path).
    mode=2: flip the END byte (frame otherwise valid — PS_END bad-byte path).
    mode=3: claim an oversized on-wire length (COMM_MAX_PAYLOAD+50 — receiver's
        length-guard path, tests recovery/resync without ever reading the crc).
    All three land on the same rx_drops/crc_drops counter — the firmware side
    doesn't distinguish detection *reason*, only that a frame was rejected.
    """
    import struct
    send_frame(build_frame(struct.pack("<BBBB", CMD_ID_TEST_INJECT_CORRUPT, count, target, mode)))


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


def send_set_telem_transport(suppress: bool):
    """Send CMD_ID_SET_TELEM_TRANSPORT — tells the ESP32 which link the GUI is
    reading, so it can suppress WiFi UDP telemetry sends while USB is active
    (only takes effect when the ESP32 is built with WIFI_TRANSPORT_GATING=1).
    No-op when the active source is "teensy" or unset — the ESP32 isn't part
    of that path anyway."""
    import struct
    send_frame(build_frame(struct.pack("<BB", CMD_ID_SET_TELEM_TRANSPORT, 1 if suppress else 0)))


def send_log_delete(file_index: int):
    """Send CMD_ID_LOG / LOG_SUB_DELETE — erases one .wlog file."""
    import struct
    send_frame(build_frame(struct.pack("<BBH", CMD_ID_LOG, LOG_SUB_DELETE, file_index)))


# ── Command reliability (effect-confirmed retry, UARTplat.md Phase 5) ────────
#
# Design choice: effect-confirmed retry at the GUI layer, no wire-protocol
# change. A transport-level ACK was considered and rejected: the ESP32
# re-stamps `seq` when forwarding commands to the Teensy (forward_to_teensy,
# esp32/src/main.cpp), so GUI seq numbers don't survive the hop, and every
# discrete command already has an observable confirmation in telemetry. This
# keeps all three ends untouched at the protocol level.
#
# Excluded on purpose (never wrapped in send_reliable): CMD_ID_PING and
# streamed setpoints (HIP_SUB_MIT, WHEEL_SUB_SEND) — superseded by the next
# frame by design, so "confirming" one is meaningless.


class CommandAlertBus(QObject):
    """Singleton: ReliableCommand emits here on final failure/rejection.
    MainWindow connects this once to the status-bar link-alert banner (the
    same one LinkHealthWidget uses) so every caller gets a visible failure
    with no per-tab wiring required."""

    alert = pyqtSignal(str)   # one-line message

    _instance: "CommandAlertBus | None" = None

    @classmethod
    def instance(cls) -> "CommandAlertBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class ReliableCommand(QObject):
    """Sends immediately, then retries on a timeout until confirm_predicate
    matches an incoming TelemetryBus packet or `retries` is exhausted. If
    reject_predicate matches first, stops immediately (no more retries) —
    the firmware told us "no", resending won't change that.

    Fire-and-forget: callers normally don't keep the returned instance, so
    it holds itself in `_in_flight` from construction until it finishes —
    a PyQt signal connection alone does *not* keep a plain QObject alive
    (verified: without this, the instance was garbage-collected before its
    own retry timer ever fired, silently disabling retries entirely). Only
    `_finish()` removes it, making it eligible for GC.
    """

    _in_flight: set["ReliableCommand"] = set()

    def __init__(self, send_fn, confirm_predicate, retries: int = 3, timeout_ms: int = 250,
                 on_fail=None, reject_predicate=None, label: str = "Command"):
        super().__init__()
        self._send_fn      = send_fn
        self._confirm       = confirm_predicate
        self._reject        = reject_predicate
        self._retries_left  = retries
        self._timeout_ms    = timeout_ms
        self._on_fail       = on_fail
        self._label         = label
        self._done          = False

        ReliableCommand._in_flight.add(self)

        from .telemetry_bus import TelemetryBus
        self._bus = TelemetryBus.instance()
        self._bus.packet.connect(self._on_packet)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

        self._send_fn()
        self._timer.start(self._timeout_ms)

    def cancel(self):
        """Stop retrying/listening without reporting a failure."""
        self._finish()

    def _finish(self):
        if self._done:
            return
        self._done = True
        self._timer.stop()
        self._bus.packet.disconnect(self._on_packet)
        ReliableCommand._in_flight.discard(self)

    def _fail(self, message: str):
        self._finish()
        if self._on_fail is not None:
            self._on_fail(message)
        else:
            CommandAlertBus.instance().alert.emit(message)

    def _on_packet(self, info: dict):
        if self._done:
            return
        if self._reject is not None and self._reject(info):
            self._fail(f"{self._label} rejected by firmware")
            return
        if self._confirm(info):
            self._finish()

    def _on_timeout(self):
        if self._done:
            return
        if self._retries_left > 0:
            self._retries_left -= 1
            self._send_fn()
            self._timer.start(self._timeout_ms)
        else:
            self._fail(f"{self._label} not confirmed after 3 retries — link degraded?")


def send_reliable(send_fn, confirm_predicate, retries: int = 3, timeout_ms: int = 250,
                   on_fail=None, reject_predicate=None, label: str = "Command") -> ReliableCommand:
    """Fire send_fn now; retry up to `retries` times, `timeout_ms` apart,
    until confirm_predicate(packet) matches a TelemetryBus packet. See
    ReliableCommand for the full contract. Returns the tracker instance —
    only needed if the caller wants to `.cancel()` it early."""
    return ReliableCommand(send_fn, confirm_predicate, retries=retries, timeout_ms=timeout_ms,
                            on_fail=on_fail, reject_predicate=reject_predicate, label=label)
