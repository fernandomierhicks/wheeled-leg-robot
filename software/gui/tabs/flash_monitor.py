import os
import shutil
import struct as _struct
from datetime import datetime
from pathlib import Path

import serial
from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QThread, QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMenu, QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from .port_manager import SerialPortManager
from .telem_format import (
    crc8 as _crc8, decode_telem_a as _decode_telem_a, decode_telem_b as _decode_telem_b,
    decode_telem_full as _decode_telem_full,
)
from .theme import BG, BORDER, BLUE, DIM, GREEN, ORANGE, RED, SURFACE, TEXT

# ── Paths ─────────────────────────────────────────────────────────────────────

_GUI_DIR = Path(__file__).parent.parent
_FW_ROOT = _GUI_DIR / ".." / ".." / "firmware" / "robot_teensy"
LOG_DIR  = _GUI_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

_pio_candidate = Path.home() / ".platformio" / "penv" / "Scripts" / "pio.exe"
PIO_EXE = str(_pio_candidate) if _pio_candidate.exists() else (shutil.which("pio") or "pio")

PIO_DIR = {
    "teensy": str((_FW_ROOT / "teensy").resolve()),
    "esp32":  str((_FW_ROOT / "esp32").resolve()),
}

FLASH_ACTIONS: dict[str, dict] = {
    "teensy": {
        "main":  [("▶  Flash Main",      ["run",  "-e", "teensy41",    "-t", "upload"])],
        "tests": [
            ("⊕  IMU Test",       ["test", "-e", "test_teensy", "-f", "test_imu", "--without-testing"]),
            ("⚙  AK45 Test",      ["test", "-e", "test_teensy", "-f", "test_hip_motor"]),
            ("⚙  ODrive Test",    ["test", "-e", "test_teensy", "-f", "test_wheel_motor"]),
            ("◈  RC iBUS Test",   ["test", "-e", "test_teensy", "-f", "test_rc"]),
            ("⇄  Comm USB",        ["test", "-e", "test_comm_usb", "-f", "test_comm_usb"]),
            ("⇄  T↔ESP32 Link",   ["test", "-e", "test_teensy", "-f", "test_telemetry"]),
            ("⚙  AK45 UART",      ["run",  "-e", "ak45_uart_demo", "-t", "upload"]),
            ("★  RGB LED",        ["run",  "-e", "rgb_led_demo",   "-t", "upload"]),
            ("♪  Buzzer",         ["test", "-e", "test_teensy",    "-f", "test_buzzer"]),
        ],
    },
    "esp32": {
        "main":  [("▶  Flash Main",      ["run",  "-e", "esp32dev",   "-t", "upload"])],
        "tests": [
            ("★  TFT Screen",     ["run",  "-e", "tft_benchmark", "-t", "upload"]),
            ("⊕  Laser Test",     ["run",  "-e", "vl53l1x_demo",  "-t", "upload"]),
            ("★  Neopixel",       ["run",  "-e", "neopixel_demo", "-t", "upload"]),
            ("⇄  T↔ESP32 Link",   ["test", "-e", "test_esp32",    "-f", "test_telemetry"]),
        ],
    },
}

DEVICE_COLOR = {"teensy": BLUE,   "esp32": ORANGE}
DEVICE_LABEL = {"teensy": "TEENSY 4.1", "esp32": "ESP32 DevKit V1"}
DEVICE_BAUD  = {"teensy": "115200",     "esp32": "921600"}

BAUD_OPTIONS = ["9600", "115200", "230400", "460800", "921600", "1000000", "1200000"]


def _detect_lan_ip() -> str | None:
    """Best-effort local LAN IP (no traffic sent — UDP connect() just picks
    the outbound interface). Used to bake the unicast telemetry target into
    the ESP32 build at flash time."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return None

# ── Serial reader thread ──────────────────────────────────────────────────────

def _decode_printable_serial_line(raw: bytes) -> str | None:
    """Decode direct serial diagnostics without treating framed binary as text."""
    raw = raw.rstrip(b"\r")
    if not raw or len(raw) > 512:
        return None
    if any(byte != 0x09 and not 0x20 <= byte <= 0x7E for byte in raw):
        return None
    text = raw.decode("ascii")
    return text if text.strip() else None


class SerialReader(QThread):
    line_received = pyqtSignal(str)
    raw_data      = pyqtSignal(bytes)
    disconnected  = pyqtSignal()

    def __init__(self, ser: serial.Serial, log_path: Path):
        super().__init__()
        self._ser     = ser
        self._running = True
        # Keep diagnostics buffered. Framed CommLink traffic is binary and can
        # contain incidental newlines, so it must never be line-flushed here.
        self._log     = open(log_path, "a", buffering=64 * 1024, encoding="utf-8")
        self._log.write(f"\n── session {datetime.now().isoformat(timespec='seconds')} ──\n")
        self._log.flush()  # make session boundaries visible while the GUI is running

    def run(self):
        buf = b""
        while self._running:
            try:
                n = self._ser.in_waiting
                if n:
                    chunk = self._ser.read(n)
                    self.raw_data.emit(chunk)
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        text = _decode_printable_serial_line(raw)
                        if text is not None:
                            self.line_received.emit(text)
                            self._log.write(text + "\n")
                    # Binary frames need not contain a newline. Bound the text
                    # candidate buffer while raw_data continues uninterrupted.
                    if len(buf) > 4096:
                        buf = buf[-512:]
                else:
                    self.msleep(5)
            except Exception:
                self._log.flush()
                self.disconnected.emit()
                return
        self._log.flush()
        self._log.close()

    def send(self, text: str):
        try:
            self._ser.write((text + "\r\n").encode())
        except Exception:
            pass

    def stop(self):
        self._running = False
        self.wait(300)

# ── Pio runner (QProcess, non-blocking) ───────────────────────────────────────

class PioRunner(QObject):
    line_ready = pyqtSignal(str, str)   # text, colour
    finished   = pyqtSignal(bool)       # success

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: QProcess | None = None

    def start(self, args: list[str], cwd: str, extra_env: dict[str, str] | None = None):
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(cwd)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        if extra_env:
            env = QProcessEnvironment.systemEnvironment()
            for k, v in extra_env.items():
                env.insert(k, v)
            self._proc.setProcessEnvironment(env)
        self._proc.readyReadStandardOutput.connect(self._on_data)
        self._proc.finished.connect(self._on_finished)
        self._proc.start(PIO_EXE, args)

    def kill(self):
        if self._proc and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.kill()
            self._proc.waitForFinished(2000)

    @property
    def running(self) -> bool:
        return bool(
            self._proc and
            self._proc.state() != QProcess.ProcessState.NotRunning
        )

    def _on_data(self):
        raw = bytes(self._proc.readAllStandardOutput())
        for line in raw.decode("utf-8", errors="replace").splitlines():
            self.line_ready.emit(line, DIM)

    def _on_finished(self, code: int, _status):
        ok  = (code == 0)
        msg = "✓ Success" if ok else f"✗ Failed (exit {code})"
        self.line_ready.emit(msg, GREEN if ok else RED)
        self.finished.emit(ok)

# ── Packet decoder ───────────────────────────────────────────────────────────

_COMM_MAGIC    = bytes([0xAA, 0x55])  # two-byte start marker
_COMM_END      = 0xEF
_HEADER_SZ     = 8   # magic(2) + type + version + source + seq + len_lo + len_hi
_OVERHEAD      = 10  # header(8) + checksum(1) + end(1)
_COMM_MAX_PAYLOAD = 512  # mirrors firmware CommLink.h COMM_MAX_PAYLOAD — a corrupted length
                          # field otherwise stalls _parse() waiting for bytes that will never
                          # arrive, swallowing subsequent legitimate datagrams into the search
                          # (Fix 2 equivalent; found via WiFi corruption-injection testing,
                          # Phase 9, UARTplat.md)
_TELEM_VERSION = 11  # must match TELEM_VERSION in shared/comm_protocol.h
_TELEM_A_LEN   = 118
_TELEM_B_LEN   = 129

_TYPE_NAMES  = {
    0x01: "TELEM", 0x02: "CMD", 0x03: "ACK", 0x04: "LOG",
    0x05: "CALIB", 0x06: "PARAM", 0x07: "TOF",
    0x10: "TELEM_A", 0x11: "TELEM_B",
    0x12: "LOG_INFO", 0x13: "LOG_DATA",
    0x14: "WIFI_DIAG", 0x15: "TELEM_FULL_WIFI",
    0x16: "COMMAND_RESULT",
}
_SRC_NAMES   = {0x01: "TEENSY", 0x02: "ESP32", 0x03: "PC"}
_LOG_LEVELS  = {0x01: "INFO", 0x02: "WARN", 0x03: "ERROR"}
_COMMAND_STATUS = {0: "APPLIED", 1: "ACCEPTED", 2: "REJECTED"}
_COMMAND_REASONS = {
    0: "NONE", 1: "BAD_VERSION", 2: "BAD_LENGTH", 3: "UNKNOWN_COMMAND",
    4: "INVALID_ENUM", 5: "INVALID_TARGET", 6: "NONFINITE",
    7: "WRONG_STATE", 8: "NOT_FOUND", 9: "READONLY",
    10: "GUARD_REJECTED", 11: "OPERATION_FAILED",
}
_FMT_WIFI_DIAG   = "<IbBBBIIHHHHBB"  # WifiDiagPayload V1, 26 bytes, packed/no padding
_WIFI_DIAG_LEN   = 26
_FMT_WIFI_DIAG_V2_EXT = "<BBHHHHH"   # V2 additions appended after the V1 26 bytes (12 bytes)
_WIFI_DIAG_V2_LEN     = 38
# _decode_telem_a/_b, _FMT_TELEM_A/_B, _STATE_NAMES, _FAULT_NAMES,
# _FAULT_DESCRIPTIONS now live in telem_format.py (shared with wlog_to_csv.py).


class PacketDecoder(QObject):
    packet_decoded = pyqtSignal(dict)

    def __init__(self, device: str = "", parent=None, skip_emit_types: frozenset[int] = frozenset()):
        super().__init__(parent)
        self._device      = device
        self._skip_emit_types = skip_emit_types  # ptypes to parse/count but not forward downstream
        self._buf         = b""
        self._telem_a_buf: dict | None = None  # holds decoded TELEM_A waiting for TELEM_B
        # Link-health counters for this transport (audit W1/W3/W4) — attached to
        # every emitted packet as link_* keys, surfaced in the Raw Data tab.
        self._crc_drops   = 0   # frames discarded on bad CRC-8
        self._seq_gaps    = 0   # frames lost per seq discontinuities
        self._pair_drops  = 0   # A/B halves rejected for non-adjacent seq
        self._seq_last: int | None = None

    def feed(self, data: bytes):
        self._buf += data
        self._parse()

    def _parse(self):
        while True:
            idx = self._buf.find(_COMM_MAGIC)
            if idx < 0:
                # Keep the last byte — could be start of a magic pair
                self._buf = self._buf[-1:] if self._buf else b""
                return
            if idx > 0:
                self._buf = self._buf[idx:]
            if len(self._buf) < _HEADER_SZ:
                return
            ptype   = self._buf[2]
            version = self._buf[3]
            source  = self._buf[4]
            seq     = self._buf[5]
            length  = self._buf[6] | (self._buf[7] << 8)
            if length > _COMM_MAX_PAYLOAD:
                self._crc_drops += 1
                self._buf = self._buf[1:]
                continue
            total   = _OVERHEAD + length
            if len(self._buf) < total:
                return
            payload  = self._buf[8:8 + length]
            checksum = self._buf[8 + length]
            end_byte = self._buf[9 + length]
            if end_byte != _COMM_END:
                self._buf = self._buf[1:]
                continue
            # Enforce the checksum (audit W1): a corrupted frame with intact
            # magic/END must not inject garbage telemetry — drop and count.
            if _crc8(self._buf[2:8 + length]) != checksum:
                self._crc_drops += 1
                self._buf = self._buf[total:]
                continue
            # Per-link loss metric (audit W3): seq gap between valid frames.
            if self._seq_last is not None:
                self._seq_gaps += (seq - self._seq_last - 1) & 0xFF
            self._seq_last = seq
            info: dict = {
                "ptype":     ptype,
                "version":   version,
                "source":    source,
                "type_name": _TYPE_NAMES.get(ptype, f"0x{ptype:02X}"),
                "src_name":  _SRC_NAMES.get(source, f"0x{source:02X}"),
                "seq":       seq,
                "length":    length,
                "checksum":  checksum,
                "crc_ok":    True,
                "link_crc_drops":  self._crc_drops,
                "link_seq_gaps":   self._seq_gaps,
                "link_pair_drops": self._pair_drops,
            }
            try:
                if ptype == 0x04 and length >= 2:
                    info["log_level"] = _LOG_LEVELS.get(payload[0], f"L{payload[0]}")
                    info["log_msg"]   = payload[1:].decode("utf-8", errors="replace")
                elif ptype == 0x06 and length >= 35:
                    param_id, value, min_val, max_val, flags = _struct.unpack_from("<HfffB", payload)
                    name = payload[15:35].rstrip(b'\x00').decode("utf-8", errors="replace")
                    info.update({
                        "param_id": param_id, "param_value": value,
                        "param_min": min_val, "param_max": max_val,
                        "param_flags": flags, "param_name": name,
                    })
                elif ptype == 0x05 and length >= 14:
                    axis, event, pos, mn, mx = _struct.unpack_from("<BBfff", payload)
                    info.update({
                        "calib_axis": axis, "calib_event": event,
                        "calib_pos_rad": pos, "calib_min_rad": mn, "calib_max_rad": mx,
                    })
                elif ptype == 0x10 and length == _TELEM_A_LEN:
                    if version != _TELEM_VERSION:
                        # Version mismatch — fall through to the normal emit path below
                        # so this reaches TelemetryBus and main.py's status-bar banner
                        # (set_version_mismatch) actually fires. Previously this branch
                        # always `continue`d before the emit, so a mismatch produced no
                        # telemetry AND no visible error — just silence.
                        info["version_mismatch"] = True
                        info["got_version"]       = version
                        info["expected_version"]  = _TELEM_VERSION
                    else:
                        # TELEM_A — store and wait for TELEM_B before emitting
                        self._telem_a_buf = {**info, **_decode_telem_a(payload)}
                        self._buf = self._buf[total:]
                        continue  # don't emit yet
                elif ptype == 0x11 and length == _TELEM_B_LEN and self._telem_a_buf is not None:
                    # TELEM_B — complete the packet and emit as a unified telemetry
                    # dict. Pairing requires B.seq == A.seq + 1 (audit W4): over UDP,
                    # reordering/loss can otherwise pair halves from different ticks.
                    a_seq = self._telem_a_buf.get("seq", -2)
                    if version == _TELEM_VERSION and ((a_seq + 1) & 0xFF) == seq:
                        info = {**self._telem_a_buf, **info, **_decode_telem_b(payload)}
                        info["ptype"]     = 0x01   # appear as TELEM to PacketInspector
                        info["type_name"] = "TELEM"
                    elif version == _TELEM_VERSION:
                        self._pair_drops += 1
                        info["link_pair_drops"] = self._pair_drops
                    self._telem_a_buf = None
                elif ptype == 0x12 and length >= 16:
                    # LOG_INFO — SD-log directory/transfer metadata (LogInfoPayload).
                    # Consumed by LogTransferManager — never routed to TelemetryBus.
                    info_type, file_index, file_size, total_chunks, crc32, status = \
                        _struct.unpack_from("<BHIIIB", payload)
                    info.update({
                        "log_info_type":   info_type,
                        "log_file_index":  file_index,
                        "log_file_size":   file_size,
                        "log_total_chunks": total_chunks,
                        "log_crc32":       crc32,
                        "log_status":      status,
                    })
                elif ptype == 0x13 and length >= 8:
                    # LOG_DATA — one .wlog file chunk (LogDataHeader + raw bytes).
                    # Consumed by LogTransferManager — never routed to TelemetryBus.
                    file_index, chunk_index, data_len = _struct.unpack_from("<HIH", payload)
                    info.update({
                        "log_file_index":  file_index,
                        "log_chunk_index": chunk_index,
                        "log_data":        bytes(payload[8:8 + data_len]),
                    })
                elif ptype == 0x14 and length >= _WIFI_DIAG_LEN:
                    # WIFI_DIAG — ESP32-only link/loop diagnostics (WifiDiagPayload).
                    (esp_uptime_ms, rssi_dbm, wifi_channel, wifi_status, tx_power_raw,
                     free_heap, min_free_heap, loop_max_us, udp_send_max_us,
                     wifi_reconnect_count, udp_send_fail_count,
                     build_variant_flags, active_telem_transport) = \
                        _struct.unpack_from(_FMT_WIFI_DIAG, payload)
                    info.update({
                        "wifi_esp_uptime_ms":        esp_uptime_ms,
                        "wifi_rssi_dbm":             rssi_dbm,
                        "wifi_channel":              wifi_channel,
                        "wifi_status":               wifi_status,
                        "wifi_tx_power_raw":         tx_power_raw,
                        "wifi_free_heap":            free_heap,
                        "wifi_min_free_heap":        min_free_heap,
                        "wifi_loop_max_us":          loop_max_us,
                        "wifi_udp_send_max_us":      udp_send_max_us,
                        "wifi_reconnect_count":      wifi_reconnect_count,
                        "wifi_udp_send_fail_count":  udp_send_fail_count,
                        "wifi_build_variant_flags":  build_variant_flags,
                        "wifi_active_telem_transport": active_telem_transport,
                    })
                    if length >= _WIFI_DIAG_V2_LEN:
                        # V2 — Teensy-link status, the "ESP32 alive" carrier: this
                        # packet arrives at 5 Hz independent of the Teensy link, so
                        # it's what lets the GUI show ESP32-alive-but-Teensy-silent.
                        (teensy_link_up, _reserved2, ms_since_teensy, uart_crc_drops,
                         uart_seq_gaps, uplink_queue_drops, tcp_send_max_us) = \
                            _struct.unpack_from(_FMT_WIFI_DIAG_V2_EXT, payload, _WIFI_DIAG_LEN)
                        info.update({
                            "wifi_teensy_link_up":     bool(teensy_link_up),
                            "wifi_ms_since_teensy":    ms_since_teensy,
                            "wifi_uart_crc_drops":     uart_crc_drops,
                            "wifi_uart_seq_gaps":      uart_seq_gaps,
                            "wifi_uplink_queue_drops": uplink_queue_drops,
                            "wifi_tcp_send_max_us":    tcp_send_max_us,
                        })
                elif ptype == 0x15 and length == 247:
                    # TELEM_FULL_WIFI — WIFI_TELEM_COMBINED variant: the full
                    # TelemetryPayload as one datagram. Remap to look like a normal
                    # TELEM packet (transparent to every existing tab).
                    if version != _TELEM_VERSION:
                        info["version_mismatch"] = True
                        info["got_version"]      = version
                        info["expected_version"] = _TELEM_VERSION
                    else:
                        info.update(_decode_telem_full(payload))
                        info["ptype"]     = 0x01
                        info["type_name"] = "TELEM"
                elif ptype == 0x16 and version == 1 and length == 8:
                    request_id, command_id, status, reason, state = _struct.unpack_from("<IBBBB", payload)
                    info.update({
                        "request_id": request_id,
                        "command_id": command_id,
                        "command_status": status,
                        "command_status_name": _COMMAND_STATUS.get(status, str(status)),
                        "command_reason": reason,
                        "command_reason_name": _COMMAND_REASONS.get(reason, str(reason)),
                        "command_state": state,
                        "command_accepted": status != 2,
                    })
            except Exception:
                pass
            self.packet_decoded.emit(info)
            if ptype not in self._skip_emit_types:
                from .source_manager import SourceManager
                if ptype in (0x12, 0x13):
                    # Same duplicate-suppression as telemetry below: with Teensy USB,
                    # ESP32 USB, and WiFi all potentially connected simultaneously,
                    # every transport independently decodes and would otherwise
                    # re-emit the identical LOG_INFO/LOG_DATA packet (e.g. one
                    # directory ENTRY per active link) — only the active source emits.
                    if SourceManager.instance().is_active(self._device):
                        from .log_bus import LogPacketBus
                        LogPacketBus.instance().packet.emit(info)
                elif ptype == 0x16:
                    if SourceManager.instance().is_active(self._device):
                        from .comm_commands import CommandResultBus
                        CommandResultBus.instance().result.emit(info)
                else:
                    from .telemetry_bus import TelemetryBus
                    bus = TelemetryBus.instance()
                    if not bus.playback_active and SourceManager.instance().is_active(self._device):
                        bus.publish(info)
            self._buf = self._buf[total:]


class PacketInspector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending_info: dict | None = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(100)
        self._refresh_timer.timeout.connect(self._flush_pending)
        self._refresh_timer.start()

        frame = QFrame(self)
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet(
            f"QFrame {{ background: #0a0a14; border: 1px solid {BORDER}; border-radius: 3px; }}"
        )

        def _kv(name: str):
            k = QLabel(name + ":")
            k.setStyleSheet(f"color: {DIM}; font-size: 10px; border: none;")
            v = QLabel("—")
            v.setStyleSheet(f"color: {TEXT}; font-size: 10px; font-weight: bold; border: none;")
            v.setMinimumWidth(52)
            return k, v

        self._v: dict[str, QLabel] = {}
        all_keys = ["Type", "Src", "Seq", "Len", "CRC",
                    "Pitch", "Rate", "WheelV", "State", "T(ms)",
                    "HipL", "HipR", "WhlTauL", "WhlTauR", "Fault"]
        for key in all_keys:
            _, val = _kv(key)
            self._v[key] = val

        def _row(keys):
            row = QHBoxLayout()
            row.setSpacing(2)
            for k in keys:
                lbl_k, _ = _kv(k)
                row.addWidget(lbl_k)
                row.addWidget(self._v[k])
                row.addSpacing(6)
            row.addStretch()
            return row

        title = QLabel("Last Packet")
        title.setStyleSheet(f"color: {DIM}; font-size: 10px; font-style: italic; border: none;")

        inner = QVBoxLayout(frame)
        inner.setContentsMargins(6, 3, 6, 3)
        inner.setSpacing(1)
        inner.addWidget(title)
        inner.addLayout(_row(["Type", "Src", "Seq", "Len", "CRC"]))
        inner.addLayout(_row(["Pitch", "Rate", "WheelV", "State", "T(ms)"]))
        inner.addLayout(_row(["HipL", "HipR", "WhlTauL", "WhlTauR", "Fault"]))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

    def _set(self, key: str, text: str, color: str = TEXT):
        lbl = self._v[key]
        lbl.setText(text)
        lbl.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold; border: none;")

    def update_packet(self, info: dict):
        self._pending_info = info

    def _flush_pending(self):
        info = self._pending_info
        self._pending_info = None
        if info is None:
            return
        self._set("Type", info.get("type_name", "—"))
        self._set("Src",  info.get("src_name",  "—"))
        self._set("Seq",  str(info.get("seq", "—")))
        self._set("Len",  str(info.get("length", "—")))
        crc = info.get("crc_ok")
        self._set("CRC", "OK" if crc else "FAIL", GREEN if crc else RED)
        if info.get("ptype") == 0x01:
            self._set("Pitch",  f"{info['pitch_rad']:+.3f}")
            self._set("Rate",   f"{info['pitch_rate_rads']:+.3f}")
            self._set("WheelV", f"{info['wheel_vel_avg']:+.3f}")
            self._set("State",  info.get("state_name", "—"))
            self._set("T(ms)",  str(info.get("timestamp_ms", "—")))
            self._set("HipL",   f"{info['hip_l_pos_rad']:+.3f}")
            self._set("HipR",   f"{info['hip_r_pos_rad']:+.3f}")
            self._set("WhlTauL", f"{info['whl_tau_l']:+.3f}")
            self._set("WhlTauR", f"{info['whl_tau_r']:+.3f}")
            fault = info.get("fault_name", "—")
            self._set("Fault", fault, RED if fault != "NONE" else TEXT)
        else:
            for k in ["Pitch", "Rate", "WheelV", "State", "T(ms)", "HipL", "HipR", "WhlTauL", "WhlTauR", "Fault"]:
                self._set(k, "—")


# ── Combined device panel ─────────────────────────────────────────────────────

class DevicePanel(QWidget):
    def __init__(self, device: str):
        super().__init__()
        self._device      = device
        self._reader: SerialReader | None = None
        self._pio         = PioRunner(self)
        self._auto_scroll = True
        self._flashing    = False
        # Set only by the user's own Disconnect click (not by flash-release or a
        # real link drop) so _auto_scan() doesn't silently reopen the port a
        # moment later — the ESP32's USB-serial bridge stays enumerated across a
        # target reset, so auto-scan would otherwise fight every Disconnect.
        self._user_disconnected = False
        self._log_path    = LOG_DIR / f"{device}.log"
        self._decoder     = PacketDecoder(device, self)

        pm = SerialPortManager.instance()
        pm.port_released.connect(self._on_port_released)
        self._pio.line_ready.connect(lambda t, c: self._append(t, c))
        self._pio.finished.connect(self._on_flash_done)

        color = DEVICE_COLOR.get(device, TEXT)

        # ── Header ──────────────────────────────────────────────────────────
        title = QLabel(DEVICE_LABEL.get(device, device.upper()))
        title.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        self._status_lbl = QLabel("● Closed")
        self._status_lbl.setStyleSheet(f"color: {RED};")
        hdr = QHBoxLayout()
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(self._status_lbl)

        # ── Port / baud row ──────────────────────────────────────────────────
        self._port_combo = QComboBox()
        self._port_combo.setMinimumWidth(120)
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(30)
        refresh_btn.clicked.connect(self._refresh_ports)
        self._baud_combo = QComboBox()
        self._baud_combo.addItems(BAUD_OPTIONS)
        self._baud_combo.setCurrentText(DEVICE_BAUD.get(device, "115200"))
        self._baud_combo.setFixedWidth(95)
        self._conn_btn = QPushButton("Connect")
        self._conn_btn.setCheckable(True)
        self._conn_btn.setFixedWidth(95)
        self._conn_btn.clicked.connect(self._toggle_connect)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Port:"))
        port_row.addWidget(self._port_combo)
        port_row.addWidget(refresh_btn)
        port_row.addSpacing(6)
        port_row.addWidget(QLabel("Baud:"))
        port_row.addWidget(self._baud_combo)
        port_row.addStretch()
        port_row.addWidget(self._conn_btn)

        # ── Flash buttons ────────────────────────────────────────────────────
        self._flash_btns: list[QPushButton] = []
        self._cancel_btn = QPushButton("✕  Cancel Flash")
        self._cancel_btn.setStyleSheet(f"color: {RED}; border-color: {RED};")
        self._cancel_btn.hide()
        self._cancel_btn.clicked.connect(self._cancel_flash)

        actions = FLASH_ACTIONS.get(device, {"main": [], "tests": []})

        main_col = QVBoxLayout()
        main_col.setSpacing(4)
        for label, args in actions["main"]:
            btn = QPushButton(label)
            btn.setMinimumHeight(40)
            btn.setStyleSheet(
                f"QPushButton {{ color: {color}; border: 1px solid {color}; "
                f"font-weight: bold; font-size: 13px; padding: 0 12px; }}"
                f"QPushButton:hover {{ background: {color}22; }}"
                f"QPushButton:disabled {{ color: {DIM}; border-color: {DIM}; }}"
            )
            btn.clicked.connect(lambda _, a=args: self._flash(a))
            self._flash_btns.append(btn)
            main_col.addWidget(btn)
        main_col.addStretch()

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {BORDER};")

        tests_lbl = QLabel("Tests")
        tests_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px; font-style: italic;")

        test_grid = QGridLayout()
        test_grid.setSpacing(4)
        for i, (label, args) in enumerate(actions["tests"]):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, a=args: self._flash(a))
            self._flash_btns.append(btn)
            test_grid.addWidget(btn, i // 2, i % 2)

        tests_col = QVBoxLayout()
        tests_col.setSpacing(3)
        tests_col.addWidget(tests_lbl)
        tests_col.addLayout(test_grid)

        flash_row = QHBoxLayout()
        flash_row.addLayout(main_col)
        flash_row.addSpacing(8)
        flash_row.addWidget(sep)
        flash_row.addSpacing(8)
        flash_row.addLayout(tests_col)
        flash_row.addStretch()
        flash_row.addWidget(self._cancel_btn)

        # ── Output ──────────────────────────────────────────────────────────
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setMaximumBlockCount(5000)
        self._output.setFont(QFont("Consolas", 10))
        self._output.setStyleSheet(
            f"background: #0a0a14; color: {TEXT}; border: 1px solid {BORDER};"
        )
        self._output.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._output.customContextMenuRequested.connect(self._show_output_menu)

        # ── Packet inspector ─────────────────────────────────────────────────
        self._inspector = PacketInspector()
        self._decoder.packet_decoded.connect(self._inspector.update_packet)

        # ── Send row ─────────────────────────────────────────────────────────
        self._send_input = QLineEdit()
        self._send_input.setPlaceholderText("Send to device (Enter)…")
        self._send_input.returnPressed.connect(self._send)
        send_btn = QPushButton("Send")
        send_btn.setFixedWidth(55)
        send_btn.clicked.connect(self._send)
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(55)
        clear_btn.clicked.connect(self._output.clear)
        scroll_chk = QCheckBox("Auto-scroll")
        scroll_chk.setChecked(True)
        scroll_chk.toggled.connect(lambda v: setattr(self, "_auto_scroll", v))
        self._scroll_chk = scroll_chk

        bot_row = QHBoxLayout()
        bot_row.addWidget(self._send_input)
        bot_row.addWidget(send_btn)
        bot_row.addWidget(clear_btn)
        bot_row.addWidget(scroll_chk)

        # ── Layout ──────────────────────────────────────────────────────────
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        lay.addLayout(hdr)
        lay.addLayout(port_row)
        lay.addLayout(flash_row)
        lay.addWidget(self._output)
        lay.addWidget(self._inspector)
        lay.addLayout(bot_row)

        # ── Auto-scan ────────────────────────────────────────────────────────
        self._scan_timer = QTimer(self)
        self._scan_timer.timeout.connect(self._auto_scan)
        self._scan_timer.start(2000)
        self._refresh_ports()
        self._auto_scan()

    # ── Port management ───────────────────────────────────────────────────────

    def _refresh_ports(self):
        pm   = SerialPortManager.instance()
        auto = pm.scan()
        prev = self._port_combo.currentText().split(" ")[0]
        self._port_combo.blockSignals(True)
        self._port_combo.clear()
        if self._device in auto:
            self._port_combo.addItem(f"{auto[self._device]} (auto)")
        for p in pm.all_ports():
            if not (self._device in auto and p == auto[self._device]):
                self._port_combo.addItem(p)
        idx = self._port_combo.findText(prev)
        if idx < 0:
            idx = self._port_combo.findText(prev + " (auto)")
        if idx >= 0:
            self._port_combo.setCurrentIndex(idx)
        self._port_combo.blockSignals(False)

    def _port_path(self) -> str:
        return self._port_combo.currentText().split(" ")[0]

    def _auto_scan(self):
        if self._reader or self._flashing or self._user_disconnected:
            return
        if self._device in SerialPortManager.instance().scan():
            self._refresh_ports()
            self._open_port()

    def _toggle_connect(self, checked: bool):
        if checked:
            self._user_disconnected = False
            self._open_port()
        else:
            self._user_disconnected = True
            self._close_port()

    def _open_port(self):
        port = self._port_path()
        if not port:
            return
        baud = int(self._baud_combo.currentText())
        ser  = SerialPortManager.instance().acquire(self._device, port, baud)
        if ser is None:
            self._append(f"[error] could not open {port}", RED)
            self._conn_btn.setChecked(False)
            return
        if self._device == "teensy":
            # Teensy's if(Serial) checks DTR; assert it so the firmware sends USB telemetry.
            # Safe for Teensy — unlike ESP32 which has an autoreset circuit triggered by DTR.
            try:
                ser.dtr = True
            except Exception:
                pass
        self._reader = SerialReader(ser, self._log_path)
        self._reader.line_received.connect(lambda t: self._append(t, TEXT))
        self._reader.raw_data.connect(self._decoder.feed)
        self._reader.disconnected.connect(self._on_disconnected)
        self._reader.start()
        self._set_status(True, "Open")
        self._conn_btn.setText("Disconnect")

    def _close_port(self, reason: str = "Closed", *, external: bool = False):
        if self._reader:
            self._reader.stop()
            self._reader = None
        if not external:
            SerialPortManager.instance().release(self._device)
        color = ORANGE if "flash" in reason.lower() or "releas" in reason.lower() else RED
        self._status_lbl.setStyleSheet(f"color: {color};")
        self._status_lbl.setText(f"● {reason}")
        self._conn_btn.setChecked(False)
        self._conn_btn.setText("Connect")

    def shutdown(self):
        """Stop background work synchronously before QApplication exits."""
        self._scan_timer.stop()
        self._flashing = True
        if self._pio.running:
            self._pio.kill()
        self._close_port("Closed", external=False)

    def _set_status(self, connected: bool, label: str):
        color = GREEN if connected else RED
        self._status_lbl.setStyleSheet(f"color: {color};")
        self._status_lbl.setText(f"● {label}")
        self._conn_btn.setChecked(connected)

    def _on_port_released(self, device: str):
        if device == self._device and self._reader:
            self._append("[port released for flashing — will reconnect when done]", ORANGE)
            self._close_port("Released", external=True)

    def _on_disconnected(self):
        self._append("[disconnected]", RED)
        self._close_port("Disconnected", external=False)

    # ── Flash ─────────────────────────────────────────────────────────────────

    def _flash(self, base_args: list[str]):
        if self._pio.running:
            return
        port = self._port_path()

        # release serial port before pio takes it; block auto-reconnect until done
        self._flashing = True
        self._close_port("Flashing…", external=False)
        self._output.clear()

        args = list(base_args)
        if port:
            args += ["--upload-port", port]

        # ESP32 main firmware defaults to unicast WiFi telemetry (WIFI_TELEM_MODE=1
        # in config.h) — it needs this machine's current LAN IP baked in at flash
        # time via a build flag, since it can change (DHCP) and isn't known at
        # compile time otherwise.
        extra_env = None
        if self._device == "esp32" and "esp32dev" in args and "upload" in args:
            ip = _detect_lan_ip()
            if ip:
                self._append(f"[unicast telemetry target: {ip}]", DIM)
                extra_env = {"PLATFORMIO_BUILD_FLAGS": f'-DWIFI_UNICAST_IP=\\"{ip}\\"'}
            else:
                self._append("[warning: could not detect LAN IP — unicast telemetry needs "
                              "WIFI_UNICAST_IP set manually]", ORANGE)

        self._append(f"$ pio {' '.join(args)}", BLUE)
        self._set_flash_busy(True)
        self._pio.start(args, PIO_DIR[self._device], extra_env)

    def _cancel_flash(self):
        self._pio.kill()

    def _on_flash_done(self, ok: bool):
        self._set_flash_busy(False)
        QTimer.singleShot(3500, self._reconnect_after_flash)

    def _reconnect_after_flash(self):
        self._flashing = False
        self._auto_scan()

    def _set_flash_busy(self, busy: bool):
        for btn in self._flash_btns:
            btn.setEnabled(not busy)
        self._cancel_btn.setVisible(busy)
        self._conn_btn.setEnabled(not busy)

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _send(self):
        text = self._send_input.text().strip()
        if text and self._reader:
            self._reader.send(text)
            self._send_input.clear()

    def _append(self, text: str, color: str = TEXT):
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cur = self._output.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.setCharFormat(fmt)
        cur.insertText(text + "\n")
        if self._auto_scroll:
            sb = self._output.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _show_output_menu(self, pos):
        menu = self._output.createStandardContextMenu()
        menu.addSeparator()
        autoscroll_act = menu.addAction("Auto-scroll")
        autoscroll_act.setCheckable(True)
        autoscroll_act.setChecked(self._auto_scroll)
        autoscroll_act.toggled.connect(self._scroll_chk.setChecked)
        clear_act = menu.addAction("Clear")
        clear_act.triggered.connect(self._output.clear)
        menu.exec(self._output.mapToGlobal(pos))

# ── Tab ───────────────────────────────────────────────────────────────────────

class FlashMonitorTab(QWidget):
    def __init__(self):
        super().__init__()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._panels = [DevicePanel("teensy"), DevicePanel("esp32")]
        for panel in self._panels:
            splitter.addWidget(panel)
        splitter.setSizes([1, 1])
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(splitter)

    def shutdown(self):
        for panel in self._panels:
            panel.shutdown()
