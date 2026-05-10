import os
import shutil
import struct as _struct
from datetime import datetime
from pathlib import Path

import serial
from PyQt6.QtCore import QObject, QProcess, QThread, QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from port_manager import SerialPortManager
from theme import BG, BORDER, BLUE, DIM, GREEN, ORANGE, RED, SURFACE, TEXT

# ── Paths ─────────────────────────────────────────────────────────────────────

_GUI_DIR = Path(__file__).parent
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
        "main":  [("Flash Main",         ["run",  "-e", "teensy41",    "-t", "upload"])],
        "tests": [
            ("Flash IMU Test",    ["test", "-e", "test_teensy", "-f", "test_imu"]),
            ("Flash AK45 Test",   ["test", "-e", "test_teensy", "-f", "test_hip_motor"]),
            ("Flash ODrive Test", ["test", "-e", "test_teensy", "-f", "test_wheel_motor"]),
            ("Flash RC Test",     ["test", "-e", "test_teensy", "-f", "test_rc"]),
            ("Flash Teensy-ESP32 Interface Test", ["test", "-e", "test_teensy", "-f", "test_telemetry"]),
            ("Flash UART HW Test",                ["test", "-e", "test_teensy", "-f", "test_uart_hw"]),
            ("Flash Test Neopixel",               ["run",  "-e", "neopixel_demo", "-t", "upload"]),
            ("Flash iBUS Test",                   ["run",  "-e", "ibus_demo",     "-t", "upload"]),
        ],
    },
    "esp32": {
        "main":  [("Flash Main",         ["run",  "-e", "esp32s3",    "-t", "upload"])],
        "tests": [
            ("Flash OLED Test",                   ["run",  "-e", "esp32dev",      "-t", "upload"]),
            ("Flash Laser Test",                  ["run",  "-e", "vl53l1x_demo",  "-t", "upload"]),
            ("Flash Test Neopixel",               ["run",  "-e", "neopixel_demo", "-t", "upload"]),
            ("Flash Teensy-ESP32 Interface Test", ["test", "-e", "test_esp32",    "-f", "test_telemetry"]),
            ("Flash UART HW Test",                ["test", "-e", "test_esp32",    "-f", "test_uart_hw"]),
        ],
    },
}

DEVICE_COLOR = {"teensy": BLUE,   "esp32": ORANGE}
DEVICE_LABEL = {"teensy": "TEENSY 4.1", "esp32": "ESP32 DevKit V1"}
DEVICE_BAUD  = {"teensy": "115200",     "esp32": "115200"}

BAUD_OPTIONS = ["9600", "115200", "230400", "460800", "1000000", "1200000"]

# ── Serial reader thread ──────────────────────────────────────────────────────

class SerialReader(QThread):
    line_received = pyqtSignal(str)
    raw_data      = pyqtSignal(bytes)
    disconnected  = pyqtSignal()

    def __init__(self, ser: serial.Serial, log_path: Path):
        super().__init__()
        self._ser     = ser
        self._running = True
        self._log     = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
        self._log.write(f"\n── session {datetime.now().isoformat(timespec='seconds')} ──\n")

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
                        text = raw.decode("utf-8", errors="replace").rstrip("\r")
                        self.line_received.emit(text)
                        self._log.write(text + "\n")
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

    def start(self, args: list[str], cwd: str):
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(cwd)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_data)
        self._proc.finished.connect(self._on_finished)
        self._proc.start(PIO_EXE, args)

    def kill(self):
        if self._proc and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.kill()

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

_COMM_START  = 0xFF
_COMM_END    = 0xFE
_HEADER_SZ   = 7   # start + type + version + source + seq + len_lo + len_hi
_OVERHEAD    = 9   # header(7) + checksum(1) + end(1)

_TYPE_NAMES  = {0x01: "TELEM", 0x02: "CMD", 0x03: "ACK", 0x04: "LOG"}
_SRC_NAMES   = {0x01: "TEENSY", 0x02: "ESP32", 0x03: "PC"}
_STATE_NAMES = {0: "STARTUP", 1: "CALIB", 2: "RUNNING", 3: "ESTOP"}
_LOG_LEVELS  = {0x01: "INFO", 0x02: "WARN", 0x03: "ERROR"}


class PacketDecoder(QObject):
    packet_decoded = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buf = b""

    def feed(self, data: bytes):
        self._buf += data
        self._parse()

    def _parse(self):
        while True:
            idx = self._buf.find(bytes([_COMM_START]))
            if idx < 0:
                self._buf = b""
                return
            if idx > 0:
                self._buf = self._buf[idx:]
            if len(self._buf) < _HEADER_SZ:
                return
            ptype   = self._buf[1]
            version = self._buf[2]
            source  = self._buf[3]
            seq     = self._buf[4]
            length  = self._buf[5] | (self._buf[6] << 8)
            total   = _OVERHEAD + length
            if len(self._buf) < total:
                return
            payload  = self._buf[7:7 + length]
            checksum = self._buf[7 + length]
            end_byte = self._buf[7 + length + 1]
            if end_byte != _COMM_END:
                self._buf = self._buf[1:]
                continue
            xor = ptype ^ version ^ source ^ seq ^ self._buf[5] ^ self._buf[6]
            for b in payload:
                xor ^= b
            info: dict = {
                "ptype":     ptype,
                "version":   version,
                "source":    source,
                "type_name": _TYPE_NAMES.get(ptype, f"0x{ptype:02X}"),
                "src_name":  _SRC_NAMES.get(source, f"0x{source:02X}"),
                "seq":       seq,
                "length":    length,
                "checksum":  checksum,
                "crc_ok":    (xor == checksum),
            }
            try:
                if ptype == 0x04 and length >= 2:
                    info["log_level"] = _LOG_LEVELS.get(payload[0], f"L{payload[0]}")
                    info["log_msg"]   = payload[1:].decode("utf-8", errors="replace")
                elif ptype == 0x01 and length >= 41:
                    ts, pitch, pitch_rate, wheel_vel, hip_l, hip_r, cmd_l, cmd_r, roll, yaw, state = \
                        _struct.unpack_from("<IfffffffffB", payload)
                    info.update({
                        "timestamp_ms":    ts,
                        "pitch_rad":       pitch,
                        "pitch_rate_rads": pitch_rate,
                        "wheel_vel_avg":   wheel_vel,
                        "hip_l_pos_rad":   hip_l,
                        "hip_r_pos_rad":   hip_r,
                        "cmd_l":           cmd_l,
                        "cmd_r":           cmd_r,
                        "roll_rad":        roll,
                        "yaw_rad":         yaw,
                        "state_name":      _STATE_NAMES.get(state, str(state)),
                    })
            except Exception:
                pass
            self.packet_decoded.emit(info)
            from telemetry_bus import TelemetryBus
            TelemetryBus.instance().packet.emit(info)
            self._buf = self._buf[total:]


class PacketInspector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

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
                    "HipL", "HipR", "CmdL", "CmdR"]
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
        inner.addLayout(_row(["HipL", "HipR", "CmdL", "CmdR"]))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

    def _set(self, key: str, text: str, color: str = TEXT):
        lbl = self._v[key]
        lbl.setText(text)
        lbl.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold; border: none;")

    def update_packet(self, info: dict):
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
            self._set("CmdL",   f"{info['cmd_l']:+.3f}")
            self._set("CmdR",   f"{info['cmd_r']:+.3f}")
        else:
            for k in ["Pitch", "Rate", "WheelV", "State", "T(ms)", "HipL", "HipR", "CmdL", "CmdR"]:
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
        self._log_path    = LOG_DIR / f"{device}.log"
        self._decoder     = PacketDecoder(self)

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
        self._cancel_btn = QPushButton("Cancel Flash")
        self._cancel_btn.setStyleSheet(f"color: {RED}; border-color: {RED};")
        self._cancel_btn.hide()
        self._cancel_btn.clicked.connect(self._cancel_flash)

        actions = FLASH_ACTIONS.get(device, {"main": [], "tests": []})

        main_col = QVBoxLayout()
        main_col.setSpacing(4)
        for label, args in actions["main"]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, a=args: self._flash(a))
            self._flash_btns.append(btn)
            main_col.addWidget(btn)
        main_col.addStretch()

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {BORDER};")

        test_grid = QGridLayout()
        test_grid.setSpacing(4)
        for i, (label, args) in enumerate(actions["tests"]):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, a=args: self._flash(a))
            self._flash_btns.append(btn)
            test_grid.addWidget(btn, i // 2, i % 2)

        flash_row = QHBoxLayout()
        flash_row.addLayout(main_col)
        flash_row.addSpacing(8)
        flash_row.addWidget(sep)
        flash_row.addSpacing(8)
        flash_row.addLayout(test_grid)
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
        if self._reader or self._flashing:
            return
        if self._device in SerialPortManager.instance().scan():
            self._refresh_ports()
            self._open_port()

    def _toggle_connect(self, checked: bool):
        if checked:
            self._open_port()
        else:
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
        self._close_port("Disconnected", external=True)

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

        self._append(f"$ pio {' '.join(args)}", BLUE)
        self._set_flash_busy(True)
        self._pio.start(args, PIO_DIR[self._device])

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

# ── Tab ───────────────────────────────────────────────────────────────────────

class FlashMonitorTab(QWidget):
    def __init__(self):
        super().__init__()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(DevicePanel("teensy"))
        splitter.addWidget(DevicePanel("esp32"))
        splitter.setSizes([1, 1])
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(splitter)
