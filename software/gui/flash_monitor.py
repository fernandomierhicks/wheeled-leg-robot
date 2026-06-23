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
        "main":  [("▶  Flash Main",      ["run",  "-e", "teensy41",    "-t", "upload"])],
        "tests": [
            ("⊕  IMU Test",       ["test", "-e", "test_teensy", "-f", "test_imu"]),
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
            ("★  TFT Screen",     ["run",  "-e", "esp32dev",      "-t", "upload"]),
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

_COMM_MAGIC    = bytes([0xAA, 0x55])  # two-byte start marker
_COMM_END      = 0xEF
_HEADER_SZ     = 8   # magic(2) + type + version + source + seq + len_lo + len_hi
_OVERHEAD      = 10  # header(8) + checksum(1) + end(1)
_TELEM_VERSION = 6   # must match TELEM_VERSION in shared/comm_protocol.h
_TELEM_A_LEN   = 118
_TELEM_B_LEN   = 120

_TYPE_NAMES  = {
    0x01: "TELEM", 0x02: "CMD", 0x03: "ACK", 0x04: "LOG",
    0x05: "CALIB", 0x06: "PARAM", 0x07: "TOF",
    0x10: "TELEM_A", 0x11: "TELEM_B",
}
_SRC_NAMES   = {0x01: "TEENSY", 0x02: "ESP32", 0x03: "PC"}
_STATE_NAMES = {0: "STARTUP", 1: "CALIBRATION", 2: "STANDBY", 3: "RUNNING", 4: "ESTOP", 5: "MANUAL", 6: "CMD_REJECT", 7: "JUMPING"}
_FAULT_NAMES = {
    0x00: "NONE",
    0x01: "IMU_ERROR",
    0x02: "HIP_INIT_TIMEOUT",
    0x03: "HIP_FEEDBACK_LOST",
    0x04: "HIP_LARGE_POS_CMD",
    0x05: "CALIBRATION_TIMEOUT",
    0x06: "HUMAN_ESTOP",
    0x07: "PARAM_OUT_OF_BOUNDS",
    0x08: "PITCH_WATCHDOG",
    0x09: "WHEEL_RUNAWAY",
}
_FAULT_DESCRIPTIONS = {
    0x00: "",
    0x01: "IMU reported ERROR during startup",
    0x02: "No CAN reply from hip motors within 2 s of boot",
    0x03: "Hip CAN feedback timed out during operation (> 20 ms)",
    0x04: "Commanded hip position jump exceeded MAX_HIP_DELTA_RAD",
    0x05: "Hardstop not found within CALIB_SAFETY_BOUND_RAD",
    0x06: "Human triggered ESTOP",
    0x07: "Param write out of bounds — value outside [min, max]",
    0x08: "|pitch| > 50° for > 200 ms — robot tipped",
    0x09: "Wheel velocity exceeded 2× soft governor limit",
}
_LOG_LEVELS  = {0x01: "INFO", 0x02: "WARN", 0x03: "ERROR"}

# Struct formats for split telemetry (V6, packed, little-endian)
# TELEM_A: bytes 0-117 of TelemetryPayload
_FMT_TELEM_A = "<I9fBB3f14HB6f2I3B"
# TELEM_B: bytes 118-237 of TelemetryPayload
# 4H tof_dist, 2f rates, 3f accel, 2f hip_vel, 10f hip_cmd, 6f ctrl, 3f ff, H health, BB diag, I loop
_FMT_TELEM_B = "<4H2f3f2f10f6f3fHBBI"


def _decode_telem_a(payload: bytes) -> dict:
    (ts,
     pitch, pitch_rate, wheel_vel, hip_l, hip_r, whl_tau_l, whl_tau_r, roll, yaw,
     state, fault,
     test_val, hip_l_curr, hip_r_curr,
     *ibus_and_alive,
     wm_l_vel, wm_r_vel, wm_l_pos, wm_r_pos, wm_l_vbus, wm_r_vbus,
     wm_l_err, wm_r_err,
     wm_l_st, wm_r_st, wm_mode) = _struct.unpack(_FMT_TELEM_A, payload)
    ibus_ch   = list(ibus_and_alive[:14])
    ibus_alive = bool(ibus_and_alive[14])
    _NO_DATA = 0xFFFF
    return {
        "timestamp_ms":    ts,
        "pitch_rad":       pitch,
        "pitch_rate_rads": pitch_rate,
        "wheel_vel_avg":   wheel_vel,
        "hip_l_pos_rad":   hip_l,
        "hip_r_pos_rad":   hip_r,
        "whl_tau_l":       whl_tau_l,
        "whl_tau_r":       whl_tau_r,
        "roll_rad":        roll,
        "yaw_rad":         yaw,
        "robot_state":     state,
        "state_name":      _STATE_NAMES.get(state, str(state)),
        "fault_code":      fault,
        "fault_name":      _FAULT_NAMES.get(fault, f"0x{fault:02X}"),
        "fault_description": _FAULT_DESCRIPTIONS.get(fault, "Unknown fault"),
        "test_val":        test_val,
        "hip_l_current_a": hip_l_curr,
        "hip_r_current_a": hip_r_curr,
        "ibus_ch":         ibus_ch,
        "ibus_alive":      ibus_alive,
        "wm_l_vel_turns_s": wm_l_vel,
        "wm_r_vel_turns_s": wm_r_vel,
        "wm_l_pos_turns":   wm_l_pos,
        "wm_r_pos_turns":   wm_r_pos,
        "wm_l_vbus":        wm_l_vbus,
        "wm_r_vbus":        wm_r_vbus,
        "wm_l_error":       wm_l_err,
        "wm_r_error":       wm_r_err,
        "wm_l_state":       wm_l_st,
        "wm_r_state":       wm_r_st,
        "wm_mode":          wm_mode,
    }


def _decode_telem_b(payload: bytes) -> dict:
    (tof0, tof1, tof2, tof3,
     roll_rate, yaw_rate,
     accel_x, accel_y, accel_z,
     hip_l_vel, hip_r_vel,
     hip_l_cmd_p, hip_r_cmd_p, hip_l_cmd_v, hip_r_cmd_v,
     hip_l_cmd_kp, hip_r_cmd_kp, hip_l_cmd_kd, hip_r_cmd_kd,
     hip_l_cmd_tff, hip_r_cmd_tff,
     theta_ref, v_ref, tau_sym, tau_yaw, vel_err_int, yaw_err_int,
     ff1, ff2, ff4,
     health_flags, imu_loss_pct, jump_state, loop_count) = _struct.unpack(_FMT_TELEM_B, payload)
    _NO_DATA = 0xFFFF
    tof_front = min((d for d in [tof0, tof1] if d != _NO_DATA), default=_NO_DATA)
    tof_rear  = min((d for d in [tof2, tof3] if d != _NO_DATA), default=_NO_DATA)
    return {
        "tof_dist_mm":         [tof0, tof1, tof2, tof3],
        "tof_front_min_mm":    tof_front,
        "tof_rear_min_mm":     tof_rear,
        "roll_rate_rads":      roll_rate,
        "yaw_rate_rads":       yaw_rate,
        "accel_x_ms2":         accel_x,
        "accel_y_ms2":         accel_y,
        "accel_z_ms2":         accel_z,
        "hip_l_vel_rads":      hip_l_vel,
        "hip_r_vel_rads":      hip_r_vel,
        "hip_l_cmd_pos_rad":   hip_l_cmd_p,
        "hip_r_cmd_pos_rad":   hip_r_cmd_p,
        "hip_l_cmd_vel_rads":  hip_l_cmd_v,
        "hip_r_cmd_vel_rads":  hip_r_cmd_v,
        "hip_l_cmd_kp":        hip_l_cmd_kp,
        "hip_r_cmd_kp":        hip_r_cmd_kp,
        "hip_l_cmd_kd":        hip_l_cmd_kd,
        "hip_r_cmd_kd":        hip_r_cmd_kd,
        "hip_l_cmd_tff":       hip_l_cmd_tff,
        "hip_r_cmd_tff":       hip_r_cmd_tff,
        "theta_ref":           theta_ref,
        "v_ref":               v_ref,
        "tau_sym":             tau_sym,
        "tau_yaw":             tau_yaw,
        "vel_err_integral":    vel_err_int,
        "yaw_err_integral":    yaw_err_int,
        "ff1_out":             ff1,
        "ff2_out":             ff2,
        "ff4_out":             ff4,
        "health_flags":        health_flags,
        "imu_packet_loss_pct": imu_loss_pct,
        "jump_state":          jump_state,
        "loop_count":          loop_count,
    }


class PacketDecoder(QObject):
    packet_decoded = pyqtSignal(dict)

    def __init__(self, device: str = "", parent=None):
        super().__init__(parent)
        self._device      = device
        self._buf         = b""
        self._telem_a_buf: dict | None = None  # holds decoded TELEM_A waiting for TELEM_B

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
            total   = _OVERHEAD + length
            if len(self._buf) < total:
                return
            payload  = self._buf[8:8 + length]
            checksum = self._buf[8 + length]
            end_byte = self._buf[9 + length]
            if end_byte != _COMM_END:
                self._buf = self._buf[1:]
                continue
            xor = ptype ^ version ^ source ^ seq ^ self._buf[6] ^ self._buf[7]
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
                    # TELEM_A — store and wait for TELEM_B before emitting
                    if version != _TELEM_VERSION:
                        info["version_mismatch"] = True
                        info["got_version"]       = version
                        info["expected_version"]  = _TELEM_VERSION
                    else:
                        self._telem_a_buf = {**info, **_decode_telem_a(payload)}
                    self._buf = self._buf[total:]
                    continue  # don't emit yet
                elif ptype == 0x11 and length == _TELEM_B_LEN and self._telem_a_buf is not None:
                    # TELEM_B — complete the packet and emit as a unified telemetry dict
                    if version == _TELEM_VERSION:
                        info = {**self._telem_a_buf, **info, **_decode_telem_b(payload)}
                        info["ptype"]     = 0x01   # appear as TELEM to PacketInspector
                        info["type_name"] = "TELEM"
                    self._telem_a_buf = None
            except Exception:
                pass
            self.packet_decoded.emit(info)
            from telemetry_bus import TelemetryBus
            from source_manager import SourceManager
            if SourceManager.instance().is_active(self._device):
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
