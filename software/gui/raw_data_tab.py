from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit,
    QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from telemetry_bus import TelemetryBus
from theme import BORDER, BLUE, DIM, GREEN, ORANGE, RED, SURFACE, TEXT


class RawDataTab(QWidget):
    def __init__(self):
        super().__init__()

        self._pkt_count   = 0   # total since start
        self._rate_count  = 0   # ticks in current second
        self._rate_hz     = 0.0

        # ── Header bar ────────────────────────────────────────────────────────
        self._pkt_lbl  = _stat("Pkt #",  "—")
        self._rate_lbl = _stat("Rate",   "—")
        self._src_lbl  = _stat("Source", "—")
        self._crc_lbl  = _stat("CRC",    "—")

        hdr = QHBoxLayout()
        hdr.setSpacing(24)
        for lbl in (self._pkt_lbl, self._rate_lbl, self._src_lbl, self._crc_lbl):
            hdr.addWidget(lbl)
        hdr.addStretch()

        # ── Scrollable field grid ─────────────────────────────────────────────
        self._fields: dict[str, QLabel] = {}

        content = QWidget()
        content.setStyleSheet(f"background: transparent;")
        grid = QGridLayout(content)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(32)
        grid.setVerticalSpacing(6)

        row = 0

        row = self._section(grid, row, "Frame")
        row = self._add_row(grid, row,
            ("START",    "—"),  ("END",      "—"),
        )
        row = self._add_row(grid, row,
            ("TYPE",     "—"),  ("VERSION",  "—"),
        )
        row = self._add_row(grid, row,
            ("SOURCE",   "—"),  ("SEQ",      "—"),
        )
        row = self._add_row(grid, row,
            ("LEN",      "—"),  ("CHECKSUM", "—"),
        )

        row = self._section(grid, row, "Telemetry Payload")
        row = self._add_row(grid, row,
            ("timestamp_ms",   "—"),  ("robot_state",     "—"),
        )
        row = self._add_row(grid, row,
            ("pitch_rad",      "—"),  ("pitch_rate_rads", "—"),
        )
        row = self._add_row(grid, row,
            ("roll_rad",       "—"),  ("yaw_rad",         "—"),
        )
        row = self._add_row(grid, row,
            ("wheel_vel_avg",  "—"),  ("",                ""),
        )
        row = self._add_row(grid, row,
            ("hip_l_pos_rad",  "—"),  ("hip_r_pos_rad",   "—"),
        )
        row = self._add_row(grid, row,
            ("cmd_l",          "—"),  ("cmd_r",           "—"),
        )

        grid.setRowStretch(row, 1)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        # ── Log box ───────────────────────────────────────────────────────────
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        self._log.setPlaceholderText("Waiting for log messages...")
        self._log.setStyleSheet(
            f"background: #0a0a14; color: {TEXT}; border: 1px solid {BORDER};"
            f" font-family: Consolas; font-size: 11px;"
        )
        self._log.setMaximumHeight(180)
        self._log.setMinimumHeight(80)

        log_hdr = QLabel("Device Log")
        log_hdr.setStyleSheet(f"color: {BLUE}; font-weight: bold; font-size: 11px;")

        log_pane = QWidget()
        log_lay = QVBoxLayout(log_pane)
        log_lay.setContentsMargins(0, 0, 0, 0)
        log_lay.setSpacing(4)
        log_lay.addWidget(log_hdr)
        log_lay.addWidget(self._log)

        # ── Outer layout ──────────────────────────────────────────────────────
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(12)
        lay.addLayout(hdr)
        lay.addWidget(_hline())
        lay.addWidget(scroll, stretch=1)
        lay.addWidget(_hline())
        lay.addWidget(log_pane)

        # ── Rate timer ────────────────────────────────────────────────────────
        self._rate_timer = QTimer(self)
        self._rate_timer.timeout.connect(self._tick_rate)
        self._rate_timer.start(1000)

        TelemetryBus.instance().packet.connect(self._on_packet)

    # ── Layout helpers ────────────────────────────────────────────────────────

    def _section(self, grid: QGridLayout, row: int, title: str) -> int:
        if row > 0:
            sep = _hline()
            grid.addWidget(sep, row, 0, 1, 4)
            row += 1
        lbl = QLabel(title)
        lbl.setStyleSheet(f"color: {BLUE}; font-weight: bold; font-size: 11px;")
        grid.addWidget(lbl, row, 0, 1, 4)
        return row + 1

    def _add_row(self, grid: QGridLayout, row: int,
                 left: tuple[str, str], right: tuple[str, str]) -> int:
        for col_offset, (name, default) in enumerate([(left, 0), (right, 2)]):
            field_name, field_default = name
            if not field_name:
                continue
            k = QLabel(field_name)
            k.setStyleSheet(f"color: {DIM}; font-size: 11px;")
            k.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            k.setMinimumWidth(130)
            v = QLabel(field_default)
            v.setStyleSheet(
                f"color: {TEXT}; font-size: 13px; font-weight: bold;"
                f" font-family: Consolas; background: {SURFACE};"
                f" border: 1px solid {BORDER}; border-radius: 3px;"
                f" padding: 2px 8px;"
            )
            v.setMinimumWidth(140)
            grid.addWidget(k, row, col_offset * 2)
            grid.addWidget(v, row, col_offset * 2 + 1)
            self._fields[field_name] = v
        return row + 1

    # ── Update ────────────────────────────────────────────────────────────────

    def _set(self, key: str, text: str, color: str = TEXT):
        lbl = self._fields.get(key)
        if lbl is None:
            return
        lbl.setText(text)
        lbl.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: bold;"
            f" font-family: Consolas; background: {SURFACE};"
            f" border: 1px solid {BORDER}; border-radius: 3px;"
            f" padding: 2px 8px;"
        )

    def _on_packet(self, info: dict):
        self._pkt_count += 1
        self._rate_count += 1

        # Header bar
        self._pkt_lbl.findChild(QLabel, "val").setText(str(self._pkt_count))
        src   = info.get("src_name", "—")
        color = BLUE if src == "TEENSY" else ORANGE
        _set_stat(self._src_lbl, src, color)
        crc = info.get("crc_ok")
        _set_stat(self._crc_lbl, "OK" if crc else "FAIL", GREEN if crc else RED)

        # Frame fields
        self._set("START",    "0xFF")
        self._set("END",      "0xFE")
        self._set("TYPE",     f"{info.get('type_name','—')}  (0x{info.get('ptype',0):02X})")
        self._set("VERSION",  str(info.get("version", "—")))
        self._set("SOURCE",   f"{info.get('src_name','—')}  (0x{info.get('source', 0):02X})")
        self._set("SEQ",      str(info.get("seq", "—")))
        self._set("LEN",      str(info.get("length", "—")))
        chk = info.get("checksum")
        self._set("CHECKSUM", f"0x{chk:02X}" if chk is not None else "—")

        # Payload
        if info.get("ptype") == 0x01:
            state = info.get("state_name", "—")
            state_color = GREEN if state == "RUNNING" else (RED if state == "ESTOP" else TEXT)
            self._set("timestamp_ms",   str(info.get("timestamp_ms",   "—")))
            self._set("robot_state",    state, state_color)
            self._set("pitch_rad",      f"{info['pitch_rad']:+.6f}")
            self._set("pitch_rate_rads",f"{info['pitch_rate_rads']:+.6f}")
            self._set("roll_rad",       f"{info['roll_rad']:+.6f}")
            self._set("yaw_rad",        f"{info['yaw_rad']:+.6f}")
            self._set("wheel_vel_avg",  f"{info['wheel_vel_avg']:+.6f}")
            self._set("hip_l_pos_rad",  f"{info['hip_l_pos_rad']:+.6f}")
            self._set("hip_r_pos_rad",  f"{info['hip_r_pos_rad']:+.6f}")
            self._set("cmd_l",          f"{info['cmd_l']:+.6f}")
            self._set("cmd_r",          f"{info['cmd_r']:+.6f}")
        elif info.get("ptype") == 0x04:
            self._append_log(info.get("log_level", "INFO"), info.get("log_msg", ""))
        else:
            for k in ("timestamp_ms", "robot_state", "pitch_rad", "pitch_rate_rads",
                      "roll_rad", "yaw_rad", "wheel_vel_avg",
                      "hip_l_pos_rad", "hip_r_pos_rad", "cmd_l", "cmd_r"):
                self._set(k, "—")

    def _append_log(self, level: str, msg: str):
        colors = {"INFO": GREEN, "WARN": ORANGE, "ERROR": RED}
        color  = colors.get(level, TEXT)
        ts     = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line   = f"[{ts}] [{level}] {msg}"
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cur = self._log.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.setCharFormat(fmt)
        cur.insertText(line + "\n")
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _tick_rate(self):
        self._rate_hz   = float(self._rate_count)
        self._rate_count = 0
        _set_stat(self._rate_lbl, f"{self._rate_hz:.0f} Hz",
                  GREEN if self._rate_hz > 100 else (ORANGE if self._rate_hz > 0 else DIM))


# ── Small helpers ─────────────────────────────────────────────────────────────

def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f


def _stat(label: str, default: str) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    k = QLabel(label + ":")
    k.setStyleSheet(f"color: {DIM}; font-size: 11px;")
    v = QLabel(default)
    v.setObjectName("val")
    v.setStyleSheet(f"color: {TEXT}; font-size: 13px; font-weight: bold; font-family: Consolas;")
    lay.addWidget(k)
    lay.addWidget(v)
    return w


def _set_stat(widget: QWidget, text: str, color: str = TEXT):
    lbl = widget.findChild(QLabel, "val")
    if lbl:
        lbl.setText(text)
        lbl.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: bold; font-family: Consolas;"
        )
