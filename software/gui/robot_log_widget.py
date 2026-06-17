from datetime import datetime

from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from telemetry_bus import TelemetryBus
from theme import BLUE, BORDER, DIM, GREEN, ORANGE, RED, SURFACE, TEXT

_AXIS_NAMES = {0x00: "BOTH", 0x01: "L", 0x02: "R"}
_CALIB_EVENTS = {
    0x01: ("CALIB START",        BLUE),
    0x02: ("CALIB BOTTOM FOUND", TEXT),
    0x03: ("CALIB LIMITS",       GREEN),
    0x04: ("CALIB DONE",         GREEN),
    0x05: ("CALIB FAULT",        RED),
}

_LEVEL_COLORS = {"INFO": GREEN, "WARN": ORANGE, "ERROR": RED}


def _filter_btn(label: str, color: str) -> QPushButton:
    b = QPushButton(label)
    b.setCheckable(True)
    b.setChecked(True)
    b.setFixedHeight(20)
    b.setStyleSheet(
        f"QPushButton{{background:{SURFACE};color:{DIM};font-size:10px;font-weight:bold;"
        f"border:1px solid {BORDER};border-radius:3px;padding:0px 8px}}"
        f"QPushButton:hover{{background:{BORDER}}}"
        f"QPushButton:checked{{background:{color}22;color:{color};"
        f"border:1px solid {color}}}"
    )
    return b


class RobotLogWidget(QWidget):
    """Scrolling log of all device log and calibration event packets, with level filter."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._show = {"INFO": True, "WARN": True, "ERROR": True}

        # ── Filter toolbar ────────────────────────────────────────────────────
        hdr = QLabel("Robot Log")
        hdr.setStyleSheet(f"color: {BLUE}; font-weight: bold; font-size: 11px;")

        self._btn_info  = _filter_btn("INFO",  GREEN)
        self._btn_warn  = _filter_btn("WARN",  ORANGE)
        self._btn_error = _filter_btn("ERROR", RED)

        self._btn_info.toggled.connect( lambda v: self._set_filter("INFO",  v))
        self._btn_warn.toggled.connect( lambda v: self._set_filter("WARN",  v))
        self._btn_error.toggled.connect(lambda v: self._set_filter("ERROR", v))

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(6)
        toolbar.addWidget(hdr)
        toolbar.addStretch()
        toolbar.addWidget(self._btn_info)
        toolbar.addWidget(self._btn_warn)
        toolbar.addWidget(self._btn_error)

        # ── Text area ─────────────────────────────────────────────────────────
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        self._log.setPlaceholderText("Waiting for robot log messages…")
        self._log.setStyleSheet(
            f"background: #0a0a14; color: {TEXT}; border: 1px solid {BORDER};"
            f" font-family: Consolas; font-size: 11px;"
        )
        self._log.setFixedHeight(140)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lay.addLayout(toolbar)
        lay.addWidget(self._log)

        TelemetryBus.instance().packet.connect(self._on_packet)

    def _set_filter(self, level: str, visible: bool):
        self._show[level] = visible

    def _append(self, text: str, color: str):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cur = self._log.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.setCharFormat(fmt)
        cur.insertText(f"[{ts}] {text}\n")
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _on_packet(self, info: dict):
        ptype = info.get("ptype")

        if ptype == 0x04:
            level = info.get("log_level", "INFO")
            if not self._show.get(level, True):
                return
            msg = info.get("log_msg", "")
            self._append(f"[{level}] {msg}", _LEVEL_COLORS.get(level, TEXT))

        elif ptype == 0x05:
            event = info.get("calib_event")
            axis  = info.get("calib_axis")
            axis_name  = _AXIS_NAMES.get(axis, f"0x{axis:02X}")
            event_name, color = _CALIB_EVENTS.get(event, (f"CALIB 0x{event:02X}", TEXT))
            pos = info.get("calib_pos_rad", 0.0)
            mn  = info.get("calib_min_rad", 0.0)
            mx  = info.get("calib_max_rad", 0.0)

            if event == 0x01:
                self._append("── calibration started ──", BLUE)
            elif event == 0x02:
                self._append(f"{axis_name}: bottom hardstop found, zeroed @ {pos:+.3f} rad", color)
            elif event == 0x03:
                self._append(f"{axis_name}: limits [{mn:+.3f}, {mx:+.3f}] rad  (range {pos:+.3f})", color)
            elif event == 0x04:
                self._append(f"{axis_name}: done, holding @ {pos:+.3f} rad  limits [{mn:+.3f}, {mx:+.3f}]", color)
            elif event == 0x05:
                self._append(f"{axis_name}: FAULT — hardstop not found @ {pos:+.3f} rad", color)
            else:
                self._append(f"{axis_name}: {event_name}  pos={pos:+.3f}", color)
