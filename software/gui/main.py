import sys
import os
import psutil
from collections import deque

def _kill_other_instances():
    current_pid = os.getpid()
    current_script = os.path.abspath(__file__)
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if proc.info['pid'] == current_pid:
            continue
        cmdline = proc.info.get('cmdline') or []
        if any(os.path.abspath(arg) == current_script for arg in cmdline if arg):
            proc.kill()

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QSplitter, QWidget,
    QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QMessageBox, QMenu,
)
from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import QFont, QPainter, QColor, QPen, QAction, QIcon

from tabs.theme import APP_STYLE, BORDER, TEXT, DIM, GREEN, ORANGE, RED, BLUE, YELLOW, WHITE
from tabs.robot_log_widget import RobotLogWidget
from tabs.flash_monitor import FlashMonitorTab
from tabs.hip_motors import HipMotorsTab
from tabs.imu_tab import ImuTab, ImuMiniWidget
from tabs.params_tab import ParamsTab
from tabs.raw_data_tab import RawDataTab
from tabs.robot_visualizer_tab import RobotVisualizerTab
from tabs.radio_tab import RadioTab
from tabs.wheel_motors import WheelMotorsTab
from tabs.controllers_tab import ControllersTab
from tabs.log_playback import LogsTab, LogPlaybackController
from tabs.log_transfer import LogTransferManager
from tabs.telemetry_bus import TelemetryBus
from tabs.source_manager import SourceManager, TRANSPORT_LABEL
from tabs.comm_commands import send_set_mode, send_reboot, send_soft_clear, STATE_STARTUP, STATE_STANDBY, STATE_ESTOP

_BG = "#0b0b18"

_ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.ico")

# ── Test-val mini chart ───────────────────────────────────────────────────────

class TestValMiniWidget(QWidget):
    """Rolling chart of the 2 Hz sine test_val field — dashboard comms health check."""

    _BUF = 200  # samples; at 50 Hz telemetry = 4 s of history

    def __init__(self):
        super().__init__()
        self.setMaximumWidth(230)

        self._buf = deque([0.0] * self._BUF, maxlen=self._BUF)

        plot = pg.PlotWidget()
        plot.setBackground(_BG)
        plot.setFixedHeight(170)
        plot.setYRange(-1.2, 1.2)
        plot.hideAxis("bottom")
        plot.getAxis("left").setStyle(tickFont=pg.Qt.QtGui.QFont("Consolas", 8))
        plot.getAxis("left").setTextPen(pg.mkPen(DIM))
        plot.showGrid(x=False, y=True, alpha=0.12)
        plot.setMouseEnabled(x=False, y=False)
        plot.setMenuEnabled(False)

        self._curve = plot.plot(list(self._buf), pen=pg.mkPen(GREEN, width=1.5))

        self._lbl = QLabel("~  —")
        self._lbl.setStyleSheet(
            f"color: {DIM}; font-size: 11px; font-family: Consolas;"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 4)
        lay.setSpacing(4)
        lay.addWidget(plot)
        lay.addWidget(self._lbl)

        TelemetryBus.instance().packet.connect(self._on_packet)

    def _on_packet(self, info: dict):
        val = info.get("test_val")
        if val is None:
            return
        self._buf.append(val)
        self._curve.setData(list(self._buf))
        self._lbl.setStyleSheet(
            f"color: {TEXT}; font-size: 11px; font-family: Consolas;"
        )
        self._lbl.setText(f"~  {val:+.4f}")

# ── ToF distance mini widget ─────────────────────────────────────────────────

class TofMiniWidget(QWidget):
    """Four VL53L1X distance bars: sensors 0-1 forward (green), 2-3 backward (orange)."""

    _MAX_MM    = 2000   # display range [mm]
    _WARN_MM   = 400    # highlight threshold [mm]
    _NO_DATA   = 0xFFFF

    def __init__(self):
        super().__init__()
        self.setMaximumWidth(200)

        self._bars:   list[QFrame]  = []
        self._labels: list[QLabel]  = []
        self._stale_lbl = QLabel("TOF — no data")
        self._stale_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px; font-family: Consolas;")
        self._stale_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        bar_row = QHBoxLayout()
        bar_row.setSpacing(6)
        bar_row.setContentsMargins(0, 0, 0, 0)

        _COLORS = [GREEN, GREEN, ORANGE, ORANGE]
        _NAMES  = ["F-L", "F-R", "R-L", "R-R"]
        for i in range(4):
            col = _COLORS[i]
            name_lbl = QLabel(_NAMES[i])
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_lbl.setStyleSheet(f"color: {DIM}; font-size: 9px;")

            bar_bg = QFrame()
            bar_bg.setFixedWidth(30)
            bar_bg.setFixedHeight(120)
            bar_bg.setStyleSheet(f"background: #111; border: 1px solid {BORDER};")

            bar_fill = QFrame(bar_bg)
            bar_fill.setFixedWidth(28)
            bar_fill.move(1, 1)
            bar_fill.setFixedHeight(0)
            bar_fill.setStyleSheet(f"background: {col};")

            dist_lbl = QLabel("—")
            dist_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dist_lbl.setStyleSheet(f"color: {DIM}; font-size: 9px; font-family: Consolas;")

            col_lay = QVBoxLayout()
            col_lay.setSpacing(2)
            col_lay.setContentsMargins(0, 0, 0, 0)
            col_lay.addWidget(name_lbl)
            col_lay.addWidget(bar_bg)
            col_lay.addWidget(dist_lbl)
            bar_row.addLayout(col_lay)

            self._bars.append(bar_fill)
            self._labels.append(dist_lbl)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        lay.addLayout(bar_row)
        lay.addWidget(self._stale_lbl)

        TelemetryBus.instance().packet.connect(self._on_packet)

    def _on_packet(self, info: dict):
        dists = info.get("tof_dist_mm")
        if dists is None:
            return

        stale = info.get("tof_stale", False)
        age   = info.get("tof_age_ms", self._NO_DATA)
        self._stale_lbl.setText(
            "TOF stale" if stale else f"TOF  age {age} ms"
        )
        self._stale_lbl.setStyleSheet(
            f"color: {RED if stale else DIM}; font-size: 10px; font-family: Consolas;"
        )

        _WARN_COLORS = [RED, RED, RED, RED]
        _OK_COLORS   = [GREEN, GREEN, ORANGE, ORANGE]
        for i, d in enumerate(dists):
            bar    = self._bars[i]
            lbl    = self._labels[i]
            parent = bar.parent()
            if d == self._NO_DATA:
                bar.setFixedHeight(0)
                lbl.setText("—")
                lbl.setStyleSheet(f"color: {DIM}; font-size: 9px; font-family: Consolas;")
            else:
                frac   = max(0.0, 1.0 - d / TofMiniWidget._MAX_MM)
                height = int(frac * (parent.height() - 2))
                bar.setFixedHeight(max(1, height))
                bar.move(1, parent.height() - 1 - bar.height())
                warn   = d < TofMiniWidget._WARN_MM
                color  = _WARN_COLORS[i] if warn else _OK_COLORS[i]
                bar.setStyleSheet(f"background: {color};")
                lbl.setText(f"{d} mm")
                lbl.setStyleSheet(
                    f"color: {color}; font-size: 9px; font-weight: {'bold' if warn else 'normal'};"
                    f" font-family: Consolas;"
                )


# ── Placeholder tabs ──────────────────────────────────────────────────────────

class _PlaceholderTab(QWidget):
    def __init__(self, name: str):
        super().__init__()
        lbl = QLabel(name)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFont(QFont("Segoe UI", 22))
        lbl.setStyleSheet(f"color: {BORDER};")
        QVBoxLayout(self).addWidget(lbl)

class DashboardTab(QWidget):
    def __init__(self):
        super().__init__()
        imu      = ImuMiniWidget()
        test_val = TestValMiniWidget()
        tof      = TofMiniWidget()

        top = QHBoxLayout()
        top.setContentsMargins(8, 8, 0, 0)
        top.setSpacing(12)
        top.addWidget(imu)
        top.addWidget(test_val)
        top.addWidget(tof)
        top.addStretch(7)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(top)
        outer.addStretch(1)

# ── Fault severity (mirrors comm_protocol.h fault_severity()) ─────────────────
_FAULT_SEVERITY = {
    0x06: "SOFT",        # HUMAN_ESTOP
    0x09: "SOFT",        # WHEEL_RUNAWAY
    0x08: "REPOSITION",  # PITCH_WATCHDOG
    0x05: "RECALIBRATE", # CALIBRATION_TIMEOUT (reposition + re-run calib)
    0x04: "GUI_FIX",     # HIP_LARGE_POS_CMD
    0x01: "REBOOT",      # IMU_ERROR
    0x02: "REBOOT",      # HIP_INIT_TIMEOUT
    0x03: "REBOOT",      # HIP_FEEDBACK_LOST
    0x0A: "REBOOT",      # IMU_LOST
    0x0B: "REBOOT",      # WHEEL_FEEDBACK_LOST
}

# ── Battery status widget ─────────────────────────────────────────────────────

class BatteryStatusWidget(QWidget):
    """Battery icon + voltage label for the status bar. Blinks at 1 Hz when critically low."""
    _VMAX = 25.2   # 4.2 V/cell × 6S
    _VMIN = 21.0   # 3.5 V/cell × 6S (display empty)
    _NBARS = 5
    # A wheel side that's disabled (wheel_l/r_enable=0) never gets its vbus
    # requested by firmware, so it reads a permanent 0.0 — exclude it from the
    # average instead of dragging a real ~24V reading down to ~12V.
    _VBUS_MIN_VALID = 5.0

    def __init__(self):
        super().__init__()
        self.setFixedWidth(72)
        self._vbus: float | None = None
        self._blink_on = True
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(500)
        self._blink_timer.timeout.connect(self._on_blink)
        TelemetryBus.instance().packet.connect(self._on_packet)

    def _num_bars(self, vbus: float) -> int:
        pct = max(0.0, min(1.0, (vbus - self._VMIN) / (self._VMAX - self._VMIN)))
        return round(pct * self._NBARS)

    def _on_packet(self, info: dict):
        vl = info.get("wm_l_vbus")
        vr = info.get("wm_r_vbus")
        if vl is None or vr is None:
            return
        readings = [v for v in (vl, vr) if v > self._VBUS_MIN_VALID]
        if not readings:
            return
        vbus = sum(readings) / len(readings)
        low = self._num_bars(vbus) <= 1
        if low and not self._blink_timer.isActive():
            self._blink_timer.start()
        elif not low and self._blink_timer.isActive():
            self._blink_timer.stop()
            self._blink_on = True
        self._vbus = vbus
        self.update()

    def set_connected(self, connected: bool):
        if not connected:
            self._blink_timer.stop()
            self._blink_on = True
            self._vbus = None
            self.update()

    def _on_blink(self):
        self._blink_on = not self._blink_on
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        w, h = self.width(), self.height()

        # Icon geometry
        BW, BH = 22, 12
        NW, NH = 3, 6
        bx = 2
        by = (h - BH) // 2

        if self._vbus is None:
            p.setPen(QPen(QColor(DIM)))
            p.drawRect(bx, by, BW - 1, BH - 1)
            p.fillRect(bx + BW, by + (BH - NH) // 2, NW, NH, QColor(DIM))
            p.end()
            return

        bars = self._num_bars(self._vbus)
        low  = bars <= 1
        pct  = bars / self._NBARS
        col  = QColor(GREEN if pct > 0.5 else ORANGE if pct > 0.2 else RED)

        if low and not self._blink_on:
            dim = QColor(100, 0, 0)
            p.setPen(QPen(dim))
            p.drawRect(bx, by, BW - 1, BH - 1)
            p.fillRect(bx + BW, by + (BH - NH) // 2, NW, NH, dim)
        else:
            p.setPen(QPen(QColor(WHITE)))
            p.drawRect(bx, by, BW - 1, BH - 1)
            p.fillRect(bx + BW, by + (BH - NH) // 2, NW, NH, QColor(WHITE))
            p.fillRect(bx + 2, by + 2, BW - 4, BH - 4, QColor(15, 15, 20))
            # 5 bars × 2px wide, 1px gap, 2px left padding
            for i in range(self._NBARS):
                if i < bars:
                    p.fillRect(bx + 4 + i * 3, by + 3, 2, BH - 6, col)

        # Voltage text
        p.setPen(QPen(col))
        p.setFont(QFont("Consolas", 9))
        tx = bx + BW + NW + 4
        p.drawText(tx, 0, w - tx, h,
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   f"{self._vbus:.1f}V")
        p.end()


# ── Radio signal widget ───────────────────────────────────────────────────────

class RadioSignalWidget(QWidget):
    """RC TX signal indicator — 4 ascending vertical bars (cellular-style, not WiFi arcs).
    Green = signal, red = no signal, dim = no telemetry data yet."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(52, 24)
        self._alive: bool | None = None
        TelemetryBus.instance().packet.connect(self._on_packet)

    def _on_packet(self, info: dict):
        alive = info.get("ibus_alive")
        if alive is None:
            return
        new = bool(alive)
        if new != self._alive:
            self._alive = new
            self.update()

    def set_connected(self, connected: bool):
        if not connected and self._alive is not None:
            self._alive = None
            self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # 4 bars, ascending heights, left-aligned
        n_bars = 4
        bar_w  = 4
        gap    = 3
        total  = n_bars * bar_w + (n_bars - 1) * gap
        x0     = 2
        base_y = h - 4

        if self._alive is None:
            bar_color = QColor(DIM)
            n_lit = 0
        elif self._alive:
            bar_color = QColor(GREEN)
            n_lit = n_bars
        else:
            bar_color = QColor(RED)
            n_lit = 1  # one short bar to suggest "no signal"

        dim_color = QColor(DIM)
        dim_color.setAlpha(60)

        for i in range(n_bars):
            bh = 4 + i * 4   # 4, 8, 12, 16 px
            x  = x0 + i * (bar_w + gap)
            y  = base_y - bh
            col = bar_color if i < n_lit else dim_color
            p.fillRect(x, y, bar_w, bh, col)

        # "RC" label
        lbl_color = QColor(GREEN if self._alive else (RED if self._alive is not None else DIM))
        p.setPen(QPen(lbl_color))
        p.setFont(QFont("Consolas", 8))
        p.drawText(x0 + total + 3, 0, w - (x0 + total + 3), h,
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   "RC")
        p.end()


# ── Status bar ────────────────────────────────────────────────────────────────

def _vsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f


class LogMiniControls(QWidget):
    """Compact SD-logging + playback controls for the status bar — start/stop
    a log or play/pause an open .wlog from any tab, without switching to
    the Logs tab. Both drive the same LogTransferManager/LogPlaybackController
    singletons the Logs tab uses, so state always stays in sync."""

    def __init__(self):
        super().__init__()
        self._xfer = LogTransferManager.instance()
        self._pb   = LogPlaybackController.instance()

        self._rec_lbl = QLabel("● REC")
        self._rec_lbl.setStyleSheet(f"color: {DIM}; font-weight: bold; font-size: 12px;")
        self._rec_lbl.setToolTip("Lit while an SD log is actively recording")

        self._btn_log = QPushButton()
        self._btn_log.setFixedWidth(64)
        self._btn_log.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_log.clicked.connect(self._on_log_clicked)
        self._style_log_button(False)

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(26)
        self._btn_play.setEnabled(False)
        self._btn_play.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_play.setToolTip("Open a .wlog in the Logs tab to enable playback")
        self._btn_play.clicked.connect(self._pb.toggle)
        self._btn_play.setStyleSheet(
            f"QPushButton{{background:transparent;color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:3px;padding:2px}}"
            f"QPushButton:disabled{{color:{DIM};border:1px solid {BORDER}}}"
            f"QPushButton:hover{{background:{BORDER}}}"
        )

        self._pb_lbl = QLabel("")
        self._pb_lbl.setStyleSheet(f"color: {DIM}; font-size: 11px; font-family: Consolas;")
        self._pb_lbl.setMaximumWidth(140)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(6)
        lay.addWidget(self._rec_lbl)
        lay.addWidget(self._btn_log)
        lay.addWidget(self._btn_play)
        lay.addWidget(self._pb_lbl)

        self._xfer.logging_state_changed.connect(self._on_logging_state)
        self._pb.file_opened.connect(self._on_file_opened)
        self._pb.playing_changed.connect(self._on_playing_changed)
        self._pb.position_changed.connect(self._on_position_changed)

    def _style_log_button(self, active: bool):
        color = RED if active else GREEN
        self._btn_log.setText("■ Stop" if active else "● Log")
        self._btn_log.setToolTip(
            "Stop the active SD log" if active else
            "Start logging until Stop is clicked (use the Logs tab for a timed duration)"
        )
        self._btn_log.setStyleSheet(
            f"QPushButton{{background:transparent;color:{color};border:1px solid {color};"
            f"border-radius:3px;padding:2px 6px;font-size:11px}}"
            f"QPushButton:hover{{background:{BORDER}}}"
        )

    def _on_log_clicked(self):
        if self._xfer.is_logging():
            self._xfer.stop_logging()
        else:
            self._xfer.start_logging(0)

    def _on_logging_state(self, active: bool):
        self._style_log_button(active)
        self._rec_lbl.setStyleSheet(
            f"color: {RED if active else DIM}; font-weight: bold; font-size: 12px;"
        )

    def _on_file_opened(self, path: str, count: int, sample_rate_hz: int):
        from pathlib import Path
        self._btn_play.setEnabled(True)
        self._btn_play.setToolTip("Play/pause the open .wlog")
        self._pb_lbl.setText(Path(path).name)

    def _on_playing_changed(self, playing: bool):
        self._btn_play.setText("⏸" if playing else "▶")

    def _on_position_changed(self, idx: int, total: int):
        self._pb_lbl.setToolTip(f"{idx} / {total}")


class StatusBar:
    def __init__(self, sb):
        self._source = QPushButton("● —")
        self._source.setFlat(True)
        self._source.setCursor(Qt.CursorShape.PointingHandCursor)
        self._source.setToolTip("Click to select telemetry source")
        self._source.setStyleSheet(
            f"QPushButton{{color:{DIM};background:transparent;border:none;"
            f"font-size:13px;padding:0 4px}}"
            f"QPushButton:hover{{color:{TEXT};text-decoration:underline}}"
        )
        self._source.clicked.connect(self._on_source_clicked)
        self._transport = QLabel("—")
        self._dt        = QLabel("dt: —")
        self._mode      = QLabel("—")
        self._mode.setStyleSheet(f"color: {DIM};")
        self._profile   = QLabel("")
        self._profile.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-weight: bold;")
        self._batt      = BatteryStatusWidget()
        self._radio     = RadioSignalWidget()
        self._conn      = QLabel("● Disconnected")
        self._conn.setStyleSheet(f"color: {RED};")

        self._current_state      = ""
        self._current_fault_code = 0

        self._btn_estop = QPushButton("ESTOP")
        self._btn_estop.setStyleSheet(
            f"QPushButton{{background:{RED};color:white;font-weight:bold;font-size:14px;"
            f"border:1px solid {RED};border-radius:4px;padding:6px 24px;margin:2px 6px}}"
            f"QPushButton:hover{{background:#ff6e60}}"
            f"QPushButton:pressed{{background:#b2362a}}"
        )
        self._btn_estop.clicked.connect(lambda: send_set_mode(STATE_ESTOP))

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setEnabled(False)
        self._btn_reset.clicked.connect(self._on_reset_clicked)
        self._reset_style_normal = (
            f"QPushButton{{background:#4a1a1a;color:white;"
            f"border:1px solid {BORDER};border-radius:3px;padding:2px 10px}}"
            f"QPushButton:hover{{background:#4a1a1acc}}"
            f"QPushButton:pressed{{background:#4a1a1a88}}"
            f"QPushButton:disabled{{background:transparent;color:{DIM};"
            f"border:1px solid {BORDER}}}"
        )
        self._reset_style_soft = (
            f"QPushButton{{background:#1a4a1a;color:white;font-weight:bold;"
            f"border:1px solid {GREEN};border-radius:3px;padding:2px 10px}}"
            f"QPushButton:hover{{background:#1a6a1a}}"
            f"QPushButton:pressed{{background:#0f3a0f}}"
        )
        self._reset_style_orange = (
            f"QPushButton{{background:#4a2a00;color:white;"
            f"border:1px solid {ORANGE};border-radius:3px;padding:2px 10px}}"
            f"QPushButton:hover{{background:#6a3a00}}"
            f"QPushButton:pressed{{background:#3a1a00}}"
        )
        self._btn_reset.setStyleSheet(self._reset_style_normal)

        self._btn_reboot = QPushButton("Reboot")
        self._btn_reboot.setStyleSheet(self._btn_reset.styleSheet())
        self._btn_reboot.setEnabled(False)
        self._btn_reboot.clicked.connect(self._on_reboot_clicked)

        for w in (self._source, _vsep(), self._transport, _vsep(), self._dt, _vsep(), self._mode, self._profile, _vsep(), self._batt, _vsep(), self._radio):
            sb.addWidget(w)
        self._mismatch_lbl = QLabel("FIRMWARE MISMATCH — reflash ESP32/Teensy")
        self._mismatch_lbl.setStyleSheet(
            f"color: white; background: {RED}; font-weight: bold;"
            f" padding: 2px 10px; border-radius: 3px;"
        )
        self._mismatch_lbl.setVisible(False)

        self._log_controls = LogMiniControls()

        sb.addPermanentWidget(self._mismatch_lbl)
        sb.addPermanentWidget(_vsep())
        sb.addPermanentWidget(self._log_controls)
        sb.addPermanentWidget(_vsep())
        sb.addPermanentWidget(self._btn_estop)
        sb.addPermanentWidget(_vsep())
        sb.addPermanentWidget(self._btn_reset)
        sb.addPermanentWidget(self._btn_reboot)
        sb.addPermanentWidget(_vsep())
        sb.addPermanentWidget(self._conn)

    def _on_reset_clicked(self):
        if self._current_state != "ESTOP":
            return
        severity = _FAULT_SEVERITY.get(self._current_fault_code, "REBOOT")
        if severity == "SOFT":
            send_soft_clear()
        elif severity in ("REPOSITION", "RECALIBRATE"):
            extra = ("\n\nYou will need to re-trigger calibration (CH5 or GUI) after reset."
                     if severity == "RECALIBRATE" else "")
            reply = QMessageBox.question(
                None, "Confirm Reset",
                f"Robot fell. Stand it upright, then click Yes to reset.{extra}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                send_set_mode(STATE_STARTUP)
        elif severity == "GUI_FIX":
            reply = QMessageBox.question(
                None, "Fix Parameter First",
                "A parameter value caused this ESTOP.\n"
                "Please fix the offending parameter in the Params tab, then click Yes to reset.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                send_set_mode(STATE_STARTUP)
        # REBOOT: button is disabled; should not be reached

    def _on_reboot_clicked(self):
        reply = QMessageBox.question(
            None, "Reboot Teensy",
            "This will fully reset the Teensy MCU and re-run setup() from scratch.\n"
            "All motors and state will drop momentarily. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            send_reboot()

    def _on_source_clicked(self):
        from tabs.source_manager import SourceManager, PRIORITY
        sm = SourceManager.instance()
        menu = QMenu()
        menu.setStyleSheet(
            f"QMenu{{background:#1a1a2e;color:{TEXT};border:1px solid {BORDER};"
            f"font-size:13px}}"
            f"QMenu::item:selected{{background:{BORDER}}}"
            f"QMenu::item:disabled{{color:{DIM}}}"
        )

        auto_action = menu.addAction("Auto (priority order)")
        auto_action.setCheckable(True)
        auto_action.setChecked(sm.override is None)
        menu.addSeparator()

        for device in PRIORITY:
            label = device.upper()
            connected = device in sm.connected
            action = menu.addAction(f"● {label}" if connected else f"○ {label} (not connected)")
            action.setCheckable(True)
            action.setChecked(sm.override == device)
            action.setEnabled(connected)
            action.setData(device)

        chosen = menu.exec(self._source.mapToGlobal(self._source.rect().bottomLeft()))
        if chosen is None:
            return
        if chosen is auto_action:
            sm.set_override(None)
        elif chosen.data():
            sm.set_override(chosen.data())

    def set_source(self, src: str):
        from tabs.source_manager import SourceManager
        sm = SourceManager.instance()
        color = {
            "TEENSY": BLUE,
            "ESP32":  ORANGE,
        }.get(src, DIM)
        is_override = sm.override is not None
        label = f"● {src}" if src != "—" else "● —"
        if is_override:
            label += " ✎"
        self._source.setStyleSheet(
            f"QPushButton{{color:{color};background:transparent;border:none;"
            f"font-size:13px;padding:0 4px}}"
            f"QPushButton:hover{{color:{color};text-decoration:underline}}"
        )
        self._source.setText(label)
        tip = "Click to select telemetry source"
        if is_override:
            tip += f" (override active — Auto uses {', '.join(p.upper() for p in ['esp32','teensy'])} priority)"
        self._source.setToolTip(tip)

    def set_transport(self, transport: str):
        self._transport.setText(transport)

    def set_dt(self, dt_ms: float):
        self._dt.setText(f"dt: {dt_ms:.1f} ms")

    def set_mode(self, state: str, fault: str = "", fault_description: str = "", fault_code: int = 0):
        self._current_state      = state
        self._current_fault_code = fault_code
        if state == "—":
            self._mode.setStyleSheet(f"color: {DIM};")
            self._mode.setText("—")
            self._mode.setToolTip("")
            self._btn_reset.setText("Reset")
            self._btn_reset.setStyleSheet(self._reset_style_normal)
            self._btn_reset.setEnabled(False)
            self._btn_reboot.setEnabled(False)
            return
        color = {"RUNNING": GREEN, "ESTOP": RED, "CALIBRATION": BLUE, "STANDBY": YELLOW, "STARTUP": WHITE}.get(state, DIM)
        self._mode.setStyleSheet(f"color: {color}; font-weight: bold;")
        is_fault = state == "ESTOP" and fault and fault != "NONE"
        label = f"{state}  [{fault}]" if is_fault else state
        self._mode.setText(label)
        self._mode.setToolTip(fault_description if is_fault else "")
        self._btn_reboot.setEnabled(True)

        if state != "ESTOP":
            self._btn_reset.setText("Reset")
            self._btn_reset.setStyleSheet(self._reset_style_normal)
            self._btn_reset.setEnabled(False)
            self._btn_reset.setToolTip("")
            return

        severity = _FAULT_SEVERITY.get(fault_code, "REBOOT")
        if severity == "SOFT":
            self._btn_reset.setText("Clear ESTOP")
            self._btn_reset.setStyleSheet(self._reset_style_soft)
            self._btn_reset.setEnabled(True)
            self._btn_reset.setToolTip("Click to clear ESTOP and return to STANDBY (no re-init)")
        elif severity in ("REPOSITION", "RECALIBRATE"):
            self._btn_reset.setText("Reset")
            self._btn_reset.setStyleSheet(self._reset_style_orange)
            self._btn_reset.setEnabled(True)
            tip = ("Stand robot upright first, then click Reset" +
                   (" — recalibration required after reset" if severity == "RECALIBRATE" else ""))
            self._btn_reset.setToolTip(tip)
        elif severity == "GUI_FIX":
            self._btn_reset.setText("Reset")
            self._btn_reset.setStyleSheet(self._reset_style_orange)
            self._btn_reset.setEnabled(True)
            self._btn_reset.setToolTip("Fix the offending parameter in the Params tab before resetting")
        else:  # REBOOT
            self._btn_reset.setText("Reset")
            self._btn_reset.setStyleSheet(self._reset_style_normal)
            self._btn_reset.setEnabled(False)
            self._btn_reset.setToolTip("Hardware fault — power cycle robot and reboot required")

    def set_profile(self, profile: int):
        label = f"  P{profile + 1}" if isinstance(profile, int) and 0 <= profile <= 2 else ""
        color = {0: DIM, 1: GREEN, 2: ORANGE}.get(profile, DIM)
        self._profile.setStyleSheet(f"color: {color}; font-family: Consolas; font-weight: bold;")
        self._profile.setText(label)

    def set_connected(self, connected: bool):
        if connected:
            self._conn.setStyleSheet(f"color: {GREEN};")
            self._conn.setText("● Connected")
        else:
            self._conn.setStyleSheet(f"color: {RED};")
            self._conn.setText("● Disconnected")
            self._mismatch_lbl.setVisible(False)
            self._batt.set_connected(False)
            self._radio.set_connected(False)

    def set_version_mismatch(self, got: int, expected: int):
        self._mismatch_lbl.setText(
            f"FIRMWARE MISMATCH (telem v{got} != v{expected}) — reflash ESP32/Teensy"
        )
        self._mismatch_lbl.setVisible(True)

    def clear_version_mismatch(self):
        self._mismatch_lbl.setVisible(False)

def _as_list(value) -> list:
    """QSettings round-trips a single-item list as a bare string on some
    platforms/formats — normalize back to a list so callers can iterate."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


# ── Floating tab window ───────────────────────────────────────────────────────

class FloatingTabWindow(QMainWindow):
    """A tab popped out of the main window into its own top-level window.

    Closing the window docks the tab back (to the left pane) rather than
    destroying it — every tab holds live TelemetryBus subscriptions that
    can't be recreated, since MainWindow only constructs each one once.
    """

    def __init__(self, title: str, widget: QWidget, main_window: "MainWindow"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(700, 700)
        self._main_window = main_window
        self._title = title
        self._widget = widget
        self._returned = False
        self.setCentralWidget(widget)
        # QTabWidget explicitly hides pages that aren't the active tab, and
        # that hidden flag persists across reparenting — force it visible.
        widget.show()

        self.menuBar().addAction("Put Back in Main Window", self.close)

    def closeEvent(self, event):
        # Reached both from the menu action above and from the OS close (X)
        # button — either way, return the tab to the main window instead of
        # destroying it (it holds live TelemetryBus subscriptions that can't
        # be recreated). Guarded so a second close() call is a harmless no-op
        # rather than re-triggering the dock logic.
        if not self._returned:
            self._returned = True
            self._main_window._dock_floating_tab(self)
        event.accept()

    def take_widget(self) -> QWidget:
        """Detach and return the hosted widget without destroying it."""
        widget = self.centralWidget()
        self.setCentralWidget(QWidget())  # leave something trivial in its place
        return widget


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wheeled-Leg Robot")
        self.resize(1400, 900)

        # Wire up singletons BEFORE creating tabs — DevicePanel._auto_scan()
        # fires synchronously during tab construction and emits port_opened;
        # SourceManager must already exist to catch those signals.
        self.status = StatusBar(self.statusBar())

        self._last_ts_ms: float | None = None
        self._disconnect_timer = QTimer(self)
        self._disconnect_timer.setSingleShot(True)
        self._disconnect_timer.setInterval(3000)
        self._disconnect_timer.timeout.connect(lambda: self.status.set_connected(False))

        TelemetryBus.instance().packet.connect(self._on_packet)

        sm = SourceManager.instance()
        sm.source_changed.connect(self._on_source_changed)

        from tabs.wifi_transport import WifiTransport
        WifiTransport.instance().start()

        self._log_pane = RobotLogWidget()
        self._log_pane.setContentsMargins(8, 2, 8, 6)

        # QFrame (not QWidget) so a thick border can be toggled reliably via
        # QSS as the obvious "you are looking at replayed data" indicator —
        # top-level window chrome borders are OS-dependent and unreliable.
        central = QFrame()
        central.setObjectName("central")
        self._central_style_live = f"QFrame#central {{ background: {_BG}; border: 3px solid transparent; }}"
        self._central_style_playback = f"QFrame#central {{ background: {_BG}; border: 3px solid {ORANGE}; }}"
        central.setStyleSheet(self._central_style_live)

        # Up to 2 tab panes side-by-side in a splitter, with the robot log
        # always pinned full-width below — right-click a tab for "Split
        # Right"/"Move to ... Pane"/"Float in New Window" (see
        # _show_tab_context_menu). The right pane is created on demand and
        # collapses away once its last tab leaves, so with nothing split or
        # floated this looks exactly like a single plain QTabWidget.
        self._split = QSplitter(Qt.Orientation.Horizontal)
        self._left_pane = self._make_tab_pane()
        self._right_pane = None
        self._floating: dict[QWidget, FloatingTabWindow] = {}
        self._split.addWidget(self._left_pane)

        central_lay = QVBoxLayout(central)
        central_lay.setContentsMargins(0, 0, 0, 0)
        central_lay.setSpacing(0)
        central_lay.addWidget(self._split, 1)
        central_lay.addWidget(self._log_pane, 0)
        self.setCentralWidget(central)
        self._central = central

        tab_defs = [
            ("Visualizer",      RobotVisualizerTab()),
            ("Dashboard",       DashboardTab()),
            ("IMU",             ImuTab()),
            ("Raw Data",        RawDataTab()),
            ("Hip Motors",      HipMotorsTab()),
            ("Parameters",      ParamsTab()),
            ("Wheel Motors",    WheelMotorsTab()),
            ("Controllers",     ControllersTab()),
            ("Radio",           RadioTab()),
            ("Logs",            LogsTab()),
            ("Flash & Monitor", FlashMonitorTab()),
        ]
        for title, widget in tab_defs:
            self._left_pane.addTab(widget, title)
        self._tab_widgets = {title: widget for title, widget in tab_defs}

        self._restore_layout()

        self._base_title = self.windowTitle()
        TelemetryBus.instance().playback_state_changed.connect(self._on_playback_state)

        # 10 Hz heartbeat feeding the firmware MANUAL-mode GUI watchdog (500 ms):
        # if the GUI dies, pings stop and the robot exits MANUAL / idles wheels.
        from tabs.comm_commands import send_ping
        self._ping_timer = QTimer(self)
        self._ping_timer.setInterval(100)
        self._ping_timer.timeout.connect(send_ping)
        self._ping_timer.start()

        self._on_source_changed(sm.active)

    def _on_playback_state(self, active: bool):
        self._central.setStyleSheet(self._central_style_playback if active else self._central_style_live)
        self.setWindowTitle(f"[PLAYBACK] {self._base_title}" if active else self._base_title)

    # ── Split-pane / floating tab management ──────────────────────────────

    def _make_tab_pane(self) -> QTabWidget:
        pane = QTabWidget()
        pane.setMovable(True)
        pane.tabBar().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        pane.tabBar().customContextMenuRequested.connect(
            lambda pos, pane=pane: self._show_tab_context_menu(pane, pos)
        )
        return pane

    def _show_tab_context_menu(self, pane: QTabWidget, pos):
        idx = pane.tabBar().tabAt(pos)

        menu = QMenu(self)
        if idx != -1:
            widget = pane.widget(idx)
            title = pane.tabText(idx)

            if pane is self._left_pane:
                label = "Split Right →" if self._right_pane is None else "Move to Right Pane"
                menu.addAction(label, lambda: self._move_tab_to_pane(widget, title, "right"))
            else:
                menu.addAction("Move to Left Pane", lambda: self._move_tab_to_pane(widget, title, "left"))
            # Visualizer/IMU embed a pyqtgraph OpenGL view, which doesn't survive
            # being reparented into a new top-level window's GL context —
            # observed to throw GL errors or hard-crash the app.
            if title not in ("Visualizer", "IMU"):
                menu.addAction("Float in New Window", lambda: self._float_tab(widget, title))
            menu.addSeparator()

        menu.addAction("Restore Default Layout", self._restore_default_layout)
        menu.exec(pane.tabBar().mapToGlobal(pos))

    def _restore_default_layout(self):
        """Undo split panes / floated windows and put every tab back in a
        single left pane, in the original registration order."""
        for win in list(self._floating.values()):
            win.close()  # closeEvent docks the tab back into the left pane

        if self._right_pane is not None:
            for title in [self._right_pane.tabText(i) for i in range(self._right_pane.count())]:
                self._move_tab_to_pane(self._tab_widgets[title], title, "left")

        self._reorder_pane_tabs(self._left_pane, list(self._tab_widgets.keys()))
        self._left_pane.setCurrentIndex(0)

    def _remove_tab_from_current_location(self, widget: QWidget):
        for pane in (self._left_pane, self._right_pane):
            if pane is None:
                continue
            idx = pane.indexOf(widget)
            if idx != -1:
                pane.removeTab(idx)
                return
        win = self._floating.pop(widget, None)
        if win is not None:
            win.take_widget()
            win._returned = True  # caller is placing the widget elsewhere itself
            win.close()

    def _move_tab_to_pane(self, widget: QWidget, title: str, target: str):
        self._remove_tab_from_current_location(widget)
        if target == "right" and self._right_pane is None:
            self._right_pane = self._make_tab_pane()
            self._split.addWidget(self._right_pane)
            self._split.setSizes([1, 1])
        pane = self._left_pane if target == "left" else self._right_pane
        pane.addTab(widget, title)
        pane.setCurrentWidget(widget)
        self._collapse_empty_pane()

    def _float_tab(self, widget: QWidget, title: str):
        self._remove_tab_from_current_location(widget)
        win = FloatingTabWindow(title, widget, self)
        win.show()
        self._floating[widget] = win
        self._collapse_empty_pane()

    def _dock_floating_tab(self, win: "FloatingTabWindow"):
        title = win._title
        widget = win.take_widget()
        self._floating.pop(widget, None)
        self._left_pane.addTab(widget, title)
        self._left_pane.setCurrentWidget(widget)

    def _collapse_empty_pane(self):
        if self._right_pane is not None and self._right_pane.count() == 0:
            self._right_pane.setParent(None)
            self._right_pane.deleteLater()
            self._right_pane = None
        if self._left_pane.count() == 0 and self._right_pane is not None:
            self._left_pane, self._right_pane = self._right_pane, self._left_pane
            self._right_pane.setParent(None)
            self._right_pane.deleteLater()
            self._right_pane = None

    # ── Split/float layout persistence ────────────────────────────────────

    def closeEvent(self, event):
        self._save_layout()
        super().closeEvent(event)

    def _reorder_pane_tabs(self, pane: QTabWidget, order: list[str]):
        for target_idx, title in enumerate(order):
            for i in range(pane.count()):
                if pane.tabText(i) == title:
                    if i != target_idx:
                        pane.tabBar().moveTab(i, target_idx)
                    break

    def _save_layout(self):
        settings = QSettings("WheeledLegRobot", "GUI")
        settings.setValue("mainWindow/geometry", self.saveGeometry())
        settings.setValue(
            "layout/left_tabs",
            [self._left_pane.tabText(i) for i in range(self._left_pane.count())],
        )
        settings.setValue(
            "layout/right_tabs",
            [self._right_pane.tabText(i) for i in range(self._right_pane.count())]
            if self._right_pane is not None else [],
        )
        if self._right_pane is not None:
            settings.setValue("layout/splitter_sizes", self._split.sizes())
        settings.setValue("layout/floating_tabs", [win._title for win in self._floating.values()])
        for win in self._floating.values():
            settings.setValue(f"layout/floating_geometry/{win._title}", win.saveGeometry())

    def _restore_layout(self):
        settings = QSettings("WheeledLegRobot", "GUI")
        geometry = settings.value("mainWindow/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        for title in _as_list(settings.value("layout/right_tabs")):
            widget = self._tab_widgets.get(title)
            if widget is not None:
                self._move_tab_to_pane(widget, title, "right")

        sizes = settings.value("layout/splitter_sizes")
        if sizes and self._right_pane is not None:
            try:
                self._split.setSizes([int(s) for s in sizes])
            except (TypeError, ValueError):
                pass

        for title in _as_list(settings.value("layout/floating_tabs")):
            widget = self._tab_widgets.get(title)
            if widget is None:
                continue
            self._float_tab(widget, title)
            win = self._floating.get(widget)
            win_geometry = settings.value(f"layout/floating_geometry/{title}")
            if win is not None and win_geometry is not None:
                win.restoreGeometry(win_geometry)

        self._reorder_pane_tabs(self._left_pane, _as_list(settings.value("layout/left_tabs")))
        if self._right_pane is not None:
            self._reorder_pane_tabs(self._right_pane, _as_list(settings.value("layout/right_tabs")))

    def _on_source_changed(self, device: str):
        if device:
            self.status.set_source(device.upper())
            self.status.set_transport(TRANSPORT_LABEL.get(device, "Unknown"))
        else:
            self.status.set_connected(False)
            self.status.set_source("—")
            self.status.set_transport("—")
            self.status.set_mode("—")
            self._last_ts_ms = None

    def _on_packet(self, info: dict):
        self.status.set_connected(True)
        self._disconnect_timer.start()
        if info.get("version_mismatch"):
            self.status.set_version_mismatch(
                info.get("got_version", "?"), info.get("expected_version", "?")
            )
            return
        if info.get("ptype") == 0x01:
            self.status.clear_version_mismatch()
        ts = info.get("timestamp_ms")
        if ts is not None and self._last_ts_ms is not None:
            self.status.set_dt(ts - self._last_ts_ms)
        if ts is not None:
            self._last_ts_ms = ts
        state = info.get("state_name")
        if state:
            self.status.set_mode(state, info.get("fault_name", ""), info.get("fault_description", ""),
                                 info.get("fault_code", 0))
        profile = info.get("active_profile")
        if profile is not None:
            self.status.set_profile(profile)

# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    _kill_other_instances()

    if sys.platform == "win32":
        # Without an explicit AppUserModelID, Windows groups this process under
        # python.exe's own taskbar icon instead of the one we set below.
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("WheeledLegRobot.GUI")

    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    app.setWindowIcon(QIcon(_ICON_PATH))
    win = MainWindow()
    win.setWindowIcon(QIcon(_ICON_PATH))
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
