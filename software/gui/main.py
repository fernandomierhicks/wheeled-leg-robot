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
    QApplication, QMainWindow, QTabWidget, QWidget,
    QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QMessageBox, QMenu,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPainter, QColor, QPen

from theme import APP_STYLE, BORDER, TEXT, DIM, GREEN, ORANGE, RED, BLUE, YELLOW, WHITE
from robot_log_widget import RobotLogWidget
from flash_monitor import FlashMonitorTab
from hip_motors import HipMotorsTab
from imu_tab import ImuTab, ImuMiniWidget
from params_tab import ParamsTab
from raw_data_tab import RawDataTab
from robot_visualizer_tab import RobotVisualizerTab
from radio_tab import RadioTab
from wheel_motors import WheelMotorsTab
from controllers_tab import ControllersTab
from telemetry_bus import TelemetryBus
from source_manager import SourceManager, TRANSPORT_LABEL
from comm_commands import send_set_mode, send_reboot, send_soft_clear, STATE_STARTUP, STATE_STANDBY, STATE_ESTOP

_BG = "#0b0b18"

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
    0x07: "GUI_FIX",     # PARAM_OUT_OF_BOUNDS
    0x04: "GUI_FIX",     # HIP_LARGE_POS_CMD
    0x01: "REBOOT",      # IMU_ERROR
    0x02: "REBOOT",      # HIP_INIT_TIMEOUT
    0x03: "REBOOT",      # HIP_FEEDBACK_LOST
}

# ── Battery status widget ─────────────────────────────────────────────────────

class BatteryStatusWidget(QWidget):
    """Battery icon + voltage label for the status bar. Blinks at 1 Hz when critically low."""
    _VMAX = 25.2   # 4.2 V/cell × 6S
    _VMIN = 21.0   # 3.5 V/cell × 6S (display empty)
    _NBARS = 5

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
        vbus = (vl + vr) * 0.5
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


# ── Status bar ────────────────────────────────────────────────────────────────

def _vsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f

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
        self._test      = QLabel("~: —")
        self._test.setStyleSheet(f"color: {DIM}; font-family: Consolas;")
        self._batt      = BatteryStatusWidget()
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

        for w in (self._source, _vsep(), self._transport, _vsep(), self._dt, _vsep(), self._mode, _vsep(), self._test, _vsep(), self._batt):
            sb.addWidget(w)
        self._mismatch_lbl = QLabel("FIRMWARE MISMATCH — reflash ESP32/Teensy")
        self._mismatch_lbl.setStyleSheet(
            f"color: white; background: {RED}; font-weight: bold;"
            f" padding: 2px 10px; border-radius: 3px;"
        )
        self._mismatch_lbl.setVisible(False)

        sb.addPermanentWidget(self._mismatch_lbl)
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
        from source_manager import SourceManager, PRIORITY
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
        from source_manager import SourceManager
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

    def set_test_val(self, val: float):
        self._test.setStyleSheet(f"color: {TEXT}; font-family: Consolas;")
        self._test.setText(f"~: {val:+.3f}")

    def set_connected(self, connected: bool):
        if connected:
            self._conn.setStyleSheet(f"color: {GREEN};")
            self._conn.setText("● Connected")
        else:
            self._conn.setStyleSheet(f"color: {RED};")
            self._conn.setText("● Disconnected")
            self._mismatch_lbl.setVisible(False)
            self._batt.set_connected(False)

    def set_version_mismatch(self, got: int, expected: int):
        self._mismatch_lbl.setText(
            f"FIRMWARE MISMATCH (telem v{got} != v{expected}) — reflash ESP32/Teensy"
        )
        self._mismatch_lbl.setVisible(True)

    def clear_version_mismatch(self):
        self._mismatch_lbl.setVisible(False)

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

        from wifi_transport import WifiTransport
        WifiTransport.instance().start()

        tabs = QTabWidget()
        tabs.addTab(DashboardTab(),       "Dashboard")
        tabs.addTab(ImuTab(),             "IMU")
        tabs.addTab(RawDataTab(),         "Raw Data")
        tabs.addTab(HipMotorsTab(),       "Hip Motors")
        tabs.addTab(ParamsTab(),          "Parameters")
        tabs.addTab(RobotVisualizerTab(), "Visualizer")
        tabs.addTab(WheelMotorsTab(),     "Wheel Motors")
        tabs.addTab(ControllersTab(),     "Controllers")
        tabs.addTab(RadioTab(),           "Radio")
        tabs.addTab(FlashMonitorTab(),    "Flash & Monitor")
        self._flash_tab_idx = tabs.count() - 1

        self._log_pane = RobotLogWidget()
        self._log_pane.setContentsMargins(8, 2, 8, 6)

        central = QWidget()
        central_lay = QVBoxLayout(central)
        central_lay.setContentsMargins(0, 0, 0, 0)
        central_lay.setSpacing(0)
        central_lay.addWidget(tabs)
        central_lay.addWidget(self._log_pane)
        self.setCentralWidget(central)

        tabs.currentChanged.connect(self._on_tab_changed)

        self._on_source_changed(sm.active)

    def _on_tab_changed(self, idx: int):
        self._log_pane.setVisible(idx != self._flash_tab_idx)

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
        test_val = info.get("test_val")
        if test_val is not None:
            self.status.set_test_val(test_val)

# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    _kill_other_instances()
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
