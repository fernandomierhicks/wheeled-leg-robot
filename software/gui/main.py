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
    QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from theme import APP_STYLE, BORDER, TEXT, DIM, GREEN, ORANGE, RED, BLUE, YELLOW, WHITE
from flash_monitor import FlashMonitorTab
from hip_motors import HipMotorsTab
from imu_tab import ImuTab, ImuMiniWidget
from params_tab import ParamsTab
from raw_data_tab import RawDataTab
from robot_visualizer_tab import RobotVisualizerTab
from radio_tab import RadioTab
from telemetry_bus import TelemetryBus
from source_manager import SourceManager, TRANSPORT_LABEL
from comm_commands import send_set_mode, send_reboot, STATE_STARTUP, STATE_ESTOP

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

        top = QHBoxLayout()
        top.setContentsMargins(8, 8, 0, 0)
        top.setSpacing(12)
        top.addWidget(imu)
        top.addWidget(test_val)
        top.addStretch(7)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(top)
        outer.addStretch(1)

class WheelMotorsTab(_PlaceholderTab):
    def __init__(self): super().__init__("Wheel Motors")

# ── Status bar ────────────────────────────────────────────────────────────────

def _vsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f

class StatusBar:
    def __init__(self, sb):
        self._source    = QLabel("● —")
        self._transport = QLabel("—")
        self._dt        = QLabel("dt: —")
        self._mode      = QLabel("—")
        self._mode.setStyleSheet(f"color: {DIM};")
        self._test      = QLabel("~: —")
        self._test.setStyleSheet(f"color: {DIM}; font-family: Consolas;")
        self._conn      = QLabel("● Disconnected")
        self._conn.setStyleSheet(f"color: {RED};")

        self._btn_estop = QPushButton("ESTOP")
        self._btn_estop.setStyleSheet(
            f"QPushButton{{background:{RED};color:white;font-weight:bold;font-size:14px;"
            f"border:1px solid {RED};border-radius:4px;padding:6px 24px;margin:2px 6px}}"
            f"QPushButton:hover{{background:#ff6e60}}"
            f"QPushButton:pressed{{background:#b2362a}}"
        )
        self._btn_estop.clicked.connect(lambda: send_set_mode(STATE_ESTOP))

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setStyleSheet(
            f"QPushButton{{background:#4a1a1a;color:white;"
            f"border:1px solid {BORDER};border-radius:3px;padding:2px 10px}}"
            f"QPushButton:hover{{background:#4a1a1acc}}"
            f"QPushButton:pressed{{background:#4a1a1a88}}"
            f"QPushButton:disabled{{background:transparent;color:{DIM};"
            f"border:1px solid {BORDER}}}"
        )
        self._btn_reset.setEnabled(False)
        self._btn_reset.clicked.connect(lambda: send_set_mode(STATE_STARTUP))

        self._btn_reboot = QPushButton("Reboot")
        self._btn_reboot.setStyleSheet(self._btn_reset.styleSheet())
        self._btn_reboot.setEnabled(False)
        self._btn_reboot.clicked.connect(self._on_reboot_clicked)

        for w in (self._source, _vsep(), self._transport, _vsep(), self._dt, _vsep(), self._mode, _vsep(), self._test):
            sb.addWidget(w)
        sb.addPermanentWidget(self._btn_estop)
        sb.addPermanentWidget(_vsep())
        sb.addPermanentWidget(self._btn_reset)
        sb.addPermanentWidget(self._btn_reboot)
        sb.addPermanentWidget(_vsep())
        sb.addPermanentWidget(self._conn)

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

    def set_source(self, src: str):
        color = {
            "TEENSY": BLUE,
            "ESP32":  ORANGE,
        }.get(src, DIM)
        self._source.setStyleSheet(f"color: {color};")
        self._source.setText(f"● {src}" if src != "—" else "● —")

    def set_transport(self, transport: str):
        self._transport.setText(transport)

    def set_dt(self, dt_ms: float):
        self._dt.setText(f"dt: {dt_ms:.1f} ms")

    def set_mode(self, state: str, fault: str = "", fault_description: str = ""):
        if state == "—":
            self._mode.setStyleSheet(f"color: {DIM};")
            self._mode.setText("—")
            self._mode.setToolTip("")
            self._btn_reset.setEnabled(False)
            self._btn_reboot.setEnabled(False)
            return
        color = {"RUNNING": GREEN, "ESTOP": RED, "CALIBRATION": BLUE, "STANDBY": YELLOW, "STARTUP": WHITE}.get(state, DIM)
        self._mode.setStyleSheet(f"color: {color}; font-weight: bold;")
        is_fault = state == "ESTOP" and fault and fault != "NONE"
        label = f"{state}  [{fault}]" if is_fault else state
        self._mode.setText(label)
        self._mode.setToolTip(fault_description if is_fault else "")
        self._btn_reset.setEnabled(state == "ESTOP")
        self._btn_reboot.setEnabled(True)

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

        tabs = QTabWidget()
        tabs.addTab(DashboardTab(),       "Dashboard")
        tabs.addTab(ImuTab(),             "IMU")
        tabs.addTab(RawDataTab(),         "Raw Data")
        tabs.addTab(HipMotorsTab(),       "Hip Motors")
        tabs.addTab(ParamsTab(),          "Parameters")
        tabs.addTab(RobotVisualizerTab(), "Visualizer")
        tabs.addTab(WheelMotorsTab(),     "Wheel Motors")
        tabs.addTab(RadioTab(),           "Radio")
        tabs.addTab(FlashMonitorTab(),    "Flash & Monitor")
        self.setCentralWidget(tabs)

        self._on_source_changed(sm.active)

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
        ts = info.get("timestamp_ms")
        if ts is not None and self._last_ts_ms is not None:
            self.status.set_dt(ts - self._last_ts_ms)
        if ts is not None:
            self._last_ts_ms = ts
        state = info.get("state_name")
        if state:
            self.status.set_mode(state, info.get("fault_name", ""), info.get("fault_description", ""))
        test_val = info.get("test_val")
        if test_val is not None:
            self.status.set_test_val(test_val)
        self._disconnect_timer.start()

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
