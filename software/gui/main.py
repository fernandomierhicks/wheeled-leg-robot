import sys
import os
import psutil

def _kill_other_instances():
    current_pid = os.getpid()
    current_script = os.path.abspath(__file__)
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if proc.info['pid'] == current_pid:
            continue
        cmdline = proc.info.get('cmdline') or []
        if any(os.path.abspath(arg) == current_script for arg in cmdline if arg):
            proc.kill()

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QLabel, QVBoxLayout, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from theme import APP_STYLE, BORDER, TEXT, DIM, GREEN, ORANGE, RED, BLUE
from flash_monitor import FlashMonitorTab
from hip_motors import HipMotorsTab
from imu_tab import ImuTab
from raw_data_tab import RawDataTab

# ── Placeholder tabs ──────────────────────────────────────────────────────────

class _PlaceholderTab(QWidget):
    def __init__(self, name: str):
        super().__init__()
        lbl = QLabel(name)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFont(QFont("Segoe UI", 22))
        lbl.setStyleSheet(f"color: {BORDER};")
        QVBoxLayout(self).addWidget(lbl)

class DashboardTab(_PlaceholderTab):
    def __init__(self): super().__init__("Dashboard")

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
        self._conn      = QLabel("● Disconnected")
        self._conn.setStyleSheet(f"color: {RED};")

        for w in (self._source, _vsep(), self._transport, _vsep(), self._dt):
            sb.addWidget(w)
        sb.addPermanentWidget(_vsep())
        sb.addPermanentWidget(self._conn)

    def set_source(self, src: str):
        color = BLUE if src == "TEENSY" else ORANGE
        self._source.setStyleSheet(f"color: {color};")
        self._source.setText(f"● {src}")

    def set_transport(self, transport: str):
        self._transport.setText(transport)

    def set_dt(self, dt_ms: float):
        self._dt.setText(f"dt: {dt_ms:.1f} ms")

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

        tabs = QTabWidget()
        tabs.addTab(DashboardTab(),    "Dashboard")
        tabs.addTab(ImuTab(),          "IMU")
        tabs.addTab(RawDataTab(),      "Raw Data")
        tabs.addTab(HipMotorsTab(),    "Hip Motors")
        tabs.addTab(WheelMotorsTab(),  "Wheel Motors")
        tabs.addTab(FlashMonitorTab(), "Flash & Monitor")
        self.setCentralWidget(tabs)

        self.status = StatusBar(self.statusBar())

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
