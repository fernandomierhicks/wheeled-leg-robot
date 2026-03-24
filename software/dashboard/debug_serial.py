"""debug_serial.py — Serial debug GUI for the wheeled-leg robot firmware.

Connects to the Arduino UNO R4 WiFi running the debug build (1 Mbps)
and provides an interactive terminal with buttons for boot-menu toggles
and runtime commands.  Also includes a Flash Firmware tab for one-click
PlatformIO builds and uploads.

Usage:
    python debug_serial.py              # auto-detect COM port
    python debug_serial.py --port COM5  # explicit port
"""

import argparse
import os
import subprocess
import sys
import threading
import time

import serial
import serial.tools.list_ports
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

FIRMWARE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "firmware")
)

# PlatformIO CLI — VSCode extension installs into ~/.platformio, not on PATH
_PIO_VENV = os.path.join(os.path.expanduser("~"), ".platformio", "penv", "Scripts", "pio.exe")
PIO_CMD = _PIO_VENV if os.path.isfile(_PIO_VENV) else "pio"

BAUD_RATE = 1_000_000


# ── Serial reader thread ────────────────────────────────────────────────────

class SerialReader(QtCore.QObject):
    """Reads serial data in a background thread, emits lines to the GUI."""
    line_received = QtCore.Signal(str)
    disconnected = QtCore.Signal()

    def __init__(self, ser):
        super().__init__()
        self.ser = ser
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        buf = b""
        while not self._stop.is_set():
            try:
                n = self.ser.in_waiting
                if n > 0:
                    chunk = self.ser.read(n)
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        text = line.decode("utf-8", errors="replace").rstrip("\r")
                        self.line_received.emit(text)
                else:
                    time.sleep(0.01)
            except (serial.SerialException, OSError):
                self.disconnected.emit()
                return


# ── Main window ─────────────────────────────────────────────────────────────

class DebugWindow(QtWidgets.QMainWindow):
    def __init__(self, port_name=None):
        super().__init__()
        self.setWindowTitle("Debug Serial — Wheeled-Leg Robot")
        self.resize(900, 640)

        self.ser = None
        self.reader = None
        self.reader_thread = None
        self._flash_proc = None  # running pio subprocess

        # ── Central tab widget ──
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs)

        self.tabs.addTab(self._build_serial_tab(port_name), "Serial Debug")
        self.tabs.addTab(self._build_flash_tab(), "Flash Firmware")

    # ── Serial Debug tab ───────────────────────────────────────────────────

    def _build_serial_tab(self, port_name):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        font = QtGui.QFont("Consolas", 10)
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)

        # ── Connection bar ──
        conn_bar = QtWidgets.QHBoxLayout()
        layout.addLayout(conn_bar)

        conn_bar.addWidget(QtWidgets.QLabel("Port:"))
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setMinimumWidth(160)
        conn_bar.addWidget(self.port_combo)

        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_ports)
        conn_bar.addWidget(self.refresh_btn)

        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self._toggle_connection)
        conn_bar.addWidget(self.connect_btn)

        self.status_label = QtWidgets.QLabel("Disconnected")
        self.status_label.setStyleSheet("color: #e44; font-weight: bold;")
        conn_bar.addWidget(self.status_label)
        conn_bar.addStretch()

        # ── Terminal output ──
        self.terminal = QtWidgets.QPlainTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setMaximumBlockCount(5000)
        self.terminal.setFont(font)
        self.terminal.setStyleSheet(
            "background-color: #1e1e1e; color: #dcdcdc; border: 1px solid #555;"
        )
        layout.addWidget(self.terminal, stretch=1)

        # ── Boot menu buttons ──
        boot_group = QtWidgets.QGroupBox("Boot Menu")
        boot_layout = QtWidgets.QHBoxLayout(boot_group)
        boot_buttons = [
            ("I — IMU",      "i"),
            ("L — LED",      "l"),
            ("W — Watchdog", "w"),
            ("F — WiFi",     "f"),
            ("T — Telemetry","t"),
            ("C — Commands", "c"),
            ("D — Defaults", "d"),
            ("G — GO",       "g"),
        ]
        for label, char in boot_buttons:
            btn = QtWidgets.QPushButton(label)
            btn.setFixedHeight(32)
            btn.clicked.connect(lambda _, ch=char: self._send(ch))
            boot_layout.addWidget(btn)
        layout.addWidget(boot_group)

        # ── Runtime command buttons ──
        rt_group = QtWidgets.QGroupBox("Runtime Commands")
        rt_layout = QtWidgets.QHBoxLayout(rt_group)
        rt_buttons = [
            ("P — Profiler", "p"),
            ("S — State",    "s"),
            ("H — Help",     "h"),
        ]
        for label, char in rt_buttons:
            btn = QtWidgets.QPushButton(label)
            btn.setFixedHeight(32)
            btn.clicked.connect(lambda _, ch=char: self._send(ch))
            rt_layout.addWidget(btn)
        rt_layout.addStretch()

        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.setFixedHeight(32)
        self.clear_btn.clicked.connect(self.terminal.clear)
        rt_layout.addWidget(self.clear_btn)
        layout.addWidget(rt_group)

        # ── Manual send bar ──
        send_bar = QtWidgets.QHBoxLayout()
        layout.addLayout(send_bar)
        self.send_input = QtWidgets.QLineEdit()
        self.send_input.setPlaceholderText("Type text and press Enter to send...")
        self.send_input.setFont(font)
        self.send_input.returnPressed.connect(self._send_input_text)
        send_bar.addWidget(self.send_input)
        send_btn = QtWidgets.QPushButton("Send")
        send_btn.clicked.connect(self._send_input_text)
        send_bar.addWidget(send_btn)

        # ── Populate ports ──
        self._refresh_ports()
        if port_name:
            idx = self.port_combo.findText(port_name)
            if idx >= 0:
                self.port_combo.setCurrentIndex(idx)
            else:
                self.port_combo.addItem(port_name)
                self.port_combo.setCurrentIndex(self.port_combo.count() - 1)
            QtCore.QTimer.singleShot(200, self._toggle_connection)

        return tab

    # ── Flash Firmware tab ─────────────────────────────────────────────────

    def _build_flash_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        font = QtGui.QFont("Consolas", 10)
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)

        # Firmware flavors: (label, description, env, targets, upload?)
        flavors = [
            ("USB-UART (default)", "Standard build, USB upload",
             "uno_r4_wifi", [], True),
            ("WiFi Telemetry",     "WiFi telemetry + commands, USB upload",
             "wifi",        [], True),
            ("WiFi + OTA",         "WiFi build, upload over-the-air",
             "ota",         [], True),
            ("Debug",              "Debug harness with boot menu, USB upload",
             "debug",       [], True),
        ]

        # ── Buttons grid ──
        btn_group = QtWidgets.QGroupBox("Select Firmware to Flash")
        btn_layout = QtWidgets.QGridLayout(btn_group)

        self._flash_buttons = []
        for row, (label, desc, env, targets, upload) in enumerate(flavors):
            name_label = QtWidgets.QLabel(f"<b>{label}</b><br/><small>{desc}</small>")
            btn_layout.addWidget(name_label, row, 0)

            build_btn = QtWidgets.QPushButton("Build")
            build_btn.setFixedHeight(36)
            build_btn.setFixedWidth(100)
            build_btn.clicked.connect(
                lambda _, e=env, t=targets: self._run_pio(e, t, upload=False)
            )
            btn_layout.addWidget(build_btn, row, 1)

            upload_btn = QtWidgets.QPushButton("Build + Upload")
            upload_btn.setFixedHeight(36)
            upload_btn.setFixedWidth(140)
            upload_btn.clicked.connect(
                lambda _, e=env, t=targets: self._run_pio(e, t, upload=True)
            )
            btn_layout.addWidget(upload_btn, row, 2)

            self._flash_buttons.extend([build_btn, upload_btn])

        layout.addWidget(btn_group)

        # ── Output console ──
        self.flash_output = QtWidgets.QPlainTextEdit()
        self.flash_output.setReadOnly(True)
        self.flash_output.setMaximumBlockCount(5000)
        self.flash_output.setFont(font)
        self.flash_output.setStyleSheet(
            "background-color: #1e1e1e; color: #dcdcdc; border: 1px solid #555;"
        )
        layout.addWidget(self.flash_output, stretch=1)

        # ── Bottom bar ──
        bottom = QtWidgets.QHBoxLayout()
        layout.addLayout(bottom)

        self.flash_status = QtWidgets.QLabel("Ready")
        self.flash_status.setStyleSheet("font-weight: bold;")
        bottom.addWidget(self.flash_status)
        bottom.addStretch()

        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setFixedHeight(32)
        clear_btn.clicked.connect(self.flash_output.clear)
        bottom.addWidget(clear_btn)

        # Timer to poll subprocess output
        self._flash_timer = QtCore.QTimer()
        self._flash_timer.timeout.connect(self._poll_flash_output)

        return tab

    # ── PlatformIO helpers ─────────────────────────────────────────────────

    def _run_pio(self, env, extra_targets, upload=False):
        if self._flash_proc and self._flash_proc.poll() is None:
            self.flash_output.appendPlainText("[GUI] A build is already running.")
            return

        cmd = [PIO_CMD, "run", "-e", env]
        if upload:
            cmd += ["-t", "upload"]
        for t in extra_targets:
            cmd += ["-t", t]

        self.flash_output.appendPlainText(f"[GUI] $ {' '.join(cmd)}")
        self.flash_output.appendPlainText(f"[GUI] Working dir: {FIRMWARE_DIR}")
        self.flash_status.setText(f"Running: {' '.join(cmd)}")
        self.flash_status.setStyleSheet("color: #fc4; font-weight: bold;")

        for btn in self._flash_buttons:
            btn.setEnabled(False)

        try:
            self._flash_proc = subprocess.Popen(
                cmd,
                cwd=FIRMWARE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            self.flash_output.appendPlainText(
                "[GUI] ERROR: 'pio' not found. Is PlatformIO CLI installed and on PATH?"
            )
            self._flash_done(ok=False)
            return

        self._flash_timer.start(100)

    def _poll_flash_output(self):
        if self._flash_proc is None:
            self._flash_timer.stop()
            return

        # Read available lines (non-blocking via poll + readline)
        while True:
            line = self._flash_proc.stdout.readline()
            if not line:
                break
            self.flash_output.appendPlainText(line.rstrip("\n"))

        if self._flash_proc.poll() is not None:
            # Process finished — drain remaining output
            rest = self._flash_proc.stdout.read()
            if rest:
                for line in rest.splitlines():
                    self.flash_output.appendPlainText(line)
            ok = self._flash_proc.returncode == 0
            self._flash_done(ok=ok)

    def _flash_done(self, ok):
        self._flash_timer.stop()
        self._flash_proc = None
        for btn in self._flash_buttons:
            btn.setEnabled(True)
        if ok:
            self.flash_status.setText("Success")
            self.flash_status.setStyleSheet("color: #4e4; font-weight: bold;")
            self.flash_output.appendPlainText("[GUI] === BUILD/UPLOAD SUCCEEDED ===")
        else:
            self.flash_status.setText("Failed")
            self.flash_status.setStyleSheet("color: #e44; font-weight: bold;")
            self.flash_output.appendPlainText("[GUI] === BUILD/UPLOAD FAILED ===")

    # ── Port management ──────────────────────────────────────────────────────

    def _refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in sorted(ports, key=lambda x: x.device):
            desc = f"{p.device}  ({p.description})" if p.description != p.device else p.device
            self.port_combo.addItem(desc, userData=p.device)

    # ── Connect / disconnect ─────────────────────────────────────────────────

    def _toggle_connection(self):
        if self.ser and self.ser.is_open:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self.port_combo.currentData()
        if not port:
            port = self.port_combo.currentText().split()[0]
        try:
            self.ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
        except serial.SerialException as e:
            self._append(f"[GUI] Failed to open {port}: {e}")
            return

        self._append(f"[GUI] Connected to {port} @ {BAUD_RATE} baud")
        self.status_label.setText(f"Connected: {port}")
        self.status_label.setStyleSheet("color: #4e4; font-weight: bold;")
        self.connect_btn.setText("Disconnect")

        # Start reader thread
        self.reader = SerialReader(self.ser)
        self.reader.line_received.connect(self._append)
        self.reader.disconnected.connect(self._on_disconnect)
        self.reader_thread = threading.Thread(target=self.reader.run, daemon=True)
        self.reader_thread.start()

    def _disconnect(self):
        if self.reader:
            self.reader.stop()
            self.reader = None
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self.status_label.setText("Disconnected")
        self.status_label.setStyleSheet("color: #e44; font-weight: bold;")
        self.connect_btn.setText("Connect")
        self._append("[GUI] Disconnected")

    def _on_disconnect(self):
        self._disconnect()
        self._append("[GUI] Serial connection lost")

    # ── Send helpers ─────────────────────────────────────────────────────────

    def _send(self, text):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(text.encode("utf-8"))
            except serial.SerialException as e:
                self._append(f"[GUI] Send error: {e}")
        else:
            self._append("[GUI] Not connected")

    def _send_input_text(self):
        text = self.send_input.text()
        if text:
            self._send(text)
            self._append(f"[TX] {text}")
            self.send_input.clear()

    # ── Terminal output ──────────────────────────────────────────────────────

    def _append(self, text):
        self.terminal.appendPlainText(text)

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._disconnect()
        event.accept()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Debug serial GUI for wheeled-leg robot")
    parser.add_argument("--port", "-p", default=None,
                        help="Serial port (e.g. COM5). Auto-detect if omitted.")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(55, 55, 55))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
    app.setPalette(palette)

    win = DebugWindow(port_name=args.port)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
