"""odrive_gui.py — ODrive 3.6 unified tuning GUI (Step 1: Core + Connect + Status Bar).

Usage:
    python odrive_gui.py

Requires: pip install PySide6 odrive
"""

import logging
import os
import sys

import psutil
from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QButtonGroup,
    QFrame, QTabWidget, QMessageBox,
)

from core.odrive_manager import ODriveManager
from core.odrive_errors import (
    decode_errors_str, AXIS_ERRORS, MOTOR_ERRORS, CONTROLLER_ERRORS,
)
from core.constants import POLL_MS
from cmd.cmd_interface import CmdInterface
from tabs.tab_setup import TabSetup
from tabs.tab_control import TabControl
from tabs.tab_anticogging import TabAnticogging
from tabs.tab_inspector import TabInspector
from tabs.tab_terminal import TabTerminal
from tabs.tab_presets import TabPresets
from ui.theme import (
    DARK_STYLE, CLR_OK, CLR_WARN, CLR_ERR, CLR_INFO, CLR_MUTED,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
_LOG_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_LOG_DIR, "odrive_gui.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("odrive_gui")

# File handler — continuous log for Claude to read
_fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                    datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(_fh)


# ── Connect worker (runs in QThread so GUI stays responsive) ──────────────────

class ConnectWorker(QThread):
    success = Signal(object)
    failed = Signal(str)

    def __init__(self, manager: ODriveManager):
        super().__init__()
        self._mgr = manager

    def run(self):
        try:
            odrv = self._mgr.connect()
            self.success.emit(odrv)
        except Exception as e:
            self.failed.emit(str(e))


class ReconnectWorker(QThread):
    success = Signal(object)
    failed = Signal(str)

    def __init__(self, manager: ODriveManager):
        super().__init__()
        self._mgr = manager

    def run(self):
        try:
            self._mgr.disconnect()
            odrv = self._mgr.connect()
            self.success.emit(odrv)
        except Exception as e:
            self.failed.emit(str(e))


# ── Main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, manager: ODriveManager, cmd_interface: CmdInterface):
        super().__init__()
        self.setWindowTitle("ODrive Unified  —  Motor Tuning")
        self.resize(900, 700)

        self._mgr = manager
        self._cmd = cmd_interface
        self._connect_worker: ConnectWorker | None = None
        self._reconnect_worker: ReconnectWorker | None = None
        self._erase_reboot_timer = QTimer(self)
        self._erase_reboot_timer.setSingleShot(True)
        self._erase_reboot_timer.timeout.connect(self._reconnect_after_erase)

        # Poll timer (100 ms) for live readouts
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_MS)
        self._poll_timer.timeout.connect(self._poll)

        self._build()
        self._poll_timer.start()
        self._cmd.start()
        self._cmd.connect_finished.connect(self._on_cmd_connect)
        self._tab_setup.reconnected.connect(self._on_setup_reconnected)
        self._tab_anticog.reconnected.connect(self._on_setup_reconnected)
        self._tab_presets.reconnected.connect(self._on_setup_reconnected)

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # ── top bar ───────────────────────────────────────────────────────────
        top = QHBoxLayout()

        # Axis selector
        top.addWidget(QLabel("Axis:"))
        self._ax_grp = QButtonGroup(self)
        self._rb_ax = []
        for i in range(2):
            rb = QRadioButton(f"  {i}  ")
            rb.setChecked(i == 0)
            self._ax_grp.addButton(rb, i)
            top.addWidget(rb)
            self._rb_ax.append(rb)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #444;")
        top.addWidget(sep)

        # Connect button
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setFixedWidth(90)
        self.btn_connect.clicked.connect(self._toggle_connect)
        top.addWidget(self.btn_connect)

        # Status label
        self.lbl_status = QLabel("Not connected")
        self.lbl_status.setStyleSheet(f"color: {CLR_MUTED}; font-family: monospace;")
        top.addWidget(self.lbl_status)
        top.addStretch()

        # Erase Config button (far right, away from other controls)
        self.btn_erase = QPushButton("Erase Config")
        self.btn_erase.setFixedWidth(110)
        self.btn_erase.setStyleSheet(f"color: {CLR_ERR};")
        self.btn_erase.clicked.connect(self._on_erase_config)
        top.addWidget(self.btn_erase)

        root.addLayout(top)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444;")
        root.addWidget(line)

        # ── Vbus + axis state readout ─────────────────────────────────────────
        readout = QHBoxLayout()

        self.lbl_vbus = QLabel("Vbus: —")
        self.lbl_vbus.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {CLR_OK};")
        readout.addWidget(self.lbl_vbus)

        self.lbl_axis_state = QLabel("State: —")
        self.lbl_axis_state.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {CLR_OK};")
        readout.addWidget(self.lbl_axis_state)

        self.lbl_errors = QLabel("Errors: —")
        self.lbl_errors.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {CLR_OK};")
        readout.addWidget(self.lbl_errors)

        readout.addStretch()
        root.addLayout(readout)

        # ── tabs ──────────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        # Real Setup tab (Step 3)
        self._tab_setup = TabSetup(self._mgr, self._ax_idx)
        self.tabs.addTab(self._tab_setup, "Setup")

        # Real Control tab (Step 4)
        self._tab_control = TabControl(self._mgr, self._ax_idx)
        self.tabs.addTab(self._tab_control, "Control")

        # Real Anticogging tab (Step 5)
        self._tab_anticog = TabAnticogging(self._mgr, self._ax_idx)
        self.tabs.addTab(self._tab_anticog, "Anticogging")

        # Inspector tab (Step 6)
        self._tab_inspector = TabInspector(self._mgr, self._ax_idx)
        self.tabs.addTab(self._tab_inspector, "Inspector")

        # Presets tab (Step 8)
        self._tab_presets = TabPresets(self._mgr, self._ax_idx)
        self.tabs.addTab(self._tab_presets, "Presets")

        # Real Terminal tab (Step 2)
        self._tab_terminal = TabTerminal(self._cmd)
        self.tabs.addTab(self._tab_terminal, "Terminal")


    # ── Axis helper ───────────────────────────────────────────────────────────

    def _ax_idx(self) -> int:
        return self._ax_grp.checkedId()

    # ── Connection ────────────────────────────────────────────────────────────

    def _toggle_connect(self):
        if self._mgr.connected:
            self._mgr.disconnect()
            self.btn_connect.setText("Connect")
            self._set_status("Disconnected", CLR_MUTED)
            self._clear_readouts()
            return

        self.btn_connect.setEnabled(False)
        self._set_status("Searching for ODrive...", CLR_INFO)
        self._connect_worker = ConnectWorker(self._mgr)
        self._connect_worker.success.connect(self._on_connect_ok)
        self._connect_worker.failed.connect(self._on_connect_fail)
        self._connect_worker.start()

    def _on_connect_ok(self, odrv):
        info = self._mgr.device_info()
        if info:
            fw_tag = f"fw{info['fw']}"
            if info["fw_unreleased"]:
                fw_tag += " (dev)"
            self._set_status(
                f"Connected — hw{info['hw']}  {fw_tag}  serial {info['serial']}",
                CLR_OK,
            )
        else:
            self._set_status("Connected", CLR_OK)
        self.btn_connect.setText("Disconnect")
        self.btn_connect.setEnabled(True)
        self._refresh_tabs()
        log.info("GUI: connect success")

    def _refresh_tabs(self):
        """Read config from ODrive into all tab forms."""
        self._tab_setup.on_connected()
        self._tab_control.on_connected()

    def _on_connect_fail(self, msg):
        self._set_status(f"Connect failed: {msg}", CLR_ERR)
        self.btn_connect.setText("Connect")
        self.btn_connect.setEnabled(True)
        log.error("GUI: connect failed: %s", msg)

    def _on_setup_reconnected(self):
        """Handle reconnect after reboot triggered by Setup tab."""
        if self._mgr.connected:
            info = self._mgr.device_info()
            if info:
                self._set_status(f"Reconnected — hw{info['hw']}  fw{info['fw']}", CLR_OK)
            else:
                self._set_status("Reconnected", CLR_OK)
            self.btn_connect.setText("Disconnect")
            self.btn_connect.setEnabled(True)
            self._tab_control.on_connected()

    def _on_cmd_connect(self, success: bool, msg: str):
        """Handle connect/disconnect triggered via inbox __CONNECT__ command."""
        if success:
            self._set_status(msg, CLR_OK)
            self.btn_connect.setText("Disconnect")
            self.btn_connect.setEnabled(True)
            self._refresh_tabs()
        else:
            self._set_status(msg, CLR_ERR)
            self.btn_connect.setText("Connect")
            self.btn_connect.setEnabled(True)

    # ── Erase Config ────────────────────────────────────────────────────────

    _ERASE_REBOOT_WAIT_MS = 4000

    def _on_erase_config(self):
        if not self._mgr.connected:
            self._set_status("Not connected — cannot erase", CLR_ERR)
            return

        reply = QMessageBox.warning(
            self,
            "Erase Configuration",
            "This will erase ALL configuration and restore factory defaults.\n"
            "The ODrive will reboot.\n\n"
            "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        log.info("GUI: erasing configuration (factory reset)")
        try:
            self._mgr.odrv.erase_configuration()
        except Exception:
            pass  # erase triggers reboot — USB exception expected

        self.btn_connect.setEnabled(False)
        self.btn_erase.setEnabled(False)
        self._mgr.disconnect()
        self._set_status("Erased — rebooting ODrive...", CLR_WARN)
        self._clear_readouts()
        self._erase_reboot_timer.start(self._ERASE_REBOOT_WAIT_MS)

    def _reconnect_after_erase(self):
        self._set_status("Reconnecting after erase...", CLR_INFO)
        self._reconnect_worker = ReconnectWorker(self._mgr)
        self._reconnect_worker.success.connect(self._on_erase_reconnect_ok)
        self._reconnect_worker.failed.connect(self._on_erase_reconnect_fail)
        self._reconnect_worker.start()

    def _on_erase_reconnect_ok(self, odrv):
        info = self._mgr.device_info()
        if info:
            self._set_status(
                f"Factory defaults restored — hw{info['hw']}  fw{info['fw']}",
                CLR_OK,
            )
        else:
            self._set_status("Factory defaults restored", CLR_OK)
        self.btn_connect.setText("Disconnect")
        self.btn_connect.setEnabled(True)
        self.btn_erase.setEnabled(True)
        self._refresh_tabs()
        log.info("GUI: erase complete, reconnected")

    def _on_erase_reconnect_fail(self, msg):
        self._set_status(f"Erase OK but reconnect failed: {msg}", CLR_ERR)
        self.btn_connect.setText("Connect")
        self.btn_connect.setEnabled(True)
        self.btn_erase.setEnabled(True)
        log.error("GUI: reconnect after erase failed: %s", msg)

    def _set_status(self, msg: str, color: str):
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet(f"color: {color}; font-family: monospace;")

    # ── Poll (100 ms) ─────────────────────────────────────────────────────────

    def _poll(self):
        # Tab poll updates (must run even when disconnected)
        self._tab_setup.poll_update()
        self._tab_control.poll_update()
        self._tab_anticog.poll_update()

        if not self._mgr.connected:
            return
        try:
            ax = self._ax_idx()

            # Vbus
            vbus = self._mgr.vbus_voltage()
            if vbus is not None:
                color = CLR_OK if vbus > 10.0 else CLR_WARN
                self.lbl_vbus.setText(f"Vbus: {vbus:.1f} V")
                self.lbl_vbus.setStyleSheet(
                    f"font-family: monospace; font-size: 12px; color: {color};")

            # Axis state
            state_str = self._mgr.axis_state_str(ax)
            self.lbl_axis_state.setText(f"State: {state_str}")

            # Errors — encoder suppressed: mode 256 (CUI) reads fine but
            # raises spurious SPI errors that drown out real faults.
            errs = self._mgr.axis_errors(ax)
            err_tables = [("axis", AXIS_ERRORS), ("motor", MOTOR_ERRORS),
                          ("controller", CONTROLLER_ERRORS)]
            any_err = any(errs[k] != 0 for k, _ in err_tables)
            if any_err:
                parts = []
                for key, table in err_tables:
                    val = errs[key]
                    if val != 0:
                        parts.append(f"{key}: {decode_errors_str(val, table)}")
                err_text = " | ".join(parts)
                self.lbl_errors.setText(f"Errors: {err_text}")
                self.lbl_errors.setStyleSheet(
                    f"font-family: monospace; font-size: 12px; color: {CLR_ERR};")
            else:
                self.lbl_errors.setText("Errors: none")
                self.lbl_errors.setStyleSheet(
                    f"font-family: monospace; font-size: 12px; color: {CLR_OK};")

        except Exception:
            # USB disconnect or stale handle
            self._mgr.disconnect()
            self.btn_connect.setText("Connect")
            self._set_status("Disconnected (USB lost)", CLR_ERR)
            self._clear_readouts()

    def _clear_readouts(self):
        self.lbl_vbus.setText("Vbus: —")
        self.lbl_vbus.setStyleSheet(
            f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")
        self.lbl_axis_state.setText("State: —")
        self.lbl_axis_state.setStyleSheet(
            f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")
        self.lbl_errors.setText("Errors: —")
        self.lbl_errors.setStyleSheet(
            f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")


# ── Kill stale processes that may hold the ODrive USB interface ───────────────

# Substrings in the command line that indicate an ODrive-related Python process.
_ODRIVE_KEYWORDS = ["odrive_gui", "odrivetool", "odrive_gui_v2"]

def _kill_stale_odrive_processes():
    """Find and kill other Python processes that might hold the ODrive USB handle."""
    my_pid = os.getpid()
    killed = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.pid == my_pid:
                continue
            name = (proc.info["name"] or "").lower()
            if "python" not in name:
                continue
            cmdline = " ".join(proc.info["cmdline"] or []).lower()
            if any(kw in cmdline for kw in _ODRIVE_KEYWORDS):
                log.warning("Killing stale process PID %d: %s", proc.pid, cmdline)
                proc.kill()
                killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if killed:
        # Give OS a moment to release USB handles
        psutil.wait_procs(
            [psutil.Process(p) for p in killed if psutil.pid_exists(p)],
            timeout=3,
        )
        log.info("Killed %d stale process(es): %s", len(killed), killed)
    else:
        log.info("No stale ODrive processes found.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    _kill_stale_odrive_processes()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)

    manager = ODriveManager()
    cmd = CmdInterface(manager)
    win = MainWindow(manager, cmd)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
