"""odrive_gui.py — PySide6 configuration panel for ODrive 3.6 (firmware 0.5.x)

Usage:
    python odrive_gui.py

Requires: pip install PySide6 odrive pyserial
"""

import sys
import time
from collections import deque

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget,
    QFrame,
)
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtGui import QPainter, QColor

import odrive

# ── Constants ─────────────────────────────────────────────────────────────────
MOTOR_TYPE_NAMES  = ["High Current (0)", "Gimbal (2)", "ACIM (3)"]
MOTOR_TYPE_VALUES = [0, 2, 3]

AXIS_STATES = {
    0: "Undefined", 1: "Idle", 2: "Startup Sequence",
    3: "Full Calibration", 4: "Motor Calibration",
    6: "Encoder Index Search", 7: "Encoder Offset Calibration",
    8: "Closed Loop", 9: "Lockin Spin", 10: "Encoder Dir Find",
    11: "Homing",
}

PLOT_LEN = 300
SIG_COLORS = {
    "Vbus (V)":       "#64ffb4",
    "Ax0 Iq (A)":    "#ff6464",
    "Ax1 Iq (A)":    "#ffa03c",
    "Ax0 Vel (t/s)": "#50a0ff",
    "Ax1 Vel (t/s)": "#a050ff",
    "Ax0 Pos (t)":   "#ffdc3c",
    "Ax1 Pos (t)":   "#c8ff50",
}


# ── Connect worker (runs find_any in a thread) ────────────────────────────────
class ConnectWorker(QThread):
    success = Signal(object)
    failed  = Signal(str)

    def run(self):
        try:
            odrv = odrive.find_any()
            self.success.emit(odrv)
        except Exception as e:
            self.failed.emit(str(e))


# ── Axis config widget ────────────────────────────────────────────────────────
class AxisWidget(QGroupBox):
    def __init__(self, ax_idx):
        super().__init__(f"Axis {ax_idx}")
        self.ax_idx = ax_idx
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        # Live state row
        state_row = QHBoxLayout()
        self.lbl_state = QLabel("State: —")
        self.lbl_temp  = QLabel("Temp: —")
        state_row.addWidget(self.lbl_state)
        state_row.addStretch()
        state_row.addWidget(self.lbl_temp)
        layout.addLayout(state_row)

        line = QFrame(); line.setFrameShape(QFrame.HLine)
        layout.addWidget(line)

        # Config form
        form = QFormLayout()
        self.motor_type = QComboBox()
        self.motor_type.addItems(MOTOR_TYPE_NAMES)
        self.pole_pairs    = QSpinBox();       self.pole_pairs.setRange(1, 50);      self.pole_pairs.setValue(7)
        self.current_lim   = QDoubleSpinBox(); self.current_lim.setRange(0, 60);     self.current_lim.setValue(10); self.current_lim.setSuffix(" A")
        self.cal_current   = QDoubleSpinBox(); self.cal_current.setRange(0, 60);     self.cal_current.setValue(10); self.cal_current.setSuffix(" A")
        self.vel_lim       = QDoubleSpinBox(); self.vel_lim.setRange(0, 500);        self.vel_lim.setValue(2);      self.vel_lim.setSuffix(" t/s")
        self.encoder_cpr   = QSpinBox();       self.encoder_cpr.setRange(1, 65536);  self.encoder_cpr.setValue(8192)
        self.pre_cal_motor = QCheckBox()
        self.pre_cal_enc   = QCheckBox()

        form.addRow("Motor type",             self.motor_type)
        form.addRow("Pole pairs",             self.pole_pairs)
        form.addRow("Current limit",          self.current_lim)
        form.addRow("Calibration current",    self.cal_current)
        form.addRow("Velocity limit",         self.vel_lim)
        form.addRow("Encoder CPR",            self.encoder_cpr)
        form.addRow("Pre-calibrated motor",   self.pre_cal_motor)
        form.addRow("Pre-calibrated encoder", self.pre_cal_enc)
        layout.addLayout(form)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply")
        self.btn_cal   = QPushButton("Full Calibration")
        self.btn_idle  = QPushButton("Set Idle")
        for b in (self.btn_apply, self.btn_cal, self.btn_idle):
            b.setEnabled(False)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

    def set_connected(self, on):
        for b in (self.btn_apply, self.btn_cal, self.btn_idle):
            b.setEnabled(on)

    def load(self, axis):
        mc = axis.motor.config
        ec = axis.encoder.config
        mt = getattr(mc, "motor_type", 0)
        idx = MOTOR_TYPE_VALUES.index(mt) if mt in MOTOR_TYPE_VALUES else 0
        self.motor_type.setCurrentIndex(idx)
        self.pole_pairs.setValue(getattr(mc, "pole_pairs", 7))
        self.current_lim.setValue(float(getattr(mc, "current_lim", 10)))
        self.cal_current.setValue(float(getattr(mc, "calibration_current", 10)))
        self.vel_lim.setValue(float(getattr(axis.controller.config, "vel_limit", 2)))
        self.encoder_cpr.setValue(getattr(ec, "cpr", 8192))
        self.pre_cal_motor.setChecked(bool(getattr(mc, "pre_calibrated", False)))
        self.pre_cal_enc.setChecked(bool(getattr(ec, "pre_calibrated", False)))

    def apply(self, axis):
        mc = axis.motor.config
        ec = axis.encoder.config
        mc.motor_type          = MOTOR_TYPE_VALUES[self.motor_type.currentIndex()]
        mc.pole_pairs          = self.pole_pairs.value()
        mc.current_lim         = self.current_lim.value()
        mc.calibration_current = self.cal_current.value()
        axis.controller.config.vel_limit = self.vel_lim.value()
        ec.cpr                 = self.encoder_cpr.value()
        mc.pre_calibrated      = self.pre_cal_motor.isChecked()
        ec.pre_calibrated      = self.pre_cal_enc.isChecked()

    def update_live(self, axis):
        state_id = getattr(axis, "current_state", 0)
        self.lbl_state.setText(f"State: {AXIS_STATES.get(state_id, str(state_id))}")
        try:
            temp = axis.motor.get_inverter_temp()
            self.lbl_temp.setText(f"Temp: {temp:.1f} °C")
        except Exception:
            self.lbl_temp.setText("Temp: —")


# ── Main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ODrive 3.6 Config")
        self.resize(560, 860)
        self.odrv = None
        self.connect_worker = None

        self._build_ui()
        self._build_plot()

        self.poll_timer = QTimer()
        self.poll_timer.setInterval(100)
        self.poll_timer.timeout.connect(self._poll)
        self._plot_t0 = None

    # ── UI build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)

        # Connection bar
        conn_row = QHBoxLayout()
        self.btn_connect    = QPushButton("Connect (USB)")
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setEnabled(False)
        conn_row.addWidget(self.btn_connect)
        conn_row.addWidget(self.btn_disconnect)
        conn_row.addStretch()
        root.addLayout(conn_row)

        # Status bar
        status_row = QHBoxLayout()
        self.lbl_status = QLabel("Not connected")
        self.lbl_status.setStyleSheet("color: #aaa")
        self.lbl_vbus = QLabel("Vbus: —")
        self.lbl_vbus.setStyleSheet("color: #64ffb4; font-weight: bold")
        status_row.addWidget(self.lbl_status)
        status_row.addStretch()
        status_row.addWidget(self.lbl_vbus)
        root.addLayout(status_row)

        line = QFrame(); line.setFrameShape(QFrame.HLine)
        root.addWidget(line)

        # Tabs
        tabs = QTabWidget()
        root.addWidget(tabs)

        # Config tab
        cfg_tab = QWidget()
        cfg_layout = QVBoxLayout(cfg_tab)
        self.axis_widgets = [AxisWidget(0), AxisWidget(1)]
        for aw in self.axis_widgets:
            cfg_layout.addWidget(aw)
            aw.btn_apply.clicked.connect(lambda _, a=aw: self._apply_axis(a))
            aw.btn_cal.clicked.connect(lambda _, a=aw: self._calibrate(a))
            aw.btn_idle.clicked.connect(lambda _, a=aw: self._idle(a))

        line2 = QFrame(); line2.setFrameShape(QFrame.HLine)
        cfg_layout.addWidget(line2)
        btn_row2 = QHBoxLayout()
        self.btn_save   = QPushButton("Save to Flash")
        self.btn_reboot = QPushButton("Reboot ODrive")
        self.btn_erase  = QPushButton("Erase Config")
        self.btn_erase.setStyleSheet("color: #ff6464")
        for b in (self.btn_save, self.btn_reboot, self.btn_erase):
            b.setEnabled(False)
            btn_row2.addWidget(b)
        cfg_layout.addLayout(btn_row2)
        cfg_layout.addStretch()
        tabs.addTab(cfg_tab, "Config")

        # Plot tab (populated in _build_plot)
        self.plot_tab = QWidget()
        tabs.addTab(self.plot_tab, "Live Plot")

        # Wire signals
        self.btn_connect.clicked.connect(self._connect)
        self.btn_disconnect.clicked.connect(self._disconnect)
        self.btn_save.clicked.connect(self._save)
        self.btn_reboot.clicked.connect(self._reboot)
        self.btn_erase.clicked.connect(self._erase)

    def _build_plot(self):
        layout = QVBoxLayout(self.plot_tab)

        # Checkbox row
        chk_row = QHBoxLayout()
        self.sig_checks = {}
        for name, color in SIG_COLORS.items():
            chk = QCheckBox(name)
            chk.setChecked(True)
            chk.setStyleSheet(f"color: {color}")
            self.sig_checks[name] = chk
            chk_row.addWidget(chk)
        chk_row.addStretch()
        layout.addLayout(chk_row)

        # Chart
        self.chart = QChart()
        self.chart.setBackgroundBrush(QColor("#1e1e1e"))
        self.chart.setPlotAreaBackgroundBrush(QColor("#252526"))
        self.chart.setPlotAreaBackgroundVisible(True)
        self.chart.legend().setVisible(False)
        self.chart.setMargins(__import__('PySide6.QtCore', fromlist=['QMargins']).QMargins(4, 4, 4, 4))

        self.axis_x = QValueAxis(); self.axis_x.setTitleText("Time (s)")
        self.axis_x.setLabelFormat("%.1f")
        self.axis_x.setGridLineColor(QColor("#404040"))
        self.axis_x.setLabelsColor(QColor("#ccc"))
        self.axis_x.setTitleBrush(QColor("#ccc"))

        self.axis_y = QValueAxis(); self.axis_y.setTitleText("Value")
        self.axis_y.setLabelFormat("%.2f")
        self.axis_y.setGridLineColor(QColor("#404040"))
        self.axis_y.setLabelsColor(QColor("#ccc"))
        self.axis_y.setTitleBrush(QColor("#ccc"))

        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)

        self.series_map = {}
        self.plot_bufs = {name: deque(maxlen=PLOT_LEN) for name in SIG_COLORS}
        self.plot_t    = deque(maxlen=PLOT_LEN)

        for name, color in SIG_COLORS.items():
            s = QLineSeries()
            s.setName(name)
            pen = s.pen(); pen.setColor(QColor(color)); pen.setWidth(2); s.setPen(pen)
            self.chart.addSeries(s)
            s.attachAxis(self.axis_x)
            s.attachAxis(self.axis_y)
            self.series_map[name] = s

        view = QChartView(self.chart)
        view.setRenderHint(QPainter.Antialiasing)
        view.setMinimumHeight(380)
        layout.addWidget(view)

        btn_clear = QPushButton("Clear Plot")
        btn_clear.clicked.connect(self._clear_plot)
        layout.addWidget(btn_clear)

    # ── Connect / Disconnect ──────────────────────────────────────────────────
    def _connect(self):
        self.btn_connect.setEnabled(False)
        self._set_status("Searching for ODrive (USB)…", "#ffff64")
        self.connect_worker = ConnectWorker()
        self.connect_worker.success.connect(self._on_connected)
        self.connect_worker.failed.connect(self._on_connect_failed)
        self.connect_worker.start()

    def _on_connected(self, odrv):
        self.odrv = odrv
        self._set_status(f"Connected — serial: {hex(odrv.serial_number)}", "#64ff64")
        self.btn_disconnect.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_reboot.setEnabled(True)
        self.btn_erase.setEnabled(True)
        for aw in self.axis_widgets:
            aw.set_connected(True)
            aw.load(self.odrv.axis0 if aw.ax_idx == 0 else self.odrv.axis1)
        self._plot_t0 = time.time()
        self.poll_timer.start()

    def _on_connect_failed(self, msg):
        self._set_status(f"Not found: {msg}", "#ff6464")
        self.btn_connect.setEnabled(True)

    def _disconnect(self):
        self.poll_timer.stop()
        self.odrv = None
        self._set_status("Disconnected", "#aaa")
        self.lbl_vbus.setText("Vbus: —")
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_reboot.setEnabled(False)
        self.btn_erase.setEnabled(False)
        for aw in self.axis_widgets:
            aw.set_connected(False)
            aw.lbl_state.setText("State: —")
            aw.lbl_temp.setText("Temp: —")

    # ── Poll ──────────────────────────────────────────────────────────────────
    def _poll(self):
        if not self.odrv:
            return
        try:
            vbus = self.odrv.vbus_voltage
            self.lbl_vbus.setText(f"Vbus: {vbus:.2f} V")

            for aw in self.axis_widgets:
                axis = self.odrv.axis0 if aw.ax_idx == 0 else self.odrv.axis1
                aw.update_live(axis)

            # Plot
            t = time.time() - self._plot_t0
            self.plot_t.append(t)
            sigs = {
                "Vbus (V)":       vbus,
                "Ax0 Iq (A)":    self.odrv.axis0.motor.current_control.Iq_measured,
                "Ax1 Iq (A)":    self.odrv.axis1.motor.current_control.Iq_measured,
                "Ax0 Vel (t/s)": self.odrv.axis0.encoder.vel_estimate,
                "Ax1 Vel (t/s)": self.odrv.axis1.encoder.vel_estimate,
                "Ax0 Pos (t)":   self.odrv.axis0.encoder.pos_estimate,
                "Ax1 Pos (t)":   self.odrv.axis1.encoder.pos_estimate,
            }
            for name, val in sigs.items():
                try:
                    self.plot_bufs[name].append(float(val))
                except Exception:
                    self.plot_bufs[name].append(float("nan"))

            t_list = list(self.plot_t)
            all_vals = []
            for name, series in self.series_map.items():
                if self.sig_checks[name].isChecked():
                    pts = list(self.plot_bufs[name])
                    series.replace([
                        __import__('PySide6.QtCore', fromlist=['QPointF']).QPointF(t_list[i], pts[i])
                        for i in range(min(len(t_list), len(pts)))
                        if pts[i] == pts[i]  # skip nan
                    ])
                    all_vals.extend(v for v in pts if v == v)
                else:
                    series.clear()

            if t_list:
                self.axis_x.setRange(t_list[0], t_list[-1] + 0.5)
            if all_vals:
                lo, hi = min(all_vals), max(all_vals)
                pad = (hi - lo) * 0.1 or 0.5
                self.axis_y.setRange(lo - pad, hi + pad)

        except Exception:
            pass

    def _clear_plot(self):
        self.plot_t.clear()
        for b in self.plot_bufs.values():
            b.clear()
        for s in self.series_map.values():
            s.clear()

    # ── Actions ───────────────────────────────────────────────────────────────
    def _apply_axis(self, aw):
        if not self.odrv:
            return
        axis = self.odrv.axis0 if aw.ax_idx == 0 else self.odrv.axis1
        try:
            aw.apply(axis)
            self._set_status(f"Axis {aw.ax_idx} applied (not saved)", "#64c8ff")
        except Exception as e:
            self._set_status(f"Apply error: {e}", "#ff6464")

    def _calibrate(self, aw):
        if not self.odrv:
            return
        try:
            import odrive.enums as enums
            axis = self.odrv.axis0 if aw.ax_idx == 0 else self.odrv.axis1
            axis.requested_state = enums.AXIS_STATE_FULL_CALIBRATION_SEQUENCE
            self._set_status(f"Axis {aw.ax_idx}: calibration started…", "#ffc850")
        except Exception as e:
            self._set_status(f"Calibration error: {e}", "#ff6464")

    def _idle(self, aw):
        if not self.odrv:
            return
        try:
            import odrive.enums as enums
            axis = self.odrv.axis0 if aw.ax_idx == 0 else self.odrv.axis1
            axis.requested_state = enums.AXIS_STATE_IDLE
            self._set_status(f"Axis {aw.ax_idx}: Idle", "#aaa")
        except Exception as e:
            self._set_status(f"Idle error: {e}", "#ff6464")

    def _save(self):
        try:
            self.odrv.save_configuration()
            self._set_status("Saved to flash.", "#64ff64")
        except Exception as e:
            self._set_status(f"Save error: {e}", "#ff6464")

    def _reboot(self):
        try:
            self.poll_timer.stop()
            self.odrv.reboot()
        except Exception:
            pass
        self._disconnect()
        self._set_status("Rebooting… reconnect when ready.", "#ffc850")

    def _erase(self):
        try:
            self.odrv.erase_configuration()
            self._set_status("Config erased.", "#ff6464")
        except Exception as e:
            self._set_status(f"Erase error: {e}", "#ff6464")

    def _set_status(self, msg, color="#ffff64"):
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet(f"color: {color}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    from PySide6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor("#2d2d2d"))
    palette.setColor(QPalette.WindowText,      QColor("#ddd"))
    palette.setColor(QPalette.Base,            QColor("#1e1e1e"))
    palette.setColor(QPalette.AlternateBase,   QColor("#252526"))
    palette.setColor(QPalette.Text,            QColor("#ddd"))
    palette.setColor(QPalette.Button,          QColor("#3c3c3c"))
    palette.setColor(QPalette.ButtonText,      QColor("#ddd"))
    palette.setColor(QPalette.Highlight,       QColor("#0078d4"))
    palette.setColor(QPalette.HighlightedText, QColor("#fff"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())
