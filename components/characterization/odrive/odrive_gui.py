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
    QFrame, QTextEdit, QRadioButton, QButtonGroup, QSizePolicy, QSplitter,
    QScrollArea,
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


# ── Motor control tab ─────────────────────────────────────────────────────────
CONTROL_MODES = {
    "Position": 3,
    "Velocity": 2,
    "Torque":   1,
}
CONTROL_UNITS = {"Position": "turns", "Velocity": "t/s", "Torque": "Nm"}


class MotorControlTab(QWidget):
    """Full-featured motor controller tab (axis selector + commands + PID)."""

    log_message = Signal(str, str)   # (text, color)  — forwarded to terminal log

    def __init__(self):
        super().__init__()
        self.odrv = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── Axis selector ────────────────────────────────────────────────────
        ax_row = QHBoxLayout()
        ax_row.addWidget(QLabel("Axis:"))
        self._ax_grp = QButtonGroup(self)
        self._rb_ax = []
        for i in range(2):
            rb = QRadioButton(str(i))
            rb.setChecked(i == 0)
            self._ax_grp.addButton(rb, i)
            ax_row.addWidget(rb)
            self._rb_ax.append(rb)
        ax_row.addStretch()
        root.addLayout(ax_row)

        # ── Control ──────────────────────────────────────────────────────────
        ctrl_box = QGroupBox("Control")
        ctrl_form = QFormLayout(ctrl_box)
        ctrl_form.setSpacing(6)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(CONTROL_MODES.keys()))
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        ctrl_form.addRow("Mode", self.mode_combo)

        sp_row = QHBoxLayout()
        self.setpoint_spin = QDoubleSpinBox()
        self.setpoint_spin.setRange(-1000, 1000)
        self.setpoint_spin.setDecimals(4)
        self.setpoint_spin.setSingleStep(0.1)
        self.setpoint_spin.setFixedWidth(110)
        self.lbl_units = QLabel("turns")
        sp_row.addWidget(self.setpoint_spin)
        sp_row.addWidget(self.lbl_units)
        sp_row.addStretch()
        ctrl_form.addRow("Setpoint", sp_row)

        cmd_row = QHBoxLayout()
        self.btn_send       = QPushButton("Send")
        self.btn_cl         = QPushButton("Enter Closed Loop")
        self.btn_stop       = QPushButton("Stop Motion (Idle)")
        self.btn_stop.setStyleSheet("color: #ff9a3c")
        for b in (self.btn_send, self.btn_cl, self.btn_stop):
            b.setEnabled(False)
            cmd_row.addWidget(b)
        ctrl_form.addRow("", cmd_row)
        root.addWidget(ctrl_box)

        # ── PID Gains ────────────────────────────────────────────────────────
        pid_box = QGroupBox("PID Gains")
        pid_form = QFormLayout(pid_box)
        pid_form.setSpacing(6)

        def _dspin(lo, hi, dec=4, step=0.01):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setFixedWidth(110)
            return s

        self.spin_pos_gain    = _dspin(0, 500,  dec=3, step=1)
        self.spin_vel_gain    = _dspin(0, 10,   dec=6, step=0.001)
        self.spin_vel_int     = _dspin(0, 100,  dec=6, step=0.01)
        self.spin_vel_lim     = _dspin(0, 500,  dec=3, step=1)
        self.spin_cur_lim     = _dspin(0, 60,   dec=2, step=0.5)
        self.spin_cur_int     = _dspin(0, 1000, dec=3, step=1)

        pid_form.addRow("pos_gain",              self.spin_pos_gain)
        pid_form.addRow("vel_gain",              self.spin_vel_gain)
        pid_form.addRow("vel_integrator_gain",   self.spin_vel_int)
        pid_form.addRow("vel_limit (t/s)",       self.spin_vel_lim)
        pid_form.addRow("current_lim (A)",       self.spin_cur_lim)
        pid_form.addRow("current_ctrl_bandwidth",self.spin_cur_int)

        pid_btns = QHBoxLayout()
        self.btn_read_pid  = QPushButton("Read from ODrive")
        self.btn_apply_pid = QPushButton("Apply Gains")
        self.btn_apply_pid.setStyleSheet("color: #64ffb4")
        for b in (self.btn_read_pid, self.btn_apply_pid):
            b.setEnabled(False)
            pid_btns.addWidget(b)
        pid_form.addRow("", pid_btns)
        root.addWidget(pid_box)

        # ── Error / utility ──────────────────────────────────────────────────
        err_box = QGroupBox("Error / Utility")
        err_row = QHBoxLayout(err_box)
        self.btn_reset_err  = QPushButton("Reset Errors")
        self.btn_reset_err.setStyleSheet("color: #ff6464")
        self.btn_reset_err.setEnabled(False)
        self.btn_reset_both = QPushButton("Reset Both Axes")
        self.btn_reset_both.setStyleSheet("color: #ffa03c")
        self.btn_reset_both.setEnabled(False)
        for b in (self.btn_reset_err, self.btn_reset_both):
            err_row.addWidget(b)
        err_row.addStretch()
        root.addWidget(err_box)

        # ── Live readout ─────────────────────────────────────────────────────
        live_box = QGroupBox("Live Readout")
        live_form = QFormLayout(live_box)
        self.lbl_live_state    = QLabel("—")
        self.lbl_live_pos      = QLabel("—")
        self.lbl_live_vel      = QLabel("—")
        self.lbl_live_iq       = QLabel("—")
        self.lbl_live_setpoint = QLabel("—")
        self.lbl_live_precal_m  = QLabel("—")
        self.lbl_live_precal_e  = QLabel("—")
        self.lbl_live_enc_rdy   = QLabel("—")
        self.lbl_live_input_mode= QLabel("—")
        self.lbl_live_vel_lim   = QLabel("—")
        self.lbl_live_cur_lim   = QLabel("—")
        self.lbl_live_err       = QLabel("—")
        for name, lbl in [("State", self.lbl_live_state), ("Position (t)", self.lbl_live_pos),
                           ("Velocity (t/s)", self.lbl_live_vel), ("Iq (A)", self.lbl_live_iq),
                           ("Setpoint (ODrive)", self.lbl_live_setpoint),
                           ("Motor pre-cal", self.lbl_live_precal_m),
                           ("Encoder pre-cal", self.lbl_live_precal_e),
                           ("Encoder ready", self.lbl_live_enc_rdy),
                           ("input_mode", self.lbl_live_input_mode),
                           ("vel_limit (t/s)", self.lbl_live_vel_lim),
                           ("current_lim (A)", self.lbl_live_cur_lim),
                           ("Error", self.lbl_live_err)]:
            lbl.setStyleSheet("font-family: monospace; font-size: 12px; color: #64ffb4")
            live_form.addRow(name, lbl)
        root.addWidget(live_box)

        root.addStretch()

        # ── Wire signals ─────────────────────────────────────────────────────
        self.btn_send.clicked.connect(self._send_setpoint)
        self.btn_cl.clicked.connect(self._enter_closed_loop)
        self.btn_stop.clicked.connect(self._stop_motion)
        self.btn_read_pid.clicked.connect(self._read_pid)
        self.btn_apply_pid.clicked.connect(self._apply_pid)
        self.btn_reset_err.clicked.connect(self._reset_errors)
        self.btn_reset_both.clicked.connect(self._reset_both)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _axis(self):
        if not self.odrv:
            return None
        return self.odrv.axis0 if self._ax_grp.checkedId() == 0 else self.odrv.axis1

    def _ax_idx(self):
        return self._ax_grp.checkedId()

    def _log(self, msg, color="#64ffb4"):
        self.log_message.emit(msg, color)

    def _on_mode_changed(self, mode_name):
        self.lbl_units.setText(CONTROL_UNITS.get(mode_name, ""))

    def set_connected(self, on):
        for b in (self.btn_send, self.btn_cl, self.btn_stop,
                  self.btn_read_pid, self.btn_apply_pid,
                  self.btn_reset_err, self.btn_reset_both):
            b.setEnabled(on)
        if on:
            self._read_pid()

    # ── Actions ───────────────────────────────────────────────────────────────
    def _send_setpoint(self):
        axis = self._axis()
        if not axis:
            return
        mode_name = self.mode_combo.currentText()
        mode_val  = CONTROL_MODES[mode_name]
        val       = self.setpoint_spin.value()
        try:
            import odrive.enums as enums
            axis.controller.config.control_mode = mode_val
            axis.controller.config.input_mode   = 1  # INPUT_MODE_PASSTHROUGH
            if mode_name == "Position":
                axis.controller.input_pos = val
            elif mode_name == "Velocity":
                axis.controller.input_vel = val
            else:
                axis.controller.input_torque = val
            self._log(f"Ax{self._ax_idx()} → {mode_name} setpoint = {val:.4f}")
        except Exception as e:
            self._log(f"Send error: {e}", "#ff6464")

    def _enter_closed_loop(self):
        axis = self._axis()
        if not axis:
            return
        try:
            import odrive.enums as enums
            mode_name = self.mode_combo.currentText()
            mode_val  = CONTROL_MODES[mode_name]
            axis.controller.config.control_mode = mode_val
            axis.controller.config.input_mode   = 1  # PASSTHROUGH — must be set before CL

            # Zero setpoints and flush integrator so re-arm is always clean
            axis.controller.input_pos     = axis.encoder.pos_estimate  # hold current pos
            axis.controller.input_vel     = 0.0
            axis.controller.input_torque  = 0.0
            try:
                axis.controller.vel_integrator_current = 0.0  # flush integrator (fw 0.5.x)
            except Exception:
                pass

            # Clear any lingering errors before entering CL
            axis.error             = 0
            axis.motor.error       = 0
            axis.encoder.error     = 0
            axis.controller.error  = 0

            axis.requested_state = enums.AXIS_STATE_CLOSED_LOOP_CONTROL
            self._log(f"Ax{self._ax_idx()} → Closed Loop ({mode_name}) — setpoints zeroed, integrator flushed", "#64ffb4")
        except Exception as e:
            self._log(f"CL error: {e}", "#ff6464")

    def _stop_motion(self):
        axis = self._axis()
        if not axis:
            return
        try:
            import odrive.enums as enums
            axis.requested_state = enums.AXIS_STATE_IDLE
            self._log(f"Ax{self._ax_idx()} → Idle (stopped)", "#ffa03c")
        except Exception as e:
            self._log(f"Stop error: {e}", "#ff6464")

    def _read_pid(self):
        axis = self._axis()
        if not axis:
            return
        try:
            cc = axis.controller.config
            self.spin_pos_gain.setValue(float(getattr(cc, "pos_gain", 20)))
            self.spin_vel_gain.setValue(float(getattr(cc, "vel_gain", 0.16)))
            self.spin_vel_int.setValue(float(getattr(cc, "vel_integrator_gain", 0.32)))
            self.spin_vel_lim.setValue(float(getattr(cc, "vel_limit", 2)))
            self.spin_cur_lim.setValue(float(getattr(axis.motor.config, "current_lim", 10)))
            self.spin_cur_int.setValue(float(getattr(axis.motor.config, "current_control_bandwidth", 1000)))
            self._log(f"Ax{self._ax_idx()} PID read OK")
        except Exception as e:
            self._log(f"Read PID error: {e}", "#ff6464")

    def _apply_pid(self):
        axis = self._axis()
        if not axis:
            return
        try:
            cc = axis.controller.config
            cc.pos_gain              = self.spin_pos_gain.value()
            cc.vel_gain              = self.spin_vel_gain.value()
            cc.vel_integrator_gain   = self.spin_vel_int.value()
            cc.vel_limit             = self.spin_vel_lim.value()
            axis.motor.config.current_lim                  = self.spin_cur_lim.value()
            axis.motor.config.current_control_bandwidth    = self.spin_cur_int.value()
            self._log(f"Ax{self._ax_idx()} PID applied (not saved)", "#64ffb4")
        except Exception as e:
            self._log(f"Apply PID error: {e}", "#ff6464")

    def _reset_errors(self):
        axis = self._axis()
        if not axis:
            return
        ax_i = self._ax_idx()
        self._log(f"── Reset Ax{ax_i} errors ──")
        for label, obj, field in [
            (f"axis{ax_i}.error",            axis,            "error"),
            (f"axis{ax_i}.motor.error",      axis.motor,      "error"),
            (f"axis{ax_i}.encoder.error",    axis.encoder,    "error"),
            (f"axis{ax_i}.controller.error", axis.controller, "error"),
        ]:
            try:
                setattr(obj, field, 0)
                self._log(f"  {label} = 0")
            except Exception as e:
                self._log(f"  {label} ERR: {e}", "#ff6464")

    def _reset_both(self):
        if not self.odrv:
            return
        self._log("── Reset ALL axis errors ──")
        for ax_i, axis in enumerate([self.odrv.axis0, self.odrv.axis1]):
            for obj, field in [(axis, "error"), (axis.motor, "error"),
                               (axis.encoder, "error"), (axis.controller, "error")]:
                try:
                    setattr(obj, field, 0)
                except Exception:
                    pass
        self._log("Done — all errors cleared", "#64ffb4")

    def update_live(self, odrv):
        """Called by the poll timer. odrv may be None."""
        self.odrv = odrv
        axis = self._axis()
        if not axis:
            for lbl in (self.lbl_live_state, self.lbl_live_pos,
                        self.lbl_live_vel, self.lbl_live_iq,
                        self.lbl_live_setpoint, self.lbl_live_precal_m,
                        self.lbl_live_precal_e, self.lbl_live_enc_rdy,
                        self.lbl_live_input_mode, self.lbl_live_vel_lim,
                        self.lbl_live_cur_lim, self.lbl_live_err):
                lbl.setText("—")
            return
        try:
            state_id = getattr(axis, "current_state", 0)
            self.lbl_live_state.setText(AXIS_STATES.get(state_id, str(state_id)))
            self.lbl_live_pos.setText(f"{axis.encoder.pos_estimate:.4f}")
            self.lbl_live_vel.setText(f"{axis.encoder.vel_estimate:.4f}")
            self.lbl_live_iq.setText(f"{axis.motor.current_control.Iq_measured:.4f}")
            # Setpoint readback — show whichever input matches the active control mode
            _ctrl_mode_map = {3: ("input_pos",     "t"),
                              2: ("input_vel",     "t/s"),
                              1: ("input_torque",  "Nm")}
            try:
                cm = axis.controller.config.control_mode
                attr, unit = _ctrl_mode_map.get(cm, ("input_vel", "t/s"))
                sp_val = getattr(axis.controller, attr, 0.0)
                self.lbl_live_setpoint.setText(f"{float(sp_val):.4f} {unit}")
            except Exception:
                self.lbl_live_setpoint.setText("—")
            # Pre-calibration / encoder readiness
            def _bool_lbl(lbl, val):
                ok = bool(val)
                lbl.setText("YES" if ok else "NO")
                lbl.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {'#64ffb4' if ok else '#ff6464'}")
            _bool_lbl(self.lbl_live_precal_m, getattr(axis.motor.config,   "pre_calibrated", False))
            _bool_lbl(self.lbl_live_precal_e, getattr(axis.encoder.config, "pre_calibrated", False))
            _bool_lbl(self.lbl_live_enc_rdy,  getattr(axis.encoder,        "is_ready",       False))

            # Volatile settings that silently block motion
            INPUT_MODE_NAMES = {0: "0-INACTIVE ⚠", 1: "1-PASSTHROUGH", 2: "2-VEL_RAMP",
                                 3: "3-POS_FILTER", 5: "5-TRAP_TRAJ"}
            im = getattr(axis.controller.config, "input_mode", 0)
            im_txt = INPUT_MODE_NAMES.get(int(im), str(im))
            self.lbl_live_input_mode.setText(im_txt)
            self.lbl_live_input_mode.setStyleSheet(
                f"font-family: monospace; font-size: 12px; color: {'#ff6464' if int(im) == 0 else '#64ffb4'}")

            vl = getattr(axis.controller.config, "vel_limit", 0)
            self.lbl_live_vel_lim.setText(f"{float(vl):.2f}")
            self.lbl_live_vel_lim.setStyleSheet(
                f"font-family: monospace; font-size: 12px; color: {'#ff6464' if float(vl) == 0 else '#64ffb4'}")

            cl = getattr(axis.motor.config, "current_lim", 0)
            self.lbl_live_cur_lim.setText(f"{float(cl):.2f}")
            self.lbl_live_cur_lim.setStyleSheet(
                f"font-family: monospace; font-size: 12px; color: {'#ff6464' if float(cl) == 0 else '#64ffb4'}")

            err = getattr(axis, "error", 0)
            if err:
                self.lbl_live_err.setText(f"0x{err:08X}")
                self.lbl_live_err.setStyleSheet("font-family: monospace; font-size: 12px; color: #ff6464")
            else:
                self.lbl_live_err.setText("None")
                self.lbl_live_err.setStyleSheet("font-family: monospace; font-size: 12px; color: #64ffb4")
        except Exception:
            pass


# ── Main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ODrive 3.6 Config")
        self.resize(1100, 860)
        self.odrv = None
        self.connect_worker = None

        self._build_ui()
        self._build_plot()

        self.poll_timer = QTimer()
        self.poll_timer.setInterval(100)
        self.poll_timer.timeout.connect(self._poll)
        self._plot_t0 = None
        self._was_calibrating = [False, False]

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

        # Combined Motor Control + Live Plot tab
        combined = QWidget()
        combined_layout = QHBoxLayout(combined)
        combined_layout.setContentsMargins(2, 2, 2, 2)
        splitter = QSplitter(Qt.Horizontal)

        # Left: motor control (in a scroll area so it never clips)
        self.motor_ctrl_tab = MotorControlTab()
        self.motor_ctrl_tab.log_message.connect(self._append_log)
        scroll = QScrollArea()
        scroll.setWidget(self.motor_ctrl_tab)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(310)
        scroll.setMaximumWidth(420)
        splitter.addWidget(scroll)

        # Right: live plot (populated in _build_plot)
        self.plot_tab = QWidget()
        splitter.addWidget(self.plot_tab)
        splitter.setSizes([340, 700])

        combined_layout.addWidget(splitter)
        tabs.addTab(combined, "Motor Control")

        # Terminal tab
        term_tab = QWidget()
        term_layout = QVBoxLayout(term_tab)
        btn_cfg_enc = QPushButton("Configure Encoder on M0 (AS5048A SPI)")
        btn_cfg_enc.clicked.connect(self._configure_encoder_m0)
        term_layout.addWidget(btn_cfg_enc)
        btn_diag = QPushButton("Diagnose M0 Encoder")
        btn_diag.clicked.connect(self._diagnose_encoder_m0)
        term_layout.addWidget(btn_diag)
        btn_reset_errors = QPushButton("Reset M0 Errors")
        btn_reset_errors.clicked.connect(self._reset_errors_m0)
        term_layout.addWidget(btn_reset_errors)
        btn_version = QPushButton("Read Firmware Version")
        btn_version.clicked.connect(self._read_version)
        term_layout.addWidget(btn_version)
        btn_full_dump = QPushButton("Full State Dump (both axes)")
        btn_full_dump.setStyleSheet("color: #64c8ff; font-weight: bold")
        btn_full_dump.clicked.connect(self._full_state_dump)
        term_layout.addWidget(btn_full_dump)
        self.term_log = QTextEdit()
        self.term_log.setReadOnly(True)
        self.term_log.setStyleSheet("font-family: monospace; font-size: 11px;")
        term_layout.addWidget(self.term_log)
        tabs.addTab(term_tab, "Terminal")

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
        self.sig_labels = {}
        for name, color in SIG_COLORS.items():
            chk = QCheckBox(name)
            chk.setChecked(True)
            chk.setStyleSheet(f"color: {color}")
            self.sig_checks[name] = chk
            chk_row.addWidget(chk)
            lbl = QLabel("—")
            lbl.setStyleSheet(f"color: {color}; font-family: monospace; font-size: 11px;")
            lbl.setFixedWidth(58)
            self.sig_labels[name] = lbl
            chk_row.addWidget(lbl)
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
        self.motor_ctrl_tab.odrv = self.odrv
        self.motor_ctrl_tab.set_connected(True)
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
        self.motor_ctrl_tab.set_connected(False)
        self.motor_ctrl_tab.update_live(None)

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
                state = getattr(axis, "current_state", 0)
                calibrating = state in (3, 4, 7)
                if self._was_calibrating[aw.ax_idx] and not calibrating:
                    err = getattr(axis, "error", 0)
                    if err:
                        self._set_status(f"Axis {aw.ax_idx} calibration FAILED — error {hex(err)}", "#ff6464")
                    else:
                        self._set_status(f"Axis {aw.ax_idx} calibration complete.", "#64ff64")
                self._was_calibrating[aw.ax_idx] = calibrating

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
                    fval = float(val)
                    self.plot_bufs[name].append(fval)
                    self.sig_labels[name].setText(f"{fval:.3f}")
                except Exception:
                    self.plot_bufs[name].append(float("nan"))
                    self.sig_labels[name].setText("—")

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

            self.motor_ctrl_tab.update_live(self.odrv)

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

    def _diagnose_encoder_m0(self):
        if not self.odrv:
            self.term_log.append("ERROR: Not connected.")
            return
        self.term_log.append("── M0 Encoder Diagnostic ──────────────────")
        enc = self.odrv.axis0.encoder
        checks = [
            ("encoder.config.mode",              lambda: enc.config.mode),
            ("encoder.config.abs_spi_cs_gpio_pin", lambda: enc.config.abs_spi_cs_gpio_pin),
            ("encoder.config.cpr",               lambda: enc.config.cpr),
            ("encoder.is_ready",                 lambda: enc.is_ready),
            ("encoder.error",                    lambda: hex(enc.error)),
            ("axis0.error",                      lambda: hex(self.odrv.axis0.error)),
            ("encoder.shadow_count",             lambda: enc.shadow_count),
            ("encoder.pos_estimate",             lambda: enc.pos_estimate),
        ]
        for label, fn in checks:
            try:
                val = fn()
                color = "#64ff64" if label not in ("encoder.error", "axis0.error") or val in ("0x0", "0x00") else "#ff6464"
                self.term_log.append(f"<span style='color:{color}'>{label} = {val}</span>")
            except Exception as e:
                self.term_log.append(f"<span style='color:#ff6464'>{label} → ERR: {e}</span>")
        self.term_log.append("───────────────────────────────────────────")

    def _configure_encoder_m0(self):
        if not self.odrv:
            self.term_log.append("ERROR: Not connected.")
            return
        cmds = [
            ("axis0.encoder.config.mode = ENCODER_MODE_SPI_ABS_AMS (0x101 = 257)",
             lambda: setattr(self.odrv.axis0.encoder.config, "mode", 257)),
            ("axis0.encoder.config.abs_spi_cs_gpio_pin = 3",
             lambda: setattr(self.odrv.axis0.encoder.config, "abs_spi_cs_gpio_pin", 3)),
            ("axis0.encoder.config.cpr = 16384",
             lambda: setattr(self.odrv.axis0.encoder.config, "cpr", 16384)),
            ("save_configuration()",
             lambda: self.odrv.save_configuration()),
        ]
        for label, fn in cmds:
            try:
                fn()
                self.term_log.append(f"<span style='color:#64ff64'>OK</span>  {label}")
            except Exception as e:
                self.term_log.append(f"<span style='color:#ff6464'>ERR</span> {label} → {e}")
                return
        self.term_log.append("<span style='color:#ffc850'>ODrive will reboot. Reconnect when ready.</span>")
        self._disconnect()

    def _read_version(self):
        if not self.odrv:
            self.term_log.append("ERROR: Not connected.")
            return
        try:
            hw = self.odrv.hw_version_major, self.odrv.hw_version_minor, self.odrv.hw_version_variant
            fw = self.odrv.fw_version_major, self.odrv.fw_version_minor, self.odrv.fw_version_revision
            self.term_log.append(f"Hardware: v{hw[0]}.{hw[1]} variant {hw[2]}")
            self.term_log.append(f"Firmware: v{fw[0]}.{fw[1]}.{fw[2]}")
        except Exception as e:
            self.term_log.append(f"<span style='color:#ff6464'>ERR reading version: {e}</span>")

    def _reset_errors_m0(self):
        if not self.odrv:
            self.term_log.append("ERROR: Not connected.")
            return
        self.term_log.append("── Reset M0 Errors ─────────────────────────")
        targets = [
            ("axis0.error",         lambda: setattr(self.odrv.axis0,         "error", 0)),
            ("axis0.motor.error",   lambda: setattr(self.odrv.axis0.motor,   "error", 0)),
            ("axis0.encoder.error", lambda: setattr(self.odrv.axis0.encoder, "error", 0)),
            ("axis0.controller.error", lambda: setattr(self.odrv.axis0.controller, "error", 0)),
        ]
        for label, fn in targets:
            try:
                fn()
                self.term_log.append(f"<span style='color:#64ff64'>OK</span>  {label} = 0")
            except Exception as e:
                self.term_log.append(f"<span style='color:#ff6464'>ERR</span> {label} → {e}")
        self.term_log.append("Run Diagnose to confirm errors cleared.")
        self.term_log.append("───────────────────────────────────────────")

    def _full_state_dump(self):
        if not self.odrv:
            self.term_log.append("ERROR: Not connected.")
            return
        self.term_log.append("═══ FULL STATE DUMP ════════════════════════")
        for ax_i, axis in enumerate([self.odrv.axis0, self.odrv.axis1]):
            self.term_log.append(f"<b>── Axis {ax_i} ──────────────────────────────</b>")
            checks = [
                # State & errors
                ("current_state",                   lambda a=axis: AXIS_STATES.get(a.current_state, a.current_state)),
                ("error",                           lambda a=axis: hex(a.error)),
                ("motor.error",                     lambda a=axis: hex(a.motor.error)),
                ("encoder.error",                   lambda a=axis: hex(a.encoder.error)),
                ("controller.error",                lambda a=axis: hex(a.controller.error)),
                # Calibration
                ("motor.is_calibrated",             lambda a=axis: a.motor.is_calibrated),
                ("motor.config.pre_calibrated",     lambda a=axis: a.motor.config.pre_calibrated),
                ("encoder.is_ready",                lambda a=axis: a.encoder.is_ready),
                ("encoder.config.pre_calibrated",   lambda a=axis: a.encoder.config.pre_calibrated),
                # Motor config
                ("motor.config.motor_type",         lambda a=axis: a.motor.config.motor_type),
                ("motor.config.pole_pairs",         lambda a=axis: a.motor.config.pole_pairs),
                ("motor.config.phase_resistance",   lambda a=axis: f"{a.motor.config.phase_resistance:.4f} Ω"),
                ("motor.config.phase_inductance",   lambda a=axis: f"{a.motor.config.phase_inductance:.6f} H"),
                ("motor.config.torque_constant",    lambda a=axis: f"{a.motor.config.torque_constant:.4f}"),
                ("motor.config.current_lim",        lambda a=axis: f"{a.motor.config.current_lim:.2f} A"),
                # Controller config
                ("controller.config.control_mode",  lambda a=axis: a.controller.config.control_mode),
                ("controller.config.input_mode",    lambda a=axis: a.controller.config.input_mode),
                ("controller.config.vel_limit",     lambda a=axis: f"{a.controller.config.vel_limit:.2f} t/s"),
                ("controller.config.pos_gain",      lambda a=axis: a.controller.config.pos_gain),
                ("controller.config.vel_gain",      lambda a=axis: a.controller.config.vel_gain),
                ("controller.config.vel_integrator_gain", lambda a=axis: a.controller.config.vel_integrator_gain),
                # Live setpoints
                ("controller.input_vel",            lambda a=axis: f"{a.controller.input_vel:.4f}"),
                ("controller.input_pos",            lambda a=axis: f"{a.controller.input_pos:.4f}"),
                # Live measurements
                ("encoder.vel_estimate",            lambda a=axis: f"{a.encoder.vel_estimate:.4f}"),
                ("encoder.pos_estimate",            lambda a=axis: f"{a.encoder.pos_estimate:.4f}"),
                ("motor.current_control.Iq_measured", lambda a=axis: f"{a.motor.current_control.Iq_measured:.4f} A"),
                ("motor.current_control.Iq_setpoint", lambda a=axis: f"{a.motor.current_control.Iq_setpoint:.4f} A"),
            ]
            for label, fn in checks:
                try:
                    val = fn()
                    # Flag likely-bad values in red
                    bad = (("error" in label and str(val) not in ("0x0", "0", "False")) or
                           (label == "motor.config.torque_constant" and float(str(val)) == 0) or
                           (label == "motor.config.phase_resistance" and float(str(val).split()[0]) == 0) or
                           (label == "controller.config.input_mode" and str(val) == "0") or
                           (label == "motor.is_calibrated" and not val) or
                           (label == "encoder.is_ready" and not val))
                    color = "#ff6464" if bad else "#ccc"
                    self.term_log.append(f"  <span style='color:{color}'>{label} = {val}</span>")
                except Exception as e:
                    self.term_log.append(f"  <span style='color:#888'>{label} → n/a ({e})</span>")
        self.term_log.append("════════════════════════════════════════════")

    def _append_log(self, msg, color="#64ffb4"):
        self.term_log.append(f"<span style='color:{color}'>{msg}</span>")

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
