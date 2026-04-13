"""tab_control.py — Motor control + live charts in a split pane.

Left pane:  mode select, setpoint, gains, errors, live readouts (scrollable).
Right pane: pyqtgraph live telemetry charts.
"""

import logging
import math
import time

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QRadioButton, QButtonGroup,
    QDoubleSpinBox, QScrollArea, QSplitter, QCheckBox,
    QDialog, QDialogButtonBox, QTextEdit,
)

from core.constants import AXIS_STATES
from core.odrive_errors import (
    decode_errors, decode_errors_str,
    AXIS_ERRORS, MOTOR_ERRORS, ENCODER_ERRORS, CONTROLLER_ERRORS,
    AXIS_ERROR_HINTS, MOTOR_ERROR_HINTS, ENCODER_ERROR_HINTS, CONTROLLER_ERROR_HINTS,
)
from tabs.tab_charts import TabCharts
from ui.theme import CLR_OK, CLR_WARN, CLR_ERR, CLR_INFO, CLR_CAL, CLR_MUTED, CLR_PANEL, CLR_LABEL

log = logging.getLogger("odrive_gui")


def _dspin(lo, hi, default, dec=2, step=0.5, suffix=""):
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(dec)
    sb.setSingleStep(step)
    sb.setValue(default)
    if suffix:
        sb.setSuffix(f"  {suffix}")
    return sb


def _colored(label: QLabel, text: str, color: str):
    label.setText(text)
    label.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {color};")


class TabControl(QWidget):
    """Motor control + live charts in a horizontal split pane."""

    def __init__(self, manager, get_axis_idx, parent=None):
        super().__init__(parent)
        self._mgr = manager
        self._get_axis_idx = get_axis_idx
        self._last_err = {"axis": 0, "motor": 0, "encoder": 0, "controller": 0}

        # Setpoint tracking (last commanded value — shown as dashed overlay in charts)
        self._cur_sp_vel = 0.0
        self._cur_sp_pos = 0.0

        # Waveform generator state
        self._wave_timer = QTimer(self)
        self._wave_timer.setInterval(20)  # 50 Hz
        self._wave_timer.timeout.connect(self._wave_tick)
        self._wave_type = None   # "square" | "sine" | None
        self._wave_mode = None   # "vel" | "pos"
        self._wave_t0 = 0.0
        self._wave_offset = 0.0  # center for position waveform

        self._build()

    def _ax_idx(self) -> int:
        return self._get_axis_idx()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        # ── Left pane: controls ──────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(250)
        scroll.setMaximumWidth(340)

        inner = QWidget()
        scroll.setWidget(inner)
        left = QVBoxLayout(inner)
        left.setSpacing(8)
        left.setContentsMargins(8, 8, 8, 8)

        left.addWidget(self._build_errors_group())
        left.addWidget(self._build_control_group())
        left.addWidget(self._build_gains_group())
        left.addWidget(self._build_ff_group())
        left.addWidget(self._build_quick_buttons())
        left.addWidget(self._build_readout_group())
        left.addStretch()

        splitter.addWidget(scroll)

        # ── Right pane: live charts ──────────────────────────────────────────
        self._charts = TabCharts(self._mgr, self._get_axis_idx, self._get_setpoints)
        splitter.addWidget(self._charts)

        # Charts get the lion's share of space
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ── Errors group ──────────────────────────────────────────────────────────

    def _build_errors_group(self):
        box = QGroupBox("Errors")
        col = QVBoxLayout(box)
        col.setSpacing(4)

        self._err_labels = {}
        self._err_btns = {}
        for key in ("axis", "motor", "encoder", "controller"):
            row = QHBoxLayout()
            row.setSpacing(4)
            lbl_name = QLabel(f"{key.capitalize()}:")
            lbl_name.setFixedWidth(80)
            lbl_name.setStyleSheet(f"color: {CLR_MUTED};")
            lbl_val = QLabel("0x0000  none")
            lbl_val.setStyleSheet("font-family: monospace; font-size: 12px;")
            lbl_val.setWordWrap(True)
            btn_q = QPushButton("?")
            btn_q.setFixedWidth(22)
            btn_q.setFixedHeight(20)
            btn_q.setStyleSheet(f"color: {CLR_INFO}; font-weight: bold; padding: 0; border-radius: 3px;")
            row.addWidget(lbl_name)
            row.addWidget(lbl_val, stretch=1)
            row.addWidget(btn_q)
            col.addLayout(row)
            self._err_labels[key] = lbl_val
            self._err_btns[key] = btn_q

        # Wire hint popups
        tables = {
            "axis": (AXIS_ERRORS, AXIS_ERROR_HINTS),
            "motor": (MOTOR_ERRORS, MOTOR_ERROR_HINTS),
            "encoder": (ENCODER_ERRORS, ENCODER_ERROR_HINTS),
            "controller": (CONTROLLER_ERRORS, CONTROLLER_ERROR_HINTS),
        }
        for key, (names, hints) in tables.items():
            self._err_btns[key].clicked.connect(
                lambda checked=False, k=key, n=names, h=hints:
                    self._show_error_popup(k.capitalize() + " Errors", self._last_err[k], n, h)
            )

        self.btn_clear_errors = QPushButton("Clear Errors")
        self.btn_clear_errors.setEnabled(False)
        self.btn_clear_errors.clicked.connect(self._clear_errors)
        col.addWidget(self.btn_clear_errors)

        return box

    def _show_error_popup(self, title: str, code: int, names: dict, hints: dict):
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(480, 360)
        layout = QVBoxLayout(dlg)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setStyleSheet(f"background: {CLR_PANEL}; color: {CLR_LABEL}; font-family: monospace;")

        if code == 0:
            txt.setHtml(f'<p style="color:{CLR_OK};">No errors (0x0000)</p>')
        else:
            html = f'<p><b>Raw value:</b> 0x{code:08X}</p><hr/>'
            known_mask = 0
            for mask, name in names.items():
                if code & mask:
                    known_mask |= mask
                    hint = hints.get(mask, "No additional information.")
                    hint_html = hint.replace("\n", "<br/>")
                    html += (
                        f'<p style="color:{CLR_ERR};"><b>0x{mask:08X} — {name}</b></p>'
                        f'<p style="color:{CLR_LABEL}; margin-left:12px;">{hint_html}</p><hr/>'
                    )
            unknown = code & ~known_mask
            if unknown:
                html += (
                    f'<p style="color:{CLR_WARN};"><b>0x{unknown:08X} — UNKNOWN BITS</b></p>'
                    f'<p style="color:{CLR_LABEL}; margin-left:12px;">'
                    f'These bits are not in the known error table for fw 0.5.6.</p><hr/>'
                )
            txt.setHtml(html)

        layout.addWidget(txt)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btns.accepted.connect(dlg.accept)
        layout.addWidget(btns)
        dlg.exec()

    def _clear_errors(self):
        try:
            self._mgr.execute("clear_errors")
            # Feed watchdog so it doesn't immediately re-fault
            for i in range(2):
                ax = self._mgr.get_axis(i)
                if ax is not None:
                    try:
                        ax.watchdog_feed()
                    except Exception:
                        pass
            log.info("Control: errors cleared + watchdog fed")
        except Exception as e:
            log.error("Control: clear errors failed: %s", e)

    # ── Control group ─────────────────────────────────────────────────────────

    def _build_control_group(self):
        box = QGroupBox("Control")
        col = QVBoxLayout(box)
        col.setSpacing(6)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._mode_grp = QButtonGroup(self)
        self._rb_vel = QRadioButton("Vel")
        self._rb_pos = QRadioButton("Pos")
        self._rb_trq = QRadioButton("Trq")
        self._rb_vel.setChecked(True)
        for i, rb in enumerate((self._rb_vel, self._rb_pos, self._rb_trq)):
            self._mode_grp.addButton(rb, i)
            mode_row.addWidget(rb)
        mode_row.addStretch()
        col.addLayout(mode_row)

        # Setpoint
        sp_row = QHBoxLayout()
        sp_row.addWidget(QLabel("SP:"))
        self.sp_setpoint = _dspin(-1000, 1000, 0.0, dec=3, step=0.1)
        self._lbl_sp_units = QLabel("t/s")
        sp_row.addWidget(self.sp_setpoint)
        sp_row.addWidget(self._lbl_sp_units)
        col.addLayout(sp_row)

        self._rb_vel.toggled.connect(self._on_mode_changed)
        self._rb_pos.toggled.connect(self._on_mode_changed)
        self._rb_trq.toggled.connect(self._on_mode_changed)
        self.sp_setpoint.valueChanged.connect(self._send_setpoint)

        # Enable / Disable buttons
        btn_row = QHBoxLayout()
        self.btn_enable = QPushButton("Enable")
        self.btn_disable = QPushButton("Idle")
        self.btn_enable.setStyleSheet(f"color: {CLR_OK};")
        self.btn_disable.setStyleSheet(f"color: {CLR_WARN};")
        for b in (self.btn_enable, self.btn_disable):
            b.setEnabled(False)
            btn_row.addWidget(b)
        col.addLayout(btn_row)

        self.btn_enable.clicked.connect(self._enable_closed_loop)
        self.btn_disable.clicked.connect(self._disable_idle)

        return box

    def _on_mode_changed(self):
        if self._rb_vel.isChecked():
            self._lbl_sp_units.setText("t/s")
        elif self._rb_pos.isChecked():
            self._lbl_sp_units.setText("turns")
        else:
            self._lbl_sp_units.setText("Nm")

    def _get_control_mode(self) -> int:
        """Return ODrive control_mode int: 2=vel, 3=pos, 1=torque."""
        if self._rb_vel.isChecked():
            return 2
        elif self._rb_pos.isChecked():
            return 3
        return 1

    def _send_setpoint(self, val):
        # Manual setpoint change from spinbox cancels any running waveform
        if self._wave_timer.isActive():
            self._wave_timer.stop()
            self._wave_type = None
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            return
        try:
            mode = self._get_control_mode()
            if mode == 3:
                axis.controller.input_pos = val
                self._cur_sp_pos = float(val)
            elif mode == 2:
                axis.controller.input_vel = val
                self._cur_sp_vel = float(val)
            else:
                axis.controller.input_torque = val
        except Exception as e:
            log.error("Control: setpoint write error: %s", e)

    def _get_setpoints(self):
        """Return (vel_sp, pos_sp) for the charts overlay."""
        return self._cur_sp_vel, self._cur_sp_pos

    def _enable_closed_loop(self):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            return
        try:
            axis.controller.config.control_mode = self._get_control_mode()
            axis.controller.config.input_mode = 1  # PASSTHROUGH
            # Zero all setpoints before enabling
            for attr in ("input_pos", "input_vel", "input_torque"):
                try:
                    setattr(axis.controller, attr, 0.0)
                except Exception:
                    pass
            axis.requested_state = 8  # CLOSED_LOOP_CONTROL
            log.info("Control: axis%d -> closed loop (mode=%d)", self._ax_idx(), self._get_control_mode())
        except Exception as e:
            log.error("Control: enable error: %s", e)

    def _disable_idle(self):
        if self._wave_timer.isActive():
            self._wave_timer.stop()
        self._wave_type = None
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            return
        try:
            axis.requested_state = 1  # IDLE
            self.sp_setpoint.setValue(0.0)
            self._cur_sp_vel = 0.0
            self._cur_sp_pos = 0.0
            log.info("Control: axis%d -> idle", self._ax_idx())
        except Exception as e:
            log.error("Control: disable error: %s", e)

    # ── Gains group ───────────────────────────────────────────────────────────

    def _build_gains_group(self):
        box = QGroupBox("Gains")
        form = QFormLayout(box)
        form.setSpacing(6)

        self.sp_pos_gain = _dspin(0, 1000, 20.0, dec=3, step=0.5)
        self.sp_vel_gain = _dspin(0, 10, 0.16, dec=5, step=0.001)
        self.sp_vel_int = _dspin(0, 10, 0.32, dec=5, step=0.001)
        self.sp_vel_lim = _dspin(0, 500, 20.0, dec=2, step=0.5, suffix="t/s")

        form.addRow("pos_gain", self.sp_pos_gain)
        form.addRow("vel_gain", self.sp_vel_gain)
        form.addRow("vel_int", self.sp_vel_int)
        form.addRow("vel_limit", self.sp_vel_lim)

        btn_row = QHBoxLayout()
        self.btn_read_gains = QPushButton("Read")
        self.btn_apply_gains = QPushButton("Apply")
        self.btn_apply_gains.setStyleSheet(f"color: {CLR_INFO};")
        for b in (self.btn_read_gains, self.btn_apply_gains):
            b.setEnabled(False)
            btn_row.addWidget(b)
        form.addRow("", btn_row)

        self.btn_read_gains.clicked.connect(self._read_gains)
        self.btn_apply_gains.clicked.connect(self._apply_gains)

        return box

    def on_connected(self):
        """Called by MainWindow after any successful connect — refresh gains."""
        self._read_gains()
        self._read_ff_flags()

    def _read_gains(self):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            return
        try:
            cc = axis.controller.config
            self.sp_pos_gain.setValue(float(getattr(cc, "pos_gain", 20.0)))
            self.sp_vel_gain.setValue(float(getattr(cc, "vel_gain", 0.16)))
            self.sp_vel_int.setValue(float(getattr(cc, "vel_integrator_gain", 0.32)))
            self.sp_vel_lim.setValue(float(getattr(cc, "vel_limit", 20.0)))
            log.info("Control: axis%d gains read OK", self._ax_idx())
        except Exception as e:
            log.error("Control: read gains error: %s", e)

    def _apply_gains(self):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            return
        try:
            cc = axis.controller.config
            cc.pos_gain = self.sp_pos_gain.value()
            cc.vel_gain = self.sp_vel_gain.value()
            cc.vel_integrator_gain = self.sp_vel_int.value()
            cc.vel_limit = self.sp_vel_lim.value()
            log.info("Control: axis%d gains applied (vel_gain=%.5f vel_int=%.5f)",
                     self._ax_idx(), cc.vel_gain, cc.vel_integrator_gain)
        except Exception as e:
            log.error("Control: apply gains error: %s", e)

    # ── Feedforward / anticogging toggles ────────────────────────────────────

    def _build_ff_group(self):
        box = QGroupBox("Feedforwards")
        col = QVBoxLayout(box)
        col.setSpacing(4)

        # Track whether user is actively toggling to avoid fighting with poll read-back
        self._ff_user_toggling = False

        self.cb_rwl_ff = QCheckBox("R·I + ωL·I feedforward  (R_wL_FF)")
        self.cb_bemf_ff = QCheckBox("Back-EMF feedforward  (bEMF_FF)")
        self.cb_anticog = QCheckBox("Anticogging enabled")

        self.lbl_rwl_reg = QLabel("register: —")
        self.lbl_bemf_reg = QLabel("register: —")
        self.lbl_anticog_reg = QLabel("register: —")

        _reg_style = f"font-family: monospace; font-size: 11px; color: {CLR_MUTED}; margin-left: 20px;"
        for lbl in (self.lbl_rwl_reg, self.lbl_bemf_reg, self.lbl_anticog_reg):
            lbl.setStyleSheet(_reg_style)

        for cb in (self.cb_rwl_ff, self.cb_bemf_ff, self.cb_anticog):
            cb.setEnabled(False)

        col.addWidget(self.cb_rwl_ff)
        col.addWidget(self.lbl_rwl_reg)
        col.addWidget(self.cb_bemf_ff)
        col.addWidget(self.lbl_bemf_reg)
        col.addWidget(self.cb_anticog)
        col.addWidget(self.lbl_anticog_reg)

        self.cb_rwl_ff.toggled.connect(
            lambda on: self._user_write_flag("motor.config.R_wL_FF_enable", on, "R_wL_FF")
        )
        self.cb_bemf_ff.toggled.connect(
            lambda on: self._user_write_flag("motor.config.bEMF_FF_enable", on, "bEMF_FF")
        )
        self.cb_anticog.toggled.connect(
            lambda on: self._user_write_flag(
                "controller.config.anticogging.anticogging_enabled", on, "anticogging"
            )
        )

        self.lbl_anticog_valid = QLabel("anticogging_valid: —")
        self.lbl_anticog_valid.setStyleSheet(f"font-family: monospace; font-size: 11px; color: {CLR_MUTED};")
        col.addWidget(self.lbl_anticog_valid)

        self.lbl_anticog_precal = QLabel("pre_calibrated: —")
        self.lbl_anticog_precal.setStyleSheet(f"font-family: monospace; font-size: 11px; color: {CLR_MUTED};")
        col.addWidget(self.lbl_anticog_precal)

        return box

    def _user_write_flag(self, attr_path: str, value: bool, label: str):
        """Write flag from user checkbox click — sets guard so poll doesn't fight."""
        self._ff_user_toggling = True
        self._write_flag(attr_path, value, label)
        # Allow poll read-back after a short delay so the write has time to settle
        QTimer.singleShot(500, self._clear_ff_toggle_guard)

    def _clear_ff_toggle_guard(self):
        self._ff_user_toggling = False

    def _write_flag(self, attr_path: str, value: bool, label: str):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            return
        try:
            obj = axis
            parts = attr_path.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], bool(value))
            log.info("Control: axis%d %s = %s", self._ax_idx(), label, value)
        except Exception as e:
            log.error("Control: write %s error: %s", label, e)

    def _read_ff_flags(self):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            return
        try:
            rwl = bool(getattr(axis.motor.config, "R_wL_FF_enable", False))
            bemf = bool(getattr(axis.motor.config, "bEMF_FF_enable", False))
            ac = bool(getattr(axis.controller.config.anticogging, "anticogging_enabled", False))
            ac_valid = bool(getattr(axis.controller, "anticogging_valid", False))
            ac_precal = bool(getattr(axis.controller.config.anticogging, "pre_calibrated", False))

            # Update checkboxes only if user isn't actively toggling
            if not self._ff_user_toggling:
                for cb, v in ((self.cb_rwl_ff, rwl), (self.cb_bemf_ff, bemf), (self.cb_anticog, ac)):
                    cb.blockSignals(True)
                    cb.setChecked(v)
                    cb.blockSignals(False)

            # Always update register readback labels
            _colored(self.lbl_rwl_reg, f"register: {rwl}",
                     CLR_OK if rwl else CLR_MUTED)
            _colored(self.lbl_bemf_reg, f"register: {bemf}",
                     CLR_OK if bemf else CLR_MUTED)
            _colored(self.lbl_anticog_reg, f"register: {ac}",
                     CLR_OK if ac else CLR_MUTED)

            _colored(self.lbl_anticog_valid, f"anticogging_valid: {ac_valid}",
                     CLR_OK if ac_valid else CLR_WARN)
            _colored(self.lbl_anticog_precal, f"pre_calibrated: {ac_precal}",
                     CLR_OK if ac_precal else CLR_WARN)
        except Exception as e:
            log.error("Control: read FF flags error: %s", e)

    # ── Quick buttons ─────────────────────────────────────────────────────────

    def _build_quick_buttons(self):
        box = QGroupBox("Quick Actions")
        col = QVBoxLayout(box)
        col.setSpacing(4)

        row1 = QHBoxLayout()
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setStyleSheet(f"color: {CLR_ERR}; font-weight: bold;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_motor)
        row1.addWidget(self.btn_stop)

        self.btn_spin_test = QPushButton("Spin Test")
        self.btn_spin_test.setStyleSheet(f"color: {CLR_CAL};")
        self.btn_spin_test.setEnabled(False)
        self.btn_spin_test.clicked.connect(self._quick_spin_test)
        row1.addWidget(self.btn_spin_test)
        col.addLayout(row1)

        # Waveform parameters
        wf_params = QHBoxLayout()
        wf_params.addWidget(QLabel("Amp:"))
        self.sp_wave_amp = _dspin(0, 100, 1.0, dec=3, step=0.1)
        wf_params.addWidget(self.sp_wave_amp)
        wf_params.addWidget(QLabel("Period:"))
        self.sp_wave_period = _dspin(0.05, 60, 2.0, dec=2, step=0.1, suffix="s")
        wf_params.addWidget(self.sp_wave_period)
        col.addLayout(wf_params)

        # Waveform buttons — mode (vel/pos) is taken from the Control radio
        wf_btns = QHBoxLayout()
        self.btn_square = QPushButton("Square")
        self.btn_square.setStyleSheet(f"color: {CLR_INFO};")
        self.btn_square.setEnabled(False)
        self.btn_square.clicked.connect(lambda: self._start_waveform("square"))
        wf_btns.addWidget(self.btn_square)

        self.btn_sine = QPushButton("Sine")
        self.btn_sine.setStyleSheet(f"color: {CLR_INFO};")
        self.btn_sine.setEnabled(False)
        self.btn_sine.clicked.connect(lambda: self._start_waveform("sine"))
        wf_btns.addWidget(self.btn_sine)

        self.btn_stop_wave = QPushButton("Stop Wave")
        self.btn_stop_wave.setStyleSheet(f"color: {CLR_WARN};")
        self.btn_stop_wave.setEnabled(False)
        self.btn_stop_wave.clicked.connect(self._stop_waveform)
        wf_btns.addWidget(self.btn_stop_wave)
        col.addLayout(wf_btns)

        return box

    # ── Waveform generator ───────────────────────────────────────────────────

    def _start_waveform(self, wave_type: str):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            return
        mode = self._get_control_mode()
        if mode not in (2, 3):
            log.warning("Control: waveform requires Vel or Pos mode (got torque)")
            return
        try:
            axis.controller.config.control_mode = mode
            axis.controller.config.input_mode = 1  # PASSTHROUGH

            if mode == 3:
                self._wave_mode = "pos"
                self._wave_offset = float(getattr(axis.encoder, "pos_estimate", 0.0))
            else:
                self._wave_mode = "vel"
                self._wave_offset = 0.0

            if getattr(axis, "current_state", 0) != 8:
                for attr in ("input_pos", "input_vel", "input_torque"):
                    try:
                        setattr(axis.controller, attr, 0.0)
                    except Exception:
                        pass
                axis.requested_state = 8  # CLOSED_LOOP_CONTROL

            self._wave_type = wave_type
            self._wave_t0 = time.perf_counter()
            self._wave_timer.start()
            log.info("Control: %s wave on %s axis%d amp=%.3f period=%.3fs",
                     wave_type, self._wave_mode, self._ax_idx(),
                     self.sp_wave_amp.value(), self.sp_wave_period.value())
        except Exception as e:
            log.error("Control: waveform start error: %s", e)

    def _stop_waveform(self):
        if self._wave_timer.isActive():
            self._wave_timer.stop()
        was_active = self._wave_type is not None
        self._wave_type = None
        if not was_active:
            return
        # Hold the last position (pos mode) or zero velocity (vel mode)
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            return
        try:
            if self._wave_mode == "pos":
                axis.controller.input_pos = self._cur_sp_pos
            else:
                axis.controller.input_vel = 0.0
                self._cur_sp_vel = 0.0
        except Exception as e:
            log.error("Control: waveform stop error: %s", e)
        log.info("Control: waveform stopped")

    def _wave_tick(self):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None or self._wave_type is None:
            return
        amp = float(self.sp_wave_amp.value())
        period = max(float(self.sp_wave_period.value()), 0.01)
        t = time.perf_counter() - self._wave_t0
        phase = (t % period) / period  # 0..1
        if self._wave_type == "square":
            val = amp if phase < 0.5 else -amp
        elif self._wave_type == "sine":
            val = amp * math.sin(2.0 * math.pi * phase)
        else:
            return
        try:
            if self._wave_mode == "pos":
                target = self._wave_offset + val
                axis.controller.input_pos = target
                self._cur_sp_pos = target
            else:
                axis.controller.input_vel = val
                self._cur_sp_vel = val
        except Exception as e:
            log.error("Control: wave write error: %s", e)

    def _stop_motor(self):
        """Emergency stop — set both axes to idle."""
        if self._wave_timer.isActive():
            self._wave_timer.stop()
        self._wave_type = None
        try:
            self._mgr.safe_set("axis0.requested_state", 1)
            self._mgr.safe_set("axis1.requested_state", 1)
            self.sp_setpoint.setValue(0.0)
            self._cur_sp_vel = 0.0
            self._cur_sp_pos = 0.0
            log.info("Control: STOP — both axes to idle")
        except Exception as e:
            log.error("Control: stop error: %s", e)

    def _quick_spin_test(self):
        """Enter closed loop velocity at 1 t/s for a quick sanity check."""
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            return
        try:
            axis.controller.config.control_mode = 2  # VELOCITY
            axis.controller.config.input_mode = 1    # PASSTHROUGH
            axis.controller.input_vel = 0.0
            axis.requested_state = 8  # CLOSED_LOOP
            axis.controller.input_vel = 1.0
            self._rb_vel.setChecked(True)
            self.sp_setpoint.setValue(1.0)
            log.info("Control: axis%d quick spin test at 1 t/s", self._ax_idx())
        except Exception as e:
            log.error("Control: spin test error: %s", e)

    # ── Live readout ──────────────────────────────────────────────────────────

    def _build_readout_group(self):
        box = QGroupBox("Live Readout")
        form = QFormLayout(box)
        form.setSpacing(6)

        self.lbl_state = QLabel("—")
        self.lbl_pos = QLabel("—")
        self.lbl_vel = QLabel("—")
        self.lbl_iq = QLabel("—")
        self.lbl_vbus = QLabel("—")
        self.lbl_power = QLabel("—")

        for lbl in (self.lbl_state, self.lbl_pos, self.lbl_vel,
                    self.lbl_iq, self.lbl_vbus, self.lbl_power):
            lbl.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")

        form.addRow("State", self.lbl_state)
        form.addRow("Position", self.lbl_pos)
        form.addRow("Velocity", self.lbl_vel)
        form.addRow("Iq", self.lbl_iq)
        form.addRow("Vbus", self.lbl_vbus)
        form.addRow("Power", self.lbl_power)

        return box

    # ── Poll update (called by MainWindow every 100ms) ────────────────────────

    def poll_update(self):
        connected = self._mgr.connected
        axis = self._mgr.get_axis(self._ax_idx()) if connected else None

        # Button state
        self.btn_clear_errors.setEnabled(connected)
        self.btn_enable.setEnabled(connected)
        self.btn_disable.setEnabled(connected)
        self.btn_read_gains.setEnabled(connected)
        self.btn_apply_gains.setEnabled(connected)
        self.btn_stop.setEnabled(connected)
        self.btn_spin_test.setEnabled(connected)
        self.btn_square.setEnabled(connected)
        self.btn_sine.setEnabled(connected)
        self.btn_stop_wave.setEnabled(connected)
        self.cb_rwl_ff.setEnabled(connected)
        self.cb_bemf_ff.setEnabled(connected)
        self.cb_anticog.setEnabled(connected)

        if axis is None:
            for lbl in (self.lbl_state, self.lbl_pos, self.lbl_vel,
                        self.lbl_iq, self.lbl_vbus, self.lbl_power):
                _colored(lbl, "—", CLR_MUTED)
            for lbl in self._err_labels.values():
                _colored(lbl, "—", CLR_MUTED)
            self._charts.poll_update()
            return

        # Errors
        err_tables = {
            "axis": (AXIS_ERRORS, "error"),
            "motor": (MOTOR_ERRORS, "motor.error"),
            "encoder": (ENCODER_ERRORS, "encoder.error"),
            "controller": (CONTROLLER_ERRORS, "controller.error"),
        }
        for key, (table, attr_path) in err_tables.items():
            val = self._mgr.safe_get(f"axis{self._ax_idx()}.{attr_path}", 0)
            self._last_err[key] = val
            if val == 0:
                _colored(self._err_labels[key], "none", CLR_OK)
            else:
                text = decode_errors_str(val, table)
                _colored(self._err_labels[key], f"0x{val:04X}  {text}", CLR_ERR)

        # Live readouts
        try:
            state = getattr(axis, "current_state", 0)
            state_name = AXIS_STATES.get(state, f"State {state}")
            color = CLR_OK if state == 8 else (CLR_WARN if state == 1 else CLR_INFO)
            _colored(self.lbl_state, state_name, color)

            pos = getattr(axis.encoder, "pos_estimate", 0.0)
            _colored(self.lbl_pos, f"{pos:.4f} turns", CLR_OK)

            vel = getattr(axis.encoder, "vel_estimate", 0.0)
            _colored(self.lbl_vel, f"{vel:.4f} t/s", CLR_OK)

            iq = getattr(axis.motor.current_control, "Iq_measured", 0.0)
            _colored(self.lbl_iq, f"{iq:.3f} A", CLR_OK if abs(iq) < 5 else CLR_WARN)

            vbus = self._mgr.vbus_voltage()
            if vbus is not None:
                _colored(self.lbl_vbus, f"{vbus:.2f} V", CLR_OK if vbus > 10 else CLR_WARN)

            elec_power = getattr(axis.controller, "electrical_power", 0.0)
            mech_power = getattr(axis.controller, "mechanical_power", 0.0)
            _colored(self.lbl_power, f"E={elec_power:.1f} M={mech_power:.1f} W", CLR_OK)

        except Exception:
            pass

        # Update FF register labels
        self._read_ff_flags()

        # Update charts
        self._charts.poll_update()
