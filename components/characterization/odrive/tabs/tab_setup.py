"""tab_setup.py — Motor & encoder setup, calibration, flash/verify.

All ODrive operations are delegated to core.odrive_operations so that
the same logic can be triggered from Claude inbox commands.
"""

import logging

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QScrollArea, QCheckBox,
)

from core.constants import (
    AXIS_STATES, MOTOR_TYPE_NAMES, MOTOR_TYPE_VALUES,
    ENCODER_TYPE_NAMES, ENCODER_TYPE_VALUES,
)
from core.odrive_operations import (
    read_config, flash_config, snapshot_params, verify_params,
    auto_tune_after_cal, lock_startup_flags, check_startup_flags,
)
from ui.theme import CLR_OK, CLR_WARN, CLR_ERR, CLR_INFO, CLR_CAL, CLR_MUTED

log = logging.getLogger("odrive_gui")

# Calibration states
_CAL_IDLE = 0
_CAL_RUNNING = 1
_CAL_DONE = 2
_CAL_FAIL = 3

_REBOOT_WAIT_MS = 4000


# ── Reconnect worker (QThread) ───────────────────────────────────────────────

class _ReconnectWorker(QThread):
    success = Signal(object)
    failed = Signal(str)

    def __init__(self, manager):
        super().__init__()
        self._mgr = manager

    def run(self):
        try:
            self._mgr.disconnect()
            odrv = self._mgr.connect()
            self.success.emit(odrv)
        except Exception as e:
            self.failed.emit(str(e))


# ── Helpers ──────────────────────────────────────────────────────────────────

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


# ── Setup Tab ────────────────────────────────────────────────────────────────

class TabSetup(QWidget):
    """Motor/encoder config, calibration, flash/verify."""

    # Emitted when Setup reconnects after reboot so MainWindow can update
    reconnected = Signal()

    def __init__(self, manager, get_axis_idx, parent=None):
        super().__init__(parent)
        self._mgr = manager
        self._get_axis_idx = get_axis_idx
        self._cal_state = _CAL_IDLE
        self._expected_params: dict = {}
        self._reboot_timer = QTimer(self)
        self._reboot_timer.setSingleShot(True)
        self._reboot_timer.timeout.connect(self._reconnect_after_reboot)
        self._reconnect_worker: _ReconnectWorker | None = None
        self._post_reboot_action: str = ""  # "verify" or "cal_done"
        self._build()

    def _ax_idx(self) -> int:
        return self._get_axis_idx()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        root = QVBoxLayout(inner)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        # Two-column: motor + encoder
        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        top_row.addWidget(self._build_motor_group())
        top_row.addWidget(self._build_encoder_group())
        root.addLayout(top_row)

        # Safety status
        root.addWidget(self._build_safety_group())

        # Action buttons
        root.addWidget(self._build_action_buttons())

        # Calibration progress
        root.addWidget(self._build_cal_progress())

        # Live status
        root.addWidget(self._build_live_status())

        root.addStretch()

        # Wire encoder type toggle
        self.enc_type_combo.currentIndexChanged.connect(self._on_enc_type_changed)
        self._on_enc_type_changed(0)

    def _build_motor_group(self):
        box = QGroupBox("Motor Parameters")
        form = QFormLayout(box)
        form.setSpacing(8)

        self.motor_type = QComboBox()
        self.motor_type.addItems(MOTOR_TYPE_NAMES)

        self.pole_pairs = QSpinBox()
        self.pole_pairs.setRange(1, 50)
        self.pole_pairs.setValue(7)

        self.current_lim    = _dspin(0, 60, 10, dec=2, step=0.5, suffix="A")
        self.cal_current    = _dspin(0, 60, 10, dec=2, step=0.5, suffix="A")
        self.vel_lim        = _dspin(0, 500, 2, dec=2, step=0.5, suffix="t/s")
        self.phase_res      = _dspin(0, 10, 0, dec=4, step=0.001, suffix="Ohm")
        self.phase_ind      = _dspin(0, 0.1, 0, dec=6, step=0.000001, suffix="H")
        self.res_calib_volt = _dspin(0, 24, 2, dec=2, step=0.5, suffix="V")
        self.torque_const   = _dspin(0, 10, 0.04, dec=5, step=0.001, suffix="Nm/A")

        # Phase R/L are read-only display — set by calibration
        self.phase_res.setEnabled(False)
        self.phase_ind.setEnabled(False)

        form.addRow("Motor type",               self.motor_type)
        form.addRow("Pole pairs",               self.pole_pairs)
        form.addRow("Current limit",            self.current_lim)
        form.addRow("Calibration current",      self.cal_current)
        form.addRow("Velocity limit",           self.vel_lim)
        form.addRow("Phase resistance",         self.phase_res)
        form.addRow("Phase inductance",         self.phase_ind)
        form.addRow("Cal max voltage (R meas)", self.res_calib_volt)
        form.addRow("Torque constant",          self.torque_const)

        hint = QLabel("Phase R/L auto-populated after calibration. Raise cal voltage if R cal fails.")
        hint.setStyleSheet(f"color: {CLR_MUTED}; font-size: 10px;")
        form.addRow("", hint)

        return box

    def _build_encoder_group(self):
        box = QGroupBox("Encoder Parameters")
        form = QFormLayout(box)
        form.setSpacing(8)

        self.enc_type_combo = QComboBox()
        self.enc_type_combo.addItems(ENCODER_TYPE_NAMES)

        self.enc_cs_pin = QSpinBox()
        self.enc_cs_pin.setRange(1, 8)
        self.enc_cs_pin.setValue(3)

        self.enc_cpr = QSpinBox()
        self.enc_cpr.setRange(4, 65536)
        self.enc_cpr.setValue(16384)

        self.enc_precal_chk = QCheckBox("Pre-calibrated")
        self.enc_precal_chk.setEnabled(False)

        self.motor_precal_chk = QCheckBox("Motor pre-calibrated")
        self.motor_precal_chk.setEnabled(False)

        form.addRow("Encoder type",    self.enc_type_combo)
        form.addRow("SPI CS GPIO pin", self.enc_cs_pin)
        form.addRow("CPR",             self.enc_cpr)
        form.addRow("",                self.enc_precal_chk)
        form.addRow("",                self.motor_precal_chk)

        self.btn_setup_encoder = QPushButton("Setup Encoder")
        self.btn_setup_encoder.setToolTip(
            "Write encoder type, CPR, and SPI CS pin to ODrive RAM"
        )
        self.btn_setup_encoder.setEnabled(False)
        self.btn_setup_encoder.clicked.connect(self._on_setup_encoder)
        form.addRow("", self.btn_setup_encoder)

        return box

    def _build_safety_group(self):
        box = QGroupBox("Boot Safety — Startup Motion Flags")
        row = QHBoxLayout(box)

        self.lbl_startup_safe = QLabel("Not connected")
        self.lbl_startup_safe.setStyleSheet(f"font-family: monospace; color: {CLR_MUTED};")
        row.addWidget(self.lbl_startup_safe)
        row.addStretch()

        note = QLabel("On every Flash: startup_* = False, enable_brake_resistor = True")
        note.setStyleSheet(f"color: {CLR_MUTED}; font-size: 10px;")
        row.addWidget(note)

        return box

    def _build_action_buttons(self):
        box = QGroupBox("Actions")
        col = QVBoxLayout(box)

        row1 = QHBoxLayout()
        self.btn_read   = QPushButton("Read from ODrive")
        self.btn_flash  = QPushButton("Flash to ODrive")
        self.btn_flash.setStyleSheet(f"color: {CLR_INFO};")
        self.btn_cal    = QPushButton("Calibrate Motor + Encoder")
        self.btn_cal.setStyleSheet(f"color: {CLR_CAL};")
        self.btn_verify = QPushButton("Reboot && Verify")
        self.btn_verify.setStyleSheet(f"color: {CLR_WARN};")

        for b in (self.btn_read, self.btn_flash, self.btn_cal, self.btn_verify):
            b.setEnabled(False)
            row1.addWidget(b)

        col.addLayout(row1)

        self.btn_read.clicked.connect(self._on_read)
        self.btn_flash.clicked.connect(self._on_flash)
        self.btn_cal.clicked.connect(self._on_calibrate)
        self.btn_verify.clicked.connect(self._on_reboot_verify)

        return box

    def _build_cal_progress(self):
        box = QGroupBox("Calibration / Flash Progress")
        form = QFormLayout(box)
        form.setSpacing(6)

        self.lbl_cal_state = QLabel("—")
        self.lbl_cal_state.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")
        self.lbl_cal_msg = QLabel("—")
        self.lbl_cal_msg.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")

        form.addRow("State",   self.lbl_cal_state)
        form.addRow("Message", self.lbl_cal_msg)

        return box

    def _build_live_status(self):
        box = QGroupBox("Live Status")
        form = QFormLayout(box)
        form.setSpacing(6)

        self.lbl_axis_state  = QLabel("—")
        self.lbl_motor_cal   = QLabel("—")
        self.lbl_enc_ready   = QLabel("—")
        self.lbl_startup_chk = QLabel("—")
        for lbl in (self.lbl_axis_state, self.lbl_motor_cal,
                    self.lbl_enc_ready, self.lbl_startup_chk):
            lbl.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")

        form.addRow("Axis state",         self.lbl_axis_state)
        form.addRow("Motor calibrated",   self.lbl_motor_cal)
        form.addRow("Encoder ready",      self.lbl_enc_ready)
        form.addRow("Startup flags safe", self.lbl_startup_chk)

        return box

    def _on_enc_type_changed(self, idx):
        is_spi = ENCODER_TYPE_VALUES[idx] in (256, 257)
        self.enc_cs_pin.setEnabled(is_spi)

    # ── Button state management ───────────────────────────────────────────────

    def _set_buttons(self, enabled: bool):
        for b in (self.btn_read, self.btn_flash, self.btn_cal, self.btn_verify,
                  self.btn_setup_encoder):
            b.setEnabled(enabled)

    # ── Setup Encoder ────────────────────────────────────────────────────────

    def _on_setup_encoder(self):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            _colored(self.lbl_cal_state, "Not connected", CLR_ERR)
            return

        try:
            ec = axis.encoder.config
            enc_type = ENCODER_TYPE_VALUES[self.enc_type_combo.currentIndex()]
            ec.mode = enc_type
            ec.cpr = self.enc_cpr.value()
            if enc_type in (256, 257):  # SPI absolute
                ec.abs_spi_cs_gpio_pin = self.enc_cs_pin.value()
                # ODrive v3.6 hardware SPI3 sits on GPIO 3/4/5. If any of
                # them is left in ANALOG_IN (default after factory erase),
                # the STM32 pin buffer is disconnected and the SPI bus
                # never drives — symptom is spi_error_rate ≈ 1.0 with one
                # frozen pos_abs read at boot. Force all three to DIGITAL.
                ocfg = self._mgr.odrv.config
                ocfg.gpio3_mode = 0  # DIGITAL
                ocfg.gpio4_mode = 0  # DIGITAL
                ocfg.gpio5_mode = 0  # DIGITAL

            log.info("Setup: axis%d encoder configured — mode=%d CPR=%d CS=%d  gpio3/4/5=DIGITAL",
                     self._ax_idx(), enc_type, self.enc_cpr.value(),
                     self.enc_cs_pin.value())
        except Exception as e:
            _colored(self.lbl_cal_state, "Encoder setup failed", CLR_ERR)
            _colored(self.lbl_cal_msg, str(e), CLR_ERR)
            log.error("Setup: encoder setup failed: %s", e)
            return

        # Save + reboot so the SPI peripheral initialises at boot
        _colored(self.lbl_cal_state, "Saving encoder config, rebooting...", CLR_WARN)
        _colored(self.lbl_cal_msg,
                 f"mode={enc_type}  CPR={self.enc_cpr.value()}  "
                 f"CS pin={self.enc_cs_pin.value()}",
                 CLR_WARN)
        try:
            self._mgr.odrv.save_configuration()
        except Exception:
            pass  # save triggers reboot — USB exception expected

        self._post_reboot_action = ""
        self._start_reboot_sequence()

    # ── Read ──────────────────────────────────────────────────────────────────

    def _on_read(self):
        try:
            cfg = read_config(self._mgr, self._ax_idx())
        except Exception as e:
            _colored(self.lbl_cal_state, "Read failed", CLR_ERR)
            _colored(self.lbl_cal_msg, str(e), CLR_ERR)
            log.error("Setup: read failed: %s", e)
            return

        # Populate form
        mt = cfg["motor_type"]
        idx = MOTOR_TYPE_VALUES.index(mt) if mt in MOTOR_TYPE_VALUES else 0
        self.motor_type.setCurrentIndex(idx)
        self.pole_pairs.setValue(cfg["pole_pairs"])
        self.current_lim.setValue(cfg["current_lim"])
        self.cal_current.setValue(cfg["calibration_current"])
        self.vel_lim.setValue(cfg["vel_limit"])
        self.phase_res.setValue(cfg["phase_resistance"])
        self.phase_ind.setValue(cfg["phase_inductance"])
        self.res_calib_volt.setValue(cfg["resistance_calib_max_voltage"])
        self.torque_const.setValue(cfg["torque_constant"])

        enc_mode = cfg["encoder_mode"]
        enc_idx = ENCODER_TYPE_VALUES.index(enc_mode) if enc_mode in ENCODER_TYPE_VALUES else 0
        self.enc_type_combo.setCurrentIndex(enc_idx)
        self.enc_cs_pin.setValue(cfg["abs_spi_cs_gpio_pin"])
        self.enc_cpr.setValue(cfg["cpr"])
        self.enc_precal_chk.setChecked(cfg["encoder_pre_calibrated"])
        self.motor_precal_chk.setChecked(cfg["motor_pre_calibrated"])

        _colored(self.lbl_cal_state, "Config read OK", CLR_OK)
        _colored(self.lbl_cal_msg,
                 f"R={cfg['phase_resistance']:.4f}  L={cfg['phase_inductance']:.6f}  "
                 f"vel_gain={cfg['vel_gain']:.4f}  vel_int={cfg['vel_integrator_gain']:.4f}",
                 CLR_OK)
        log.info("Setup: axis%d config read OK", self._ax_idx())

    # ── Flash ─────────────────────────────────────────────────────────────────

    def _on_flash(self):
        params = self._form_to_params()
        try:
            self._expected_params = flash_config(self._mgr, self._ax_idx(), params)
        except Exception as e:
            _colored(self.lbl_cal_state, "Flash failed", CLR_ERR)
            _colored(self.lbl_cal_msg, str(e), CLR_ERR)
            log.error("Setup: flash failed: %s", e)
            return

        _colored(self.lbl_cal_state, "Flashed + saved, rebooting...", CLR_WARN)
        _colored(self.lbl_cal_msg, f"Waiting {_REBOOT_WAIT_MS // 1000}s then reconnecting", CLR_WARN)
        self._post_reboot_action = "verify"
        self._start_reboot_sequence()

    def _form_to_params(self) -> dict:
        return {
            "motor_type":                   MOTOR_TYPE_VALUES[self.motor_type.currentIndex()],
            "pole_pairs":                   self.pole_pairs.value(),
            "current_lim":                  self.current_lim.value(),
            "calibration_current":          self.cal_current.value(),
            "resistance_calib_max_voltage": self.res_calib_volt.value(),
            "torque_constant":              self.torque_const.value(),
            "vel_limit":                    self.vel_lim.value(),
            "encoder_mode":                 ENCODER_TYPE_VALUES[self.enc_type_combo.currentIndex()],
            "abs_spi_cs_gpio_pin":          self.enc_cs_pin.value(),
            "cpr":                          self.enc_cpr.value(),
        }

    # ── Calibrate ─────────────────────────────────────────────────────────────

    def _on_calibrate(self):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            return

        state = getattr(axis, "current_state", 0)
        if state != 1:  # not IDLE
            _colored(self.lbl_cal_state, "Axis must be IDLE before calibrating", CLR_WARN)
            return

        # Write encoder config to RAM before calibration
        try:
            ec = axis.encoder.config
            enc_type = ENCODER_TYPE_VALUES[self.enc_type_combo.currentIndex()]
            ec.mode = enc_type
            ec.cpr = self.enc_cpr.value()
            if enc_type in (256, 257):
                ec.abs_spi_cs_gpio_pin = self.enc_cs_pin.value()
        except Exception as e:
            _colored(self.lbl_cal_state, "Encoder config write failed", CLR_ERR)
            _colored(self.lbl_cal_msg, str(e), CLR_ERR)
            return

        self._cal_state = _CAL_RUNNING
        self._set_buttons(False)
        _colored(self.lbl_cal_state, "Calibrating...", CLR_CAL)
        _colored(self.lbl_cal_msg, "FULL_CALIBRATION_SEQUENCE started", CLR_CAL)

        try:
            axis.requested_state = 3  # FULL_CALIBRATION_SEQUENCE
            log.info("Setup: axis%d calibration started", self._ax_idx())
        except Exception as e:
            self._cal_state = _CAL_FAIL
            self._set_buttons(True)
            _colored(self.lbl_cal_state, "FAILED", CLR_ERR)
            _colored(self.lbl_cal_msg, str(e), CLR_ERR)

    def _poll_calibration(self, axis):
        """Called from poll_update() while calibrating."""
        state = getattr(axis, "current_state", 0)
        err = getattr(axis, "error", 0)
        state_name = AXIS_STATES.get(state, f"State {state}")

        _colored(self.lbl_cal_state, f"Calibrating — {state_name}", CLR_CAL)

        if state == 1:  # back to IDLE — finished
            if err != 0:
                self._cal_state = _CAL_FAIL
                from core.odrive_errors import decode_errors_str, AXIS_ERRORS
                desc = decode_errors_str(err, AXIS_ERRORS)
                _colored(self.lbl_cal_state, "FAILED", CLR_ERR)
                _colored(self.lbl_cal_msg, f"Axis error 0x{err:04X}: {desc}", CLR_ERR)
                log.error("Setup: axis%d calibration FAILED — 0x%04X: %s",
                         self._ax_idx(), err, desc)
                self._set_buttons(True)
            else:
                # Success — auto-tune and save
                self._cal_state = _CAL_DONE
                _colored(self.lbl_cal_state, "Calibration OK, auto-tuning...", CLR_OK)
                try:
                    tuned = auto_tune_after_cal(self._mgr, self._ax_idx())
                    _colored(self.lbl_cal_state, "Done — saved + rebooting", CLR_OK)
                    _colored(self.lbl_cal_msg,
                             f"vel_gain={tuned['vel_gain']:.4f}  "
                             f"R={tuned['phase_resistance']:.4f}  "
                             f"L={tuned['phase_inductance']:.6f}",
                             CLR_OK)
                    log.info("Setup: axis%d calibration + auto-tune complete", self._ax_idx())
                    # Reboot sequence (save_configuration already triggered reboot)
                    self._post_reboot_action = "cal_done"
                    self._start_reboot_sequence()
                except Exception as e:
                    _colored(self.lbl_cal_msg, f"Auto-tune warning: {e}", CLR_WARN)
                    log.warning("Setup: auto-tune warning: %s", e)
                    self._set_buttons(True)

    # ── Reboot & Verify ───────────────────────────────────────────────────────

    def _on_reboot_verify(self):
        try:
            self._expected_params = snapshot_params(self._mgr, self._ax_idx())
        except Exception as e:
            _colored(self.lbl_cal_state, "Snapshot failed", CLR_ERR)
            _colored(self.lbl_cal_msg, str(e), CLR_ERR)
            return

        try:
            self._mgr.odrv.reboot()
        except Exception:
            pass  # expected

        _colored(self.lbl_cal_state, "Rebooting...", CLR_WARN)
        _colored(self.lbl_cal_msg, f"Waiting {_REBOOT_WAIT_MS // 1000}s then reconnecting", CLR_WARN)
        self._post_reboot_action = "verify"
        self._start_reboot_sequence()

    # ── Reboot sequence (shared) ──────────────────────────────────────────────

    def _start_reboot_sequence(self):
        self._set_buttons(False)
        self._mgr.disconnect()
        self._reboot_timer.start(_REBOOT_WAIT_MS)

    def _reconnect_after_reboot(self):
        _colored(self.lbl_cal_state, "Reconnecting...", CLR_INFO)
        _colored(self.lbl_cal_msg, "Searching for ODrive...", CLR_INFO)
        self._reconnect_worker = _ReconnectWorker(self._mgr)
        self._reconnect_worker.success.connect(self._on_reconnect_ok)
        self._reconnect_worker.failed.connect(self._on_reconnect_fail)
        self._reconnect_worker.start()

    def _on_reconnect_ok(self, odrv):
        log.info("Setup: reconnected after reboot")
        if self._post_reboot_action == "verify" and self._expected_params:
            actual = snapshot_params(self._mgr, self._ax_idx())
            results = verify_params(self._expected_params, actual)
            passed = sum(1 for r in results if r["ok"])
            total = len(results)
            all_ok = (passed == total)

            color = CLR_OK if all_ok else CLR_ERR
            _colored(self.lbl_cal_state,
                     "All parameters PASS" if all_ok else "Some parameters FAILED",
                     color)
            _colored(self.lbl_cal_msg, f"{passed}/{total} parameters match", color)

            for r in results:
                mark = "PASS" if r["ok"] else "FAIL"
                log.info("  %s  %s: expected=%s  got=%s",
                         mark, r["key"], r["expected"], r["actual"])
        else:
            _colored(self.lbl_cal_state, "Reconnected", CLR_OK)
            _colored(self.lbl_cal_msg, "Ready", CLR_OK)

        self._set_buttons(True)
        self.reconnected.emit()
        # Read config to refresh form
        self._on_read()

    def _on_reconnect_fail(self, msg):
        _colored(self.lbl_cal_state, "Reconnect FAILED", CLR_ERR)
        _colored(self.lbl_cal_msg, msg, CLR_ERR)
        log.error("Setup: reconnect failed: %s", msg)

    # ── Poll update (called by MainWindow every 100ms) ────────────────────────

    def poll_update(self):
        """Called from MainWindow's poll timer."""
        if not self._mgr.connected:
            _colored(self.lbl_axis_state,  "—", CLR_MUTED)
            _colored(self.lbl_motor_cal,   "—", CLR_MUTED)
            _colored(self.lbl_enc_ready,   "—", CLR_MUTED)
            _colored(self.lbl_startup_chk, "Not connected", CLR_MUTED)
            _colored(self.lbl_startup_safe, "Not connected", CLR_MUTED)
            if self._cal_state != _CAL_RUNNING:
                self._set_buttons(False)
            return

        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            return

        # Enable buttons when connected (unless calibrating)
        if self._cal_state != _CAL_RUNNING:
            self._set_buttons(True)

        # Calibration polling
        if self._cal_state == _CAL_RUNNING:
            self._poll_calibration(axis)

        # Live status
        try:
            state = getattr(axis, "current_state", 0)
            _colored(self.lbl_axis_state, AXIS_STATES.get(state, f"State {state}"), CLR_OK)

            motor_cal = getattr(axis.motor, "is_calibrated", False)
            _colored(self.lbl_motor_cal,
                     "YES" if motor_cal else "NO",
                     CLR_OK if motor_cal else CLR_ERR)

            enc_rdy = getattr(axis.encoder, "is_ready", False)
            _colored(self.lbl_enc_ready,
                     "YES" if enc_rdy else "NO",
                     CLR_OK if enc_rdy else CLR_ERR)

            startup_ok = check_startup_flags(axis)
            _colored(self.lbl_startup_chk,
                     "All OFF — safe" if startup_ok else "WARNING: motion on boot!",
                     CLR_OK if startup_ok else CLR_ERR)
            _colored(self.lbl_startup_safe,
                     "All startup_* = False" if startup_ok else "WARNING — motion flags enabled!",
                     CLR_OK if startup_ok else CLR_ERR)

        except Exception:
            pass  # USB glitch — main poll will handle disconnect
