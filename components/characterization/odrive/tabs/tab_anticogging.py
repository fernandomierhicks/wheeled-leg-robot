"""tab_anticogging.py — Anticogging calibration workflow tab.

Consolidates run_anticogging.py + check_anticog_nvm.py + probe_anticogging.py
into a single GUI tab with sequential step buttons, progress bar, and quality test.

All long-running operations are poll-based (no QThread) so the GUI stays
responsive and the ODrive USB bus isn't contested between threads.
"""

import logging
import statistics
import time

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QDoubleSpinBox,
    QProgressBar, QScrollArea,
)

from core.constants import AXIS_STATES, COGGING_MAP_SIZE
from core.odrive_operations import anticogging_prerequisites
from ui.theme import CLR_OK, CLR_WARN, CLR_ERR, CLR_INFO, CLR_CAL, CLR_MUTED

log = logging.getLogger("odrive_gui")

# ── State machine ────────────────────────────────────────────────────────────
_IDLE = 0
_VEL_TEST = 1
_CALIBRATING = 2
_QUALITY_ON = 3       # sampling Iq with anticogging ON
_QUALITY_OFF = 4      # sampling Iq with anticogging OFF
_SAVING = 5           # reboot in progress

_VEL_TEST_TIMEOUT = 10.0    # seconds
_QUALITY_SAMPLE_SEC = 4.0   # seconds per ON/OFF phase
_SWEEP_TIMEOUT = 600.0      # 10 min max for full sweep
_REBOOT_WAIT_MS = 4000

# Threshold presets
THRESHOLD_PRESETS = {
    "Good (5)": 5.0,
    "Quick (100)": 100.0,
}


# ── Reconnect worker (QThread — blocking USB enumeration) ────────────────────

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

def _colored(label: QLabel, text: str, color: str):
    label.setText(text)
    label.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {color};")


# ── Anticogging Tab ──────────────────────────────────────────────────────────

class TabAnticogging(QWidget):
    """Anticogging calibration workflow: prereqs -> vel test -> sweep -> quality -> save."""

    # Emitted after save+reboot reconnect so MainWindow can update status bar
    reconnected = Signal()

    def __init__(self, manager, get_axis_idx, parent=None):
        super().__init__(parent)
        self._mgr = manager
        self._get_axis_idx = get_axis_idx

        # State machine
        self._state = _IDLE
        self._t0 = 0.0               # start time for current operation
        self._prev_idx = -1           # last logged sweep index
        self._quality_samples = []    # Iq samples for current phase
        self._quality_on_result = None  # (mean, stdev, n) from ON phase

        # Sequential step progress (0=none, 1=prereq, 2=vel, 3=cal, 4=quality)
        self._step_done = 0

        # Reboot / reconnect
        self._reconnect_worker = None
        self._reboot_timer = QTimer(self)
        self._reboot_timer.setSingleShot(True)
        self._reboot_timer.timeout.connect(self._reconnect_after_reboot)

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

        root.addWidget(self._build_prereq_group())
        root.addWidget(self._build_calibration_group())
        root.addWidget(self._build_quality_group())
        root.addWidget(self._build_save_group())
        root.addWidget(self._build_live_status())
        root.addStretch()

    # ── 1. Prerequisites ─────────────────────────────────────────────────────

    def _build_prereq_group(self):
        box = QGroupBox("1. Prerequisites Check")
        col = QVBoxLayout(box)

        row = QHBoxLayout()
        self.btn_prereq = QPushButton("Check Prerequisites")
        self.btn_prereq.clicked.connect(self._on_prereq_check)
        row.addWidget(self.btn_prereq)
        self.lbl_prereq_status = QLabel("--")
        self.lbl_prereq_status.setStyleSheet(
            f"font-family: monospace; color: {CLR_MUTED};")
        row.addWidget(self.lbl_prereq_status, stretch=1)
        col.addLayout(row)

        # Individual check labels (6 checks)
        self._prereq_labels = []
        for _ in range(6):
            lbl = QLabel("")
            lbl.setStyleSheet(
                f"font-family: monospace; font-size: 11px; color: {CLR_MUTED};")
            col.addWidget(lbl)
            self._prereq_labels.append(lbl)

        return box

    def _on_prereq_check(self):
        result = anticogging_prerequisites(self._mgr, self._ax_idx())

        for i, check in enumerate(result["checks"]):
            if i < len(self._prereq_labels):
                color = CLR_OK if check["ok"] else CLR_ERR
                mark = "PASS" if check["ok"] else "FAIL"
                self._prereq_labels[i].setText(
                    f"  {mark}  {check['name']}: {check['msg']}")
                self._prereq_labels[i].setStyleSheet(
                    f"font-family: monospace; font-size: 11px; color: {color};")

        # Clear unused labels
        for i in range(len(result["checks"]), len(self._prereq_labels)):
            self._prereq_labels[i].setText("")

        if result["ok"]:
            _colored(self.lbl_prereq_status, "All prerequisites PASS", CLR_OK)
            self._step_done = max(self._step_done, 1)
            log.info("Anticogging: prerequisites OK")
        else:
            _colored(self.lbl_prereq_status, "Prerequisites FAILED", CLR_ERR)
            log.warning("Anticogging: prerequisites failed")

        self._update_buttons()

    # ── 2. Calibration (velocity test + sweep) ───────────────────────────────

    def _build_calibration_group(self):
        box = QGroupBox("2. Anticogging Calibration")
        col = QVBoxLayout(box)

        # Threshold row
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Threshold preset:"))
        self.combo_threshold = QComboBox()
        self.combo_threshold.addItems(list(THRESHOLD_PRESETS.keys()))
        self.combo_threshold.currentTextChanged.connect(self._on_preset_changed)
        thresh_row.addWidget(self.combo_threshold)

        thresh_row.addWidget(QLabel("  Custom:"))
        self.sp_threshold = QDoubleSpinBox()
        self.sp_threshold.setRange(0.1, 1000)
        self.sp_threshold.setDecimals(1)
        self.sp_threshold.setSingleStep(1.0)
        self.sp_threshold.setValue(5.0)
        thresh_row.addWidget(self.sp_threshold)
        thresh_row.addStretch()
        col.addLayout(thresh_row)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_vel_test = QPushButton("Velocity Smoke Test")
        self.btn_vel_test.setStyleSheet(f"color: {CLR_CAL};")
        self.btn_vel_test.setEnabled(False)
        self.btn_vel_test.clicked.connect(self._on_vel_test)
        btn_row.addWidget(self.btn_vel_test)

        self.btn_start_cal = QPushButton("Start Calibration")
        self.btn_start_cal.setStyleSheet(f"color: {CLR_INFO};")
        self.btn_start_cal.setEnabled(False)
        self.btn_start_cal.clicked.connect(self._on_start_calibration)
        btn_row.addWidget(self.btn_start_cal)

        self.btn_abort = QPushButton("Abort")
        self.btn_abort.setStyleSheet(f"color: {CLR_ERR};")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self._on_abort)
        btn_row.addWidget(self.btn_abort)
        col.addLayout(btn_row)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, COGGING_MAP_SIZE)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m  (%p%)")
        col.addWidget(self.progress)

        # Status labels
        self.lbl_cal_status = QLabel("--")
        self.lbl_cal_status.setStyleSheet(
            f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")
        col.addWidget(self.lbl_cal_status)

        self.lbl_cal_detail = QLabel("--")
        self.lbl_cal_detail.setStyleSheet(
            f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")
        col.addWidget(self.lbl_cal_detail)

        return box

    def _on_preset_changed(self, text):
        val = THRESHOLD_PRESETS.get(text, 5.0)
        self.sp_threshold.setValue(val)

    def _on_vel_test(self):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            return

        try:
            self._mgr.execute("clear_errors")
            axis.controller.config.control_mode = 2   # VELOCITY
            axis.controller.config.input_mode = 1     # PASSTHROUGH
            axis.controller.input_vel = 0.0
            axis.requested_state = 8                   # CLOSED_LOOP
            axis.controller.input_vel = 2.0
        except Exception as e:
            _colored(self.lbl_cal_status, f"Vel test entry failed: {e}", CLR_ERR)
            log.error("Anticogging: vel test entry failed: %s", e)
            return

        self._state = _VEL_TEST
        self._t0 = time.monotonic()
        _colored(self.lbl_cal_status, "Velocity test: commanding 2 t/s...", CLR_CAL)
        _colored(self.lbl_cal_detail, "", CLR_MUTED)
        self._update_buttons()
        log.info("Anticogging: velocity smoke test started")

    def _on_start_calibration(self):
        axis = self._mgr.get_axis(self._ax_idx())
        odrv = self._mgr.odrv
        if axis is None or odrv is None:
            return

        threshold = self.sp_threshold.value()

        try:
            # Disable stale anticogging map (fights itself during calibration)
            axis.controller.config.anticogging.anticogging_enabled = False

            # Allow reasonable regen current
            odrv.config.dc_max_negative_current = -5.0

            # Set calibration thresholds
            axis.controller.config.anticogging.calib_vel_threshold = threshold
            axis.controller.config.anticogging.calib_pos_threshold = threshold

            # Clear errors and enter position control
            self._mgr.execute("clear_errors")
            axis.controller.config.control_mode = 3   # POSITION
            axis.controller.config.input_mode = 1     # PASSTHROUGH
            axis.requested_state = 8                   # CLOSED_LOOP
        except Exception as e:
            _colored(self.lbl_cal_status, f"Setup failed: {e}", CLR_ERR)
            log.error("Anticogging: calibration setup failed: %s", e)
            return

        # Start the anticogging calibration
        try:
            axis.controller.start_anticogging_calibration()
        except Exception as e:
            _colored(self.lbl_cal_status,
                     f"start_anticogging_calibration() failed: {e}", CLR_ERR)
            log.error("Anticogging: start failed: %s", e)
            try:
                axis.requested_state = 1  # IDLE
            except Exception:
                pass
            return

        self._state = _CALIBRATING
        self._t0 = time.monotonic()
        self._prev_idx = -1
        self.progress.setValue(0)
        _colored(self.lbl_cal_status, "Anticogging sweep started...", CLR_CAL)
        _colored(self.lbl_cal_detail,
                 f"Threshold: {threshold}  Map: 0/{COGGING_MAP_SIZE}", CLR_MUTED)
        self._update_buttons()
        log.info("Anticogging: calibration started (threshold=%.1f)", threshold)

    def _on_abort(self):
        """Abort any running operation -- go IDLE."""
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is not None:
            try:
                axis.controller.input_vel = 0.0
            except Exception:
                pass
            try:
                axis.requested_state = 1  # IDLE
            except Exception:
                pass

        old_state = self._state
        self._state = _IDLE
        _colored(self.lbl_cal_status, "Aborted", CLR_WARN)
        self._update_buttons()
        log.info("Anticogging: aborted (was state %d)", old_state)

    # ── 3. Quality Test ──────────────────────────────────────────────────────

    def _build_quality_group(self):
        box = QGroupBox("3. Quality Test")
        col = QVBoxLayout(box)

        row = QHBoxLayout()
        self.btn_quality = QPushButton("Run Quality Test")
        self.btn_quality.setStyleSheet(f"color: {CLR_CAL};")
        self.btn_quality.setEnabled(False)
        self.btn_quality.clicked.connect(self._on_quality_test)
        row.addWidget(self.btn_quality)
        row.addStretch()
        col.addLayout(row)

        form = QFormLayout()
        form.setSpacing(4)
        self.lbl_q_on = QLabel("--")
        self.lbl_q_off = QLabel("--")
        self.lbl_q_result = QLabel("--")
        for lbl in (self.lbl_q_on, self.lbl_q_off, self.lbl_q_result):
            lbl.setStyleSheet(
                f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")
        form.addRow("Anticogging ON:", self.lbl_q_on)
        form.addRow("Anticogging OFF:", self.lbl_q_off)
        form.addRow("Result:", self.lbl_q_result)
        col.addLayout(form)

        return box

    def _on_quality_test(self):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            return

        try:
            self._mgr.execute("clear_errors")
            axis.controller.config.control_mode = 3   # POSITION
            axis.controller.config.input_mode = 1     # PASSTHROUGH
            axis.requested_state = 8                   # CLOSED_LOOP

            # Hold current position
            pos = axis.encoder.pos_estimate
            axis.controller.input_pos = pos

            # Enable anticogging for ON phase
            axis.controller.config.anticogging.anticogging_enabled = True
        except Exception as e:
            _colored(self.lbl_q_result, f"Setup failed: {e}", CLR_ERR)
            log.error("Anticogging: quality test setup failed: %s", e)
            return

        self._state = _QUALITY_ON
        self._t0 = time.monotonic()
        self._quality_samples = []
        self._quality_on_result = None
        _colored(self.lbl_q_on, "Sampling Iq with anticogging ON...", CLR_CAL)
        _colored(self.lbl_q_off, "--", CLR_MUTED)
        _colored(self.lbl_q_result, "Running...", CLR_CAL)
        self._update_buttons()
        log.info("Anticogging: quality test started (ON phase)")

    # ── 4. Save ──────────────────────────────────────────────────────────────

    def _build_save_group(self):
        box = QGroupBox("4. Save")
        col = QVBoxLayout(box)

        row = QHBoxLayout()
        self.btn_save = QPushButton("Save Map + Reboot")
        self.btn_save.setStyleSheet(f"color: {CLR_INFO};")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._on_save)
        row.addWidget(self.btn_save)

        self.lbl_save_status = QLabel("--")
        self.lbl_save_status.setStyleSheet(
            f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")
        row.addWidget(self.lbl_save_status, stretch=1)
        col.addLayout(row)

        note = QLabel(
            "Sets anticogging_enabled=True, pre_calibrated=True, "
            "saves config, reboots.")
        note.setStyleSheet(f"color: {CLR_MUTED}; font-size: 10px;")
        col.addWidget(note)

        return box

    def _on_save(self):
        axis = self._mgr.get_axis(self._ax_idx())
        odrv = self._mgr.odrv
        if axis is None or odrv is None:
            _colored(self.lbl_save_status, "Not connected", CLR_ERR)
            return

        try:
            axis.controller.config.anticogging.anticogging_enabled = True
            axis.controller.config.anticogging.pre_calibrated = True
            log.info("Anticogging: flags set (enabled=True, pre_calibrated=True)")
        except Exception as e:
            _colored(self.lbl_save_status, f"Flag set failed: {e}", CLR_ERR)
            return

        try:
            odrv.save_configuration()
        except Exception:
            pass  # expected -- triggers reboot

        self._state = _SAVING
        _colored(self.lbl_save_status,
                 f"Saved, rebooting... ({_REBOOT_WAIT_MS // 1000}s wait)",
                 CLR_WARN)
        self._mgr.disconnect()
        self._reboot_timer.start(_REBOOT_WAIT_MS)
        self._update_buttons()
        log.info("Anticogging: save + reboot")

    def _reconnect_after_reboot(self):
        _colored(self.lbl_save_status, "Reconnecting...", CLR_INFO)
        self._reconnect_worker = _ReconnectWorker(self._mgr)
        self._reconnect_worker.success.connect(self._on_reconnect_ok)
        self._reconnect_worker.failed.connect(self._on_reconnect_fail)
        self._reconnect_worker.start()

    def _on_reconnect_ok(self, odrv):
        self._state = _IDLE

        # Verify flags persisted
        axis = self._mgr.get_axis(self._ax_idx())
        if axis:
            acog = axis.controller.config.anticogging
            enabled = getattr(acog, "anticogging_enabled", False)
            precal = getattr(acog, "pre_calibrated", False)
            if enabled and precal:
                _colored(self.lbl_save_status,
                         "Saved + verified: enabled=True, pre_calibrated=True",
                         CLR_OK)
                log.info("Anticogging: save verified OK")
            else:
                _colored(self.lbl_save_status,
                         f"WARNING: enabled={enabled}, pre_calibrated={precal}",
                         CLR_WARN)
                log.warning("Anticogging: save verification mismatch")
        else:
            _colored(self.lbl_save_status,
                     "Reconnected (could not verify)", CLR_WARN)

        self._update_buttons()
        self.reconnected.emit()

    def _on_reconnect_fail(self, msg):
        self._state = _IDLE
        _colored(self.lbl_save_status, f"Reconnect failed: {msg}", CLR_ERR)
        log.error("Anticogging: reconnect failed: %s", msg)
        self._update_buttons()

    # ── Live status ──────────────────────────────────────────────────────────

    def _build_live_status(self):
        box = QGroupBox("Live Status")
        form = QFormLayout(box)
        form.setSpacing(4)

        self.lbl_acog_enabled = QLabel("--")
        self.lbl_acog_precal = QLabel("--")
        self.lbl_acog_index = QLabel("--")
        self.lbl_acog_thresholds = QLabel("--")
        for lbl in (self.lbl_acog_enabled, self.lbl_acog_precal,
                    self.lbl_acog_index, self.lbl_acog_thresholds):
            lbl.setStyleSheet(
                f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")

        form.addRow("anticogging_enabled", self.lbl_acog_enabled)
        form.addRow("pre_calibrated", self.lbl_acog_precal)
        form.addRow("index", self.lbl_acog_index)
        form.addRow("thresholds", self.lbl_acog_thresholds)

        return box

    # ── Button state management ──────────────────────────────────────────────

    def _update_buttons(self):
        busy = self._state != _IDLE
        connected = self._mgr.connected

        self.btn_prereq.setEnabled(connected and not busy)
        self.btn_vel_test.setEnabled(
            connected and not busy and self._step_done >= 1)
        self.btn_start_cal.setEnabled(
            connected and not busy and self._step_done >= 2)
        self.btn_quality.setEnabled(
            connected and not busy and self._step_done >= 3)
        self.btn_save.setEnabled(
            connected and not busy and self._step_done >= 3)
        self.btn_abort.setEnabled(
            busy and self._state in (_VEL_TEST, _CALIBRATING))

    # ── Poll update (called by MainWindow every 100ms) ───────────────────────

    def poll_update(self):
        """Called from MainWindow's poll timer."""
        # Always update buttons on connection state changes
        if not self._mgr.connected and self._state == _IDLE:
            self._update_buttons()
            for lbl in (self.lbl_acog_enabled, self.lbl_acog_precal,
                        self.lbl_acog_index, self.lbl_acog_thresholds):
                _colored(lbl, "--", CLR_MUTED)
            return

        # State machine polling
        if self._state == _VEL_TEST:
            self._poll_vel_test()
        elif self._state == _CALIBRATING:
            self._poll_calibration()
        elif self._state == _QUALITY_ON:
            self._poll_quality_on()
        elif self._state == _QUALITY_OFF:
            self._poll_quality_off()

        # Live status (always when connected)
        if self._mgr.connected:
            self._poll_live_status()

    def _poll_vel_test(self):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            self._abort_to_idle("Connection lost during velocity test")
            return

        elapsed = time.monotonic() - self._t0
        try:
            vel = axis.encoder.vel_estimate
            iq = axis.motor.current_control.Iq_measured
            err = axis.error or axis.motor.error or axis.encoder.error
        except Exception:
            self._abort_to_idle("USB error during velocity test")
            return

        _colored(self.lbl_cal_detail,
                 f"t={elapsed:.1f}s  vel={vel:+.4f} t/s  Iq={iq:+.3f} A",
                 CLR_CAL)

        if err:
            self._vel_test_stop(axis)
            self._state = _IDLE
            _colored(self.lbl_cal_status,
                     "Velocity test FAILED -- errors during test", CLR_ERR)
            log.error("Anticogging: vel test failed -- errors")
            self._update_buttons()
            return

        if abs(vel) > 0.3:
            # Success
            self._vel_test_stop(axis)
            self._state = _IDLE
            self._step_done = max(self._step_done, 2)
            _colored(self.lbl_cal_status,
                     "Velocity test PASSED -- motor moves OK", CLR_OK)
            _colored(self.lbl_cal_detail,
                     f"vel={vel:+.4f} t/s at t={elapsed:.1f}s", CLR_OK)
            log.info("Anticogging: velocity test passed (vel=%.4f)", vel)
            self._update_buttons()
            return

        if elapsed > _VEL_TEST_TIMEOUT:
            self._vel_test_stop(axis)
            self._state = _IDLE
            _colored(self.lbl_cal_status,
                     f"Velocity test FAILED -- no movement in "
                     f"{_VEL_TEST_TIMEOUT:.0f}s", CLR_ERR)
            log.error("Anticogging: vel test timeout")
            self._update_buttons()

    def _vel_test_stop(self, axis):
        """Stop motor after velocity test."""
        try:
            axis.controller.input_vel = 0.0
        except Exception:
            pass
        try:
            axis.requested_state = 1  # IDLE
        except Exception:
            pass

    def _poll_calibration(self):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            self._abort_to_idle("Connection lost during calibration")
            return

        elapsed = time.monotonic() - self._t0
        try:
            acog = axis.controller.config.anticogging
            idx = acog.index
            vel = axis.encoder.vel_estimate
            iq = axis.motor.current_control.Iq_measured
            state = axis.current_state
            err = axis.error or axis.motor.error or axis.encoder.error
        except Exception:
            self._abort_to_idle("USB error during calibration")
            return

        self.progress.setValue(idx)
        _colored(self.lbl_cal_detail,
                 f"idx={idx}/{COGGING_MAP_SIZE}  vel={vel:+.4f}  "
                 f"Iq={iq:+.3f}  t={elapsed:.0f}s", CLR_CAL)

        # Log every 100 indices
        if idx != self._prev_idx:
            if idx % 100 == 0 or idx == COGGING_MAP_SIZE:
                log.info("Anticogging: sweep idx=%d vel=%.4f Iq=%.3f t=%.0fs",
                         idx, vel, iq, elapsed)
            self._prev_idx = idx

        if err:
            try:
                axis.requested_state = 1
            except Exception:
                pass
            self._state = _IDLE
            _colored(self.lbl_cal_status,
                     "Calibration FAILED -- errors during sweep", CLR_ERR)
            log.error("Anticogging: calibration error during sweep")
            self._update_buttons()
            return

        # Axis returns to IDLE when sweep is done (ignore first 3s of startup)
        if state == 1 and elapsed > 3.0:
            try:
                axis.controller.remove_anticogging_bias()
                log.info("Anticogging: remove_anticogging_bias() OK")
            except Exception as e:
                log.warning("Anticogging: remove_anticogging_bias(): %s", e)

            self._state = _IDLE
            self._step_done = max(self._step_done, 3)
            self.progress.setValue(COGGING_MAP_SIZE)
            _colored(self.lbl_cal_status,
                     f"Calibration COMPLETE -- {elapsed:.0f}s, {idx} entries",
                     CLR_OK)
            _colored(self.lbl_cal_detail,
                     "Ready for quality test or save", CLR_OK)
            log.info("Anticogging: calibration complete (%.0fs, idx=%d)",
                     elapsed, idx)
            self._update_buttons()
            return

        if elapsed > _SWEEP_TIMEOUT:
            try:
                axis.requested_state = 1
            except Exception:
                pass
            self._state = _IDLE
            _colored(self.lbl_cal_status,
                     "Calibration TIMEOUT -- 10 min exceeded", CLR_ERR)
            log.error("Anticogging: calibration timeout")
            self._update_buttons()

    def _poll_quality_on(self):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            self._abort_to_idle("Connection lost during quality test")
            return

        elapsed = time.monotonic() - self._t0
        try:
            iq = abs(axis.motor.current_control.Iq_measured)
        except Exception:
            self._abort_to_idle("USB error during quality test")
            return

        self._quality_samples.append(iq)
        n = len(self._quality_samples)
        _colored(self.lbl_q_on,
                 f"Sampling... n={n}  latest Iq={iq:.4f} A", CLR_CAL)

        if elapsed >= _QUALITY_SAMPLE_SEC:
            # ON phase done -- compute stats
            mean_on = statistics.mean(self._quality_samples)
            std_on = (statistics.stdev(self._quality_samples)
                      if n > 1 else 0.0)
            self._quality_on_result = (mean_on, std_on, n)
            _colored(self.lbl_q_on,
                     f"mean={mean_on:.4f} A  stdev={std_on:.4f} A  (n={n})",
                     CLR_OK)

            # Switch to OFF phase
            try:
                axis.controller.config.anticogging.anticogging_enabled = False
            except Exception:
                pass

            self._state = _QUALITY_OFF
            self._t0 = time.monotonic()
            self._quality_samples = []
            _colored(self.lbl_q_off,
                     "Sampling Iq with anticogging OFF...", CLR_CAL)
            log.info("Anticogging: quality ON done (mean=%.4f std=%.4f n=%d)",
                     mean_on, std_on, n)

    def _poll_quality_off(self):
        axis = self._mgr.get_axis(self._ax_idx())
        if axis is None:
            self._abort_to_idle("Connection lost during quality test")
            return

        elapsed = time.monotonic() - self._t0
        try:
            iq = abs(axis.motor.current_control.Iq_measured)
        except Exception:
            self._abort_to_idle("USB error during quality test")
            return

        self._quality_samples.append(iq)
        n = len(self._quality_samples)
        _colored(self.lbl_q_off,
                 f"Sampling... n={n}  latest Iq={iq:.4f} A", CLR_CAL)

        if elapsed >= _QUALITY_SAMPLE_SEC:
            mean_off = statistics.mean(self._quality_samples)
            std_off = (statistics.stdev(self._quality_samples)
                       if n > 1 else 0.0)
            _colored(self.lbl_q_off,
                     f"mean={mean_off:.4f} A  stdev={std_off:.4f} A  (n={n})",
                     CLR_OK)

            # Restore anticogging ON and go idle
            try:
                axis.controller.config.anticogging.anticogging_enabled = True
                axis.requested_state = 1  # IDLE
            except Exception:
                pass

            # Compute result
            mean_on, std_on, n_on = self._quality_on_result
            if std_off > 0 and std_on < std_off * 0.8:
                reduction = (1 - std_on / std_off) * 100
                _colored(self.lbl_q_result,
                         f"Map VALID -- Iq stdev reduced by {reduction:.0f}%",
                         CLR_OK)
                log.info("Anticogging: quality VALID (%.0f%% reduction)",
                         reduction)
            elif std_on <= std_off:
                _colored(self.lbl_q_result,
                         "Marginal effect -- stdev similar ON vs OFF",
                         CLR_WARN)
                log.info("Anticogging: quality marginal")
            else:
                _colored(self.lbl_q_result,
                         "Map may be corrupt -- ON increased variance",
                         CLR_ERR)
                log.warning("Anticogging: quality FAILED -- ON worse than OFF")

            self._state = _IDLE
            self._step_done = max(self._step_done, 4)
            self._update_buttons()
            log.info("Anticogging: quality OFF done (mean=%.4f std=%.4f n=%d)",
                     mean_off, std_off, n)

    def _poll_live_status(self):
        ax = self._ax_idx()
        prefix = f"axis{ax}.controller.config.anticogging"

        enabled = self._mgr.safe_get(f"{prefix}.anticogging_enabled")
        precal = self._mgr.safe_get(f"{prefix}.pre_calibrated")
        idx = self._mgr.safe_get(f"{prefix}.index")
        vel_th = self._mgr.safe_get(f"{prefix}.calib_vel_threshold")
        pos_th = self._mgr.safe_get(f"{prefix}.calib_pos_threshold")

        _colored(self.lbl_acog_enabled,
                 str(enabled), CLR_OK if enabled else CLR_MUTED)
        _colored(self.lbl_acog_precal,
                 str(precal), CLR_OK if precal else CLR_MUTED)
        _colored(self.lbl_acog_index, str(idx), CLR_OK)
        if vel_th is not None and pos_th is not None:
            _colored(self.lbl_acog_thresholds,
                     f"vel={vel_th}  pos={pos_th}", CLR_OK)

    def _abort_to_idle(self, msg):
        """Emergency transition to IDLE with error message."""
        self._state = _IDLE
        _colored(self.lbl_cal_status, msg, CLR_ERR)
        log.error("Anticogging: %s", msg)
        self._update_buttons()
