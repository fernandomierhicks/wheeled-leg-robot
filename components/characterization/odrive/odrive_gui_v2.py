# -*- coding: utf-8 -*-
"""odrive_gui_v2.py — ODrive 3.6 tuning GUI (PySide6)

Tabs:
  1. Motor Setup   — configure, calibrate, verify
  2. Motor Control — drive + gains + anticogging + plot  [stub]
  3. Terminal      — REPL + command-inbox bridge

Usage:
    python odrive_gui_v2.py

Requires: pip install PySide6 odrive
"""

import math
import sys
import time
import datetime
from collections import deque
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QObject, QPointF, QMargins
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget,
    QFrame, QTextEdit, QLineEdit, QRadioButton, QButtonGroup, QScrollArea,
    QSizePolicy, QSplitter, QProgressBar,
    QTableWidget, QTableWidgetItem,
)
from PySide6.QtGui import QColor, QFont, QPen, QPainter

try:
    from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
    _HAS_CHARTS = True
except ImportError:
    _HAS_CHARTS = False

import odrive
from odrive.enums import (
    AXIS_STATE_IDLE,
    AXIS_STATE_FULL_CALIBRATION_SEQUENCE,
    AXIS_STATE_MOTOR_CALIBRATION,
    AXIS_STATE_ENCODER_OFFSET_CALIBRATION,
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    CONTROL_MODE_POSITION_CONTROL,
    CONTROL_MODE_VELOCITY_CONTROL,
    CONTROL_MODE_TORQUE_CONTROL,
    INPUT_MODE_PASSTHROUGH,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
CMD_INBOX = _HERE / "odrive_cmd_inbox.txt"
CMD_LOG   = _HERE / "odrive_cmd_log.txt"

# ── Constants ──────────────────────────────────────────────────────────────────
MOTOR_TYPE_NAMES  = ["High Current (0)", "Gimbal (2)", "ACIM (3)"]
MOTOR_TYPE_VALUES = [0, 2, 3]

ENCODER_TYPE_NAMES  = ["SPI Absolute AMS (257)", "Incremental (0)"]
ENCODER_TYPE_VALUES = [257, 0]

AXIS_STATES = {
    0: "Undefined", 1: "Idle", 2: "Startup Sequence",
    3: "Full Calibration", 4: "Motor Calibration",
    6: "Encoder Index Search", 7: "Encoder Offset Calibration",
    8: "Closed Loop", 9: "Lockin Spin", 10: "Encoder Dir Find", 11: "Homing",
}

COGGING_MAP_SIZE = 3600   # fixed in ODrive 0.5.6 firmware (controller.cpp)

AXIS_ERRORS = {
    # Source: odrive.enums — fw 0.5.6 (verified via installed odrive Python package)
    0x000001:  "INVALID_STATE",
    0x000800:  "WATCHDOG_TIMER_EXPIRED",
    0x001000:  "MIN_ENDSTOP_PRESSED",
    0x002000:  "MAX_ENDSTOP_PRESSED",
    0x004000:  "ESTOP_REQUESTED",
    0x020000:  "HOMING_WITHOUT_ENDSTOP",
    0x040000:  "OVER_TEMP",
    0x080000:  "UNKNOWN_POSITION",
}

AXIS_ERROR_HINTS = {
    0x000001:  "Requested state is invalid from the current state. Clear errors and ensure axis is Idle before commanding.",
    0x000800:  "Watchdog timer expired — no keepalive received in time.",
    0x001000:  "Min endstop triggered.",
    0x002000:  "Max endstop triggered.",
    0x004000:  "Emergency stop was requested.",
    0x020000:  "Homing attempted without endstop configured.",
    0x040000:  "Axis over-temperature. Let ODrive cool. Check airflow and FET thermistor limits.",
    0x080000:  "Position unknown — absolute encoder not ready or not configured. Check encoder is_ready.",
}

MOTOR_ERRORS = {
    # Source: odrive.enums — fw 0.5.6 (verified via installed odrive Python package)
    0x00000001: "PHASE_RESISTANCE_OUT_OF_RANGE",
    0x00000002: "PHASE_INDUCTANCE_OUT_OF_RANGE",
    0x00000008: "DRV_FAULT",
    0x00000010: "CONTROL_DEADLINE_MISSED",
    0x00000080: "MODULATION_MAGNITUDE",
    0x00000400: "CURRENT_SENSE_SATURATION",
    0x00001000: "CURRENT_LIMIT_VIOLATION",
    0x00010000: "MODULATION_IS_NAN",
    0x00020000: "MOTOR_THERMISTOR_OVER_TEMP",
    0x00040000: "FET_THERMISTOR_OVER_TEMP",
    0x00080000: "TIMER_UPDATE_MISSED",
    0x00100000: "CURRENT_MEASUREMENT_UNAVAILABLE",
    0x00200000: "CONTROLLER_FAILED",
    0x00400000: "I_BUS_OUT_OF_RANGE",
    0x00800000: "BRAKE_RESISTOR_DISARMED",
    0x01000000: "SYSTEM_LEVEL",
    0x02000000: "BAD_TIMING",
    0x04000000: "UNKNOWN_PHASE_ESTIMATE",
    0x08000000: "UNKNOWN_PHASE_VEL",
    0x10000000: "UNKNOWN_TORQUE",
    0x20000000: "UNKNOWN_CURRENT_COMMAND",
    0x40000000: "UNKNOWN_CURRENT_MEASUREMENT",
    0x80000000: "UNKNOWN_VBUS_VOLTAGE",
}

MOTOR_ERROR_HINTS = {
    0x00000001: "Phase resistance measured out of range during calibration.\n"
                "Fix: Check motor wiring for shorts/opens. Increase resistance_calib_max_voltage "
                "(try 4–8 V). Verify pole pairs are correct.",
    0x00000002: "Phase inductance measured out of range during calibration.\n"
                "Fix: Check motor wiring. May need to increase calibration current for high-inductance motors.",
    0x00000008: "Gate driver chip (DRV8301/DRV8323) asserted FAULT.\n"
                "Causes: overcurrent trip, short circuit, gate driver undervoltage, overtemperature.\n"
                "Fix: Power off immediately. Check motor wiring for phase-to-phase shorts. "
                "Let ODrive cool. Check bus voltage is stable and within spec. Reduce current_lim.",
    0x00000010: "Control loop deadline missed — MCU could not complete the control step in time.\n"
                "Fix: Reboot ODrive. If persistent, may indicate firmware or hardware issue.",
    0x00000080: "PWM modulation magnitude too high — back-EMF too high for bus voltage at this speed.\n"
                "Fix: Reduce vel_limit. Increase bus voltage. Reduce kV or increase bus voltage headroom.",
    0x00000400: "Current sense ADC saturated — actual current exceeded measurement range.\n"
                "Fix: Reduce load. Check for motor wiring shorts. Reduce current_lim.",
    0x00001000: "Motor current exceeded current_lim while running.\n"
                "Fix: Reduce velocity setpoint or load. Increase current_lim if motor can handle it. "
                "Check for mechanical binding or stall.",
    0x00010000: "Modulation value became NaN — numerical instability in the controller.\n"
                "Fix: Reduce gains (vel_gain, vel_integrator_gain). Clear errors and recalibrate.",
    0x00020000: "Motor thermistor over-temperature.\n"
                "Fix: Let motor cool. Check thermistor wiring. Reduce current_lim.",
    0x00040000: "FET thermistor over-temperature (ODrive board too hot).\n"
                "Fix: Let ODrive cool. Improve airflow. Reduce current_lim.",
    0x00080000: "Timer update missed — control loop timing slip.\n"
                "Fix: Reboot ODrive.",
    0x00100000: "Current measurement unavailable — ADC data not ready when needed.\n"
                "Fix: Reboot ODrive. Check for power supply noise.",
    0x00200000: "Motor controller subsystem failed. See controller error register.\n"
                "Fix: Check controller error for root cause (e.g. OVERSPEED, SPINOUT_DETECTED).",
    0x00400000: "I_bus (DC bus current) measurement out of range.\n"
                "Fix: Check bus current draw. Reduce load or current_lim.",
    0x00800000: "Brake resistor disarmed — motor-level flag for brake not enabled.\n"
                "Fix: Set odrv0.config.enable_brake_resistor = True and save_configuration().",
    0x01000000: "System-level fault — cascading error from another subsystem.\n"
                "Most common cause: brake resistor not enabled (enable_brake_resistor = False).\n"
                "Fix: Enable brake resistor via Flash, clear errors, recalibrate.",
    0x02000000: "Bad control loop timing — phase estimates computed at wrong time.\n"
                "Fix: Reboot ODrive.",
    0x04000000: "Phase angle estimate unavailable — encoder not ready or not calibrated.\n"
                "Fix: Ensure encoder is configured, calibrated, and is_ready = True before closed loop.",
    0x08000000: "Phase velocity estimate unavailable.\n"
                "Fix: Check encoder is_ready. Ensure encoder config is saved and pre_calibrated = True.",
    0x10000000: "Torque estimate unavailable — gains or motor constants not set.\n"
                "Fix: Run full calibration. Ensure torque_constant and pole_pairs are set.",
    0x20000000: "Current command unavailable — controller could not produce a valid current setpoint.\n"
                "Fix: Check controller mode and input_mode are set correctly.",
    0x40000000: "Current measurement unavailable at the time it was needed.\n"
                "Fix: Reboot ODrive. Check for power supply noise.",
    0x80000000: "Vbus voltage measurement unavailable — cannot compute voltage commands.\n"
                "Fix: Check power supply. Enable brake resistor to prevent vbus spikes.",
}

CONTROLLER_ERRORS = {
    # Source: odrive.enums — fw 0.5.6 (verified via installed odrive Python package)
    0x01: "OVERSPEED",
    0x02: "INVALID_INPUT_MODE",
    0x04: "UNSTABLE_GAIN",
    0x08: "INVALID_MIRROR_AXIS",
    0x10: "INVALID_LOAD_ENCODER",
    0x20: "INVALID_ESTIMATE",
    0x40: "INVALID_CIRCULAR_RANGE",
    0x80: "SPINOUT_DETECTED",
}

CONTROLLER_ERROR_HINTS = {
    0x01: "Motor exceeded vel_limit. Reduce velocity setpoint or increase vel_limit.",
    0x02: "Invalid input_mode for the current control_mode.",
    0x04: "Control gains are unstable. Reduce vel_gain or vel_integrator_gain.",
    0x08: "Invalid mirror axis configuration.",
    0x10: "Invalid load encoder axis configuration.",
    0x20: "Position/velocity estimate is invalid. Check encoder is_ready and calibration.",
    0x40: "Invalid circular range configuration.",
    0x80: "Spinout detected — electrical and mechanical power disagree. "
          "Check motor wiring direction, encoder direction, or reduce spinout thresholds.",
}

ENCODER_ERRORS = {
    0x01:  "UNSTABLE_GAIN",
    0x02:  "CPR_POLEPAIRS_MISMATCH",
    0x04:  "NO_RESPONSE",
    0x08:  "UNSUPPORTED_ENCODER_MODE",
    0x10:  "ILLEGAL_HALL_STATE",
    0x20:  "INDEX_NOT_FOUND_YET",
    0x40:  "ABS_SPI_TIMEOUT",
    0x80:  "ABS_SPI_COM_FAIL",
    0x100: "ABS_SPI_NOT_READY",
    0x200: "HALL_NOT_CALIBRATED_YET",
}

ENCODER_ERROR_HINTS = {
    0x01:  "Encoder gain is unstable — position estimate diverging.\n"
           "Fix: Check motor wiring. Verify CPR setting. Check encoder power supply.",
    0x02:  "CPR does not match pole pairs — encoder and motor settings are inconsistent.\n"
           "Fix: Verify CPR = encoder resolution (e.g. 8192 for AS5047). "
           "Verify pole_pairs matches your motor.",
    0x04:  "No response from encoder — SPI/communication failure.\n"
           "Fix: Check encoder wiring (SPI: SCK, MISO, MOSI, CS). Check CS GPIO pin setting. "
           "Verify 3.3 V power to encoder.",
    0x08:  "Encoder mode not supported by this firmware version.",
    0x10:  "Illegal Hall sensor state (only for Hall-effect encoders).",
    0x20:  "Index pulse not found yet (incremental encoders with index).\n"
           "Fix: Perform encoder index search first, or use an absolute encoder.",
    0x40:  "SPI communication timed out waiting for ABS encoder response.\n"
           "Fix: Check SPI wiring. Verify CS GPIO pin. Check encoder power supply.",
    0x80:  "SPI communication with ABS encoder failed (bad data/CRC).\n"
           "Fix: Check SPI wiring for noise. Shorten cables. Verify encoder part number.",
    0x100: "ABS encoder not ready (still initialising).\n"
           "Fix: Wait a moment after power-on before commanding motion. "
           "Check encoder power supply ramp time.",
    0x200: "Hall sensor not yet calibrated.\n"
           "Fix: Run encoder offset calibration for Hall-effect encoder. Not relevant for SPI ABS.",
}

POLL_MS   = 100
PLOT_LEN  = 300
INBOX_MS  = 500

QUICK_CMDS = {
    "Full State Dump":     "print(odrv)",
    "Firmware Version":    "(odrv.hw_version_major, odrv.fw_version_minor, odrv.fw_version_revision)",
    "Reset Errors (both)": "ax0.clear_errors(); ax1.clear_errors()",
    "Diagnose Encoder":    "(ax0.encoder.is_ready, ax0.encoder.error, ax0.encoder.shadow_count)",
}

# ── Style helpers ──────────────────────────────────────────────────────────────
CLR_BG      = "#2d2d2d"
CLR_PANEL   = "#252526"
CLR_OK      = "#64ffb4"
CLR_WARN    = "#ffa03c"
CLR_ERR     = "#ff6464"
CLR_INFO    = "#64c8ff"
CLR_CAL     = "#ffff64"
CLR_LABEL   = "#cccccc"
CLR_MUTED   = "#888888"

DARK_STYLE = f"""
QMainWindow, QWidget {{ background: {CLR_BG}; color: {CLR_LABEL}; }}
QGroupBox {{
    border: 1px solid #444; border-radius: 4px; margin-top: 10px;
    padding-top: 6px; color: {CLR_LABEL};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: #aaa; }}
QPushButton {{
    background: #3c3c3c; border: 1px solid #555; border-radius: 3px;
    padding: 4px 10px; color: {CLR_LABEL};
}}
QPushButton:hover {{ background: #4a4a4a; }}
QPushButton:disabled {{ color: #555; border-color: #444; }}
QComboBox, QSpinBox, QDoubleSpinBox {{
    background: #3c3c3c; border: 1px solid #555; border-radius: 3px;
    padding: 2px 4px; color: {CLR_LABEL};
}}
QLabel {{ color: {CLR_LABEL}; }}
QTextEdit {{ background: {CLR_PANEL}; color: {CLR_LABEL}; border: 1px solid #444; }}
QRadioButton {{ color: {CLR_LABEL}; }}
QCheckBox {{ color: {CLR_LABEL}; }}
QTabWidget::pane {{ border: 1px solid #444; }}
QTabBar::tab {{
    background: #3c3c3c; border: 1px solid #444; border-bottom: none;
    padding: 5px 14px; color: #aaa;
}}
QTabBar::tab:selected {{ background: {CLR_BG}; color: {CLR_LABEL}; }}
"""


def _status_label(text="—"):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {CLR_OK};")
    return lbl


def _colored(lbl: QLabel, text: str, color: str):
    lbl.setText(text)
    lbl.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {color};")


def _dspin(lo, hi, val=0.0, dec=4, step=0.01, suffix="", width=120):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(f" {suffix}")
    s.setFixedWidth(width)
    return s


# ── Connect worker ─────────────────────────────────────────────────────────────
class ConnectWorker(QThread):
    success = Signal(object)
    failed  = Signal(str)

    def run(self):
        try:
            odrv = odrive.find_any()
            self.success.emit(odrv)
        except Exception as e:
            self.failed.emit(str(e))


class _PreflightWorker(QThread):
    """Runs anticogging pre-flight smoke test off the main thread."""
    log = Signal(str, str)  # (message, color)

    def __init__(self, odrv, axis_idx):
        super().__init__()
        self._odrv = odrv
        self._axis_idx = axis_idx

    def run(self):
        axis = self._odrv.axis0 if self._axis_idx == 0 else self._odrv.axis1
        try:
            saved_mode  = axis.controller.config.control_mode
            saved_input = axis.controller.config.input_mode

            axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            axis.controller.config.input_mode   = INPUT_MODE_PASSTHROUGH
            axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
            time.sleep(0.3)

            if axis.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                self.log.emit(f"  FAIL  could not enter closed loop (state={axis.current_state})", CLR_ERR)
                axis.requested_state = AXIS_STATE_IDLE
                return

            axis.controller.input_vel = 2.0
            t0 = time.monotonic()
            moved = False
            while time.monotonic() - t0 < 2.0:
                if abs(getattr(axis.encoder, "vel_estimate", 0.0)) > 0.3:
                    moved = True
                    break
                time.sleep(0.05)

            axis.controller.input_vel = 0.0
            time.sleep(0.2)
            axis.requested_state = AXIS_STATE_IDLE
            time.sleep(0.3)

            axis.controller.config.control_mode = saved_mode
            axis.controller.config.input_mode   = saved_input

            if moved:
                vel = getattr(axis.encoder, "vel_estimate", 0.0)
                iq  = getattr(axis.motor.current_control, "Iq_measured", 0.0)
                self.log.emit(f"  PASS  motor moved (vel≈{vel:+.2f} t/s  Iq≈{iq:+.2f} A) — safe to calibrate", CLR_OK)
            else:
                self.log.emit("  FAIL  motor did not reach 0.3 t/s in 2 s — check gains/power before calibrating", CLR_ERR)
        except Exception as e:
            self.log.emit(f"Pre-flight error: {e}", CLR_ERR)
            try:
                axis.controller.input_vel = 0.0
                axis.requested_state = AXIS_STATE_IDLE
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Motor Setup
# ══════════════════════════════════════════════════════════════════════════════
class MotorSetupTab(QWidget):
    """Tab 1: configure motor + encoder parameters, calibrate, verify."""

    log_message = Signal(str, str)   # (text, colour)

    # States used during calibration polling
    _CAL_IDLE    = 0
    _CAL_RUNNING = 1
    _CAL_DONE    = 2
    _CAL_FAIL    = 3

    _REBOOT_WAIT_MS = 4000   # ms to wait after reboot before reconnecting

    def __init__(self, get_odrv, get_axis_idx):
        """
        get_odrv()      → current odrv object or None
        get_axis_idx()  → 0 or 1
        """
        super().__init__()
        self._get_odrv     = get_odrv
        self._get_axis_idx = get_axis_idx
        self._cal_state    = self._CAL_IDLE
        self._expected_params: dict = {}   # populated before reboot for verify
        self._reboot_timer = QTimer(self)
        self._reboot_timer.setSingleShot(True)
        self._reboot_timer.timeout.connect(self._reconnect_after_reboot)
        self._reconnect_worker: ConnectWorker | None = None
        self._build()

    # ── build UI ───────────────────────────────────────────────────────────────
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

        # ── top two-column row ─────────────────────────────────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        top_row.addWidget(self._build_motor_group())
        top_row.addWidget(self._build_encoder_group())
        root.addLayout(top_row)

        # ── safety status ──────────────────────────────────────────────────────
        root.addWidget(self._build_safety_group())

        # ── action buttons ─────────────────────────────────────────────────────
        root.addWidget(self._build_action_buttons())

        # ── calibration progress ───────────────────────────────────────────────
        root.addWidget(self._build_cal_progress())

        # ── live status ───────────────────────────────────────────────────────
        root.addWidget(self._build_live_status())

        root.addStretch()

        # ── wire encoder type toggle ──────────────────────────────────────────
        self.enc_type_combo.currentIndexChanged.connect(self._on_enc_type_changed)
        self._on_enc_type_changed(0)

    # ── sub-builders ──────────────────────────────────────────────────────────
    def _build_motor_group(self):
        box = QGroupBox("Motor Parameters")
        form = QFormLayout(box)
        form.setSpacing(8)

        self.motor_type  = QComboBox()
        self.motor_type.addItems(MOTOR_TYPE_NAMES)

        self.pole_pairs  = QSpinBox()
        self.pole_pairs.setRange(1, 50)
        self.pole_pairs.setValue(7)

        self.current_lim    = _dspin(0, 60, 10, dec=2, step=0.5, suffix="A")
        self.cal_current    = _dspin(0, 60, 10, dec=2, step=0.5, suffix="A")
        self.vel_lim        = _dspin(0, 500, 2, dec=2, step=0.5, suffix="t/s")
        self.phase_res      = _dspin(0, 10, 0, dec=4, step=0.001, suffix="Ohm")
        self.phase_ind      = _dspin(0, 0.1, 0, dec=6, step=0.000001, suffix="H")
        self.res_calib_volt = _dspin(0, 24, 2, dec=2, step=0.5, suffix="V")
        self.torque_const   = _dspin(0, 10, 0.04, dec=5, step=0.001, suffix="Nm/A")

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
        self.enc_cs_pin.setValue(3)   # default M0=GPIO3

        self.enc_cpr = QSpinBox()
        self.enc_cpr.setRange(4, 65536)
        self.enc_cpr.setValue(8192)

        self.enc_precal_chk = QCheckBox("Pre-calibrated (read-only)")
        self.enc_precal_chk.setEnabled(False)

        form.addRow("Encoder type",   self.enc_type_combo)
        form.addRow("SPI CS GPIO pin",self.enc_cs_pin)
        form.addRow("CPR",            self.enc_cpr)
        form.addRow("",               self.enc_precal_chk)

        hint = QLabel("CPR only used for incremental encoders")
        hint.setStyleSheet(f"color: {CLR_MUTED}; font-size: 10px;")
        form.addRow("", hint)

        return box

    def _build_safety_group(self):
        box = QGroupBox("Boot Safety — Startup Motion Flags")
        row = QHBoxLayout(box)

        self.lbl_startup_safe = QLabel("Not connected")
        self.lbl_startup_safe.setStyleSheet(f"font-family: monospace; color: {CLR_MUTED};")
        row.addWidget(self.lbl_startup_safe)
        row.addStretch()

        note = QLabel("On every Flash: startup_* = False, enable_overspeed_error = False, enable_brake_resistor = True.")
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

        self.btn_read.setToolTip("Read current ODrive config into form fields")
        self.btn_flash.setToolTip(
            "Write form values to ODrive + lock all startup_* flags = False + save_configuration().\n"
            "No motion is triggered."
        )
        self.btn_cal.setToolTip(
            "Run AXIS_STATE_FULL_CALIBRATION_SEQUENCE.\n"
            "Motor will spin briefly. Encoder will be indexed."
        )
        self.btn_verify.setToolTip(
            "Reboot ODrive, reconnect, read back all written parameters\n"
            "and compare against last-flashed values."
        )

        self.btn_read.clicked.connect(self._read_config)
        self.btn_flash.clicked.connect(self._flash_config)
        self.btn_cal.clicked.connect(self._start_calibration)
        self.btn_verify.clicked.connect(self._reboot_and_verify)

        return box

    def _build_cal_progress(self):
        box = QGroupBox("Calibration Progress")
        form = QFormLayout(box)
        form.setSpacing(6)

        self.lbl_cal_state = _status_label("—")
        self.lbl_cal_msg   = _status_label("—")

        form.addRow("State",   self.lbl_cal_state)
        form.addRow("Message", self.lbl_cal_msg)

        return box

    def _build_live_status(self):
        box = QGroupBox("Live Status")
        form = QFormLayout(box)
        form.setSpacing(6)

        self.lbl_axis_state   = _status_label("—")
        self.lbl_motor_cal    = _status_label("—")
        self.lbl_enc_ready    = _status_label("—")
        self.lbl_startup_chk  = _status_label("—")
        self.lbl_acog_enabled = _status_label("—")
        self.lbl_acog_cal     = _status_label("—")

        form.addRow("Axis state",           self.lbl_axis_state)
        form.addRow("Motor calibrated",     self.lbl_motor_cal)
        form.addRow("Encoder ready",        self.lbl_enc_ready)
        form.addRow("Startup flags safe",   self.lbl_startup_chk)
        form.addRow("Anticogging enabled",  self.lbl_acog_enabled)
        form.addRow("Anticogging cal'd",    self.lbl_acog_cal)

        return box

    # ── encoder type toggle ────────────────────────────────────────────────────
    def _on_enc_type_changed(self, idx):
        is_spi = (ENCODER_TYPE_VALUES[idx] == 257)
        self.enc_cs_pin.setEnabled(is_spi)
        self.enc_cpr.setEnabled(True)

    # ── helpers ────────────────────────────────────────────────────────────────
    def _odrv(self):
        return self._get_odrv()

    def _axis(self):
        odrv = self._odrv()
        if odrv is None:
            return None
        try:
            return odrv.axis0 if self._get_axis_idx() == 0 else odrv.axis1
        except AttributeError:
            return None  # EmptyInterface after disconnect/reboot

    def _ax_idx(self):
        return self._get_axis_idx()

    def _log(self, msg, color=CLR_OK):
        self.log_message.emit(msg, color)

    def _set_buttons_enabled(self, connected: bool, calibrating: bool = False):
        self.btn_read.setEnabled(connected and not calibrating)
        self.btn_flash.setEnabled(connected and not calibrating)
        self.btn_cal.setEnabled(connected and not calibrating)
        self.btn_verify.setEnabled(connected and not calibrating)

    # ── startup flags lockout ──────────────────────────────────────────────────
    def _lock_startup_flags(self, axis):
        """Always force all startup motion flags False."""
        flags = [
            "startup_motor_calibration",
            "startup_encoder_offset_calibration",
            "startup_encoder_index_search",
            "startup_closed_loop_control",
            "startup_homing",
        ]
        for flag in flags:
            try:
                setattr(axis.config, flag, False)
            except Exception:
                pass

    def _check_startup_flags(self, axis) -> bool:
        """Returns True if all startup flags are safely False."""
        flags = [
            "startup_motor_calibration",
            "startup_encoder_offset_calibration",
            "startup_encoder_index_search",
            "startup_closed_loop_control",
            "startup_homing",
        ]
        for flag in flags:
            if getattr(axis.config, flag, False):
                return False
        return True

    # ── read config ────────────────────────────────────────────────────────────
    def _read_config(self):
        axis = self._axis()
        if axis is None:
            return
        try:
            mc = axis.motor.config
            ec = axis.encoder.config

            mt = getattr(mc, "motor_type", 0)
            idx = MOTOR_TYPE_VALUES.index(mt) if mt in MOTOR_TYPE_VALUES else 0
            self.motor_type.setCurrentIndex(idx)
            self.pole_pairs.setValue(getattr(mc, "pole_pairs", 7))
            self.current_lim.setValue(float(getattr(mc, "current_lim", 10)))
            self.cal_current.setValue(float(getattr(mc, "calibration_current", 10)))
            self.vel_lim.setValue(float(getattr(axis.controller.config, "vel_limit", 2)))
            self.phase_res.setValue(float(getattr(mc, "phase_resistance", 0)))
            self.phase_ind.setValue(float(getattr(mc, "phase_inductance", 0)))
            self.res_calib_volt.setValue(float(getattr(mc, "resistance_calib_max_voltage", 2)))
            self.torque_const.setValue(float(getattr(mc, "torque_constant", 0.04)))

            enc_mode = getattr(ec, "mode", 0)
            enc_idx = ENCODER_TYPE_VALUES.index(enc_mode) if enc_mode in ENCODER_TYPE_VALUES else 1
            self.enc_type_combo.setCurrentIndex(enc_idx)
            self.enc_cs_pin.setValue(getattr(ec, "abs_spi_cs_gpio_pin", 3))
            self.enc_cpr.setValue(getattr(ec, "cpr", 8192))
            self.enc_precal_chk.setChecked(getattr(ec, "pre_calibrated", False))

            try:
                acog    = axis.controller.config.anticogging
                enabled = getattr(acog, "anticogging_enabled", False)
                pre_cal = getattr(acog, "pre_calibrated",      False)
                ratio   = getattr(acog, "cogging_ratio",       None)
                ratio_s = f"  cogging_ratio={ratio:.4f}" if ratio is not None else ""
                self._log(
                    f"  anticogging: enabled={enabled}  pre_calibrated={pre_cal}{ratio_s}",
                    CLR_OK if (enabled and pre_cal) else CLR_MUTED,
                )
            except Exception:
                pass
            self._log(f"Ax{self._ax_idx()} config read OK", CLR_OK)
        except Exception as e:
            self._log(f"Read error: {e}", CLR_ERR)

    # ── flash config ───────────────────────────────────────────────────────────
    def _flash_config(self):
        axis = self._axis()
        odrv = self._odrv()
        if axis is None or odrv is None:
            self._log("Flash: not connected", CLR_ERR)
            return
        try:
            mc = axis.motor.config
            ec = axis.encoder.config
            cc = axis.controller.config

            # Motor
            mc.motor_type                   = MOTOR_TYPE_VALUES[self.motor_type.currentIndex()]
            mc.pole_pairs                   = self.pole_pairs.value()
            mc.current_lim                  = self.current_lim.value()
            mc.calibration_current          = self.cal_current.value()
            cc.vel_limit                    = self.vel_lim.value()
            try:
                mc.resistance_calib_max_voltage = self.res_calib_volt.value()
            except Exception:
                pass  # attribute absent on some firmware builds
            try:
                mc.torque_constant = self.torque_const.value()
            except Exception:
                pass

            # Encoder
            enc_type = ENCODER_TYPE_VALUES[self.enc_type_combo.currentIndex()]
            ec.mode  = enc_type
            ec.cpr   = self.enc_cpr.value()
            if enc_type == 257:   # SPI Absolute AMS
                ec.abs_spi_cs_gpio_pin = self.enc_cs_pin.value()

            # Safety lockout — always
            self._lock_startup_flags(axis)

            # Controller safety defaults — always written on flash
            cc.enable_overspeed_error = False

            # Brake resistor — always enable; config wipe resets this to False
            # which causes BRAKE_CURRENT_OUT_OF_RANGE + SYSTEM_LEVEL on any regen
            odrv.config.enable_brake_resistor = True

            # Anticogging — mark calibration complete on every flash
            try:
                acog = axis.controller.config.anticogging
                acog.anticogging_enabled = True
                acog.pre_calibrated      = True
                self._log(f"Ax{self._ax_idx()} anticogging marked pre_calibrated", CLR_MUTED)
            except Exception as acog_err:
                self._log(f"Anticogging flag write failed: {acog_err}", CLR_WARN)

            # Snapshot before save — ODrive reboots on save_configuration() so we
            # can't read params back after the call.
            self._expected_params = self._snapshot_params(axis)

            try:
                odrv.save_configuration()
            except Exception:
                pass  # expected — ODrive reboots immediately after saving

            self._log(f"Ax{self._ax_idx()} flashed + saved + rebooting…", CLR_OK)
            self._start_reboot_sequence()

        except Exception as e:
            self._log(f"Flash error: {e}", CLR_ERR)

    def _start_reboot_sequence(self):
        """Shared reboot + auto-reconnect sequence used by flash and save-anticogging."""
        self._set_buttons_enabled(False)
        self.log_message.emit("__DISCONNECTED__", "")
        _colored(self.lbl_cal_state, "Rebooting…", CLR_WARN)
        _colored(self.lbl_cal_msg, f"Waiting {self._REBOOT_WAIT_MS // 1000}s then reconnecting", CLR_WARN)
        self._reboot_timer.start(self._REBOOT_WAIT_MS)

    def _snapshot_params(self, axis) -> dict:
        """Snapshot key params for post-reboot verification."""
        mc = axis.motor.config
        ec = axis.encoder.config
        cc = axis.controller.config
        return {
            "motor_type":                    getattr(mc, "motor_type", None),
            "pole_pairs":                    getattr(mc, "pole_pairs", None),
            "current_lim":                   getattr(mc, "current_lim", None),
            "calibration_current":           getattr(mc, "calibration_current", None),
            "resistance_calib_max_voltage":  getattr(mc, "resistance_calib_max_voltage", None),
            "torque_constant":               getattr(mc, "torque_constant", None),
            "vel_limit":                     getattr(cc, "vel_limit", None),
            "enable_overspeed_error":        getattr(cc, "enable_overspeed_error", None),
            "enable_brake_resistor":         getattr(self._odrv().config, "enable_brake_resistor", None),
            "encoder_mode":                  getattr(ec, "mode", None),
            "abs_spi_cs_gpio_pin":           getattr(ec, "abs_spi_cs_gpio_pin", None),
            "cpr":                           getattr(ec, "cpr", None),
            "startup_motor_calibration":             getattr(axis.config, "startup_motor_calibration", None),
            "startup_encoder_offset_calibration":    getattr(axis.config, "startup_encoder_offset_calibration", None),
            "startup_encoder_index_search":          getattr(axis.config, "startup_encoder_index_search", None),
            "startup_closed_loop_control":           getattr(axis.config, "startup_closed_loop_control", None),
            "startup_homing":                        getattr(axis.config, "startup_homing", None),
        }

    # ── calibration ────────────────────────────────────────────────────────────
    def _start_calibration(self):
        axis = self._axis()
        odrv = self._odrv()
        if axis is None or odrv is None:
            return
        if axis.current_state != AXIS_STATE_IDLE:
            self._log("Axis must be Idle before calibrating", CLR_WARN)
            return

        # Write encoder config to RAM first — motor can't turn until encoder is ready.
        # No save here: save_configuration() reboots the ODrive. Config is applied in
        # RAM immediately and will be saved automatically on calibration completion.
        try:
            ec = axis.encoder.config
            enc_type = ENCODER_TYPE_VALUES[self.enc_type_combo.currentIndex()]
            ec.mode = enc_type
            ec.cpr  = self.enc_cpr.value()
            if enc_type == 257:   # SPI Absolute AMS
                ec.abs_spi_cs_gpio_pin = self.enc_cs_pin.value()
            self._log(f"Ax{self._ax_idx()} encoder config applied (mode={enc_type} cpr={ec.cpr})", CLR_INFO)
        except Exception as e:
            self._log(f"Encoder config write failed: {e}", CLR_ERR)
            return

        self._cal_state = self._CAL_RUNNING
        self._set_buttons_enabled(True, calibrating=True)
        _colored(self.lbl_cal_state, "Calibrating…", CLR_CAL)
        _colored(self.lbl_cal_msg, "AXIS_STATE_FULL_CALIBRATION_SEQUENCE started", CLR_CAL)

        try:
            axis.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
            self._log(f"Ax{self._ax_idx()} full calibration started", CLR_CAL)
        except Exception as e:
            self._cal_state = self._CAL_FAIL
            self._set_buttons_enabled(True)
            _colored(self.lbl_cal_state, "FAILED", CLR_ERR)
            _colored(self.lbl_cal_msg, str(e), CLR_ERR)
            self._log(f"Calibration start error: {e}", CLR_ERR)

    def _poll_calibration(self, axis):
        """Called from update() while _cal_state == _CAL_RUNNING."""
        state = getattr(axis, "current_state", 0)
        err   = getattr(axis, "error", 0)
        state_name = AXIS_STATES.get(state, f"State {state}")

        _colored(self.lbl_cal_state, f"Calibrating — {state_name}", CLR_CAL)

        if state == AXIS_STATE_IDLE:
            # Finished (back to idle)
            if err != 0:
                self._cal_state = self._CAL_FAIL
                desc = _decode_errors(err, AXIS_ERRORS)
                _colored(self.lbl_cal_state, "FAILED", CLR_ERR)
                _colored(self.lbl_cal_msg, f"Axis error 0x{err:04X}: {desc}", CLR_ERR)
                self._log(f"Ax{self._ax_idx()} calibration FAILED — 0x{err:04X}: {desc}", CLR_ERR)
            else:
                self._cal_state = self._CAL_DONE
                try:
                    mc = axis.motor.config
                    cc = axis.controller.config

                    # Set pre-calibrated flags
                    mc.pre_calibrated              = True
                    axis.encoder.config.pre_calibrated = True

                    # ── Post-cal velocity loop tuning ──────────────────────────
                    # Compute vel_gain from measured phase_resistance so the
                    # velocity loop is correctly scaled to this specific motor.
                    # ODrive recommended formula:
                    #   vel_gain = 0.5 * torque_constant / phase_resistance
                    # Clamp to [0.01, 2.0] so bad calibration can't blow up gains.
                    R  = mc.phase_resistance
                    kT = mc.torque_constant
                    if R and R > 0.001:
                        vel_gain = max(0.01, min(2.0, 0.5 * kT / R))
                    else:
                        vel_gain = cc.vel_gain  # leave unchanged if R not measured
                    vel_int  = 0.5 * vel_gain   # integrator at half vel_gain bandwidth

                    cc.vel_gain             = vel_gain
                    cc.vel_integrator_gain  = vel_int
                    self._log(
                        f"  vel_gain={vel_gain:.4f}  vel_integrator_gain={vel_int:.4f}"
                        f"  (R={R:.4f} Ω  kT={kT:.4f} Nm/A)",
                        CLR_INFO
                    )

                    # Raise vel_limit — default 2.0 turns/s is too tight for any
                    # real motion; overspeed error is already forced False on flash
                    # but the vel_limit also gates the torque mode vel limiter.
                    cc.vel_limit = max(cc.vel_limit, 20.0)
                    self._log(f"  vel_limit → {cc.vel_limit:.1f} turns/s", CLR_INFO)

                    # Loosen spinout thresholds (new in 0.5.6).
                    # Defaults (10 W / -10 W) fire easily at vel=0 when the
                    # integrator winds up against cogging torque.
                    try:
                        cc.spinout_electrical_power_threshold = 120.0
                        cc.spinout_mechanical_power_threshold = -120.0
                        self._log("  spinout thresholds -> +/-120 W", CLR_INFO)
                    except Exception:
                        pass  # attribute absent on some builds

                    # ── Brake resistor + regen fix ─────────────────────────────
                    # UNKNOWN_VBUS_VOLTAGE motor error on closed-loop entry is
                    # caused by dc_max_negative_current = -0.01 A (regen ~off)
                    # combined with enable_brake_resistor = False.  The motor
                    # thread's internal vbus validation fails even at vel=0.
                    # Fix: arm the brake resistor and allow real regen current.
                    try:
                        odrv = self._odrv()
                        odrv.config.enable_brake_resistor = True
                        odrv.config.dc_max_negative_current = -5.0
                        self._log("  brake resistor enabled, dc_max_negative_current -> -5.0 A", CLR_INFO)
                    except Exception as e:
                        self._log(f"  brake resistor fix warning: {e}", CLR_WARN)

                    self._odrv().save_configuration()
                except Exception as e:
                    self._log(f"Post-cal save warning: {e}", CLR_WARN)
                _colored(self.lbl_cal_state, "Done", CLR_OK)
                _colored(self.lbl_cal_msg, "pre_calibrated set, vel gains tuned, brake resistor armed, config saved", CLR_OK)
                self._log(f"Ax{self._ax_idx()} calibration complete — gains tuned + saved", CLR_OK)

            self._set_buttons_enabled(True)

    # ── reboot & verify ────────────────────────────────────────────────────────
    def _reboot_and_verify(self):
        odrv = self._odrv()
        axis = self._axis()
        if odrv is None or axis is None:
            self._log("Reboot & Verify: not connected", CLR_ERR)
            return
        try:
            # Take snapshot of what we expect to read back
            self._expected_params = self._snapshot_params(axis)
        except Exception as e:
            self._log(f"Reboot & Verify snapshot error: {e}", CLR_ERR)
            return
        self._set_buttons_enabled(False)
        _colored(self.lbl_cal_state, "Rebooting…", CLR_WARN)
        _colored(self.lbl_cal_msg, f"Waiting {self._REBOOT_WAIT_MS // 1000}s then reconnecting", CLR_WARN)
        self._log("Rebooting ODrive for verification…", CLR_WARN)

        try:
            odrv.reboot()
        except Exception:
            pass  # expected — connection drops on reboot

        # Signal MainWindow that odrv is gone
        self.log_message.emit("__DISCONNECTED__", "")

        self._reboot_timer.start(self._REBOOT_WAIT_MS)

    def _reconnect_after_reboot(self):
        _colored(self.lbl_cal_state, "Reconnecting…", CLR_INFO)
        _colored(self.lbl_cal_msg, "Searching for ODrive…", CLR_INFO)
        self._reconnect_worker = ConnectWorker()
        self._reconnect_worker.success.connect(self._on_verify_reconnect)
        self._reconnect_worker.failed.connect(self._on_verify_reconnect_fail)
        self._reconnect_worker.start()

    def _on_verify_reconnect(self, odrv):
        # Store odrv so MainWindow can grab it directly (no second find_any)
        self._reconnected_odrv = odrv
        self.log_message.emit(f"__RECONNECTED__:{id(odrv)}", "")
        self._verify_params(odrv)
        self._set_buttons_enabled(True)

    def _on_verify_reconnect_fail(self, msg):
        _colored(self.lbl_cal_state, "Reconnect FAILED", CLR_ERR)
        _colored(self.lbl_cal_msg, msg, CLR_ERR)
        self._log(f"Reconnect failed: {msg}", CLR_ERR)
        self._set_buttons_enabled(False)

    def _verify_params(self, odrv):
        axis = odrv.axis0 if self._ax_idx() == 0 else odrv.axis1
        actual = self._snapshot_params(axis)
        expected = self._expected_params

        all_pass = True
        lines = []
        for key, exp_val in expected.items():
            act_val = actual.get(key)
            # Compare with tolerance for floats
            if isinstance(exp_val, float) and isinstance(act_val, float):
                ok = abs(exp_val - act_val) < 1e-5
            else:
                ok = (exp_val == act_val)
            mark = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            lines.append(f"  {mark}  {key}: expected={exp_val}  got={act_val}")

        result_color = CLR_OK if all_pass else CLR_ERR
        result_text  = "All parameters PASS" if all_pass else "Some parameters FAILED — check log"
        _colored(self.lbl_cal_state, result_text, result_color)
        _colored(self.lbl_cal_msg, f"{sum(1 for l in lines if 'PASS' in l)}/{len(lines)} parameters match", result_color)

        self._log("─── Verify Results ───", CLR_INFO)
        for line in lines:
            color = CLR_OK if "PASS" in line else CLR_ERR
            self._log(line, color)
        self._log("─────────────────────", CLR_INFO)

    # ── poll update (called by MainWindow every 100ms) ─────────────────────────
    def update(self, odrv, axis_idx):
        axis = None
        if odrv is not None:
            try:
                axis = odrv.axis0 if axis_idx == 0 else odrv.axis1
            except AttributeError:
                pass  # EmptyInterface after disconnect/reboot

        connected = axis is not None

        if not connected:
            _colored(self.lbl_axis_state,   "—", CLR_MUTED)
            _colored(self.lbl_motor_cal,    "—", CLR_MUTED)
            _colored(self.lbl_enc_ready,    "—", CLR_MUTED)
            _colored(self.lbl_startup_chk,  "Not connected", CLR_MUTED)
            _colored(self.lbl_startup_safe, "Not connected", CLR_MUTED)
            _colored(self.lbl_acog_enabled, "—", CLR_MUTED)
            _colored(self.lbl_acog_cal,     "—", CLR_MUTED)
            self._set_buttons_enabled(False)
            return

        # Calibration polling
        if self._cal_state == self._CAL_RUNNING:
            self._poll_calibration(axis)

        # Live status
        try:
            state = getattr(axis, "current_state", 0)
            state_name = AXIS_STATES.get(state, f"State {state}")
            _colored(self.lbl_axis_state, state_name, CLR_OK)

            motor_cal = getattr(axis.motor, "is_calibrated", False)
            _colored(self.lbl_motor_cal,
                     "YES" if motor_cal else "NO",
                     CLR_OK if motor_cal else CLR_ERR)

            enc_rdy = getattr(axis.encoder, "is_ready", False)
            _colored(self.lbl_enc_ready,
                     "YES" if enc_rdy else "NO",
                     CLR_OK if enc_rdy else CLR_ERR)

            startup_ok = self._check_startup_flags(axis)
            _colored(self.lbl_startup_chk,
                     "All OFF — safe" if startup_ok else "WARNING: motion on boot enabled!",
                     CLR_OK if startup_ok else CLR_ERR)
            _colored(self.lbl_startup_safe,
                     "All startup_* flags = False" if startup_ok else "WARNING — motion flags enabled!",
                     CLR_OK if startup_ok else CLR_ERR)

            try:
                acog    = axis.controller.config.anticogging
                enabled = getattr(acog, "anticogging_enabled", False)
                pre_cal = getattr(acog, "pre_calibrated",      False)
                valid   = getattr(axis.controller, "anticogging_valid", False)
                _colored(self.lbl_acog_enabled,
                         "YES" if enabled else "NO",
                         CLR_OK if enabled else CLR_MUTED)
                if valid:
                    _colored(self.lbl_acog_cal, "Map valid", CLR_OK)
                elif pre_cal:
                    _colored(self.lbl_acog_cal, "pre_calibrated (reboot to load)", CLR_WARN)
                else:
                    _colored(self.lbl_acog_cal, "not calibrated", CLR_MUTED)
            except Exception:
                pass
        except Exception:
            pass

        if self._cal_state in (self._CAL_IDLE, self._CAL_DONE, self._CAL_FAIL):
            self._set_buttons_enabled(connected)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Motor Control
# ══════════════════════════════════════════════════════════════════════════════
CLR_IQ  = "#ff6464"
CLR_VEL = "#50a0ff"
CLR_POS = "#ffdc3c"
CLR_VQ  = "#50e040"


class MotorControlTab(QWidget):
    log_message      = Signal(str, str)
    reboot_requested = Signal()

    _ANTICOG_IDLE    = 0
    _ANTICOG_RUNNING = 1
    _ANTICOG_DONE    = 2

    def __init__(self, get_odrv, get_axis_idx):
        super().__init__()
        self._get_odrv          = get_odrv
        self._get_axis_idx      = get_axis_idx
        self._anticog_state     = self._ANTICOG_IDLE
        self._anticog_step      = 0          # 0=idle, 1=prereqs ok, 2=closed loop ok
        self._anticog_saved_gains: dict = {}
        self._anticog_pos_buf   = deque(maxlen=600)  # (time, pos) — 60 s at 100 ms poll
        self._t0                = None
        self._last_axis_idx     = -1
        self._t_buf             = deque(maxlen=PLOT_LEN)
        self._iq_buf            = deque(maxlen=PLOT_LEN)
        self._vel_buf           = deque(maxlen=PLOT_LEN)
        self._pos_buf           = deque(maxlen=PLOT_LEN)
        self._build()

    # ── helpers ────────────────────────────────────────────────────────────────
    def _odrv(self):
        return self._get_odrv()

    def _axis(self):
        odrv = self._odrv()
        if odrv is None:
            return None
        try:
            return odrv.axis0 if self._get_axis_idx() == 0 else odrv.axis1
        except AttributeError:
            return None  # EmptyInterface after disconnect/reboot

    def _ax_idx(self):
        return self._get_axis_idx()

    def _log(self, msg, color=CLR_OK):
        self.log_message.emit(msg, color)

    # ── UI build ───────────────────────────────────────────────────────────────
    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # Left scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        inner = QWidget()
        scroll.setWidget(inner)
        left = QVBoxLayout(inner)
        left.setSpacing(8)
        left.setContentsMargins(8, 8, 8, 8)

        left.addWidget(self._build_errors_group())
        left.addWidget(self._build_control_group())
        left.addWidget(self._build_gains_group())
        left.addWidget(self._build_anticogging_group())
        left.addWidget(self._build_readout_group())
        left.addStretch()

        splitter.addWidget(scroll)
        splitter.addWidget(self._build_chart_panel())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

    # ── Errors group ───────────────────────────────────────────────────────────
    def _build_errors_group(self):
        self._err_box = QGroupBox("Errors")
        col = QVBoxLayout(self._err_box)
        col.setSpacing(4)

        def _err_row(label_text):
            row = QHBoxLayout()
            row.setSpacing(4)
            lbl_name = QLabel(f"{label_text}:")
            lbl_name.setFixedWidth(74)
            lbl_name.setStyleSheet(f"color: {CLR_MUTED};")
            lbl_val = QLabel("0x0000  none")
            lbl_val.setStyleSheet("font-family: monospace;")
            lbl_val.setTextFormat(Qt.RichText)
            lbl_val.setWordWrap(True)
            btn_q = QPushButton("?")
            btn_q.setFixedWidth(22)
            btn_q.setFixedHeight(20)
            btn_q.setStyleSheet(
                f"color: {CLR_INFO}; font-weight: bold; padding: 0; border-radius: 3px;"
            )
            row.addWidget(lbl_name)
            row.addWidget(lbl_val, stretch=1)
            row.addWidget(btn_q)
            return row, lbl_val, btn_q

        row_ax,  self.lbl_err_axis,    self.btn_err_axis    = _err_row("Axis")
        row_mot, self.lbl_err_motor,   self.btn_err_motor   = _err_row("Motor")
        row_enc, self.lbl_err_encoder, self.btn_err_encoder = _err_row("Encoder")
        row_ctl, self.lbl_err_ctrl,    self.btn_err_ctrl    = _err_row("Controller")

        for row in (row_ax, row_mot, row_enc, row_ctl):
            col.addLayout(row)

        self.btn_err_axis.clicked.connect(
            lambda: self._show_error_popup("Axis Errors", self._last_err_axis, AXIS_ERRORS, AXIS_ERROR_HINTS))
        self.btn_err_motor.clicked.connect(
            lambda: self._show_error_popup("Motor Errors", self._last_err_motor, MOTOR_ERRORS, MOTOR_ERROR_HINTS))
        self.btn_err_encoder.clicked.connect(
            lambda: self._show_error_popup("Encoder Errors", self._last_err_encoder, ENCODER_ERRORS, ENCODER_ERROR_HINTS))
        self.btn_err_ctrl.clicked.connect(
            lambda: self._show_error_popup("Controller Errors", self._last_err_ctrl, CONTROLLER_ERRORS, CONTROLLER_ERROR_HINTS))

        self._last_err_axis = self._last_err_motor = self._last_err_encoder = self._last_err_ctrl = 0

        self.btn_clear_errors = QPushButton("Clear Errors")
        self.btn_clear_errors.setEnabled(False)
        self.btn_clear_errors.setToolTip("Requires: connected")
        self.btn_clear_errors.clicked.connect(self._clear_errors)
        col.addWidget(self.btn_clear_errors)

        return self._err_box

    def _show_error_popup(self, title: str, code: int, names: dict, hints: dict):
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(480, 360)
        outer = QVBoxLayout(dlg)

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
                    f'These bits are not in the known error table for fw 0.5.6. '
                    f'Check the ODrive firmware source or forum for 0x{unknown:08X}.</p><hr/>'
                )
            txt.setHtml(html)

        outer.addWidget(txt)
        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        outer.addWidget(btns)
        dlg.exec()

    # ── Control group ──────────────────────────────────────────────────────────
    def _build_control_group(self):
        box = QGroupBox("Control")
        col = QVBoxLayout(box)
        col.setSpacing(6)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._mode_grp = QButtonGroup(self)
        self._rb_pos = QRadioButton("Position")
        self._rb_vel = QRadioButton("Velocity")
        self._rb_trq = QRadioButton("Torque")
        self._rb_vel.setChecked(True)
        for i, rb in enumerate((self._rb_pos, self._rb_vel, self._rb_trq)):
            self._mode_grp.addButton(rb, i)
            mode_row.addWidget(rb)
        mode_row.addStretch()
        col.addLayout(mode_row)

        sp_row = QHBoxLayout()
        sp_row.addWidget(QLabel("Setpoint:"))
        self.sp_setpoint = _dspin(-1000, 1000, 0.0, dec=3, step=0.1)
        self._lbl_sp_units = QLabel("t/s")
        sp_row.addWidget(self.sp_setpoint)
        sp_row.addWidget(self._lbl_sp_units)
        sp_row.addStretch()
        col.addLayout(sp_row)

        self._rb_pos.toggled.connect(self._on_mode_changed)
        self._rb_vel.toggled.connect(self._on_mode_changed)
        self._rb_trq.toggled.connect(self._on_mode_changed)
        self.sp_setpoint.valueChanged.connect(self._send_setpoint)

        _TIP = "Requires: connected + motor calibrated + encoder ready + no errors"
        btn_row = QHBoxLayout()
        self.btn_enable  = QPushButton("Enable / Closed Loop")
        self.btn_disable = QPushButton("Disable / Idle")
        self.btn_enable.setStyleSheet(f"color: {CLR_OK};")
        self.btn_disable.setStyleSheet(f"color: {CLR_WARN};")
        for b in (self.btn_enable, self.btn_disable):
            b.setEnabled(False)
            b.setToolTip(_TIP)
            btn_row.addWidget(b)
        col.addLayout(btn_row)

        self.btn_enable.clicked.connect(self._enable_closed_loop)
        self.btn_disable.clicked.connect(self._disable_idle)

        return box

    def _on_mode_changed(self):
        if self._rb_pos.isChecked():
            self._lbl_sp_units.setText("t")
        elif self._rb_vel.isChecked():
            self._lbl_sp_units.setText("t/s")
        else:
            self._lbl_sp_units.setText("Nm")

    def _get_control_mode(self):
        if self._rb_pos.isChecked():
            return CONTROL_MODE_POSITION_CONTROL
        elif self._rb_vel.isChecked():
            return CONTROL_MODE_VELOCITY_CONTROL
        return CONTROL_MODE_TORQUE_CONTROL

    def _send_setpoint(self, val):
        axis = self._axis()
        if axis is None:
            return
        try:
            mode = self._get_control_mode()
            if mode == CONTROL_MODE_POSITION_CONTROL:
                axis.controller.input_pos = val
            elif mode == CONTROL_MODE_VELOCITY_CONTROL:
                axis.controller.input_vel = val
            else:
                axis.controller.input_torque = val
        except Exception as e:
            self._log(f"Setpoint write error: {e}", CLR_ERR)

    def _enable_closed_loop(self):
        axis = self._axis()
        if axis is None:
            return
        try:
            axis.controller.config.control_mode = self._get_control_mode()
            axis.controller.config.input_mode   = INPUT_MODE_PASSTHROUGH
            for attr in ("input_pos", "input_vel", "input_torque"):
                try:
                    setattr(axis.controller, attr, 0.0)
                except Exception:
                    pass
            axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
            self._log(f"Ax{self._ax_idx()} → Closed Loop", CLR_OK)
        except Exception as e:
            self._log(f"Enable error: {e}", CLR_ERR)

    def _disable_idle(self):
        axis = self._axis()
        if axis is None:
            return
        try:
            axis.requested_state = AXIS_STATE_IDLE
            self._log(f"Ax{self._ax_idx()} → Idle", CLR_WARN)
        except Exception as e:
            self._log(f"Disable error: {e}", CLR_ERR)

    # ── Gains group ────────────────────────────────────────────────────────────
    def _build_gains_group(self):
        box = QGroupBox("Gains")
        form = QFormLayout(box)
        form.setSpacing(6)

        self.sp_pos_gain  = _dspin(0, 1000, 20.0, dec=3, step=0.5)
        self.sp_vel_gain  = _dspin(0, 10,   0.16, dec=5, step=0.001)
        self.sp_vel_int   = _dspin(0, 10,   0.32, dec=5, step=0.001)
        self.sp_vel_lim_g = _dspin(0, 500,  2.0,  dec=2, step=0.5, suffix="t/s")

        form.addRow("pos_gain",            self.sp_pos_gain)
        form.addRow("vel_gain",            self.sp_vel_gain)
        form.addRow("vel_integrator_gain", self.sp_vel_int)
        form.addRow("vel_limit",           self.sp_vel_lim_g)

        _TIP = "Requires: connected + motor calibrated + encoder ready + no errors"
        btn_row = QHBoxLayout()
        self.btn_read_gains  = QPushButton("Read Gains")
        self.btn_apply_gains = QPushButton("Apply Gains")
        self.btn_apply_gains.setStyleSheet(f"color: {CLR_INFO};")
        for b in (self.btn_read_gains, self.btn_apply_gains):
            b.setEnabled(False)
            b.setToolTip(_TIP)
            btn_row.addWidget(b)
        form.addRow("", btn_row)

        self.btn_read_gains.clicked.connect(self._read_gains)
        self.btn_apply_gains.clicked.connect(self._apply_gains)

        return box

    def _read_gains(self):
        axis = self._axis()
        if axis is None:
            return
        try:
            cc = axis.controller.config
            self.sp_pos_gain.setValue(float(getattr(cc, "pos_gain", 20.0)))
            self.sp_vel_gain.setValue(float(getattr(cc, "vel_gain", 0.16)))
            self.sp_vel_int.setValue(float(getattr(cc, "vel_integrator_gain", 0.32)))
            self.sp_vel_lim_g.setValue(float(getattr(cc, "vel_limit", 2.0)))
            self._log(f"Ax{self._ax_idx()} gains read OK", CLR_OK)
        except Exception as e:
            self._log(f"Read gains error: {e}", CLR_ERR)

    def _apply_gains(self):
        axis = self._axis()
        if axis is None:
            return
        try:
            cc = axis.controller.config
            cc.pos_gain            = self.sp_pos_gain.value()
            cc.vel_gain            = self.sp_vel_gain.value()
            cc.vel_integrator_gain = self.sp_vel_int.value()
            cc.vel_limit           = self.sp_vel_lim_g.value()
            self._log(f"Ax{self._ax_idx()} gains applied", CLR_OK)
        except Exception as e:
            self._log(f"Apply gains error: {e}", CLR_ERR)

    # ── Anticogging group ──────────────────────────────────────────────────────
    def _build_anticogging_group(self):
        box = QGroupBox("Anticogging")
        col = QVBoxLayout(box)
        col.setSpacing(6)

        form = QFormLayout()
        form.setSpacing(4)
        self.sp_acog_vel_threshold  = _dspin(0, 500,  5.0,  dec=1, step=1.0,   suffix="t/s")
        self.sp_acog_vel_threshold.setToolTip("calib_vel_threshold — higher = faster sweep, lower quality. ~5 = good, ~100 = quick test")
        self.sp_acog_pos_threshold  = _dspin(0, 500,  5.0,  dec=1, step=1.0,   suffix="turns")
        self.sp_acog_pos_threshold.setToolTip("calib_pos_threshold — higher = faster sweep, lower quality. ~5 = good, ~100 = quick test")
        form.addRow("vel_threshold",      self.sp_acog_vel_threshold)
        form.addRow("pos_threshold",      self.sp_acog_pos_threshold)
        col.addLayout(form)

        # Quick-fill buttons
        thresh_row = QHBoxLayout()
        btn_thresh_good  = QPushButton("Good (5)")
        btn_thresh_quick = QPushButton("Quick test (100)")
        btn_thresh_good.setToolTip("vel/pos thresholds = 5  — normal quality, ~30 min")
        btn_thresh_quick.setToolTip("vel/pos thresholds = 100 — fast end-to-end test, low quality, ~2 min")
        btn_thresh_good.clicked.connect(lambda: (
            self.sp_acog_vel_threshold.setValue(5.0),
            self.sp_acog_pos_threshold.setValue(5.0),
        ))
        btn_thresh_quick.clicked.connect(lambda: (
            self.sp_acog_vel_threshold.setValue(100.0),
            self.sp_acog_pos_threshold.setValue(100.0),
        ))
        thresh_row.addWidget(btn_thresh_good)
        thresh_row.addWidget(btn_thresh_quick)
        col.addLayout(thresh_row)

        # Step buttons — each unlocks the next, 2-column grid
        self.btn_acog_preflight = QPushButton("0. Pre-flight")
        self.btn_acog_preflight.setEnabled(False)
        self.btn_acog_preflight.setToolTip("Spin motor briefly in velocity mode to confirm it can move before committing to 30-min calibration")

        self.btn_acog_prereqs = QPushButton("1. Check Prerequisites")
        self.btn_acog_prereqs.setEnabled(False)

        self.btn_acog_enter_cl = QPushButton("2. Enter Closed Loop")
        self.btn_acog_enter_cl.setEnabled(False)

        self.btn_acog_start = QPushButton("3. Start Calibration")
        self.btn_acog_start.setStyleSheet(f"color: {CLR_CAL};")
        self.btn_acog_start.setEnabled(False)

        step_grid = QGridLayout()
        step_grid.setSpacing(4)
        step_grid.addWidget(self.btn_acog_preflight, 0, 0)
        step_grid.addWidget(self.btn_acog_prereqs,   0, 1)
        step_grid.addWidget(self.btn_acog_enter_cl,  1, 0)
        step_grid.addWidget(self.btn_acog_start,     1, 1)
        col.addLayout(step_grid)

        # Progress bar
        self.pb_acog = QProgressBar()
        self.pb_acog.setRange(0, COGGING_MAP_SIZE)
        self.pb_acog.setValue(0)
        self.pb_acog.setFormat("%v / " + str(COGGING_MAP_SIZE) + "  (%p%)")
        self.pb_acog.setTextVisible(True)
        col.addWidget(self.pb_acog)

        btn_row2 = QHBoxLayout()
        self.chk_acog_enabled  = QCheckBox("Enabled")
        self.btn_acog_apply_en = QPushButton("Apply Enable")
        self.btn_acog_apply_en.setEnabled(False)
        self.btn_acog_apply_en.setToolTip("Requires: connected + motor calibrated + encoder ready + no errors")
        btn_row2.addWidget(self.chk_acog_enabled)
        btn_row2.addWidget(self.btn_acog_apply_en)
        btn_row2.addStretch()
        col.addLayout(btn_row2)

        self.lbl_acog_status = QLabel("Status: Idle — click Step 1")
        self.lbl_acog_status.setStyleSheet(f"color: {CLR_MUTED}; font-family: monospace;")
        col.addWidget(self.lbl_acog_status)

        self.btn_acog_preflight.clicked.connect(self._preflight_anticogging)
        self.btn_acog_prereqs.clicked.connect(self._check_anticog_prereqs)
        self.btn_acog_enter_cl.clicked.connect(self._enter_anticog_closed_loop)
        self.btn_acog_start.clicked.connect(self._start_anticogging)
        self.btn_acog_apply_en.clicked.connect(self._apply_anticog_enable)

        return box

    def _preflight_anticogging(self):
        """Spin motor at 2 t/s for 2 s to confirm it can move before calibration."""
        odrv = self._odrv()
        if odrv is None:
            return
        self._log("── Anticogging Pre-flight: Velocity Smoke Test ──", CLR_MUTED)
        self.btn_acog_preflight.setEnabled(False)
        self._preflight_worker = _PreflightWorker(odrv, self._ax_idx())
        self._preflight_worker.log.connect(lambda msg, clr: self._log(msg, clr))
        self._preflight_worker.finished.connect(lambda: self.btn_acog_preflight.setEnabled(True))
        self._preflight_worker.start()

    def _check_anticog_prereqs(self):
        axis = self._axis()
        if axis is None:
            return
        self._log("── Anticogging Step 1: Check Prerequisites ──", CLR_MUTED)
        ok = True
        checks = [
            ("encoder.is_ready",    getattr(axis.encoder, "is_ready",      False), True),
            ("motor.is_calibrated", getattr(axis.motor,   "is_calibrated", False), True),
            ("axis.error",          getattr(axis, "error", -1),                    0),
            ("axis.current_state",  getattr(axis, "current_state", -1),            1),
        ]
        for label, val, expect in checks:
            if val == expect:
                self._log(f"  PASS  {label} = {val}", CLR_OK)
            else:
                self._log(f"  FAIL  {label} = {val}  (expected {expect})", CLR_ERR)
                ok = False
        if ok:
            self._anticog_step = 1
            self.btn_acog_enter_cl.setEnabled(True)
            self.lbl_acog_status.setText("Step 1 PASS — click Step 2")
            self.lbl_acog_status.setStyleSheet(f"color: {CLR_OK}; font-family: monospace;")
        else:
            self._anticog_step = 0
            self.btn_acog_enter_cl.setEnabled(False)
            self.btn_acog_start.setEnabled(False)
            self.lbl_acog_status.setText("Step 1 FAIL — see log")
            self.lbl_acog_status.setStyleSheet(f"color: {CLR_ERR}; font-family: monospace;")

    def _enter_anticog_closed_loop(self):
        axis = self._axis()
        if axis is None:
            return
        self._log("── Anticogging Step 2: Enter Closed Loop ──", CLR_MUTED)
        try:
            if axis.controller is None:
                raise RuntimeError("axis.controller is None — ODrive may be in a bad state; try reconnecting")
            axis.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
            axis.controller.config.input_mode   = INPUT_MODE_PASSTHROUGH
            axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL

            # Check state after 300ms without blocking the GUI
            QTimer.singleShot(300, self._verify_anticog_closed_loop)
        except Exception as e:
            self._log(f"Step 2 error: {e}", CLR_ERR)

    def _verify_anticog_closed_loop(self):
        axis = self._axis()
        if axis is None:
            return
        try:
            state = getattr(axis, "current_state", -1)
            error = getattr(axis, "error", -1)
            ok = (state == 8 and error == 0)

            self._log(f"  {'PASS' if state==8 else 'FAIL'}  axis.current_state = {state}  (expected 8)", CLR_OK if state == 8 else CLR_ERR)
            self._log(f"  {'PASS' if error==0 else 'FAIL'}  axis.error = {error}  (expected 0)",         CLR_OK if error == 0 else CLR_ERR)

            if ok:
                self._anticog_step = 2
                self.btn_acog_start.setEnabled(True)
                self.lbl_acog_status.setText("Step 2 PASS — click Step 3")
                self.lbl_acog_status.setStyleSheet(f"color: {CLR_OK}; font-family: monospace;")
            else:
                axis.requested_state = AXIS_STATE_IDLE
                self._anticog_step = 1
                self.btn_acog_start.setEnabled(False)
                self.lbl_acog_status.setText("Step 2 FAIL — see log")
                self.lbl_acog_status.setStyleSheet(f"color: {CLR_ERR}; font-family: monospace;")
        except Exception as e:
            self._log(f"Step 2 error: {e}", CLR_ERR)

    def _start_anticogging(self):
        axis = self._axis()
        odrv = self._odrv()
        if axis is None or odrv is None:
            return
        try:
            cc = axis.controller.config

            # Save gains and ODrive-level settings we are about to change
            self._anticog_saved_gains = {
                "pos_gain":            getattr(cc, "pos_gain", 20.0),
                "vel_gain":            getattr(cc, "vel_gain", 0.16),
                "vel_integrator_gain": getattr(cc, "vel_integrator_gain", 0.32),
                "vel_limit":           getattr(cc, "vel_limit", 2.0),
            }
            self._anticog_saved_neg_cur = getattr(odrv.config, "dc_max_negative_current", -10.0)

            # ── Pre-calibration setup (mirrors run_anticogging.py) ─────────────
            # 1. Disable anticogging so the stale map doesn't fight the sweep
            cc.anticogging.anticogging_enabled = False
            self._log("  anticogging_enabled → False (re-enabled on save)", CLR_MUTED)

            # 2. Allow regen current (default -0.01 A clamps braking → motor stalls)
            odrv.config.dc_max_negative_current = -5.0
            self._log(f"  dc_max_negative_current: {self._anticog_saved_neg_cur:.2f} → -5.0 A", CLR_MUTED)

            # 3. Apply calibration thresholds from spinboxes
            vt = self.sp_acog_vel_threshold.value()
            pt = self.sp_acog_pos_threshold.value()
            cc.anticogging.calib_vel_threshold = vt
            cc.anticogging.calib_pos_threshold = pt
            self._log(f"  calib_vel_threshold={vt}  calib_pos_threshold={pt}", CLR_MUTED)

            # Axis is already in closed loop position control from Step 2
            self._anticog_pos_buf.clear()
            self._anticog_calib_seen_running = False  # completion guard
            axis.controller.start_anticogging_calibration()
            self._anticog_state = self._ANTICOG_RUNNING
            self.btn_acog_start.setEnabled(False)
            self.pb_acog.setValue(0)
            self.lbl_acog_status.setText("Status: Calibrating…")
            self.lbl_acog_status.setStyleSheet(f"color: {CLR_CAL}; font-family: monospace;")
            self._log(f"Ax{self._ax_idx()} anticogging calibration started", CLR_CAL)
        except Exception as e:
            self._log(f"Anticogging start error: {e}", CLR_ERR)
            self._anticog_state = self._ANTICOG_IDLE

    def _poll_anticogging(self, axis):
        import time as _time
        try:
            acog  = axis.controller.config.anticogging
            idx   = getattr(acog, "index", 0)
            pos   = float(getattr(axis.encoder, "pos_estimate", 0.0))
            now   = _time.monotonic()

            # Progress bar (idx counts up from 0 to COGGING_MAP_SIZE during sweep)
            self.pb_acog.setValue(min(idx, COGGING_MAP_SIZE))

            # Stall detection: keep a 60-second rolling window of positions.
            # If the total range of positions in that window is < 0.001 turns, warn.
            self._anticog_pos_buf.append((now, pos))
            oldest_allowed = now - 60.0
            recent = [p for t, p in self._anticog_pos_buf if t >= oldest_allowed]
            if len(recent) >= 60:  # need at least 6 s of data before warning
                pos_range = max(recent) - min(recent)
                if pos_range < 0.001:
                    self._log(f"  WARN  motor position unchanged for 60 s (range {pos_range:.5f} t) — may be stalled", CLR_WARN)
                    self._anticog_pos_buf.clear()  # reset so we don't spam

            # Completion detection via calib_anticogging flag:
            #   True  → calibration running
            #   False → calibration finished (after we saw it True)
            calib_running = getattr(acog, "calib_anticogging", None)
            if calib_running is True:
                self._anticog_calib_seen_running = True
            done = (calib_running is False and getattr(self, "_anticog_calib_seen_running", False))

            pct = idx * 100 // COGGING_MAP_SIZE
            self.lbl_acog_status.setText(f"Calibrating — idx {idx}/{COGGING_MAP_SIZE} ({pct}%)")
            if done:
                self._anticog_calib_seen_running = False   # reset for next run
                self._anticog_state = self._ANTICOG_DONE
                self.pb_acog.setValue(COGGING_MAP_SIZE)
                try:
                    axis.controller.remove_anticogging_bias()
                except Exception as e:
                    self._log(f"remove_anticogging_bias warning: {e}", CLR_WARN)
                try:
                    odrv = self._odrv()
                    if odrv is not None:
                        odrv.config.dc_max_negative_current = getattr(self, "_anticog_saved_neg_cur", -10.0)
                    cc = axis.controller.config
                    for k, v in self._anticog_saved_gains.items():
                        setattr(cc, k, v)
                    self._log(f"Ax{self._ax_idx()} anticogging done — gains and dc_max_negative_current restored", CLR_OK)
                except Exception as e:
                    self._log(f"Gain restore warning: {e}", CLR_WARN)
                self.lbl_acog_status.setText("Status: Done — click Save")
                self.lbl_acog_status.setStyleSheet(f"color: {CLR_OK}; font-family: monospace;")
        except Exception:
            pass

    def _save_anticogging(self):
        axis = self._axis()
        odrv = self._odrv()
        if axis is None or odrv is None:
            return
        try:
            # Guard: refuse to save if calibration is still actively running
            calib_active = getattr(axis.controller.config.anticogging, "calib_anticogging", False)
            if calib_active:
                self._log("Save blocked — calibration still running (calib_anticogging=True). Wait for motor to stop.", CLR_WARN)
                return

            acog = axis.controller.config.anticogging
            acog.anticogging_enabled = True
            acog.pre_calibrated      = True
            # Note: calib_anticogging is read-only (firmware-controlled); cannot be cleared from Python

            try:
                odrv.save_configuration()
            except Exception:
                pass  # expected — ODrive reboots immediately after saving
            try:
                odrv.reboot()
            except Exception:
                pass  # expected — connection drops on reboot

            self._log(f"Ax{self._ax_idx()} anticogging saved + rebooting…", CLR_OK)
            self.reboot_requested.emit()
        except Exception as e:
            self._log(f"Anticogging save error: {e}", CLR_ERR)

    def _apply_anticog_enable(self):
        axis = self._axis()
        if axis is None:
            return
        try:
            axis.controller.config.anticogging.anticogging_enabled = self.chk_acog_enabled.isChecked()
            self._log(f"Ax{self._ax_idx()} anticogging enabled={self.chk_acog_enabled.isChecked()}", CLR_OK)
        except Exception as e:
            self._log(f"Anticogging enable error: {e}", CLR_ERR)

    # ── Live Readout group ─────────────────────────────────────────────────────
    def _build_readout_group(self):
        box = QGroupBox("Live Readout")
        form = QFormLayout(box)
        form.setSpacing(4)

        self.lbl_rd_state    = _status_label("—")
        self.lbl_rd_pos      = _status_label("—")
        self.lbl_rd_vel      = _status_label("—")
        self.lbl_rd_iq       = _status_label("—")
        self.lbl_rd_setpoint = _status_label("—")
        self.lbl_rd_vbus     = _status_label("—")

        form.addRow("State",     self.lbl_rd_state)
        form.addRow("Pos (t)",   self.lbl_rd_pos)
        form.addRow("Vel (t/s)", self.lbl_rd_vel)
        form.addRow("Iq (A)",    self.lbl_rd_iq)
        form.addRow("Setpoint",  self.lbl_rd_setpoint)
        form.addRow("Vbus (V)",  self.lbl_rd_vbus)

        return box

    # ── Chart panel ────────────────────────────────────────────────────────────
    def _build_chart_panel(self):
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setContentsMargins(6, 6, 6, 6)
        col.setSpacing(6)

        if _HAS_CHARTS:
            self._chart = QChart()
            self._chart.setBackgroundBrush(QColor(CLR_PANEL))
            self._chart.setPlotAreaBackgroundBrush(QColor("#1e1e1e"))
            self._chart.setPlotAreaBackgroundVisible(True)
            self._chart.legend().setVisible(False)
            self._chart.layout().setContentsMargins(0, 0, 0, 0)
            self._chart.setMargins(QMargins(4, 4, 4, 4))

            self._ser_iq  = QLineSeries()
            self._ser_vel = QLineSeries()
            self._ser_pos = QLineSeries()
            self._ser_iq.setColor(QColor(CLR_IQ))
            self._ser_vel.setColor(QColor(CLR_VEL))
            self._ser_pos.setColor(QColor(CLR_POS))

            for s in (self._ser_iq, self._ser_vel, self._ser_pos):
                self._chart.addSeries(s)

            self._ax_t = QValueAxis()
            self._ax_t.setLabelsColor(QColor(CLR_LABEL))
            self._ax_t.setGridLineColor(QColor("#3a3a3a"))
            self._ax_t.setTitleText("t (s)")
            self._ax_t.setTitleBrush(QColor(CLR_MUTED))

            self._ax_y = QValueAxis()
            self._ax_y.setLabelsColor(QColor(CLR_LABEL))
            self._ax_y.setGridLineColor(QColor("#3a3a3a"))

            self._chart.addAxis(self._ax_t, Qt.AlignBottom)
            self._chart.addAxis(self._ax_y, Qt.AlignLeft)
            for s in (self._ser_iq, self._ser_vel, self._ser_pos):
                s.attachAxis(self._ax_t)
                s.attachAxis(self._ax_y)

            chart_view = QChartView(self._chart)
            chart_view.setRenderHint(QPainter.Antialiasing)
            chart_view.setMinimumHeight(250)
            col.addWidget(chart_view, stretch=1)
        else:
            self._chart = None
            no_chart = QLabel("PySide6.QtCharts not available.\npip install PySide6-Addons")
            no_chart.setAlignment(Qt.AlignCenter)
            no_chart.setStyleSheet(f"color: {CLR_MUTED};")
            col.addWidget(no_chart, stretch=1)

        legend_row = QHBoxLayout()
        self.chk_show_iq  = QCheckBox("Iq")
        self.chk_show_vel = QCheckBox("Vel")
        self.chk_show_pos = QCheckBox("Pos")
        for chk in (self.chk_show_iq, self.chk_show_vel, self.chk_show_pos):
            chk.setChecked(True)
        self.chk_show_iq.setStyleSheet(f"color: {CLR_IQ};")
        self.chk_show_vel.setStyleSheet(f"color: {CLR_VEL};")
        self.chk_show_pos.setStyleSheet(f"color: {CLR_POS};")

        self.lbl_val_iq  = QLabel("0.00 A")
        self.lbl_val_vel = QLabel("0.00 t/s")
        self.lbl_val_pos = QLabel("0.00 t")
        for lbl in (self.lbl_val_iq, self.lbl_val_vel, self.lbl_val_pos):
            lbl.setStyleSheet("font-family: monospace; font-size: 11px;")

        legend_row.addWidget(self.chk_show_iq)
        legend_row.addWidget(self.lbl_val_iq)
        legend_row.addWidget(self.chk_show_vel)
        legend_row.addWidget(self.lbl_val_vel)
        legend_row.addWidget(self.chk_show_pos)
        legend_row.addWidget(self.lbl_val_pos)
        legend_row.addStretch()

        self.btn_clear_plot = QPushButton("Clear Plot")
        self.btn_clear_plot.setFixedWidth(80)
        self.btn_clear_plot.clicked.connect(self._clear_plot)
        legend_row.addWidget(self.btn_clear_plot)
        col.addLayout(legend_row)

        if _HAS_CHARTS:
            self.chk_show_iq.toggled.connect(self._ser_iq.setVisible)
            self.chk_show_vel.toggled.connect(self._ser_vel.setVisible)
            self.chk_show_pos.toggled.connect(self._ser_pos.setVisible)

        return panel

    def _clear_plot(self):
        self._t_buf.clear()
        self._iq_buf.clear()
        self._vel_buf.clear()
        self._pos_buf.clear()
        self._t0 = None
        if _HAS_CHARTS and self._chart:
            for s in (self._ser_iq, self._ser_vel, self._ser_pos):
                s.replace([])

    # ── Button gate ────────────────────────────────────────────────────────────
    def _update_button_gates(self, ok: bool, connected: bool):
        self.btn_clear_errors.setEnabled(connected)
        for b in (self.btn_enable, self.btn_disable, self.btn_read_gains,
                  self.btn_apply_gains, self.btn_acog_apply_en):
            b.setEnabled(ok)
        # Step 0 (pre-flight) and Step 1 available whenever connected; later steps gate on prior step passing
        not_running = self._anticog_state != self._ANTICOG_RUNNING
        self.btn_acog_preflight.setEnabled(connected and not_running)
        self.btn_acog_prereqs.setEnabled(connected and not_running)
        self.btn_acog_enter_cl.setEnabled(connected and self._anticog_step >= 1 and not_running)
        self.btn_acog_start.setEnabled(connected and self._anticog_step >= 2 and not_running)

    def _update_error_box(self, has_error: bool):
        if has_error:
            self._err_box.setStyleSheet(
                "QGroupBox { background: #3a1a1a; border-color: #ff6464; }"
            )
        else:
            self._err_box.setStyleSheet("")

    def _clear_errors(self):
        axis = self._axis()
        if axis is None:
            return
        try:
            axis.clear_errors()
            self._log(f"Ax{self._ax_idx()} errors cleared (axis + motor + encoder + controller)", CLR_OK)
        except Exception as e:
            self._log(f"Clear errors failed: {e}", CLR_ERR)

    # ── poll update (called by MainWindow every 100ms) ─────────────────────────
    def update(self, odrv, axis_idx):
        axis = None
        if odrv is not None:
            try:
                axis = odrv.axis0 if axis_idx == 0 else odrv.axis1
            except AttributeError:
                pass  # EmptyInterface after disconnect/reboot

        connected = axis is not None

        # Axis switch → clear plot buffers
        if axis_idx != self._last_axis_idx:
            self._last_axis_idx = axis_idx
            self._clear_plot()

        if not connected:
            for lbl in (self.lbl_rd_state, self.lbl_rd_pos, self.lbl_rd_vel,
                        self.lbl_rd_iq, self.lbl_rd_setpoint, self.lbl_rd_vbus):
                _colored(lbl, "—", CLR_MUTED)
            self._update_button_gates(False, False)
            self._update_error_box(False)
            return

        try:
            ax_err   = getattr(axis,             "error", 0)
            mot_err  = getattr(axis.motor,       "error", 0)
            enc_err  = getattr(axis.encoder,     "error", 0)
            ctrl_err = getattr(axis.controller,  "error", 0)

            # Cache for popup buttons
            self._last_err_axis    = ax_err
            self._last_err_motor   = mot_err
            self._last_err_encoder = enc_err
            self._last_err_ctrl    = ctrl_err

            def _fmt_verbose(code, names):
                if code == 0:
                    return f'<span style="color:{CLR_OK}">0x00000000  none</span>'
                bits = [name for mask, name in names.items() if code & mask]
                known_mask = sum(m for m in names if code & m)
                unknown = code & ~known_mask
                parts = bits[:]
                if unknown:
                    parts.append(f"UNKNOWN(0x{unknown:08X})")
                label = " | ".join(parts) if parts else ""
                return (f'<span style="color:{CLR_ERR}">0x{code:08X}</span>'
                        f'<span style="color:{CLR_WARN}">  {label}</span>')

            self.lbl_err_axis.setText(_fmt_verbose(ax_err,   AXIS_ERRORS))
            self.lbl_err_motor.setText(_fmt_verbose(mot_err,  MOTOR_ERRORS))
            self.lbl_err_encoder.setText(_fmt_verbose(enc_err, ENCODER_ERRORS))
            self.lbl_err_ctrl.setText(_fmt_verbose(ctrl_err,  CONTROLLER_ERRORS))

            has_error = any(e != 0 for e in (ax_err, mot_err, enc_err, ctrl_err))
            self._update_error_box(has_error)

            mot_cal = getattr(axis.motor,   "is_calibrated", False)
            enc_rdy = getattr(axis.encoder, "is_ready",      False)
            ok = mot_cal and enc_rdy and not has_error
            self._update_button_gates(ok, connected)

            # Anticogging poll
            if self._anticog_state == self._ANTICOG_RUNNING:
                self._poll_anticogging(axis)

            # Live readout
            state_name = AXIS_STATES.get(getattr(axis, "current_state", 0),
                                         f"State {getattr(axis, 'current_state', 0)}")
            _colored(self.lbl_rd_state, state_name, CLR_OK)

            pos = float(getattr(axis.encoder, "pos_estimate", 0.0))
            vel = float(getattr(axis.encoder, "vel_estimate", 0.0))
            try:
                iq = float(axis.motor.current_control.Iq_measured)
            except Exception:
                iq = 0.0

            try:
                cc = axis.controller
                if self._rb_pos.isChecked():
                    sp_val = float(cc.input_pos)
                elif self._rb_vel.isChecked():
                    sp_val = float(cc.input_vel)
                else:
                    sp_val = float(cc.input_torque)
            except Exception as e:
                self._log(f"Setpoint readout error: {e}", CLR_WARN)
                sp_val = 0.0

            try:
                vbus = float(odrv.vbus_voltage)
            except Exception:
                vbus = 0.0

            _colored(self.lbl_rd_pos,      f"{pos:.4f} t",   CLR_OK)
            _colored(self.lbl_rd_vel,      f"{vel:.4f} t/s", CLR_OK)
            _colored(self.lbl_rd_iq,       f"{iq:.4f} A",    CLR_OK)
            _colored(self.lbl_rd_setpoint, f"{sp_val:.4f}",  CLR_OK)
            _colored(self.lbl_rd_vbus,     f"{vbus:.2f} V",  CLR_OK)

            self.lbl_val_iq.setText(f"{iq:.2f} A")
            self.lbl_val_vel.setText(f"{vel:.2f} t/s")
            self.lbl_val_pos.setText(f"{pos:.2f} t")

            # Plot
            if _HAS_CHARTS and self._chart:
                now = time.monotonic()
                if self._t0 is None:
                    self._t0 = now
                t = now - self._t0
                self._t_buf.append(t)
                self._iq_buf.append(iq)
                self._vel_buf.append(vel)
                self._pos_buf.append(pos)

                self._ser_iq.replace([QPointF(x, y) for x, y in zip(self._t_buf, self._iq_buf)])
                self._ser_vel.replace([QPointF(x, y) for x, y in zip(self._t_buf, self._vel_buf)])
                self._ser_pos.replace([QPointF(x, y) for x, y in zip(self._t_buf, self._pos_buf)])

                if len(self._t_buf) > 1:
                    self._ax_t.setRange(self._t_buf[0], self._t_buf[-1])
                    all_vals = list(self._iq_buf) + list(self._vel_buf) + list(self._pos_buf)
                    mn, mx = min(all_vals), max(all_vals)
                    margin = max(0.5, (mx - mn) * 0.1)
                    self._ax_y.setRange(mn - margin, mx + margin)

        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Terminal
# ══════════════════════════════════════════════════════════════════════════════
class TerminalTab(QWidget):
    def __init__(self, get_odrv, get_axis_idx):
        super().__init__()
        self._get_odrv     = get_odrv
        self._get_axis_idx = get_axis_idx
        self._build()

        self._inbox_timer = QTimer(self)
        self._inbox_timer.setInterval(INBOX_MS)
        self._inbox_timer.timeout.connect(self._poll_inbox)
        self._inbox_timer.start()

    # ── build UI ───────────────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # Log output
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Courier New", 10))
        root.addWidget(self._log_view, stretch=1)

        # Command input row
        cmd_row = QHBoxLayout()
        self._cmd_input = QLineEdit()
        self._cmd_input.setPlaceholderText("Enter Python expression or statement…")
        self._cmd_input.setStyleSheet(
            f"background: #3c3c3c; border: 1px solid #555; border-radius: 3px;"
            f" padding: 3px 6px; color: {CLR_LABEL}; font-family: Courier New; font-size: 11px;"
        )
        self._cmd_input.returnPressed.connect(self._send_cmd)
        cmd_row.addWidget(self._cmd_input, stretch=1)
        btn_send = QPushButton("Send")
        btn_send.setFixedWidth(70)
        btn_send.clicked.connect(self._send_cmd)
        cmd_row.addWidget(btn_send)
        root.addLayout(cmd_row)

        # Quick buttons row
        quick_row = QHBoxLayout()
        for label in QUICK_CMDS:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, l=label: self._run_quick(l))
            quick_row.addWidget(btn)
        quick_row.addStretch()
        root.addLayout(quick_row)

        # Inbox status row
        inbox_row = QHBoxLayout()
        self._lbl_inbox = QLabel(
            f"Command Inbox: ACTIVE \u25cf  "
            f"inbox: {CMD_INBOX.name}  log: {CMD_LOG.name}"
        )
        self._lbl_inbox.setStyleSheet(
            f"color: {CLR_OK}; font-family: monospace; font-size: 10px;"
        )
        inbox_row.addWidget(self._lbl_inbox)
        inbox_row.addStretch()
        root.addLayout(inbox_row)

    # ── public log method (called by MainWindow for all tab log messages) ──────
    def log(self, text: str, color: str = CLR_LABEL):
        escaped = (text.replace("&", "&amp;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;")
                       .replace("\n", "<br>"))
        self._log_view.append(
            f'<span style="color:{color}; font-family:Courier New,monospace;">{escaped}</span>'
        )
        sb = self._log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── eval namespace ─────────────────────────────────────────────────────────
    def _ns(self):
        odrv = self._get_odrv()
        ax_idx = self._get_axis_idx()
        ax0 = odrv.axis0 if odrv else None
        ax1 = odrv.axis1 if odrv else None
        axis = (ax0 if ax_idx == 0 else ax1) if odrv else None
        return {"odrv": odrv, "ax0": ax0, "ax1": ax1, "axis": axis}

    def _eval_cmd(self, cmd: str):
        ns = self._ns()
        try:
            result = eval(cmd, ns)
            return result, None
        except SyntaxError:
            pass
        try:
            exec(cmd, ns)
            return None, None
        except Exception as e:
            return None, e

    # ── command dispatch ───────────────────────────────────────────────────────
    def _send_cmd(self):
        cmd = self._cmd_input.text().strip()
        if not cmd:
            return
        self._cmd_input.clear()
        self._run_cmd(cmd)

    def _run_cmd(self, cmd: str):
        ts = time.strftime("%H:%M:%S")
        self.log(f"[{ts}] >> {cmd}", CLR_INFO)
        result, err = self._eval_cmd(cmd)
        if err is not None:
            self.log(f"[ERR] {err}", CLR_ERR)
        elif result is not None:
            self.log(f"[OK] {repr(result)}", CLR_OK)
        else:
            self.log("[OK]", CLR_OK)

    def _run_quick(self, label: str):
        self._run_cmd(QUICK_CMDS[label])

    # ── inbox polling (500 ms) ─────────────────────────────────────────────────
    def _poll_inbox(self):
        if not CMD_INBOX.exists():
            return
        try:
            content = CMD_INBOX.read_text(encoding="utf-8")
        except Exception:
            return
        lines = [l for l in content.splitlines() if l.strip()]
        if not lines:
            return
        # Truncate inbox immediately before processing
        try:
            CMD_INBOX.write_text("", encoding="utf-8")
        except Exception:
            pass
        odrv = self._get_odrv()
        log_entries: list[str] = []
        for cmd in lines:
            ts = time.strftime("%H:%M:%S")
            self.log(f"[{ts}] >> {cmd}", CLR_INFO)
            if odrv is None:
                msg = "[ERR] ODrive not connected"
                self.log(msg, CLR_ERR)
            else:
                result, err = self._eval_cmd(cmd)
                if err is not None:
                    msg = f"[ERR] {err}"
                    self.log(msg, CLR_ERR)
                elif result is not None:
                    msg = f"[OK] {repr(result)}"
                    self.log(msg, CLR_OK)
                else:
                    msg = "[OK]"
                    self.log(msg, CLR_OK)
            log_entries.append(f"[{ts}] >> {cmd}\n{msg}")
        try:
            with CMD_LOG.open("a", encoding="utf-8") as f:
                for entry in log_entries:
                    f.write(entry + "\n")
        except Exception:
            pass

    def update(self, odrv, axis_idx):
        pass  # inbox handled by own 500 ms timer


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _decode_errors(code: int, table: dict) -> str:
    if code == 0:
        return "none"
    parts = [name for mask, name in table.items() if code & mask]
    return ", ".join(parts) if parts else f"0x{code:04X}"


# ══════════════════════════════════════════════════════════════════════════════
#  KV Test Worker
# ══════════════════════════════════════════════════════════════════════════════
class KVTestWorker(QThread):
    progress = Signal(str)
    sample   = Signal(float, float, float)   # vel, Iq, V_q
    result   = Signal(dict)
    error    = Signal(str)

    def __init__(self, odrv, axis_idx, spin_vel, settle_s, n_samples):
        super().__init__()
        self._odrv      = odrv
        self._axis_idx  = axis_idx
        self._spin_vel  = spin_vel
        self._settle_s  = settle_s
        self._n_samples = n_samples

    def run(self):
        axis = None
        try:
            odrv = self._odrv
            axis = odrv.axis0 if self._axis_idx == 0 else odrv.axis1

            # --- Force IDLE first so config changes take effect cleanly ---
            self.progress.emit("Returning to Idle…")
            axis.requested_state = AXIS_STATE_IDLE
            t0 = time.monotonic()
            while time.monotonic() - t0 < 3.0:
                if getattr(axis, "current_state", 0) == AXIS_STATE_IDLE:
                    break
                time.sleep(0.05)

            self.progress.emit("Clearing errors…")
            for obj in (axis, axis.motor, axis.encoder, axis.controller):
                try:
                    obj.error = 0
                except Exception:
                    pass

            # Ensure vel_limit can accommodate the spin velocity
            try:
                if axis.controller.config.vel_limit < self._spin_vel * 1.2:
                    axis.controller.config.vel_limit = self._spin_vel * 1.5
            except Exception:
                pass

            self.progress.emit("Setting velocity control…")
            axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            axis.controller.config.input_mode   = INPUT_MODE_PASSTHROUGH
            axis.controller.input_vel = 0.0   # start at 0; ramp after entering closed-loop

            self.progress.emit("Entering closed-loop…")
            axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL

            t0 = time.monotonic()
            while time.monotonic() - t0 < 5.0:
                if getattr(axis, "current_state", 0) == AXIS_STATE_CLOSED_LOOP_CONTROL:
                    break
                time.sleep(0.05)
            else:
                ax_err  = getattr(axis, "error", 0)
                mot_err = getattr(axis.motor, "error", 0)
                self.error.emit(
                    f"Timed out entering closed-loop. "
                    f"axis.error=0x{ax_err:08X} motor.error=0x{mot_err:08X}. "
                    "Check motor/encoder calibration."
                )
                return

            # Ramp to target velocity now that we're in closed-loop
            self.progress.emit(f"Ramping to {self._spin_vel:.1f} t/s…")
            axis.controller.input_vel = self._spin_vel

            self.progress.emit(f"Settling ({self._settle_s:.1f} s)…")
            time.sleep(self._settle_s)

            if getattr(axis, "error", 0) != 0:
                self.error.emit(f"Axis error after settling: 0x{axis.error:08X}")
                axis.requested_state = AXIS_STATE_IDLE
                return

            self.progress.emit(f"Sampling ({self._n_samples} samples)…")
            vel_samples, iq_samples, vq_samples = [], [], []
            for _ in range(self._n_samples):
                vel = float(getattr(axis.encoder, "vel_estimate", 0.0))
                try:
                    iq = float(axis.motor.current_control.Iq_measured)
                except Exception:
                    iq = 0.0
                try:
                    vq = float(axis.motor.current_control.v_current_control_integral_q)
                except AttributeError:
                    self.error.emit(
                        "v_current_control_integral_q not available. Verify firmware 0.5.6.\n"
                        "Test from Terminal: ax0.motor.current_control.v_current_control_integral_q"
                    )
                    axis.controller.input_vel = 0
                    time.sleep(0.3)
                    axis.requested_state = AXIS_STATE_IDLE
                    return
                vel_samples.append(vel)
                iq_samples.append(iq)
                vq_samples.append(vq)
                self.sample.emit(vel, iq, vq)
                time.sleep(0.05)

            self.progress.emit("Stopping motor…")
            axis.controller.input_vel = 0
            time.sleep(0.3)
            axis.requested_state = AXIS_STATE_IDLE

            vel_mean = sum(vel_samples) / len(vel_samples)
            iq_mean  = sum(iq_samples)  / len(iq_samples)
            vq_mean  = sum(vq_samples)  / len(vq_samples)

            if abs(vq_mean) < 0.01:
                self.error.emit(
                    f"V_q_mean too small ({vq_mean:.4f} V). Motor may not be spinning or "
                    "integral not converging. Try a higher spin velocity."
                )
                return

            KV = (vel_mean * 60.0) / abs(vq_mean)
            Kt = 60.0 / (2.0 * math.pi * KV)

            self.result.emit({
                "KV": KV, "Kt": Kt,
                "vq_mean": vq_mean, "iq_mean": iq_mean, "vel_mean": vel_mean,
                "vel_samples": vel_samples, "iq_samples": iq_samples, "vq_samples": vq_samples,
            })

        except Exception as e:
            self.error.emit(f"KV test error: {e}")
            if axis is not None:
                try:
                    axis.requested_state = AXIS_STATE_IDLE
                except Exception:
                    pass


# ══════════════════════════════════════════════════════════════════════════════
#  Friction Sweep Worker
# ══════════════════════════════════════════════════════════════════════════════
class FrictionSweepWorker(QThread):
    progress = Signal(str)
    point    = Signal(float, float)   # vel, Iq_mean
    done     = Signal()
    error    = Signal(str)

    def __init__(self, odrv, axis_idx, vel_min, vel_max, steps, dwell_s):
        super().__init__()
        self._odrv     = odrv
        self._axis_idx = axis_idx
        self._vel_min  = vel_min
        self._vel_max  = vel_max
        self._steps    = steps
        self._dwell_s  = dwell_s

    def run(self):
        axis = None
        try:
            odrv = self._odrv
            axis = odrv.axis0 if self._axis_idx == 0 else odrv.axis1

            for obj in (axis, axis.motor, axis.encoder, axis.controller):
                try:
                    obj.error = 0
                except Exception:
                    pass

            axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            axis.controller.config.input_mode   = INPUT_MODE_PASSTHROUGH
            axis.controller.input_vel = self._vel_min

            self.progress.emit("Entering closed-loop…")
            axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL

            t0 = time.monotonic()
            while time.monotonic() - t0 < 5.0:
                if getattr(axis, "current_state", 0) == AXIS_STATE_CLOSED_LOOP_CONTROL:
                    break
                time.sleep(0.05)
            else:
                self.error.emit("Timed out waiting for closed-loop.")
                return

            n = max(2, self._steps)
            vels = [self._vel_min + (self._vel_max - self._vel_min) * i / (n - 1) for i in range(n)]

            for vel in vels:
                if getattr(axis, "error", 0) != 0:
                    self.error.emit(f"Axis error during sweep: 0x{axis.error:08X}")
                    break
                axis.controller.input_vel = vel
                self.progress.emit(f"vel = {vel:.2f} t/s…")
                time.sleep(self._dwell_s)

                iq_vals = []
                for _ in range(10):
                    try:
                        iq_vals.append(float(axis.motor.current_control.Iq_measured))
                    except Exception:
                        iq_vals.append(0.0)
                    time.sleep(0.05)

                self.point.emit(vel, sum(iq_vals) / len(iq_vals))

            axis.controller.input_vel = 0
            time.sleep(0.3)
            axis.requested_state = AXIS_STATE_IDLE
            self.done.emit()

        except Exception as e:
            self.error.emit(f"Friction sweep error: {e}")
            if axis is not None:
                try:
                    axis.requested_state = AXIS_STATE_IDLE
                except Exception:
                    pass


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Motor Characterization
# ══════════════════════════════════════════════════════════════════════════════
class MotorCharacterizationTab(QWidget):
    log_message = Signal(str, str)

    def __init__(self, get_odrv, get_axis_idx):
        super().__init__()
        self._get_odrv     = get_odrv
        self._get_axis_idx = get_axis_idx
        self._kv_worker:    KVTestWorker | None        = None
        self._sweep_worker: FrictionSweepWorker | None = None
        self._kv_results:   dict | None                = None
        self._sweep_points: list                       = []
        self._t_buf   = deque(maxlen=PLOT_LEN)
        self._vel_buf = deque(maxlen=PLOT_LEN)
        self._iq_buf  = deque(maxlen=PLOT_LEN)
        self._vq_buf  = deque(maxlen=PLOT_LEN)
        self._t0      = None
        self._build()

    def _odrv(self):
        return self._get_odrv()

    def _axis(self):
        odrv = self._odrv()
        if odrv is None:
            return None
        try:
            return odrv.axis0 if self._get_axis_idx() == 0 else odrv.axis1
        except AttributeError:
            return None

    def _log(self, msg, color=CLR_OK):
        self.log_message.emit(msg, color)

    # ── build UI ───────────────────────────────────────────────────────────────
    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        inner = QWidget()
        scroll.setWidget(inner)
        left = QVBoxLayout(inner)
        left.setSpacing(10)
        left.setContentsMargins(8, 8, 8, 8)
        left.addWidget(self._build_kv_group())
        left.addWidget(self._build_sweep_group())
        left.addStretch()
        splitter.addWidget(scroll)

        splitter.addWidget(self._build_chart_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ── KV group ───────────────────────────────────────────────────────────────
    def _build_kv_group(self):
        box = QGroupBox("KV Measurement (no-load back-EMF)")
        root = QVBoxLayout(box)

        form = QFormLayout()
        form.setSpacing(4)
        self._spin_vel_sb = _dspin(0.5, 20.0, 3.0, dec=1, step=0.5, suffix="t/s")
        self._settle_sb   = _dspin(0.5, 10.0, 2.0, dec=1, step=0.5, suffix="s")
        self._n_samples_sb = QSpinBox()
        self._n_samples_sb.setRange(5, 100)
        self._n_samples_sb.setValue(20)
        self._n_samples_sb.setFixedWidth(80)
        form.addRow("Spin velocity", self._spin_vel_sb)
        form.addRow("Settle time",   self._settle_sb)
        form.addRow("Samples",       self._n_samples_sb)
        root.addLayout(form)

        btn_row = QHBoxLayout()
        self._btn_kv_run = QPushButton("Run KV Test")
        self._btn_kv_run.clicked.connect(self._start_kv_test)
        self._lbl_kv_status = QLabel("Idle")
        self._lbl_kv_status.setStyleSheet(f"color: {CLR_MUTED}; font-family: monospace;")
        btn_row.addWidget(self._btn_kv_run)
        btn_row.addWidget(self._lbl_kv_status, stretch=1)
        root.addLayout(btn_row)

        res_form = QFormLayout()
        res_form.setSpacing(4)

        def _res_lbl():
            l = QLabel("—")
            l.setStyleSheet(f"font-family: monospace; font-size: 13px; color: {CLR_INFO};")
            return l

        self._lbl_kv    = _res_lbl()
        self._lbl_kt    = _res_lbl()
        self._lbl_vq    = _res_lbl()
        self._lbl_iq_kv = _res_lbl()
        res_form.addRow("KV (RPM/V)",         self._lbl_kv)
        res_form.addRow("Kt (Nm/A)",          self._lbl_kt)
        res_form.addRow("V_q — back-EMF (V)", self._lbl_vq)
        res_form.addRow("Iq — no-load (A)",   self._lbl_iq_kv)
        root.addLayout(res_form)

        self._btn_kv_write = QPushButton("Write Kt + recompute gains → ODrive & save file")
        self._btn_kv_write.setEnabled(False)
        self._btn_kv_write.clicked.connect(self._write_kv_results)
        root.addWidget(self._btn_kv_write)

        return box

    # ── Friction sweep group ───────────────────────────────────────────────────
    def _build_sweep_group(self):
        box = QGroupBox("No-Load Friction Sweep (Iq vs velocity)")
        root = QVBoxLayout(box)

        form = QFormLayout()
        form.setSpacing(4)
        self._sw_vel_min = _dspin(0.1, 10.0,  0.5, dec=1, step=0.5, suffix="t/s")
        self._sw_vel_max = _dspin(1.0, 30.0, 10.0, dec=1, step=1.0, suffix="t/s")
        self._sw_steps   = QSpinBox()
        self._sw_steps.setRange(2, 30)
        self._sw_steps.setValue(10)
        self._sw_steps.setFixedWidth(80)
        self._sw_dwell   = _dspin(0.5, 10.0, 1.5, dec=1, step=0.5, suffix="s")
        form.addRow("Vel min",    self._sw_vel_min)
        form.addRow("Vel max",    self._sw_vel_max)
        form.addRow("Steps",      self._sw_steps)
        form.addRow("Dwell/step", self._sw_dwell)
        root.addLayout(form)

        btn_row = QHBoxLayout()
        self._btn_sweep_run = QPushButton("Run Friction Sweep")
        self._btn_sweep_run.clicked.connect(self._start_friction_sweep)
        self._lbl_sweep_status = QLabel("Idle")
        self._lbl_sweep_status.setStyleSheet(f"color: {CLR_MUTED}; font-family: monospace;")
        btn_row.addWidget(self._btn_sweep_run)
        btn_row.addWidget(self._lbl_sweep_status, stretch=1)
        root.addLayout(btn_row)

        self._sweep_table = QTableWidget(0, 2)
        self._sweep_table.setHorizontalHeaderLabels(["Vel (t/s)", "Iq_mean (A)"])
        self._sweep_table.horizontalHeader().setStretchLastSection(True)
        self._sweep_table.setMaximumHeight(150)
        self._sweep_table.setEditTriggers(QTableWidget.NoEditTriggers)
        root.addWidget(self._sweep_table)

        self._btn_sweep_save = QPushButton("Save sweep to motor_characterization.py")
        self._btn_sweep_save.setEnabled(False)
        self._btn_sweep_save.clicked.connect(self._save_sweep)
        root.addWidget(self._btn_sweep_save)

        return box

    # ── Chart panel ────────────────────────────────────────────────────────────
    def _build_chart_panel(self):
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setContentsMargins(6, 6, 6, 6)
        col.setSpacing(4)

        if _HAS_CHARTS:
            self._chart = QChart()
            self._chart.setBackgroundBrush(QColor(CLR_PANEL))
            self._chart.setPlotAreaBackgroundBrush(QColor("#1e1e1e"))
            self._chart.setPlotAreaBackgroundVisible(True)
            self._chart.legend().setVisible(False)
            self._chart.layout().setContentsMargins(0, 0, 0, 0)
            self._chart.setMargins(QMargins(4, 4, 4, 4))

            self._ser_vel      = QLineSeries()
            self._ser_iq_t     = QLineSeries()
            self._ser_vq       = QLineSeries()
            self._ser_friction = QLineSeries()
            self._ser_vel.setColor(QColor(CLR_VEL))
            self._ser_iq_t.setColor(QColor(CLR_IQ))
            self._ser_vq.setColor(QColor(CLR_VQ))
            self._ser_friction.setColor(QColor(CLR_WARN))

            for s in (self._ser_vel, self._ser_iq_t, self._ser_vq, self._ser_friction):
                self._chart.addSeries(s)

            self._ax_x = QValueAxis()
            self._ax_x.setLabelsColor(QColor(CLR_LABEL))
            self._ax_x.setGridLineColor(QColor("#3a3a3a"))
            self._ax_x.setTitleText("t (s)")
            self._ax_x.setTitleBrush(QColor(CLR_MUTED))

            self._ax_y = QValueAxis()
            self._ax_y.setLabelsColor(QColor(CLR_LABEL))
            self._ax_y.setGridLineColor(QColor("#3a3a3a"))

            self._chart.addAxis(self._ax_x, Qt.AlignBottom)
            self._chart.addAxis(self._ax_y, Qt.AlignLeft)
            for s in (self._ser_vel, self._ser_iq_t, self._ser_vq, self._ser_friction):
                s.attachAxis(self._ax_x)
                s.attachAxis(self._ax_y)

            self._ser_friction.setVisible(False)

            chart_view = QChartView(self._chart)
            chart_view.setRenderHint(QPainter.Antialiasing)
            chart_view.setMinimumHeight(250)
            col.addWidget(chart_view, stretch=1)
        else:
            self._chart = None
            no_chart = QLabel("PySide6.QtCharts not available.\npip install PySide6-Addons")
            no_chart.setAlignment(Qt.AlignCenter)
            no_chart.setStyleSheet(f"color: {CLR_MUTED};")
            col.addWidget(no_chart, stretch=1)

        legend_row = QHBoxLayout()
        for label, color in (("● Vel", CLR_VEL), ("● Iq", CLR_IQ),
                              ("● V_q", CLR_VQ), ("● Friction Iq", CLR_WARN)):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {color}; font-family: monospace;")
            legend_row.addWidget(lbl)
        legend_row.addStretch()
        self._btn_clear_chart = QPushButton("Clear")
        self._btn_clear_chart.setFixedWidth(60)
        self._btn_clear_chart.clicked.connect(self._clear_chart)
        legend_row.addWidget(self._btn_clear_chart)
        col.addLayout(legend_row)

        return panel

    def _clear_chart(self):
        self._t_buf.clear()
        self._vel_buf.clear()
        self._iq_buf.clear()
        self._vq_buf.clear()
        self._t0 = None
        if _HAS_CHARTS and self._chart:
            for s in (self._ser_vel, self._ser_iq_t, self._ser_vq, self._ser_friction):
                s.replace([])

    # ── KV test ────────────────────────────────────────────────────────────────
    def _start_kv_test(self):
        odrv = self._odrv()
        if odrv is None:
            _colored(self._lbl_kv_status, "Not connected", CLR_WARN)
            return
        if self._kv_worker and self._kv_worker.isRunning():
            return

        self._clear_chart()
        if _HAS_CHARTS and self._chart:
            self._ser_vel.setVisible(True)
            self._ser_iq_t.setVisible(True)
            self._ser_vq.setVisible(True)
            self._ser_friction.setVisible(False)
            self._ax_x.setTitleText("t (s)")

        self._btn_kv_run.setEnabled(False)
        self._btn_kv_write.setEnabled(False)
        _colored(self._lbl_kv_status, "Running…", CLR_INFO)

        self._kv_worker = KVTestWorker(
            odrv, self._get_axis_idx(),
            self._spin_vel_sb.value(),
            self._settle_sb.value(),
            self._n_samples_sb.value(),
        )
        self._kv_worker.progress.connect(lambda msg: _colored(self._lbl_kv_status, msg, CLR_INFO))
        self._kv_worker.sample.connect(self._on_kv_sample)
        self._kv_worker.result.connect(self._on_kv_result)
        self._kv_worker.error.connect(self._on_kv_error)
        self._kv_worker.start()

    def _on_kv_sample(self, vel, iq, vq):
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        t = now - self._t0
        self._t_buf.append(t)
        self._vel_buf.append(vel)
        self._iq_buf.append(iq)
        self._vq_buf.append(vq)
        if not (_HAS_CHARTS and self._chart):
            return
        self._ser_vel.replace([QPointF(x, y) for x, y in zip(self._t_buf, self._vel_buf)])
        self._ser_iq_t.replace([QPointF(x, y) for x, y in zip(self._t_buf, self._iq_buf)])
        self._ser_vq.replace([QPointF(x, y) for x, y in zip(self._t_buf, self._vq_buf)])
        if len(self._t_buf) > 1:
            self._ax_x.setRange(self._t_buf[0], self._t_buf[-1])
            all_v = list(self._vel_buf) + list(self._iq_buf) + list(self._vq_buf)
            mn, mx = min(all_v), max(all_v)
            margin = max(0.5, (mx - mn) * 0.1)
            self._ax_y.setRange(mn - margin, mx + margin)

    def _on_kv_result(self, res):
        self._kv_results = res
        self._btn_kv_run.setEnabled(True)
        _colored(self._lbl_kv_status, "Done", CLR_OK)
        self._lbl_kv.setText(f"{res['KV']:.2f}")
        self._lbl_kt.setText(f"{res['Kt']:.5f}")
        self._lbl_vq.setText(f"{res['vq_mean']:.4f}")
        self._lbl_iq_kv.setText(f"{res['iq_mean']:.4f}")
        self._btn_kv_write.setEnabled(True)
        self._log(
            f"KV={res['KV']:.2f} RPM/V  Kt={res['Kt']:.5f} Nm/A  "
            f"V_q={res['vq_mean']:.4f} V  Iq={res['iq_mean']:.4f} A", CLR_OK
        )

    def _on_kv_error(self, msg):
        self._btn_kv_run.setEnabled(True)
        _colored(self._lbl_kv_status, "Error", CLR_ERR)
        self._log(f"KV test error: {msg}", CLR_ERR)

    # ── Friction sweep ─────────────────────────────────────────────────────────
    def _start_friction_sweep(self):
        odrv = self._odrv()
        if odrv is None:
            _colored(self._lbl_sweep_status, "Not connected", CLR_WARN)
            return
        if self._sweep_worker and self._sweep_worker.isRunning():
            return

        self._sweep_points.clear()
        self._sweep_table.setRowCount(0)
        self._btn_sweep_save.setEnabled(False)

        if _HAS_CHARTS and self._chart:
            self._ser_vel.setVisible(False)
            self._ser_iq_t.setVisible(False)
            self._ser_vq.setVisible(False)
            self._ser_friction.setVisible(True)
            self._ser_friction.replace([])
            self._ax_x.setTitleText("vel (t/s)")

        self._btn_sweep_run.setEnabled(False)
        _colored(self._lbl_sweep_status, "Running…", CLR_INFO)

        self._sweep_worker = FrictionSweepWorker(
            odrv, self._get_axis_idx(),
            self._sw_vel_min.value(), self._sw_vel_max.value(),
            self._sw_steps.value(),   self._sw_dwell.value(),
        )
        self._sweep_worker.progress.connect(lambda msg: _colored(self._lbl_sweep_status, msg, CLR_INFO))
        self._sweep_worker.point.connect(self._on_sweep_point)
        self._sweep_worker.done.connect(self._on_sweep_done)
        self._sweep_worker.error.connect(self._on_sweep_error)
        self._sweep_worker.start()

    def _on_sweep_point(self, vel, iq_mean):
        self._sweep_points.append((vel, iq_mean))
        row = self._sweep_table.rowCount()
        self._sweep_table.insertRow(row)
        self._sweep_table.setItem(row, 0, QTableWidgetItem(f"{vel:.3f}"))
        self._sweep_table.setItem(row, 1, QTableWidgetItem(f"{iq_mean:.4f}"))
        if not (_HAS_CHARTS and self._chart) or not self._sweep_points:
            return
        pts  = [QPointF(v, iq) for v, iq in self._sweep_points]
        self._ser_friction.replace(pts)
        vels = [p[0] for p in self._sweep_points]
        iqs  = [p[1] for p in self._sweep_points]
        self._ax_x.setRange(min(vels) - 0.1, max(vels) + 0.1)
        mn, mx = min(iqs), max(iqs)
        margin = max(0.05, (mx - mn) * 0.15)
        self._ax_y.setRange(mn - margin, mx + margin)

    def _on_sweep_done(self):
        self._btn_sweep_run.setEnabled(True)
        _colored(self._lbl_sweep_status, "Done", CLR_OK)
        self._btn_sweep_save.setEnabled(True)
        self._log(f"Friction sweep done: {len(self._sweep_points)} points", CLR_OK)

    def _on_sweep_error(self, msg):
        self._btn_sweep_run.setEnabled(True)
        _colored(self._lbl_sweep_status, "Error", CLR_ERR)
        self._log(f"Friction sweep error: {msg}", CLR_ERR)

    # ── Write / save ───────────────────────────────────────────────────────────
    def _write_kv_results(self):
        axis = self._axis()
        if axis is None or self._kv_results is None:
            return
        res = self._kv_results
        Kt  = res["Kt"]

        try:
            phase_r = float(axis.motor.config.phase_resistance)
        except Exception:
            phase_r = 0.0
        try:
            phase_l = float(axis.motor.config.phase_inductance)
        except Exception:
            phase_l = 0.0

        errors = []
        try:
            axis.motor.config.torque_constant = Kt
        except Exception as e:
            errors.append(f"torque_constant: {e}")

        vel_gain = vel_int_gain = None
        if phase_r > 1e-6:
            vel_gain    = 0.5 * Kt / phase_r
            vel_int_gain = 0.5 * vel_gain
            try:
                axis.controller.config.vel_gain            = vel_gain
                axis.controller.config.vel_integrator_gain = vel_int_gain
            except Exception as e:
                errors.append(f"gains: {e}")
        else:
            errors.append("phase_resistance not calibrated — gains not recomputed")

        if errors:
            self._log("Write warnings: " + "; ".join(errors), CLR_WARN)
        else:
            self._log(
                f"Wrote Kt={Kt:.5f} Nm/A, vel_gain={vel_gain:.4f}, "
                f"vel_int_gain={vel_int_gain:.4f} → Ax{self._get_axis_idx()}", CLR_OK
            )
        self._write_char_file(res, phase_r, phase_l, vel_gain, vel_int_gain)

    def _save_sweep(self):
        self._write_char_file(None, None, None, None, None)

    def _write_char_file(self, kv_res, phase_r, phase_l, vel_gain, vel_int_gain):
        out_path = _HERE / "motor_characterization.py"

        existing = {}
        if out_path.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("_mc_existing", out_path)
                mod  = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for key in ("MOTOR_KV_RPM_V", "MOTOR_KT_NM_A", "MOTOR_PHASE_R", "MOTOR_PHASE_L",
                            "MOTOR_VEL_GAIN", "MOTOR_VEL_INT_GAIN", "FRICTION_SWEEP"):
                    if hasattr(mod, key):
                        existing[key] = getattr(mod, key)
            except Exception:
                pass

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        ax  = self._get_axis_idx()

        if kv_res is not None:
            existing["MOTOR_KV_RPM_V"] = round(kv_res["KV"], 4)
            existing["MOTOR_KT_NM_A"]  = round(kv_res["Kt"], 6)
        if phase_r is not None and phase_r > 1e-6:
            existing["MOTOR_PHASE_R"] = round(phase_r, 6)
        if phase_l is not None and phase_l > 1e-9:
            existing["MOTOR_PHASE_L"] = round(phase_l, 9)
        if vel_gain is not None:
            existing["MOTOR_VEL_GAIN"]     = round(vel_gain, 6)
            existing["MOTOR_VEL_INT_GAIN"] = round(vel_int_gain, 6)
        if self._sweep_points:
            existing["FRICTION_SWEEP"] = list(self._sweep_points)

        def _v(key):
            return repr(existing[key]) if key in existing else "None"

        lines = [
            "# Auto-generated by odrive_gui_v2.py — Motor Characterization Results",
            f"# Last updated: {now}   Axis: {ax}",
            "",
            f"MOTOR_KV_RPM_V     = {_v('MOTOR_KV_RPM_V'):<14}  # RPM/V (no-load back-EMF test)",
            f"MOTOR_KT_NM_A      = {_v('MOTOR_KT_NM_A'):<14}  # Nm/A  (= 60 / (2π × KV))",
            f"MOTOR_PHASE_R      = {_v('MOTOR_PHASE_R'):<14}  # Ω  (from ODrive calibration)",
            f"MOTOR_PHASE_L      = {_v('MOTOR_PHASE_L'):<14}  # H  (from ODrive calibration)",
            f"MOTOR_VEL_GAIN     = {_v('MOTOR_VEL_GAIN'):<14}  # computed: 0.5 × Kt / R",
            f"MOTOR_VEL_INT_GAIN = {_v('MOTOR_VEL_INT_GAIN'):<14}  # computed: 0.5 × vel_gain",
            "",
            "# No-load friction sweep: [(vel_turns_per_s, Iq_A), ...]",
            "FRICTION_SWEEP = [",
        ]
        for v, iq in existing.get("FRICTION_SWEEP", []):
            lines.append(f"    ({v:.4f}, {iq:.6f}),")
        lines += ["]", ""]

        try:
            out_path.write_text("\n".join(lines), encoding="utf-8")
            self._log(f"Saved: {out_path}", CLR_OK)
        except Exception as e:
            self._log(f"File write error: {e}", CLR_ERR)

    # ── poll update (called by MainWindow every POLL_MS) ───────────────────────
    def update(self, odrv, axis_idx):
        pass   # workers drive their own state; no continuous polling needed


# ══════════════════════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ODrive v2  —  Motor Tuning")
        self.resize(900, 700)

        self._odrv: object = None
        self._connect_worker: ConnectWorker | None = None

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_MS)
        self._poll_timer.timeout.connect(self._poll)

        self._build()
        self._poll_timer.start()

    # ── UI build ───────────────────────────────────────────────────────────────
    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # ── top bar ────────────────────────────────────────────────────────────
        top = QHBoxLayout()

        # Axis selector
        top.addWidget(QLabel("Axis:"))
        self._ax_grp = QButtonGroup(self)
        self._rb_ax  = []
        for i in range(2):
            rb = QRadioButton(f"  {i}  ")
            rb.setChecked(i == 0)
            self._ax_grp.addButton(rb, i)
            top.addWidget(rb)
            self._rb_ax.append(rb)
        self._ax_grp.idClicked.connect(self._on_axis_changed)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
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

        root.addLayout(top)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #444;")
        root.addWidget(line)

        # ── tabs ───────────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.tab_setup   = MotorSetupTab(lambda: self._odrv, lambda: self._ax_idx())
        self.tab_control = MotorControlTab(lambda: self._odrv, lambda: self._ax_idx())
        self.tab_term    = TerminalTab(lambda: self._odrv, lambda: self._ax_idx())
        self.tab_char    = MotorCharacterizationTab(lambda: self._odrv, lambda: self._ax_idx())

        self.tabs.addTab(self.tab_setup,   "1 · Motor Setup")
        self.tabs.addTab(self.tab_control, "2 · Motor Control")
        self.tabs.addTab(self.tab_term,    "3 · Terminal")
        self.tabs.addTab(self.tab_char,    "4 · Characterization")

        # Wire log signals → status label for now
        self.tab_setup.log_message.connect(self._on_log)
        self.tab_control.log_message.connect(self._on_log)
        self.tab_char.log_message.connect(self._on_log)
        self.tab_control.reboot_requested.connect(self.tab_setup._start_reboot_sequence)

        # Auto-read gains when switching to Motor Control tab
        self.tabs.currentChanged.connect(self._on_tab_changed)

    # ── axis helper ────────────────────────────────────────────────────────────
    def _ax_idx(self) -> int:
        return self._ax_grp.checkedId()

    def _on_axis_changed(self, idx: int):
        pass  # tabs read _ax_idx() live on each poll

    def _on_tab_changed(self, index: int):
        if index == 1 and self._odrv is not None:
            self.tab_control._read_gains()

    # ── connection ─────────────────────────────────────────────────────────────
    def _toggle_connect(self):
        if self._odrv is not None:
            self._odrv = None
            self.btn_connect.setText("Connect")
            self._set_status("Disconnected", CLR_MUTED)
            return

        self.btn_connect.setEnabled(False)
        self._set_status("Searching…", CLR_INFO)
        self._connect_worker = ConnectWorker()
        self._connect_worker.success.connect(self._on_connect_ok)
        self._connect_worker.failed.connect(self._on_connect_fail)
        self._connect_worker.start()

    def _on_connect_ok(self, odrv):
        self._odrv = odrv
        hw = f"hw{odrv.hw_version_major}.{odrv.hw_version_minor}.{odrv.hw_version_variant}"
        fw_unreleased = getattr(odrv, "fw_version_unreleased", 1)
        fw = f"fw{odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}"
        fw_tag = f"{fw} (dev)" if fw_unreleased else fw
        self._set_status(f"Connected — {hw}  {fw_tag}", CLR_OK)
        self.btn_connect.setText("Disconnect")
        self.btn_connect.setEnabled(True)
        # Auto-read config into Tab 1
        self.tab_setup._read_config()

    def _on_connect_fail(self, msg):
        self._set_status(f"Failed: {msg}", CLR_ERR)
        self.btn_connect.setText("Connect")
        self.btn_connect.setEnabled(True)

    def _set_status(self, msg: str, color: str):
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet(f"color: {color}; font-family: monospace;")

    # ── log handler ────────────────────────────────────────────────────────────
    def _on_log(self, text: str, color: str):
        if text == "__DISCONNECTED__":
            self._odrv = None
            self.btn_connect.setText("Connect")
            self._set_status("Disconnected (rebooting…)", CLR_WARN)
            return
        if text.startswith("__RECONNECTED__:"):
            # Tab 1 already found + verified the new odrv — grab it directly
            odrv = getattr(self.tab_setup, "_reconnected_odrv", None)
            if odrv is not None:
                self._odrv = odrv
                hw = f"hw{odrv.hw_version_major}.{odrv.hw_version_minor}.{odrv.hw_version_variant}"
                self._set_status(f"Reconnected — {hw}", CLR_OK)
            else:
                self._set_status("Reconnected", CLR_OK)
            self.btn_connect.setText("Disconnect")
            self.btn_connect.setEnabled(True)
            return
        # Normal log → update status bar + append to terminal
        self._set_status(text[:80], color)
        self.tab_term.log(text, color)

    def _on_reconnect_after_reboot(self, odrv):
        self._odrv = odrv
        hw = f"hw{odrv.hw_version_major}.{odrv.hw_version_minor}.{odrv.hw_version_variant}"
        self._set_status(f"Reconnected — {hw}", CLR_OK)
        self.btn_connect.setText("Disconnect")
        self.btn_connect.setEnabled(True)

    # ── poll ───────────────────────────────────────────────────────────────────
    def _poll(self):
        try:
            ax = self._ax_idx()
            self.tab_setup.update(self._odrv, ax)
            self.tab_control.update(self._odrv, ax)
            self.tab_term.update(self._odrv, ax)
            self.tab_char.update(self._odrv, ax)
        except Exception:
            # USB disconnect or stale odrv — go to disconnected state
            self._odrv = None
            self.btn_connect.setText("Connect")
            self._set_status("Disconnected (USB lost)", CLR_ERR)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
