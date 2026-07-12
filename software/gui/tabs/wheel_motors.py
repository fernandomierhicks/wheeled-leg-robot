"""wheel_motors.py — Wheel Motors tab (ODrive, CAN, node 0 = L / node 1 = R).

Displays live velocity + position for both wheel axes. Controls wheel mode
(IDLE / VELOCITY / POSITION / TORQUE), per-wheel setpoints, sine/square test
waveforms, error clearing, and a differential drive spin-in-place slider.

Telemetry fields used (TelemetryPayload V3, comm_protocol.h):
    wm_l_vel_turns_s, wm_r_vel_turns_s   — wheel velocity [turns/s]
    wm_l_pos_turns, wm_r_pos_turns       — wheel position [turns]
    wm_l_vbus, wm_r_vbus                 — ODrive bus voltage [V]
    wm_l_error, wm_r_error               — ODrive Axis_Error bitmask
    wm_l_state, wm_r_state               — ODrive Axis_State (1=IDLE, 8=CLOSED_LOOP)
    wm_mode                              — current WheelMode (0-3)
"""

import math
import time
from collections import deque

import pyqtgraph as pg
pg.setConfigOptions(antialias=False)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

from .telemetry_bus import TelemetryBus
from .theme import BG, BLUE, BORDER, DIM, GREEN, ORANGE, RED, SURFACE, TEXT
from .comm_commands import (
    send_set_mode, send_wheel_set_mode, send_wheel_setpoint, send_wheel_clear_errors,
    WHEEL_MODE_IDLE, WHEEL_MODE_VELOCITY, WHEEL_MODE_POSITION, WHEEL_MODE_TORQUE,
    STATE_MANUAL, STATE_STANDBY,
)

_BUF = 750  # rolling chart samples (~15 s at 50 Hz telemetry)

# ODrive Axis_State enum values
_AXIS_STATE_NAMES = {
    1: "IDLE",
    2: "STARTUP_SEQ",
    3: "FULL_CAL",
    4: "MOTOR_CAL",
    6: "ENCODER_INDEX",
    7: "ENCODER_OFFSET",
    8: "CLOSED_LOOP",
    11: "LOCKIN_SPIN",
    12: "ENCODER_DIR",
    13: "HOMING",
    14: "ENCODER_HALL_POL",
    15: "ENCODER_HALL_DIR",
}

# ODrive Axis_Error bitmask (v0.5.x)
_AXIS_ERROR_BITS: list[tuple[int, str]] = [
    (0x00000001, "INVALID_STATE"),
    (0x00000002, "DC_BUS_UNDER_VOLTAGE"),
    (0x00000004, "DC_BUS_OVER_VOLTAGE"),
    (0x00000008, "CURRENT_MEAS_TIMEOUT"),
    (0x00000010, "BRAKE_RESISTOR_DISARMED"),
    (0x00000020, "MOTOR_DISARMED"),
    (0x00000040, "MOTOR_FAILED"),
    (0x00000080, "SENSORLESS_EST_FAILED"),
    (0x00000100, "ENCODER_FAILED"),
    (0x00000200, "CONTROLLER_FAILED"),
    (0x00000800, "WATCHDOG_TIMER_EXPIRED"),
    (0x00004000, "ESTOP_REQUESTED"),
    (0x00400000, "OVER_TEMP"),
]

_WHEEL_MODE_NAMES = {
    WHEEL_MODE_IDLE:     "IDLE",
    WHEEL_MODE_VELOCITY: "VELOCITY",
    WHEEL_MODE_POSITION: "POSITION",
    WHEEL_MODE_TORQUE:   "TORQUE",
}

# Setpoint range/step by mode
_SETPOINT_CFG = {
    WHEEL_MODE_VELOCITY: (-20.0,  20.0,  0.1,  0.0, 2, "turns/s"),
    WHEEL_MODE_POSITION: (-100.0, 100.0, 1.0,  0.0, 2, "turns"),
    WHEEL_MODE_TORQUE:   (-5.0,   5.0,   0.05, 0.0, 3, "N·m"),
}


# ── Small UI helpers ──────────────────────────────────────────────────────────

def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f


def _readout(parent_lay: QVBoxLayout, label: str, color: str = TEXT) -> QLabel:
    row = QHBoxLayout()
    row.setSpacing(6)
    k = QLabel(label + ":")
    k.setStyleSheet(f"color: {DIM}; font-size: 11px;")
    v = QLabel("—")
    v.setStyleSheet(
        f"color: {color}; font-size: 13px; font-weight: bold; font-family: Consolas;"
    )
    row.addWidget(k)
    row.addStretch()
    row.addWidget(v)
    parent_lay.addLayout(row)
    return v


def _spin(lo: float, hi: float, step: float, val: float, dec: int) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(val)
    s.setDecimals(dec)
    s.setFixedWidth(80)
    s.setStyleSheet(
        f"QDoubleSpinBox{{background:{BG};color:{TEXT};"
        f"font-family:Consolas;font-size:10px;"
        f"border:1px solid {BORDER};border-radius:3px;padding:1px 4px}}"
        f"QDoubleSpinBox::up-button,QDoubleSpinBox::down-button{{width:14px}}"
        f"QDoubleSpinBox:disabled{{background:{SURFACE};color:{DIM}}}"
    )
    return s


def _toggle_btn(label: str, bg: str) -> QPushButton:
    b = QPushButton(label)
    b.setCheckable(True)
    b.setStyleSheet(
        f"QPushButton{{background:{SURFACE};color:{TEXT};"
        f"border:1px solid {BORDER};border-radius:3px;padding:4px 10px}}"
        f"QPushButton:hover{{background:{BORDER}}}"
        f"QPushButton:checked{{background:{bg};color:white;font-weight:bold}}"
        f"QPushButton:disabled{{background:{SURFACE};color:{DIM};"
        f"border:1px solid {BORDER}}}"
    )
    return b


def _decode_errors(word: int) -> str:
    if word == 0:
        return "OK"
    flags = [name for mask, name in _AXIS_ERROR_BITS if word & mask]
    return " | ".join(flags) if flags else f"0x{word:08X}"


# ── Per-wheel panel ───────────────────────────────────────────────────────────

class _WheelPanel(QWidget):
    """Live readout + mini chart + controls for one ODrive wheel axis."""

    def __init__(self, side: str, node_id: int, send_fn):
        super().__init__()
        self._side = side
        self._send_fn = send_fn  # callable(fw_val: float) → sends setpoint for this wheel only
        self._vel_buf:  deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._cmd_buf:  deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._latest_vel  = 0.0
        self._latest_pos  = 0.0
        self._latest_vbus = 0.0
        self._latest_err  = 0
        self._latest_state_id = 0
        self._latest_cmd  = 0.0
        self._in_manual   = False
        self._wm_mode     = WHEEL_MODE_IDLE

        self.setObjectName("WheelPanel")
        self.setStyleSheet(
            f"#WheelPanel {{background: {SURFACE}; border: 1px solid {BORDER};"
            f" border-radius: 4px;}}"
            f"QLabel, QPushButton, QDoubleSpinBox {{border: none;}}"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        # Header
        hdr = QLabel(
            f"<b>Wheel {side}</b>"
            f"  <span style='color:{DIM};font-size:10px'>(node {node_id})</span>"
        )
        hdr.setStyleSheet(f"color: {TEXT}; font-size: 12px;")
        lay.addWidget(hdr)
        lay.addWidget(_hline())

        # Live readouts
        self._lbl_vel   = _readout(lay, "Velocity", BLUE)
        self._lbl_pos   = _readout(lay, "Position", TEXT)
        self._lbl_vbus  = _readout(lay, "Vbus",     GREEN)
        self._lbl_state = _readout(lay, "State",    DIM)
        self._lbl_error = _readout(lay, "Error",    GREEN)
        lay.addWidget(_hline())

        # Mini chart — velocity (blue) + commanded setpoint (orange dashed)
        self._chart = pg.PlotWidget()
        self._chart.setBackground(BG)
        self._chart.setMaximumHeight(110)
        self._chart.setMinimumHeight(80)
        self._chart.showGrid(x=True, y=True, alpha=0.12)
        self._chart.setXRange(0, _BUF)
        self._chart.getAxis("bottom").setStyle(showValues=False)
        leg = self._chart.addLegend(offset=(4, 4), verSpacing=-4)
        self._crv_vel = self._chart.plot(
            list(self._vel_buf),
            pen=pg.mkPen(BLUE, width=1.5),
            name="vel (t/s)",
        )
        self._crv_cmd = self._chart.plot(
            list(self._cmd_buf),
            pen=pg.mkPen(ORANGE, width=1.2, style=Qt.PenStyle.DashLine),
            name="cmd",
        )
        lay.addWidget(self._chart)
        lay.addWidget(_hline())

        # ── Controls (enabled only in MANUAL) ────────────────────────────────
        self._ctrl = QWidget()
        ctrl_lay = QVBoxLayout(self._ctrl)
        ctrl_lay.setContentsMargins(0, 0, 0, 0)
        ctrl_lay.setSpacing(6)

        # Setpoint row
        sp_row = QHBoxLayout()
        sp_row.setSpacing(4)
        sp_lbl = QLabel("Setpoint:")
        sp_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        self._sp_set = _spin(-20.0, 20.0, 0.1, 0.0, 2)
        self._lbl_sp_unit = QLabel("turns/s")
        self._lbl_sp_unit.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        btn_send = QPushButton("Send")
        btn_send.setStyleSheet(
            f"QPushButton{{background:#1a4a7a;color:white;"
            f"border:1px solid {BORDER};border-radius:3px;padding:3px 10px}}"
            f"QPushButton:hover{{background:#2a5a8a}}"
            f"QPushButton:disabled{{background:{SURFACE};color:{DIM};"
            f"border:1px solid {BORDER}}}"
        )
        btn_send.clicked.connect(self._on_send_clicked)
        sp_row.addWidget(sp_lbl)
        sp_row.addWidget(self._sp_set)
        sp_row.addWidget(self._lbl_sp_unit)
        sp_row.addWidget(btn_send)
        sp_row.addStretch()
        ctrl_lay.addLayout(sp_row)

        # Test wave row
        ctrl_lay.addWidget(_hline())
        wave_row = QHBoxLayout()
        wave_row.setSpacing(4)
        wlbl = QLabel("Wave:")
        wlbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        wave_row.addWidget(wlbl)
        amp_lbl = QLabel("Amp")
        amp_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        self._sp_amp  = _spin(0.0, 20.0, 0.1, 1.0, 2)
        freq_lbl = QLabel("Hz")
        freq_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        self._sp_freq = _spin(0.05, 5.0, 0.05, 0.2, 2)
        wave_row.addWidget(amp_lbl)
        wave_row.addWidget(self._sp_amp)
        wave_row.addWidget(freq_lbl)
        wave_row.addWidget(self._sp_freq)
        self._btn_sine   = _toggle_btn("Sine",   "#1a4a7a")
        self._btn_square = _toggle_btn("Square", "#1a4a7a")
        self._btn_sine.clicked.connect(lambda: self._toggle_wave("sine"))
        self._btn_square.clicked.connect(lambda: self._toggle_wave("square"))
        wave_row.addWidget(self._btn_sine)
        wave_row.addWidget(self._btn_square)
        wave_row.addStretch()
        ctrl_lay.addLayout(wave_row)

        # Wave timer
        self._wave_type    = None
        self._wave_t0      = 0.0
        self._wave_center  = 0.0
        self._wave_timer   = QTimer(self)
        self._wave_timer.timeout.connect(self._wave_tick)

        self._ctrl.setEnabled(False)
        lay.addWidget(self._ctrl)

        # Clear Errors (always enabled)
        ctrl_lay.addWidget(_hline())
        btn_clr = QPushButton("Clear Errors")
        btn_clr.setStyleSheet(
            f"QPushButton{{background:{SURFACE};color:{DIM};"
            f"border:1px solid {BORDER};border-radius:3px;padding:3px 10px}}"
            f"QPushButton:hover{{background:{BORDER};color:{TEXT}}}"
        )
        btn_clr.clicked.connect(send_wheel_clear_errors)
        ctrl_lay.addWidget(btn_clr)

        lay.addStretch()

        # Redraw at 20 Hz, decoupled from telemetry rate
        self._redraw_timer = QTimer(self)
        self._redraw_timer.timeout.connect(self._redraw)
        self._redraw_timer.start(50)

    # ── public API ────────────────────────────────────────────────────────────

    def set_controls_enabled(self, enabled: bool):
        self._ctrl.setEnabled(enabled)
        self._in_manual = enabled
        if not enabled:
            self._stop_wave()

    def set_mode(self, mode: int):
        self._wm_mode = mode
        cfg = _SETPOINT_CFG.get(mode)
        if cfg:
            lo, hi, step, val, dec, unit = cfg
            self._sp_set.blockSignals(True)
            self._sp_set.setRange(lo, hi)
            self._sp_set.setSingleStep(step)
            self._sp_set.setDecimals(dec)
            self._sp_set.setValue(val)
            self._sp_set.blockSignals(False)
            self._lbl_sp_unit.setText(unit)

    def update_data(self, vel: float, pos: float, vbus: float,
                    error: int, state_id: int):
        self._latest_vel      = vel
        self._latest_pos      = pos
        self._latest_vbus     = vbus
        self._latest_err      = error
        self._latest_state_id = state_id
        self._vel_buf.append(vel)
        self._cmd_buf.append(self._latest_cmd)

    # ── commands ──────────────────────────────────────────────────────────────

    def _to_fw_units(self, val: float) -> float:
        """Convert spinbox value (turns/s or turns) to firmware units (rad/s or rad)."""
        if self._wm_mode in (WHEEL_MODE_VELOCITY, WHEEL_MODE_POSITION):
            return val * (2 * math.pi)
        return val  # TORQUE: N·m passed through unchanged

    def _send_setpoint(self):
        val = self._sp_set.value()
        self._latest_cmd = val
        fw = self._to_fw_units(val)
        self._send_fn(fw)

    def _on_send_clicked(self):
        self._stop_wave()
        self._send_setpoint()

    # ── wave generator ────────────────────────────────────────────────────────

    def _toggle_wave(self, kind: str):
        if self._wave_type == kind:
            self._stop_wave()
            return
        self._wave_type   = kind
        self._wave_center = (self._latest_pos if self._wm_mode == WHEEL_MODE_POSITION
                             else 0.0)
        self._wave_t0     = time.monotonic()
        self._btn_sine.setChecked(kind == "sine")
        self._btn_square.setChecked(kind == "square")
        self._wave_timer.start(50)  # 20 Hz

    def _stop_wave(self):
        self._wave_timer.stop()
        self._wave_type = None
        self._btn_sine.setChecked(False)
        self._btn_square.setChecked(False)

    def _wave_tick(self):
        t     = time.monotonic() - self._wave_t0
        freq  = self._sp_freq.value()
        amp   = self._sp_amp.value()
        phase = 2 * math.pi * freq * t
        if self._wave_type == "sine":
            offset = amp * math.sin(phase)
        else:
            offset = amp if math.sin(phase) >= 0 else -amp
        val = self._wave_center + offset
        self._sp_set.setValue(val)
        self._latest_cmd = val
        self._send_fn(self._to_fw_units(val))

    # ── redraw ────────────────────────────────────────────────────────────────

    def _redraw(self):
        # Velocity with RPM
        rpm = self._latest_vel * 60.0
        self._lbl_vel.setText(f"{self._latest_vel:+.3f} t/s  ({rpm:+.1f} rpm)")

        # Position
        self._lbl_pos.setText(f"{self._latest_pos:+.3f} turns")

        # Vbus with colour
        vbus_color = GREEN if self._latest_vbus > 20.0 else ORANGE
        self._lbl_vbus.setStyleSheet(
            f"color: {vbus_color}; font-size: 13px; font-weight: bold; font-family: Consolas;"
        )
        self._lbl_vbus.setText(f"{self._latest_vbus:.2f} V")

        # Axis state
        state_str = _AXIS_STATE_NAMES.get(self._latest_state_id,
                                          str(self._latest_state_id))
        self._lbl_state.setText(state_str)

        # Error
        err_str   = _decode_errors(self._latest_err)
        err_color = RED if self._latest_err else GREEN
        self._lbl_error.setStyleSheet(
            f"color: {err_color}; font-size: 13px; font-weight: bold; font-family: Consolas;"
        )
        self._lbl_error.setText(err_str)

        # Charts
        self._crv_vel.setData(list(self._vel_buf))
        self._crv_cmd.setData(list(self._cmd_buf))

        # Auto-fit Y every 5th redraw
        if not hasattr(self, "_rd_count"):
            self._rd_count = 0
        self._rd_count += 1
        if self._rd_count % 5 == 0:
            all_vals = list(self._vel_buf) + list(self._cmd_buf)
            lo, hi = min(all_vals), max(all_vals)
            span = max(hi - lo, 1.0)
            mid  = (lo + hi) / 2
            self._chart.setYRange(mid - span * 0.6, mid + span * 0.6, padding=0.05)


# ── Main tab ──────────────────────────────────────────────────────────────────

class WheelMotorsTab(QWidget):
    def __init__(self):
        super().__init__()

        self._fw_L: float = 0.0  # last firmware setpoint sent to left wheel
        self._fw_R: float = 0.0  # last firmware setpoint sent to right wheel
        self._last_state_id: int | None = None
        self._wm_mode = WHEEL_MODE_IDLE
        self._vbus_L_buf: deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._vbus_R_buf: deque = deque([0.0] * _BUF, maxlen=_BUF)

        self._panel_L = _WheelPanel("L", 0, self._send_L)
        self._panel_R = _WheelPanel("R", 1, self._send_R)

        # ── Mode bar ─────────────────────────────────────────────────────────
        mode_bar = QWidget()
        mode_bar.setObjectName("WModeBar")
        mode_bar.setFixedHeight(36)
        mode_bar.setStyleSheet(
            f"#WModeBar {{background: {SURFACE}; border: 1px solid {BORDER};"
            f" border-radius: 4px;}}"
            f"QLabel, QPushButton {{border: none;}}"
        )
        mb = QHBoxLayout(mode_bar)
        mb.setContentsMargins(10, 4, 10, 4)
        mb.setSpacing(10)

        self._lbl_robot_state = QLabel("—")
        self._lbl_robot_state.setStyleSheet(
            f"color: {DIM}; font-size: 12px; font-weight: bold; font-family: Consolas;"
        )
        self._lbl_wm_mode = QLabel("ODrive: —")
        self._lbl_wm_mode.setStyleSheet(
            f"color: {DIM}; font-size: 11px; font-family: Consolas;"
        )
        lbl_wd = QLabel("⚠ ODrive watchdog expires ~5 s after last setpoint — normal behaviour")
        lbl_wd.setStyleSheet(f"color: {ORANGE}; font-size: 10px;")

        mb.addWidget(self._lbl_robot_state)
        mb.addWidget(self._lbl_wm_mode)
        mb.addStretch()
        mb.addWidget(lbl_wd)

        self._btn_enter = QPushButton("Enter Manual")
        self._btn_enter.setStyleSheet(
            f"QPushButton{{background:#1a4a6a;color:white;"
            f"border:1px solid {BORDER};border-radius:3px;padding:4px 12px}}"
            f"QPushButton:hover{{background:#2a5a7a}}"
            f"QPushButton:disabled{{background:{SURFACE};color:{DIM};"
            f"border:1px solid {BORDER}}}"
        )
        self._btn_exit = QPushButton("Exit Manual")
        self._btn_exit.setStyleSheet(
            f"QPushButton{{background:#4a2a1a;color:white;"
            f"border:1px solid {BORDER};border-radius:3px;padding:4px 12px}}"
            f"QPushButton:hover{{background:#5a3a2a}}"
            f"QPushButton:disabled{{background:{SURFACE};color:{DIM};"
            f"border:1px solid {BORDER}}}"
        )
        self._btn_enter.clicked.connect(lambda: send_set_mode(STATE_MANUAL))
        self._btn_exit.clicked.connect(lambda: send_set_mode(STATE_STANDBY))
        mb.addWidget(self._btn_enter)
        mb.addWidget(self._btn_exit)

        # ── Centre column ─────────────────────────────────────────────────────
        centre = QWidget()
        centre.setFixedWidth(130)
        c_lay = QVBoxLayout(centre)
        c_lay.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        c_lay.setContentsMargins(4, 0, 4, 0)
        c_lay.setSpacing(6)

        mode_lbl = QLabel("ODrive Mode")
        mode_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mode_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px; font-weight: bold;")
        c_lay.addWidget(mode_lbl)

        self._mode_btns: dict[int, QPushButton] = {}
        for mode, label in [
            (WHEEL_MODE_IDLE,     "IDLE"),
            (WHEEL_MODE_VELOCITY, "VEL"),
            (WHEEL_MODE_POSITION, "POS"),
            (WHEEL_MODE_TORQUE,   "TRQ"),
        ]:
            b = QPushButton(label)
            b.setStyleSheet(
                f"QPushButton{{background:{SURFACE};color:{TEXT};"
                f"border:1px solid {BORDER};border-radius:3px;padding:3px 6px}}"
                f"QPushButton:hover{{background:{BORDER}}}"
                f"QPushButton:checked{{background:#1a4a7a;color:white;font-weight:bold}}"
                f"QPushButton:disabled{{background:{SURFACE};color:{DIM};"
                f"border:1px solid {BORDER}}}"
            )
            b.setCheckable(True)
            b.clicked.connect(lambda _, m=mode: self._on_mode_btn(m))
            b.setEnabled(False)
            c_lay.addWidget(b)
            self._mode_btns[mode] = b

        c_lay.addSpacing(8)

        btn_clr_both = QPushButton("Clear Both")
        btn_clr_both.setStyleSheet(
            f"QPushButton{{background:{SURFACE};color:{DIM};"
            f"border:1px solid {BORDER};border-radius:3px;padding:3px 6px}}"
            f"QPushButton:hover{{background:{BORDER};color:{TEXT}}}"
        )
        btn_clr_both.clicked.connect(send_wheel_clear_errors)
        c_lay.addWidget(btn_clr_both)
        c_lay.addStretch()

        # ── Diff drive slider (spin-in-place) ─────────────────────────────────
        diff_widget = QWidget()
        diff_lay = QVBoxLayout(diff_widget)
        diff_lay.setContentsMargins(10, 4, 10, 4)
        diff_lay.setSpacing(4)
        diff_hdr = QLabel("Spin-in-place  (±5 t/s)")
        diff_hdr.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        self._diff_lbl = QLabel("0.0 t/s")
        self._diff_lbl.setStyleSheet(
            f"color: {TEXT}; font-size: 10px; font-family: Consolas;"
        )
        diff_hdr_row = QHBoxLayout()
        diff_hdr_row.addWidget(diff_hdr)
        diff_hdr_row.addStretch()
        diff_hdr_row.addWidget(self._diff_lbl)
        diff_lay.addLayout(diff_hdr_row)
        self._diff_slider = QSlider(Qt.Orientation.Horizontal)
        self._diff_slider.setRange(-500, 500)   # ×0.01 → ±5 t/s
        self._diff_slider.setValue(0)
        self._diff_slider.setEnabled(False)
        self._diff_slider.valueChanged.connect(self._on_diff_changed)
        diff_lay.addWidget(self._diff_slider)

        # ── Motor panels row ──────────────────────────────────────────────────
        panels_row = QHBoxLayout()
        panels_row.setSpacing(8)
        panels_row.addWidget(self._panel_L, stretch=1)
        panels_row.addWidget(centre)
        panels_row.addWidget(self._panel_R, stretch=1)

        # ── Vbus chart ────────────────────────────────────────────────────────
        vbus_widget = QWidget()
        vbus_widget.setObjectName("VbusPanel")
        vbus_widget.setStyleSheet(
            f"#VbusPanel {{background: {SURFACE}; border: 1px solid {BORDER};"
            f" border-radius: 4px;}}"
        )
        vbus_lay = QVBoxLayout(vbus_widget)
        vbus_lay.setContentsMargins(8, 4, 8, 4)
        vbus_lay.setSpacing(2)
        vbus_hdr = QLabel("Bus Voltage")
        vbus_hdr.setStyleSheet(f"color: {DIM}; font-size: 10px; font-weight: bold;")
        vbus_lay.addWidget(vbus_hdr)
        self._vbus_chart = pg.PlotWidget()
        self._vbus_chart.setBackground(BG)
        self._vbus_chart.setFixedHeight(90)
        self._vbus_chart.showGrid(x=True, y=True, alpha=0.12)
        self._vbus_chart.setXRange(0, _BUF)
        self._vbus_chart.getAxis("bottom").setStyle(showValues=False)
        self._vbus_chart.addLegend(offset=(4, 4), verSpacing=-4)
        self._crv_vbus_L = self._vbus_chart.plot(
            list(self._vbus_L_buf),
            pen=pg.mkPen(GREEN, width=1.5),
            name="Vbus L",
        )
        self._crv_vbus_R = self._vbus_chart.plot(
            list(self._vbus_R_buf),
            pen=pg.mkPen(ORANGE, width=1.5),
            name="Vbus R",
        )
        vbus_lay.addWidget(self._vbus_chart)

        self._vbus_redraw = QTimer(self)
        self._vbus_redraw.timeout.connect(self._redraw_vbus)
        self._vbus_redraw.start(50)

        # ── Outer layout ──────────────────────────────────────────────────────
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        lay.addWidget(mode_bar)
        lay.addLayout(panels_row)
        lay.addWidget(diff_widget)
        lay.addWidget(vbus_widget)

        TelemetryBus.instance().packet.connect(self._on_packet)

    # ── State labels ──────────────────────────────────────────────────────────

    _STATE_LABELS = {
        0: ("STARTUP",     "#aaaaaa"),
        1: ("CALIBRATION", "#4488ff"),
        2: ("STANDBY",     "#ffcc00"),
        3: ("RUNNING",     "#44ff88"),
        4: ("ESTOP",       "#ff4444"),
        5: ("MANUAL",      "#00ccff"),
        6: ("CMD_REJECT",  "#ff8800"),
    }

    # ── Per-wheel send helpers ────────────────────────────────────────────────

    def _send_L(self, fw: float):
        self._fw_L = fw
        send_wheel_setpoint(self._fw_L, self._fw_R)

    def _send_R(self, fw: float):
        self._fw_R = fw
        send_wheel_setpoint(self._fw_L, self._fw_R)

    # ── Commands ──────────────────────────────────────────────────────────────

    def _on_mode_btn(self, mode: int):
        send_wheel_set_mode(mode)
        # Update both panels' spinbox config immediately
        self._panel_L.set_mode(mode)
        self._panel_R.set_mode(mode)
        self._update_mode_btn_highlights(mode)

    def _update_mode_btn_highlights(self, mode: int):
        for m, b in self._mode_btns.items():
            b.setChecked(m == mode)

    # ── Diff slider ───────────────────────────────────────────────────────────

    def _on_diff_changed(self, raw: int):
        val = raw * 0.01  # turns/s (display)
        self._diff_lbl.setText(f"{val:+.2f} t/s")
        fw = val * (2 * math.pi)  # → rad/s for firmware
        self._fw_L = fw
        self._fw_R = -fw
        self._panel_L._latest_cmd =  val
        self._panel_R._latest_cmd = -val
        send_wheel_setpoint(self._fw_L, self._fw_R)

    # ── Telemetry ─────────────────────────────────────────────────────────────

    def _on_packet(self, info: dict):
        if info.get("ptype") != 0x01:
            return

        state_id = info.get("robot_state", 0)
        fault    = info.get("fault_name", "")

        if state_id != self._last_state_id or (state_id == 4 and fault and fault != "NONE"):
            self._last_state_id = state_id
            label, color = self._STATE_LABELS.get(state_id, (f"STATE {state_id}", "#aaaaaa"))
            if state_id == 4 and fault and fault != "NONE":
                label = f"ESTOP [{fault}]"
            self._lbl_robot_state.setText(label)
            self._lbl_robot_state.setStyleSheet(
                f"color: {color}; font-size: 12px; font-weight: bold; font-family: Consolas;"
            )

            in_manual = (state_id == 5)
            self._panel_L.set_controls_enabled(in_manual)
            self._panel_R.set_controls_enabled(in_manual)
            self._diff_slider.setEnabled(in_manual)
            for b in self._mode_btns.values():
                b.setEnabled(in_manual)

            # Enter VELOCITY at 0 on MANUAL entry so ODrive goes CLOSED_LOOP
            # and encoder estimates start flowing — same pattern as test_wheel_motor.cpp.
            # Clear errors first: the watchdog-expired fault from the previous session
            # must be dismissed before the ODrive can re-arm on the next MANUAL entry.
            if in_manual:
                self._fw_L = 0.0
                self._fw_R = 0.0
                send_wheel_clear_errors()
                QTimer.singleShot(150, lambda: (
                    send_wheel_set_mode(WHEEL_MODE_VELOCITY),
                    send_wheel_setpoint(0.0, 0.0),
                ))

        # Wheel telemetry (V3 only)
        wm_mode = info.get("wm_mode")
        if wm_mode is not None:
            if wm_mode != self._wm_mode:
                self._wm_mode = wm_mode
                self._update_mode_btn_highlights(wm_mode)
                self._panel_L.set_mode(wm_mode)
                self._panel_R.set_mode(wm_mode)
            mode_str = _WHEEL_MODE_NAMES.get(wm_mode, str(wm_mode))
            self._lbl_wm_mode.setText(f"ODrive: {mode_str}")
            self._lbl_wm_mode.setStyleSheet(
                f"color: {TEXT}; font-size: 11px; font-family: Consolas;"
            )

        vbus_l = info.get("wm_l_vbus", 0.0)
        vbus_r = info.get("wm_r_vbus", 0.0)
        self._panel_L.update_data(
            vel=info.get("wm_l_vel_turns_s", 0.0),
            pos=info.get("wm_l_pos_turns",   0.0),
            vbus=vbus_l,
            error=info.get("wm_l_error",      0),
            state_id=info.get("wm_l_state",   0),
        )
        self._panel_R.update_data(
            vel=info.get("wm_r_vel_turns_s", 0.0),
            pos=info.get("wm_r_pos_turns",   0.0),
            vbus=vbus_r,
            error=info.get("wm_r_error",      0),
            state_id=info.get("wm_r_state",   0),
        )
        self._vbus_L_buf.append(vbus_l)
        self._vbus_R_buf.append(vbus_r)

    def _redraw_vbus(self):
        self._crv_vbus_L.setData(list(self._vbus_L_buf))
        self._crv_vbus_R.setData(list(self._vbus_R_buf))
        all_v = list(self._vbus_L_buf) + list(self._vbus_R_buf)
        lo, hi = min(all_v), max(all_v)
        span = max(hi - lo, 1.0)
        mid  = (lo + hi) / 2
        self._vbus_chart.setYRange(mid - span * 0.6, mid + span * 0.6, padding=0.05)
