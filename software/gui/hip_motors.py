"""hip_motors.py — Hip Motors tab (AK45-10, MIT CAN mode).

Displays live position + command for both hip axes. The "Command" trace
shows torque command (cmd_l/cmd_r) outside MANUAL/CALIBRATION, the MIT
position setpoint (p°, from Send MIT / test wave) while in MANUAL mode,
or the firmware's hardstop-seek ramp (cmd_l/cmd_r, echoed in rad) while
in CALIBRATION — so position tracking can be gauged against the actual
position trace.
Controls: Enable / Disable / Zero per-motor and both,
          MIT position command (p°, Kp, Kd, τff).

Telemetry fields used (from TelemetryPayload / comm_protocol.h):
    hip_l_pos_rad, hip_r_pos_rad       — hip encoder position [rad]
    cmd_l, cmd_r                       — torque command sent to each hip [N·m]
    hip_l_current_a, hip_r_current_a   — hip phase current [A]

Commands are framed via the CommLink protocol (shared/comm_protocol.h) and
written to the Teensy serial port through SerialPortManager.
"""

import math
import struct
import time
from collections import deque
from datetime import datetime

import pyqtgraph as pg
pg.setConfigOptions(antialias=False)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget, QSizePolicy,
)

from telemetry_bus import TelemetryBus
from theme import BG, BLUE, BORDER, DIM, GREEN, ORANGE, RED, SURFACE, TEXT
from comm_commands import build_frame, send_frame, send_set_mode, CMD_ID_HIP

_BUF = 750          # rolling chart samples (~15 s at 50 Hz telemetry)

# Command IDs (comm_protocol.h CMD_ID_*)
_CMD_ID_HIP      = CMD_ID_HIP

# Hip motor IDs
_HIP_MOTOR_BOTH  = 0x00
_HIP_MOTOR_L     = 0x01
_HIP_MOTOR_R     = 0x02

# Hip sub-commands
_HIP_CMD_DISABLE = 0x00
_HIP_CMD_ENABLE  = 0x01
_HIP_CMD_ZERO    = 0x02
_HIP_CMD_MIT     = 0x03

# Robot state IDs (robot_state.h RobotStateEnum)
_STATE_CALIBRATION = 1
_STATE_STANDBY     = 2
_STATE_MANUAL      = 5

# Keep old alias so nothing else breaks
_CMD_HIP_MOTOR = _CMD_ID_HIP

# Calibration event sub-types (comm_protocol.h CALIB_EVENT_*)
_CALIB_EVENT_START        = 0x01
_CALIB_EVENT_BOTTOM_FOUND = 0x02
_CALIB_EVENT_LIMITS       = 0x03
_CALIB_EVENT_DONE         = 0x04
_CALIB_EVENT_FAULT        = 0x05


# ── Small UI helpers ──────────────────────────────────────────────────────────

def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f


def _readout(parent_lay: QVBoxLayout, label: str,
             color: str = TEXT) -> QLabel:
    """Add a dim-label / value-label row; return the value QLabel."""
    row = QHBoxLayout()
    row.setSpacing(6)
    k = QLabel(label + ":")
    k.setStyleSheet(f"color: {DIM}; font-size: 11px;")
    v = QLabel("—")
    v.setStyleSheet(
        f"color: {color}; font-size: 13px; font-weight: bold;"
        f" font-family: Consolas;"
    )
    row.addWidget(k)
    row.addStretch()
    row.addWidget(v)
    parent_lay.addLayout(row)
    return v


def _colored_btn(label: str, bg: str) -> QPushButton:
    b = QPushButton(label)
    b.setStyleSheet(
        f"QPushButton{{background:{bg};color:white;"
        f"border:1px solid {BORDER};border-radius:3px;padding:4px 10px}}"
        f"QPushButton:hover{{background:{bg}cc}}"
        f"QPushButton:pressed{{background:{bg}88}}"
        f"QPushButton:disabled{{background:{SURFACE};color:{DIM};"
        f"border:1px solid {BORDER}}}"
    )
    return b


def _small_btn(label: str, accent: str) -> QPushButton:
    """Muted, compact button — for manual overrides tucked out of the way."""
    b = QPushButton(label)
    b.setStyleSheet(
        f"QPushButton{{background:{SURFACE};color:{DIM};font-size:9px;"
        f"border:1px solid {BORDER};border-radius:3px;padding:2px 8px}}"
        f"QPushButton:hover{{background:{accent}55;color:{TEXT}}}"
        f"QPushButton:pressed{{background:{accent}aa;color:white}}"
        f"QPushButton:disabled{{background:{SURFACE};color:{DIM};"
        f"border:1px solid {BORDER}}}"
    )
    return b


def _preset_btn_style(bg: str, selected: bool) -> str:
    """Preset button style — highlighted border when this preset is active."""
    border_color = ORANGE if selected else BORDER
    weight = "font-weight:bold;" if selected else ""
    return (
        f"QPushButton{{background:{bg};color:white;{weight}"
        f"border:2px solid {border_color};border-radius:3px;padding:3px 9px}}"
        f"QPushButton:hover{{background:{bg}cc}}"
        f"QPushButton:disabled{{background:{SURFACE};color:{DIM};"
        f"border:2px solid {BORDER}}}"
    )


def _toggle_btn(label: str, bg: str) -> QPushButton:
    """Checkable button that highlights `bg` while active (for wave start/stop)."""
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


def _spin(lo: float, hi: float, step: float,
          val: float, dec: int) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(val)
    s.setDecimals(dec)
    s.setFixedWidth(72)
    s.setStyleSheet(
        f"QDoubleSpinBox{{background:{BG};color:{TEXT};"
        f"font-family:Consolas;font-size:10px;"
        f"border:1px solid {BORDER};border-radius:3px;padding:1px 4px}}"
        f"QDoubleSpinBox::up-button,QDoubleSpinBox::down-button{{width:14px}}"
        f"QDoubleSpinBox:disabled{{background:{SURFACE};color:{DIM}}}"
    )
    return s


# ── Per-motor panel ───────────────────────────────────────────────────────────

class _MotorPanel(QWidget):
    """Live readout + mini chart + controls for one AK45 hip axis."""

    def __init__(self, motor_id: int, title: str, node_hex: str):
        super().__init__()
        self._mid = motor_id
        self._pos_buf: deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._cmd_buf: deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._cur_buf: deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._latest_pos_deg = 0.0
        self._latest_tau_nm  = 0.0
        self._latest_current_a = 0.0
        self._latest_cmd_pos_deg = 0.0
        self._in_manual = False
        self._in_calibration = False
        self._redraw_count = 0

        self.setObjectName("MotorPanel")
        self.setStyleSheet(
            f"#MotorPanel {{background: {SURFACE}; border: 1px solid {BORDER};"
            f" border-radius: 4px;}}"
            f"QLabel, QPushButton, QDoubleSpinBox {{border: none;}}"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        # Header
        hdr = QLabel(
            f"<b>{title}</b>"
            f"  <span style='color:{DIM};font-size:10px'>({node_hex})</span>"
        )
        hdr.setStyleSheet(f"color: {TEXT}; font-size: 12px;")
        lay.addWidget(hdr)
        lay.addWidget(_hline())

        # Live readouts
        self._lbl_pos    = _readout(lay, "Position", BLUE)
        self._lbl_cmd    = _readout(lay, "Command", ORANGE)
        self._lbl_cur    = _readout(lay, "Current", GREEN)
        self._lbl_limits = _readout(lay, "Limits", DIM)
        lay.addWidget(_hline())

        # Mini chart — pos (blue) + tau (orange) + current (green)
        self._chart = pg.PlotWidget()
        self._chart.setBackground(BG)
        self._chart.setMaximumHeight(115)
        self._chart.setMinimumHeight(80)
        self._chart.showGrid(x=True, y=True, alpha=0.12)
        self._chart.setXRange(0, _BUF)
        self._chart.getAxis("bottom").setStyle(showValues=False)
        leg = self._chart.addLegend(offset=(4, 4), verSpacing=-4)
        self._crv_pos = self._chart.plot(
            list(self._pos_buf), pen=pg.mkPen(BLUE,   width=1.5), name="pos (°)")
        self._crv_cmd = self._chart.plot(
            list(self._cmd_buf), pen=pg.mkPen(ORANGE, width=1.2), name="cmd")
        self._crv_cur = self._chart.plot(
            list(self._cur_buf), pen=pg.mkPen(GREEN,  width=1.2), name="current (A)")
        lay.addWidget(self._chart)
        lay.addWidget(_hline())

        # ── Controls container (disabled until MANUAL mode) ───────────────────
        self._ctrl = QWidget()
        ctrl_lay = QVBoxLayout(self._ctrl)
        ctrl_lay.setContentsMargins(0, 0, 0, 0)
        ctrl_lay.setSpacing(6)

        # ── Presets: fill Kp / Kd ────────────────────────────────────────────
        self._presets = [
            ("Springy", 15.0,  1.5, "#3a5a3a"),
            ("Normal",  50.0,  3.0, "#3a3a5a"),
            ("Stiff",  200.0,  4.5, "#5a3a2a"),
        ]
        self._preset_buttons: dict[str, QPushButton] = {}
        self._selected_preset: str | None = None
        pre_row = QHBoxLayout()
        pre_row.setSpacing(6)
        pre_lbl = QLabel("Preset:")
        pre_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        pre_row.addWidget(pre_lbl)
        for name, kp_val, kd_val, bg in self._presets:
            b = QPushButton(name)
            b.setProperty("preset_bg", bg)
            b.clicked.connect(lambda _, n=name, kp=kp_val, kd=kd_val: self._apply_preset(n, kp, kd))
            pre_row.addWidget(b)
            self._preset_buttons[name] = b
        pre_row.addStretch()
        ctrl_lay.addLayout(pre_row)

        # ── MIT spinboxes ─────────────────────────────────────────────────────
        mit_row = QHBoxLayout()
        mit_row.setSpacing(4)
        self._sp_p  = _spin(-716, 716, 1.0, 0.0, 1)
        self._sp_kp = _spin(0, 500,   5.0, 15.0, 1)
        self._sp_kd = _spin(0,   5,   0.1,  1.5, 2)
        self._sp_tf = _spin(-8,  8,   0.1,  0.0, 2)
        for lbl_txt, sp in [("p°", self._sp_p), ("Kp", self._sp_kp),
                             ("Kd", self._sp_kd), ("τff", self._sp_tf)]:
            k = QLabel(lbl_txt)
            k.setStyleSheet(f"color: {DIM}; font-size: 10px;")
            mit_row.addWidget(k)
            mit_row.addWidget(sp)
        mit_row.addStretch()
        ctrl_lay.addLayout(mit_row)

        # ── Angle limits + Send ───────────────────────────────────────────────
        lim_row = QHBoxLayout()
        lim_row.setSpacing(4)
        lim_lbl = QLabel("Limits:")
        lim_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        lim_row.addWidget(lim_lbl)
        lo_lbl = QLabel("min")
        lo_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        self._sp_min = _spin(-716, 716, 5.0, -90.0, 1)
        hi_lbl = QLabel("max")
        hi_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        self._sp_max = _spin(-716, 716, 5.0, +90.0, 1)
        lim_row.addWidget(lo_lbl)
        lim_row.addWidget(self._sp_min)
        lim_row.addWidget(hi_lbl)
        lim_row.addWidget(self._sp_max)
        lim_row.addSpacing(8)
        btn_mit = QPushButton("Send MIT")
        btn_mit.setStyleSheet(
            f"QPushButton{{background:#1a4a7a;color:white;"
            f"border:1px solid {BORDER};border-radius:3px;padding:3px 10px}}"
            f"QPushButton:hover{{background:#2a5a8a}}"
            f"QPushButton:disabled{{background:{SURFACE};color:{DIM};"
            f"border:1px solid {BORDER}}}"
        )
        btn_mit.clicked.connect(self._on_send_mit_clicked)
        lim_row.addWidget(btn_mit)
        lim_row.addStretch()
        ctrl_lay.addLayout(lim_row)

        # ── Test wave generator ───────────────────────────────────────────────
        ctrl_lay.addWidget(_hline())
        wave_row = QHBoxLayout()
        wave_row.setSpacing(4)
        wave_lbl = QLabel("Test wave:")
        wave_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        wave_row.addWidget(wave_lbl)

        amp_lbl = QLabel("Amp°")
        amp_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        self._sp_amp = _spin(0, 90, 1.0, 10.0, 1)
        freq_lbl = QLabel("Hz")
        freq_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        self._sp_freq = _spin(0.05, 5.0, 0.05, 0.5, 2)
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

        # ── Enable / Disable / Zero — manual overrides, tucked out of the way ──
        ctrl_lay.addWidget(_hline())
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        for lbl, cv, col in [("Enable",  _HIP_CMD_ENABLE,  "#2e7d32"),
                              ("Disable", _HIP_CMD_DISABLE, "#7d2e2e"),
                              ("Zero",    _HIP_CMD_ZERO,    "#4a4a2e")]:
            b = _small_btn(lbl, col)
            b.clicked.connect(lambda _, m=motor_id, c=cv: self._simple_cmd(m, c))
            btn_row.addWidget(b)
        btn_row.addStretch()
        ctrl_lay.addLayout(btn_row)

        # Wave generator state: centered on p° (the spinbox value when started),
        # oscillating with the Kp/Kd/limits/τff currently set in the panel —
        # so presets (Springy/Normal/Stiff) apply to the wave too.
        self._wave_type      = None   # None | "sine" | "square"
        self._wave_t0        = 0.0
        self._wave_center_deg = 0.0
        self._wave_timer = QTimer(self)
        self._wave_timer.timeout.connect(self._wave_tick)

        # Springy is the default preset for both motors
        springy_name, springy_kp, springy_kd, _ = self._presets[0]
        self._apply_preset(springy_name, springy_kp, springy_kd)

        self._ctrl.setEnabled(False)
        lay.addWidget(self._ctrl)
        lay.addStretch()

        # Redraw mini chart at a fixed rate, decoupled from telemetry rate
        self._redraw_timer = QTimer(self)
        self._redraw_timer.timeout.connect(self._redraw_chart)
        self._redraw_timer.start(50)  # 20 Hz

    def set_controls_enabled(self, enabled: bool):
        self._ctrl.setEnabled(enabled)
        self._in_manual = enabled
        if not enabled:
            self._stop_wave()

    def set_calibration_active(self, active: bool):
        self._in_calibration = active

    def set_calib_limits(self, min_rad: float | None, max_rad: float | None):
        if min_rad is None:
            self._lbl_limits.setText("—")
        else:
            self._lbl_limits.setText(f"[{min_rad:+.3f}, {max_rad:+.3f}] rad")

    # ── command helpers ───────────────────────────────────────────────────────

    def _simple_cmd(self, motor_id: int, hip_sub: int):
        payload = struct.pack("<BBB", _CMD_ID_HIP, motor_id, hip_sub)
        send_frame(build_frame(payload))

    def _apply_preset(self, name: str, kp: float, kd: float):
        self._sp_kp.setValue(kp)
        self._sp_kd.setValue(kd)
        self._selected_preset = name
        self._update_preset_styles()

    def _update_preset_styles(self):
        for name, b in self._preset_buttons.items():
            bg = b.property("preset_bg")
            b.setStyleSheet(_preset_btn_style(bg, name == self._selected_preset))

    def _send_mit(self):
        # Clamp p° to [min, max] and snap the spinbox so it stays in range visually
        lo_deg  = self._sp_min.value()
        hi_deg  = self._sp_max.value()
        p_deg   = max(lo_deg, min(hi_deg, self._sp_p.value()))
        self._sp_p.setValue(p_deg)
        self._latest_cmd_pos_deg = p_deg
        p_rad   = math.radians(p_deg)
        payload = struct.pack("<BBBfffff",
                              _CMD_ID_HIP, self._mid, _HIP_CMD_MIT,
                              p_rad, 0.0,
                              self._sp_kp.value(), self._sp_kd.value(),
                              self._sp_tf.value())
        send_frame(build_frame(payload))

    def _on_send_mit_clicked(self):
        self._stop_wave()
        self._send_mit()

    # ── test wave generator ───────────────────────────────────────────────────

    def _toggle_wave(self, kind: str):
        if self._wave_type == kind:
            self._stop_wave()
            return
        self._wave_type       = kind
        self._wave_center_deg = self._sp_p.value()
        self._wave_t0         = time.monotonic()
        self._btn_sine.setChecked(kind == "sine")
        self._btn_square.setChecked(kind == "square")
        self._wave_timer.start(20)  # 50 Hz setpoint update

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
        else:  # "square"
            offset = amp if math.sin(phase) >= 0 else -amp
        self._sp_p.setValue(self._wave_center_deg + offset)
        self._send_mit()

    # ── data update (called from HipMotorsTab._on_packet) ────────────────────

    def update_data(self, pos_rad: float, tau_nm: float, current_a: float):
        pos_deg = math.degrees(pos_rad)
        self._latest_pos_deg = pos_deg
        self._latest_tau_nm  = tau_nm
        self._latest_current_a = current_a

        self._pos_buf.append(pos_deg)
        self._cur_buf.append(current_a)
        # In MANUAL mode the orange trace tracks the position setpoint
        # (Send MIT / test wave) so it can be compared against actual
        # position to gauge tracking. In CALIBRATION it tracks the
        # firmware's hardstop-seek ramp (echoed via cmd_l/cmd_r, in rad).
        # Otherwise it shows commanded torque.
        if self._in_manual:
            self._cmd_buf.append(self._latest_cmd_pos_deg)
        elif self._in_calibration:
            self._latest_cmd_pos_deg = math.degrees(tau_nm)
            self._cmd_buf.append(self._latest_cmd_pos_deg)
        else:
            self._cmd_buf.append(tau_nm)

    def _redraw_chart(self):
        self._lbl_pos.setText(f"{self._latest_pos_deg:+.1f}°")
        if self._in_manual or self._in_calibration:
            self._lbl_cmd.setText(f"{self._latest_cmd_pos_deg:+.1f}°")
        else:
            self._lbl_cmd.setText(f"{self._latest_tau_nm:+.2f} N·m")
        self._lbl_cur.setText(f"{self._latest_current_a:+.2f} A")

        self._crv_pos.setData(list(self._pos_buf))
        self._crv_cmd.setData(list(self._cmd_buf))
        self._crv_cur.setData(list(self._cur_buf))

        # auto-fit mini chart Y to all traces together — only every 5th
        # redraw (4 Hz at 20 Hz refresh); the Y range doesn't need to track
        # every single frame and setYRange forces an axis/repaint pass.
        self._redraw_count += 1
        if self._redraw_count % 5 == 0:
            lo = min(min(self._pos_buf), min(self._cmd_buf), min(self._cur_buf))
            hi = max(max(self._pos_buf), max(self._cmd_buf), max(self._cur_buf))
            span = max(hi - lo, 5.0)
            mid  = (lo + hi) / 2
            self._chart.setYRange(mid - span * 0.6, mid + span * 0.6, padding=0.05)


# ── Main tab ──────────────────────────────────────────────────────────────────

class HipMotorsTab(QWidget):
    def __init__(self):
        super().__init__()

        self._panel_L = _MotorPanel(1, "Hip L", "0x41")
        self._panel_R = _MotorPanel(2, "Hip R", "0x42")

        # ── Centre: Both controls ─────────────────────────────────────────────
        both = QWidget()
        both.setEnabled(False)
        both.setFixedWidth(120)
        both_lay = QVBoxLayout(both)
        both_lay.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        both_lay.setContentsMargins(4, 0, 4, 0)
        both_lay.setSpacing(10)
        lbl = QLabel("Both")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"color: {DIM}; font-size: 11px; font-weight: bold;")
        both_lay.addWidget(lbl)
        both_lay.addStretch()
        # Manual overrides, tucked out of the way at the bottom
        for lbl_txt, cv, col in [
                ("Enable",  _HIP_CMD_ENABLE,  "#2e7d32"),
                ("Disable", _HIP_CMD_DISABLE, "#7d2e2e"),
                ("Zero",    _HIP_CMD_ZERO,    "#4a4a2e")]:
            b = _small_btn(lbl_txt, col)
            b.clicked.connect(lambda _, c=cv: self._both_cmd(c))
            both_lay.addWidget(b)

        self._both = both

        # ── Top row ───────────────────────────────────────────────────────────
        top = QWidget()
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(8)
        top_lay.addWidget(self._panel_L, stretch=1)
        top_lay.addWidget(both)
        top_lay.addWidget(self._panel_R, stretch=1)

        # ── Manual mode bar ───────────────────────────────────────────────────
        mode_bar = QWidget()
        mode_bar.setObjectName("ModeBar")
        mode_bar.setFixedHeight(36)
        mode_bar.setStyleSheet(
            f"#ModeBar {{background: {SURFACE}; border: 1px solid {BORDER};"
            f" border-radius: 4px;}}"
            f"QLabel, QPushButton {{border: none;}}"
        )
        mb_lay = QHBoxLayout(mode_bar)
        mb_lay.setContentsMargins(10, 4, 10, 4)
        mb_lay.setSpacing(10)

        self._lbl_mode = QLabel("STANDBY")
        self._lbl_mode.setStyleSheet(
            f"color: {ORANGE}; font-size: 12px; font-weight: bold;"
            f" font-family: Consolas;"
        )
        mb_lay.addWidget(self._lbl_mode)
        mb_lay.addStretch()

        mode_hint = QLabel("Hip commands are only executed in MANUAL mode")
        mode_hint.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        mb_lay.addWidget(mode_hint)
        mb_lay.addStretch()

        self._btn_calibrate = _colored_btn("Calibrate", "#4a3a1a")
        self._btn_enter     = _colored_btn("Enter Manual", "#1a4a6a")
        self._btn_exit      = _colored_btn("Exit Manual",  "#4a2a1a")
        self._btn_calibrate.clicked.connect(lambda: self._set_mode(_STATE_CALIBRATION))
        self._btn_enter.clicked.connect(lambda: self._set_mode(_STATE_MANUAL))
        self._btn_exit .clicked.connect(lambda: self._set_mode(_STATE_STANDBY))
        self._btn_calibrate.setEnabled(False)
        mb_lay.addWidget(self._btn_calibrate)
        mb_lay.addWidget(self._btn_enter)
        mb_lay.addWidget(self._btn_exit)

        # ── Calibration event log ─────────────────────────────────────────────
        log_hdr = QLabel("Calibration Log")
        log_hdr.setStyleSheet(
            f"color: {BLUE}; font-weight: bold; font-size: 11px;"
        )
        self._calib_log = QPlainTextEdit()
        self._calib_log.setReadOnly(True)
        self._calib_log.setMaximumBlockCount(200)
        self._calib_log.setPlaceholderText("Waiting for calibration events…")
        self._calib_log.setStyleSheet(
            f"background: #0a0a14; color: {TEXT}; border: 1px solid {BORDER};"
            f" font-family: Consolas; font-size: 11px;"
        )
        self._calib_log.setMaximumHeight(160)
        self._calib_log.setMinimumHeight(70)

        log_pane = QWidget()
        log_pane_lay = QVBoxLayout(log_pane)
        log_pane_lay.setContentsMargins(0, 4, 0, 0)
        log_pane_lay.setSpacing(4)
        log_pane_lay.addWidget(log_hdr)
        log_pane_lay.addWidget(self._calib_log)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        lay.addWidget(mode_bar)
        lay.addWidget(top)
        lay.addWidget(log_pane)

        TelemetryBus.instance().packet.connect(self._on_packet)

        self._last_state_id: int | None = None

        # AK45 MIT mode drops out silently without periodic re-entry — while
        # in MANUAL, re-send "enter MIT" to both hips every 4 s.
        self._mit_reinforce_timer = QTimer(self)
        self._mit_reinforce_timer.timeout.connect(
            lambda: self._both_cmd(_HIP_CMD_ENABLE))

    # ── commands ──────────────────────────────────────────────────────────────

    def _set_mode(self, target: int):
        send_set_mode(target)

    def _both_cmd(self, hip_sub: int):
        payload = struct.pack("<BBB", _CMD_ID_HIP, _HIP_MOTOR_BOTH, hip_sub)
        send_frame(build_frame(payload))

    # ── calib log ─────────────────────────────────────────────────────────────

    _AXIS_NAMES  = {0x00: "BOTH", 0x01: "L", 0x02: "R"}
    _EVENT_NAMES = {
        0x01: ("START",        BLUE),
        0x02: ("BOTTOM FOUND", TEXT),
        0x03: ("LIMITS",       GREEN),
        0x04: ("DONE",         GREEN),
        0x05: ("FAULT",        RED),
    }

    def _append_calib_log(self, text: str, color: str = TEXT):
        ts  = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cur = self._calib_log.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.setCharFormat(fmt)
        cur.insertText(f"[{ts}] {text}\n")
        self._calib_log.verticalScrollBar().setValue(
            self._calib_log.verticalScrollBar().maximum()
        )

    # ── telemetry ─────────────────────────────────────────────────────────────

    _STATE_LABELS = {
        0: ("STARTUP",     "#aaaaaa"),
        1: ("CALIBRATION", "#4488ff"),
        2: ("STANDBY",     "#ffcc00"),
        3: ("RUNNING",     "#44ff88"),
        4: ("ESTOP",       "#ff4444"),
        5: ("MANUAL",      "#00ccff"),
    }

    def _on_packet(self, info: dict):
        ptype = info.get("ptype")

        if ptype == 0x05:
            event = info.get("calib_event")
            axis  = info.get("calib_axis")
            event_name, color = self._EVENT_NAMES.get(event, (f"0x{event:02X}", TEXT))
            axis_name = self._AXIS_NAMES.get(axis, f"0x{axis:02X}")
            pos = info.get("calib_pos_rad", 0.0)
            mn  = info.get("calib_min_rad", 0.0)
            mx  = info.get("calib_max_rad", 0.0)

            if event == _CALIB_EVENT_START:
                self._calib_log.clear()
                self._append_calib_log("── calibration started ──", BLUE)
                self._panel_L.set_calib_limits(None, None)
                self._panel_R.set_calib_limits(None, None)
            elif event == _CALIB_EVENT_BOTTOM_FOUND:
                self._append_calib_log(
                    f"{axis_name}: bottom hardstop found, zeroed @ {pos:+.3f} rad", color)
            elif event == _CALIB_EVENT_LIMITS:
                self._append_calib_log(
                    f"{axis_name}: limits [{mn:+.3f}, {mx:+.3f}] rad  (range {pos:+.3f})", color)
                if axis == _HIP_MOTOR_L:
                    self._panel_L.set_calib_limits(mn, mx)
                elif axis == _HIP_MOTOR_R:
                    self._panel_R.set_calib_limits(mn, mx)
            elif event == _CALIB_EVENT_DONE:
                self._append_calib_log(
                    f"{axis_name}: done, holding @ {pos:+.3f} rad  limits [{mn:+.3f}, {mx:+.3f}]",
                    color)
                if axis == _HIP_MOTOR_L:
                    self._panel_L.set_calib_limits(mn, mx)
                elif axis == _HIP_MOTOR_R:
                    self._panel_R.set_calib_limits(mn, mx)
            elif event == _CALIB_EVENT_FAULT:
                self._append_calib_log(
                    f"{axis_name}: FAULT — hardstop not found @ {pos:+.3f} rad", color)
            else:
                self._append_calib_log(f"{axis_name}: {event_name}  pos={pos:+.3f}", color)
            return

        if ptype == 0x04:
            msg = info.get("log_msg", "")
            if "calib" in msg.lower():
                level = info.get("log_level", "INFO")
                log_color = {
                    "INFO": DIM, "WARN": ORANGE, "ERROR": RED,
                }.get(level, DIM)
                self._append_calib_log(f"[{level}] {msg}", log_color)
            return

        if ptype != 0x01:
            return

        state_id = info.get("robot_state", 0)
        fault = info.get("fault_name", "")

        # Mode bar / control-enable state only changes occasionally — skip the
        # (relatively expensive) setStyleSheet/setEnabled calls on every
        # telemetry packet (~500 Hz) to keep the UI thread free for input.
        if state_id != self._last_state_id or (state_id == 4 and fault and fault != "NONE"):
            self._last_state_id = state_id
            label, color = self._STATE_LABELS.get(state_id, (f"STATE {state_id}", "#aaaaaa"))
            if state_id == 4 and fault and fault != "NONE":  # ESTOP
                label = f"ESTOP [{fault}]"
            self._lbl_mode.setText(label)
            self._lbl_mode.setStyleSheet(
                f"color: {color}; font-size: 12px; font-weight: bold; font-family: Consolas;"
            )

            in_manual = (state_id == _STATE_MANUAL)
            in_calibration = (state_id == _STATE_CALIBRATION)
            self._panel_L.set_controls_enabled(in_manual)
            self._panel_R.set_controls_enabled(in_manual)
            self._panel_L.set_calibration_active(in_calibration)
            self._panel_R.set_calibration_active(in_calibration)
            self._both.setEnabled(in_manual)
            self._btn_calibrate.setEnabled(state_id == _STATE_STANDBY)

            # Entering MANUAL: arm both hips into MIT mode and keep
            # reinforcing it. Leaving MANUAL: just stop the reinforcement —
            # do not send an explicit "exit MIT" / disable command.
            if in_manual:
                self._both_cmd(_HIP_CMD_ENABLE)
                self._mit_reinforce_timer.start(4000)
            else:
                self._mit_reinforce_timer.stop()

        pos_l = info.get("hip_l_pos_rad", 0.0)
        pos_r = info.get("hip_r_pos_rad", 0.0)
        tau_l = info.get("cmd_l", 0.0)
        tau_r = info.get("cmd_r", 0.0)
        cur_l = info.get("hip_l_current_a", 0.0)
        cur_r = info.get("hip_r_current_a", 0.0)

        # Update per-motor panels
        self._panel_L.update_data(pos_l, tau_l, cur_l)
        self._panel_R.update_data(pos_r, tau_r, cur_r)
