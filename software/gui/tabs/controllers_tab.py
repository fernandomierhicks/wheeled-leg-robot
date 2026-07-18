import math
import time
from collections import deque

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel,
    QScrollArea, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from .comm_commands import send_param_get, send_param_set
from .telemetry_bus import TelemetryBus
from .theme import BG, BORDER, BLUE, DIM, GREEN, ORANGE, RED, SURFACE, TEXT, YELLOW

_BUF = 300  # rolling chart samples

_HEALTH_LQR_ACTIVE       = 1 << 6
_HEALTH_VEL_PI_SAT       = 1 << 7
_HEALTH_YAW_PI_SAT       = 1 << 8
_HEALTH_WM_L_VEL_LIMITED = 1 << 9
_HEALTH_WM_R_VEL_LIMITED = 1 << 10

# LQR gain params (Phase 5) — runtime-tunable in firmware (param_ids.h), no
# longer hardcoded here. Requested once on first telemetry and kept current
# via PARAM_REPORT echoes (ptype 0x06), so edits made in the Params tab
# (Control > LQR Gains) show up here too. alpha=0 -> retracted, alpha=1 -> extended.
_PARAM_LQR_K_PITCH_RET = 0x0424
_PARAM_LQR_K_RATE_RET  = 0x0425
_PARAM_LQR_K_PITCH_EXT = 0x0426
_PARAM_LQR_K_RATE_EXT  = 0x0427
_PARAM_LQR_K_VEL       = 0x0428

_GAIN_PARAM_LABELS: dict[int, str] = {
    _PARAM_LQR_K_PITCH_RET: "K_pitch_ret",
    _PARAM_LQR_K_RATE_RET:  "K_rate_ret",
    _PARAM_LQR_K_PITCH_EXT: "K_pitch_ext",
    _PARAM_LQR_K_RATE_EXT:  "K_rate_ext",
    _PARAM_LQR_K_VEL:       "K_vel",
}

# Sim-pitch injection params (param_ids.h) — bench-test only, no arming
# interlock in firmware, so this control must be disarmed before running for real.
_PARAM_SIM_PITCH_RAD          = 0x0401
_PARAM_ENABLE_SIM_PITCH_RAD   = 0x0420
_PARAM_SIM_PITCH_RATE_RAD_S   = 0x0421
_PARAM_ENABLE_SIM_PITCH_RATE  = 0x0422

_SIM_PITCH_MAX_DEG   = 90.0   # matches firmware clamp on sim_pitch_rad (+-1.5708 rad)
_SIM_PITCH_RATE_MAX  = 10.0   # matches firmware clamp on sim_pitch_rate (rad/s)
_SIM_SLIDER_STEPS    = 900    # 0.1 deg resolution


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {BLUE}; font-weight: bold; font-size: 11px; padding-top: 6px;"
    )
    return lbl


class _ValueCell(QWidget):
    """Name + numeric readout stacked vertically."""

    def __init__(self, name: str, unit: str = ""):
        super().__init__()
        k = QLabel(f"{name}  [{unit}]" if unit else name)
        k.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        k.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._val = QLabel("—")
        self._val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._val.setMinimumWidth(106)
        self._last_css = ""
        self._set_style(TEXT)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        lay.addWidget(k)
        lay.addWidget(self._val)

    def _set_style(self, color: str):
        css = (
            f"color: {color}; font-size: 14px; font-weight: bold;"
            f" font-family: Consolas; background: {SURFACE};"
            f" border: 1px solid {BORDER}; border-radius: 3px;"
            f" padding: 4px 8px;"
        )
        if css != self._last_css:
            self._last_css = css
            self._val.setStyleSheet(css)

    def set(self, value: float | int, fmt: str = "+.4f", color: str = TEXT):
        self._val.setText(format(value, fmt))
        self._set_style(color)


class _StatusPill(QLabel):
    """Small badge — lit or dimmed."""

    def __init__(self, text: str, active_color: str = GREEN):
        super().__init__(text)
        self._on  = (
            f"color: #fff; background: {active_color}; border: 1px solid {active_color};"
            f" border-radius: 10px; padding: 2px 10px; font-size: 11px; font-weight: bold;"
        )
        self._off = (
            f"color: {DIM}; background: {SURFACE}; border: 1px solid {BORDER};"
            f" border-radius: 10px; padding: 2px 10px; font-size: 11px;"
        )
        self.setStyleSheet(self._off)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_active(self, on: bool):
        self.setStyleSheet(self._on if on else self._off)


def _make_plot(ylabel: str) -> pg.PlotWidget:
    w = pg.PlotWidget()
    w.setBackground(BG)
    w.setLabel("left", ylabel, color=DIM)
    w.showGrid(x=False, y=True, alpha=0.15)
    w.getAxis("bottom").setStyle(showValues=False)
    w.setXRange(0, _BUF)
    w.enableAutoRange(axis="y")  # Y scales to whatever's actually in the buffer
    w.setMouseEnabled(x=False, y=True)
    w.setMenuEnabled(False)
    return w


def _buf() -> deque:
    return deque([0.0] * _BUF, maxlen=_BUF)


class _SeriesLabel(QLabel):
    """Colour-coded variable name + live numeric readout — doubles as the
    chart's legend entry, always kept current regardless of the Live toggle
    (which only pauses the plot redraw, not this readout)."""

    def __init__(self, name: str, color: str, fmt: str = "+.3f"):
        super().__init__()
        self._name = name
        self._color = color
        self._fmt = fmt
        self.setStyleSheet(f"color: {TEXT}; font-size: 13px; font-family: Consolas;")
        self.update_value(0.0)

    def update_value(self, value: float):
        self.setText(
            f'<span style="color:{self._color};font-weight:bold;">{self._name}</span> '
            f'{value:{self._fmt}}'
        )


_PAUSED_OPACITY = 0.35
_TITLE_COL_WIDTH = 118


def _chart_row(plot: pg.PlotWidget, series_labels: list[_SeriesLabel],
                checked: bool) -> tuple[QWidget, QCheckBox]:
    """Lay a chart out with its Live checkbox and colour-coded series
    name+value readouts in a column to the left of the plot (instead of
    above it), so the chart itself runs taller rather than the row running
    longer. The readouts double as the legend — no separate legend needed."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)

    left = QWidget()
    left.setFixedWidth(_TITLE_COL_WIDTH)
    left_lay = QVBoxLayout(left)
    left_lay.setContentsMargins(2, 2, 2, 2)
    left_lay.setSpacing(6)

    chk = QCheckBox("Live")
    chk.setChecked(checked)
    chk.setStyleSheet(f"color: {DIM}; font-size: 11px;")
    left_lay.addWidget(chk)

    for lbl in series_labels:
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        left_lay.addWidget(lbl)
    left_lay.addStretch()
    row.addWidget(left)

    effect = QGraphicsOpacityEffect(plot)
    plot.setGraphicsEffect(effect)

    def _apply(on: bool):
        effect.setOpacity(1.0 if on else _PAUSED_OPACITY)

    chk.toggled.connect(_apply)
    _apply(checked)

    row.addWidget(plot, stretch=1)
    return container, chk


def _chart_divider() -> QFrame:
    f = QFrame()
    f.setFixedHeight(3)
    f.setStyleSheet("background-color: #cccccc;")
    return f


class _SimPitchInjector(QWidget):
    """Bench-test slider: drags a fake pitch into the LQR (in place of real
    IMU feedback) and derives the matching fake pitch rate from how fast the
    slider moves. Arms/disarms PARAM_ENABLE_SIM_PITCH_RAD/RATE — there is no
    firmware interlock blocking RUNNING/JUMPING while armed, so this must be
    unchecked before arming the robot for real.
    """

    def __init__(self):
        super().__init__()
        self._armed         = False
        self._last_pitch_rad = 0.0
        self._last_t         = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        chk_row = QHBoxLayout()
        chk_row.setSpacing(6)
        self._chk = QCheckBox("Armed (overrides real IMU pitch)")
        self._chk.setStyleSheet(f"color: {RED}; font-weight: bold; font-size: 11px;")
        self._chk.toggled.connect(self._on_toggle)
        chk_row.addWidget(self._chk)
        chk_row.addStretch()
        outer.addLayout(chk_row)

        slider_row = QHBoxLayout()
        slider_row.setSpacing(6)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(-_SIM_SLIDER_STEPS, _SIM_SLIDER_STEPS)
        self._slider.setValue(0)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._slider.sliderReleased.connect(self._on_released)
        slider_row.addWidget(self._slider, stretch=1)
        self._lbl = QLabel("+0.0°")
        self._lbl.setStyleSheet(
            f"color: {ORANGE}; font-family: Consolas; font-size: 10px;"
        )
        self._lbl.setFixedWidth(52)
        slider_row.addWidget(self._lbl)
        outer.addLayout(slider_row)

    def _raw_to_rad(self, raw: int) -> float:
        return math.radians(raw / _SIM_SLIDER_STEPS * _SIM_PITCH_MAX_DEG)

    def _on_toggle(self, checked: bool):
        self._armed = checked
        self._slider.setEnabled(checked)
        if not checked:
            self._slider.blockSignals(True)
            self._slider.setValue(0)
            self._slider.blockSignals(False)
            self._lbl.setText("+0.0°")
            self._last_pitch_rad = 0.0
            self._last_t = None
            send_param_set(_PARAM_SIM_PITCH_RAD, 0.0)
            send_param_set(_PARAM_SIM_PITCH_RATE_RAD_S, 0.0)
        send_param_set(_PARAM_ENABLE_SIM_PITCH_RAD, 1.0 if checked else 0.0)
        send_param_set(_PARAM_ENABLE_SIM_PITCH_RATE, 1.0 if checked else 0.0)

    def _on_slider_changed(self, raw: int):
        if not self._armed:
            return
        pitch_rad = self._raw_to_rad(raw)
        now  = time.monotonic()
        rate = 0.0
        if self._last_t is not None:
            dt = now - self._last_t
            if dt > 1e-3:
                rate = (pitch_rad - self._last_pitch_rad) / dt
                rate = max(-_SIM_PITCH_RATE_MAX, min(_SIM_PITCH_RATE_MAX, rate))
        self._last_pitch_rad = pitch_rad
        self._last_t = now
        self._lbl.setText(f"{math.degrees(pitch_rad):+.1f}°")
        send_param_set(_PARAM_SIM_PITCH_RAD, pitch_rad)
        send_param_set(_PARAM_SIM_PITCH_RATE_RAD_S, rate)

    def _on_released(self):
        if not self._armed:
            return
        self._last_t = None  # next drag starts its rate estimate fresh
        send_param_set(_PARAM_SIM_PITCH_RATE_RAD_S, 0.0)


class ControllersTab(QWidget):
    def __init__(self):
        super().__init__()

        # ── Rolling buffers ───────────────────────────────────────────────────
        self._b_pitch        = _buf()
        self._b_theta_ref    = _buf()
        self._b_vel_avg      = _buf()
        self._b_v_ref        = _buf()
        self._b_tau_sym      = _buf()
        self._b_whl_tau_l    = _buf()
        self._b_whl_tau_r    = _buf()
        self._b_omega_cmd    = _buf()
        self._b_yaw_rate     = _buf()
        self._b_tau_yaw      = _buf()
        self._b_ff1          = _buf()
        self._b_ff2          = _buf()

        # ── Chart 1: Pitch vs θ_ref ───────────────────────────────────────────
        p1 = _make_plot("deg")
        self._c_pitch_line     = p1.plot(list(self._b_pitch),     pen=pg.mkPen(GREEN,  width=1.5))
        self._c_theta_ref_line = p1.plot(list(self._b_theta_ref), pen=pg.mkPen(YELLOW, width=1.5))
        self._lbl_pitch     = _SeriesLabel("Pitch", GREEN, "+.2f")
        self._lbl_theta_ref = _SeriesLabel("θ_ref", YELLOW, "+.2f")

        # ── Chart 2: Wheel velocity vs v_ref ──────────────────────────────────
        p2 = _make_plot("m/s")
        self._c_vel_avg_line = p2.plot(list(self._b_vel_avg), pen=pg.mkPen(GREEN,  width=1.5))
        self._c_v_ref_line   = p2.plot(list(self._b_v_ref),   pen=pg.mkPen(YELLOW, width=1.5))
        self._lbl_vel_avg = _SeriesLabel("Wheel Vel", GREEN, "+.3f")
        self._lbl_v_ref   = _SeriesLabel("v_ref", YELLOW, "+.3f")

        # ── Chart 3: Torques (tau_sym, whl_tau_l, whl_tau_r) ─────────────────────────
        p3 = _make_plot("N·m")
        self._c_tau_sym_line   = p3.plot(list(self._b_tau_sym),   pen=pg.mkPen(BLUE,   width=2))
        self._c_whl_tau_l_line = p3.plot(list(self._b_whl_tau_l), pen=pg.mkPen(GREEN,  width=1))
        self._c_whl_tau_r_line = p3.plot(list(self._b_whl_tau_r), pen=pg.mkPen(ORANGE, width=1))
        self._lbl_tau_sym   = _SeriesLabel("τ_sym", BLUE, "+.3f")
        self._lbl_whl_tau_l = _SeriesLabel("cmd_L", GREEN, "+.3f")
        self._lbl_whl_tau_r = _SeriesLabel("cmd_R", ORANGE, "+.3f")

        # ── Chart 4: Yaw rate reference vs measured ───────────────────────────
        p4 = _make_plot("deg/s")
        self._c_omega_cmd_line = p4.plot(list(self._b_omega_cmd), pen=pg.mkPen(YELLOW, width=1.5))
        self._c_yaw_rate_line  = p4.plot(list(self._b_yaw_rate),  pen=pg.mkPen(GREEN,  width=1.5))
        self._lbl_omega_cmd = _SeriesLabel("ω_cmd", YELLOW, "+.2f")
        self._lbl_yaw_rate  = _SeriesLabel("ω_meas", GREEN, "+.2f")

        # ── Chart 5: Yaw torque ───────────────────────────────────────────────
        p5 = _make_plot("N·m")
        self._c_tau_yaw_line = p5.plot(list(self._b_tau_yaw), pen=pg.mkPen(ORANGE, width=1.5))
        self._lbl_tau_yaw = _SeriesLabel("τ_yaw", ORANGE, "+.3f")

        # ── Chart 6: Feedforward outputs ──────────────────────────────────────
        p6 = _make_plot("N·m")
        self._c_ff1_line = p6.plot(list(self._b_ff1), pen=pg.mkPen(BLUE,  width=1.5))
        self._c_ff2_line = p6.plot(list(self._b_ff2), pen=pg.mkPen(GREEN, width=1.5))
        self._lbl_ff1 = _SeriesLabel("ff1 hip", BLUE, "+.3f")
        self._lbl_ff2 = _SeriesLabel("ff2 grav", GREEN, "+.3f")

        p1_w, self._chk_p1 = _chart_row(p1, [self._lbl_pitch, self._lbl_theta_ref], checked=True)
        p2_w, self._chk_p2 = _chart_row(p2, [self._lbl_vel_avg, self._lbl_v_ref], checked=False)
        p3_w, self._chk_p3 = _chart_row(
            p3, [self._lbl_tau_sym, self._lbl_whl_tau_l, self._lbl_whl_tau_r], checked=False)
        p4_w, self._chk_p4 = _chart_row(p4, [self._lbl_omega_cmd, self._lbl_yaw_rate], checked=False)
        p5_w, self._chk_p5 = _chart_row(p5, [self._lbl_tau_yaw], checked=False)
        p6_w, self._chk_p6 = _chart_row(p6, [self._lbl_ff1, self._lbl_ff2], checked=False)

        chart_panel = QWidget()
        cl = QVBoxLayout(chart_panel)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(4)
        cl.addWidget(p1_w, 3)
        cl.addWidget(_chart_divider())
        cl.addWidget(p2_w, 2)
        cl.addWidget(_chart_divider())
        cl.addWidget(p3_w, 3)
        cl.addWidget(_chart_divider())
        cl.addWidget(p4_w, 2)
        cl.addWidget(_chart_divider())
        cl.addWidget(p5_w, 2)
        cl.addWidget(_chart_divider())
        cl.addWidget(p6_w, 2)

        # ── Health flag pills ─────────────────────────────────────────────────
        self._pill_lqr     = _StatusPill("LQR Active",  GREEN)
        self._pill_vel_sat = _StatusPill("Vel PI Sat",  YELLOW)
        self._pill_yaw_sat = _StatusPill("Yaw PI Sat",  YELLOW)
        self._pill_wm_l    = _StatusPill("WM-L Lim",    ORANGE)
        self._pill_wm_r    = _StatusPill("WM-R Lim",    ORANGE)

        flags_row = QHBoxLayout()
        flags_row.setSpacing(8)
        flags_row.setContentsMargins(0, 0, 0, 4)
        for p in (self._pill_lqr, self._pill_vel_sat, self._pill_yaw_sat,
                  self._pill_wm_l, self._pill_wm_r):
            flags_row.addWidget(p)
        flags_row.addStretch()

        # ── Live LQR gain labels (from ParamRegistry, not hardcoded) ────────────
        gains_row = QHBoxLayout()
        gains_row.setContentsMargins(0, 0, 0, 0)
        gains_row.setSpacing(0)
        self._gain_lbls: dict[int, QLabel] = {}
        for pid, name in _GAIN_PARAM_LABELS.items():
            g = QLabel(f"{name} = —")
            g.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-size: 10px; padding: 0 12px 0 0;")
            gains_row.addWidget(g)
            self._gain_lbls[pid] = g
        gains_row.addStretch()
        self._gains_requested = False

        # ── Value cells ───────────────────────────────────────────────────────
        self._v_pitch     = _ValueCell("pitch",     "deg")
        self._v_pitch_r   = _ValueCell("pitch_rate","deg/s")
        self._v_vel_avg   = _ValueCell("wheel_vel", "m/s")
        self._v_tau_sym   = _ValueCell("τ_sym",     "N·m")
        self._v_whl_tau_l     = _ValueCell("cmd_L",     "N·m")
        self._v_whl_tau_r     = _ValueCell("cmd_R",     "N·m")

        self._v_theta_ref  = _ValueCell("θ_ref",     "deg")
        self._v_v_ref      = _ValueCell("v_ref",     "m/s")

        self._v_omega_cmd  = _ValueCell("ω_cmd",     "deg/s")
        self._v_tau_yaw    = _ValueCell("τ_yaw",     "N·m")

        self._v_ff1        = _ValueCell("ff1_hip",   "N·m")
        self._v_ff2        = _ValueCell("ff2_grav",  "N·m")

        self._v_jump      = _ValueCell("jump_state", "")
        self._v_loop      = _ValueCell("loop_count", "")

        def _row(*cells) -> QHBoxLayout:
            h = QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            for c in cells:
                h.addWidget(c)
            h.addStretch()
            return h

        info_w = QWidget()
        info_w.setStyleSheet("background: transparent;")
        il = QVBoxLayout(info_w)
        il.setContentsMargins(12, 8, 12, 8)
        il.setSpacing(4)

        il.addLayout(flags_row)
        il.addWidget(_hline())

        il.addWidget(_section_label("Sim Pitch Injection (bench only — no arming interlock)"))
        self._sim_pitch = _SimPitchInjector()
        il.addWidget(self._sim_pitch)
        il.addWidget(_hline())

        il.addWidget(_section_label("LQR Balance Controller"))
        il.addLayout(gains_row)
        il.addLayout(_row(self._v_pitch, self._v_pitch_r, self._v_vel_avg))
        il.addLayout(_row(self._v_tau_sym, self._v_whl_tau_l, self._v_whl_tau_r))
        il.addWidget(_hline())

        il.addWidget(_section_label("Velocity PI  (Phase 3 — active when non-zero)"))
        il.addLayout(_row(self._v_v_ref, self._v_theta_ref))
        il.addWidget(_hline())

        il.addWidget(_section_label("Yaw PI  (Phase 4 — active when non-zero)"))
        il.addLayout(_row(self._v_omega_cmd, self._v_tau_yaw))
        il.addWidget(_hline())

        il.addWidget(_section_label("Feedforward  (Phase 6 — active when non-zero)"))
        il.addLayout(_row(self._v_ff1, self._v_ff2))
        il.addWidget(_hline())

        il.addWidget(_section_label("Diagnostics"))
        il.addLayout(_row(self._v_jump, self._v_loop))
        il.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(info_w)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(chart_panel)
        splitter.setSizes([400, 800])

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        TelemetryBus.instance().packet.connect(self._on_packet)

        self._chart_timer = QTimer()
        self._chart_timer.setInterval(50)  # 20 Hz chart refresh
        self._chart_timer.timeout.connect(self._refresh_charts)
        self._chart_timer.start()

    # ── Packet handler ────────────────────────────────────────────────────────

    def _on_packet(self, info: dict):
        ptype = info.get("ptype")

        if ptype == 0x06:
            pid = info.get("param_id")
            val = info.get("param_value")
            lbl = self._gain_lbls.get(pid)
            if lbl is not None and val is not None:
                lbl.setText(f"{_GAIN_PARAM_LABELS[pid]} = {val:+.5g}")
            return

        if ptype != 0x01:
            return

        if not self._gains_requested:
            self._gains_requested = True
            for pid in _GAIN_PARAM_LABELS:
                send_param_get(pid)

        pitch    = info.get("pitch_rad", 0.0)
        pitch_r  = info.get("pitch_rate_rads", 0.0)
        vel_avg  = info.get("wheel_vel_avg", 0.0)
        whl_tau_l    = info.get("whl_tau_l", 0.0)
        whl_tau_r    = info.get("whl_tau_r", 0.0)
        tau_sym  = info.get("tau_sym", 0.0)
        tau_yaw   = info.get("tau_yaw", 0.0)
        theta     = info.get("theta_ref", 0.0)
        v_ref     = info.get("v_ref", 0.0)
        omega_cmd = info.get("omega_cmd_rds", 0.0)
        yaw_rate  = info.get("yaw_rate_rads", 0.0)
        ff1       = info.get("ff1_out", 0.0)
        ff2       = info.get("ff2_out", 0.0)
        flags    = info.get("health_flags", 0)
        jump     = info.get("jump_state", 0)
        loop     = info.get("loop_count", 0)

        # Angular quantities arrive in radians (firmware units) — display in
        # degrees. Threshold checks below stay on the raw radian value so they
        # don't need re-deriving in a different unit.
        pitch_deg      = math.degrees(pitch)
        pitch_r_deg    = math.degrees(pitch_r)
        theta_deg      = math.degrees(theta)
        omega_cmd_deg  = math.degrees(omega_cmd)
        yaw_rate_deg   = math.degrees(yaw_rate)

        # ── Health flags ──────────────────────────────────────────────────────
        self._pill_lqr.set_active(bool(flags & _HEALTH_LQR_ACTIVE))
        self._pill_vel_sat.set_active(bool(flags & _HEALTH_VEL_PI_SAT))
        self._pill_yaw_sat.set_active(bool(flags & _HEALTH_YAW_PI_SAT))
        self._pill_wm_l.set_active(bool(flags & _HEALTH_WM_L_VEL_LIMITED))
        self._pill_wm_r.set_active(bool(flags & _HEALTH_WM_R_VEL_LIMITED))

        # ── LQR numeric cells ─────────────────────────────────────────────────
        self._v_pitch.set(pitch_deg, "+.2f",
                          RED if abs(pitch) > 0.5 else (ORANGE if abs(pitch) > 0.3 else GREEN))
        self._v_pitch_r.set(pitch_r_deg, "+.2f")
        self._v_vel_avg.set(vel_avg)
        self._v_tau_sym.set(tau_sym, "+.4f",
                            RED if abs(tau_sym) > 4.5 else (ORANGE if abs(tau_sym) > 3.0 else TEXT))
        self._v_whl_tau_l.set(whl_tau_l)
        self._v_whl_tau_r.set(whl_tau_r)

        # ── Velocity PI cells ─────────────────────────────────────────────────
        self._v_v_ref.set(v_ref)
        self._v_theta_ref.set(theta_deg, "+.2f")

        # ── Yaw PI cells ──────────────────────────────────────────────────────
        self._v_omega_cmd.set(omega_cmd_deg, "+.2f")
        self._v_tau_yaw.set(tau_yaw, "+.4f",
                            ORANGE if abs(tau_yaw) > 1.0 else TEXT)

        # ── Feedforward cells ─────────────────────────────────────────────────
        self._v_ff1.set(ff1)
        self._v_ff2.set(ff2)

        # ── Diagnostic cells ──────────────────────────────────────────────────
        self._v_jump.set(int(jump), "d", GREEN if jump > 0 else TEXT)
        self._v_loop.set(int(loop), "d")

        # ── Chart series readouts (always current, independent of Live toggle) ─
        self._lbl_pitch.update_value(pitch_deg)
        self._lbl_theta_ref.update_value(theta_deg)
        self._lbl_vel_avg.update_value(vel_avg)
        self._lbl_v_ref.update_value(v_ref)
        self._lbl_tau_sym.update_value(tau_sym)
        self._lbl_whl_tau_l.update_value(whl_tau_l)
        self._lbl_whl_tau_r.update_value(whl_tau_r)
        self._lbl_omega_cmd.update_value(omega_cmd_deg)
        self._lbl_yaw_rate.update_value(yaw_rate_deg)
        self._lbl_tau_yaw.update_value(tau_yaw)
        self._lbl_ff1.update_value(ff1)
        self._lbl_ff2.update_value(ff2)

        # ── Rolling chart buffers (charts redrawn by timer) ───────────────────
        self._b_pitch.append(pitch_deg)
        self._b_theta_ref.append(theta_deg)
        self._b_vel_avg.append(vel_avg)
        self._b_v_ref.append(v_ref)
        self._b_tau_sym.append(tau_sym)
        self._b_whl_tau_l.append(whl_tau_l)
        self._b_whl_tau_r.append(whl_tau_r)
        self._b_omega_cmd.append(omega_cmd_deg)
        self._b_yaw_rate.append(yaw_rate_deg)
        self._b_tau_yaw.append(tau_yaw)
        self._b_ff1.append(ff1)
        self._b_ff2.append(ff2)

    def _refresh_charts(self):
        if self._chk_p1.isChecked():
            self._c_pitch_line.setData(list(self._b_pitch))
            self._c_theta_ref_line.setData(list(self._b_theta_ref))
        if self._chk_p2.isChecked():
            self._c_vel_avg_line.setData(list(self._b_vel_avg))
            self._c_v_ref_line.setData(list(self._b_v_ref))
        if self._chk_p3.isChecked():
            self._c_tau_sym_line.setData(list(self._b_tau_sym))
            self._c_whl_tau_l_line.setData(list(self._b_whl_tau_l))
            self._c_whl_tau_r_line.setData(list(self._b_whl_tau_r))
        if self._chk_p4.isChecked():
            self._c_omega_cmd_line.setData(list(self._b_omega_cmd))
            self._c_yaw_rate_line.setData(list(self._b_yaw_rate))
        if self._chk_p5.isChecked():
            self._c_tau_yaw_line.setData(list(self._b_tau_yaw))
        if self._chk_p6.isChecked():
            self._c_ff1_line.setData(list(self._b_ff1))
            self._c_ff2_line.setData(list(self._b_ff2))
