from collections import deque

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel,
    QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from telemetry_bus import TelemetryBus
from theme import BG, BORDER, BLUE, DIM, GREEN, ORANGE, RED, SURFACE, TEXT, YELLOW

_BUF = 300  # rolling chart samples

_HEALTH_LQR_ACTIVE       = 1 << 6
_HEALTH_VEL_PI_SAT       = 1 << 7
_HEALTH_YAW_PI_SAT       = 1 << 8
_HEALTH_WM_L_VEL_LIMITED = 1 << 9
_HEALTH_WM_R_VEL_LIMITED = 1 << 10

# Static LQR gains from control_loop.cpp (not sent in telemetry)
_K_PITCH      = -9.77113533
_K_PITCH_RATE = -1.88054364
_K_VEL        = -7.13051190e-03


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


def _make_plot(title: str, ylabel: str) -> pg.PlotWidget:
    w = pg.PlotWidget()
    w.setBackground(BG)
    w.setTitle(title, color=TEXT, size="10pt")
    w.setLabel("left", ylabel, color=DIM)
    w.showGrid(x=False, y=True, alpha=0.15)
    w.getAxis("bottom").setStyle(showValues=False)
    w.setXRange(0, _BUF)
    w.setMouseEnabled(x=False, y=True)
    w.setMenuEnabled(False)
    return w


def _buf() -> deque:
    return deque([0.0] * _BUF, maxlen=_BUF)


class ControllersTab(QWidget):
    def __init__(self):
        super().__init__()

        # ── Rolling buffers ───────────────────────────────────────────────────
        self._b_pitch     = _buf()
        self._b_theta_ref = _buf()
        self._b_vel_avg   = _buf()
        self._b_v_ref     = _buf()
        self._b_tau_sym   = _buf()
        self._b_whl_tau_l     = _buf()
        self._b_whl_tau_r     = _buf()
        self._b_tau_yaw   = _buf()
        self._b_vel_int   = _buf()
        self._b_yaw_int   = _buf()
        self._b_ff1       = _buf()
        self._b_ff2       = _buf()
        self._b_ff4       = _buf()

        # ── Chart 1: Pitch vs θ_ref ───────────────────────────────────────────
        p1 = _make_plot("Pitch vs Lean Setpoint", "rad")
        p1.setYRange(-0.5, 0.5)
        self._c_pitch_line     = p1.plot(list(self._b_pitch),     pen=pg.mkPen(GREEN,  width=1.5), name="pitch")
        self._c_theta_ref_line = p1.plot(list(self._b_theta_ref), pen=pg.mkPen(YELLOW, width=1.5), name="θ_ref")
        p1.addLegend(offset=(5, 5))

        # ── Chart 2: Wheel velocity vs v_ref ──────────────────────────────────
        p2 = _make_plot("Wheel Velocity vs Reference", "m/s")
        p2.setYRange(-2.0, 2.0)
        self._c_vel_avg_line = p2.plot(list(self._b_vel_avg), pen=pg.mkPen(GREEN,  width=1.5), name="vel_avg")
        self._c_v_ref_line   = p2.plot(list(self._b_v_ref),   pen=pg.mkPen(YELLOW, width=1.5), name="v_ref")
        p2.addLegend(offset=(5, 5))

        # ── Chart 3: Torques (tau_sym, whl_tau_l, whl_tau_r) ─────────────────────────
        p3 = _make_plot("Torque Outputs", "N·m")
        p3.setYRange(-6.0, 6.0)
        self._c_tau_sym_line = p3.plot(list(self._b_tau_sym), pen=pg.mkPen(BLUE,   width=2),   name="τ_sym")
        self._c_whl_tau_l_line   = p3.plot(list(self._b_whl_tau_l),   pen=pg.mkPen(GREEN,  width=1),   name="cmd_L")
        self._c_whl_tau_r_line   = p3.plot(list(self._b_whl_tau_r),   pen=pg.mkPen(ORANGE, width=1),   name="cmd_R")
        p3.addLegend(offset=(5, 5))

        # ── Chart 4: Yaw torque ───────────────────────────────────────────────
        p4 = _make_plot("Yaw PI Torque", "N·m")
        p4.setYRange(-2.0, 2.0)
        self._c_tau_yaw_line = p4.plot(list(self._b_tau_yaw), pen=pg.mkPen(ORANGE, width=1.5), name="τ_yaw")
        p4.addLegend(offset=(5, 5))

        # ── Chart 5: PI integrators ───────────────────────────────────────────
        p5 = _make_plot("PI Integrator States", "rad / N·m·s")
        p5.setYRange(-0.3, 0.3)
        self._c_vel_int_line = p5.plot(list(self._b_vel_int), pen=pg.mkPen(YELLOW, width=1.5), name="vel_int")
        self._c_yaw_int_line = p5.plot(list(self._b_yaw_int), pen=pg.mkPen(ORANGE, width=1.5), name="yaw_int")
        p5.addLegend(offset=(5, 5))

        # ── Chart 6: Feedforward outputs ──────────────────────────────────────
        p6 = _make_plot("Feedforward Outputs (Phase 6)", "N·m / rad")
        p6.setYRange(-2.0, 2.0)
        self._c_ff1_line = p6.plot(list(self._b_ff1), pen=pg.mkPen(BLUE,   width=1.5), name="ff1 hip")
        self._c_ff2_line = p6.plot(list(self._b_ff2), pen=pg.mkPen(GREEN,  width=1.5), name="ff2 grav")
        self._c_ff4_line = p6.plot(list(self._b_ff4), pen=pg.mkPen(ORANGE, width=1.5), name="ff4 cent")
        p6.addLegend(offset=(5, 5))

        chart_panel = QWidget()
        cl = QVBoxLayout(chart_panel)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(4)
        cl.addWidget(p1, 3)
        cl.addWidget(p2, 2)
        cl.addWidget(p3, 3)
        cl.addWidget(p4, 2)
        cl.addWidget(p5, 2)
        cl.addWidget(p6, 2)

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

        # ── Static LQR gain labels ─────────────────────────────────────────────
        gains_row = QHBoxLayout()
        gains_row.setContentsMargins(0, 0, 0, 0)
        gains_row.setSpacing(0)
        for label, val in (("K_pitch", _K_PITCH), ("K_ṗ", _K_PITCH_RATE), ("K_vel", _K_VEL)):
            g = QLabel(f"{label} = {val:+.5g}")
            g.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-size: 10px; padding: 0 18px 0 0;")
            gains_row.addWidget(g)
        gains_row.addStretch()

        # ── Value cells ───────────────────────────────────────────────────────
        self._v_pitch     = _ValueCell("pitch",     "rad")
        self._v_pitch_r   = _ValueCell("pitch_rate","rad/s")
        self._v_vel_avg   = _ValueCell("wheel_vel", "m/s")
        self._v_tau_sym   = _ValueCell("τ_sym",     "N·m")
        self._v_whl_tau_l     = _ValueCell("cmd_L",     "N·m")
        self._v_whl_tau_r     = _ValueCell("cmd_R",     "N·m")

        self._v_theta_ref = _ValueCell("θ_ref",     "rad")
        self._v_v_ref     = _ValueCell("v_ref",     "m/s")
        self._v_vel_int   = _ValueCell("vel_int",   "rad")

        self._v_tau_yaw   = _ValueCell("τ_yaw",     "N·m")
        self._v_yaw_int   = _ValueCell("yaw_int",   "N·m·s")

        self._v_ff1       = _ValueCell("ff1_hip",   "N·m")
        self._v_ff2       = _ValueCell("ff2_grav",  "N·m")
        self._v_ff4       = _ValueCell("ff4_cent",  "rad")

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

        il.addWidget(_section_label("LQR Balance Controller"))
        il.addLayout(gains_row)
        il.addLayout(_row(self._v_pitch, self._v_pitch_r, self._v_vel_avg))
        il.addLayout(_row(self._v_tau_sym, self._v_whl_tau_l, self._v_whl_tau_r))
        il.addWidget(_hline())

        il.addWidget(_section_label("Velocity PI  (Phase 3 — active when non-zero)"))
        il.addLayout(_row(self._v_theta_ref, self._v_v_ref, self._v_vel_int))
        il.addWidget(_hline())

        il.addWidget(_section_label("Yaw PI  (Phase 4 — active when non-zero)"))
        il.addLayout(_row(self._v_tau_yaw, self._v_yaw_int))
        il.addWidget(_hline())

        il.addWidget(_section_label("Feedforward  (Phase 6 — active when non-zero)"))
        il.addLayout(_row(self._v_ff1, self._v_ff2, self._v_ff4))
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
        if info.get("ptype") != 0x01:
            return

        pitch    = info.get("pitch_rad", 0.0)
        pitch_r  = info.get("pitch_rate_rads", 0.0)
        vel_avg  = info.get("wheel_vel_avg", 0.0)
        whl_tau_l    = info.get("whl_tau_l", 0.0)
        whl_tau_r    = info.get("whl_tau_r", 0.0)
        tau_sym  = info.get("tau_sym", 0.0)
        tau_yaw  = info.get("tau_yaw", 0.0)
        theta    = info.get("theta_ref", 0.0)
        v_ref    = info.get("v_ref", 0.0)
        vel_int  = info.get("vel_err_integral", 0.0)
        yaw_int  = info.get("yaw_err_integral", 0.0)
        ff1      = info.get("ff1_out", 0.0)
        ff2      = info.get("ff2_out", 0.0)
        ff4      = info.get("ff4_out", 0.0)
        flags    = info.get("health_flags", 0)
        jump     = info.get("jump_state", 0)
        loop     = info.get("loop_count", 0)

        # ── Health flags ──────────────────────────────────────────────────────
        self._pill_lqr.set_active(bool(flags & _HEALTH_LQR_ACTIVE))
        self._pill_vel_sat.set_active(bool(flags & _HEALTH_VEL_PI_SAT))
        self._pill_yaw_sat.set_active(bool(flags & _HEALTH_YAW_PI_SAT))
        self._pill_wm_l.set_active(bool(flags & _HEALTH_WM_L_VEL_LIMITED))
        self._pill_wm_r.set_active(bool(flags & _HEALTH_WM_R_VEL_LIMITED))

        # ── LQR numeric cells ─────────────────────────────────────────────────
        self._v_pitch.set(pitch, "+.4f",
                          RED if abs(pitch) > 0.5 else (ORANGE if abs(pitch) > 0.3 else GREEN))
        self._v_pitch_r.set(pitch_r)
        self._v_vel_avg.set(vel_avg)
        self._v_tau_sym.set(tau_sym, "+.4f",
                            RED if abs(tau_sym) > 4.5 else (ORANGE if abs(tau_sym) > 3.0 else TEXT))
        self._v_whl_tau_l.set(whl_tau_l)
        self._v_whl_tau_r.set(whl_tau_r)

        # ── Velocity PI cells ─────────────────────────────────────────────────
        self._v_theta_ref.set(theta)
        self._v_v_ref.set(v_ref)
        self._v_vel_int.set(vel_int)

        # ── Yaw PI cells ──────────────────────────────────────────────────────
        self._v_tau_yaw.set(tau_yaw, "+.4f",
                            ORANGE if abs(tau_yaw) > 1.0 else TEXT)
        self._v_yaw_int.set(yaw_int)

        # ── Feedforward cells ─────────────────────────────────────────────────
        self._v_ff1.set(ff1)
        self._v_ff2.set(ff2)
        self._v_ff4.set(ff4)

        # ── Diagnostic cells ──────────────────────────────────────────────────
        self._v_jump.set(int(jump), "d", GREEN if jump > 0 else TEXT)
        self._v_loop.set(int(loop), "d")

        # ── Rolling chart buffers (charts redrawn by timer) ───────────────────
        self._b_pitch.append(pitch)
        self._b_theta_ref.append(theta)
        self._b_vel_avg.append(vel_avg)
        self._b_v_ref.append(v_ref)
        self._b_tau_sym.append(tau_sym)
        self._b_whl_tau_l.append(whl_tau_l)
        self._b_whl_tau_r.append(whl_tau_r)
        self._b_tau_yaw.append(tau_yaw)
        self._b_vel_int.append(vel_int)
        self._b_yaw_int.append(yaw_int)
        self._b_ff1.append(ff1)
        self._b_ff2.append(ff2)
        self._b_ff4.append(ff4)

    def _refresh_charts(self):
        self._c_pitch_line.setData(list(self._b_pitch))
        self._c_theta_ref_line.setData(list(self._b_theta_ref))
        self._c_vel_avg_line.setData(list(self._b_vel_avg))
        self._c_v_ref_line.setData(list(self._b_v_ref))
        self._c_tau_sym_line.setData(list(self._b_tau_sym))
        self._c_whl_tau_l_line.setData(list(self._b_whl_tau_l))
        self._c_whl_tau_r_line.setData(list(self._b_whl_tau_r))
        self._c_tau_yaw_line.setData(list(self._b_tau_yaw))
        self._c_vel_int_line.setData(list(self._b_vel_int))
        self._c_yaw_int_line.setData(list(self._b_yaw_int))
        self._c_ff1_line.setData(list(self._b_ff1))
        self._c_ff2_line.setData(list(self._b_ff2))
        self._c_ff4_line.setData(list(self._b_ff4))
