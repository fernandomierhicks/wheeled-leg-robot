"""sandbox_fastchart.py — sandbox.py with pyqtgraph telemetry instead of matplotlib.

Identical simulation / control logic; _plot_process replaced with pyqtgraph.
Full 5×2 live chart layout at 60 Hz, 15-second rolling window.

Charts:
  [0,0] Pitch (deg)          [0,1] Pitch Rate (deg/s)
  [1,0] Velocity (m/s)       [1,1] Yaw Rate (deg/s)
  [2,0] Hip Joints (deg)     [2,1] Roll (deg)
  [3,0] Wheel Torque (N·m)   [3,1] Hip Torque (N·m)  — both with ± limit lines
  [4,0] Battery V + I_bat    [4,1] Motor Currents (A)
Status bar: SoC% | Batt Temp | I_bat | I_bat max | hover Y-value   (labels @ 3 Hz)

Telemetry tuple — 25 values:
   0  t
   1  pitch_deg        2  pitch_ref_deg
   3  pitch_rate_deg
   4  vel_ms           5  v_cmd_ms
   6  yaw_rate_deg     7  omega_cmd_deg
   8  hip_L_deg        9  hip_R_deg
  10  hip_cmd_L_deg   11  hip_cmd_R_deg   ← per-leg (differ due to roll leveling)
  12  roll_deg
  13  tau_whl_L       14  tau_whl_R
  15  v_batt          16  batt_temp_c     17  soc_pct
  18  I_whl_L         19  I_whl_R
  20  I_hip_L         21  I_hip_R         22  I_total
  23  tau_hip_L       24  tau_hip_R

Usage:
    python sandbox_fastchart.py
    python sandbox_fastchart.py --slowmo 2
"""
import ctypes
import math
import multiprocessing as mp
import os
import sys
import threading
import time
import argparse
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

try:
    import pygame
    import os as _os
    _os.environ['SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS'] = '1'
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sim_config import (
    ROBOT, Q_NOM, Q_RET, Q_EXT, WHEEL_R,
    HIP_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT, WHEEL_OMEGA_NOLOAD,
    LEG_K_S, LEG_B_S, LEG_K_ROLL, LEG_D_ROLL,
    HIP_POSITION_KP, HIP_POSITION_KD,
    ROLL_NOISE_STD_RAD, HIP_SAFE_MIN, HIP_SAFE_MAX,
    LQR_Q_PITCH, LQR_Q_PITCH_RATE, LQR_Q_VEL, LQR_R,
    PITCH_NOISE_STD_RAD, PITCH_RATE_NOISE_STD_RAD_S,
    CTRL_STEPS, THETA_REF_RATE_LIMIT,
    VELOCITY_PI_KP, VELOCITY_PI_KI,
    YAW_PI_KP, YAW_PI_KI,
    BATT_V_NOM, WHEEL_KT, HIP_KT_OUTPUT, BATT_I_QUIESCENT,
    USE_LATENCY_MODEL, SENSOR_DELAY_S, ACTUATOR_DELAY_S,
)
from physics import build_xml, build_assets, get_equilibrium_pitch
from scenarios import init_sim, get_pitch_and_rate, lqr_torque, VelocityPI, YawPI, motor_taper, _motor_currents
from battery_model import BatteryModel
from lqr_design import compute_gain_table
import scenarios as _scenarios

RENDER_HZ    = 60
TELEMETRY_HZ = 60
WINDOW_S     = 15.0

_HIP_MAX_Q   = Q_EXT + math.radians(10)
_HIP_NOM_PCT = (Q_NOM - Q_RET) / (_HIP_MAX_Q - Q_RET) * 100.0


# ---------------------------------------------------------------------------
# Sandbox obstacle layout  (unchanged from sandbox.py)
# ---------------------------------------------------------------------------
SANDBOX_OBSTACLES = [
    dict(shape="sphere", x= 3.0, y= 0.0, r=1.20, h=0.06),
    dict(shape="sphere", x=-2.5, y= 1.5, r=0.80, h=0.05),
    dict(shape="sphere", x= 5.5, y=-2.0, r=1.00, h=0.07),
    dict(shape="sphere", x=-5.0, y=-1.0, r=0.90, h=0.06),

    dict(shape="capsule", x= 1.5, y= 0.0, r=0.020, length=1.20),
    dict(shape="capsule", x=-1.2, y= 0.5, r=0.025, length=0.80),
    dict(shape="capsule", x= 4.0, y= 1.0, r=0.030, length=1.00),
    dict(shape="capsule", x= 2.5, y=-1.5, r=0.020, length=0.70),
    dict(shape="capsule", x=-3.5, y=-2.0, r=0.035, length=1.20),

    dict(shape="ramp", x= 2.0, y=-0.5, angle_deg= 8, length=0.60, width=0.50, h=0.050),
    dict(shape="ramp", x=-2.0, y= 2.0, angle_deg= 6, length=0.80, width=0.60, h=0.040),
    dict(shape="ramp", x= 6.0, y= 0.5, angle_deg=10, length=0.50, width=0.60, h=0.060),
    dict(shape="ramp", x=-4.0, y=-3.0, angle_deg= 7, length=0.70, width=0.50, h=0.045),

    dict(shape="box", x= 1.0, y= 2.0, rx=0.12, ry=0.20, h=0.02),
    dict(shape="box", x=-1.5, y=-2.5, rx=0.15, ry=0.15, h=0.03),
    dict(shape="cyl", x= 4.5, y= 3.0, r=0.10,           h=0.05),
    dict(shape="box", x= 7.0, y=-1.5, rx=0.20, ry=0.25, h=0.07),
    dict(shape="box", x=-6.0, y= 2.0, rx=0.20, ry=0.20, h=0.06),
]

SANDBOX_PROPS = [
    dict(type="can",           x= 0.8, y= 0.3),
    dict(type="can",           x= 0.8, y=-0.3),
    dict(type="bottle",        x= 1.5, y=-0.8),
    dict(type="ball",          x= 0.6, y= 0.8),
    dict(type="ball",          x= 0.5, y=-0.6),
    dict(type="cardboard_box", x= 2.0, y= 0.5),
]


# ---------------------------------------------------------------------------
# Window positioning helpers (Windows)
# ---------------------------------------------------------------------------
def _screen_size():
    u32 = ctypes.windll.user32
    u32.SetProcessDPIAware()
    return u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)

def _move_window(hwnd, x, y, w, h):
    ctypes.windll.user32.SetWindowPos(hwnd, 0, x, y, w, h, 0x0004)

def _find_window_by_partial_title(partial: str) -> int:
    result = [0]
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
    def _cb(hwnd, _):
        if ctypes.windll.user32.IsWindowVisible(hwnd):
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            if partial.lower() in buf.value.lower():
                result[0] = hwnd
                return False
        return True
    ctypes.windll.user32.EnumWindows(EnumWindowsProc(_cb), 0)
    return result[0]

def _position_mujoco_right(delay: float = 2.0):
    time.sleep(delay)
    sw, sh = _screen_size()
    half = sw // 2
    for _ in range(20):
        hwnd = _find_window_by_partial_title("MuJoCo")
        if hwnd:
            _move_window(hwnd, half, 0, half, sh)
            return
        time.sleep(0.2)


# ---------------------------------------------------------------------------
# pyqtgraph telemetry panel  (separate process)
# ---------------------------------------------------------------------------
def _plot_process(data_q: mp.Queue, cmd_q: mp.Queue,
                  window_s: float, title: str,
                  wheel_limit: float, hip_limit: float) -> None:
    import sys as _sys
    import time as _time
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

    pg.setConfigOptions(antialias=False, useOpenGL=True, enableExperimental=True)
    app = QtWidgets.QApplication(_sys.argv)

    # ── Main window ───────────────────────────────────────────────────────────
    main_win = QtWidgets.QMainWindow()
    main_win.setWindowTitle(title)
    main_win.setStyleSheet("background:#12121e;")
    central = QtWidgets.QWidget()
    main_win.setCentralWidget(central)
    vbox = QtWidgets.QVBoxLayout(central)
    vbox.setContentsMargins(4, 4, 4, 2)
    vbox.setSpacing(2)

    glw = pg.GraphicsLayoutWidget()
    glw.setBackground("#12121e")
    vbox.addWidget(glw, stretch=1)

    # Status bar
    status_row = QtWidgets.QWidget()
    status_row.setStyleSheet("background:#1a1a2e; border-radius:4px;")
    hbox = QtWidgets.QHBoxLayout(status_row)
    hbox.setContentsMargins(6, 3, 6, 3)
    hbox.setSpacing(0)

    _SL = ("color:#e8e8e8; font-family:Consolas,monospace; "
           "font-size:11px; font-weight:bold; padding:0 14px 0 0;")

    lbl_soc      = QtWidgets.QLabel("SoC: --")
    lbl_temp     = QtWidgets.QLabel("Temp: --")
    lbl_ibat     = QtWidgets.QLabel("I_bat: --")
    lbl_ibat_max = QtWidgets.QLabel("I_bat max: --")
    hover_lbl    = QtWidgets.QLabel("")

    for lbl in (lbl_soc, lbl_temp, lbl_ibat, lbl_ibat_max):
        lbl.setStyleSheet(_SL)
        hbox.addWidget(lbl)

    hbox.addStretch()
    hover_lbl.setStyleSheet(
        "color:#80ffb0; font-family:Consolas,monospace; font-size:11px; padding:0 12px;")
    hbox.addWidget(hover_lbl)

    btn = QtWidgets.QPushButton("⟳  Restart")
    btn.setFixedHeight(26)
    btn.setStyleSheet(
        "QPushButton{background:#3a3a5e;color:white;font-size:11px;"
        "border-radius:4px;padding:0 10px}"
        "QPushButton:hover{background:#5a5a9e}")
    btn.clicked.connect(lambda: cmd_q.put_nowait("RESTART"))
    hbox.addWidget(btn)
    vbox.addWidget(status_row)

    # ── Controller enable/disable checkboxes ──────────────────────────────────
    ctrl_row = QtWidgets.QWidget()
    ctrl_row.setStyleSheet("background:#1a1a2e; border-radius:4px;")
    hbox_ctrl = QtWidgets.QHBoxLayout(ctrl_row)
    hbox_ctrl.setContentsMargins(6, 3, 6, 3)
    hbox_ctrl.setSpacing(16)

    lbl_ctrl = QtWidgets.QLabel("Controllers:")
    lbl_ctrl.setStyleSheet(_SL)
    hbox_ctrl.addWidget(lbl_ctrl)

    _CB_STYLE = ("QCheckBox{color:#e8e8e8;font-family:Consolas,monospace;"
                 "font-size:11px;font-weight:bold;spacing:5px}"
                 "QCheckBox::indicator{width:14px;height:14px}"
                 "QCheckBox::indicator:checked{background:#4a9eff;border:1px solid #4a9eff;border-radius:2px}"
                 "QCheckBox::indicator:unchecked{background:#333;border:1px solid #666;border-radius:2px}")

    _ctrl_defs = [
        ("LQR",        "LQR Balance"),
        ("VelPI",      "Vel PI"),
        ("YawPI",      "Yaw PI"),
        ("Suspension", "Suspension"),
        ("RollLev",    "Roll Leveling"),
    ]
    for key, label in _ctrl_defs:
        cb = QtWidgets.QCheckBox(label)
        cb.setChecked(True)
        cb.setStyleSheet(_CB_STYLE)
        cb.stateChanged.connect(
            lambda state, k=key: cmd_q.put_nowait(("CTRL_EN", k, bool(state))))
        hbox_ctrl.addWidget(cb)

    hbox_ctrl.addStretch()

    # Hip mode toggle button
    _btn_hip = QtWidgets.QPushButton("Hip: Impedance")
    _btn_hip.setFixedHeight(24)
    _btn_hip.setCheckable(True)
    _btn_hip.setChecked(False)   # False = Impedance, True = Position PD
    _btn_hip.setStyleSheet(
        "QPushButton{background:#3a3a5e;color:#e8e8e8;font-family:Consolas,monospace;"
        "font-size:11px;font-weight:bold;border-radius:4px;padding:0 10px}"
        "QPushButton:checked{background:#5a3a1e;color:#ffb060}"
        "QPushButton:hover{background:#4a4a7e}")

    def _on_hip_toggle(checked):
        _btn_hip.setText("Hip: Pos PD" if checked else "Hip: Impedance")
        cmd_q.put_nowait(("HIP_MODE", "position_pd" if checked else "impedance"))

    _btn_hip.toggled.connect(_on_hip_toggle)
    hbox_ctrl.addWidget(_btn_hip)

    vbox.addWidget(ctrl_row)

    # Position window on left half of primary monitor
    try:
        screen = app.primaryScreen()
        rect   = screen.geometry()
        main_win.setGeometry(rect.x(), rect.y(), rect.width() // 2, rect.height())
    except Exception:
        main_win.resize(960, 1000)

    main_win.show()

    # ── Style helpers ─────────────────────────────────────────────────────────
    TICK_FONT  = QtGui.QFont("Consolas", 9)
    TICK_PEN   = pg.mkColor('#d8d8d8')
    DASH       = QtCore.Qt.PenStyle.DashLine
    W          = 1.4

    def _p(row, col, ttl, ylabel):
        pl = glw.addPlot(row=row, col=col)
        pl.setTitle(
            f'<span style="color:#e0e0e0;font-size:9pt;font-weight:600">{ttl}</span>')
        pl.setLabel(
            "left",
            f'<span style="color:#c8c8c8;font-size:9pt">{ylabel}</span>')
        pl.showGrid(x=True, y=True, alpha=0.20)
        for ax_name in ("left", "bottom"):
            ax = pl.getAxis(ax_name)
            ax.setTextPen(TICK_PEN)
            ax.setPen(pg.mkPen('#555'))
            ax.setStyle(tickFont=TICK_FONT)
        pl.setXRange(-window_s, 0, padding=0.02)
        return pl

    def _leg(pl, ncols=1):
        leg = pl.addLegend(offset=(6, 6), verSpacing=-4, colCount=ncols)
        leg.setBrush(pg.mkBrush(18, 18, 36, 210))
        leg.setPen(pg.mkPen('#444'))
        leg.setLabelTextColor(pg.mkColor('#cccccc'))
        return leg

    def _limits(pl, val, color='#ff4444'):
        for sign in (+1, -1):
            anchor = (0.05, 1.1) if sign > 0 else (0.05, -0.1)
            il = pg.InfiniteLine(
                pos=sign * val, angle=0,
                pen=pg.mkPen(color, width=1.2, style=DASH),
                label=f'{"+" if sign>0 else "−"}{val:.1f}',
                labelOpts={"color": color, "anchors": [anchor, anchor]})
            pl.addItem(il)

    # ── Row 0: Pitch | Pitch Rate ─────────────────────────────────────────────
    p_pitch = _p(0, 0, "Pitch", "deg")
    _leg(p_pitch)
    ln_pitch     = p_pitch.plot(pen=pg.mkPen('#60d0ff', width=W), name="pitch")
    ln_pitch_ref = p_pitch.plot(pen=pg.mkPen('#ff6060', width=W, style=DASH), name="cmd")

    p_prate = _p(0, 1, "Pitch Rate", "deg/s")
    _leg(p_prate)
    ln_prate = p_prate.plot(pen=pg.mkPen('#ffa040', width=W), name="pitch rate")

    # ── Row 1: Velocity | Yaw Rate ────────────────────────────────────────────
    p_vel = _p(1, 0, "Velocity", "m/s")
    _leg(p_vel)
    ln_vel  = p_vel.plot(pen=pg.mkPen('#60d0ff', width=W), name="vel")
    ln_vcmd = p_vel.plot(pen=pg.mkPen('#ff6060', width=W, style=DASH), name="cmd")

    p_yaw = _p(1, 1, "Yaw Rate", "deg/s")
    _leg(p_yaw)
    ln_yaw  = p_yaw.plot(pen=pg.mkPen('#60d0ff', width=W), name="ω")
    ln_ocmd = p_yaw.plot(pen=pg.mkPen('#ff6060', width=W, style=DASH), name="cmd")

    # ── Row 2: Hip Joints | Roll ──────────────────────────────────────────────
    # Four lines: actual L/R (solid) + cmd L/R (dashed, differ due to roll leveling)
    p_hip = _p(2, 0, "Hip Joints", "deg")
    _leg(p_hip, ncols=2)
    ln_hip_L      = p_hip.plot(pen=pg.mkPen('#60d0ff', width=W),          name="act L")
    ln_hip_R      = p_hip.plot(pen=pg.mkPen('#80ff80', width=W),          name="act R")
    ln_hip_cmd_L  = p_hip.plot(pen=pg.mkPen('#ff7070', width=W, style=DASH), name="cmd L")
    ln_hip_cmd_R  = p_hip.plot(pen=pg.mkPen('#ffe060', width=W, style=DASH), name="cmd R")

    p_roll = _p(2, 1, "Roll", "deg")
    _leg(p_roll)
    ln_roll = p_roll.plot(pen=pg.mkPen('#ffa040', width=W), name="roll")

    # ── Row 3: Wheel Torque | Hip Torque  (± limit lines) ────────────────────
    p_tau = _p(3, 0, "Wheel Torque", "N·m")
    _leg(p_tau)
    _limits(p_tau, wheel_limit)
    ln_tau_L = p_tau.plot(pen=pg.mkPen('#60d0ff', width=W), name="L")
    ln_tau_R = p_tau.plot(pen=pg.mkPen('#80ff80', width=W), name="R")

    p_htau = _p(3, 1, "Hip Torque", "N·m")
    _leg(p_htau)
    _limits(p_htau, hip_limit)
    ln_htau_L = p_htau.plot(pen=pg.mkPen('#60d0ff', width=W), name="L")
    ln_htau_R = p_htau.plot(pen=pg.mkPen('#80ff80', width=W), name="R")

    # ── Row 4: Battery V + I_total (dual Y) | Motor Currents ─────────────────
    p_batt = _p(4, 0, "Battery", "V")
    leg_batt = _leg(p_batt)
    ln_vbatt = p_batt.plot(pen=pg.mkPen('#60d0ff', width=W), name="V_bat")

    # Second Y axis for battery current
    p_batt.showAxis('right')
    ax_right = p_batt.getAxis('right')
    ax_right.setLabel('<span style="color:#ffa040;font-size:9pt">A</span>')
    ax_right.setTextPen(TICK_PEN)
    ax_right.setPen(pg.mkPen('#555'))
    ax_right.setStyle(tickFont=TICK_FONT)

    _batt_vb2 = pg.ViewBox()
    p_batt.scene().addItem(_batt_vb2)
    ax_right.linkToView(_batt_vb2)
    _batt_vb2.setXLink(p_batt)

    ln_ibat_batt = pg.PlotDataItem(pen=pg.mkPen('#ffa040', width=W), name="I_bat")
    _batt_vb2.addItem(ln_ibat_batt)
    leg_batt.addItem(ln_ibat_batt, "I_bat")

    def _sync_batt():
        _batt_vb2.setGeometry(p_batt.vb.sceneBoundingRect())
        _batt_vb2.linkedViewChanged(p_batt.vb, _batt_vb2.XAxis)

    p_batt.vb.sigResized.connect(_sync_batt)

    p_cur = _p(4, 1, "Motor Currents", "A")
    _leg(p_cur, ncols=2)
    ln_iwhl_L = p_cur.plot(pen=pg.mkPen('#60d0ff', width=W),            name="whl L")
    ln_iwhl_R = p_cur.plot(pen=pg.mkPen('#80ff80', width=W),            name="whl R")
    ln_ihip_L = p_cur.plot(pen=pg.mkPen('#ffa040', width=W),            name="hip L")
    ln_ihip_R = p_cur.plot(pen=pg.mkPen('#ffe060', width=W),            name="hip R")
    ln_itotal = p_cur.plot(pen=pg.mkPen('#e0e0e0', width=W, style=DASH), name="total")

    # ── Mouse hover Y label ───────────────────────────────────────────────────
    named_plots = [
        ("Pitch",         p_pitch), ("Pitch Rate",     p_prate),
        ("Velocity",      p_vel),   ("Yaw Rate",       p_yaw),
        ("Hip Joints",    p_hip),   ("Roll",           p_roll),
        ("Wheel Torque",  p_tau),   ("Hip Torque",     p_htau),
        ("Battery",       p_batt),  ("Motor Currents", p_cur),
    ]

    def _on_mouse(evt):
        pos = evt[0]
        for name, pl in named_plots:
            if pl.sceneBoundingRect().contains(pos):
                mp = pl.vb.mapSceneToView(pos)
                hover_lbl.setText(f"▶ {name}   y = {mp.y():.3f}")
                return
        hover_lbl.setText("")

    _proxy = pg.SignalProxy(glw.scene().sigMouseMoved, rateLimit=60, slot=_on_mouse)

    # ── Ring buffers — 25 channels ────────────────────────────────────────────
    MAXLEN = int(window_s * TELEMETRY_HZ) + 200
    (t_buf,
     pitch_buf, pitch_ref_buf, pitch_rate_buf,
     vel_buf, v_cmd_buf,
     yaw_rate_buf, omega_cmd_buf,
     hip_L_buf, hip_R_buf, hip_cmd_L_buf, hip_cmd_R_buf,
     roll_buf,
     tau_whl_L_buf, tau_whl_R_buf,
     v_batt_buf, batt_temp_buf, soc_buf,
     I_whl_L_buf, I_whl_R_buf, I_hip_L_buf, I_hip_R_buf, I_total_buf,
     tau_hip_L_buf, tau_hip_R_buf,
     ) = (deque(maxlen=MAXLEN) for _ in range(25))

    all_bufs = [
        t_buf, pitch_buf, pitch_ref_buf, pitch_rate_buf,
        vel_buf, v_cmd_buf, yaw_rate_buf, omega_cmd_buf,
        hip_L_buf, hip_R_buf, hip_cmd_L_buf, hip_cmd_R_buf, roll_buf,
        tau_whl_L_buf, tau_whl_R_buf,
        v_batt_buf, batt_temp_buf, soc_buf,
        I_whl_L_buf, I_whl_R_buf, I_hip_L_buf, I_hip_R_buf, I_total_buf,
        tau_hip_L_buf, tau_hip_R_buf,
    ]

    all_lines = [
        ln_pitch, ln_pitch_ref, ln_prate,
        ln_vel, ln_vcmd, ln_yaw, ln_ocmd,
        ln_hip_L, ln_hip_R, ln_hip_cmd_L, ln_hip_cmd_R, ln_roll,
        ln_tau_L, ln_tau_R, ln_vbatt, ln_ibat_batt,
        ln_iwhl_L, ln_iwhl_R, ln_ihip_L, ln_ihip_R, ln_itotal,
        ln_htau_L, ln_htau_R,
    ]

    i_bat_max       = [0.0]
    _last_stat_t    = [0.0]
    _stat_interval  = 1.0 / 3.0   # status labels update at 3 Hz

    def _reset_bufs():
        for b in all_bufs:
            b.clear()
        for ln in all_lines:
            ln.setData([], [])
        i_bat_max[0] = 0.0
        for lbl in (lbl_soc, lbl_temp, lbl_ibat, lbl_ibat_max):
            lbl.setText(lbl.text().split(":")[0] + ": --")

    # ── 60 Hz update callback ─────────────────────────────────────────────────
    def _update():
        # Drain queue — just fill buffers, track running i_bat_max
        while True:
            try:
                item = data_q.get_nowait()
            except Exception:
                break
            if item is None:
                app.quit()
                return
            if item == "RESET":
                _reset_bufs()
                continue

            (t, pitch, pitch_ref, pitch_rate,
             vel, v_cmd, yaw_rate, omega_cmd,
             hip_L, hip_R, hip_cmd_L, hip_cmd_R, roll,
             tau_whl_L, tau_whl_R,
             v_batt, batt_temp, soc,
             I_whl_L, I_whl_R, I_hip_L, I_hip_R, I_total,
             tau_hip_L, tau_hip_R) = item

            t_buf.append(t)
            pitch_buf.append(pitch);          pitch_ref_buf.append(pitch_ref)
            pitch_rate_buf.append(pitch_rate)
            vel_buf.append(vel);              v_cmd_buf.append(v_cmd)
            yaw_rate_buf.append(yaw_rate);    omega_cmd_buf.append(omega_cmd)
            hip_L_buf.append(hip_L);         hip_R_buf.append(hip_R)
            hip_cmd_L_buf.append(hip_cmd_L); hip_cmd_R_buf.append(hip_cmd_R)
            roll_buf.append(roll)
            tau_whl_L_buf.append(tau_whl_L); tau_whl_R_buf.append(tau_whl_R)
            v_batt_buf.append(v_batt);        batt_temp_buf.append(batt_temp)
            soc_buf.append(soc)
            I_whl_L_buf.append(I_whl_L);     I_whl_R_buf.append(I_whl_R)
            I_hip_L_buf.append(I_hip_L);     I_hip_R_buf.append(I_hip_R)
            I_total_buf.append(I_total)
            i_bat_max[0] = max(i_bat_max[0], I_total)
            tau_hip_L_buf.append(tau_hip_L); tau_hip_R_buf.append(tau_hip_R)

        if len(t_buf) < 2:
            return

        # Status labels — throttled to 3 Hz
        import time as _t
        now = _t.perf_counter()
        if now - _last_stat_t[0] >= _stat_interval:
            lbl_soc.setText(f"SoC: {soc_buf[-1]:.1f}%")
            lbl_temp.setText(f"Temp: {batt_temp_buf[-1]:.1f}°C")
            lbl_ibat.setText(f"I_bat: {I_total_buf[-1]:.1f} A")
            lbl_ibat_max.setText(f"I_bat max: {i_bat_max[0]:.1f} A")
            _last_stat_t[0] = now

        # Compute visible window slice
        tb    = np.array(t_buf)
        sim_t = float(tb[-1])
        t0    = max(0.0, sim_t - window_s)
        idx   = int(np.searchsorted(tb, t0))
        xw    = tb[idx:] - sim_t   # x = seconds relative to now

        def _a(buf): return np.array(buf)[idx:]

        ln_pitch.setData(xw, _a(pitch_buf))
        ln_pitch_ref.setData(xw, _a(pitch_ref_buf))
        ln_prate.setData(xw, _a(pitch_rate_buf))
        ln_vel.setData(xw, _a(vel_buf))
        ln_vcmd.setData(xw, _a(v_cmd_buf))
        ln_yaw.setData(xw, _a(yaw_rate_buf))
        ln_ocmd.setData(xw, _a(omega_cmd_buf))
        ln_hip_L.setData(xw, _a(hip_L_buf))
        ln_hip_R.setData(xw, _a(hip_R_buf))
        ln_hip_cmd_L.setData(xw, _a(hip_cmd_L_buf))
        ln_hip_cmd_R.setData(xw, _a(hip_cmd_R_buf))
        ln_roll.setData(xw, _a(roll_buf))
        ln_tau_L.setData(xw, _a(tau_whl_L_buf))
        ln_tau_R.setData(xw, _a(tau_whl_R_buf))
        ln_vbatt.setData(xw, _a(v_batt_buf))
        ln_ibat_batt.setData(xw, _a(I_total_buf))
        ln_iwhl_L.setData(xw, _a(I_whl_L_buf))
        ln_iwhl_R.setData(xw, _a(I_whl_R_buf))
        ln_ihip_L.setData(xw, _a(I_hip_L_buf))
        ln_ihip_R.setData(xw, _a(I_hip_R_buf))
        ln_itotal.setData(xw, _a(I_total_buf))
        ln_htau_L.setData(xw, _a(tau_hip_L_buf))
        ln_htau_R.setData(xw, _a(tau_hip_R_buf))

        # Keep second viewbox geometry in sync
        _sync_batt()

    timer = QtCore.QTimer()
    timer.setInterval(int(1000 / TELEMETRY_HZ))
    timer.timeout.connect(_update)
    timer.start()

    app.exec()


# ---------------------------------------------------------------------------
# Main sandbox viewer
# ---------------------------------------------------------------------------
def sandbox(slowmo: float = 1.0):
    JOY_DEADZONE = 0.08
    _joy         = None
    _joy_axes    = {}
    if _PYGAME_OK:
        pygame.display.init()
        pygame.display.set_mode((1, 1), pygame.NOFRAME)
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            _joy = pygame.joystick.Joystick(0)
            _joy.init()
            print(f"Joystick: {_joy.get_name()}")
            print(f"  Axes: {_joy.get_numaxes()}  Buttons: {_joy.get_numbuttons()}")
        else:
            print("Joystick: none detected — sliders only")

    print("Building sandbox arena...")
    xml    = build_xml(sandbox_obstacles=SANDBOX_OBSTACLES, prop_bodies=SANDBOX_PROPS,
                       floor_size=(25, 25))
    assets = build_assets()
    model  = mujoco.MjModel.from_xml_string(xml, assets)
    data   = mujoco.MjData(model)

    _scenarios.LQR_K_TABLE = compute_gain_table(
        ROBOT, Q_diag=[LQR_Q_PITCH, LQR_Q_PITCH_RATE, LQR_Q_VEL], R_val=LQR_R)
    _scenarios.USE_PD_CONTROLLER = False
    _scenarios.USE_YAW_PI        = True
    _scenarios.VELOCITY_PI_KP    = VELOCITY_PI_KP
    _scenarios.VELOCITY_PI_KI    = VELOCITY_PI_KI
    _scenarios.YAW_PI_KP_GAIN    = YAW_PI_KP
    _scenarios.YAW_PI_KI_GAIN    = YAW_PI_KI

    print(f"  LQR: Q=[{LQR_Q_PITCH:.4g}, {LQR_Q_PITCH_RATE:.4g}, {LQR_Q_VEL:.4g}]  R={LQR_R:.4g}")
    print(f"  VelocityPI: Kp={VELOCITY_PI_KP:.4g}  Ki={VELOCITY_PI_KI:.4g}")
    print(f"  YawPI:      Kp={YAW_PI_KP:.4g}  Ki={YAW_PI_KI:.4g}")
    print(f"  Obstacles: {len(SANDBOX_OBSTACLES)}")
    print()

    free_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")
    qpos_free = model.jnt_qposadr[free_id]
    dof_free  = model.jnt_dofadr[free_id]
    d_yaw     = dof_free + 5

    def _dof(name):  return model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _qpos(name): return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]

    dof_hip_L  = _dof("hip_L");         dof_hip_R  = _dof("hip_R")
    qpos_hip_L = _qpos("hip_L");        qpos_hip_R = _qpos("hip_R")
    dof_whl_L  = _dof("wheel_spin_L");  dof_whl_R  = _dof("wheel_spin_R")

    act_hip_L   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_act_L")
    act_hip_R   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_act_R")
    act_wheel_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_act_L")
    act_wheel_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_act_R")

    box_bid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    d_pitch   = dof_free + 4

    PHYSICS_HZ      = round(1.0 / model.opt.timestep)
    steps_per_frame = max(1, PHYSICS_HZ // RENDER_HZ)
    _dt_ctrl        = model.opt.timestep * CTRL_STEPS

    _n_sens = max(1, round(SENSOR_DELAY_S   / _dt_ctrl)) if USE_LATENCY_MODEL else 1
    _n_act  = max(1, round(ACTUATOR_DELAY_S / _dt_ctrl)) if USE_LATENCY_MODEL else 1

    from physics import get_equilibrium_pitch as _gep
    _pitch0_lat = _gep(ROBOT, Q_NOM)
    _sens_buf = deque([(_pitch0_lat, 0.0, 0.0)] * max(1, _n_sens), maxlen=max(1, _n_sens))
    _ctrl_buf = deque([(0.0, 0.0)] * max(1, _n_act),  maxlen=max(1, _n_act))

    vel_pi  = VelocityPI(kp=VELOCITY_PI_KP, ki=VELOCITY_PI_KI, dt=_dt_ctrl)
    yaw_pi  = YawPI(kp=YAW_PI_KP, ki=YAW_PI_KI, dt=_dt_ctrl)
    battery = BatteryModel()
    battery.reset()
    _v_batt = [BATT_V_NOM]

    _v_desired     = [0.0]
    _omega_desired = [0.0]
    _hip_pct       = [_HIP_NOM_PCT]

    # Controller enable flags (toggled via UI checkboxes)
    _en_lqr        = [True]
    _en_vel_pi     = [True]
    _en_yaw_pi     = [True]
    _en_suspension = [True]
    _en_roll_lev   = [True]
    # Hip controller mode: "impedance" (soft, Phase 4+) or "position_pd" (stiff, S1-S7)
    _hip_mode      = ["impedance"]

    def _hip_target():
        pct = max(0.0, min(100.0, _hip_pct[0])) / 100.0
        return Q_RET + pct * (_HIP_MAX_Q - Q_RET)

    _pitch_d        = _gep(ROBOT, Q_NOM)
    _pitch_rate_d   = 0.0
    _wheel_vel_d    = 0.0
    _theta_ref      = 0.0
    _prev_theta_ref = [0.0]
    _vel_est_ms     = 0.0

    _hip_L_deg      = math.degrees(Q_NOM)
    _hip_R_deg      = math.degrees(Q_NOM)
    _hip_cmd_L_deg  = math.degrees(Q_NOM)   # per-leg cmd (includes roll correction)
    _hip_cmd_R_deg  = math.degrees(Q_NOM)
    _roll_deg       = 0.0
    _pitch_rate_deg = 0.0
    _tau_whl_L      = 0.0
    _tau_whl_R      = 0.0
    _tau_hip_L      = 0.0
    _tau_hip_R      = 0.0
    _batt_v         = BATT_V_NOM
    _batt_soc       = 100.0
    _batt_temp      = 25.0
    _I_whl_L        = 0.0
    _I_whl_R        = 0.0
    _I_hip_L        = 0.0
    _I_hip_R        = 0.0
    _I_total        = BATT_I_QUIESCENT

    rng  = np.random.default_rng(0)
    step = 0
    prev_sim_t = 0.0
    last_push  = -1.0

    def _init():
        init_sim(model, data)

    _init()

    data_q = mp.Queue(maxsize=12000)
    cmd_q  = mp.Queue(maxsize=32)
    plot_proc = mp.Process(
        target=_plot_process,
        args=(data_q, cmd_q, WINDOW_S, "Sandbox — free drive",
              WHEEL_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT),
        daemon=True)
    plot_proc.start()

    threading.Thread(target=_position_mujoco_right, args=(1.5,), daemon=True).start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 45
        viewer.cam.elevation = -18
        viewer.cam.distance  = 3.0
        viewer.cam.lookat    = np.array([0.0, 0.0, 0.30])

        # Wait for plot window and MuJoCo viewer to load and position
        time.sleep(2.5)

        def _reset_state():
            nonlocal step, prev_sim_t, _pitch_d, _pitch_rate_d
            nonlocal _wheel_vel_d, _theta_ref, _vel_est_ms
            nonlocal _hip_L_deg, _hip_R_deg, _hip_cmd_L_deg, _hip_cmd_R_deg
            nonlocal _roll_deg, _pitch_rate_deg, _tau_whl_L, _tau_whl_R
            nonlocal _tau_hip_L, _tau_hip_R
            nonlocal _batt_v, _batt_soc, _batt_temp
            nonlocal _I_whl_L, _I_whl_R, _I_hip_L, _I_hip_R, _I_total
            _init()
            step = 0; prev_sim_t = 0.0
            _pitch_d = _gep(ROBOT, Q_NOM); _pitch_rate_d = 0.0; _wheel_vel_d = 0.0
            _theta_ref = 0.0; _vel_est_ms = 0.0; _prev_theta_ref[0] = 0.0
            _v_desired[0] = 0.0; _omega_desired[0] = 0.0
            _hip_pct[0] = _HIP_NOM_PCT
            _hip_L_deg = _hip_R_deg = math.degrees(Q_NOM)
            _hip_cmd_L_deg = _hip_cmd_R_deg = math.degrees(Q_NOM)
            _roll_deg = _pitch_rate_deg = 0.0
            _tau_whl_L = _tau_whl_R = _tau_hip_L = _tau_hip_R = 0.0
            battery.reset(); _v_batt[0] = BATT_V_NOM
            _batt_v = BATT_V_NOM; _batt_soc = 100.0; _batt_temp = 25.0
            _I_whl_L = _I_whl_R = _I_hip_L = _I_hip_R = 0.0
            _I_total = BATT_I_QUIESCENT
            vel_pi.reset(); yaw_pi.reset()
            _p0 = _gep(ROBOT, Q_NOM)
            _sens_buf.clear(); _sens_buf.extend([(_p0, 0.0, 0.0)] * max(1, _n_sens))
            _ctrl_buf.clear(); _ctrl_buf.extend([(0.0, 0.0)] * max(1, _n_act))
            if not data_q.full(): data_q.put_nowait("RESET")

        while viewer.is_running():
            frame_start = time.perf_counter()
            sim_t = float(data.time)

            try:
                cmd = cmd_q.get_nowait()
                if cmd == "RESTART":
                    _reset_state()
                elif isinstance(cmd, tuple):
                    if   cmd[0] == "V_DESIRED":     _v_desired[0]     = float(cmd[1])
                    elif cmd[0] == "OMEGA_DESIRED":  _omega_desired[0] = float(cmd[1])
                    elif cmd[0] == "HIP_PCT":        _hip_pct[0]       = float(cmd[1])
                    elif cmd[0] == "CTRL_EN":
                        key, val = cmd[1], cmd[2]
                        if   key == "LQR":        _en_lqr[0]        = val
                        elif key == "VelPI":      _en_vel_pi[0]     = val
                        elif key == "YawPI":      _en_yaw_pi[0]     = val
                        elif key == "Suspension": _en_suspension[0] = val
                        elif key == "RollLev":    _en_roll_lev[0]   = val
                    elif cmd[0] == "HIP_MODE":
                        _hip_mode[0] = cmd[1]
            except Exception:
                pass

            if sim_t < prev_sim_t - 0.01:
                _reset_state()
            prev_sim_t = sim_t

            if _joy is not None:
                for _ev in pygame.event.get():
                    if _ev.type == pygame.JOYAXISMOTION:
                        _joy_axes[_ev.axis] = _ev.value
                    elif _ev.type == pygame.JOYBUTTONDOWN and _ev.button == 0:
                        _reset_state()

                raw_v = _joy_axes.get(1, 0.0)
                raw_w = _joy_axes.get(2, 0.0)
                raw_h = _joy_axes.get(3, 0.0)

                v_joy = -raw_v if abs(raw_v) > JOY_DEADZONE else 0.0
                w_joy = -raw_w if abs(raw_w) > JOY_DEADZONE else 0.0
                h_joy = (1.0 - raw_h) / 2.0 * 100.0 if abs(raw_h) > JOY_DEADZONE else None

                _v_desired[0]     = v_joy * 3.0
                _omega_desired[0] = w_joy * 5.0
                if h_joy is not None:
                    _hip_pct[0] = h_joy

            for _ in range(steps_per_frame):
                if step % CTRL_STEPS == 0:
                    pitch_true, pitch_rate_true = get_pitch_and_rate(data, box_bid, d_pitch)
                    pitch      = pitch_true      + rng.normal(0, PITCH_NOISE_STD_RAD)
                    pitch_rate = pitch_rate_true + rng.normal(0, PITCH_RATE_NOISE_STD_RAD_S)
                    wheel_vel  = (data.qvel[dof_whl_L] + data.qvel[dof_whl_R]) / 2.0
                    q_hip_avg  = (data.qpos[qpos_hip_L] + data.qpos[qpos_hip_R]) / 2.0

                    _sens_buf.append((pitch, pitch_rate, wheel_vel))
                    _pitch_d, _pitch_rate_d, _wheel_vel_d = _sens_buf[0]
                    _vel_est_ms = _wheel_vel_d * WHEEL_R

                    v_ref_rads = _v_desired[0] / WHEEL_R
                    if _en_vel_pi[0]:
                        theta_ref = vel_pi.update(_v_desired[0], _vel_est_ms)
                        d_max     = THETA_REF_RATE_LIMIT * _dt_ctrl
                        theta_ref = float(np.clip(theta_ref,
                                                   _prev_theta_ref[0] - d_max,
                                                   _prev_theta_ref[0] + d_max))
                    else:
                        vel_pi.reset()
                        theta_ref = 0.0
                    _prev_theta_ref[0] = theta_ref
                    _theta_ref         = theta_ref

                    if _en_lqr[0]:
                        tau_sym = lqr_torque(_pitch_d, _pitch_rate_d, _wheel_vel_d,
                                             q_hip_avg, v_ref=v_ref_rads, theta_ref=theta_ref)
                    else:
                        tau_sym = 0.0

                    yaw_rate = data.qvel[d_yaw]
                    if _en_yaw_pi[0]:
                        tau_yaw = yaw_pi.update(_omega_desired[0], yaw_rate)
                    else:
                        yaw_pi.reset()
                        tau_yaw = 0.0

                    _ctrl_buf.append((tau_sym - tau_yaw, tau_sym + tau_yaw))
                    _tau_L_d, _tau_R_d = _ctrl_buf[0]
                    data.ctrl[act_wheel_L] = motor_taper(_tau_L_d, data.qvel[dof_whl_L], _v_batt[0])
                    data.ctrl[act_wheel_R] = motor_taper(_tau_R_d, data.qvel[dof_whl_R], _v_batt[0])

                    q_sym  = _hip_target()
                    q_roll = data.xquat[box_bid]
                    roll_true = math.atan2(
                        2.0 * (q_roll[0]*q_roll[1] + q_roll[2]*q_roll[3]),
                        1.0 - 2.0 * (q_roll[1]**2  + q_roll[2]**2))
                    roll_rate = data.qvel[dof_free + 3]
                    roll_meas = roll_true + rng.normal(0, ROLL_NOISE_STD_RAD)
                    if _en_roll_lev[0]:
                        delta_q = LEG_K_ROLL * roll_meas + LEG_D_ROLL * roll_rate
                    else:
                        delta_q = 0.0
                    q_nom_L   = float(np.clip(q_sym + delta_q, HIP_SAFE_MIN, HIP_SAFE_MAX))
                    q_nom_R   = float(np.clip(q_sym - delta_q, HIP_SAFE_MIN, HIP_SAFE_MAX))

                    for qpos_hip, dof_hip, act_hip, q_nom_leg in [
                        (qpos_hip_L, dof_hip_L, act_hip_L, q_nom_L),
                        (qpos_hip_R, dof_hip_R, act_hip_R, q_nom_R),
                    ]:
                        if not _en_suspension[0]:
                            data.ctrl[act_hip] = 0.0
                            continue
                        q_hip  = data.qpos[qpos_hip]
                        dq_hip = data.qvel[dof_hip]
                        if _hip_mode[0] == "position_pd":
                            # Matches S1–S7 optimizer: stiff position servo
                            tau_h = (HIP_POSITION_KP * (Q_NOM - q_hip)
                                     - HIP_POSITION_KD * dq_hip)
                        else:
                            # Default: soft impedance (Phase 4+)
                            tau_h = -(LEG_K_S * (q_hip - q_nom_leg) + LEG_B_S * dq_hip)
                        data.ctrl[act_hip] = np.clip(tau_h,
                                                     -HIP_IMPEDANCE_TORQUE_LIMIT,
                                                      HIP_IMPEDANCE_TORQUE_LIMIT)

                    _v_batt[0] = battery.step(_dt_ctrl, _motor_currents(
                        float(data.ctrl[act_wheel_L]), float(data.ctrl[act_wheel_R]),
                        float(data.ctrl[act_hip_L]),   float(data.ctrl[act_hip_R])))

                    _hip_L_deg      = math.degrees(data.qpos[qpos_hip_L])
                    _hip_R_deg      = math.degrees(data.qpos[qpos_hip_R])
                    _hip_cmd_L_deg  = math.degrees(q_nom_L)   # true per-leg cmd
                    _hip_cmd_R_deg  = math.degrees(q_nom_R)
                    _roll_deg       = math.degrees(roll_true)
                    _pitch_rate_deg = math.degrees(pitch_rate_true)
                    _tau_whl_L      = float(data.ctrl[act_wheel_L])
                    _tau_whl_R      = float(data.ctrl[act_wheel_R])
                    _tau_hip_L      = float(data.ctrl[act_hip_L])
                    _tau_hip_R      = float(data.ctrl[act_hip_R])
                    _batt_v         = battery.v_terminal
                    _batt_soc       = battery.soc_pct
                    _batt_temp      = battery.temperature_c
                    _I_whl_L        = abs(_tau_whl_L) / WHEEL_KT
                    _I_whl_R        = abs(_tau_whl_R) / WHEEL_KT
                    _I_hip_L        = abs(_tau_hip_L) / HIP_KT_OUTPUT
                    _I_hip_R        = abs(_tau_hip_R) / HIP_KT_OUTPUT
                    _I_total        = battery.i_total

                mujoco.mj_step(model, data)
                step += 1

            viewer.sync()

            robot_pos = data.xpos[box_bid]
            viewer.cam.lookat[0] = robot_pos[0]
            viewer.cam.lookat[1] = robot_pos[1]

            # Push 25-value telemetry tuple
            wall_now = time.perf_counter()
            if wall_now - last_push >= 1.0 / TELEMETRY_HZ and not data_q.full():
                pitch_true, _ = get_pitch_and_rate(data, box_bid, d_pitch)
                q_hip_avg = (data.qpos[qpos_hip_L] + data.qpos[qpos_hip_R]) / 2.0
                pitch_ref_display = _gep(ROBOT, q_hip_avg) + _theta_ref
                data_q.put_nowait((
                    sim_t,
                    math.degrees(pitch_true),
                    math.degrees(pitch_ref_display),
                    _pitch_rate_deg,
                    _vel_est_ms,
                    _v_desired[0],
                    math.degrees(data.qvel[d_yaw]),
                    math.degrees(_omega_desired[0]),
                    _hip_L_deg,
                    _hip_R_deg,
                    _hip_cmd_L_deg,
                    _hip_cmd_R_deg,
                    _roll_deg,
                    _tau_whl_L,
                    _tau_whl_R,
                    _batt_v,
                    _batt_temp,
                    _batt_soc,
                    _I_whl_L,
                    _I_whl_R,
                    _I_hip_L,
                    _I_hip_R,
                    _I_total,
                    _tau_hip_L,
                    _tau_hip_R,
                ))
                last_push = wall_now

            # HUD — world-frame axes at origin
            viewer.user_scn.ngeom = 0
            _r  = 0.007; _hl = 0.10
            _o  = np.array([0.0, 0.0, 0.01])
            _ax = [
                (np.array([_hl, 0,  0  ]),
                 np.array([[0,0,1],[0,1,0],[-1,0,0]], dtype=np.float32),
                 np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32)),
                (np.array([0,  _hl, 0  ]),
                 np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float32),
                 np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32)),
                (np.array([0,  0,  _hl ]),
                 np.eye(3, dtype=np.float32),
                 np.array([0.1, 0.4, 1.0, 1.0], dtype=np.float32)),
            ]
            for i, (off, mat, rgba) in enumerate(_ax):
                g = viewer.user_scn.geoms[i]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER,
                                    [_r, _hl, 0], (_o + off).tolist(),
                                    mat.flatten().tolist(), rgba)
                g.label = (b"+X" if i==0 else b"+Y" if i==1 else b"+Z")
            viewer.user_scn.ngeom = 3

            elapsed = time.perf_counter() - frame_start
            sleep_t = slowmo / RENDER_HZ - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    data_q.put(None)
    plot_proc.join(timeout=2)
    if plot_proc.is_alive():
        plot_proc.terminate()
    if _PYGAME_OK:
        pygame.quit()
    os._exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Sandbox arena — pyqtgraph telemetry, 5×2 layout, 15s window.",
        epilog="python sandbox_fastchart.py\npython sandbox_fastchart.py --slowmo 2")
    ap.add_argument("--slowmo", type=float, default=1.0,
                    help="Slow-motion factor (e.g. 2 = half real-time)")
    args = ap.parse_args()
    sandbox(slowmo=args.slowmo)


if __name__ == "__main__":
    mp.freeze_support()
    main()
