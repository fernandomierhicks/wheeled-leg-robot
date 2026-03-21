"""live_hip.py — Real-time hip angle chart alongside MuJoCo viewer.

Runs the same hip sweep as verify_phase0.py (Q_NOM→Q_RET→Q_EXT→Q_NOM, 6s loop)
while plotting target vs actual hip angle live in a pyqtgraph window.

Run:  python -m master_sim.viz.live_hip
"""
import sys
import time
import math
import threading

import numpy as np

# IMPORTANT: mujoco must be imported BEFORE pyqtgraph on Windows to avoid
# OpenGL DLL init conflict (WinError 1114).
import mujoco
import mujoco.viewer
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from master_sim.defaults import DEFAULT_PARAMS
from master_sim.physics import build_xml, build_assets, init_sim

# Import style constants directly to avoid re-importing pyqtgraph via visualizer
BG_COLOR    = "#12121e"
BAR_COLOR   = "#1a1a2e"
PALETTE     = ["#60d0ff", "#ff6060", "#80ff80", "#ffa040", "#ffe060",
               "#e080ff", "#ff80b0", "#40ffd0", "#a0a0ff", "#e0e0e0"]
LINE_WIDTH  = 1.8
GRID_ALPHA  = 0.20
TICK_COLOR  = "#d8d8d8"


# ── Chart config ─────────────────────────────────────────────────────────────
HISTORY_SEC = 12.0        # rolling window
MAX_POINTS  = 3000        # ring buffer size
UPDATE_HZ   = 30          # chart refresh rate


def main():
    robot = DEFAULT_PARAMS.robot

    # ── Build MuJoCo model (welded in air, same as verify_phase0) ────────
    xml = build_xml(robot, weld_body=True)
    assets = build_assets()
    model = mujoco.MjModel.from_xml_string(xml, assets)
    data = mujoco.MjData(model)
    init_sim(model, data, robot, q_hip_init=robot.Q_NOM)

    # Joint / actuator indices
    jnt_hip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_L")
    act_hip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_act_L")
    act_hip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_act_R")
    jnt_hip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_R")
    qa_L = model.jnt_qposadr[jnt_hip_L]
    qa_R = model.jnt_qposadr[jnt_hip_R]
    da_L = model.jnt_dofadr[jnt_hip_L]
    da_R = model.jnt_dofadr[jnt_hip_R]

    KP, KD = 50.0, 3.0
    PERIOD = 6.0

    # ── Shared ring buffers (sim thread → Qt thread) ─────────────────────
    buf_t      = np.zeros(MAX_POINTS)
    buf_target = np.zeros(MAX_POINTS)
    buf_actual = np.zeros(MAX_POINTS)
    buf_idx    = [0]       # mutable counter
    sim_running = [True]

    # ── Sim thread (MuJoCo viewer + physics) ─────────────────────────────
    def sim_loop():
        with mujoco.viewer.launch_passive(model, data) as viewer:
            t_start = time.time()
            while viewer.is_running() and sim_running[0]:
                t_wall = time.time() - t_start
                t_sweep = t_wall % PERIOD

                # Target (same ramp as verify_phase0)
                if t_sweep < 2.0:
                    frac = t_sweep / 2.0
                    q_target = robot.Q_NOM + frac * (robot.Q_RET - robot.Q_NOM)
                elif t_sweep < 4.0:
                    frac = (t_sweep - 2.0) / 2.0
                    q_target = robot.Q_RET + frac * (robot.Q_EXT - robot.Q_RET)
                else:
                    frac = (t_sweep - 4.0) / 2.0
                    q_target = robot.Q_EXT + frac * (robot.Q_NOM - robot.Q_EXT)

                # PD torque
                for qa, da, act in [(qa_L, da_L, act_hip_L),
                                    (qa_R, da_R, act_hip_R)]:
                    err = q_target - data.qpos[qa]
                    dq  = data.qvel[da]
                    data.ctrl[act] = np.clip(KP * err - KD * dq, -7.0, 7.0)

                mujoco.mj_step(model, data)
                viewer.sync()

                # Record into ring buffer (left hip)
                i = buf_idx[0] % MAX_POINTS
                buf_t[i]      = t_wall
                buf_target[i] = math.degrees(q_target)
                buf_actual[i] = math.degrees(data.qpos[qa_L])
                buf_idx[0] += 1

                time.sleep(0.002)

        sim_running[0] = False

    sim_thread = threading.Thread(target=sim_loop, daemon=True)
    sim_thread.start()

    # ── PyQtGraph live window ────────────────────────────────────────────
    pg.setConfigOptions(antialias=False, useOpenGL=True, enableExperimental=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Live Hip Angle — master_sim")
    win.setStyleSheet(f"background:{BG_COLOR};")
    central = QtWidgets.QWidget()
    win.setCentralWidget(central)
    vbox = QtWidgets.QVBoxLayout(central)
    vbox.setContentsMargins(4, 4, 4, 2)
    vbox.setSpacing(2)

    glw = pg.GraphicsLayoutWidget()
    glw.setBackground(BG_COLOR)
    vbox.addWidget(glw, stretch=1)

    # Status bar
    status = QtWidgets.QWidget()
    status.setStyleSheet(f"background:{BAR_COLOR}; border-radius:4px;")
    hbox = QtWidgets.QHBoxLayout(status)
    hbox.setContentsMargins(6, 3, 6, 3)
    _SL = ("color:#e8e8e8; font-family:Consolas,monospace; "
           "font-size:11px; font-weight:bold; padding:0 14px 0 0;")
    lbl = QtWidgets.QLabel("Hip Angle (left leg)")
    lbl.setStyleSheet(_SL)
    hbox.addWidget(lbl)
    hbox.addStretch()
    val_lbl = QtWidgets.QLabel("")
    val_lbl.setStyleSheet(
        "color:#80ffb0; font-family:Consolas,monospace; font-size:11px;")
    hbox.addWidget(val_lbl)
    vbox.addWidget(status)

    # Chart panel (inline — avoids importing visualizer which triggers pyqtgraph)
    from pyqtgraph.Qt import QtGui
    pl = glw.addPlot(row=0, col=0)
    pl.setTitle('<span style="color:#e0e0e0;font-size:9pt;font-weight:600">'
                'Hip Angle (left leg)</span>')
    pl.setLabel("left", '<span style="color:#c8c8c8;font-size:9pt">deg</span>')
    pl.setLabel("bottom", '<span style="color:#c8c8c8;font-size:9pt">time (s)</span>')
    pl.showGrid(x=True, y=True, alpha=GRID_ALPHA)
    tick_font = QtGui.QFont("Consolas", 9)
    for ax_name in ("left", "bottom"):
        ax = pl.getAxis(ax_name)
        ax.setTextPen(pg.mkColor(TICK_COLOR))
        ax.setPen(pg.mkPen("#555"))
        ax.setStyle(tickFont=tick_font)
    leg = pl.addLegend(offset=(6, 6), verSpacing=-4)
    leg.setBrush(pg.mkBrush(18, 18, 36, 210))
    leg.setPen(pg.mkPen("#444"))
    leg.setLabelTextColor(pg.mkColor("#cccccc"))

    # Limit lines at Q_RET and Q_EXT
    q_ret_deg = math.degrees(robot.Q_RET)
    q_ext_deg = math.degrees(robot.Q_EXT)
    q_nom_deg = math.degrees(robot.Q_NOM)
    pl.addItem(pg.InfiniteLine(pos=q_ret_deg, angle=0,
               pen=pg.mkPen("#ff8844", width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
               label=f"Q_RET {q_ret_deg:.1f}°",
               labelOpts={"color": "#ff8844"}))
    pl.addItem(pg.InfiniteLine(pos=q_ext_deg, angle=0,
               pen=pg.mkPen("#ff8844", width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
               label=f"Q_EXT {q_ext_deg:.1f}°",
               labelOpts={"color": "#ff8844"}))
    pl.addItem(pg.InfiniteLine(pos=q_nom_deg, angle=0,
               pen=pg.mkPen("#88ff88", width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
               label=f"Q_NOM {q_nom_deg:.1f}°",
               labelOpts={"color": "#88ff88"}))

    curve_target = pl.plot([], [], pen=pg.mkPen(PALETTE[0], width=LINE_WIDTH),
                           name="target")
    curve_actual = pl.plot([], [], pen=pg.mkPen(PALETTE[1], width=LINE_WIDTH),
                           name="actual")

    # ── Timer-driven update ──────────────────────────────────────────────
    def update():
        n = buf_idx[0]
        if n == 0:
            return

        if n <= MAX_POINTS:
            t = buf_t[:n]
            tgt = buf_target[:n]
            act = buf_actual[:n]
        else:
            # Unwrap ring buffer
            start = n % MAX_POINTS
            t   = np.concatenate([buf_t[start:],   buf_t[:start]])
            tgt = np.concatenate([buf_target[start:], buf_target[:start]])
            act = np.concatenate([buf_actual[start:], buf_actual[:start]])

        # Rolling window
        t_max = t[-1]
        t_min = max(0, t_max - HISTORY_SEC)
        mask = t >= t_min
        t_w   = t[mask]
        tgt_w = tgt[mask]
        act_w = act[mask]

        curve_target.setData(t_w, tgt_w)
        curve_actual.setData(t_w, act_w)
        pl.setXRange(t_min, t_max, padding=0.02)

        val_lbl.setText(f"target={tgt_w[-1]:.1f}°  actual={act_w[-1]:.1f}°  "
                        f"err={tgt_w[-1]-act_w[-1]:.2f}°")

        if not sim_running[0]:
            timer.stop()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(int(1000 / UPDATE_HZ))

    win.resize(900, 400)
    win.show()
    app.exec()

    sim_running[0] = False
    sim_thread.join(timeout=2)
    print("Done.")


if __name__ == "__main__":
    main()
