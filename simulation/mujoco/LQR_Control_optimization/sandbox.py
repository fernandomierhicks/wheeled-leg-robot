"""sandbox.py — Interactive free-drive sandbox arena.

Scattered obstacles 1–8 cm tall (boxes and cylinders).
Drive the robot freely:
  - v_desired slider:  forward / backward speed  [-1.2, 1.2] m/s
  - ω_desired slider:  yaw rate (left/right)      [-2.0, 2.0] rad/s
  - Hip height slider: leg height 0 % (up) → 100 % (down)

Gains: baselined LQR + VelocityPI + YawPI from sim_config.py.

Usage:
    python sandbox.py
    python sandbox.py --slowmo 2
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
    HIP_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT,
    LEG_K_S, LEG_B_S, LEG_K_ROLL, LEG_D_ROLL,
    ROLL_NOISE_STD_RAD, HIP_SAFE_MIN, HIP_SAFE_MAX,
    LQR_Q_PITCH, LQR_Q_PITCH_RATE, LQR_Q_VEL, LQR_R,
    PITCH_NOISE_STD_RAD, PITCH_RATE_NOISE_STD_RAD_S,
    CTRL_STEPS, THETA_REF_RATE_LIMIT,
    VELOCITY_PI_KP, VELOCITY_PI_KI,
    YAW_PI_KP, YAW_PI_KI,
)
from physics import build_xml, build_assets, get_equilibrium_pitch
from scenarios import init_sim, get_pitch_and_rate, lqr_torque, VelocityPI, YawPI
from lqr_design import compute_gain_table
import scenarios as _scenarios

RENDER_HZ    = 60
TELEMETRY_HZ = 100
WINDOW_S     = 8.0

# Sandbox hip range: cap max extension 10° closer to neutral than Q_EXT
_HIP_MAX_Q   = Q_EXT + math.radians(10)
# Nominal hip position expressed as 0-100 % of sandbox stroke
_HIP_NOM_PCT = (Q_NOM - Q_RET) / (_HIP_MAX_Q - Q_RET) * 100.0


# ---------------------------------------------------------------------------
# Sandbox obstacle layout
# Colors by height: ≤2 cm = yellow-green, ≤4 cm = orange,
#                   ≤6 cm = red-orange,   >6 cm = red
# shape="box": rx, ry = half-sizes in X/Y   shape="cyl": r = radius
# ---------------------------------------------------------------------------
# ── Sandbox terrain obstacles ────────────────────────────────────────────────
# Mix of gentle shapes (sphere mounds, capsule bumps, ramps) and a few sharp steps.
# sphere: large sphere mostly below ground — only top h metres emerge as smooth mound
# capsule: cylinder lying on side — rounded cross-section, bump height = radius r
# ramp: tilted box, angle_deg from horizontal, length=slope length, h=peak height
# box/cyl: traditional abrupt steps (used sparingly)
SANDBOX_OBSTACLES = [
    # ── Gentle sphere mounds — large smooth hills ────────────────────────────
    dict(shape="sphere", x= 3.0, y= 0.0, r=1.20, h=0.06),  # broad 6 cm mound ahead
    dict(shape="sphere", x=-2.5, y= 1.5, r=0.80, h=0.05),  # 5 cm mound left-rear
    dict(shape="sphere", x= 5.5, y=-2.0, r=1.00, h=0.07),  # 7 cm far mound
    dict(shape="sphere", x=-5.0, y=-1.0, r=0.90, h=0.06),  # 6 cm far-left mound

    # ── Capsule rounded bumps — gentle speed bumps ───────────────────────────
    dict(shape="capsule", x= 1.5, y= 0.0, r=0.020, length=1.20),  # 2 cm near bump
    dict(shape="capsule", x=-1.2, y= 0.5, r=0.025, length=0.80),  # 2.5 cm behind
    dict(shape="capsule", x= 4.0, y= 1.0, r=0.030, length=1.00),  # 3 cm far bump
    dict(shape="capsule", x= 2.5, y=-1.5, r=0.020, length=0.70),  # 2 cm off-axis
    dict(shape="capsule", x=-3.5, y=-2.0, r=0.035, length=1.20),  # 3.5 cm far-rear

    # ── Ramps — gradual rise then abrupt back edge ───────────────────────────
    dict(shape="ramp", x= 2.0, y=-0.5, angle_deg= 8, length=0.60, width=0.50, h=0.050),
    dict(shape="ramp", x=-2.0, y= 2.0, angle_deg= 6, length=0.80, width=0.60, h=0.040),
    dict(shape="ramp", x= 6.0, y= 0.5, angle_deg=10, length=0.50, width=0.60, h=0.060),
    dict(shape="ramp", x=-4.0, y=-3.0, angle_deg= 7, length=0.70, width=0.50, h=0.045),

    # ── A few abrupt steps for comparison (keep these sparse) ───────────────
    dict(shape="box", x= 1.0, y= 2.0, rx=0.12, ry=0.20, h=0.02),  # 2 cm sharp step
    dict(shape="box", x=-1.5, y=-2.5, rx=0.15, ry=0.15, h=0.03),  # 3 cm step
    dict(shape="cyl", x= 4.5, y= 3.0, r=0.10,           h=0.05),  # 5 cm puck
    dict(shape="box", x= 7.0, y=-1.5, rx=0.20, ry=0.25, h=0.07),  # 7 cm far step
    dict(shape="box", x=-6.0, y= 2.0, rx=0.20, ry=0.20, h=0.06),  # 6 cm far step
]

# ── Real-world props for scale reference — free bodies, can be knocked around ──
# Soda can ≈ 66 mm dia × 122 mm tall (385 g full)
# Water bottle ≈ 80 mm dia × 250 mm tall (520 g full)
# Tennis ball ≈ 66 mm dia (57 g)
# Cardboard box ≈ 300 × 200 × 200 mm (300 g)
SANDBOX_PROPS = [
    dict(type="can",           x= 0.8, y= 0.3),   # soda can — right in front
    dict(type="can",           x= 0.8, y=-0.3),   # soda can — pair it
    dict(type="bottle",        x= 1.5, y=-0.8),   # water bottle off to right
    dict(type="ball",          x= 0.6, y= 0.8),   # tennis ball — rolls on contact
    dict(type="ball",          x= 0.5, y=-0.6),   # second tennis ball
    dict(type="cardboard_box", x= 2.0, y= 0.5),   # box — taller obstacle
]


# ---------------------------------------------------------------------------
# Window positioning (Windows)
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
# Matplotlib telemetry panel (separate process)
# Telemetry tuple: (t, pitch_deg, theta_ref_deg, yaw_rate_rads,
#                   pitch_rate_deg_s, vel_ms, v_cmd, omega_cmd)
# ---------------------------------------------------------------------------
def _plot_process(data_q: mp.Queue, cmd_q: mp.Queue,
                  window_s: float, title: str) -> None:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    MAXLEN = int(window_s * TELEMETRY_HZ) + 500
    (t_buf, pitch_buf, theta_ref_buf, yaw_rate_buf,
     pitch_rate_buf, vel_buf, v_cmd_buf, omega_cmd_buf) = (deque(maxlen=MAXLEN) for _ in range(8))

    plt.ion()
    fig, axes = plt.subplots(4, 1, figsize=(6, 9))
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(hspace=0.60, top=0.91, bottom=0.35, left=0.18, right=0.96)

    specs = [
        ("Pitch",         "deg",     "#60d0ff"),
        ("Yaw Rate",      "rad/s",   "#f08040"),
        ("Pitch Rate",    "deg/s",   "#b060ff"),
        ("Robot Velocity","m/s",     "#50e080"),
    ]
    lines = []
    for ax, (ttl, unit, col) in zip(axes, specs):
        ax.set_facecolor("#1e1e2e")
        for sp in ax.spines.values(): sp.set_edgecolor("#555")
        ax.tick_params(colors="lightgray", labelsize=7)
        ax.grid(True, color="#333", linewidth=0.4, linestyle="--")
        ax.set_title(ttl, color="white", fontsize=9, pad=3)
        ax.set_ylabel(unit, color=col, fontsize=8)
        ax.set_xlabel("sim time [s]", color="lightgray", fontsize=7)
        ln, = ax.plot([], [], color=col, linewidth=1.5)
        lines.append(ln)

    axes[2].axhline(0.0, color="#666688", linewidth=0.8, linestyle="--")
    axes[1].axhline(0.0, color="#666688", linewidth=0.8, linestyle="--")

    ln_theta_ref, = axes[0].plot([], [], color="#ff6060", linewidth=1.0, linestyle="--")
    axes[0].legend(handles=[lines[0], ln_theta_ref],
                   labels=["pitch", "pitch cmd"],
                   loc="upper right", fontsize=6,
                   facecolor="#2a2a3e", edgecolor="#555", labelcolor="lightgray")

    ln_omega_cmd, = axes[1].plot([], [], color="#ffb060", linewidth=1.0, linestyle="--")
    axes[1].legend(handles=[lines[1], ln_omega_cmd],
                   labels=["yaw rate", "cmd ω"],
                   loc="upper right", fontsize=6,
                   facecolor="#2a2a3e", edgecolor="#555", labelcolor="lightgray")

    ln_v_cmd, = axes[3].plot([], [], color="#ff9040", linewidth=1.0, linestyle="--")
    axes[3].legend(handles=[lines[3], ln_v_cmd],
                   labels=["actual vel", "cmd vel"],
                   loc="upper right", fontsize=6,
                   facecolor="#2a2a3e", edgecolor="#555", labelcolor="lightgray")

    # ── Sliders ─────────────────────────────────────────────────────────────
    ax_sld_v = fig.add_axes([0.18, 0.25, 0.60, 0.025])
    ax_sld_v.set_facecolor("#2a2a3e")
    sld_v = Slider(ax_sld_v, "v (m/s)", -3.0, 3.0, valinit=0.0, color="#50e080")
    sld_v.label.set_color("lightgray"); sld_v.label.set_fontsize(8)
    sld_v.valtext.set_color("#ff6060"); sld_v.valtext.set_fontsize(8)

    ax_sld_w = fig.add_axes([0.18, 0.20, 0.60, 0.025])
    ax_sld_w.set_facecolor("#2a2a3e")
    sld_w = Slider(ax_sld_w, "ω (rad/s)", -5.0, 5.0, valinit=0.0, color="#f08040")
    sld_w.label.set_color("lightgray"); sld_w.label.set_fontsize(8)
    sld_w.valtext.set_color("#ffb060"); sld_w.valtext.set_fontsize(8)

    hip_nom_pct = (Q_NOM - Q_RET) / (Q_EXT - Q_RET) * 100.0
    ax_sld_h = fig.add_axes([0.18, 0.15, 0.60, 0.025])
    ax_sld_h.set_facecolor("#2a2a3e")
    sld_h = Slider(ax_sld_h, "Hip % (0=up)", 0.0, 100.0, valinit=hip_nom_pct, color="#c080ff")
    sld_h.label.set_color("lightgray"); sld_h.label.set_fontsize(8)
    sld_h.valtext.set_color("#c080ff"); sld_h.valtext.set_fontsize(8)

    # ── Buttons ─────────────────────────────────────────────────────────────
    drive_active = [True]
    ax_drv = fig.add_axes([0.05, 0.08, 0.20, 0.045])
    btn_drv = Button(ax_drv, "Drive ON", color="#2a5e3a", hovercolor="#5a5a9e")
    btn_drv.label.set_color("white"); btn_drv.label.set_fontsize(8)

    turn_active = [True]
    ax_trn = fig.add_axes([0.28, 0.08, 0.20, 0.045])
    btn_trn = Button(ax_trn, "Turn ON", color="#5e3a2a", hovercolor="#5a5a9e")
    btn_trn.label.set_color("white"); btn_trn.label.set_fontsize(8)

    ax_rst = fig.add_axes([0.74, 0.08, 0.20, 0.045])
    btn_rst = Button(ax_rst, "Restart", color="#3a3a5e", hovercolor="#5a5a9e")
    btn_rst.label.set_color("white"); btn_rst.label.set_fontsize(9)
    btn_rst.on_clicked(lambda _: cmd_q.put_nowait("RESTART"))

    _from_joy = [False]   # guard: suppress cmd echo when joystick updates sliders

    def _toggle_drive(_):
        drive_active[0] = not drive_active[0]
        if drive_active[0]:
            btn_drv.label.set_text("Drive ON")
            btn_drv.ax.set_facecolor("#2a5e3a")
            cmd_q.put_nowait(("V_DESIRED", sld_v.val))
        else:
            btn_drv.label.set_text("Drive OFF")
            btn_drv.ax.set_facecolor("#3a3a5e")
            cmd_q.put_nowait(("V_DESIRED", 0.0))
        fig.canvas.draw_idle()
    btn_drv.on_clicked(_toggle_drive)

    def _toggle_turn(_):
        turn_active[0] = not turn_active[0]
        if turn_active[0]:
            btn_trn.label.set_text("Turn ON")
            btn_trn.ax.set_facecolor("#5e3a2a")
            cmd_q.put_nowait(("OMEGA_DESIRED", sld_w.val))
        else:
            btn_trn.label.set_text("Turn OFF")
            btn_trn.ax.set_facecolor("#3a3a5e")
            cmd_q.put_nowait(("OMEGA_DESIRED", 0.0))
        fig.canvas.draw_idle()
    btn_trn.on_clicked(_toggle_turn)

    def _on_v(_):
        if _from_joy[0]: return
        if drive_active[0]: cmd_q.put_nowait(("V_DESIRED", sld_v.val))
    sld_v.on_changed(_on_v)

    def _on_w(_):
        if _from_joy[0]: return
        if turn_active[0]: cmd_q.put_nowait(("OMEGA_DESIRED", sld_w.val))
    sld_w.on_changed(_on_w)

    def _on_h(_):
        if _from_joy[0]: return
        cmd_q.put_nowait(("HIP_PCT", sld_h.val))
    sld_h.on_changed(_on_h)

    fig.suptitle(title, color="white", fontsize=9)
    fig.show()

    try:
        import ctypes as _ct
        _ct.windll.user32.SetProcessDPIAware()
        sw = _ct.windll.user32.GetSystemMetrics(0)
        sh = _ct.windll.user32.GetSystemMetrics(1)
        half = sw // 2
        win = fig.canvas.manager.window
        win.geometry(f"{half}x{sh}+0+0")
        win.update_idletasks()
        fig.set_size_inches(half / fig.dpi, sh / fig.dpi, forward=True)
    except Exception:
        pass

    while plt.fignum_exists(fig.number):
        items = []
        while True:
            try: items.append(data_q.get_nowait())
            except Exception: break
        if not items:
            plt.pause(1.0 / 30)
            continue

        for item in items:
            if item is None: return
            if item == "RESET":
                for buf in [t_buf, pitch_buf, theta_ref_buf, yaw_rate_buf,
                            pitch_rate_buf, vel_buf, v_cmd_buf, omega_cmd_buf]:
                    buf.clear()
                for ln in lines: ln.set_data([], [])
                ln_theta_ref.set_data([], [])
                ln_omega_cmd.set_data([], [])
                fig.canvas.flush_events()
                continue
            if isinstance(item, tuple) and item[0] == "JOY_V":
                _from_joy[0] = True; sld_v.set_val(item[1]); _from_joy[0] = False
                continue
            if isinstance(item, tuple) and item[0] == "JOY_W":
                _from_joy[0] = True; sld_w.set_val(item[1]); _from_joy[0] = False
                continue
            if isinstance(item, tuple) and item[0] == "JOY_H":
                _from_joy[0] = True; sld_h.set_val(item[1]); _from_joy[0] = False
                continue
            t, p, th_ref, yaw_rate, pitch_rate, vel, v_cmd, omega_cmd = item
            t_buf.append(t); pitch_buf.append(p); theta_ref_buf.append(th_ref)
            yaw_rate_buf.append(yaw_rate); pitch_rate_buf.append(pitch_rate)
            vel_buf.append(vel); v_cmd_buf.append(v_cmd); omega_cmd_buf.append(omega_cmd)

        if len(t_buf) < 2: continue
        tb    = list(t_buf)
        sim_t = tb[-1]
        t0    = max(0.0, sim_t - window_s)
        idx   = next((i for i, t in enumerate(tb) if t >= t0), 0)
        tw    = tb[idx:]

        for i, (ln, ax, buf) in enumerate(zip(
                lines, axes,
                [pitch_buf, yaw_rate_buf, pitch_rate_buf, vel_buf])):
            bw = list(buf)[idx:]
            ln.set_data(tw, bw)
            ax.set_xlim(t0, sim_t + 0.5)
            if len(bw) > 1:
                if i == 0:
                    tr_bw = list(theta_ref_buf)[idx:]
                    all_vals = bw + (tr_bw if tr_bw else [])
                    lo, hi = min(all_vals), max(all_vals)
                elif i == 1:
                    oc_bw = list(omega_cmd_buf)[idx:]
                    all_vals = bw + (oc_bw if oc_bw else [])
                    lo, hi = min(all_vals), max(all_vals)
                elif i == 3:
                    vc_bw = list(v_cmd_buf)[idx:]
                    all_vals = bw + (vc_bw if vc_bw else [])
                    lo, hi = min(all_vals), max(all_vals)
                else:
                    lo, hi = min(bw), max(bw)
                span = max(hi - lo, 0.1)
                ax.set_ylim(lo - span * 0.15, hi + span * 0.15)

        tr_bw = list(theta_ref_buf)[idx:]
        ln_theta_ref.set_data(tw, tr_bw)
        oc_bw = list(omega_cmd_buf)[idx:]
        ln_omega_cmd.set_data(tw, oc_bw)
        vc_bw = list(v_cmd_buf)[idx:]
        ln_v_cmd.set_data(tw, vc_bw)
        fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Main sandbox viewer
# ---------------------------------------------------------------------------
def sandbox(slowmo: float = 1.0):
    # ── Joystick init ────────────────────────────────────────────────────────
    # Axis mapping (standard Xbox / most gamepads):
    #   Axis 1: Left stick Y  (up=-1, down=+1) → v_desired  (negate: up=forward)
    #   Axis 2: Right stick X (left=-1, right=+1) → omega    (negate: right=turn right)
    #   Axis 3: Right stick Y (up=-1, down=+1) → hip %      (up=retract, down=extend)
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

    # Compute LQR gain table from sim_config baselines
    _scenarios.LQR_K_TABLE = compute_gain_table(
        ROBOT,
        Q_diag=[LQR_Q_PITCH, LQR_Q_PITCH_RATE, LQR_Q_VEL],
        R_val=LQR_R,
    )
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

    # Address lookups
    free_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")
    qpos_free = model.jnt_qposadr[free_id]
    dof_free  = model.jnt_dofadr[free_id]
    d_yaw     = dof_free + 5   # world-frame ωz

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

    from physics import get_equilibrium_pitch as _gep
    vel_pi = VelocityPI(kp=VELOCITY_PI_KP, ki=VELOCITY_PI_KI, dt=_dt_ctrl)
    yaw_pi = YawPI(kp=YAW_PI_KP, ki=YAW_PI_KI, dt=_dt_ctrl)

    # Mutable control targets
    _v_desired     = [0.0]
    _omega_desired = [0.0]
    _hip_pct       = [_HIP_NOM_PCT]   # 0=retracted(up), 100=extended(down)

    def _hip_target():
        pct = max(0.0, min(100.0, _hip_pct[0])) / 100.0
        return Q_RET + pct * (_HIP_MAX_Q - Q_RET)

    # Delayed sensor state
    _pitch_d     = _gep(ROBOT, Q_NOM)
    _pitch_rate_d = 0.0
    _wheel_vel_d  = 0.0
    _theta_ref    = 0.0
    _prev_theta_ref = [0.0]
    _vel_est_ms   = 0.0

    rng  = np.random.default_rng(0)
    step = 0
    prev_sim_t = 0.0
    last_push  = -1.0

    def _init():
        init_sim(model, data)

    _init()

    # Telemetry process
    data_q = mp.Queue(maxsize=4000)
    cmd_q  = mp.Queue(maxsize=32)
    plot_proc = mp.Process(
        target=_plot_process,
        args=(data_q, cmd_q, WINDOW_S, "Sandbox — free drive"),
        daemon=True)
    plot_proc.start()

    threading.Thread(target=_position_mujoco_right, args=(1.5,), daemon=True).start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 45
        viewer.cam.elevation = -18
        viewer.cam.distance  = 3.0
        viewer.cam.lookat    = np.array([0.0, 0.0, 0.30])

        def _reset_state():
            nonlocal step, prev_sim_t, _pitch_d, _pitch_rate_d
            nonlocal _wheel_vel_d, _theta_ref, _vel_est_ms
            _init()
            step = 0; prev_sim_t = 0.0
            _pitch_d = _gep(ROBOT, Q_NOM); _pitch_rate_d = 0.0; _wheel_vel_d = 0.0
            _theta_ref = 0.0; _vel_est_ms = 0.0; _prev_theta_ref[0] = 0.0
            _v_desired[0] = 0.0; _omega_desired[0] = 0.0
            _hip_pct[0] = _HIP_NOM_PCT
            vel_pi.reset(); yaw_pi.reset()
            if not data_q.full(): data_q.put_nowait("RESET")

        while viewer.is_running():
            frame_start = time.perf_counter()
            sim_t = float(data.time)

            # Process commands from matplotlib
            try:
                cmd = cmd_q.get_nowait()
                if cmd == "RESTART":
                    _reset_state()
                elif isinstance(cmd, tuple):
                    if   cmd[0] == "V_DESIRED":     _v_desired[0]     = float(cmd[1])
                    elif cmd[0] == "OMEGA_DESIRED":  _omega_desired[0] = float(cmd[1])
                    elif cmd[0] == "HIP_PCT":        _hip_pct[0]       = float(cmd[1])
            except Exception:
                pass

            if sim_t < prev_sim_t - 0.01:
                _reset_state()
            prev_sim_t = sim_t

            # ── Joystick poll ────────────────────────────────────────────────
            if _joy is not None:
                for _ev in pygame.event.get():
                    if _ev.type == pygame.JOYAXISMOTION:
                        _joy_axes[_ev.axis] = _ev.value
                    elif _ev.type == pygame.JOYBUTTONDOWN and _ev.button == 0:
                        _reset_state()   # Button A = restart

                raw_v = _joy_axes.get(1, 0.0)   # left stick Y
                raw_w = _joy_axes.get(2, 0.0)   # right stick X
                raw_h = _joy_axes.get(3, 0.0)   # right stick Y

                v_joy = -raw_v if abs(raw_v) > JOY_DEADZONE else 0.0
                w_joy = -raw_w if abs(raw_w) > JOY_DEADZONE else 0.0
                # hip: stick full up (-1) = extended/low (100%), full down (+1) = retracted/tall (0%)
                h_joy = (1.0 - raw_h) / 2.0 * 100.0 if abs(raw_h) > JOY_DEADZONE else None

                _v_desired[0]     = v_joy * 3.0
                _omega_desired[0] = w_joy * 5.0
                if h_joy is not None:
                    _hip_pct[0] = h_joy

                # Mirror joystick position to matplotlib sliders
                if not data_q.full():
                    data_q.put_nowait(("JOY_V", _v_desired[0]))
                    data_q.put_nowait(("JOY_W", _omega_desired[0]))
                    if h_joy is not None:
                        data_q.put_nowait(("JOY_H", _hip_pct[0]))

            # Physics + control loop
            for _ in range(steps_per_frame):
                if step % CTRL_STEPS == 0:
                    pitch_true, pitch_rate_true = get_pitch_and_rate(data, box_bid, d_pitch)
                    pitch      = pitch_true      + rng.normal(0, PITCH_NOISE_STD_RAD)
                    pitch_rate = pitch_rate_true + rng.normal(0, PITCH_RATE_NOISE_STD_RAD_S)
                    wheel_vel  = (data.qvel[dof_whl_L] + data.qvel[dof_whl_R]) / 2.0
                    q_hip_avg  = (data.qpos[qpos_hip_L] + data.qpos[qpos_hip_R]) / 2.0

                    _pitch_d, _pitch_rate_d, _wheel_vel_d = pitch, pitch_rate, wheel_vel
                    _vel_est_ms = _wheel_vel_d * WHEEL_R

                    # VelocityPI → θ_ref
                    v_ref_rads  = _v_desired[0] / WHEEL_R
                    theta_ref   = vel_pi.update(_v_desired[0], _vel_est_ms)
                    d_max       = THETA_REF_RATE_LIMIT * _dt_ctrl
                    theta_ref   = float(np.clip(theta_ref,
                                                _prev_theta_ref[0] - d_max,
                                                _prev_theta_ref[0] + d_max))
                    _prev_theta_ref[0] = theta_ref
                    _theta_ref         = theta_ref

                    # LQR symmetric torque
                    tau_sym = lqr_torque(_pitch_d, _pitch_rate_d, _wheel_vel_d,
                                         q_hip_avg, v_ref=v_ref_rads, theta_ref=theta_ref)

                    # YawPI differential torque
                    yaw_rate = data.qvel[d_yaw]
                    tau_yaw  = yaw_pi.update(_omega_desired[0], yaw_rate)

                    data.ctrl[act_wheel_L] = np.clip(tau_sym - tau_yaw, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
                    data.ctrl[act_wheel_R] = np.clip(tau_sym + tau_yaw, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)

                    # Hip impedance + roll leveling — same logic as _run_sim_loop
                    q_sym = _hip_target()
                    q_roll = data.xquat[box_bid]   # [w, x, y, z]
                    roll_true = math.atan2(
                        2.0 * (q_roll[0]*q_roll[1] + q_roll[2]*q_roll[3]),
                        1.0 - 2.0 * (q_roll[1]**2  + q_roll[2]**2))
                    roll_rate = data.qvel[dof_free + 3]
                    roll_meas = roll_true + rng.normal(0, ROLL_NOISE_STD_RAD)
                    delta_q   = LEG_K_ROLL * roll_meas + LEG_D_ROLL * roll_rate
                    q_nom_L   = float(np.clip(q_sym + delta_q, HIP_SAFE_MIN, HIP_SAFE_MAX))
                    q_nom_R   = float(np.clip(q_sym - delta_q, HIP_SAFE_MIN, HIP_SAFE_MAX))
                    for qpos_hip, dof_hip, act_hip, q_nom_leg in [
                        (qpos_hip_L, dof_hip_L, act_hip_L, q_nom_L),
                        (qpos_hip_R, dof_hip_R, act_hip_R, q_nom_R),
                    ]:
                        q_hip  = data.qpos[qpos_hip]
                        dq_hip = data.qvel[dof_hip]
                        tau_h  = -(LEG_K_S * (q_hip - q_nom_leg) + LEG_B_S * dq_hip)
                        data.ctrl[act_hip] = np.clip(tau_h,
                                                     -HIP_IMPEDANCE_TORQUE_LIMIT,
                                                      HIP_IMPEDANCE_TORQUE_LIMIT)

                mujoco.mj_step(model, data)
                step += 1

            viewer.sync()

            # Camera follows robot
            robot_pos = data.xpos[box_bid]
            viewer.cam.lookat[0] = robot_pos[0]
            viewer.cam.lookat[1] = robot_pos[1]

            # Push telemetry
            wall_now = time.perf_counter()
            if wall_now - last_push >= 1.0 / TELEMETRY_HZ and not data_q.full():
                pitch_true, pitch_rate_true = get_pitch_and_rate(data, box_bid, d_pitch)
                q_hip_avg  = (data.qpos[qpos_hip_L] + data.qpos[qpos_hip_R]) / 2.0
                _pitch_ff  = _gep(ROBOT, q_hip_avg)
                pitch_ref_display = _pitch_ff + _theta_ref
                data_q.put_nowait((
                    sim_t,
                    math.degrees(pitch_true),
                    math.degrees(pitch_ref_display),
                    data.qvel[d_yaw],
                    math.degrees(pitch_rate_true),
                    _vel_est_ms,
                    _v_desired[0],
                    _omega_desired[0],
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Interactive sandbox arena — free drive with obstacles.",
        epilog="python sandbox.py\npython sandbox.py --slowmo 2")
    ap.add_argument("--slowmo", type=float, default=1.0,
                    help="Slow-motion factor (e.g. 2 = half real-time)")
    args = ap.parse_args()
    sandbox(slowmo=args.slowmo)


if __name__ == "__main__":
    mp.freeze_support()
    main()
