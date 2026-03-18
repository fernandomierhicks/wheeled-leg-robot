"""replay.py — Visual replay of any logged run in MuJoCo viewer.

Loads controller gains and scenario from the CSV, re-runs the exact same
simulation, and shows the MuJoCo viewer with a live matplotlib telemetry panel.

Usage:
    python replay.py                    # replay best run (by fitness)
    python replay.py 42                 # replay run_id = 42
    python replay.py --top 3            # replay 3rd-best run
    python replay.py --list             # list all runs
    python replay.py --resim 42         # re-simulate run 42 and overwrite CSV row
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


# ---------------------------------------------------------------------------
# Window auto-positioning (Windows only)
# ---------------------------------------------------------------------------
def _screen_size():
    u32 = ctypes.windll.user32
    u32.SetProcessDPIAware()
    return u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)


def _move_window(hwnd, x, y, w, h):
    SWP_NOZORDER = 0x0004
    ctypes.windll.user32.SetWindowPos(hwnd, 0, x, y, w, h, SWP_NOZORDER)


def _find_window_by_partial_title(partial: str) -> int:
    """Return HWND of first visible top-level window whose title contains `partial`."""
    result = [0]
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)

    def _cb(hwnd, _):
        if ctypes.windll.user32.IsWindowVisible(hwnd):
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            if partial.lower() in buf.value.lower():
                result[0] = hwnd
                return False  # stop enumeration
        return True

    ctypes.windll.user32.EnumWindows(EnumWindowsProc(_cb), 0)
    return result[0]


def _position_mujoco_right(delay: float = 2.0):
    """Background thread: wait for MuJoCo window then snap it to right half."""
    time.sleep(delay)
    sw, sh = _screen_size()
    half = sw // 2
    for _ in range(20):          # retry for up to 4 s
        hwnd = _find_window_by_partial_title("MuJoCo")
        if hwnd:
            _move_window(hwnd, half, 0, half, sh)
            return
        time.sleep(0.2)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sim_config import (
    ROBOT, Q_NOM, WHEEL_R,
    HIP_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT,
    LEG_K_S, LEG_B_S,
    LQR_Q_PITCH, LQR_Q_PITCH_RATE, LQR_Q_VEL, LQR_R,
    PITCH_NOISE_STD_RAD, PITCH_RATE_NOISE_STD_RAD_S,
    CTRL_STEPS, PITCH_STEP_RAD, THETA_REF_RATE_LIMIT,
    S5_BUMPS,
    YAW_PI_KP, YAW_PI_KI,
)
from physics import build_xml, build_assets, get_equilibrium_pitch
from run_log import (
    CSV_PATH, load_run, load_all_runs, get_best_run,
    list_runs, overwrite_run,
)
import scenarios as _scenarios
from scenarios import (
    init_sim, get_pitch_and_rate, balance_torque, lqr_torque, evaluate,
    VelocityPI, YawPI,
    FALL_THRESHOLD, BALANCE_DURATION,
    DISTURBANCE_TIME, DISTURBANCE_FORCE, DISTURBANCE_DUR,
    s1_dist_fn, s2_dist_fn, s3_velocity_profile, _leg_cycle_profile,
)
from lqr_design import compute_gain_table

RENDER_HZ    = 60
TELEMETRY_HZ = 100   # how often to push data to the plot process
WINDOW_S     = 8.0   # rolling time window shown in matplotlib


# ---------------------------------------------------------------------------
# CSV → gains dict
# ---------------------------------------------------------------------------
def _row_to_gains(row: dict) -> dict:
    # For LQR runs, KP/KD may be empty; use defaults
    def safe_float(val, default):
        if val is None or val == '' or val == 'nan':
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    return {
        "KP":     safe_float(row.get("KP"),     60.0),
        "KD":     safe_float(row.get("KD"),      5.0),
        "KP_pos": safe_float(row.get("KP_pos"),  0.30),
        "KP_vel": safe_float(row.get("KP_vel"),  0.30),
    }


def _get_best_run_id(rank: int = 1, csv_path: str = None) -> int:
    rows = load_all_runs(csv_path or CSV_PATH)
    pass_rows = []
    for r in rows:
        if r.get("status") == "PASS":
            try:
                pass_rows.append((float(r["fitness"]), int(r["run_id"])))
            except (ValueError, KeyError):
                pass
    if not pass_rows:
        raise ValueError("No PASS runs found in CSV")
    pass_rows.sort(key=lambda x: x[0])   # ascending fitness (lower = better)
    if rank > len(pass_rows):
        raise ValueError(f"Only {len(pass_rows)} PASS runs exist, cannot get rank {rank}")
    return pass_rows[rank - 1][1]


# ---------------------------------------------------------------------------
# Matplotlib telemetry (separate process)
# ---------------------------------------------------------------------------
def _plot_process(data_q: mp.Queue, cmd_q: mp.Queue,
                  window_s: float, title: str) -> None:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    MAXLEN = int(window_s * TELEMETRY_HZ) + 500
    t_buf, pitch_buf, theta_ref_buf, yaw_rate_buf, pitch_rate_buf, vel_buf, v_cmd_buf, omega_cmd_buf = (deque(maxlen=MAXLEN) for _ in range(8))

    plt.ion()
    fig, axes = plt.subplots(4, 1, figsize=(6, 9))
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(hspace=0.60, top=0.91, bottom=0.28, left=0.18, right=0.96)

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
    # Zero reference line on pitch rate panel
    axes[2].axhline(0.0, color="#666688", linewidth=0.8, linestyle="--")
    # Zero reference line on yaw rate panel
    axes[1].axhline(0.0, color="#666688", linewidth=0.8, linestyle="--")
    # Second line on pitch axes: θ_ref commanded by VelocityPI or slider
    ln_theta_ref, = axes[0].plot([], [], color="#ff6060", linewidth=1.0,
                                 linestyle="--")
    axes[0].legend(handles=[lines[0], ln_theta_ref],
                   labels=["pitch", "pitch cmd (ff+θ_ref)"],
                   loc="upper right", fontsize=6,
                   facecolor="#2a2a3e", edgecolor="#555", labelcolor="lightgray")
    # Second line on yaw axes: commanded yaw rate (omega_desired)
    ln_omega_cmd, = axes[1].plot([], [], color="#ffb060", linewidth=1.0, linestyle="--")
    axes[1].legend(handles=[lines[1], ln_omega_cmd],
                   labels=["yaw rate", "cmd ω"],
                   loc="upper right", fontsize=6,
                   facecolor="#2a2a3e", edgecolor="#555", labelcolor="lightgray")
    # Second line on velocity axes: commanded velocity (staircase / slider)
    ln_v_cmd, = axes[3].plot([], [], color="#ff9040", linewidth=1.0, linestyle="--")
    axes[3].legend(handles=[lines[3], ln_v_cmd],
                   labels=["actual vel", "cmd vel"],
                   loc="upper right", fontsize=6,
                   facecolor="#2a2a3e", edgecolor="#555", labelcolor="lightgray")

    # ── Widgets ─────────────────────────────────────────────────────────────
    # v_desired slider  [left, bottom, width, height]
    ax_sld = fig.add_axes([0.18, 0.19, 0.60, 0.03])
    ax_sld.set_facecolor("#2a2a3e")
    sld = Slider(ax_sld, "v_desired (m/s)", -1.2, 1.2, valinit=0.0, color="#50e080")
    sld.label.set_color("lightgray"); sld.label.set_fontsize(8)
    sld.valtext.set_color("#ff6060"); sld.valtext.set_fontsize(8)

    # ω_desired slider
    ax_sld_omega = fig.add_axes([0.18, 0.13, 0.60, 0.03])
    ax_sld_omega.set_facecolor("#2a2a3e")
    sld_omega = Slider(ax_sld_omega, "ω_desired (rad/s)", -2.0, 2.0, valinit=0.0, color="#f08040")
    sld_omega.label.set_color("lightgray"); sld_omega.label.set_fontsize(8)
    sld_omega.valtext.set_color("#ffb060"); sld_omega.valtext.set_fontsize(8)

    # Drive ON/OFF toggle
    override_active = [False]
    ax_tog = fig.add_axes([0.05, 0.06, 0.18, 0.05])
    btn_tog = Button(ax_tog, "Drive OFF", color="#3a3a5e", hovercolor="#5a5a9e")
    btn_tog.label.set_color("white"); btn_tog.label.set_fontsize(8)

    def _toggle(_):
        override_active[0] = not override_active[0]
        if override_active[0]:
            btn_tog.label.set_text("Drive ON")
            btn_tog.ax.set_facecolor("#2a5e3a")
            cmd_q.put_nowait(("V_DESIRED", sld.val))
        else:
            btn_tog.label.set_text("Drive OFF")
            btn_tog.ax.set_facecolor("#3a3a5e")
            cmd_q.put_nowait(("V_DESIRED", 0.0))
        fig.canvas.draw_idle()
    btn_tog.on_clicked(_toggle)

    def _on_slider(_):
        if override_active[0]:
            cmd_q.put_nowait(("V_DESIRED", sld.val))
    sld.on_changed(_on_slider)

    # Turn ON/OFF toggle
    turn_active = [False]
    ax_turn = fig.add_axes([0.26, 0.06, 0.18, 0.05])
    btn_turn = Button(ax_turn, "Turn OFF", color="#3a3a5e", hovercolor="#5a5a9e")
    btn_turn.label.set_color("white"); btn_turn.label.set_fontsize(8)

    def _toggle_turn(_):
        turn_active[0] = not turn_active[0]
        if turn_active[0]:
            btn_turn.label.set_text("Turn ON")
            btn_turn.ax.set_facecolor("#5e3a2a")
            cmd_q.put_nowait(("OMEGA_DESIRED", sld_omega.val))
        else:
            btn_turn.label.set_text("Turn OFF")
            btn_turn.ax.set_facecolor("#3a3a5e")
            cmd_q.put_nowait(("OMEGA_DESIRED", 0.0))
        fig.canvas.draw_idle()
    btn_turn.on_clicked(_toggle_turn)

    def _on_omega_slider(_):
        if turn_active[0]:
            cmd_q.put_nowait(("OMEGA_DESIRED", sld_omega.val))
    sld_omega.on_changed(_on_omega_slider)

    # Staircase override toggle (S3 manual mode)
    staircase_manual = [False]
    ax_sc = fig.add_axes([0.47, 0.06, 0.18, 0.05])
    btn_sc = Button(ax_sc, "Staircase", color="#3a3a5e", hovercolor="#5a5a9e")
    btn_sc.label.set_color("white"); btn_sc.label.set_fontsize(8)

    def _toggle_staircase(_):
        staircase_manual[0] = not staircase_manual[0]
        if staircase_manual[0]:
            btn_sc.label.set_text("Manual")
            btn_sc.ax.set_facecolor("#5e4a2a")
            cmd_q.put_nowait("STAIRCASE_OFF")
            # Also activate drive so slider works immediately
            override_active[0] = True
            btn_tog.label.set_text("Drive ON")
            btn_tog.ax.set_facecolor("#2a5e3a")
            cmd_q.put_nowait(("V_DESIRED", sld.val))
        else:
            btn_sc.label.set_text("Staircase")
            btn_sc.ax.set_facecolor("#3a3a5e")
            cmd_q.put_nowait("STAIRCASE_ON")
            override_active[0] = False
            btn_tog.label.set_text("Drive OFF")
            btn_tog.ax.set_facecolor("#3a3a5e")
            cmd_q.put_nowait(("V_DESIRED", 0.0))
        fig.canvas.draw_idle()
    btn_sc.on_clicked(_toggle_staircase)

    # Restart button
    ax_btn = fig.add_axes([0.78, 0.06, 0.18, 0.05])
    btn    = Button(ax_btn, "Restart", color="#3a3a5e", hovercolor="#5a5a9e")
    btn.label.set_color("white"); btn.label.set_fontsize(9)
    btn.on_clicked(lambda _: cmd_q.put_nowait("RESTART"))

    fig.suptitle(title, color="white", fontsize=9)
    fig.show()

    # Snap matplotlib window to left half of primary monitor
    try:
        import ctypes as _ct
        _ct.windll.user32.SetProcessDPIAware()
        sw = _ct.windll.user32.GetSystemMetrics(0)
        sh = _ct.windll.user32.GetSystemMetrics(1)
        half = sw // 2
        win = fig.canvas.manager.window
        win.geometry(f"{half}x{sh}+0+0")
        win.update_idletasks()   # flush geometry so canvas knows its new size
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
                for buf in [t_buf, pitch_buf, theta_ref_buf, yaw_rate_buf, pitch_rate_buf, vel_buf, v_cmd_buf, omega_cmd_buf]:
                    buf.clear()
                for ln in lines: ln.set_data([], [])
                ln_theta_ref.set_data([], [])
                ln_omega_cmd.set_data([], [])
                fig.canvas.flush_events()
                continue
            t, p, th_ref, yaw_rate, pitch_rate, vel, v_cmd, omega_cmd = item
            t_buf.append(t); pitch_buf.append(p); theta_ref_buf.append(th_ref)
            yaw_rate_buf.append(yaw_rate); pitch_rate_buf.append(pitch_rate)
            vel_buf.append(vel); v_cmd_buf.append(v_cmd); omega_cmd_buf.append(omega_cmd)

        if len(t_buf) < 2: continue
        tb     = list(t_buf)
        sim_t  = tb[-1]
        t0     = max(0.0, sim_t - window_s)
        idx    = next((i for i, t in enumerate(tb) if t >= t0), 0)
        tw     = tb[idx:]

        for i, (ln, ax, buf) in enumerate(zip(lines, axes, [pitch_buf, yaw_rate_buf, pitch_rate_buf, vel_buf])):
            bw = list(buf)[idx:]
            ln.set_data(tw, bw)
            ax.set_xlim(t0, sim_t + 0.5)
            if len(bw) > 1:
                if i == 0:
                    # Autoscale pitch axes to both pitch and θ_ref signals
                    tr_bw = list(theta_ref_buf)[idx:]
                    all_vals = bw + (tr_bw if tr_bw else [])
                    lo, hi = min(all_vals), max(all_vals)
                elif i == 1:
                    # Autoscale yaw axes to both measured and commanded yaw rate
                    oc_bw = list(omega_cmd_buf)[idx:]
                    all_vals = bw + (oc_bw if oc_bw else [])
                    lo, hi = min(all_vals), max(all_vals)
                elif i == 3:
                    # Autoscale velocity axes to both actual and commanded velocity
                    vc_bw_s = list(v_cmd_buf)[idx:]
                    all_vals = bw + (vc_bw_s if vc_bw_s else [])
                    lo, hi = min(all_vals), max(all_vals)
                else:
                    lo, hi = min(bw), max(bw)
                span = max(hi - lo, 0.1)
                ax.set_ylim(lo - span * 0.15, hi + span * 0.15)
        # Update θ_ref line on pitch axes
        tr_bw = list(theta_ref_buf)[idx:]
        ln_theta_ref.set_data(tw, tr_bw)
        # Update commanded yaw rate line on yaw axes
        oc_bw = list(omega_cmd_buf)[idx:]
        ln_omega_cmd.set_data(tw, oc_bw)
        # Update commanded velocity line on velocity axes
        vc_bw = list(v_cmd_buf)[idx:]
        ln_v_cmd.set_data(tw, vc_bw)
        fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Main replay
# ---------------------------------------------------------------------------
def replay(run_id: int = None, baseline: bool = False, scenario_override: str = None,
           freefall: bool = False, slowmo: float = 1.0, csv_path: str = None):
    """Launch MuJoCo viewer for the given run_id (best run if None).

    Pass baseline=True to skip CSV and use the baselined LQR gains from
    sim_config.py directly.  scenario_override sets the scenario regardless
    of what is stored in the CSV row.
    """
    if baseline:
        scenario_default = scenario_override or "balance_disturbance"
        row = dict(
            run_id=0, label="baseline_lqr",
            scenario=scenario_default,
            Q_PITCH=LQR_Q_PITCH, Q_PITCH_RATE=LQR_Q_PITCH_RATE,
            Q_VEL=LQR_Q_VEL, R=LQR_R,
            fitness="(baseline)",
        )
    else:
        csv_path = csv_path or CSV_PATH
        if run_id is None:
            run_id = _get_best_run_id(rank=1, csv_path=csv_path)
        row = load_run(run_id, csv_path)

    gains = _row_to_gains(row)
    label = row.get("label", "?")
    scenario = scenario_override or row.get("scenario", "balance")
    logged_fit = row.get("fitness", "?")

    # Load LQR gains from CSV row if present, otherwise fall back to sim_config defaults
    def _safe_float(v, default):
        try: return float(v) if v not in (None, '', 'nan') else default
        except (ValueError, TypeError): return default

    q_pitch      = max(1e-6, _safe_float(row.get("Q_PITCH"),      LQR_Q_PITCH))
    q_pitch_rate = max(1e-6, _safe_float(row.get("Q_PITCH_RATE"), LQR_Q_PITCH_RATE))
    q_vel        = max(1e-6, _safe_float(row.get("Q_VEL"),        LQR_Q_VEL))
    r_val        = max(1e-6, _safe_float(row.get("R"),             LQR_R))
    kp_v         = _safe_float(row.get("KP_V"),    _scenarios.VELOCITY_PI_KP)
    ki_v         = _safe_float(row.get("KI_V"),    _scenarios.VELOCITY_PI_KI)
    kp_yaw       = _safe_float(row.get("KP_YAW"),  YAW_PI_KP)
    ki_yaw       = _safe_float(row.get("KI_YAW"),  YAW_PI_KI)
    _scenarios.LQR_K_TABLE = compute_gain_table(
        ROBOT, Q_diag=[q_pitch, q_pitch_rate, q_vel], R_val=r_val
    )
    _scenarios.USE_PD_CONTROLLER = False
    use_lqr = True
    print(f"  LQR weights: Q=[{q_pitch:.4g}, {q_pitch_rate:.4g}, {q_vel:.4g}]  R={r_val:.4g}")
    print(f"  VelocityPI:  Kp_v={kp_v:.4g}  Ki_v={ki_v:.4g}")
    print(f"  YawPI:       Kp_yaw={kp_yaw:.4g}  Ki_yaw={ki_yaw:.4g}")

    print(f"\nReplaying run {run_id}  [{label}]  scenario={scenario}")
    print(f"  Logged fitness: {logged_fit}")
    print(f"  Gains: {gains}")

    # Build model — S5 gets bump obstacles in the world
    _bumps = S5_BUMPS if scenario == "5_VEL_PI_leg_cycling" else None
    xml    = build_xml(bumps=_bumps)
    assets = build_assets()
    model  = mujoco.MjModel.from_xml_string(xml, assets)
    data   = mujoco.MjData(model)

    # Address lookups
    free_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")
    qpos_free = model.jnt_qposadr[free_id]
    dof_free  = model.jnt_dofadr[free_id]

    def _dof(name): return model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _qpos(name): return model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)]

    d_yaw       = dof_free + 5   # world-frame ωz; positive = CCW = left turn
    dof_hip_L   = _dof("hip_L");          dof_hip_R   = _dof("hip_R")
    qpos_hip_L  = _qpos("hip_L");         qpos_hip_R  = _qpos("hip_R")
    dof_whl_L   = _dof("wheel_spin_L");   dof_whl_R   = _dof("wheel_spin_R")

    act_hip_L   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_act_L")
    act_hip_R   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_act_R")
    act_wheel_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_act_L")
    act_wheel_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_act_R")

    PHYSICS_HZ      = round(1.0 / model.opt.timestep)
    steps_per_frame = max(1, PHYSICS_HZ // RENDER_HZ)

    def _init():
        init_sim(model, data)
        if scenario == "1_LQR_pitch_step":
            # Apply +PITCH_STEP_RAD perturbation on top of equilibrium
            from physics import get_equilibrium_pitch as _gep2
            theta = _gep2(ROBOT, Q_NOM) + PITCH_STEP_RAD
            data.qpos[qpos_free + 3] = math.cos(theta / 2.0)
            data.qpos[qpos_free + 4] = 0.0
            data.qpos[qpos_free + 5] = math.sin(theta / 2.0)
            data.qpos[qpos_free + 6] = 0.0
            mujoco.mj_forward(model, data)

    _init()

    # Telemetry process
    plot_title = f"run {run_id} | {label} | fit={logged_fit}"
    data_q  = mp.Queue(maxsize=4000)
    cmd_q   = mp.Queue(maxsize=16)
    plot_proc = mp.Process(target=_plot_process,
                           args=(data_q, cmd_q, WINDOW_S, plot_title),
                           daemon=True)
    plot_proc.start()

    # Controller state
    step            = 0
    prev_sim_t      = 0.0
    last_push       = -1.0
    pitch_integral  = 0.0
    odo_x           = 0.0
    rng             = np.random.default_rng(0)
    box_bid         = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    d_pitch         = model.jnt_dofadr[mujoco.mj_name2id(
                          model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")] + 4
    disturbance_applied = False

    # Resolve disturbance function once from scenarios.py (single source of truth)
    _generic_dist = lambda t: (DISTURBANCE_FORCE
                               if DISTURBANCE_TIME <= t < DISTURBANCE_TIME + DISTURBANCE_DUR
                               else 0.0)
    _scenario_dist_fn = {
        "1_LQR_pitch_step":          s1_dist_fn,
        "4_leg_height_gain_sched":   s1_dist_fn,
        "2_VEL_PI_disturbance":      s2_dist_fn,
        "5_VEL_PI_leg_cycling":      s2_dist_fn,
        "balance_disturbance":       _generic_dist,
        "lqr_combined":              _generic_dist,
        "2_LQR_impulse_recovery":    _generic_dist,
    }.get(scenario, lambda t: 0.0)

    # 1-step sensor delay buffer — models ~2 ms BNO086 I2C + CAN latency.
    from physics import get_equilibrium_pitch as _gep
    _pitch_d      = _gep(ROBOT, Q_NOM)
    _pitch_rate_d = 0.0
    _wheel_vel_d  = 0.0
    _theta_ref          = 0.0   # last θ_ref sent to controller (for telemetry)
    _prev_theta_ref     = [0.0] # rate-limiter state (mutable for nested scope)
    _vel_est_ms         = 0.0   # last velocity estimate fed into VelocityPI (m/s)
    _v_desired          = [0.0] # velocity setpoint [m/s]; slider writes here via cmd_q
    _omega_desired      = [0.0] # yaw rate setpoint [rad/s]; slider writes here via cmd_q
    _staircase_active   = [True]  # False when user switches to manual slider control

    # Outer Velocity PI (v_desired=0 in balance mode → position hold)
    _dt_ctrl = model.opt.timestep * CTRL_STEPS
    vel_pi = VelocityPI(kp=kp_v, ki=ki_v, dt=_dt_ctrl)
    yaw_pi = YawPI(kp=kp_yaw, ki=ki_yaw, dt=_dt_ctrl)

    # Snap MuJoCo window to right half once it appears
    threading.Thread(target=_position_mujoco_right, args=(1.5,), daemon=True).start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 35
        viewer.cam.elevation = -15
        viewer.cam.distance  =  2.5
        viewer.cam.lookat    = np.array([0.0, 0.0, 0.30])

        def _reset_state():
            nonlocal step, prev_sim_t, pitch_integral, odo_x
            nonlocal _pitch_d, _pitch_rate_d, _wheel_vel_d, _theta_ref, _vel_est_ms
            _init()
            step = 0; prev_sim_t = 0.0
            pitch_integral = 0.0; odo_x = 0.0
            _pitch_d = _gep(ROBOT, Q_NOM); _pitch_rate_d = 0.0; _wheel_vel_d = 0.0
            _theta_ref = 0.0; _vel_est_ms = 0.0
            _prev_theta_ref[0] = 0.0
            _staircase_active[0] = True
            _v_desired[0] = 0.0
            _omega_desired[0] = 0.0
            vel_pi.reset()
            yaw_pi.reset()
            if not data_q.full(): data_q.put_nowait("RESET")

        while viewer.is_running():
            frame_start = time.perf_counter()
            sim_t = float(data.time)

            try:
                cmd = cmd_q.get_nowait()
                if cmd == "RESTART":
                    _reset_state()
                elif isinstance(cmd, tuple) and cmd[0] == "V_DESIRED":
                    _v_desired[0] = float(cmd[1])
                elif isinstance(cmd, tuple) and cmd[0] == "OMEGA_DESIRED":
                    _omega_desired[0] = float(cmd[1])
                elif cmd == "STAIRCASE_OFF":
                    _staircase_active[0] = False
                elif cmd == "STAIRCASE_ON":
                    _staircase_active[0] = True
            except Exception:
                pass
            if sim_t < prev_sim_t - 0.01:
                _reset_state()
            prev_sim_t = sim_t

            # Physics: 2000 Hz.  Controller: 500 Hz = every CTRL_STEPS steps.
            dt = model.opt.timestep * CTRL_STEPS
            for _ in range(steps_per_frame):
                if step % CTRL_STEPS == 0:
                    pitch_true, pitch_rate_true = get_pitch_and_rate(data, box_bid, d_pitch)
                    pitch      = pitch_true      + rng.normal(0, PITCH_NOISE_STD_RAD)
                    pitch_rate = pitch_rate_true + rng.normal(0, PITCH_RATE_NOISE_STD_RAD_S)
                    wheel_vel  = (data.qvel[dof_whl_L] + data.qvel[dof_whl_R]) / 2.0

                    q_hip_avg = (data.qpos[qpos_hip_L] + data.qpos[qpos_hip_R]) / 2.0

                    # Rotate delay buffer (0ms: update before use)
                    _pitch_d, _pitch_rate_d, _wheel_vel_d = pitch, pitch_rate, wheel_vel

                    if freefall:
                        # No controller — all actuators zeroed, robot falls freely
                        data.ctrl[act_wheel_L] = 0.0
                        data.ctrl[act_wheel_R] = 0.0
                        data.ctrl[act_hip_L]   = 0.0
                        data.ctrl[act_hip_R]   = 0.0
                    elif use_lqr:
                        if scenario in ("1_LQR_pitch_step", "2_LQR_impulse_recovery",
                                        "4_leg_height_gain_sched"):
                            # LQR isolation scenarios — VelocityPI disabled
                            theta_ref   = 0.0
                            _vel_est_ms = _wheel_vel_d * WHEEL_R
                        elif scenario == "5_VEL_PI_leg_cycling" and _staircase_active[0]:
                            # S5: auto-run staircase profile (can switch to manual via button)
                            _vel_est_ms = _wheel_vel_d * WHEEL_R
                            _v_desired[0] = s3_velocity_profile(float(data.time))
                            theta_ref   = vel_pi.update(_v_desired[0], _vel_est_ms)
                            _d_max = THETA_REF_RATE_LIMIT * _dt_ctrl
                            theta_ref = float(np.clip(
                                theta_ref,
                                _prev_theta_ref[0] - _d_max,
                                _prev_theta_ref[0] + _d_max))
                            _prev_theta_ref[0] = theta_ref
                            _theta_ref  = theta_ref
                        else:
                            _vel_est_ms = _wheel_vel_d * WHEEL_R
                            if scenario == "3_VEL_PI_staircase" and _staircase_active[0]:
                                _v_desired[0] = s3_velocity_profile(float(data.time))
                            theta_ref   = vel_pi.update(_v_desired[0], _vel_est_ms)
                            _d_max = THETA_REF_RATE_LIMIT * _dt_ctrl
                            theta_ref = float(np.clip(
                                theta_ref,
                                _prev_theta_ref[0] - _d_max,
                                _prev_theta_ref[0] + _d_max))
                            _prev_theta_ref[0] = theta_ref
                        _theta_ref  = theta_ref
                        v_ref_rads = _v_desired[0] / WHEEL_R
                        tau_sym = lqr_torque(_pitch_d, _pitch_rate_d, _wheel_vel_d, q_hip_avg,
                                             v_ref=v_ref_rads, theta_ref=theta_ref)
                        yaw_rate = data.qvel[d_yaw]
                        tau_yaw = yaw_pi.update(_omega_desired[0], yaw_rate) if _scenarios.USE_YAW_PI else 0.0
                        data.ctrl[act_wheel_L] = np.clip(tau_sym - tau_yaw, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
                        data.ctrl[act_wheel_R] = np.clip(tau_sym + tau_yaw, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
                        _q_hip_tgt = (_leg_cycle_profile(float(data.time))
                                      if scenario in ("4_leg_height_gain_sched",
                                                       "5_VEL_PI_leg_cycling")
                                      else Q_NOM)
                        for qpos_hip, dof_hip, act_hip in [
                            (qpos_hip_L, dof_hip_L, act_hip_L),
                            (qpos_hip_R, dof_hip_R, act_hip_R),
                        ]:
                            q_hip   = data.qpos[qpos_hip]
                            dq_hip  = data.qvel[dof_hip]
                            tau_hip = -(LEG_K_S * (q_hip - _q_hip_tgt) + LEG_B_S * dq_hip)
                            data.ctrl[act_hip] = np.clip(tau_hip, -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)
                    else:
                        tau_wheel, pitch_integral, odo_x = balance_torque(
                            _pitch_d, _pitch_rate_d, pitch_integral, odo_x,
                            _wheel_vel_d, q_hip_avg, dt, gains)
                        data.ctrl[act_wheel_L] = tau_wheel
                        data.ctrl[act_wheel_R] = tau_wheel
                        for qpos_hip, dof_hip, act_hip in [
                            (qpos_hip_L, dof_hip_L, act_hip_L),
                            (qpos_hip_R, dof_hip_R, act_hip_R),
                        ]:
                            q_hip   = data.qpos[qpos_hip]
                            dq_hip  = data.qvel[dof_hip]
                            tau_hip = -(LEG_K_S * (q_hip - Q_NOM) + LEG_B_S * dq_hip)
                            data.ctrl[act_hip] = np.clip(tau_hip, -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)

                # Apply disturbance impulse — function resolved once before the loop
                data.xfrc_applied[box_bid, 0] = _scenario_dist_fn(data.time)

                mujoco.mj_step(model, data)
                step += 1

            viewer.sync()

            # Camera follow — track robot XY, keep Z fixed at body height
            robot_pos = data.xpos[box_bid]
            viewer.cam.lookat[0] = robot_pos[0]
            viewer.cam.lookat[1] = robot_pos[1]

            # Push telemetry
            wall_now = time.perf_counter()
            if wall_now - last_push >= 1.0 / TELEMETRY_HZ and not data_q.full():
                pitch_true, pitch_rate_true = get_pitch_and_rate(data, box_bid, d_pitch)
                q_hip_avg = (data.qpos[qpos_hip_L] + data.qpos[qpos_hip_R]) / 2.0
                _pitch_ff_now = _gep(ROBOT, q_hip_avg)
                # For LQR isolation scenarios use fixed Q_NOM reference so the
                # reference line is a clean horizontal (hip barely moves anyway).
                if scenario in ("1_LQR_pitch_step", "2_LQR_impulse_recovery"):
                    _pitch_ref_display = _gep(ROBOT, Q_NOM)  # fixed reference — hip doesn't move
                else:
                    _pitch_ref_display = _pitch_ff_now + _theta_ref
                hip_q_avg_tel = (data.qpos[qpos_hip_L] + data.qpos[qpos_hip_R]) / 2.0
                data_q.put_nowait((sim_t,
                                   math.degrees(pitch_true),
                                   math.degrees(_pitch_ref_display),
                                   data.qvel[d_yaw],
                                   math.degrees(pitch_rate_true),
                                   _vel_est_ms,
                                   _v_desired[0],
                                   _omega_desired[0]))
                last_push = wall_now

            # HUD overlay + world frame axes
            viewer.user_scn.ngeom = 0
            # XYZ frame at world origin (just above floor).
            # Cylinders are aligned along their local Z, so we rotate to each world axis.
            _r   = 0.007          # shaft radius [m]
            _hl  = 0.10           # half-length [m]
            _o   = np.array([0.0, 0.0, 0.01])   # origin, 1 cm above floor
            _ax  = [
                # (center offset,  rotation matrix rows,                rgba)
                (np.array([_hl, 0,  0  ]),  # +X  red
                 np.array([[0,0,1],[0,1,0],[-1,0,0]], dtype=np.float32),
                 np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32)),
                (np.array([0,  _hl, 0  ]),  # +Y  green
                 np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float32),
                 np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32)),
                (np.array([0,  0,  _hl ]),  # +Z  blue
                 np.eye(3, dtype=np.float32),
                 np.array([0.1, 0.4, 1.0, 1.0], dtype=np.float32)),
            ]
            for i, (off, mat, rgba) in enumerate(_ax):
                g = viewer.user_scn.geoms[i]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER,
                                    [_r, _hl, 0],
                                    (_o + off).tolist(),
                                    mat.flatten().tolist(),
                                    rgba)
                g.label = (b"+X" if i==0 else b"+Y" if i==1 else b"+Z")
            viewer.user_scn.ngeom = 3
            # HUD label on invisible sphere
            g = viewer.user_scn.geoms[3]
            hud = f"run {run_id} | {label[:20]}"
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                [0.006, 0, 0], [-0.30, 0.15, 0.65],
                                np.eye(3).flatten(),
                                np.array([0.4, 1.0, 0.4, 0.0], dtype=np.float32))  # alpha=0: invisible
            g.label = hud.encode()[:99]
            viewer.user_scn.ngeom = 4

            elapsed = time.perf_counter() - frame_start
            sleep_t = slowmo / RENDER_HZ - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    data_q.put(None)
    plot_proc.join(timeout=2)


# ---------------------------------------------------------------------------
# Re-simulate and overwrite
# ---------------------------------------------------------------------------
def resim(run_id: int):
    """Re-run the scenario for run_id and overwrite its CSV row."""
    row   = load_run(run_id)
    gains = _row_to_gains(row)
    label = row.get("label", f"resim_{run_id}")
    scenario = row.get("scenario", "balance")
    print(f"Re-simulating run {run_id} [{label}] ...")
    new_row = evaluate(gains, scenario=scenario, label=label,
                       run_id=run_id, csv_path=CSV_PATH)
    overwrite_run(run_id, new_row)
    print(f"Run {run_id} overwritten: fitness={new_row.get('fitness','?')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Replay or re-simulate logged runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python replay.py                 # replay best run
  python replay.py 42              # replay run_id=42
  python replay.py --top 3         # replay 3rd-best run
  python replay.py --list          # list all runs
  python replay.py --resim 42      # re-simulate run 42 and overwrite CSV
""")
    ap.add_argument("run_id", nargs="?", type=int, default=None,
                    help="run_id to replay")
    ap.add_argument("--top",   type=int, default=None,
                    help="Replay Nth-best run by fitness")
    ap.add_argument("--list",  action="store_true",
                    help="Print all runs and exit")
    ap.add_argument("--resim", type=int, default=None,
                    help="Re-simulate run_id and overwrite CSV row")
    ap.add_argument("--baseline", action="store_true",
                    help="Replay baselined LQR gains from sim_config (disturbance scenario)")
    ap.add_argument("--scenario", type=str, default=None,
                    help="Override scenario: balance | balance_disturbance | lqr_combined | 1_LQR_pitch_step | 2_LQR_impulse_recovery")
    ap.add_argument("--freefall", action="store_true",
                    help="Seed robot and run with zero torque — no controller active")
    ap.add_argument("--slowmo", type=float, default=1.0,
                    help="Slow-motion factor (e.g. 10 = 10x slower than real-time)")
    ap.add_argument("--csv", type=str, default=None,
                    help="Path to CSV file to load runs from (default: results.csv)")
    args = ap.parse_args()

    if args.list:
        list_runs()
        return

    if args.resim is not None:
        resim(args.resim)
        return

    if args.freefall:
        replay(baseline=True, scenario_override="balance", freefall=True,
               slowmo=args.slowmo)
        return

    if args.baseline:
        replay(baseline=True, scenario_override=args.scenario, slowmo=args.slowmo)
        return

    run_id = args.run_id
    if args.top is not None:
        run_id = _get_best_run_id(rank=args.top, csv_path=args.csv)
        print(f"Top-{args.top} run -> run_id={run_id}")

    replay(run_id, scenario_override=args.scenario, slowmo=args.slowmo,
           csv_path=args.csv)


if __name__ == "__main__":
    mp.freeze_support()
    main()
