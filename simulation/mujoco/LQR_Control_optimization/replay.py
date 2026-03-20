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
    LEG_K_ROLL, LEG_D_ROLL, ROLL_NOISE_STD_RAD, HIP_SAFE_MIN, HIP_SAFE_MAX,
    LQR_Q_PITCH, LQR_Q_PITCH_RATE, LQR_Q_VEL, LQR_R,
    PITCH_NOISE_STD_RAD, PITCH_RATE_NOISE_STD_RAD_S,
    CTRL_STEPS, PITCH_STEP_RAD, THETA_REF_RATE_LIMIT,
    S5_BUMPS, S8_BUMPS, S8_DRIVE_SPEED,
    YAW_PI_KP, YAW_PI_KI,
    BATT_V_NOM,
)
from physics import build_xml, build_assets, get_equilibrium_pitch
from run_log import (
    CSV_PATH, load_run, load_all_runs, get_best_run,
    list_runs, overwrite_run,
)
import scenarios as _scenarios
from scenarios import (
    init_sim, get_pitch_and_rate, balance_torque, lqr_torque, evaluate,
    VelocityPI, YawPI, motor_taper, _motor_currents,
    FALL_THRESHOLD, BALANCE_DURATION,
    DISTURBANCE_TIME, DISTURBANCE_FORCE, DISTURBANCE_DUR,
    s1_dist_fn, s2_dist_fn, s3_velocity_profile, _leg_cycle_profile,
)
from battery_model import BatteryModel
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
    # 18 data channels
    (t_buf,
     pitch_buf, pitch_ref_buf,          # panel [0,0] — Pitch
     vel_buf, v_cmd_buf,                # panel [0,1] — Velocity
     yaw_buf, omega_cmd_buf,            # panel [1,0] — Yaw Rate
     hip_L_buf, hip_R_buf, hip_cmd_buf, # panel [1,1] — Hip joints
     roll_buf,                          # panel [2,0] — Roll
     pitch_rate_buf,                    # panel [2,1] — Pitch Rate
     tau_L_buf, tau_R_buf,              # panel [3,0] — Wheel Torques
     delta_q_buf,                       # panel [3,1] — Suspension Δq
     v_batt_buf, soc_buf, batt_temp_buf,# panel [4,0/1] — Battery
     ) = (deque(maxlen=MAXLEN) for _ in range(18))

    plt.ion()
    fig, axes = plt.subplots(5, 2, figsize=(13, 11))
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(hspace=0.65, wspace=0.38,
                        top=0.91, bottom=0.22, left=0.10, right=0.97)

    BG  = "#1e1e2e"
    GRD = "#333333"

    # ── Panel specs: (title, ylabel, primary_colour) ────────────────────────
    specs = [
        [("Pitch",            "deg",   "#60d0ff"),   ("Velocity",       "m/s",   "#50e080")],
        [("Yaw Rate",         "rad/s", "#f08040"),   ("Hip Joints",     "deg",   "#d0a0ff")],
        [("Roll",             "deg",   "#ff6090"),   ("Pitch Rate",     "deg/s", "#b060ff")],
        [("Wheel Torques",    "N·m",   "#ffcc44"),   ("Suspension Δq",  "deg",   "#44ddcc")],
        [("Battery Voltage",  "V",     "#ff9944"),   ("SoC / Temp",     "%",     "#44ffcc")],
    ]

    def _style_ax(ax, ttl, unit, col):
        ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_edgecolor("#555")
        ax.tick_params(colors="lightgray", labelsize=6)
        ax.grid(True, color=GRD, linewidth=0.4, linestyle="--")
        ax.set_title(ttl, color="white", fontsize=8, pad=2)
        ax.set_ylabel(unit, color=col, fontsize=7)
        ax.set_xlabel("t [s]", color="#888", fontsize=6)

    for r in range(5):
        for c in range(2):
            ttl, unit, col = specs[r][c]
            _style_ax(axes[r][c], ttl, unit, col)

    # Primary lines (one per panel)
    ln_pitch,    = axes[0][0].plot([], [], color="#60d0ff", lw=1.5)
    ln_vel,      = axes[0][1].plot([], [], color="#50e080", lw=1.5)
    ln_yaw,      = axes[1][0].plot([], [], color="#f08040", lw=1.5)
    ln_hip_L,    = axes[1][1].plot([], [], color="#d0a0ff", lw=1.5)
    ln_roll,     = axes[2][0].plot([], [], color="#ff6090", lw=1.5)
    ln_prate,    = axes[2][1].plot([], [], color="#b060ff", lw=1.5)
    ln_tau_L,    = axes[3][0].plot([], [], color="#ffcc44", lw=1.5)
    ln_delta_q,  = axes[3][1].plot([], [], color="#44ddcc", lw=1.5)
    ln_vbatt,    = axes[4][0].plot([], [], color="#ff9944", lw=1.5)
    ln_soc,      = axes[4][1].plot([], [], color="#44ffcc", lw=1.5)

    # Secondary / command overlay lines
    ln_pitch_ref, = axes[0][0].plot([], [], color="#ff6060", lw=1.0, ls="--")
    ln_v_cmd,     = axes[0][1].plot([], [], color="#b0ff80", lw=1.0, ls="--")
    ln_omega_cmd, = axes[1][0].plot([], [], color="#ffb060", lw=1.0, ls="--")
    ln_hip_R,     = axes[1][1].plot([], [], color="#a060cc", lw=1.2, ls="--")
    ln_hip_cmd,   = axes[1][1].plot([], [], color="#ffffff", lw=0.8, ls=":")
    ln_tau_R,     = axes[3][0].plot([], [], color="#ff8844", lw=1.2, ls="--")
    ln_vbatt_nom, = axes[4][0].plot([], [], color="#ffffff", lw=0.8, ls="--")
    ln_temp,      = axes[4][1].plot([], [], color="#ff6090", lw=1.2, ls="--")

    # Twin-y axis for temperature on SoC panel
    ax_temp = axes[4][1].twinx()
    ax_temp.set_facecolor(BG)
    ax_temp.tick_params(colors="lightgray", labelsize=6)
    ax_temp.set_ylabel("°C", color="#ff6090", fontsize=7)
    ln_temp, = ax_temp.plot([], [], color="#ff6090", lw=1.2, ls="--")

    # Zero/reference lines
    axes[1][0].axhline(0.0, color="#666688", lw=0.7, ls="--")
    axes[2][0].axhline(0.0, color="#666688", lw=0.7, ls="--")
    axes[2][1].axhline(0.0, color="#666688", lw=0.7, ls="--")
    axes[3][0].axhline(0.0, color="#666688", lw=0.7, ls="--")
    axes[3][1].axhline(0.0, color="#666688", lw=0.7, ls="--")

    # Legends
    _leg_kw = dict(loc="upper right", fontsize=5,
                   facecolor="#2a2a3e", edgecolor="#555", labelcolor="lightgray")
    axes[0][0].legend([ln_pitch, ln_pitch_ref], ["pitch", "cmd"], **_leg_kw)
    axes[0][1].legend([ln_vel,   ln_v_cmd],     ["actual", "cmd"], **_leg_kw)
    axes[1][0].legend([ln_yaw,   ln_omega_cmd], ["yaw ω", "cmd ω"], **_leg_kw)
    axes[1][1].legend([ln_hip_L, ln_hip_R, ln_hip_cmd],
                      ["hip L", "hip R", "cmd"], **_leg_kw)
    axes[3][0].legend([ln_tau_L, ln_tau_R], ["τ_L", "τ_R"], **_leg_kw)
    axes[4][0].legend([ln_vbatt, ln_vbatt_nom], ["V_term", "V_nom"], **_leg_kw)
    axes[4][1].legend([ln_soc,   ln_temp],       ["SoC %", "T °C"],  **_leg_kw)

    # ── Widgets ─────────────────────────────────────────────────────────────
    ax_sld = fig.add_axes([0.10, 0.19, 0.80, 0.025])
    ax_sld.set_facecolor("#2a2a3e")
    sld = Slider(ax_sld, "v_desired (m/s)", -1.2, 1.2, valinit=0.0, color="#50e080")
    sld.label.set_color("lightgray"); sld.label.set_fontsize(7)
    sld.valtext.set_color("#ff6060"); sld.valtext.set_fontsize(7)

    ax_sld_omega = fig.add_axes([0.10, 0.14, 0.80, 0.025])
    ax_sld_omega.set_facecolor("#2a2a3e")
    sld_omega = Slider(ax_sld_omega, "ω_desired (rad/s)", -2.0, 2.0, valinit=0.0, color="#f08040")
    sld_omega.label.set_color("lightgray"); sld_omega.label.set_fontsize(7)
    sld_omega.valtext.set_color("#ffb060"); sld_omega.valtext.set_fontsize(7)

    override_active = [False]
    ax_tog = fig.add_axes([0.04, 0.06, 0.14, 0.05])
    btn_tog = Button(ax_tog, "Drive OFF", color="#3a3a5e", hovercolor="#5a5a9e")
    btn_tog.label.set_color("white"); btn_tog.label.set_fontsize(7)

    def _toggle(_):
        override_active[0] = not override_active[0]
        if override_active[0]:
            btn_tog.label.set_text("Drive ON");  btn_tog.ax.set_facecolor("#2a5e3a")
            cmd_q.put_nowait(("V_DESIRED", sld.val))
        else:
            btn_tog.label.set_text("Drive OFF"); btn_tog.ax.set_facecolor("#3a3a5e")
            cmd_q.put_nowait(("V_DESIRED", 0.0))
        fig.canvas.draw_idle()
    btn_tog.on_clicked(_toggle)
    sld.on_changed(lambda _: cmd_q.put_nowait(("V_DESIRED", sld.val)) if override_active[0] else None)

    turn_active = [False]
    ax_turn = fig.add_axes([0.21, 0.06, 0.14, 0.05])
    btn_turn = Button(ax_turn, "Turn OFF", color="#3a3a5e", hovercolor="#5a5a9e")
    btn_turn.label.set_color("white"); btn_turn.label.set_fontsize(7)

    def _toggle_turn(_):
        turn_active[0] = not turn_active[0]
        if turn_active[0]:
            btn_turn.label.set_text("Turn ON");  btn_turn.ax.set_facecolor("#5e3a2a")
            cmd_q.put_nowait(("OMEGA_DESIRED", sld_omega.val))
        else:
            btn_turn.label.set_text("Turn OFF"); btn_turn.ax.set_facecolor("#3a3a5e")
            cmd_q.put_nowait(("OMEGA_DESIRED", 0.0))
        fig.canvas.draw_idle()
    btn_turn.on_clicked(_toggle_turn)
    sld_omega.on_changed(lambda _: cmd_q.put_nowait(("OMEGA_DESIRED", sld_omega.val)) if turn_active[0] else None)

    staircase_manual = [False]
    ax_sc = fig.add_axes([0.38, 0.06, 0.14, 0.05])
    btn_sc = Button(ax_sc, "Staircase", color="#3a3a5e", hovercolor="#5a5a9e")
    btn_sc.label.set_color("white"); btn_sc.label.set_fontsize(7)

    def _toggle_staircase(_):
        staircase_manual[0] = not staircase_manual[0]
        if staircase_manual[0]:
            btn_sc.label.set_text("Manual"); btn_sc.ax.set_facecolor("#5e4a2a")
            cmd_q.put_nowait("STAIRCASE_OFF")
            override_active[0] = True
            btn_tog.label.set_text("Drive ON"); btn_tog.ax.set_facecolor("#2a5e3a")
            cmd_q.put_nowait(("V_DESIRED", sld.val))
        else:
            btn_sc.label.set_text("Staircase"); btn_sc.ax.set_facecolor("#3a3a5e")
            cmd_q.put_nowait("STAIRCASE_ON")
            override_active[0] = False
            btn_tog.label.set_text("Drive OFF"); btn_tog.ax.set_facecolor("#3a3a5e")
            cmd_q.put_nowait(("V_DESIRED", 0.0))
        fig.canvas.draw_idle()
    btn_sc.on_clicked(_toggle_staircase)

    ax_btn = fig.add_axes([0.84, 0.06, 0.12, 0.05])
    btn    = Button(ax_btn, "Restart", color="#3a3a5e", hovercolor="#5a5a9e")
    btn.label.set_color("white"); btn.label.set_fontsize(7)
    btn.on_clicked(lambda _: cmd_q.put_nowait("RESTART"))

    fig.suptitle(title, color="white", fontsize=8)
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
        win.update_idletasks()
        fig.set_size_inches(half / fig.dpi, sh / fig.dpi, forward=True)
    except Exception:
        pass

    all_bufs = [t_buf, pitch_buf, pitch_ref_buf, vel_buf, v_cmd_buf,
                yaw_buf, omega_cmd_buf, hip_L_buf, hip_R_buf, hip_cmd_buf,
                roll_buf, pitch_rate_buf, tau_L_buf, tau_R_buf, delta_q_buf,
                v_batt_buf, soc_buf, batt_temp_buf]
    all_sec_lines = [ln_pitch_ref, ln_v_cmd, ln_omega_cmd,
                     ln_hip_R, ln_hip_cmd, ln_tau_R]

    # ── Blit setup ───────────────────────────────────────────────────────────
    # Mark every data line animated so it's excluded from the static background
    _animated_lines = [
        ln_pitch, ln_pitch_ref, ln_vel, ln_v_cmd, ln_yaw, ln_omega_cmd,
        ln_hip_L, ln_hip_R, ln_hip_cmd, ln_roll, ln_prate,
        ln_tau_L, ln_tau_R, ln_delta_q, ln_vbatt, ln_vbatt_nom,
        ln_soc, ln_temp,
    ]
    for _ln in _animated_lines:
        _ln.set_animated(True)

    fig.canvas.draw()   # initial full draw to capture clean backgrounds
    _ax_list = list(axes.flat) + [ax_temp]
    _bgs     = [fig.canvas.copy_from_bbox(ax.bbox) for ax in _ax_list]

    _RESCALE_EVERY = 6   # full redraw + background recapture every 6 rendered frames
    _render_frame  = 0
    _do_rescale    = True
    _last_draw = 0.0
    while plt.fignum_exists(fig.number):
        items = []
        while True:
            try: items.append(data_q.get_nowait())
            except Exception: break
        if not items:
            plt.pause(0.005)
            continue

        for item in items:
            if item is None: return
            if item == "RESET":
                for buf in all_bufs: buf.clear()
                for ln in [ln_pitch, ln_vel, ln_yaw, ln_hip_L, ln_roll,
                           ln_prate, ln_tau_L, ln_delta_q, ln_vbatt, ln_soc] + all_sec_lines:
                    ln.set_data([], [])
                fig.canvas.flush_events()
                continue
            (t, pitch, pitch_ref, vel, v_cmd, yaw, omega_cmd,
             hip_L, hip_R, hip_cmd, roll, prate, tau_L, tau_R, delta_q,
             v_batt, soc, batt_temp) = item
            t_buf.append(t)
            pitch_buf.append(pitch);      pitch_ref_buf.append(pitch_ref)
            vel_buf.append(vel);          v_cmd_buf.append(v_cmd)
            yaw_buf.append(yaw);          omega_cmd_buf.append(omega_cmd)
            hip_L_buf.append(hip_L);      hip_R_buf.append(hip_R);  hip_cmd_buf.append(hip_cmd)
            roll_buf.append(roll)
            pitch_rate_buf.append(prate)
            tau_L_buf.append(tau_L);      tau_R_buf.append(tau_R)
            delta_q_buf.append(delta_q)
            v_batt_buf.append(v_batt);    soc_buf.append(soc);  batt_temp_buf.append(batt_temp)

        if len(t_buf) < 2: continue
        tb    = list(t_buf)
        sim_t = tb[-1]
        t0    = max(0.0, sim_t - window_s)
        idx   = next((i for i, t in enumerate(tb) if t >= t0), 0)
        tw    = tb[idx:]

        _render_frame += 1
        _do_rescale = (_render_frame % _RESCALE_EVERY == 0)

        def _autoscale(ax, *bufs):
            if not _do_rescale: return
            all_vals = []
            for b in bufs:
                all_vals += list(b)[idx:]
            if len(all_vals) < 2: return
            lo, hi = min(all_vals), max(all_vals)
            span = max(hi - lo, 0.05)
            ax.set_xlim(t0, sim_t + 0.5)
            ax.set_ylim(lo - span * 0.15, hi + span * 0.15)

        # Update each panel
        def _upd(ln, ax, buf, *extra_bufs):
            bw = list(buf)[idx:]
            ln.set_data(tw, bw)
            _autoscale(ax, buf, *extra_bufs)

        _upd(ln_pitch,   axes[0][0], pitch_buf,    pitch_ref_buf)
        _upd(ln_vel,     axes[0][1], vel_buf,       v_cmd_buf)
        _upd(ln_yaw,     axes[1][0], yaw_buf,       omega_cmd_buf)
        _upd(ln_hip_L,   axes[1][1], hip_L_buf,     hip_R_buf, hip_cmd_buf)
        _upd(ln_roll,    axes[2][0], roll_buf)
        _upd(ln_prate,   axes[2][1], pitch_rate_buf)
        _upd(ln_tau_L,   axes[3][0], tau_L_buf,     tau_R_buf)
        _upd(ln_delta_q, axes[3][1], delta_q_buf)

        # Battery voltage panel
        vb_bw = list(v_batt_buf)[idx:]
        if vb_bw:
            ln_vbatt.set_data(tw, vb_bw)
            if _do_rescale:
                axes[4][0].set_xlim(t0, sim_t + 0.5)
                lo, hi = min(vb_bw), max(vb_bw)
                span = max(hi - lo, 0.5)
                axes[4][0].set_ylim(lo - span * 0.15, hi + span * 0.15)
            ln_vbatt_nom.set_data([t0, sim_t + 0.5], [BATT_V_NOM, BATT_V_NOM])

        # SoC + temperature panel
        soc_bw  = list(soc_buf)[idx:]
        temp_bw = list(batt_temp_buf)[idx:]
        if soc_bw:
            ln_soc.set_data(tw, soc_bw)
            if _do_rescale:
                axes[4][1].set_xlim(t0, sim_t + 0.5)
                axes[4][1].set_ylim(max(0.0, min(soc_bw) - 5), 105)
        if temp_bw:
            ln_temp.set_data(tw, temp_bw)
            if _do_rescale:
                ax_temp.set_xlim(t0, sim_t + 0.5)
                lo_t, hi_t = min(temp_bw), max(temp_bw)
                span_t = max(hi_t - lo_t, 2.0)
                ax_temp.set_ylim(lo_t - span_t * 0.2, hi_t + span_t * 0.5)

        # Secondary lines
        ln_pitch_ref.set_data(tw, list(pitch_ref_buf)[idx:])
        ln_v_cmd.set_data(tw,     list(v_cmd_buf)[idx:])
        ln_omega_cmd.set_data(tw, list(omega_cmd_buf)[idx:])
        ln_hip_R.set_data(tw,     list(hip_R_buf)[idx:])
        ln_hip_cmd.set_data(tw,   list(hip_cmd_buf)[idx:])
        ln_tau_R.set_data(tw,     list(tau_R_buf)[idx:])

        # ── Render ───────────────────────────────────────────────────────────
        _now = time.monotonic()
        if _now - _last_draw < 1.0 / 60:
            continue   # skip render — will catch up next iteration
        _last_draw = _now

        if _do_rescale:
            # Full redraw on rescale frames — recapture static backgrounds
            fig.canvas.draw()
            _bgs[:] = [fig.canvas.copy_from_bbox(ax.bbox) for ax in _ax_list]
        else:
            # Blit: restore cached background, draw only animated lines on top
            for ax, bg in zip(_ax_list, _bgs):
                fig.canvas.restore_region(bg)
                for ln in ax.get_lines():
                    if ln.get_animated():
                        ax.draw_artist(ln)
                fig.canvas.blit(ax.bbox)
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

    # Build model — S5 gets full-width bumps, S8 gets one-sided sandbox obstacles
    if scenario == "5_VEL_PI_leg_cycling":
        xml = build_xml(bumps=S5_BUMPS)
    elif scenario == "8_terrain_compliance":
        xml = build_xml(sandbox_obstacles=S8_BUMPS)
    else:
        xml = build_xml()
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
        "2_leg_height_gain_sched":   s1_dist_fn,
        "3_VEL_PI_disturbance":      s2_dist_fn,
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
    _hip_sym_ref        = [Q_NOM]  # common-mode hip target; updated each control step
    _staircase_active   = [True]  # False when user switches to manual slider control

    # Outer Velocity PI (v_desired=0 in balance mode → position hold)
    _dt_ctrl = model.opt.timestep * CTRL_STEPS
    vel_pi = VelocityPI(kp=kp_v, ki=ki_v, dt=_dt_ctrl)
    yaw_pi = YawPI(kp=kp_yaw, ki=ki_yaw, dt=_dt_ctrl)
    battery = BatteryModel(); battery.reset()
    _v_batt = [BATT_V_NOM]

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
            battery.reset(); _v_batt[0] = BATT_V_NOM
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
                                        "2_leg_height_gain_sched"):
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
                        elif scenario == "8_terrain_compliance":
                            # S8: constant forward drive at S8_DRIVE_SPEED (1.0 m/s)
                            _vel_est_ms = _wheel_vel_d * WHEEL_R
                            if _staircase_active[0]:
                                _v_desired[0] = S8_DRIVE_SPEED
                            theta_ref = vel_pi.update(_v_desired[0], _vel_est_ms)
                            _d_max = THETA_REF_RATE_LIMIT * _dt_ctrl
                            theta_ref = float(np.clip(
                                theta_ref,
                                _prev_theta_ref[0] - _d_max,
                                _prev_theta_ref[0] + _d_max))
                            _prev_theta_ref[0] = theta_ref
                            _theta_ref  = theta_ref
                        else:
                            _vel_est_ms = _wheel_vel_d * WHEEL_R
                            if scenario == "4_VEL_PI_staircase" and _staircase_active[0]:
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
                        data.ctrl[act_wheel_L] = motor_taper(tau_sym - tau_yaw, data.qvel[dof_whl_L], _v_batt[0])
                        data.ctrl[act_wheel_R] = motor_taper(tau_sym + tau_yaw, data.qvel[dof_whl_R], _v_batt[0])
                        # Leg impedance + roll leveling
                        _q_hip_sym = (_leg_cycle_profile(float(data.time))
                                      if scenario in ("2_leg_height_gain_sched",
                                                       "5_VEL_PI_leg_cycling")
                                      else Q_NOM)
                        _hip_sym_ref[0] = _q_hip_sym
                        _q_roll   = data.xquat[box_bid]
                        _roll_true = math.atan2(
                            2.0 * (_q_roll[0]*_q_roll[1] + _q_roll[2]*_q_roll[3]),
                            1.0 - 2.0 * (_q_roll[1]**2  + _q_roll[2]**2))
                        _roll_rate = data.qvel[dof_free + 3]
                        _roll_meas = _roll_true + rng.normal(0, ROLL_NOISE_STD_RAD)
                        _delta_q   = LEG_K_ROLL * _roll_meas + LEG_D_ROLL * _roll_rate
                        _q_nom_L   = float(np.clip(_q_hip_sym + _delta_q, HIP_SAFE_MIN, HIP_SAFE_MAX))
                        _q_nom_R   = float(np.clip(_q_hip_sym - _delta_q, HIP_SAFE_MIN, HIP_SAFE_MAX))
                        for qpos_hip, dof_hip, act_hip, _q_nom_leg in [
                            (qpos_hip_L, dof_hip_L, act_hip_L, _q_nom_L),
                            (qpos_hip_R, dof_hip_R, act_hip_R, _q_nom_R),
                        ]:
                            q_hip   = data.qpos[qpos_hip]
                            dq_hip  = data.qvel[dof_hip]
                            tau_hip = -(LEG_K_S * (q_hip - _q_nom_leg) + LEG_B_S * dq_hip)
                            data.ctrl[act_hip] = np.clip(tau_hip, -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)
                        _v_batt[0] = battery.step(_dt_ctrl, _motor_currents(
                            float(data.ctrl[act_wheel_L]), float(data.ctrl[act_wheel_R]),
                            float(data.ctrl[act_hip_L]),   float(data.ctrl[act_hip_R])))
                    else:
                        tau_wheel, pitch_integral, odo_x = balance_torque(
                            _pitch_d, _pitch_rate_d, pitch_integral, odo_x,
                            _wheel_vel_d, q_hip_avg, dt, gains)
                        data.ctrl[act_wheel_L] = motor_taper(tau_wheel, data.qvel[dof_whl_L], _v_batt[0])
                        data.ctrl[act_wheel_R] = motor_taper(tau_wheel, data.qvel[dof_whl_R], _v_batt[0])
                        for qpos_hip, dof_hip, act_hip in [
                            (qpos_hip_L, dof_hip_L, act_hip_L),
                            (qpos_hip_R, dof_hip_R, act_hip_R),
                        ]:
                            q_hip   = data.qpos[qpos_hip]
                            dq_hip  = data.qvel[dof_hip]
                            tau_hip = -(LEG_K_S * (q_hip - Q_NOM) + LEG_B_S * dq_hip)
                            data.ctrl[act_hip] = np.clip(tau_hip, -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)
                        _v_batt[0] = battery.step(_dt_ctrl, _motor_currents(
                            float(data.ctrl[act_wheel_L]), float(data.ctrl[act_wheel_R]),
                            float(data.ctrl[act_hip_L]),   float(data.ctrl[act_hip_R])))

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
                if scenario in ("1_LQR_pitch_step", "2_LQR_impulse_recovery"):
                    _pitch_ref_display = _gep(ROBOT, Q_NOM)
                else:
                    _pitch_ref_display = _pitch_ff_now + _theta_ref
                # Roll angle from body quaternion (all scenarios)
                _q_r = data.xquat[box_bid]
                _roll_deg_tel = math.degrees(math.atan2(
                    2.0 * (_q_r[0]*_q_r[1] + _q_r[2]*_q_r[3]),
                    1.0 - 2.0 * (_q_r[1]**2 + _q_r[2]**2)))
                data_q.put_nowait((
                    sim_t,
                    math.degrees(pitch_true),                                    # 1  pitch
                    math.degrees(_pitch_ref_display),                            # 2  pitch_cmd
                    _vel_est_ms,                                                 # 3  vel
                    _v_desired[0],                                               # 4  v_cmd
                    data.qvel[d_yaw],                                            # 5  yaw_rate
                    _omega_desired[0],                                           # 6  omega_cmd
                    math.degrees(data.qpos[qpos_hip_L]),                         # 7  hip_L
                    math.degrees(data.qpos[qpos_hip_R]),                         # 8  hip_R
                    math.degrees(_hip_sym_ref[0]),                               # 9  hip_cmd
                    _roll_deg_tel,                                               # 10 roll
                    math.degrees(pitch_rate_true),                               # 11 pitch_rate
                    float(data.ctrl[act_wheel_L]),                               # 12 tau_L
                    float(data.ctrl[act_wheel_R]),                               # 13 tau_R
                    math.degrees(data.qpos[qpos_hip_L] - data.qpos[qpos_hip_R]),# 14 delta_q
                    battery.v_terminal,                                          # 15 v_batt
                    battery.soc_pct,                                             # 16 soc_pct
                    battery.temperature_c,                                       # 17 batt_temp
                ))
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
    if plot_proc.is_alive():
        plot_proc.terminate()


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
