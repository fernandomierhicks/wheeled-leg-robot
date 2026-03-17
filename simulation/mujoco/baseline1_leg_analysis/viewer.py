"""viewer.py — MuJoCo GUI viewer for baseline-1 two-leg robot.

Usage:
    python simulation/mujoco/baseline1_leg_analysis/viewer.py

Features:
    - Two-leg model (left +Y, right -Y), independent L/R control ready
    - Realistic component geoms (chassis, arms, battery, Arduino, ODESC, bearings)
    - Buttons: Drive Fwd (hold) | Drive Bwd (hold) | Restart | Neutral | Crouch | Jump
    - Slow-motion from JUMP start through landing + 0.5 s
    - Matplotlib telemetry panel — four tabs:
        • Motion      : Pitch, Wheel Torque, Jump Height  (live scrolling)
        • Femur       : Femur Axial+GRF, Femur Lateral, Bearing A
        • Tib/Cpl     : Tibia Axial+Coupler Axial, Tibia Lateral, Bearing C+E
        • Bearings    : Bearings A+W, Bearings C+E, Bearing F+GRF
      All tabs include a live peak-value table (shown below each force tab).
    - Force log written to force_log.csv in this directory each run.
      Columns: time_s, pitch_deg, height_mm,
               fax_fem_N, flat_fem_N, fax_tib_N, flat_tib_N, fax_cpl_N,
               fbear_A_N, fbear_C_N, fbear_E_N, fbear_F_N, fbear_W_N,
               grf_L_N
    - CG marker + speed indicator overlay
"""
import csv
import math
import multiprocessing as mp
import os
import sys
import time
from collections import deque

try:
    import pygame
    import os as _os
    _os.environ['SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS'] = '1'
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_config import *
from physics import solve_ik, get_equilibrium_pitch, build_xml, build_assets

RENDER_HZ           = 60
PUSH_HZ             = 200
WINDOW_S            = 12.0
JUMP_SLOWMO         = 10
DRIVE_PITCH_OFFSET  = 0.30
DRIVE_RAMP_S        = 0.20
CROUCH_DURATION_V   = 0.83
NEUTRAL_RAMP_S      = 0.50
TURN_TORQUE         = 0.15

STATE_NEUTRAL = 0
STATE_CROUCH  = 1
STATE_JUMP    = 2

# CSV columns (order must match what the sim loop sends)
_CSV_HEADER = [
    'time_s', 'pitch_deg', 'height_mm',
    'fax_fem_N', 'flat_fem_N',
    'fax_tib_N', 'flat_tib_N',
    'fax_cpl_N',
    'fbear_A_N', 'fbear_C_N', 'fbear_E_N', 'fbear_F_N', 'fbear_W_N',
    'grf_L_N',
]

# Keys used in max_abs dict (must be a subset of CSV columns, without _N suffix)
_STRUCT_KEYS = [
    'fax_fem', 'flat_fem',
    'fax_tib', 'flat_tib',
    'fax_cpl',
    'fbear_A', 'fbear_C', 'fbear_E', 'fbear_F', 'fbear_W',
    'grf_L',
]


# ---------------------------------------------------------------------------
# Matplotlib telemetry process
# ---------------------------------------------------------------------------
def _plot_process(q: mp.Queue, cmd_q: mp.Queue, window_s: float) -> None:
    import matplotlib; matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    MAXLEN = int(window_s * 1000) + 500

    # ── Data buffers ──────────────────────────────────────────────────────────
    t_buf         = deque(maxlen=MAXLEN)
    pitch_buf     = deque(maxlen=MAXLEN)
    torque_buf    = deque(maxlen=MAXLEN)
    height_buf    = deque(maxlen=MAXLEN)
    fax_fem_buf   = deque(maxlen=MAXLEN)
    flat_fem_buf  = deque(maxlen=MAXLEN)
    fax_tib_buf   = deque(maxlen=MAXLEN)
    flat_tib_buf  = deque(maxlen=MAXLEN)
    fax_cpl_buf   = deque(maxlen=MAXLEN)
    fbear_A_buf   = deque(maxlen=MAXLEN)
    fbear_C_buf   = deque(maxlen=MAXLEN)
    fbear_E_buf   = deque(maxlen=MAXLEN)
    fbear_F_buf   = deque(maxlen=MAXLEN)
    fbear_W_buf   = deque(maxlen=MAXLEN)
    grf_L_buf     = deque(maxlen=MAXLEN)

    max_abs = {k: 0.0 for k in _STRUCT_KEYS}

    # ── CSV log ───────────────────────────────────────────────────────────────
    _log_dir    = os.path.dirname(os.path.abspath(__file__))
    _log_path   = os.path.join(_log_dir, 'force_log.csv')
    _log_file   = open(_log_path, 'w', newline='')
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(_CSV_HEADER)

    # ── Figure layout ─────────────────────────────────────────────────────────
    plt.ion()
    fig, axes_m = plt.subplots(3, 1, figsize=(6, 8))
    fig.patch.set_facecolor("#1e1e2e")
    # bottom=0.42 leaves room for the peak table between plots and drive buttons
    plt.subplots_adjust(hspace=0.50, top=0.88, bottom=0.42, left=0.18, right=0.96)

    BTN_COL  = "#3a3a5e"
    BTN_HOV  = "#5a5a9e"
    BTN_ACTV = "#1a5a1a"

    # ── Helper: style one axes ────────────────────────────────────────────────
    def _style_ax(ax, title, ylabel, col):
        ax.set_facecolor("#1e1e2e")
        for sp in ax.spines.values(): sp.set_edgecolor("#555")
        ax.tick_params(colors="lightgray", labelsize=7)
        ax.grid(True, color="#333", linewidth=0.4, linestyle="--")
        ax.set_title(title, color="white", fontsize=9, pad=3)
        ax.set_ylabel(ylabel, color=col, fontsize=8)
        ax.set_xlabel("sim time [s]", color="lightgray", fontsize=7)

    # ── Motion tab axes (created by subplots) ─────────────────────────────────
    spec_motion = [
        ("Pitch",        "deg", "#60d0ff"),
        ("Wheel Torque", "N·m", "#f08040"),
        ("Jump Height",  "mm",  "#60ff60"),
    ]
    lines_m = []
    for ax, (ttl, unit, col) in zip(axes_m, spec_motion):
        _style_ax(ax, ttl, unit, col)
        ln, = ax.plot([], [], color=col, linewidth=1.5)
        lines_m.append(ln)

    # Capture axes positions after draw so overlay axes can reuse them
    fig.canvas.draw()
    positions = [ax.get_position() for ax in axes_m]

    # ── Factory: create 3 overlay axes (one content tab) ─────────────────────
    def _make_tab_axes(specs):
        """
        specs: list of 3 dicts, each:
          title, ylabel, col_L, lbl_L, [col_R, lbl_R]   (col_R optional)
        Returns (axes_list, lines_L_list, lines_R_list)
        """
        axs, lnLs, lnRs = [], [], []
        for pos, sp in zip(positions, specs):
            ax = fig.add_axes(pos)
            _style_ax(ax, sp['title'], sp['ylabel'], sp['col_L'])
            lnL, = ax.plot([], [], color=sp['col_L'], linewidth=1.5, label=sp['lbl_L'])
            lnR = None
            if sp.get('col_R'):
                lnR, = ax.plot([], [], color=sp['col_R'], linewidth=1.2,
                               linestyle="--", label=sp['lbl_R'])
            ax.legend(loc="upper right", fontsize=7,
                      facecolor="#2a2a3e", edgecolor="#555", labelcolor="white")
            ax.set_visible(False)
            axs.append(ax); lnLs.append(lnL); lnRs.append(lnR)
        return axs, lnLs, lnRs

    # Tab 1 — Femur
    axes_f, lines_fL, lines_fR = _make_tab_axes([
        dict(title="Femur Axial Force",       ylabel="N", col_L="#60d0ff", lbl_L="Axial",
             col_R="#60ff60", lbl_R="GRF L"),
        dict(title="Femur Lateral → Bending", ylabel="N", col_L="#f08040", lbl_L="Lateral"),
        dict(title="Bearing A  (hip)",         ylabel="N", col_L="#e060ff", lbl_L="Bearing A"),
    ])

    # Tab 2 — Tibia / Coupler
    axes_tc, lines_tcL, lines_tcR = _make_tab_axes([
        dict(title="Tibia Axial + Coupler Axial", ylabel="N", col_L="#60d0ff", lbl_L="Tibia ax",
             col_R="#f0d040", lbl_R="Coupler ax"),
        dict(title="Tibia Lateral → Bending",     ylabel="N", col_L="#f08040", lbl_L="Tib lateral"),
        dict(title="Bearing C (knee) + E (4-bar)", ylabel="N", col_L="#e060ff", lbl_L="Bearing C",
             col_R="#60ffff", lbl_R="Bearing E"),
    ])

    # Tab 3 — Bearings overview
    axes_b, lines_bL, lines_bR = _make_tab_axes([
        dict(title="Bearings A (hip) + W (wheel)", ylabel="N", col_L="#ff6060", lbl_L="A (hip)",
             col_R="#60d0ff", lbl_R="W (wheel)"),
        dict(title="Bearings C (knee) + E (4-bar)", ylabel="N", col_L="#e060ff", lbl_L="C (knee)",
             col_R="#60ffff", lbl_R="E (4-bar)"),
        dict(title="Bearing F (coupler) + GRF L",  ylabel="N", col_L="#f0d040", lbl_L="F (coupler)",
             col_R="#60ff60", lbl_R="GRF L"),
    ])

    # ── Peak table (shown on all non-Motion tabs) ─────────────────────────────
    # Positioned between subplots (bottom≈0.42) and drive buttons (top≈0.265)
    ax_tbl = fig.add_axes([0.03, 0.255, 0.94, 0.148])
    ax_tbl.set_facecolor("#161626")
    ax_tbl.axis("off")
    ax_tbl.set_visible(False)

    _TBL_INIT = [
        ["Force Signal",    "Peak [N]", "Bearing",        "Peak [N]"],
        ["Femur axial",     "—",        "A  (hip)",        "—"],
        ["Femur lateral",   "—",        "C  (knee)",       "—"],
        ["Tibia axial",     "—",        "E  (4-bar)",      "—"],
        ["Tibia lateral",   "—",        "F  (coupler)",    "—"],
        ["Coupler axial",   "—",        "W  (wheel)",      "—"],
        ["GRF L (wheel)",   "—",        "",                ""],
    ]
    _tbl_obj = ax_tbl.table(
        cellText=_TBL_INIT, loc="center", cellLoc="left", edges="horizontal")
    _tbl_obj.auto_set_font_size(False)
    _tbl_obj.set_fontsize(6.5)
    _tbl_obj.scale(1, 1.15)
    for col in range(4):
        _tbl_obj[0, col].set_facecolor("#2a2a4e")
        _tbl_obj[0, col].get_text().set_color("lightgray")
        _tbl_obj[0, col].get_text().set_fontweight("bold")
    for row in range(1, len(_TBL_INIT)):
        for col in range(4):
            _tbl_obj[row, col].set_facecolor("#161626")
            _tbl_obj[row, col].get_text().set_color("white")
    for row in range(1, len(_TBL_INIT)):
        for col in (1, 3):
            _tbl_obj[row, col].get_text().set_ha("right")

    def _update_table():
        def _fmt(k):
            return f"{max_abs[k]:.1f}" if max_abs[k] > 0 else "—"
        _tbl_obj[1, 1].get_text().set_text(_fmt('fax_fem'))
        _tbl_obj[2, 1].get_text().set_text(_fmt('flat_fem'))
        _tbl_obj[3, 1].get_text().set_text(_fmt('fax_tib'))
        _tbl_obj[4, 1].get_text().set_text(_fmt('flat_tib'))
        _tbl_obj[5, 1].get_text().set_text(_fmt('fax_cpl'))
        _tbl_obj[6, 1].get_text().set_text(_fmt('grf_L'))
        _tbl_obj[1, 3].get_text().set_text(_fmt('fbear_A'))
        _tbl_obj[2, 3].get_text().set_text(_fmt('fbear_C'))
        _tbl_obj[3, 3].get_text().set_text(_fmt('fbear_E'))
        _tbl_obj[4, 3].get_text().set_text(_fmt('fbear_F'))
        _tbl_obj[5, 3].get_text().set_text(_fmt('fbear_W'))

    # ── Tab state + 4-button toggle ───────────────────────────────────────────
    _tab = [0]   # 0=Motion  1=Femur  2=Tib/Cpl  3=Bearings

    def _set_tab(t):
        _tab[0] = t
        for ax in axes_m:   ax.set_visible(t == 0)
        for ax in axes_f:   ax.set_visible(t == 1)
        for ax in axes_tc:  ax.set_visible(t == 2)
        for ax in axes_b:   ax.set_visible(t == 3)
        ax_tbl.set_visible(t != 0)
        for i, btn in enumerate([btn_tab_m, btn_tab_f, btn_tab_tc, btn_tab_b]):
            btn.color = "#3a3a9e" if i == t else BTN_COL
        fig.canvas.draw_idle()

    tab_bw = 0.213; tab_gap = 0.013
    tab_xs = [0.05 + i * (tab_bw + tab_gap) for i in range(4)]
    ax_tab_m  = fig.add_axes([tab_xs[0], 0.928, tab_bw, 0.040])
    ax_tab_f  = fig.add_axes([tab_xs[1], 0.928, tab_bw, 0.040])
    ax_tab_tc = fig.add_axes([tab_xs[2], 0.928, tab_bw, 0.040])
    ax_tab_b  = fig.add_axes([tab_xs[3], 0.928, tab_bw, 0.040])

    btn_tab_m  = Button(ax_tab_m,  "Motion",   color="#3a3a9e", hovercolor=BTN_HOV)
    btn_tab_f  = Button(ax_tab_f,  "Femur",    color=BTN_COL,   hovercolor=BTN_HOV)
    btn_tab_tc = Button(ax_tab_tc, "Tib/Cpl",  color=BTN_COL,   hovercolor=BTN_HOV)
    btn_tab_b  = Button(ax_tab_b,  "Bearings", color=BTN_COL,   hovercolor=BTN_HOV)
    for btn in [btn_tab_m, btn_tab_f, btn_tab_tc, btn_tab_b]:
        btn.label.set_color("white"); btn.label.set_fontsize(8)
    btn_tab_m.on_clicked( lambda _: _set_tab(0))
    btn_tab_f.on_clicked( lambda _: _set_tab(1))
    btn_tab_tc.on_clicked(lambda _: _set_tab(2))
    btn_tab_b.on_clicked( lambda _: _set_tab(3))

    # ── Drive / action buttons ────────────────────────────────────────────────
    ax_fwd  = fig.add_axes([0.05,  0.210, 0.42, 0.042])
    ax_bwd  = fig.add_axes([0.53,  0.210, 0.42, 0.042])

    bw, gap = 0.197, 0.017
    bx = [0.05 + i * (bw + gap) for i in range(4)]
    ax_rst  = fig.add_axes([bx[0], 0.158, bw, 0.042])
    ax_neu  = fig.add_axes([bx[1], 0.158, bw, 0.042])
    ax_crch = fig.add_axes([bx[2], 0.158, bw, 0.042])
    ax_jmp  = fig.add_axes([bx[3], 0.158, bw, 0.042])

    # ── Joystick gizmo ────────────────────────────────────────────────────────
    _GZ_LABELS = ['Drive (fwd ▶)', 'Turn (right ▶)', 'Hip (crouch ▶)']
    _GZ_COLORS = ['#60ff60',        '#f08040',         '#60d0ff']
    _gizmo_dots = []; _gizmo_bars = []
    for i, (lbl, col) in enumerate(zip(_GZ_LABELS, _GZ_COLORS)):
        ax_g = fig.add_axes([0.07 + i * 0.31, 0.067, 0.26, 0.070])
        ax_g.set_facecolor('#161626'); ax_g.set_xlim(-1.15, 1.15)
        ax_g.set_ylim(-0.5, 0.5); ax_g.axis('off')
        ax_g.set_title(lbl, color=col, fontsize=7, pad=2)
        ax_g.plot([-1, 1], [0, 0], color='#444', lw=3, solid_capstyle='round', zorder=0)
        ax_g.plot([0, 0], [-0.25, 0.25], color='#333', lw=1, zorder=1)
        bar, = ax_g.plot([0, 0], [0, 0], color=col, lw=4, solid_capstyle='round', zorder=2)
        dot, = ax_g.plot([0], [0], 'o', color=col, ms=9, zorder=3)
        _gizmo_dots.append(dot); _gizmo_bars.append(bar)

    btn_fwd  = Button(ax_fwd,  "Drive Fwd >>", color=BTN_COL,    hovercolor=BTN_HOV)
    btn_bwd  = Button(ax_bwd,  "<< Drive Bwd", color=BTN_COL,    hovercolor=BTN_HOV)
    btn_rst  = Button(ax_rst,  "Restart",      color=BTN_COL,    hovercolor=BTN_HOV)
    btn_neu  = Button(ax_neu,  "Neutral",      color=BTN_COL,    hovercolor=BTN_HOV)
    btn_crch = Button(ax_crch, "Crouch",       color=BTN_COL,    hovercolor=BTN_HOV)
    btn_jmp  = Button(ax_jmp,  "Jump",         color="#5a1a1a",   hovercolor="#9a2a2a")

    for b in [btn_fwd, btn_bwd, btn_rst, btn_neu, btn_crch, btn_jmp]:
        b.label.set_color("white"); b.label.set_fontsize(9)

    driving = [0]

    def _send(msg):
        try: cmd_q.put_nowait(msg)
        except Exception: pass

    def on_press(event):
        if event.inaxes == ax_fwd:
            driving[0] = 1;  ax_fwd.set_facecolor(BTN_ACTV); fig.canvas.draw_idle()
            _send("DRIVE_FWD_ON")
        elif event.inaxes == ax_bwd:
            driving[0] = -1; ax_bwd.set_facecolor(BTN_ACTV); fig.canvas.draw_idle()
            _send("DRIVE_BWD_ON")

    def on_release(event):
        if driving[0] == 1:
            ax_fwd.set_facecolor(BTN_COL); fig.canvas.draw_idle(); _send("DRIVE_FWD_OFF")
        elif driving[0] == -1:
            ax_bwd.set_facecolor(BTN_COL); fig.canvas.draw_idle(); _send("DRIVE_BWD_OFF")
        driving[0] = 0

    fig.canvas.mpl_connect('button_press_event',   on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)

    btn_rst.on_clicked( lambda _: _send("RESTART"))
    btn_neu.on_clicked( lambda _: _send("NEUTRAL"))
    btn_crch.on_clicked(lambda _: _send("CROUCH"))
    btn_jmp.on_clicked( lambda _: _send("JUMP"))

    def on_key_press(event):
        k = event.key
        if   k == 'up':    driving[0] = 1;  ax_fwd.set_facecolor(BTN_ACTV); fig.canvas.draw_idle(); _send("DRIVE_FWD_ON")
        elif k == 'down':  driving[0] = -1; ax_bwd.set_facecolor(BTN_ACTV); fig.canvas.draw_idle(); _send("DRIVE_BWD_ON")
        elif k == 'left':  _send("TURN_LEFT_ON")
        elif k == 'right': _send("TURN_RIGHT_ON")

    def on_key_release(event):
        k = event.key
        if k in ('up', 'down'):
            ax_fwd.set_facecolor(BTN_COL) if driving[0] == 1 else ax_bwd.set_facecolor(BTN_COL)
            fig.canvas.draw_idle()
            _send("DRIVE_FWD_OFF" if driving[0] == 1 else "DRIVE_BWD_OFF")
            driving[0] = 0
        elif k in ('left', 'right'):
            _send("TURN_OFF")

    fig.canvas.mpl_connect('key_press_event',   on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    fig.text(0.5, 0.018,
             "↑ / ↓ drive   ← / → turn   (click this window first)",
             ha='center', color='#888888', fontsize=7)

    fig.suptitle(
        f"Baseline-1 (2-leg)  |  L_f={ROBOT['L_femur']*1000:.0f}mm  "
        f"Lc={ROBOT['Lc']*1000:.0f}mm  "
        f"F_X={ROBOT['F_X']*1000:.0f}mm",
        color="white", fontsize=9)
    fig.show()

    all_bufs = [t_buf, pitch_buf, torque_buf, height_buf,
                fax_fem_buf, flat_fem_buf, fax_tib_buf, flat_tib_buf, fax_cpl_buf,
                fbear_A_buf, fbear_C_buf, fbear_E_buf, fbear_F_buf, fbear_W_buf,
                grf_L_buf]

    # ── Main update loop ──────────────────────────────────────────────────────
    while plt.fignum_exists(fig.number):
        items = []
        while True:
            try: items.append(q.get_nowait())
            except Exception: break
        if not items:
            plt.pause(1.0 / 60)
            continue

        for item in items:
            if item is None:
                _log_file.close()
                return
            if item == "RESET":
                for buf in all_bufs: buf.clear()
                for ln in lines_m: ln.set_data([], [])
                for lnLs, lnRs in [(lines_fL, lines_fR),
                                    (lines_tcL, lines_tcR),
                                    (lines_bL, lines_bR)]:
                    for lnL, lnR in zip(lnLs, lnRs):
                        lnL.set_data([], [])
                        if lnR is not None: lnR.set_data([], [])
                for k in max_abs: max_abs[k] = 0.0
                ax_fwd.set_facecolor(BTN_COL); ax_bwd.set_facecolor(BTN_COL)
                driving[0] = 0
                fig.canvas.flush_events()
                continue

            (tt, pt, trq_wL, h,
             joy_y, joy_z, joy_t,
             fax_f, flat_f,
             fax_t, flat_t,
             fax_c,
             fbear_a, fbear_c, fbear_e, fbear_f, fbear_w,
             grf_l) = item

            t_buf.append(tt);       pitch_buf.append(pt)
            torque_buf.append(trq_wL); height_buf.append(h)
            fax_fem_buf.append(fax_f);   flat_fem_buf.append(flat_f)
            fax_tib_buf.append(fax_t);   flat_tib_buf.append(flat_t)
            fax_cpl_buf.append(fax_c)
            fbear_A_buf.append(fbear_a); fbear_C_buf.append(fbear_c)
            fbear_E_buf.append(fbear_e); fbear_F_buf.append(fbear_f)
            fbear_W_buf.append(fbear_w); grf_L_buf.append(grf_l)

            # Update peak values
            for k, v in [('fax_fem', fax_f), ('flat_fem', flat_f),
                         ('fax_tib', fax_t), ('flat_tib', flat_t),
                         ('fax_cpl', fax_c),
                         ('fbear_A', fbear_a), ('fbear_C', fbear_c),
                         ('fbear_E', fbear_e), ('fbear_F', fbear_f),
                         ('fbear_W', fbear_w), ('grf_L',   grf_l)]:
                if abs(v) > max_abs[k]:
                    max_abs[k] = abs(v)

            # Update joystick gizmo
            for dot, bar, v in zip(_gizmo_dots, _gizmo_bars,
                                   [-joy_y, -joy_z, joy_t]):
                v = float(max(-1.0, min(1.0, v)))
                dot.set_xdata([v]); bar.set_xdata([0.0, v])

            # CSV log
            _log_writer.writerow([
                f"{tt:.4f}", f"{pt:.3f}", f"{h:.1f}",
                f"{fax_f:.2f}", f"{flat_f:.2f}",
                f"{fax_t:.2f}", f"{flat_t:.2f}",
                f"{fax_c:.2f}",
                f"{fbear_a:.2f}", f"{fbear_c:.2f}",
                f"{fbear_e:.2f}", f"{fbear_f:.2f}", f"{fbear_w:.2f}",
                f"{grf_l:.2f}",
            ])

        if len(t_buf) < 2:
            continue

        tb, sim_t = list(t_buf), t_buf[-1]
        t0  = max(0.0, sim_t - window_s)
        idx = next((i for i, tt in enumerate(tb) if tt >= t0), 0)
        tw  = tb[idx:]

        def _window(buf):
            return list(buf)[idx:]

        def _autoscale(ax, *bufs, min_span=0.5):
            all_vals = []
            for b in bufs:
                all_vals.extend(b)
            if len(all_vals) < 2:
                return
            lo, hi = min(all_vals), max(all_vals)
            span = max(hi - lo, min_span)
            ax.set_xlim(t0, sim_t + 0.5)
            ax.set_ylim(lo - span * 0.15, hi + span * 0.15)

        tab = _tab[0]

        if tab == 0:
            # Motion tab
            for ln, ax, buf, ms in zip(lines_m, axes_m,
                                       [pitch_buf, torque_buf, height_buf],
                                       [0.05, 0.05, 10]):
                d = _window(buf)
                ln.set_data(tw, d)
                _autoscale(ax, d, min_span=ms)

        elif tab == 1:
            # Femur tab
            panel_data = [
                (fax_fem_buf,  grf_L_buf),
                (flat_fem_buf, None),
                (fbear_A_buf,  None),
            ]
            for (bL, bR), lnL, lnR, ax in zip(panel_data, lines_fL, lines_fR, axes_f):
                dL = _window(bL)
                lnL.set_data(tw, dL)
                dR = _window(bR) if bR is not None else []
                if lnR is not None: lnR.set_data(tw, dR)
                _autoscale(ax, dL, dR)
            _update_table()

        elif tab == 2:
            # Tibia / Coupler tab
            panel_data = [
                (fax_tib_buf,  fax_cpl_buf),
                (flat_tib_buf, None),
                (fbear_C_buf,  fbear_E_buf),
            ]
            for (bL, bR), lnL, lnR, ax in zip(panel_data, lines_tcL, lines_tcR, axes_tc):
                dL = _window(bL)
                lnL.set_data(tw, dL)
                dR = _window(bR) if bR is not None else []
                if lnR is not None: lnR.set_data(tw, dR)
                _autoscale(ax, dL, dR)
            _update_table()

        else:
            # Bearings tab
            panel_data = [
                (fbear_A_buf, fbear_W_buf),
                (fbear_C_buf, fbear_E_buf),
                (fbear_F_buf, grf_L_buf),
            ]
            for (bL, bR), lnL, lnR, ax in zip(panel_data, lines_bL, lines_bR, axes_b):
                dL = _window(bL)
                lnL.set_data(tw, dL)
                dR = _window(bR) if bR is not None else []
                if lnR is not None: lnR.set_data(tw, dR)
                _autoscale(ax, dL, dR)
            _update_table()

        fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------
def run_viewer():
    p   = ROBOT
    xml = build_xml(p)
    model = mujoco.MjModel.from_xml_string(xml, assets=build_assets())
    data  = mujoco.MjData(model)

    # Fix 4-bar equality constraint anchors for both legs
    for eq_name in ["4bar_close_L", "4bar_close_R"]:
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
        model.eq_data[eq_id, 0:3] = [-p['Lc'], 0.0, 0.0]
        model.eq_data[eq_id, 3:6] = [0.0, 0.0, p['L_stub']]

    # ── Joystick ──────────────────────────────────────────────────────────────
    _joy = None
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
            print("Joystick: none detected — keyboard/mouse only")
    JOY_DEADZONE = 0.08

    def jqp(n): return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def jdof(n): return model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def bid(n):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    def gid(n):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)

    # ── Body / geom IDs ───────────────────────────────────────────────────────
    femur_bid_L   = bid("femur_L")
    tibia_bid_L   = bid("tibia_L")
    coupler_bid_L = bid("coupler_L")
    wheel_bid_L   = bid("wheel_asm_L")
    wheel_bid_R   = bid("wheel_asm_R")
    box_bid       = bid("box")
    tire_gid_L    = gid("wheel_tire_geom_L")

    # Equality constraint ID and type constant for 4-bar E force extraction
    eq_id_L   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "4bar_close_L")
    _EQ_TYPE  = mujoco.mjtConstraint.mjCNSTR_EQUALITY

    s_root   = jqp("root_free")
    s_hip_L  = jqp("hip_L");         s_hip_R  = jqp("hip_R")
    s_hF_L   = jqp("hinge_F_L");     s_hF_R   = jqp("hinge_F_R")
    s_knee_L = jqp("knee_joint_L");  s_knee_R = jqp("knee_joint_R")
    d_root   = jdof("root_free")
    d_pitch  = d_root + 4
    d_hip_L  = jdof("hip_L");        d_hip_R  = jdof("hip_R")
    d_whl_L  = jdof("wheel_spin_L"); d_whl_R  = jdof("wheel_spin_R")

    # Pre-allocate reusable array for contact force queries
    _f6 = np.zeros(6)

    # ── Full structural computation for left leg ──────────────────────────────
    def _struct_all_L():
        """
        Return 11-tuple:
          (fax_fem, flat_fem,
           fax_tib, flat_tib,
           fax_cpl,
           fbear_A, fbear_C, fbear_E, fbear_F, fbear_W,
           grf_L)
        All values in [N].
        """
        # ── Femur (A→C) ───────────────────────────────────────────────────────
        cfrc_fem = data.cfrc_int[femur_bid_L]
        dv = data.xpos[tibia_bid_L] - data.xpos[femur_bid_L]
        n  = math.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
        ax_fem = dv / n if n > 1e-6 else np.array([1., 0., 0.])
        F_fem  = cfrc_fem[3:6]
        fax_f  = float(np.dot(F_fem, ax_fem))
        flat_f = float(math.sqrt(max(0., float(np.dot(F_fem, F_fem)) - fax_f**2)))
        # Bearing A radial: rotation axis ≈ Y → radial = √(Fx²+Fz²)
        fbear_A = float(math.sqrt(F_fem[0]**2 + F_fem[2]**2))

        # ── GRF (left wheel contact) ──────────────────────────────────────────
        grf = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 == tire_gid_L or c.geom2 == tire_gid_L:
                mujoco.mj_contactForce(model, data, i, _f6)
                grf += _f6[0]

        # ── E force vector from 4-bar equality constraint ─────────────────────
        F_E_vec = np.zeros(3)
        k = 0
        for i in range(data.nefc):
            if data.efc_type[i] == _EQ_TYPE and data.efc_id[i] == eq_id_L:
                if k < 3:
                    F_E_vec[k] = data.efc_force[i]
                k += 1
        fbear_E = float(np.linalg.norm(F_E_vec))
        fbear_F = fbear_E   # two-force coupler: |F_F| ≈ |F_E|

        # ── Tibia (C→W) ───────────────────────────────────────────────────────
        cfrc_tib = data.cfrc_int[tibia_bid_L]
        F_C_vec  = cfrc_tib[3:6] - F_E_vec
        fbear_C  = float(np.linalg.norm(F_C_vec))
        dv_t = data.xpos[wheel_bid_L] - data.xpos[tibia_bid_L]
        n_t  = math.sqrt(dv_t[0]**2 + dv_t[1]**2 + dv_t[2]**2)
        ax_t = dv_t / n_t if n_t > 1e-6 else np.array([0., 0., -1.])
        fax_t  = float(np.dot(cfrc_tib[3:6], ax_t))
        flat_t = float(math.sqrt(max(0., float(np.dot(cfrc_tib[3:6], cfrc_tib[3:6])) - fax_t**2)))

        # ── Coupler (F→E) axial ───────────────────────────────────────────────
        cfrc_cpl = data.cfrc_int[coupler_bid_L]
        R_cpl    = data.xmat[coupler_bid_L].reshape(3, 3)
        ax_cpl   = R_cpl @ np.array([-1., 0., 0.])
        fax_c    = float(np.dot(cfrc_cpl[3:6], ax_cpl))

        # ── Bearing W (wheel axle) — radial only ──────────────────────────────
        cfrc_whl = data.cfrc_int[wheel_bid_L]
        fbear_W  = float(math.sqrt(cfrc_whl[3]**2 + cfrc_whl[5]**2))

        return (fax_f, flat_f,
                fax_t, flat_t,
                fax_c,
                fbear_A, fbear_C, fbear_E, fbear_F, fbear_W,
                float(grf))

    def _init():
        mujoco.mj_resetData(model, data)
        ik = solve_ik(Q_NEUTRAL, p)
        if not ik: raise RuntimeError("IK failed at Q_NEUTRAL")
        for s_hF, s_hip, s_knee in [
            (s_hF_L, s_hip_L, s_knee_L),
            (s_hF_R, s_hip_R, s_knee_R),
        ]:
            data.qpos[s_hF]   = ik['q_coupler_F']
            data.qpos[s_hip]  = ik['q_hip']
            data.qpos[s_knee] = ik['q_knee']
        mujoco.mj_forward(model, data)
        wz = (data.xpos[wheel_bid_L][2] + data.xpos[wheel_bid_R][2]) / 2.0
        data.qpos[s_root + 2] += WHEEL_R - wz
        theta = get_equilibrium_pitch(p, Q_NEUTRAL)
        data.qpos[s_root + 3] = math.cos(theta / 2)
        data.qpos[s_root + 4] = 0.0
        data.qpos[s_root + 5] = math.sin(theta / 2)
        data.qpos[s_root + 6] = 0.0
        mujoco.mj_forward(model, data)

    _init()

    PHYSICS_HZ      = round(1.0 / model.opt.timestep)
    steps_per_frame = max(1, int(PHYSICS_HZ / RENDER_HZ))
    last_push_wall  = -1.0
    prev_sim_t      = 0.0

    plot_q    = mp.Queue(maxsize=4000)
    cmd_q     = mp.Queue(maxsize=64)
    plot_proc = mp.Process(
        target=_plot_process, args=(plot_q, cmd_q, WINDOW_S), daemon=True)
    plot_proc.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 35
        viewer.cam.elevation = -15
        viewer.cam.distance  = 2.5
        viewer.cam.lookat    = np.array([0.0, 0.0, 0.30])

        pitch_integral       = 0.0
        odo_x                = 0.0
        drive_pitch_offset   = 0.0
        drive_pitch_target   = 0.0
        turn_offset          = 0.0
        grounded             = True
        leg_state            = STATE_NEUTRAL
        current_hip_target   = Q_NEUTRAL
        jump_triggered       = False
        jump_start_t         = 0.0
        crouch_start_t       = 0.0
        neutral_ramp_start_t = -999.0
        neutral_ramp_start_q = Q_NEUTRAL
        was_airborne         = False
        land_t               = -999.0
        prev_grounded        = True
        slowmo_active        = False
        max_height_m         = 0.0
        joy_prev_trigger     = False
        _joy_drive_active    = False
        _joy_turn_active     = False
        _throttle_moved      = False
        _joy_axes = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        def _reset_state():
            nonlocal pitch_integral, odo_x, drive_pitch_offset, drive_pitch_target, turn_offset
            nonlocal grounded, leg_state, current_hip_target, joy_prev_trigger
            nonlocal jump_triggered, jump_start_t, crouch_start_t
            nonlocal neutral_ramp_start_t, neutral_ramp_start_q
            nonlocal was_airborne, land_t, prev_grounded, slowmo_active, max_height_m
            nonlocal _throttle_moved
            _init()
            pitch_integral = 0.0; odo_x = 0.0
            drive_pitch_offset = 0.0; drive_pitch_target = 0.0; turn_offset = 0.0
            grounded = True
            leg_state = STATE_NEUTRAL; current_hip_target = Q_NEUTRAL
            jump_triggered = False; jump_start_t = 0.0; crouch_start_t = 0.0
            neutral_ramp_start_t = -999.0; neutral_ramp_start_q = Q_NEUTRAL
            was_airborne = False; land_t = -999.0; prev_grounded = True
            slowmo_active = False; max_height_m = 0.0; joy_prev_trigger = False
            _throttle_moved = False
            if not plot_q.full(): plot_q.put_nowait("RESET")

        while viewer.is_running():
            frame_start = time.perf_counter()
            sim_t = float(data.time)

            # Drain command queue
            while True:
                try:
                    cmd = cmd_q.get_nowait()
                    if cmd == "RESTART":
                        _reset_state()
                    elif cmd == "DRIVE_FWD_ON":
                        drive_pitch_target =  DRIVE_PITCH_OFFSET; odo_x = 0.0
                    elif cmd == "DRIVE_FWD_OFF":
                        drive_pitch_target = 0.0
                    elif cmd == "DRIVE_BWD_ON":
                        drive_pitch_target = -DRIVE_PITCH_OFFSET; odo_x = 0.0
                    elif cmd == "DRIVE_BWD_OFF":
                        drive_pitch_target = 0.0
                    elif cmd == "TURN_LEFT_ON":
                        turn_offset =  TURN_TORQUE
                    elif cmd == "TURN_RIGHT_ON":
                        turn_offset = -TURN_TORQUE
                    elif cmd == "TURN_OFF":
                        turn_offset = 0.0
                    elif cmd == "NEUTRAL":
                        neutral_ramp_start_t = sim_t
                        neutral_ramp_start_q = data.qpos[s_hip_L]
                        leg_state = STATE_NEUTRAL
                        jump_triggered = False; was_airborne = False; land_t = -999.0
                    elif cmd == "CROUCH":
                        if leg_state == STATE_NEUTRAL:
                            leg_state = STATE_CROUCH; crouch_start_t = sim_t
                    elif cmd == "JUMP":
                        if leg_state in (STATE_NEUTRAL, STATE_CROUCH) and grounded:
                            leg_state = STATE_JUMP; jump_start_t = sim_t
                            jump_triggered = True; was_airborne = False; land_t = -999.0
                except Exception:
                    break

            if sim_t < prev_sim_t - 0.01:
                _reset_state()
            prev_sim_t = sim_t

            # ── Joystick poll ─────────────────────────────────────────────────
            _joy_raw_y, _joy_raw_z, _joy_raw_t = 0.0, 0.0, 0.0
            if _joy is not None:
                for _ev in pygame.event.get():
                    if _ev.type == pygame.JOYAXISMOTION:
                        _joy_axes[_ev.axis] = _ev.value
                        if _ev.axis == 3:
                            _throttle_moved = True
                    elif _ev.type == pygame.JOYBUTTONDOWN and _ev.button == 0:
                        if leg_state in (STATE_NEUTRAL, STATE_CROUCH) and grounded:
                            leg_state = STATE_JUMP; jump_start_t = sim_t
                            jump_triggered = True; was_airborne = False; land_t = -999.0
                    elif _ev.type == pygame.JOYBUTTONUP and _ev.button == 0:
                        joy_prev_trigger = False

                _joy_raw_y = _joy_axes.get(1, 0.0)
                _joy_raw_z = _joy_axes.get(2, 0.0)
                _joy_raw_t = _joy_axes.get(3, 0.0)
                _ey = _joy_raw_y if abs(_joy_raw_y) > JOY_DEADZONE else 0.0
                _ez = _joy_raw_z if abs(_joy_raw_z) > JOY_DEADZONE else 0.0
                if _ey != 0.0:
                    drive_pitch_target = -_ey * DRIVE_PITCH_OFFSET; odo_x = 0.0
                    _joy_drive_active = True
                elif _joy_drive_active:
                    drive_pitch_target = 0.0; _joy_drive_active = False
                if _ez != 0.0:
                    turn_offset = -_ez * TURN_TORQUE; _joy_turn_active = True
                elif _joy_turn_active:
                    turn_offset = 0.0; _joy_turn_active = False

            # Slowmo
            if leg_state == STATE_JUMP and not slowmo_active:
                slowmo_active = True
            if slowmo_active and land_t > jump_start_t and sim_t > land_t + 0.5:
                slowmo_active = False
            slow_now = slowmo_active
            n_steps  = max(1, steps_per_frame // JUMP_SLOWMO) if slow_now else steps_per_frame

            for _ in range(n_steps):
                sim_t = data.time

                q_quat = data.xquat[box_bid]
                pitch_true      = math.asin(max(-1.0, min(1.0,
                    2 * (q_quat[0]*q_quat[2] - q_quat[3]*q_quat[1]))))
                pitch_rate_true = data.qvel[d_pitch]
                pitch      = pitch_true      + np.random.normal(0, PITCH_NOISE_STD_RAD)
                pitch_rate = pitch_rate_true + np.random.normal(0, PITCH_RATE_NOISE_STD_RAD_S)

                hip_q     = (data.qpos[s_hip_L] + data.qpos[s_hip_R]) / 2.0
                hip_omega = (data.qvel[d_hip_L]  + data.qvel[d_hip_R])  / 2.0
                wheel_vel = (data.qvel[d_whl_L]  + data.qvel[d_whl_R])  / 2.0

                accel_mag = np.linalg.norm(
                    data.sensor("accel").data + np.random.normal(0, ACCEL_NOISE_STD, 3))
                if   accel_mag < 3.0: grounded = False
                elif accel_mag > 7.0: grounded = True

                if not grounded: was_airborne = True
                just_landed = was_airborne and grounded and land_t < jump_start_t
                if just_landed:
                    land_t = sim_t; odo_x = 0.0; pitch_integral = 0.0

                _ramp_rate = DRIVE_PITCH_OFFSET / DRIVE_RAMP_S
                _dt = model.opt.timestep
                drive_pitch_offset += np.clip(
                    drive_pitch_target - drive_pitch_offset,
                    -_ramp_rate * _dt, _ramp_rate * _dt)

                pitch_ff = get_equilibrium_pitch(p, hip_q)
                if grounded:
                    vel_est = (wheel_vel + pitch_rate) * WHEEL_R
                    if abs(drive_pitch_target) < 1e-6:
                        odo_x += vel_est * model.opt.timestep
                    else:
                        odo_x = 0.0
                    pitch_fb = np.clip(
                        -(POSITION_KP * odo_x + VELOCITY_KP * vel_est),
                        -MAX_PITCH_CMD, MAX_PITCH_CMD)
                    target_pitch = pitch_ff + pitch_fb + drive_pitch_offset
                else:
                    target_pitch = 0.0; pitch_integral = 0.0

                pitch_error    = pitch - target_pitch
                pitch_integral = np.clip(
                    pitch_integral + pitch_error * model.opt.timestep, -1.0, 1.0)
                kp    = PITCH_KP * (2.0 if not grounded else 1.0)
                kd    = PITCH_KD * (2.0 if not grounded else 1.0)
                u_bal = kp * pitch_error + PITCH_KI * pitch_integral + kd * pitch_rate
                data.ctrl[2] = np.clip(u_bal - turn_offset, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
                data.ctrl[3] = np.clip(u_bal + turn_offset, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)

                if leg_state == STATE_JUMP:
                    ramp_in     = min(1.0, (sim_t - jump_start_t) / JUMP_RAMP_S)
                    ramp_out    = min(1.0, max(0.0, (hip_q - Q_EXT) / JUMP_RAMPDOWN))
                    speed_scale = max(0.0, 1.0 - abs(hip_omega) / OMEGA_MAX)
                    u_hip = -HIP_TORQUE_LIMIT * ramp_in * ramp_out * speed_scale
                    if (hip_q <= Q_EXT + 0.05) or (
                            not grounded and (sim_t - jump_start_t) > 0.05):
                        leg_state = STATE_NEUTRAL; current_hip_target = Q_NEUTRAL
                    data.ctrl[0] = np.clip(u_hip, -HIP_TORQUE_LIMIT, HIP_TORQUE_LIMIT)
                    data.ctrl[1] = np.clip(u_hip, -HIP_TORQUE_LIMIT, HIP_TORQUE_LIMIT)
                else:
                    if leg_state == STATE_CROUCH:
                        frac = min(1.0, (sim_t - crouch_start_t) / CROUCH_DURATION_V)
                        current_hip_target = Q_NEUTRAL + frac * (Q_RET - Q_NEUTRAL)
                    elif leg_state == STATE_NEUTRAL:
                        ramp_elapsed = sim_t - neutral_ramp_start_t
                        if ramp_elapsed < NEUTRAL_RAMP_S:
                            frac = ramp_elapsed / NEUTRAL_RAMP_S
                            current_hip_target = (neutral_ramp_start_q
                                + frac * (Q_NEUTRAL - neutral_ramp_start_q))
                        else:
                            current_hip_target = Q_NEUTRAL
                    if _joy is not None and _throttle_moved:
                        t_norm = (_joy_raw_t + 1.0) / 2.0
                        current_hip_target = Q_EXT + t_norm * (Q_RET - Q_EXT)
                    data.ctrl[0] = np.clip(
                        HIP_KP_SUSP * (current_hip_target - data.qpos[s_hip_L])
                        - HIP_KD_SUSP * data.qvel[d_hip_L],
                        -HIP_TORQUE_LIMIT, HIP_TORQUE_LIMIT)
                    data.ctrl[1] = np.clip(
                        HIP_KP_SUSP * (current_hip_target - data.qpos[s_hip_R])
                        - HIP_KD_SUSP * data.qvel[d_hip_R],
                        -HIP_TORQUE_LIMIT, HIP_TORQUE_LIMIT)

                mujoco.mj_step(model, data)

            viewer.sync()

            # ── Telemetry push ────────────────────────────────────────────────
            wall_now = time.perf_counter()
            if wall_now - last_push_wall >= (1.0 / PUSH_HZ) and not plot_q.full():
                h_mm = max(0.0, (data.xpos[wheel_bid_L][2] - WHEEL_R) * 1000.0)
                (fax_f, flat_f,
                 fax_t, flat_t,
                 fax_c,
                 fbear_a, fbear_c, fbear_e, fbear_f, fbear_w,
                 grf_l) = _struct_all_L()
                plot_q.put_nowait((
                    sim_t,
                    math.degrees(pitch_true),
                    data.ctrl[2],
                    h_mm,
                    _joy_raw_y, _joy_raw_z, _joy_raw_t,
                    fax_f,   flat_f,
                    fax_t,   flat_t,
                    fax_c,
                    fbear_a, fbear_c, fbear_e, fbear_f, fbear_w,
                    grf_l,
                ))
                last_push_wall = wall_now

            # ── Scene overlays ────────────────────────────────────────────────
            viewer.user_scn.ngeom = 0
            g0 = viewer.user_scn.geoms[0]
            label = f"1/{JUMP_SLOWMO}x" if slow_now else "1x"
            rgba0 = [1.0, 0.55, 0.0, 1.0] if slow_now else [0.4, 1.0, 0.4, 1.0]
            mujoco.mjv_initGeom(
                g0, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0],
                [-0.25, 0.15, 0.60], np.eye(3).flatten(),
                np.array(rgba0, dtype=np.float32))
            g0.label = label.encode()[:99]

            com = data.subtree_com[box_bid].copy()
            g1  = viewer.user_scn.geoms[1]
            mujoco.mjv_initGeom(
                g1, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],
                com, np.eye(3).flatten(),
                np.array([1.0, 0.95, 0.0, 0.85], dtype=np.float32))
            g1.label = b"CG"
            viewer.user_scn.ngeom = 2

            max_height_m = max(max_height_m, data.xpos[wheel_bid_L][2] - WHEEL_R)

            elapsed = time.perf_counter() - frame_start
            if (sleep_t := 1.0 / RENDER_HZ - elapsed) > 0:
                time.sleep(sleep_t)

    print(f"\nMax jump height: {max_height_m * 1000:.1f} mm")
    _log_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Force log saved → {os.path.join(_log_dir, 'force_log.csv')}")
    print(f"Run size_recommend.py to get tube/bearing recommendations.")
    plot_q.put(None)
    plot_proc.join(timeout=2)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mp.freeze_support()
    run_viewer()
