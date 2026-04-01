"""visualizer.py — Unified visualization for master_sim.

Replaces both replay.py and sandbox_fastchart.py with a single file.

Modes:
    --mode chart   : CSV chart viewer (default if CSV files given)
    --mode replay  : Run a scenario, show pyqtgraph 5x2 telemetry + MuJoCo viewer
    --mode sandbox : Interactive arena (Phase 6.1b — TODO)

Usage examples:
    # Replay scenario s01 with default gains:
    python -m master_sim.viz --mode replay --scenario s01_lqr_pitch_step

    # Replay with MuJoCo viewer alongside:
    python -m master_sim.viz --mode replay --scenario s04_vel_pi_staircase --viewer

    # CSV chart viewer (unchanged from Phase 6a):
    python -m master_sim.viz logs/S1_LQR_pitch_step.csv
    python -m master_sim.viz logs/S1_LQR_pitch_step.csv --x run_id --y fitness
"""
import argparse
import math
import multiprocessing as mp
import os
import sys
import threading
import time
from collections import deque

# Allow running this file directly: python visualizer.py
# Adds simulation/mujoco/ to sys.path so "import master_sim" works.
_MASTER_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MUJOCO_DIR = os.path.dirname(_MASTER_SIM_DIR)
if _MUJOCO_DIR not in sys.path:
    sys.path.insert(0, _MUJOCO_DIR)

import numpy as np

# IMPORTANT: mujoco must be imported BEFORE pyqtgraph on Windows to avoid
# OpenGL DLL init conflict (WinError 1114).
import mujoco
import mujoco.viewer

import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui


# ── Style constants (from sandbox_fastchart.py) ──────────────────────────────
BG_COLOR    = "#12121e"
BAR_COLOR   = "#1a1a2e"
TICK_COLOR  = "#d8d8d8"
GRID_ALPHA  = 0.20
LINE_WIDTH  = 1.8

# Window layout padding (pixels) — avoid vertical taskbar on the left
# and leave a gap between fastchart and MuJoCo windows.
PAD_LEFT    = 90    # clear vertical taskbar
PAD_TOP     = 35    # keep title bar visible
PAD_MID     = 10    # gap between the two side-by-side windows

# 10-color palette: vivid, distinguishable on dark background
PALETTE = [
    "#60d0ff",   # cyan
    "#ff6060",   # red
    "#80ff80",   # green
    "#ffa040",   # orange
    "#ffe060",   # yellow
    "#e080ff",   # purple
    "#ff80b0",   # pink
    "#40ffd0",   # teal
    "#a0a0ff",   # lavender
    "#e0e0e0",   # white-grey
]

DASH = QtCore.Qt.PenStyle.DashLine

TELEMETRY_HZ = 60
WINDOW_S     = 15.0


_N_STATIC_GEOMS = 3   # world-axes geoms (X, Y, Z arrows) — preserved across frames


def _add_world_axes(viewer, length=0.3, radius=0.006):
    """Draw RGB XYZ axis arrows at the world origin in the MuJoCo viewer."""
    z_up = np.array([0.0, 0.0, 1.0])

    def _rotation_z_to(d_norm):
        """Rotation matrix that maps Z-up to d_norm (column-major flat)."""
        if np.allclose(d_norm, z_up):
            return np.eye(3).flatten()
        if np.allclose(d_norm, -z_up):
            return np.diag([1.0, -1.0, -1.0]).flatten()
        v = np.cross(z_up, d_norm)
        s = np.linalg.norm(v)
        c = np.dot(z_up, d_norm)
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        return R.flatten()

    axes = [
        ([1, 0, 0], [1, 0, 0, 0.9]),   # X = red
        ([0, 1, 0], [0, 1, 0, 0.9]),   # Y = green
        ([0, 0, 1], [0, 0, 1, 0.9]),   # Z = blue
    ]
    scn = viewer.user_scn
    for direction, rgba in axes:
        if scn.ngeom >= scn.maxgeom:
            break
        d = np.array(direction, dtype=np.float64)
        mat = _rotation_z_to(d)
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[radius, radius, length],
            pos=np.array([0.0, 0.0, 0.0]),
            mat=mat,
            rgba=np.array(rgba, dtype=np.float32),
        )
        scn.ngeom += 1


def _add_force_arrow(viewer, data, body_id, force_z, scale=0.04, radius=0.012):
    """Draw a +Z arrow at *body_id* proportional to *force_z*.  No-op if zero."""
    if abs(force_z) < 1e-9:
        return
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        return
    pos = data.xpos[body_id].copy()
    length = abs(force_z) * scale
    # Arrow points +Z by default; flip rotation if force is negative
    if force_z >= 0:
        mat = np.eye(3).flatten()
    else:
        mat = np.diag([1.0, -1.0, -1.0]).flatten()
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=[radius, radius, length],
        pos=pos,
        mat=mat,
        rgba=np.array([1.0, 0.5, 0.0, 0.9], dtype=np.float32),
    )
    scn.ngeom += 1


def _add_knee_spring_sphere(viewer, data, body_id, active=False, radius=0.015):
    """Draw a 3 cm sphere at *body_id* for the knee spring.

    Always visible as grey; turns purple when spring torque is active.
    """
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        return
    pos = data.xpos[body_id].copy()
    if active:
        rgba = np.array([0.6, 0.0, 0.8, 0.9], dtype=np.float32)   # purple
    else:
        rgba = np.array([0.5, 0.5, 0.5, 0.7], dtype=np.float32)   # grey
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    scn.ngeom += 1


# ═══════════════════════════════════════════════════════════════════════════════
# SANDBOX ARENA — 28 static obstacles + 6 movable props
# ═══════════════════════════════════════════════════════════════════════════════

SANDBOX_OBSTACLES = (
    # Spheres (4)
    dict(shape="sphere",  x= 3.0, y= 0.0, r=1.20, h=0.06),
    dict(shape="sphere",  x=-2.5, y= 1.5, r=0.80, h=0.05),
    dict(shape="sphere",  x= 5.5, y=-2.0, r=1.00, h=0.07),
    dict(shape="sphere",  x=-5.0, y=-1.0, r=0.90, h=0.06),
    # Capsules (5)
    dict(shape="capsule", x= 1.5, y= 0.0, r=0.020, length=1.20),
    dict(shape="capsule", x=-1.2, y= 0.5, r=0.025, length=0.80),
    dict(shape="capsule", x= 4.0, y= 1.0, r=0.030, length=1.00),
    dict(shape="capsule", x= 2.5, y=-1.5, r=0.020, length=0.70),
    dict(shape="capsule", x=-3.5, y=-2.0, r=0.035, length=1.20),
    # Ramps (4)
    dict(shape="ramp", x= 2.0, y=-0.5, angle_deg= 8, length=0.60, width=0.50, h=0.050),
    dict(shape="ramp", x=-2.0, y= 2.0, angle_deg= 6, length=0.80, width=0.60, h=0.040),
    dict(shape="ramp", x= 6.0, y= 0.5, angle_deg=10, length=0.50, width=0.60, h=0.060),
    dict(shape="ramp", x=-4.0, y=-3.0, angle_deg= 7, length=0.70, width=0.50, h=0.045),
    # Boxes (4)
    dict(shape="box", x= 1.0, y= 2.0, rx=0.12, ry=0.20, h=0.02),
    dict(shape="box", x=-1.5, y=-2.5, rx=0.15, ry=0.15, h=0.03),
    dict(shape="box", x= 7.0, y=-1.5, rx=0.20, ry=0.25, h=0.07),
    dict(shape="box", x=-6.0, y= 2.0, rx=0.20, ry=0.20, h=0.06),
    # Cylinder (1)
    dict(shape="cyl", x= 4.5, y= 3.0, r=0.10, h=0.05),
)

SANDBOX_PROPS = (
    dict(type="can",           x= 0.8, y= 0.3),
    dict(type="can",           x= 0.8, y=-0.3),
    dict(type="bottle",        x= 1.5, y=-0.8),
    dict(type="ball",          x= 0.6, y= 0.8),
    dict(type="ball",          x= 0.5, y=-0.6),
    dict(type="cardboard_box", x= 2.0, y= 0.5),
)


def sandbox_world():
    """Return a WorldConfig for the sandbox arena (28 obstacles, 6 props, 25×25 m floor)."""
    from master_sim_jump.scenarios.base import WorldConfig
    return WorldConfig(
        sandbox_obstacles=SANDBOX_OBSTACLES,
        prop_bodies=SANDBOX_PROPS,
        floor_size=(25.0, 25.0, 0.1),
    )


# ── Reusable chart panel builder ─────────────────────────────────────────────
class ChartPanel:
    """Creates a styled pyqtgraph plot panel — reused by future live visualizer."""

    TICK_FONT = None  # set once after QApplication exists

    @classmethod
    def _ensure_font(cls):
        if cls.TICK_FONT is None:
            cls.TICK_FONT = QtGui.QFont("Consolas", 9)

    @staticmethod
    def create(glw, row, col, title, ylabel, xrange=None):
        """Add a plot to a GraphicsLayoutWidget and return it, fully styled."""
        ChartPanel._ensure_font()
        pl = glw.addPlot(row=row, col=col)
        pl.setTitle(
            f'<span style="color:#e0e0e0;font-size:9pt;font-weight:600">{title}</span>')
        pl.setLabel(
            "left",
            f'<span style="color:#c8c8c8;font-size:9pt">{ylabel}</span>')
        pl.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        for ax_name in ("left", "bottom"):
            ax = pl.getAxis(ax_name)
            ax.setTextPen(pg.mkColor(TICK_COLOR))
            ax.setPen(pg.mkPen("#555"))
            ax.setStyle(tickFont=ChartPanel.TICK_FONT)
        if xrange is not None:
            pl.setXRange(*xrange, padding=0.02)
        return pl

    @staticmethod
    def add_legend(pl, ncols=1):
        leg = pl.addLegend(offset=(6, 6), verSpacing=-4, colCount=ncols)
        leg.setBrush(pg.mkBrush(18, 18, 36, 210))
        leg.setPen(pg.mkPen("#444"))
        leg.setLabelTextColor(pg.mkColor("#cccccc"))
        return leg

    @staticmethod
    def add_limit_lines(pl, val, color="#ff4444"):
        for sign in (+1, -1):
            anchor = (0.05, 1.1) if sign > 0 else (0.05, -0.1)
            il = pg.InfiniteLine(
                pos=sign * val, angle=0,
                pen=pg.mkPen(color, width=1.2, style=DASH),
                label=f'{"+" if sign > 0 else "−"}{val:.1f}',
                labelOpts={"color": color, "anchors": [anchor, anchor]})
            pl.addItem(il)


# ── CSV loading ──────────────────────────────────────────────────────────────
def load_csv(path, filter_expr=None):
    """Load a CSV into a DataFrame, optionally filtering rows.

    filter_expr: "col=val" string — keeps only rows where col == val.
    """
    df = pd.read_csv(path)
    if filter_expr:
        col, val = filter_expr.split("=", 1)
        col = col.strip()
        val = val.strip()
        if col in df.columns:
            # Try numeric comparison first, fall back to string
            try:
                val_num = float(val)
                df = df[df[col] == val_num]
            except ValueError:
                df = df[df[col].astype(str) == val]
    return df


def auto_numeric_columns(df, exclude=None):
    """Return list of numeric column names, excluding specified ones."""
    exclude = set(exclude or [])
    exclude.update({"run_id", "timestamp"})
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude]


# ── Main viewer ──────────────────────────────────────────────────────────────
def show(csv_paths, x_col=None, y_cols=None, filter_expr=None,
         title=None, max_cols=2):
    """Open a pyqtgraph window plotting CSV data.

    Parameters
    ----------
    csv_paths : list[str]
        One or more CSV file paths.
    x_col : str | None
        Column for X axis (default: row index).
    y_cols : list[str] | None
        Columns to plot. None = auto-detect all numeric.
    filter_expr : str | None
        "col=val" filter applied to every CSV.
    title : str | None
        Window title.
    max_cols : int
        Max chart columns in grid layout.
    """
    pg.setConfigOptions(antialias=False, useOpenGL=True, enableExperimental=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # ── Main window ──────────────────────────────────────────────────────
    win = QtWidgets.QMainWindow()
    win.setWindowTitle(title or "master_sim — Chart Viewer")
    win.setStyleSheet(f"background:{BG_COLOR};")
    central = QtWidgets.QWidget()
    win.setCentralWidget(central)
    vbox = QtWidgets.QVBoxLayout(central)
    vbox.setContentsMargins(4, 4, 4, 2)
    vbox.setSpacing(2)

    glw = pg.GraphicsLayoutWidget()
    glw.setBackground(BG_COLOR)
    vbox.addWidget(glw, stretch=1)

    # ── Status / hover bar ───────────────────────────────────────────────
    status_row = QtWidgets.QWidget()
    status_row.setStyleSheet(f"background:{BAR_COLOR}; border-radius:4px;")
    hbox = QtWidgets.QHBoxLayout(status_row)
    hbox.setContentsMargins(6, 3, 6, 3)
    hbox.setSpacing(0)

    _SL = ("color:#e8e8e8; font-family:Consolas,monospace; "
           "font-size:11px; font-weight:bold; padding:0 14px 0 0;")

    lbl_info = QtWidgets.QLabel(f"Files: {len(csv_paths)}")
    lbl_info.setStyleSheet(_SL)
    hbox.addWidget(lbl_info)
    hbox.addStretch()

    hover_lbl = QtWidgets.QLabel("")
    hover_lbl.setStyleSheet(
        "color:#80ffb0; font-family:Consolas,monospace; font-size:11px; padding:0 12px;")
    hbox.addWidget(hover_lbl)
    vbox.addWidget(status_row)

    # ── Load data and create panels ──────────────────────────────────────
    named_plots = []

    for file_idx, csv_path in enumerate(csv_paths):
        df = load_csv(csv_path, filter_expr)
        basename = os.path.basename(csv_path)
        n_rows = len(df)

        if n_rows == 0:
            print(f"  Warning: {basename} has 0 rows after filtering, skipping.")
            continue

        # Determine columns to plot
        if y_cols:
            cols = [c for c in y_cols if c in df.columns]
            if not cols:
                print(f"  Warning: none of {y_cols} found in {basename}, "
                      f"available: {list(df.columns)}")
                continue
        else:
            cols = auto_numeric_columns(df, exclude={x_col} if x_col else None)
            if not cols:
                print(f"  Warning: no numeric columns in {basename}")
                continue

        # X data
        if x_col and x_col in df.columns:
            x_data = df[x_col].values.astype(float)
            x_label = x_col
        else:
            x_data = np.arange(n_rows, dtype=float)
            x_label = "row"

        # Create one panel per y-column, arranged in grid
        for col_idx, col_name in enumerate(cols):
            grid_row = file_idx * ((len(cols) + max_cols - 1) // max_cols) + col_idx // max_cols
            grid_col = col_idx % max_cols

            panel_title = f"{basename} — {col_name}" if len(csv_paths) > 1 else col_name
            pl = ChartPanel.create(glw, grid_row, grid_col, panel_title, col_name)
            pl.setLabel("bottom",
                        f'<span style="color:#c8c8c8;font-size:9pt">{x_label}</span>')

            y_data = pd.to_numeric(df[col_name], errors="coerce").values
            valid = ~np.isnan(y_data) & ~np.isnan(x_data)

            if valid.sum() == 0:
                continue

            color = PALETTE[col_idx % len(PALETTE)]
            pl.plot(x_data[valid], y_data[valid],
                    pen=pg.mkPen(color, width=LINE_WIDTH),
                    symbol="o", symbolSize=4,
                    symbolBrush=pg.mkBrush(color),
                    symbolPen=None)

            named_plots.append((panel_title, pl))

        lbl_info.setText(
            f"Files: {len(csv_paths)} | "
            f"Rows: {n_rows}{' (filtered)' if filter_expr else ''} | "
            f"Panels: {len(named_plots)}")

    # ── Mouse hover ──────────────────────────────────────────────────────
    def _on_mouse(evt):
        pos = evt[0]
        for name, pl in named_plots:
            if pl.sceneBoundingRect().contains(pos):
                mp = pl.vb.mapSceneToView(pos)
                hover_lbl.setText(f"▶ {name}   x={mp.x():.3f}  y={mp.y():.4f}")
                return
        hover_lbl.setText("")

    _proxy = pg.SignalProxy(glw.scene().sigMouseMoved, rateLimit=60, slot=_on_mouse)

    # ── Enforce minimum Y span of 2 units on all static panels ───────────
    _MIN_Y_SPAN = 2.0
    _fixed_range_plots = {id(pl_hip)}
    for _, pl in named_plots:
        if id(pl) in _fixed_range_plots:
            continue
        d_lo, d_hi = float('inf'), float('-inf')
        for item in pl.listDataItems():
            yd = item.yData
            if yd is not None and len(yd) > 0:
                d_lo = min(d_lo, float(np.nanmin(yd)))
                d_hi = max(d_hi, float(np.nanmax(yd)))
        if d_lo <= d_hi:
            span = d_hi - d_lo
            if span < _MIN_Y_SPAN:
                mid = (d_lo + d_hi) * 0.5
                half = _MIN_Y_SPAN * 0.5
                pl.setYRange(mid - half, mid + half, padding=0)

    # ── Window sizing ────────────────────────────────────────────────────
    try:
        screen = app.primaryScreen()
        rect = screen.geometry()
        win.setGeometry(rect.x() + PAD_LEFT, rect.y() + PAD_TOP,
                        min(rect.width() - PAD_LEFT, 1600),
                        min(rect.height() - PAD_TOP, 1000))
    except Exception:
        win.resize(1400, 900)

    win.show()
    app.exec()


# ═══════════════════════════════════════════════════════════════════════════════
# REPLAY MODE — run scenario via sim_loop.run(), display pyqtgraph telemetry
# ═══════════════════════════════════════════════════════════════════════════════

class TelemetryRecorder:
    """Callback for sim_loop.run() — collects tick data into numpy arrays."""

    def __init__(self):
        self._ticks = []

    def __call__(self, tick: dict):
        self._ticks.append(tick)

    def to_arrays(self) -> dict:
        """Convert collected ticks to dict of numpy arrays."""
        if not self._ticks:
            return {}
        keys = self._ticks[0].keys()
        return {k: np.array([t[k] for t in self._ticks]) for k in keys}

    @property
    def n_ticks(self):
        return len(self._ticks)


def _replay_run_sim(scenario_name: str, params=None, rng_seed: int = 0):
    """Run scenario headlessly and return (metrics, telemetry_arrays, params)."""
    from master_sim_jump.defaults import DEFAULT_PARAMS
    from master_sim_jump.scenarios import SCENARIOS
    from master_sim_jump.sim_loop import run

    if params is None:
        params = DEFAULT_PARAMS
    cfg = SCENARIOS[scenario_name]
    recorder = TelemetryRecorder()

    print(f"  Running scenario: {cfg.display_name} ({cfg.duration:.1f}s) ...")
    t0 = time.perf_counter()
    metrics = run(params, cfg, callbacks=[recorder], rng_seed=rng_seed)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.2f}s — {recorder.n_ticks} ticks, "
          f"status={metrics['status']}, fitness={cfg.fitness_fn(metrics):.4f}")

    return metrics, recorder.to_arrays(), params, cfg



def replay_show(telemetry: dict, metrics: dict, scenario_name: str,
                display_name: str = ""):
    """Show pyqtgraph 5x2 telemetry panel from recorded sim data.

    Parameters
    ----------
    telemetry : dict of numpy arrays from TelemetryRecorder.to_arrays()
    metrics   : dict from sim_loop.run()
    scenario_name : e.g. "s01_lqr_pitch_step"
    display_name  : human-readable scenario name
    """
    from master_sim_jump.defaults import DEFAULT_PARAMS
    robot = DEFAULT_PARAMS.robot

    pg.setConfigOptions(antialias=False, useOpenGL=True, enableExperimental=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    t = telemetry.get('t', np.array([]))
    if len(t) == 0:
        print("  No telemetry data to display.")
        return

    # Derived signals
    pitch_deg     = np.degrees(telemetry['pitch'])
    pitch_rate_deg = np.degrees(telemetry['pitch_rate'])
    roll_deg      = np.degrees(telemetry['roll'])
    hip_q_deg     = np.degrees(telemetry['hip_q_avg'])
    v_measured    = telemetry['v_measured']
    v_target      = telemetry['v_target']
    yaw_rate      = telemetry['yaw_rate']
    omega_tgt     = telemetry['omega_tgt']
    tau_whl_L     = telemetry['tau_whl_L']
    tau_whl_R     = telemetry['tau_whl_R']
    v_batt        = telemetry['v_batt']

    # Compute pitch reference (equilibrium + theta_ref) for display
    from master_sim_jump.physics import get_equilibrium_pitch
    pitch_ff = np.array([get_equilibrium_pitch(robot, q)
                         for q in telemetry['hip_q_avg']])
    pitch_ref_deg = np.degrees(pitch_ff + telemetry['theta_ref'])

    # Compute delta_q (L-R hip difference) — not in tick data, use hip_q_avg as proxy
    # For now show hip_q_avg (both sides identical in headless sim)

    # ── Window setup ──────────────────────────────────────────────────────
    status_str = metrics.get('status', '?')
    fit_val = metrics.get('fitness', '?')
    win_title = (f"master_sim Replay — {display_name or scenario_name} | "
                 f"status={status_str} | fitness={fit_val}")

    win = QtWidgets.QMainWindow()
    win.setWindowTitle(win_title)
    win.setStyleSheet(f"background:{BG_COLOR};")
    central = QtWidgets.QWidget()
    win.setCentralWidget(central)
    vbox = QtWidgets.QVBoxLayout(central)
    vbox.setContentsMargins(4, 4, 4, 2)
    vbox.setSpacing(2)

    glw = pg.GraphicsLayoutWidget()
    glw.setBackground(BG_COLOR)
    vbox.addWidget(glw, stretch=1)

    # ── Status bar ────────────────────────────────────────────────────────
    status_row = QtWidgets.QWidget()
    status_row.setStyleSheet(f"background:{BAR_COLOR}; border-radius:4px;")
    hbox = QtWidgets.QHBoxLayout(status_row)
    hbox.setContentsMargins(6, 3, 6, 3)
    hbox.setSpacing(0)

    _SL = ("color:#e8e8e8; font-family:Consolas,monospace; "
           "font-size:11px; font-weight:bold; padding:0 14px 0 0;")
    lbl_info = QtWidgets.QLabel(
        f"{display_name} | {len(t)} ticks | "
        f"RMS pitch={metrics.get('rms_pitch_deg', '?'):.2f} deg | "
        f"survived={metrics.get('survived_s', '?')}s")
    lbl_info.setStyleSheet(_SL)
    hbox.addWidget(lbl_info)
    hbox.addStretch()

    hover_lbl = QtWidgets.QLabel("")
    hover_lbl.setStyleSheet(
        "color:#80ffb0; font-family:Consolas,monospace; font-size:11px; padding:0 12px;")
    hbox.addWidget(hover_lbl)
    vbox.addWidget(status_row)

    # ── Fitness breakdown table ────────────────────────────────────────────
    bd = metrics.get('fitness_breakdown', {})
    if bd:
        # Sort by value descending (worst penalty first), skip zero terms
        items = [(k, v) for k, v in sorted(bd.items(), key=lambda x: -x[1]) if v != 0.0]
        total = sum(bd.values())
        bd_row = QtWidgets.QWidget()
        bd_row.setStyleSheet(f"background:{BAR_COLOR}; border-radius:4px;")
        bd_hbox = QtWidgets.QHBoxLayout(bd_row)
        bd_hbox.setContentsMargins(6, 2, 6, 2)
        bd_hbox.setSpacing(0)
        _BDL = ("color:#e8e8e8; font-family:Consolas,monospace; "
                "font-size:11px; padding:0 4px 0 0;")
        _BDH = ("color:#ff6060; font-family:Consolas,monospace; "
                "font-size:11px; font-weight:bold; padding:0 4px 0 0;")
        parts = []
        for i, (name, val) in enumerate(items):
            # Highlight the worst (first) non-FELL term
            style = _BDH if i == 0 else _BDL
            parts.append(f'<span style="color:{"#ff6060" if i == 0 else "#e8e8e8"}">'
                         f'{name}={val:.4f}</span>')
        text = " | ".join(parts) + f'  ||  <b>TOTAL={total:.4f}</b>'
        lbl_bd = QtWidgets.QLabel(text)
        lbl_bd.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lbl_bd.setStyleSheet("color:#e8e8e8; font-family:Consolas,monospace; "
                             "font-size:11px; padding:0;")
        bd_hbox.addWidget(lbl_bd)
        bd_hbox.addStretch()
        vbox.addWidget(bd_row)

    # ── 5x2 Panel layout (matching replay.py) ─────────────────────────────
    # Row 0: Pitch, Velocity
    # Row 1: Yaw Rate, Hip Angle
    # Row 2: Roll, Pitch Rate
    # Row 3: Wheel Torques, (unused / battery current placeholder)
    # Row 4: Battery Voltage, SoC
    xr = (float(t[0]), float(t[-1]))

    # [0,0] Pitch
    pl_pitch = ChartPanel.create(glw, 0, 0, "Pitch", "deg", xr)
    ChartPanel.add_legend(pl_pitch)
    pl_pitch.plot(t, pitch_deg,     pen=pg.mkPen(PALETTE[0], width=LINE_WIDTH), name="pitch")
    pl_pitch.plot(t, pitch_ref_deg, pen=pg.mkPen(PALETTE[1], width=1.0, style=DASH), name="ref")

    # [0,1] Velocity
    pl_vel = ChartPanel.create(glw, 0, 1, "Velocity", "m/s", xr)
    ChartPanel.add_legend(pl_vel)
    pl_vel.plot(t, v_measured, pen=pg.mkPen(PALETTE[2], width=LINE_WIDTH), name="actual")
    pl_vel.plot(t, v_target,   pen=pg.mkPen(PALETTE[3], width=1.0, style=DASH), name="cmd")

    # [1,0] Yaw Rate
    pl_yaw = ChartPanel.create(glw, 1, 0, "Yaw Rate", "rad/s", xr)
    ChartPanel.add_legend(pl_yaw)
    pl_yaw.plot(t, yaw_rate,  pen=pg.mkPen(PALETTE[3], width=LINE_WIDTH), name="yaw w")
    pl_yaw.plot(t, omega_tgt, pen=pg.mkPen(PALETTE[3], width=1.0, style=DASH), name="cmd w")
    pl_yaw.addLine(y=0, pen=pg.mkPen("#666688", width=0.7, style=DASH))
    _yaw_cmd_peak = float(np.max(np.abs(omega_tgt))) if len(omega_tgt) else 1.0
    _yaw_margin = math.radians(10.0)
    pl_yaw.setYRange(-_yaw_cmd_peak - _yaw_margin, _yaw_cmd_peak + _yaw_margin, padding=0)

    # [1,1] Hip Angle
    pl_hip = ChartPanel.create(glw, 1, 1, "Hip Angle", "deg", xr)
    pl_hip.plot(t, hip_q_deg, pen=pg.mkPen(PALETTE[5], width=LINE_WIDTH))
    _hip_lo = math.degrees(min(robot.Q_RET, robot.Q_EXT)) - 3.0
    _hip_hi = math.degrees(max(robot.Q_RET, robot.Q_EXT)) + 3.0
    pl_hip.setYRange(_hip_lo, _hip_hi, padding=0)
    pl_hip.addLine(y=math.degrees(robot.Q_RET), pen=pg.mkPen('#ff4444', width=1.5, style=DASH))
    pl_hip.addLine(y=math.degrees(robot.Q_EXT), pen=pg.mkPen('#ff4444', width=1.5, style=DASH))

    # [2,0] Roll
    pl_roll = ChartPanel.create(glw, 2, 0, "Roll", "deg", xr)
    pl_roll.plot(t, roll_deg, pen=pg.mkPen(PALETTE[6], width=LINE_WIDTH))
    pl_roll.addLine(y=0, pen=pg.mkPen("#666688", width=0.7, style=DASH))

    # [2,1] Pitch Rate
    pl_prate = ChartPanel.create(glw, 2, 1, "Pitch Rate", "deg/s", xr)
    pl_prate.plot(t, pitch_rate_deg, pen=pg.mkPen(PALETTE[5], width=LINE_WIDTH))
    pl_prate.addLine(y=0, pen=pg.mkPen("#666688", width=0.7, style=DASH))

    # [3,0] Hip Torques
    tau_hip_L = telemetry.get('tau_hip_L', np.zeros_like(t))
    tau_hip_R = telemetry.get('tau_hip_R', np.zeros_like(t))
    tau_spring_L = telemetry.get('tau_spring_L', np.zeros_like(t))
    tau_spring_R = telemetry.get('tau_spring_R', np.zeros_like(t))
    pl_htau = ChartPanel.create(glw, 3, 0, "Hip Torques", "N-m", xr)
    ChartPanel.add_legend(pl_htau)
    pl_htau.plot(t, tau_hip_L, pen=pg.mkPen(PALETTE[5], width=LINE_WIDTH), name="hip_L")
    pl_htau.plot(t, tau_hip_R, pen=pg.mkPen(PALETTE[8], width=1.2, style=DASH), name="hip_R")
    pl_htau.plot(t, tau_spring_L, pen=pg.mkPen('#ff60ff', width=1.2), name="spring_L")
    pl_htau.plot(t, tau_spring_R, pen=pg.mkPen('#ffaa44', width=1.2, style=DASH), name="spring_R")

    # [3,1] Wheel Torques
    pl_tau = ChartPanel.create(glw, 3, 1, "Wheel Torques", "N-m", xr)
    ChartPanel.add_legend(pl_tau)
    pl_tau.plot(t, tau_whl_L, pen=pg.mkPen(PALETTE[4], width=LINE_WIDTH), name="t_L")
    pl_tau.plot(t, tau_whl_R, pen=pg.mkPen(PALETTE[3], width=1.2, style=DASH), name="t_R")
    pl_tau.addLine(y=0, pen=pg.mkPen("#666688", width=0.7, style=DASH))

    # [4,0] Battery Voltage
    pl_batt = ChartPanel.create(glw, 4, 0, "Battery Voltage", "V", xr)
    pl_batt.plot(t, v_batt, pen=pg.mkPen(PALETTE[3], width=LINE_WIDTH))
    if len(v_batt) > 0:
        from master_sim_jump.defaults import DEFAULT_PARAMS as _dp
        pl_batt.addLine(y=_dp.battery.V_nom, pen=pg.mkPen("#ffffff", width=0.8, style=DASH))

    # [4,1] Position X
    pos_x = telemetry.get('pos_x', np.zeros_like(t))
    pl_pos = ChartPanel.create(glw, 4, 1, "Position X", "m", xr)
    pl_pos.plot(t, pos_x, pen=pg.mkPen(PALETTE[7], width=LINE_WIDTH))

    # [5, span 2] Wheel Jump Height (Z above resting ground contact)
    wheel_z_L_raw = telemetry.get('wheel_z_L', np.full_like(t, robot.wheel_r))
    wheel_z_R_raw = telemetry.get('wheel_z_R', np.full_like(t, robot.wheel_r))
    wheel_h_L   = np.maximum(wheel_z_L_raw - robot.wheel_r, 0.0) * 1000.0  # mm
    wheel_h_R   = np.maximum(wheel_z_R_raw - robot.wheel_r, 0.0) * 1000.0  # mm
    wheel_h_avg = (wheel_h_L + wheel_h_R) * 0.5
    ChartPanel._ensure_font()
    pl_wh = glw.addPlot(row=5, col=0, colspan=2)
    pl_wh.setTitle('<span style="color:#e0e0e0;font-size:9pt;font-weight:600">Wheel Jump Height</span>')
    pl_wh.setLabel("left", '<span style="color:#c8c8c8;font-size:9pt">mm</span>')
    pl_wh.showGrid(x=True, y=True, alpha=GRID_ALPHA)
    for _ax in ("left", "bottom"):
        _a = pl_wh.getAxis(_ax)
        _a.setTextPen(pg.mkColor(TICK_COLOR))
        _a.setPen(pg.mkPen("#555"))
        _a.setStyle(tickFont=ChartPanel.TICK_FONT)
    pl_wh.setXRange(*xr, padding=0.02)
    ChartPanel.add_legend(pl_wh)
    pl_wh.plot(t, wheel_h_L,   pen=pg.mkPen(PALETTE[0], width=LINE_WIDTH),       name="L")
    pl_wh.plot(t, wheel_h_R,   pen=pg.mkPen(PALETTE[1], width=1.2, style=DASH),  name="R")
    pl_wh.plot(t, wheel_h_avg, pen=pg.mkPen(PALETTE[9], width=1.5),              name="avg")
    pl_wh.addLine(y=0, pen=pg.mkPen("#666688", width=0.7, style=DASH))
    # Vertical markers: airborne / landed from robot's own mode transitions
    _modes = telemetry.get('mode', np.full(len(t), 'BALANCE', dtype=object))
    if len(_modes) > 1:
        _air_idxs  = np.where((_modes[1:] == 'FLYING')  & (_modes[:-1] != 'FLYING'))[0]
        _land_idxs = np.where((_modes[1:] == 'LANDING') & (_modes[:-1] == 'FLYING'))[0]
        if len(_air_idxs):
            pl_wh.addItem(pg.InfiniteLine(
                pos=t[_air_idxs[0]], angle=90,
                pen=pg.mkPen('#ffff44', width=1.5, style=DASH),
                label='airborne', labelOpts={'color': '#ffff44', 'position': 0.95,
                                             'fill': pg.mkBrush(0, 0, 0, 120)}))
            _after_air = _land_idxs[_land_idxs > _air_idxs[0]]
            if len(_after_air):
                pl_wh.addItem(pg.InfiniteLine(
                    pos=t[_after_air[0]], angle=90,
                    pen=pg.mkPen('#ff8844', width=1.5, style=DASH),
                    label='landed', labelOpts={'color': '#ff8844', 'position': 0.85,
                                               'fill': pg.mkBrush(0, 0, 0, 120)}))

    # ── Mouse hover ───────────────────────────────────────────────────────
    all_plots = [
        ("Pitch", pl_pitch), ("Velocity", pl_vel),
        ("Yaw Rate", pl_yaw), ("Hip Angle", pl_hip),
        ("Roll", pl_roll), ("Pitch Rate", pl_prate),
        ("Hip Torques", pl_htau), ("Wheel Torques", pl_tau),
        ("Battery V", pl_batt), ("Position X", pl_pos),
        ("Wheel Jump Height", pl_wh),
    ]

    def _on_mouse(evt):
        pos = evt[0]
        for name, pl in all_plots:
            if pl.sceneBoundingRect().contains(pos):
                mp = pl.vb.mapSceneToView(pos)
                hover_lbl.setText(f"  {name}   t={mp.x():.3f}s  val={mp.y():.4f}")
                return
        hover_lbl.setText("")

    _proxy = pg.SignalProxy(glw.scene().sigMouseMoved, rateLimit=60, slot=_on_mouse)

    # ── Window sizing — left half of screen (with taskbar padding) ──────────
    try:
        screen = app.primaryScreen()
        rect = screen.geometry()
        half_w = rect.width() // 2
        win.setGeometry(rect.x() + PAD_LEFT, rect.y() + PAD_TOP,
                        half_w - PAD_LEFT - PAD_MID // 2, rect.height() - PAD_TOP)
    except Exception:
        win.resize(900, 1000)

    win.show()
    return app, win, _proxy  # keep references alive


def replay(scenario_name: str, with_viewer: bool = False,
           params=None, rng_seed: int = 0):
    """Full replay: run scenario in real-time with live scrolling telemetry.

    Uses the same dual-process architecture as sandbox():
      Main process  — MuJoCo viewer + physics + control loop
      Child process — pyqtgraph 10-panel live telemetry (identical to sandbox)

    Parameters
    ----------
    scenario_name : key in SCENARIOS, e.g. "s01_lqr_pitch_step"
    with_viewer   : ignored (viewer always shown for real-time pacing)
    params        : SimParams override (default: DEFAULT_PARAMS)
    rng_seed      : reproducible noise seed
    """
    from master_sim_jump.defaults import DEFAULT_PARAMS
    from master_sim_jump.scenarios import SCENARIOS
    from master_sim_jump.sim_loop import (
        build_model_and_data, init_sim, SimController, get_pitch_and_rate,
        apply_disturbance,
    )

    if params is None:
        params = DEFAULT_PARAMS
    cfg = SCENARIOS[scenario_name]
    robot = params.robot

    print(f"  Replay: {cfg.display_name} ({cfg.duration:.1f}s)")

    model, data = build_model_and_data(params, cfg.world)
    init_sim(model, data, params)
    if cfg.init_fn:
        cfg.init_fn(model, data, params)

    RENDER_HZ = 60
    PHYSICS_HZ = round(1.0 / model.opt.timestep)
    steps_per_frame = max(1, PHYSICS_HZ // RENDER_HZ)
    ctrl_steps = params.timing.ctrl_steps

    # Scenario profiles
    v_profile_fn     = cfg.v_profile or (lambda t: 0.0)
    theta_ref_fn     = cfg.theta_ref_profile
    omega_profile_fn = cfg.omega_profile
    hip_profile_fn   = cfg.hip_profile
    hip_vel_fn       = cfg.hip_vel_profile

    # Controller enable flags from scenario (single source of truth)
    flags = cfg.tick_flags

    # Controller — single source of truth (shared with scenarios/optimizer)
    ctrl = SimController(model, data, params, rng_seed=rng_seed)
    step = 0
    _last_tick = {}
    last_push  = -1.0

    # Launch plot process (identical to sandbox)
    wheel_limit = params.motors.wheel.torque_limit
    hip_limit   = params.motors.hip.impedance_torque_limit
    _vp_init = _vel_pi_gains_dict(params)
    _lqr_init = _lqr_gains_dict(params)
    _yaw_init = _yaw_pi_gains_dict(params)
    _susp_init = _suspension_gains_dict(params)
    _hl_init = _hip_limits_dict(params)
    _rg_init = _robot_geom_dict(params)

    data_q = mp.Queue(maxsize=12000)
    cmd_q  = mp.Queue(maxsize=32)
    plot_proc = mp.Process(
        target=_plot_process,
        args=(data_q, cmd_q, WINDOW_S,
              f"Replay — {cfg.display_name}",
              wheel_limit, hip_limit, False, _vp_init, _lqr_init,
              _ctrl_defaults_from_cfg(cfg), _yaw_init, _susp_init,
              _hl_init, _rg_init),
        daemon=True)
    plot_proc.start()
    print(f"  Plot process started (PID {plot_proc.pid})")

    # Screen size for window positioning
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
        sw = ctypes.windll.user32.GetSystemMetrics(0)
        sh = ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        sw, sh = 1920, 1080

    scenario_logged = False
    sim_fell = False

    def _reset_replay():
        nonlocal step, _last_tick, last_push, scenario_logged, sim_fell
        mujoco.mj_resetData(model, data)
        init_sim(model, data, params)
        if cfg.init_fn:
            cfg.init_fn(model, data, params)
        ctrl.reset(model, data)
        step = 0
        _last_tick = {}
        last_push = -1.0
        scenario_logged = False
        sim_fell = False
        if not data_q.full():
            data_q.put_nowait("RESET")
        print("  Replay restarted.")

    print("  Launching MuJoCo viewer — scenario running in real-time ...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 35
        viewer.cam.elevation = -15
        viewer.cam.distance  = 2.5
        viewer.cam.lookat    = np.array([0.0, 0.0, 0.30])
        _add_world_axes(viewer)

        # Snap viewer to right half (Windows, with mid-padding)
        def _snap():
            time.sleep(1.5)
            try:
                import ctypes as _ct
                _EnumProc = _ct.WINFUNCTYPE(_ct.c_bool, _ct.c_int, _ct.c_int)
                half = sw // 2
                mj_x = half + PAD_MID // 2
                mj_w = sw - mj_x
                def _cb(hwnd, _):
                    if _ct.windll.user32.IsWindowVisible(hwnd):
                        buf = _ct.create_unicode_buffer(256)
                        _ct.windll.user32.GetWindowTextW(hwnd, buf, 256)
                        if "mujoco" in buf.value.lower():
                            SWP = 0x0004
                            _ct.windll.user32.SetWindowPos(
                                hwnd, 0, mj_x, 0, mj_w, sh, SWP)
                            return False
                    return True
                _ct.windll.user32.EnumWindows(_EnumProc(_cb), 0)
            except Exception:
                pass
        threading.Thread(target=_snap, daemon=True).start()

        while viewer.is_running():
            frame_start = time.perf_counter()

            # Drain command queue from plot process
            try:
                while True:
                    cmd = cmd_q.get_nowait()
                    if cmd == "RESTART":
                        _reset_replay()
                    elif isinstance(cmd, tuple):
                        if cmd[0] == "GAIN_VP":
                            ctrl.update_velocity_pi_gain(cmd[1], cmd[2])
                        elif cmd[0] == "GAIN_LQR":
                            ctrl.update_lqr_gain(cmd[1], cmd[2])
                        elif cmd[0] == "GAIN_YAW":
                            ctrl.update_yaw_pi_gain(cmd[1], cmd[2])
                        elif cmd[0] == "GAIN_SUSP":
                            ctrl.update_suspension_gain(cmd[1], cmd[2])
                        elif cmd[0] == "ROBOT_GEOM":
                            ctrl.update_robot_geom(cmd[1], cmd[2])
            except Exception:
                pass

            for _ in range(steps_per_frame):
                if sim_fell:
                    break

                if step % ctrl_steps == 0:
                    t_now = float(data.time)
                    past_duration = t_now >= cfg.duration

                    if not scenario_logged and past_duration:
                        scenario_logged = True
                        print(f"  Scenario complete ({cfg.duration:.1f}s). "
                              f"Continuing — close viewer to exit.")

                    # After scenario duration, hold final profile values
                    v_target     = v_profile_fn(t_now) if not past_duration else 0.0
                    theta_ref_c  = (theta_ref_fn(t_now) if theta_ref_fn and not past_duration else 0.0)
                    omega_target = (omega_profile_fn(t_now) if omega_profile_fn and not past_duration else 0.0)
                    q_hip_target = hip_profile_fn(t_now) if hip_profile_fn else robot.Q_NOM
                    dq_hip_tgt   = hip_vel_fn(t_now) if hip_vel_fn else 0.0

                    # Jump trigger (mirrors sim_loop.run)
                    if (cfg.jump_time is not None and
                            t_now >= cfg.jump_time):
                        ctrl.jump_ctrl.trigger()

                    tick = ctrl.tick(
                        model, data,
                        v_target_ms=v_target,
                        theta_ref_cmd=theta_ref_c,
                        omega_target=omega_target,
                        q_hip_target=q_hip_target,
                        dq_hip_target=dq_hip_tgt,
                        **flags)

                    _last_tick = tick

                    if tick['fell']:
                        sim_fell = True
                        print(f"  Robot fell at t={t_now:.2f}s. "
                              f"Close viewer to exit.")
                        break

                # Disturbance only during scenario duration
                _vis_bid, _vis_fz = None, 0.0
                if data.time < cfg.duration:
                    _vis_bid, _vis_fz = apply_disturbance(
                        data, data.time, cfg,
                        ctrl.box_bid, ctrl.wheel_bid_L, ctrl.wheel_bid_R)

                mujoco.mj_step(model, data)
                step += 1

            # Camera follow
            viewer.cam.lookat[0] = data.xpos[ctrl.box_bid][0]
            viewer.cam.lookat[1] = data.xpos[ctrl.box_bid][1]
            # Reset dynamic geoms (keep static world-axes)
            viewer.user_scn.ngeom = _N_STATIC_GEOMS
            if _vis_bid is not None:
                _add_force_arrow(viewer, data, _vis_bid, _vis_fz)
            # Always-visible knee spring spheres (grey idle, purple when active)
            _add_knee_spring_sphere(viewer, data, ctrl.tibia_bid_L,
                active=abs(_last_tick.get('tau_spring_L', 0.0)) > 1e-6 if _last_tick else False)
            _add_knee_spring_sphere(viewer, data, ctrl.tibia_bid_R,
                active=abs(_last_tick.get('tau_spring_R', 0.0)) > 1e-6 if _last_tick else False)
            viewer.sync()

            # Push 27-value telemetry tuple at ~60 Hz
            wall_now = time.perf_counter()
            if (wall_now - last_push >= 1.0 / TELEMETRY_HZ
                    and not data_q.full() and _last_tick):
                tk = _last_tick
                pitch_true_now, _ = get_pitch_and_rate(
                    data, ctrl.box_bid, ctrl.d_pitch)
                pitch_ref_display = tk['pitch_ff'] + tk['theta_ref']
                v_cmd = v_profile_fn(float(data.time))
                _ocm = ctrl.yaw_pi_ctrl.gains.omega_cmd_max
                omega_cmd = max(-_ocm, min(_ocm,
                    omega_profile_fn(float(data.time))
                    if omega_profile_fn else 0.0))
                data_q.put_nowait((
                    float(data.time),
                    math.degrees(pitch_true_now),
                    math.degrees(pitch_ref_display),
                    math.degrees(tk['pitch_rate']),
                    tk['v_measured'],
                    v_cmd,
                    math.degrees(tk['yaw_rate']),
                    math.degrees(omega_cmd),
                    math.degrees(tk['hip_q_L']),
                    math.degrees(tk['hip_q_R']),
                    math.degrees(tk['q_nom_L']),
                    math.degrees(tk['q_nom_R']),
                    math.degrees(tk['roll']),
                    tk['tau_whl_L'],
                    tk['tau_whl_R'],
                    tk['v_batt'],
                    tk['batt_temp'],
                    tk['batt_soc'],
                    abs(tk['tau_whl_L']) / params.motors.wheel.Kt,
                    abs(tk['tau_whl_R']) / params.motors.wheel.Kt,
                    abs(tk['tau_hip_L']) / params.motors.hip.Kt_output,
                    abs(tk['tau_hip_R']) / params.motors.hip.Kt_output,
                    tk['i_total'],
                    tk['tau_hip_L'],
                    tk['tau_hip_R'],
                    tk['tau_spring_L'],
                    tk['tau_spring_R'],
                    max(tk['wheel_z_L'] - params.robot.wheel_r, 0.0) * 1000.0,
                    max(tk['wheel_z_R'] - params.robot.wheel_r, 0.0) * 1000.0,
                    tk.get('mode', 'BALANCE'),
                    tk.get('az_imu', 0.0),
                ))
                last_push = wall_now

            elapsed = time.perf_counter() - frame_start
            sleep_t = 1.0 / RENDER_HZ - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    # Clean up
    data_q.put(None)
    plot_proc.join(timeout=2)
    if plot_proc.is_alive():
        plot_proc.terminate()
    print("  Replay closed.")
    os._exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# SANDBOX MODE — interactive arena with live telemetry (Step 6.2)
# ═══════════════════════════════════════════════════════════════════════════════

# 28 static obstacles — spheres, capsules, ramps, boxes, cylinders
SANDBOX_OBSTACLES = [
    # Spheres (4) — elevated terrain bumps
    dict(shape="sphere", x= 3.0, y= 0.0, r=1.20, h=0.06),
    dict(shape="sphere", x=-2.5, y= 1.5, r=0.80, h=0.05),
    dict(shape="sphere", x= 5.5, y=-2.0, r=1.00, h=0.07),
    dict(shape="sphere", x=-5.0, y=-1.0, r=0.90, h=0.06),
    # Capsules (5) — thin poles
    dict(shape="capsule", x= 1.5, y= 0.0, r=0.020, length=1.20),
    dict(shape="capsule", x=-1.2, y= 0.5, r=0.025, length=0.80),
    dict(shape="capsule", x= 4.0, y= 1.0, r=0.030, length=1.00),
    dict(shape="capsule", x= 2.5, y=-1.5, r=0.020, length=0.70),
    dict(shape="capsule", x=-3.5, y=-2.0, r=0.035, length=1.20),
    # Ramps (4)
    dict(shape="ramp", x= 2.0, y=-0.5, angle_deg= 8, length=0.60, width=0.50, h=0.050),
    dict(shape="ramp", x=-2.0, y= 2.0, angle_deg= 6, length=0.80, width=0.60, h=0.040),
    dict(shape="ramp", x= 6.0, y= 0.5, angle_deg=10, length=0.50, width=0.60, h=0.060),
    dict(shape="ramp", x=-4.0, y=-3.0, angle_deg= 7, length=0.70, width=0.50, h=0.045),
    # Boxes (4) + cylinder (1)
    dict(shape="box", x= 1.0, y= 2.0, rx=0.12, ry=0.20, h=0.02),
    dict(shape="box", x=-1.5, y=-2.5, rx=0.15, ry=0.15, h=0.03),
    dict(shape="cyl", x= 4.5, y= 3.0, r=0.10,           h=0.05),
    dict(shape="box", x= 7.0, y=-1.5, rx=0.20, ry=0.25, h=0.07),
    dict(shape="box", x=-6.0, y= 2.0, rx=0.20, ry=0.20, h=0.06),
    # Staircase — 3 steps, real-world dims (0.18 m rise, 0.28 m run, 1.0 m wide), purple
    dict(shape="box", x= 8.14, y= 0.0, rx=0.14, ry=0.50, h=0.18, rgba="0.55 0.10 0.80 1.0"),
    dict(shape="box", x= 8.42, y= 0.0, rx=0.14, ry=0.50, h=0.36, rgba="0.55 0.10 0.80 1.0"),
    dict(shape="box", x= 8.70, y= 0.0, rx=0.14, ry=0.50, h=0.54, rgba="0.55 0.10 0.80 1.0"),
]

# 6 movable prop bodies — scale-reference objects
SANDBOX_PROPS = [
    dict(type="can",           x= 0.8, y= 0.3),
    dict(type="can",           x= 0.8, y=-0.3),
    dict(type="bottle",        x= 1.5, y=-0.8),
    dict(type="ball",          x= 0.6, y= 0.8),
    dict(type="ball",          x= 0.5, y=-0.6),
    dict(type="cardboard_box", x= 2.0, y= 0.5),
]


def sandbox_world() -> 'WorldConfig':
    """Return a WorldConfig for the sandbox arena (28 obstacles + 6 props, 25m floor)."""
    from master_sim_jump.scenarios.base import WorldConfig
    return WorldConfig(
        sandbox_obstacles=tuple(SANDBOX_OBSTACLES),
        prop_bodies=tuple(SANDBOX_PROPS),
        floor_size=(25.0, 25.0, 0.1),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SANDBOX — dual-process: main = MuJoCo sim, child = pyqtgraph telemetry
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_process(data_q: mp.Queue, cmd_q: mp.Queue,
                  window_s: float, title: str,
                  wheel_limit: float, hip_limit: float,
                  has_gamepad: bool = False,
                  vel_pi_gains: dict = None,
                  lqr_gains: dict = None,
                  ctrl_defaults: dict = None,
                  yaw_pi_gains: dict = None,
                  suspension_gains: dict = None,
                  hip_limits: dict = None,
                  robot_geom: dict = None) -> None:
    """Child process — independent Qt app with 11-panel pyqtgraph telemetry.

    Communication:
        data_q (main→plot): 29-value telemetry tuples at ~60 Hz
        cmd_q  (plot→main): UI commands (RESTART, CTRL_EN, HIP_MODE)
    """
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

    # ── Tab widget: Main (telemetry) + Mechanical (structural loads) ────────
    tab_widget = QtWidgets.QTabWidget()
    tab_widget.setStyleSheet(
        "QTabWidget::pane{border:0;background:#12121e}"
        "QTabBar::tab{background:#1a1a2e;color:#aaa;padding:4px 14px;"
        "font-family:Consolas,monospace;font-size:11px;border:1px solid #333;"
        "border-bottom:0;border-top-left-radius:4px;border-top-right-radius:4px}"
        "QTabBar::tab:selected{background:#12121e;color:#e0e0e0;font-weight:bold}"
        "QTabBar::tab:hover{color:#60d0ff}")
    vbox.addWidget(tab_widget, stretch=1)

    glw = pg.GraphicsLayoutWidget()
    glw.setBackground("#12121e")
    tab_widget.addTab(glw, "Main")

    # ── Fitness breakdown overlay (hidden until scenario completes) ─────────
    _fitness_overlay = QtWidgets.QLabel(glw)
    _fitness_overlay.setStyleSheet(
        "background:rgba(26,26,46,230); color:#e8e8e8; "
        "font-family:Consolas,monospace; font-size:12px; "
        "border:1px solid #3a3a5e; border-radius:6px; padding:8px 12px;")
    _fitness_overlay.setTextFormat(QtCore.Qt.TextFormat.RichText)
    _fitness_overlay.hide()
    _fitness_overlay.setWordWrap(True)
    _fitness_overlay.mousePressEvent = lambda e: _fitness_overlay.hide()

    def _show_fitness_breakdown(bd, total, status):
        """Build HTML table and show overlay in top-right of chart area."""
        items = [(k, v) for k, v in sorted(bd.items(), key=lambda x: -x[1]) if v != 0.0]
        if not items:
            return
        rows = ""
        for i, (name, val) in enumerate(items):
            color = "#ff6060" if i == 0 else "#e8e8e8"
            bar_w = min(int(val / max(total, 0.001) * 120), 120) if total > 0 else 0
            rows += (f'<tr><td style="color:{color};padding:1px 8px 1px 0">{name}</td>'
                     f'<td style="color:{color};text-align:right;padding:1px 4px">'
                     f'{val:.4f}</td>'
                     f'<td><div style="background:{color};width:{bar_w}px;'
                     f'height:8px;border-radius:2px"></div></td></tr>')
        html = (f'<div style="color:#80ff80;font-size:11px;margin-bottom:4px">'
                f'status={status} | click to dismiss</div>'
                f'<table>{rows}'
                f'<tr><td style="color:#60d0ff;padding:3px 8px 0 0;'
                f'border-top:1px solid #3a3a5e"><b>TOTAL</b></td>'
                f'<td style="color:#60d0ff;text-align:right;padding:3px 4px 0 0;'
                f'border-top:1px solid #3a3a5e"><b>{total:.4f}</b></td>'
                f'<td></td></tr></table>')
        _fitness_overlay.setText(html)
        _fitness_overlay.adjustSize()
        # Position top-right of the chart area
        pw = glw.width()
        ow = _fitness_overlay.sizeHint().width()
        _fitness_overlay.move(max(pw - ow - 12, 4), 8)
        _fitness_overlay.raise_()
        _fitness_overlay.show()

    # ── Status bar ─────────────────────────────────────────────────────────────
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

    def _safe_cmd(cmd):
        """Put a command on the queue, silently dropping if full."""
        try:
            cmd_q.put_nowait(cmd)
        except Exception:
            pass

    def _send_restart():
        # Drain stale commands so the queue never stays full
        while not cmd_q.empty():
            try:
                cmd_q.get_nowait()
            except Exception:
                break
        try:
            cmd_q.put_nowait("RESTART")
        except Exception:
            pass
    btn.clicked.connect(_send_restart)
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

    _CB_STYLE = (
        "QCheckBox{color:#e8e8e8;font-family:Consolas,monospace;"
        "font-size:11px;font-weight:bold;spacing:5px}"
        "QCheckBox::indicator{width:14px;height:14px}"
        "QCheckBox::indicator:checked{background:#4a9eff;"
        "border:1px solid #4a9eff;border-radius:2px}"
        "QCheckBox::indicator:unchecked{background:#333;"
        "border:1px solid #666;border-radius:2px}")

    _cd = ctrl_defaults or {}
    _ctrl_defs = [
        ("LQR",        "LQR Balance"),
        ("VelPI",      "Vel PI"),
        ("YawPI",      "Yaw PI"),
        ("Suspension", "Suspension"),
        ("RollLev",    "Roll Leveling"),
        ("HipFF",      "Hip FF"),
        ("GravFF",     "Grav FF"),
        ("CoMFF",      "CoM FF"),
        ("TurnFF",     "Turn FF"),
        ("KneeSpr",    "Knee Spring"),
    ]
    _ctrl_widgets = {}   # key → QCheckBox, for CONFIG_UPDATE
    for key, label in _ctrl_defs:
        cb = QtWidgets.QCheckBox(label)
        cb.setChecked(_cd.get(key, True))
        cb.setStyleSheet(_CB_STYLE)
        cb.stateChanged.connect(
            lambda state, k=key: _safe_cmd(("CTRL_EN", k, bool(state))))
        hbox_ctrl.addWidget(cb)
        _ctrl_widgets[key] = cb

    hbox_ctrl.addStretch()

    # Hip mode toggle button
    _hip_is_pd = not _cd.get("Suspension", True)   # position PD when suspension off
    _btn_hip = QtWidgets.QPushButton("Hip: Pos PD" if _hip_is_pd else "Hip: Impedance")
    _btn_hip.setFixedHeight(24)
    _btn_hip.setCheckable(True)
    _btn_hip.setChecked(_hip_is_pd)   # False = Impedance, True = Position PD
    _btn_hip.setStyleSheet(
        "QPushButton{background:#3a3a5e;color:#e8e8e8;font-family:Consolas,monospace;"
        "font-size:11px;font-weight:bold;border-radius:4px;padding:0 10px}"
        "QPushButton:checked{background:#5a3a1e;color:#ffb060}"
        "QPushButton:hover{background:#4a4a7e}")

    def _on_hip_toggle(checked):
        _btn_hip.setText("Hip: Pos PD" if checked else "Hip: Impedance")
        _safe_cmd(("HIP_MODE", "position_pd" if checked else "impedance"))

    _btn_hip.toggled.connect(_on_hip_toggle)
    hbox_ctrl.addWidget(_btn_hip)

    # Jump trigger button (sandbox mode)
    _btn_jump = QtWidgets.QPushButton("Jump!")
    _btn_jump.setFixedHeight(24)
    _btn_jump.setStyleSheet(
        "QPushButton{background:#5a1e3a;color:#ff80b0;font-family:Consolas,monospace;"
        "font-size:11px;font-weight:bold;border-radius:4px;padding:0 10px}"
        "QPushButton:hover{background:#7a2e5a}")
    _btn_jump.clicked.connect(lambda: _safe_cmd(("JUMP",)))
    hbox_ctrl.addWidget(_btn_jump)

    vbox.addWidget(ctrl_row)

    # ── Spinbox tracking dicts (for Save button) ────────────────────────────
    # Maps field_name → (QDoubleSpinBox, to_rad_factor_or_None)
    _vp_spinboxes = {}
    _lqr_spinboxes = {}
    _yaw_spinboxes = {}
    _susp_spinboxes = {}
    _hip_lim_spinboxes = {}

    _SPIN_STYLE = (
        "QDoubleSpinBox{background:#2a2a4e;color:#80ffb0;border:1px solid #444;"
        "border-radius:3px;font-family:Consolas,monospace;font-size:11px;"
        "font-weight:bold;padding:1px 4px;min-width:70px}"
        "QDoubleSpinBox::up-button,QDoubleSpinBox::down-button{"
        "background:#3a3a5e;border:1px solid #555;width:14px}"
        "QDoubleSpinBox::up-button:hover,QDoubleSpinBox::down-button:hover{"
        "background:#5a5a9e}")
    _SPIN_LBL = ("color:#e8e8e8;font-family:Consolas,monospace;"
                 "font-size:11px;font-weight:bold;")

    import math as _math

    # ── Gain group panel: combobox + stacked pages ───────────────────────────
    gains_outer = QtWidgets.QWidget()
    gains_outer.setStyleSheet("background:#1a1a2e; border-radius:4px;")
    hbox_gains_outer = QtWidgets.QHBoxLayout(gains_outer)
    hbox_gains_outer.setContentsMargins(6, 3, 6, 3)
    hbox_gains_outer.setSpacing(10)

    _gain_combo = QtWidgets.QComboBox()
    _gain_combo.addItems(["VelPI", "LQR", "YawPI", "Suspension", "Hip Limits"])
    _gain_combo.setFixedWidth(100)
    _gain_combo.setStyleSheet(
        "QComboBox{background:#2a2a4e;color:#ffd080;border:1px solid #555;"
        "border-radius:3px;font-family:Consolas,monospace;font-size:11px;"
        "font-weight:bold;padding:1px 6px}"
        "QComboBox::drop-down{border:0;width:18px}"
        "QComboBox QAbstractItemView{background:#2a2a4e;color:#ffd080;"
        "selection-background-color:#3a3a6e}")
    hbox_gains_outer.addWidget(_gain_combo)

    _gain_stack = QtWidgets.QStackedWidget()
    _gain_stack.setStyleSheet("background:transparent")
    hbox_gains_outer.addWidget(_gain_stack, stretch=1)
    _gain_combo.currentIndexChanged.connect(_gain_stack.setCurrentIndex)

    # ── Page 0: VelocityPI ───────────────────────────────────────────────────
    _vp = vel_pi_gains or {}
    _page_vp = QtWidgets.QWidget()
    _page_vp.setStyleSheet("background:transparent")
    _hb_vp = QtWidgets.QHBoxLayout(_page_vp)
    _hb_vp.setContentsMargins(0, 0, 0, 0)
    _hb_vp.setSpacing(12)

    _gain_defs = [
        ("Kp",          "Kp",                   0.0,   1.0,    0.005, 4, _vp.get("Kp", 0.0103),                               None),
        ("Ki",          "Ki",                   0.0,   2.0,    0.01,  4, _vp.get("Ki", 0.0554),                                None),
        ("Kff",         "Kff",                  0.0,   1.0,    0.005, 4, _vp.get("Kff", 0.102),                                None),
        ("θ_max°",      "theta_max",            5.0,   90.0,   1.0,   1, _math.degrees(_vp.get("theta_max", 0.8)),             _math.pi/180),
        ("rate_lim°/s", "theta_ref_rate_limit", 50.0, 12000.0, 50.0, 0, _math.degrees(_vp.get("theta_ref_rate_limit", 50.0)), _math.pi/180),
    ]
    for disp, field, lo, hi, step, dec, default, to_rad in _gain_defs:
        lbl = QtWidgets.QLabel(disp); lbl.setStyleSheet(_SPIN_LBL)
        _hb_vp.addWidget(lbl)
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(lo, hi); sb.setSingleStep(step); sb.setDecimals(dec)
        sb.setValue(default); sb.setStyleSheet(_SPIN_STYLE)
        if to_rad is not None:
            sb.valueChanged.connect(lambda val, f=field, fac=to_rad: _safe_cmd(("GAIN_VP", f, val * fac)))
        else:
            sb.valueChanged.connect(lambda val, f=field: _safe_cmd(("GAIN_VP", f, val)))
        _hb_vp.addWidget(sb)
        _vp_spinboxes[field] = (sb, to_rad)
    _hb_vp.addStretch()
    _gain_stack.addWidget(_page_vp)

    # ── Page 1: LQR ─────────────────────────────────────────────────────────
    _lq = lqr_gains or {}
    _page_lqr = QtWidgets.QWidget()
    _page_lqr.setStyleSheet("background:transparent")
    _hb_lqr = QtWidgets.QHBoxLayout(_page_lqr)
    _hb_lqr.setContentsMargins(0, 0, 0, 0)
    _hb_lqr.setSpacing(12)

    _lqr_defs = [
        ("Q_pitch",      "Q_pitch",      0.01,  50.0,  0.01, 3, _lq.get("Q_pitch", 1.0)),
        ("Q_pitch_rate", "Q_pitch_rate", 0.001, 10.0,  0.01, 4, _lq.get("Q_pitch_rate", 1.0)),
        ("R",            "R",            0.0,  100.0,  0.1,  4, _lq.get("R", 1e-5)),
    ]
    for disp, fld, lo, hi, step, dec, default in _lqr_defs:
        lbl = QtWidgets.QLabel(disp); lbl.setStyleSheet(_SPIN_LBL)
        _hb_lqr.addWidget(lbl)
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(lo, hi); sb.setSingleStep(step); sb.setDecimals(dec)
        sb.setValue(default); sb.setStyleSheet(_SPIN_STYLE)
        sb.valueChanged.connect(lambda val, f=fld: _safe_cmd(("GAIN_LQR", f, val)))
        _hb_lqr.addWidget(sb)
        _lqr_spinboxes[fld] = (sb, None)
    _hb_lqr.addStretch()
    _gain_stack.addWidget(_page_lqr)

    # ── Page 2: YawPI ───────────────────────────────────────────────────────
    _yp = yaw_pi_gains or {}
    _page_yaw = QtWidgets.QWidget()
    _page_yaw.setStyleSheet("background:transparent")
    _hb_yaw = QtWidgets.QHBoxLayout(_page_yaw)
    _hb_yaw.setContentsMargins(0, 0, 0, 0)
    _hb_yaw.setSpacing(12)

    _yaw_defs = [
        ("Kp",      "Kp",           0.0,  2.0,  0.01, 4, _yp.get("Kp", 0.3)),
        ("Ki",      "Ki",           0.0,  5.0,  0.05, 4, _yp.get("Ki", 1.2)),
        ("τ_max",   "torque_max",   0.05, 3.0,  0.05, 2, _yp.get("torque_max", 0.5)),
        ("int_max", "int_max",      0.05, 3.0,  0.05, 2, _yp.get("int_max", 0.5)),
        ("ω_max",   "omega_cmd_max", 0.1, 6.28, 0.05, 2, _yp.get("omega_cmd_max", 1.05)),
    ]
    for disp, fld, lo, hi, step, dec, default in _yaw_defs:
        lbl = QtWidgets.QLabel(disp); lbl.setStyleSheet(_SPIN_LBL)
        _hb_yaw.addWidget(lbl)
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(lo, hi); sb.setSingleStep(step); sb.setDecimals(dec)
        sb.setValue(default); sb.setStyleSheet(_SPIN_STYLE)
        sb.valueChanged.connect(lambda val, f=fld: _safe_cmd(("GAIN_YAW", f, val)))
        _hb_yaw.addWidget(sb)
        _yaw_spinboxes[fld] = (sb, None)
    _hb_yaw.addStretch()
    _gain_stack.addWidget(_page_yaw)

    # ── Page 3: Suspension ──────────────────────────────────────────────────
    _sg = suspension_gains or {}
    _page_susp = QtWidgets.QWidget()
    _page_susp.setStyleSheet("background:transparent")
    _hb_susp = QtWidgets.QHBoxLayout(_page_susp)
    _hb_susp.setContentsMargins(0, 0, 0, 0)
    _hb_susp.setSpacing(12)

    _susp_defs = [
        ("K_s",    "K_s",    0.0,  100.0, 0.5,  2, _sg.get("K_s", 20.0)),
        ("B_s",    "B_s",    0.0,    5.0, 0.05, 3, _sg.get("B_s", 0.65)),
        ("K_roll", "K_roll", 0.0,  500.0, 1.0,  1, _sg.get("K_roll", 120.0)),
        ("D_roll", "D_roll", 0.0,    5.0, 0.05, 2, _sg.get("D_roll", 0.15)),
    ]
    for disp, fld, lo, hi, step, dec, default in _susp_defs:
        lbl = QtWidgets.QLabel(disp); lbl.setStyleSheet(_SPIN_LBL)
        _hb_susp.addWidget(lbl)
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(lo, hi); sb.setSingleStep(step); sb.setDecimals(dec)
        sb.setValue(default); sb.setStyleSheet(_SPIN_STYLE)
        sb.valueChanged.connect(lambda val, f=fld: _safe_cmd(("GAIN_SUSP", f, val)))
        _hb_susp.addWidget(sb)
        _susp_spinboxes[fld] = (sb, None)
    _hb_susp.addStretch()
    _gain_stack.addWidget(_page_susp)

    # ── Page 4: Hip Limits ──────────────────────────────────────────────────
    _hl = hip_limits or {}
    _page_hl = QtWidgets.QWidget()
    _page_hl.setStyleSheet("background:transparent")
    _hb_hl = QtWidgets.QHBoxLayout(_page_hl)
    _hb_hl.setContentsMargins(0, 0, 0, 0)
    _hb_hl.setSpacing(12)

    _DEG2RAD = math.pi / 180.0
    _hip_lim_defs = [
        ("Q_ext", "Q_EXT", -120.0, -30.0, 0.5, 2, math.degrees(_hl.get("Q_EXT", -1.43161))),
        ("Q_ret", "Q_RET", -120.0, -30.0, 0.5, 2, math.degrees(_hl.get("Q_RET", -0.78705))),
    ]
    for disp, fld, lo, hi, step, dec, default in _hip_lim_defs:
        lbl = QtWidgets.QLabel(disp); lbl.setStyleSheet(_SPIN_LBL)
        _hb_hl.addWidget(lbl)
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(lo, hi); sb.setSingleStep(step); sb.setDecimals(dec)
        sb.setValue(default); sb.setStyleSheet(_SPIN_STYLE)
        sb.valueChanged.connect(lambda val, f=fld: _safe_cmd(("ROBOT_GEOM", f, math.radians(val))))
        _hb_hl.addWidget(sb)
        _hip_lim_spinboxes[fld] = sb
    _hb_hl.addStretch()
    _gain_stack.addWidget(_page_hl)

    vbox.addWidget(gains_outer)

    # ── Save Gains → params.py button ─────────────────────────────────────────
    def _save_gains_to_params():
        """Read current spinbox values and write them as baseline into params.py."""
        import re as _re
        import os as _os

        params_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "params.py")

        try:
            with open(params_path, "r", encoding="utf-8") as f:
                src = f.read()
        except Exception as exc:
            print(f"  [Save] ERROR reading params.py: {exc}")
            return

        original = src

        def _replace_field_in_class(text, class_name, field_name, new_val):
            """Replace a field default within a specific dataclass in params.py.

            Finds 'class <class_name>' then the first occurrence of
            '<field_name>: float = <value>' within that class body.
            """
            # Find the class definition
            cls_pattern = _re.compile(
                r'^class\s+' + _re.escape(class_name) + r'\b',
                _re.MULTILINE)
            cls_match = cls_pattern.search(text)
            if not cls_match:
                print(f"  [Save] WARNING: class {class_name} not found")
                return text

            # Search for the field within the class body (from class start onward)
            cls_start = cls_match.start()
            field_pattern = _re.compile(
                r'^(\s+' + _re.escape(field_name) + r'\s*:\s*float\s*=\s*)'
                r'([^\s#]+)'
                r'(.*)',
                _re.MULTILINE)
            field_match = field_pattern.search(text, cls_start)
            if not field_match:
                print(f"  [Save] WARNING: {class_name}.{field_name} not found")
                return text

            return (text[:field_match.start()]
                    + field_match.group(1) + str(new_val) + field_match.group(3)
                    + text[field_match.end():])

        # VelocityPI gains
        for field_name in ("Kp", "Ki", "Kff", "theta_max", "theta_ref_rate_limit"):
            sb, to_rad = _vp_spinboxes[field_name]
            val = sb.value()
            if to_rad is not None:
                val = val * to_rad  # convert deg display back to rad
            val = round(val, 6)
            src = _replace_field_in_class(src, "VelocityPIGains", field_name, val)

        # LQR gains
        for field_name, (sb, _) in _lqr_spinboxes.items():
            val = round(sb.value(), 6)
            src = _replace_field_in_class(src, "LQRGains", field_name, val)

        # YawPI gains
        for field_name, (sb, _) in _yaw_spinboxes.items():
            val = round(sb.value(), 6)
            src = _replace_field_in_class(src, "YawPIGains", field_name, val)

        # Suspension gains
        for field_name, (sb, _) in _susp_spinboxes.items():
            val = round(sb.value(), 6)
            src = _replace_field_in_class(src, "SuspensionGains", field_name, val)

        # Hip limits (displayed in degrees, saved in radians)
        for field_name, sb in _hip_lim_spinboxes.items():
            val = round(math.radians(sb.value()), 6)
            src = _replace_field_in_class(src, "RobotGeometry", field_name, val)

        if src == original:
            print("  [Save] No changes detected — params.py unchanged.")
            return

        try:
            with open(params_path, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"  [Save] Gains written to {params_path}")
            _btn_save.setText("✓ Saved!")
            _btn_save.setStyleSheet(
                "QPushButton{background:#1e5a3a;color:#80ff80;font-family:Consolas,monospace;"
                "font-size:11px;font-weight:bold;border-radius:4px;padding:0 14px}"
                "QPushButton:hover{background:#2e7a5a}")
            QtCore.QTimer.singleShot(2000, lambda: (
                _btn_save.setText("Save Gains → params.py"),
                _btn_save.setStyleSheet(
                    "QPushButton{background:#3a5a1e;color:#c0ff60;font-family:Consolas,monospace;"
                    "font-size:11px;font-weight:bold;border-radius:4px;padding:0 14px}"
                    "QPushButton:hover{background:#5a7a2e}")))
        except Exception as exc:
            print(f"  [Save] ERROR writing params.py: {exc}")
            _btn_save.setText(f"Error: {exc}")

    save_row = QtWidgets.QWidget()
    save_row.setStyleSheet("background:#1a1a2e; border-radius:4px;")
    hbox_save = QtWidgets.QHBoxLayout(save_row)
    hbox_save.setContentsMargins(6, 3, 6, 3)
    hbox_save.setSpacing(12)
    hbox_save.addStretch()

    _btn_save = QtWidgets.QPushButton("Save Gains → params.py")
    _btn_save.setFixedHeight(26)
    _btn_save.setStyleSheet(
        "QPushButton{background:#3a5a1e;color:#c0ff60;font-family:Consolas,monospace;"
        "font-size:11px;font-weight:bold;border-radius:4px;padding:0 14px}"
        "QPushButton:hover{background:#5a7a2e}")
    _btn_save.clicked.connect(_save_gains_to_params)
    hbox_save.addWidget(_btn_save)
    hbox_save.addStretch()
    vbox.addWidget(save_row)

    # ── Fallback sliders (shown when no gamepad detected) ─────────────────────
    if not has_gamepad:
        slider_row = QtWidgets.QWidget()
        slider_row.setStyleSheet("background:#1a1a2e; border-radius:4px;")
        hbox_sl = QtWidgets.QHBoxLayout(slider_row)
        hbox_sl.setContentsMargins(6, 3, 6, 3)
        hbox_sl.setSpacing(12)

        _SL_LABEL = ("color:#e8e8e8; font-family:Consolas,monospace; "
                     "font-size:11px; font-weight:bold;")
        _SL_SLIDER = (
            "QSlider::groove:horizontal{background:#333;height:6px;border-radius:3px}"
            "QSlider::handle:horizontal{background:#4a9eff;width:14px;margin:-4px 0;"
            "border-radius:7px}"
            "QSlider::sub-page:horizontal{background:#4a9eff;border-radius:3px}")

        def _make_slider(label_text, min_val, max_val, default, scale, cmd_name):
            lbl = QtWidgets.QLabel(label_text)
            lbl.setStyleSheet(_SL_LABEL)
            lbl.setFixedWidth(50)
            hbox_sl.addWidget(lbl)

            val_lbl = QtWidgets.QLabel(f"{default / scale:+.1f}")
            val_lbl.setStyleSheet(
                "color:#80ffb0; font-family:Consolas,monospace; "
                "font-size:11px; font-weight:bold; min-width:40px;")

            sl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            sl.setMinimum(min_val)
            sl.setMaximum(max_val)
            sl.setValue(default)
            sl.setFixedWidth(140)
            sl.setStyleSheet(_SL_SLIDER)

            def _on_change(v, s=scale, n=cmd_name, vl=val_lbl):
                real = v / s
                vl.setText(f"{real:+.1f}" if abs(real) >= 0.05 else " 0.0")
                _safe_cmd((n, real))

            sl.valueChanged.connect(_on_change)
            hbox_sl.addWidget(sl)
            hbox_sl.addWidget(val_lbl)
            return sl

        # Velocity: ±3.0 m/s, resolution 0.1
        _sl_vel = _make_slider("Vel:", -30, 30, 0, 10.0, "V_DESIRED")
        # Yaw: ±5.0 rad/s, resolution 0.1
        _sl_yaw = _make_slider("Yaw:", -50, 50, 0, 10.0, "OMEGA_DESIRED")
        # Hip: 0–100%, resolution 1
        _sl_hip = _make_slider("Hip%:", 0, 100, 50, 1.0, "HIP_PCT")

        hbox_sl.addStretch()

        # Reset sliders button
        btn_reset_sl = QtWidgets.QPushButton("Zero")
        btn_reset_sl.setFixedHeight(24)
        btn_reset_sl.setStyleSheet(
            "QPushButton{background:#3a3a5e;color:white;font-size:11px;"
            "border-radius:4px;padding:0 8px}"
            "QPushButton:hover{background:#5a5a9e}")

        def _zero_sliders():
            _sl_vel.setValue(0)
            _sl_yaw.setValue(0)
            _sl_hip.setValue(50)

        btn_reset_sl.clicked.connect(_zero_sliders)
        hbox_sl.addWidget(btn_reset_sl)

        vbox.addWidget(slider_row)

    # Position window on left half of primary monitor (with taskbar padding)
    try:
        screen = app.primaryScreen()
        rect   = screen.geometry()
        half_w = rect.width() // 2
        main_win.setGeometry(rect.x() + PAD_LEFT, rect.y() + PAD_TOP,
                             half_w - PAD_LEFT - PAD_MID // 2, rect.height() - PAD_TOP)
    except Exception:
        main_win.resize(960, 1000)

    main_win.show()

    # ── Style helpers ─────────────────────────────────────────────────────────
    TICK_FONT = QtGui.QFont("Consolas", 9)
    TICK_PEN  = pg.mkColor('#d8d8d8')
    _DASH     = QtCore.Qt.PenStyle.DashLine
    W         = 1.4

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
        pl.disableAutoRange(axis='y')   # we manage Y range in _update()
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
                pen=pg.mkPen(color, width=1.2, style=_DASH),
                label=f'{"+" if sign > 0 else "−"}{val:.1f}',
                labelOpts={"color": color, "anchors": [anchor, anchor]})
            pl.addItem(il)

    # ── Row 0: Pitch | Pitch Rate ─────────────────────────────────────────────
    p_pitch = _p(0, 0, "Pitch", "deg")
    _leg(p_pitch)
    ln_pitch     = p_pitch.plot(pen=pg.mkPen('#60d0ff', width=W), name="pitch")
    ln_pitch_ref = p_pitch.plot(pen=pg.mkPen('#ff6060', width=W, style=_DASH), name="cmd")

    p_prate = _p(0, 1, "Pitch Rate", "deg/s")
    _leg(p_prate)
    ln_prate = p_prate.plot(pen=pg.mkPen('#ffa040', width=W), name="pitch rate")

    # ── Row 1: Velocity | Yaw Rate ────────────────────────────────────────────
    p_vel = _p(1, 0, "Velocity", "m/s")
    _leg(p_vel)
    ln_vel  = p_vel.plot(pen=pg.mkPen('#60d0ff', width=W), name="vel")
    ln_vcmd = p_vel.plot(pen=pg.mkPen('#ff6060', width=W, style=_DASH), name="cmd")

    p_yaw = _p(1, 1, "Yaw Rate", "deg/s")
    _leg(p_yaw)
    from master_sim_jump.defaults import DEFAULT_PARAMS as _dp
    _yaw_cmd_peak_dps = math.degrees(_dp.gains.yaw_pi.omega_cmd_max)
    p_yaw.setYRange(-_yaw_cmd_peak_dps - 10.0, _yaw_cmd_peak_dps + 10.0, padding=0)
    ln_yaw  = p_yaw.plot(pen=pg.mkPen('#60d0ff', width=W), name="ω")
    ln_ocmd = p_yaw.plot(pen=pg.mkPen('#ff6060', width=W, style=_DASH), name="cmd")

    # ── Row 2: Hip Joints (4 lines: actual L/R + cmd L/R) | Roll ──────────────
    p_hip = _p(2, 0, "Hip Joints", "deg")
    _leg(p_hip, ncols=2)
    from master_sim_jump.defaults import DEFAULT_PARAMS as _dp
    _robot = _dp.robot
    _hip_lo = math.degrees(min(_robot.Q_RET, _robot.Q_EXT)) - 3.0
    _hip_hi = math.degrees(max(_robot.Q_RET, _robot.Q_EXT)) + 3.0
    p_hip.setYRange(_hip_lo, _hip_hi, padding=0)
    p_hip.addLine(y=math.degrees(_robot.Q_RET), pen=pg.mkPen('#ff4444', width=1.5, style=_DASH))
    p_hip.addLine(y=math.degrees(_robot.Q_EXT), pen=pg.mkPen('#ff4444', width=1.5, style=_DASH))
    ln_hip_L     = p_hip.plot(pen=pg.mkPen('#60d0ff', width=W), name="act L")
    ln_hip_R     = p_hip.plot(pen=pg.mkPen('#80ff80', width=W), name="act R")
    ln_hip_cmd_L = p_hip.plot(
        pen=pg.mkPen('#ff7070', width=W, style=_DASH), name="cmd L")
    ln_hip_cmd_R = p_hip.plot(
        pen=pg.mkPen('#ffe060', width=W, style=_DASH), name="cmd R")

    p_roll = _p(2, 1, "Roll", "deg")
    _leg(p_roll)
    ln_roll = p_roll.plot(pen=pg.mkPen('#ffa040', width=W), name="roll")

    # ── Row 3: Hip Torque (±limit) | Wheel Torque (±limit) ────────────────────
    p_htau = _p(3, 0, "Hip Torque", "N·m")
    _leg(p_htau)
    _limits(p_htau, hip_limit)
    p_htau.setYRange(-hip_limit * 1.1, hip_limit * 1.1, padding=0)
    ln_htau_L = p_htau.plot(pen=pg.mkPen('#60d0ff', width=W), name="L")
    ln_htau_R = p_htau.plot(pen=pg.mkPen('#80ff80', width=W), name="R")
    ln_htau_spr_L = p_htau.plot(pen=pg.mkPen('#ff60ff', width=W), name="spr L")
    ln_htau_spr_R = p_htau.plot(pen=pg.mkPen('#ffaa44', width=W), name="spr R")

    p_tau = _p(3, 1, "Wheel Torque", "N·m")
    _leg(p_tau)
    _limits(p_tau, wheel_limit)
    p_tau.setYRange(-wheel_limit * 1.1, wheel_limit * 1.1, padding=0)
    ln_tau_L = p_tau.plot(pen=pg.mkPen('#60d0ff', width=W), name="L")
    ln_tau_R = p_tau.plot(pen=pg.mkPen('#80ff80', width=W), name="R")

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
    ln_iwhl_L = p_cur.plot(pen=pg.mkPen('#60d0ff', width=W), name="whl L")
    ln_iwhl_R = p_cur.plot(pen=pg.mkPen('#80ff80', width=W), name="whl R")
    ln_ihip_L = p_cur.plot(pen=pg.mkPen('#ffa040', width=W), name="hip L")
    ln_ihip_R = p_cur.plot(pen=pg.mkPen('#ffe060', width=W), name="hip R")
    ln_itotal = p_cur.plot(
        pen=pg.mkPen('#e0e0e0', width=W, style=_DASH), name="total")

    # ── Row 5 (full-width): Wheel Jump Height ─────────────────────────────────
    p_wh = glw.addPlot(row=5, col=0, colspan=2)
    p_wh.setTitle('<span style="color:#e0e0e0;font-size:9pt;font-weight:600">Wheel Jump Height</span>')
    p_wh.setLabel("left", '<span style="color:#c8c8c8;font-size:9pt">mm</span>')
    p_wh.showGrid(x=True, y=True, alpha=0.20)
    for _ax_name in ("left", "bottom"):
        _ax = p_wh.getAxis(_ax_name)
        _ax.setTextPen(TICK_PEN)
        _ax.setPen(pg.mkPen('#555'))
        _ax.setStyle(tickFont=TICK_FONT)
    p_wh.setXRange(-window_s, 0, padding=0.02)
    p_wh.disableAutoRange(axis='y')
    p_wh.addLine(y=0, pen=pg.mkPen("#666688", width=0.7, style=_DASH))
    _leg(p_wh)
    ln_wh_L   = p_wh.plot(pen=pg.mkPen('#60d0ff', width=W),              name="L")
    ln_wh_R   = p_wh.plot(pen=pg.mkPen('#ff6060', width=W, style=_DASH), name="R")
    ln_wh_avg = p_wh.plot(pen=pg.mkPen('#e0e0e0', width=1.5),            name="avg")
    _ln_air  = pg.InfiniteLine(angle=90, movable=False,
                               pen=pg.mkPen('#ffff44', width=1.5, style=_DASH),
                               label='airborne',
                               labelOpts={'color': '#ffff44', 'position': 0.95,
                                          'fill': pg.mkBrush(0, 0, 0, 120)})
    _ln_land = pg.InfiniteLine(angle=90, movable=False,
                               pen=pg.mkPen('#ff8844', width=1.5, style=_DASH),
                               label='landed',
                               labelOpts={'color': '#ff8844', 'position': 0.85,
                                          'fill': pg.mkBrush(0, 0, 0, 120)})
    p_wh.addItem(_ln_air);  _ln_air.hide()
    p_wh.addItem(_ln_land); _ln_land.hide()

    # ── Mouse hover Y label ───────────────────────────────────────────────────
    named_plots = [
        ("Pitch",         p_pitch), ("Pitch Rate",     p_prate),
        ("Velocity",      p_vel),   ("Yaw Rate",       p_yaw),
        ("Hip Joints",    p_hip),   ("Roll",           p_roll),
        ("Hip Torque",    p_htau),  ("Wheel Torque",   p_tau),
        ("Battery",       p_batt),  ("Motor Currents", p_cur),
        ("Wheel Height",  p_wh),
    ]

    def _on_mouse(evt):
        pos = evt[0]
        for name, pl in named_plots:
            if pl.sceneBoundingRect().contains(pos):
                mp_ = pl.vb.mapSceneToView(pos)
                hover_lbl.setText(f"▶ {name}   y = {mp_.y():.3f}")
                return
        hover_lbl.setText("")

    _proxy = pg.SignalProxy(
        glw.scene().sigMouseMoved, rateLimit=60, slot=_on_mouse)

    # ══════════════════════════════════════════════════════════════════════════
    # MECHANICAL TAB — bearing forces, link loads, motor currents vs limits
    # ══════════════════════════════════════════════════════════════════════════
    glw_mech = pg.GraphicsLayoutWidget()
    glw_mech.setBackground("#12121e")
    tab_widget.addTab(glw_mech, "Mechanical")

    _rg = robot_geom or {}
    _has_mech = bool(_rg)

    def _mp(row, col, ttl, ylabel):
        """Add a plot to the Mechanical tab grid."""
        pl = glw_mech.addPlot(row=row, col=col)
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
        pl.disableAutoRange(axis='y')
        return pl

    def _mlimits(pl, val, label_text=None, color='#ff4444'):
        """Add ±limit red dashed lines to a mechanical plot."""
        for sign in (+1, -1):
            anchor = (0.05, 1.1) if sign > 0 else (0.05, -0.1)
            txt = label_text or f'{val:.0f}'
            il = pg.InfiniteLine(
                pos=sign * val, angle=0,
                pen=pg.mkPen(color, width=1.2, style=_DASH),
                label=f'{"+" if sign > 0 else "−"}{txt}',
                labelOpts={"color": color, "anchors": [anchor, anchor]})
            pl.addItem(il)

    def _mlimit_pos(pl, val, label_text=None, color='#ff4444'):
        """Add a single positive limit red dashed line."""
        txt = label_text or f'{val:.0f}'
        il = pg.InfiniteLine(
            pos=val, angle=0,
            pen=pg.mkPen(color, width=1.2, style=_DASH),
            label=txt,
            labelOpts={"color": color, "anchors": [(0.05, 1.1), (0.05, 1.1)]})
        pl.addItem(il)

    # ── Row 0: Bearing A (hip, 608) | Bearing C (knee, 608) ─────────────────
    _C0_608  = _rg.get('C0_608', 1370.0)
    _C0_6001 = _rg.get('C0_6001', 2850.0)

    pm_brg_a = _mp(0, 0, "Bearing A — Hip (608)", "N")
    _mlimit_pos(pm_brg_a, _C0_608, f'C₀={_C0_608:.0f}N')
    pm_brg_a.setYRange(0, _C0_608 * 1.2, padding=0)
    _leg(pm_brg_a)
    ln_brg_a_L = pm_brg_a.plot(pen=pg.mkPen('#60d0ff', width=W), name="L")
    ln_brg_a_R = pm_brg_a.plot(pen=pg.mkPen('#80ff80', width=W), name="R")

    pm_brg_c = _mp(0, 1, "Bearing C — Knee (608)", "N")
    _mlimit_pos(pm_brg_c, _C0_608, f'C₀={_C0_608:.0f}N')
    pm_brg_c.setYRange(0, _C0_608 * 1.2, padding=0)
    _leg(pm_brg_c)
    ln_brg_c_L = pm_brg_c.plot(pen=pg.mkPen('#60d0ff', width=W), name="L")
    ln_brg_c_R = pm_brg_c.plot(pen=pg.mkPen('#80ff80', width=W), name="R")

    # ── Row 1: Bearing E (stub, 6001) | Bearing W (wheel, 608) ──────────────
    pm_brg_e = _mp(1, 0, "Bearing E — Stub (6001)", "N")
    _mlimit_pos(pm_brg_e, _C0_6001, f'C₀={_C0_6001:.0f}N')
    pm_brg_e.setYRange(0, _C0_6001 * 1.2, padding=0)
    _leg(pm_brg_e)
    ln_brg_e_L = pm_brg_e.plot(pen=pg.mkPen('#60d0ff', width=W), name="L")
    ln_brg_e_R = pm_brg_e.plot(pen=pg.mkPen('#80ff80', width=W), name="R")

    pm_brg_w = _mp(1, 1, "Bearing W — Wheel (608)", "N")
    _mlimit_pos(pm_brg_w, _C0_608, f'C₀={_C0_608:.0f}N')
    pm_brg_w.setYRange(0, _C0_608 * 1.2, padding=0)
    _leg(pm_brg_w)
    ln_brg_w_L = pm_brg_w.plot(pen=pg.mkPen('#60d0ff', width=W), name="L")
    ln_brg_w_R = pm_brg_w.plot(pen=pg.mkPen('#80ff80', width=W), name="R")

    # ── Row 2: Femur axial | Tibia axial ────────────────────────────────────
    _Fy_fem = _rg.get('F_yield_femur', 2107.0)
    _Fy_tib = _rg.get('F_yield_tibia', 2912.0)

    pm_fem = _mp(2, 0, "Femur Axial Load", "N")
    _mlimit_pos(pm_fem, _Fy_fem, f'yield={_Fy_fem:.0f}N')
    pm_fem.setYRange(0, _Fy_fem * 1.2, padding=0)
    _leg(pm_fem)
    ln_fem_L = pm_fem.plot(pen=pg.mkPen('#60d0ff', width=W), name="L")
    ln_fem_R = pm_fem.plot(pen=pg.mkPen('#80ff80', width=W), name="R")

    pm_tib = _mp(2, 1, "Tibia Axial Load", "N")
    _mlimit_pos(pm_tib, _Fy_tib, f'yield={_Fy_tib:.0f}N')
    pm_tib.setYRange(0, _Fy_tib * 1.2, padding=0)
    _leg(pm_tib)
    ln_tib_L = pm_tib.plot(pen=pg.mkPen('#60d0ff', width=W), name="L")
    ln_tib_R = pm_tib.plot(pen=pg.mkPen('#80ff80', width=W), name="R")

    # ── Row 3: Coupler axial | Hip motor current ─────────────────────────────
    _Fy_coup = _rg.get('F_yield_coupler', 6381.0)
    _I_max_hip = _rg.get('I_max_hip', 20.0)
    _I_max_wheel = _rg.get('I_max_wheel', 50.0)

    pm_coup = _mp(3, 0, "Coupler Axial Load", "N")
    _mlimit_pos(pm_coup, _Fy_coup, f'yield={_Fy_coup:.0f}N')
    pm_coup.setYRange(0, _Fy_coup * 1.2, padding=0)
    _leg(pm_coup)
    ln_coup_L = pm_coup.plot(pen=pg.mkPen('#60d0ff', width=W), name="L")
    ln_coup_R = pm_coup.plot(pen=pg.mkPen('#80ff80', width=W), name="R")

    pm_icur = _mp(3, 1, "Motor Currents vs Limits", "A")
    _mlimits(pm_icur, _I_max_hip, f'{_I_max_hip:.0f}A hip', '#ff8844')
    _mlimits(pm_icur, _I_max_wheel, f'{_I_max_wheel:.0f}A whl', '#ff4444')
    pm_icur.setYRange(-_I_max_wheel * 1.15, _I_max_wheel * 1.15, padding=0)
    _leg(pm_icur, ncols=2)
    ln_mi_hip_L  = pm_icur.plot(pen=pg.mkPen('#ffa040', width=W), name="hip L")
    ln_mi_hip_R  = pm_icur.plot(pen=pg.mkPen('#ffe060', width=W), name="hip R")
    ln_mi_whl_L  = pm_icur.plot(pen=pg.mkPen('#60d0ff', width=W), name="whl L")
    ln_mi_whl_R  = pm_icur.plot(pen=pg.mkPen('#80ff80', width=W), name="whl R")

    mech_lines = [
        ln_brg_a_L, ln_brg_a_R, ln_brg_c_L, ln_brg_c_R,
        ln_brg_e_L, ln_brg_e_R, ln_brg_w_L, ln_brg_w_R,
        ln_fem_L, ln_fem_R, ln_tib_L, ln_tib_R,
        ln_coup_L, ln_coup_R,
        ln_mi_hip_L, ln_mi_hip_R, ln_mi_whl_L, ln_mi_whl_R,
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # IMU TAB — vertical specific force (az) and jump-phase transition thresholds
    # ══════════════════════════════════════════════════════════════════════════
    glw_imu = pg.GraphicsLayoutWidget()
    glw_imu.setBackground("#12121e")
    tab_widget.addTab(glw_imu, "IMU")

    _sg = suspension_gains or {}
    _freefall_az = _sg.get('freefall_az_threshold', 2.0)
    _landing_az  = _sg.get('landing_az_threshold', 15.0)

    def _ip(ttl, ylabel):
        pl = glw_imu.addPlot()
        pl.setTitle(
            f'<span style="color:#e0e0e0;font-size:9pt;font-weight:600">{ttl}</span>')
        pl.setLabel(
            "left",
            f'<span style="color:#c8c8c8;font-size:9pt">{ylabel}</span>')
        pl.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        for ax_name in ("left", "bottom"):
            ax = pl.getAxis(ax_name)
            ax.setTextPen(TICK_PEN)
            ax.setPen(pg.mkPen('#555'))
            ax.setStyle(tickFont=TICK_FONT)
        pl.setXRange(-window_s, 0, padding=0.02)
        return pl

    p_az = _ip("IMU Vertical Specific Force", "m/s²")

    # Nominal on-ground reference (9.81 m/s²)
    p_az.addItem(pg.InfiniteLine(
        pos=9.81, angle=0,
        pen=pg.mkPen('#606060', width=1.0, style=_DASH),
        label='ground ≈9.81',
        labelOpts={'color': '#808080', 'anchors': [(0.05, 1.1), (0.05, 1.1)]}))

    # Freefall threshold (EXTEND→FLYING when az drops below this)
    p_az.addItem(pg.InfiniteLine(
        pos=_freefall_az, angle=0,
        pen=pg.mkPen('#ffe060', width=1.5, style=_DASH),
        label=f'freefall < {_freefall_az:.1f}',
        labelOpts={'color': '#ffe060', 'anchors': [(0.05, 1.1), (0.05, 1.1)]}))

    # Landing threshold (FLYING→LANDING when az spikes above this)
    p_az.addItem(pg.InfiniteLine(
        pos=_landing_az, angle=0,
        pen=pg.mkPen('#ff8844', width=1.5, style=_DASH),
        label=f'landing > {_landing_az:.1f}',
        labelOpts={'color': '#ff8844', 'anchors': [(0.05, -0.1), (0.05, -0.1)]}))

    p_az.setYRange(-2.0, _landing_az * 1.25, padding=0)

    ln_az = p_az.plot(pen=pg.mkPen('#60d0ff', width=LINE_WIDTH), name="az_imu")

    # ── Ring buffers — 29 channels ────────────────────────────────────────────
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
     tau_spring_L_buf, tau_spring_R_buf,
     wheel_h_L_buf, wheel_h_R_buf,
     ) = (deque(maxlen=MAXLEN) for _ in range(29))

    all_bufs = [
        t_buf, pitch_buf, pitch_ref_buf, pitch_rate_buf,
        vel_buf, v_cmd_buf, yaw_rate_buf, omega_cmd_buf,
        hip_L_buf, hip_R_buf, hip_cmd_L_buf, hip_cmd_R_buf, roll_buf,
        tau_whl_L_buf, tau_whl_R_buf,
        v_batt_buf, batt_temp_buf, soc_buf,
        I_whl_L_buf, I_whl_R_buf, I_hip_L_buf, I_hip_R_buf, I_total_buf,
        tau_hip_L_buf, tau_hip_R_buf,
        tau_spring_L_buf, tau_spring_R_buf,
        wheel_h_L_buf, wheel_h_R_buf,
    ]

    all_lines = [
        ln_pitch, ln_pitch_ref, ln_prate,
        ln_vel, ln_vcmd, ln_yaw, ln_ocmd,
        ln_hip_L, ln_hip_R, ln_hip_cmd_L, ln_hip_cmd_R, ln_roll,
        ln_tau_L, ln_tau_R, ln_vbatt, ln_ibat_batt,
        ln_iwhl_L, ln_iwhl_R, ln_ihip_L, ln_ihip_R, ln_itotal,
        ln_htau_L, ln_htau_R,
        ln_htau_spr_L, ln_htau_spr_R,
        ln_wh_L, ln_wh_R, ln_wh_avg,
    ]

    # ── Mechanical tab ring buffers (18 channels) ─────────────────────────────
    (mech_brg_a_L_buf, mech_brg_a_R_buf,
     mech_brg_c_L_buf, mech_brg_c_R_buf,
     mech_brg_e_L_buf, mech_brg_e_R_buf,
     mech_brg_w_L_buf, mech_brg_w_R_buf,
     mech_fem_L_buf, mech_fem_R_buf,
     mech_tib_L_buf, mech_tib_R_buf,
     mech_coup_L_buf, mech_coup_R_buf,
     mech_ihip_L_buf, mech_ihip_R_buf,
     mech_iwhl_L_buf, mech_iwhl_R_buf,
     ) = (deque(maxlen=MAXLEN) for _ in range(18))

    mech_bufs = [
        mech_brg_a_L_buf, mech_brg_a_R_buf,
        mech_brg_c_L_buf, mech_brg_c_R_buf,
        mech_brg_e_L_buf, mech_brg_e_R_buf,
        mech_brg_w_L_buf, mech_brg_w_R_buf,
        mech_fem_L_buf, mech_fem_R_buf,
        mech_tib_L_buf, mech_tib_R_buf,
        mech_coup_L_buf, mech_coup_R_buf,
        mech_ihip_L_buf, mech_ihip_R_buf,
        mech_iwhl_L_buf, mech_iwhl_R_buf,
    ]

    # ── IMU tab ring buffer ───────────────────────────────────────────────────
    az_imu_buf = deque(maxlen=MAXLEN)

    # ── 4-bar force estimation from hip torque + kinematics ──────────────────
    def _estimate_forces(q_hip_deg, tau_hip, tau_whl):
        """Estimate bearing reaction forces and link axial loads [N].

        Uses static torque equilibrium at the hip pivot:
          F_femur ≈ |tau_hip| / L_femur  (axial load in femur)
        and propagates through the 4-bar to estimate loads at each pivot.

        Returns dict with keys: brg_a, brg_c, brg_e, brg_w, fem, tib, coup
        All values are |force| in Newtons.
        """
        if not _has_mech:
            return dict(brg_a=0, brg_c=0, brg_e=0, brg_w=0,
                        fem=0, tib=0, coup=0)

        import math as _m
        q_hip_rad = _m.radians(q_hip_deg)
        L_f = _rg['L_femur']
        L_s = _rg['L_stub']
        L_t = _rg['L_tibia']
        Lc  = _rg['Lc']
        F_X = _rg['F_X']
        F_Z = _rg['F_Z']
        A_Z = _rg['A_Z']
        wheel_r = _rg['wheel_r']

        # Half of body weight on one leg (gravity load)
        g = 9.81
        m_total = (_rg['m_box'] + 2 * _rg['m_femur'] + 2 * _rg['m_tibia']
                   + 2 * _rg['m_coupler'] + 2 * _rg['m_wheel'])
        W_leg = m_total * g / 2.0  # weight per leg

        # IK to get pivot positions
        C_x = -L_f * _m.cos(q_hip_rad)
        C_z = A_Z + L_f * _m.sin(q_hip_rad)
        dx, dz = C_x - F_X, C_z - F_Z
        R = _m.sqrt(dx*dx + dz*dz)
        if R < 1e-9:
            return dict(brg_a=0, brg_c=0, brg_e=0, brg_w=0,
                        fem=0, tib=0, coup=0)

        # Femur: hip torque → axial force along femur
        F_femur = abs(tau_hip) / max(L_f, 0.01) + W_leg * abs(_m.cos(q_hip_rad))

        # Bearing A (hip pivot): sees femur force + half body weight
        F_brg_a = _m.sqrt(F_femur**2 + W_leg**2)

        # Bearing C (knee pivot): femur and tibia meet — vector sum
        # Tibia carries weight + wheel torque reaction
        F_tibia = W_leg + abs(tau_whl) / max(wheel_r, 0.01)
        F_brg_c = _m.sqrt(F_femur**2 + F_tibia**2)

        # Coupler: tension/compression from 4-bar constraint
        # Approximate: torque about C from stub creates coupler force
        F_coupler = F_femur * L_s / max(Lc, 0.01) + W_leg * 0.3

        # Bearing E (stub-coupler joint): sees coupler + stub forces
        F_brg_e = _m.sqrt(F_coupler**2 + (F_tibia * L_s / max(L_t, 0.01))**2)

        # Bearing W (wheel axle): weight + ground reaction + wheel torque
        F_brg_w = W_leg + abs(tau_whl) / max(wheel_r, 0.01)

        return dict(
            brg_a=F_brg_a, brg_c=F_brg_c, brg_e=F_brg_e, brg_w=F_brg_w,
            fem=F_femur, tib=F_tibia, coup=F_coupler)

    i_bat_max      = [0.0]
    _last_stat_t   = [0.0]
    _stat_interval = 1.0 / 3.0   # status labels update at 3 Hz
    _airborne_t    = [None]       # sim-time of last airborne detection
    _landing_t     = [None]       # sim-time of last landing detection
    _prev_mode     = ['BALANCE']  # previous robot_mode for edge detection

    def _reset_bufs():
        for b in all_bufs:
            b.clear()
        for b in mech_bufs:
            b.clear()
        az_imu_buf.clear()
        for ln in all_lines:
            ln.setData([], [])
        for ln in mech_lines:
            ln.setData([], [])
        ln_az.setData([], [])
        i_bat_max[0] = 0.0
        for lbl in (lbl_soc, lbl_temp, lbl_ibat, lbl_ibat_max):
            lbl.setText(lbl.text().split(":")[0] + ": --")
        _airborne_t[0] = None; _landing_t[0] = None; _prev_mode[0] = 'BALANCE'
        _ln_air.hide(); _ln_land.hide()

    # ── 60 Hz update callback ─────────────────────────────────────────────────
    def _update():
        # Drain queue — fill buffers, track running i_bat_max
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
                _fitness_overlay.hide()
                continue
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "TITLE":
                main_win.setWindowTitle(item[1])
                continue
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "CONFIG_UPDATE":
                new_cd = item[1]
                for key, cb in _ctrl_widgets.items():
                    cb.blockSignals(True)
                    cb.setChecked(new_cd.get(key, True))
                    cb.blockSignals(False)
                hip_pd = not new_cd.get("Suspension", True)
                _btn_hip.blockSignals(True)
                _btn_hip.setChecked(hip_pd)
                _btn_hip.setText("Hip: Pos PD" if hip_pd else "Hip: Impedance")
                _btn_hip.blockSignals(False)
                continue
            if (isinstance(item, tuple) and len(item) == 4
                    and item[0] == "FITNESS"):
                _show_fitness_breakdown(item[1], item[2], item[3])
                continue

            (t, pitch, pitch_ref, pitch_rate,
             vel, v_cmd, yaw_rate, omega_cmd,
             hip_L, hip_R, hip_cmd_L, hip_cmd_R, roll,
             tau_whl_L, tau_whl_R,
             v_batt, batt_temp, soc,
             I_whl_L, I_whl_R, I_hip_L, I_hip_R, I_total,
             tau_hip_L, tau_hip_R,
             tau_spring_L, tau_spring_R,
             wh_L_mm, wh_R_mm,
             robot_mode, az_imu) = item

            t_buf.append(t)
            pitch_buf.append(pitch);          pitch_ref_buf.append(pitch_ref)
            pitch_rate_buf.append(pitch_rate)
            vel_buf.append(vel);              v_cmd_buf.append(v_cmd)
            yaw_rate_buf.append(yaw_rate);    omega_cmd_buf.append(omega_cmd)
            hip_L_buf.append(hip_L);          hip_R_buf.append(hip_R)
            hip_cmd_L_buf.append(hip_cmd_L);  hip_cmd_R_buf.append(hip_cmd_R)
            roll_buf.append(roll)
            tau_whl_L_buf.append(tau_whl_L);  tau_whl_R_buf.append(tau_whl_R)
            v_batt_buf.append(v_batt);        batt_temp_buf.append(batt_temp)
            soc_buf.append(soc)
            I_whl_L_buf.append(I_whl_L);      I_whl_R_buf.append(I_whl_R)
            I_hip_L_buf.append(I_hip_L);      I_hip_R_buf.append(I_hip_R)
            I_total_buf.append(I_total)
            i_bat_max[0] = max(i_bat_max[0], I_total)
            tau_hip_L_buf.append(tau_hip_L);  tau_hip_R_buf.append(tau_hip_R)
            tau_spring_L_buf.append(tau_spring_L);  tau_spring_R_buf.append(tau_spring_R)
            wheel_h_L_buf.append(wh_L_mm);          wheel_h_R_buf.append(wh_R_mm)
            az_imu_buf.append(az_imu)

            # ── Airborne / landing edge detection from robot's own mode ──
            if robot_mode == 'FLYING' and _prev_mode[0] != 'FLYING':
                _airborne_t[0] = t
                _landing_t[0]  = None
            elif robot_mode == 'LANDING' and _prev_mode[0] == 'FLYING':
                _landing_t[0] = t
            _prev_mode[0] = robot_mode

            # ── Mechanical force estimation ──────────────────────────────
            if _has_mech:
                fl = _estimate_forces(hip_L, tau_hip_L, tau_whl_L)
                fr = _estimate_forces(hip_R, tau_hip_R, tau_whl_R)
                mech_brg_a_L_buf.append(fl['brg_a']); mech_brg_a_R_buf.append(fr['brg_a'])
                mech_brg_c_L_buf.append(fl['brg_c']); mech_brg_c_R_buf.append(fr['brg_c'])
                mech_brg_e_L_buf.append(fl['brg_e']); mech_brg_e_R_buf.append(fr['brg_e'])
                mech_brg_w_L_buf.append(fl['brg_w']); mech_brg_w_R_buf.append(fr['brg_w'])
                mech_fem_L_buf.append(fl['fem']);   mech_fem_R_buf.append(fr['fem'])
                mech_tib_L_buf.append(fl['tib']);   mech_tib_R_buf.append(fr['tib'])
                mech_coup_L_buf.append(fl['coup']); mech_coup_R_buf.append(fr['coup'])
                mech_ihip_L_buf.append(I_hip_L);   mech_ihip_R_buf.append(I_hip_R)
                mech_iwhl_L_buf.append(I_whl_L);   mech_iwhl_R_buf.append(I_whl_R)

        if len(t_buf) < 2:
            return

        # Status labels — throttled to 3 Hz
        now = _time.perf_counter()
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

        def _a(buf):
            return np.array(buf)[idx:]

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
        ln_htau_spr_L.setData(xw, _a(tau_spring_L_buf))
        ln_htau_spr_R.setData(xw, _a(tau_spring_R_buf))

        wh_L_arr = _a(wheel_h_L_buf)
        wh_R_arr = _a(wheel_h_R_buf)
        wh_avg   = (wh_L_arr + wh_R_arr) * 0.5
        ln_wh_L.setData(xw, wh_L_arr)
        ln_wh_R.setData(xw, wh_R_arr)
        ln_wh_avg.setData(xw, wh_avg)
        _wh_hi = max(float(wh_L_arr.max()) if len(wh_L_arr) else 0.0,
                     float(wh_R_arr.max()) if len(wh_R_arr) else 0.0, 10.0)
        p_wh.setYRange(0.0, _wh_hi * 1.1, padding=0)
        # Move airborne/landing markers with the scrolling window
        for _ln_marker, _mt in ((_ln_air, _airborne_t[0]), (_ln_land, _landing_t[0])):
            if _mt is not None:
                _xm = _mt - sim_t
                if -window_s <= _xm <= 0:
                    _ln_marker.setValue(_xm); _ln_marker.show()
                else:
                    _ln_marker.hide()
            else:
                _ln_marker.hide()

        _sync_batt()

        # ── Update Mechanical tab charts ─────────────────────────────────────
        if _has_mech and len(mech_brg_a_L_buf) > 1:
            ln_brg_a_L.setData(xw, _a(mech_brg_a_L_buf))
            ln_brg_a_R.setData(xw, _a(mech_brg_a_R_buf))
            ln_brg_c_L.setData(xw, _a(mech_brg_c_L_buf))
            ln_brg_c_R.setData(xw, _a(mech_brg_c_R_buf))
            ln_brg_e_L.setData(xw, _a(mech_brg_e_L_buf))
            ln_brg_e_R.setData(xw, _a(mech_brg_e_R_buf))
            ln_brg_w_L.setData(xw, _a(mech_brg_w_L_buf))
            ln_brg_w_R.setData(xw, _a(mech_brg_w_R_buf))
            ln_fem_L.setData(xw, _a(mech_fem_L_buf))
            ln_fem_R.setData(xw, _a(mech_fem_R_buf))
            ln_tib_L.setData(xw, _a(mech_tib_L_buf))
            ln_tib_R.setData(xw, _a(mech_tib_R_buf))
            ln_coup_L.setData(xw, _a(mech_coup_L_buf))
            ln_coup_R.setData(xw, _a(mech_coup_R_buf))
            ln_mi_hip_L.setData(xw, _a(mech_ihip_L_buf))
            ln_mi_hip_R.setData(xw, _a(mech_ihip_R_buf))
            ln_mi_whl_L.setData(xw, _a(mech_iwhl_L_buf))
            ln_mi_whl_R.setData(xw, _a(mech_iwhl_R_buf))

        # ── Update IMU tab chart ──────────────────────────────────────────────
        if len(az_imu_buf) > 1:
            ln_az.setData(xw, _a(az_imu_buf))

        # Auto-fit Y to data, but enforce minimum span of 2 units.
        # Compute data range from all curves in each plot, then expand
        # symmetrically around the midpoint if the span is too small.
        _MIN_Y_SPAN = 2.0
        _fixed_range_plots = {id(p_tau), id(p_htau), id(p_hip), id(p_yaw), id(p_wh)}
        for _, pl in named_plots:
            if id(pl) in _fixed_range_plots:
                continue
            # Gather min/max across all PlotDataItems in this plot
            d_lo, d_hi = float('inf'), float('-inf')
            for item in pl.listDataItems():
                yd = item.yData
                if yd is not None and len(yd) > 0:
                    d_lo = min(d_lo, float(yd.min()))
                    d_hi = max(d_hi, float(yd.max()))
            if d_lo > d_hi:          # no data yet
                continue
            span = d_hi - d_lo
            if span < _MIN_Y_SPAN:
                mid = (d_lo + d_hi) * 0.5
                half = _MIN_Y_SPAN * 0.5
                pl.setYRange(mid - half, mid + half, padding=0)
            else:
                pl.setYRange(d_lo, d_hi, padding=0.05)

    timer = QtCore.QTimer()
    timer.setInterval(int(1000 / TELEMETRY_HZ))
    timer.timeout.connect(_update)
    timer.start()

    app.exec()


def _vel_pi_gains_dict(params) -> dict:
    """Extract VelocityPI gains as a plain dict for passing to plot process."""
    g = params.gains.velocity_pi
    return dict(Kp=g.Kp, Ki=g.Ki, Kff=g.Kff, theta_max=g.theta_max,
                theta_ref_rate_limit=g.theta_ref_rate_limit)


def _lqr_gains_dict(params) -> dict:
    """Extract LQR cost weights as a plain dict for passing to plot process."""
    g = params.gains.lqr
    return dict(Q_pitch=g.Q_pitch, Q_pitch_rate=g.Q_pitch_rate, R=g.R)


def _yaw_pi_gains_dict(params) -> dict:
    """Extract YawPI gains as a plain dict for passing to plot process."""
    g = params.gains.yaw_pi
    return dict(Kp=g.Kp, Ki=g.Ki, torque_max=g.torque_max, int_max=g.int_max,
                omega_cmd_max=g.omega_cmd_max)


def _suspension_gains_dict(params) -> dict:
    """Extract Suspension + Roll gains as a plain dict for passing to plot process."""
    g = params.gains.suspension
    return dict(K_s=g.K_s, B_s=g.B_s, K_roll=g.K_roll, D_roll=g.D_roll,
                freefall_az_threshold=g.freefall_az_threshold,
                landing_az_threshold=g.landing_az_threshold)


def _hip_limits_dict(params) -> dict:
    """Extract Q_EXT / Q_RET as a plain dict for passing to plot process."""
    r = params.robot
    return dict(Q_EXT=r.Q_EXT, Q_RET=r.Q_RET)


def _robot_geom_dict(params) -> dict:
    """Extract robot geometry + motor params as a plain dict for the plot process."""
    r = params.robot
    m = params.motors
    return dict(
        L_femur=r.L_femur, L_stub=r.L_stub, L_tibia=r.L_tibia,
        Lc=r.Lc, F_X=r.F_X, F_Z=r.F_Z, A_Z=r.A_Z,
        wheel_r=r.wheel_r, leg_y=r.leg_y,
        m_box=r.m_box, m_femur=r.m_femur, m_tibia=r.m_tibia,
        m_coupler=r.m_coupler, m_wheel=r.m_wheel,
        Kt_hip=m.hip.Kt_output, Kt_wheel=m.wheel.Kt,
        I_max_hip=params.limits.max_motor_current_hip,
        I_max_wheel=params.limits.max_motor_current_wheel,
        # Bearing static ratings [N] from COMPONENTS.md
        C0_608=1370.0, C0_6001=2850.0,
        # Tube yield force [N] — 6061-T6, SF=1.0 at these peak loads
        # (from COMPONENTS.md: femur 920N × SF2.29, tibia 1234N × SF2.36, coupler 1102N × SF5.79)
        F_yield_femur=920.0 * 2.29,    # ~2107 N
        F_yield_tibia=1234.0 * 2.36,   # ~2912 N
        F_yield_coupler=1102.0 * 5.79, # ~6381 N
    )


def _ctrl_defaults_from_cfg(cfg) -> dict:
    """Build checkbox default states from a ScenarioConfig (or None for sandbox)."""
    if cfg is None:
        # Sandbox: everything enabled
        return {"LQR": True, "VelPI": True, "YawPI": True,
                "Suspension": True, "RollLev": True, "HipFF": True,
                "GravFF": True, "CoMFF": True, "TurnFF": True,
                "KneeSpr": False}
    ac = cfg.active_controllers
    imp = cfg.hip_mode in ("impedance", "jump")
    return {
        "LQR":        "lqr" in ac,
        "VelPI":      "velocity_pi" in ac,
        "YawPI":      "yaw_pi" in ac,
        "Suspension": imp,
        "RollLev":    imp,
        "HipFF":      True,
        "GravFF":     True,
        "CoMFF":      True,
        "TurnFF":     True,
        "KneeSpr":    cfg.hip_mode == "jump",
    }


def sandbox(rng_seed: int = 0):
    """Launch sandbox mode — MuJoCo viewer + pyqtgraph telemetry (dual-process).

    Main process: MuJoCo physics + control loop + gamepad
    Child process: pyqtgraph 10-panel telemetry window

    Uses SimController.tick() — the SAME control cascade as scenarios/optimizer.
    """
    from master_sim_jump.defaults import DEFAULT_PARAMS
    from master_sim_jump.sim_loop import (
        build_model_and_data, init_sim, SimController,
        get_pitch_and_rate,
    )
    from master_sim_jump.physics import get_equilibrium_pitch
    from master_sim_jump.controllers.jump import RobotMode

    params = DEFAULT_PARAMS
    robot = params.robot
    world = sandbox_world()

    print("  Building sandbox arena ...")
    print(f"    Obstacles: {len(SANDBOX_OBSTACLES)}")
    print(f"    Props:     {len(SANDBOX_PROPS)}")
    print(f"    Floor:     {world.floor_size[0]:.0f} x {world.floor_size[1]:.0f} m")

    model, data = build_model_and_data(params, world)
    init_sim(model, data, params)

    RENDER_HZ = 60
    PHYSICS_HZ = round(1.0 / model.opt.timestep)
    steps_per_frame = max(1, PHYSICS_HZ // RENDER_HZ)
    ctrl_steps = params.timing.ctrl_steps

    # ── Controller — single source of truth (shared with scenarios) ──────────
    ctrl = SimController(model, data, params, rng_seed=rng_seed)

    step = 0

    # Controller enable flags (toggled via plot process UI)
    _en_lqr        = [True]
    _en_vel_pi     = [True]
    _en_yaw_pi     = [True]
    _en_suspension = [True]
    _en_roll_lev   = [True]
    _en_ff1        = [True]
    _en_ff2        = [True]
    _en_ff3        = [True]
    _en_ff4        = [True]
    _en_knee_spr   = [False]
    _hip_mode      = ["impedance"]

    # Command targets (v, omega, hip %)
    _v_desired     = [0.0]
    _omega_desired = [0.0]
    _HIP_MIN_Q     = robot.Q_EXT
    _HIP_MAX_Q     = robot.Q_RET
    _HIP_NOM_PCT   = (robot.Q_NOM - _HIP_MIN_Q) / (_HIP_MAX_Q - _HIP_MIN_Q) * 100.0
    _hip_pct       = [_HIP_NOM_PCT]
    _jump_active   = [False]

    def _hip_target():
        pct = max(0.0, min(100.0, _hip_pct[0])) / 100.0
        return _HIP_MIN_Q + pct * (_HIP_MAX_Q - _HIP_MIN_Q)

    # Last tick data (for telemetry push between control steps)
    _last_tick = {}
    last_push  = -1.0

    # ── Gamepad init ─────────────────────────────────────────────────────────
    JOY_DEADZONE = 0.08
    JOY_HIP_SATURATION = 0.85  # physical stick rarely reaches ±1.0; treat ±this as full range
    _joy = None
    _joy_axes = {}

    try:
        import os as _os
        _os.environ['SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS'] = '1'
        import pygame
        pygame.display.init()
        pygame.display.set_mode((1, 1), pygame.NOFRAME)
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            _joy = pygame.joystick.Joystick(0)
            _joy.init()
            print(f"  Joystick: {_joy.get_name()}")
            print(f"    Axes: {_joy.get_numaxes()}  Buttons: {_joy.get_numbuttons()}")
        else:
            print("  Joystick: none detected — sliders only")
    except ImportError:
        print("  pygame not installed — sliders only")

    has_gamepad = _joy is not None

    # ── Launch plot process ───────────────────────────────────────────────────
    wheel_limit = params.motors.wheel.torque_limit
    hip_limit   = params.motors.hip.impedance_torque_limit
    _vp_init = _vel_pi_gains_dict(params)
    _lqr_init = _lqr_gains_dict(params)
    _yaw_init = _yaw_pi_gains_dict(params)
    _susp_init = _suspension_gains_dict(params)
    _hl_init = _hip_limits_dict(params)
    _rg_init = _robot_geom_dict(params)

    data_q = mp.Queue(maxsize=12000)
    cmd_q  = mp.Queue(maxsize=32)
    _ctrl_init_sb = _ctrl_defaults_from_cfg(None)
    plot_proc = mp.Process(
        target=_plot_process,
        args=(data_q, cmd_q, WINDOW_S, "Sandbox — free drive",
              wheel_limit, hip_limit, has_gamepad, _vp_init, _lqr_init,
              _ctrl_init_sb, _yaw_init, _susp_init, _hl_init, _rg_init),
        daemon=True)
    plot_proc.start()
    print(f"  Plot process started (PID {plot_proc.pid})")

    # Position MuJoCo viewer to right half of screen
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
        sw = ctypes.windll.user32.GetSystemMetrics(0)
        sh = ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        sw, sh = 1920, 1080

    print("  Launching MuJoCo viewer — robot balancing at standstill ...")

    def _reset_state():
        nonlocal step, last_push, _last_tick
        mujoco.mj_resetData(model, data)
        init_sim(model, data, params)
        ctrl.reset(model, data)
        step = 0
        _v_desired[0] = 0.0; _omega_desired[0] = 0.0
        _hip_pct[0] = _HIP_NOM_PCT
        _jump_active[0] = False
        _last_tick = {}; last_push = -1.0
        if not data_q.full():
            data_q.put_nowait("RESET")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 35
        viewer.cam.elevation = -15
        viewer.cam.distance  = 2.5
        viewer.cam.lookat    = np.array([0.0, 0.0, 0.30])
        _add_world_axes(viewer)

        # Snap viewer to right half (Windows, with mid-padding)
        def _snap():
            time.sleep(1.5)
            try:
                import ctypes as _ct
                _EnumProc = _ct.WINFUNCTYPE(_ct.c_bool, _ct.c_int, _ct.c_int)
                half = sw // 2
                mj_x = half + PAD_MID // 2
                mj_w = sw - mj_x
                def _cb(hwnd, _):
                    if _ct.windll.user32.IsWindowVisible(hwnd):
                        buf = _ct.create_unicode_buffer(256)
                        _ct.windll.user32.GetWindowTextW(hwnd, buf, 256)
                        if "mujoco" in buf.value.lower():
                            SWP = 0x0004
                            _ct.windll.user32.SetWindowPos(
                                hwnd, 0, mj_x, 0, mj_w, sh, SWP)
                            return False
                    return True
                _ct.windll.user32.EnumWindows(_EnumProc(_cb), 0)
            except Exception:
                pass
        threading.Thread(target=_snap, daemon=True).start()

        while viewer.is_running():
            frame_start = time.perf_counter()
            sim_t = float(data.time)

            # ── Drain command queue from plot process ──
            try:
                cmd = cmd_q.get_nowait()
                if cmd == "RESTART":
                    _reset_state()
                elif isinstance(cmd, tuple):
                    if cmd[0] == "V_DESIRED":
                        _v_desired[0] = float(cmd[1])
                    elif cmd[0] == "OMEGA_DESIRED":
                        _omega_desired[0] = float(cmd[1])
                    elif cmd[0] == "HIP_PCT":
                        _hip_pct[0] = float(cmd[1])
                    elif cmd[0] == "CTRL_EN":
                        key, val = cmd[1], cmd[2]
                        if   key == "LQR":        _en_lqr[0]        = val
                        elif key == "VelPI":      _en_vel_pi[0]     = val
                        elif key == "YawPI":      _en_yaw_pi[0]     = val
                        elif key == "Suspension": _en_suspension[0] = val
                        elif key == "RollLev":    _en_roll_lev[0]   = val
                        elif key == "HipFF":      _en_ff1[0]        = val
                        elif key == "GravFF":     _en_ff2[0]        = val
                        elif key == "CoMFF":      _en_ff3[0]        = val
                        elif key == "TurnFF":     _en_ff4[0]        = val
                        elif key == "KneeSpr":    _en_knee_spr[0]   = val
                    elif cmd[0] == "HIP_MODE":
                        _hip_mode[0] = cmd[1]
                    elif cmd[0] == "GAIN_VP":
                        ctrl.update_velocity_pi_gain(cmd[1], cmd[2])
                    elif cmd[0] == "GAIN_LQR":
                        ctrl.update_lqr_gain(cmd[1], cmd[2])
                    elif cmd[0] == "GAIN_YAW":
                        ctrl.update_yaw_pi_gain(cmd[1], cmd[2])
                    elif cmd[0] == "GAIN_SUSP":
                        ctrl.update_suspension_gain(cmd[1], cmd[2])
                    elif cmd[0] == "ROBOT_GEOM":
                        ctrl.update_robot_geom(cmd[1], cmd[2])
                    elif cmd[0] == "JUMP":
                        _jump_active[0] = True
                        ctrl.jump_ctrl.trigger()
            except Exception:
                pass

            # ── Gamepad polling (event-based — direct read needs focus on Windows) ──
            if _joy is not None:
                for _ev in pygame.event.get():
                    if _ev.type == pygame.JOYAXISMOTION:
                        _joy_axes[_ev.axis] = _ev.value
                    elif _ev.type == pygame.JOYBUTTONDOWN and _ev.button == 0:
                        _jump_active[0] = True
                        ctrl.jump_ctrl.trigger()

                raw_v = _joy_axes.get(1, 0.0)   # left stick Y
                raw_w = _joy_axes.get(2, 0.0)   # right stick X
                raw_h = _joy_axes.get(3, 0.0)   # right stick Y

                _v_desired[0] = (-raw_v * 3.0) if abs(raw_v) > JOY_DEADZONE else 0.0
                _omega_desired[0] = (-raw_w * 5.0) if abs(raw_w) > JOY_DEADZONE else 0.0
                if abs(raw_h) > JOY_DEADZONE:
                    clamped_h = max(-JOY_HIP_SATURATION, min(JOY_HIP_SATURATION, raw_h))
                    _hip_pct[0] = (JOY_HIP_SATURATION + clamped_h) / (2.0 * JOY_HIP_SATURATION) * 100.0

            for _ in range(steps_per_frame):
                if step % ctrl_steps == 0:
                    # ── Single source of truth: SimController.tick() ──
                    tick = ctrl.tick(
                        model, data,
                        v_target_ms=_v_desired[0],
                        omega_target=_omega_desired[0],
                        q_hip_target=_hip_target(),
                        use_lqr=_en_lqr[0],
                        use_velocity_pi=_en_vel_pi[0],
                        use_yaw_pi=_en_yaw_pi[0],
                        use_impedance=(_hip_mode[0] != "position_pd"),
                        use_roll_leveling=_en_roll_lev[0],
                        use_suspension=_en_suspension[0],
                        use_ff1=_en_ff1[0],
                        use_ff2=_en_ff2[0],
                        use_ff3=_en_ff3[0],
                        use_ff4=_en_ff4[0],
                        use_knee_spring=_en_knee_spr[0],
                        jump_active=_jump_active[0])

                    # Auto-clear jump flag when jump controller returns to BALANCE
                    if _jump_active[0] and ctrl.jump_ctrl.mode == RobotMode.BALANCE:
                        _jump_active[0] = False

                    _last_tick = tick

                    # ── Fall check → auto-reset ──
                    if tick['fell']:
                        _reset_state()
                        break

                mujoco.mj_step(model, data)
                step += 1

            # Camera follow
            viewer.cam.lookat[0] = data.xpos[ctrl.box_bid][0]
            viewer.cam.lookat[1] = data.xpos[ctrl.box_bid][1]
            # Reset dynamic geoms (keep static world-axes)
            viewer.user_scn.ngeom = _N_STATIC_GEOMS
            # Always-visible knee spring spheres (grey idle, purple when active)
            _add_knee_spring_sphere(viewer, data, ctrl.tibia_bid_L,
                active=abs(_last_tick.get('tau_spring_L', 0.0)) > 1e-6 if _last_tick else False)
            _add_knee_spring_sphere(viewer, data, ctrl.tibia_bid_R,
                active=abs(_last_tick.get('tau_spring_R', 0.0)) > 1e-6 if _last_tick else False)
            viewer.sync()

            # ── Push 27-value telemetry tuple at ~60 Hz ──
            wall_now = time.perf_counter()
            if (wall_now - last_push >= 1.0 / TELEMETRY_HZ
                    and not data_q.full() and _last_tick):
                tk = _last_tick
                # Re-read pitch for freshest display value
                pitch_true_now, _ = get_pitch_and_rate(
                    data, ctrl.box_bid, ctrl.d_pitch)
                pitch_ref_display = tk['pitch_ff'] + tk['theta_ref']
                data_q.put_nowait((
                    sim_t,
                    math.degrees(pitch_true_now),
                    math.degrees(pitch_ref_display),
                    math.degrees(tk['pitch_rate']),
                    tk['v_measured'],
                    _v_desired[0],
                    math.degrees(tk['yaw_rate']),
                    math.degrees(_omega_desired[0]),
                    math.degrees(tk['hip_q_L']),
                    math.degrees(tk['hip_q_R']),
                    math.degrees(tk['q_nom_L']),
                    math.degrees(tk['q_nom_R']),
                    math.degrees(tk['roll']),
                    tk['tau_whl_L'],
                    tk['tau_whl_R'],
                    tk['v_batt'],
                    tk['batt_temp'],
                    tk['batt_soc'],
                    abs(tk['tau_whl_L']) / params.motors.wheel.Kt,
                    abs(tk['tau_whl_R']) / params.motors.wheel.Kt,
                    abs(tk['tau_hip_L']) / params.motors.hip.Kt_output,
                    abs(tk['tau_hip_R']) / params.motors.hip.Kt_output,
                    tk['i_total'],
                    tk['tau_hip_L'],
                    tk['tau_hip_R'],
                    tk['tau_spring_L'],
                    tk['tau_spring_R'],
                    max(tk['wheel_z_L'] - params.robot.wheel_r, 0.0) * 1000.0,
                    max(tk['wheel_z_R'] - params.robot.wheel_r, 0.0) * 1000.0,
                    tk.get('mode', 'BALANCE'),
                    tk.get('az_imu', 0.0),
                ))
                last_push = wall_now

            elapsed = time.perf_counter() - frame_start
            sleep_t = 1.0 / RENDER_HZ - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    # Clean up
    if _joy is not None:
        pygame.quit()
    data_q.put(None)
    plot_proc.join(timeout=2)
    if plot_proc.is_alive():
        plot_proc.terminate()
    print("  Sandbox closed.")
    os._exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED MODE — single function for both replay & sandbox, with live switching
# ═══════════════════════════════════════════════════════════════════════════════

def run_unified(initial_scenario: str = "sandbox",
                switch_q: mp.Queue = None,
                rng_seed: int = 0):
    """Run the visualizer with optional live scenario switching.

    Parameters
    ----------
    initial_scenario : "sandbox" or a key in SCENARIOS (e.g. "s01_lqr_pitch_step")
    switch_q         : mp.Queue receiving ("SWITCH", scenario_key) from launcher.
                       If None, no external switching (backward-compat CLI usage).
    rng_seed         : reproducible noise seed
    """
    from master_sim_jump.defaults import DEFAULT_PARAMS
    from master_sim_jump.params import SimParams
    from master_sim_jump.scenarios import SCENARIOS
    from master_sim_jump.sim_loop import (
        build_model_and_data, init_sim, SimController, get_pitch_and_rate,
        apply_disturbance,
    )
    from master_sim_jump.controllers.jump import RobotMode

    params = DEFAULT_PARAMS
    robot  = params.robot

    # ── Gamepad init (once, survives all switches) ────────────────────────────
    JOY_DEADZONE = 0.08
    JOY_HIP_SATURATION = 0.85  # physical stick rarely reaches ±1.0; treat ±this as full range
    _joy = None
    _joy_axes = {}

    try:
        import os as _os
        _os.environ['SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS'] = '1'
        import pygame
        pygame.display.init()
        pygame.display.set_mode((1, 1), pygame.NOFRAME)
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            _joy = pygame.joystick.Joystick(0)
            _joy.init()
            print(f"  Joystick: {_joy.get_name()}")
        else:
            print("  Joystick: none detected — sliders only")
    except ImportError:
        print("  pygame not installed — sliders only")

    has_gamepad = _joy is not None

    # ── Screen metrics (once) ─────────────────────────────────────────────────
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
        sw = ctypes.windll.user32.GetSystemMetrics(0)
        sh = ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        sw, sh = 1920, 1080

    # ── Helper: resolve scenario key → (cfg_or_None, world, title) ────────────
    def _resolve(key):
        """Return (cfg, world, title).  cfg is None for sandbox."""
        if key == "sandbox":
            return None, sandbox_world(), "Sandbox — free drive"
        cfg = SCENARIOS[key]
        return cfg, cfg.world, f"Replay — {cfg.display_name}"

    # ── Launch plot process ONCE (survives all switches) ──────────────────────
    wheel_limit = params.motors.wheel.torque_limit
    hip_limit   = params.motors.hip.impedance_torque_limit
    _vp_init = _vel_pi_gains_dict(params)
    _lqr_init = _lqr_gains_dict(params)
    _yaw_init = _yaw_pi_gains_dict(params)
    _susp_init = _suspension_gains_dict(params)
    _hl_init = _hip_limits_dict(params)
    _rg_init = _robot_geom_dict(params)

    data_q = mp.Queue(maxsize=12000)
    cmd_q  = mp.Queue(maxsize=32)

    init_cfg, _, init_title = _resolve(initial_scenario)
    _ctrl_init = _ctrl_defaults_from_cfg(init_cfg)
    plot_proc = mp.Process(
        target=_plot_process,
        args=(data_q, cmd_q, WINDOW_S, init_title,
              wheel_limit, hip_limit, has_gamepad, _vp_init, _lqr_init,
              _ctrl_init, _yaw_init, _susp_init, _hl_init, _rg_init),
        daemon=True)
    plot_proc.start()
    print(f"  Plot process started (PID {plot_proc.pid})")

    # ── Mutable state shared across switches ──────────────────────────────────
    current_key = initial_scenario

    # Sandbox-mode interactive targets (initialised from scenario defaults)
    _en_lqr        = [_ctrl_init.get("LQR", True)]
    _en_vel_pi     = [_ctrl_init.get("VelPI", True)]
    _en_yaw_pi     = [_ctrl_init.get("YawPI", True)]
    _en_suspension = [_ctrl_init.get("Suspension", True)]
    _en_roll_lev   = [_ctrl_init.get("RollLev", True)]
    _en_ff1        = [_ctrl_init.get("HipFF", True)]
    _en_ff2        = [_ctrl_init.get("GravFF", True)]
    _en_ff3        = [_ctrl_init.get("CoMFF", True)]
    _en_ff4        = [_ctrl_init.get("TurnFF", True)]
    _en_knee_spr   = [_ctrl_init.get("KneeSpr", False)]
    _hip_mode      = ["position_pd" if not _ctrl_init.get("Suspension", True) else "impedance"]
    _v_desired     = [0.0]
    _omega_desired = [0.0]
    _HIP_MIN_Q     = robot.Q_EXT
    _HIP_MAX_Q     = robot.Q_RET
    _HIP_NOM_PCT   = (robot.Q_NOM - _HIP_MIN_Q) / (_HIP_MAX_Q - _HIP_MIN_Q) * 100.0
    _hip_pct       = [_HIP_NOM_PCT]
    _jump_active   = [False]

    def _hip_target():
        pct = max(0.0, min(100.0, _hip_pct[0])) / 100.0
        return _HIP_MIN_Q + pct * (_HIP_MAX_Q - _HIP_MIN_Q)

    def _reset_sandbox_targets():
        _v_desired[0] = 0.0
        _omega_desired[0] = 0.0
        _hip_pct[0] = _HIP_NOM_PCT
        _jump_active[0] = False

    # ── OUTER LOOP — one iteration per world ──────────────────────────────────
    user_quit = False

    while not user_quit:
        # Reload params.py on every world rebuild so optimizer changes are picked up
        import importlib
        import master_sim_jump.params as _params_mod
        importlib.reload(_params_mod)
        params = _params_mod.SimParams()

        cfg, world, title = _resolve(current_key)

        print(f"  Building world for: {title}")
        model, data = build_model_and_data(params, world)
        init_sim(model, data, params)
        if cfg and cfg.init_fn:
            cfg.init_fn(model, data, params)

        RENDER_HZ    = 60
        PHYSICS_HZ   = round(1.0 / model.opt.timestep)
        steps_per_frame_normal = max(1, PHYSICS_HZ // RENDER_HZ)
        steps_per_frame_slow   = max(1, steps_per_frame_normal // 8)
        steps_per_frame = steps_per_frame_normal
        ctrl_steps   = params.timing.ctrl_steps

        ctrl = SimController(model, data, params, rng_seed=rng_seed)

        # Replay-mode state
        if cfg is not None:
            v_profile_fn     = cfg.v_profile or (lambda t: 0.0)
            theta_ref_fn     = cfg.theta_ref_profile
            omega_profile_fn = cfg.omega_profile
            hip_profile_fn   = cfg.hip_profile
            hip_vel_fn       = cfg.hip_vel_profile
            flags = cfg.tick_flags

        step         = 0
        _last_tick   = {}
        last_push    = -1.0
        scenario_logged = False
        sim_fell     = False
        world_switch = False     # set True to break inner loop for new world
        _slowmo_liftoff_t = [None]   # sim-time when FLYING started (mutable container)
        _slowmo_landing_t = [None]   # sim-time when robot landed after a jump

        def _reset_sim():
            nonlocal step, _last_tick, last_push, scenario_logged, sim_fell
            mujoco.mj_resetData(model, data)
            init_sim(model, data, params)
            if cfg and cfg.init_fn:
                cfg.init_fn(model, data, params)
            ctrl.reset(model, data)
            step = 0
            _last_tick = {}
            last_push = -1.0
            scenario_logged = False
            sim_fell = False
            _slowmo_liftoff_t[0] = None
            _slowmo_landing_t[0] = None
            _reset_sandbox_targets()
            if not data_q.full():
                data_q.put_nowait("RESET")

        # Send title + reset to plot process
        if not data_q.full():
            data_q.put_nowait("RESET")
        if not data_q.full():
            data_q.put_nowait(("TITLE", title))

        print(f"  Launching MuJoCo viewer ...")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth   = 35
            viewer.cam.elevation = -15
            viewer.cam.distance  = 2.5
            viewer.cam.lookat    = np.array([0.0, 0.0, 0.30])
            _add_world_axes(viewer)

            # Snap viewer to right half (Windows, with mid-padding)
            def _snap():
                time.sleep(1.5)
                try:
                    import ctypes as _ct
                    _EnumProc = _ct.WINFUNCTYPE(_ct.c_bool, _ct.c_int, _ct.c_int)
                    half = sw // 2
                    mj_x = half + PAD_MID // 2
                    mj_w = sw - mj_x
                    def _cb(hwnd, _):
                        if _ct.windll.user32.IsWindowVisible(hwnd):
                            buf = _ct.create_unicode_buffer(256)
                            _ct.windll.user32.GetWindowTextW(hwnd, buf, 256)
                            if "mujoco" in buf.value.lower():
                                SWP = 0x0004
                                _ct.windll.user32.SetWindowPos(
                                    hwnd, 0, mj_x, 0, mj_w, sh, SWP)
                                return False
                        return True
                    _ct.windll.user32.EnumWindows(_EnumProc(_cb), 0)
                except Exception:
                    pass
            threading.Thread(target=_snap, daemon=True).start()

            # ── INNER LOOP — frames within one world ──────────────────────
            while viewer.is_running():
                frame_start = time.perf_counter()

                # ── Drain switch_q (from launcher) ──
                if switch_q is not None:
                    try:
                        while True:
                            sw_cmd = switch_q.get_nowait()
                            if (isinstance(sw_cmd, tuple) and len(sw_cmd) == 2
                                    and sw_cmd[0] == "SWITCH"):
                                new_key = sw_cmd[1]
                                if new_key == current_key:
                                    # Same scenario — just restart
                                    _reset_sim()
                                    continue

                                new_cfg, new_world, new_title = _resolve(new_key)

                                if new_world == world:
                                    # ── SAME world — instant switch ──
                                    current_key = new_key
                                    cfg = new_cfg
                                    if cfg is not None:
                                        v_profile_fn     = cfg.v_profile or (lambda t: 0.0)
                                        theta_ref_fn     = cfg.theta_ref_profile
                                        omega_profile_fn = cfg.omega_profile
                                        hip_profile_fn   = cfg.hip_profile
                                        hip_vel_fn       = cfg.hip_vel_profile
                                        flags = cfg.tick_flags
                                        new_cd = _ctrl_defaults_from_cfg(cfg)
                                        _en_lqr[0]        = new_cd["LQR"]
                                        _en_vel_pi[0]     = new_cd["VelPI"]
                                        _en_yaw_pi[0]     = new_cd["YawPI"]
                                        _en_suspension[0] = new_cd["Suspension"]
                                        _en_roll_lev[0]   = new_cd["RollLev"]
                                        _en_ff1[0]        = new_cd["HipFF"]
                                        _en_ff2[0]        = new_cd["GravFF"]
                                        _en_ff3[0]        = new_cd["CoMFF"]
                                        _en_ff4[0]        = new_cd["TurnFF"]
                                    title = new_title
                                    _reset_sim()
                                    if cfg and cfg.init_fn:
                                        cfg.init_fn(model, data, params)
                                    if not data_q.full():
                                        data_q.put_nowait(("TITLE", title))
                                    if cfg is not None and not data_q.full():
                                        data_q.put_nowait(("CONFIG_UPDATE", new_cd))
                                    print(f"  Switched (same world): {title}")
                                else:
                                    # ── DIFFERENT world — rebuild ──
                                    current_key = new_key
                                    world_switch = True
                                    if not data_q.full():
                                        data_q.put_nowait("RESET")
                                    print(f"  World change → rebuilding: {new_title}")
                                    break
                    except Exception:
                        pass

                if world_switch:
                    break

                # ── Drain cmd_q (from plot process UI) ──
                try:
                    while True:
                        cmd = cmd_q.get_nowait()
                        if cmd == "RESTART":
                            _reset_sim()
                            if cfg and cfg.init_fn:
                                cfg.init_fn(model, data, params)
                        elif isinstance(cmd, tuple):
                            if cmd[0] == "V_DESIRED":
                                _v_desired[0] = float(cmd[1])
                            elif cmd[0] == "OMEGA_DESIRED":
                                _omega_desired[0] = float(cmd[1])
                            elif cmd[0] == "HIP_PCT":
                                _hip_pct[0] = float(cmd[1])
                            elif cmd[0] == "CTRL_EN":
                                key, val = cmd[1], cmd[2]
                                if   key == "LQR":        _en_lqr[0]        = val
                                elif key == "VelPI":      _en_vel_pi[0]     = val
                                elif key == "YawPI":      _en_yaw_pi[0]     = val
                                elif key == "Suspension": _en_suspension[0] = val
                                elif key == "RollLev":    _en_roll_lev[0]   = val
                                elif key == "HipFF":      _en_ff1[0]        = val
                                elif key == "GravFF":     _en_ff2[0]        = val
                                elif key == "CoMFF":      _en_ff3[0]        = val
                                elif key == "KneeSpr":    _en_knee_spr[0]   = val
                            elif cmd[0] == "HIP_MODE":
                                _hip_mode[0] = cmd[1]
                            elif cmd[0] == "GAIN_VP":
                                ctrl.update_velocity_pi_gain(cmd[1], cmd[2])
                            elif cmd[0] == "GAIN_LQR":
                                ctrl.update_lqr_gain(cmd[1], cmd[2])
                            elif cmd[0] == "GAIN_YAW":
                                ctrl.update_yaw_pi_gain(cmd[1], cmd[2])
                            elif cmd[0] == "GAIN_SUSP":
                                ctrl.update_suspension_gain(cmd[1], cmd[2])
                            elif cmd[0] == "ROBOT_GEOM":
                                ctrl.update_robot_geom(cmd[1], cmd[2])
                            elif cmd[0] == "JUMP":
                                _jump_active[0] = True
                                ctrl.jump_ctrl.trigger()
                except Exception:
                    pass

                # ── Gamepad polling (sandbox mode) ──
                if cfg is None and _joy is not None:
                    for _ev in pygame.event.get():
                        if _ev.type == pygame.JOYAXISMOTION:
                            _joy_axes[_ev.axis] = _ev.value
                        elif _ev.type == pygame.JOYBUTTONDOWN and _ev.button == 0:
                            _jump_active[0] = True
                            ctrl.jump_ctrl.trigger()

                    raw_v = _joy_axes.get(1, 0.0)
                    raw_w = _joy_axes.get(2, 0.0)
                    raw_h = _joy_axes.get(3, 0.0)
                    _v_desired[0]     = (-raw_v * 3.0) if abs(raw_v) > JOY_DEADZONE else 0.0
                    _omega_desired[0] = (-raw_w * 5.0) if abs(raw_w) > JOY_DEADZONE else 0.0
                    if abs(raw_h) > JOY_DEADZONE:
                        clamped_h = max(-JOY_HIP_SATURATION, min(JOY_HIP_SATURATION, raw_h))
                        _hip_pct[0] = (JOY_HIP_SATURATION + clamped_h) / (2.0 * JOY_HIP_SATURATION) * 100.0

                # ── Slow-mo when jump is active ──
                # Slow-mo for entire flight, then continues 1s past landing.
                jmode = ctrl.jump_ctrl.mode
                if jmode == RobotMode.FLYING and _slowmo_liftoff_t[0] is None:
                    _slowmo_liftoff_t[0] = float(data.time)
                    _slowmo_landing_t[0] = None
                elif jmode == RobotMode.BALANCE and _slowmo_liftoff_t[0] is not None:
                    # Just landed — record landing time and clear liftoff tracker
                    if _slowmo_landing_t[0] is None:
                        _slowmo_landing_t[0] = float(data.time)
                    _slowmo_liftoff_t[0] = None

                in_flight_slow = (jmode != RobotMode.BALANCE
                                  and _slowmo_liftoff_t[0] is not None)
                post_landing_slow = (_slowmo_landing_t[0] is not None
                                     and (float(data.time) - _slowmo_landing_t[0]) <= 1.0)
                if in_flight_slow or post_landing_slow:
                    steps_per_frame = steps_per_frame_slow
                else:
                    steps_per_frame = steps_per_frame_normal
                    if _slowmo_landing_t[0] is not None and not post_landing_slow:
                        _slowmo_landing_t[0] = None  # clean up after slow-mo window

                # ── Physics stepping ──
                for _ in range(steps_per_frame):
                    if sim_fell:
                        break
                    if step % ctrl_steps == 0:
                        t_now = float(data.time)

                        if cfg is not None:
                            # ── Replay mode ──
                            past_duration = t_now >= cfg.duration
                            if not scenario_logged and past_duration:
                                scenario_logged = True
                                # Compute fitness breakdown via headless run
                                try:
                                    from master_sim_jump.scenarios import evaluate
                                    _metrics = evaluate(params, cfg.name)
                                    _bd = _metrics.get('fitness_breakdown', {})
                                    _fit = _metrics.get('fitness', '?')
                                    _status = _metrics.get('status', '?')
                                    if _bd and not data_q.full():
                                        data_q.put_nowait(("FITNESS", _bd, _fit, _status))
                                except Exception as _e:
                                    print(f"  Fitness breakdown failed: {_e}")
                                print(f"  Scenario complete ({cfg.duration:.1f}s). "
                                      f"Continuing — close viewer to exit.")

                            v_target     = v_profile_fn(t_now) if not past_duration else 0.0
                            theta_ref_c  = (theta_ref_fn(t_now) if theta_ref_fn and not past_duration else 0.0)
                            omega_target = (omega_profile_fn(t_now)
                                            if omega_profile_fn and not past_duration else 0.0)
                            q_hip_target = hip_profile_fn(t_now) if hip_profile_fn else robot.Q_NOM
                            dq_hip_tgt   = hip_vel_fn(t_now) if hip_vel_fn else 0.0

                            # Jump trigger (replay mode)
                            if (cfg.jump_time is not None and
                                    t_now >= cfg.jump_time):
                                ctrl.jump_ctrl.trigger()

                            tick = ctrl.tick(
                                model, data,
                                v_target_ms=v_target,
                                theta_ref_cmd=theta_ref_c,
                                omega_target=omega_target,
                                q_hip_target=q_hip_target,
                                dq_hip_target=dq_hip_tgt,
                                **flags)

                            _last_tick = tick
                            if tick['fell']:
                                sim_fell = True
                                print(f"  Robot fell at t={t_now:.2f}s.")
                                break
                        else:
                            # ── Sandbox mode ──
                            tick = ctrl.tick(
                                model, data,
                                v_target_ms=_v_desired[0],
                                omega_target=_omega_desired[0],
                                q_hip_target=_hip_target(),
                                use_lqr=_en_lqr[0],
                                use_velocity_pi=_en_vel_pi[0],
                                use_yaw_pi=_en_yaw_pi[0],
                                use_impedance=(_hip_mode[0] != "position_pd"),
                                use_roll_leveling=_en_roll_lev[0],
                                use_suspension=_en_suspension[0],
                                use_ff1=_en_ff1[0],
                                use_ff2=_en_ff2[0],
                                use_ff3=_en_ff3[0],
                        use_ff4=_en_ff4[0],
                                use_knee_spring=_en_knee_spr[0],
                                jump_active=_jump_active[0])

                            # Auto-clear jump flag when jump controller returns to BALANCE
                            if _jump_active[0] and ctrl.jump_ctrl.mode == RobotMode.BALANCE:
                                _jump_active[0] = False

                            _last_tick = tick
                            if tick['fell']:
                                _reset_sim()
                                break

                    # Disturbance (replay only, during scenario duration)
                    _vis_bid2, _vis_fz2 = None, 0.0
                    if cfg is not None and data.time < cfg.duration:
                        _vis_bid2, _vis_fz2 = apply_disturbance(
                            data, data.time, cfg,
                            ctrl.box_bid, ctrl.wheel_bid_L, ctrl.wheel_bid_R)

                    mujoco.mj_step(model, data)
                    step += 1

                # Camera follow
                viewer.cam.lookat[0] = data.xpos[ctrl.box_bid][0]
                viewer.cam.lookat[1] = data.xpos[ctrl.box_bid][1]
                # Reset dynamic geoms (keep static world-axes)
                viewer.user_scn.ngeom = _N_STATIC_GEOMS
                if _vis_bid2 is not None:
                    _add_force_arrow(viewer, data, _vis_bid2, _vis_fz2)
                # Always-visible knee spring spheres (grey idle, purple when active)
                _add_knee_spring_sphere(viewer, data, ctrl.tibia_bid_L,
                    active=abs(_last_tick.get('tau_spring_L', 0.0)) > 1e-6 if _last_tick else False)
                _add_knee_spring_sphere(viewer, data, ctrl.tibia_bid_R,
                    active=abs(_last_tick.get('tau_spring_R', 0.0)) > 1e-6 if _last_tick else False)
                viewer.sync()

                # ── Push 27-value telemetry tuple at ~60 Hz ──
                wall_now = time.perf_counter()
                if (wall_now - last_push >= 1.0 / TELEMETRY_HZ
                        and not data_q.full() and _last_tick):
                    tk = _last_tick
                    pitch_true_now, _ = get_pitch_and_rate(
                        data, ctrl.box_bid, ctrl.d_pitch)
                    pitch_ref_display = tk['pitch_ff'] + tk['theta_ref']

                    _ocm = ctrl.yaw_pi_ctrl.gains.omega_cmd_max
                    if cfg is not None:
                        v_cmd_display = v_profile_fn(float(data.time))
                        omega_cmd_display = max(-_ocm, min(_ocm,
                            omega_profile_fn(float(data.time))
                            if omega_profile_fn else 0.0))
                    else:
                        v_cmd_display = _v_desired[0]
                        omega_cmd_display = max(-_ocm, min(_ocm, _omega_desired[0]))

                    data_q.put_nowait((
                        float(data.time),
                        math.degrees(pitch_true_now),
                        math.degrees(pitch_ref_display),
                        math.degrees(tk['pitch_rate']),
                        tk['v_measured'],
                        v_cmd_display,
                        math.degrees(tk['yaw_rate']),
                        math.degrees(omega_cmd_display),
                        math.degrees(tk['hip_q_L']),
                        math.degrees(tk['hip_q_R']),
                        math.degrees(tk['q_nom_L']),
                        math.degrees(tk['q_nom_R']),
                        math.degrees(tk['roll']),
                        tk['tau_whl_L'],
                        tk['tau_whl_R'],
                        tk['v_batt'],
                        tk['batt_temp'],
                        tk['batt_soc'],
                        abs(tk['tau_whl_L']) / params.motors.wheel.Kt,
                        abs(tk['tau_whl_R']) / params.motors.wheel.Kt,
                        abs(tk['tau_hip_L']) / params.motors.hip.Kt_output,
                        abs(tk['tau_hip_R']) / params.motors.hip.Kt_output,
                        tk['i_total'],
                        tk['tau_hip_L'],
                        tk['tau_hip_R'],
                        tk['tau_spring_L'],
                        tk['tau_spring_R'],
                        max(tk['wheel_z_L'] - params.robot.wheel_r, 0.0) * 1000.0,
                        max(tk['wheel_z_R'] - params.robot.wheel_r, 0.0) * 1000.0,
                        tk.get('mode', 'BALANCE'),
                        tk.get('az_imu', 0.0),
                    ))
                    last_push = wall_now

                elapsed = time.perf_counter() - frame_start
                sleep_t = 1.0 / RENDER_HZ - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        # Viewer closed — was it a world switch or user quit?
        if not world_switch:
            user_quit = True

    # ── Clean up ──────────────────────────────────────────────────────────────
    if _joy is not None:
        pygame.quit()
    data_q.put(None)
    plot_proc.join(timeout=2)
    if plot_proc.is_alive():
        plot_proc.terminate()
    print("  Unified visualizer closed.")
    os._exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="master_sim visualization — replay scenarios or chart CSV data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Replay scenario with pyqtgraph telemetry:
  python -m master_sim.viz --mode replay --scenario s01_lqr_pitch_step
  python -m master_sim.viz --mode replay --scenario s04_vel_pi_staircase --viewer

  # CSV chart viewer:
  python -m master_sim.viz logs/S1_LQR_pitch_step.csv
  python -m master_sim.viz logs/S1*.csv --y fitness,rms_pitch_deg --filter status=PASS
""")
    ap.add_argument("csv", nargs="*", default=[],
                    help="CSV file(s) to plot (chart mode)")
    ap.add_argument("--mode", choices=["chart", "replay", "sandbox"],
                    default=None,
                    help="Visualization mode (default: chart if CSVs given, else replay)")
    ap.add_argument("--scenario", type=str, default=None,
                    help="Scenario name for replay mode (e.g. s01_lqr_pitch_step)")
    ap.add_argument("--viewer", action="store_true",
                    help="Also launch MuJoCo viewer alongside telemetry (replay mode)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for reproducible noise (default: 0)")
    # Chart mode options
    ap.add_argument("--x", dest="x_col", default=None,
                    help="X-axis column for chart mode (default: row index)")
    ap.add_argument("--y", dest="y_cols", default=None,
                    help="Comma-separated Y columns for chart mode (default: all numeric)")
    ap.add_argument("--filter", dest="filter_expr", default=None,
                    help="Row filter as col=val (e.g. status=PASS)")
    ap.add_argument("--title", default=None, help="Window title")
    ap.add_argument("--max-cols", type=int, default=2,
                    help="Max chart columns in grid (default: 2)")
    ap.add_argument("--list", action="store_true",
                    help="List available scenarios and exit")
    args = ap.parse_args()

    # List scenarios
    if args.list:
        from master_sim_jump.scenarios import SCENARIOS
        print("Available scenarios:")
        for name, cfg in sorted(SCENARIOS.items(), key=lambda x: x[1].order):
            print(f"  {name:36s} {cfg.display_name} ({cfg.duration:.0f}s)")
        return

    # Auto-detect mode: sandbox if no args, chart if CSVs given
    mode = args.mode
    if mode is None:
        mode = "chart" if args.csv else "sandbox"

    if mode == "replay":
        if not args.scenario:
            ap.error("--scenario is required for replay mode")
        run_unified(args.scenario, rng_seed=args.seed)

    elif mode == "sandbox":
        run_unified("sandbox", rng_seed=args.seed)

    else:  # chart
        if not args.csv:
            ap.error("CSV file(s) required for chart mode")
        y_cols = [c.strip() for c in args.y_cols.split(",")] if args.y_cols else None
        show(args.csv, x_col=args.x_col, y_cols=y_cols,
             filter_expr=args.filter_expr, title=args.title,
             max_cols=args.max_cols)


if __name__ == "__main__":
    mp.freeze_support()
    main()
