"""diag_delay.py — Standalone diagnostic for sensor delay, motor delay & predictor.

Runs S1 (LQR pitch step), prints numerical sanity checks, then opens a
pyqtgraph window showing true vs delayed vs predicted traces.

Usage (from simulation/mujoco/):
    python -m master_sim.diag_delay
"""
import math
import os
import sys
import time

# Allow running directly: python diag_delay.py
_MASTER_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_MUJOCO_DIR = os.path.dirname(_MASTER_SIM_DIR)
if _MUJOCO_DIR not in sys.path:
    sys.path.insert(0, _MUJOCO_DIR)

import numpy as np

# mujoco before pyqtgraph (Windows OpenGL DLL conflict)
import mujoco  # noqa: F401

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

# ── Style (matches visualizer.py) ───────────────────────────────────────────
BG_COLOR   = "#12121e"
TICK_COLOR = "#d8d8d8"
GRID_ALPHA = 0.20
LINE_WIDTH = 1.8
PAD_LEFT   = 90
PAD_TOP    = 35

CYAN   = "#60d0ff"
RED    = "#ff6060"
GREEN  = "#80ff80"
ORANGE = "#ffa040"
YELLOW = "#ffe060"
PURPLE = "#e080ff"

RAD2DEG = 180.0 / math.pi

# ── Data collection ─────────────────────────────────────────────────────────

class TelemetryRecorder:
    def __init__(self):
        self._ticks = []

    def __call__(self, tick: dict):
        self._ticks.append(tick)

    def to_arrays(self) -> dict:
        if not self._ticks:
            return {}
        keys = self._ticks[0].keys()
        return {k: np.array([t[k] for t in self._ticks]) for k in keys}


def run_sim():
    """Run S1 scenario and return telemetry arrays + params."""
    from master_sim_jump.defaults import DEFAULT_PARAMS
    from master_sim_jump.scenarios import SCENARIOS
    from master_sim_jump.sim_loop import run

    params = DEFAULT_PARAMS
    cfg = SCENARIOS["s01_lqr_pitch_step"]
    rec = TelemetryRecorder()

    print(f"Running: {cfg.display_name} ({cfg.duration:.1f}s) ...")
    t0 = time.perf_counter()
    metrics = run(params, cfg, callbacks=[rec], rng_seed=42)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s — {rec._ticks.__len__()} ticks, "
          f"status={metrics['status']}")

    return rec.to_arrays(), params


# ── Numerical checks ────────────────────────────────────────────────────────

def run_checks(d, params):
    """Print PASS/FAIL sanity checks for delay buffer & predictor."""
    dt_ctrl = 1.0 / params.timing.ctrl_hz
    n_sens = round(params.latency.sensor_delay_s / dt_ctrl) \
        if params.latency.sensor_delay_s > 0 else 0
    n_act = round(params.latency.actuator_delay_s / dt_ctrl) \
        if params.latency.actuator_delay_s > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Delay diagnostics — n_sens={n_sens}, n_act={n_act}")
    print(f"{'='*60}")

    all_pass = True

    # --- Check 1: sensor delay buffer correctness ---
    if n_sens > 0:
        # delayed[k] should equal noisy[k - n_sens]
        noisy = d["pitch_noisy"]
        delayed = d["pitch_delayed"]
        N = len(noisy)
        mismatches = 0
        for k in range(n_sens, N):
            if noisy[k - n_sens] != delayed[k]:
                mismatches += 1
        ok = mismatches == 0
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] Sensor delay buffer: pitch_delayed[k] == "
              f"pitch_noisy[k-{n_sens}]  ({mismatches}/{N - n_sens} mismatches)")
        if not ok:
            all_pass = False
    else:
        print("  [SKIP] Sensor delay buffer: n_sens=0 (passthrough)")

    # --- Check 2: predictor improves estimate ---
    if n_sens > 0:
        true_p = d["pitch"]
        delayed_p = d["pitch_delayed"]
        predicted_p = d["pitch_predicted"]
        # Skip transient at start
        start = max(n_sens, 50)
        err_delayed = np.mean(np.abs(delayed_p[start:] - true_p[start:]))
        err_predicted = np.mean(np.abs(predicted_p[start:] - true_p[start:]))
        ok = err_predicted < err_delayed
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] Predictor reduces pitch error: "
              f"delayed MAE={err_delayed*RAD2DEG:.4f} deg, "
              f"predicted MAE={err_predicted*RAD2DEG:.4f} deg")
        if not ok:
            all_pass = False

        err_delayed_r = np.mean(np.abs(d["pitch_rate_delayed"][start:] - d["pitch_rate"][start:]))
        err_predicted_r = np.mean(np.abs(d["pitch_rate_predicted"][start:] - d["pitch_rate"][start:]))
        ok2 = err_predicted_r < err_delayed_r
        tag = "PASS" if ok2 else "FAIL"
        print(f"  [{tag}] Predictor reduces pitch_rate error: "
              f"delayed MAE={err_delayed_r*RAD2DEG:.4f} deg/s, "
              f"predicted MAE={err_predicted_r*RAD2DEG:.4f} deg/s")
        if not ok2:
            all_pass = False
    else:
        print("  [SKIP] Predictor check: n_sens=0 (no prediction)")

    # --- Check 3: actuator delay buffer correctness ---
    if n_act > 0:
        cmd = d["tau_cmd_L"]
        delayed_t = d["tau_delayed_L"]
        N = len(cmd)
        mismatches = 0
        for k in range(n_act, N):
            if cmd[k - n_act] != delayed_t[k]:
                mismatches += 1
        ok = mismatches == 0
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] Actuator delay buffer: tau_delayed_L[k] == "
              f"tau_cmd_L[k-{n_act}]  ({mismatches}/{N - n_act} mismatches)")
        if not ok:
            all_pass = False
    else:
        print("  [SKIP] Actuator delay buffer: n_act=0 (passthrough)")

    print(f"{'='*60}")
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'='*60}\n")
    return all_pass


# ── Chart ────────────────────────────────────────────────────────────────────

def show_charts(d, params):
    """Open pyqtgraph window with 5-row diagnostic chart."""
    dt_ctrl = 1.0 / params.timing.ctrl_hz
    n_sens = round(params.latency.sensor_delay_s / dt_ctrl) \
        if params.latency.sensor_delay_s > 0 else 0
    n_act = round(params.latency.actuator_delay_s / dt_ctrl) \
        if params.latency.actuator_delay_s > 0 else 0

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    win = pg.GraphicsLayoutWidget(show=False)
    win.setWindowTitle(f"Delay Diagnostic — n_sens={n_sens}  n_act={n_act}")
    win.setBackground(BG_COLOR)

    screen = app.primaryScreen().availableGeometry()
    w = screen.width() - PAD_LEFT - 20
    h = screen.height() - PAD_TOP - 20
    win.setGeometry(PAD_LEFT, PAD_TOP, w, h)

    t = d["t"]

    def make_plot(row, title, ylabel):
        p = win.addPlot(row=row, col=0, title=title)
        p.setLabel("bottom", "Time", units="s")
        p.setLabel("left", ylabel)
        p.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        for ax in ("bottom", "left"):
            p.getAxis(ax).setTickFont(pg.QtGui.QFont("Consolas", 9))
            p.getAxis(ax).setTextPen(TICK_COLOR)
            p.getAxis(ax).label.setFont(pg.QtGui.QFont("Consolas", 9))
        p.setTitle(title, color=TICK_COLOR, size="11pt")
        # Legend before adding items so auto-populate works
        p.addLegend(offset=(10, 10), labelTextColor=TICK_COLOR)
        # Rect zoom mode (left-drag draws rectangle to zoom)
        p.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        return p

    def add_line(plot, x, y, color, name, width=LINE_WIDTH, dash=False):
        pen_args = {"color": color, "width": width}
        if dash:
            pen_args["style"] = QtCore.Qt.PenStyle.DashLine
        pen = pg.mkPen(**pen_args)
        plot.plot(x, y, pen=pen, name=name)

    # Row 0 — Pitch
    p0 = make_plot(0, "Pitch", "deg")
    add_line(p0, t, d["pitch"] * RAD2DEG, CYAN, "true")
    add_line(p0, t, d["pitch_delayed"] * RAD2DEG, RED, "delayed", dash=True)
    add_line(p0, t, d["pitch_predicted"] * RAD2DEG, GREEN, "predicted")

    # Row 1 — Pitch Rate
    p1 = make_plot(1, "Pitch Rate", "deg/s")
    add_line(p1, t, d["pitch_rate"] * RAD2DEG, CYAN, "true")
    add_line(p1, t, d["pitch_rate_delayed"] * RAD2DEG, RED, "delayed", dash=True)
    add_line(p1, t, d["pitch_rate_predicted"] * RAD2DEG, GREEN, "predicted")

    # Row 2 — Wheel Velocity
    p2 = make_plot(2, "Wheel Velocity", "rad/s")
    add_line(p2, t, d["wheel_vel"], CYAN, "true")
    add_line(p2, t, d["wheel_vel_delayed"], RED, "delayed", dash=True)
    add_line(p2, t, d["wheel_vel_predicted"], GREEN, "predicted")

    # Row 3 — Torque (symmetric, L wheel shown)
    p3 = make_plot(3, "Wheel Torque (L)", "N-m")
    add_line(p3, t, d["tau_cmd_L"], CYAN, "commanded")
    add_line(p3, t, d["tau_delayed_L"], ORANGE, "delayed", dash=True)
    add_line(p3, t, d["tau_whl_L"], GREEN, "applied (ctrl)")

    # Row 4 — Prediction Error
    p4 = make_plot(4, "Prediction Error", "deg")
    err_pitch = (d["pitch_predicted"] - d["pitch"]) * RAD2DEG
    err_rate  = (d["pitch_rate_predicted"] - d["pitch_rate"]) * RAD2DEG
    add_line(p4, t, err_pitch, CYAN, "pitch err")
    add_line(p4, t, err_rate, PURPLE, "pitch_rate err")

    # Link X axes
    for p in (p1, p2, p3, p4):
        p.setXLink(p0)

    win.show()
    app.exec()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    d, params = run_sim()
    run_checks(d, params)
    show_charts(d, params)


if __name__ == "__main__":
    main()
