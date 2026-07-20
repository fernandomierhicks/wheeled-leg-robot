"""log_analyzer_tab.py — "Log Analyzer" tab: visual cross-check for wlog_metrics.

Lets Fernando open any .wlog and see the exact numbers analyze_hw_run.py would
report, plotted — because both call the same analysis/wlog_metrics.py functions
on the same decoded arrays, a number on screen here *is* the number that would
drive Claude's accept/reject decision, not a re-derivation of it (tuning.md §4c).

Pick a plot flavor from the combo box; each flavor redraws the one shared
PlotWidget and refreshes the metadata panel next to it.
"""

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)

from analysis.wlog_metrics import (
    HIP_CURRENT_EPS_A, SETTLE_DEG, compute_metrics, decode_wlog,
)
from .log_transfer import LOG_DIR
from .theme import BG, BLUE, BORDER, DIM, GREEN, ORANGE, RED, SURFACE, TEXT, YELLOW

pg.setConfigOptions(antialias=True, background=BG, foreground=TEXT)

_FLAVORS = ["LQR", "Vel-PI", "Yaw-PI", "Torque/Current", "Health/Saturation", "Gain-schedule check"]

# (label, health_flags bit, color) — shared/comm_protocol.h HEALTH_FLAG_*
_HEALTH_ROWS = [
    ("VEL_PI_SAT",       1 << 7,  RED),
    ("YAW_PI_SAT",       1 << 8,  ORANGE),
    ("WM_L_VEL_LIMITED", 1 << 9,  BLUE),
    ("WM_R_VEL_LIMITED", 1 << 10, GREEN),
    ("LOOP_OVERRUN",     1 << 11, YELLOW),
]


def _btn_style(color: str) -> str:
    return (
        f"QPushButton {{ background: {SURFACE}; color: {color}; border: 1px solid {color}; "
        f"border-radius: 3px; padding: 4px 12px; }}"
        f"QPushButton:hover {{ background: {BORDER}; }}"
    )


class LogAnalyzerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._run = None       # analysis.wlog_metrics.DecodedRun | None
        self._metrics = None   # dict | None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ── Top bar: open + flavor selector ─────────────────────────────────
        bar = QHBoxLayout()
        btn_open = QPushButton("Open .wlog...")
        btn_open.setStyleSheet(_btn_style(BLUE))
        btn_open.clicked.connect(self._on_open)
        bar.addWidget(btn_open)

        self._lbl_file = QLabel("No file loaded")
        self._lbl_file.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        bar.addWidget(self._lbl_file)
        bar.addStretch()

        bar.addWidget(QLabel("Flavor:"))
        self._flavor_combo = QComboBox()
        self._flavor_combo.addItems(_FLAVORS)
        self._flavor_combo.currentIndexChanged.connect(self._redraw)
        bar.addWidget(self._flavor_combo)
        outer.addLayout(bar)

        # ── Body: chart | metadata panel ────────────────────────────────────
        body = QSplitter(Qt.Orientation.Horizontal)

        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.12)
        self._legend = self._plot.addLegend(offset=(5, 5))
        body.addWidget(self._plot)

        meta_frame = QFrame()
        meta_frame.setStyleSheet(
            f"QFrame {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 4px; }}"
        )
        self._meta_layout = QVBoxLayout(meta_frame)
        meta_title = QLabel("Metrics")
        meta_title.setStyleSheet(f"color: {TEXT}; font-weight: bold; font-size: 13px;")
        self._meta_layout.addWidget(meta_title)
        self._meta_layout.addStretch()
        meta_frame.setMinimumWidth(220)
        meta_frame.setMaximumWidth(320)
        body.addWidget(meta_frame)
        body.setSizes([700, 260])
        outer.addWidget(body, 1)

        self._safety_lbl = QLabel("")
        self._safety_lbl.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        outer.addWidget(self._safety_lbl)

    # ── File loading ─────────────────────────────────────────────────────────

    def _on_open(self):
        start_dir = str(LOG_DIR) if LOG_DIR.exists() else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Open .wlog", start_dir, "WLOG files (*.WLOG *.wlog)")
        if path:
            self._load(Path(path))

    def _load(self, path: Path):
        try:
            run = decode_wlog(path)
        except (ValueError, OSError) as e:
            self._run = None
            self._metrics = None
            self._lbl_file.setText(f"Failed to open {path.name}: {e}")
            self._redraw()
            return
        self._run = run
        self._metrics = compute_metrics(run)
        self._lbl_file.setText(
            f"{path.name}  ({run.count} records @ {run.sample_rate_hz} Hz, telem_v{run.telem_version})"
        )
        self._redraw()

    # ── Redraw ───────────────────────────────────────────────────────────────

    def _set_meta_rows(self, rows: dict):
        # Drop everything after the title (index 0), including the previous
        # trailing stretch, then rebuild.
        while self._meta_layout.count() > 1:
            item = self._meta_layout.takeAt(1)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for k, v in rows.items():
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            kl = QLabel(f"{k}:")
            kl.setStyleSheet(f"color: {DIM}; font-size: 11px;")
            vl = QLabel(str(v))
            vl.setStyleSheet(f"color: {TEXT}; font-size: 12px; font-weight: bold; font-family: Consolas;")
            row.addWidget(kl)
            row.addStretch()
            row.addWidget(vl)
            self._meta_layout.addWidget(row_w)
        self._meta_layout.addStretch()

    def _redraw(self, *_args):
        self._plot.clear()
        self._legend.clear()
        self._plot.getAxis("left").setTicks(None)

        if self._run is None:
            self._set_meta_rows({})
            self._safety_lbl.setText("")
            return

        run, m = self._run, self._metrics
        t = run.t_s
        f = run.fields
        flavor = self._flavor_combo.currentText()
        self._plot.setLabel("bottom", "time", units="s", color=DIM)

        if flavor == "LQR":
            self._plot.setLabel("left", "pitch [deg] / rate [deg/s]", color=DIM)
            self._plot.plot(t, np.degrees(f["pitch_rad"]), pen=pg.mkPen(BLUE, width=1.5), name="pitch_deg")
            self._plot.plot(t, np.degrees(f["pitch_rate_rads"]), pen=pg.mkPen(ORANGE, width=1.2), name="pitch_rate_dps")
            for sign in (1, -1):
                self._plot.addLine(y=sign * SETTLE_DEG, pen=pg.mkPen(GREEN, style=Qt.PenStyle.DashLine))
            self._set_meta_rows({
                "rms_pitch_deg": m["rms_pitch_deg"],
                "ise_pitch": m["ise_pitch"],
                "settle_time_s": m["settle_time_s"],
                "max_pitch_deg (overshoot)": m["max_pitch_deg"],
            })

        elif flavor == "Vel-PI":
            self._plot.setLabel("left", "velocity [m/s]", color=DIM)
            self._plot.plot(t, f["v_ref"], pen=pg.mkPen(GREEN, width=1.5), name="v_ref")
            self._plot.plot(t, f["wheel_vel_avg"], pen=pg.mkPen(BLUE, width=1.2), name="wheel_vel_avg_ms")
            self._set_meta_rows({"vel_track_rms_ms": m["vel_track_rms_ms"]})

        elif flavor == "Yaw-PI":
            self._plot.setLabel("left", "yaw rate [rad/s]", color=DIM)
            self._plot.plot(t, f["omega_cmd_rds"], pen=pg.mkPen(GREEN, width=1.5), name="omega_cmd_rds")
            self._plot.plot(t, f["yaw_rate_rads"], pen=pg.mkPen(BLUE, width=1.2), name="yaw_rate_rads")
            self._set_meta_rows({"yaw_track_rms_rads": m["yaw_track_rms_rads"]})

        elif flavor == "Torque/Current":
            self._plot.setLabel("left", "current [A] / torque [N·m]", color=DIM)
            self._plot.plot(t, f["hip_l_current_a"], pen=pg.mkPen(RED, width=1.2), name="hip_l_current_a")
            self._plot.plot(t, f["hip_r_current_a"], pen=pg.mkPen(ORANGE, width=1.2), name="hip_r_current_a")
            self._plot.plot(t, f["whl_tau_l"], pen=pg.mkPen(BLUE, width=1.2), name="whl_tau_l")
            self._plot.plot(t, f["whl_tau_r"], pen=pg.mkPen(GREEN, width=1.2), name="whl_tau_r")
            for sign in (1, -1):
                self._plot.addLine(y=sign * HIP_CURRENT_EPS_A, pen=pg.mkPen(YELLOW, style=Qt.PenStyle.DashLine))
            self._set_meta_rows({
                "max_hip_l_current_a": m["max_hip_l_current_a"],
                "max_hip_r_current_a": m["max_hip_r_current_a"],
            })

        elif flavor == "Health/Saturation":
            self._plot.setLabel("left", "", color=DIM)
            flags = f["health_flags"].astype(np.int64)
            ticks = []
            for i, (name, bit, color) in enumerate(_HEALTH_ROWS):
                trace = ((flags & bit) != 0).astype(np.float64) * 0.8 + i * 1.2
                self._plot.plot(t, trace, pen=pg.mkPen(color, width=1.5), name=name)
                ticks.append((i * 1.2 + 0.4, name))
            self._plot.getAxis("left").setTicks([ticks])
            self._set_meta_rows({k: f"{v * 100:.1f}%" for k, v in m["health_fractions"].items()})

        elif flavor == "Gain-schedule check":
            if run.has_gain_sched_alpha:
                self._plot.setLabel("left", "gain_sched_alpha", color=DIM)
                self._plot.plot(t, f["gain_sched_alpha"], pen=pg.mkPen(BLUE, width=1.5), name="gain_sched_alpha")
                self._plot.addLine(y=0.0, pen=pg.mkPen(GREEN, style=Qt.PenStyle.DashLine))
                self._set_meta_rows(m["gain_sched_alpha_range"])
            else:
                txt = pg.TextItem(
                    "gain_sched_alpha not in telemetry yet\n(§1a pending — see tuning.md)",
                    color=DIM, anchor=(0.5, 0.5),
                )
                self._plot.addItem(txt)
                txt.setPos(float(t[len(t) // 2]) if len(t) else 0.0, 0.0)
                self._set_meta_rows({"gain_sched_alpha": "N/A (§1a pending)"})

        if m["fault_fired"]:
            self._safety_lbl.setStyleSheet(f"color: {RED}; font-size: 11px;")
            self._safety_lbl.setText(f"FAULT fired mid-run: codes {m['faults_seen']}")
        else:
            self._safety_lbl.setStyleSheet(f"color: {GREEN}; font-size: 11px;")
            self._safety_lbl.setText("No fault fired during this run.")
