"""Parameter-aware visual analysis for SD WLOG and host JSONL controller trials.

The numerical fitness calculations remain in analysis.wlog_metrics and retain
their firmware/SI units.  This tab is a display layer: all angular signals and
angular metrics are converted to degrees, controller limits are read from the
paired .PARAMS sidecar, and saturation is shown beside the constrained signal.
"""

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)

from analysis.param_sidecar import (
    ParamSidecar, active_profile_series, load_host_param_sidecar, load_matching_sidecar,
)
from analysis.wlog_metrics import (
    HIP_CURRENT_EPS_A, SETTLE_DEG, compute_metrics, decode_run,
)
from .log_paths import RUNS_DIR
from .theme import BG, BLUE, BORDER, DIM, GREEN, ORANGE, RED, SURFACE, TEXT, WHITE, YELLOW

pg.setConfigOptions(antialias=True, background=BG, foreground=TEXT)

_FLAVORS = ["Overview", "LQR", "Vel-PI", "Yaw-PI", "Torque / Current", "Gain schedule"]

_HF_VEL_PI_SAT = 1 << 7
_HF_YAW_PI_SAT = 1 << 8
_HF_WM_L_VEL_LIMITED = 1 << 9
_HF_WM_R_VEL_LIMITED = 1 << 10
_HF_LOOP_OVERRUN = 1 << 11


def _btn_style(color: str) -> str:
    return (
        f"QPushButton {{ background: {SURFACE}; color: {color}; border: 1px solid {color}; "
        f"border-radius: 3px; padding: 4px 12px; }}"
        f"QPushButton:hover {{ background: {BORDER}; }}"
    )


def _mask_intervals(t: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Compress a boolean sample mask into visible time intervals."""
    t = np.asarray(t, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if t.size == 0 or mask.size != t.size or not np.any(mask):
        return []
    transitions = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(transitions == 1) + 1)
    ends = list(np.flatnonzero(transitions == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size)
    nominal_dt = float(np.median(np.diff(t))) if t.size > 1 else 0.002
    return [
        (float(t[start]), float(t[end]) if end < t.size else float(t[-1] + nominal_dt))
        for start, end in zip(starts, ends)
    ]


def _event_summary(t: np.ndarray, mask: np.ndarray) -> str:
    intervals = _mask_intervals(t, mask)
    if not intervals:
        return "none"
    durations = [end - start for start, end in intervals]
    return f"{len(intervals)} event{'s' if len(intervals) != 1 else ''}, {sum(durations):.3f} s total"


def _parameter_change_groups(sidecar: ParamSidecar | None,
                             sample_t_micros: np.ndarray,
                             sample_t_s: np.ndarray) -> list[tuple[float, tuple]]:
    """Align real parameter CHANGE events to the nearest log sample.

    DUMP rows describe the initial snapshot and are intentionally omitted.
    Events sharing a sample are grouped so the overview draws one marker.
    """
    if sidecar is None:
        return []
    sample_t_micros = np.asarray(sample_t_micros, dtype=np.uint32)
    sample_t_s = np.asarray(sample_t_s, dtype=np.float64)
    if sample_t_micros.size == 0 or sample_t_s.size != sample_t_micros.size:
        return []

    elapsed_us = np.concatenate((
        np.asarray([0], dtype=np.uint64),
        np.cumsum(np.diff(sample_t_micros).astype(np.uint32).astype(np.uint64),
                  dtype=np.uint64),
    ))
    start = int(sample_t_micros[0])
    grouped: dict[int, list] = {}
    for event in sidecar.events:
        if event.event.upper() != "CHANGE":
            continue
        delta = ((int(event.t_micros) - start + (1 << 31)) % (1 << 32)) - (1 << 31)
        index = 0 if delta <= 0 else int(np.searchsorted(elapsed_us, delta, side="left"))
        if index < sample_t_s.size:
            grouped.setdefault(index, []).append(event)
    return [(float(sample_t_s[index]), tuple(events))
            for index, events in sorted(grouped.items())]


def _parameter_change_label(events: tuple, max_names: int = 2) -> str:
    shown = [f"{event.name}={event.value:g}" for event in events[:max_names]]
    if len(events) > max_names:
        shown.append(f"+{len(events) - max_names}")
    return ", ".join(shown)


def _transparent_brush(color: str, alpha: int = 34):
    qcolor = QColor(color)
    qcolor.setAlpha(alpha)
    return pg.mkBrush(qcolor)


def _split_legend_name(name: str) -> tuple[str, str]:
    """Split "Short name — verbose description" into (short, full-text-for-tooltip)."""
    short, sep, _rest = name.partition(" — ")
    return (short, name) if sep else (name, name)


def _set_legend_tooltip(legend: pg.LegendItem | None, item: pg.PlotDataItem, tooltip: str):
    if legend is None:
        return
    for sample, label in legend.items:
        if sample.item is item:
            sample.setToolTip(tooltip)
            label.setToolTip(tooltip)
            return


class LogAnalyzerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._run = None
        self._metrics = None
        self._params = None
        self._param_error = ""
        self._plots: list[pg.PlotWidget] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        bar = QHBoxLayout()
        btn_open = QPushButton("Open log…")
        btn_open.setStyleSheet(_btn_style(BLUE))
        btn_open.clicked.connect(self._on_open)
        bar.addWidget(btn_open)

        self._lbl_file = QLabel("No file loaded")
        self._lbl_file.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        bar.addWidget(self._lbl_file)
        bar.addStretch()

        bar.addWidget(QLabel("View:"))
        self._flavor_combo = QComboBox()
        self._flavor_combo.addItems(_FLAVORS)
        self._flavor_combo.currentIndexChanged.connect(self._redraw)
        bar.addWidget(self._flavor_combo)
        outer.addLayout(bar)

        body = QSplitter(Qt.Orientation.Horizontal)

        chart_frame = QWidget()
        self._charts_layout = QVBoxLayout(chart_frame)
        self._charts_layout.setContentsMargins(0, 0, 0, 0)
        self._charts_layout.setSpacing(5)
        body.addWidget(chart_frame)

        meta_frame = QFrame()
        meta_frame.setStyleSheet(
            f"QFrame {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 4px; }}"
        )
        self._meta_layout = QVBoxLayout(meta_frame)
        meta_title = QLabel("Metrics and limits")
        meta_title.setStyleSheet(f"color: {TEXT}; font-weight: bold; font-size: 13px;")
        self._meta_layout.addWidget(meta_title)
        self._meta_layout.addStretch()
        meta_frame.setMinimumWidth(285)
        meta_frame.setMaximumWidth(390)
        body.addWidget(meta_frame)
        body.setSizes([760, 330])
        outer.addWidget(body, 1)

        self._safety_lbl = QLabel("")
        self._safety_lbl.setWordWrap(True)
        self._safety_lbl.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        outer.addWidget(self._safety_lbl)

    # ── File loading ────────────────────────────────────────────────────────

    def _on_open(self):
        start_dir = str(RUNS_DIR) if RUNS_DIR.exists() else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open controller log", start_dir,
            "Controller logs (*.WLOG *.wlog *.jsonl)")
        if path:
            self._load(Path(path))

    def _load(self, path: Path):
        try:
            run = decode_run(path)
        except (ValueError, OSError) as exc:
            self._run = None
            self._metrics = None
            self._params = None
            self._lbl_file.setText(f"Failed to open {path.name}: {exc}")
            self._redraw()
            return

        self._run = run
        self._metrics = compute_metrics(run)
        self._params = None
        self._param_error = ""
        try:
            self._params = (
                load_host_param_sidecar(path) if run.source_kind == "host"
                else load_matching_sidecar(path)
            )
        except (ValueError, OSError) as exc:
            self._param_error = str(exc)

        sidecar_status = (
            f"limits: {self._params.path.name}" if self._params is not None
            else "limits unavailable (parameter snapshot not found)"
        )
        if self._param_error:
            sidecar_status = "limits unavailable (.PARAMS invalid)"
        self._lbl_file.setText(
            f"{path.name}  •  {run.count} records @ {run.sample_rate_hz} Hz  •  "
            f"telem v{run.telem_version}  •  {sidecar_status}"
        )
        self._redraw()

    # ── Plot and metadata helpers ──────────────────────────────────────────

    def _clear_plots(self):
        self._plots.clear()
        while self._charts_layout.count():
            item = self._charts_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _new_plot(self, title: str, axis_text: str, units: str = "",
                  auto_si_prefix: bool = True) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setTitle(title, color=TEXT, size="11pt")
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setLabel("left", axis_text, units=units or None, color=DIM)
        plot.getAxis("left").enableAutoSIPrefix(auto_si_prefix)
        plot.addLegend(offset=(5, 5), verSpacing=-2)
        if self._plots:
            plot.setXLink(self._plots[0])
        self._plots.append(plot)
        self._charts_layout.addWidget(plot, 1)
        return plot

    def _finish_plots(self):
        for plot in self._plots[:-1]:
            plot.hideAxis("bottom")
        if self._plots:
            self._plots[-1].setLabel("bottom", "Time", units="s", color=DIM)

    @staticmethod
    def _curve(plot: pg.PlotWidget, t: np.ndarray, values: np.ndarray,
               color: str, name: str, width: float = 1.35,
               style: Qt.PenStyle = Qt.PenStyle.SolidLine):
        short_name, tooltip = _split_legend_name(name)
        curve = plot.plot(t, values, pen=pg.mkPen(color, width=width, style=style), name=short_name)
        _set_legend_tooltip(plot.plotItem.legend, curve, tooltip)

    def _limit_pair(self, plot: pg.PlotWidget, t: np.ndarray, limit,
                    color: str, upper_name: str, lower_name: str,
                    style: Qt.PenStyle = Qt.PenStyle.DashLine):
        values = np.asarray(limit, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(t.size, float(values), dtype=np.float64)
        if values.size != t.size or not np.any(np.isfinite(values)):
            return
        self._curve(plot, t, values, color, upper_name, 1.0, style)
        self._curve(plot, t, -values, color, lower_name, 1.0, style)

    @staticmethod
    def _add_regions(plot: pg.PlotWidget, t: np.ndarray, mask: np.ndarray,
                     color: str, legend_name: str):
        intervals = _mask_intervals(t, mask)
        for start, end in intervals:
            region = pg.LinearRegionItem(
                values=(start, end), movable=False, brush=_transparent_brush(color),
                pen=pg.mkPen(color, width=0.6), orientation="vertical")
            region.setZValue(-20)
            plot.addItem(region)
        if intervals and plot.plotItem.legend is not None:
            short_name, tooltip = _split_legend_name(legend_name)
            sample = pg.PlotDataItem(pen=pg.mkPen(color, width=5))
            plot.plotItem.legend.addItem(sample, short_name)
            _set_legend_tooltip(plot.plotItem.legend, sample, tooltip)

    @staticmethod
    def _add_parameter_changes(plot: pg.PlotWidget, changes: list[tuple[float, tuple]],
                               show_labels: bool = False):
        """Draw synchronized vertical markers without affecting auto-range."""
        for t_s, events in changes:
            full_label = _parameter_change_label(events, max_names=len(events))
            marker = pg.InfiniteLine(
                pos=t_s, angle=90, movable=False,
                pen=pg.mkPen(YELLOW, width=1.0, style=Qt.PenStyle.DashLine),
                label=_parameter_change_label(events) if show_labels else None,
                labelOpts={"color": YELLOW, "position": 0.92},
            )
            marker.setToolTip(f"Parameter change at {t_s:.3f} s: {full_label}")
            marker.setZValue(30)
            plot.addItem(marker, ignoreBounds=True)

        if changes and plot.plotItem.legend is not None:
            sample = pg.PlotDataItem(
                pen=pg.mkPen(YELLOW, width=1.5, style=Qt.PenStyle.DashLine))
            plot.plotItem.legend.addItem(sample, "Parameter change")
            _set_legend_tooltip(
                plot.plotItem.legend, sample,
                "Firmware parameter value changed at this time; hover a marker for details.")

    def _param_series(self, name: str) -> np.ndarray | None:
        if self._params is None or self._run is None:
            return None
        return self._params.series(name, self._run.t_micros)

    def _profile_limit(self, suffix: str) -> np.ndarray | None:
        if self._run is None or "active_profile" not in self._run.fields:
            return None
        return active_profile_series(
            self._params, suffix, self._run.fields["active_profile"], self._run.t_micros)

    def _set_meta_rows(self, rows: list[tuple[str, str, str]]):
        while self._meta_layout.count() > 1:
            item = self._meta_layout.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for label, value, description in rows:
            row_w = QWidget()
            row = QVBoxLayout(row_w)
            row.setContentsMargins(0, 2, 0, 4)
            top = QHBoxLayout()
            name_label = QLabel(label)
            name_label.setStyleSheet(f"color: {DIM}; font-size: 11px;")
            value_label = QLabel(value)
            value_label.setStyleSheet(
                f"color: {TEXT}; font-size: 12px; font-weight: bold; font-family: Consolas;")
            top.addWidget(name_label)
            top.addStretch()
            top.addWidget(value_label)
            row.addLayout(top)
            if description:
                detail = QLabel(description)
                detail.setWordWrap(True)
                detail.setStyleSheet(f"color: {DIM}; font-size: 10px;")
                row.addWidget(detail)
            self._meta_layout.addWidget(row_w)
        self._meta_layout.addStretch()

    def _profile_text(self) -> str:
        if self._run is None or "active_profile" not in self._run.fields:
            return "unknown"
        profiles = np.unique(self._run.fields["active_profile"].astype(np.int64)) + 1
        return " → ".join(f"P{profile}" for profile in profiles)

    # ── Redraw ─────────────────────────────────────────────────────────────

    def _redraw(self, *_args):
        self._clear_plots()
        if self._run is None:
            self._set_meta_rows([])
            self._safety_lbl.setText("")
            return

        run, metrics = self._run, self._metrics
        t, fields = run.t_s, run.fields
        health = fields["health_flags"].astype(np.int64)
        flavor = self._flavor_combo.currentText()

        if flavor == "Overview":
            changes = _parameter_change_groups(self._params, run.t_micros, t)

            velocity_plot = self._new_plot(
                "Forward command and measured motion", "Velocity", "m/s",
                auto_si_prefix=False)
            self._curve(velocity_plot, t, fields["v_ref"], GREEN,
                        "Velocity command vref — requested forward speed (m/s)", 1.6)
            self._curve(velocity_plot, t, fields["wheel_vel_avg"], BLUE,
                        "Average wheel velocity — measured forward speed (m/s)", 1.45)
            self._add_parameter_changes(velocity_plot, changes, show_labels=True)

            yaw_plot = self._new_plot(
                "Yaw command and measured turn rate", "Yaw rate", "°/s",
                auto_si_prefix=False)
            self._curve(yaw_plot, t, np.degrees(fields["omega_cmd_rds"]), GREEN,
                        "Yaw command ωcmd — requested turn rate (°/s)", 1.6)
            self._curve(yaw_plot, t, np.degrees(fields["yaw_rate_rads"]), BLUE,
                        "Measured yaw rate — IMU turn rate (°/s)", 1.45)
            self._add_parameter_changes(yaw_plot, changes)

            pitch_plot = self._new_plot(
                "Pitch/roll and controller lean target", "Angle", "°",
                auto_si_prefix=False)
            self._curve(pitch_plot, t, np.degrees(fields["pitch_rad"]), BLUE,
                        "Pitch — measured body tilt (°)", 1.6)
            self._curve(pitch_plot, t, np.degrees(fields["theta_ref"]), ORANGE,
                        "Lean target θref — controller target angle (°)", 1.4)
            self._curve(pitch_plot, t, np.degrees(fields["roll_rad"]), RED,
                        "Roll — measured body roll (°)", 1.4)
            self._add_parameter_changes(pitch_plot, changes)

            change_text = "none"
            if changes:
                event_count = sum(len(events) for _time, events in changes)
                change_text = f"{event_count} change{'s' if event_count != 1 else ''} at " + \
                    ", ".join(f"{time_s:.2f} s" for time_s, _events in changes[:4])
                if len(changes) > 4:
                    change_text += f", +{len(changes) - 4} more times"
            self._set_meta_rows([
                ("Source", "Host JSONL" if run.source_kind == "host" else "SD WLOG",
                 "Both sources use the same decoded-run and chart pipeline."),
                ("Duration", f"{metrics['duration_s']:.3f} s",
                 f"{run.count:,} telemetry samples at approximately {run.sample_rate_hz} Hz."),
                ("Parameter changes", change_text,
                 "Dashed yellow markers show actual value changes; initial snapshot rows are omitted."),
                ("Peak velocity command", f"{np.max(np.abs(fields['v_ref'])):.3f} m/s",
                 "Largest requested forward or reverse speed."),
                ("Peak yaw command", f"{np.max(np.abs(np.degrees(fields['omega_cmd_rds']))):.2f} °/s",
                 "Largest requested left or right turn rate."),
                ("Pitch range",
                 f"{np.min(np.degrees(fields['pitch_rad'])):.2f}° to "
                 f"{np.max(np.degrees(fields['pitch_rad'])):.2f}°",
                 "Full measured body-pitch range over the log."),
                ("Active profile", self._profile_text(),
                 "Speed profile reported by telemetry."),
            ])

        elif flavor == "LQR":
            pitch_deg = np.degrees(fields["pitch_rad"])
            target_deg = np.degrees(fields["theta_ref"])
            error_deg = pitch_deg - target_deg
            rate_dps = np.degrees(fields["pitch_rate_rads"])

            angle_plot = self._new_plot("Body angle and controller target", "Angle", "°")
            self._curve(angle_plot, t, pitch_deg, BLUE,
                        "Pitch — measured body tilt (°)", 1.6)
            self._curve(angle_plot, t, target_deg, GREEN,
                        "Lean target θref — requested balance angle (°)", 1.4)

            error_plot = self._new_plot("Balance tracking error", "Pitch error", "°")
            self._curve(error_plot, t, error_deg, YELLOW,
                        "Pitch error — measured pitch minus lean target (°)", 1.5)
            self._limit_pair(
                error_plot, t, SETTLE_DEG, GREEN,
                f"Settling-band upper — +{SETTLE_DEG:g}° tracking error",
                f"Settling-band lower — −{SETTLE_DEG:g}° tracking error")

            rate_plot = self._new_plot("Pitch damping", "Pitch rate", "°/s")
            self._curve(rate_plot, t, rate_dps, ORANGE,
                        "Pitch rate — body rotation speed about wheel axle (°/s)", 1.35)

            ise_deg2_s = metrics["ise_pitch"] * (180.0 / np.pi) ** 2
            self._set_meta_rows([
                ("RMS pitch error", f"{metrics['rms_pitch_deg']:.3f}°",
                 "Typical deviation from the current lean target."),
                ("Maximum pitch error", f"{metrics['max_pitch_deg']:.3f}°",
                 "Worst measured deviation from the lean target."),
                ("Pitch-error ISE", f"{ise_deg2_s:.3f} °²·s",
                 "Accumulated squared tracking error over the entire log."),
                ("Settling time", f"{metrics['settle_time_s']:.3f} s",
                 f"First continuous 0.5 s inside the ±{SETTLE_DEG:g}° error band."),
                ("Maximum pitch rate", f"{np.max(np.abs(rate_dps)):.2f} °/s",
                 "Largest absolute body rotation rate."),
                ("Active profile", self._profile_text(),
                 "CH9 speed profile reported by telemetry."),
            ])

        elif flavor == "Vel-PI":
            target = fields["v_ref"]
            measured = fields["wheel_vel_avg"]
            error = target - measured
            theta_deg = np.degrees(fields["theta_ref"])
            vel_sat = (health & _HF_VEL_PI_SAT) != 0

            velocity_plot = self._new_plot("Forward-velocity tracking", "Velocity", "m/s")
            self._curve(velocity_plot, t, target, GREEN,
                        "Velocity target — commanded forward speed (m/s)", 1.6)
            self._curve(velocity_plot, t, measured, BLUE,
                        "Measured velocity — average wheel-derived speed (m/s)", 1.4)
            self._curve(velocity_plot, t, error, YELLOW,
                        "Velocity error — target minus measured speed (m/s)", 1.0)
            radio_vel_limit = self._profile_limit("vel_max")
            if radio_vel_limit is not None:
                self._limit_pair(
                    velocity_plot, t, radio_vel_limit, WHITE,
                    "Radio-profile upper speed — CH9 profile bound (m/s)",
                    "Radio-profile lower speed — CH9 profile bound (m/s)")

            lean_plot = self._new_plot("Velocity-PI lean command", "Lean target", "°")
            self._curve(lean_plot, t, theta_deg, ORANGE,
                        "Lean target θref — acceleration angle requested by velocity PI (°)", 1.5)
            theta_limit = self._param_series("vel_pi_theta_max")
            if theta_limit is not None:
                self._limit_pair(
                    lean_plot, t, np.degrees(theta_limit), RED,
                    "Velocity-PI upper angle limit — positive lean clamp (°)",
                    "Velocity-PI lower angle limit — negative lean clamp (°)")
            self._add_regions(
                lean_plot, t, vel_sat, RED,
                "Velocity PI saturated — lean command reached its angle clamp")

            wheel_plot = self._new_plot("Wheel-speed governor", "Wheel speed", "turns/s")
            self._curve(wheel_plot, t, fields["wm_l_vel_turns_s"], BLUE,
                        "Left wheel speed — ODrive velocity feedback (turns/s)", 1.25)
            self._curve(wheel_plot, t, fields["wm_r_vel_turns_s"], GREEN,
                        "Right wheel speed — ODrive velocity feedback (turns/s)", 1.25)
            wheel_limit = self._param_series("wm_vel_limit")
            if wheel_limit is not None:
                self._limit_pair(
                    wheel_plot, t, wheel_limit, YELLOW,
                    "Soft governor upper limit — torque blocked above this speed",
                    "Soft governor lower limit — torque blocked below this speed")
                self._limit_pair(
                    wheel_plot, t, 2.0 * wheel_limit, RED,
                    "Runaway-fault upper threshold — 2× soft wheel-speed limit",
                    "Runaway-fault lower threshold — 2× soft wheel-speed limit",
                    Qt.PenStyle.DashDotLine)
            left_governor = (health & _HF_WM_L_VEL_LIMITED) != 0
            right_governor = (health & _HF_WM_R_VEL_LIMITED) != 0
            self._add_regions(wheel_plot, t, left_governor, BLUE,
                              "Left-wheel governor active")
            self._add_regions(wheel_plot, t, right_governor, GREEN,
                              "Right-wheel governor active")

            theta_rate_limit = self._param_series("vel_pi_rate_lim")
            theta_rate_dps = np.gradient(theta_deg, t) if t.size > 1 else np.zeros_like(t)
            rate_limited = np.zeros_like(theta_deg, dtype=bool)
            rate_limit_text = "unavailable"
            if theta_rate_limit is not None:
                rate_limit_dps = np.degrees(theta_rate_limit)
                rate_limited = np.abs(theta_rate_dps) >= 0.98 * rate_limit_dps
                rate_limit_text = f"{np.nanmax(rate_limit_dps):.2f} °/s"

            self._set_meta_rows([
                ("RMS velocity error", f"{metrics['vel_track_rms_ms']:.4f} m/s",
                 "Typical difference between target and measured speed."),
                ("Maximum velocity error", f"{np.max(np.abs(error)):.4f} m/s",
                 "Worst target-versus-measured difference."),
                ("Velocity-PI angle saturation", _event_summary(t, vel_sat),
                 "Intervals where θref reached vel_pi_theta_max."),
                ("Lean slew limit", rate_limit_text,
                 f"Derived near-limit activity: {_event_summary(t, rate_limited)}."),
                ("Left-wheel governor", _event_summary(t, left_governor),
                 "Intervals where the left wheel exceeded the soft speed limit."),
                ("Right-wheel governor", _event_summary(t, right_governor),
                 "Intervals where the right wheel exceeded the soft speed limit."),
                ("Active profile", self._profile_text(),
                 "Radio speed lines are profile limits; GUI motion commands bypass radio scaling."),
            ])

        elif flavor == "Yaw-PI":
            target_dps = np.degrees(fields["omega_cmd_rds"])
            measured_dps = np.degrees(fields["yaw_rate_rads"])
            error_dps = target_dps - measured_dps
            yaw_sat = (health & _HF_YAW_PI_SAT) != 0

            rate_plot = self._new_plot("Yaw-rate tracking", "Yaw rate", "°/s")
            self._curve(rate_plot, t, target_dps, GREEN,
                        "Yaw-rate target — commanded turn rate (°/s)", 1.6)
            self._curve(rate_plot, t, measured_dps, BLUE,
                        "Measured yaw rate — IMU turn rate (°/s)", 1.4)
            self._curve(rate_plot, t, error_dps, YELLOW,
                        "Yaw-rate error — target minus measured turn rate (°/s)", 1.0)
            radio_yaw_limit = self._profile_limit("yaw_max")
            if radio_yaw_limit is not None:
                self._limit_pair(
                    rate_plot, t, np.degrees(radio_yaw_limit), WHITE,
                    "Radio-profile upper yaw rate — CH9 profile bound (°/s)",
                    "Radio-profile lower yaw rate — CH9 profile bound (°/s)")

            torque_plot = self._new_plot("Yaw-controller effort", "Yaw torque", "N·m")
            self._curve(torque_plot, t, fields["tau_yaw"], ORANGE,
                        "Yaw torque τyaw — differential wheel-torque contribution (N·m)", 1.5)
            yaw_torque_limit = self._param_series("yaw_pi_torque_max")
            if yaw_torque_limit is not None:
                self._limit_pair(
                    torque_plot, t, yaw_torque_limit, RED,
                    "Yaw-controller upper torque limit — positive clamp (N·m)",
                    "Yaw-controller lower torque limit — negative clamp (N·m)")
            self._add_regions(torque_plot, t, yaw_sat, RED,
                              "Yaw PI saturated — differential torque reached its clamp")

            self._set_meta_rows([
                ("RMS yaw-rate error", f"{np.degrees(metrics['yaw_track_rms_rads']):.3f} °/s",
                 "Typical difference between commanded and measured turn rate."),
                ("Maximum yaw-rate error", f"{np.max(np.abs(error_dps)):.3f} °/s",
                 "Worst commanded-versus-measured turn-rate difference."),
                ("Maximum yaw torque", f"{np.max(np.abs(fields['tau_yaw'])):.4f} N·m",
                 "Largest differential torque requested by the yaw controller."),
                ("Yaw-torque saturation", _event_summary(t, yaw_sat),
                 "Intervals where τyaw reached yaw_pi_torque_max."),
                ("Active profile", self._profile_text(),
                 "Radio yaw lines are profile limits; GUI motion commands bypass radio scaling."),
            ])

        elif flavor == "Torque / Current":
            torque_limit = self._profile_limit("torque_lim")
            torque_plot = self._new_plot("Wheel torque output", "Torque", "N·m")
            self._curve(torque_plot, t, fields["whl_tau_l"], BLUE,
                        "Left-wheel torque — final command sent to left ODrive (N·m)", 1.45)
            self._curve(torque_plot, t, fields["whl_tau_r"], GREEN,
                        "Right-wheel torque — final command sent to right ODrive (N·m)", 1.45)
            self._curve(torque_plot, t, fields["tau_sym"], ORANGE,
                        "Balance torque τsym — symmetric LQR contribution before mixing (N·m)", 1.0)
            self._curve(torque_plot, t, fields["tau_yaw"], YELLOW,
                        "Yaw torque τyaw — differential turning contribution before mixing (N·m)", 1.0)
            if torque_limit is not None:
                self._limit_pair(
                    torque_plot, t, torque_limit, RED,
                    "Active wheel-torque upper limit — selected profile clamp (N·m)",
                    "Active wheel-torque lower limit — selected profile clamp (N·m)")
                torque_sat = (
                    (np.abs(fields["whl_tau_l"]) >= 0.98 * torque_limit)
                    | (np.abs(fields["whl_tau_r"]) >= 0.98 * torque_limit)
                )
                self._add_regions(torque_plot, t, torque_sat, RED,
                                  "Wheel torque at active profile limit")
            else:
                torque_sat = np.zeros(t.size, dtype=bool)

            current_plot = self._new_plot("Hip current during Phase-1 tuning", "Current", "A")
            self._curve(current_plot, t, fields["hip_l_current_a"], RED,
                        "Left-hip current — measured AK45 current (A)", 1.35)
            self._curve(current_plot, t, fields["hip_r_current_a"], ORANGE,
                        "Right-hip current — measured AK45 current (A)", 1.35)
            self._limit_pair(
                current_plot, t, HIP_CURRENT_EPS_A, YELLOW,
                "Phase-1 upper idle-current threshold — tuning safety check (A)",
                "Phase-1 lower idle-current threshold — tuning safety check (A)")

            torque_limit_text = "unavailable"
            if torque_limit is not None:
                unique_limits = np.unique(np.round(torque_limit, 6))
                torque_limit_text = " → ".join(f"±{value:g} N·m" for value in unique_limits)
            self._set_meta_rows([
                ("Maximum left-wheel torque", f"{np.max(np.abs(fields['whl_tau_l'])):.4f} N·m",
                 "Largest final torque command sent to the left ODrive."),
                ("Maximum right-wheel torque", f"{np.max(np.abs(fields['whl_tau_r'])):.4f} N·m",
                 "Largest final torque command sent to the right ODrive."),
                ("Active wheel-torque limit", torque_limit_text,
                 "Reconstructed from active_profile and profile torque settings."),
                ("Wheel-torque saturation", _event_summary(t, torque_sat),
                 "Intervals within 2% of the active profile torque clamp."),
                ("Maximum left-hip current", f"{metrics['max_hip_l_current_a']:.4f} A",
                 "Phase-1 hips should remain close to zero current."),
                ("Maximum right-hip current", f"{metrics['max_hip_r_current_a']:.4f} A",
                 "Phase-1 hips should remain close to zero current."),
                ("Active profile", self._profile_text(),
                 "Profile changes also change the wheel-torque limit."),
            ])

        elif flavor == "Gain schedule":
            alpha_plot = self._new_plot("Leg-height gain schedule", "Schedule position α")
            if run.has_gain_sched_alpha:
                alpha = fields["gain_sched_alpha"]
                self._curve(alpha_plot, t, alpha, BLUE,
                            "Gain-schedule position α — 0=retracted, 1=extended", 1.6)
                self._curve(alpha_plot, t, np.zeros_like(t), GREEN,
                            "Retracted LQR anchor — α=0", 1.0, Qt.PenStyle.DashLine)
                self._curve(alpha_plot, t, np.ones_like(t), ORANGE,
                            "Extended LQR anchor — α=1", 1.0, Qt.PenStyle.DashLine)
                alpha_range = metrics["gain_sched_alpha_range"]
                self._set_meta_rows([
                    ("Minimum α", f"{alpha_range['min']:.4f}",
                     "Smallest logged gain-schedule blend."),
                    ("Maximum α", f"{alpha_range['max']:.4f}",
                     "Largest logged gain-schedule blend."),
                    ("Mean α", f"{alpha_range['mean']:.4f}",
                     "Average retracted-to-extended blend."),
                    ("Active profile", self._profile_text(),
                     "Speed profile is independent of leg-height gain scheduling."),
                ])
            else:
                text = pg.TextItem("Gain-schedule telemetry is unavailable in this log.",
                                   color=DIM, anchor=(0.5, 0.5))
                alpha_plot.addItem(text)
                text.setPos(float(t[len(t) // 2]) if len(t) else 0.0, 0.5)
                self._set_meta_rows([
                    ("Gain-schedule position α", "unavailable",
                     "This telemetry version did not record the schedule blend."),
                ])

        self._finish_plots()
        self._update_status(t, health, metrics)

    def _update_status(self, t: np.ndarray, health: np.ndarray, metrics: dict):
        warnings = []
        if metrics["fault_fired"]:
            warnings.append(f"FAULT codes {metrics['faults_seen']}")
        if np.any((health & _HF_VEL_PI_SAT) != 0):
            warnings.append("velocity-PI angle saturation occurred")
        if np.any((health & _HF_YAW_PI_SAT) != 0):
            warnings.append("yaw-torque saturation occurred")
        if np.any((health & (_HF_WM_L_VEL_LIMITED | _HF_WM_R_VEL_LIMITED)) != 0):
            warnings.append("wheel-speed governor activated")
        overrun = (health & _HF_LOOP_OVERRUN) != 0
        if np.any(overrun):
            warnings.append(
                f"loop-overrun warning active: {_event_summary(t, overrun)} "
                "(the flag stays asserted for 1 s after an overrun)")
        if self._params is None:
            warnings.append("parameter limits unavailable because no parameter snapshot was found")
        if self._param_error:
            warnings.append(f"parameter sidecar could not be parsed: {self._param_error}")

        if warnings:
            self._safety_lbl.setStyleSheet(f"color: {YELLOW}; font-size: 11px;")
            self._safety_lbl.setText("Review: " + "; ".join(warnings) + ".")
        else:
            self._safety_lbl.setStyleSheet(f"color: {GREEN}; font-size: 11px;")
            self._safety_lbl.setText("No fault, controller saturation, wheel governor, or loop-overrun warning was recorded.")
