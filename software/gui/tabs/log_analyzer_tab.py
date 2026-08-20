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
    QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from analysis.jump_analysis import (
    CURRENT_PHASE_NAMES, LEGACY_PHASE_NAMES, analyze_jumps, jump_focus_mask,
    phase_name,
)
from analysis.param_sidecar import (
    ParamSidecar, active_profile_series, load_host_param_sidecar, load_matching_sidecar,
)
from analysis.leg_height_sweep import PlateauMetrics, fit_trim_schedule, plateau_report
from analysis.wlog_metrics import (
    HIP_TORQUE_LIMIT_NM, SETTLE_DEG, compute_metrics, decode_run,
)
from .generated_protocol import STATE_NAMES
from .log_paths import RUNS_DIR
from .theme import BG, BLUE, BORDER, DIM, GREEN, ORANGE, RED, SURFACE, TEXT, WHITE, YELLOW

pg.setConfigOptions(antialias=True, background=BG, foreground=TEXT)

_FLAVORS = ["Overview", "Jumping", "LQR", "Vel-PI", "Yaw-PI", "Hip",
            "Torque / Current", "Gain schedule", "Leg-height sweep"]

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


def _discrete_change_events(t: np.ndarray, values: np.ndarray,
                            label_fn) -> list[tuple[float, str, str]]:
    """Detect transitions in a discrete-valued telemetry field (e.g. robot_state,
    active_profile). label_fn(old, new) builds the marker text; the initial
    value at t[0] is not itself a "change" and is not reported."""
    values = np.asarray(values, dtype=np.int64)
    if values.size < 2:
        return []
    change_idx = np.flatnonzero(np.diff(values)) + 1
    events = []
    for i in change_idx:
        label = label_fn(int(values[i - 1]), int(values[i]))
        events.append((float(t[i]), label, label))
    return events


def _mode_change_events(t: np.ndarray, robot_state: np.ndarray) -> list[tuple[float, str, str]]:
    return _discrete_change_events(
        t, robot_state,
        lambda old, new: f"{STATE_NAMES.get(old, str(old))} → {STATE_NAMES.get(new, str(new))}")


def _profile_change_events(t: np.ndarray, active_profile: np.ndarray) -> list[tuple[float, str, str]]:
    return _discrete_change_events(t, active_profile, lambda old, new: f"P{old + 1} → P{new + 1}")


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
        self._plateau_cache: list[PlateauMetrics] | None = None
        self._plots: list[pg.PlotWidget] = []
        self._param_changes: list[tuple[float, str, str]] = []
        self._mode_changes: list[tuple[float, str, str]] = []
        self._profile_changes: list[tuple[float, str, str]] = []

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
        chart_scroll = QScrollArea()
        chart_scroll.setWidgetResizable(True)
        chart_scroll.setFrameShape(QFrame.Shape.NoFrame)
        chart_scroll.setWidget(chart_frame)
        body.addWidget(chart_scroll)

        meta_frame = QFrame()
        meta_frame.setStyleSheet(
            f"QFrame {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 4px; }}"
        )
        self._meta_layout = QVBoxLayout(meta_frame)
        meta_title = QLabel("Metrics and limits")
        meta_title.setStyleSheet(f"color: {TEXT}; font-weight: bold; font-size: 13px;")
        self._meta_layout.addWidget(meta_title)
        self._meta_layout.addStretch()
        meta_scroll = QScrollArea()
        meta_scroll.setWidgetResizable(True)
        meta_scroll.setFrameShape(QFrame.Shape.NoFrame)
        meta_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        meta_scroll.setWidget(meta_frame)
        meta_scroll.setMinimumWidth(285)
        meta_scroll.setMaximumWidth(390)
        body.addWidget(meta_scroll)
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
            self._plateau_cache = None
            self._lbl_file.setText(f"Failed to open {path.name}: {exc}")
            self._redraw()
            return

        self._run = run
        self._metrics = compute_metrics(run)
        self._params = None
        self._param_error = ""
        self._plateau_cache = None
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
        plot.setMinimumHeight(165)
        plot.addLegend(offset=(5, 5), verSpacing=-2)
        is_first = not self._plots
        if self._plots:
            plot.setXLink(self._plots[0])
        self._plots.append(plot)
        self._charts_layout.addWidget(plot, 1)

        # Change markers: drawn on every plot in every flavor so context (what
        # mode/profile/parameter changed) is visible no matter which signal
        # is being inspected; only the first (topmost) plot gets on-chart text
        # labels since the x-axis is shared across the whole flavor.
        self._add_event_markers(
            plot, self._mode_changes, WHITE, "Mode change", "Robot mode changed",
            style=Qt.PenStyle.DashDotLine, width=1.3, show_labels=is_first)
        self._add_event_markers(
            plot, self._profile_changes, ORANGE, "Profile change", "Speed profile changed",
            style=Qt.PenStyle.DotLine, width=1.3, show_labels=is_first)
        self._add_event_markers(
            plot, self._param_changes, YELLOW, "Parameter change", "Firmware parameter changed",
            style=Qt.PenStyle.DashLine, width=1.0, show_labels=is_first)
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
    def _add_event_markers(plot: pg.PlotWidget, events: list[tuple[float, str, str]],
                           color: str, legend_name: str, tooltip_prefix: str,
                           style: Qt.PenStyle = Qt.PenStyle.DashLine,
                           width: float = 1.0, show_labels: bool = False):
        """Draw synchronized vertical markers (mode/profile/parameter changes)
        without affecting auto-range. events are (t_s, short_label, full_label)."""
        for t_s, short_label, full_label in events:
            marker = pg.InfiniteLine(
                pos=t_s, angle=90, movable=False,
                pen=pg.mkPen(color, width=width, style=style),
                label=short_label if show_labels else None,
                labelOpts={"color": color, "position": 0.92},
            )
            marker.setToolTip(f"{tooltip_prefix} at {t_s:.3f} s: {full_label}")
            marker.setZValue(30)
            plot.addItem(marker, ignoreBounds=True)

        if events and plot.plotItem.legend is not None:
            sample = pg.PlotDataItem(pen=pg.mkPen(color, width=width + 0.5, style=style))
            plot.plotItem.legend.addItem(sample, legend_name)
            _set_legend_tooltip(
                plot.plotItem.legend, sample,
                f"{tooltip_prefix}; hover a marker for details.")

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

        param_change_groups = _parameter_change_groups(self._params, run.t_micros, t)
        self._param_changes = [
            (time_s, _parameter_change_label(events),
             _parameter_change_label(events, max_names=len(events)))
            for time_s, events in param_change_groups
        ]
        self._mode_changes = _mode_change_events(t, fields["robot_state"])
        self._profile_changes = _profile_change_events(t, fields["active_profile"])

        if flavor == "Overview":
            velocity_plot = self._new_plot(
                "Forward command and measured motion", "Velocity", "m/s",
                auto_si_prefix=False)
            self._curve(velocity_plot, t, fields["v_ref"], GREEN,
                        "Velocity command vref — requested forward speed (m/s)", 1.6)
            self._curve(velocity_plot, t, fields["wheel_vel_avg"], BLUE,
                        "Average wheel velocity — measured forward speed (m/s)", 1.45)

            yaw_plot = self._new_plot(
                "Yaw command and measured turn rate", "Yaw rate", "°/s",
                auto_si_prefix=False)
            self._curve(yaw_plot, t, np.degrees(fields["omega_cmd_rds"]), GREEN,
                        "Yaw command ωcmd — requested turn rate (°/s)", 1.6)
            self._curve(yaw_plot, t, np.degrees(fields["yaw_rate_rads"]), BLUE,
                        "Measured yaw rate — IMU turn rate (°/s)", 1.45)

            pitch_plot = self._new_plot(
                "Pitch/roll and controller lean target", "Angle", "°",
                auto_si_prefix=False)
            self._curve(pitch_plot, t, np.degrees(fields["pitch_rad"]), BLUE,
                        "Pitch — measured body tilt (°)", 1.6)
            self._curve(pitch_plot, t, np.degrees(fields["theta_ref"] + fields["pitch_trim_rad"]), ORANGE,
                        "Lean target θref+trim — true LQR balance point (°)", 1.4)
            self._curve(pitch_plot, t, np.degrees(fields["roll_rad"]), RED,
                        "Roll — measured body roll (°)", 1.4)

            change_text = "none"
            if param_change_groups:
                event_count = sum(len(events) for _time, events in param_change_groups)
                change_text = f"{event_count} change{'s' if event_count != 1 else ''} at " + \
                    ", ".join(f"{time_s:.2f} s" for time_s, _events in param_change_groups[:4])
                if len(param_change_groups) > 4:
                    change_text += f", +{len(param_change_groups) - 4} more times"
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

        elif flavor == "Jumping":
            self._draw_jumping(t, fields, run)

        elif flavor == "LQR":
            pitch_deg = np.degrees(fields["pitch_rad"])
            target_deg = np.degrees(fields["theta_ref"] + fields["pitch_trim_rad"])
            error_deg = pitch_deg - target_deg
            rate_dps = np.degrees(fields["pitch_rate_rads"])

            angle_plot = self._new_plot("Body angle and controller target", "Angle", "°")
            self._curve(angle_plot, t, pitch_deg, BLUE,
                        "Pitch — measured body tilt (°)", 1.6)
            self._curve(angle_plot, t, target_deg, GREEN,
                        "Lean target θref+trim — true LQR balance point (°)", 1.4)

            error_plot = self._new_plot("Balance tracking error", "Pitch error", "°")
            self._curve(error_plot, t, error_deg, YELLOW,
                        "Pitch error — measured pitch minus trimmed lean target (°)", 1.5)
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
                 "Untrimmed θref, matching the sim's fitness formula (analysis/wlog_metrics.py) — "
                 "the chart's error curve above is trimmed and may differ."),
                ("Maximum pitch error", f"{metrics['max_pitch_deg']:.3f}°",
                 "Untrimmed θref, matching the sim's fitness formula — see note above."),
                ("Pitch-error ISE", f"{ise_deg2_s:.3f} °²·s",
                 "Accumulated squared tracking error over the entire log (untrimmed θref)."),
                ("Settling time", f"{metrics['settle_time_s']:.3f} s",
                 f"First continuous 0.5 s inside the ±{SETTLE_DEG:g}° error band (untrimmed θref)."),
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
            # Asymmetric, gain-scheduled clamp (theta_max_fwd/bwd_ret/ext) —
            # these reference lines show the retracted-anchor (alpha=0) bound
            # from the .PARAMS sidecar dump, not the live per-tick blend with
            # the extended-leg value (that blend isn't logged).
            theta_fwd_limit = self._param_series("theta_max_fwd_ret")
            theta_bwd_limit = self._param_series("theta_max_bwd_ret")
            if theta_fwd_limit is not None and np.any(np.isfinite(theta_fwd_limit)):
                self._curve(lean_plot, t, np.degrees(theta_fwd_limit),
                            RED, "Velocity-PI upper angle limit — forward lean clamp, retracted (°)",
                            1.0, Qt.PenStyle.DashLine)
            if theta_bwd_limit is not None and np.any(np.isfinite(theta_bwd_limit)):
                self._curve(lean_plot, t, -np.degrees(theta_bwd_limit),
                            RED, "Velocity-PI lower angle limit — backward lean clamp, retracted (°)",
                            1.0, Qt.PenStyle.DashLine)
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

        elif flavor == "Hip":
            hip_l_pos_deg = np.degrees(fields["hip_l_pos_rad"])
            hip_l_cmd_deg = np.degrees(fields["hip_l_cmd_pos_rad"])
            hip_r_pos_deg = np.degrees(fields["hip_r_pos_rad"])
            hip_r_cmd_deg = np.degrees(fields["hip_r_cmd_pos_rad"])
            hip_l_err_deg = hip_l_pos_deg - hip_l_cmd_deg
            hip_r_err_deg = hip_r_pos_deg - hip_r_cmd_deg

            left_pos_plot = self._new_plot("Left hip position", "Angle", "°")
            self._curve(left_pos_plot, t, hip_l_pos_deg, BLUE,
                        "Left-hip angle — measured AK45 position (°)", 1.6)
            self._curve(left_pos_plot, t, hip_l_cmd_deg, GREEN,
                        "Left-hip target — commanded MIT position (°)", 1.4)

            right_pos_plot = self._new_plot("Right hip position", "Angle", "°")
            self._curve(right_pos_plot, t, hip_r_pos_deg, BLUE,
                        "Right-hip angle — measured AK45 position (°)", 1.6)
            self._curve(right_pos_plot, t, hip_r_cmd_deg, GREEN,
                        "Right-hip target — commanded MIT position (°)", 1.4)

            torque_plot = self._new_plot("Hip torque and limit", "Torque", "N·m")
            self._curve(torque_plot, t, fields["hip_l_torque_nm"], RED,
                        "Left-hip torque — measured AK45 shaft torque (N·m)", 1.35)
            self._curve(torque_plot, t, fields["hip_r_torque_nm"], ORANGE,
                        "Right-hip torque — measured AK45 shaft torque (N·m)", 1.35)
            self._limit_pair(
                torque_plot, t, HIP_TORQUE_LIMIT_NM, YELLOW,
                "Hip upper torque limit (N·m)", "Hip lower torque limit (N·m)")

            self._set_meta_rows([
                ("Maximum left-hip position error", f"{np.max(np.abs(hip_l_err_deg)):.3f}°",
                 "Worst measured-versus-target difference for the left hip."),
                ("Maximum right-hip position error", f"{np.max(np.abs(hip_r_err_deg)):.3f}°",
                 "Worst measured-versus-target difference for the right hip."),
                ("Maximum left-hip torque", f"{metrics['max_hip_l_torque_nm']:.4f} N·m",
                 f"Hip torque limit is ±{HIP_TORQUE_LIMIT_NM:g} N·m."),
                ("Maximum right-hip torque", f"{metrics['max_hip_r_torque_nm']:.4f} N·m",
                 f"Hip torque limit is ±{HIP_TORQUE_LIMIT_NM:g} N·m."),
                ("Active profile", self._profile_text(),
                 "Speed profile is independent of hip motion."),
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

            hip_trq_plot = self._new_plot("Hip torque during Phase-1 tuning", "Torque", "N·m")
            self._curve(hip_trq_plot, t, fields["hip_l_torque_nm"], RED,
                        "Left-hip torque — measured AK45 shaft torque (N·m)", 1.35)
            self._curve(hip_trq_plot, t, fields["hip_r_torque_nm"], ORANGE,
                        "Right-hip torque — measured AK45 shaft torque (N·m)", 1.35)
            self._limit_pair(
                hip_trq_plot, t, HIP_TORQUE_LIMIT_NM, YELLOW,
                "Phase-1 upper idle-torque threshold — tuning safety check (N·m)",
                "Phase-1 lower idle-torque threshold — tuning safety check (N·m)")

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
                ("Maximum left-hip torque", f"{metrics['max_hip_l_torque_nm']:.4f} N·m",
                 "Phase-1 hips should remain close to zero torque."),
                ("Maximum right-hip torque", f"{metrics['max_hip_r_torque_nm']:.4f} N·m",
                 "Phase-1 hips should remain close to zero torque."),
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

        elif flavor == "Leg-height sweep":
            self._draw_leg_height_sweep(t, fields, run)

        self._finish_plots()
        self._update_status(t, health, metrics)

    # ── Jumping ────────────────────────────────────────────────────────────

    def _draw_jumping(self, t: np.ndarray, fields: dict, run):
        """Draw phase-aligned launch, flight, landing, and recovery evidence."""
        def sidecar_or(name: str, default: float) -> float:
            value = self._sidecar_value(name)
            return default if value is None else value

        air_accel = sidecar_or("jump_air_accel_z", -3.0)
        land_accel = sidecar_or("jump_land_accel_z", 1.5)
        gyro_impulse = sidecar_or("jump_land_gyro_imp", 2.5)
        min_air_s = sidecar_or("jump_land_min_air", 0.16)
        modern_hint = (
            True if self._params is not None
            and "jmp_handoff_timeout" in self._params.names else None
        )
        episodes = analyze_jumps(
            t, fields, modern_phases=modern_hint,
            airborne_accel_z=air_accel, landing_accel_z=land_accel,
            gyro_impulse=gyro_impulse, min_air_s=min_air_s,
        )

        if not episodes:
            plot = self._new_plot("Jumping", "Jump phase")
            text = pg.TextItem("No JUMPING episode is present in this log.",
                               color=DIM, anchor=(0.5, 0.5))
            plot.addItem(text)
            text.setPos(float(t[len(t) // 2]) if len(t) else 0.0, 0.0)
            self._set_meta_rows([(
                "Jumping episodes", "none found",
                "Load a capture containing robot_state=JUMPING (7). Start the SD log "
                "in STANDBY before commanding a jump.")])
            return

        jump_mask = np.asarray(fields["robot_state"], dtype=np.int64) == 7
        focus = jump_focus_mask(t, episodes, before_s=0.25, after_s=0.75)

        # Plot only the focused samples. Feeding a long WLOG to every curve with
        # most rows replaced by NaN creates very large Qt painter paths and can
        # crash pyqtgraph's Windows renderer. Keep one NaN separator after each
        # focus interval so separate attempts never get joined by a line.
        plot_mask = focus.copy()
        if plot_mask.size > 1:
            plot_mask[1:] |= focus[:-1] & ~focus[1:]
        plot_indices = np.flatnonzero(plot_mask)
        plot_t = t[plot_indices]

        def focused(values, jump_only: bool = False) -> np.ndarray:
            values = np.asarray(values, dtype=np.float64)
            visible = jump_mask if jump_only else focus
            return np.where(visible[plot_indices], values[plot_indices], np.nan)

        def parameter_series(name: str, default: float) -> np.ndarray:
            values = self._param_series(name)
            if values is None:
                return np.full(t.size, default, dtype=np.float64)
            return np.asarray(values, dtype=np.float64)

        phase_events: list[tuple[float, str, str]] = []
        landing_events: list[tuple[float, str, str]] = []
        for episode in episodes:
            for phase, index in episode.phase_entries:
                name = phase_name(phase, episode.modern_phases)
                label = f"J{episode.number} {name}"
                phase_events.append((float(t[index]), label, label))
            if episode.landing_index is not None:
                offset = float(t[episode.landing_index] - t[episode.i0])
                label = f"J{episode.number} touchdown"
                landing_events.append((
                    float(t[episode.landing_index]), label,
                    f"{label} at +{offset:.3f} s ({episode.landing_source})"))

        def add_jump_markers(plot: pg.PlotWidget, show_labels: bool = False):
            self._add_event_markers(
                plot, phase_events, WHITE, "Jump phase",
                "Jump phase transition", style=Qt.PenStyle.DotLine,
                width=1.0, show_labels=show_labels)
            self._add_event_markers(
                plot, landing_events, RED, "Touchdown",
                "Landing detection", style=Qt.PenStyle.DashDotLine,
                width=1.5, show_labels=show_labels)

        # Phase timeline. Phase zero is a real CROUCH, not an inactive value;
        # robot_state is therefore the mask that distinguishes it outside jumps.
        phase_plot = self._new_plot("Jump phase timeline", "Phase")
        phase_values = focused(fields["jump_state"], jump_only=True)
        self._curve(phase_plot, plot_t, phase_values, BLUE,
                    "Jump phase — firmware phase code", 2.0)
        names = (CURRENT_PHASE_NAMES if any(e.modern_phases for e in episodes)
                 else LEGACY_PHASE_NAMES)
        phase_plot.getAxis("left").setTicks([[(code, name) for code, name in names.items()]])
        phase_plot.setYRange(-0.35, max(names) + 0.35, padding=0)
        add_jump_markers(phase_plot, show_labels=True)

        # Pilot command + final-crouch nudge and wheel-derived response.
        velocity_plot = self._new_plot(
            "Forward command, nudge, and measured motion", "Velocity", "m/s",
            auto_si_prefix=False)
        self._curve(velocity_plot, plot_t, focused(fields["v_ref"]), GREEN,
                    "Effective velocity target — pilot command plus active jump nudge (m/s)", 1.6)
        self._curve(velocity_plot, plot_t, focused(fields["wheel_vel_avg"]), BLUE,
                    "Wheel-derived velocity — meaningful on the floor, free-spin proxy in flight (m/s)", 1.4)
        add_jump_markers(velocity_plot)

        # Full attitude and the exact LQR target. Watchdog/barrier references
        # are reconstructed with the logged gain-schedule alpha when available.
        attitude_plot = self._new_plot(
            "Body attitude, balance target, and pitch safety bounds", "Angle", "°",
            auto_si_prefix=False)
        pitch_deg = np.degrees(fields["pitch_rad"])
        roll_deg = np.degrees(fields["roll_rad"])
        target_deg = np.degrees(fields["theta_ref"] + fields["pitch_trim_rad"])
        self._curve(attitude_plot, plot_t, focused(pitch_deg), BLUE,
                    "Pitch — measured body angle (°)", 1.7)
        self._curve(attitude_plot, plot_t, focused(roll_deg), RED,
                    "Roll — measured lateral body angle (°)", 1.35)
        self._curve(attitude_plot, plot_t, focused(target_deg), GREEN,
                    "LQR target — θref plus applied pitch trim (°)", 1.35)

        alpha = (np.asarray(fields["gain_sched_alpha"], dtype=np.float64)
                 if "gain_sched_alpha" in fields else np.zeros(t.size))
        wd_fwd_ret = self._param_series("pitch_wd_fwd_ret")
        wd_bwd_ret = self._param_series("pitch_wd_bwd_ret")
        wd_fwd = wd_bwd = None
        if wd_fwd_ret is not None:
            wd_fwd_ext = self._param_series("pitch_wd_fwd_ext")
            wd_fwd = np.asarray(wd_fwd_ret) + alpha * (
                np.asarray(wd_fwd_ext if wd_fwd_ext is not None else wd_fwd_ret)
                - np.asarray(wd_fwd_ret))
            self._curve(attitude_plot, plot_t, focused(np.degrees(wd_fwd)), ORANGE,
                        "Forward pitch watchdog — live scheduled threshold (°)",
                        1.0, Qt.PenStyle.DashLine)
        if wd_bwd_ret is not None:
            wd_bwd_ext = self._param_series("pitch_wd_bwd_ext")
            wd_bwd = np.asarray(wd_bwd_ret) + alpha * (
                np.asarray(wd_bwd_ext if wd_bwd_ext is not None else wd_bwd_ret)
                - np.asarray(wd_bwd_ret))
            self._curve(attitude_plot, plot_t, focused(-np.degrees(wd_bwd)), ORANGE,
                        "Backward pitch watchdog — live scheduled threshold (°)",
                        1.0, Qt.PenStyle.DashLine)
        barrier_ret = self._param_series("lqr_barrier_th_ret")
        if barrier_ret is not None:
            barrier_ext = self._param_series("lqr_barrier_th_ext")
            barrier = np.asarray(barrier_ret) + alpha * (
                np.asarray(barrier_ext if barrier_ext is not None else barrier_ret)
                - np.asarray(barrier_ret))
            self._curve(attitude_plot, plot_t, focused(-np.degrees(barrier)), YELLOW,
                        "Backward recovery barrier — extra LQR torque begins here (°)",
                        1.0, Qt.PenStyle.DotLine)
        add_jump_markers(attitude_plot)

        rates_plot = self._new_plot(
            "IMU angular rates", "Angular rate", "°/s", auto_si_prefix=False)
        self._curve(rates_plot, plot_t, focused(np.degrees(fields["pitch_rate_rads"])), BLUE,
                    "Pitch rate — rotation about the wheel axle (°/s)", 1.5)
        self._curve(rates_plot, plot_t, focused(np.degrees(fields["roll_rate_rads"])), RED,
                    "Roll rate — lateral rotation (°/s)", 1.2)
        self._curve(rates_plot, plot_t, focused(np.degrees(fields["yaw_rate_rads"])), GREEN,
                    "Yaw rate — heading rotation (°/s)", 1.2)
        add_jump_markers(rates_plot)

        accel_plot = self._new_plot(
            "IMU gravity-removed linear acceleration", "Acceleration", "m/s²",
            auto_si_prefix=False)
        self._curve(accel_plot, plot_t, focused(fields["accel_x_ms2"]), BLUE,
                    "Body-X acceleration — forward (+) / backward (−) (m/s²)", 1.2)
        self._curve(accel_plot, plot_t, focused(fields["accel_y_ms2"]), GREEN,
                    "Body-Y acceleration — left (+) / right (−) (m/s²)", 1.2)
        self._curve(accel_plot, plot_t, focused(fields["accel_z_ms2"]), RED,
                    "Body-Z acceleration — landing detector input (m/s²)", 1.5)
        self._curve(accel_plot, plot_t, focused(parameter_series("jump_air_accel_z", air_accel)),
                    ORANGE, "Airborne latch threshold — jump_air_accel_z (m/s²)",
                    1.0, Qt.PenStyle.DashLine)
        self._curve(accel_plot, plot_t, focused(parameter_series("jump_land_accel_z", land_accel)),
                    YELLOW, "Landing rebound threshold — jump_land_accel_z (m/s²)",
                    1.0, Qt.PenStyle.DashLine)
        add_jump_markers(accel_plot)

        hip_pos_plot = self._new_plot("Hip position and command", "Hip angle", "°")
        self._curve(hip_pos_plot, plot_t, focused(np.degrees(fields["hip_l_pos_rad"])), BLUE,
                    "Left hip measured position (°)", 1.5)
        self._curve(hip_pos_plot, plot_t, focused(np.degrees(fields["hip_l_cmd_pos_rad"])),
                    GREEN, "Left hip commanded position (°)", 1.1, Qt.PenStyle.DashLine)
        self._curve(hip_pos_plot, plot_t, focused(np.degrees(fields["hip_r_pos_rad"])), RED,
                    "Right hip measured position (°)", 1.5)
        self._curve(hip_pos_plot, plot_t, focused(np.degrees(fields["hip_r_cmd_pos_rad"])),
                    ORANGE, "Right hip commanded position (°)", 1.1, Qt.PenStyle.DashLine)
        add_jump_markers(hip_pos_plot)

        hip_vel_plot = self._new_plot("Hip extension/retraction rates", "Hip rate", "°/s")
        self._curve(hip_vel_plot, plot_t, focused(np.degrees(fields["hip_l_vel_rads"])), BLUE,
                    "Left hip measured velocity (°/s)", 1.35)
        self._curve(hip_vel_plot, plot_t, focused(np.degrees(fields["hip_r_vel_rads"])), RED,
                    "Right hip measured velocity (°/s)", 1.35)
        add_jump_markers(hip_vel_plot)

        hip_torque_plot = self._new_plot("Hip launch and landing effort", "Hip torque", "N·m")
        self._curve(hip_torque_plot, plot_t, focused(fields["hip_l_torque_nm"]), BLUE,
                    "Left hip measured torque (N·m)", 1.4)
        self._curve(hip_torque_plot, plot_t, focused(fields["hip_r_torque_nm"]), RED,
                    "Right hip measured torque (N·m)", 1.4)
        self._limit_pair(
            hip_torque_plot, plot_t, focused(np.full(t.size, HIP_TORQUE_LIMIT_NM)), YELLOW,
            "Hip positive protocol limit (N·m)", "Hip negative protocol limit (N·m)")
        add_jump_markers(hip_torque_plot)

        wheel_speed_plot = self._new_plot("Wheel speed and recovery authority", "Wheel speed", "turns/s")
        self._curve(wheel_speed_plot, plot_t, focused(fields["wm_l_vel_turns_s"]), BLUE,
                    "Left wheel velocity — ODrive feedback (turns/s)", 1.4)
        self._curve(wheel_speed_plot, plot_t, focused(fields["wm_r_vel_turns_s"]), GREEN,
                    "Right wheel velocity — ODrive feedback (turns/s)", 1.4)
        normal_wheel_limit = parameter_series("wm_vel_limit", 6.0)
        handoff_wheel_limit = parameter_series("jmp_handoff_vel_lim", np.nan)
        self._limit_pair(
            wheel_speed_plot, plot_t, focused(normal_wheel_limit), YELLOW,
            "RUNNING wheel governor upper limit (turns/s)",
            "RUNNING wheel governor lower limit (turns/s)")
        if np.any(np.isfinite(handoff_wheel_limit)):
            inherited = np.where(handoff_wheel_limit > 0.0,
                                 handoff_wheel_limit, normal_wheel_limit)
            self._limit_pair(
                wheel_speed_plot, plot_t, focused(inherited), ORANGE,
                "Jump-handoff wheel governor upper limit (turns/s)",
                "Jump-handoff wheel governor lower limit (turns/s)",
                Qt.PenStyle.DashDotLine)
        health = fields["health_flags"].astype(np.int64)
        self._add_regions(wheel_speed_plot, t, focus & ((health & _HF_WM_L_VEL_LIMITED) != 0),
                          BLUE, "Left-wheel governor active")
        self._add_regions(wheel_speed_plot, t, focus & ((health & _HF_WM_R_VEL_LIMITED) != 0),
                          GREEN, "Right-wheel governor active")
        add_jump_markers(wheel_speed_plot)

        wheel_torque_plot = self._new_plot("Wheel balance and yaw torque", "Wheel torque", "N·m")
        self._curve(wheel_torque_plot, plot_t, focused(fields["whl_tau_l"]), BLUE,
                    "Left wheel final commanded torque (N·m)", 1.45)
        self._curve(wheel_torque_plot, plot_t, focused(fields["whl_tau_r"]), GREEN,
                    "Right wheel final commanded torque (N·m)", 1.45)
        self._curve(wheel_torque_plot, plot_t, focused(fields["tau_sym"]), ORANGE,
                    "Symmetric LQR torque — pitch/reaction-wheel contribution (N·m)", 1.1)
        self._curve(wheel_torque_plot, plot_t, focused(fields["tau_yaw"]), YELLOW,
                    "Differential yaw torque contribution (N·m)", 1.0)
        handoff_torque = self._param_series("jmp_handoff_torque")
        if handoff_torque is not None:
            running_torque = self._profile_limit("torque_lim")
            inherited = np.asarray(handoff_torque)
            if running_torque is not None:
                inherited = np.where(inherited > 0.0, inherited, running_torque)
            self._limit_pair(
                wheel_torque_plot, plot_t, focused(inherited), RED,
                "Jump-handoff positive torque authority (N·m)",
                "Jump-handoff negative torque authority (N·m)",
                Qt.PenStyle.DashDotLine)
        add_jump_markers(wheel_torque_plot)

        # Compact per-attempt report. These numbers use exactly the same sample
        # indices as the phase/landing markers above.
        rows: list[tuple[str, str, str]] = [(
            "Jumping episodes", str(len(episodes)),
            "Curves include 0.25 s before each request and 0.75 s after each exit. "
            "Gaps between attempts remain on the shared time axis."), (
            "Landing detector",
            f"az {air_accel:g}/{land_accel:g} · gyro {gyro_impulse:g}",
            f"Airborne/landing Z thresholds [m/s²], gyro impulse [rad/s], and "
            f"{min_air_s:.2f} s launch blanking. Legacy logs use offline inference; "
            "new logs use the live LANDING phase."),
        ]

        minimum_pitch_index = int(np.nanargmin(np.where(jump_mask, fields["pitch_rad"], np.nan)))
        watchdog_margin = None
        if wd_bwd is not None:
            watchdog_margin = np.degrees(wd_bwd[minimum_pitch_index]) - abs(pitch_deg[minimum_pitch_index])

        for episode in episodes:
            local = slice(episode.i0, episode.i1 + 1)
            start = float(t[episode.i0])
            entries = " · ".join(
                f"{phase_name(phase, episode.modern_phases)} +{t[index] - start:.3f}s"
                for phase, index in episode.phase_entries)
            if episode.landing_index is None:
                landing_text = "no touchdown"
                landing_detail = episode.landing_source
            else:
                li = episode.landing_index
                landing_text = f"land +{t[li] - start:.3f}s"
                landing_detail = (
                    f"{episode.landing_source}; pitch {pitch_deg[li]:+.2f}°, "
                    f"roll {roll_deg[li]:+.2f}°, pitch rate "
                    f"{np.degrees(fields['pitch_rate_rads'][li]):+.1f}°/s")
            rows.append((
                f"Jump {episode.number} · {t[episode.i0]:.3f}s",
                landing_text,
                f"{entries}. {landing_detail}. Minimum pitch "
                f"{np.min(pitch_deg[local]):+.2f}°, max |roll| "
                f"{np.max(np.abs(roll_deg[local])):.2f}°, max wheel speed "
                f"{max(np.max(np.abs(fields['wm_l_vel_turns_s'][local])), np.max(np.abs(fields['wm_r_vel_turns_s'][local]))):.2f} turns/s, "
                f"max hip torque {max(np.max(np.abs(fields['hip_l_torque_nm'][local])), np.max(np.abs(fields['hip_r_torque_nm'][local]))):.2f} N·m."))

        accel = np.column_stack((fields["accel_x_ms2"], fields["accel_y_ms2"], fields["accel_z_ms2"]))
        fresh_accel = np.zeros(t.size, dtype=bool)
        if t.size > 1:
            fresh_accel[1:] = np.any(np.abs(np.diff(accel, axis=0)) > 1e-6, axis=1)
        jump_duration = sum(max(0.0, float(t[e.i1] - t[e.i0])) for e in episodes)
        fresh_count = int(np.count_nonzero(fresh_accel & jump_mask))
        fresh_rate = fresh_count / jump_duration if jump_duration > 0.0 else 0.0
        rows.extend([(
            "Worst backward pitch",
            f"{pitch_deg[minimum_pitch_index]:+.2f}°",
            (f"Margin to the scheduled backward watchdog was {watchdog_margin:+.2f}°."
             if watchdog_margin is not None else
             "Pitch-watchdog parameters were unavailable, so margin cannot be reconstructed.")), (
            "Fresh accel reports in JUMPING",
            f"{fresh_count} · {fresh_rate:.1f} Hz",
            "Estimated by changes in the held acceleration vector. A rate far below "
            "the configured 50 Hz indicates an old report-dropping firmware or sensor/link trouble."),
        ])
        self._set_meta_rows(rows)

        visible = np.flatnonzero(focus)
        if visible.size and self._plots:
            self._plots[0].setXRange(float(t[visible[0]]), float(t[visible[-1]]), padding=0.01)

    # ── Leg-height sweep ───────────────────────────────────────────────────

    def _sidecar_value(self, name: str) -> float | None:
        """A parameter's value at capture time, or None if it wasn't recorded."""
        if self._params is None:
            return None
        try:
            value = self._params.initial_value(name)
        except (KeyError, ValueError):
            return None
        return None if value is None else float(value)

    def _plateaus(self) -> list[PlateauMetrics]:
        """Per-leg-height metrics, computed once per loaded file.

        Every clamp is read from the .PARAMS sidecar rather than assumed: a
        saturation percentage measured against the wrong limit is worse than
        no number at all, so plateau_report() reports zero when a limit is
        genuinely unknown.
        """
        if self._plateau_cache is None and self._run is not None:
            torque_limit = self._profile_limit("torque_lim")
            self._plateau_cache = plateau_report(
                self._run,
                torque_limit_nm=(float(np.nanmax(torque_limit))
                                 if torque_limit is not None
                                 and np.any(np.isfinite(torque_limit)) else None),
                rate_lim=self._sidecar_value("vel_pi_rate_lim"),
                theta_max_fwd=self._sidecar_value("theta_max_fwd_ret"),
                theta_max_bwd=self._sidecar_value("theta_max_bwd_ret"),
            )
        return self._plateau_cache or []

    def _draw_leg_height_sweep(self, t: np.ndarray, fields: dict, run):
        if not run.has_gain_sched_alpha:
            plot = self._new_plot("Leg-height sweep", "Schedule position α")
            text = pg.TextItem("Gain-schedule telemetry is unavailable in this log.",
                               color=DIM, anchor=(0.5, 0.5))
            plot.addItem(text)
            text.setPos(float(t[len(t) // 2]) if len(t) else 0.0, 0.5)
            self._set_meta_rows([("Leg-height sweep", "unavailable",
                                  "This telemetry version did not record α.")])
            return

        plateaus = self._plateaus()

        def mask_for(selected) -> np.ndarray:
            mask = np.zeros(t.size, dtype=bool)
            for plateau in selected:
                mask[plateau.i0:plateau.i1 + 1] = True
            return mask

        settled = [p for p in plateaus if p.equilibrium]
        unsettled = [p for p in plateaus if not p.equilibrium]

        alpha_plot = self._new_plot("Leg height and detected plateaus",
                                    "Schedule position α")
        self._curve(alpha_plot, t, fields["gain_sched_alpha"], BLUE,
                    "Gain-schedule position α — 0=retracted, 1=extended", 1.6)
        self._add_regions(alpha_plot, t, mask_for(settled), GREEN,
                          "Settled plateau — in equilibrium, balance point is trustworthy")
        self._add_regions(alpha_plot, t, mask_for(unsettled), YELLOW,
                          "Unsettled plateau — still drifting, balance point is provisional")

        pitch_plot = self._new_plot("Pitch against the scheduled balance point",
                                    "Angle", "°", auto_si_prefix=False)
        self._curve(pitch_plot, t, np.degrees(fields["pitch_rad"]), BLUE,
                    "Pitch — measured body tilt (°)", 1.5)
        self._curve(pitch_plot, t, np.degrees(fields["pitch_trim_rad"]), ORANGE,
                    "Applied trim — the balance point the schedule believes in (°)", 1.3)
        for plateau in plateaus:
            marker = pg.PlotDataItem(
                [plateau.t0, plateau.t1],
                [np.degrees(plateau.balance_rad)] * 2,
                pen=pg.mkPen(GREEN if plateau.equilibrium else YELLOW,
                             width=2.2, style=Qt.PenStyle.DashLine))
            marker.setZValue(20)
            pitch_plot.addItem(marker)
        if plateaus and pitch_plot.plotItem.legend is not None:
            sample = pg.PlotDataItem(pen=pg.mkPen(GREEN, width=2.2,
                                                  style=Qt.PenStyle.DashLine))
            pitch_plot.plotItem.legend.addItem(sample, "Measured balance point")
            _set_legend_tooltip(
                pitch_plot.plotItem.legend, sample,
                "Mean pitch over each plateau's settled tail. Where this sits away "
                "from the applied trim, the trim schedule is wrong by the gap.")

        err_plot = self._new_plot("LQR pitch error", "Pitch error", "°",
                                  auto_si_prefix=False)
        self._curve(err_plot, t,
                    np.degrees(fields["pitch_rad"] - fields["theta_ref"]
                               - fields["pitch_trim_rad"]), RED,
                    "Pitch error — pitch − θref − trim, the quantity the LQR regulates (°)",
                    1.3)

        slew_plot = self._new_plot("Velocity-PI lean-target slew rate", "Slew rate",
                                   "°/s", auto_si_prefix=False)
        slew = np.zeros_like(t)
        if t.size > 1:
            slew[1:] = np.diff(fields["theta_ref"]) * run.sample_rate_hz
        self._curve(slew_plot, t, np.degrees(slew), BLUE,
                    "dθref/dt — how fast the velocity PI is moving the lean target (°/s)",
                    1.1)
        rate_lim = self._sidecar_value("vel_pi_rate_lim")
        if rate_lim:
            self._limit_pair(
                slew_plot, t, np.degrees(rate_lim), YELLOW,
                "Upper vel_pi_rate_lim — θref slew clamp (°/s)",
                "Lower vel_pi_rate_lim — θref slew clamp (°/s)")

        self._set_meta_rows(self._leg_height_rows(plateaus))

    @staticmethod
    def _leg_height_rows(plateaus: list[PlateauMetrics]) -> list[tuple[str, str, str]]:
        if not plateaus:
            return [("Leg-height plateaus", "none found",
                     "No stretch of RUNNING held α flat for long enough to measure. "
                     "Hold each leg height still for several seconds.")]

        rows: list[tuple[str, str, str]] = []
        for plateau in plateaus:
            gate = ("settled" if plateau.equilibrium else
                    f"NOT settled (drift {plateau.drift_turns_s:+.3f} turns/s)")
            rows.append((
                f"α {plateau.alpha:.3f} · {plateau.duration_s:.0f} s",
                f"{np.degrees(plateau.balance_rad):+.2f}°",
                f"Balance point, {gate}. Applied trim "
                f"{np.degrees(plateau.applied_trim_rad):+.2f}°, so the schedule is off by "
                f"{np.degrees(plateau.applied_trim_rad - plateau.balance_rad):+.2f}°. "
                f"Pitch error {np.degrees(plateau.rms_pitch_err_rad):.2f}° RMS "
                f"({np.degrees(plateau.pitch_err_bands['vel_pi']):.2f}° of it in the "
                f"0.3–1.5 Hz velocity-PI band, "
                f"{np.degrees(plateau.pitch_err_bands['lqr']):.2f}° in the 1.5–4 Hz LQR band). "
                f"θref slew-limited {100 * plateau.rate_limit['duty_frac']:.0f}% of the time "
                f"in runs averaging {1000 * plateau.rate_limit['mean_run_s']:.0f} ms. "
                f"Backward lean clamp hit {100 * plateau.theta_bwd_sat_frac:.1f}%, "
                f"wheel torque clamp {100 * plateau.torque_sat_frac:.1f}%. "
                f"Hip sag {np.degrees(plateau.hip_sag_l_rad):+.2f}°/"
                f"{np.degrees(plateau.hip_sag_r_rad):+.2f}° at "
                f"{plateau.mean_hip_torque_nm:+.2f} N·m."))

        settled = [p for p in plateaus if p.equilibrium]
        if len(settled) >= 2:
            fit = fit_trim_schedule([p.alpha for p in settled],
                                    [p.balance_rad for p in settled])
            note = (f"Least-squares fit of the {fit['n_points']} settled plateaus to "
                    f"control_safety.h's scheduled_pitch_trim(); worst residual "
                    f"{np.degrees(fit['max_residual_rad']):.2f}°.")
            if fit["extrapolated"]:
                note += (f" The sweep only reached α={fit['alpha_span'][1]:.2f}, so "
                         f"trim_ext (the α=1 value) is an extrapolation, not a "
                         f"measurement — do not trust it above the measured span.")
            rows.append((
                "Fitted trim schedule",
                f"{fit['trim_ret']:+.4f} / {fit['trim_ext']:+.4f} / {fit['trim_curve']:+.4f}",
                "lqr_pitch_trim_ret / _ext / _curve [rad]. " + note))
        else:
            rows.append((
                "Fitted trim schedule", "not enough settled heights",
                "At least two plateaus must pass the equilibrium gate before a trim "
                "schedule can be fitted."))
        return rows

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
