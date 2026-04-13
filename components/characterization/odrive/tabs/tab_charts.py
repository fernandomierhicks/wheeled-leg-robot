"""tab_charts.py — Live pyqtgraph telemetry plots (Step 7).

Four stacked ring-buffer plots: Vbus, Iq_measured, vel_estimate, pos_estimate.
Toggle checkboxes, clear button, per-signal statistics (mean, stdev).
Based on the ring-buffer pattern from misc/fastchart.py.
"""

import logging
import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QLabel,
)

import pyqtgraph as pg

from ui.theme import CLR_OK, CLR_INFO, CLR_WARN, CLR_ERR, CLR_MUTED, CLR_PANEL

log = logging.getLogger("odrive_gui")

# ── Tunables ─────────────────────────────────────────────────────────────────
WINDOW_SECONDS = 30.0   # visible time window
BUFFER_SIZE = 1200       # ~2 min at 10 Hz poll rate

# Signal definitions: (key, display_label, unit, hex_color)
SIGNALS = [
    ("vbus", "Vbus",     "V",     CLR_OK),    # green
    ("iq",   "Iq",       "A",     CLR_INFO),   # blue
    ("vel",  "Velocity", "t/s",   CLR_WARN),   # orange
    ("pos",  "Position", "turns", CLR_ERR),    # red
]


# ── Ring buffer ──────────────────────────────────────────────────────────────

class RingBuffer:
    """Fixed-size numpy ring buffer for time-series data."""

    def __init__(self, size: int):
        self.size = size
        self.t = np.zeros(size)
        self.y = np.zeros(size)
        self.head = 0
        self.count = 0

    def append(self, t_val: float, y_val: float):
        self.t[self.head] = t_val
        self.y[self.head] = y_val
        self.head = (self.head + 1) % self.size
        self.count += 1

    def get_ordered(self):
        """Return (t_array, y_array) in chronological order."""
        if self.count == 0:
            return np.array([]), np.array([])
        n = min(self.count, self.size)
        if n < self.size:
            return self.t[:n].copy(), self.y[:n].copy()
        idx = np.arange(self.head, self.head + self.size) % self.size
        return self.t[idx], self.y[idx]

    def clear(self):
        self.t[:] = 0
        self.y[:] = 0
        self.head = 0
        self.count = 0

    def stats(self):
        """Return (mean, stdev) or (None, None) if empty."""
        if self.count == 0:
            return None, None
        n = min(self.count, self.size)
        data = self.y[:n] if n < self.size else self.y
        return float(np.mean(data)), float(np.std(data))


# ── Charts tab ───────────────────────────────────────────────────────────────

class TabCharts(QWidget):
    """Live telemetry charts with pyqtgraph ring-buffer plots."""

    def __init__(self, manager, get_axis_idx, get_setpoints=None, parent=None):
        super().__init__(parent)
        self._mgr = manager
        self._get_axis_idx = get_axis_idx
        self._get_setpoints = get_setpoints  # () -> (vel_sp, pos_sp) or None
        self._t0 = time.perf_counter()
        self._buffers = {sig[0]: RingBuffer(BUFFER_SIZE) for sig in SIGNALS}
        self._sp_buffers = {
            "vel": RingBuffer(BUFFER_SIZE),
            "pos": RingBuffer(BUFFER_SIZE),
        }
        self._build()

    def _ax_idx(self) -> int:
        return self._get_axis_idx()

    # ── Build UI ─────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Controls row ─────────────────────────────────────────────────────
        ctrl_row = QHBoxLayout()

        self._checkboxes = {}
        for key, label, _unit, color in SIGNALS:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {color};")
            cb.toggled.connect(self._on_checkbox_toggled)
            ctrl_row.addWidget(cb)
            self._checkboxes[key] = cb

        ctrl_row.addStretch()

        btn_clear = QPushButton("Clear")
        btn_clear.setFixedWidth(70)
        btn_clear.clicked.connect(self._clear_all)
        ctrl_row.addWidget(btn_clear)

        root.addLayout(ctrl_row)

        # ── pyqtgraph plots ──────────────────────────────────────────────────
        pg.setConfigOptions(antialias=False, useOpenGL=True)

        self._graphics = pg.GraphicsLayoutWidget()
        self._graphics.setBackground(CLR_PANEL)
        root.addWidget(self._graphics, stretch=1)

        self._plots = {}
        self._curves = {}

        for i, (key, label, unit, color) in enumerate(SIGNALS):
            if i > 0:
                self._graphics.nextRow()

            plot = self._graphics.addPlot()
            plot.setLabel("left", f"{label} ({unit})")
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setXRange(-WINDOW_SECONDS, 0, padding=0)
            plot.enableAutoRange(axis="y")
            # Minimum ±1 Y range for iq/vel/pos (avoid x0.001 scale)
            if key in ("iq", "vel", "pos"):
                plot.setLimits(minYRange=2.0)
            plot.setMouseEnabled(x=False, y=True)

            # Only show x-axis label on the bottom plot
            if i < len(SIGNALS) - 1:
                plot.getAxis("bottom").setStyle(showValues=False)
            else:
                plot.setLabel("bottom", "time (s)")

            # Link x-axes so pan/zoom stays in sync
            if i > 0:
                plot.setXLink(self._plots[SIGNALS[0][0]])

            curve = plot.plot(pen=pg.mkPen(color=color, width=2))
            self._plots[key] = plot
            self._curves[key] = curve

        # Setpoint overlay curves on vel and pos plots (dashed white)
        self._sp_curves = {}
        for key in ("vel", "pos"):
            sp_pen = pg.mkPen(color="#e0e0e0", width=1, style=Qt.PenStyle.DashLine)
            self._sp_curves[key] = self._plots[key].plot(pen=sp_pen, name="setpoint")

        # ── Statistics row ───────────────────────────────────────────────────
        stats_row = QHBoxLayout()
        self._stat_labels = {}
        for key, label, _unit, color in SIGNALS:
            lbl = QLabel(f"{label}: --")
            lbl.setStyleSheet(
                f"font-family: monospace; font-size: 11px; color: {color};")
            stats_row.addWidget(lbl)
            self._stat_labels[key] = lbl
        stats_row.addStretch()
        root.addLayout(stats_row)

    # ── Checkbox toggle ──────────────────────────────────────────────────────

    def _on_checkbox_toggled(self):
        for key in self._plots:
            self._plots[key].setVisible(self._checkboxes[key].isChecked())

    # ── Clear ────────────────────────────────────────────────────────────────

    def _clear_all(self):
        for buf in self._buffers.values():
            buf.clear()
        for buf in self._sp_buffers.values():
            buf.clear()
        for curve in self._curves.values():
            curve.setData([], [])
        for curve in self._sp_curves.values():
            curve.setData([], [])
        for key, label, _unit, _color in SIGNALS:
            self._stat_labels[key].setText(f"{label}: --")
        self._t0 = time.perf_counter()
        log.info("Charts: cleared all buffers")

    # ── Autotuner injection ────────────────────────────────────────────────────

    def inject_autotune_sample(self, t_rel_from_test: float, vel: float):
        """Feed a velocity sample from the autotuner relay test into the chart.

        The autotuner runs in its own thread at ~200 Hz.  We convert its
        relative timestamp to our chart time-base and push into the vel buffer.
        """
        t_chart = time.perf_counter() - self._t0
        self._buffers["vel"].append(t_chart, vel)

    # ── Poll update (called by MainWindow every 100 ms) ──────────────────────

    def poll_update(self):
        if not self._mgr.connected:
            return

        t_now = time.perf_counter() - self._t0
        ax = self._ax_idx()

        # ── Sample signals ───────────────────────────────────────────────────
        try:
            vbus = self._mgr.vbus_voltage()
            if vbus is not None:
                self._buffers["vbus"].append(t_now, vbus)
        except Exception:
            pass

        try:
            axis = self._mgr.get_axis(ax)
            if axis is not None:
                iq = getattr(axis.motor.current_control, "Iq_measured", None)
                if iq is not None:
                    self._buffers["iq"].append(t_now, float(iq))

                vel = getattr(axis.encoder, "vel_estimate", None)
                if vel is not None:
                    self._buffers["vel"].append(t_now, float(vel))

                pos = getattr(axis.encoder, "pos_estimate", None)
                if pos is not None:
                    self._buffers["pos"].append(t_now, float(pos))
        except Exception:
            pass

        # Setpoint overlay samples (mirror of last commanded value)
        if self._get_setpoints is not None:
            try:
                sp_vel, sp_pos = self._get_setpoints()
                self._sp_buffers["vel"].append(t_now, float(sp_vel))
                self._sp_buffers["pos"].append(t_now, float(sp_pos))
            except Exception:
                pass

        # ── Update curves + stats ────────────────────────────────────────────
        for key, label, unit, _color in SIGNALS:
            if not self._checkboxes[key].isChecked():
                continue

            t_arr, y_arr = self._buffers[key].get_ordered()
            if len(t_arr) == 0:
                continue

            # x = 0 is "now"; negative is past
            t_rel = t_arr - t_now
            mask = t_rel > -WINDOW_SECONDS
            self._curves[key].setData(t_rel[mask], y_arr[mask])

            mean, std = self._buffers[key].stats()
            if mean is not None:
                self._stat_labels[key].setText(
                    f"{label}: \u03bc={mean:.3f} \u03c3={std:.3f} {unit}")

            # Update setpoint overlay for vel/pos plots
            if key in self._sp_curves:
                sp_t, sp_y = self._sp_buffers[key].get_ordered()
                if len(sp_t) > 0:
                    sp_rel = sp_t - t_now
                    sp_mask = sp_rel > -WINDOW_SECONDS
                    self._sp_curves[key].setData(sp_rel[sp_mask], sp_y[sp_mask])
