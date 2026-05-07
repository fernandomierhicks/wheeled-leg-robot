import math
from collections import deque

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from telemetry_bus import TelemetryBus
from theme import BG, BORDER, BLUE, DIM, GREEN, ORANGE, RED, SURFACE, TEXT

pg.setConfigOptions(antialias=True, background=BG, foreground=TEXT)

_BUF = 500   # rolling chart samples (~1 s at 500 Hz)
_AXIS_LEN = 0.8


def _placeholder_chart(title: str, ylabel: str, note: str,
                        series: list[tuple[str, str]]) -> pg.PlotWidget:
    w = pg.PlotWidget()
    w.setTitle(title, color=TEXT, size="11pt")
    w.setLabel("left", ylabel, color=DIM)
    w.showGrid(x=True, y=True, alpha=0.12)
    w.setYRange(-1, 1)
    w.setXRange(0, _BUF)
    w.getAxis("bottom").setStyle(showValues=False)
    leg = w.addLegend(offset=(5, 5))
    for name, color in series:
        w.plot([0] * _BUF, pen=pg.mkPen(color, width=1,
               style=Qt.PenStyle.DashLine), name=name)
    txt = pg.TextItem(note, color=DIM, anchor=(0.5, 0.5))
    w.addItem(txt)
    txt.setPos(_BUF // 2, 0)
    return w


def _build_robot(view: gl.GLViewWidget) -> list[tuple[np.ndarray, gl.GLLinePlotItem]]:
    """Draw a small two-wheeled balancing robot chassis at the origin.
    Returns (pts_body_frame, line_item) pairs so they can be rotated each frame."""
    items: list[tuple[np.ndarray, gl.GLLinePlotItem]] = []

    # Body proportions — +X forward, +Y left, +Z up
    bx, by, bz = 0.07, 0.16, 0.18   # body half-extents
    wr  = 0.22                        # wheel radius
    wyo = by + 0.05                   # wheel Y offset from centre

    BODY  = (0.80, 0.80, 0.82, 0.90)
    WHEEL = (0.50, 0.52, 0.58, 0.85)

    def _line(pts: np.ndarray, color: tuple, width: float = 1.8):
        item = gl.GLLinePlotItem(pos=pts.astype(np.float32),
                                 color=color, width=width, antialias=True)
        view.addItem(item)
        items.append((pts.copy(), item))

    # ── Body box ─────────────────────────────────────────────────────────────
    v = np.array([
        [-bx, -by, -bz], [+bx, -by, -bz], [+bx, +by, -bz], [-bx, +by, -bz],  # bottom
        [-bx, -by, +bz], [+bx, -by, +bz], [+bx, +by, +bz], [-bx, +by, +bz],  # top
    ])
    _line(v[[0,1,2,3,0]], BODY, 2.0)   # bottom rect
    _line(v[[4,5,6,7,4]], BODY, 2.0)   # top rect
    _line(v[[0,4,5,1]],   BODY, 2.0)   # front + back uprights
    _line(v[[2,6,7,3]],   BODY, 2.0)

    # ── Wheels ────────────────────────────────────────────────────────────────
    N = 24
    a = np.linspace(0, 2 * np.pi, N + 1)
    xs = wr * np.sin(a)
    zs = wr * np.cos(a)

    for yo in (-wyo, +wyo):
        # rim
        rim = np.column_stack([xs, np.full(N + 1, yo), zs])
        _line(rim, WHEEL, 2.2)
        # two spokes (cross)
        for dx, dz in ((wr, 0.0), (0.0, wr)):
            _line(np.array([[-dx, yo, -dz], [dx, yo, dz]]), WHEEL, 1.4)

    return items


class ImuTab(QWidget):
    def __init__(self):
        super().__init__()

        # ── 3-D orientation view ──────────────────────────────────────────────
        self._gl = gl.GLViewWidget()
        self._gl.opts["distance"]  = 3.5
        self._gl.opts["elevation"] = 25
        self._gl.opts["azimuth"]   = -50
        self._gl.setBackgroundColor(pg.mkColor(BG))
        self._gl.setMinimumSize(320, 280)

        grid = gl.GLGridItem()
        grid.setSize(3, 3, 1)
        grid.setSpacing(0.5, 0.5, 0.5)
        grid.setColor((255, 255, 255, 18))
        self._gl.addItem(grid)

        _axes = [
            (np.array([_AXIS_LEN, 0.0, 0.0]), (1.0, 0.3, 0.3, 1.0)),  # X red
            (np.array([0.0, _AXIS_LEN, 0.0]), (0.3, 1.0, 0.3, 1.0)),  # Y green
            (np.array([0.0, 0.0, _AXIS_LEN]), (0.3, 0.6, 1.0, 1.0)),  # Z blue
        ]
        self._axis_lines: list[tuple[np.ndarray, gl.GLLinePlotItem]] = []
        for vec, color in _axes:
            pts  = np.array([[0, 0, 0], vec], dtype=np.float32)
            line = gl.GLLinePlotItem(pos=pts, color=color, width=5.0, antialias=True)
            self._gl.addItem(line)
            self._axis_lines.append((vec.copy(), line))

        self._body_lines = _build_robot(self._gl)

        # ── Readout panel ─────────────────────────────────────────────────────
        ro = QWidget()
        ro.setStyleSheet(
            f"QWidget {{ background: {SURFACE}; border: 1px solid {BORDER};"
            f" border-radius: 3px; }}"
            f"QLabel {{ border: none; }}"
        )
        ro.setMinimumWidth(155)
        ro.setMaximumWidth(200)
        ro_lay = QVBoxLayout(ro)
        ro_lay.setContentsMargins(14, 12, 14, 12)
        ro_lay.setSpacing(10)

        hdr = QLabel("Game Rotation Vector")
        hdr.setStyleSheet(f"color: {BLUE}; font-weight: bold; font-size: 11px;")
        ro_lay.addWidget(hdr)

        self._vals: dict[str, QLabel] = {}
        for name in ["Pitch", "Roll", "Yaw", "Pitch rate", "State"]:
            row = QHBoxLayout()
            row.setSpacing(4)
            k = QLabel(name + ":")
            k.setStyleSheet(f"color: {DIM}; font-size: 11px;")
            v = QLabel("—")
            v.setStyleSheet(
                f"color: {TEXT}; font-size: 12px; font-weight: bold;"
                f" font-family: Consolas;"
            )
            row.addWidget(k)
            row.addStretch()
            row.addWidget(v)
            ro_lay.addLayout(row)
            self._vals[name] = v

        ro_lay.addStretch()
        legend = QLabel(
            f'<span style="color:#ff6666">■</span> X fwd&nbsp;&nbsp;'
            f'<span style="color:#66ff66">■</span> Y left&nbsp;&nbsp;'
            f'<span style="color:#6699ff">■</span> Z up'
        )
        legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        legend.setStyleSheet(f"font-size: 10px; color: {DIM};")
        ro_lay.addWidget(legend)

        # ── Top: GL view + readout ────────────────────────────────────────────
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self._gl)
        top.addWidget(ro)
        top.setSizes([520, 160])
        top.setHandleWidth(5)

        # ── Gyro chart (pitch rate live; roll/yaw placeholder) ────────────────
        self._gyro_buf: deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._gyro_w = pg.PlotWidget()
        self._gyro_w.setTitle("Gyroscope", color=TEXT, size="11pt")
        self._gyro_w.setLabel("left", "rad/s", color=DIM)
        self._gyro_w.showGrid(x=True, y=True, alpha=0.12)
        self._gyro_w.setXRange(0, _BUF)
        self._gyro_w.getAxis("bottom").setStyle(showValues=False)
        self._gyro_w.addLegend(offset=(5, 5))
        self._gyro_curve = self._gyro_w.plot(
            list(self._gyro_buf),
            pen=pg.mkPen(BLUE, width=1.5),
            name="pitch rate",
        )
        for name, color in [("roll rate", DIM), ("yaw rate", ORANGE)]:
            self._gyro_w.plot(
                [0] * _BUF,
                pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                name=name,
            )

        # ── Accel chart (placeholder) ─────────────────────────────────────────
        self._accel_w = _placeholder_chart(
            "Accelerometer", "m/s²",
            "No data — not in telemetry yet",
            [("X", RED), ("Y", GREEN), ("Z", BLUE)],
        )

        # ── Mag chart (placeholder) ───────────────────────────────────────────
        self._mag_w = _placeholder_chart(
            "Magnetometer", "µT",
            "No data — not in telemetry yet",
            [("X", RED), ("Y", GREEN), ("Z", BLUE)],
        )

        # ── Bottom: gyro | accel | mag ────────────────────────────────────────
        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.addWidget(self._gyro_w)
        bottom.addWidget(self._accel_w)
        bottom.addWidget(self._mag_w)
        bottom.setSizes([1, 1, 1])
        bottom.setHandleWidth(5)

        # ── Outer layout ──────────────────────────────────────────────────────
        outer = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(top)
        outer.addWidget(bottom)
        outer.setSizes([480, 220])
        outer.setHandleWidth(5)
        outer.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(outer)

        TelemetryBus.instance().packet.connect(self._on_packet)

    # ── Telemetry handler ─────────────────────────────────────────────────────

    def _on_packet(self, info: dict):
        if info.get("ptype") != 0x01:
            return

        pitch = info.get("pitch_rad",       0.0)
        roll  = info.get("roll_rad",        0.0)
        yaw   = info.get("yaw_rad",         0.0)
        rate  = info.get("pitch_rate_rads", 0.0)
        state = info.get("state_name",      "—")

        # Readout
        self._vals["Pitch"].setText(f"{math.degrees(pitch):+.1f}°")
        self._vals["Roll"].setText(f"{math.degrees(roll):+.1f}°")
        self._vals["Yaw"].setText(f"{math.degrees(yaw):+.1f}°")
        self._vals["Pitch rate"].setText(f"{rate:+.3f} r/s")
        color = GREEN if state == "RUNNING" else (RED if state == "ESTOP" else TEXT)
        self._vals["State"].setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: bold; font-family: Consolas;"
        )
        self._vals["State"].setText(state)

        # Rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll),  math.sin(roll)
        cy, sy = math.cos(yaw),   math.sin(yaw)
        R = np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,    cp*sr,             cp*cr            ],
        ])
        for vec0, line in self._axis_lines:
            pts = np.array([[0, 0, 0], R @ vec0], dtype=np.float32)
            line.setData(pos=pts)
        for pts0, line in self._body_lines:
            line.setData(pos=(R @ pts0.T).T.astype(np.float32))

        # Gyro chart
        self._gyro_buf.append(rate)
        self._gyro_curve.setData(list(self._gyro_buf))
