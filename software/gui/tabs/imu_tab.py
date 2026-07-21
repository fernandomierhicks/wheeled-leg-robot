import math
from collections import deque

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from .telemetry_bus import TelemetryBus
from .theme import BG, BORDER, BLUE, DIM, GREEN, ORANGE, RED, SURFACE, TEXT, YELLOW, WHITE

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
        for name in ["Pitch", "Roll", "Yaw", "Pitch rate", "Roll rate", "Yaw rate", "IMU loss", "State"]:
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

        # ── Gyro chart (all 3 rates live) ────────────────────────────────────
        self._pitch_rate_buf: deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._roll_rate_buf:  deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._yaw_rate_buf:   deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._gyro_w = pg.PlotWidget()
        self._gyro_w.setTitle("Gyroscope", color=TEXT, size="11pt")
        self._gyro_w.setLabel("left", "rad/s", color=DIM)
        self._gyro_w.showGrid(x=True, y=True, alpha=0.12)
        self._gyro_w.setXRange(0, _BUF)
        self._gyro_w.getAxis("bottom").setStyle(showValues=False)
        self._gyro_w.addLegend(offset=(5, 5))
        self._pitch_rate_curve = self._gyro_w.plot(
            list(self._pitch_rate_buf), pen=pg.mkPen(BLUE,   width=1.5), name="pitch rate")
        self._roll_rate_curve  = self._gyro_w.plot(
            list(self._roll_rate_buf),  pen=pg.mkPen(GREEN,  width=1.5), name="roll rate")
        self._yaw_rate_curve   = self._gyro_w.plot(
            list(self._yaw_rate_buf),   pen=pg.mkPen(ORANGE, width=1.5), name="yaw rate")

        # ── Accel chart (live) ────────────────────────────────────────────────
        self._ax_buf: deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._ay_buf: deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._az_buf: deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._accel_w = pg.PlotWidget()
        self._accel_w.setTitle("Accelerometer", color=TEXT, size="11pt")
        self._accel_w.setLabel("left", "m/s²", color=DIM)
        self._accel_w.showGrid(x=True, y=True, alpha=0.12)
        self._accel_w.setXRange(0, _BUF)
        self._accel_w.getAxis("bottom").setStyle(showValues=False)
        self._accel_w.addLegend(offset=(5, 5))
        self._ax_curve = self._accel_w.plot(list(self._ax_buf), pen=pg.mkPen(RED,   width=1.5), name="X fwd")
        self._ay_curve = self._accel_w.plot(list(self._ay_buf), pen=pg.mkPen(GREEN, width=1.5), name="Y left")
        self._az_curve = self._accel_w.plot(list(self._az_buf), pen=pg.mkPen(BLUE,  width=1.5), name="Z up")

        # ── Mag chart (placeholder — no magnetometer in telemetry) ────────────
        self._mag_w = _placeholder_chart(
            "Magnetometer", "µT",
            "No magnetometer in telemetry",
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

        # 3-D scene + chart redraws are expensive relative to raw packet rate —
        # decouple them from packet arrival jitter onto a fixed-interval timer
        # holding the latest orientation, same pattern as controllers_tab.py's
        # chart timer / robot_visualizer_tab.py's scene timer. Cheap per-packet
        # work (readout labels, buffer appends) still happens in _on_packet.
        self._latest_orientation: tuple[float, float, float] | None = None
        self._scene_timer = QTimer(self)
        self._scene_timer.setInterval(33)  # ~30 Hz
        self._scene_timer.timeout.connect(self._redraw_scene)
        self._scene_timer.start()

    # ── Telemetry handler ─────────────────────────────────────────────────────

    def _on_packet(self, info: dict):
        if info.get("ptype") == 0x01 and not self.isVisible():
            return
        if info.get("ptype") != 0x01:
            return

        pitch      = info.get("pitch_rad",         0.0)
        roll       = info.get("roll_rad",           0.0)
        yaw        = info.get("yaw_rad",            0.0)
        pitch_rate = info.get("pitch_rate_rads",    0.0)
        roll_rate  = info.get("roll_rate_rads",     0.0)
        yaw_rate   = info.get("yaw_rate_rads",      0.0)
        accel_x    = info.get("accel_x_ms2",        0.0)
        accel_y    = info.get("accel_y_ms2",        0.0)
        accel_z    = info.get("accel_z_ms2",        0.0)
        imu_loss   = info.get("imu_packet_loss_pct", 0)
        state      = info.get("state_name",         "—")

        # Readout
        self._vals["Pitch"].setText(f"{math.degrees(pitch):+.1f}°")
        self._vals["Roll"].setText(f"{math.degrees(roll):+.1f}°")
        self._vals["Yaw"].setText(f"{math.degrees(yaw):+.1f}°")
        self._vals["Pitch rate"].setText(f"{pitch_rate:+.3f} r/s")
        self._vals["Roll rate"].setText(f"{roll_rate:+.3f} r/s")
        self._vals["Yaw rate"].setText(f"{yaw_rate:+.3f} r/s")
        loss_color = RED if imu_loss > 10 else (YELLOW if imu_loss > 0 else GREEN)
        self._vals["IMU loss"].setStyleSheet(
            f"color: {loss_color}; font-size: 12px; font-weight: bold; font-family: Consolas;"
        )
        self._vals["IMU loss"].setText(f"{imu_loss}%")
        state_color = {"RUNNING": GREEN, "ESTOP": RED, "CALIBRATION": BLUE,
                       "STANDBY": YELLOW, "STARTUP": WHITE}.get(state, TEXT)
        self._vals["State"].setStyleSheet(
            f"color: {state_color}; font-size: 12px; font-weight: bold; font-family: Consolas;"
        )
        self._vals["State"].setText(state)

        # Orientation for the 3-D scene, and rolling chart buffers — cheap, kept
        # per-packet. Actual redraw deferred to _redraw_scene() on a timer.
        self._latest_orientation = (pitch, roll, yaw)

        self._pitch_rate_buf.append(pitch_rate)
        self._roll_rate_buf.append(roll_rate)
        self._yaw_rate_buf.append(yaw_rate)

        self._ax_buf.append(accel_x)
        self._ay_buf.append(accel_y)
        self._az_buf.append(accel_z)

    def _redraw_scene(self):
        if self._latest_orientation is not None:
            pitch, roll, yaw = self._latest_orientation
            _apply_rotation(pitch, roll, yaw, self._axis_lines, self._body_lines)

        self._pitch_rate_curve.setData(list(self._pitch_rate_buf))
        self._roll_rate_curve.setData(list(self._roll_rate_buf))
        self._yaw_rate_curve.setData(list(self._yaw_rate_buf))

        self._ax_curve.setData(list(self._ax_buf))
        self._ay_curve.setData(list(self._ay_buf))
        self._az_curve.setData(list(self._az_buf))


# ── Compact IMU widget for embedding in other tabs ────────────────────────────

def _apply_rotation(pitch: float, roll: float, yaw: float,
                    axis_lines: list, body_lines: list):
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll),  math.sin(roll)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    R = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ])
    for vec0, line in axis_lines:
        line.setData(pos=np.array([[0, 0, 0], R @ vec0], dtype=np.float32))
    for pts0, line in body_lines:
        line.setData(pos=(R @ pts0.T).T.astype(np.float32))


class ImuMiniWidget(QWidget):
    """Small 3-D orientation view + gyro readout for embedding in the dashboard."""

    def __init__(self):
        super().__init__()
        self.setMaximumWidth(230)

        self._gl = gl.GLViewWidget()
        self._gl.opts["distance"]  = 3.5
        self._gl.opts["elevation"] = 25
        self._gl.opts["azimuth"]   = -50
        self._gl.setBackgroundColor(pg.mkColor(BG))
        self._gl.setFixedHeight(170)

        self._axis_lines: list[tuple[np.ndarray, gl.GLLinePlotItem]] = []
        for vec, color in [
            (np.array([_AXIS_LEN, 0.0, 0.0]), (1.0, 0.3, 0.3, 1.0)),
            (np.array([0.0, _AXIS_LEN, 0.0]), (0.3, 1.0, 0.3, 1.0)),
            (np.array([0.0, 0.0, _AXIS_LEN]), (0.3, 0.6, 1.0, 1.0)),
        ]:
            pts  = np.array([[0, 0, 0], vec], dtype=np.float32)
            line = gl.GLLinePlotItem(pos=pts, color=color, width=4.0, antialias=True)
            self._gl.addItem(line)
            self._axis_lines.append((vec.copy(), line))

        self._body_lines = _build_robot(self._gl)

        self._dot = QLabel("●  IMU")
        self._dot.setStyleSheet(f"color: {DIM}; font-size: 12px;")
        self._gyro_lbl = QLabel("gyro_z:  —")
        self._gyro_lbl.setStyleSheet(
            f"color: {TEXT}; font-size: 11px; font-family: Consolas;"
        )

        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self._dot)
        row.addWidget(self._gyro_lbl)
        row.addStretch()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 4)
        lay.setSpacing(4)
        lay.addWidget(self._gl)
        lay.addLayout(row)

        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.setInterval(3000)
        self._timeout.timeout.connect(
            lambda: self._dot.setStyleSheet(f"color: {DIM}; font-size: 12px;")
        )

        TelemetryBus.instance().packet.connect(self._on_packet)

    def _on_packet(self, info: dict):
        if info.get("ptype") == 0x01 and not self.isVisible():
            return
        if info.get("ptype") != 0x01:
            return
        pitch = info.get("pitch_rad",       0.0)
        roll  = info.get("roll_rad",        0.0)
        yaw   = info.get("yaw_rad",         0.0)
        rate  = info.get("pitch_rate_rads", 0.0)

        self._dot.setStyleSheet(f"color: {GREEN}; font-size: 12px;")
        self._gyro_lbl.setText(f"gyro_z:  {rate:+.3f}")
        self._timeout.start()
        _apply_rotation(pitch, roll, yaw, self._axis_lines, self._body_lines)
