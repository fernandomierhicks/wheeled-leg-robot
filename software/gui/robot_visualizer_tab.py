"""Robot Visualizer Tab — live 3-D 4-bar linkage viewer with coordinate frames."""

import math

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from telemetry_bus import TelemetryBus
from theme import BG, BLUE, BORDER, DIM, GREEN, ORANGE, RED, SURFACE, TEXT, YELLOW, WHITE

pg.setConfigOptions(antialias=True, background=BG, foreground=TEXT)

# ── Baseline-1 geometry (params.py RobotGeometry, run_id 51167) ───────────────
_L_FEMUR = 0.17378   # hip → knee A→C  [m]
_L_STUB  = 0.03513   # tibia stub C→E  [m]
_L_TIBIA = 0.12939   # tibia C→W       [m]
_LC      = 0.15081   # coupler F→E     [m]
_F_X     = -0.05887  # fixed pivot X   [m]
_F_Z     = -0.01821  # fixed pivot Z   [m]
_A_Z     = -0.0235   # hip motor Z     [m]
_LEG_Y   =  0.1430   # leg-plane Y (±) [m]
_WHEEL_R =  0.075    # wheel radius    [m]
_Q_NOM   = -1.082366 # (Q_RET+Q_EXT)/2 [rad]

# Body box half-extents [m]  (approximate chassis)
_BX, _BY, _BZ = 0.10, 0.07, 0.05

# Coordinate-frame arrow lengths [m]
_FR_BODY  = 0.09    # body/IMU frame (slightly larger)
_FR_JOINT = 0.055   # hip & wheel frames


# ── 4-bar IK (ported from simulation/mujoco/master_sim_jump/physics.py) ───────

def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _solve_ik(q_hip: float) -> dict | None:
    """4-bar forward kinematics. Returns pivot XZ positions in body frame, or
    None near singularity."""
    C_x = -_L_FEMUR * math.cos(q_hip)
    C_z =  _A_Z + _L_FEMUR * math.sin(q_hip)
    dx, dz = C_x - _F_X, C_z - _F_Z
    R = math.sqrt(dx * dx + dz * dz)
    if R < 1e-9:
        return None
    K = (_LC**2 - dx**2 - dz**2 - _L_STUB**2) / (2.0 * _L_STUB)
    if abs(K) / R >= 0.95:
        return None
    phi   = math.atan2(dz, dx)
    asinv = math.asin(max(-1.0, min(1.0, K / R)))
    a1    = _wrap(asinv - phi)
    a2    = _wrap(math.pi - asinv - phi)
    alpha = a1 if abs(a1 - q_hip) <= abs(a2 - q_hip) else a2
    E_x = C_x + _L_STUB * math.sin(alpha)
    E_z = C_z + _L_STUB * math.cos(alpha)
    W_x = C_x - _L_TIBIA * math.sin(alpha)
    W_z = C_z - _L_TIBIA * math.cos(alpha)
    return dict(
        A=(0.0, _A_Z), C=(C_x, C_z), E=(E_x, E_z),
        F=(_F_X, _F_Z), W=(W_x, W_z),
    )


# ── 3-D helpers ───────────────────────────────────────────────────────────────

def _rot3(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """Rotation matrix  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll),  math.sin(roll)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ])


def _xz_to_world(xz_pairs: list[tuple], y: float,
                 R: np.ndarray) -> np.ndarray:
    """Convert [(x,z), …] body-frame points → world-frame float32 array."""
    body = np.array([[x, y, z] for x, z in xz_pairs])
    return (R @ body.T).T.astype(np.float32)


def _wheel_pts_body(W_xz: tuple, y: float, N: int = 28) -> np.ndarray:
    """Wheel-rim circle in body frame (wheel spins around Y-axis)."""
    wx, wz = W_xz
    ang = np.linspace(0.0, 2.0 * math.pi, N + 1)
    return np.column_stack([
        wx + _WHEEL_R * np.sin(ang),
        np.full(N + 1, y),
        wz + _WHEEL_R * np.cos(ang),
    ])


def _line_item(view: gl.GLViewWidget, pts: np.ndarray,
               color: tuple, width: float = 2.5) -> gl.GLLinePlotItem:
    item = gl.GLLinePlotItem(pos=pts.astype(np.float32),
                             color=color, width=width, antialias=True)
    view.addItem(item)
    return item


# ── Coordinate-frame helpers ──────────────────────────────────────────────────

def _make_frame(view: gl.GLViewWidget, length: float) -> list:
    """Three GLLinePlotItems for X (red) / Y (green) / Z (blue) axes."""
    colors = [
        (1.00, 0.22, 0.22, 1.0),   # X red
        (0.22, 1.00, 0.22, 1.0),   # Y green
        (0.22, 0.55, 1.00, 1.0),   # Z blue
    ]
    items = []
    for color in colors:
        item = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=color, width=3.5, antialias=True,
        )
        view.addItem(item)
        items.append(item)
    return items  # [x_item, y_item, z_item]


def _set_frame(items: list, p_body: np.ndarray, R: np.ndarray,
               length: float) -> None:
    """Update frame axes at body-local position p_body using rotation R."""
    o = R @ p_body
    for i, item in enumerate(items):
        axis = np.zeros(3)
        axis[i] = length
        item.setData(pos=np.array([o, o + R @ axis], dtype=np.float32))


# ── Tab ───────────────────────────────────────────────────────────────────────

class RobotVisualizerTab(QWidget):

    def __init__(self):
        super().__init__()

        # ── GL view ──────────────────────────────────────────────────────────
        self._gl = gl.GLViewWidget()
        self._gl.opts["distance"]  = 1.5
        self._gl.opts["elevation"] = 18
        self._gl.opts["azimuth"]   = -55
        self._gl.setBackgroundColor(pg.mkColor(BG))
        self._gl.setMinimumSize(400, 300)

        # Static floor grid
        grid = gl.GLGridItem()
        grid.setSize(1.5, 1.5, 1)
        grid.setSpacing(0.1, 0.1, 0.1)
        grid.setColor((255, 255, 255, 20))
        grid.translate(0.0, 0.0, -0.35)
        self._gl.addItem(grid)

        # World-axis reference lines (dim, world-static)
        for pts, col in [
            (np.array([[0,0,0],[0.25,0,0]]), (1.0,0.2,0.2,0.25)),
            (np.array([[0,0,0],[0,0.25,0]]), (0.2,1.0,0.2,0.25)),
            (np.array([[0,0,0],[0,0,0.25]]), (0.2,0.5,1.0,0.25)),
        ]:
            self._gl.addItem(gl.GLLinePlotItem(
                pos=pts.astype(np.float32), color=col, width=1.5, antialias=True))

        # ── Body box (rotated each frame) ────────────────────────────────────
        bx, by, bz = _BX, _BY, _BZ
        v = np.array([
            [-bx,-by,-bz],[+bx,-by,-bz],[+bx,+by,-bz],[-bx,+by,-bz],
            [-bx,-by,+bz],[+bx,-by,+bz],[+bx,+by,+bz],[-bx,+by,+bz],
        ])
        BOX = (0.65, 0.65, 0.72, 0.85)
        self._box_pairs: list[tuple[np.ndarray, gl.GLLinePlotItem]] = []
        for edge in [v[[0,1,2,3,0]], v[[4,5,6,7,4]],
                     v[[0,4,5,1]],   v[[2,6,7,3]]]:
            item = _line_item(self._gl, edge, BOX, 2.0)
            self._box_pairs.append((edge.copy(), item))

        # ── 4-bar leg line items ──────────────────────────────────────────────
        nom = _solve_ik(_Q_NOM)
        self._leg_L = self._build_leg(+1, nom)
        self._leg_R = self._build_leg(-1, nom)

        # ── Coordinate frames ─────────────────────────────────────────────────
        # body/IMU at chassis origin
        self._fr_body    = _make_frame(self._gl, _FR_BODY)
        # hip motor shaft (fixed in body)
        self._fr_hip_L   = _make_frame(self._gl, _FR_JOINT)
        self._fr_hip_R   = _make_frame(self._gl, _FR_JOINT)
        # wheel centre (moves with hip angle)
        self._fr_wheel_L = _make_frame(self._gl, _FR_JOINT)
        self._fr_wheel_R = _make_frame(self._gl, _FR_JOINT)

        # Initial draw at identity rotation, nominal hip angle
        self._redraw(_Q_NOM, _Q_NOM, 0.0, 0.0, 0.0)

        # ── Right info panel ─────────────────────────────────────────────────
        self._lbl: dict[str, QLabel] = {}
        panel = self._build_panel()

        # ── Layout ───────────────────────────────────────────────────────────
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._gl)
        split.addWidget(panel)
        split.setSizes([700, 195])
        split.setHandleWidth(5)
        split.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(split)

        TelemetryBus.instance().packet.connect(self._on_packet)

    # ── Builder helpers ───────────────────────────────────────────────────────

    def _build_leg(self, y_sign: int, ik: dict | None) -> dict:
        y  = y_sign * _LEG_Y
        R0 = np.eye(3)
        if ik is None:
            ik = {'A': (0.0, _A_Z), 'C': (0.0, _A_Z - 0.01),
                  'E': (0.01, _A_Z + _L_STUB), 'F': (_F_X, _F_Z),
                  'W': (0.0, _A_Z - _L_TIBIA)}
        return dict(
            y       = y,
            femur   = _line_item(self._gl, _xz_to_world([ik['A'], ik['C']], y, R0),
                                 (1.00, 0.55, 0.00, 1.0), 3.5),
            tibia   = _line_item(self._gl, _xz_to_world([ik['C'], ik['W']], y, R0),
                                 (0.30, 0.80, 1.00, 1.0), 3.5),
            stub    = _line_item(self._gl, _xz_to_world([ik['C'], ik['E']], y, R0),
                                 (0.40, 1.00, 0.40, 1.0), 2.5),
            coupler = _line_item(self._gl, _xz_to_world([ik['F'], ik['E']], y, R0),
                                 (0.90, 0.30, 0.90, 1.0), 2.5),
            wheel   = _line_item(self._gl,
                                 (_wheel_pts_body(ik['W'], y) @ R0.T).astype(np.float32),
                                 (0.72, 0.72, 0.72, 0.80), 2.0),
        )

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(
            f"QWidget {{ background: {SURFACE}; border: 1px solid {BORDER};"
            f" border-radius: 3px; }}"
            f"QLabel  {{ border: none; }}"
        )
        panel.setMinimumWidth(170)
        panel.setMaximumWidth(215)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        hdr = QLabel("Robot Visualizer")
        hdr.setStyleSheet(f"color: {BLUE}; font-weight: bold; font-size: 11px;")
        lay.addWidget(hdr)

        for name in ["Hip L", "Hip R", "Ext L", "Ext R",
                     "Pitch", "Roll", "Yaw", "State"]:
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
            lay.addLayout(row)
            self._lbl[name] = v

        lay.addStretch()

        link_legend = QLabel(
            '<b style="color:#888">Links</b><br>'
            f'<span style="color:#ff8c00">■</span> Femur (A→C)<br>'
            f'<span style="color:#4dccff">■</span> Tibia (C→W)<br>'
            f'<span style="color:#66ff66">■</span> Stub  (C→E)<br>'
            f'<span style="color:#e64de6">■</span> Coupler (F→E)'
        )
        link_legend.setStyleSheet(f"font-size: 10px; color: {DIM}; line-height: 160%;")
        lay.addWidget(link_legend)

        fr_legend = QLabel(
            '<b style="color:#888">Frames</b><br>'
            f'<span style="color:#ff3838">■</span> +X fwd &nbsp;'
            f'<span style="color:#38ff38">■</span> +Y left &nbsp;'
            f'<span style="color:#388cff">■</span> +Z up<br>'
            "• Body/IMU (large)<br>"
            "• Hip A  L / R<br>"
            "• Wheel W  L / R"
        )
        fr_legend.setStyleSheet(f"font-size: 10px; color: {DIM}; line-height: 160%;")
        lay.addWidget(fr_legend)

        return panel

    # ── Core update ───────────────────────────────────────────────────────────

    def _redraw(self, q_l: float, q_r: float,
                pitch: float, roll: float, yaw: float) -> None:
        R = _rot3(pitch, roll, yaw)

        # Body box
        for pts0, item in self._box_pairs:
            item.setData(pos=(R @ pts0.T).T.astype(np.float32))

        # Legs
        ik_l = _solve_ik(q_l)
        ik_r = _solve_ik(q_r)
        self._update_leg(self._leg_L, ik_l, R)
        self._update_leg(self._leg_R, ik_r, R)

        # Body/IMU frame at chassis origin
        _set_frame(self._fr_body, np.zeros(3), R, _FR_BODY)

        # Hip motor frames (fixed in body frame at A position)
        _set_frame(self._fr_hip_L, np.array([0.0, +_LEG_Y, _A_Z]), R, _FR_JOINT)
        _set_frame(self._fr_hip_R, np.array([0.0, -_LEG_Y, _A_Z]), R, _FR_JOINT)

        # Wheel frames (move with hip angle)
        if ik_l:
            _set_frame(self._fr_wheel_L,
                       np.array([ik_l['W'][0], +_LEG_Y, ik_l['W'][1]]), R, _FR_JOINT)
        if ik_r:
            _set_frame(self._fr_wheel_R,
                       np.array([ik_r['W'][0], -_LEG_Y, ik_r['W'][1]]), R, _FR_JOINT)

    def _update_leg(self, leg: dict, ik: dict | None, R: np.ndarray) -> None:
        if ik is None:
            return
        y = leg['y']
        leg['femur'].setData(pos=_xz_to_world([ik['A'], ik['C']], y, R))
        leg['tibia'].setData(pos=_xz_to_world([ik['C'], ik['W']], y, R))
        leg['stub'].setData(pos=_xz_to_world([ik['C'], ik['E']], y, R))
        leg['coupler'].setData(pos=_xz_to_world([ik['F'], ik['E']], y, R))
        w_body = _wheel_pts_body(ik['W'], y)
        leg['wheel'].setData(pos=(R @ w_body.T).T.astype(np.float32))

    # ── Telemetry handler ─────────────────────────────────────────────────────

    def _on_packet(self, info: dict) -> None:
        if info.get("ptype") != 0x01:
            return

        q_l   = info.get("hip_l_pos_rad", _Q_NOM)
        q_r   = info.get("hip_r_pos_rad", _Q_NOM)
        pitch = info.get("pitch_rad",     0.0)
        roll  = info.get("roll_rad",      0.0)
        yaw   = info.get("yaw_rad",       0.0)
        state = info.get("state_name",    "—")

        self._redraw(q_l, q_r, pitch, roll, yaw)

        ik_l = _solve_ik(q_l)
        ik_r = _solve_ik(q_r)

        self._lbl["Hip L"].setText(f"{math.degrees(q_l):+.1f}°")
        self._lbl["Hip R"].setText(f"{math.degrees(q_r):+.1f}°")
        self._lbl["Ext L"].setText(
            f"{ik_l['W'][1]*1000:+.0f} mm" if ik_l else "singularity")
        self._lbl["Ext R"].setText(
            f"{ik_r['W'][1]*1000:+.0f} mm" if ik_r else "singularity")
        self._lbl["Pitch"].setText(f"{math.degrees(pitch):+.1f}°")
        self._lbl["Roll"].setText(f"{math.degrees(roll):+.1f}°")
        self._lbl["Yaw"].setText(f"{math.degrees(yaw):+.1f}°")

        color = {"RUNNING": GREEN, "ESTOP": RED, "CALIBRATION": BLUE,
                 "STANDBY": YELLOW, "STARTUP": WHITE}.get(state, TEXT)
        self._lbl["State"].setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: bold;"
            f" font-family: Consolas;"
        )
        self._lbl["State"].setText(state)
