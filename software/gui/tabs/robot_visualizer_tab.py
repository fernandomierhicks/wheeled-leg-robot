"""Robot Visualizer Tab — cockpit instrument panels surrounding live 3-D scene."""

import math

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PyQt6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from .comm_commands import send_param_get_all
from .telemetry_bus import TelemetryBus
from .theme import BG, BLUE, BORDER, DIM, GREEN, ORANGE, RED, SURFACE, TEXT, YELLOW, WHITE

# Per-motor presence params (firmware ParamRegistry GROUP_SYSTEM) — read-only
# indicators here; toggle via the Parameters tab's param browser.
_PARAM_HIP_L_ENABLE   = 0x0005
_PARAM_HIP_R_ENABLE   = 0x0006
_PARAM_WHEEL_L_ENABLE = 0x0007
_PARAM_WHEEL_R_ENABLE = 0x0008

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

_R_FEMUR   = 0.007
_R_TIBIA   = 0.008
_R_COUPLER = 0.005
_R_STUB    = 0.008

_HIP_MOTOR_R = 0.0265
_HIP_MOTOR_L = 0.043

_WHEEL_MOTOR_R = 0.025
_WHEEL_MOTOR_L = 0.065
_WHEEL_T       = 0.0325
_N_SPOKES      = 3
_SPOKE_FADE    = 0.88

_BX, _BY, _BZ = 0.10, 0.07, 0.05

# IMU (BNO086) block — representative breakout-board size (30 × 20 × 4 mm)
_IMU_BX  = 0.015
_IMU_BY  = 0.010
_IMU_BZ  = 0.002
_IMU_POS = np.array([0.0, -0.03, _BZ + 0.002], dtype=float)  # sits on top face of body box, offset right
_FR_IMU  = 0.040
# Chip mounted face-down: pitch_rate = -gyro.y, roll = +gyro.x, yaw = +gyro.z
# → IMU-X = body+X, IMU-Y = body−Y, IMU-Z = body−Z
_R_IMU_MOUNT = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)

_FR_BODY  = 0.09
_FR_JOINT = 0.055

_R_MOUNT_LEFT  = np.array([[1, 0, 0], [0, 0, 1], [0,-1, 0]], dtype=float)
_R_MOUNT_RIGHT = np.array([[1, 0, 0], [0, 0,-1], [0, 1, 0]], dtype=float)
_R_MOUNT_IDENT = np.eye(3)

_HIP_L_P1 = np.array([0.0,  _LEG_Y,                _A_Z])
_HIP_L_P2 = np.array([0.0,  _LEG_Y - _HIP_MOTOR_L, _A_Z])
_HIP_R_P1 = np.array([0.0, -_LEG_Y,                _A_Z])
_HIP_R_P2 = np.array([0.0, -_LEG_Y + _HIP_MOTOR_L, _A_Z])

_BOX_VERTS_BASE = np.array([
    [-_BX,-_BY,-_BZ],[+_BX,-_BY,-_BZ],[+_BX,+_BY,-_BZ],[-_BX,+_BY,-_BZ],
    [-_BX,-_BY,+_BZ],[+_BX,-_BY,+_BZ],[+_BX,+_BY,+_BZ],[-_BX,+_BY,+_BZ],
], dtype=np.float32)

_BOX_FACES = np.array([
    [0,1,2],[0,2,3],
    [4,6,5],[4,7,6],
    [0,5,1],[0,4,5],
    [2,3,7],[2,7,6],
    [0,3,7],[0,7,4],
    [1,2,6],[1,6,5],
], dtype=np.uint32)

_IMU_VERTS_BASE = (np.array([
    [-_IMU_BX, -_IMU_BY, -_IMU_BZ], [+_IMU_BX, -_IMU_BY, -_IMU_BZ],
    [+_IMU_BX, +_IMU_BY, -_IMU_BZ], [-_IMU_BX, +_IMU_BY, -_IMU_BZ],
    [-_IMU_BX, -_IMU_BY, +_IMU_BZ], [+_IMU_BX, -_IMU_BY, +_IMU_BZ],
    [+_IMU_BX, +_IMU_BY, +_IMU_BZ], [-_IMU_BX, +_IMU_BY, +_IMU_BZ],
], dtype=np.float32) + _IMU_POS.astype(np.float32))

_TOF_DOWN_ANGLE = math.radians(20)
_TOF_HALF_SQ    = 0.0085
_TOF_MAX_MM     = 2000
_TOF_WARN_MM    = 400
_TOF_NEAR_MM    = 200
_TOF_NO_DATA    = 0xFFFF

_TOF_DEFS = (
    (np.array([+_BX,  0.0,  0.015]),
     np.array([1.0, 0.0, 0.0])),
    (np.array([+_BX,  0.0, -0.015]),
     np.array([math.cos(_TOF_DOWN_ANGLE), 0.0, -math.sin(_TOF_DOWN_ANGLE)])),
    (np.array([-_BX,  0.0,  0.015]),
     np.array([-1.0, 0.0, 0.0])),
    (np.array([-_BX,  0.0, -0.015]),
     np.array([-math.cos(_TOF_DOWN_ANGLE), 0.0, -math.sin(_TOF_DOWN_ANGLE)])),
)
_TOF_BEAM_COLORS = (
    (1.00, 1.00, 0.30, 1.0),
    (1.00, 0.65, 0.10, 1.0),
    (0.30, 1.00, 1.00, 1.0),
    (0.20, 0.70, 1.00, 1.0),
)
_TOF_SQ_COLOR    = (1.00, 0.85, 0.35, 1.0)
_TOF_LABEL_NAMES = ("F-Horiz", "F-Down", "R-Horiz", "R-Down")

_EMPTY_MD = gl.MeshData(
    vertexes=np.zeros((3, 3), dtype=np.float32),
    faces=np.array([[0, 1, 2]], dtype=np.uint32),
)

# ── Health flag bit positions ─────────────────────────────────────────────────
_HF_HIP_L_OK        = 0
_HF_HIP_R_OK        = 1
_HF_WM_L_OK         = 2
_HF_WM_R_OK         = 3
_HF_HIP_LIMITS_VALID= 4
_HF_IMU_NOMINAL     = 5
_HF_LQR_ACTIVE      = 6
_HF_VEL_PI_SAT      = 7
_HF_YAW_PI_SAT      = 8
_HF_WM_L_VEL_LIM    = 9
_HF_WM_R_VEL_LIM    = 10

# ── ODrive axis states ────────────────────────────────────────────────────────
_ODRIVE_CLOSED = 8
_ODRIVE_IDLE   = 1

# ── WheelMode enum ────────────────────────────────────────────────────────────
_WM_MODE_LABELS = {0: "IDLE", 1: "VEL", 2: "POS", 3: "TRQ"}

# ── Jump state labels ─────────────────────────────────────────────────────────
_JUMP_STATE_LABELS = {0: "IDLE", 1: "CROUCH", 2: "LAUNCH", 3: "AIRBORNE",
                      4: "LAND", 5: "RECOVER", 6: "DONE"}


# ── 4-bar IK ─────────────────────────────────────────────────────────────────

def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _solve_ik(q_hip: float) -> dict | None:
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
        F=(_F_X, _F_Z), W=(W_x, W_z), alpha=alpha,
    )


def _solve_ik_right(q_r: float) -> dict | None:
    return _solve_ik(q_r)


# Fallback pose used when the live hip angle doesn't solve (e.g. raw encoder
# reading before this session's hip calibration has run) — keeps the leg/wheel
# rendered at a reasonable pose instead of vanishing, so wheel-spin telemetry
# stays visible even with an uncalibrated hip. _Q_NOM is the geometry's nominal
# operating point and is always solvable.
_IK_L_FALLBACK = _solve_ik(_Q_NOM)
_IK_R_FALLBACK = _solve_ik_right(_Q_NOM)


# ── 3-D helpers ───────────────────────────────────────────────────────────────

def _ry(q: float) -> np.ndarray:
    cq, sq = math.cos(q), math.sin(q)
    return np.array([[cq, 0, sq], [0, 1, 0], [-sq, 0, cq]])


def _rot3(pitch: float, roll: float, yaw: float) -> np.ndarray:
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll),  math.sin(roll)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ])


def _cylinder_mesh(p1: np.ndarray, p2: np.ndarray,
                   radius: float, n: int = 8) -> gl.MeshData:
    d = p2 - p1
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return _EMPTY_MD
    zv = d / L
    ref = np.array([1., 0., 0.]) if abs(zv[0]) < 0.9 else np.array([0., 1., 0.])
    xv = np.cross(zv, ref); xv /= np.linalg.norm(xv)
    yv = np.cross(zv, xv)
    ang = np.linspace(0., 2. * np.pi, n, endpoint=False)
    ring = radius * (np.outer(np.cos(ang), xv) + np.outer(np.sin(ang), yv))
    bot = (p1 + ring).astype(np.float32)
    top = (p2 + ring).astype(np.float32)
    verts = np.vstack([bot, top])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n+j], [i, n+j, n+i]]
    return gl.MeshData(vertexes=verts, faces=np.array(faces, dtype=np.uint32))


def _circle_pts(center: np.ndarray, r: float, axis_u: np.ndarray, axis_v: np.ndarray,
                n: int = 20) -> np.ndarray:
    """Points around a circle of radius r in the plane spanned by axis_u/axis_v."""
    ang = np.linspace(0.0, 2.0 * np.pi, n)
    return center + r * (np.outer(np.cos(ang), axis_u) + np.outer(np.sin(ang), axis_v))


def _arc_pts(center: np.ndarray, r: float, axis_u: np.ndarray, axis_v: np.ndarray,
            a0: float, a1: float, n: int = 14) -> np.ndarray:
    """Points along an arc (radians a0..a1) in the plane spanned by axis_u/axis_v."""
    ang = np.linspace(a0, a1, n)
    return center + r * (np.outer(np.cos(ang), axis_u) + np.outer(np.sin(ang), axis_v))


def _make_frame(view: gl.GLViewWidget, length: float) -> list:
    colors = [
        (1.00, 0.22, 0.22, 1.0),
        (0.22, 1.00, 0.22, 1.0),
        (0.22, 0.55, 1.00, 1.0),
    ]
    items = []
    for color in colors:
        item = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=color, width=3.5, antialias=True,
        )
        view.addItem(item)
        items.append(item)
    return items


def _set_frame(items: list, origin_world: np.ndarray, R_orient: np.ndarray,
               length: float, R_mount: np.ndarray | None = None) -> None:
    R_eff = R_orient @ (R_mount if R_mount is not None else _R_MOUNT_IDENT)
    for i, item in enumerate(items):
        axis = np.zeros(3)
        axis[i] = length
        item.setData(pos=np.array([origin_world,
                                   origin_world + R_eff @ axis], dtype=np.float32))


def _mesh_item(view: gl.GLViewWidget, md: gl.MeshData,
               color: tuple) -> gl.GLMeshItem:
    item = gl.GLMeshItem(meshdata=md, smooth=True, drawEdges=False,
                         color=color, glOptions='translucent')
    view.addItem(item)
    return item


# ── Orientation markings (rotate rigidly with the body, no translation) ───────
# "Happy face" on the front (+X) face, just above the front ToF sensor pair
# (mounted at z=+0.015, see _TOF_DEFS), so it's an at-a-glance front-side marker.
_AXIS_Y = np.array([0.0, 1.0, 0.0])
_AXIS_Z = np.array([0.0, 0.0, 1.0])

_FACE_X       = _BX + 0.0015  # a hair proud of the box face to avoid z-fighting
_FACE_CENTER  = np.array([_FACE_X, 0.0, 0.033])
_FACE_R       = 0.014
_EYE_R        = 0.0022
_EYE_DY       = 0.006
_EYE_DZ       = 0.004
_SMILE_CENTER = _FACE_CENTER + np.array([0.0, 0.0, -0.002])
_SMILE_R      = 0.008
_SMILE_A0, _SMILE_A1 = math.radians(200), math.radians(340)

_FACE_COLOR = (0.0, 0.0, 0.0, 1.0)

_FACE_OUTLINE_BASE = _circle_pts(_FACE_CENTER, _FACE_R, _AXIS_Y, _AXIS_Z, n=24).astype(np.float32)
_FACE_EYE_L_BASE = _circle_pts(_FACE_CENTER + np.array([0.0, _EYE_DY, _EYE_DZ]),
                               _EYE_R, _AXIS_Y, _AXIS_Z, n=10).astype(np.float32)
_FACE_EYE_R_BASE = _circle_pts(_FACE_CENTER + np.array([0.0, -_EYE_DY, _EYE_DZ]),
                               _EYE_R, _AXIS_Y, _AXIS_Z, n=10).astype(np.float32)
_FACE_SMILE_BASE = _arc_pts(_SMILE_CENTER, _SMILE_R, _AXIS_Y, _AXIS_Z,
                            _SMILE_A0, _SMILE_A1, n=14).astype(np.float32)

# Forward arrow on the top (+Z) face, pointing +X. Drawn as three disconnected
# segments (shaft + two head wings) via GLLinePlotItem mode='lines'.
_ARROW_Z            = _BZ + 0.0015
_ARROW_TAIL_X       = -0.045
_ARROW_TIP_X        = 0.085
_ARROW_HEAD_LEN     = 0.028
_ARROW_HEAD_HALF_W  = 0.018
_ARROW_COLOR        = (0.0, 0.0, 0.0, 1.0)

_FWD_ARROW_BASE = np.array([
    [_ARROW_TAIL_X, 0.0, _ARROW_Z], [_ARROW_TIP_X, 0.0, _ARROW_Z],
    [_ARROW_TIP_X,  0.0, _ARROW_Z], [_ARROW_TIP_X - _ARROW_HEAD_LEN,  _ARROW_HEAD_HALF_W, _ARROW_Z],
    [_ARROW_TIP_X,  0.0, _ARROW_Z], [_ARROW_TIP_X - _ARROW_HEAD_LEN, -_ARROW_HEAD_HALF_W, _ARROW_Z],
], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Shared style helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_group(title: str) -> QGroupBox:
    gb = QGroupBox(title)
    gb.setStyleSheet(
        f"QGroupBox {{ color: {DIM}; font-size: 10px; font-weight: bold;"
        f" border: 1px solid {BORDER}; border-radius: 4px;"
        f" margin-top: 10px; padding: 4px 4px 4px 4px; }}"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 6px; padding: 0 3px; }}"
    )
    return gb


def _mono_label(text: str = "—", color: str = TEXT, size: int = 10) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {color}; font-size: {size}px; font-family: Consolas;"
    )
    return lbl


def _dim_label(text: str, size: int = 9) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {DIM}; font-size: {size}px;")
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
# Custom painted instrument widgets
# ─────────────────────────────────────────────────────────────────────────────

class LedIndicator(QWidget):
    """Small colored circle LED."""

    _C = {"green": GREEN, "yellow": YELLOW, "red": RED, "blue": BLUE,
          "orange": ORANGE, "gray": DIM}

    def __init__(self, size: int = 9):
        super().__init__()
        self._color = QColor(DIM)
        self._sz = size
        self.setFixedSize(size + 4, size + 4)

    def set_state(self, state: str) -> None:
        self._color = QColor(self._C.get(state, DIM))
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self._color))
        p.drawEllipse(2, 2, self._sz, self._sz)


class AngleArcWidget(QWidget):
    """Half-circle arc gauge.

    Draws a 180° semicircle (top half).  Left edge = -max_val, right = +max_val.
    Actual value: thick white needle.  Optional setpoint: thin yellow needle.
    Rate bar: small horizontal bar below the arc.
    """

    def __init__(self, title: str, max_val: float, unit: str = "°",
                 show_setpoint: bool = False, max_rate: float = 1.0):
        super().__init__()
        self._title = title
        self._max_val = max(float(max_val), 1e-9)
        self._unit = unit
        self._show_sp = show_setpoint
        self._max_rate = max(float(max_rate), 1e-9)
        self._actual   = 0.0
        self._setpoint = 0.0
        self._rate     = 0.0
        self.setMinimumHeight(78)
        self.setMaximumHeight(92)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, actual: float, setpoint: float = 0.0, rate: float = 0.0) -> None:
        self._actual, self._setpoint, self._rate = actual, setpoint, rate
        self.update()

    def _qt_deg(self, val: float) -> float:
        """Map value to Qt arc angle in degrees (0=3-oclock, 90=12-oclock, 180=9-oclock)."""
        norm = (val + self._max_val) / (2.0 * self._max_val)
        return (1.0 - max(0.0, min(1.0, norm))) * 180.0

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = float(self.width()), float(self.height())
        margin  = 7.0
        rate_h  = 11.0
        text_h  = 13.0
        arc_h   = h - rate_h - text_h

        r  = min((w - 2.0 * margin) / 2.0, arc_h - 2.0)
        cx = w / 2.0
        cy = arc_h  # pivot at bottom of arc area

        # Track arc
        rect = QRectF(cx - r, cy - r, 2.0 * r, 2.0 * r)
        pen = QPen(QColor(BORDER))
        pen.setWidth(3)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(rect, 0, 180 * 16)

        def _needle(val: float, color: str, width: int) -> None:
            a = math.radians(self._qt_deg(val))
            ex, ey = cx + r * math.cos(a), cy - r * math.sin(a)
            pen2 = QPen(QColor(color))
            pen2.setWidth(width)
            pen2.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen2)
            p.drawLine(QPointF(cx, cy), QPointF(ex, ey))

        _needle(self._actual, WHITE, 3)
        if self._show_sp:
            _needle(self._setpoint, RED, 1)

        # Pivot dot
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(BORDER)))
        p.drawEllipse(QPointF(cx, cy), 3.5, 3.5)

        # Value text (inside arc, near center)
        val_color = RED if abs(self._actual) > self._max_val * 0.88 else TEXT
        p.setPen(QPen(QColor(val_color)))
        p.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        p.drawText(QRectF(0.0, cy - r * 0.68, w, 15.0),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                   f"{self._actual:+.1f}{self._unit}")

        # Title below arc
        p.setPen(QPen(QColor(DIM)))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(QRectF(0.0, arc_h + 1.0, w, text_h),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                   self._title)

        # Rate bar
        by = h - rate_h + 2.0
        bw = w - 2.0 * margin
        bh = 6.0
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(BORDER)))
        p.drawRoundedRect(QRectF(margin, by, bw, bh), 2.0, 2.0)
        norm = max(-1.0, min(1.0, self._rate / self._max_rate))
        fill = abs(norm) * bw / 2.0
        fx   = cx if norm >= 0.0 else cx - fill
        if fill > 0.5:
            p.setBrush(QBrush(QColor(BLUE)))
            p.drawRoundedRect(QRectF(fx, by, fill, bh), 2.0, 2.0)


class BannerBarWidget(QWidget):
    """Single-row distance banner: label | colored fill bar | mm overlay."""

    def __init__(self, label: str, max_mm: int = 2000):
        super().__init__()
        self._label  = label
        self._max_mm = float(max_mm)
        self._dist   = -1
        self.setFixedHeight(22)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_dist(self, dist_mm: int) -> None:
        self._dist = dist_mm
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = float(self.width()), float(self.height())
        lw   = 46.0
        pad  = 3.0
        bx   = lw + pad
        bw   = w - bx - pad
        bh   = h - 4.0
        by   = 2.0

        # Bar background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(BORDER)))
        p.drawRoundedRect(QRectF(bx, by, bw, bh), 3.0, 3.0)

        no_data = (self._dist < 0 or self._dist == _TOF_NO_DATA)
        if not no_data:
            frac = max(0.0, min(1.0, float(self._dist) / self._max_mm))
            fill_col = (RED    if self._dist < _TOF_NEAR_MM else
                        YELLOW if self._dist < _TOF_WARN_MM else GREEN)
            p.setBrush(QBrush(QColor(fill_col)))
            p.drawRoundedRect(QRectF(bx, by, frac * bw, bh), 3.0, 3.0)

        # Label
        p.setPen(QPen(QColor(DIM)))
        p.setFont(QFont("Consolas", 8))
        p.drawText(QRectF(0.0, 0.0, lw - 2.0, h),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   self._label)

        # Value overlay
        val_str = "—" if no_data else f"{self._dist} mm"
        p.setPen(QPen(QColor(TEXT)))
        p.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        p.drawText(QRectF(bx, 0.0, bw - 3.0, h),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   val_str)


class FillBarWidget(QWidget):
    """Left-to-right fill bar for positive quantities (e.g. current)."""

    def __init__(self, label: str, max_val: float, unit: str = "",
                 warn: float = 0.65, crit: float = 0.88):
        super().__init__()
        self._label   = label
        self._max_val = max(float(max_val), 1e-9)
        self._unit    = unit
        self._warn    = warn
        self._crit    = crit
        self._value   = 0.0
        self.setFixedHeight(18)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, v: float) -> None:
        self._value = v
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())
        lw = 28.0; pad = 3.0
        bx = lw + pad; bw = w - bx - pad
        by = 3.0; bh = h - 6.0

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(BORDER)))
        p.drawRoundedRect(QRectF(bx, by, bw, bh), 2.0, 2.0)

        frac = max(0.0, min(1.0, self._value / self._max_val))
        color = RED if frac > self._crit else (ORANGE if frac > self._warn else GREEN)
        if frac > 0.01:
            p.setBrush(QBrush(QColor(color)))
            p.drawRoundedRect(QRectF(bx, by, frac * bw, bh), 2.0, 2.0)

        p.setPen(QPen(QColor(DIM)))
        p.setFont(QFont("Consolas", 8))
        p.drawText(QRectF(0.0, 0.0, lw - 1.0, h),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   self._label)
        p.setPen(QPen(QColor(TEXT)))
        p.setFont(QFont("Consolas", 8))
        p.drawText(QRectF(bx, 0.0, bw - 2.0, h),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"{self._value:.2f}{self._unit}")


class DeviationBarWidget(QWidget):
    """Centred bidirectional bar: fill from centre proportional to (value - setpoint)."""

    def __init__(self, label: str, full_range: float, unit: str = ""):
        super().__init__()
        self._label      = label
        self._full_range = max(float(full_range), 1e-9)
        self._unit       = unit
        self._value      = 0.0
        self._setpoint   = 0.0
        self.setFixedHeight(18)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, value: float, setpoint: float = 0.0) -> None:
        self._value, self._setpoint = value, setpoint
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())
        lw = 32.0; pad = 3.0
        bx = lw + pad; bw = w - bx - pad
        by = (h - 7.0) / 2.0; bh = 7.0
        cx = bx + bw / 2.0

        # Background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(BORDER)))
        p.drawRoundedRect(QRectF(bx, by, bw, bh), 2.0, 2.0)

        # Centre tick
        pen = QPen(QColor(DIM))
        pen.setWidth(1)
        p.setPen(pen)
        p.drawLine(QPointF(cx, by), QPointF(cx, by + bh))

        # Deviation fill
        dev  = self._value - self._setpoint
        norm = max(-1.0, min(1.0, dev / self._full_range))
        fill = abs(norm) * bw / 2.0
        fx   = cx if norm >= 0.0 else cx - fill
        an   = abs(norm)
        col  = RED if an > 0.7 else (ORANGE if an > 0.3 else GREEN)
        if fill > 0.5:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(col)))
            p.drawRoundedRect(QRectF(fx, by, fill, bh), 2.0, 2.0)

        # Label
        p.setPen(QPen(QColor(DIM)))
        p.setFont(QFont("Consolas", 8))
        p.drawText(QRectF(0.0, 0.0, lw - 1.0, h),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   self._label)

        # Value text
        p.setPen(QPen(QColor(TEXT)))
        p.setFont(QFont("Consolas", 8))
        p.drawText(QRectF(bx, 0.0, bw - 2.0, h),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"{self._value:+.2f}{self._unit}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab
# ─────────────────────────────────────────────────────────────────────────────

class RobotVisualizerTab(QWidget):

    def __init__(self):
        super().__init__()

        # ── GL view ──────────────────────────────────────────────────────────
        self._gl = gl.GLViewWidget()
        self._gl.opts["distance"]  = 1.5
        self._gl.opts["elevation"] = 18
        self._gl.opts["azimuth"]   = -55
        self._gl.setBackgroundColor(pg.mkColor(BG))
        self._gl.setMinimumSize(300, 260)

        grid = gl.GLGridItem()
        grid.setSize(1.5, 1.5, 1)
        grid.setSpacing(0.1, 0.1, 0.1)
        grid.setColor((255, 255, 255, 20))
        grid.translate(0.0, 0.0, -0.35)
        self._gl.addItem(grid)

        for pts, col in [
            (np.array([[0,0,0],[0.25,0,0]]), (1.0,0.2,0.2,0.25)),
            (np.array([[0,0,0],[0,0.25,0]]), (0.2,1.0,0.2,0.25)),
            (np.array([[0,0,0],[0,0,0.25]]), (0.2,0.5,1.0,0.25)),
        ]:
            self._gl.addItem(gl.GLLinePlotItem(
                pos=pts.astype(np.float32), color=col, width=1.5, antialias=True))

        self._box_mesh = _mesh_item(
            self._gl,
            gl.MeshData(vertexes=_BOX_VERTS_BASE.copy(), faces=_BOX_FACES),
            (0.65, 0.65, 0.72, 0.40),
        )

        self._face_outline = gl.GLLinePlotItem(
            pos=_FACE_OUTLINE_BASE, color=_FACE_COLOR, width=2.0, antialias=True)
        self._face_eye_l = gl.GLLinePlotItem(
            pos=_FACE_EYE_L_BASE, color=_FACE_COLOR, width=2.0, antialias=True)
        self._face_eye_r = gl.GLLinePlotItem(
            pos=_FACE_EYE_R_BASE, color=_FACE_COLOR, width=2.0, antialias=True)
        self._face_smile = gl.GLLinePlotItem(
            pos=_FACE_SMILE_BASE, color=_FACE_COLOR, width=2.5, antialias=True)
        for item in (self._face_outline, self._face_eye_l, self._face_eye_r, self._face_smile):
            self._gl.addItem(item)

        self._fwd_arrow = gl.GLLinePlotItem(
            pos=_FWD_ARROW_BASE, color=_ARROW_COLOR, width=2.5, antialias=True, mode='lines')
        self._gl.addItem(self._fwd_arrow)

        self._imu_box = _mesh_item(
            self._gl,
            gl.MeshData(vertexes=_IMU_VERTS_BASE.copy(), faces=_BOX_FACES),
            (0.20, 0.80, 0.90, 0.92),
        )
        self._fr_imu = _make_frame(self._gl, _FR_IMU)

        R0 = np.eye(3)
        self._hip_motor_L = _mesh_item(
            self._gl,
            _cylinder_mesh(R0 @ _HIP_L_P1, R0 @ _HIP_L_P2, _HIP_MOTOR_R, n=16),
            (0.20, 0.20, 0.24, 0.90),
        )
        self._hip_motor_R = _mesh_item(
            self._gl,
            _cylinder_mesh(R0 @ _HIP_R_P1, R0 @ _HIP_R_P2, _HIP_MOTOR_R, n=16),
            (0.20, 0.20, 0.24, 0.90),
        )

        nom = _solve_ik(_Q_NOM)
        self._leg_L = self._build_leg(+1, nom, R0)
        self._leg_R = self._build_leg(-1, nom, R0)

        self._fr_body    = _make_frame(self._gl, _FR_BODY)
        self._fr_hip_L   = _make_frame(self._gl, _FR_JOINT)
        self._fr_hip_R   = _make_frame(self._gl, _FR_JOINT)
        self._fr_wheel_L = _make_frame(self._gl, _FR_JOINT)
        self._fr_wheel_R = _make_frame(self._gl, _FR_JOINT)

        self._tof_dist_mm: list[int] = [_TOF_NO_DATA] * 4
        self._tof_squares: list[gl.GLLinePlotItem] = []
        self._tof_beams:   list[gl.GLLinePlotItem] = []

        self._spokes_L = self._build_spokes()
        self._spokes_R = self._build_spokes()

        self._redraw(_Q_NOM, _Q_NOM, 0.0, 0.0, 0.0)

        for i in range(len(_TOF_DEFS)):
            sq = gl.GLLinePlotItem(
                pos=np.zeros((5, 3), dtype=np.float32),
                color=_TOF_SQ_COLOR, width=2.0, antialias=True,
            )
            self._gl.addItem(sq)
            self._tof_squares.append(sq)
            beam = gl.GLLinePlotItem(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=_TOF_BEAM_COLORS[i], width=2.5, antialias=True,
            )
            self._gl.addItem(beam)
            self._tof_beams.append(beam)
        self._update_tof_items(np.eye(3))

        # ── 3-column cockpit layout ───────────────────────────────────────────
        self._enable_leds: dict[int, LedIndicator] = {}
        self._params_requested = False

        left  = self._build_left_panel()
        right = self._build_right_panel()

        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        root.addWidget(left,      0)
        root.addWidget(self._gl,  1)
        root.addWidget(right,     0)

        TelemetryBus.instance().packet.connect(self._on_packet)

    # ── Left panel ───────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(212)
        w.setStyleSheet(f"QWidget {{ background: transparent; }}")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self._build_orientation_group())
        lay.addWidget(self._build_tof_group())
        lay.addWidget(self._build_radio_group())
        lay.addStretch()
        return w

    def _build_orientation_group(self) -> QGroupBox:
        gb = _make_group("ORIENTATION")
        lay = QVBoxLayout(gb)
        lay.setContentsMargins(4, 6, 4, 4)
        lay.setSpacing(2)

        # Three arc gauges side by side: pitch (with setpoint), roll, yaw
        arc_row = QHBoxLayout()
        arc_row.setSpacing(2)
        self._pitch_arc = AngleArcWidget("Pitch",  45.0, "°", show_setpoint=True, max_rate=5.0)
        self._roll_arc  = AngleArcWidget("Roll",   30.0, "°", show_setpoint=False, max_rate=3.0)
        self._yaw_arc   = AngleArcWidget("Yaw",   180.0, "°", show_setpoint=False, max_rate=3.0)
        for arc in (self._pitch_arc, self._roll_arc, self._yaw_arc):
            arc_row.addWidget(arc)
        lay.addLayout(arc_row)

        # IMU health row
        health_row = QHBoxLayout()
        health_row.setSpacing(4)
        self._imu_led = LedIndicator(9)
        self._imu_loss_lbl = _dim_label("IMU —")
        health_row.addWidget(self._imu_led)
        health_row.addWidget(self._imu_loss_lbl)
        health_row.addStretch()
        lay.addLayout(health_row)

        return gb

    def _build_tof_group(self) -> QGroupBox:
        gb = _make_group("TOF RANGES")
        lay = QVBoxLayout(gb)
        lay.setContentsMargins(4, 6, 4, 4)
        lay.setSpacing(3)

        self._tof_banners: list[BannerBarWidget] = []
        for name in _TOF_LABEL_NAMES:
            bar = BannerBarWidget(name)
            self._tof_banners.append(bar)
            lay.addWidget(bar)

        return gb

    def _build_radio_group(self) -> QGroupBox:
        gb = _make_group("RADIO / SPEED")
        lay = QVBoxLayout(gb)
        lay.setContentsMargins(4, 6, 4, 4)
        lay.setSpacing(3)

        # RC link + speed profile
        top_row = QHBoxLayout()
        top_row.setSpacing(4)
        self._rc_led  = LedIndicator(9)
        self._rc_lbl  = _dim_label("LOST", size=9)
        top_row.addWidget(self._rc_led)
        top_row.addWidget(self._rc_lbl)
        top_row.addStretch()

        # Profile squares P1 / P2 / P3
        self._profile_lbls: list[QLabel] = []
        for i, name in enumerate(("P1", "P2", "P3")):
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedSize(22, 16)
            lbl.setStyleSheet(
                f"color: {DIM}; font-size: 9px; font-weight: bold;"
                f" background: {BORDER}; border-radius: 2px;"
            )
            self._profile_lbls.append(lbl)
            top_row.addWidget(lbl)
        lay.addLayout(top_row)

        # Velocity deviation bar (actual vs setpoint)
        self._vel_dev_bar = DeviationBarWidget("V", full_range=1.5, unit=" m/s")
        self._yaw_dev_bar = DeviationBarWidget("ω", full_range=2.0, unit=" r/s")
        lay.addWidget(self._vel_dev_bar)
        lay.addWidget(self._yaw_dev_bar)

        return gb

    # ── Right panel ──────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(238)
        w.setStyleSheet(f"QWidget {{ background: transparent; }}")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self._build_hip_group())
        lay.addWidget(self._build_wheel_group())
        lay.addWidget(self._build_controller_group())
        lay.addStretch()
        return w

    def _enable_indicator(self, param_id: int) -> LedIndicator:
        """Read-only LED reflecting a firmware presence param (hip_l/r_enable,
        wheel_l/r_enable) — green when the motor is enabled, gray when disabled.
        Toggle the underlying param from the Parameters tab, not from here."""
        led = LedIndicator(7)
        led.setToolTip(f"Motor enabled (param 0x{param_id:04X}) — set via Parameters tab")
        self._enable_leds[param_id] = led
        return led

    def _presence_row(self, param_ids: tuple[int, int]) -> QHBoxLayout:
        """Standalone 'motor present' row — LED + text label per side, kept off
        the crowded health-LED row so it stays legible."""
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(_dim_label("present:", 9))
        for side, pid in zip(("L", "R"), param_ids):
            led = self._enable_indicator(pid)
            sub = QHBoxLayout()
            sub.setSpacing(3)
            sub.addWidget(led)
            sub.addWidget(_dim_label(side, 9))
            row.addLayout(sub)
            row.addSpacing(10)
        row.addStretch()
        return row

    def _build_hip_group(self) -> QGroupBox:
        gb = _make_group("HIP MOTORS")
        lay = QVBoxLayout(gb)
        lay.setContentsMargins(4, 6, 4, 4)
        lay.setSpacing(3)

        # Two arc gauges side by side
        arc_row = QHBoxLayout()
        arc_row.setSpacing(4)
        self._hip_arcs: list[AngleArcWidget] = []
        for side in ("HIP L", "HIP R"):
            arc = AngleArcWidget(side, max_val=143.0, unit="°",
                                 show_setpoint=True, max_rate=200.0)
            self._hip_arcs.append(arc)
            arc_row.addWidget(arc)
        lay.addLayout(arc_row)

        # Per-side info: health LED + extension
        info_row = QHBoxLayout()
        info_row.setSpacing(4)
        self._hip_leds: list[LedIndicator] = []
        self._hip_ext_lbls: list[QLabel] = []
        for _side in ("L", "R"):
            led = LedIndicator(8)
            self._hip_leds.append(led)
            ext = _mono_label("ext: —", size=9)
            self._hip_ext_lbls.append(ext)
            sub = QHBoxLayout()
            sub.setSpacing(2)
            sub.addWidget(led)
            sub.addWidget(ext)
            sub.addStretch()
            info_row.addLayout(sub)
        lay.addLayout(info_row)

        # Motor-present row (hip_l/r_enable) — separate line, easier to read
        lay.addLayout(self._presence_row((_PARAM_HIP_L_ENABLE, _PARAM_HIP_R_ENABLE)))

        # Current bars
        curr_row = QHBoxLayout()
        curr_row.setSpacing(4)
        self._hip_curr_bars: list[FillBarWidget] = []
        for _side in ("L", "R"):
            bar = FillBarWidget("I", max_val=8.0, unit="A")
            self._hip_curr_bars.append(bar)
            curr_row.addWidget(bar)
        lay.addLayout(curr_row)

        return gb

    def _build_wheel_group(self) -> QGroupBox:
        gb = _make_group("WHEELS")
        lay = QVBoxLayout(gb)
        lay.setContentsMargins(4, 6, 4, 4)
        lay.setSpacing(3)

        # Column headers
        hdr = QHBoxLayout()
        hdr.addWidget(_dim_label("LEFT", 8))
        hdr.addStretch()
        hdr.addWidget(_dim_label("RIGHT", 8))
        lay.addLayout(hdr)

        # Speed deviation bars
        self._whl_vel_bars: list[DeviationBarWidget] = []
        for label in ("vL", "vR"):
            bar = DeviationBarWidget(label, full_range=8.0, unit=" t/s")
            self._whl_vel_bars.append(bar)
            lay.addWidget(bar)

        # Torque fill bars
        self._whl_tau_bars: list[FillBarWidget] = []
        for label in ("τL", "τR"):
            bar = FillBarWidget(label, max_val=2.0, unit=" Nm")
            self._whl_tau_bars.append(bar)
            lay.addWidget(bar)

        # ODrive state row
        state_row = QHBoxLayout()
        state_row.setSpacing(4)
        self._whl_state_leds: list[LedIndicator] = []
        self._whl_state_lbls: list[QLabel] = []
        for _side in ("L", "R"):
            led = LedIndicator(8)
            lbl = _dim_label("IDLE", 8)
            self._whl_state_leds.append(led)
            self._whl_state_lbls.append(lbl)
            sub = QHBoxLayout()
            sub.setSpacing(2)
            sub.addWidget(led)
            sub.addWidget(lbl)
            sub.addStretch()
            state_row.addLayout(sub)
        self._whl_mode_lbl = _dim_label("—", 8)
        state_row.addWidget(self._whl_mode_lbl)
        lay.addLayout(state_row)

        # Motor-present row (wheel_l/r_enable) — separate line, easier to read
        lay.addLayout(self._presence_row((_PARAM_WHEEL_L_ENABLE, _PARAM_WHEEL_R_ENABLE)))

        return gb

    def _build_controller_group(self) -> QGroupBox:
        gb = _make_group("CONTROLLERS")
        lay = QVBoxLayout(gb)
        lay.setContentsMargins(4, 6, 4, 4)
        lay.setSpacing(3)

        # LED indicator row: LQR, VelPI, YawPI, FF1, FF2, Jump
        self._ctrl_leds: dict[str, LedIndicator] = {}
        self._ctrl_sat_lbl: dict[str, QLabel] = {}

        led_row1 = QHBoxLayout()
        led_row1.setSpacing(2)
        for name in ("LQR", "VelPI", "YawPI"):
            led = LedIndicator(9)
            lbl = _dim_label(name, 9)
            self._ctrl_leds[name] = led
            sub = QHBoxLayout(); sub.setSpacing(2)
            sub.addWidget(led); sub.addWidget(lbl)
            led_row1.addLayout(sub)
            led_row1.addSpacing(4)
        led_row1.addStretch()
        lay.addLayout(led_row1)

        led_row2 = QHBoxLayout()
        led_row2.setSpacing(2)
        for name in ("FF1", "FF2", "Jump"):
            led = LedIndicator(9)
            lbl = _dim_label(name, 9)
            self._ctrl_leds[name] = led
            sub = QHBoxLayout(); sub.setSpacing(2)
            sub.addWidget(led); sub.addWidget(lbl)
            led_row2.addLayout(sub)
            led_row2.addSpacing(4)
        self._jump_state_lbl = _dim_label("", 8)
        led_row2.addWidget(self._jump_state_lbl)
        led_row2.addStretch()
        lay.addLayout(led_row2)

        # Robot state label
        self._state_lbl = _mono_label("—", size=10)
        lay.addWidget(self._state_lbl)

        # ── Balance section ────────────────────────────────────────────────
        lay.addWidget(self._section_sep("BALANCE"))
        self._bal_err_bar = DeviationBarWidget("err", full_range=0.25, unit=" r")
        self._bal_tau_lbl = _mono_label("τ: — Nm", size=9)
        lay.addWidget(self._bal_err_bar)
        lay.addWidget(self._bal_tau_lbl)

        # ── Velocity section ───────────────────────────────────────────────
        lay.addWidget(self._section_sep("VELOCITY"))
        self._v_err_bar  = DeviationBarWidget("err", full_range=1.5, unit=" m/s")
        self._lean_lbl   = _mono_label("lean: —", size=9)
        lay.addWidget(self._v_err_bar)
        lay.addWidget(self._lean_lbl)

        # ── Yaw section ────────────────────────────────────────────────────
        lay.addWidget(self._section_sep("YAW"))
        self._yaw_err_bar = DeviationBarWidget("err", full_range=1.5, unit=" r/s")
        self._yaw_tau_lbl = _mono_label("τ: — Nm", size=9)
        lay.addWidget(self._yaw_err_bar)
        lay.addWidget(self._yaw_tau_lbl)

        return gb

    @staticmethod
    def _section_sep(title: str) -> QLabel:
        lbl = QLabel(f"— {title} ")
        lbl.setStyleSheet(
            f"color: {DIM}; font-size: 8px; font-weight: bold;"
            f" border-bottom: 1px solid {BORDER}; margin-top: 2px;"
        )
        return lbl

    # ── GL builder helpers (unchanged) ────────────────────────────────────────

    def _build_leg(self, y_sign: int, ik: dict | None, R: np.ndarray) -> dict:
        y = y_sign * _LEG_Y
        if ik is None:
            ik = {'A': (0.0, _A_Z), 'C': (0.0, _A_Z - 0.01),
                  'E': (0.01, _A_Z + _L_STUB), 'F': (_F_X, _F_Z),
                  'W': (0.0, _A_Z - _L_TIBIA)}

        def _pw(xz): return R @ np.array([xz[0], y, xz[1]])
        def _pw3(x, yy, z): return R @ np.array([x, yy, z])

        Wx, Wz = ik['W']
        return dict(
            y       = y,
            femur   = _mesh_item(self._gl,
                                 _cylinder_mesh(_pw(ik['A']), _pw(ik['C']), _R_FEMUR),
                                 (1.00, 0.55, 0.00, 0.85)),
            tibia   = _mesh_item(self._gl,
                                 _cylinder_mesh(_pw(ik['C']), _pw(ik['W']), _R_TIBIA),
                                 (0.30, 0.80, 1.00, 0.85)),
            stub    = _mesh_item(self._gl,
                                 _cylinder_mesh(_pw(ik['C']), _pw(ik['E']), _R_STUB),
                                 (0.40, 1.00, 0.40, 0.85)),
            coupler = _mesh_item(self._gl,
                                 _cylinder_mesh(_pw(ik['F']), _pw(ik['E']), _R_COUPLER),
                                 (0.90, 0.30, 0.90, 0.85)),
            wheel   = _mesh_item(self._gl,
                                 _cylinder_mesh(_pw3(Wx, y - _WHEEL_T, Wz),
                                                _pw3(Wx, y + _WHEEL_T, Wz),
                                                _WHEEL_R, n=28),
                                 (0.45, 0.45, 0.45, 0.65)),
            wheel_motor = _mesh_item(self._gl,
                                     _cylinder_mesh(_pw3(Wx, y - _WHEEL_MOTOR_L/2, Wz),
                                                    _pw3(Wx, y + _WHEEL_MOTOR_L/2, Wz),
                                                    _WHEEL_MOTOR_R, n=16),
                                     (0.18, 0.18, 0.20, 0.90)),
        )

    def _build_spokes(self) -> list:
        items = []
        for _ in range(_N_SPOKES):
            item = gl.GLLinePlotItem(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(0.80, 0.80, 0.85, 0.90), width=2.5, antialias=True,
            )
            self._gl.addItem(item)
            items.append(item)
        return items

    # ── Core GL update (unchanged) ────────────────────────────────────────────

    def _redraw(self, q_l: float, q_r: float,
                pitch: float, roll: float, yaw: float,
                whl_angle_l: float = 0.0, whl_angle_r: float = 0.0) -> None:
        R = _rot3(pitch, roll, yaw)

        rot_verts = (_BOX_VERTS_BASE @ R.T).astype(np.float32)
        self._box_mesh.setMeshData(
            meshdata=gl.MeshData(vertexes=rot_verts, faces=_BOX_FACES))

        self._face_outline.setData(pos=(_FACE_OUTLINE_BASE @ R.T).astype(np.float32))
        self._face_eye_l.setData(pos=(_FACE_EYE_L_BASE @ R.T).astype(np.float32))
        self._face_eye_r.setData(pos=(_FACE_EYE_R_BASE @ R.T).astype(np.float32))
        self._face_smile.setData(pos=(_FACE_SMILE_BASE @ R.T).astype(np.float32))
        self._fwd_arrow.setData(pos=(_FWD_ARROW_BASE @ R.T).astype(np.float32))

        self._hip_motor_L.setMeshData(
            meshdata=_cylinder_mesh(R @ _HIP_L_P1, R @ _HIP_L_P2, _HIP_MOTOR_R, n=16))
        self._hip_motor_R.setMeshData(
            meshdata=_cylinder_mesh(R @ _HIP_R_P1, R @ _HIP_R_P2, _HIP_MOTOR_R, n=16))

        ik_l = _solve_ik(q_l)       or _IK_L_FALLBACK
        ik_r = _solve_ik_right(q_r) or _IK_R_FALLBACK
        self._update_leg(self._leg_L, ik_l, R)
        self._update_leg(self._leg_R, ik_r, R)

        if ik_l:
            self._update_wheel_spokes(self._spokes_L, ik_l['W'], +_LEG_Y, R, whl_angle_l)
        if ik_r:
            self._update_wheel_spokes(self._spokes_R, ik_r['W'], -_LEG_Y, R, whl_angle_r)

        _set_frame(self._fr_body, np.zeros(3), R, _FR_BODY)

        imu_verts = (_IMU_VERTS_BASE @ R.T).astype(np.float32)
        self._imu_box.setMeshData(meshdata=gl.MeshData(vertexes=imu_verts, faces=_BOX_FACES))
        _set_frame(self._fr_imu, R @ _IMU_POS, R @ _R_IMU_MOUNT, _FR_IMU)

        _set_frame(self._fr_hip_L,
                   R @ np.array([0.0, +_LEG_Y, _A_Z]),
                   R @ _ry(q_l), _FR_JOINT, _R_MOUNT_LEFT)
        _set_frame(self._fr_hip_R,
                   R @ np.array([0.0, -_LEG_Y, _A_Z]),
                   R @ _ry(q_r), _FR_JOINT, _R_MOUNT_RIGHT)

        if ik_l:
            _set_frame(self._fr_wheel_L,
                       R @ np.array([ik_l['W'][0], +_LEG_Y, ik_l['W'][1]]),
                       R @ _ry(ik_l['alpha'] + whl_angle_l), _FR_JOINT, _R_MOUNT_LEFT)
        if ik_r:
            _set_frame(self._fr_wheel_R,
                       R @ np.array([ik_r['W'][0], -_LEG_Y, ik_r['W'][1]]),
                       R @ _ry(ik_r['alpha'] + whl_angle_r), _FR_JOINT, _R_MOUNT_RIGHT)

        self._update_tof_items(R)

    def _update_leg(self, leg: dict, ik: dict | None, R: np.ndarray) -> None:
        if ik is None:
            return
        y = leg['y']

        def _pw(xz): return R @ np.array([xz[0], y, xz[1]])
        def _pw3(x, yy, z): return R @ np.array([x, yy, z])

        Wx, Wz = ik['W']
        leg['femur'].setMeshData(
            meshdata=_cylinder_mesh(_pw(ik['A']), _pw(ik['C']), _R_FEMUR))
        leg['tibia'].setMeshData(
            meshdata=_cylinder_mesh(_pw(ik['C']), _pw(ik['W']), _R_TIBIA))
        leg['stub'].setMeshData(
            meshdata=_cylinder_mesh(_pw(ik['C']), _pw(ik['E']), _R_STUB))
        leg['coupler'].setMeshData(
            meshdata=_cylinder_mesh(_pw(ik['F']), _pw(ik['E']), _R_COUPLER))
        leg['wheel'].setMeshData(
            meshdata=_cylinder_mesh(_pw3(Wx, y - _WHEEL_T, Wz),
                                    _pw3(Wx, y + _WHEEL_T, Wz),
                                    _WHEEL_R, n=28))
        leg['wheel_motor'].setMeshData(
            meshdata=_cylinder_mesh(_pw3(Wx, y - _WHEEL_MOTOR_L/2, Wz),
                                    _pw3(Wx, y + _WHEEL_MOTOR_L/2, Wz),
                                    _WHEEL_MOTOR_R, n=16))

    def _update_wheel_spokes(self, spokes: list, W_xz: tuple, y: float,
                              R: np.ndarray, angle: float) -> None:
        Wx, Wz = W_xz
        center = R @ np.array([Wx, y, Wz])
        x_ax = R @ np.array([1.0, 0.0, 0.0])
        z_ax = R @ np.array([0.0, 0.0, 1.0])
        step = 2.0 * math.pi / len(spokes)
        for i, spoke in enumerate(spokes):
            theta = -angle + i * step
            radial = _WHEEL_R * _SPOKE_FADE * (math.cos(theta) * x_ax + math.sin(theta) * z_ax)
            spoke.setData(pos=np.array([
                (center - radial).astype(np.float32),
                (center + radial).astype(np.float32),
            ]))

    def _update_tof_items(self, R: np.ndarray) -> None:
        if not self._tof_squares:
            return
        for i, (p_body, d_body) in enumerate(_TOF_DEFS):
            p_world = R @ p_body
            n_world = R @ d_body

            ref = np.array([0., 0., 1.]) if abs(n_world[2]) < 0.9 else np.array([0., 1., 0.])
            t1 = np.cross(n_world, ref); t1 /= np.linalg.norm(t1)
            t2 = np.cross(n_world, t1);  t2 /= np.linalg.norm(t2)
            s  = _TOF_HALF_SQ
            sq_pts = np.array([
                p_world + s*t1 + s*t2,
                p_world - s*t1 + s*t2,
                p_world - s*t1 - s*t2,
                p_world + s*t1 - s*t2,
                p_world + s*t1 + s*t2,
            ], dtype=np.float32)
            self._tof_squares[i].setData(pos=sq_pts)

            d_mm    = self._tof_dist_mm[i]
            no_data = (d_mm == _TOF_NO_DATA)
            if no_data or d_mm >= _TOF_MAX_MM:
                alpha  = 0.0
                dist_m = 0.5
            else:
                alpha  = 1.0 - d_mm / _TOF_MAX_MM
                dist_m = d_mm * 1e-3
            if not no_data and d_mm < _TOF_WARN_MM:
                beam_color = (1.0, 0.25, 0.25, 1.0)
            else:
                r, g, b, _ = _TOF_BEAM_COLORS[i]
                beam_color = (r, g, b, alpha)
            end = (p_world + dist_m * n_world).astype(np.float32)
            self._tof_beams[i].setData(
                pos=np.array([p_world, end], dtype=np.float32),
                color=beam_color,
            )

    # ── Telemetry handler ─────────────────────────────────────────────────────

    def _on_packet(self, info: dict) -> None:
        ptype = info.get("ptype")

        if ptype == 0x01 and not self._params_requested:
            send_param_get_all()
            self._params_requested = True

        if ptype == 0x06:
            pid = info.get("param_id")
            val = info.get("param_value")
            led = self._enable_leds.get(pid)
            if led is not None and val is not None:
                led.set_state("green" if val >= 0.5 else "gray")
            return

        if ptype != 0x01:
            return

        # ── Raw fields ───────────────────────────────────────────────────────
        q_l   = info.get("hip_l_pos_rad",    _Q_NOM)
        q_r   = info.get("hip_r_pos_rad",    _Q_NOM)
        pitch = info.get("pitch_rad",          0.0)
        roll  = info.get("roll_rad",           0.0)
        yaw   = info.get("yaw_rad",            0.0)
        pitch_rate = info.get("pitch_rate_rads", 0.0)
        roll_rate  = info.get("roll_rate_rads",  0.0)
        yaw_rate   = info.get("yaw_rate_rads",   0.0)

        hip_l_cmd = info.get("hip_l_cmd_pos_rad", q_l)
        hip_r_cmd = info.get("hip_r_cmd_pos_rad", q_r)
        hip_l_cur = info.get("hip_l_current_a",   0.0)
        hip_r_cur = info.get("hip_r_current_a",   0.0)

        wm_l_vel = info.get("wm_l_vel_turns_s", 0.0)
        wm_r_vel = info.get("wm_r_vel_turns_s", 0.0)
        wm_l_state = info.get("wm_l_state", 0)
        wm_r_state = info.get("wm_r_state", 0)
        wm_l_error = info.get("wm_l_error", 0)
        wm_r_error = info.get("wm_r_error", 0)
        wm_mode    = info.get("wm_mode",    0)
        tau_l = info.get("whl_tau_l", 0.0)
        tau_r = info.get("whl_tau_r", 0.0)

        theta_ref    = info.get("theta_ref",      0.0)
        v_ref        = info.get("v_ref",          0.0)
        omega_cmd    = info.get("omega_cmd_rds",  0.0)
        tau_sym      = info.get("tau_sym",        0.0)
        tau_yaw      = info.get("tau_yaw",        0.0)
        ff1_out      = info.get("ff1_out",        0.0)
        ff2_out      = info.get("ff2_out",        0.0)
        vel_avg      = info.get("wheel_vel_avg",  0.0)
        health_flags = info.get("health_flags",   0)
        imu_loss     = info.get("imu_packet_loss_pct", 0)
        jump_state   = info.get("jump_state",     0)
        active_profile = info.get("active_profile", None)
        ibus_alive   = info.get("ibus_alive",     0)
        state        = info.get("state_name",     "—")

        tof = info.get("tof_dist_mm")
        if isinstance(tof, (list, tuple)) and len(tof) == 4:
            self._tof_dist_mm = list(tof)

        whl_angle_l = info.get("wm_l_pos_turns", 0.0) * 2.0 * math.pi
        whl_angle_r = info.get("wm_r_pos_turns", 0.0) * 2.0 * math.pi

        # ── 3-D scene ────────────────────────────────────────────────────────
        self._redraw(q_l, q_r, pitch, roll, yaw, whl_angle_l, whl_angle_r)

        # ── IK for leg extension (health-indicator text; matches _redraw's
        # sign convention — right hip uses q_r directly, same sign as left) ──
        ik_l = _solve_ik(q_l)
        ik_r = _solve_ik_right(q_r)

        # ── Left panel: Orientation ──────────────────────────────────────────
        pitch_deg = math.degrees(pitch)
        roll_deg  = math.degrees(roll)
        yaw_deg   = math.degrees(yaw)
        # Wrap yaw to ±180
        yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0

        self._pitch_arc.set_value(pitch_deg,
                                  setpoint=math.degrees(theta_ref),
                                  rate=pitch_rate)
        self._roll_arc.set_value(roll_deg,  rate=roll_rate)
        self._yaw_arc.set_value(yaw_deg,    rate=yaw_rate)

        imu_ok = bool(health_flags & (1 << _HF_IMU_NOMINAL))
        self._imu_led.set_state("green" if imu_ok else "red")
        loss_color = RED if imu_loss > 10 else (YELLOW if imu_loss > 3 else DIM)
        self._imu_loss_lbl.setStyleSheet(f"color: {loss_color}; font-size: 9px;")
        self._imu_loss_lbl.setText(f"IMU {imu_loss}% loss")

        # ── Left panel: ToF ──────────────────────────────────────────────────
        for i, bar in enumerate(self._tof_banners):
            bar.set_dist(self._tof_dist_mm[i])

        # ── Left panel: Radio/Speed ───────────────────────────────────────────
        rc_ok = bool(ibus_alive)
        self._rc_led.set_state("green" if rc_ok else "red")
        self._rc_lbl.setStyleSheet(
            f"color: {'#00e676' if rc_ok else '#f44336'}; font-size: 9px;")
        self._rc_lbl.setText("LINK" if rc_ok else "LOST")

        if active_profile is not None:
            for i, lbl in enumerate(self._profile_lbls):
                active = (i == active_profile)
                lbl.setStyleSheet(
                    f"color: {'#fff' if active else DIM}; font-size: 9px;"
                    f" font-weight: bold;"
                    f" background: {BLUE if active else BORDER}; border-radius: 2px;"
                )

        self._vel_dev_bar.set_value(vel_avg, setpoint=v_ref)
        self._yaw_dev_bar.set_value(yaw_rate, setpoint=omega_cmd)

        # ── Right panel: Hips ─────────────────────────────────────────────────
        hip_l_deg = math.degrees(q_l)
        hip_r_deg = math.degrees(q_r)
        hip_l_cmd_deg = math.degrees(hip_l_cmd)
        hip_r_cmd_deg = math.degrees(hip_r_cmd)

        self._hip_arcs[0].set_value(hip_l_deg, setpoint=hip_l_cmd_deg,
                                    rate=math.degrees(info.get("hip_l_vel_rads", 0.0)))
        self._hip_arcs[1].set_value(hip_r_deg, setpoint=hip_r_cmd_deg,
                                    rate=math.degrees(info.get("hip_r_vel_rads", 0.0)))

        hip_ok = [bool(health_flags & (1 << _HF_HIP_L_OK)),
                  bool(health_flags & (1 << _HF_HIP_R_OK))]
        for i, led in enumerate(self._hip_leds):
            led.set_state("green" if hip_ok[i] else "red")

        ext_vals = [
            (f"ext {ik_l['W'][1]*1000:+.0f}mm" if ik_l else "singularity"),
            (f"ext {ik_r['W'][1]*1000:+.0f}mm" if ik_r else "singularity"),
        ]
        for i, lbl in enumerate(self._hip_ext_lbls):
            lbl.setText(ext_vals[i])

        self._hip_curr_bars[0].set_value(abs(hip_l_cur))
        self._hip_curr_bars[1].set_value(abs(hip_r_cur))

        # ── Right panel: Wheels ───────────────────────────────────────────────
        self._whl_vel_bars[0].set_value(wm_l_vel)
        self._whl_vel_bars[1].set_value(wm_r_vel)
        self._whl_tau_bars[0].set_value(abs(tau_l))
        self._whl_tau_bars[1].set_value(abs(tau_r))

        def _odrive_state(state_val: int, error_val: int) -> tuple[str, str]:
            if error_val:
                return "red", "ERR"
            if state_val == _ODRIVE_CLOSED:
                return "green", "CLOSED"
            if state_val == _ODRIVE_IDLE:
                return "gray", "IDLE"
            return "yellow", f"ST{state_val}"

        for i, (sv, ev) in enumerate([(wm_l_state, wm_l_error), (wm_r_state, wm_r_error)]):
            col, txt = _odrive_state(sv, ev)
            self._whl_state_leds[i].set_state(col)
            self._whl_state_lbls[i].setText(txt)

        wm_lim_l = bool(health_flags & (1 << _HF_WM_L_VEL_LIM))
        wm_lim_r = bool(health_flags & (1 << _HF_WM_R_VEL_LIM))
        mode_str = _WM_MODE_LABELS.get(wm_mode, f"M{wm_mode}")
        if wm_lim_l or wm_lim_r:
            mode_str += " LIM"
        self._whl_mode_lbl.setText(mode_str)

        # ── Right panel: Controllers ──────────────────────────────────────────
        lqr_active  = bool(health_flags & (1 << _HF_LQR_ACTIVE))
        vel_pi_sat  = bool(health_flags & (1 << _HF_VEL_PI_SAT))
        yaw_pi_sat  = bool(health_flags & (1 << _HF_YAW_PI_SAT))
        ff1_active  = (abs(ff1_out) > 0.001)
        ff2_active  = (abs(ff2_out) > 0.001)
        jump_active = (jump_state > 0)

        self._ctrl_leds["LQR"].set_state("green" if lqr_active else "gray")
        self._ctrl_leds["VelPI"].set_state("yellow" if vel_pi_sat else
                                           ("green" if lqr_active else "gray"))
        self._ctrl_leds["YawPI"].set_state("yellow" if yaw_pi_sat else
                                           ("green" if lqr_active else "gray"))
        self._ctrl_leds["FF1"].set_state("green" if ff1_active else "gray")
        self._ctrl_leds["FF2"].set_state("green" if ff2_active else "gray")
        self._ctrl_leds["Jump"].set_state("orange" if jump_active else "gray")

        jump_txt = _JUMP_STATE_LABELS.get(jump_state, f"JS{jump_state}") if jump_active else ""
        self._jump_state_lbl.setText(jump_txt)

        s_color = {"RUNNING": GREEN, "ESTOP": RED, "CALIBRATION": BLUE,
                   "STANDBY": YELLOW, "STARTUP": WHITE}.get(state, DIM)
        self._state_lbl.setStyleSheet(
            f"color: {s_color}; font-size: 10px; font-family: Consolas; font-weight: bold;"
        )
        self._state_lbl.setText(state)

        # Balance
        pitch_err = pitch - theta_ref
        self._bal_err_bar.set_value(pitch_err)
        self._bal_tau_lbl.setText(f"τ_sym: {tau_sym:+.3f} Nm")

        # Velocity
        vel_err = vel_avg - v_ref
        self._v_err_bar.set_value(vel_err)
        self._lean_lbl.setText(f"lean θ: {math.degrees(theta_ref):+.1f}°")

        # Yaw
        yaw_err = yaw_rate - omega_cmd
        self._yaw_err_bar.set_value(yaw_err)
        self._yaw_tau_lbl.setText(f"τ_yaw: {tau_yaw:+.3f} Nm")
