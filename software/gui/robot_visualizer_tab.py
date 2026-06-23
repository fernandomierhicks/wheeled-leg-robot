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

# Link tube radii [m] — from COMPONENTS.md Al tube OD
_R_FEMUR   = 0.007   # 14 mm OD tube
_R_TIBIA   = 0.008   # 16 mm OD tube
_R_COUPLER = 0.005   # 10 mm OD tube
_R_STUB    = 0.008   # same as tibia

# Hip motor (CubeMars AK45-10): Φ53×43 mm — COMPONENTS.md
_HIP_MOTOR_R = 0.0265  # 53 mm diameter / 2
_HIP_MOTOR_L = 0.043   # 43 mm length; motor body extends inward from A along ±Y

# Wheel motor (Maytech MTO5065-70-HA-C): Ø50×65 mm — COMPONENTS.md
# The motor IS the wheel hub; the tire mounts around the bell.
_WHEEL_MOTOR_R = 0.025   # 50 mm diameter / 2
_WHEEL_MOTOR_L = 0.065   # 65 mm length
_WHEEL_T       = 0.0325  # tire half-thickness = motor half-length (±32.5 mm)

# Body box half-extents [m]  (approximate chassis)
_BX, _BY, _BZ = 0.10, 0.07, 0.05

# Coordinate-frame arrow lengths [m]
_FR_BODY  = 0.09    # body/IMU frame (slightly larger)
_FR_JOINT = 0.055   # hip & wheel frames

# Motor mount transforms: local +Z is the shaft/rotation axis, which points
# outward from the robot midplane.  Left motors → body +Y; right → body -Y.
#   Left  Rx(-90°): maps local +Z → body +Y
#   Right Rx(+90°): maps local +Z → body -Y
# Body/IMU uses identity — the firmware already remaps the BNO086 gyro x/y
# axes in IMU.cpp (_pitch_rate = gyro.y, _roll_rate = gyro.x) so that IMU
# output arrives pre-expressed in body frame; that swap is the firmware-side
# equivalent of this same mounting-transform concept.
_R_MOUNT_LEFT  = np.array([[1, 0, 0], [0, 0, 1], [0,-1, 0]], dtype=float)  # Rx(-90°)
_R_MOUNT_RIGHT = np.array([[1, 0, 0], [0, 0,-1], [0, 1, 0]], dtype=float)  # Rx(+90°)
_R_MOUNT_IDENT = np.eye(3)

# Hip motor body-frame endpoints (output shaft face at A, other end inward)
# Left:  output at +Y=_LEG_Y, body extends in -Y direction
# Right: output at -Y=_LEG_Y, body extends in +Y direction
_HIP_L_P1 = np.array([0.0,  _LEG_Y,                _A_Z])
_HIP_L_P2 = np.array([0.0,  _LEG_Y - _HIP_MOTOR_L, _A_Z])
_HIP_R_P1 = np.array([0.0, -_LEG_Y,                _A_Z])
_HIP_R_P2 = np.array([0.0, -_LEG_Y + _HIP_MOTOR_L, _A_Z])

# Box base vertices and face indices (half-extents _BX/_BY/_BZ, unrotated)
_BOX_VERTS_BASE = np.array([
    [-_BX,-_BY,-_BZ],[+_BX,-_BY,-_BZ],[+_BX,+_BY,-_BZ],[-_BX,+_BY,-_BZ],
    [-_BX,-_BY,+_BZ],[+_BX,-_BY,+_BZ],[+_BX,+_BY,+_BZ],[-_BX,+_BY,+_BZ],
], dtype=np.float32)

_BOX_FACES = np.array([
    [0,1,2],[0,2,3],  # -Z face
    [4,6,5],[4,7,6],  # +Z face
    [0,5,1],[0,4,5],  # -Y face
    [2,3,7],[2,7,6],  # +Y face
    [0,3,7],[0,7,4],  # -X face
    [1,2,6],[1,6,5],  # +X face
], dtype=np.uint32)

# ── ToF sensor geometry (body-frame, four VL53L1X sensors) ──────────────────
_TOF_DOWN_ANGLE = math.radians(20)          # angled sensors tilt below horizontal
_TOF_HALF_SQ    = 0.0085                    # sensor body square half-size [m]
_TOF_MAX_MM     = 2000                      # beam fades to invisible at this distance [mm]
_TOF_WARN_MM    = 400                       # beam turns red below this distance [mm]
_TOF_NO_DATA    = 0xFFFF

# Each entry: (body_pos [m], beam_unit_vec) — sensors 0,1 on +X face, 2,3 on -X face
# Horiz sensor at Z=+0.015; down sensor directly below at Z=-0.015, same Y
_TOF_DEFS = (
    (np.array([+_BX,  0.0,  0.015]),
     np.array([1.0, 0.0, 0.0])),                                                   # 0 front-horiz
    (np.array([+_BX,  0.0, -0.015]),
     np.array([math.cos(_TOF_DOWN_ANGLE), 0.0, -math.sin(_TOF_DOWN_ANGLE)])),      # 1 front-down
    (np.array([-_BX,  0.0,  0.015]),
     np.array([-1.0, 0.0, 0.0])),                                                  # 2 rear-horiz
    (np.array([-_BX,  0.0, -0.015]),
     np.array([-math.cos(_TOF_DOWN_ANGLE), 0.0, -math.sin(_TOF_DOWN_ANGLE)])),     # 3 rear-down
)
_TOF_BEAM_COLORS = (
    (1.00, 1.00, 0.30, 1.0),   # front-h  yellow
    (1.00, 0.65, 0.10, 1.0),   # front-d  orange
    (0.30, 1.00, 1.00, 1.0),   # rear-h   cyan
    (0.20, 0.70, 1.00, 1.0),   # rear-d   sky-blue
)
_TOF_SQ_COLOR    = (1.00, 0.85, 0.35, 1.0)  # gold outline for sensor body squares
_TOF_LABEL_NAMES = ("F-Horiz", "F-Down", "R-Horiz", "R-Down")

# Minimal valid mesh used as a no-op placeholder (e.g. degenerate cylinder)
_EMPTY_MD = gl.MeshData(
    vertexes=np.zeros((3, 3), dtype=np.float32),
    faces=np.array([[0, 1, 2]], dtype=np.uint32),
)


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
        F=(_F_X, _F_Z), W=(W_x, W_z), alpha=alpha,
    )


def _solve_ik_right(q_r: float) -> dict | None:
    """FK for the right leg. The right motor shaft faces -Y so its angle sign
    is physically reversed relative to the left; negate before solving."""
    return _solve_ik(-q_r)


# ── 3-D helpers ───────────────────────────────────────────────────────────────

def _ry(q: float) -> np.ndarray:
    """Rotation matrix around Y axis by q radians."""
    cq, sq = math.cos(q), math.sin(q)
    return np.array([[cq, 0, sq], [0, 1, 0], [-sq, 0, cq]])


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


# ── Mesh data builder ─────────────────────────────────────────────────────────

def _cylinder_mesh(p1: np.ndarray, p2: np.ndarray,
                   radius: float, n: int = 8) -> gl.MeshData:
    """Open tube cylinder from world-frame p1 to p2 with n-sided cross-section."""
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


def _set_frame(items: list, origin_world: np.ndarray, R_orient: np.ndarray,
               length: float, R_mount: np.ndarray | None = None) -> None:
    """Draw frame arrows starting at world-frame origin, oriented by R_orient @ R_mount.
    Caller is responsible for transforming the origin to world frame (R_body @ p_body)
    so that hip/wheel angle rotations don't accidentally shift the origin position."""
    R_eff = R_orient @ (R_mount if R_mount is not None else _R_MOUNT_IDENT)
    for i, item in enumerate(items):
        axis = np.zeros(3)
        axis[i] = length
        item.setData(pos=np.array([origin_world,
                                   origin_world + R_eff @ axis], dtype=np.float32))


# ── Mesh item factory ─────────────────────────────────────────────────────────

def _mesh_item(view: gl.GLViewWidget, md: gl.MeshData,
               color: tuple) -> gl.GLMeshItem:
    item = gl.GLMeshItem(meshdata=md, smooth=True, drawEdges=False,
                         color=color, glOptions='translucent')
    view.addItem(item)
    return item


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

        # ── Body box mesh ─────────────────────────────────────────────────────
        self._box_mesh = _mesh_item(
            self._gl,
            gl.MeshData(vertexes=_BOX_VERTS_BASE.copy(), faces=_BOX_FACES),
            (0.65, 0.65, 0.72, 0.40),
        )

        # ── Hip motor meshes (body-fixed, AK45-10: Φ53×43 mm) ────────────────
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

        # ── 4-bar leg mesh items ──────────────────────────────────────────────
        nom = _solve_ik(_Q_NOM)
        self._leg_L = self._build_leg(+1, nom, R0)
        self._leg_R = self._build_leg(-1, nom, R0)

        # ── Coordinate frames ─────────────────────────────────────────────────
        # body/IMU at chassis origin
        self._fr_body    = _make_frame(self._gl, _FR_BODY)
        # hip motor shaft (fixed in body)
        self._fr_hip_L   = _make_frame(self._gl, _FR_JOINT)
        self._fr_hip_R   = _make_frame(self._gl, _FR_JOINT)
        # wheel centre (moves with hip angle)
        self._fr_wheel_L = _make_frame(self._gl, _FR_JOINT)
        self._fr_wheel_R = _make_frame(self._gl, _FR_JOINT)

        # ToF state must exist before _redraw (which calls _update_tof_items)
        self._tof_dist_mm: list[int] = [_TOF_NO_DATA] * 4
        self._tof_squares: list[gl.GLLinePlotItem] = []
        self._tof_beams:   list[gl.GLLinePlotItem] = []

        # Initial draw at identity rotation, nominal hip angle
        self._redraw(_Q_NOM, _Q_NOM, 0.0, 0.0, 0.0)

        # ── ToF sensor visualizations ─────────────────────────────────────────
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

    def _build_leg(self, y_sign: int, ik: dict | None,
                   R: np.ndarray) -> dict:
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
            # Tire: wide cylinder along body ±Y at wheel centre W
            wheel   = _mesh_item(self._gl,
                                 _cylinder_mesh(_pw3(Wx, y - _WHEEL_T, Wz),
                                                _pw3(Wx, y + _WHEEL_T, Wz),
                                                _WHEEL_R, n=28),
                                 (0.45, 0.45, 0.45, 0.65)),
            # Motor hub: same axis, smaller radius (MTO5065: Ø50 mm)
            wheel_motor = _mesh_item(self._gl,
                                     _cylinder_mesh(_pw3(Wx, y - _WHEEL_MOTOR_L/2, Wz),
                                                    _pw3(Wx, y + _WHEEL_MOTOR_L/2, Wz),
                                                    _WHEEL_MOTOR_R, n=16),
                                     (0.18, 0.18, 0.20, 0.90)),
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

        tof_hdr = QLabel("ToF Ranges")
        tof_hdr.setStyleSheet(
            f"color: {DIM}; font-size: 10px; font-weight: bold; margin-top: 4px;")
        lay.addWidget(tof_hdr)

        for name in _TOF_LABEL_NAMES:
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
            "• Body/IMU (large, body frame)<br>"
            "• Hip A  L / R (+Z = shaft out)<br>"
            "• Wheel W  L / R (+Z = shaft out)"
        )
        fr_legend.setStyleSheet(f"font-size: 10px; color: {DIM}; line-height: 160%;")
        lay.addWidget(fr_legend)

        return panel

    # ── Core update ───────────────────────────────────────────────────────────

    def _redraw(self, q_l: float, q_r: float,
                pitch: float, roll: float, yaw: float) -> None:
        R = _rot3(pitch, roll, yaw)

        # Body box — rotate base verts and update mesh
        rot_verts = (_BOX_VERTS_BASE @ R.T).astype(np.float32)
        self._box_mesh.setMeshData(
            meshdata=gl.MeshData(vertexes=rot_verts, faces=_BOX_FACES))

        # Hip motors (AK45-10) — body-fixed, rotate with body only
        self._hip_motor_L.setMeshData(
            meshdata=_cylinder_mesh(R @ _HIP_L_P1, R @ _HIP_L_P2, _HIP_MOTOR_R, n=16))
        self._hip_motor_R.setMeshData(
            meshdata=_cylinder_mesh(R @ _HIP_R_P1, R @ _HIP_R_P2, _HIP_MOTOR_R, n=16))

        # Legs
        ik_l = _solve_ik(q_l)
        ik_r = _solve_ik_right(q_r)
        self._update_leg(self._leg_L, ik_l, R)
        self._update_leg(self._leg_R, ik_r, R)

        # Body/IMU frame at chassis origin
        _set_frame(self._fr_body, np.zeros(3), R, _FR_BODY)

        # Hip motor frames — origin is body-frame A rotated by body R only.
        # R_orient = R @ Ry(q) so the axes spin with the output shaft.
        # R_mount orients the shaft indicator along ±Y (blue = shaft out).
        _set_frame(self._fr_hip_L,
                   R @ np.array([0.0, +_LEG_Y, _A_Z]),
                   R @ _ry(q_l), _FR_JOINT, _R_MOUNT_LEFT)
        _set_frame(self._fr_hip_R,
                   R @ np.array([0.0, -_LEG_Y, _A_Z]),
                   R @ _ry(-q_r), _FR_JOINT, _R_MOUNT_RIGHT)

        # Wheel frames — origin is body-frame W rotated by body R only.
        # R_orient = R @ Ry(alpha) so axes follow the tibia angle.
        if ik_l:
            _set_frame(self._fr_wheel_L,
                       R @ np.array([ik_l['W'][0], +_LEG_Y, ik_l['W'][1]]),
                       R @ _ry(ik_l['alpha']), _FR_JOINT, _R_MOUNT_LEFT)
        if ik_r:
            _set_frame(self._fr_wheel_R,
                       R @ np.array([ik_r['W'][0], -_LEG_Y, ik_r['W'][1]]),
                       R @ _ry(ik_r['alpha']), _FR_JOINT, _R_MOUNT_RIGHT)

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
        # Tire: axis along body ±Y, so endpoints differ only in Y
        leg['wheel'].setMeshData(
            meshdata=_cylinder_mesh(_pw3(Wx, y - _WHEEL_T, Wz),
                                    _pw3(Wx, y + _WHEEL_T, Wz),
                                    _WHEEL_R, n=28))
        # Motor hub inside the tire
        leg['wheel_motor'].setMeshData(
            meshdata=_cylinder_mesh(_pw3(Wx, y - _WHEEL_MOTOR_L/2, Wz),
                                    _pw3(Wx, y + _WHEEL_MOTOR_L/2, Wz),
                                    _WHEEL_MOTOR_R, n=16))

    def _update_tof_items(self, R: np.ndarray) -> None:
        if not self._tof_squares:
            return
        for i, (p_body, d_body) in enumerate(_TOF_DEFS):
            p_world = R @ p_body
            n_world = R @ d_body

            # sensor square — outline in the plane perpendicular to the beam
            ref = np.array([0., 0., 1.]) if abs(n_world[2]) < 0.9 else np.array([0., 1., 0.])
            t1 = np.cross(n_world, ref); t1 /= np.linalg.norm(t1)
            t2 = np.cross(n_world, t1);  t2 /= np.linalg.norm(t2)
            s  = _TOF_HALF_SQ
            sq_pts = np.array([
                p_world + s*t1 + s*t2,
                p_world - s*t1 + s*t2,
                p_world - s*t1 - s*t2,
                p_world + s*t1 - s*t2,
                p_world + s*t1 + s*t2,   # close the loop
            ], dtype=np.float32)
            self._tof_squares[i].setData(pos=sq_pts)

            # range beam — alpha fades with distance, red when close
            d_mm    = self._tof_dist_mm[i]
            no_data = (d_mm == _TOF_NO_DATA)
            if no_data or d_mm >= _TOF_MAX_MM:
                alpha  = 0.0
                dist_m = 0.5
            else:
                alpha  = 1.0 - d_mm / _TOF_MAX_MM
                dist_m = d_mm * 1e-3
            if not no_data and d_mm < _TOF_WARN_MM:
                beam_color = (1.0, 0.25, 0.25, 1.0)   # red when close
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
        if info.get("ptype") != 0x01:
            return

        q_l   = info.get("hip_l_pos_rad", _Q_NOM)
        q_r   = info.get("hip_r_pos_rad", _Q_NOM)
        pitch = info.get("pitch_rad",     0.0)
        roll  = info.get("roll_rad",      0.0)
        yaw   = info.get("yaw_rad",       0.0)
        state = info.get("state_name",    "—")

        tof = info.get("tof_dist_mm")
        if isinstance(tof, (list, tuple)) and len(tof) == 4:
            self._tof_dist_mm = list(tof)

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

        for i, lname in enumerate(_TOF_LABEL_NAMES):
            d = self._tof_dist_mm[i]
            self._lbl[lname].setText("—" if d == _TOF_NO_DATA else f"{d} mm")
