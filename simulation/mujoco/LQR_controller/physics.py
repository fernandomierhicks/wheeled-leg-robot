"""physics.py — 4-bar kinematics, equilibrium pitch, and MJCF builder.

Two-leg model: left leg at +Y, right leg at -Y.
"""
import math
import struct
import numpy as np

from sim_config import (
    ROBOT, MOTOR_MASS, LEG_Y, WHEEL_R,
    HIP_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT,
)

_SINGULARITY_LIM = 0.95


# ---------------------------------------------------------------------------
def generate_torus_stl(R_t: float, r_t: float,
                       N_theta: int = 48, N_phi: int = 24) -> bytes:
    """Return binary STL bytes for a torus centred at the origin.

    The torus lies in the X-Z plane (rotation axis = Y).
      x = (R_t + r_t*cos(phi)) * sin(theta)
      y =  r_t * sin(phi)
      z = (R_t + r_t*cos(phi)) * cos(theta)
    """
    def _pt(theta: float, phi: float):
        r = R_t + r_t * math.cos(phi)
        return (r * math.sin(theta), r_t * math.sin(phi), r * math.cos(theta))

    def _cross(a, b):
        return (a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0])

    def _norm3(v):
        m = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) or 1.0
        return (v[0]/m, v[1]/m, v[2]/m)

    tris = []
    for i in range(N_theta):
        for j in range(N_phi):
            t0 = 2 * math.pi * i       / N_theta
            t1 = 2 * math.pi * (i + 1) / N_theta
            p0 = 2 * math.pi * j       / N_phi
            p1 = 2 * math.pi * (j + 1) / N_phi
            v00 = _pt(t0, p0); v10 = _pt(t1, p0)
            v11 = _pt(t1, p1); v01 = _pt(t0, p1)
            for tri in ((v00, v10, v11), (v00, v11, v01)):
                e1 = tuple(tri[1][k] - tri[0][k] for k in range(3))
                e2 = tuple(tri[2][k] - tri[0][k] for k in range(3))
                n  = _norm3(_cross(e1, e2))
                tris.append((n, tri[0], tri[1], tri[2]))

    buf = bytearray(80)                          # 80-byte header
    buf += struct.pack('<I', len(tris))
    for n, v1, v2, v3 in tris:
        buf += struct.pack('<fff', *n)
        buf += struct.pack('<fff', *v1)
        buf += struct.pack('<fff', *v2)
        buf += struct.pack('<fff', *v3)
        buf += struct.pack('<H', 0)              # attribute byte count
    return bytes(buf)


def build_assets() -> dict:
    """Return assets dict for mujoco.MjModel.from_xml_string(..., assets=...)."""
    stl = generate_torus_stl(R_t=0.055, r_t=0.020, N_theta=48, N_phi=24)
    return {"torus_wheel.stl": stl}


# ---------------------------------------------------------------------------
def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
def solve_ik(q_hip: float, p: dict) -> dict | None:
    """4-bar inverse kinematics.  Returns pivot positions + joint angles,
    or None if the linkage is at a singularity."""
    L_f = p['L_femur']; L_s = p['L_stub']; L_t = p['L_tibia']
    Lc  = p['Lc'];      F_X = p['F_X'];    F_Z = p['F_Z']; A_Z = p['A_Z']

    C_x = -L_f * math.cos(q_hip)
    C_z =  A_Z + L_f * math.sin(q_hip)
    dx, dz = C_x - F_X, C_z - F_Z
    R = math.sqrt(dx*dx + dz*dz)
    if R < 1e-9: return None

    K  = (Lc**2 - dx**2 - dz**2 - L_s**2) / (2.0 * L_s)
    kr = abs(K) / R
    if kr >= _SINGULARITY_LIM: return None

    phi   = math.atan2(dz, dx)
    asinv = math.asin(max(-1.0, min(1.0, K / R)))
    a1    = _wrap(asinv - phi)
    a2    = _wrap(math.pi - asinv - phi)
    qk1   = a1 - q_hip;  qk2 = a2 - q_hip
    alpha = a1 if abs(qk1) <= abs(qk2) else a2

    q_knee      = alpha - q_hip
    E_x         = C_x + L_s * math.sin(alpha)
    E_z         = C_z + L_s * math.cos(alpha)
    W_x         = C_x - L_t * math.sin(alpha)
    W_z         = C_z - L_t * math.cos(alpha)
    q_coupler_F = math.atan2(E_z - F_Z, F_X - E_x)

    return dict(
        q_hip=q_hip, q_knee=q_knee, q_coupler_F=q_coupler_F,
        C=(C_x, C_z), E=(E_x, E_z), F=(F_X, F_Z), W=(W_x, W_z),
        W_z=W_z, KR_ratio=kr, alpha=alpha,
    )


# ---------------------------------------------------------------------------
def get_equilibrium_pitch(p: dict, q_hip: float) -> float:
    """Body pitch (rad) that places system CoM directly over wheel centre W.
    Accounts for two symmetric legs (×2 on all leg-mass contributions)."""
    ik = solve_ik(q_hip, p)
    if ik is None: return 0.0

    n = 2  # two legs; Y-symmetry means only X-Z CoM matters
    m_sys = p['m_box'];  mx = 0.0;  mz = 0.0

    # Hip motors (n motors, each at (0, A_Z) in body frame)
    m_sys += n * MOTOR_MASS
    mz    += n * MOTOR_MASS * p['A_Z']

    # Femur midpoint (A→C)
    C_x, C_z = ik['C']
    m_sys += n * p['m_femur']
    mx    += n * p['m_femur'] * C_x / 2.0
    mz    += n * p['m_femur'] * (p['A_Z'] + C_z) / 2.0

    # Coupler midpoint (F→E)
    E_x, E_z = ik['E']
    F_X, F_Z = ik['F']
    m_sys += n * p['m_coupler']
    mx    += n * p['m_coupler'] * (F_X + E_x) / 2.0
    mz    += n * p['m_coupler'] * (F_Z + E_z) / 2.0

    # Tibia CoM (centre of the C→E / C→W bar)
    L_s, L_t = p['L_stub'], p['L_tibia']
    alpha     = ik['alpha']
    off       = (L_s - L_t) / 2.0
    m_sys += n * p['m_tibia']
    mx    += n * p['m_tibia'] * (C_x + off * math.sin(alpha))
    mz    += n * p['m_tibia'] * (C_z + off * math.cos(alpha))

    # Wheel at W
    W_x, W_z = ik['W']
    m_sys += n * p['m_wheel']
    mx    += n * p['m_wheel'] * W_x
    mz    += n * p['m_wheel'] * W_z

    com_x = mx / m_sys
    com_z = mz / m_sys
    dx = com_x - W_x
    dz = com_z - W_z
    if abs(dz) < 1e-4: return 0.0
    return -math.atan2(dx, dz)


# ---------------------------------------------------------------------------
def find_stroke(p: dict):
    """Scan valid IK range and return (Q_ret, Q_ext) at W_z extrema."""
    MARGIN = math.radians(3.0)
    Q_MAX  = -0.35
    angles = np.linspace(-2.5, 0.5, 1200)
    valid  = [(q, solve_ik(q, p)) for q in angles]
    valid  = [(q, r) for q, r in valid if r is not None]
    if len(valid) < 20: return None

    qs, wzs = zip(*[(q, r['W_z']) for q, r in valid])
    trimmed = [(q, wz) for q, wz in zip(qs, wzs)
               if qs[0] + MARGIN <= q <= min(qs[-1] - MARGIN, Q_MAX)]
    if not trimmed: return None

    q_ret = max(trimmed, key=lambda x: x[1])[0]
    q_ext = min(trimmed, key=lambda x: x[1])[0]
    if q_ret < q_ext: q_ret, q_ext = q_ext, q_ret
    return float(q_ret), float(q_ext)


# ---------------------------------------------------------------------------
class SimParams:
    """Encapsulate robot geometry parameters for clean IK/kinematics access."""
    def __init__(self):
        self.d = ROBOT.copy()

    def __getitem__(self, key):
        return self.d[key]

    def items(self):
        return self.d.items()


def build_xml(p: dict) -> str:
    """Generate MJCF XML for two-leg balanced sim with realistic component models."""
    L_f = p['L_femur']; L_s = p['L_stub']; L_t = p['L_tibia']
    Lc  = p['Lc'];      F_X = p['F_X'];    F_Z = p['F_Z']; A_Z = p['A_Z']

    tib_cz  = (L_s - L_t) / 2.0
    tib_hsz = (L_s + L_t) / 2.0

    # Hip motor centre Y (motor half-width = 21.5 mm; inner face at chassis)
    motor_y_L =  LEG_Y - 0.0215
    motor_y_R = -(LEG_Y - 0.0215)

    # Wheel assembly Y offset from wheel_asm body (outboard)
    W_Y = 0.036

    # Wheel mass split: motor body + tyre
    maytech = min(MOTOR_MASS, p['m_wheel'])
    tyre    = max(0.005, p['m_wheel'] - maytech)

    # ── Body frame geometry ──────────────────────────────────────────────────
    # Central electronics chassis
    ch_hx, ch_hy, ch_hz = 0.070, 0.050, 0.052

    # Side arm plates: span from chassis Y-edge to leg plane, covering both pivots
    arm_x_lo = F_X - 0.010               # past coupler pivot in -X
    arm_x_hi = 0.015                      # past hip pivot in +X
    arm_cx   = (arm_x_lo + arm_x_hi) / 2.0
    arm_hx_v = (arm_x_hi - arm_x_lo)   / 2.0
    arm_cy   = (ch_hy + LEG_Y) / 2.0
    arm_hy_v = (LEG_Y - ch_hy) / 2.0
    arm_z_lo = min(A_Z, F_Z) - 0.015
    arm_z_hi = max(A_Z, F_Z) + 0.020
    arm_cz   = (arm_z_lo + arm_z_hi) / 2.0
    arm_hz_v = (arm_z_hi - arm_z_lo)   / 2.0

    # Electronics positions (inside chassis, all massless — visual only)
    bat_z = -(ch_hz - 0.022)    # battery near chassis bottom
    ode_z =   ch_hz - 0.010     # ODESC near chassis top
    ard_z =   ch_hz - 0.030     # Arduino mid-top
    imu_z =   ch_hz - 0.003     # BNO086 on top surface

    def _leg(side: str) -> str:
        """Return XML fragment for one complete leg assembly."""
        sy  =  LEG_Y if side == 'L' else -LEG_Y   # Y sign
        wy  =  W_Y   if side == 'L' else -W_Y      # wheel offset Y
        # wheel_marker Y: further outboard than motor centre
        wm  =  (W_Y + 0.016) if side == 'L' else -(W_Y + 0.016)

        return f"""\
      <!-- ══ {side} LEG ══════════════════════════════════════════════════ -->
      <body name="coupler_{side}" pos="{F_X:.5f} {sy:.5f} {F_Z:.5f}">
        <inertial pos="0 0 0" mass="{p['m_coupler']:.4f}" diaginertia="1.796e-5 1.796e-5 2.004e-7" />
        <joint name="hinge_F_{side}" type="hinge" axis="0 1 0" damping="0.001"/>
        <!-- 608 bearing at F pivot (OD 22mm, ID 8mm, W 7mm) -->
        <geom type="cylinder" pos="0 0 0" euler="90 0 0"
              size="0.011 0.0035" rgba="0.85 0.85 0.85 0.90"
              contype="0" conaffinity="0" mass="{p['m_bearing']:.4f}"/>
        <geom name="coupler_geom_{side}" type="cylinder" fromto="0 0 0 {-Lc:.5f} 0 0"
              size="0.005" rgba="0.30 0.90 0.90 0.55"
              solref="0.02 1"
              contype="1" conaffinity="1"/>
        <!-- 608 bearing at E pivot -->
        <geom type="cylinder" pos="{-Lc:.5f} 0 0" euler="90 0 0"
              size="0.011 0.0035" rgba="0.85 0.85 0.85 0.90"
              contype="0" conaffinity="0" mass="{p['m_bearing']:.4f}"/>
      </body>
      <body name="femur_{side}" pos="0 {sy:.5f} {A_Z:.5f}">
        <inertial pos="0 0 0" mass="{p['m_femur']:.4f}" diaginertia="4.8848e-5 4.8848e-5 8.16e-7" />
        <joint name="hip_{side}" type="hinge" axis="0 1 0"
               range="-180 180" armature="0.01" damping="0.05"/>
        <geom name="femur_geom_{side}" type="cylinder" fromto="0 0 0 {-L_f:.5f} 0 0"
              size="0.007" rgba="0.94 0.75 0.25 0.55"
              solref="0.02 1"
              contype="1" conaffinity="1"/>
        <!-- 608 bearing at knee C -->
        <geom name="knee_geom_{side}" type="cylinder" pos="{-L_f:.5f} 0 0"
              euler="90 0 0" size="0.011 0.0035" rgba="0.85 0.85 0.85 0.90"
              contype="0" conaffinity="0" mass="{p['m_bearing']:.4f}" solref="0.02 1"/>
        <body name="tibia_{side}" pos="{-L_f:.5f} 0 0">
          <inertial pos="0 0 {tib_cz:.5f}" mass="{p['m_tibia']:.4f}" diaginertia="4.178e-5 4.178e-5 1.034e-6" />
          <joint name="knee_joint_{side}" type="hinge" axis="0 1 0"
                 range="-60 60" damping="0.001"/>
          <geom name="tibia_geom_{side}" type="cylinder" fromto="0 0 {L_s:.5f} 0 0 {-L_t:.5f}"
                size="0.008" rgba="0.75 0.45 0.90 0.55"
                solref="0.02 1"
                contype="1" conaffinity="1"/>
          <!-- 608 bearing at E (stub tip) -->
          <geom name="stub_geom_{side}" type="cylinder" pos="0 0 {L_s:.5f}"
                euler="90 0 0" size="0.011 0.0035" rgba="0.85 0.85 0.85 0.90"
                contype="0" conaffinity="0" mass="{p['m_bearing']:.4f}" solref="0.02 1"/>
          <body name="wheel_asm_{side}" pos="0 0 {-L_t:.5f}">
            <joint name="wheel_spin_{side}" type="hinge" axis="0 1 0"
                   damping="0.001"/>
            <!-- Maytech MTO7052HBM hub motor (Φ70mm × 52mm) -->
            <geom name="motor_body_geom_{side}" type="cylinder"
                  pos="0 {wy:.4f} 0" euler="90 0 0"
                  size="0.035 0.026" rgba="0.15 0.15 0.15 0.80"
                  mass="{maytech:.4f}" solref="0.02 1"
                  contype="1" conaffinity="1"/>
            <!-- Ground contact cylinder (invisible; torus mesh provides visual) -->
            <!-- solref: timeconst=0.06s (soft TPU tire), dampratio=0.9 (well-damped) -->
            <!-- solimp: dmin=0.7 dmax=0.95 width=0.005 (gradual contact ramp) -->
            <geom name="wheel_tire_geom_{side}" type="cylinder"
                  pos="0 {wy:.4f} 0" euler="90 0 0"
                  size="{WHEEL_R} 0.015" rgba="0 0 0 0"
                  friction="0.8 0.01 0.001" solref="0.06 0.9"
                  solimp="0.7 0.95 0.005"
                  mass="{tyre:.4f}" contype="1" conaffinity="1"/>
            <!-- Donut torus mesh (major R=55mm, minor r=20mm) -->
            <geom name="wheel_torus_{side}" type="mesh" mesh="torus_wheel"
                  pos="0 {wy:.4f} 0"
                  rgba="0.25 0.70 0.35 0.90"
                  contype="0" conaffinity="0" mass="0"/>
            <!-- Spin marker (red dot, rotates with wheel) -->
            <geom name="wheel_marker_{side}" type="box"
                  pos="0 {wm:.4f} {WHEEL_R*0.82:.5f}"
                  size="0.007 0.012 0.007" rgba="1.0 0.15 0.15 1.0"
                  contype="0" conaffinity="0" mass="0" solref="0.02 1"/>
          </body>
        </body>
      </body>"""

    # ── Arena features: platforms and ramps ────────────────────────────────
    # Platforms: (cx, cy, half_x, half_y, half_z, "r g b")
    #   Platform sits on the ground — geom pos z = half_z.
    # Ramps: (cx, cy, half_len, half_width, half_thickness, euler_deg, "r g b")
    #   Tilted box, euler="0 pitch_deg 0".  pitch_deg>0 rises in +X direction.
    _platforms = [
        # Short/low platforms (3–5 cm tall)
        ( 0.85,  0.60, 0.25, 0.20, 0.015, "0.55 0.44 0.32"),   # 3 cm
        (-0.80,  1.10, 0.30, 0.22, 0.020, "0.50 0.43 0.34"),   # 4 cm
        ( 1.60, -0.55, 0.28, 0.25, 0.018, "0.53 0.41 0.30"),   # 3.5 cm
        (-1.25, -0.90, 0.22, 0.18, 0.013, "0.52 0.46 0.35"),   # 2.5 cm
        ( 0.50,  1.85, 0.35, 0.28, 0.025, "0.55 0.44 0.32"),   # 5 cm
        # Taller platforms (7–11 cm)
        ( 1.20,  1.55, 0.30, 0.28, 0.040, "0.42 0.34 0.24"),   # 8 cm
        (-1.00,  0.90, 0.28, 0.25, 0.045, "0.40 0.33 0.26"),   # 9 cm
        ( 1.70, -1.50, 0.32, 0.28, 0.038, "0.44 0.36 0.26"),   # 7.5 cm
        (-1.55, -1.60, 0.25, 0.22, 0.055, "0.40 0.33 0.26"),   # 11 cm
        ( 2.00,  0.15, 0.30, 0.26, 0.035, "0.42 0.34 0.24"),   # 7 cm
        (-2.00,  1.80, 0.28, 0.24, 0.043, "0.40 0.33 0.26"),   # 8.5 cm
        ( 0.15, -2.05, 0.35, 0.30, 0.048, "0.42 0.34 0.24"),   # 9.5 cm
    ]

    # Ramps: box tilted on Y-axis.  Ramp base flush with floor.
    # half_len = half the slope length, half_w = half width (Y), half_t = half thickness.
    # pitch_deg: positive tilts +X end upward.
    # Ramp centre z = half_t / cos(pitch) lifted so underside just touches floor.
    _ramps = [
        # (cx,   cy,  half_len, half_w, half_t, pitch_deg, "r g b")
        ( 1.00, -1.50, 0.28, 0.20, 0.020,   6, "0.38 0.50 0.35"),  # gentle ~6°
        (-1.60,  0.00, 0.32, 0.22, 0.020,  -7, "0.36 0.48 0.33"),  # faces -X
        ( 0.00,  1.30, 0.26, 0.20, 0.020,   5, "0.38 0.50 0.35"),  # gentle
        (-0.80, -1.80, 0.30, 0.22, 0.020,   8, "0.36 0.48 0.33"),  # moderate
        ( 1.90,  1.60, 0.28, 0.20, 0.020,  -6, "0.38 0.50 0.35"),  # faces -X
    ]

    def _env_xml(platforms, ramps):
        import math as _math
        lines = ["    <!-- ── Arena platforms ──────────────────────────────────────────── -->"]
        for i, (cx, cy, hx, hy, hz, rgb) in enumerate(platforms):
            lines.append(
                f'    <geom name="plat_{i}" type="box" '
                f'pos="{cx:.3f} {cy:.3f} {hz:.4f}" '
                f'size="{hx:.3f} {hy:.3f} {hz:.4f}" '
                f'rgba="{rgb} 1.0" '
                f'friction="0.8 0.01 0.001" solref="0.02 1"/>'
            )
        lines.append("    <!-- ── Arena ramps ───────────────────────────────────────────────── -->")
        for i, (cx, cy, hl, hw, ht, pitch_deg, rgb) in enumerate(ramps):
            pitch_r = _math.radians(pitch_deg)
            # Lift centre so the bottom face (at -ht in local Z) just clears z=0
            cz = ht / _math.cos(pitch_r) + hl * abs(_math.sin(pitch_r))
            lines.append(
                f'    <geom name="ramp_{i}" type="box" '
                f'pos="{cx:.3f} {cy:.3f} {cz:.4f}" '
                f'size="{hl:.3f} {hw:.3f} {ht:.4f}" '
                f'euler="{0} {pitch_deg} {0}" '
                f'rgba="{rgb} 1.0" '
                f'friction="0.8 0.01 0.001" solref="0.02 1"/>'
            )
        return "\n".join(lines)

    return f"""<mujoco model="baseline1_2leg">
  <option gravity="0 0 -9.81" timestep="0.0005" solver="Newton"
          iterations="200" tolerance="1e-10"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <!-- Torus wheel mesh (major R=55mm, minor r=20mm, X-Z plane, Y=rot axis) -->
    <mesh name="torus_wheel" file="torus_wheel.stl" scale="1 1 1"/>
    <!-- Checkerboard floor: each checker square = 10 cm (texrepeat 50 on 5m half-size) -->
    <texture name="checker" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.50 0.50 0.50" rgb2="0.28 0.28 0.28" mark="none"/>
    <material name="floor_mat" texture="checker" texrepeat="50 50" reflectance="0.12"/>
    <!-- Wall material: warm concrete -->
    <texture name="wall_tex" type="2d" builtin="flat" width="1" height="1"
             rgb1="0.68 0.64 0.58"/>
    <material name="wall_mat" texture="wall_tex" reflectance="0.04"/>
  </asset>
  <worldbody>
    <light name="sun"   pos="0 0 4"    dir="0 0 -1"   diffuse="0.80 0.80 0.75" castshadow="true"/>
    <light name="front" pos="0 -3 2.5" dir="0  1 -0.8" diffuse="0.40 0.40 0.45"/>
    <light name="back"  pos="0  3 2.5" dir="0 -1 -0.8" diffuse="0.40 0.40 0.45"/>

    <!-- ── Ground: soft checkerboard 5 m × 5 m arena ──────────────────── -->
    <geom name="ground" type="plane" size="5 5 0.1"
          material="floor_mat" condim="3" friction="0.8 0.01 0.001"
          solref="0.04 1" solimp="0.9 0.95 0.001"/>

    <!-- ── Arena walls: 5 m × 5 m enclosure, 0.5 m tall ──────────────── -->
    <!-- North wall (+X face, inner edge at x=+2.5) -->
    <geom name="wall_N" type="box" pos=" 2.55 0 0.25" size="0.05 2.60 0.25"
          material="wall_mat" friction="0.6 0.01 0.001" solref="0.02 1"/>
    <!-- South wall (-X face) -->
    <geom name="wall_S" type="box" pos="-2.55 0 0.25" size="0.05 2.60 0.25"
          material="wall_mat" friction="0.6 0.01 0.001" solref="0.02 1"/>
    <!-- East wall (+Y face) -->
    <geom name="wall_E" type="box" pos="0  2.55 0.25" size="2.50 0.05 0.25"
          material="wall_mat" friction="0.6 0.01 0.001" solref="0.02 1"/>
    <!-- West wall (-Y face) -->
    <geom name="wall_W" type="box" pos="0 -2.55 0.25" size="2.50 0.05 0.25"
          material="wall_mat" friction="0.6 0.01 0.001" solref="0.02 1"/>

{_env_xml(_platforms, _ramps)}

    <!-- ── Body ───────────────────────────────────────────────────────── -->
    <body name="box" pos="0 0 0.45">
      <joint name="root_free" type="free"/>
      <inertial pos="0 0 0" mass="{p['m_box']:.4f}"
                diaginertia="8.0e-4 1.2e-3 1.2e-3"/>
      <!-- Central electronics chassis -->
      <geom name="chassis" type="box"
            size="{ch_hx:.4f} {ch_hy:.4f} {ch_hz:.4f}"
            rgba="0.25 0.35 0.55 0.85" mass="0"
            contype="0" conaffinity="0"/>
      <!-- Side arm plates -->
      <geom name="arm_L" type="box"
            pos="{arm_cx:.5f} {arm_cy:.5f} {arm_cz:.5f}"
            size="{arm_hx_v:.5f} {arm_hy_v:.5f} {arm_hz_v:.5f}"
            rgba="0.35 0.45 0.65 0.70" mass="0"
            contype="0" conaffinity="0"/>
      <geom name="arm_R" type="box"
            pos="{arm_cx:.5f} {-arm_cy:.5f} {arm_cz:.5f}"
            size="{arm_hx_v:.5f} {arm_hy_v:.5f} {arm_hz_v:.5f}"
            rgba="0.35 0.45 0.65 0.70" mass="0"
            contype="0" conaffinity="0"/>
      <!-- AK45-10 hip motors (Phi53mm x 43mm) -->
      <geom name="motor_L" type="cylinder"
            pos="0 {motor_y_L:.5f} {A_Z:.5f}" euler="90 0 0"
            size="0.0265 0.0215" rgba="0.15 0.15 0.15 0.90"
            mass="{MOTOR_MASS:.4f}"
            contype="0" conaffinity="0"/>
      <geom name="motor_R" type="cylinder"
            pos="0 {motor_y_R:.5f} {A_Z:.5f}" euler="90 0 0"
            size="0.0265 0.0215" rgba="0.15 0.15 0.15 0.90"
            mass="{MOTOR_MASS:.4f}"
            contype="0" conaffinity="0"/>
      <!-- Electronics (massless, visual only) -->
      <geom type="box" pos="0 0 {bat_z:.5f}" size="0.060 0.038 0.018"
            rgba="0.20 0.20 0.80 0.70" mass="0"
            contype="0" conaffinity="0"/>
      <geom type="box" pos="0 0 {ode_z:.5f}" size="0.050 0.030 0.008"
            rgba="0.80 0.30 0.10 0.80" mass="0"
            contype="0" conaffinity="0"/>
      <geom type="box" pos="0 0 {ard_z:.5f}" size="0.054 0.027 0.006"
            rgba="0.10 0.60 0.10 0.80" mass="0"
            contype="0" conaffinity="0"/>
      <geom type="box" pos="0 0 {imu_z:.5f}" size="0.010 0.010 0.003"
            rgba="0.80 0.80 0.10 0.90" mass="0"
            contype="0" conaffinity="0"/>
      <site name="imu_site" pos="0 0 {imu_z:.5f}" size="0.002"/>
      {_leg('L')}
      {_leg('R')}
    </body>
  </worldbody>

  <equality>
    <connect name="4bar_close_L" body1="coupler_L" body2="tibia_L"
             anchor="0 0 0"
             solref="0.002 1" solimp="0.9999 0.9999 0.001"/>
    <connect name="4bar_close_R" body1="coupler_R" body2="tibia_R"
             anchor="0 0 0"
             solref="0.002 1" solimp="0.9999 0.9999 0.001"/>
  </equality>

  <actuator>
    <motor name="hip_act_L"   joint="hip_L"        gear="1"
           ctrllimited="true"
           ctrlrange="{-HIP_TORQUE_LIMIT:.1f} {HIP_TORQUE_LIMIT:.1f}"/>
    <motor name="hip_act_R"   joint="hip_R"        gear="1"
           ctrllimited="true"
           ctrlrange="{-HIP_TORQUE_LIMIT:.1f} {HIP_TORQUE_LIMIT:.1f}"/>
    <motor name="wheel_act_L" joint="wheel_spin_L" gear="1"
           ctrllimited="true"
           ctrlrange="{-WHEEL_TORQUE_LIMIT:.1f} {WHEEL_TORQUE_LIMIT:.1f}"/>
    <motor name="wheel_act_R" joint="wheel_spin_R" gear="1"
           ctrllimited="true"
           ctrlrange="{-WHEEL_TORQUE_LIMIT:.1f} {WHEEL_TORQUE_LIMIT:.1f}"/>
  </actuator>

  <sensor>
    <accelerometer name="accel" site="imu_site"/>
  </sensor>
</mujoco>"""
