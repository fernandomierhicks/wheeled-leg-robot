"""
eval_jump_balanced.py — Balanced 4-bar jump evaluator.

Combines the parametric geometry generation of eval_jump.py with the 
active balance controller from balance_sim.py.

Features:
  - Full physics simulation with IMU noise and delayed control.
  - Active PID balance controller on wheels.
  - Active Impedance/Torque control on hip for jumping.
  - Reaction wheel air control.
  - Geometry feasibility checks.

Run:
    python simulation/mujoco/4bar_optimization_with_balancing/eval_jump_balanced.py
"""

import csv
import datetime
import math
import os
import sys

import mujoco
import numpy as np

# ── File paths ──────────────────────────────────────────────────────────────
_DIR     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_DIR, "results_balanced.csv")

# ── Robot Physical Constants ────────────────────────────────────────────────
LEG_Y            = 0.1430    # Y-offset of leg plane from box centre [m]
MOTOR_MASS       = 0.260     # AK45-10 [kg]
MOTOR_R_MM       = 26.5      # AK45-10 housing radius [mm]
MIN_AF_CLEAR_MM  = 12.5      # F-bearing to motor clearance
MIN_TIBIA_SPAN_MM = 25.0     # min C–E distance
SINGULARITY_LIM  = 0.95      # 4-bar closure limit
MIN_STROKE_DEG   = 15.0      # min valid stroke
STROKE_MARGIN_RAD = math.radians(3.0)
Q_HIP_RETRACT_MAX = -0.35    # Max hip angle for retraction
BOX_HALF_X       = 0.050     # body box half-width
W_X_MARGIN       = 0.000     # wheel overhang allowed
MIN_JUMP_DZ      = 0.010     # min jump extension height
WHEEL_R          = 0.075     # [m]

# ── Default Geometry ────────────────────────────────────────────────────────
DEFAULT = dict(
    L_femur  = 0.100,    L_stub   = 0.025,    L_tibia  = 0.115,
    Lc       = 0.110,    F_X      = -0.020,   F_Z      =  0.01116,
    A_Z      = -0.0235,  m_box    = 0.477,
    m_femur  = 0.025,    m_tibia  = 0.035,    m_coupler= 0.015,
    m_wheel  = 0.410,
)

# ── Balance & Physics Parameters (Matching balance_sim.py) ──────────────────
PITCH_KP = 60.0       # [N·m/rad]
PITCH_KI = 0.0        # [N·m/(rad·s)]
PITCH_KD = 5.0        # [N·m·s/rad]

POSITION_KP = 0.30    # [rad/m]
VELOCITY_KP = 0.30    # [rad/(m/s)]
MAX_PITCH_CMD = 0.25  # [rad]

WHEEL_TORQUE_LIMIT = 7.0

# Hip Control
HIP_KP = 30.0
HIP_KD = 1.0
HIP_TORQUE_LIMIT = 7.0
OMEGA_MAX = 18.85
JUMP_RAMP_S = 0.010
JUMP_RAMPDOWN = 0.15
CROUCH_DURATION_S = 1.5   # [s] time to ramp from neutral to fully retracted

# Simulation / Noise
SIM_DURATION_S = 8.0
ACCEL_NOISE_STD = 0.2
PITCH_NOISE_STD_RAD = math.radians(0.1)
PITCH_RATE_NOISE_STD_RAD_S = math.radians(0.5)

# ── CSV columns ─────────────────────────────────────────────────────────────
CSV_COLS = [
    "run_id", "label", "timestamp",
    "L_femur_mm", "L_stub_mm", "L_tibia_mm", "Lc_mm",
    "F_X_mm", "F_Z_mm", "A_Z_mm",
    "AF_mm", "AF_motor_clearance_mm",
    "m_box_g", "total_mass_g",
    "Q_retracted_rad", "Q_extended_rad", "stroke_deg",
    "jump_height_mm",
    "status", "fail_reason", "bearing_warnings"
]


# ---------------------------------------------------------------------------
# Helper: Wrap Angle
# ---------------------------------------------------------------------------
def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Helper: CSV Logger
# ---------------------------------------------------------------------------
def _next_run_id():
    if not os.path.exists(CSV_PATH): return 1
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            return max(1, sum(1 for _ in csv.reader(f)))
    except: return 1

def _log_csv(row: dict):
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if write_header: w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLS})


# ---------------------------------------------------------------------------
# Kinematics: Inverse Kinematics
# ---------------------------------------------------------------------------
def solve_ik(q_hip: float, p: dict) -> dict | None:
    L_f = p['L_femur']; L_s = p['L_stub']; L_t = p['L_tibia']
    Lc  = p['Lc'];      F_X = p['F_X'];    F_Z = p['F_Z']; A_Z = p['A_Z']

    C_x = -L_f * math.cos(q_hip)
    C_z =  A_Z + L_f * math.sin(q_hip)
    dx, dz = C_x - F_X, C_z - F_Z
    R = math.sqrt(dx*dx + dz*dz)
    if R < 1e-9: return None

    K = (Lc**2 - dx**2 - dz**2 - L_s**2) / (2.0 * L_s)
    kr = abs(K) / R
    if kr >= SINGULARITY_LIM: return None

    phi   = math.atan2(dz, dx)
    asinv = math.asin(max(-1.0, min(1.0, K / R)))
    a1    = _wrap(asinv - phi)
    a2    = _wrap(math.pi - asinv - phi)
    qk1   = a1 - q_hip; qk2 = a2 - q_hip
    alpha = a1 if abs(qk1) <= abs(qk2) else a2

    q_knee      = alpha - q_hip
    E_x         = C_x + L_s * math.sin(alpha)
    E_z         = C_z + L_s * math.cos(alpha)
    W_x         = C_x - L_t * math.sin(alpha)
    W_z         = C_z - L_t * math.cos(alpha)
    q_coupler_F = math.atan2(E_z - F_Z, F_X - E_x)

    return dict(q_hip=q_hip, q_knee=q_knee, q_coupler_F=q_coupler_F,
                C=(C_x, C_z), E=(E_x, E_z), F=(F_X, F_Z), W=(W_x, W_z), 
                W_z=W_z, KR_ratio=kr, alpha=alpha)

def get_equilibrium_pitch(p: dict, q_hip: float) -> float:
    """
    Calculate the body pitch (rad) required to align the System CoM
    vertically over the Wheel Center W.
    Returns theta such that CoM_x_world = W_x_world.
    """
    ik = solve_ik(q_hip, p)
    if ik is None: return 0.0

    # 1. Component Masses & Positions in Body Frame
    # Body Box
    m_sys = p['m_box']
    mx_sys = 0.0  # Box CoM at (0,0) in body frame
    mz_sys = 0.0

    # Hip Motor A (at 0, A_Z)
    m_sys += MOTOR_MASS
    mz_sys += MOTOR_MASS * p['A_Z']

    # Femur (Midpoint of A-C)
    # A=(0, A_Z), C from IK
    C_x, C_z = ik['C']
    fem_x = C_x / 2.0
    fem_z = (p['A_Z'] + C_z) / 2.0
    m_sys += p['m_femur']
    mx_sys += p['m_femur'] * fem_x
    mz_sys += p['m_femur'] * fem_z

    # Coupler (Midpoint of F-E)
    # F from params, E from IK
    E_x, E_z = ik['E']
    F_X, F_Z = ik['F']
    cpl_x = (F_X + E_x) / 2.0
    cpl_z = (F_Z + E_z) / 2.0
    m_sys += p['m_coupler']
    mx_sys += p['m_coupler'] * cpl_x
    mz_sys += p['m_coupler'] * cpl_z

    # Tibia (Geometry center is at tib_cz in Tibia frame)
    # Tibia frame origin is C, Z axis is line C-E.
    # Tibia CoM is at distance (L_s - L_t)/2 from C along C-E line.
    # This corresponds to vector direction 'alpha' in IK.
    # tib_cz = (L_s - L_t)/2
    L_s, L_t = p['L_stub'], p['L_tibia']
    tib_offset = (L_s - L_t) / 2.0
    # Direction: sin(alpha), cos(alpha)
    alpha = ik['alpha']
    tib_x = C_x + tib_offset * math.sin(alpha)
    tib_z = C_z + tib_offset * math.cos(alpha)
    m_sys += p['m_tibia']
    mx_sys += p['m_tibia'] * tib_x
    mz_sys += p['m_tibia'] * tib_z

    # Wheel (at W)
    W_x, W_z = ik['W']
    m_w = p['m_wheel']
    m_sys += m_w
    mx_sys += m_w * W_x
    mz_sys += m_w * W_z

    # 2. System CoM
    com_x = mx_sys / m_sys
    com_z = mz_sys / m_sys

    # 3. Calculate Balance Angle
    # We want to rotate Body by theta so that (CoM - W) becomes vertical.
    # Vector V = CoM - W = (dx, dz)
    # Rotated V' = [dx cos - dz sin, dx sin + dz cos]
    # We want V'_x = 0  => dx cos = dz sin => tan(theta) = dx/dz
    # Note: MuJoCo Pitch is rotation about Y. Positive pitch = Nose Down (X -> -Z).
    # If CoM is forward of wheel (dx > 0), we need to pitch UP (negative angle).
    # theta = -atan(dx/dz)
    
    dx = com_x - W_x
    dz = com_z - W_z  # Typically positive (CoM above wheel)
    
    if abs(dz) < 1e-4: return 0.0
    return -math.atan2(dx, dz)

# ---------------------------------------------------------------------------
# Feasibility Checks
# ---------------------------------------------------------------------------
def check_feasibility(p: dict):
    computed, warn_list = {}, []
    
    # Bearings
    stub_mm = p['L_stub'] * 1000.0
    if stub_mm < MIN_TIBIA_SPAN_MM:
        warn_list.append(f"stub {stub_mm:.1f}<{MIN_TIBIA_SPAN_MM:.0f}")
    
    AF_mm = math.sqrt(p['F_X']**2 + (p['F_Z'] - p['A_Z'])**2) * 1000.0
    af_clear = AF_mm - MOTOR_R_MM
    computed.update(AF_mm=round(AF_mm,2), AF_motor_clearance_mm=round(af_clear,2))
    if af_clear < MIN_AF_CLEAR_MM:
        warn_list.append(f"AF_clear {af_clear:.1f}<{MIN_AF_CLEAR_MM:.1f}")

    # Closure
    angles = np.linspace(-2.5, 0.5, 1200)
    valid = [q for q in angles if solve_ik(q, p) is not None]
    if not valid:
        return False, "4-bar closure fails everywhere", computed, "|".join(warn_list)
    
    stroke = abs(math.degrees(valid[-1] - valid[0]))
    if stroke < MIN_STROKE_DEG:
        return False, f"stroke {stroke:.1f}<{MIN_STROKE_DEG}", computed, "|".join(warn_list)

    return True, "", computed, "|".join(warn_list)


# ---------------------------------------------------------------------------
# Find Stroke
# ---------------------------------------------------------------------------
def find_stroke(p: dict):
    angles = np.linspace(-2.5, 0.5, 1200)
    valid = [(q, solve_ik(q, p)) for q in angles]
    valid = [(q, r) for q, r in valid if r is not None]
    if len(valid) < 20: return None

    qs, wzs = zip(*[(q, r['W_z']) for q, r in valid])
    q_lo, q_hi = qs[0] + STROKE_MARGIN_RAD, min(qs[-1] - STROKE_MARGIN_RAD, Q_HIP_RETRACT_MAX)
    trimmed = [(q, wz) for q, wz in zip(qs, wzs) if q_lo <= q <= q_hi]
    if not trimmed: return None

    q_ret = max(trimmed, key=lambda x: x[1])[0]
    q_ext = min(trimmed, key=lambda x: x[1])[0]
    if q_ret < q_ext: q_ret, q_ext = q_ext, q_ret
    return float(q_ret), float(q_ext)


# ---------------------------------------------------------------------------
# Validate Stroke
# ---------------------------------------------------------------------------
def validate_stroke(p: dict, Q_ret: float, Q_ext: float):
    ik_ret = solve_ik(Q_ret, p)
    ik_ext = solve_ik(Q_ext, p)
    if not ik_ret or not ik_ext: return False, "IK fail at endpoints"

    dz = ik_ret['W_z'] - ik_ext['W_z']
    if dz < MIN_JUMP_DZ:
        return False, f"jump dz {dz*1000:.1f}mm < {MIN_JUMP_DZ*1000:.0f}mm"

    wx = ik_ret['W'][0]
    if abs(wx) > BOX_HALF_X + W_X_MARGIN:
        return False, f"wheel X {wx*1000:.1f}mm outside box"

    return True, ""


# ---------------------------------------------------------------------------
# Build XML (with IMU and Balance Sim Params)
# ---------------------------------------------------------------------------
def build_xml(p: dict) -> str:
    L_f = p['L_femur']; L_s = p['L_stub']; L_t = p['L_tibia']
    Lc  = p['Lc'];      F_X = p['F_X'];    F_Z = p['F_Z']; A_Z = p['A_Z']
    tib_cz  = (L_s - L_t) / 2.0
    tib_hsz = (L_s + L_t) / 2.0
    motor_y = LEG_Y - 0.0215
    maytech_mass = min(MOTOR_MASS, p['m_wheel'])
    tyre_mass    = max(0.005, p['m_wheel'] - maytech_mass)

    return f"""<mujoco model="balance_leg_opt">
  <option gravity="0 0 -9.81" timestep="0.0005" solver="Newton" iterations="200" tolerance="1e-10"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <worldbody>
    <light name="main" pos="0 -1 1" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" pos="0 0 0" size="5 5 0.01" rgba="0.3 0.3 0.3 1" contype="1" conaffinity="1" friction="0.8 0.005 0.0001" solref="0.02 1"/>
    
    <body name="box" pos="0 0 0.40">
      <joint name="root_x" type="slide" axis="1 0 0" pos="0 0 0" damping="0.5"/>
      <joint name="root_z" type="slide" axis="0 0 1" pos="0 0 0" damping="0.5"/>
      <joint name="root_pitch" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.01"/>
      
      <geom name="box_geom" type="box" size="0.05 0.10 0.05" rgba="0.29 0.56 0.85 0.55" contype="1" conaffinity="1" mass="{p['m_box']:.4f}" solref="0.02 1"/>
      <site name="imu" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>
      <geom name="motor_geom" type="cylinder" pos="0 {motor_y:.4f} {A_Z:.5f}" euler="90 0 0" size="0.0265 0.0215" rgba="0.87 0.36 0.36 0.55" contype="1" conaffinity="1" mass="{MOTOR_MASS:.3f}" solref="0.02 1"/>
      
      <body name="coupler" pos="{F_X:.5f} {LEG_Y:.4f} {F_Z:.5f}">
        <joint name="hinge_F" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.001"/>
        <geom type="cylinder" pos="0 0 0" euler="90 0 0" size="0.011 0.010" rgba="0 1 1 0.9" contype="0" conaffinity="0" mass="0"/>
        <geom name="coupler_geom" type="box" pos="{-Lc/2:.5f} 0 0" size="{Lc/2:.5f} 0.01 0.01" rgba="0.30 0.90 0.90 0.55" contype="1" conaffinity="1" mass="{p['m_coupler']:.4f}" solref="0.02 1"/>
      </body>
      
      <body name="femur" pos="0 {LEG_Y:.4f} {A_Z:.5f}">
        <joint name="hip" type="hinge" axis="0 1 0" pos="0 0 0" range="-180 180" armature="0.01" damping="0.05"/>
        <geom type="cylinder" pos="0 0 0" euler="90 0 0" size="0.005 0.015" rgba="1 1 0 0.9" contype="0" conaffinity="0" mass="0"/>
        <geom name="femur_geom" type="box" pos="{-L_f/2:.5f} 0 0" size="{L_f/2:.5f} 0.01 0.01" rgba="0.94 0.75 0.25 0.55" contype="1" conaffinity="1" mass="{p['m_femur']:.4f}" solref="0.02 1"/>
        <geom name="knee_geom" type="cylinder" pos="{-L_f:.5f} 0 0" euler="90 0 0" size="0.011 0.010" rgba="1 1 1 0.9" contype="0" conaffinity="0" mass="0" solref="0.02 1"/>
        
        <body name="tibia" pos="{-L_f:.5f} 0 0">
          <joint name="knee_joint" type="hinge" axis="0 1 0" pos="0 0 0" range="-60 60" damping="0.001"/>
          <geom name="tibia_geom" type="box" pos="0 0 {tib_cz:.5f}" size="0.010 0.010 {tib_hsz:.5f}" rgba="0.75 0.45 0.90 0.55" contype="1" conaffinity="1" mass="{p['m_tibia']:.4f}" solref="0.02 1"/>
          <geom name="stub_geom" type="cylinder" pos="0 0 {L_s:.5f}" euler="90 0 0" size="0.011 0.010" rgba="1 0.3 0.3 0.9" contype="0" conaffinity="0" mass="0" solref="0.02 1"/>
          
          <body name="wheel_asm" pos="0 0 {-L_t:.5f}">
            <joint name="wheel_spin" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.001"/>
            <geom name="motor_body_geom" type="cylinder" pos="0 0.036 0" euler="90 0 0" size="0.035 0.026" rgba="0.15 0.15 0.15 0.55" contype="1" conaffinity="1" mass="{maytech_mass:.4f}" solref="0.02 1"/>
            <geom name="wheel_tire_geom" type="cylinder" pos="0 0.036 0" euler="90 0 0" size="{WHEEL_R} 0.015" rgba="0.25 0.70 0.35 0.80" contype="1" conaffinity="1" friction="0.8 0.01 0.001" solref="0.02 1" solimp="0.9 0.99 0.001" mass="{tyre_mass:.4f}"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <connect name="4bar_close" body1="coupler" body2="tibia" anchor="0 0 0" solref="0.00005 1" solimp="0.9999 0.9999 0.0001"/>
  </equality>
  <sensor>
    <accelerometer name="accel" site="imu"/>
  </sensor>
  <actuator>
    <motor name="hip_motor" joint="hip" ctrlrange="-{HIP_TORQUE_LIMIT} {HIP_TORQUE_LIMIT}" gear="1"/>
    <motor name="wheel_motor" joint="wheel_spin" ctrlrange="-{WHEEL_TORQUE_LIMIT} {WHEEL_TORQUE_LIMIT}" gear="1"/>
  </actuator>
</mujoco>"""


# ---------------------------------------------------------------------------
# Balanced Headless Simulation
# ---------------------------------------------------------------------------
def run_balanced_sim(p: dict, Q_ret: float, Q_ext: float) -> tuple[float, str]:
    """
    Run simulation with active balancing and jump logic.
    Returns (max_jump_height_mm, fail_reason).
    """
    xml = build_xml(p)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Override closure anchors
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "4bar_close")
    model.eq_data[eq_id, 0:3] = [-p['Lc'], 0.0, 0.0]
    model.eq_data[eq_id, 3:6] = [0.0, 0.0, p['L_stub']]

    # ID Lookups
    s_pitch = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch")]
    s_hip   = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip")]
    s_root_z= model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_z")]
    s_hF    = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_F")]
    s_knee  = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")]

    d_pitch = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch")]
    d_hip   = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip")]
    d_wheel = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_spin")]

    wheel_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_asm")
    gid_knee  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "knee_geom")
    gid_stub  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "stub_geom")

    # ── Init Pose (Start in Crouch) ─────────────────────────────────────────
    # Start in NEUTRAL position (between retracted and extended)
    # This prevents the "starting from extreme" jitters
    Q_neutral = Q_ret + 0.30 * (Q_ext - Q_ret)
    
    ik = solve_ik(Q_neutral, p)
    if not ik: return 0.0, "Init IK failed"
    
    data.qpos[s_hF]   = ik['q_coupler_F']
    data.qpos[s_hip]  = ik['q_hip']
    data.qpos[s_knee] = ik['q_knee']
    
    # Place on ground
    mujoco.mj_forward(model, data)
    wheel_z = data.xpos[wheel_bid][2]
    data.qpos[s_root_z] += WHEEL_R - wheel_z
    
    # Seed with calculated equilibrium pitch to minimize startup drift
    theta_eq = get_equilibrium_pitch(p, Q_neutral)
    data.qpos[s_pitch] = theta_eq
    mujoco.mj_forward(model, data)

    # ── State Machine ───────────────────────────────────────────────────────
    # 1. BALANCE (stabilize for 0.5s)
    # 2. JUMP (trigger jump)
    # 3. FLIGHT/LAND (track height, retract legs)
    
    sim_t = 0.0
    pitch_integral = 0.0
    odo_x = 0.0
    grounded = True
    
    leg_state = "NEUTRAL" # NEUTRAL -> CROUCH -> JUMP -> NEUTRAL
    current_hip_target = Q_neutral

    jump_triggered = False
    jump_start_t = 0.0
    crouch_start_t = 0.0
    
    max_height_m = 0.0
    
    # We run step-by-step
    while data.time < SIM_DURATION_S:
        sim_t = data.time
        
        # ── Sensors ──
        pitch_true = data.qpos[s_pitch]
        pitch_rate_true = data.qvel[d_pitch]
        
        # Add Noise
        pitch = pitch_true + np.random.normal(0, PITCH_NOISE_STD_RAD)
        pitch_rate = pitch_rate_true + np.random.normal(0, PITCH_RATE_NOISE_STD_RAD_S)
        
        wheel_vel = data.qvel[d_wheel]
        
        # Ground Detection (IMU)
        accel_raw = data.sensor("accel").data
        accel_noisy = accel_raw + np.random.normal(0, ACCEL_NOISE_STD, 3)
        accel_mag = np.linalg.norm(accel_noisy)
        
        if accel_mag < 3.0: grounded = False
        elif accel_mag > 7.0: grounded = True
        
        # ── Balance Controller ──
        # Feedforward: Analytical balance pitch for current stance
        # This removes the steady-state error the integral term would otherwise have to hunt for
        pitch_ff = get_equilibrium_pitch(p, data.qpos[s_hip])

        target_pitch = 0.0
        if grounded:
            vel_est = (wheel_vel + pitch_rate) * WHEEL_R
            odo_x += vel_est * model.opt.timestep
            
            # Feedback: Position/Velocity -> Pitch adjustment
            pitch_fb = -(POSITION_KP * odo_x + VELOCITY_KP * vel_est)
            pitch_fb = np.clip(pitch_fb, -MAX_PITCH_CMD, MAX_PITCH_CMD)
            
            target_pitch = pitch_ff + pitch_fb
        else:
            # Air Control: Reaction Wheel
            target_pitch = 0.0
            pitch_integral = 0.0
            
        pitch_error = pitch - target_pitch
        pitch_integral += pitch_error * model.opt.timestep
        pitch_integral = np.clip(pitch_integral, -1.0, 1.0)
        
        u_bal = (PITCH_KP * pitch_error +
                 PITCH_KI * pitch_integral +
                 PITCH_KD * pitch_rate)
        data.ctrl[1] = np.clip(u_bal, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
        
        # ── Jump State Machine ──
        
        # Sequence: Neutral (0-2s) -> Crouch (2-3.5s) -> Jump (3.5s)
        if sim_t > 2.0 and leg_state == "NEUTRAL" and not jump_triggered:
            leg_state = "CROUCH"
            crouch_start_t = sim_t
            # target ramps gradually — updated each step below
            
        if sim_t > 3.5 and leg_state == "CROUCH" and not jump_triggered:
            leg_state = "JUMP"
            jump_start_t = sim_t
            jump_triggered = True
            
        # Hip Control
        hip_q = data.qpos[s_hip]
        hip_omega = data.qvel[d_hip]
        
        if leg_state == "JUMP":
            # Feed-forward torque profile
            ramp_in = min(1.0, (sim_t - jump_start_t) / JUMP_RAMP_S)
            # Ramp down as we approach Q_ext
            ramp_out = min(1.0, max(0.0, (hip_q - Q_ext) / JUMP_RAMPDOWN))
            # Speed limit torque scaling
            speed_scale = max(0.0, 1.0 - abs(hip_omega) / OMEGA_MAX)
            
            u_hip = -HIP_TORQUE_LIMIT * ramp_in * ramp_out * speed_scale
            
            # Retract logic: if extended OR airborne for a bit
            if (hip_q <= Q_ext + 0.05) or (not grounded and (sim_t - jump_start_t) > 0.05):
                leg_state = "NEUTRAL"
                current_hip_target = Q_neutral # Retract to neutral for landing
        else:
            # PD Hold — ramp target gently during CROUCH to avoid wheel lift-off
            if leg_state == "CROUCH":
                crouch_frac = min(1.0, (sim_t - crouch_start_t) / CROUCH_DURATION_S)
                current_hip_target = Q_neutral + crouch_frac * (Q_ret - Q_neutral)
            u_hip = HIP_KP * (current_hip_target - hip_q) - HIP_KD * hip_omega
            
        data.ctrl[0] = np.clip(u_hip, -HIP_TORQUE_LIMIT, HIP_TORQUE_LIMIT)
        
        # ── Step ──
        mujoco.mj_step(model, data)
        
        # ── Safety Checks ──
        # Fail if robot falls over (pitch > 60 deg)
        if abs(pitch_true) > 1.0:
            return 0.0, "fell over (pitch > 1 rad)"
            
        # Fail if mechanical crash (joints hitting ground)
        if (gid_knee >= 0 and data.geom_xpos[gid_knee][2] < 0.0) or \
           (gid_stub >= 0 and data.geom_xpos[gid_stub][2] < 0.0):
             return 0.0, "mechanical crash: joint below ground"
             
        # Track Height
        w_z = data.xpos[wheel_bid][2]
        max_height_m = max(max_height_m, w_z - WHEEL_R)
        
        # Stop if we landed and stabilized? 
        # Allow more time for the full sequence
        if jump_triggered and sim_t > jump_start_t + 2.5:
            break
            
    return max_height_m * 1000.0, ""


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------
def evaluate(p: dict, label: str = "", forced_run_id: int | None = None) -> dict:
    run_id = forced_run_id if forced_run_id else _next_run_id()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calc mass/geometry stats
    AF_mm = math.sqrt(p['F_X']**2 + (p['F_Z'] - p['A_Z'])**2) * 1000.0
    af_clear = AF_mm - MOTOR_R_MM
    total_mass = (p['m_box'] + p['m_femur'] + p['m_tibia'] + 
                  p['m_coupler'] + p['m_wheel'] + MOTOR_MASS) * 1000.0

    row = dict(
        run_id=run_id, label=label, timestamp=ts,
        L_femur_mm=round(p['L_femur']*1000,2), L_stub_mm=round(p['L_stub']*1000,2),
        L_tibia_mm=round(p['L_tibia']*1000,2), Lc_mm=round(p['Lc']*1000,2),
        F_X_mm=round(p['F_X']*1000,2), F_Z_mm=round(p['F_Z']*1000,2),
        A_Z_mm=round(p['A_Z']*1000,2), AF_mm=round(AF_mm,2),
        AF_motor_clearance_mm=round(af_clear,2),
        m_box_g=round(p['m_box']*1000,1), total_mass_g=round(total_mass,1)
    )

    # 1. Feasibility
    ok, reason, _, warnings = check_feasibility(p)
    row['bearing_warnings'] = warnings
    if not ok:
        row.update(status="FAIL", fail_reason=reason)
        print(f"[{run_id}] {label:20s} FAIL geom: {reason}")
        _log_csv(row); return row

    # 2. Find Stroke
    stroke = find_stroke(p)
    if not stroke:
        row.update(status="FAIL", fail_reason="stroke not found")
        print(f"[{run_id}] {label:20s} FAIL stroke not found")
        _log_csv(row); return row
    Q_ret, Q_ext = stroke

    # 3. Validate Stroke
    ok, reason = validate_stroke(p, Q_ret, Q_ext)
    if not ok:
        row.update(status="FAIL", fail_reason=reason)
        print(f"[{run_id}] {label:20s} FAIL stroke invalid: {reason}")
        _log_csv(row); return row
    
    row.update(Q_retracted_rad=round(Q_ret,4), Q_extended_rad=round(Q_ext,4),
               stroke_deg=round(abs(math.degrees(Q_ret-Q_ext)),2))

    # 4. Balanced Simulation
    h_mm, err = run_balanced_sim(p, Q_ret, Q_ext)
    if err:
        row.update(status="FAIL", fail_reason=err)
        print(f"[{run_id}] {label:20s} FAIL sim: {err}")
        _log_csv(row); return row

    row.update(jump_height_mm=round(h_mm,2), status="PASS", fail_reason="")
    print(f"[{run_id}] {label:20s} PASS: {h_mm:.1f} mm  (stroke {row['stroke_deg']}°)")
    _log_csv(row)
    return row


if __name__ == "__main__":
    # Run a single baseline test to verify the new Balanced Evaluator works
    print("="*60)
    print("Running Balanced 4-Bar Jump Evaluator")
    print("="*60)
    
    # Using visual-sim geometry (violates stub constraint but works for test)
    TEST_PARAMS = DEFAULT.copy()
    TEST_PARAMS.update({
        "L_stub": 0.015,
        "F_X":   -0.015,
        "F_Z":    0.0025,
    })
    
    evaluate(TEST_PARAMS, label="balanced_baseline")