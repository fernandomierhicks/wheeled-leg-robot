"""
eval_jump.py — headless 4-bar jump evaluator.

Feasibility checks enforced:
  1. Tibia bearing spacing : L_stub >= 25 mm  (C–E distance covers 608 OD=22 mm)
  2. Motor clearance       : distance from F-bearing centre to motor housing edge >= 12.5 mm
                             i.e. |AF| >= MOTOR_R_MM + MIN_AF_CLEAR_MM = 26.5+12.5 = 39 mm
  3. 4-bar closure         : |K/R| < SINGULARITY_LIM at every angle in stroke
  4. Stroke size           : valid stroke >= MIN_STROKE_DEG

Auto-finds Q_retracted / Q_extended by sweeping hip angles and picking the
retracted (W highest) and extended (W lowest) positions within the valid range.

Results are appended to results.csv next to this file.

Run directly for a single 2-run experiment (baseline vs stub +1 mm):
    python simulation/mujoco/4bar_optimization/eval_jump.py
"""

import csv
import datetime
import math
import os

import mujoco
import numpy as np

# ── File paths ──────────────────────────────────────────────────────────────
_DIR     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_DIR, "results.csv")

# ── Fixed physical constants ────────────────────────────────────────────────
LEG_Y            = 0.1430    # Y-offset of leg plane from box centre [m] (motor mount)
MOTOR_MASS       = 0.260     # AK45-10 [kg] — not swept
MOTOR_R_MM       = 26.5      # AK45-10 housing radius [mm]  (Ø53 mm)
MIN_AF_CLEAR_MM  = 12.5      # min distance: F-bearing centre to motor housing EDGE [mm]
                              # → hard constraint: |AF| >= 26.5 + 12.5 = 39.0 mm
MIN_TIBIA_SPAN_MM = 25.0     # min C–E distance [mm]  (608 OD=22 mm, rounded up)
SINGULARITY_LIM  = 0.95      # |K/R| must stay BELOW this everywhere in stroke
MIN_STROKE_DEG   = 15.0      # stroke must span at least this many degrees
STROKE_MARGIN_RAD = math.radians(3.0)  # safety trim from each end of valid range
# Jump stroke must start with femur angled below horizontal so wheel is under the body.
# Angles more positive than this (femur nearly horizontal) produce mechanically valid
# 4-bar closure but poor jump geometry (leg points sideways).
Q_HIP_RETRACT_MAX = -0.35    # [rad]  ~−20°: hip must be at least this negative for Q_ret

# Jump geometry constraints (checked by validate_stroke)
BOX_HALF_X   = 0.050    # body box half-width in X [m]  (100 mm box)
W_X_MARGIN   = 0.000    # allowed wheel overhang beyond box edge [m]
              # → wheel must be strictly under box footprint (±50 mm) at Q_ret
MIN_JUMP_DZ  = 0.010    # min W_z drop from Q_ret→Q_ext [m] (leg must push body up)

WHEEL_R      = 0.075    # [m]
GROUNDED_TOL = 0.015    # wheel contact threshold [m]
CTRL_HZ      = 60       # control update rate [Hz] — match visual sim (viewer at 60fps)

# ── Default geometry (current working design) ───────────────────────────────
DEFAULT = dict(
    L_femur  = 0.100,    # A→C  [m]
    L_stub   = 0.025,    # C→E tibia stub (above knee)  [m]
    L_tibia  = 0.115,    # C→W to wheel centre  [m]
    Lc       = 0.110,    # F→E coupler length  [m]
    F_X      = -0.020,   # F x-position in box frame  [m]  (A is at x=0)
    F_Z      =  0.01116, # F z-position in box frame  [m]
    A_Z      = -0.0235,  # A z-position in box frame  [m]  (hip motor centre)
    m_box    = 0.477,    # body electronics + structure  [kg]
    m_femur  = 0.025,
    m_tibia  = 0.035,
    m_coupler= 0.015,
    m_wheel  = 0.410,    # Maytech motor + TPU tyre  [kg]
)

# ── Control ─────────────────────────────────────────────────────────────────
KP            = 20.0
KD            =  1.0
MAX_TORQUE    =  7.0
HOLD_TORQUE   =  2.5     # AK45-10 hold torque [N·m]  (peak=7, hold=2.5 per datasheet)
OMEGA_MAX     = 18.85    # AK45-10 output shaft no-load [rad/s]
JUMP_RAMP_S   =  0.010   # torque ramp-up duration [s]
JUMP_RAMPDOWN =  0.15    # taper torque to 0 in last 0.15 rad before Q_extended
CROUCH_TIME   =  0.5
STABLE_TIME  =  0.4
WHEEL_CLEARANCE = 0.001  # drop height above ground [m]
SIM_DURATION_S  = 8.0

# State labels
FALLING=0; CROUCHING=1; JUMPING=2; LANDING=3

# ── CSV columns ─────────────────────────────────────────────────────────────
CSV_COLS = [
    "run_id", "label", "timestamp",
    "L_femur_mm", "L_stub_mm", "L_tibia_mm", "Lc_mm",
    "F_X_mm", "F_Z_mm", "A_Z_mm",
    "AF_mm", "AF_motor_clearance_mm",
    "m_box_g", "m_femur_g", "m_tibia_g", "m_coupler_g", "m_wheel_g", "total_mass_g",
    "Q_retracted_rad", "Q_extended_rad", "stroke_deg",
    "jump_height_mm",
    "status",            # PASS or FAIL (geometry/closure only — not bearing constraints)
    "fail_reason",
    "bearing_warnings",  # pipe-separated list of violated bearing constraints; empty = OK
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _next_run_id():
    """Return next integer run ID by counting existing CSV rows."""
    if not os.path.exists(CSV_PATH):
        return 1
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            return max(1, sum(1 for _ in csv.reader(f)))  # header counts as 1
    except UnicodeDecodeError:
        # Fallback for files created with system-default encoding
        with open(CSV_PATH, newline="", encoding="latin-1") as f:
            return max(1, sum(1 for _ in csv.reader(f)))


def _log_csv(row: dict):
    write_header = not os.path.exists(CSV_PATH)
    # Always write/append with UTF-8 for consistency.
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLS})


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def solve_ik(q_hip: float, p: dict) -> dict | None:
    """
    Full IK for one hip angle.
    Returns dict with joint angles and key world positions, or None if:
      - closure fails (no real solution)
      - |K/R| >= SINGULARITY_LIM  (near singularity)
    """
    L_f = p['L_femur']; L_s = p['L_stub']; L_t = p['L_tibia']
    Lc  = p['Lc'];      F_X = p['F_X'];    F_Z = p['F_Z']; A_Z = p['A_Z']

    C_x = -L_f * math.cos(q_hip)
    C_z =  A_Z + L_f * math.sin(q_hip)

    dx = C_x - F_X
    dz = C_z - F_Z
    R  = math.sqrt(dx * dx + dz * dz)
    if R < 1e-9:
        return None

    K = (Lc**2 - dx**2 - dz**2 - L_s**2) / (2.0 * L_s)

    kr = abs(K) / R
    if kr >= SINGULARITY_LIM:
        return None  # near or past singularity

    phi   = math.atan2(dz, dx)
    asinv = math.asin(max(-1.0, min(1.0, K / R)))
    a1    = _wrap(asinv - phi)
    a2    = _wrap(math.pi - asinv - phi)
    qk1   = a1 - q_hip
    qk2   = a2 - q_hip
    alpha = a1 if abs(qk1) <= abs(qk2) else a2

    q_knee      = alpha - q_hip
    E_x         = C_x + L_s * math.sin(alpha)
    E_z         = C_z + L_s * math.cos(alpha)
    W_x         = C_x - L_t * math.sin(alpha)
    W_z         = C_z - L_t * math.cos(alpha)
    q_coupler_F = math.atan2(E_z - F_Z, F_X - E_x)

    return dict(
        q_hip=q_hip, q_knee=q_knee, q_coupler_F=q_coupler_F,
        C=(C_x, C_z), E=(E_x, E_z), W=(W_x, W_z),
        W_z=W_z, KR_ratio=kr,
    )


# ---------------------------------------------------------------------------
# Feasibility checks
# ---------------------------------------------------------------------------
def check_feasibility(p: dict) -> tuple[bool, str, dict, str]:
    """
    Returns (ok, fail_reason, computed_values, bearing_warnings).

    Hard failures (FAIL status):
      - 4-bar closure fails at every swept angle
      - valid stroke < MIN_STROKE_DEG

    Bearing warnings (logged but do NOT fail the run — may be workable in mech design):
      - tibia stub < MIN_TIBIA_SPAN_MM  (C–E spacing for 608 bearing)
      - F-to-motor-housing clearance < MIN_AF_CLEAR_MM
    """
    computed  = {}
    warn_list = []

    # ── Bearing warning 1: tibia stub spacing ───────────────────────────────
    stub_mm = p['L_stub'] * 1000.0
    if stub_mm < MIN_TIBIA_SPAN_MM:
        warn_list.append(f"tibia_stub {stub_mm:.1f}mm<{MIN_TIBIA_SPAN_MM:.0f}mm")

    # ── Bearing warning 2: F-bearing clearance from motor housing ───────────
    AF_mm        = math.sqrt(p['F_X']**2 + (p['F_Z'] - p['A_Z'])**2) * 1000.0
    af_clearance = AF_mm - MOTOR_R_MM
    computed['AF_mm']                 = round(AF_mm, 2)
    computed['AF_motor_clearance_mm'] = round(af_clearance, 2)
    if af_clearance < MIN_AF_CLEAR_MM:
        warn_list.append(f"AF_clear {af_clearance:.1f}mm<{MIN_AF_CLEAR_MM:.1f}mm")

    bearing_warnings = "|".join(warn_list)

    # ── Hard check: 4-bar closure ────────────────────────────────────────────
    angles = np.linspace(-2.5, 0.5, 1200)
    valid  = [q for q in angles if solve_ik(q, p) is not None]
    if not valid:
        return False, "4-bar closure fails at every swept angle", computed, bearing_warnings

    # ── Hard check: stroke wide enough ──────────────────────────────────────
    stroke_deg = abs(math.degrees(valid[-1] - valid[0]))
    if stroke_deg < MIN_STROKE_DEG:
        return False, (f"valid stroke only {stroke_deg:.1f}° < {MIN_STROKE_DEG:.0f}° minimum"), computed, bearing_warnings

    return True, "", computed, bearing_warnings


# ---------------------------------------------------------------------------
# Auto-find jump stroke
# ---------------------------------------------------------------------------
def find_stroke(p: dict) -> tuple[float, float] | None:
    """
    Sweep hip angles → find Q_retracted (leg short, W highest) and
    Q_extended (leg long, W lowest) within the valid range.
    Returns (Q_retracted, Q_extended) or None.
    """
    angles = np.linspace(-2.5, 0.5, 1200)
    results = [(q, solve_ik(q, p)) for q in angles]
    valid   = [(q, r) for q, r in results if r is not None]

    if len(valid) < 20:
        return None

    qs   = [q for q, _ in valid]
    w_zs = [r['W_z'] for _, r in valid]

    # Trim safety margin from each end, and enforce retraction angle constraint
    q_lo = qs[0]  + STROKE_MARGIN_RAD
    q_hi = min(qs[-1] - STROKE_MARGIN_RAD, Q_HIP_RETRACT_MAX)
    if q_lo >= q_hi:
        return None

    trimmed = [(q, wz) for q, wz in zip(qs, w_zs) if q_lo <= q <= q_hi]
    if not trimmed:
        return None

    # Q_retracted: angle where W_z is HIGHEST (leg most retracted)
    q_ret = max(trimmed, key=lambda x: x[1])[0]
    # Q_extended:  angle where W_z is LOWEST  (leg most extended)
    q_ext = min(trimmed, key=lambda x: x[1])[0]

    # Sanity: retracted should be less negative than extended
    if q_ret < q_ext:
        q_ret, q_ext = q_ext, q_ret

    return float(q_ret), float(q_ext)


# ---------------------------------------------------------------------------
# Validate jump stroke geometry
# ---------------------------------------------------------------------------
def validate_stroke(p: dict, Q_ret: float, Q_ext: float) -> tuple[bool, str]:
    """
    Two geometric checks that find_stroke cannot guarantee:

    1. Jump direction: W_z must DROP from Q_ret to Q_ext (body rises when leg extends).
       find_stroke's sanity-check can swap Q_ret/Q_ext to fix motor direction, but after
       the swap W_z(Q_ret) may still be LOWER than W_z(Q_ext) → body would go down.

    2. Wheel under box: at Q_ret (starting position) the wheel centre X must be within
       the body box footprint ± W_X_MARGIN.  Geometries where the wheel is far to the
       side cannot be balanced.

    Returns (ok, reason).  Call after find_stroke succeeds.
    """
    ik_ret = solve_ik(Q_ret, p)
    ik_ext = solve_ik(Q_ext, p)
    if ik_ret is None or ik_ext is None:
        return False, "IK failed at stroke endpoints"

    # Check 1: jump direction (body must rise)
    dz = ik_ret['W_z'] - ik_ext['W_z']   # positive = wheel moves down = body rises
    if dz < MIN_JUMP_DZ:
        return False, (f"jump direction wrong: W_z delta={dz*1000:.1f}mm "
                       f"(need >{MIN_JUMP_DZ*1000:.0f}mm drop, "
                       f"Q_ret W_z={ik_ret['W_z']*1000:.1f}mm "
                       f"Q_ext W_z={ik_ext['W_z']*1000:.1f}mm)")

    # Check 2: wheel under box at Q_ret
    w_x_ret  = ik_ret['W'][0]
    wx_mm    = w_x_ret * 1000.0
    limit_mm = (BOX_HALF_X + W_X_MARGIN) * 1000.0
    if abs(w_x_ret) > BOX_HALF_X + W_X_MARGIN:
        return False, (f"wheel not under box at Q_ret: "
                       f"W_x={wx_mm:.1f}mm (limit ±{limit_mm:.0f}mm)")

    return True, ""


# ---------------------------------------------------------------------------
# Dynamic XML builder
# ---------------------------------------------------------------------------
def build_xml(p: dict) -> str:
    L_f = p['L_femur']; L_s = p['L_stub']; L_t = p['L_tibia']
    Lc  = p['Lc'];      F_X = p['F_X'];    F_Z = p['F_Z']; A_Z = p['A_Z']
    # Tibia geom: spans E (stub end, +L_s above C) down to W (-L_t below C)
    tib_cz  = (L_s - L_t) / 2.0    # geom centre z in tibia frame (negative)
    tib_hsz = (L_s + L_t) / 2.0    # geom half-size z
    # Motor body visual: centre is half motor-length (21.5 mm) inboard of leg plane
    motor_y = LEG_Y - 0.0215
    # Wheel mass split: Maytech hub motor (fixed) + TPU tyre
    maytech_mass = min(MOTOR_MASS, p['m_wheel'])
    tyre_mass    = max(0.005, p['m_wheel'] - maytech_mass)

    return f"""<mujoco model="4bar_jump_eval">
  <option gravity="0 0 -9.81" timestep="0.000025"
          solver="Newton" iterations="200" tolerance="1e-10"/>

  <visual>
    <global offwidth="1280" offheight="720" azimuth="270" elevation="0"/>
  </visual>

  <worldbody>
    <light name="main" pos="0 -1 1" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>
    <light name="fill" pos="0  1 1" dir="0 -1 -1" diffuse="0.3 0.3 0.3"/>

    <!-- Ground plane -->
    <geom name="ground" type="plane" pos="0 0 0" size="2 2 0.01"
          rgba="0.3 0.3 0.3 1" contype="1" conaffinity="1"
          friction="0.8 0.005 0.0001"/>

    <!-- XYZ axis markers -->
    <geom name="axis_x" type="cylinder" fromto="0 0 0.001  0.06 0 0.001"
          size="0.003" rgba="1 0 0 1" contype="0" conaffinity="0" mass="0"/>
    <geom name="axis_y" type="cylinder" fromto="0 0 0.001  0 0.06 0.001"
          size="0.003" rgba="0 1 0 1" contype="0" conaffinity="0" mass="0"/>
    <geom name="axis_z" type="cylinder" fromto="0 0 0.001  0 0 0.061"
          size="0.003" rgba="0 0 1 1" contype="0" conaffinity="0" mass="0"/>

    <!-- Body box: slides vertically only -->
    <body name="box" pos="0 0 0.40">
      <joint name="body_slide" type="slide" axis="0 0 1" pos="0 0 0"/>

      <geom name="box_geom" type="box" size="0.05 0.10 0.05"
            rgba="0.29 0.56 0.85 0.55" contype="0" conaffinity="0"
            mass="{p['m_box']:.4f}"/>

      <!-- AK45-10 motor: Phi53 mm x 43 mm -->
      <geom name="motor_geom" type="cylinder"
            pos="0 {motor_y:.4f} {A_Z:.5f}" euler="90 0 0"
            size="0.0265 0.0215" rgba="0.87 0.36 0.36 0.55"
            contype="0" conaffinity="0" mass="{MOTOR_MASS:.3f}"/>

      <!-- Coupler pivot F -->
      <body name="coupler" pos="{F_X:.5f} {LEG_Y:.4f} {F_Z:.5f}">
        <joint name="hinge_F" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.001"/>
        <!-- 608 bearing at F: OD=22 mm, width=7 mm (2x half-width=10 mm) -->
        <geom type="cylinder" pos="0 0 0" euler="90 0 0" size="0.011 0.010"
              rgba="0 1 1 0.9" contype="0" conaffinity="0" mass="0"/>
        <geom name="coupler_geom" type="box"
              pos="{-Lc/2:.5f} 0 0" size="{Lc/2:.5f} 0.01 0.01"
              rgba="0.30 0.90 0.90 0.55" contype="0" conaffinity="0"
              mass="{p['m_coupler']:.4f}"/>
      </body>

      <!-- Hip motor pivot A -> femur -->
      <body name="femur" pos="0 {LEG_Y:.4f} {A_Z:.5f}">
        <joint name="hip" type="hinge" axis="0 1 0" pos="0 0 0"
               range="-180 180" armature="0.01" damping="0.05"/>
        <!-- Yellow dot at A (hip pivot) -->
        <geom type="cylinder" pos="0 0 0" euler="90 0 0" size="0.005 0.015"
              rgba="1 1 0 0.9" contype="0" conaffinity="0" mass="0"/>
        <geom name="femur_geom" type="box"
              pos="{-L_f/2:.5f} 0 0" size="{L_f/2:.5f} 0.01 0.01"
              rgba="0.94 0.75 0.25 0.55" contype="0" conaffinity="0"
              mass="{p['m_femur']:.4f}"/>
        <!-- 608 bearing at C (knee): OD=22 mm. Named for ground collision check. -->
        <geom name="knee_geom" type="cylinder" pos="{-L_f:.5f} 0 0" euler="90 0 0" size="0.011 0.010"
              rgba="1 1 1 0.9" contype="0" conaffinity="0" mass="0"/>

        <!-- Knee pivot C -> tibia -->
        <body name="tibia" pos="{-L_f:.5f} 0 0">
          <joint name="knee_joint" type="hinge" axis="0 1 0" pos="0 0 0"
                 range="-60 60" damping="0.001"/>
          <geom name="tibia_geom" type="box"
                pos="0 0 {tib_cz:.5f}" size="0.010 0.010 {tib_hsz:.5f}"
                rgba="0.75 0.45 0.90 0.55" contype="0" conaffinity="0"
                mass="{p['m_tibia']:.4f}"/>
          <!-- 608 bearing at E (tibia-coupler): OD=22 mm. Named for collision check. -->
          <geom name="stub_geom" type="cylinder" pos="0 0 {L_s:.5f}" euler="90 0 0" size="0.011 0.010"
                rgba="1 0.3 0.3 0.9" contype="0" conaffinity="0" mass="0"/>

          <!-- Wheel centre W -->
          <body name="wheel_asm" pos="0 0 {-L_t:.5f}">
            <joint name="wheel_spin" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.001"/>
            <!-- Maytech MTO7052 motor body: visual -->
            <geom name="motor_body_geom" type="cylinder"
                  pos="0 0.036 0" euler="90 0 0"
                  size="0.035 0.026" rgba="0.15 0.15 0.15 0.55"
                  contype="0" conaffinity="0" mass="{maytech_mass:.4f}"/>
            <!-- TPU tyre: Phi150 mm x 30 mm, collides with ground -->
            <geom name="wheel_tire_geom" type="cylinder"
                  pos="0 0.036 0" euler="90 0 0"
                  size="0.075 0.015" rgba="0.25 0.70 0.35 0.80"
                  contype="1" conaffinity="1"
                  friction="0.8 0.01 0.001"
                  solref="0.01 1.0" solimp="0.9 0.99 0.001"
                  mass="{tyre_mass:.4f}"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- 4-bar closure: eq_data anchors overridden in Python after loading -->
  <equality>
    <connect name="4bar_close" body1="coupler" body2="tibia"
             anchor="0 0 0"
             solref="0.00005 1" solimp="0.9999 0.9999 0.0001"/>
  </equality>

  <actuator>
    <motor name="hip_motor" joint="hip" ctrlrange="-7.0 7.0" gear="1"/>
  </actuator>

  <statistic center="0 0 0.30" extent="0.75"/>
</mujoco>"""


# ---------------------------------------------------------------------------
# Headless simulation
# ---------------------------------------------------------------------------
def run_headless(p: dict, Q_ret: float, Q_ext: float,
                 kp: float | None = None,
                 kd: float | None = None,
                 jump_ramp_s: float | None = None, 
                 xml_string: str | None = None) -> tuple[float, str]:
    """
    Run headless MuJoCo sim. 
    Returns (max_jump_height_mm, fail_reason_str).
    If success, fail_reason is empty string.
    Stops after the first successful landing (one jump cycle).
    kp / kd / jump_ramp_s override module-level constants when provided.
    xml_string overrides build_xml(p) when provided (for validation against original XML).
    """
    _KP   = kp          if kp          is not None else KP
    _KD   = kd          if kd          is not None else KD
    _RAMP = jump_ramp_s if jump_ramp_s is not None else JUMP_RAMP_S

    xml   = xml_string if xml_string is not None else build_xml(p)
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)

    # Match visual sim: apply one control decision per CTRL_HZ interval,
    # then batch the physics steps between decisions.
    physics_hz      = round(1.0 / model.opt.timestep)
    steps_per_ctrl  = max(1, physics_hz // CTRL_HZ)

    # Override equality constraint anchors
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "4bar_close")
    model.eq_data[eq_id, 0:3] = [-p['Lc'], 0.0,        0.0       ]
    model.eq_data[eq_id, 3:6] = [  0.0,    0.0,  p['L_stub']     ]

    def _jqpos(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid]
    def _jdof(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_dofadr[jid]
    def _bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    s_slide   = _jqpos("body_slide")
    s_hF      = _jqpos("hinge_F")
    s_hip     = _jqpos("hip")
    s_knee    = _jqpos("knee_joint")
    d_hip     = _jdof("hip")
    wheel_bid = _bid("wheel_asm")

    # Initialise pose at a neutral "pre-crouch" angle, on the ground.
    q_crouch_start = -0.8
    ik = solve_ik(q_crouch_start, p)
    if ik is None:
        # Fallback to starting directly at Q_ret if pre-crouch is invalid
        q_crouch_start = Q_ret
        ik = solve_ik(q_crouch_start, p)
    if ik is None:
        return 0.0, "sim init failed: IK invalid at start"

    data.qpos[s_hF]   = ik['q_coupler_F']
    data.qpos[s_hip]  = ik['q_hip']
    data.qpos[s_knee] = ik['q_knee']
    data.ctrl[0]      = 0.0
    mujoco.mj_forward(model, data)
    # Place wheel on ground
    wheel_z = float(data.xpos[wheel_bid][2])
    data.qpos[s_slide] += WHEEL_R - wheel_z
    mujoco.mj_forward(model, data)

    # State machine
    state          = CROUCHING
    crouch_start_t = 0.0
    jump_start_t   = 0.0
    grounded_since = None
    max_height_m   = 0.0
    torque         = 0.0

    # Safety geoms to check for ground collision (links/joints cannot go below 0)
    gid_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "knee_geom")
    gid_stub = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "stub_geom")
    
    HEADLESS_STABLE_S = 0.10   # shorter settle window than visual sim (0.4s)

    while float(data.time) < SIM_DURATION_S:
        sim_t    = float(data.time)
        hip_q    = float(data.qpos[s_hip])
        omega    = float(data.qvel[d_hip])
        wheel_z  = float(data.xpos[wheel_bid][2])
        grounded = wheel_z < (WHEEL_R + GROUNDED_TOL)

        if state == CROUCHING:
            alpha_t = min(1.0, (sim_t - crouch_start_t) / CROUCH_TIME)
            q_des   = q_crouch_start + (Q_ret - q_crouch_start) * alpha_t
            torque  = _KP * (q_des - hip_q) + _KD * (0.0 - omega)
            torque  = float(np.clip(torque, -HOLD_TORQUE, HOLD_TORQUE))
            if alpha_t >= 1.0:
                state = JUMPING
                jump_start_t = sim_t

        elif state == JUMPING:
            ramp_in  = min(1.0, (sim_t - jump_start_t) / _RAMP) if _RAMP > 0 else 1.0
            ramp_out = min(1.0, max(0.0, (hip_q - Q_ext) / JUMP_RAMPDOWN))
            torque   = -MAX_TORQUE * ramp_in * ramp_out * max(0.0, 1.0 - abs(omega) / OMEGA_MAX)
            if not grounded or hip_q <= Q_ext + 0.02:
                state = LANDING; torque = 0.0; grounded_since = None

        elif state == LANDING:
            torque = _KP * (Q_ext - hip_q) + _KD * (0.0 - omega)
            torque = float(np.clip(torque, -HOLD_TORQUE, HOLD_TORQUE))
            if grounded:
                if grounded_since is None:
                    grounded_since = sim_t
                elif sim_t - grounded_since >= HEADLESS_STABLE_S:
                    break   # one jump cycle complete
            else:
                grounded_since = None

        data.ctrl[0] = torque
        for _ in range(steps_per_ctrl):
            mujoco.mj_step(model, data)

        # Check mechanical constraints (no joints/links allowed below ground)
        # We check the center of the bearing geoms. If center < 0, it's a hard fail.
        if (gid_knee >= 0 and data.geom_xpos[gid_knee][2] < 0.0) or \
           (gid_stub >= 0 and data.geom_xpos[gid_stub][2] < 0.0):
            return 0.0, "mechanical crash: joint below ground"

        w_z = float(data.xpos[wheel_bid][2])
        max_height_m = max(max_height_m, w_z - WHEEL_R)

    return max_height_m * 1000.0, ""


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------
def evaluate(p: dict, label: str = "", forced_run_id: int | None = None, write_csv: bool = True) -> dict:
    """
    Run full pipeline: feasibility → stroke → sim → CSV log.
    Returns the result row dict.
    """
    run_id = forced_run_id if forced_run_id is not None else _next_run_id()
    ts     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Derived geometry values (always logged)
    AF_mm        = math.sqrt(p['F_X']**2 + (p['F_Z'] - p['A_Z'])**2) * 1000.0
    af_clearance = AF_mm - MOTOR_R_MM
    total_mass_g = (p['m_box'] + p['m_femur'] + p['m_tibia'] +
                    p['m_coupler'] + p['m_wheel'] + MOTOR_MASS) * 1000.0

    row = dict(
        run_id=run_id, label=label, timestamp=ts,
        L_femur_mm=round(p['L_femur']*1000,2),
        L_stub_mm =round(p['L_stub'] *1000,2),
        L_tibia_mm=round(p['L_tibia']*1000,2),
        Lc_mm     =round(p['Lc']     *1000,2),
        F_X_mm    =round(p['F_X']    *1000,2),
        F_Z_mm    =round(p['F_Z']    *1000,2),
        A_Z_mm    =round(p['A_Z']    *1000,2),
        AF_mm     =round(AF_mm, 2),
        AF_motor_clearance_mm=round(af_clearance, 2),
        m_box_g   =round(p['m_box']    *1000,1),
        m_femur_g =round(p['m_femur']  *1000,1),
        m_tibia_g =round(p['m_tibia']  *1000,1),
        m_coupler_g=round(p['m_coupler']*1000,1),
        m_wheel_g =round(p['m_wheel']  *1000,1),
        total_mass_g=round(total_mass_g,1),
    )

    # Feasibility
    ok, reason, computed, bearing_warnings = check_feasibility(p)
    row.update(computed)
    row['bearing_warnings'] = bearing_warnings
    if bearing_warnings:
        print(f"  WARN bearing: {bearing_warnings}")

    if not ok:
        row.update(Q_retracted_rad="", Q_extended_rad="", stroke_deg="",
                   jump_height_mm="", status="FAIL", fail_reason=reason)
        print(f"[{run_id}] {label:20s}  FAIL  —  {reason}")
        if write_csv: _log_csv(row)
        return row

    # Stroke
    stroke = find_stroke(p)
    if stroke is None:
        row.update(Q_retracted_rad="", Q_extended_rad="", stroke_deg="",
                   jump_height_mm="", status="FAIL", fail_reason="stroke search failed")
        print(f"[{run_id}] {label:20s}  FAIL  —  stroke search failed")
        if write_csv: _log_csv(row)
        return row

    Q_ret, Q_ext = stroke

    # Validate jump direction and wheel-under-box
    sv_ok, sv_reason = validate_stroke(p, Q_ret, Q_ext)
    if not sv_ok:
        row.update(Q_retracted_rad="", Q_extended_rad="", stroke_deg="",
                   jump_height_mm="", status="FAIL", fail_reason=sv_reason)
        print(f"[{run_id}] {label:20s}  FAIL  —  {sv_reason}")
        if write_csv: _log_csv(row)
        return row

    stroke_deg   = abs(math.degrees(Q_ret - Q_ext))
    row['Q_retracted_rad'] = round(Q_ret, 5)
    row['Q_extended_rad']  = round(Q_ext, 5)
    row['stroke_deg']      = round(stroke_deg, 2)

    # Headless sim
    h_mm, sim_err = run_headless(p, Q_ret, Q_ext)
    if sim_err:
        row.update(jump_height_mm="", status="FAIL", fail_reason=sim_err)
        print(f"[{run_id}] {label:20s}  FAIL  —  {sim_err}")
        if write_csv: _log_csv(row)
        return row

    row['jump_height_mm'] = round(h_mm, 2)
    row['status']         = "PASS"
    row['fail_reason']    = ""
    print(f"[{run_id}] {label:20s}  PASS  —  {h_mm:.1f} mm jump  "
          f"(stroke {stroke_deg:.1f}°  Q_ret={Q_ret:.3f}  Q_ext={Q_ext:.3f})")
    if write_csv: _log_csv(row)
    return row


# ---------------------------------------------------------------------------
# Single experiment: baseline vs stub +1 mm
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import copy, sys

    # ── Validation mode ──────────────────────────────────────────────────────
    # python eval_jump.py --validate
    # Replicates 4bar_jump_sim/4bar_jump_sim.py exactly (no bearing constraint)
    # and compares to its known output of ~45.8 mm.
    if "--validate" in sys.argv:
        print("=" * 60)
        print("VALIDATION: match 4bar_jump_sim/4bar_jump_sim.py")
        print("  Geometry : L_stub=15mm  F_X=-15mm  F_Z=2.5mm")
        print("  Control  : KP=80  KD=3  no torque ramp")
        print("  Q stroke : fixed  Q_ret=-0.58  Q_ext=-1.27")
        print("  Reference: 45.8 mm  (from 4bar_jump_sim original)")
        print("=" * 60)

        p_orig = {
            **copy.deepcopy(DEFAULT),
            "L_stub": 0.015,   # original (violates bearing constraint — intentional)
            "F_X":   -0.015,
            "F_Z":    0.0025,
        }

        # Auto-find stroke for information
        stroke_auto = find_stroke(p_orig)
        if stroke_auto:
            print(f"  Auto stroke: Q_ret={stroke_auto[0]:.3f}  Q_ext={stroke_auto[1]:.3f}"
                  f"  ({abs(math.degrees(stroke_auto[0]-stroke_auto[1])):.1f} deg)")
        else:
            print("  Auto stroke: not found")

        # Load original XML (uses hinge_F=0.5, solref=0.002, friction=0 0 0)
        orig_xml_path = os.path.join(_DIR, "..", "4bar_jump_sim", "4bar_jump.xml")
        orig_xml = None
        if os.path.exists(orig_xml_path):
            with open(orig_xml_path) as f:
                orig_xml = f.read()
            print(f"  Loaded original XML: {os.path.normpath(orig_xml_path)}")
        else:
            print(f"  WARNING: original XML not found at {orig_xml_path}")

        # A: Original XML + original control  (should match reference 1:1)
        if orig_xml:
            h_orig_xml, _ = run_headless(p_orig, Q_ret=-0.58, Q_ext=-1.27,
                                      kp=80, kd=3, jump_ramp_s=0,
                                      xml_string=orig_xml)
            print(f"\n  [A] Original XML + KP=80/KD=3/no-ramp  : {h_orig_xml:.1f} mm"
                  f"  <-- should match reference")

        # B: build_xml + original control  (isolates XML parameter differences)
        h_fixed, _ = run_headless(p_orig, Q_ret=-0.58, Q_ext=-1.27,
                               kp=80, kd=3, jump_ramp_s=0)
        print(f"  [B] build_xml  + KP=80/KD=3/no-ramp  : {h_fixed:.1f} mm"
              f"  <-- diff vs A = XML params (damping/solref/friction)")

        # C: build_xml + auto Q (tests stroke finder)
        if stroke_auto:
            h_auto, _ = run_headless(p_orig,
                                  Q_ret=stroke_auto[0], Q_ext=stroke_auto[1],
                                  kp=80, kd=3, jump_ramp_s=0)
            print(f"  [C] build_xml  + auto Q             : {h_auto:.1f} mm"
                  f"  <-- diff vs B = stroke finder error")

        print(f"\n  Reference (4bar_jump_sim headless)    : see [A] above")
        raise SystemExit(0)

    # ── Baseline experiment: match 4bar_jump_sim/4bar_jump_sim.py ───────────
    # This uses the exact geometry from the visual sim (stub=15mm violates
    # bearing constraint but is logged as a warning, not a failure).
    VISUAL_SIM_REFERENCE_MM = 31.9   # measured from 4bar_jump_sim.py with realistic mass/torque

    VISUAL_SIM_PARAMS = {
        **copy.deepcopy(DEFAULT),
        "L_stub": 0.015,   # original visual-sim value (bearing constraint violated — OK)
        "F_X":   -0.015,
        "F_Z":    0.0025,
    }

    print("=" * 60)
    print("Run 1: visual-sim baseline (4bar_jump_sim.py geometry)")
    print(f"  stub=15mm  F_X=-15mm  F_Z=2.5mm")
    print(f"  Reference from 4bar_jump_sim.py: {VISUAL_SIM_REFERENCE_MM:.1f} mm")
    print(f"CSV output: {CSV_PATH}")
    print("=" * 60)

    r0 = evaluate(VISUAL_SIM_PARAMS, label="visual_sim_baseline")

    print()
    print("-" * 60)
    if r0.get("status") == "PASS":
        diff = r0["jump_height_mm"] - VISUAL_SIM_REFERENCE_MM
        print(f"Eval result : {r0['jump_height_mm']:.1f} mm")
        print(f"Visual sim  : {VISUAL_SIM_REFERENCE_MM:.1f} mm  (4bar_jump_sim.py)")
        print(f"Difference  : {diff:+.1f} mm  ({diff/VISUAL_SIM_REFERENCE_MM*100:+.1f}%)")
    else:
        print(f"FAIL: {r0.get('fail_reason','')}")
