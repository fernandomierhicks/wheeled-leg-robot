"""scenarios.py — Headless simulation scenarios for LQR_Control_optimization.

Balance controller PD gains are passed in via `gains` dict and wired directly
into the controller.  The evolutionary optimizer searches over these gains.
"""
import math
import datetime
import numpy as np
import mujoco

from sim_config import (
    ROBOT, Q_NOM, Q_RET, Q_EXT, WHEEL_R,
    HIP_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT, WHEEL_OMEGA_NOLOAD, WHEEL_KT,
    BATT_V_NOM, HIP_KT_OUTPUT, BATT_I_QUIESCENT,
    LEG_K_S, LEG_B_S,
    LEG_K_ROLL, LEG_D_ROLL, ROLL_NOISE_STD_RAD, HIP_SAFE_MIN, HIP_SAFE_MAX,
    MAX_PITCH_CMD,
    PITCH_NOISE_STD_RAD, PITCH_RATE_NOISE_STD_RAD_S, ACCEL_NOISE_STD,
    CTRL_STEPS,
    LQR_Q_PITCH, LQR_Q_PITCH_RATE, LQR_Q_VEL, LQR_R,
    VELOCITY_PI_KP as _DEFAULT_KP_V,
    VELOCITY_PI_KI as _DEFAULT_KI_V,
    VELOCITY_PI_THETA_MAX, VELOCITY_PI_INT_MAX, THETA_REF_RATE_LIMIT,
    YAW_PI_KP as _DEFAULT_KP_YAW,
    YAW_PI_KI as _DEFAULT_KI_YAW,
    YAW_PI_TORQUE_MAX, YAW_PI_INT_MAX,
    DRIVE_SLOW_SPEED, DRIVE_MEDIUM_SPEED, DRIVE_DURATION, DRIVE_REV_TIME,
    OBSTACLE_HEIGHT, OBSTACLE_DURATION,
    PITCH_STEP_RAD, SCENARIO_1_DURATION, SCENARIO_2_DURATION, SCENARIO_3_DURATION, SCENARIO_5_DURATION,
    S2_DIST1_TIME, S2_DIST1_FORCE, S2_DIST1_DUR,
    S2_DIST2_TIME, S2_DIST2_FORCE, S2_DIST2_DUR,
    SCENARIO_4_DURATION, LEG_CYCLE_PERIOD, LEG_CYCLE_Q_RET,
    S5_BUMPS,
    SCENARIO_6_DURATION, YAW_TURN_RATE, YAW_ERR_START,
    SCENARIO_7_DURATION, DRIVE_TURN_SPEED, DRIVE_TURN_YAW_RATE,
    SCENARIO_8_DURATION, S8_DRIVE_SPEED, S8_BUMPS,
)
from physics import build_xml, build_assets, solve_ik, get_equilibrium_pitch
from run_log import log_run, next_run_id, CSV_PATH
from lqr_design import interpolate_gains, compute_gain_table
from battery_model import BatteryModel

# ── Motor back-EMF taper ─────────────────────────────────────────────────────
def motor_taper(tau_cmd: float, omega_wheel: float,
                v_batt: float = BATT_V_NOM) -> float:
    """Clamp wheel torque by linear back-EMF taper, voltage-scaled.

    ω_noload ∝ V_terminal — lower battery voltage reduces available top speed and
    therefore also reduces the torque available at high wheel speeds.
    At the nominal rated voltage (BATT_V_NOM) the behaviour is identical to before.
    """
    omega_noload = WHEEL_OMEGA_NOLOAD * (v_batt / BATT_V_NOM)
    taper = max(0.0, 1.0 - abs(omega_wheel) / omega_noload)
    t_max = WHEEL_TORQUE_LIMIT * taper
    return float(np.clip(tau_cmd, -t_max, t_max))


def _motor_currents(tau_whl_L: float, tau_whl_R: float,
                    tau_hip_L: float, tau_hip_R: float) -> float:
    """Sum all motor currents plus quiescent electronics load [A].

    Uses the commanded (clamped) torques as a proxy for actual phase current.
    I_wheel = |τ| / Kt_wheel,  I_hip = |τ_output| / Kt_output_shaft.
    """
    I_whl = (abs(tau_whl_L) + abs(tau_whl_R)) / WHEEL_KT
    I_hip = (abs(tau_hip_L) + abs(tau_hip_R)) / HIP_KT_OUTPUT
    return I_whl + I_hip + BATT_I_QUIESCENT


# ── Controller mode ──────────────────────────────────────────────────────────
USE_PD_CONTROLLER = True  # Toggle between PD (True) and LQR (False)
USE_VELOCITY_PI   = True  # When True, outer VelocityPI loop provides theta_ref for LQR
                           # Set False to optimize LQR gains only (no outer loop)
USE_YAW_PI        = True  # When True, YawPI provides differential tau_yaw
                           # Set False to run symmetric torque only (no turning)

# ── Velocity PI gains — mutable module globals (overridden by optimizer worker) ─
# Outer loop: velocity error → theta_ref (lean angle command for LQR).
# Positive theta_ref = lean forward = drive forward.
VELOCITY_PI_KP = _DEFAULT_KP_V   # [rad/(m/s)]
VELOCITY_PI_KI = _DEFAULT_KI_V   # [rad/m]

# ── Yaw PI gains — mutable module globals (overridden by optimizer worker) ────
# Differential torque: tau_yaw = YawPI(omega_desired - omega_measured)
# tau_L = tau_sym + tau_yaw,  tau_R = tau_sym − tau_yaw
YAW_PI_KP_GAIN = _DEFAULT_KP_YAW   # [N·m / (rad/s)]
YAW_PI_KI_GAIN = _DEFAULT_KI_YAW   # [N·m / rad]

# ── Roll leveling gains — mutable module globals (overridden by optimizer) ────
# Differential hip offset: δq = K_ROLL*roll + D_ROLL*roll_rate
# q_nom_L = q_nom_sym + δq,  q_nom_R = q_nom_sym - δq
LEG_K_ROLL_GAIN = LEG_K_ROLL   # [rad/rad]
LEG_D_ROLL_GAIN = LEG_D_ROLL   # [rad·s/rad]

# ── Initialize LQR gain table at module load ────────────────────────────────
LQR_K_TABLE = compute_gain_table(
    ROBOT,
    Q_diag=[LQR_Q_PITCH, LQR_Q_PITCH_RATE, LQR_Q_VEL],
    R_val=LQR_R
)

# ── Scenario parameters ──────────────────────────────────────────────────────
BALANCE_DURATION  = 5.0    # [s] total simulation time
SETTLE_THRESHOLD  = 2.0    # [deg] |pitch| below this = settled
SETTLE_WINDOW     = 0.5    # [s]  must stay settled this long to record
FALL_THRESHOLD    = 0.785  # [rad] ~45° — robot considered fallen

# ── Disturbance scenario parameters ───────────────────────────────────────────
DISTURBANCE_TIME  = 2.5    # [s] when to apply the impulse (mid-run, after settling)
DISTURBANCE_FORCE = 1.0    # [N] horizontal (X) impulse magnitude
DISTURBANCE_DUR   = 0.2    # [s] duration of force application

# ── Scenario 1 mid-run disturbances ───────────────────────────────────────────
S1_DIST1_TIME  = 2.0   # [s] first kick (forward)
S1_DIST1_FORCE = 4.0   # [N] horizontal (+X)
S1_DIST1_DUR   = 0.2   # [s]
S1_DIST2_TIME  = 3.0   # [s] second kick (backward, opposite direction)
S1_DIST2_FORCE = -4.0  # [N] horizontal (−X)
S1_DIST2_DUR   = 0.2   # [s]

# ── Fitness weights ──────────────────────────────────────────────────────────
W_RMS      = 1.0     # RMS pitch error [deg]
W_TRAVEL   = 1.0     # wheel travel [m] — penalise drifting/oscillating wheels
W_FALL     = 200.0   # fell-over penalty
W_RECOVERY = 1.0     # recovery from disturbance (RMS pitch error post-disturbance)
W_LIFTOFF  = 50.0    # penalty per second any wheel is off the ground (bouncing)
W_PITCH_RATE = 0.05   # 1_LQR_pitch_step: weight on ISE_pitch_rate (damps oscillation)

LIFTOFF_THRESHOLD = WHEEL_R + 0.005   # 5 mm above nominal contact = airborne

# ---------------------------------------------------------------------------
# Scenario building blocks — module-level so replay.py can import them
# ---------------------------------------------------------------------------
def s1_dist_fn(t: float) -> float:
    """S1: +4 N at t=2 s for 0.2 s, then −4 N at t=3 s for 0.2 s."""
    if S1_DIST1_TIME <= t < S1_DIST1_TIME + S1_DIST1_DUR:
        return S1_DIST1_FORCE
    if S1_DIST2_TIME <= t < S1_DIST2_TIME + S1_DIST2_DUR:
        return S1_DIST2_FORCE
    return 0.0


def s2_dist_fn(t: float) -> float:
    """S2: +1 N at t=2 s for 0.2 s, then −1 N at t=3 s for 0.2 s."""
    if S2_DIST1_TIME <= t < S2_DIST1_TIME + S2_DIST1_DUR:
        return S2_DIST1_FORCE
    if S2_DIST2_TIME <= t < S2_DIST2_TIME + S2_DIST2_DUR:
        return S2_DIST2_FORCE
    return 0.0


def s3_velocity_profile(t: float) -> float:
    """S3 staircase: 0 → +0.3 → +0.6 → +1.0 → −0.5 → −1.0 → 0 m/s."""
    if   t <  1.0: return  0.0
    elif t <  3.0: return  0.3
    elif t <  5.0: return  0.6
    elif t <  7.0: return  1.0
    elif t <  9.0: return -0.5
    elif t < 11.0: return -1.0
    else:          return  0.0

def _leg_cycle_profile(t: float) -> float:
    """Sinusoidal leg-height cycle between LEG_CYCLE_Q_RET and Q_EXT.

    Period = LEG_CYCLE_PERIOD (4 s).  Half-period = 2 s per extreme-to-extreme sweep.
    Starts at mid-stroke at t=0, first moves toward Q_EXT (extended/low).

    LEG_CYCLE_Q_RET = Q_NOM ≈ -0.676 rad  (retracted limit — avoids 4-bar instability)
    Q_EXT           = -1.432 rad           (extended — low)
    """
    center = (LEG_CYCLE_Q_RET + Q_EXT) / 2.0
    amp    = (Q_EXT - LEG_CYCLE_Q_RET) / 2.0   # negative: Q_EXT < LEG_CYCLE_Q_RET
    return center + amp * math.sin(2.0 * math.pi * t / LEG_CYCLE_PERIOD)


# ── Combined scenario weights (for LQR optimization) ────────────────────────
# Phase 2 combined: balance (gate) + disturbance + drive-slow + drive-medium + obstacle
W_BALANCE      = 0.10   # still balance (just a gate really)
W_DISTURBANCE  = 0.35   # disturbance recovery (proven)
W_DRIVE_SLOW   = 0.20   # 0.3 m/s fwd+bwd tracking
W_DRIVE_MED    = 0.20   # 0.8 m/s fwd+bwd tracking (harder)
W_DRIVE_STEP   = 0.15   # 3 cm floor step crossing robustness
# Sum = 1.00

# ── Drive/obstacle fitness weights ───────────────────────────────────────────
W_VEL_ERR  = 3.0    # RMS velocity tracking error [m/s] — 3× to penalise steady-state offset harder
W_YAW_ERR  = 3.0    # RMS yaw rate tracking error [rad/s] — same weight as velocity


# ---------------------------------------------------------------------------
# VelocityPI — outer loop: velocity error → lean angle command
# ---------------------------------------------------------------------------
class VelocityPI:
    """Converts velocity error into a lean-angle setpoint (theta_ref) for LQR.

    theta_ref > 0  →  lean forward  →  drive forward
    theta_ref < 0  →  lean backward →  drive backward

    The LQR state becomes: [pitch - pitch_ff + theta_ref, pitch_rate, wheel_vel - v_ref]
    so positive theta_ref shifts the equilibrium forward, making the LQR apply
    forward wheel torque to compensate.
    """
    def __init__(self, kp: float, ki: float, dt: float):
        self.kp  = kp
        self.ki  = ki
        self.dt  = dt
        self.integral = 0.0

    def update(self, v_desired_ms: float, v_measured_ms: float) -> float:
        v_err = v_desired_ms - v_measured_ms
        self.integral = float(np.clip(
            self.integral + v_err * self.dt,
            -VELOCITY_PI_INT_MAX, VELOCITY_PI_INT_MAX))
        theta_ref = float(np.clip(
            self.kp * v_err + self.ki * self.integral,
            -VELOCITY_PI_THETA_MAX, VELOCITY_PI_THETA_MAX))
        return theta_ref

    def reset(self):
        self.integral = 0.0


# ---------------------------------------------------------------------------
# YawPI — outer loop: yaw rate error → differential wheel torque
# ---------------------------------------------------------------------------
class YawPI:
    """Converts yaw rate error into differential wheel torque (tau_yaw).

    tau_yaw > 0  →  tau_L += tau_yaw, tau_R -= tau_yaw  →  left turn (CCW)
    tau_yaw < 0  →  tau_L -= |tau_yaw|, tau_R += |tau_yaw|  →  right turn (CW)

    Yaw rate measured from data.qvel[5] (world-frame ωz).
    Positive ωz = CCW = left turn when viewed from above.

    Orthogonal to LQR/VelocityPI: average wheel torque (tau_sym) is unaffected.
    """
    def __init__(self, kp: float, ki: float, dt: float):
        self.kp  = kp
        self.ki  = ki
        self.dt  = dt
        self.integral = 0.0

    def update(self, omega_desired: float, omega_measured: float) -> float:
        err = omega_desired - omega_measured
        self.integral = float(np.clip(
            self.integral + err * self.dt,
            -YAW_PI_INT_MAX, YAW_PI_INT_MAX))
        tau_yaw = self.kp * err + self.ki * self.integral
        return float(np.clip(tau_yaw, -YAW_PI_TORQUE_MAX, YAW_PI_TORQUE_MAX))

    def reset(self):
        self.integral = 0.0


# ---------------------------------------------------------------------------
# Shared sensor / controller helpers (used by both scenarios and replay)
# ---------------------------------------------------------------------------
def get_pitch_and_rate(data, box_bid: int, d_pitch: int):
    """Return (pitch_rad, pitch_rate_rad_s) from body world quaternion."""
    q = data.xquat[box_bid]
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q[0]*q[2] - q[3]*q[1]))))
    return pitch, float(data.qvel[d_pitch])


def balance_torque(pitch: float, pitch_rate: float, pitch_integral: float,
                   odo_x: float, wheel_vel: float, hip_q_avg: float,
                   dt: float, gains: dict) -> tuple:
    """PD balance controller.  Gains are passed in from the optimizer.

    gains keys: KP, KD, KP_pos, KP_vel
    Returns (tau_wheel, new_pitch_integral, new_odo_x).
    """
    Kp     = gains['KP']
    Kd     = gains['KD']
    Kp_pos = gains['KP_pos']
    Kp_vel = gains['KP_vel']

    pitch_ff = get_equilibrium_pitch(ROBOT, hip_q_avg)
    vel_est  = (wheel_vel + pitch_rate) * WHEEL_R
    odo_x   += vel_est * dt
    pitch_fb = float(np.clip(
        -(Kp_pos * odo_x + Kp_vel * vel_est),
        -MAX_PITCH_CMD, MAX_PITCH_CMD))
    target_pitch   = pitch_ff + pitch_fb
    pitch_error    = pitch - target_pitch
    pitch_integral = float(np.clip(pitch_integral + pitch_error * dt, -1.0, 1.0))
    u_bal      = Kp * pitch_error + Kd * pitch_rate
    tau_wheel  = float(np.clip(u_bal, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT))
    return tau_wheel, pitch_integral, odo_x


def lqr_torque(pitch: float, pitch_rate: float, wheel_vel: float,
               hip_q_avg: float, v_ref: float = 0.0,
               theta_ref: float = 0.0) -> float:
    """3-state LQR balance controller with gain scheduling.

    State: [pitch − pitch_ff + theta_ref, pitch_rate, wheel_vel − v_ref]

    theta_ref (lean command from VelocityPI):
      +  → lean forward  → drive forward  (positive = forward)
      0  → static balance (default)
      −  → lean backward → drive backward
    v_ref: wheel angular velocity feedforward [rad/s] (v_desired_ms / WHEEL_R)

    K is interpolated online based on hip angle (leg position).
    Returns tau_wheel (symmetric) for both wheels.
    """
    pitch_ff = get_equilibrium_pitch(ROBOT, hip_q_avg)
    K = interpolate_gains(LQR_K_TABLE, hip_q_avg)

    # State vector: pitch_error with lean command, pitch_rate, wheel_vel_error
    # theta_ref > 0 = forward drive command; subtract to create negative torque = forward motion
    # (empirically verified: negative wheel torque drives robot forward in this geometry)
    x = np.array([pitch - pitch_ff - theta_ref, pitch_rate, wheel_vel - v_ref])

    # Control: u = -K @ x
    u = float(-np.dot(K, x))

    # Clamp to motor limit
    tau_wheel = float(np.clip(u, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT))

    return tau_wheel


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------
def _build_model_and_data(p=None, obstacle_height=0.0, bumps=None, sandbox_obstacles=None):
    if p is None: p = ROBOT
    xml    = build_xml(p, obstacle_height=obstacle_height, bumps=bumps,
                       sandbox_obstacles=sandbox_obstacles)
    assets = build_assets()
    model  = mujoco.MjModel.from_xml_string(xml, assets)
    data   = mujoco.MjData(model)
    return model, data


def init_sim(model, data, p=None, q_hip_init=None):
    """Reset and place robot at q_hip_init (defaults to Q_NOM)."""
    if p is None: p = ROBOT
    if q_hip_init is None: q_hip_init = Q_NOM

    mujoco.mj_resetData(model, data)

    # Fix 4-bar closure anchors
    for side in ('L', 'R'):
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                   f"4bar_close_{side}")
        if eq_id >= 0:
            model.eq_data[eq_id, 0:3] = [-p['Lc'], 0.0, 0.0]
            model.eq_data[eq_id, 3:6] = [0.0, 0.0, p['L_stub']]

    def _jqp(name): return model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)]

    s_root   = _jqp("root_free")
    s_hF_L   = _jqp("hinge_F_L");    s_hF_R   = _jqp("hinge_F_R")
    s_hip_L  = _jqp("hip_L");        s_hip_R  = _jqp("hip_R")
    s_knee_L = _jqp("knee_joint_L"); s_knee_R = _jqp("knee_joint_R")

    ik = solve_ik(q_hip_init, p)
    if ik is None:
        raise RuntimeError(f"IK failed at q_hip={q_hip_init:.3f}")

    for s_hF, s_hip, s_knee in [
        (s_hF_L, s_hip_L, s_knee_L),
        (s_hF_R, s_hip_R, s_knee_R),
    ]:
        data.qpos[s_hF]   = ik['q_coupler_F']
        data.qpos[s_hip]  = ik['q_hip']
        data.qpos[s_knee] = ik['q_knee']

    mujoco.mj_forward(model, data)

    wheel_bid_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_asm_L")
    wheel_bid_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_asm_R")
    wz = (data.xpos[wheel_bid_L][2] + data.xpos[wheel_bid_R][2]) / 2.0
    data.qpos[s_root + 2] += WHEEL_R - wz

    theta = get_equilibrium_pitch(p, q_hip_init)
    data.qpos[s_root + 3] = math.cos(theta / 2.0)
    data.qpos[s_root + 4] = 0.0
    data.qpos[s_root + 5] = math.sin(theta / 2.0)
    data.qpos[s_root + 6] = 0.0

    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Shared LQR simulation loop (drive + obstacle scenarios)
# ---------------------------------------------------------------------------
def _run_sim_loop(model, data, duration: float, v_profile_fn,
                  dist_fn=None,
                  add_noise: bool = True, rng=None,
                  target_hip_q=None,
                  omega_profile_fn=None) -> dict:
    """Core LQR simulation loop used by drive and obstacle scenarios.

    v_profile_fn(t: float) -> v_target_ms [float]
        Returns desired linear velocity [m/s] at time t.
        Use lambda t: 0.0 for static balance.

    omega_profile_fn(t: float) -> omega_desired_rads [float]  (optional)
        Returns desired yaw rate [rad/s] at time t.
        None (default) → omega_desired = 0.0 → tau_yaw = 0.0 → symmetric drive.
        Positive = CCW = left turn when viewed from above.

    target_hip_q: float, callable(t)->float, or None.
        Sets the impedance controller's target hip angle each step.
        None (default) → hold Q_NOM as before.

    dist_fn: optional callable(t)->force [N] applied to body X axis.
    Returns a metrics dict with keys compatible with run_balance_* returns.
    """
    # Resolve hip target into a callable regardless of input type
    if target_hip_q is None:
        _hip_fn = lambda t: Q_NOM
    elif callable(target_hip_q):
        _hip_fn = target_hip_q
    else:
        _hip_fn = lambda t: target_hip_q
    if rng is None:
        rng = np.random.default_rng(None)

    # ── Address lookups ─────────────────────────────────────────────────────
    def _jqp(name): return model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _jdof(name): return model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _act(name):  return mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    def _bid(name):  return mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, name)

    d_root  = _jdof("root_free")
    d_pitch = d_root + 4
    d_yaw   = d_root + 5   # world-frame ωz; positive = CCW = left turn
    s_root  = _jqp("root_free")

    s_hip_L = _jqp("hip_L");   s_hip_R = _jqp("hip_R")
    d_hip_L = _jdof("hip_L");  d_hip_R = _jdof("hip_R")
    d_whl_L = _jdof("wheel_spin_L"); d_whl_R = _jdof("wheel_spin_R")

    act_hip_L   = _act("hip_act_L");   act_hip_R   = _act("hip_act_R")
    act_wheel_L = _act("wheel_act_L"); act_wheel_R = _act("wheel_act_R")

    box_bid     = _bid("box")
    wheel_bid_L = _bid("wheel_asm_L")
    wheel_bid_R = _bid("wheel_asm_R")

    # ── Controller state ────────────────────────────────────────────────────
    dt = model.opt.timestep * CTRL_STEPS
    vel_pi = VelocityPI(kp=VELOCITY_PI_KP, ki=VELOCITY_PI_KI, dt=dt)
    yaw_pi = YawPI(kp=YAW_PI_KP_GAIN, ki=YAW_PI_KI_GAIN, dt=dt)
    _prev_theta_ref = 0.0

    # 1-step sensor delay buffer (0 ms — update before use)
    _pitch_d      = get_equilibrium_pitch(ROBOT, Q_NOM)
    _pitch_rate_d = 0.0
    _wheel_vel_d  = 0.0

    # ── Metric accumulators ─────────────────────────────────────────────────
    pitch_sq_sum      = 0.0
    roll_sq_sum       = 0.0   # body roll RMS
    max_roll          = 0.0   # peak |roll| over the run [deg]
    vel_sq_sum        = 0.0   # velocity tracking error
    max_pitch         = 0.0
    wheel_travel_m    = 0.0
    wheel_liftoff_s   = 0.0
    n_samples         = 0
    n_vel             = 0
    survived_s        = duration
    settle_time       = duration
    settled           = False
    settle_start      = None

    VEL_ERR_START = 1.0   # skip first 1.0 s (settle period)
    prev_v_target = 0.0
    yaw_sq_sum    = 0.0   # yaw rate tracking error accumulator
    n_yaw         = 0

    # ── Battery model ────────────────────────────────────────────────────────
    battery = BatteryModel()
    battery.reset()
    v_batt = BATT_V_NOM   # initialise to rated voltage; updated each ctrl step

    # ── Simulation loop ─────────────────────────────────────────────────────
    step = 0
    while data.time < duration:
        if step % CTRL_STEPS == 0:
            # ── Sensors ──────────────────────────────────────────────────────
            q_quat = data.xquat[box_bid]
            pitch_true = math.asin(max(-1.0, min(1.0,
                2.0 * (q_quat[0] * q_quat[2] - q_quat[3] * q_quat[1]))))
            pitch_rate_true = data.qvel[d_pitch]

            if add_noise:
                pitch      = pitch_true      + rng.normal(0, PITCH_NOISE_STD_RAD)
                pitch_rate = pitch_rate_true + rng.normal(0, PITCH_RATE_NOISE_STD_RAD_S)
            else:
                pitch, pitch_rate = pitch_true, pitch_rate_true

            wheel_vel = (data.qvel[d_whl_L] + data.qvel[d_whl_R]) / 2.0
            hip_q_avg = (data.qpos[s_hip_L] + data.qpos[s_hip_R]) / 2.0
            pitch_ff  = get_equilibrium_pitch(ROBOT, hip_q_avg)

            # ── Velocity profile & PI ─────────────────────────────────────────
            v_target_ms = v_profile_fn(data.time)
            v_measured_ms = wheel_vel * WHEEL_R
            v_ref_rads    = v_target_ms / WHEEL_R

            # Reset PI integrator on direction reversal (prevents windup carryover)
            if (prev_v_target != 0.0 and
                    math.copysign(1, v_target_ms) != math.copysign(1, prev_v_target)):
                vel_pi.reset()
            prev_v_target = v_target_ms

            theta_ref = vel_pi.update(v_target_ms, v_measured_ms)
            _d_max = THETA_REF_RATE_LIMIT * dt
            theta_ref = float(np.clip(
                theta_ref, _prev_theta_ref - _d_max, _prev_theta_ref + _d_max))
            _prev_theta_ref = theta_ref

            # ── Delay buffer (0 ms: update before use) ────────────────────────
            _pitch_d, _pitch_rate_d, _wheel_vel_d = pitch, pitch_rate, wheel_vel

            # ── LQR controller (symmetric torque) ─────────────────────────────
            tau_sym = lqr_torque(
                _pitch_d, _pitch_rate_d, _wheel_vel_d, hip_q_avg,
                v_ref=v_ref_rads, theta_ref=theta_ref)

            # ── Yaw PI (differential torque) ──────────────────────────────────
            # tau_yaw > 0 → left turn (CCW); orthogonal to tau_sym (avg unchanged)
            yaw_rate    = data.qvel[d_yaw]   # world-frame ωz [rad/s]
            omega_tgt   = omega_profile_fn(data.time) if omega_profile_fn else 0.0
            tau_yaw     = yaw_pi.update(omega_tgt, yaw_rate) if USE_YAW_PI else 0.0

            # tau_yaw > 0 = left turn: right wheel gets more torque than left
            data.ctrl[act_wheel_L] = motor_taper(tau_sym - tau_yaw, data.qvel[d_whl_L], v_batt)
            data.ctrl[act_wheel_R] = motor_taper(tau_sym + tau_yaw, data.qvel[d_whl_R], v_batt)

            # ── Yaw tracking error ─────────────────────────────────────────────
            if omega_profile_fn and data.time >= YAW_ERR_START:
                yaw_sq_sum += (omega_tgt - yaw_rate) ** 2
                n_yaw      += 1

            # ── Leg impedance + roll leveling ────────────────────────────────
            # Common-mode: both legs track the same height profile (Q_NOM or cycling)
            _q_hip_sym = _hip_fn(data.time)

            # Roll leveling: differential δq keeps box at 0° roll.
            # Sign: positive roll = left side UP (right-hand rule about +X forward).
            # δq > 0 → q_nom_L += δq (retract left) + q_nom_R -= δq (extend right).
            # Verify sign with lateral disturbance; negate LEG_K_ROLL if inverted.
            q_roll    = data.xquat[box_bid]   # world quaternion [w, x, y, z]
            roll_true = math.atan2(
                2.0 * (q_roll[0]*q_roll[1] + q_roll[2]*q_roll[3]),
                1.0 - 2.0 * (q_roll[1]**2  + q_roll[2]**2))
            roll_rate = data.qvel[d_root + 3]  # ωx world-frame [rad/s]

            if add_noise:
                roll_meas = roll_true + rng.normal(0, ROLL_NOISE_STD_RAD)
            else:
                roll_meas = roll_true

            delta_q = LEG_K_ROLL_GAIN * roll_meas + LEG_D_ROLL_GAIN * roll_rate

            q_nom_L = float(np.clip(_q_hip_sym + delta_q, HIP_SAFE_MIN, HIP_SAFE_MAX))
            q_nom_R = float(np.clip(_q_hip_sym - delta_q, HIP_SAFE_MIN, HIP_SAFE_MAX))

            for s_hip, d_hip, act_hip, q_nom_leg in [
                (s_hip_L, d_hip_L, act_hip_L, q_nom_L),
                (s_hip_R, d_hip_R, act_hip_R, q_nom_R),
            ]:
                q_hip  = data.qpos[s_hip]
                dq_hip = data.qvel[d_hip]
                tau_hip = -(LEG_K_S * (q_hip - q_nom_leg) + LEG_B_S * dq_hip)
                data.ctrl[act_hip] = np.clip(
                    tau_hip, -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)

            # ── Battery step (update v_batt for next control tick) ────────────
            v_batt = battery.step(dt, _motor_currents(
                float(data.ctrl[act_wheel_L]), float(data.ctrl[act_wheel_R]),
                float(data.ctrl[act_hip_L]),   float(data.ctrl[act_hip_R])))

            # ── Metrics ───────────────────────────────────────────────────────
            pitch_err_deg = math.degrees(abs(pitch_true - pitch_ff))
            pitch_sq_sum += pitch_err_deg ** 2
            roll_deg_now  = math.degrees(roll_true)
            roll_sq_sum  += roll_deg_now ** 2
            max_roll      = max(max_roll, abs(roll_deg_now))
            max_pitch     = max(max_pitch, pitch_err_deg)
            vel_est       = (wheel_vel + pitch_rate_true) * WHEEL_R
            wheel_travel_m += abs(vel_est) * dt
            n_samples += 1

            if data.time >= VEL_ERR_START:
                vel_sq_sum += (v_target_ms - v_measured_ms) ** 2
                n_vel += 1

            if (data.xpos[wheel_bid_L][2] > LIFTOFF_THRESHOLD or
                    data.xpos[wheel_bid_R][2] > LIFTOFF_THRESHOLD):
                wheel_liftoff_s += dt

            if not settled:
                if pitch_err_deg < SETTLE_THRESHOLD:
                    if settle_start is None:
                        settle_start = data.time
                    elif data.time - settle_start >= SETTLE_WINDOW:
                        settle_time = settle_start
                        settled     = True
                else:
                    settle_start = None

            if abs(pitch_true) > FALL_THRESHOLD:
                survived_s = data.time
                break

        # ── Disturbance impulse ───────────────────────────────────────────────
        data.xfrc_applied[box_bid, 0] = dist_fn(data.time) if dist_fn else 0.0

        mujoco.mj_step(model, data)
        step += 1

    # ── Compute final metrics ────────────────────────────────────────────────
    rms_pitch_deg      = math.sqrt(pitch_sq_sum / max(1, n_samples))
    rms_roll_deg       = math.sqrt(roll_sq_sum  / max(1, n_samples))
    max_roll_deg       = max_roll
    rms_vel_ms         = (math.sqrt(vel_sq_sum / max(1, n_vel))
                          if n_vel > 0 else 0.0)
    yaw_track_rms_rads = (math.sqrt(yaw_sq_sum / max(1, n_yaw))
                          if n_yaw > 0 else 0.0)
    final_x            = data.qpos[s_root]
    fell               = survived_s < duration - 0.05

    return dict(
        rms_pitch_deg           = round(rms_pitch_deg,      4),
        rms_roll_deg            = round(rms_roll_deg,       4),
        max_roll_deg            = round(max_roll_deg,       4),
        vel_track_rms_ms        = round(rms_vel_ms,         4),
        yaw_track_rms_rads      = round(yaw_track_rms_rads, 4),
        max_pitch_deg           = round(max_pitch,           4),
        wheel_travel_m          = round(wheel_travel_m,      4),
        wheel_liftoff_s         = round(wheel_liftoff_s,     4),
        final_x_m               = round(final_x,             3),
        settle_time_s           = round(settle_time,         3),
        survived_s              = round(survived_s,          3),
        fell                    = fell,
        status                  = "FAIL" if fell else "PASS",
        fail_reason             = "fell over" if fell else "",
    )


# ---------------------------------------------------------------------------
# Drive scenarios — forward then backward
# ---------------------------------------------------------------------------
def run_drive_slow_scenario(gains: dict, duration: float = DRIVE_DURATION,
                             add_noise: bool = True, rng_seed: int = None) -> dict:
    """Drive at 0.3 m/s forward for first half, backward for second half.

    Uses VelocityPI outer loop + 3-state LQR. VELOCITY_PI_KP/KI taken from
    module-level globals (set by optimizer worker).
    """
    rng = np.random.default_rng(rng_seed)
    try:
        model, data = _build_model_and_data()
        init_sim(model, data)
    except Exception as e:
        return _fail_result(f"model init: {e}")

    v_fn = lambda t: DRIVE_SLOW_SPEED if t < DRIVE_REV_TIME else -DRIVE_SLOW_SPEED
    raw = _run_sim_loop(model, data, duration, v_fn, add_noise=add_noise, rng=rng)

    fitness = (
        W_RMS     * raw['rms_pitch_deg']
        + W_VEL_ERR * raw['vel_track_rms_ms']
        + W_LIFTOFF * raw['wheel_liftoff_s']
        + (W_FALL if raw['fell'] else 0.0)
    )
    raw['fitness'] = round(fitness, 4)
    return raw


def run_drive_medium_scenario(gains: dict, duration: float = DRIVE_DURATION,
                               add_noise: bool = True, rng_seed: int = None) -> dict:
    """Drive at 0.8 m/s forward for first half, backward for second half."""
    rng = np.random.default_rng(rng_seed)
    try:
        model, data = _build_model_and_data()
        init_sim(model, data)
    except Exception as e:
        return _fail_result(f"model init: {e}")

    v_fn = lambda t: DRIVE_MEDIUM_SPEED if t < DRIVE_REV_TIME else -DRIVE_MEDIUM_SPEED
    raw = _run_sim_loop(model, data, duration, v_fn, add_noise=add_noise, rng=rng)

    fitness = (
        W_RMS     * raw['rms_pitch_deg']
        + W_VEL_ERR * raw['vel_track_rms_ms']
        + W_LIFTOFF * raw['wheel_liftoff_s']
        + (W_FALL if raw['fell'] else 0.0)
    )
    raw['fitness'] = round(fitness, 4)
    return raw


def run_obstacle_scenario(gains: dict, duration: float = OBSTACLE_DURATION,
                           add_noise: bool = True, rng_seed: int = None) -> dict:
    """Drive forward at 0.3 m/s and cross a 3 cm floor step at x=0.5 m.

    Robot starts at x=0, hits step at ~t=1.7 s, must not fall.
    """
    rng = np.random.default_rng(rng_seed)
    try:
        model, data = _build_model_and_data(obstacle_height=OBSTACLE_HEIGHT)
        init_sim(model, data)
    except Exception as e:
        return _fail_result(f"model init: {e}")

    v_fn = lambda t: DRIVE_SLOW_SPEED  # constant forward drive
    raw = _run_sim_loop(model, data, duration, v_fn, add_noise=add_noise, rng=rng)

    # Higher liftoff penalty for obstacle — bouncing over step is bad
    fitness = (
        W_RMS         * raw['rms_pitch_deg']
        + W_VEL_ERR   * raw['vel_track_rms_ms']
        + W_LIFTOFF * 2.0 * raw['wheel_liftoff_s']
        + (W_FALL if raw['fell'] else 0.0)
    )
    raw['fitness'] = round(fitness, 4)
    return raw


# ---------------------------------------------------------------------------
# Balance scenario — headless runner
# ---------------------------------------------------------------------------
def run_balance_scenario(gains: dict, duration: float = BALANCE_DURATION,
                         add_noise: bool = True, rng_seed: int = None) -> dict:
    """Run balance scenario headlessly.  Returns metrics dict.

    gains keys: KP, KD, KP_pos, KP_vel
    """
    rng = np.random.default_rng(rng_seed)

    try:
        model, data = _build_model_and_data()
        init_sim(model, data)
    except Exception as e:
        return _fail_result(f"model init: {e}")

    # ── Address lookups ─────────────────────────────────────────────────────
    def _jqp(name): return model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _jdof(name): return model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _act(name):  return mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    def _bid(name):  return mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, name)

    d_root  = _jdof("root_free")
    d_pitch = d_root + 4

    s_hip_L = _jqp("hip_L");   s_hip_R = _jqp("hip_R")
    d_hip_L = _jdof("hip_L");  d_hip_R = _jdof("hip_R")
    d_whl_L = _jdof("wheel_spin_L"); d_whl_R = _jdof("wheel_spin_R")

    act_hip_L   = _act("hip_act_L");   act_hip_R   = _act("hip_act_R")
    act_wheel_L = _act("wheel_act_L"); act_wheel_R = _act("wheel_act_R")

    box_bid      = _bid("box")
    wheel_bid_L  = _bid("wheel_asm_L")
    wheel_bid_R  = _bid("wheel_asm_R")

    # ── Controller state ────────────────────────────────────────────────────
    pitch_integral = 0.0
    odo_x          = 0.0

    # 1-step sensor delay buffer — models ~2 ms BNO086 I2C + CAN latency.
    # Controller always acts on readings from the previous 500 Hz tick.
    _pitch_d      = get_equilibrium_pitch(ROBOT, Q_NOM)
    _pitch_rate_d = 0.0
    _wheel_vel_d  = 0.0

    # ── Metric accumulators ─────────────────────────────────────────────────
    pitch_sq_sum   = 0.0
    max_pitch      = 0.0
    wheel_travel_m = 0.0
    wheel_liftoff_s = 0.0
    liftoff_kill   = False
    n_samples      = 0
    survived_s     = duration
    settle_time    = duration
    settled        = False
    settle_start   = None

    # ── Battery model ────────────────────────────────────────────────────────
    battery = BatteryModel()
    battery.reset()
    v_batt = BATT_V_NOM

    # ── Simulation loop ─────────────────────────────────────────────────────
    # Physics: 2000 Hz.  Controller (IMU + torque cmd): 500 Hz = every CTRL_STEPS steps.
    step = 0
    dt   = model.opt.timestep * CTRL_STEPS   # 0.002 s controller dt
    vel_pi = VelocityPI(kp=VELOCITY_PI_KP, ki=VELOCITY_PI_KI, dt=dt)
    while data.time < duration:
        if step % CTRL_STEPS == 0:
            # ── Sensors (500 Hz) ─────────────────────────────────────────────
            q_quat = data.xquat[box_bid]
            pitch_true = math.asin(max(-1.0, min(1.0,
                2.0 * (q_quat[0] * q_quat[2] - q_quat[3] * q_quat[1]))))
            pitch_rate_true = data.qvel[d_pitch]

            if add_noise:
                pitch      = pitch_true      + rng.normal(0, PITCH_NOISE_STD_RAD)
                pitch_rate = pitch_rate_true + rng.normal(0, PITCH_RATE_NOISE_STD_RAD_S)
            else:
                pitch, pitch_rate = pitch_true, pitch_rate_true

            wheel_vel = (data.qvel[d_whl_L] + data.qvel[d_whl_R]) / 2.0

            # ── Balance controller (uses delayed readings) ───────────────────
            hip_q_avg = (data.qpos[s_hip_L] + data.qpos[s_hip_R]) / 2.0
            pitch_ff  = get_equilibrium_pitch(ROBOT, hip_q_avg)

            # Rotate delay buffer (set to 0ms: update before use)
            _pitch_d, _pitch_rate_d, _wheel_vel_d = pitch, pitch_rate, wheel_vel

            if USE_PD_CONTROLLER:
                tau_wheel, pitch_integral, odo_x = balance_torque(
                    _pitch_d, _pitch_rate_d, pitch_integral, odo_x,
                    _wheel_vel_d, hip_q_avg, dt, gains)
            else:
                # LQR + optional outer VelocityPI (v_desired=0 → position hold)
                if USE_VELOCITY_PI:
                    vel_est_ms = _wheel_vel_d * WHEEL_R   # +wheel_vel = forward body
                    theta_ref  = vel_pi.update(0.0, vel_est_ms)
                else:
                    theta_ref = 0.0
                tau_wheel = lqr_torque(_pitch_d, _pitch_rate_d, _wheel_vel_d, hip_q_avg,
                                       v_ref=0.0, theta_ref=theta_ref)

            data.ctrl[act_wheel_L] = motor_taper(tau_wheel, data.qvel[d_whl_L], v_batt)
            data.ctrl[act_wheel_R] = motor_taper(tau_wheel, data.qvel[d_whl_R], v_batt)

            # ── Leg impedance: hold Q_NOM ────────────────────────────────────
            for s_hip, d_hip, act_hip in [
                (s_hip_L, d_hip_L, act_hip_L),
                (s_hip_R, d_hip_R, act_hip_R),
            ]:
                q_hip  = data.qpos[s_hip]
                dq_hip = data.qvel[d_hip]
                tau_hip = -(LEG_K_S * (q_hip - Q_NOM) + LEG_B_S * dq_hip)
                data.ctrl[act_hip] = np.clip(tau_hip, -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)

            # ── Battery step ─────────────────────────────────────────────────
            v_batt = battery.step(dt, _motor_currents(
                float(data.ctrl[act_wheel_L]), float(data.ctrl[act_wheel_R]),
                float(data.ctrl[act_hip_L]),   float(data.ctrl[act_hip_R])))

            # ── Metrics ─────────────────────────────────────────────────────
            pitch_err_deg  = math.degrees(abs(pitch_true - pitch_ff))
            pitch_sq_sum  += pitch_err_deg ** 2
            max_pitch      = max(max_pitch, pitch_err_deg)
            vel_est        = (wheel_vel + pitch_rate_true) * WHEEL_R
            wheel_travel_m += abs(vel_est) * dt
            n_samples     += 1

            if (data.xpos[wheel_bid_L][2] > LIFTOFF_THRESHOLD or
                    data.xpos[wheel_bid_R][2] > LIFTOFF_THRESHOLD):
                survived_s   = data.time
                liftoff_kill = True
                break  # instant kill — any wheel liftoff fails the balance scenario

            if not settled:
                if pitch_err_deg < SETTLE_THRESHOLD:
                    if settle_start is None:
                        settle_start = data.time
                    elif data.time - settle_start >= SETTLE_WINDOW:
                        settle_time = settle_start
                        settled     = True
                else:
                    settle_start = None

            if abs(pitch_true) > FALL_THRESHOLD:
                survived_s = data.time
                break

        mujoco.mj_step(model, data)
        step += 1

    # ── Final metrics ────────────────────────────────────────────────────────
    rms_pitch_deg = math.sqrt(pitch_sq_sum / max(1, n_samples))
    fell = survived_s < duration - 0.05

    # Fitness: wheel travel only — pitch is implicit (still wheels = maintained pitch).
    # Liftoff in balance = instant kill (see break above), so W_LIFTOFF not needed here.
    fitness = (
        W_TRAVEL * wheel_travel_m
        + (W_FALL if (fell or liftoff_kill) else 0.0)
    )

    return dict(
        rms_pitch_deg   = round(rms_pitch_deg,   4),
        max_pitch_deg   = round(max_pitch,        4),
        wheel_travel_m  = round(wheel_travel_m,   4),
        wheel_liftoff_s = round(wheel_liftoff_s,  4),
        settle_time_s   = round(settle_time,      3),
        survived_s      = round(survived_s,       3),
        fitness         = round(fitness,          4),
        status          = "FAIL" if (fell or liftoff_kill) else "PASS",
        fail_reason     = ("liftoff" if liftoff_kill else "fell over") if (fell or liftoff_kill) else "",
    )


# ---------------------------------------------------------------------------
# Balance with disturbance scenario — headless runner
# ---------------------------------------------------------------------------
def run_balance_with_disturbance_scenario(gains: dict, duration: float = BALANCE_DURATION,
                                          add_noise: bool = True, rng_seed: int = None) -> dict:
    """Run balance scenario with mid-run horizontal impulse disturbance.

    Applies a horizontal (X-axis) force for ~50 ms at t=2.5s to test recovery.
    Combines still-balance RMS + post-disturbance recovery RMS in fitness.

    gains keys: KP, KD, KP_pos, KP_vel
    """
    rng = np.random.default_rng(rng_seed)

    try:
        model, data = _build_model_and_data()
        init_sim(model, data)
    except Exception as e:
        return _fail_result(f"model init: {e}")

    # ── Address lookups ─────────────────────────────────────────────────────
    def _jqp(name): return model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _jdof(name): return model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _act(name):  return mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    def _bid(name):  return mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, name)

    d_root  = _jdof("root_free")
    d_pitch = d_root + 4

    s_hip_L = _jqp("hip_L");   s_hip_R = _jqp("hip_R")
    d_hip_L = _jdof("hip_L");  d_hip_R = _jdof("hip_R")
    d_whl_L = _jdof("wheel_spin_L"); d_whl_R = _jdof("wheel_spin_R")

    act_hip_L   = _act("hip_act_L");   act_hip_R   = _act("hip_act_R")
    act_wheel_L = _act("wheel_act_L"); act_wheel_R = _act("wheel_act_R")

    box_bid      = _bid("box")
    wheel_bid_L  = _bid("wheel_asm_L")
    wheel_bid_R  = _bid("wheel_asm_R")

    # ── Controller state ────────────────────────────────────────────────────
    pitch_integral = 0.0
    odo_x          = 0.0

    # 1-step sensor delay buffer — models ~2 ms BNO086 I2C + CAN latency.
    _pitch_d      = get_equilibrium_pitch(ROBOT, Q_NOM)
    _pitch_rate_d = 0.0
    _wheel_vel_d  = 0.0

    # ── Metric accumulators ─────────────────────────────────────────────────
    pitch_sq_sum   = 0.0
    pitch_sq_sum_post_dist = 0.0  # RMS pitch AFTER disturbance
    max_pitch      = 0.0
    wheel_travel_m = 0.0
    wheel_liftoff_s = 0.0
    n_samples      = 0
    n_samples_post_dist = 0
    survived_s     = duration
    settle_time    = duration
    settled        = False
    settle_start   = None
    disturbance_applied = False
    disturbance_end_time = DISTURBANCE_TIME + DISTURBANCE_DUR

    # ── Battery model ────────────────────────────────────────────────────────
    battery = BatteryModel()
    battery.reset()
    v_batt = BATT_V_NOM

    # ── Simulation loop ─────────────────────────────────────────────────────
    step = 0
    dt   = model.opt.timestep * CTRL_STEPS   # 0.002 s controller dt
    vel_pi = VelocityPI(kp=VELOCITY_PI_KP, ki=VELOCITY_PI_KI, dt=dt)
    while data.time < duration:
        if step % CTRL_STEPS == 0:
            # ── Sensors (500 Hz) ─────────────────────────────────────────────
            q_quat = data.xquat[box_bid]
            pitch_true = math.asin(max(-1.0, min(1.0,
                2.0 * (q_quat[0] * q_quat[2] - q_quat[3] * q_quat[1]))))
            pitch_rate_true = data.qvel[d_pitch]

            if add_noise:
                pitch      = pitch_true      + rng.normal(0, PITCH_NOISE_STD_RAD)
                pitch_rate = pitch_rate_true + rng.normal(0, PITCH_RATE_NOISE_STD_RAD_S)
            else:
                pitch, pitch_rate = pitch_true, pitch_rate_true

            wheel_vel = (data.qvel[d_whl_L] + data.qvel[d_whl_R]) / 2.0

            # ── Balance controller (uses delayed readings) ───────────────────
            hip_q_avg = (data.qpos[s_hip_L] + data.qpos[s_hip_R]) / 2.0
            pitch_ff  = get_equilibrium_pitch(ROBOT, hip_q_avg)

            # Rotate delay buffer (set to 0ms: update before use)
            _pitch_d, _pitch_rate_d, _wheel_vel_d = pitch, pitch_rate, wheel_vel

            if USE_PD_CONTROLLER:
                tau_wheel, pitch_integral, odo_x = balance_torque(
                    _pitch_d, _pitch_rate_d, pitch_integral, odo_x,
                    _wheel_vel_d, hip_q_avg, dt, gains)
            else:
                # LQR + optional outer VelocityPI (v_desired=0 → position hold)
                if USE_VELOCITY_PI:
                    vel_est_ms = _wheel_vel_d * WHEEL_R   # +wheel_vel = forward body
                    theta_ref  = vel_pi.update(0.0, vel_est_ms)
                else:
                    theta_ref = 0.0
                tau_wheel = lqr_torque(_pitch_d, _pitch_rate_d, _wheel_vel_d, hip_q_avg,
                                       v_ref=0.0, theta_ref=theta_ref)

            data.ctrl[act_wheel_L] = motor_taper(tau_wheel, data.qvel[d_whl_L], v_batt)
            data.ctrl[act_wheel_R] = motor_taper(tau_wheel, data.qvel[d_whl_R], v_batt)

            # ── Leg impedance: hold Q_NOM ────────────────────────────────────
            for s_hip, d_hip, act_hip in [
                (s_hip_L, d_hip_L, act_hip_L),
                (s_hip_R, d_hip_R, act_hip_R),
            ]:
                q_hip  = data.qpos[s_hip]
                dq_hip = data.qvel[d_hip]
                tau_hip = -(LEG_K_S * (q_hip - Q_NOM) + LEG_B_S * dq_hip)
                data.ctrl[act_hip] = np.clip(tau_hip, -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)

            # ── Battery step ─────────────────────────────────────────────────
            v_batt = battery.step(dt, _motor_currents(
                float(data.ctrl[act_wheel_L]), float(data.ctrl[act_wheel_R]),
                float(data.ctrl[act_hip_L]),   float(data.ctrl[act_hip_R])))

            # ── Metrics ─────────────────────────────────────────────────────
            pitch_err_deg  = math.degrees(abs(pitch_true - pitch_ff))
            pitch_sq_sum  += pitch_err_deg ** 2
            max_pitch      = max(max_pitch, pitch_err_deg)
            vel_est        = (wheel_vel + pitch_rate_true) * WHEEL_R
            wheel_travel_m += abs(vel_est) * dt
            n_samples     += 1

            # Track post-disturbance metrics separately
            if data.time >= disturbance_end_time:
                pitch_sq_sum_post_dist += pitch_err_deg ** 2
                n_samples_post_dist += 1

            if (data.xpos[wheel_bid_L][2] > LIFTOFF_THRESHOLD or
                    data.xpos[wheel_bid_R][2] > LIFTOFF_THRESHOLD):
                wheel_liftoff_s += dt

            if not settled:
                if pitch_err_deg < SETTLE_THRESHOLD:
                    if settle_start is None:
                        settle_start = data.time
                    elif data.time - settle_start >= SETTLE_WINDOW:
                        settle_time = settle_start
                        settled     = True
                else:
                    settle_start = None

            if abs(pitch_true) > FALL_THRESHOLD:
                survived_s = data.time
                break

        # ── Apply disturbance: horizontal impulse ───────────────────────────
        if (DISTURBANCE_TIME <= data.time < DISTURBANCE_TIME + DISTURBANCE_DUR
            and not disturbance_applied):
            # Apply horizontal force to the body (box)
            data.xfrc_applied[box_bid, 0] = DISTURBANCE_FORCE  # X-axis force
            disturbance_applied = True
        elif data.time >= DISTURBANCE_TIME + DISTURBANCE_DUR:
            # Clear disturbance force
            data.xfrc_applied[box_bid, 0] = 0.0

        mujoco.mj_step(model, data)
        step += 1

    # ── Final metrics ────────────────────────────────────────────────────────
    rms_pitch_deg = math.sqrt(pitch_sq_sum / max(1, n_samples))
    rms_pitch_post_dist_deg = (
        math.sqrt(pitch_sq_sum_post_dist / max(1, n_samples_post_dist))
        if n_samples_post_dist > 0 else 0.0
    )
    fell = survived_s < duration - 0.05

    # Combined fitness: still-balance RMS + post-disturbance recovery RMS
    fitness = (
        W_RMS * rms_pitch_deg
        + W_RECOVERY * rms_pitch_post_dist_deg
        + W_TRAVEL * wheel_travel_m
        + W_LIFTOFF * wheel_liftoff_s
        + (W_FALL if fell else 0.0)
    )

    return dict(
        rms_pitch_deg           = round(rms_pitch_deg,           4),
        rms_pitch_post_dist_deg = round(rms_pitch_post_dist_deg, 4),
        max_pitch_deg           = round(max_pitch,               4),
        wheel_travel_m          = round(wheel_travel_m,          4),
        wheel_liftoff_s         = round(wheel_liftoff_s,         4),
        settle_time_s           = round(settle_time,             3),
        survived_s              = round(survived_s,              3),
        fitness                 = round(fitness,                 4),
        status                  = "FAIL" if fell else "PASS",
        fail_reason             = "fell over" if fell else "",
    )


# ---------------------------------------------------------------------------
# Combined scenario v2 (balance + disturbance + drive-slow + drive-med + obstacle)
# ---------------------------------------------------------------------------
def run_combined_scenario(gains: dict, duration: float = BALANCE_DURATION,
                          add_noise: bool = True, rng_seed: int = None,
                          active_scenarios: list = None) -> dict:
    """Run scenarios and combine with weighted fitness.

    active_scenarios: list of names to include, e.g. ['balance'] or
        ['balance', 'disturbance', 'drive_slow', 'drive_med', 'obstacle'].
        None = all 5 (default, backward-compatible).

    Balance always runs as a gate. If it fails, returns 9999 immediately.
    When active_scenarios == ['balance'], returns immediately after balance
    with fitness = fitness_balance (skips remaining 4 scenarios).
    Weights for active scenarios are renormalized to sum to 1.0.

    Module globals VELOCITY_PI_KP/KI are used for VelocityPI in drive scenarios.
    gains keys: KP, KD, KP_pos, KP_vel (dummy for LQR, not used)
    """
    if active_scenarios is None:
        active_scenarios = ['balance', 'disturbance', 'drive_slow', 'drive_med', 'obstacle']

    # --- 1. Balance gate (always runs) ---
    m_bal = run_balance_scenario(gains, duration, add_noise, rng_seed)
    if m_bal['status'] != 'PASS':
        return dict(
            rms_pitch_deg=m_bal['rms_pitch_deg'],
            rms_pitch_post_dist_deg=0.0,
            max_pitch_deg=m_bal['max_pitch_deg'],
            wheel_travel_m=m_bal['wheel_travel_m'],
            vel_track_rms_ms=0.0,
            settle_time_s=m_bal['settle_time_s'],
            survived_s=m_bal['survived_s'],
            fitness_balance=m_bal['fitness'],
            fitness_disturbance=9999.0,
            fitness_drive_slow=9999.0,
            fitness_drive_med=9999.0,
            fitness_obstacle=9999.0,
            fitness=9999.0,
            status='FAIL',
            fail_reason='failed balance scenario',
        )

    # --- 2. Balance-only short-circuit ---
    if set(active_scenarios) == {'balance'}:
        return dict(
            rms_pitch_deg           = m_bal['rms_pitch_deg'],
            rms_pitch_post_dist_deg = 0.0,
            max_pitch_deg           = m_bal['max_pitch_deg'],
            wheel_travel_m          = m_bal['wheel_travel_m'],
            vel_track_rms_ms        = 0.0,
            settle_time_s           = m_bal['settle_time_s'],
            survived_s              = m_bal['survived_s'],
            fitness_balance         = round(m_bal['fitness'], 4),
            fitness_disturbance     = 0.0,
            fitness_drive_slow      = 0.0,
            fitness_drive_med       = 0.0,
            fitness_obstacle        = 0.0,
            fitness                 = round(m_bal['fitness'], 4),
            status                  = 'PASS',
            fail_reason             = '',
        )

    # --- 3. Active non-balance scenarios ---
    _dummy = dict(fitness=0.0, status='PASS', survived_s=duration,
                  max_pitch_deg=0.0, rms_pitch_post_dist_deg=0.0, vel_track_rms_ms=0.0)
    m_dist = run_balance_with_disturbance_scenario(gains, duration, add_noise, rng_seed) \
             if 'disturbance' in active_scenarios else _dummy
    m_slow = run_drive_slow_scenario(gains, DRIVE_DURATION, add_noise, rng_seed) \
             if 'drive_slow' in active_scenarios else _dummy
    m_med  = run_drive_medium_scenario(gains, DRIVE_DURATION, add_noise, rng_seed) \
             if 'drive_med' in active_scenarios else _dummy
    m_step = run_obstacle_scenario(gains, OBSTACLE_DURATION, add_noise, rng_seed) \
             if 'obstacle' in active_scenarios else _dummy

    # Renormalize weights for active scenarios only
    w = {
        'balance':     W_BALANCE     if 'balance'     in active_scenarios else 0.0,
        'disturbance': W_DISTURBANCE if 'disturbance' in active_scenarios else 0.0,
        'drive_slow':  W_DRIVE_SLOW  if 'drive_slow'  in active_scenarios else 0.0,
        'drive_med':   W_DRIVE_MED   if 'drive_med'   in active_scenarios else 0.0,
        'obstacle':    W_DRIVE_STEP  if 'obstacle'    in active_scenarios else 0.0,
    }
    total_w = sum(w.values()) or 1.0

    fit_balance = m_bal['fitness']
    fit_dist    = m_dist['fitness']
    fit_slow    = m_slow['fitness']
    fit_med     = m_med['fitness']
    fit_step    = m_step['fitness']

    combined_fit = (
        w['balance']     / total_w * fit_balance
        + w['disturbance'] / total_w * fit_dist
        + w['drive_slow']  / total_w * fit_slow
        + w['drive_med']   / total_w * fit_med
        + w['obstacle']    / total_w * fit_step
    )

    all_pass = all(m['status'] == 'PASS'
                   for m in [m_dist, m_slow, m_med, m_step]
                   if m is not _dummy)

    return dict(
        rms_pitch_deg           = m_bal['rms_pitch_deg'],
        rms_pitch_post_dist_deg = m_dist.get('rms_pitch_post_dist_deg', 0.0),
        max_pitch_deg           = max(m_bal['max_pitch_deg'],
                                      m_dist['max_pitch_deg'],
                                      m_slow['max_pitch_deg'],
                                      m_med['max_pitch_deg'],
                                      m_step['max_pitch_deg']),
        wheel_travel_m          = m_bal['wheel_travel_m'],
        vel_track_rms_ms        = m_slow.get('vel_track_rms_ms', 0.0),
        settle_time_s           = m_bal['settle_time_s'],
        survived_s              = min(m_dist['survived_s'],
                                      m_slow['survived_s'],
                                      m_med['survived_s'],
                                      m_step['survived_s']),
        fitness_balance         = round(fit_balance, 4),
        fitness_disturbance     = round(fit_dist,    4),
        fitness_drive_slow      = round(fit_slow,    4),
        fitness_drive_med       = round(fit_med,     4),
        fitness_obstacle        = round(fit_step,    4),
        fitness                 = round(combined_fit, 4),
        status                  = 'PASS' if all_pass else 'FAIL',
        fail_reason             = '' if all_pass else 'one or more scenarios failed',
    )


# ---------------------------------------------------------------------------
# 1_LQR_pitch_step — LQR pitch controller, step-response from perturbed pitch
# ---------------------------------------------------------------------------
def run_1_LQR_pitch_step(gains: dict, duration: float = BALANCE_DURATION,
                         add_noise: bool = True, rng_seed: int = None) -> dict:
    """1_LQR_pitch_step — LQR pitch step response.

    Controller under test : Balance LQR (inner loop only)
    VelocityPI            : OFF — theta_ref = 0 always
    Initial condition     : pitch = equilibrium + PITCH_STEP_RAD (~5°)
    Metric / Fitness      : ISE = integral of squared pitch error [rad²·s]
                            lower is better; W_FALL penalty if robot falls
    """
    rng = np.random.default_rng(rng_seed)
    try:
        model, data = _build_model_and_data()
        init_sim(model, data)
    except Exception as e:
        return dict(ise_rad2s=9999.0, rms_pitch_deg=999.0, max_pitch_deg=999.0,
                    settle_time_s=999.0, survived_s=0.0,
                    fitness=9999.0, status="FAIL", fail_reason=str(e))

    def _jqp(name):  return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _jdof(name): return model.jnt_dofadr [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _act(name):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    def _bid(name):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    # Apply pitch perturbation on top of equilibrium
    s_root   = _jqp("root_free")
    theta    = get_equilibrium_pitch(ROBOT, Q_NOM) + PITCH_STEP_RAD
    data.qpos[s_root + 3] = math.cos(theta / 2.0)
    data.qpos[s_root + 4] = 0.0
    data.qpos[s_root + 5] = math.sin(theta / 2.0)
    data.qpos[s_root + 6] = 0.0
    mujoco.mj_forward(model, data)

    d_root      = _jdof("root_free");    d_pitch = d_root + 4
    s_hip_L     = _jqp("hip_L");        s_hip_R = _jqp("hip_R")
    d_hip_L     = _jdof("hip_L");       d_hip_R = _jdof("hip_R")
    d_whl_L     = _jdof("wheel_spin_L"); d_whl_R = _jdof("wheel_spin_R")
    act_hip_L   = _act("hip_act_L");    act_hip_R   = _act("hip_act_R")
    act_wheel_L = _act("wheel_act_L");  act_wheel_R = _act("wheel_act_R")
    box_bid     = _bid("box")

    dt = model.opt.timestep * CTRL_STEPS
    pitch_ff_nom = get_equilibrium_pitch(ROBOT, Q_NOM)   # fixed reference

    # Accumulators
    ise_pitch      = 0.0   # ∫ (pitch - pitch_ff_nom)² dt  — rewards returning to target
    ise_pitch_rate = 0.0   # ∫ pitch_rate² dt              — rewards killing oscillation
    pitch_sq_sum   = 0.0
    max_pitch_deg  = 0.0
    n_samples      = 0
    survived_s     = duration
    settle_time    = duration   # logged only, not used in fitness
    settled        = False
    settle_start   = None

    battery = BatteryModel()
    battery.reset()
    v_batt = BATT_V_NOM

    step = 0
    while data.time < duration:
        if step % CTRL_STEPS == 0:
            q_quat = data.xquat[box_bid]
            pitch_true = math.asin(max(-1.0, min(1.0,
                2.0 * (q_quat[0] * q_quat[2] - q_quat[3] * q_quat[1]))))
            pitch_rate_true = data.qvel[d_pitch]

            if add_noise:
                pitch      = pitch_true      + rng.normal(0, PITCH_NOISE_STD_RAD)
                pitch_rate = pitch_rate_true + rng.normal(0, PITCH_RATE_NOISE_STD_RAD_S)
            else:
                pitch, pitch_rate = pitch_true, pitch_rate_true

            wheel_vel = (data.qvel[d_whl_L] + data.qvel[d_whl_R]) / 2.0
            hip_q_avg = (data.qpos[s_hip_L] + data.qpos[s_hip_R]) / 2.0

            # LQR — VelocityPI OFF (theta_ref = 0)
            tau_wheel = lqr_torque(pitch, pitch_rate, wheel_vel, hip_q_avg,
                                   v_ref=0.0, theta_ref=0.0)
            data.ctrl[act_wheel_L] = motor_taper(tau_wheel, data.qvel[d_whl_L], v_batt)
            data.ctrl[act_wheel_R] = motor_taper(tau_wheel, data.qvel[d_whl_R], v_batt)

            # Leg impedance: hold Q_NOM
            for s_hip, d_hip, act_hip in [
                (s_hip_L, d_hip_L, act_hip_L),
                (s_hip_R, d_hip_R, act_hip_R),
            ]:
                q_hip  = data.qpos[s_hip]
                dq_hip = data.qvel[d_hip]
                data.ctrl[act_hip] = np.clip(
                    -(LEG_K_S * (q_hip - Q_NOM) + LEG_B_S * dq_hip),
                    -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)

            v_batt = battery.step(dt, _motor_currents(
                float(data.ctrl[act_wheel_L]), float(data.ctrl[act_wheel_R]),
                float(data.ctrl[act_hip_L]),   float(data.ctrl[act_hip_R])))

            # Metrics — error vs fixed Q_NOM reference (not live hip)
            pitch_err_rad = pitch_true - pitch_ff_nom
            pitch_err_deg = math.degrees(abs(pitch_err_rad))
            pitch_sq_sum += pitch_err_deg ** 2
            max_pitch_deg = max(max_pitch_deg, pitch_err_deg)
            n_samples    += 1

            # ISE: pitch error and pitch rate (full window, no split)
            ise_pitch      += pitch_err_rad ** 2 * dt
            ise_pitch_rate += pitch_rate_true ** 2 * dt

            # Settle detection for logging only (not used in fitness)
            if not settled:
                if pitch_err_deg < SETTLE_THRESHOLD:
                    if settle_start is None:
                        settle_start = data.time
                    elif data.time - settle_start >= SETTLE_WINDOW:
                        settle_time = settle_start
                        settled     = True
                else:
                    settle_start = None

            # Mid-run disturbances: +4N at t=2s, −4N at t=3s
            data.xfrc_applied[box_bid, 0] = s1_dist_fn(data.time)

            if abs(pitch_true) > FALL_THRESHOLD:
                survived_s = data.time
                break

        mujoco.mj_step(model, data)
        step += 1

    fell          = survived_s < duration - 0.05
    rms_pitch_deg = math.sqrt(pitch_sq_sum / max(1, n_samples))
    fitness = (ise_pitch
               + W_PITCH_RATE * ise_pitch_rate
               + (W_FALL if fell else 0.0))

    return dict(
        ise_pitch      = round(ise_pitch,      6),
        ise_pitch_rate = round(ise_pitch_rate, 6),
        rms_pitch_deg  = round(rms_pitch_deg,  4),
        max_pitch_deg  = round(max_pitch_deg,  4),
        settle_time_s  = round(settle_time,    3),
        survived_s     = round(survived_s,     3),
        fitness        = round(fitness,        6),
        status         = "FAIL" if fell else "PASS",
        fail_reason    = "fell over" if fell else "",
    )


# ---------------------------------------------------------------------------
# 2_LQR_impulse_recovery — LQR disturbance rejection, VelocityPI OFF
# ---------------------------------------------------------------------------
def run_2_LQR_impulse_recovery(gains: dict, duration: float = BALANCE_DURATION,
                                add_noise: bool = True, rng_seed: int = None) -> dict:
    """2_LQR_impulse_recovery — LQR rejection of a horizontal impulse.

    Controller under test : Balance LQR (inner loop only)
    VelocityPI            : OFF — theta_ref = 0 always
    Initial condition     : nominal equilibrium
    Disturbance           : DISTURBANCE_FORCE [N] at t=DISTURBANCE_TIME for DISTURBANCE_DUR [s]
    Metric / Fitness      : ISE_post = integral of squared pitch error after impulse [rad²·s]
                            lower is better; W_FALL penalty if robot falls
    """
    rng = np.random.default_rng(rng_seed)
    try:
        model, data = _build_model_and_data()
        init_sim(model, data)
    except Exception as e:
        return dict(ise_post_rad2s=9999.0, rms_pitch_pre_deg=999.0,
                    rms_pitch_post_deg=999.0, max_pitch_post_deg=999.0,
                    settle_time_post_s=999.0, survived_s=0.0,
                    fitness=9999.0, status="FAIL", fail_reason=str(e))

    def _jqp(name):  return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _jdof(name): return model.jnt_dofadr [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _act(name):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    def _bid(name):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    d_root      = _jdof("root_free");     d_pitch = d_root + 4
    s_hip_L     = _jqp("hip_L");         s_hip_R = _jqp("hip_R")
    d_hip_L     = _jdof("hip_L");        d_hip_R = _jdof("hip_R")
    d_whl_L     = _jdof("wheel_spin_L"); d_whl_R = _jdof("wheel_spin_R")
    act_hip_L   = _act("hip_act_L");     act_hip_R   = _act("hip_act_R")
    act_wheel_L = _act("wheel_act_L");   act_wheel_R = _act("wheel_act_R")
    box_bid     = _bid("box")

    dt       = model.opt.timestep * CTRL_STEPS
    dist_end = DISTURBANCE_TIME + DISTURBANCE_DUR

    # Accumulators — pre and post disturbance tracked separately
    pitch_sq_sum_pre   = 0.0;  n_pre  = 0
    ise_post_rad2s     = 0.0
    pitch_sq_sum_post  = 0.0;  n_post = 0
    max_pitch_post_deg = 0.0
    survived_s         = duration
    settle_time_post   = duration
    settled_post       = False
    settle_start_post  = None

    battery = BatteryModel()
    battery.reset()
    v_batt = BATT_V_NOM

    step = 0
    while data.time < duration:
        if step % CTRL_STEPS == 0:
            q_quat = data.xquat[box_bid]
            pitch_true = math.asin(max(-1.0, min(1.0,
                2.0 * (q_quat[0] * q_quat[2] - q_quat[3] * q_quat[1]))))
            pitch_rate_true = data.qvel[d_pitch]

            if add_noise:
                pitch      = pitch_true      + rng.normal(0, PITCH_NOISE_STD_RAD)
                pitch_rate = pitch_rate_true + rng.normal(0, PITCH_RATE_NOISE_STD_RAD_S)
            else:
                pitch, pitch_rate = pitch_true, pitch_rate_true

            wheel_vel = (data.qvel[d_whl_L] + data.qvel[d_whl_R]) / 2.0
            hip_q_avg = (data.qpos[s_hip_L] + data.qpos[s_hip_R]) / 2.0
            pitch_ff  = get_equilibrium_pitch(ROBOT, hip_q_avg)

            # LQR — VelocityPI OFF (theta_ref = 0)
            tau_wheel = lqr_torque(pitch, pitch_rate, wheel_vel, hip_q_avg,
                                   v_ref=0.0, theta_ref=0.0)
            data.ctrl[act_wheel_L] = motor_taper(tau_wheel, data.qvel[d_whl_L], v_batt)
            data.ctrl[act_wheel_R] = motor_taper(tau_wheel, data.qvel[d_whl_R], v_batt)

            # Leg impedance: hold Q_NOM
            for s_hip, d_hip, act_hip in [
                (s_hip_L, d_hip_L, act_hip_L),
                (s_hip_R, d_hip_R, act_hip_R),
            ]:
                q_hip  = data.qpos[s_hip]
                dq_hip = data.qvel[d_hip]
                data.ctrl[act_hip] = np.clip(
                    -(LEG_K_S * (q_hip - Q_NOM) + LEG_B_S * dq_hip),
                    -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)

            v_batt = battery.step(dt, _motor_currents(
                float(data.ctrl[act_wheel_L]), float(data.ctrl[act_wheel_R]),
                float(data.ctrl[act_hip_L]),   float(data.ctrl[act_hip_R])))

            # Metrics — split pre / post disturbance
            pitch_err_rad = pitch_true - pitch_ff
            pitch_err_deg = math.degrees(abs(pitch_err_rad))

            if data.time < DISTURBANCE_TIME:
                pitch_sq_sum_pre += pitch_err_deg ** 2
                n_pre += 1
            elif data.time >= dist_end:
                ise_post_rad2s    += pitch_err_rad ** 2 * dt
                pitch_sq_sum_post += pitch_err_deg ** 2
                max_pitch_post_deg = max(max_pitch_post_deg, pitch_err_deg)
                n_post += 1

                if not settled_post:
                    if pitch_err_deg < SETTLE_THRESHOLD:
                        if settle_start_post is None:
                            settle_start_post = data.time
                        elif data.time - settle_start_post >= SETTLE_WINDOW:
                            settle_time_post = settle_start_post
                            settled_post     = True
                    else:
                        settle_start_post = None

            if abs(pitch_true) > FALL_THRESHOLD:
                survived_s = data.time
                break

        # Disturbance impulse
        if DISTURBANCE_TIME <= data.time < dist_end:
            data.xfrc_applied[box_bid, 0] = DISTURBANCE_FORCE
        else:
            data.xfrc_applied[box_bid, 0] = 0.0

        mujoco.mj_step(model, data)
        step += 1

    fell               = survived_s < duration - 0.05
    rms_pitch_pre_deg  = math.sqrt(pitch_sq_sum_pre  / max(1, n_pre))
    rms_pitch_post_deg = math.sqrt(pitch_sq_sum_post / max(1, n_post))
    fitness            = ise_post_rad2s + (W_FALL if fell else 0.0)

    return dict(
        ise_post_rad2s     = round(ise_post_rad2s,     5),
        rms_pitch_pre_deg  = round(rms_pitch_pre_deg,  4),
        rms_pitch_post_deg = round(rms_pitch_post_deg, 4),
        max_pitch_post_deg = round(max_pitch_post_deg, 4),
        settle_time_post_s = round(settle_time_post,   3),
        survived_s         = round(survived_s,         3),
        fitness            = round(fitness,             5),
        status             = "FAIL" if fell else "PASS",
        fail_reason        = "fell over" if fell else "",
    )


# ── VelocityPI combined scenario weights ─────────────────────────────────────
# Used by run_combined_PI_scenario. Both sub-fitnesses are normalized to [m/s].
W_PI_DISTURBANCE = 0.50   # weight on 2_VEL_PI_disturbance fitness (position hold)
W_PI_STAIRCASE   = 0.50   # weight on 3_VEL_PI_staircase fitness  (setpoint tracking)

# ---------------------------------------------------------------------------
# 2_VEL_PI_disturbance — VelocityPI outer loop, disturbance rejection
# ---------------------------------------------------------------------------
def run_2_VEL_PI_disturbance(gains: dict, duration: float = SCENARIO_2_DURATION,
                              add_noise: bool = True, rng_seed: int = None) -> dict:
    """2_VEL_PI_disturbance — VelocityPI position-hold under two impulse kicks.

    Controller under test : VelocityPI (outer) + Balance LQR (inner)
    Initial condition     : equilibrium pitch — no perturbation
    Disturbances          : +1N at t=2s, −1N at t=3s, 0.2s each
    Metric / Fitness      : total absolute wheel travel [m]  (lower = better position hold)
                            Fall → simulation killed immediately, W_FALL penalty
    Duration              : 6 s (gives PI 2.8 s to recover after last kick at t=3.2 s)
    """
    rng = np.random.default_rng(rng_seed)
    try:
        model, data = _build_model_and_data()
        init_sim(model, data)
    except Exception as e:
        return dict(rms_pitch_deg=999.0, max_pitch_deg=999.0,
                    wheel_travel_m=999.0, survived_s=0.0,
                    fitness=9999.0, status="FAIL", fail_reason=str(e))

    raw = _run_sim_loop(model, data, duration,
                        v_profile_fn=lambda t: 0.0,
                        dist_fn=s2_dist_fn,
                        add_noise=add_noise, rng=rng)

    fell    = raw['fell']
    # Normalize by duration → units [m/s], comparable to vel_track_rms_ms in S3.
    fitness = (raw['wheel_travel_m'] + (W_FALL if fell else 0.0)) / duration
    return dict(
        rms_pitch_deg  = raw['rms_pitch_deg'],
        max_pitch_deg  = raw['max_pitch_deg'],
        wheel_travel_m = raw['wheel_travel_m'],
        survived_s     = raw['survived_s'],
        fitness        = round(fitness, 4),
        status         = raw['status'],
        fail_reason    = raw['fail_reason'],
    )


# ---------------------------------------------------------------------------
# 3_VEL_PI_staircase — VelocityPI setpoint tracking across four speed steps
# ---------------------------------------------------------------------------
def run_3_VEL_PI_staircase(gains: dict, duration: float = SCENARIO_3_DURATION,
                            add_noise: bool = True, rng_seed: int = None) -> dict:
    """3_VEL_PI_staircase — VelocityPI tracking of a six-step velocity profile.

    Controller under test : VelocityPI (outer) + Balance LQR (inner)
    Velocity profile      : 0 → +0.3 → +0.6 → +1.0 → −0.5 → −1.0 → 0 m/s
    Metric / Fitness      : vel_track_rms_ms [m/s] — lower is better
                            +0.1×rms_pitch_deg penalty (secondary)
                            Fall → W_FALL penalty, simulation killed
    Duration              : 13 s

    Profile timing:
        t =  0.0 –  1.0 s  →  0.0  m/s  (settle; excluded from VEL_ERR metric)
        t =  1.0 –  3.0 s  → +0.3  m/s  gentle forward
        t =  3.0 –  5.0 s  → +0.6  m/s  step up
        t =  5.0 –  7.0 s  → +1.0  m/s  high-speed forward
        t =  7.0 –  9.0 s  → −0.5  m/s  direction reversal — integrator reset
        t =  9.0 – 11.0 s  → −1.0  m/s  high-speed backward
        t = 11.0 – 13.0 s  →  0.0  m/s  return to stop
    """
    rng = np.random.default_rng(rng_seed)
    try:
        model, data = _build_model_and_data()
        init_sim(model, data)
    except Exception as e:
        return _fail_result(f"model init: {e}")

    raw = _run_sim_loop(model, data, duration, s3_velocity_profile, add_noise=add_noise, rng=rng)

    fitness = (
        W_VEL_ERR        * raw['vel_track_rms_ms']
        + 0.1 * W_RMS    * raw['rms_pitch_deg']
        + (W_FALL if raw['fell'] else 0.0)
    )
    raw['fitness'] = round(fitness, 4)
    return raw


# ---------------------------------------------------------------------------
# combined_PI — weighted combination of S2 (hold) + S3 (tracking)
# ---------------------------------------------------------------------------
def run_combined_PI_scenario(gains: dict, add_noise: bool = True,
                              rng_seed: int = None) -> dict:
    """combined_PI — simultaneous PI position-hold and setpoint-tracking fitness.

    Runs both sub-scenarios and returns a weighted sum.
    Both sub-fitnesses are normalized to [m/s] units before weighting.

    Fitness = W_PI_DISTURBANCE × s2_fitness + W_PI_STAIRCASE × s3_fitness
    """
    rng_seed = int(np.random.default_rng(rng_seed).integers(0, 2**31))
    s2 = run_2_VEL_PI_disturbance(gains, add_noise=add_noise, rng_seed=rng_seed)
    s3 = run_3_VEL_PI_staircase(  gains, add_noise=add_noise, rng_seed=rng_seed)

    s2_fell = s2['status'] == 'FAIL'
    s3_fell = s3.get('fell', s3['status'] == 'FAIL')
    fell    = s2_fell or s3_fell
    fitness = W_PI_DISTURBANCE * s2['fitness'] + W_PI_STAIRCASE * s3['fitness']

    fail_parts = []
    if s2_fell: fail_parts.append("s2 fell")
    if s3_fell: fail_parts.append("s3 fell")

    return dict(
        s2_fitness          = s2['fitness'],
        s3_fitness          = s3['fitness'],
        s2_wheel_travel_m   = s2['wheel_travel_m'],
        s3_vel_track_rms_ms = s3['vel_track_rms_ms'],
        s2_rms_pitch_deg    = s2['rms_pitch_deg'],
        s3_rms_pitch_deg    = s3['rms_pitch_deg'],
        survived_s          = round(min(s2['survived_s'], s3['survived_s']), 3),
        fitness             = round(fitness, 4),
        status              = "FAIL" if fell else "PASS",
        fail_reason         = "; ".join(fail_parts),
    )


# ---------------------------------------------------------------------------
# 4_leg_height_gain_sched — gain scheduler validation across full leg stroke
# ---------------------------------------------------------------------------
def run_4_leg_height_gain_sched(gains: dict, duration: float = SCENARIO_4_DURATION,
                                 add_noise: bool = True, rng_seed: int = None) -> dict:
    """4_leg_height_gain_sched — LQR-only balance while legs cycle through full stroke.

    Controller under test : Balance LQR (inner loop only, gain-scheduled)
    VelocityPI            : OFF — theta_ref = 0, v_ref = 0
    Leg motion            : sinusoidal Q_RET ↔ Q_EXT, period = LEG_CYCLE_PERIOD (4 s)
                            half-period = 2 s per extreme-to-extreme sweep
    Disturbances          : none — leg cycling is the test stimulus
    Metric / Fitness      : ISE pitch  [rad²·s]  (same form as S1)
                            pitch error measured vs live pitch_ff(q_hip), not fixed Q_NOM
    Duration              : 12 s (3 full leg cycles)

    Position drift is expected and acceptable — only pitch tracking is scored.
    """
    rng = np.random.default_rng(rng_seed)
    try:
        model, data = _build_model_and_data()
        init_sim(model, data)
    except Exception as e:
        return dict(ise_rad2s=9999.0, rms_pitch_deg=999.0, max_pitch_deg=999.0,
                    survived_s=0.0, fitness=9999.0, status="FAIL", fail_reason=str(e))

    def _jqp(name):  return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _jdof(name): return model.jnt_dofadr [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _act(name):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    def _bid(name):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    d_root      = _jdof("root_free");    d_pitch = d_root + 4
    s_hip_L     = _jqp("hip_L");        s_hip_R = _jqp("hip_R")
    d_hip_L     = _jdof("hip_L");       d_hip_R = _jdof("hip_R")
    d_whl_L     = _jdof("wheel_spin_L"); d_whl_R = _jdof("wheel_spin_R")
    act_hip_L   = _act("hip_act_L");    act_hip_R   = _act("hip_act_R")
    act_wheel_L = _act("wheel_act_L");  act_wheel_R = _act("wheel_act_R")
    box_bid     = _bid("box")

    dt = model.opt.timestep * CTRL_STEPS

    ise_pitch      = 0.0
    ise_pitch_rate = 0.0
    pitch_sq_sum   = 0.0
    max_pitch_deg  = 0.0
    n_samples      = 0
    survived_s     = duration

    battery = BatteryModel()
    battery.reset()
    v_batt = BATT_V_NOM

    step = 0
    while data.time < duration:
        if step % CTRL_STEPS == 0:
            q_quat = data.xquat[box_bid]
            pitch_true = math.asin(max(-1.0, min(1.0,
                2.0 * (q_quat[0] * q_quat[2] - q_quat[3] * q_quat[1]))))
            pitch_rate_true = data.qvel[d_pitch]

            if add_noise:
                pitch      = pitch_true      + rng.normal(0, PITCH_NOISE_STD_RAD)
                pitch_rate = pitch_rate_true + rng.normal(0, PITCH_RATE_NOISE_STD_RAD_S)
            else:
                pitch, pitch_rate = pitch_true, pitch_rate_true

            wheel_vel = (data.qvel[d_whl_L] + data.qvel[d_whl_R]) / 2.0
            hip_q_avg = (data.qpos[s_hip_L] + data.qpos[s_hip_R]) / 2.0

            # LQR — VelocityPI OFF, theta_ref = 0
            tau_wheel = lqr_torque(pitch, pitch_rate, wheel_vel, hip_q_avg,
                                   v_ref=0.0, theta_ref=0.0)
            data.ctrl[act_wheel_L] = motor_taper(tau_wheel, data.qvel[d_whl_L], v_batt)
            data.ctrl[act_wheel_R] = motor_taper(tau_wheel, data.qvel[d_whl_R], v_batt)

            # Leg impedance: track cycling profile
            q_hip_tgt = _leg_cycle_profile(data.time)
            for s_hip, d_hip, act_hip in [
                (s_hip_L, d_hip_L, act_hip_L),
                (s_hip_R, d_hip_R, act_hip_R),
            ]:
                q_hip  = data.qpos[s_hip]
                dq_hip = data.qvel[d_hip]
                data.ctrl[act_hip] = np.clip(
                    -(LEG_K_S * (q_hip - q_hip_tgt) + LEG_B_S * dq_hip),
                    -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)

            v_batt = battery.step(dt, _motor_currents(
                float(data.ctrl[act_wheel_L]), float(data.ctrl[act_wheel_R]),
                float(data.ctrl[act_hip_L]),   float(data.ctrl[act_hip_R])))

            # Metrics — pitch error vs live equilibrium for current leg height
            pitch_ff      = get_equilibrium_pitch(ROBOT, hip_q_avg)
            pitch_err_rad = pitch_true - pitch_ff
            pitch_err_deg = math.degrees(abs(pitch_err_rad))
            pitch_sq_sum += pitch_err_deg ** 2
            max_pitch_deg = max(max_pitch_deg, pitch_err_deg)
            n_samples    += 1
            ise_pitch      += pitch_err_rad ** 2 * dt
            ise_pitch_rate += pitch_rate_true ** 2 * dt

            if abs(pitch_true) > FALL_THRESHOLD:
                survived_s = data.time
                break

        # S1 disturbances: +4 N at t=2 s, −4 N at t=3 s (same as 1_LQR_pitch_step)
        data.xfrc_applied[box_bid, 0] = s1_dist_fn(data.time)

        mujoco.mj_step(model, data)
        step += 1

    fell          = survived_s < duration - 0.05
    rms_pitch_deg = math.sqrt(pitch_sq_sum / max(1, n_samples))
    fitness       = ise_pitch + W_PITCH_RATE * ise_pitch_rate + (W_FALL if fell else 0.0)

    return dict(
        ise_rad2s      = round(ise_pitch,      6),
        ise_pitch_rate = round(ise_pitch_rate, 6),
        rms_pitch_deg  = round(rms_pitch_deg,  4),
        max_pitch_deg  = round(max_pitch_deg,  4),
        survived_s     = round(survived_s,     3),
        fitness        = round(fitness,        6),
        status         = "FAIL" if fell else "PASS",
        fail_reason    = "fell over" if fell else "",
    )


# ---------------------------------------------------------------------------
# 5_VEL_PI_leg_cycling — VelocityPI under full real-world conditions
# ---------------------------------------------------------------------------
def run_5_VEL_PI_leg_cycling(gains: dict, duration: float = SCENARIO_5_DURATION,
                              add_noise: bool = True, rng_seed: int = None) -> dict:
    """5_VEL_PI_leg_cycling — VelocityPI tuning with legs cycling + staircase + disturbances.

    Controller under test : VelocityPI (outer) + Balance LQR (inner, gain-scheduled)
    Velocity profile      : same S3 staircase: 0 → +0.3 → +0.6 → +1.0 → −0.5 → −1.0 → 0 m/s
    Disturbances          : S2 kicks: +1N at t=2s, −1N at t=3s (fire during +0.3m/s phase)
    Leg motion            : sinusoidal Q_RET ↔ Q_EXT, period = LEG_CYCLE_PERIOD (4 s)
    Metric / Fitness      : vel_track_rms_ms [m/s] + 0.1×rms_pitch_deg + W_FALL×fell
    Duration              : 13 s (staircase drives duration; 3+ leg cycles included)

    This is the hardest VelocityPI scenario: the robot must track velocity commands
    while simultaneously absorbing disturbance kicks and coping with changing equilibrium
    pitch as the legs extend and retract. Gains tuned here are the most transfer-robust.
    """
    rng = np.random.default_rng(rng_seed)
    try:
        model, data = _build_model_and_data(bumps=S5_BUMPS)
        init_sim(model, data)
    except Exception as e:
        return _fail_result(f"model init: {e}")

    raw = _run_sim_loop(model, data, duration,
                        v_profile_fn=s3_velocity_profile,
                        dist_fn=s2_dist_fn,
                        add_noise=add_noise, rng=rng,
                        target_hip_q=_leg_cycle_profile)

    fitness = (
        W_VEL_ERR        * raw['vel_track_rms_ms']
        + 0.1 * W_RMS    * raw['rms_pitch_deg']
        + (W_FALL if raw['fell'] else 0.0)
    )
    raw['fitness'] = round(fitness, 4)
    return raw


def run_6_YAW_PI_turn(gains: dict, duration: float = SCENARIO_6_DURATION,
                      add_noise: bool = True, rng_seed: int = None) -> dict:
    """6_YAW_PI_turn — YawPI tuning: pure 360° CCW turn at 1 rad/s.

    Controller under test : YawPI (differential) + VelocityPI (v=0 position hold) + LQR
    v_desired             : 0.0 throughout (position hold, VelocityPI active)
    omega_desired         : +1.0 rad/s from t=1.0s → one full CCW revolution (6.28s)
    Duration              : 8.0s (1s settle + 6.28s turn + 0.72s tail)
    Metric / Fitness      : W_YAW_ERR × yaw_track_rms_rads + 0.1×rms_pitch_deg + W_FALL×fell

    Pure yaw test — isolates YawPI from velocity coupling. Because tau_yaw is differential
    (average wheel velocity unchanged), LQR/VelocityPI are not directly affected by the
    yaw command. Any coupling is through centripetal physics.
    """
    rng = np.random.default_rng(rng_seed)
    try:
        model, data = _build_model_and_data()
        init_sim(model, data)
    except Exception as e:
        return _fail_result(f"model init: {e}")

    # Settle for 1s at omega=0, then command YAW_TURN_RATE (1 rad/s CCW)
    omega_fn = lambda t: YAW_TURN_RATE if t >= 1.0 else 0.0

    raw = _run_sim_loop(model, data, duration,
                        v_profile_fn=lambda t: 0.0,
                        omega_profile_fn=omega_fn,
                        add_noise=add_noise, rng=rng)

    fitness = (
        W_YAW_ERR     * raw['yaw_track_rms_rads']
        + 0.1 * W_RMS * raw['rms_pitch_deg']
        + (W_FALL if raw['fell'] else 0.0)
    )
    raw['fitness'] = round(fitness, 4)
    return raw


def run_7_DRIVE_TURN(gains: dict, duration: float = SCENARIO_7_DURATION,
                     add_noise: bool = True, rng_seed: int = None) -> dict:
    """7_DRIVE_TURN — Cross-coupling check: simultaneous drive + turn.

    Controller under test : YawPI + VelocityPI + LQR
    v_desired             : +0.3 m/s constant (gentle forward drive)
    omega_desired         : +0.5 rad/s constant (gentle left turn → curved path)
    Duration              : 8.0s
    Metric / Fitness      : 0.5×W_VEL_ERR×vel_rms + 0.5×W_YAW_ERR×yaw_rms + 0.1×rms_pitch + W_FALL×fell

    Verifies that yaw torque does not destabilize the VelocityPI or vice-versa.
    If pitch spikes during this scenario, joint re-optimization of LQR Q/R may be needed.
    """
    rng = np.random.default_rng(rng_seed)
    try:
        model, data = _build_model_and_data()
        init_sim(model, data)
    except Exception as e:
        return _fail_result(f"model init: {e}")

    raw = _run_sim_loop(model, data, duration,
                        v_profile_fn=lambda t: DRIVE_TURN_SPEED,
                        omega_profile_fn=lambda t: DRIVE_TURN_YAW_RATE,
                        add_noise=add_noise, rng=rng)

    fitness = (
        0.5 * W_VEL_ERR * raw['vel_track_rms_ms']
        + 0.5 * W_YAW_ERR * raw['yaw_track_rms_rads']
        + 0.1 * W_RMS     * raw['rms_pitch_deg']
        + (W_FALL if raw['fell'] else 0.0)
    )
    raw['fitness'] = round(fitness, 4)
    return raw


def run_8_terrain_compliance(gains: dict, duration: float = SCENARIO_8_DURATION,
                              add_noise: bool = True, rng_seed: int = None) -> dict:
    """8_terrain_compliance — Roll leveling + suspension test with one-sided bumps.

    Controller under test : LQR + VelocityPI + roll leveling impedance
    v_desired             : S8_DRIVE_SPEED (1.0 m/s) constant forward
    Obstacles             : S8_BUMPS — alternate left/right one-sided bumps (3–5 cm)
                            Each bump hits only one wheel → roll disturbance.
    Duration              : 12.0 s
    Metric / Fitness      : W_VEL_ERR*vel_rms + W_RMS*roll_rms + 0.1*pitch_rms + W_FALL*fell

    Pass criterion: roll_rms < 2°, no fall, velocity tracking maintained.
    """
    rng = np.random.default_rng(rng_seed)
    try:
        model, data = _build_model_and_data(sandbox_obstacles=S8_BUMPS)
        init_sim(model, data)
    except Exception as e:
        return _fail_result(f"model init: {e}")

    raw = _run_sim_loop(model, data, duration,
                        v_profile_fn=lambda t: S8_DRIVE_SPEED,
                        add_noise=add_noise, rng=rng)

    fitness = (
        W_VEL_ERR * raw['vel_track_rms_ms']
        + W_RMS   * raw['max_roll_deg']       # peak spike, not RMS — penalises worst-case bump
        + 0.1 * W_RMS * raw['rms_pitch_deg']
        + (W_FALL if raw['fell'] else 0.0)
    )
    raw['fitness'] = round(fitness, 4)
    return raw


def _fail_result(reason: str) -> dict:
    return dict(
        rms_pitch_deg=999.0, max_pitch_deg=999.0, wheel_travel_m=999.0,
        settle_time_s=999.0, survived_s=0.0,
        fitness=9999.0, status="FAIL", fail_reason=reason,
    )


# ---------------------------------------------------------------------------
# Top-level evaluate() — run + log to CSV
# ---------------------------------------------------------------------------
def evaluate(gains: dict, scenario: str = "balance", label: str = "",
             run_id: int = None, csv_path: str = CSV_PATH) -> dict:
    if scenario == "balance":
        metrics = run_balance_scenario(gains)
    elif scenario == "balance_disturbance":
        metrics = run_balance_with_disturbance_scenario(gains)
    elif scenario == "drive_slow":
        metrics = run_drive_slow_scenario(gains)
    elif scenario == "drive_medium":
        metrics = run_drive_medium_scenario(gains)
    elif scenario == "obstacle":
        metrics = run_obstacle_scenario(gains)
    elif scenario == "lqr_combined":
        metrics = run_combined_scenario(gains)
    elif scenario == "1_LQR_pitch_step":
        metrics = run_1_LQR_pitch_step(gains, duration=SCENARIO_1_DURATION)
    elif scenario == "2_LQR_impulse_recovery":
        metrics = run_2_LQR_impulse_recovery(gains)
    elif scenario == "2_VEL_PI_disturbance":
        metrics = run_2_VEL_PI_disturbance(gains, duration=SCENARIO_2_DURATION)
    elif scenario == "3_VEL_PI_staircase":
        metrics = run_3_VEL_PI_staircase(gains, duration=SCENARIO_3_DURATION)
    elif scenario == "combined_PI":
        metrics = run_combined_PI_scenario(gains)
    elif scenario == "4_leg_height_gain_sched":
        metrics = run_4_leg_height_gain_sched(gains, duration=SCENARIO_4_DURATION)
    elif scenario == "5_VEL_PI_leg_cycling":
        metrics = run_5_VEL_PI_leg_cycling(gains, duration=SCENARIO_5_DURATION)
    elif scenario == "6_YAW_PI_turn":
        metrics = run_6_YAW_PI_turn(gains, duration=SCENARIO_6_DURATION)
    elif scenario == "7_DRIVE_TURN":
        metrics = run_7_DRIVE_TURN(gains, duration=SCENARIO_7_DURATION)
    elif scenario == "8_terrain_compliance":
        metrics = run_8_terrain_compliance(gains, duration=SCENARIO_8_DURATION)
    else:
        raise ValueError(f"Unknown scenario: '{scenario}'")

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = dict(
        run_id    = run_id or next_run_id(csv_path),
        scenario  = scenario,
        label     = label,
        timestamp = ts,
        KP        = round(gains.get("KP",     0.0), 4),
        KD        = round(gains.get("KD",     0.0), 4),
        KP_pos    = round(gains.get("KP_pos", 0.0), 4),
        KP_vel    = round(gains.get("KP_vel", 0.0), 4),
    )
    row.update(metrics)
    log_run(row, csv_path)

    fit_str = f"{row['fitness']:.3f}" if isinstance(row.get("fitness"), float) else "?"
    print(f"[{row['run_id']:5d}] {label:<28}  {row['status']:<5}  "
          f"fitness={fit_str}  rms={row.get('rms_pitch_deg','?')}°  "
          f"travel={row.get('wheel_travel_m','?')}m  "
          f"surv={row.get('survived_s','?')}s")
    return row


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gains = dict(KP=60.0, KD=5.0, KP_pos=0.30, KP_vel=0.30)

    print("=" * 70)
    print("1_LQR_pitch_step  (LQR only, +5° initial perturbation)")
    print("=" * 70)
    result = run_1_LQR_pitch_step(gains, add_noise=False, rng_seed=0)
    for k, v in result.items():
        print(f"  {k:<30} = {v}")

    print("\n" + "=" * 70)
    print("2_LQR_impulse_recovery  (LQR only, horizontal impulse at t={:.1f}s)".format(
        DISTURBANCE_TIME))
    print("=" * 70)
    result = run_2_LQR_impulse_recovery(gains, add_noise=False, rng_seed=0)
    for k, v in result.items():
        print(f"  {k:<30} = {v}")
    print("Baseline gains are brittle under disturbance (high pitch error post-impact)")
    print("=" * 70)
