"""scenarios.py — Headless simulation scenarios for LQR_Control_optimization.

Balance controller PD gains are passed in via `gains` dict and wired directly
into the controller.  The evolutionary optimizer searches over these gains.
"""
import math
import datetime
import numpy as np
import mujoco

from sim_config import (
    ROBOT, Q_NOM, WHEEL_R,
    HIP_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT,
    LEG_K_S, LEG_B_S,
    MAX_PITCH_CMD,
    PITCH_NOISE_STD_RAD, PITCH_RATE_NOISE_STD_RAD_S, ACCEL_NOISE_STD,
    CTRL_STEPS,
    LQR_Q_PITCH, LQR_Q_PITCH_RATE, LQR_Q_VEL, LQR_R,
)
from physics import build_xml, build_assets, solve_ik, get_equilibrium_pitch
from run_log import log_run, next_run_id, CSV_PATH
from lqr_design import interpolate_gains, compute_gain_table

# ── Controller mode ──────────────────────────────────────────────────────────
USE_PD_CONTROLLER = True  # Toggle between PD (True) and LQR (False)
                          # When False, uses 3-state LQR with gain scheduling
                          # Phase 1 (Step 2) implementation

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

# ── Fitness weights ──────────────────────────────────────────────────────────
W_RMS      = 1.0     # RMS pitch error [deg]
W_TRAVEL   = 0.5     # wheel travel [m] — penalise drifting/oscillating wheels
W_FALL     = 200.0   # fell-over penalty
W_RECOVERY = 1.0     # recovery from disturbance (RMS pitch error post-disturbance)
W_LIFTOFF  = 50.0    # penalty per second any wheel is off the ground (bouncing)

LIFTOFF_THRESHOLD = WHEEL_R + 0.005   # 5 mm above nominal contact = airborne

# ── Combined scenario weights (for LQR optimization) ────────────────────────
W_BALANCE = 0.2    # weight on balance scenario fitness
W_DISTURBANCE = 0.8  # weight on disturbance scenario fitness


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
               hip_q_avg: float, v_ref: float = 0.0) -> float:
    """3-state LQR balance controller with gain scheduling.

    State: [pitch − pitch_ff, pitch_rate, wheel_vel − v_ref]
    K is interpolated online based on hip angle (leg position).

    Returns tau_wheel (symmetric) for both wheels.
    """
    pitch_ff = get_equilibrium_pitch(ROBOT, hip_q_avg)
    K = interpolate_gains(LQR_K_TABLE, hip_q_avg)

    # State vector: [pitch_error, pitch_rate, wheel_vel_error]
    x = np.array([pitch - pitch_ff, pitch_rate, wheel_vel - v_ref])

    # Control: u = -K @ x
    u = float(-np.dot(K, x))

    # Clamp to motor limit
    tau_wheel = float(np.clip(u, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT))

    return tau_wheel


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------
def _build_model_and_data(p=None):
    if p is None: p = ROBOT
    xml    = build_xml(p)
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

    # ── Metric accumulators ─────────────────────────────────────────────────
    pitch_sq_sum   = 0.0
    max_pitch      = 0.0
    wheel_travel_m = 0.0
    wheel_liftoff_s = 0.0
    n_samples      = 0
    survived_s     = duration
    settle_time    = duration
    settled        = False
    settle_start   = None

    # ── Simulation loop ─────────────────────────────────────────────────────
    # Physics: 2000 Hz.  Controller (IMU + torque cmd): 500 Hz = every CTRL_STEPS steps.
    step = 0
    dt   = model.opt.timestep * CTRL_STEPS   # 0.002 s controller dt
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

            # ── Balance controller ───────────────────────────────────────────
            hip_q_avg = (data.qpos[s_hip_L] + data.qpos[s_hip_R]) / 2.0
            pitch_ff  = get_equilibrium_pitch(ROBOT, hip_q_avg)

            if USE_PD_CONTROLLER:
                tau_wheel, pitch_integral, odo_x = balance_torque(
                    pitch, pitch_rate, pitch_integral, odo_x,
                    wheel_vel, hip_q_avg, dt, gains)
            else:
                # LQR controller (pitch_integral, odo_x not used but kept for compatibility)
                tau_wheel = lqr_torque(pitch, pitch_rate, wheel_vel, hip_q_avg, v_ref=0.0)

            data.ctrl[act_wheel_L] = tau_wheel
            data.ctrl[act_wheel_R] = tau_wheel

            # ── Leg impedance: hold Q_NOM ────────────────────────────────────
            for s_hip, d_hip, act_hip in [
                (s_hip_L, d_hip_L, act_hip_L),
                (s_hip_R, d_hip_R, act_hip_R),
            ]:
                q_hip  = data.qpos[s_hip]
                dq_hip = data.qvel[d_hip]
                tau_hip = -(LEG_K_S * (q_hip - Q_NOM) + LEG_B_S * dq_hip)
                data.ctrl[act_hip] = np.clip(tau_hip, -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)

            # ── Metrics ─────────────────────────────────────────────────────
            pitch_err_deg  = math.degrees(abs(pitch_true - pitch_ff))
            pitch_sq_sum  += pitch_err_deg ** 2
            max_pitch      = max(max_pitch, pitch_err_deg)
            vel_est        = (wheel_vel + pitch_rate_true) * WHEEL_R
            wheel_travel_m += abs(vel_est) * dt
            n_samples     += 1

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

        mujoco.mj_step(model, data)
        step += 1

    # ── Final metrics ────────────────────────────────────────────────────────
    rms_pitch_deg = math.sqrt(pitch_sq_sum / max(1, n_samples))
    fell = survived_s < duration - 0.05

    fitness = (
        W_RMS    * rms_pitch_deg
        + W_TRAVEL * wheel_travel_m
        + W_LIFTOFF * wheel_liftoff_s
        + (W_FALL if fell else 0.0)
    )

    return dict(
        rms_pitch_deg   = round(rms_pitch_deg,   4),
        max_pitch_deg   = round(max_pitch,        4),
        wheel_travel_m  = round(wheel_travel_m,   4),
        wheel_liftoff_s = round(wheel_liftoff_s,  4),
        settle_time_s   = round(settle_time,      3),
        survived_s      = round(survived_s,       3),
        fitness         = round(fitness,          4),
        status          = "FAIL" if fell else "PASS",
        fail_reason     = "fell over" if fell else "",
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

    # ── Simulation loop ─────────────────────────────────────────────────────
    step = 0
    dt   = model.opt.timestep * CTRL_STEPS   # 0.002 s controller dt
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

            # ── Balance controller ───────────────────────────────────────────
            hip_q_avg = (data.qpos[s_hip_L] + data.qpos[s_hip_R]) / 2.0
            pitch_ff  = get_equilibrium_pitch(ROBOT, hip_q_avg)

            if USE_PD_CONTROLLER:
                tau_wheel, pitch_integral, odo_x = balance_torque(
                    pitch, pitch_rate, pitch_integral, odo_x,
                    wheel_vel, hip_q_avg, dt, gains)
            else:
                # LQR controller (pitch_integral, odo_x not used but kept for compatibility)
                tau_wheel = lqr_torque(pitch, pitch_rate, wheel_vel, hip_q_avg, v_ref=0.0)

            data.ctrl[act_wheel_L] = tau_wheel
            data.ctrl[act_wheel_R] = tau_wheel

            # ── Leg impedance: hold Q_NOM ────────────────────────────────────
            for s_hip, d_hip, act_hip in [
                (s_hip_L, d_hip_L, act_hip_L),
                (s_hip_R, d_hip_R, act_hip_R),
            ]:
                q_hip  = data.qpos[s_hip]
                dq_hip = data.qvel[d_hip]
                tau_hip = -(LEG_K_S * (q_hip - Q_NOM) + LEG_B_S * dq_hip)
                data.ctrl[act_hip] = np.clip(tau_hip, -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)

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
# Combined balance + disturbance scenario (for LQR optimization)
# ---------------------------------------------------------------------------
def run_combined_scenario(gains: dict, duration: float = BALANCE_DURATION,
                          add_noise: bool = True, rng_seed: int = None) -> dict:
    """Run balance + disturbance, combine fitness with 20%/80% weighting.

    1. Run balance scenario
    2. If passes, run disturbance scenario
    3. Combined fitness = 0.2 * fit_balance + 0.8 * fit_disturbance
    4. If balance fails, return high fitness (robot must survive balance first)

    gains keys: KP, KD, KP_pos, KP_vel (dummy for LQR, not used)
    """
    # Run balance scenario first
    metrics_balance = run_balance_scenario(gains, duration, add_noise, rng_seed)

    if metrics_balance['status'] != 'PASS':
        # Failed balance scenario — immediately return high fitness
        return dict(
            rms_pitch_deg=metrics_balance['rms_pitch_deg'],
            rms_pitch_post_dist_deg=0.0,
            max_pitch_deg=metrics_balance['max_pitch_deg'],
            wheel_travel_m=metrics_balance['wheel_travel_m'],
            settle_time_s=metrics_balance['settle_time_s'],
            survived_s=metrics_balance['survived_s'],
            fitness_balance=metrics_balance['fitness'],
            fitness_disturbance=9999.0,
            fitness=9999.0,  # penalty: didn't survive balance
            status='FAIL',
            fail_reason='failed balance scenario',
        )

    # Balance passed — now run disturbance scenario
    metrics_dist = run_balance_with_disturbance_scenario(gains, duration, add_noise, rng_seed)

    fit_balance = metrics_balance['fitness']
    fit_dist = metrics_dist['fitness']
    combined_fit = W_BALANCE * fit_balance + W_DISTURBANCE * fit_dist

    return dict(
        rms_pitch_deg=metrics_balance['rms_pitch_deg'],
        rms_pitch_post_dist_deg=metrics_dist['rms_pitch_post_dist_deg'],
        max_pitch_deg=max(metrics_balance['max_pitch_deg'], metrics_dist['max_pitch_deg']),
        wheel_travel_m=metrics_balance['wheel_travel_m'] + metrics_dist['wheel_travel_m'],
        settle_time_s=metrics_balance['settle_time_s'],
        survived_s=metrics_dist['survived_s'],
        fitness_balance=round(fit_balance, 4),
        fitness_disturbance=round(fit_dist, 4),
        fitness=round(combined_fit, 4),
        status='PASS' if metrics_dist['status'] == 'PASS' else 'FAIL',
        fail_reason='',
    )


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
    elif scenario == "lqr_combined":
        metrics = run_combined_scenario(gains)
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
    print("BALANCE SCENARIO (still balance, no disturbance)")
    print("=" * 70)
    result = run_balance_scenario(gains, add_noise=False, rng_seed=0)
    for k, v in result.items():
        print(f"  {k:<30} = {v}")

    print("\n" + "=" * 70)
    print("BALANCE+DISTURBANCE SCENARIO (with mid-run horizontal impulse)")
    print("=" * 70)
    result = run_balance_with_disturbance_scenario(gains, add_noise=False, rng_seed=0)
    for k, v in result.items():
        print(f"  {k:<30} = {v}")

    print("\n" + "=" * 70)
    print("Note: Disturbance applied at t={:.1f}s for {:.0f}ms with {:.0f}N force".format(
        DISTURBANCE_TIME, DISTURBANCE_DUR * 1000, DISTURBANCE_FORCE))
    print("Baseline gains are brittle under disturbance (high pitch error post-impact)")
    print("=" * 70)
