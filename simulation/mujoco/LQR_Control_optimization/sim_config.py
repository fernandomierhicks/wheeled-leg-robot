"""sim_config.py — Single source of truth for LQR_Control_optimization.

Robot geometry: baseline-1 winning design (run_id 51167, 282.65 mm jump).
All files in this folder import from here; nothing is imported from outside.
"""
import math

# ── Robot Geometry (baseline-1, run_id 51167) ──────────────────────────────
ROBOT = dict(
    L_femur   = 0.17378,    # [m] hip-to-knee (A→C)
    L_stub    = 0.03513,    # [m] tibia stub upward (C→E)
    L_tibia   = 0.12939,    # [m] tibia downward (C→W)
    Lc        = 0.15081,    # [m] coupler link (F→E)
    F_X       = -0.05887,   # [m] coupler pivot X from body origin
    F_Z       = -0.01821,   # [m] coupler pivot Z in body frame
    A_Z       = -0.0235,    # [m] hip motor Z offset from body centre
    m_box     = 0.477,      # [kg] body box + electronics
    m_femur   = 0.0192,     # [kg] femur link
    m_tibia   = 0.0183,     # [kg] tibia link
    m_coupler = 0.0094,     # [kg] coupler link
    m_bearing = 0.012,      # [kg] 608 bearing
    m_wheel   = 0.270,      # [kg] wheel assembly (motor + hub + tyre)
)

# ── Hip stroke angles (verified against run_id 51167) ──────────────────────
Q_RET      = -0.35071   # [rad] fully retracted  (W_z highest — crouch)
Q_EXT      = -1.43161   # [rad] fully extended   (W_z lowest  — jump)
Q_NOM      = Q_RET + 0.30 * (Q_EXT - Q_RET)   # nominal stance (30% of stroke)
STROKE_DEG =  61.93     # [deg]

# ── Physical constants ──────────────────────────────────────────────────────
WHEEL_R    = 0.075    # [m] wheel radius (150 mm diameter)
LEG_Y      = 0.1430   # [m] Y-offset of leg plane from body centre
MOTOR_MASS = 0.260    # [kg] AK45-10 hip motor

# ── Motor limits ────────────────────────────────────────────────────────────
HIP_TORQUE_LIMIT            = 7.0   # [N·m] AK45-10 peak (physical spec — never exceed)
HIP_IMPEDANCE_TORQUE_LIMIT  = 2.0   # [N·m] max torque the impedance controller may use
                                     # for position-holding.  Keeps the hip backdrivable:
                                     # any disturbance > this will move the leg.
                                     # Tune this later; full 7 N·m reserved for jump/recovery.
WHEEL_TORQUE_LIMIT          = 3.67  # [N·m] 5065 130KV @ 50 A ODESC limit

# ── Balance PD controller (optimized via (1+8)-ES, run_id=221) ─────────────
# Optimized gains for smooth, efficient balance.
# Will be replaced by LQR K computation in Step 2.
PITCH_KP      = 10.1   # [N·m/rad]       (was 60.0 baseline)
PITCH_KI      =  0.0   # [N·m/(rad·s)]  unused for now
PITCH_KD      =  0.893 # [N·m·s/rad]    (was 5.0 baseline)
POSITION_KP   =  2.16  # [rad/m]    wheel odometry → pitch lean correction (was 0.30)
VELOCITY_KP   =  0.497 # [rad/(m/s)] wheel velocity feedback (was 0.30)
MAX_PITCH_CMD =  0.25  # [rad] clamp on position/velocity feedback

# ── LQR cost weights — optimized via balance-only 2-min run (2026-03-18) ──────
# State: x = [pitch − pitch_ff − θ_ref,  pitch_rate,  wheel_vel_avg − v_ref]
# u = −K @ x,  K solved from Q, R via scipy.linalg.solve_continuous_are
# Balance-only run: fitness=2.94 (wheel_travel_m, VelocityPI not yet active)
#   run_id=2904, Q_VEL=0.0269 key insight (vs Phase 2 Q_VEL≈0.0001 → drift)
LQR_Q_PITCH      =  0.654   # weight on pitch error
LQR_Q_PITCH_RATE =  0.134   # weight on pitch rate
LQR_Q_VEL        =  0.0269  # weight on wheel velocity (actively damps drift)
LQR_R            =  9.275   # weight on control effort

# ── Leg impedance (held at Q_NOM; decoupled from balance loop) ─────────────
LEG_K_S = 8.0    # [N·m/rad] spring stiffness  (matches baseline1 HIP_KP_SUSP)
LEG_B_S = 4.0    # [N·m·s/rad] damping        (matches baseline1 HIP_KD_SUSP)

# ── Sensor noise (BNO086 realistic model) ──────────────────────────────────
PITCH_NOISE_STD_RAD        = math.radians(0.1)    # [rad]
PITCH_RATE_NOISE_STD_RAD_S = math.radians(0.5)    # [rad/s]
ACCEL_NOISE_STD            = 0.2                  # [m/s²]

# ── Simulation timing ───────────────────────────────────────────────────────
SIM_TIMESTEP = 0.0005                                     # [s] MuJoCo timestep (2 kHz)
CTRL_HZ      = 500                                        # [Hz] balance controller rate
CTRL_STEPS   = round(1.0 / (SIM_TIMESTEP * CTRL_HZ))     # MuJoCo steps per control call = 4

# ── Velocity PI outer loop (drive mode) ─────────────────────────────────────
# Outer loop converts velocity error → lean angle command (theta_ref).
# theta_ref is added to pitch_ff inside lqr_torque: state[0] = pitch - pitch_ff + theta_ref.
# Positive theta_ref = lean forward command = drive forward.
# Starting gains from Control.MD; optimizer will tune via (1+8)-ES.
VELOCITY_PI_KP = 0.299   # [rad/(m/s)] proportional gain (balance-only opt, 2026-03-18, sign-fixed)
VELOCITY_PI_KI = 0.500   # [rad/m]     integral gain  (balance-only opt, 2026-03-18, sign-fixed; at upper bound)
VELOCITY_PI_THETA_MAX = 0.26   # [rad] ±15° hard clamp on theta_ref output
VELOCITY_PI_INT_MAX   = 2.0    # [rad·s] integrator anti-windup clamp

# ── Drive scenario parameters ────────────────────────────────────────────────
DRIVE_SLOW_SPEED   = 0.3    # [m/s] slow scenario target speed
DRIVE_MEDIUM_SPEED = 0.8    # [m/s] medium scenario target speed
DRIVE_DURATION     = 7.0    # [s] total drive scenario: 3.5 s fwd + 3.5 s bwd
DRIVE_REV_TIME     = 3.5    # [s] time to switch from forward to backward

# ── Obstacle scenario parameters ─────────────────────────────────────────────
OBSTACLE_DURATION  = 5.0    # [s] drive-forward-only, robot hits step at ~t=1.7 s
OBSTACLE_HEIGHT    = 0.03   # [m] 3 cm floor step
OBSTACLE_X         = 0.50   # [m] step front face X position (1 m in front of start)

# ── LQR Gain Scheduling Table ────────────────────────────────────────────────
# Computed in scenarios.py to avoid circular import with lqr_design.py
# Will be initialized on first use.
LQR_K_TABLE = None
