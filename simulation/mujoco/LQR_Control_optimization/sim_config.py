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
HIP_TORQUE_LIMIT   = 7.0    # [N·m] AK45-10 peak
WHEEL_TORQUE_LIMIT = 3.67   # [N·m] 5065 130KV @ 50 A ODESC limit

# ── Balance PD controller (optimized via (1+8)-ES, run_id=221) ─────────────
# Optimized gains for smooth, efficient balance.
# Will be replaced by LQR K computation in Step 2.
PITCH_KP      = 10.1   # [N·m/rad]       (was 60.0 baseline)
PITCH_KI      =  0.0   # [N·m/(rad·s)]  unused for now
PITCH_KD      =  0.893 # [N·m·s/rad]    (was 5.0 baseline)
POSITION_KP   =  2.16  # [rad/m]    wheel odometry → pitch lean correction (was 0.30)
VELOCITY_KP   =  0.497 # [rad/(m/s)] wheel velocity feedback (was 0.30)
MAX_PITCH_CMD =  0.25  # [rad] clamp on position/velocity feedback

# ── LQR cost weights — optimized via Phase 1 (50 gen, 400 evals) ──────────
# State: x = [pitch − θ_ref,  pitch_rate,  wheel_vel_avg − v_ref]
# u = −K @ x,  K solved from Q, R via scipy.linalg.solve_continuous_are
# Baseline (Phase 1): fitness=5.428 (balance 20% + disturbance 80%)
#   Balances cleanly: RMS pitch 1.44° over 5s
#   Recovers from disturbance: survives 1N force for 1s at t=2.5s
LQR_Q_PITCH      =   0.158  # weight on pitch error (optimized)
LQR_Q_PITCH_RATE =  0.00419 # weight on pitch rate (optimized)
LQR_Q_VEL        = 0.00196  # weight on wheel velocity error (optimized)
LQR_R            =   0.451  # weight on control effort (optimized)

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

# ── LQR Gain Scheduling Table ────────────────────────────────────────────────
# Computed in scenarios.py to avoid circular import with lqr_design.py
# Will be initialized on first use.
LQR_K_TABLE = None
