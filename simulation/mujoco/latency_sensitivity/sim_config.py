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

# ── Battery model (6S LiPo) ─────────────────────────────────────────────────
BATT_CAPACITY_AH  = 5.0         # [Ah]    pack capacity
BATT_V_FULL       = 25.2        # [V]     fully charged (4.2 V/cell × 6)
BATT_V_NOM        = 24.0        # [V]     rated operating voltage — used to normalise motor speed
BATT_V_CUTOFF     = 18.0        # [V]     low-voltage cutoff (3.0 V/cell × 6)
BATT_R0           = 0.040       # [Ω]     internal resistance at SoC=1, T=25 °C (quality 5 Ah cell)
BATT_K_SOC        = 0.50        # []      R_int quadratic rise factor at SoC=0 (50 % higher at empty)
BATT_K_TEMP       = 0.010       # [1/°C]  Arrhenius-like temp coefficient (cold → higher R)
BATT_TEMP_REF_C   = 25.0        # [°C]    reference temperature for BATT_R0
BATT_THERMAL_MASS = 800.0       # [J/°C]  thermal mass of pack (~5 Ah LiPo)
BATT_COOL_W_PER_C = 3.0         # [W/°C]  passive convective cooling coefficient
BATT_TEMP_INIT_C  = 25.0        # [°C]    initial battery temperature
BATT_SOC_INIT     = 1.0         # []      initial state of charge (fully charged)
BATT_I_QUIESCENT  = 0.30        # [A]     always-on electronics: MCU + IMU + CAN transceivers

# ── Hip motor electrical (CubeMars AK45-10 KV75, 10:1 planetary) ─────────────
# Raw motor: Kt = 60 / (KV × 2π) = 0.1273 N·m/A
# After 10:1 gearbox (torque multiplies, current does not):
#   Kt_output = Kt_motor × gear_ratio = 1.273 N·m/A
#   I_hip = |tau_output| / HIP_KT_OUTPUT  →  ~5.5 A at 7 N·m peak
HIP_KT_OUTPUT = 60.0 / (75.0 * 2 * math.pi) * 10.0   # [N·m/A] ≈ 1.273

# ── Motor limits ────────────────────────────────────────────────────────────
HIP_TORQUE_LIMIT            = 7.0   # [N·m] AK45-10 peak (physical spec — never exceed)
HIP_IMPEDANCE_TORQUE_LIMIT  = 1.0   # [N·m] max torque the impedance controller may use
                                     # for position-holding.  Keeps the hip backdrivable:
                                     # any disturbance > this will move the leg.
                                     # Lowered from 2.0 (placeholder) → 1.0 (Phase 4.1 validation).
                                     # Full 7 N·m reserved for jump/recovery.
WHEEL_TORQUE_LIMIT          = 3.67  # [N·m] 5065 130KV @ 50 A ODESC limit
WHEEL_OMEGA_NOLOAD          = 326.7 # [rad/s] ω_noload = KV × V_batt × 2π/60 = 130 × 24 × 2π/60
WHEEL_KT                    = 0.0735 # [N·m/A] torque constant (Kt = 1/KV in SI)

# ── Balance PD controller (optimized via (1+8)-ES, run_id=221) ─────────────
# Optimized gains for smooth, efficient balance.
# Will be replaced by LQR K computation in Step 2.
PITCH_KP      = 10.1   # [N·m/rad]       (was 60.0 baseline)
PITCH_KI      =  0.0   # [N·m/(rad·s)]  unused for now
PITCH_KD      =  0.893 # [N·m·s/rad]    (was 5.0 baseline)
POSITION_KP   =  2.16  # [rad/m]    wheel odometry → pitch lean correction (was 0.30)
VELOCITY_KP   =  0.497 # [rad/(m/s)] wheel velocity feedback (was 0.30)
MAX_PITCH_CMD =  0.25  # [rad] clamp on position/velocity feedback

# ── LQR cost weights — optimized via scenario 4 (2026-03-18) ──────────────────
# State: x = [pitch − pitch_ff − θ_ref,  pitch_rate,  wheel_vel_avg − v_ref]
# u = −K @ x,  K solved from Q, R via scipy.linalg.solve_continuous_are
# Scenario 4 (4_leg_height_gain_sched) 5-min run: 237 gens, 1896 evals, 12s duration
#   Best (run_id=6426): fitness=0.017938, rms_pitch=1.242°, survived=12.0s
# S1 seed (retained in docs/Control.MD): Q=[0.138282, 0.023379, 0.004591], R=9.998298
LQR_Q_PITCH      =  0.014168  # weight on pitch error
LQR_Q_PITCH_RATE =  0.033720  # weight on pitch rate
LQR_Q_VEL        =  0.000250  # weight on wheel velocity
LQR_R            = 28.734420  # weight on control effort

# ── Leg impedance (held at Q_NOM; decoupled from balance loop) ─────────────
LEG_K_S = 16.000000  # [N·m/rad] spring stiffness  (Phase 4 re-opt: 3127 gens / 25016 evals, fitness=4.0918, 2026-03-18)
LEG_B_S =  0.798710  # [N·m·s/rad] damping        (Phase 4 re-opt; was 0.8216)

# ── Roll leveling (differential hip control, Phase 4.2) ─────────────────────
# Each hip gets a differential offset δq = K_ROLL*roll + D_ROLL*roll_rate
# so the body box stays at 0° roll on sloped / uneven terrain.
#
# Sign convention (verify with lateral disturbance, negate if wrong):
#   positive roll = left side UP (right-hand rule about +X forward)
#   δq > 0 → q_nom_L += δq (retract left) + q_nom_R -= δq (extend right)
#
# Hip safe range: 10° buffer inside joint limits to avoid end-stops.
# Robot drives at Q_NOM (mid-stroke) so full ±travel is available.
LEG_K_ROLL         = 4.000000  # [rad/rad]     roll proportional gain (Phase 4 re-opt, at upper bound)
LEG_D_ROLL         = 1.000000  # [rad·s/rad]   roll rate damping (Phase 4 re-opt, at upper bound)
ROLL_NOISE_STD_RAD = math.radians(0.05)          # [rad] BNO086 roll noise model
HIP_SAFE_MIN       = Q_EXT + math.radians(10)    # [rad] -1.257 (extended limit + 10°)
HIP_SAFE_MAX       = Q_RET - math.radians(10)    # [rad] -0.526 (retracted limit - 10°)

# ── Sensor noise (BNO086 realistic model) ──────────────────────────────────
PITCH_NOISE_STD_RAD        = math.radians(0.1)    # [rad]
PITCH_RATE_NOISE_STD_RAD_S = math.radians(0.5)    # [rad/s]
ACCEL_NOISE_STD            = 0.2                  # [m/s²]

# ── Simulation timing ───────────────────────────────────────────────────────
SIM_TIMESTEP = 0.0005                                     # [s] MuJoCo timestep (2 kHz)
CTRL_HZ      = 500                                        # [Hz] balance controller rate
CTRL_STEPS   = round(1.0 / (SIM_TIMESTEP * CTRL_HZ))     # MuJoCo steps per control call = 4

# ── Latency model (Phase 6) ──────────────────────────────────────────────────
# Set USE_LATENCY_MODEL = False to revert to zero-latency behaviour for A/B comparison.
# Buffer depth = round(delay_s / SIM_TIMESTEP) — auto-adjusts if physics rate changes.
USE_LATENCY_MODEL  = True
SENSOR_DELAY_S     = 0.005    # [s] BNO086 fusion ~5 ms + I2C ~0.4 ms  → 10 steps @ 2 kHz
ACTUATOR_DELAY_S   = 0.0025   # [s] ODESC FOC ~0.5 ms + τ_elec ~2 ms   →  5 steps @ 2 kHz

SCENARIO_1_DURATION = 5.0                                 # [s] 5s — only transient recovery matters (no steady-state scoring)
SCENARIO_2_DURATION = 12.0                                # [s] 12s — extends post-kick steady-state window from 2.8s → 8.8s
SCENARIO_3_DURATION = 13.0                                # [s] 13s — staircase: 0→+0.3→+0.6→+1.0→-0.5→-1.0→0 m/s (1s settle)
SCENARIO_5_DURATION = 13.0                                # [s] 13s — same staircase as S3, legs cycling throughout

# ── Scenario 5 bump obstacles — (x_position [m], height [m]) ─────────────────
# Placed along the forward (+X) and backward (−X) drive paths.
# Bump footprint: 4 cm wide × 1 m across (covers both wheels).
# Heights alternate 1 cm / 3 cm for variety.
S5_BUMPS = [
    ( 0.8, 0.01),   # 1 cm — encountered during +0.3 m/s phase
    ( 2.0, 0.03),   # 3 cm — encountered during +0.6 m/s phase
    ( 3.5, 0.01),   # 1 cm — encountered during +1.0 m/s phase
    (-0.8, 0.03),   # 3 cm — encountered during −0.5 m/s return
    (-2.0, 0.01),   # 1 cm — encountered during −1.0 m/s return
]

# ── Velocity PI outer loop (drive mode) ─────────────────────────────────────
# Outer loop converts velocity error → lean angle command (theta_ref).
# theta_ref is added to pitch_ff inside lqr_torque: state[0] = pitch - pitch_ff + theta_ref.
# Positive theta_ref = lean forward command = drive forward.
# Starting gains from Control.MD; optimizer will tune via (1+8)-ES.
VELOCITY_PI_KP = 0.502418  # [rad/(m/s)] proportional gain — 2× for snappier accel (was 0.251209 optimizer baseline)
VELOCITY_PI_KI = 0.011405  # [rad/m]     integral gain  (S5 5-min, fitness=2.0635, vel_rms=0.502 m/s)
# Prior combined_PI baseline (retained for reference): KP_V=0.502932, KI_V=0.012678, fitness=0.61
VELOCITY_PI_THETA_MAX    = 0.26   # [rad] ±15° hard clamp on theta_ref output
VELOCITY_PI_INT_MAX      = 2.0    # [rad·s] integrator anti-windup clamp
THETA_REF_RATE_LIMIT     = 5.0    # [rad/s] max rate of change of theta_ref from VelocityPI — 2.5× for snappier accel (was 2.0)
                                   # Prevents pitch_rate spikes when lean command steps abruptly.

# ── Drive scenario parameters ────────────────────────────────────────────────
DRIVE_SLOW_SPEED   = 0.3    # [m/s] slow scenario target speed
DRIVE_MEDIUM_SPEED = 0.8    # [m/s] medium scenario target speed
DRIVE_DURATION     = 7.0    # [s] total drive scenario: 3.5 s fwd + 3.5 s bwd
DRIVE_REV_TIME     = 3.5    # [s] time to switch from forward to backward

# ── Obstacle scenario parameters ─────────────────────────────────────────────
OBSTACLE_DURATION  = 5.0    # [s] drive-forward-only, robot hits step at ~t=1.7 s
OBSTACLE_HEIGHT    = 0.03   # [m] 3 cm floor step
OBSTACLE_X         = 0.50   # [m] step front face X position (1 m in front of start)

# ── Scenario 1 parameters (LQR pitch step) ───────────────────────────────────
PITCH_STEP_RAD = math.radians(5.0)   # [rad] initial pitch perturbation for 1_LQR_pitch_step

# ── Scenario 2 disturbances (VelocityPI position hold, lighter than S1) ──────
S2_DIST1_TIME  = 2.0    # [s] first kick (forward)
S2_DIST1_FORCE = 1.0    # [N] horizontal (+X) — lighter, tests PI not LQR
S2_DIST1_DUR   = 0.2    # [s]
S2_DIST2_TIME  = 3.0    # [s] second kick (backward)
S2_DIST2_FORCE = -1.0   # [N] horizontal (−X)
S2_DIST2_DUR   = 0.2    # [s]

# ── Scenario 4 — leg-height gain scheduling validation ───────────────────────
# Like S1 (LQR only, VelocityPI OFF). Legs cycle through full stroke while
# the LQR must keep pitch near the live equilibrium pitch_ff(q_hip).
# Exercises the gain scheduler at all heights; position drift is acceptable.
SCENARIO_4_DURATION  = 12.0    # [s] 3 full leg cycles
LEG_CYCLE_PERIOD     =  4.0    # [s] one full retracted→extended→retracted cycle
                                #     half-period = 2 s (one extreme to the other)
LEG_CYCLE_Q_RET      = Q_RET - math.radians(5.0)  # [rad] S4 crouched setpoint — 5° less
                                                   # crouched than Q_RET to avoid 4-bar
                                                   # instability at fully-retracted position

# ── Scenario 6 parameters (YawPI pure turn) ──────────────────────────────────
SCENARIO_6_DURATION  = 8.0    # [s] 1s settle + 6.28s turn + 0.72s tail
YAW_TURN_RATE        = 1.0    # [rad/s] target yaw rate (one full revolution in 6.28s)
YAW_ERR_START        = 1.0    # [s] skip first 1s settle period from yaw error metric

# ── Scenario 8 parameters (one-sided bumps — roll leveling test) ─────────────
# Bumps hit only ONE wheel at a time to create a roll disturbance.
# Alternates left / right so the roll leveling controller is exercised in both directions.
# Uses sandbox_obstacles format (x, y, h, rx, ry) so y-position can be one-sided.
# Wheel planes are at y = ±LEG_Y = ±0.143 m; bump y-half-size = 0.15 m covers one wheel.
SCENARIO_8_DURATION = 12.0    # [s] constant forward drive
S8_DRIVE_SPEED      =  1.0    # [m/s] fast enough to make roll disturbances significant
S8_BUMPS = [
    {'shape': 'box', 'x':  1.2, 'y':  LEG_Y, 'h': 0.05, 'rx': 0.02, 'ry': 0.15},  # 5 cm left
    {'shape': 'box', 'x':  3.0, 'y': -LEG_Y, 'h': 0.03, 'rx': 0.02, 'ry': 0.15},  # 3 cm right
    {'shape': 'box', 'x':  5.0, 'y':  LEG_Y, 'h': 0.05, 'rx': 0.02, 'ry': 0.15},  # 5 cm left
    {'shape': 'box', 'x':  7.0, 'y': -LEG_Y, 'h': 0.03, 'rx': 0.02, 'ry': 0.15},  # 3 cm right
    {'shape': 'box', 'x':  9.5, 'y':  LEG_Y, 'h': 0.05, 'rx': 0.02, 'ry': 0.15},  # 5 cm left
]

# ── Scenario 7 parameters (drive+turn cross-coupling check) ──────────────────
SCENARIO_7_DURATION  = 8.0    # [s]
DRIVE_TURN_SPEED     = 0.3    # [m/s] forward drive during combined scenario
DRIVE_TURN_YAW_RATE  = 0.5    # [rad/s] simultaneous yaw rate during combined scenario

# ── Yaw PI outer loop ────────────────────────────────────────────────────────
# Differential torque: tau_L = tau_sym + tau_yaw,  tau_R = tau_sym − tau_yaw
# Yaw rate measured from data.qvel[5] (world-frame ωz, positive = CCW = left turn).
# Independent of pitch — symmetric (LQR/VelocityPI) and differential (YawPI) modes
# are orthogonal in control space; the average wheel velocity is unaffected by tau_yaw.
YAW_PI_KP         = 2.272   # [N·m / (rad/s)] proportional gain — (1+8)-ES, 6969 gens / 55752 evals, fitness=0.4102, 2026-03-18
YAW_PI_KI         = 1.125   # [N·m / rad]     integral gain     — (was visual 0.3/0.05; optimizer found 7.5× larger Kp)
YAW_PI_TORQUE_MAX = 0.5     # [N·m] differential torque clamp (±0.5 N·m each wheel)
YAW_PI_INT_MAX    = 0.5     # [N·m·s] integrator anti-windup

# ── LQR Gain Scheduling Table ────────────────────────────────────────────────
# Computed in scenarios.py to avoid circular import with lqr_design.py
# Will be initialized on first use.
LQR_K_TABLE = None
