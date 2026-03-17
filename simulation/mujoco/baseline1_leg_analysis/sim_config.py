"""sim_config.py — Single source of truth for baseline-1 winning geometry.

Winning design: run_id 51167 from 70,818-run evolutionary optimisation.
All downstream scripts import from here — never hardcode values elsewhere.
"""
import math

# ── Winning Geometry ─────────────────────────────────────────────────────────
# Source: results_balanced.csv, run_id 51167, jump_height=282.65 mm
ROBOT = dict(
    L_femur   = 0.17378,    # [m] hip-to-knee (A→C)
    L_stub    = 0.03513,    # [m] tibia stub upward (C→E)
    L_tibia   = 0.12939,    # [m] tibia downward (C→W)
    Lc        = 0.15081,    # [m] coupler link (F→E)
    F_X       = -0.05887,   # [m] coupler pivot X offset from body origin
    F_Z       = -0.01821,   # [m] coupler pivot Z in body frame (negative = below hip)
    A_Z       = -0.0235,    # [m] hip motor Z offset from body centre
    m_box     = 0.477,      # [kg] body box + electronics
    m_femur   = 0.0192,     # [kg] femur link (was 0.025)
    m_tibia   = 0.0183,     # [kg] tibia link (was 0.035)
    m_coupler = 0.0094,     # [kg] coupler link (was 0.015)
    m_bearing = 0.012,      # [kg] 608 bearing
    m_wheel   = 0.270,      # [kg] 5065 130KV 200g + PLA hub 45g + TPU tread 25g
)

# ── Stroke angles (computed by find_stroke, verified against CSV) ─────────────
Q_RET      = -0.35071   # [rad] hip at full retraction (W_z highest)
Q_EXT      = -1.43161   # [rad] hip at full extension  (W_z lowest)
STROKE_DEG =  61.93

# Neutral stance: 30% of the way from retracted toward extended
NEUTRAL_FRAC = 0.30
Q_NEUTRAL    = Q_RET + NEUTRAL_FRAC * (Q_EXT - Q_RET)

# ── Physical constants ────────────────────────────────────────────────────────
WHEEL_R    = 0.075    # [m] wheel radius (150 mm diameter)
LEG_Y      = 0.1430   # [m] Y-offset of leg plane from body centre
MOTOR_MASS = 0.260    # [kg] AK45-10 hip motor
MOTOR_R_MM = 26.5     # [mm] AK45-10 housing radius

# ── Balance controller ────────────────────────────────────────────────────────
PITCH_KP          = 60.0   # [N·m/rad]
PITCH_KI          = 0.0    # [N·m/(rad·s)] — unused
PITCH_KD          = 5.0    # [N·m·s/rad]
POSITION_KP       = 0.30   # [rad/m]   wheel position → pitch correction
VELOCITY_KP       = 0.30   # [rad/(m/s)] wheel velocity → pitch correction
MAX_PITCH_CMD     = 0.25   # [rad] saturation on position/velocity feedback
WHEEL_TORQUE_LIMIT = 3.67  # [N·m]  5065 130KV: Kt=0.0735 Nm/A × 50A ODESC limit

# ── Hip / jump controller ─────────────────────────────────────────────────────
HIP_KP           = 30.0    # [N·m/rad]   full stiffness (jump)
HIP_KD           = 1.0     # [N·m·s/rad] full damping  (jump)
HIP_KP_SUSP      = 8.0     # [N·m/rad]   soft suspension spring (normal/crouch)
HIP_KD_SUSP      = 4.0     # [N·m·s/rad] high suspension damping (normal/crouch)
HIP_TORQUE_LIMIT = 7.0     # [N·m]  AK45-10 peak
OMEGA_MAX        = 18.85   # [rad/s] AK45-10 KV75 @ 24V / 10:1 = 1800 RPM = 18.85 rad/s
JUMP_RAMP_S      = 0.010   # [s]   torque ramp-in at jump start
JUMP_RAMPDOWN    = 0.15    # [rad] ramp-down zone before Q_EXT
CROUCH_DURATION_S = 1.5    # [s]   gentle crouch ramp (neutral → retracted)

# ── Simulation timing ─────────────────────────────────────────────────────────
SIM_DURATION_S   = 8.0
NEUTRAL_HOLD_S   = 2.0     # balance stabilisation before crouch
CROUCH_START_S   = 2.0     # time at which crouch begins
JUMP_TRIGGER_S   = 3.5     # time at which jump fires

# ── Motor electrical models ───────────────────────────────────────────────────
# Wheel: 5065 130KV outrunner, 24 V, direct drive
#   ω_noload = 130 KV × 24 V × 2π/60 = 326.7 rad/s  →  24.5 m/s at wheel rim
#   T_peak   = Kt × I_max = (9.55/130) × 50 A = 3.67 N·m
WHEEL_OMEGA_NOLOAD = 326.7   # [rad/s]
WHEEL_TAU_ELEC     = 0.002   # [s]  CAN transport (<0.5 ms) + ODrive FOC rise (~1 ms)
WHEEL_B_FRICTION   = 0.02    # [N·m·s/rad]  outrunner bearing drag (less than hub motor)

# Hip: AK45-10  KV75, 24 V, 10:1 planetary  (ω_noload identical to OMEGA_MAX)
HIP_TAU_ELEC       = 0.002   # [s]  CAN transport (<0.5 ms) + integrated FOC rise (~1 ms)
HIP_B_FRICTION     = 0.02    # [N·m·s/rad]  planetary gearbox viscous drag

# ── Sensor noise (realistic IMU model) ───────────────────────────────────────
ACCEL_NOISE_STD            = 0.2                  # [m/s²]
PITCH_NOISE_STD_RAD        = math.radians(0.1)    # [rad]
PITCH_RATE_NOISE_STD_RAD_S = math.radians(0.5)    # [rad/s]
