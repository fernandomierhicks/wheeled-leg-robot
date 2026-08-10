"""params.py — Frozen dataclass hierarchy for all simulation parameters.

Lineage: original — first dataclass-based sim with jump support.
Forked from archive/LQR_Control_optimization + archive/latency_sensitivity,
then re-optimised with BNO086-measured noise and reduced latency model.

Every parameter that was a module-level global in sim_config.py is now a field
on one of these immutable dataclasses.  Optimizer creates modified copies via
dataclasses.replace().  No file I/O, no Pydantic, no YAML — pure Python.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple, List


# ── Control gains ──────────────────────────────────────────────────────────

################################## LQR
@dataclass(frozen=True)
class LQRGains:
    """LQR cost weights — state: [pitch-θ_ref, pitch_rate, wheel_vel_avg-v_ref].
    From 'params good gains 3_22_26.py': 500 Hz, 0 ms delay, no forecaster.
    """
    Q_pitch: float = 0.01
    Q_pitch_rate: float = 0.1884
    Q_vel: float = 0.00508442
    R: float = 100.0
################################## VEL_PI
@dataclass(frozen=True)
class VelocityPIGains:
    """Velocity PI outer loop — velocity error → lean angle."""
    Kp: float = 0.2
    Ki: float = 0.1
    Kff: float = 0.1049             # [s²·rad/m] ≈ 1/g — feed-forward: lean per unit dv_target/dt

    #to mcuh and robot linkages touch ground.
    theta_max: float = 0.698132        # [rad] ±46° clamp  Max commandable lean angle.

    int_max: float = 1.0           # [rad·s] anti-windup ( theta_max / Ki)

    #Allows for more aggresiv egains without making LQR unstable
    theta_ref_rate_limit: float = 1.745329  # [rad/s]  How fast can commanded lean angle change every tick (larger number = more aggresive)

################################## YAW
@dataclass(frozen=True)
class YawPIGains:
    """Yaw PI — differential torque for yaw rate tracking."""
    Kp: float = 0.1
    Ki: float = 0.2
    omega_cmd_max: float = 4.0  # [rad/s] 60 deg/s — max commandable yaw rate
    torque_max: float = 0.5        # [N·m] differential clamp
    int_max: float = 0.5           # [N·m·s] anti-windup

################################## SUSPENSION
@dataclass(frozen=True)
class SuspensionGains:
    """Leg impedance + roll leveling (Phase 4)."""
    K_s: float = 20.0               # [N·m/rad] spring stiffness
    B_s: float = 0.5              # [N·m·s/rad] damping
    K_roll: float = 10.0            # [rad/rad] roll proportional
    D_roll: float = 0.15           # [rad·s/rad] roll rate damping

    # Adaptive suspension: soften during free-fall, ramp back after landing
    freefall_az_threshold: float = 2.0   # [m/s²] |az_imu| below this → free-fall
    freefall_scale: float = 0.15         # K_s and B_s multiplier during free-fall
    landing_az_threshold: float = 15.0  # [m/s²] az_imu spike above this → landing
    landing_ramp_s: float = 0.75        # [s] ramp duration back to nominal gains

    @staticmethod
    def hip_safe_range(robot: RobotGeometry) -> Tuple[float, float]:
        """(hip_safe_min, hip_safe_max) — 10° buffer inside joint limits."""
        return (
            robot.Q_EXT + math.radians(10),
            robot.Q_RET - math.radians(10),
        )

################################## KNEE SPRING
@dataclass(frozen=True)
class KneeSpringParams:
    """Knee spring — disabled (found not useful for jump performance).
    All values zeroed so springs contribute no mass, torque, or inertia.
    """
    K_spring: float = 0.0             # [N·m/rad] torsional stiffness
    engage_offset: float = -0.10      # [rad] offset from Q_NOM toward Q_EXT (unused at K=0)
    B_spring: float = 0.0             # [N·m·s/rad] damping
    m_spring: float = 0.0             # [kg] mass of one spring

    # Physical spring properties (zeroed — no spring present)
    OD_m: float = 0.020               # [m] outer diameter (kept for geometry if re-enabled)
    wire_dia_m: float = 0.0           # [m] wire diameter
    body_length_m: float = 0.012      # [m] coil stack height (kept for geometry if re-enabled)
    material: str = "none"
    max_deflection_rad: float = 0.0   # [rad]

    @property
    def max_torque(self) -> float:
        """Peak spring torque [N·m] at max deflection."""
        return self.K_spring * self.max_deflection_rad

    @property
    def energy_per_spring_J(self) -> float:
        """Energy stored [J] in one spring at max deflection."""
        return 0.5 * self.K_spring * self.max_deflection_rad ** 2

    @property
    def I_sphere(self) -> float:
        """Moment of inertia [kg·m²] approximated as solid sphere of radius OD/2."""
        r = self.OD_m / 2.0
        return 0.4 * self.m_spring * r * r

################################## JUMP
@dataclass(frozen=True)
class JumpGains:
    """Jump state-machine parameters (Phase 1 — constant gains)."""
    crouch_time: float = 0.1       # [s] PD trajectory duration to Q_RET
    max_torque: float = 8.0           # [N·m] AK45-10 protocol limit
    omega_max: float = 20.0           # [rad/s] AK45-10 protocol speed limit
    ramp_up_s: float = 0.01          # [s] torque ramp-up to avoid shock
    ramp_down_rad: float = 0.05       # [rad] taper zone before Q_EXT
    retract_time: float = 0.01         # [s] Q_EXT→Q_NOM PD trajectory duration
    retract_done_rad: float = 0.05     # [rad] hip within this of Q_NOM → FLYING
    min_airborne_s: float = 0.10      # [s] minimum time in FLYING before landing can trigger
    extend_timeout_s: float = 1.0     # [s] max time in EXTEND before aborting jump
    settle_pitch_deg: float = 10.0     # [deg] pitch threshold for SETTLED
    settle_time_s: float = 0.5        # [s] stable window for SETTLED
    landing_timeout_s: float = 0.5    # [s] max time in LANDING before forcing SETTLED
    wheel_hold_gain: float = 0.15     # [N·m·s/rad] wheel velocity damping during CROUCH+EXTEND
    wheel_ff_torque: float = 0.0     # [N·m] reverse feedforward torque during EXTEND only


################################## FEED FORWARD
@dataclass(frozen=True)
class FeedforwardGains:
    """Feedforward terms (FF1, FF2, FF4).  alpha=0 disables; alpha=1 full cancellation."""
    ff1_alpha: float = 0.5   # Hip reaction torque cancellation gain
    ff2_alpha: float = 1.0   # Gravity compensation torque on wheels gain
    ff4_alpha: float = 0.0   # Centripetal turn coupling pitch offset gain


@dataclass(frozen=True)
class LegacyPDGains:
    """Balance PD controller (pre-LQR, kept for reference)."""
    pitch_Kp: float = 10.1
    pitch_Ki: float = 0.0
    pitch_Kd: float = 0.893
    position_Kp: float = 2.16
    velocity_Kp: float = 0.497
    max_pitch_cmd: float = 0.25


@dataclass(frozen=True)
class GainSet:
    """All control gains."""
    lqr: LQRGains = field(default_factory=LQRGains)
    velocity_pi: VelocityPIGains = field(default_factory=VelocityPIGains)
    yaw_pi: YawPIGains = field(default_factory=YawPIGains)
    suspension: SuspensionGains = field(default_factory=SuspensionGains)
    knee_spring: KneeSpringParams = field(default_factory=KneeSpringParams)
    jump: JumpGains = field(default_factory=JumpGains)
    feedforward: FeedforwardGains = field(default_factory=FeedforwardGains)
    legacy_pd: LegacyPDGains = field(default_factory=LegacyPDGains)


# ── Simulation settings ──────────────────────────────────────────────────

@dataclass(frozen=True)
class SimTiming:
    """MuJoCo timestep and control rate."""
    sim_timestep: float = 0.0005       # [s] 2 kHz physics
    ctrl_hz: int = 500                 # [Hz] balance controller rate

    @property
    def ctrl_steps(self) -> int:
        """MuJoCo steps per control call."""
        return round(1.0 / (self.sim_timestep * self.ctrl_hz))


@dataclass(frozen=True)
class LatencyParams:
    """Ring-buffer delays for sensor and actuator pipelines.

    Both delays are sampled at the control rate (dt_ctrl).  Buffer depth is
    round(delay_s / dt_ctrl) control steps.  Total round-trip delay seen by
    the controller = sensor_delay_s + actuator_delay_s.
    """
    sensor_delay_s: float = 0.002       # [s] 0 = disabled
    actuator_delay_s: float = 0.001     # [s] 0 = disabled


@dataclass(frozen=True)
class NoiseParams:
    """BNO086 measured noise (Test 5, 2026-03-22).
    GRV fused: 333 Hz, 30s stationary.  Raw gyro: 100 Hz, 60s.  Raw accel: 32 Hz, 30s.
    """
    pitch_std_rad: float = 0.000176        # 0.0101 deg — GRV fused pitch noise
    pitch_rate_std_rad_s: float = 0.002116 # 0.121 deg/s — raw gyro Y-axis
    accel_std: float = 0.008               # [m/s²] RMS across X/Y/Z
    roll_std_rad: float = 0.000156         # 0.0090 deg — GRV fused roll noise


@dataclass(frozen=True)
class MetricThresholds:
    """Thresholds used by sim_loop for fall detection, settle detection, etc."""
    fall_rad: float = 0.785            # [rad] ~45° — robot considered fallen
    settle_deg: float = 2.0            # [deg] |pitch| below this = settled
    settle_window_s: float = 0.5       # [s]  must stay settled this long
    vel_err_start_s: float = 1.0       # [s]  skip first 1.0 s for velocity metric
    transient_window_s: float = 1.0    # [s]  penalise |v_error| after step change
    liftoff_margin_m: float = 0.005    # [m]  wheel Z > wheel_r + this = liftoff


@dataclass(frozen=True)
class HardwareLimits:
    """Per-tick safety checks — violations logged, early-exit in optimizer."""
    max_bearing_force: float = 500.0   # [N] 608 bearing static rating
    max_motor_current_hip: float = 6.0     # [A] AK45-10 peak phase current at 7 N·m (datasheet curve @24V)
    max_motor_current_wheel: float = 50.0  # [A] ODESC limit
    max_winding_temp_C: float = 130.0  # [°C] Class B insulation


@dataclass(frozen=True)
class ScenarioTimings:
    """Duration and timing constants for all scenarios."""
    s1_duration: float = 5.0
    s2_duration: float = 12.0
    s3_duration: float = 12.0
    s4_duration: float = 13.0
    s5_duration: float = 13.0
    s6_duration: float = 8.0
    s7_duration: float = 8.0
    s8_duration: float = 12.0
    s9_duration: float = 16.0

    # S1 — LQR pitch step
    pitch_step_rad: float = field(default_factory=lambda: math.radians(5.0))

    # S2 — disturbances
    s2_dist1_time: float = 2.0
    s2_dist1_force: float = 1.0
    s2_dist1_dur: float = 0.2
    s2_dist2_time: float = 3.0
    s2_dist2_force: float = -1.0
    s2_dist2_dur: float = 0.2

    # S4 — leg cycling
    leg_cycle_period: float = 4.0

    # S6 — yaw turn
    yaw_turn_rate: float = 1.0
    yaw_err_start: float = 1.0

    # S7 — drive+turn
    drive_turn_speed: float = 1.0
    drive_turn_yaw_rate: float = 1.0472  # 60 deg/s

    # S8 — terrain compliance
    s8_drive_speed: float = 1.0

    # Drive scenario
    drive_slow_speed: float = 0.3
    drive_medium_speed: float = 0.8
    drive_duration: float = 7.0
    drive_rev_time: float = 3.5

    # Obstacle scenario
    obstacle_duration: float = 5.0
    obstacle_height: float = 0.03
    obstacle_x: float = 0.50



@dataclass(frozen=True)
class S5Bump:
    """Speed bump for Scenario 5."""
    x: float       # [m] position along X
    height: float  # [m] bump height


@dataclass(frozen=True)
class S8Bump:
    """One-sided bump for Scenario 8 (roll leveling test)."""
    shape: str
    x: float
    y: float
    h: float
    rx: float
    ry: float


# ── Battery and motor models ─────────────────────────────────────────────

@dataclass(frozen=True)
class BatteryParams:
    """6S LiPo battery model (1800 mAh, 50C cont / 100C burst, 275 g)."""
    capacity_Ah: float = 1.8
    V_full: float = 25.2           # [V] 4.2 V/cell × 6
    V_nom: float = 24.0            # [V] operating voltage (assumed fully charged, ~4.0 V/cell)
    V_cutoff: float = 18.0         # [V] 3.0 V/cell × 6
    I_cont_max: float = 90.0       # [A] 50C × 1.8 Ah continuous
    I_burst_max: float = 180.0     # [A] 100C × 1.8 Ah burst
    R0: float = 0.040              # [Ω] internal resistance at SoC=1
    K_soc: float = 0.50            # R_int rise factor at SoC=0
    K_temp: float = 0.010          # [1/°C] Arrhenius-like temp coefficient
    temp_ref_C: float = 25.0       # [°C]
    thermal_mass: float = 800.0    # [J/°C]
    cool_W_per_C: float = 3.0      # [W/°C] passive convective cooling
    temp_init_C: float = 25.0      # [°C]
    SoC_init: float = 1.0
    I_quiescent: float = 0.30      # [A] always-on electronics


@dataclass(frozen=True)
class HipMotorParams:
    """CubeMars AK45-10 KV75, 10:1 planetary."""
    KV: float = 75.0               # [RPM/V] raw motor KV
    gear_ratio: float = 10.0       # planetary gearbox ratio
    torque_limit: float = 8.0      # [N·m] AK45-10 MIT protocol output limit
    impedance_torque_limit: float = 5.0   # [N·m] S8 impedance cap
    position_Kp: float = 50.0      # [N·m/rad] position servo stiffness
    position_Kd: float = 3.0       # [N·m·s/rad] position servo damping

    @property
    def Kt_output(self) -> float:
        """Output-side torque constant [N·m/A]."""
        return 60.0 / (self.KV * 2 * math.pi) * self.gear_ratio

    # Thermal model
    R_eff: float = 2.19            # [Ω] phase-to-phase, derived from measured output Kt/current
    C_winding: float = 30.0        # [J/°C] copper thermal mass
    C_case: float = 200.0          # [J/°C] motor + gearbox housing
    R_th_wc: float = 1.0           # [°C/W] winding-to-case
    R_th_ca: float = 6.0           # [°C/W] case-to-air
    T_max_C: float = 130.0         # [°C] Class B insulation limit


@dataclass(frozen=True)
class WheelMotorParams:
    """Maytech MTO5065-70-HA-C direct drive."""
    KV: float = 70.0               # [RPM/V]
    current_limit: float = 50.0    # [A] ODESC limit

    @property
    def Kt(self) -> float:
        """Torque constant [N·m/A]."""
        return 9.55 / self.KV

    @property
    def torque_limit(self) -> float:
        """Peak torque [N·m] = Kt × I_max."""
        return self.Kt * self.current_limit

    def omega_noload(self, v_batt: float) -> float:
        """No-load speed [rad/s] at given battery voltage."""
        return self.KV * v_batt * (2 * math.pi / 60)

    # Thermal model
    R_eff: float = 0.24            # [Ω] effective winding resistance
    C_winding: float = 20.0        # [J/°C]
    C_case: float = 80.0           # [J/°C]
    R_th_wc: float = 1.5           # [°C/W]
    R_th_ca: float = 8.0           # [°C/W]
    T_max_C: float = 130.0         # [°C]


@dataclass(frozen=True)
class MotorParams:
    """All motor parameters (hip + wheel)."""
    hip: HipMotorParams = field(default_factory=HipMotorParams)
    wheel: WheelMotorParams = field(default_factory=WheelMotorParams)
    T_amb_C: float = 25.0          # [°C] ambient temperature


# ── Robot geometry ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RobotGeometry:
    """v4 four-bar geometry, with drawing coordinates stored relative to A.

    ``F_X_A``/``F_Z_A`` are the measured A-at-origin coordinates. The
    body-centre values consumed by MuJoCo are derived properties so the two
    frames cannot be confused. ``A_Z`` remains provisional until test T0.3.
    """
    L_femur: float = 0.187577       # [m] A->C
    L_stub: float = 0.039013        # [m] C->E
    L_tibia: float = 0.183411       # [m] axial C->W component in tibia frame
    w_perp: float = -0.030354       # [m] optimizer local-Z coordinate of W
    Lc: float = 0.169536            # [m] F->E
    F_X_A: float = -0.036425        # [m] F X relative to A
    F_Z_A: float = +0.037537        # [m] F Z relative to A
    A_Z: float = -0.0235            # [m] hip shaft A below body centre; unverified

    # Catalog/CAD mass prior. T0.1 replaces the aggregate values with scale data.
    # Tube masses use 6061 density=2700 kg/m^3 and the v4 centreline lengths.
    m_box: float = 0.985            # [kg] itemised COMPONENTS.md body contents
    m_femur: float = 0.020684097    # [kg] 14x1 mm tube, A-C=187.577 mm
    m_tibia: float = 0.028617429    # [kg] 16x1 mm tube, C-W chord + C-E stub
    m_coupler: float = 0.012942497  # [kg] 10x1 mm tube, F-E=169.536 mm
    m_bearing: float = 0.0175       # [kg] equivalent lump; 8 lumps = 140 g BOM total
    wheel_motor_mass: float = 0.450 # [kg] Maytech MTO5065 catalog mass
    wheel_print_mass: float = 0.070 # [kg] BOM prior; v4 wheel has not been re-sliced
    m_wheel: float = 0.520          # [kg] translating assembly prior per side

    wheel_r: float = 0.056          # [m] wheel radius (112 mm OD)
    leg_y: float = 0.1430           # [m] Y-offset of leg plane from body centre
    motor_mass: float = 0.260       # [kg] BASELINE 0.260 — AK45-10 hip motor

    @property
    def body_mass_excluding_wheels(self) -> float:
        """Catalog/CAD body prior using the same eight bearing lumps as MuJoCo."""
        return (self.m_box
                + 2.0 * (self.motor_mass + self.m_femur + self.m_tibia
                         + self.m_coupler)
                + 8.0 * self.m_bearing)

    @property
    def total_mass(self) -> float:
        """Catalog/CAD whole-robot prior, including both wheel assemblies."""
        return self.body_mass_excluding_wheels + 2.0 * self.m_wheel

    # CAD hard stops in the firmware physical convention: zero is a horizontal
    # femur and positive rotation retracts the wheel.
    Q_RET: float = field(default_factory=lambda: math.radians(+28.0))
    Q_EXT: float = field(default_factory=lambda: math.radians(-57.0))
    calib_backoff_rad: float = 0.0872665  # [rad] retract-switch backoff

    @property
    def F_X(self) -> float:
        """Body-centre X coordinate of F (A has no X offset)."""
        return self.F_X_A

    @property
    def F_Z(self) -> float:
        """Body-centre Z coordinate of F, derived from the measured A frame."""
        return self.A_Z + self.F_Z_A

    @property
    def Q_ALPHA_RET(self) -> float:
        """Physical q at gain-schedule alpha=0, inside the retract stop."""
        return self.Q_RET - self.calib_backoff_rad

    @property
    def Q_NOM(self) -> float:
        """Nominal stance (midpoint of stroke)."""
        return self.Q_RET + 0.50 * (self.Q_EXT - self.Q_RET)

    @property
    def STROKE_DEG(self) -> float:
        return math.degrees(abs(self.Q_EXT - self.Q_RET))

    def as_dict(self) -> dict:
        """Legacy dict for physics.py IK functions."""
        return dict(
            L_femur=self.L_femur, L_stub=self.L_stub, L_tibia=self.L_tibia,
            w_perp=self.w_perp, Lc=self.Lc,
            F_X=self.F_X, F_Z=self.F_Z, A_Z=self.A_Z,
            m_box=self.m_box, m_femur=self.m_femur, m_tibia=self.m_tibia,
            m_coupler=self.m_coupler, m_bearing=self.m_bearing, m_wheel=self.m_wheel,
        )


# ── Top-level SimParams ───────────────────────────────────────────────────

@dataclass(frozen=True)
class SimParams:
    """Top-level container — single source of truth for an entire simulation run.

    Immutable: optimizer creates modified copies via dataclasses.replace().
    """
    robot: RobotGeometry = field(default_factory=RobotGeometry)
    motors: MotorParams = field(default_factory=MotorParams)
    battery: BatteryParams = field(default_factory=BatteryParams)
    gains: GainSet = field(default_factory=GainSet)
    latency: LatencyParams = field(default_factory=LatencyParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    timing: SimTiming = field(default_factory=SimTiming)
    limits: HardwareLimits = field(default_factory=HardwareLimits)
    thresholds: MetricThresholds = field(default_factory=MetricThresholds)
    scenarios: ScenarioTimings = field(default_factory=ScenarioTimings)
    s5_bumps: Tuple[S5Bump, ...] = field(default_factory=lambda: (
        S5Bump(0.8, 0.01), S5Bump(2.0, 0.03), S5Bump(3.5, 0.01),
        S5Bump(-0.8, 0.03), S5Bump(-2.0, 0.01),
    ))
    s8_bumps: Tuple[S8Bump, ...] = field(default_factory=lambda: (
        S8Bump(shape='box', x=1.2, y=0.1430, h=0.025, rx=0.02, ry=0.15),  # TEMP: h/2
    ))
