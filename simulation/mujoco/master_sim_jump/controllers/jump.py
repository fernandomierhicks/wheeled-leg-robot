"""jump.py — Jump state machine controller (Phase 1: constant gains).

Five-state machine: BALANCE → CROUCH → EXTEND → FLYING → LANDING → SETTLED → BALANCE.
Only the EXTEND phase overrides hip torque; other phases use impedance or position servo.
All balance controllers (LQR, VelocityPI, YawPI) run unchanged throughout.

Hip mode per phase:
  BALANCE / LANDING / SETTLED → "impedance"  (soft suspension + roll leveling)
  CROUCH                      → "position"   (stiff PD servo to compress knee spring)
  EXTEND                      → "torque_override" (explosive extension)
  FLYING                      → "impedance"  (retract to Q_NOM — tuck + cushion landing)

Liftoff and landing are detected from the IMU vertical specific force (az):
  - az ≈ 0   → free-fall (airborne)
  - az spike  → landing impact
No ground-truth wheel heights are used, matching real-robot capability.

Suspension softening is owned by this state machine:
  susp_scale = freefall_scale during FLYING, ramping back to 1.0 during LANDING.
"""
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from master_sim_jump.params import RobotGeometry, JumpGains, SuspensionGains


class RobotMode(Enum):
    BALANCE = auto()
    CROUCH = auto()
    EXTEND = auto()
    FLYING = auto()
    LANDING = auto()
    SETTLED = auto()


@dataclass
class ModeOutput:
    mode: RobotMode
    hip_mode: str                          # "impedance" | "position" | "torque_override"
    hip_torque_override: Optional[float]   # None except during EXTEND
    q_hip_target: float                    # Hip position / impedance equilibrium target
    dq_hip_target: float                   # Hip velocity feed-forward (position mode only)
    susp_scale: float = 1.0               # K_s / B_s multiplier for adaptive suspension


class JumpController:
    """Jump state machine — controls hip trajectory and torque override.

    Liftoff is detected when az_imu drops below suspension.freefall_az_threshold
    (held for liftoff_debounce_s).  Landing is detected when az_imu spikes above
    suspension.landing_az_threshold.  No ground-truth wheel heights are used.

    Suspension softening is managed here: susp_scale is returned in ModeOutput so
    sim_loop does not need a separate adaptive-suspension block.

    Usage:
        jc = JumpController(robot, gains, dt)
        jc.trigger()                   # start CROUCH
        out = jc.update(t, hip_q_avg, hip_dq_avg, az, suspension, pitch)
    """

    def __init__(self, robot: RobotGeometry, gains: JumpGains, dt: float):
        self.robot = robot
        self.gains = gains
        self.dt = dt

        self._mode = RobotMode.BALANCE
        self._triggered = False

        # CROUCH state
        self._crouch_start: float = 0.0
        self._q_crouch_start: float = robot.Q_NOM

        # EXTEND state
        self._extend_start: float = 0.0

        # EXTEND → FLYING liftoff debounce (IMU-based)
        self._liftoff_timer: float = 0.0

        # FLYING start time — guards against false landing detection immediately after liftoff
        self._flying_start: float = 0.0

        # LANDING → SETTLED pitch settle timer
        self._settle_timer: float = 0.0
        self._landing_start: float = 0.0

        # Hold angle for FLYING/LANDING
        self._hold_angle: float = robot.Q_NOM

        # Adaptive suspension state (merged from sim_loop)
        self._susp_scale: float = 1.0
        self._ramp_elapsed: float = -1.0    # -1 = not ramping; ≥0 = seconds into ramp
        self._ramp_start_scale: float = 1.0

    @property
    def mode(self) -> RobotMode:
        return self._mode

    def trigger(self):
        """Start jump sequence (idempotent after first call)."""
        if not self._triggered:
            self._triggered = True

    def update(self, t: float, hip_q_avg: float, hip_dq_avg: float,
               az: float, suspension: SuspensionGains,
               pitch: float) -> ModeOutput:
        """Advance state machine and return hip control output.

        Parameters
        ----------
        az : float
            Delayed, noisy IMU vertical specific force [m/s²].
            ≈ 0 in free-fall, ≈ 9.81 on ground, spikes on landing impact.
        suspension : SuspensionGains
            Provides freefall_az_threshold, landing_az_threshold, freefall_scale,
            landing_ramp_s for both phase detection and gain softening.
        """
        robot = self.robot
        gains = self.gains

        # ── State transitions ────────────────────────────────────────────
        if self._mode == RobotMode.BALANCE:
            if self._triggered:
                self._mode = RobotMode.CROUCH
                self._crouch_start = t
                self._q_crouch_start = hip_q_avg

        elif self._mode == RobotMode.CROUCH:
            if t - self._crouch_start >= gains.crouch_time:
                self._mode = RobotMode.EXTEND
                self._extend_start = t
                self._liftoff_timer = 0.0

        elif self._mode == RobotMode.EXTEND:
            # Liftoff: az drops toward 0 (free-fall), debounced
            if az < suspension.freefall_az_threshold:
                self._liftoff_timer += self.dt
                if self._liftoff_timer >= gains.liftoff_debounce_s:
                    self._mode = RobotMode.FLYING
                    self._flying_start = t
                    self._hold_angle = hip_q_avg
            else:
                self._liftoff_timer = 0.0
            # Timeout: abort jump if no liftoff detected
            if t - self._extend_start >= gains.extend_timeout_s:
                self._mode = RobotMode.SETTLED

        elif self._mode == RobotMode.FLYING:
            # Landing: az spikes above threshold (impact), but only after min airborne time
            if (az > suspension.landing_az_threshold
                    and t - self._flying_start >= gains.min_airborne_s):
                self._mode = RobotMode.LANDING
                self._hold_angle = hip_q_avg
                self._settle_timer = 0.0
                self._landing_start = t

        elif self._mode == RobotMode.LANDING:
            if abs(math.degrees(pitch)) < gains.settle_pitch_deg:
                self._settle_timer += self.dt
                if self._settle_timer >= gains.settle_time_s:
                    self._mode = RobotMode.SETTLED
            else:
                self._settle_timer = 0.0
            # Timeout: force settle if pitch won't stabilise
            if t - self._landing_start >= gains.landing_timeout_s:
                self._mode = RobotMode.SETTLED

        elif self._mode == RobotMode.SETTLED:
            # Immediate transition back to BALANCE
            self._mode = RobotMode.BALANCE
            self._triggered = False

        # ── Adaptive suspension scale ────────────────────────────────────
        if self._mode == RobotMode.FLYING:
            self._susp_scale = suspension.freefall_scale
            self._ramp_elapsed = -1.0
        elif self._mode == RobotMode.LANDING:
            if self._ramp_elapsed < 0.0:
                # First tick in LANDING — start ramp from current (freefall) scale
                self._ramp_elapsed = 0.0
                self._ramp_start_scale = self._susp_scale
            self._ramp_elapsed += self.dt
            t_norm = min(1.0, self._ramp_elapsed / suspension.landing_ramp_s)
            self._susp_scale = (self._ramp_start_scale
                                + (1.0 - self._ramp_start_scale) * t_norm)
            if t_norm >= 1.0:
                self._susp_scale = 1.0
                self._ramp_elapsed = -1.0
        else:
            # BALANCE, CROUCH, EXTEND, SETTLED — nominal
            self._susp_scale = 1.0
            self._ramp_elapsed = -1.0

        # ── Compute outputs per mode ─────────────────────────────────────
        if self._mode == RobotMode.CROUCH:
            alpha = min(1.0, (t - self._crouch_start) / gains.crouch_time)
            q_target = self._q_crouch_start + (robot.Q_RET - self._q_crouch_start) * alpha
            dq_target = (robot.Q_RET - self._q_crouch_start) / gains.crouch_time
            return ModeOutput(
                mode=RobotMode.CROUCH,
                hip_mode="position",
                hip_torque_override=None,
                q_hip_target=q_target,
                dq_hip_target=dq_target,
                susp_scale=self._susp_scale,
            )

        elif self._mode == RobotMode.EXTEND:
            # Torque-speed limited output (ported from archive 4bar_jump_sim.py)
            ramp_in = min(1.0, (t - self._extend_start) / gains.ramp_up_s)

            ramp_out = 1.0
            if hip_q_avg < robot.Q_EXT + gains.ramp_down_rad:
                ramp_out = max(0.0, (hip_q_avg - robot.Q_EXT) / gains.ramp_down_rad)

            speed_limit = max(0.0, 1.0 - abs(hip_dq_avg) / gains.omega_max)
            torque = -gains.max_torque * ramp_in * ramp_out * speed_limit

            return ModeOutput(
                mode=RobotMode.EXTEND,
                hip_mode="torque_override",
                hip_torque_override=torque,
                q_hip_target=robot.Q_EXT,
                dq_hip_target=0.0,
                susp_scale=self._susp_scale,
            )

        elif self._mode == RobotMode.FLYING:
            # Retract legs to Q_NOM so wheels tuck up — looks like higher jump
            # Impedance mode so legs cushion the landing on touchdown
            return ModeOutput(
                mode=RobotMode.FLYING,
                hip_mode="impedance",
                hip_torque_override=None,
                q_hip_target=robot.Q_NOM,
                dq_hip_target=0.0,
                susp_scale=self._susp_scale,
            )

        elif self._mode == RobotMode.LANDING:
            return ModeOutput(
                mode=RobotMode.LANDING,
                hip_mode="impedance",
                hip_torque_override=None,
                q_hip_target=self._hold_angle,
                dq_hip_target=0.0,
                susp_scale=self._susp_scale,
            )

        else:
            # BALANCE or SETTLED — impedance (soft suspension + roll leveling)
            return ModeOutput(
                mode=self._mode,
                hip_mode="impedance",
                hip_torque_override=None,
                q_hip_target=robot.Q_NOM,
                dq_hip_target=0.0,
                susp_scale=self._susp_scale,
            )
