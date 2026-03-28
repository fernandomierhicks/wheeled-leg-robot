"""jump.py — Jump state machine controller (Phase 1: constant gains).

Five-state machine: BALANCE → CROUCH → EXTEND → FLYING → LANDING → SETTLED → BALANCE.
Only the EXTEND phase overrides hip torque; other phases use impedance or position servo.
All balance controllers (LQR, VelocityPI, YawPI) run unchanged throughout.

Hip mode per phase:
  BALANCE / LANDING / SETTLED → "impedance"  (soft suspension + roll leveling)
  CROUCH                      → "position"   (stiff PD servo to compress knee spring)
  EXTEND                      → "torque_override" (explosive extension)
  FLYING                      → "impedance"  (retract to Q_NOM — tuck + cushion landing)
"""
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from master_sim_jump.params import RobotGeometry, JumpGains


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


class JumpController:
    """Jump state machine — controls hip trajectory and torque override.

    Usage:
        jc = JumpController(robot, gains, dt)
        jc.trigger()                   # start CROUCH
        out = jc.update(t, hip_q_avg, hip_dq_avg,
                        wheel_z_L, wheel_z_R, wheel_r, pitch)
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

        # FLYING liftoff debounce
        self._liftoff_timer: float = 0.0

        # LANDING → SETTLED pitch settle timer
        self._settle_timer: float = 0.0
        self._landing_start: float = 0.0

        # Hold angle for FLYING/LANDING
        self._hold_angle: float = robot.Q_NOM

    @property
    def mode(self) -> RobotMode:
        return self._mode

    def trigger(self):
        """Start jump sequence (idempotent after first call)."""
        if not self._triggered:
            self._triggered = True

    def update(self, t: float, hip_q_avg: float, hip_dq_avg: float,
               wheel_z_L: float, wheel_z_R: float, wheel_r: float,
               pitch: float) -> ModeOutput:
        """Advance state machine and return hip control output."""
        robot = self.robot
        gains = self.gains
        liftoff_threshold = wheel_r + 0.005

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
            # Check liftoff: both wheels above ground
            if (wheel_z_L > liftoff_threshold and
                    wheel_z_R > liftoff_threshold):
                self._liftoff_timer += self.dt
                if self._liftoff_timer >= gains.liftoff_debounce_s:
                    self._mode = RobotMode.FLYING
                    self._hold_angle = hip_q_avg
            else:
                self._liftoff_timer = 0.0
            # Timeout: abort jump if no liftoff detected
            if t - self._extend_start >= gains.extend_timeout_s:
                self._mode = RobotMode.SETTLED

        elif self._mode == RobotMode.FLYING:
            # Either wheel touches down
            if (wheel_z_L < liftoff_threshold or
                    wheel_z_R < liftoff_threshold):
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
            )

        elif self._mode == RobotMode.LANDING:
            return ModeOutput(
                mode=RobotMode.LANDING,
                hip_mode="impedance",
                hip_torque_override=None,
                q_hip_target=self._hold_angle,
                dq_hip_target=0.0,
            )

        else:
            # BALANCE or SETTLED — impedance (soft suspension + roll leveling)
            return ModeOutput(
                mode=self._mode,
                hip_mode="impedance",
                hip_torque_override=None,
                q_hip_target=robot.Q_NOM,
                dq_hip_target=0.0,
            )
