"""Firmware-equivalent jump phase machine for the v4 MuJoCo twin.

The controller mirrors ``teensy/src/state_machine.cpp``:

    BALANCE -> CROUCH -> EXTEND -> RETRACT -> LANDING -> HANDOFF -> BALANCE

CROUCH and RETRACT use minimum-jerk trajectories whose durations are derived
from travel and configured peak speed.  EXTEND uses rate-ramped, position- and
speed-tapered torque.  Landing uses the same acceleration latch plus rolling
multi-sample gyro impulse as the firmware.  HANDOFF leaves the normal balance
controller active with its jump-specific authority and capture settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math
from typing import Mapping, Optional

from v4_twin_279mm_baseline.params import RobotGeometry


MIN_JERK_PEAK_OVER_MEAN = 1.875
PHASE_MIN_S = 0.020
GYRO_WINDOW_S = 0.012
GYRO_MIN_EVENTS = 2
GYRO_EVENT_EPS = 0.01
OVERRUN_MARGIN_S = 0.5
RETRACT_BRAKE_NOMINAL_S = 0.015


class RobotMode(Enum):
    BALANCE = auto()
    CROUCH = auto()
    EXTEND = auto()
    RETRACT = auto()
    LANDING = auto()
    HANDOFF = auto()
    FAULT = auto()


@dataclass(frozen=True)
class ModeOutput:
    mode: RobotMode
    hip_mode: str
    hip_torque_override: Optional[float]
    q_hip_target: float
    dq_hip_target: float
    susp_scale: float = 1.0
    velocity_offset_ms: float = 0.0
    handoff_active: bool = False
    landing_source: str = ""
    airborne_seen: bool = False
    failed: bool = False
    hip_torque_limit_nm: Optional[float] = None


def _minimum_jerk(u: float) -> tuple[float, float]:
    """Return unit position and unit-time derivative for a quintic profile."""
    u = min(1.0, max(0.0, float(u)))
    position = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
    derivative = 30.0 * u * u * (1.0 - u) * (1.0 - u)
    return position, derivative


def _phase_duration(travel_rad: float, peak_speed_rad_s: float) -> float:
    if peak_speed_rad_s <= 0.0:
        return PHASE_MIN_S
    return max(PHASE_MIN_S,
               MIN_JERK_PEAK_OVER_MEAN * abs(travel_rad) / peak_speed_rad_s)


def _retract_brake_sample(start_q: float, start_dq: float,
                          elapsed_s: float, duration_s: float) -> tuple[float, float]:
    """Velocity-continuous EXTEND-to-RETRACT braking blend."""
    if duration_s <= 0.0:
        return float(start_q), 0.0
    u = min(1.0, max(0.0, float(elapsed_s) / float(duration_s)))
    velocity_scale = 1.0 - 3.0 * u * u + 2.0 * u * u * u
    position_integral = u - u ** 3 + 0.5 * u ** 4
    return (float(start_q) + float(start_dq) * duration_s * position_integral,
            float(start_dq) * velocity_scale)


def _retract_brake_duration(start_q: float, start_dq: float,
                            extended_limit: float, hardstop_margin: float) -> float:
    if start_dq >= 0.0 or abs(start_dq) < 1e-9:
        return RETRACT_BRAKE_NOMINAL_S
    safe_extended = float(extended_limit) + float(hardstop_margin)
    remaining = float(start_q) - safe_extended
    if remaining <= 0.0:
        return 0.0
    return min(RETRACT_BRAKE_NOMINAL_S, 2.0 * remaining / abs(float(start_dq)))


class JumpController:
    """Stateful port of the production jump phase logic."""

    def __init__(self, robot: RobotGeometry, params: Mapping[str, float], dt: float):
        self.robot = robot
        self.params = params
        self.dt = float(dt)
        self._mode = RobotMode.BALANCE
        self._triggered = False
        self._phase_start = 0.0
        self._jump_start = 0.0
        self._deadline = math.inf
        self._nominal_q = robot.Q_NOM
        self._crouch_q = robot.Q_NOM
        self._crouch_duration = PHASE_MIN_S
        self._retract_from_q = robot.Q_EXT
        self._retract_from_dq = 0.0
        self._retract_turn_q = robot.Q_EXT
        self._retract_brake_duration = 0.0
        self._retract_q = robot.Q_NOM
        self._retract_duration = PHASE_MIN_S
        self._airborne_seen = False
        self._landing_source = ""
        self._previous_rates: tuple[float, float, float] | None = None
        self._gyro_events: list[tuple[float, float]] = []
        self._capture_since: float | None = None
        self._fault_reason = ""
        self._phase_entries: list[tuple[str, float]] = []

    @property
    def mode(self) -> RobotMode:
        return self._mode

    @property
    def complete(self) -> bool:
        return self._mode == RobotMode.BALANCE and not self._triggered

    @property
    def fault_reason(self) -> str:
        return self._fault_reason

    @property
    def phase_entries(self) -> tuple[tuple[str, float], ...]:
        return tuple(self._phase_entries)

    def trigger(self) -> None:
        if not self._triggered:
            self._triggered = True

    def _p(self, name: str) -> float:
        return float(self.params[name])

    def _set_mode(self, mode: RobotMode, now: float) -> None:
        self._mode = mode
        self._phase_start = float(now)
        self._phase_entries.append((mode.name, float(now - self._jump_start)))

    def _extension_target(self, extension_angle: float) -> float:
        # Firmware encoder zero is the retract switch.  MuJoCo uses the CAD
        # frame whose retract switch is robot.Q_RET; both have the same sign.
        return self.robot.Q_RET - float(extension_angle)

    def _begin(self, now: float, hip_q: float) -> None:
        self._jump_start = float(now)
        self._nominal_q = float(hip_q)
        self._crouch_q = self._extension_target(self._p("jump_crouch_angle"))
        self._crouch_duration = _phase_duration(
            self._crouch_q - self._nominal_q, self._p("jump_crouch_speed"))
        budget = (self._crouch_duration + self._p("jump_ext_timeout")
                  + self._p("jump_land_timeout")
                  + self._p("jmp_handoff_timeout") + OVERRUN_MARGIN_S)
        self._deadline = now + budget
        self._fault_reason = ""
        self._landing_source = ""
        self._capture_since = None
        self._phase_entries = []
        self._set_mode(RobotMode.CROUCH, now)

    def _reset_landing(self, rates: tuple[float, float, float]) -> None:
        self._airborne_seen = False
        self._landing_source = ""
        self._previous_rates = tuple(float(value) for value in rates)
        self._gyro_events = []

    def _landing_update(self, now: float, accel_xyz: tuple[float, float, float],
                        rates: tuple[float, float, float]) -> bool:
        accel_contact = False
        accel_z = float(accel_xyz[2])
        if accel_z <= self._p("jump_air_accel_z"):
            self._airborne_seen = True
        if self._airborne_seen and accel_z >= self._p("jump_land_accel_z"):
            accel_contact = True

        if self._previous_rates is not None:
            delta = math.sqrt(sum(
                (float(value) - previous) ** 2
                for value, previous in zip(rates, self._previous_rates)))
            if delta > GYRO_EVENT_EPS:
                self._gyro_events.append((float(now), delta))
        self._previous_rates = tuple(float(value) for value in rates)
        cutoff = now - GYRO_WINDOW_S
        self._gyro_events = [event for event in self._gyro_events
                             if event[0] >= cutoff]
        gyro_contact = (len(self._gyro_events) >= GYRO_MIN_EVENTS
                        and sum(value for _time, value in self._gyro_events)
                        >= self._p("jump_land_gyro_imp"))
        eligible = now - self._phase_start >= self._p("jump_land_min_air")
        if not eligible or not (accel_contact or gyro_contact):
            return False
        if accel_contact and gyro_contact:
            self._landing_source = "accel + gyro"
        elif accel_contact:
            self._landing_source = "accel rebound"
        else:
            self._landing_source = "gyro impulse"
        return True

    def update(self, t: float, hip_q_avg: float, hip_dq_avg: float,
               accel_xyz: tuple[float, float, float],
               gyro_xyz: tuple[float, float, float]) -> ModeOutput:
        """Advance launch/contact phases before the balance-controller tick."""
        now = float(t)
        if self._mode == RobotMode.BALANCE and self._triggered:
            self._begin(now, hip_q_avg)

        if self._mode not in (RobotMode.BALANCE, RobotMode.FAULT) and now > self._deadline:
            self._fault_reason = "jump phase overrun"
            self._set_mode(RobotMode.FAULT, now)

        elapsed = now - self._phase_start
        p = self.params

        if self._mode == RobotMode.CROUCH:
            if elapsed >= self._crouch_duration:
                self._set_mode(RobotMode.EXTEND, now)
                elapsed = 0.0
            else:
                s, ds_du = _minimum_jerk(elapsed / self._crouch_duration)
                travel = self._crouch_q - self._nominal_q
                remaining = self._crouch_duration - elapsed
                nudge = (self._p("jump_nudge_fwd_vel")
                         if self._p("jump_nudge_fwd_dur") > 0.0
                         and remaining <= self._p("jump_nudge_fwd_dur") else 0.0)
                return ModeOutput(
                    RobotMode.CROUCH, "position", None,
                    self._nominal_q + s * travel,
                    travel * ds_du / self._crouch_duration,
                    velocity_offset_ms=nudge)

        if self._mode == RobotMode.EXTEND:
            target = self._extension_target(self._p("jump_extend_angle"))
            to_go = hip_q_avg - target
            distance_to_limit = hip_q_avg - self.robot.Q_EXT
            cutoff = distance_to_limit < self._p("jump_hs_margin")
            near_target = to_go <= 0.0
            timed_out = elapsed >= self._p("jump_ext_timeout")
            if near_target or cutoff or timed_out:
                self._retract_from_q = float(hip_q_avg)
                self._retract_from_dq = float(hip_dq_avg)
                angle = self._p("jump_retract_angle")
                self._retract_q = (self._nominal_q if angle < 0.0
                                   else self._extension_target(angle))
                self._retract_brake_duration = _retract_brake_duration(
                    self._retract_from_q, self._retract_from_dq,
                    self.robot.Q_EXT, self._p("jump_hs_margin"))
                self._retract_turn_q, _stopped = _retract_brake_sample(
                    self._retract_from_q, self._retract_from_dq,
                    self._retract_brake_duration, self._retract_brake_duration)
                self._retract_duration = _phase_duration(
                    self._retract_q - self._retract_turn_q,
                    self._p("jump_retract_speed"))
                self._set_mode(RobotMode.RETRACT, now)
                self._reset_landing(gyro_xyz)
                elapsed = 0.0
            else:
                maximum = self._p("jump_torque_max") * self._p("jump_effort")
                ramp_in = (1.0 if maximum <= 0.0 else
                           min(1.0, elapsed * self._p("jump_torque_rate") / maximum))
                ramp_out = min(1.0, max(0.0, to_go / self._p("jump_ramp_down")))
                speed_taper = max(0.0, 1.0 - abs(hip_dq_avg) / self._p("jump_omega_max"))
                torque = -maximum * ramp_in * ramp_out * speed_taper
                return ModeOutput(
                    RobotMode.EXTEND, "torque_override", torque,
                    target, 0.0)

        if self._mode == RobotMode.RETRACT:
            if self._landing_update(now, accel_xyz, gyro_xyz):
                self._set_mode(RobotMode.LANDING, now)
                elapsed = 0.0
            elif elapsed >= self._p("jump_land_timeout"):
                self._fault_reason = "landing not detected"
                self._set_mode(RobotMode.FAULT, now)
            else:
                if elapsed < self._retract_brake_duration:
                    q_cmd, dq_cmd = _retract_brake_sample(
                        self._retract_from_q, self._retract_from_dq,
                        elapsed, self._retract_brake_duration)
                else:
                    return_elapsed = elapsed - self._retract_brake_duration
                    s, ds_du = _minimum_jerk(return_elapsed / self._retract_duration)
                    travel = self._retract_q - self._retract_turn_q
                    q_cmd = self._retract_turn_q + s * travel
                    dq_cmd = travel * ds_du / self._retract_duration
                return ModeOutput(
                    RobotMode.RETRACT, "position", None,
                    q_cmd, dq_cmd,
                    landing_source=self._landing_source,
                    airborne_seen=self._airborne_seen,
                    hip_torque_limit_nm=self._p("jump_retract_torque"))

        if self._mode == RobotMode.LANDING:
            # Firmware exposes LANDING for one 500 Hz telemetry tick, while the
            # ordinary hip/balance controller already owns the outputs.
            if elapsed >= self.dt:
                self._set_mode(RobotMode.HANDOFF, now)

        if self._mode in (RobotMode.LANDING, RobotMode.HANDOFF):
            return ModeOutput(
                self._mode, "running", None, self._nominal_q, 0.0,
                handoff_active=True, landing_source=self._landing_source,
                airborne_seen=self._airborne_seen)
        if self._mode == RobotMode.FAULT:
            return ModeOutput(
                RobotMode.FAULT, "running", None, self._nominal_q, 0.0,
                landing_source=self._landing_source,
                airborne_seen=self._airborne_seen, failed=True)
        return ModeOutput(RobotMode.BALANCE, "running", None,
                          self._nominal_q, 0.0)

    def post_control(self, t: float, *, pitch: float, pitch_rate: float,
                     wheel_l_turns_s: float, wheel_r_turns_s: float,
                     theta_ref: float, pitch_trim: float) -> None:
        """Apply firmware HANDOFF capture after the balance tick."""
        if self._mode != RobotMode.HANDOFF:
            return
        now = float(t)
        in_band = (
            abs(pitch - theta_ref - pitch_trim) < self._p("jmp_handoff_pitch")
            and abs(pitch_rate) < self._p("jmp_handoff_rate")
            and abs(wheel_l_turns_s) <= self._p("wm_vel_limit")
            and abs(wheel_r_turns_s) <= self._p("wm_vel_limit")
        )
        if in_band:
            if self._capture_since is None:
                self._capture_since = now
            if now - self._capture_since >= self._p("jmp_handoff_hold_s"):
                self._mode = RobotMode.BALANCE
                self._triggered = False
                self._phase_entries.append(("RUNNING", now - self._jump_start))
        else:
            self._capture_since = None
        if (self._mode == RobotMode.HANDOFF
                and now - self._phase_start >= self._p("jmp_handoff_timeout")):
            self._fault_reason = "handoff not captured"
            self._set_mode(RobotMode.FAULT, now)
