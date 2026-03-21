"""profiles.py — Velocity, yaw, hip, and disturbance profile functions.

All functions are pure (no module-level state) and take explicit parameters
from SimParams where needed.  Profile functions are stored as callables in
ScenarioConfig and called each tick by sim_loop.
"""
import math

from master_sim.params import RobotGeometry, ScenarioTimings


# ── Disturbance functions ────────────────────────────────────────────────────

def s1_dist_fn(t: float) -> float:
    """+4 N at t=2 s for 0.2 s, then −4 N at t=3 s for 0.2 s."""
    if 2.0 <= t < 2.2:
        return 4.0
    if 3.0 <= t < 3.2:
        return -4.0
    return 0.0


def s2_dist_fn(t: float, timings: ScenarioTimings = None) -> float:
    """+1 N at t=2 s for 0.2 s, then −1 N at t=3 s for 0.2 s."""
    if timings is None:
        t1, f1, d1 = 2.0, 1.0, 0.2
        t2, f2, d2 = 3.0, -1.0, 0.2
    else:
        t1, f1, d1 = timings.s2_dist1_time, timings.s2_dist1_force, timings.s2_dist1_dur
        t2, f2, d2 = timings.s2_dist2_time, timings.s2_dist2_force, timings.s2_dist2_dur
    if t1 <= t < t1 + d1:
        return f1
    if t2 <= t < t2 + d2:
        return f2
    return 0.0


# ── Velocity profile functions ───────────────────────────────────────────────

def s3_velocity_profile(t: float) -> float:
    """S3/S4 staircase: 0 → +0.3 → +0.6 → +1.0 → −0.5 → −1.0 → 0 m/s."""
    if   t <  1.0: return  0.0
    elif t <  3.0: return  0.3
    elif t <  5.0: return  0.6
    elif t <  7.0: return  1.0
    elif t <  9.0: return -0.5
    elif t < 11.0: return -1.0
    else:          return  0.0


def constant_velocity(speed: float):
    """Return a closure: constant velocity profile."""
    def _profile(t: float) -> float:
        return speed
    return _profile


def zero_velocity(t: float) -> float:
    return 0.0


# ── Hip (leg height) profile functions ───────────────────────────────────────

def leg_cycle_profile(t: float, robot: RobotGeometry,
                      timings: ScenarioTimings) -> float:
    """Sinusoidal leg-height cycle between leg_cycle_Q_RET and Q_EXT.

    Period = leg_cycle_period.  Starts at mid-stroke, first moves toward Q_EXT.
    """
    q_ret = timings.leg_cycle_Q_RET
    q_ext = robot.Q_EXT
    period = timings.leg_cycle_period
    center = (q_ret + q_ext) / 2.0
    amp = (q_ext - q_ret) / 2.0
    return center + amp * math.sin(2.0 * math.pi * t / period)


def leg_cycle_velocity(t: float, robot: RobotGeometry,
                       timings: ScenarioTimings) -> float:
    """Derivative of leg_cycle_profile — target hip velocity for feed-forward."""
    q_ret = timings.leg_cycle_Q_RET
    q_ext = robot.Q_EXT
    period = timings.leg_cycle_period
    amp = (q_ext - q_ret) / 2.0
    return amp * (2.0 * math.pi / period) * math.cos(2.0 * math.pi * t / period)


def make_leg_cycle_fn(robot: RobotGeometry, timings: ScenarioTimings):
    """Return a closure (t -> q_hip) for use as ScenarioConfig.hip_profile."""
    def _fn(t: float) -> float:
        return leg_cycle_profile(t, robot, timings)
    return _fn


def make_leg_cycle_vel_fn(robot: RobotGeometry, timings: ScenarioTimings):
    """Return a closure (t -> dq_hip) for hip velocity feed-forward."""
    def _fn(t: float) -> float:
        return leg_cycle_velocity(t, robot, timings)
    return _fn


# ── Yaw profile functions ───────────────────────────────────────────────────

def make_yaw_step_fn(rate: float, start_time: float = 1.0):
    """Return a closure: 0 until start_time, then constant yaw rate."""
    def _fn(t: float) -> float:
        return rate if t >= start_time else 0.0
    return _fn
