"""profiles.py — Velocity, yaw, hip, and disturbance profile functions.

All functions are pure (no module-level state) and take explicit parameters
from SimParams where needed.  Profile functions are stored as callables in
ScenarioConfig and called each tick by sim_loop.
"""
import math

from master_sim.params import RobotGeometry, ScenarioTimings


# ── Disturbance functions ────────────────────────────────────────────────────

def s1_velocity_profile(t: float) -> float:
    """S1 velocity: zero throughout (debug: removed step commands)."""
    return 0.0


def s1_theta_ref_profile(t: float) -> float:
    """S1 pitch command: 0 until t=2.5s, then −5° step."""
    if t >= 2.5:
        return math.radians(-5.0)
    return 0.0


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

def s3_vel_step_profile(t: float) -> float:
    """S3 simple step: 0 until t=6s, then +0.5 m/s."""
    return 0.5 if t >= 6.0 else 0.0


def s3_velocity_profile(t: float) -> float:
    """S3/S4 staircase: 0 → +0.3 → −0.5 → +0.7 → −1.0 → +1.2 → 0 m/s."""
    if   t <  1.0: return  0.0
    elif t <  3.0: return  0.3
    elif t <  5.0: return -0.5
    elif t <  7.0: return  0.7
    elif t <  9.0: return -1.0
    elif t < 11.0: return  1.2
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
    """Sinusoidal leg-height cycle between Q_RET and Q_EXT.

    Period = leg_cycle_period.  Starts at mid-stroke, first moves toward Q_EXT.
    """
    q_ret = robot.Q_RET
    q_ext = robot.Q_EXT
    period = timings.leg_cycle_period
    center = (q_ret + q_ext) / 2.0
    amp = (q_ext - q_ret) / 2.0
    return center + amp * math.sin(2.0 * math.pi * t / period)


def leg_cycle_velocity(t: float, robot: RobotGeometry,
                       timings: ScenarioTimings) -> float:
    """Derivative of leg_cycle_profile — target hip velocity for feed-forward."""
    q_ret = robot.Q_RET
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

def s6_velocity_profile(t: float) -> float:
    """S6 forward/backward: 0 → +0.3 → −0.3 → 0 m/s during yaw turn."""
    if   t < 1.0: return  0.0
    elif t < 3.5: return  0.3
    elif t < 6.0: return -0.3
    else:         return  0.0


def make_yaw_step_fn(rate: float, start_time: float = 1.0):
    """Return a closure: 0 until start_time, then constant yaw rate."""
    def _fn(t: float) -> float:
        return rate if t >= start_time else 0.0
    return _fn


# ── S9 Integrated scenario profiles ──────────────────────────────────────

def s9_velocity_profile(t: float) -> float:
    """S9 velocity: 0 (0-5s), 0.5 (5-7s), 1.0 (7-8.5s), -0.5 (8.5-10s), 0.5 (10-16s)."""
    if   t < 5.0:  return 0.0
    elif t < 7.0:  return 0.5
    elif t < 8.5:  return 1.0
    elif t < 10.0: return -0.5
    else:          return 0.5


def s9_yaw_profile(t: float) -> float:
    """S9 yaw: 0 (0-10s), +60°/s (10-12s), -60°/s (12-14s), 0 (14-16s)."""
    if   t < 10.0: return 0.0
    elif t < 12.0: return 1.047
    elif t < 14.0: return -1.047
    else:          return 0.0


def s9_dist_fn(t: float) -> float:
    """S9 +Z kick on left wheel hub mid-drive at Q_NOM crossing: +10N at 7.7s (0.2s)."""
    if 7.7 <= t < 7.9:
        return 7.0
    return 0.0


def s9_roll_dist_fn(t: float) -> float:
    """S9 Z-force pulses on left wheel hub: ~10N at t=6,8,11,13s (0.2s each)."""
    for t0 in (6.0, 8.0, 11.0, 13.0):
        if t0 <= t < t0 + 0.2:
            return 7.0
    return 0.0


def s9_hip_sweep(robot: RobotGeometry):
    """Return a closure: continuous sine Q_RET↔Q_EXT, period 4s, entire test.

    q(t) = mid + amp·sin(2π·t/4)
    where mid = (Q_RET+Q_EXT)/2, amp = (Q_EXT-Q_RET)/2.
    Crosses Q_NOM at t ≈ 2.26, 3.74, 6.26, 7.74, 10.26, 11.74, 14.26, 15.74 s.
    """
    mid = (robot.Q_RET + robot.Q_EXT) / 2.0
    amp = (robot.Q_EXT - robot.Q_RET) / 2.0
    def _fn(t: float) -> float:
        return mid + amp * math.sin(2.0 * math.pi * t / 4.0)
    return _fn
