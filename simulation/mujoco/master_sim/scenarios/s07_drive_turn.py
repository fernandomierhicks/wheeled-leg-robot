"""s07_drive_turn — Cross-coupling check: simultaneous drive + turn.

YawPI + VelocityPI + LQR.
v_desired = 1.0 m/s constant, omega_desired = +60 deg/s first half → −60 deg/s second half.

Fitness = 0.5*3.0*vel_rms + 0.5*3.0*yaw_rms + 0.1*rms_pitch + 200*fell
"""
from master_sim.defaults import DEFAULT_PARAMS
from master_sim.scenarios.base import ScenarioConfig
from master_sim.scenarios.profiles import constant_velocity

_timings = DEFAULT_PARAMS.scenarios


def _yaw_inversion(rate: float, t_flip: float):
    """Return closure: +rate until t_flip, then −rate."""
    def _fn(t: float) -> float:
        return rate if t < t_flip else -rate
    return _fn


def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    return (0.5 * CONFIG.W_VEL_ERR * m['vel_track_rms_ms']
            + 0.5 * CONFIG.W_YAW_ERR * m['yaw_track_rms_rads']
            + 0.1 * CONFIG.W_RMS * m['rms_pitch_deg']
            + (CONFIG.W_FALL if fell else 0.0))


# ── Scenario config ──────────────────────────────────────────────────────────

CONFIG = ScenarioConfig(
    name="s07_drive_turn",
    display_name="S7 — Drive + Turn",
    duration=_timings.s7_duration,
    active_controllers=frozenset({"lqr", "velocity_pi", "yaw_pi"}),
    hip_mode="position",
    v_profile=constant_velocity(_timings.drive_turn_speed),
    omega_profile=_yaw_inversion(_timings.drive_turn_yaw_rate,
                                 _timings.s7_duration / 2.0),
    fitness_fn=fitness,
    group="yaw_pi",
    order=7.0,
)
