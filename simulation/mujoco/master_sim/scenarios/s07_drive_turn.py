"""s07_drive_turn — Cross-coupling check: simultaneous drive + turn.

YawPI + VelocityPI + LQR.
v_desired = 0.3 m/s constant, omega_desired = 0.5 rad/s constant (CCW).

Fitness = 0.5*3.0*vel_rms + 0.5*3.0*yaw_rms + 0.1*rms_pitch + 200*fell
"""
from master_sim.defaults import DEFAULT_PARAMS
from master_sim.scenarios.base import ScenarioConfig
from master_sim.scenarios.profiles import constant_velocity, make_yaw_step_fn

_timings = DEFAULT_PARAMS.scenarios

W_VEL_ERR = 3.0
W_YAW_ERR = 3.0
W_RMS     = 1.0
W_FALL    = 200.0


def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    return (0.5 * W_VEL_ERR * m['vel_track_rms_ms']
            + 0.5 * W_YAW_ERR * m['yaw_track_rms_rads']
            + 0.1 * W_RMS * m['rms_pitch_deg']
            + (W_FALL if fell else 0.0))


# ── Scenario config ──────────────────────────────────────────────────────────

CONFIG = ScenarioConfig(
    name="s07_drive_turn",
    display_name="S7 — Drive + Turn",
    duration=8.0,                                # SCENARIO_7_DURATION
    active_controllers=frozenset({"lqr", "velocity_pi", "yaw_pi"}),
    hip_mode="position",
    v_profile=constant_velocity(_timings.drive_turn_speed),
    omega_profile=constant_velocity(_timings.drive_turn_yaw_rate),
    fitness_fn=fitness,
    group="yaw_pi",
    order=7.0,
)
