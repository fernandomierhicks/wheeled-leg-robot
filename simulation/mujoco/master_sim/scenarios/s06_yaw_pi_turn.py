"""s06_yaw_pi_turn — Pure yaw test: 360° CCW turn at 1 rad/s.

YawPI differential + VelocityPI (v=0 position hold) + LQR.
omega = 0 for first 1 s, then YAW_TURN_RATE (1 rad/s) for ~6.28 s.

Fitness = 3.0 * yaw_track_rms_rads + 0.1 * rms_pitch_deg + 200 * fell
"""
from master_sim.defaults import DEFAULT_PARAMS
from master_sim.scenarios.base import ScenarioConfig
from master_sim.scenarios.profiles import make_yaw_step_fn

_timings = DEFAULT_PARAMS.scenarios

W_YAW_ERR = 3.0
W_RMS     = 1.0
W_FALL    = 200.0


def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    return (W_YAW_ERR * m['yaw_track_rms_rads']
            + 0.1 * W_RMS * m['rms_pitch_deg']
            + (W_FALL if fell else 0.0))


# ── Scenario config ──────────────────────────────────────────────────────────

CONFIG = ScenarioConfig(
    name="s06_yaw_pi_turn",
    display_name="S6 — Yaw PI Turn",
    duration=8.0,                                # SCENARIO_6_DURATION
    active_controllers=frozenset({"lqr", "velocity_pi", "yaw_pi"}),
    hip_mode="position",
    omega_profile=make_yaw_step_fn(_timings.yaw_turn_rate, start_time=1.0),
    fitness_fn=fitness,
    group="yaw_pi",
    order=6.0,
)
