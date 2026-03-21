"""s06_yaw_pi_turn — Yaw test: 360° CCW turn at 1 rad/s with fwd/rev drive.

YawPI differential + VelocityPI (fwd +0.3 then rev −0.3 m/s) + LQR.
omega = 0 for first 1 s, then YAW_TURN_RATE (1 rad/s) for ~6.28 s.

Fitness = 3.0 * yaw_track_rms_rads + 0.1 * rms_pitch_deg + 200 * fell
"""
from master_sim.defaults import DEFAULT_PARAMS
from master_sim.scenarios.base import ScenarioConfig
from master_sim.scenarios.profiles import make_yaw_step_fn, s6_velocity_profile

_timings = DEFAULT_PARAMS.scenarios

def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    return (CONFIG.W_YAW_ERR * m['yaw_track_rms_rads']
            + 0.1 * CONFIG.W_RMS * m['rms_pitch_deg']
            + (CONFIG.W_FALL if fell else 0.0))


# ── Scenario config ──────────────────────────────────────────────────────────

CONFIG = ScenarioConfig(
    name="s06_yaw_pi_turn",
    display_name="S6 — Yaw PI Turn",
    duration=_timings.s6_duration,
    active_controllers=frozenset({"lqr", "velocity_pi", "yaw_pi"}),
    hip_mode="position",
    v_profile=s6_velocity_profile,
    omega_profile=make_yaw_step_fn(_timings.yaw_turn_rate, start_time=1.0),
    fitness_fn=fitness,
    group="yaw_pi",
    order=6.0,
)
