"""s04_vel_pi_staircase — VelocityPI setpoint tracking across six velocity steps.

VelocityPI outer + LQR inner.
Profile: 0 → +0.3 → +0.6 → +1.0 → −0.5 → −1.0 → 0 m/s  (13 s total).

Fitness = 3.0 * vel_track_rms_ms + 0.1 * rms_pitch_deg + 200 * fell
"""
from master_sim.scenarios.base import ScenarioConfig
from master_sim.scenarios.profiles import s3_velocity_profile

W_VEL_ERR = 3.0
W_RMS     = 1.0
W_FALL    = 200.0


def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    return (W_VEL_ERR * m['vel_track_rms_ms']
            + 0.1 * W_RMS * m['rms_pitch_deg']
            + (W_FALL if fell else 0.0))


# ── Scenario config ──────────────────────────────────────────────────────────

CONFIG = ScenarioConfig(
    name="s04_vel_pi_staircase",
    display_name="S4 — VelPI Staircase",
    duration=13.0,                               # SCENARIO_3_DURATION
    active_controllers=frozenset({"lqr", "velocity_pi"}),
    hip_mode="position",
    v_profile=s3_velocity_profile,
    fitness_fn=fitness,
    group="velocity_pi",
    order=4.0,
)
