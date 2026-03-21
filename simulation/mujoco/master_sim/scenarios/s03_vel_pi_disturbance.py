"""s03_vel_pi_disturbance — VelocityPI position-hold under impulse kicks.

VelocityPI outer + LQR inner.  v_target = 0 throughout (position hold).
Disturbances: +1 N at t=2 s, −1 N at t=3 s (lighter than S1).

Fitness = (wheel_travel_m + 200 * fell) / duration   [normalized to m/s]
"""
from master_sim.scenarios.base import ScenarioConfig
from master_sim.scenarios.profiles import s2_dist_fn

W_FALL = 200.0


def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    duration = m['survived_s'] if fell else 12.0  # use config duration
    return (m['wheel_travel_m'] + (W_FALL if fell else 0.0)) / 12.0


# ── Scenario config ──────────────────────────────────────────────────────────

CONFIG = ScenarioConfig(
    name="s03_vel_pi_disturbance",
    display_name="S3 — VelPI Disturbance",
    duration=12.0,                               # SCENARIO_2_DURATION
    active_controllers=frozenset({"lqr", "velocity_pi"}),
    hip_mode="position",
    dist_fn=s2_dist_fn,                           # +1N/−1N lighter kicks
    fitness_fn=fitness,
    group="velocity_pi",
    order=3.0,
)
