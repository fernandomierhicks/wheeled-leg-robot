"""s08_terrain_compliance — Roll leveling + suspension test with one-sided bumps.

LQR + VelocityPI + roll leveling impedance mode.
v_desired = 1.0 m/s constant.  One-sided bumps hit left wheel → roll disturbance.

Fitness = 3.0 * vel_rms + 1.0 * max_roll_deg + 0.1 * rms_pitch + 200 * fell
"""
from master_sim.defaults import DEFAULT_PARAMS
from master_sim.scenarios.base import ScenarioConfig, WorldConfig
from master_sim.scenarios.profiles import constant_velocity

_timings = DEFAULT_PARAMS.scenarios
_s8_bumps = DEFAULT_PARAMS.s8_bumps

W_VEL_ERR = 3.0
W_RMS     = 1.0
W_FALL    = 200.0


def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    return (W_VEL_ERR * m['vel_track_rms_ms']
            + W_RMS * m['max_roll_deg']
            + 0.1 * W_RMS * m['rms_pitch_deg']
            + (W_FALL if fell else 0.0))


# ── Scenario config ──────────────────────────────────────────────────────────

CONFIG = ScenarioConfig(
    name="s08_terrain_compliance",
    display_name="S8 — Terrain Compliance",
    duration=12.0,                               # SCENARIO_8_DURATION
    active_controllers=frozenset({"lqr", "velocity_pi"}),
    hip_mode="impedance",
    v_profile=constant_velocity(_timings.s8_drive_speed),
    world=WorldConfig(sandbox_obstacles=tuple(_s8_bumps)),
    fitness_fn=fitness,
    group="suspension",
    order=8.0,
)
