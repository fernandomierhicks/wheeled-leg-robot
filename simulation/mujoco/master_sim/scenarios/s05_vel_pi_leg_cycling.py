"""s05_vel_pi_leg_cycling — Hardest VelocityPI test.

Simultaneous velocity staircase + leg cycling + disturbances + terrain bumps.
VelocityPI outer + LQR inner (gain-scheduled).

Fitness = 3.0 * vel_track_rms_ms + 0.1 * rms_pitch_deg + 200 * fell
"""
from master_sim.defaults import DEFAULT_PARAMS
from master_sim.scenarios.base import ScenarioConfig, WorldConfig
from master_sim.scenarios.profiles import (
    s3_velocity_profile, s2_dist_fn,
    make_leg_cycle_fn, make_leg_cycle_vel_fn,
)

_robot   = DEFAULT_PARAMS.robot
_timings = DEFAULT_PARAMS.scenarios
_bumps   = DEFAULT_PARAMS.s5_bumps

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
    name="s05_vel_pi_leg_cycling",
    display_name="S5 — VelPI Leg Cycling",
    duration=13.0,                               # SCENARIO_5_DURATION
    active_controllers=frozenset({"lqr", "velocity_pi"}),
    hip_mode="position",
    v_profile=s3_velocity_profile,
    hip_profile=make_leg_cycle_fn(_robot, _timings),
    hip_vel_profile=make_leg_cycle_vel_fn(_robot, _timings),
    dist_fn=s2_dist_fn,
    world=WorldConfig(bumps=tuple(_bumps)),
    fitness_fn=fitness,
    group="velocity_pi",
    order=5.0,
)
