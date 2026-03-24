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

def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    bd = {
        'velocity': CONFIG.W_VEL_ERR * m['vel_track_rms_ms'],
        'pitch':    0.1 * CONFIG.W_RMS * m['rms_pitch_deg'],
        'FELL':     CONFIG.W_FALL if fell else 0.0,
    }
    m['fitness_breakdown'] = bd
    return sum(bd.values())


# ── Scenario config ──────────────────────────────────────────────────────────

CONFIG = ScenarioConfig(
    name="s05_vel_pi_leg_cycling",
    display_name="S5 — VelPI Leg Cycling",
    duration=_timings.s5_duration,
    active_controllers=frozenset({"lqr", "velocity_pi"}),
    hip_mode="position",
    v_profile=s3_velocity_profile,
    hip_profile=make_leg_cycle_fn(_robot, _timings),
    hip_vel_profile=make_leg_cycle_vel_fn(_robot, _timings),
    dist_fn=s2_dist_fn,
    world=WorldConfig(),  # no bumps — keep sim deterministic
    fitness_fn=fitness,
    group="velocity_pi",
    order=5.0,
)
