"""s02_leg_height_gain_sched — LQR gain-scheduler validation across full leg stroke.

LQR-only balance while legs cycle sinusoidally through full stroke.
Disturbances: +4 N / −4 N (same as S1).
Hip position servo tracks sinusoidal target with velocity feed-forward.

Fitness = ISE_pitch + 0.05 * ISE_pitch_rate + 0.01 * settle_time + 200 * fell
"""
from master_sim.defaults import DEFAULT_PARAMS
from master_sim.scenarios.base import ScenarioConfig
from master_sim.scenarios.profiles import (
    s1_dist_fn, make_leg_cycle_fn, make_leg_cycle_vel_fn,
)


_robot   = DEFAULT_PARAMS.robot
_timings = DEFAULT_PARAMS.scenarios

# ── Fitness ──────────────────────────────────────────────────────────────────

W_PITCH_RATE = 0.05
W_SETTLE     = 0.01
W_FALL       = 200.0


def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    return (m['ise_pitch']
            + W_PITCH_RATE * m['ise_pitch_rate']
            + W_SETTLE * m['settle_time_s']
            + (W_FALL if fell else 0.0))


# ── Scenario config ──────────────────────────────────────────────────────────

CONFIG = ScenarioConfig(
    name="s02_leg_height_gain_sched",
    display_name="S2 — Leg Height Gain Sched",
    duration=12.0,                               # SCENARIO_4_DURATION
    active_controllers=frozenset({"lqr"}),        # LQR only
    hip_mode="position",
    hip_profile=make_leg_cycle_fn(_robot, _timings),
    hip_vel_profile=make_leg_cycle_vel_fn(_robot, _timings),
    dist_fn=s1_dist_fn,                           # same +4N/−4N as S1
    fitness_fn=fitness,
    group="lqr",
    order=2.0,
)
