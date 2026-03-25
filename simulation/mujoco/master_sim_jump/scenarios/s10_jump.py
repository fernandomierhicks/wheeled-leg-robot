"""s10_jump — Jump scenario (Phase 1: constant gains).

1 s balance settling → 2 s crouch → extension → flight → landing → recovery.
All controllers run with baseline gains; only hip torque is overridden during EXTEND.

Fitness: survival + post-landing pitch recovery.
"""
from master_sim_jump.scenarios.base import ScenarioConfig


# ── Fitness ──────────────────────────────────────────────────────────────────

W_FELL = 200.0
W_PITCH_RMS = 1.0
W_SETTLE = 0.5

REF_PITCH_RMS = 5.0    # [deg] expected post-landing pitch excursion
REF_SETTLE_S = 6.0     # [s] generous settle budget for 8 s scenario


def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    bd = {
        'pitch_rms':  W_PITCH_RMS * m['rms_pitch_deg'] / REF_PITCH_RMS,
        'settle':     W_SETTLE * m['settle_time_s'] / REF_SETTLE_S,
        'FELL':       W_FELL if fell else 0.0,
    }
    m['fitness_breakdown'] = bd
    return sum(bd.values())


# ── Scenario config ──────────────────────────────────────────────────────────

CONFIG = ScenarioConfig(
    name="s10_jump",
    display_name="S10 — Jump",
    duration=8.0,
    active_controllers=frozenset({"lqr", "velocity_pi", "yaw_pi"}),
    hip_mode="jump",
    jump_time=1.0,
    fitness_fn=fitness,
    group="jump",
    order=10.0,
)
