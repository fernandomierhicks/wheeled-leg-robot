"""s02_leg_height_gain_sched — LQR gain-scheduler validation across full leg stroke.

LQR-only balance while legs cycle sinusoidally through full stroke.
Disturbances: +4 N / −4 N (same as S1).
Hip position servo tracks sinusoidal target with velocity feed-forward.

Fitness = weighted sum of raw metrics (weights must sum to 1.0) + fell penalty
"""
from master_sim_jump.defaults import DEFAULT_PARAMS
from master_sim_jump.scenarios.base import ScenarioConfig
from master_sim_jump.scenarios.profiles import (
    s1_dist_fn, make_leg_cycle_fn, make_leg_cycle_vel_fn,
)


_robot   = DEFAULT_PARAMS.robot
_timings = DEFAULT_PARAMS.scenarios

# ── REF values: normalise each metric so 1 REF-unit ≈ 1.0 ───────────────────
# Raw metrics live on wildly different scales (e.g. ise_pitch ~ 0.001 vs
# vel_track_rms_ms ~ 1.3).  Dividing by REF puts every term on a comparable
# dimensionless scale so that the W_ weights genuinely represent the fraction
# of fitness budget allocated to each metric.  Without REFs the weights would
# need to span 3+ orders of magnitude and "sum to 1.0" would be meaningless.
#
# Choose each REF as a "typical" or "acceptable" value for that metric — the
# exact number is not critical, it just needs to be in the right ballpark so
# no single term dominates by accident.
REF_ISE_PITCH       = 0.005
REF_ISE_PITCH_RATE  = 0.02
REF_SETTLE_S        = 2.0
REF_VEL_MS          = 0.2

# ── Weights: fraction of fitness budget per metric (must sum to 1.0) ──────────
W_PITCH      = 0.40
W_PITCH_RATE = 0.20
W_SETTLE     = 0.10
W_VEL        = 0.30
W_FELL       = 200.0


# ── Fitness ──────────────────────────────────────────────────────────────────

def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    bd = {
        'pitch (ISE)':      W_PITCH      * m['ise_pitch']      / REF_ISE_PITCH,
        'pitch_rate (ISE)': W_PITCH_RATE * m['ise_pitch_rate'] / REF_ISE_PITCH_RATE,
        'settle':           W_SETTLE     * m['settle_time_s']  / REF_SETTLE_S,
        'velocity':         W_VEL        * m['vel_track_rms_ms'] / REF_VEL_MS,
        'FELL':             W_FELL if fell else 0.0,
    }
    m['fitness_breakdown'] = bd
    return sum(bd.values())


# ── Scenario config ──────────────────────────────────────────────────────────

CONFIG = ScenarioConfig(
    name="s02_leg_height_gain_sched",
    display_name="S2 — Leg Height Gain Sched",
    duration=_timings.s2_duration,
    active_controllers=frozenset({"lqr"}),        # LQR only
    hip_mode="position",
    hip_profile=make_leg_cycle_fn(_robot, _timings),
    hip_vel_profile=make_leg_cycle_vel_fn(_robot, _timings),
    dist_fn=s1_dist_fn,                           # same +4N/−4N as S1
    fitness_fn=fitness,
    group="lqr",
    order=2.0,
)
