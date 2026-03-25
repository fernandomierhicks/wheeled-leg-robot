"""s04_vel_pi_staircase — VelocityPI setpoint tracking across six velocity steps.

VelocityPI outer + LQR inner.
Profile: 0 → +0.3 → −0.5 → +0.7 → −1.0 → +1.2 → 0 m/s  (13 s total).

Fitness = weighted sum of normalised metrics (weights sum to 1.0) + fell penalty.
"""
from master_sim_jump.defaults import DEFAULT_PARAMS
from master_sim_jump.scenarios.base import ScenarioConfig
from master_sim_jump.scenarios.profiles import s3_velocity_profile

_timings = DEFAULT_PARAMS.scenarios

# ── REF values: normalise each metric so 1 REF-unit ≈ 1.0 ───────────────────
#
# First-principles estimates for S4 (alternating ±1.2 m/s staircase, 13 s):
#
#   VEL_MS [m/s RMS]:  Larger velocity swings with sign reversals;
#       ~0.25 m/s RMS tracking error is decent.
#
#   PITCH_RATE_DPS [deg/s RMS]:  Sign reversals drive more oscillation;
#       15 deg/s RMS is typical.  Catches oscillatory settling without
#       penalising commanded lean.
#
REF_VEL_MS          = 0.25
REF_PITCH_RATE_DPS  = 15.0

# ── Weights: fraction of fitness budget per metric (must sum to 1.0) ──────────
W_VEL   = 0.75
W_RATE  = 0.25
W_FELL  = 200.0


def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    bd = {
        'velocity':   W_VEL  * m['vel_track_rms_ms']   / REF_VEL_MS,
        'pitch_rate': W_RATE * m['rms_pitch_rate_dps'] / REF_PITCH_RATE_DPS,
        'FELL':       W_FELL if fell else 0.0,
    }
    m['fitness_breakdown'] = bd
    return sum(bd.values())


# ── Scenario config ──────────────────────────────────────────────────────────

CONFIG = ScenarioConfig(
    name="s04_vel_pi_staircase",
    display_name="S4 — VelPI Staircase",
    duration=_timings.s4_duration,
    active_controllers=frozenset({"lqr", "velocity_pi"}),
    hip_mode="position",
    v_profile=s3_velocity_profile,
    max_vel_error_ms=0.5,                             # kill if |v_err| > 0.5 m/s
    fitness_fn=fitness,
    group="velocity_pi",
    order=4.0,
)
