"""s03_vel_pi_disturbance — VelocityPI velocity-tracking test.

VelocityPI outer + LQR inner.  Simple 0 → +0.5 m/s step at t=6 s
(midpoint).  No external disturbances.

Fitness = weighted sum of normalised metrics (weights sum to 1.0) + fell penalty.
"""
from master_sim.defaults import DEFAULT_PARAMS
from master_sim.scenarios.base import ScenarioConfig
from master_sim.scenarios.profiles import s3_vel_step_profile

_timings = DEFAULT_PARAMS.scenarios

# ── REF values: normalise each metric so 1 REF-unit ≈ 1.0 ───────────────────
#
# First-principles estimates for S3 (single 0.5 m/s step, no disturbance, 12 s):
#
#   VEL_MS [m/s RMS]:  Tracking a 0.5 m/s step with PI; ~0.15 m/s RMS
#       error is decent for a well-tuned controller.
#
#   PITCH_RATE_DPS [deg/s RMS]:  Mild disturbance; 10 deg/s RMS is typical.
#       Catches oscillatory settling without penalising commanded lean.
#
REF_VEL_MS          = 0.15
REF_PITCH_RATE_DPS  = 10.0

# ── Weights: fraction of fitness budget per metric (must sum to 1.0) ──────────
W_VEL   = 0.75
W_RATE  = 0.25
W_FELL  = 200.0


def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    return (W_VEL   * m['vel_track_rms_ms']   / REF_VEL_MS
            + W_RATE  * m['rms_pitch_rate_dps'] / REF_PITCH_RATE_DPS
            + (W_FELL if fell else 0.0))


# ── Scenario config ──────────────────────────────────────────────────────────

CONFIG = ScenarioConfig(
    name="s03_vel_pi_disturbance",
    display_name="S3 — VelPI Disturbance",
    duration=_timings.s3_duration,
    active_controllers=frozenset({"lqr", "velocity_pi"}),
    hip_mode="position",
    v_profile=s3_vel_step_profile,                  # 0 → +0.5 m/s step at t=6s
    dist_fn=None,                                   # no disturbance
    max_vel_error_ms=0.5,                             # kill if |v_err| > 0.5 m/s
    fitness_fn=fitness,
    group="velocity_pi",
    order=3.0,
)
