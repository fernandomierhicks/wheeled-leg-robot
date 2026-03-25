"""s09_integrated — Joint scenario exercising ALL 4 controllers + impedance.

Continuous sine leg sweep Q_RET↔Q_EXT (4s period) throughout all phases.

5 phases, 16s:
  Phase 1 (0-5s):   Balance while legs cycle — impedance only
  Phase 2 (5-10s):  Drive + wheel-hub kick at Q_NOM crossing — VelPI + impedance
  Phase 3 (10-14s): 60°/s turns + roll kicks — full cross-coupling
  Phase 4 (14-16s): Cruise at 0.5 m/s — settling

Fitness = weighted sum of REF-normalised metrics (weights sum to 1.0) + fell penalty.

Pitch metric subtracts theta_ref (VelPI lean command) so commanded lean
is not penalised as balance error.  The pitch_rate_dps term penalises
oscillatory behaviour without dominating the position-error terms.
Hip tracking and hip rate terms penalise jerky/inaccurate impedance control
during the continuous leg sweep.
"""
from master_sim_jump.defaults import DEFAULT_PARAMS
from master_sim_jump.scenarios.base import ScenarioConfig
from master_sim_jump.scenarios.profiles import (
    s9_velocity_profile, s9_yaw_profile, s9_dist_fn, s9_hip_sweep,
)

_timings = DEFAULT_PARAMS.scenarios
_robot = DEFAULT_PARAMS.robot

# ── REF values: normalise each metric so 1 REF-unit ≈ 1.0 ───────────────────
# Raw metrics live on wildly different scales (e.g. rms_pitch_deg ~ 0.5 vs
# yaw_track_rms_rads ~ 0.05).  Dividing by REF puts every term on a comparable
# dimensionless scale so that the W_ weights genuinely represent the fraction
# of fitness budget allocated to each metric.  Without REFs the weights would
# need to span orders of magnitude and "sum to 1.0" would be meaningless.
#
# Choose each REF as a "typical" or "acceptable" value for that metric — the
# exact number is not critical, it just needs to be in the right ballpark so
# no single term dominates by accident.
#
# First-principles estimates for S9 (all controllers, leg sweep, drive+turn, 16 s):
#
#   PITCH_DEG [deg RMS]:  Cross-coupling from leg sweeps, velocity steps, and
#       yaw turns all perturb pitch.  1–2° RMS is acceptable → 1.5.
#
#   PITCH_RATE_DPS [deg/s RMS]:  Penalises oscillatory settling.  With multiple
#       disturbance phases, 10–20 deg/s RMS is typical → 15.0.
#
#   VEL_MS [m/s RMS]:  Driving at 0.5 m/s with disturbances and leg cycling;
#       ~0.2 m/s tracking error is decent → 0.2.
#
#   YAW_RADS [rad/s RMS]:  60°/s (≈1.05 rad/s) commanded turns; ~0.1 rad/s
#       tracking error is good → 0.1.
#
#   ROLL_DEG [deg max]:  Roll kicks applied during Phase 3; 3–5° peak roll
#       before leveling corrects is acceptable → 4.0.
#
#   HIP_TRACK [rad RMS]:  Impedance hip position tracking error during
#       continuous leg sweep; ~0.02 rad is acceptable → 0.02.
#
#   HIP_RATE [N·m/s RMS]:  Penalises jerky impedance torque; ~2500 is
#       typical for smooth tracking → 2500.0.
#
REF_PITCH_DEG       = 1.5
REF_PITCH_RATE_DPS  = 15.0
REF_VEL_MS          = 0.2
REF_YAW_RADS        = 0.1
REF_ROLL_DEG        = 4.0
REF_HIP_TRACK       = 0.02    # rad — same as S8
REF_HIP_RATE        = 2500.0  # N·m/s — same as S8

# ── Weights: fraction of fitness budget per metric (must sum to 1.0) ──────────
W_PITCH     = 0.20
W_RATE      = 0.00
W_VEL       = 0.20
W_YAW       = 0.20
W_ROLL      = 0.20
W_HIP_TRACK = 0.10
W_HIP_RATE  = 0.10
W_FELL      = 200.0


def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    bd = {
        'pitch':     W_PITCH     * m['rms_pitch_deg']       / REF_PITCH_DEG,
        'pitch_rate':W_RATE      * m['rms_pitch_rate_dps']  / REF_PITCH_RATE_DPS,
        'velocity':  W_VEL       * m['vel_track_rms_ms']    / REF_VEL_MS,
        'yaw':       W_YAW       * m['yaw_track_rms_rads']  / REF_YAW_RADS,
        'roll':      W_ROLL      * m['max_roll_deg']         / REF_ROLL_DEG,
        'hip_track': W_HIP_TRACK * m['hip_track_rms_rad']   / REF_HIP_TRACK,
        'hip_rate':  W_HIP_RATE  * m['hip_rate_rms']         / REF_HIP_RATE,
        'FELL':      W_FELL if fell else 0.0,
    }
    m['fitness_breakdown'] = bd
    return sum(bd.values())


CONFIG = ScenarioConfig(
    name="s09_integrated",
    display_name="S9 — Integrated",
    duration=_timings.s9_duration,
    active_controllers=frozenset({"lqr", "velocity_pi", "yaw_pi"}),
    hip_mode="impedance",
    v_profile=s9_velocity_profile,
    omega_profile=s9_yaw_profile,
    hip_profile=s9_hip_sweep(_robot),
    dist_fn=s9_dist_fn,
    dist_target="wheel_L",
    fitness_fn=fitness,
    max_vel_error_ms=0.5,                             # kill if |v_err| > 0.5 m/s
    group="integrated",
    order=9.0,
    use_theta_ref_correction=True,
)
