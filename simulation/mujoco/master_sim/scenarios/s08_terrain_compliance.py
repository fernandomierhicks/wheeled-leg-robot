"""s08_terrain_compliance — Roll leveling + suspension test with force pulse.

LQR + VelocityPI + roll leveling impedance mode.
v_desired = 1.0 m/s constant.  A 7 N vertical force pulse on the left wheel hub
at t = 2.0 s for 0.2 s creates a roll disturbance.

Fitness = weighted sum of normalised metrics (weights sum to 1.0) + fell penalty.
"""
from master_sim.defaults import DEFAULT_PARAMS
from master_sim.scenarios.base import ScenarioConfig
from master_sim.scenarios.profiles import constant_velocity

_timings = DEFAULT_PARAMS.scenarios


def _roll_pulse(t: float) -> float:
    """7 N vertical force on left wheel hub for 0.2 s starting at t = 2.0 s."""
    return 7.0 if 2.0 <= t < 2.2 else 0.0


# ── REF values: normalise each metric so 1 REF-unit ≈ 1.0 ───────────────────
# Raw metrics live on wildly different scales (e.g. max_roll_deg ~ 0.25 vs
# hip_rate_rms ~ 4000).  Dividing by REF puts every term on a comparable
# dimensionless scale so that the W_ weights genuinely represent the fraction
# of fitness budget allocated to each metric.
#
# Each REF is a "typical baseline" value from the current gains (seed 42).
# The exact number is not critical — it just needs to be in the right ballpark
# so no single term dominates by accident.
#
# S8 context: 1 m/s constant drive, 7 N vertical force pulse on left wheel
# at t=2 s for 0.2 s, impedance hips, roll leveling active.
#
#   VEL_TRACK [m/s]:  RMS velocity error while absorbing force pulse.
#       Pulse disturbs pitch → LQR corrects → wheel vel deviates from 1 m/s.
#       Baseline ≈ 0.59.  Use 0.5 — "half a m/s tracking error is one unit".
#       WHY IN FITNESS: suspension gains that absorb roll but destroy forward
#       velocity tracking are useless on the real robot.
#
#   MAX_ROLL [deg]:  Peak roll angle from asymmetric force pulse.
#       This is the PRIMARY objective of S8 — the whole point of impedance
#       suspension + roll leveling is to keep the body level over uneven terrain.
#       Baseline ≈ 0.25°.  Use 0.5° — "half a degree peak roll is one unit".
#
#   RMS_PITCH [deg]:  Average pitch oscillation around equilibrium.
#       Hip impedance gains couple into pitch via the 4-bar linkage — a soft
#       suspension that absorbs roll may let the body pitch-wobble from the pulse.
#       Baseline ≈ 2.6°.  Use 3.0° — "3 degrees RMS pitch is one unit".
#       WHY IN FITNESS: guardrail to prevent the optimizer from trading pitch
#       stability for roll performance.
#
#   HIP_TRACK [rad]:  RMS hip position error (actual vs commanded q_nom).
#       Measures how well the impedance spring tracks its setpoint.  Large error
#       means the legs are bottoming out or the spring is too soft to recover.
#       Baseline ≈ 0.018 rad (≈ 1°).  Use 0.02 — "1° hip tracking error
#       is one unit".
#
#   HIP_RATE [N·m/s]:  RMS rate-of-change of hip actuator torque.
#       Penalises bang-bang behaviour where the actuator slams between ±torque
#       limits every tick.  This is mechanically destructive (gear shock) and
#       electrically wasteful (current spikes).  The impedance torque limit is
#       5 N·m and ctrl_dt = 4 ms, so a full ±5 N·m reversal every tick gives
#       10/0.004 = 2500 N·m/s.  Baseline RMS ≈ 4150 — even worse than one
#       full reversal per tick on average, confirming severe bang-bang.
#       Use 2500 — "one full torque reversal per tick is one unit".
#
REF_VEL_TRACK  = 0.5       # m/s
REF_MAX_ROLL   = 0.5       # deg
REF_RMS_PITCH  = 3.0       # deg
REF_HIP_TRACK  = 0.02      # rad
REF_HIP_RATE   = 2500.0    # N·m/s

# ── Weights: fraction of fitness budget per metric (must sum to 1.0) ──────────
W_ROLL      = 0.40   # primary objective — minimise peak roll from force pulse
W_VEL       = 0.0    # maintain forward velocity while absorbing terrain
W_PITCH     = 0.10   # don't trade pitch stability for roll compliance
W_HIP_TRACK = 0.10   # hip spring must track its setpoint, not bottom out
W_HIP_RATE  = 0.40   # penalise bang-bang torque — smooth actuator effort
W_FELL      = 200.0  # binary survival penalty (not part of the 1.0 budget)


# ── Fitness ──────────────────────────────────────────────────────────────────

def fitness(m: dict) -> float:
    fell = m.get('fell', m.get('status') == 'FAIL')
    bd = {
        'velocity':  W_VEL       * m['vel_track_rms_ms']  / REF_VEL_TRACK,
        'roll':      W_ROLL      * m['max_roll_deg']      / REF_MAX_ROLL,
        'pitch':     W_PITCH     * m['rms_pitch_deg']     / REF_RMS_PITCH,
        'hip_track': W_HIP_TRACK * m['hip_track_rms_rad'] / REF_HIP_TRACK,
        'hip_rate':  W_HIP_RATE  * m['hip_rate_rms']      / REF_HIP_RATE,
        'FELL':      W_FELL if fell else 0.0,
    }
    m['fitness_breakdown'] = bd
    return sum(bd.values())


# ── Scenario config ──────────────────────────────────────────────────────────

CONFIG = ScenarioConfig(
    name="s08_terrain_compliance",
    display_name="S8 — Terrain Compliance",
    duration=_timings.s8_duration,
    active_controllers=frozenset({"lqr", "velocity_pi"}),
    hip_mode="impedance",
    v_profile=constant_velocity(_timings.s8_drive_speed),
    roll_dist_fn=_roll_pulse,
    fitness_fn=fitness,
    group="suspension",
    order=8.0,
)
