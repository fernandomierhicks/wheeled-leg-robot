"""s01_lqr_pitch_step — LQR pitch step-response test.

Pure LQR inner-loop only.  Robot starts at equilibrium + 5° perturbation,
must recover to upright.  No external force disturbances, v_ref = 0.

Fitness = weighted sum of raw metrics + fell penalty.
Currently optimising for pitch tracking only (vel/settle weights zeroed).
"""
import math
import mujoco

from master_sim_jump.defaults import DEFAULT_PARAMS
from master_sim_jump.scenarios.base import ScenarioConfig
from master_sim_jump.scenarios.profiles import s1_velocity_profile, s1_theta_ref_profile
from master_sim_jump.physics import get_equilibrium_pitch

_timings = DEFAULT_PARAMS.scenarios


# ── Init function: apply pitch step perturbation ─────────────────────────────

def _s1_init(model, data, params):
    """Perturb pitch by PITCH_STEP_RAD on top of equilibrium."""
    robot = params.robot
    s_root = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")]
    theta = get_equilibrium_pitch(robot, robot.Q_NOM) + params.scenarios.pitch_step_rad
    data.qpos[s_root + 3] = math.cos(theta / 2.0)
    data.qpos[s_root + 4] = 0.0
    data.qpos[s_root + 5] = math.sin(theta / 2.0)
    data.qpos[s_root + 6] = 0.0
    mujoco.mj_forward(model, data)


# ── REF values: normalise each metric so 1 REF-unit ≈ 1.0 ───────────────────
#
# First-principles estimates for S1 (5° pitch step, no disturbances, 5 s):
#
#   ISE_PITCH [rad²·s]:  Assume exponential recovery with τ ≈ 0.5 s.
#       Single recovery event: ISE ≈ (0.087)² × τ/2 ≈ 0.0019 → ~0.005.
#
#   ISE_PITCH_RATE [rad²/s]:  Peak rate ≈ amplitude/τ ≈ 0.17 rad/s.
#       Single event: ISE ≈ (0.17)² × τ/2 ≈ 0.007 → ~0.02.
#
#   SETTLE_S [s]:  From 5° a well-tuned LQR should decay below ±2° in
#       roughly 1–2 s.
#
REF_ISE_PITCH       = 0.005
REF_ISE_PITCH_RATE  = 0.02
REF_SETTLE_S        = 2.0
REF_VEL_MS          = 0.2

# ── Weights: fraction of fitness budget per metric (must sum to 1.0) ──────────
W_PITCH      = 0.50
W_PITCH_RATE = 0.50
W_SETTLE     = 0.0
W_VEL        = 0.0
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
    name="s01_lqr_pitch_step",
    display_name="S1 — LQR Pitch Step",
    duration=_timings.s1_duration,
    active_controllers=frozenset({"lqr"}),       # LQR only, no VelocityPI
    hip_mode="none",
    v_profile=s1_velocity_profile,               # zero throughout
    theta_ref_profile=s1_theta_ref_profile,      # −5° step at t=2.5s
    dist_fn=None,
    init_fn=_s1_init,
    fitness_fn=fitness,
    group="lqr",
    order=1.0,
    max_liftoff_s=0.1,     # kill bouncing gains — wheels must stay on ground
)
