"""s01_lqr_pitch_step — LQR pitch step-response test.

Pure LQR inner-loop only.  Robot starts at equilibrium + 5° perturbation,
must recover under +4 N / −4 N impulse disturbances.

Fitness = ISE_pitch + 0.05 * ISE_pitch_rate + 0.01 * settle_time + 200 * fell
"""
import math
import mujoco

from master_sim.scenarios.base import ScenarioConfig
from master_sim.scenarios.profiles import s1_dist_fn
from master_sim.physics import get_equilibrium_pitch


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
    name="s01_lqr_pitch_step",
    display_name="S1 — LQR Pitch Step",
    duration=5.0,                               # SCENARIO_1_DURATION
    active_controllers=frozenset({"lqr"}),       # LQR only, no VelocityPI
    hip_mode="position",
    dist_fn=s1_dist_fn,
    init_fn=_s1_init,
    fitness_fn=fitness,
    group="lqr",
    order=1.0,
)
