"""scenarios — Scenario registry and evaluate() entry point.

All scenarios are registered in SCENARIOS dict by name.
evaluate(params, scenario_name) runs the scenario headlessly and returns metrics + fitness.
"""
from master_sim_jump.scenarios.base import ScenarioConfig, WorldConfig
from master_sim_jump.scenarios.s01_lqr_pitch_step import CONFIG as S01
from master_sim_jump.scenarios.s02_leg_height_gain_sched import CONFIG as S02
from master_sim_jump.scenarios.s03_vel_pi_disturbance import CONFIG as S03
from master_sim_jump.scenarios.s04_vel_pi_staircase import CONFIG as S04
from master_sim_jump.scenarios.s05_vel_pi_leg_cycling import CONFIG as S05
from master_sim_jump.scenarios.s06_yaw_pi_turn import CONFIG as S06
from master_sim_jump.scenarios.s07_drive_turn import CONFIG as S07
from master_sim_jump.scenarios.s08_terrain_compliance import CONFIG as S08
from master_sim_jump.scenarios.s09_integrated import CONFIG as S09
from master_sim_jump.scenarios.s10_jump import CONFIG as S10


# ── Registry ──────────────────────────────────────────────────────────────────

SCENARIOS: dict[str, ScenarioConfig] = {
    s.name: s for s in [S01, S02, S03, S04, S05, S06, S07, S08, S09, S10]
}


def evaluate(params, scenario_name: str, rng_seed: int = None) -> dict:
    """Run a named scenario headlessly and return metrics dict with 'fitness' key.

    Parameters
    ----------
    params         : SimParams
    scenario_name  : key in SCENARIOS, e.g. "s01_lqr_pitch_step"
    rng_seed       : reproducible noise seed

    Returns
    -------
    dict with all sim_loop metrics + computed 'fitness' value
    """
    from master_sim_jump.sim_loop import run

    cfg = SCENARIOS[scenario_name]
    metrics = run(params, cfg, rng_seed=rng_seed)
    metrics['fitness'] = round(cfg.fitness_fn(metrics), 6)
    return metrics
