"""Scenario registry and headless evaluation entry point.

The registry is loaded lazily.  Importing ``scenarios.base`` from low-level
simulation/identification code must not load ``DEFAULT_PARAMS`` first: doing so
would make it impossible for ``fit_robot_match`` to regenerate a deliberately
stale controller-snapshot lock after the GUI export changes.
"""

from __future__ import annotations

from .base import ScenarioConfig, WorldConfig


_SCENARIOS: dict[str, ScenarioConfig] | None = None


def get_scenarios() -> dict[str, ScenarioConfig]:
    global _SCENARIOS
    if _SCENARIOS is None:
        from .s01_lqr_pitch_step import CONFIG as s01
        from .s02_leg_height_gain_sched import CONFIG as s02
        from .s03_vel_pi_disturbance import CONFIG as s03
        from .s04_vel_pi_staircase import CONFIG as s04
        from .s05_vel_pi_leg_cycling import CONFIG as s05
        from .s06_yaw_pi_turn import CONFIG as s06
        from .s07_drive_turn import CONFIG as s07
        from .s08_terrain_compliance import CONFIG as s08
        from .s09_integrated import CONFIG as s09
        from .s10_jump import CONFIG as s10

        _SCENARIOS = {
            scenario.name: scenario
            for scenario in (s01, s02, s03, s04, s05, s06, s07, s08, s09, s10)
        }
    return _SCENARIOS


def __getattr__(name: str):
    if name == "SCENARIOS":
        return get_scenarios()
    raise AttributeError(name)


def evaluate(params, scenario_name: str, rng_seed: int | None = None) -> dict:
    """Run a named scenario and return its metrics plus fitness."""
    from v4_twin_279mm_baseline.sim_loop import run

    config = get_scenarios()[scenario_name]
    metrics = run(params, config, rng_seed=rng_seed)
    metrics["fitness"] = round(config.fitness_fn(metrics), 6)
    return metrics


__all__ = ("ScenarioConfig", "WorldConfig", "SCENARIOS", "get_scenarios", "evaluate")
