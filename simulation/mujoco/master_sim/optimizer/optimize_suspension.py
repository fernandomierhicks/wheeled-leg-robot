"""optimize_suspension.py — Suspension / roll leveling gain optimizer.

Searches LEG_K_S, LEG_B_S, LEG_K_ROLL, LEG_D_ROLL via (1+8)-ES.

    python -m master_sim.optimizer.optimize_suspension --hours 1
"""
import multiprocessing

from master_sim.optimizer.common import eval_with_gains, optimizer_main

_SCENARIO_NAME = "s08_terrain_compliance"
_GAINS_KEY = "suspension"
_PARAM_MAP = {
    "LEG_K_S": "K_s",
    "LEG_B_S": "B_s",
    "LEG_K_ROLL": "K_roll",
    "LEG_D_ROLL": "D_roll",
}


def eval_suspension(candidate: dict) -> dict:
    return eval_with_gains(candidate, _SCENARIO_NAME, _GAINS_KEY, _PARAM_MAP)


def main():
    def _set(s):
        global _SCENARIO_NAME
        _SCENARIO_NAME = s

    from master_sim.optimizer.search_space import SUSPENSION_SPACE
    optimizer_main(
        description="Suspension optimizer (1+8)-ES",
        default_scenario="s08_terrain_compliance",
        search_space=SUSPENSION_SPACE,
        eval_fn=eval_suspension,
        gains_key=_GAINS_KEY,
        param_mapping=_PARAM_MAP,
        ui_label="Suspension",
        set_scenario=_set,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
