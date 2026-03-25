"""optimize_suspension.py — Suspension / roll leveling gain optimizer.

Searches LEG_K_S, LEG_B_S, LEG_K_ROLL, LEG_D_ROLL via (1+8)-ES.

    python -m master_sim.optimizer.optimize_suspension --hours 1
"""
import multiprocessing

from master_sim_jump.optimizer.common import optimizer_main

_GAINS_KEY = "suspension"
_PARAM_MAP = {
    "LEG_K_S": "K_s",
    "LEG_B_S": "B_s",
    "LEG_K_ROLL": "K_roll",
    "LEG_D_ROLL": "D_roll",
}


def main():
    from master_sim_jump.optimizer.search_space import SUSPENSION_SPACE
    optimizer_main(
        description="Suspension optimizer (1+8)-ES",
        default_scenario="s08_terrain_compliance",
        search_space=SUSPENSION_SPACE,
        gains_key=_GAINS_KEY,
        param_mapping=_PARAM_MAP,
        ui_label="Suspension",
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
