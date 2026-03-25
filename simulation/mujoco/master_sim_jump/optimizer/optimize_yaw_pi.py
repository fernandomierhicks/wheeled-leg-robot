"""optimize_yaw_pi.py — Yaw PI gain optimizer.

Searches KP_YAW, KI_YAW via (1+8)-ES.

    python -m master_sim.optimizer.optimize_yaw_pi --hours 1
    python -m master_sim.optimizer.optimize_yaw_pi --scenario s06_yaw_pi_turn
"""
import multiprocessing

from master_sim_jump.optimizer.common import optimizer_main

_GAINS_KEY = "yaw_pi"
_PARAM_MAP = {
    "KP_YAW": "Kp",
    "KI_YAW": "Ki",
}


def main():
    from master_sim_jump.optimizer.search_space import YAW_PI_SPACE
    optimizer_main(
        description="Yaw PI optimizer (1+8)-ES",
        default_scenario="s07_drive_turn",
        search_space=YAW_PI_SPACE,
        gains_key=_GAINS_KEY,
        param_mapping=_PARAM_MAP,
        ui_label="YawPI",
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
