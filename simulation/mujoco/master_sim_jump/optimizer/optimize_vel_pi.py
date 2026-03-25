"""optimize_vel_pi.py — Velocity PI gain optimizer.

Searches KP_V, KI_V via (1+8)-ES.

    python -m master_sim.optimizer.optimize_vel_pi --hours 1
    python -m master_sim.optimizer.optimize_vel_pi --scenario s04_vel_pi_staircase
"""
import multiprocessing

from master_sim_jump.optimizer.common import optimizer_main

_GAINS_KEY = "velocity_pi"
_PARAM_MAP = {
    "KP_V":  "Kp",
    "KI_V":  "Ki",
    "KFF_V": "Kff",
}


def main():
    from master_sim_jump.optimizer.search_space import VELOCITY_PI_SPACE
    optimizer_main(
        description="Velocity PI optimizer (1+8)-ES",
        default_scenario="s04_vel_pi_staircase",
        search_space=VELOCITY_PI_SPACE,
        gains_key=_GAINS_KEY,
        param_mapping=_PARAM_MAP,
        ui_label="VelocityPI",
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
