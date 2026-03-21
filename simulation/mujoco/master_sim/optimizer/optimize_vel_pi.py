"""optimize_vel_pi.py — Velocity PI gain optimizer.

Searches KP_V, KI_V via (1+8)-ES.

    python -m master_sim.optimizer.optimize_vel_pi --hours 1
    python -m master_sim.optimizer.optimize_vel_pi --scenario s04_vel_pi_staircase
"""
import multiprocessing

from master_sim.optimizer.common import eval_with_gains, optimizer_main

_SCENARIO_NAME = "s04_vel_pi_staircase"
_GAINS_KEY = "velocity_pi"
_PARAM_MAP = {
    "KP_V": "Kp",
    "KI_V": "Ki",
}


def eval_vel_pi(candidate: dict) -> dict:
    return eval_with_gains(candidate, _SCENARIO_NAME, _GAINS_KEY, _PARAM_MAP)


def main():
    def _set(s):
        global _SCENARIO_NAME
        _SCENARIO_NAME = s

    from master_sim.optimizer.search_space import VELOCITY_PI_SPACE
    optimizer_main(
        description="Velocity PI optimizer (1+8)-ES",
        default_scenario="s04_vel_pi_staircase",
        search_space=VELOCITY_PI_SPACE,
        eval_fn=eval_vel_pi,
        gains_key=_GAINS_KEY,
        param_mapping=_PARAM_MAP,
        ui_label="VelocityPI",
        set_scenario=_set,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
