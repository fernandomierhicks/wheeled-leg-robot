"""optimize_yaw_pi.py — Yaw PI gain optimizer.

Searches KP_YAW, KI_YAW via (1+8)-ES.

    python -m master_sim.optimizer.optimize_yaw_pi --hours 1
    python -m master_sim.optimizer.optimize_yaw_pi --scenario s06_yaw_pi_turn
"""
import multiprocessing

from master_sim.optimizer.common import eval_with_gains, optimizer_main

_SCENARIO_NAME = "s07_drive_turn"
_GAINS_KEY = "yaw_pi"
_PARAM_MAP = {
    "KP_YAW": "Kp",
    "KI_YAW": "Ki",
}


def eval_yaw_pi(candidate: dict) -> dict:
    return eval_with_gains(candidate, _SCENARIO_NAME, _GAINS_KEY, _PARAM_MAP)


def main():
    def _set(s):
        global _SCENARIO_NAME
        _SCENARIO_NAME = s

    from master_sim.optimizer.search_space import YAW_PI_SPACE
    optimizer_main(
        description="Yaw PI optimizer (1+8)-ES",
        default_scenario="s07_drive_turn",
        search_space=YAW_PI_SPACE,
        eval_fn=eval_yaw_pi,
        gains_key=_GAINS_KEY,
        param_mapping=_PARAM_MAP,
        ui_label="YawPI",
        set_scenario=_set,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
