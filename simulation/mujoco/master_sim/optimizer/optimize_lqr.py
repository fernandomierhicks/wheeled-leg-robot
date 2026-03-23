"""optimize_lqr.py — LQR Q/R cost weight optimizer.

Searches Q_PITCH, Q_PITCH_RATE, Q_VEL, R via (1+8)-ES.

    python -m master_sim.optimizer.optimize_lqr --hours 1
    python -m master_sim.optimizer.optimize_lqr --scenario s01_lqr_pitch_step --hours 0.5
"""
import multiprocessing

from master_sim.optimizer.common import optimizer_main

_GAINS_KEY = "lqr"
_PARAM_MAP = {
    "Q_PITCH": "Q_pitch",
    "Q_PITCH_RATE": "Q_pitch_rate",
    "Q_VEL": "Q_vel",
    "R": "R",
}


def main():
    from master_sim.optimizer.search_space import LQR_SPACE
    optimizer_main(
        description="LQR Q/R optimizer (1+8)-ES",
        default_scenario="s02_leg_height_gain_sched",
        search_space=LQR_SPACE,
        gains_key=_GAINS_KEY,
        param_mapping=_PARAM_MAP,
        ui_label="LQR",
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
