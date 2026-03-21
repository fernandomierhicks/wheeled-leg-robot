"""optimize_yaw_pi.py — Yaw PI gain optimizer.

Searches KP_YAW, KI_YAW via (1+8)-ES.

    python -m master_sim.optimizer.optimize_yaw_pi --hours 1
    python -m master_sim.optimizer.optimize_yaw_pi --scenario s06_yaw_pi_turn
"""
import argparse
import multiprocessing
from dataclasses import replace

_SCENARIO_NAME = "s07_drive_turn"


def eval_yaw_pi(candidate: dict) -> dict:
    from master_sim.defaults import DEFAULT_PARAMS
    from master_sim.scenarios import evaluate

    p = DEFAULT_PARAMS
    new_yaw = replace(p.gains.yaw_pi,
                       Kp=candidate["KP_YAW"],
                       Ki=candidate["KI_YAW"])
    new_gains = replace(p.gains, yaw_pi=new_yaw)
    params = replace(p, gains=new_gains)

    metrics = evaluate(params, _SCENARIO_NAME)
    metrics["scenario"] = _SCENARIO_NAME
    return metrics


def main():
    global _SCENARIO_NAME

    ap = argparse.ArgumentParser(description="Yaw PI optimizer (1+8)-ES")
    ap.add_argument("--hours",    type=float, default=None)
    ap.add_argument("--iters",    type=int,   default=None)
    ap.add_argument("--workers",  type=int,   default=None)
    ap.add_argument("--scenario", type=str,   default="s07_drive_turn")
    ap.add_argument("--patience", type=int,   default=200)
    ap.add_argument("--tol",      type=float, default=1e-4)
    ap.add_argument("--seed-gains", type=str, default=None)
    args = ap.parse_args()

    _SCENARIO_NAME = args.scenario

    from master_sim.optimizer.search_space import YAW_PI_SPACE
    from master_sim.optimizer.es_engine import ESOptimizer, ESConfig
    from master_sim.optimizer.progress_ui import ProgressUI
    from master_sim.optimizer.run_log import get_scenario_csv_path
    from master_sim.defaults import DEFAULT_PARAMS

    csv_path = get_scenario_csv_path(args.scenario)

    seed = None
    if args.seed_gains:
        seed = {k.strip(): float(v.strip())
                for part in args.seed_gains.split(",")
                for k, _, v in [part.partition("=")]}
    else:
        p = DEFAULT_PARAMS.gains.yaw_pi
        seed = {"KP_YAW": p.Kp, "KI_YAW": p.Ki}

    ui = ProgressUI(f"YawPI Optimizer — {args.scenario}")
    cfg = ESConfig(patience=args.patience, tol=args.tol, n_workers=args.workers)
    opt = ESOptimizer(
        search_space=YAW_PI_SPACE,
        eval_fn=eval_yaw_pi,
        csv_path=csv_path,
        config=cfg,
        progress_fn=ui.update_from_progress,
        pause_fn=ui.wait_if_paused,
    )
    try:
        opt.run(hours=args.hours, max_iters=args.iters, seed_params=seed)
    finally:
        ui.finish()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
