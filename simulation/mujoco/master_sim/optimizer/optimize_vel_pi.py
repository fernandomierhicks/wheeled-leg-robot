"""optimize_vel_pi.py — Velocity PI gain optimizer.

Searches KP_V, KI_V via (1+8)-ES.

    python -m master_sim.optimizer.optimize_vel_pi --hours 1
    python -m master_sim.optimizer.optimize_vel_pi --scenario s04_vel_pi_staircase
"""
import argparse
import multiprocessing
from dataclasses import replace

_SCENARIO_NAME = "s04_vel_pi_staircase"


def eval_vel_pi(candidate: dict) -> dict:
    from master_sim.defaults import DEFAULT_PARAMS
    from master_sim.scenarios import evaluate

    p = DEFAULT_PARAMS
    new_vpi = replace(p.gains.velocity_pi,
                       Kp=candidate["KP_V"],
                       Ki=candidate["KI_V"])
    new_gains = replace(p.gains, velocity_pi=new_vpi)
    params = replace(p, gains=new_gains)

    metrics = evaluate(params, _SCENARIO_NAME)
    metrics["scenario"] = _SCENARIO_NAME
    return metrics


def main():
    global _SCENARIO_NAME

    ap = argparse.ArgumentParser(description="Velocity PI optimizer (1+8)-ES")
    ap.add_argument("--hours",    type=float, default=None)
    ap.add_argument("--iters",    type=int,   default=None)
    ap.add_argument("--workers",  type=int,   default=None)
    ap.add_argument("--scenario", type=str,   default="s04_vel_pi_staircase")
    ap.add_argument("--patience", type=int,   default=200)
    ap.add_argument("--tol",      type=float, default=1e-4)
    ap.add_argument("--seed-gains", type=str, default=None)
    args = ap.parse_args()

    _SCENARIO_NAME = args.scenario

    from master_sim.optimizer.search_space import VELOCITY_PI_SPACE
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
        p = DEFAULT_PARAMS.gains.velocity_pi
        seed = {"KP_V": p.Kp, "KI_V": p.Ki}

    ui = ProgressUI(f"VelocityPI Optimizer — {args.scenario}")
    cfg = ESConfig(patience=args.patience, tol=args.tol, n_workers=args.workers)
    opt = ESOptimizer(
        search_space=VELOCITY_PI_SPACE,
        eval_fn=eval_vel_pi,
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
