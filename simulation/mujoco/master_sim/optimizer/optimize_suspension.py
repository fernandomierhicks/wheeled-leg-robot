"""optimize_suspension.py — Suspension / roll leveling gain optimizer.

Searches LEG_K_S, LEG_B_S, LEG_K_ROLL, LEG_D_ROLL via (1+8)-ES.

    python -m master_sim.optimizer.optimize_suspension --hours 1
"""
import argparse
import multiprocessing
from dataclasses import replace

_SCENARIO_NAME = "s08_terrain_compliance"


def eval_suspension(candidate: dict) -> dict:
    from master_sim.defaults import DEFAULT_PARAMS
    from master_sim.scenarios import evaluate

    p = DEFAULT_PARAMS
    new_susp = replace(p.gains.suspension,
                        K_s=candidate["LEG_K_S"],
                        B_s=candidate["LEG_B_S"],
                        K_roll=candidate["LEG_K_ROLL"],
                        D_roll=candidate["LEG_D_ROLL"])
    new_gains = replace(p.gains, suspension=new_susp)
    params = replace(p, gains=new_gains)

    metrics = evaluate(params, _SCENARIO_NAME)
    metrics["scenario"] = _SCENARIO_NAME
    return metrics


def main():
    global _SCENARIO_NAME

    ap = argparse.ArgumentParser(description="Suspension optimizer (1+8)-ES")
    ap.add_argument("--hours",    type=float, default=None)
    ap.add_argument("--iters",    type=int,   default=None)
    ap.add_argument("--workers",  type=int,   default=None)
    ap.add_argument("--scenario", type=str,   default="s08_terrain_compliance")
    ap.add_argument("--patience", type=int,   default=200)
    ap.add_argument("--tol",      type=float, default=1e-4)
    ap.add_argument("--seed-gains", type=str, default=None)
    args = ap.parse_args()

    _SCENARIO_NAME = args.scenario

    from master_sim.optimizer.search_space import SUSPENSION_SPACE
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
        p = DEFAULT_PARAMS.gains.suspension
        seed = {"LEG_K_S": p.K_s, "LEG_B_S": p.B_s,
                "LEG_K_ROLL": p.K_roll, "LEG_D_ROLL": p.D_roll}

    ui = ProgressUI(f"Suspension Optimizer — {args.scenario}")
    cfg = ESConfig(patience=args.patience, tol=args.tol, n_workers=args.workers)
    opt = ESOptimizer(
        search_space=SUSPENSION_SPACE,
        eval_fn=eval_suspension,
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
