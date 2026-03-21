"""optimize_lqr.py — LQR Q/R cost weight optimizer.

Searches Q_PITCH, Q_PITCH_RATE, Q_VEL, R via (1+8)-ES.

    python -m master_sim.optimizer.optimize_lqr --hours 1
    python -m master_sim.optimizer.optimize_lqr --scenario s01_lqr_pitch_step --hours 0.5
"""
import argparse
import multiprocessing
from dataclasses import replace

# ── Eval function (module-level for pickling on Windows) ─────────────────────

_SCENARIO_NAME = "s02_leg_height_gain_sched"   # overridden by main()


def eval_lqr(candidate: dict) -> dict:
    """Evaluate one LQR candidate — called in subprocess."""
    from master_sim.defaults import DEFAULT_PARAMS
    from master_sim.scenarios import evaluate
    from master_sim.controllers.lqr import compute_gain_table

    p = DEFAULT_PARAMS
    new_lqr = replace(p.gains.lqr,
                       Q_pitch=candidate["Q_PITCH"],
                       Q_pitch_rate=candidate["Q_PITCH_RATE"],
                       Q_vel=candidate["Q_VEL"],
                       R=candidate["R"])
    new_gains = replace(p.gains, lqr=new_lqr)
    params = replace(p, gains=new_gains)

    metrics = evaluate(params, _SCENARIO_NAME)
    metrics["scenario"] = _SCENARIO_NAME
    return metrics


def main():
    global _SCENARIO_NAME

    ap = argparse.ArgumentParser(description="LQR Q/R optimizer (1+8)-ES")
    ap.add_argument("--hours",    type=float, default=None)
    ap.add_argument("--iters",    type=int,   default=None)
    ap.add_argument("--workers",  type=int,   default=None)
    ap.add_argument("--scenario", type=str,   default="s02_leg_height_gain_sched")
    ap.add_argument("--patience", type=int,   default=200)
    ap.add_argument("--tol",      type=float, default=1e-4)
    ap.add_argument("--seed-gains", type=str, default=None,
                    help='e.g. "Q_PITCH=0.063,Q_PITCH_RATE=0.00022,Q_VEL=1.1e-5,R=1.98"')
    args = ap.parse_args()

    _SCENARIO_NAME = args.scenario

    from master_sim.optimizer.search_space import LQR_SPACE
    from master_sim.optimizer.es_engine import ESOptimizer, ESConfig
    from master_sim.optimizer.progress_ui import ProgressUI
    from master_sim.optimizer.run_log import get_scenario_csv_path
    from master_sim.defaults import DEFAULT_PARAMS

    csv_path = get_scenario_csv_path(args.scenario)

    # Seed from args or defaults
    seed = None
    if args.seed_gains:
        seed = {k.strip(): float(v.strip())
                for part in args.seed_gains.split(",")
                for k, _, v in [part.partition("=")]}
    else:
        p = DEFAULT_PARAMS.gains.lqr
        seed = {"Q_PITCH": p.Q_pitch, "Q_PITCH_RATE": p.Q_pitch_rate,
                "Q_VEL": p.Q_vel, "R": p.R}

    ui = ProgressUI(f"LQR Optimizer — {args.scenario}")
    cfg = ESConfig(
        patience=args.patience, tol=args.tol,
        n_workers=args.workers,
    )
    opt = ESOptimizer(
        search_space=LQR_SPACE,
        eval_fn=eval_lqr,
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
