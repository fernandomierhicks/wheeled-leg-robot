"""optimize_integrated.py — Optimizer that auto-selects search space per scenario.

When --scenario is s09_integrated (default), tunes all 12 gains.
For any other scenario, looks up the scenario's group and only tunes the
gains relevant to that group (e.g. s01 → LQR 4 gains only).

    python -m master_sim.optimizer.optimize_integrated --hours 4
    python -m master_sim.optimizer.optimize_integrated --hours 0.07 --scenario s01_lqr_pitch_step
"""
import argparse
import multiprocessing

from master_sim_jump.optimizer.common import (
    EvalWithAllGains, default_seed_all, parse_seed_gains,
)


def _resolve_search_space(scenario_name: str):
    """Return the SearchSpace matching the scenario's group."""
    from master_sim_jump.scenarios import SCENARIOS
    from master_sim_jump.optimizer.search_space import SPACE_BY_GROUP, INTEGRATED_SPACE

    cfg = SCENARIOS.get(scenario_name)
    if cfg is None:
        raise ValueError(f"Unknown scenario: {scenario_name!r}")
    space = SPACE_BY_GROUP.get(cfg.group, INTEGRATED_SPACE)
    return space


def main():
    from master_sim_jump.optimizer.es_engine import ESOptimizer, ESConfig
    from master_sim_jump.optimizer.progress_ui import ProgressUI
    from master_sim_jump.optimizer.run_log import get_scenario_csv_path

    ap = argparse.ArgumentParser(
        description="Optimizer — auto-selects search space per scenario group")
    ap.add_argument("--hours",    type=float, default=None)
    ap.add_argument("--iters",    type=int,   default=None)
    ap.add_argument("--workers",  type=int,   default=None)
    ap.add_argument("--scenario", type=str,   default="s09_integrated")
    ap.add_argument("--patience", type=int,   default=300)
    ap.add_argument("--tol",      type=float, default=1e-4)
    ap.add_argument("--seed-gains", type=str, default=None,
                    help='e.g. "K1=0.5,K2=1.2"')
    ap.add_argument("--no-baseline", action="store_true",
                    help="Skip auto-baselining after optimizer finishes")
    ap.add_argument("--random-seed", action="store_true",
                    help="Start from a random point instead of baseline gains")
    args = ap.parse_args()

    # Picklable eval — scenario name baked in, safe for Windows spawn
    eval_fn = EvalWithAllGains(args.scenario)

    # Pick search space based on scenario group
    space = _resolve_search_space(args.scenario)

    csv_path = get_scenario_csv_path(args.scenario)

    # Seed: random, explicit, or baseline defaults
    if args.random_seed:
        seed = None
    elif args.seed_gains:
        seed = parse_seed_gains(args.seed_gains)
    else:
        all_seeds = default_seed_all()
        seed = {k: v for k, v in all_seeds.items() if k in space.params}

    print(f"Scenario: {args.scenario}  |  Search space: {space.dim}D "
          f"({', '.join(space.names)})")

    # All 12 defaults for the gains panel (active ones update live, rest dimmed)
    all_defaults = default_seed_all()
    active_names = set(space.names)

    ui = ProgressUI(f"Optimizer — {args.scenario} ({space.dim}D)",
                    all_defaults=all_defaults, active_names=active_names)
    cfg = ESConfig(patience=args.patience, tol=args.tol, n_workers=args.workers)
    opt = ESOptimizer(
        search_space=space,
        eval_fn=eval_fn,
        csv_path=csv_path,
        config=cfg,
        progress_fn=ui.update_from_progress,
        pause_fn=ui.wait_if_paused,
    )
    try:
        result = opt.run(hours=args.hours, max_iters=args.iters, seed_params=seed)
    finally:
        ui.finish()

    # Auto-baseline best gains
    if not args.no_baseline and result and result.get("best_params"):
        from master_sim_jump.optimizer.baseline import save_baseline_gains
        save_baseline_gains(
            best_params=result["best_params"],
            best_fitness=result["best_fitness"],
            scenario=args.scenario,
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
