"""common.py — Shared helpers for all optimizer entry points.

Eliminates boilerplate that was duplicated across optimize_lqr / _vel_pi /
_yaw_pi / _suspension: eval function body, seed parsing, argparse + run loop.

The eval helper uses lazy imports so that each multiprocessing worker only
loads MuJoCo / scenario code on first use (required for Windows pickling).
"""
import argparse
from dataclasses import replace


# ── Eval helper (called inside each module-level eval function) ──────────────

def eval_with_gains(candidate: dict, scenario_name: str,
                    gains_key: str, param_mapping: dict) -> dict:
    """Evaluate one candidate by replacing gains and running a scenario.

    Parameters
    ----------
    candidate : dict
        Search-space keys → float values (e.g. {"Q_PITCH": 0.06, ...}).
    scenario_name : str
        Scenario to evaluate (e.g. "s02_leg_height_gain_sched").
    gains_key : str
        Attribute name on the Gains dataclass (e.g. "lqr", "velocity_pi").
    param_mapping : dict
        Maps candidate keys → dataclass field names on the sub-gains object
        (e.g. {"Q_PITCH": "Q_pitch", "R": "R"}).
    """
    from master_sim.defaults import DEFAULT_PARAMS
    from master_sim.scenarios import evaluate

    p = DEFAULT_PARAMS
    current_sub = getattr(p.gains, gains_key)
    new_sub = replace(current_sub,
                      **{param_mapping[k]: candidate[k] for k in candidate})
    new_gains = replace(p.gains, **{gains_key: new_sub})
    params = replace(p, gains=new_gains)

    metrics = evaluate(params, scenario_name)
    metrics["scenario"] = scenario_name
    return metrics


# ── Seed helpers ─────────────────────────────────────────────────────────────

def parse_seed_gains(raw: str) -> dict:
    """Parse 'K1=v1,K2=v2,...' into {str: float}."""
    return {k.strip(): float(v.strip())
            for part in raw.split(",")
            for k, _, v in [part.partition("=")]}


def default_seed(gains_key: str, param_mapping: dict) -> dict:
    """Extract current default gains as a seed dict."""
    from master_sim.defaults import DEFAULT_PARAMS
    sub = getattr(DEFAULT_PARAMS.gains, gains_key)
    return {cand_key: getattr(sub, field_name)
            for cand_key, field_name in param_mapping.items()}


# ── Shared main() ────────────────────────────────────────────────────────────

def optimizer_main(*, description: str, default_scenario: str,
                   search_space, eval_fn, gains_key: str,
                   param_mapping: dict, ui_label: str,
                   set_scenario) -> None:
    """Common argparse + ESOptimizer loop for all four optimizers.

    Parameters
    ----------
    set_scenario : callable(str) -> None
        Callback that sets the caller's module-level _SCENARIO_NAME global
        (required for Windows multiprocessing pickling).
    """
    from master_sim.optimizer.es_engine import ESOptimizer, ESConfig
    from master_sim.optimizer.progress_ui import ProgressUI
    from master_sim.optimizer.run_log import get_scenario_csv_path

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--hours",    type=float, default=None)
    ap.add_argument("--iters",    type=int,   default=None)
    ap.add_argument("--workers",  type=int,   default=None)
    ap.add_argument("--scenario", type=str,   default=default_scenario)
    ap.add_argument("--patience", type=int,   default=200)
    ap.add_argument("--tol",      type=float, default=1e-4)
    ap.add_argument("--seed-gains", type=str, default=None,
                    help='e.g. "K1=0.5,K2=1.2"')
    args = ap.parse_args()

    set_scenario(args.scenario)

    csv_path = get_scenario_csv_path(args.scenario)
    seed = (parse_seed_gains(args.seed_gains) if args.seed_gains
            else default_seed(gains_key, param_mapping))

    ui = ProgressUI(f"{ui_label} Optimizer — {args.scenario}")
    cfg = ESConfig(patience=args.patience, tol=args.tol, n_workers=args.workers)
    opt = ESOptimizer(
        search_space=search_space,
        eval_fn=eval_fn,
        csv_path=csv_path,
        config=cfg,
        progress_fn=ui.update_from_progress,
        pause_fn=ui.wait_if_paused,
    )
    try:
        opt.run(hours=args.hours, max_iters=args.iters, seed_params=seed)
    finally:
        ui.finish()
