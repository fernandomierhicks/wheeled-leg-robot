"""common.py — Shared helpers for all optimizer entry points.

Eliminates boilerplate that was duplicated across optimize_lqr / _vel_pi /
_yaw_pi / _suspension: eval function body, seed parsing, argparse + run loop.

The eval helper uses lazy imports so that each multiprocessing worker only
loads MuJoCo / scenario code on first use (required for Windows pickling).
"""
import argparse
from dataclasses import replace


# ── Eval helper (called inside each module-level eval function) ──────────────

class EvalWithGains:
    """Picklable callable for multiprocessing — replaces module-global hack."""

    def __init__(self, scenario_name: str, gains_key: str, param_mapping: dict):
        self.scenario_name = scenario_name
        self.gains_key = gains_key
        self.param_mapping = param_mapping

    def __call__(self, candidate: dict) -> dict:
        return eval_with_gains(candidate, self.scenario_name,
                               self.gains_key, self.param_mapping)


class EvalWithAllGains:
    """Picklable callable for the integrated optimizer."""

    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name

    def __call__(self, candidate: dict) -> dict:
        return eval_with_all_gains(candidate, self.scenario_name)


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
    from master_sim_jump.defaults import DEFAULT_PARAMS
    from master_sim_jump.scenarios import evaluate

    p = DEFAULT_PARAMS
    current_sub = getattr(p.gains, gains_key)
    new_sub = replace(current_sub,
                      **{param_mapping[k]: candidate[k] for k in candidate})
    new_gains = replace(p.gains, **{gains_key: new_sub})
    params = replace(p, gains=new_gains)

    metrics = evaluate(params, scenario_name)
    metrics["scenario"] = scenario_name
    return metrics


# ── All-gains eval for integrated optimizer ──────────────────────────────────

# Maps search-space keys → (gains_key, field_name)
_ALL_GAINS_MAP = {
    "Q_PITCH":      ("lqr", "Q_pitch"),
    "Q_PITCH_RATE": ("lqr", "Q_pitch_rate"),
    "Q_VEL":        ("lqr", "Q_vel"),
    "R":            ("lqr", "R"),
    "KP_V":         ("velocity_pi", "Kp"),
    "KI_V":         ("velocity_pi", "Ki"),
    "KFF_V":        ("velocity_pi", "Kff"),
    "KP_YAW":       ("yaw_pi", "Kp"),
    "KI_YAW":       ("yaw_pi", "Ki"),
    "LEG_K_S":      ("suspension", "K_s"),
    "LEG_B_S":      ("suspension", "B_s"),
    "LEG_K_ROLL":   ("suspension", "K_roll"),
    "LEG_D_ROLL":   ("suspension", "D_roll"),
    "JUMP_CROUCH_TIME":  ("jump", "crouch_time"),
    "JUMP_MAX_TORQUE":   ("jump", "max_torque"),
    "JUMP_RAMP_UP_S":    ("jump", "ramp_up_s"),
    "JUMP_RAMP_DOWN_RAD":("jump", "ramp_down_rad"),
}


def eval_with_all_gains(candidate: dict, scenario_name: str) -> dict:
    """Evaluate one candidate by replacing ALL 4 gain groups at once."""
    from master_sim_jump.defaults import DEFAULT_PARAMS
    from master_sim_jump.scenarios import evaluate

    p = DEFAULT_PARAMS

    # Group candidate keys by gains_key
    updates = {}  # gains_key → {field: value}
    for cand_key, value in candidate.items():
        gains_key, field_name = _ALL_GAINS_MAP[cand_key]
        updates.setdefault(gains_key, {})[field_name] = value

    # Build new GainSet
    new_gains = p.gains
    for gains_key, fields in updates.items():
        current_sub = getattr(new_gains, gains_key)
        new_sub = replace(current_sub, **fields)
        new_gains = replace(new_gains, **{gains_key: new_sub})

    params = replace(p, gains=new_gains)
    metrics = evaluate(params, scenario_name)
    metrics["scenario"] = scenario_name
    return metrics


def default_seed_all() -> dict:
    """Extract all 12 default gains as a seed dict for the integrated optimizer."""
    from master_sim_jump.defaults import DEFAULT_PARAMS
    p = DEFAULT_PARAMS
    seed = {}
    for cand_key, (gains_key, field_name) in _ALL_GAINS_MAP.items():
        seed[cand_key] = getattr(getattr(p.gains, gains_key), field_name)
    return seed


# ── Seed helpers ─────────────────────────────────────────────────────────────

def parse_seed_gains(raw: str) -> dict:
    """Parse 'K1=v1,K2=v2,...' into {str: float}."""
    return {k.strip(): float(v.strip())
            for part in raw.split(",")
            for k, _, v in [part.partition("=")]}


def default_seed(gains_key: str, param_mapping: dict) -> dict:
    """Extract current default gains as a seed dict."""
    from master_sim_jump.defaults import DEFAULT_PARAMS
    sub = getattr(DEFAULT_PARAMS.gains, gains_key)
    return {cand_key: getattr(sub, field_name)
            for cand_key, field_name in param_mapping.items()}


# ── Shared main() ────────────────────────────────────────────────────────────

def optimizer_main(*, description: str, default_scenario: str,
                   search_space, eval_fn=None, gains_key: str,
                   param_mapping: dict, ui_label: str,
                   set_scenario=None) -> None:
    """Common argparse + ESOptimizer loop for all four optimizers.

    The eval function is built internally as a picklable EvalWithGains
    instance so that multiprocessing workers on Windows (spawn) use the
    correct scenario name.  Legacy eval_fn / set_scenario args are ignored.
    """
    from master_sim_jump.optimizer.es_engine import ESOptimizer, ESConfig
    from master_sim_jump.optimizer.progress_ui import ProgressUI
    from master_sim_jump.optimizer.run_log import get_scenario_csv_path

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--hours",    type=float, default=None)
    ap.add_argument("--iters",    type=int,   default=None)
    ap.add_argument("--workers",  type=int,   default=None)
    ap.add_argument("--scenario", type=str,   default=default_scenario)
    ap.add_argument("--patience", type=int,   default=200)
    ap.add_argument("--tol",      type=float, default=1e-4)
    ap.add_argument("--seed-gains", type=str, default=None,
                    help='e.g. "K1=0.5,K2=1.2"')
    ap.add_argument("--no-baseline", action="store_true",
                    help="Skip auto-baselining after optimizer finishes")
    args = ap.parse_args()

    # Picklable eval — scenario name baked in, safe for Windows spawn
    eval_fn = EvalWithGains(args.scenario, gains_key, param_mapping)

    csv_path = get_scenario_csv_path(args.scenario)
    seed = (parse_seed_gains(args.seed_gains) if args.seed_gains
            else default_seed(gains_key, param_mapping))

    all_defaults = default_seed_all()
    active_names = set(seed.keys())
    ui = ProgressUI(f"{ui_label} Optimizer — {args.scenario}",
                    all_defaults=all_defaults, active_names=active_names)
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
