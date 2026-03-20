"""optimize_lqr.py — LQR Cost Weight Optimizer (Q/R search)

Searches over 4 LQR cost parameters via (1+8)-ES:
  Q_PITCH, Q_PITCH_RATE, Q_VEL, R  — LQR cost weights

Optimizes on a configurable scenario (default: 2_leg_height_gain_sched).
VelocityPI gains are held fixed during LQR optimization.

Run:
    python optimize_lqr.py --hours 0.033  # 2 minutes
    python optimize_lqr.py --hours 1 --workers 8
    python optimize_lqr.py --scenario 1_LQR_pitch_step --hours 1
    python optimize_lqr.py --seed-gains "Q_PITCH=0.063,Q_PITCH_RATE=0.00022,Q_VEL=1.1e-5,R=1.98"
"""
import argparse
import copy
import math
import multiprocessing
import os
import sys
import time
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from progress_window import ProgressWindow

from run_log import get_best_run, next_run_id, log_run, get_scenario_csv_path
import scenarios
from lqr_design import compute_gain_table
from sim_config import ROBOT

# ── Fixed VelocityPI gains (not searched during LQR optimization) ────────────
FIXED_VELOCITY_PI_PARAMS = {
    "KP_V":  0.299,
    "KI_V":  0.500,
}

# ── Search space: LQR Q/R cost weights (wide — never clamp) ──────────────────
PARAM_RANGES = {
    "Q_PITCH":      (1e-5,  1000.0),   # weight on pitch error
    "Q_PITCH_RATE": (1e-3,  100.0),    # weight on pitch rate — floor at 1e-3 to preserve damping (latency can reduce but not eliminate)
    "Q_VEL":        (1e-7,  10.0),     # weight on wheel velocity error (often tiny with latency)
    "R":            (1e-4,  1e5),      # weight on control effort
}

# Gains whose log-space value may evolve toward zero
ZERO_FLOOR = 1e-6   # values below this threshold are clamped to 0.0

# ── Default scenario ──────────────────────────────────────────────────────────
_DEFAULT_SCENARIO = "2_leg_height_gain_sched"

# ── Seed weights (used when CSV is empty — fresh start) ──────────────────────
SEED_WEIGHTS = {
    "Q_PITCH":      0.014168,   # LQR_Control_optimization baseline (fitness=0.018, rms=1.24°)
    "Q_PITCH_RATE": 0.033720,
    "Q_VEL":        0.000250,
    "R":            28.734,
}

# ── (1+λ)-ES hyper-parameters ────────────────────────────────────────────────
LAMBDA           = 8      # offspring per generation
SIGMA_LOG_INIT   = 1.00   # initial step size in log10 space (large exploration)
SIGMA_LOG_MIN    = 0.01
SIGMA_LOG_MAX    = 1.00
SUCCESS_TARGET   = 1.0 / 5.0
ADAPT_WINDOW     = 10
_DEFAULT_PATIENCE = 200   # generations without improvement → early stop
_DEFAULT_TOL      = 1e-4  # relative improvement threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _gains_to_str(g: dict) -> str:
    return "  ".join(f"{k}={v:.3g}" for k, v in g.items())


def _default_weights() -> dict:
    """Centre-of-range defaults in log10 space."""
    out = {}
    for k, (lo, hi) in PARAM_RANGES.items():
        out[k] = 10 ** ((math.log10(lo) + math.log10(hi)) / 2.0)
    return out


def _parse_seed_gains(seed_str: str) -> dict:
    """Parse 'Q_PITCH=0.063,Q_PITCH_RATE=0.00022,Q_VEL=1.1e-5,R=1.98' → dict."""
    result = {}
    for part in seed_str.split(","):
        k, _, v = part.strip().partition("=")
        result[k.strip()] = float(v.strip())
    return result


def _load_best_weights(active_scenario: str, seed_override: dict = None) -> tuple:
    """Seed from explicit override, best CSV result, or SEED_WEIGHTS when CSV is empty."""
    if seed_override:
        return dict(seed_override), float("inf")
    csv_path = get_scenario_csv_path(active_scenario)
    row = get_best_run(scenario=active_scenario, csv_path=csv_path)
    if row is None:
        return dict(SEED_WEIGHTS), float("inf")
    defaults = _default_weights()
    weights = {}
    for k in PARAM_RANGES:
        raw = row.get(k, "") or row.get(k.lower(), "")
        try:
            weights[k] = float(raw)
        except (ValueError, TypeError):
            weights[k] = defaults[k]
    fitness = float(row.get("fitness", float("inf")))
    return weights, fitness


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------
def _eval_worker(args):
    """Evaluate one Q/R candidate (LQR only) in a subprocess."""
    Q_pitch, Q_pitch_rate, Q_vel, R, label, run_id, csv_path, scenario_name = args

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Import here to get fresh module state
    from lqr_design import compute_gain_table
    from sim_config import ROBOT
    import scenarios

    # Configure: LQR controller ON, VelocityPI OFF (or set to fixed values)
    scenarios.USE_PD_CONTROLLER = False
    scenarios.USE_VELOCITY_PI   = False  # LQR-only optimization
    scenarios.VELOCITY_PI_KP = 0.0
    scenarios.VELOCITY_PI_KI = 0.0

    # Compute LQR gains for this Q/R candidate
    try:
        K_table = compute_gain_table(
            ROBOT,
            Q_diag=[Q_pitch, Q_pitch_rate, Q_vel],
            R_val=R
        )
        # Update global K_TABLE
        scenarios.LQR_K_TABLE = K_table
    except Exception as e:
        return dict(
            run_id=run_id, scenario=scenario_name, label=label,
            timestamp="",
            Q_PITCH=Q_pitch, Q_PITCH_RATE=Q_pitch_rate, Q_VEL=Q_vel, R=R,
            rms_pitch_deg=999.0, max_pitch_deg=999.0, wheel_travel_m=999.0,
            settle_time_s=999.0, survived_s=0.0,
            fitness=9999.0,
            status="FAIL", fail_reason=f"gain computation: {e}",
        )

    # Dispatch directly to scenario runner (avoids double-logging via scenarios.evaluate)
    _RUNNERS = {
        "1_LQR_pitch_step":        lambda: scenarios.run_1_LQR_pitch_step({}, duration=scenarios.SCENARIO_1_DURATION),
        "2_leg_height_gain_sched": lambda: scenarios.run_2_leg_height_gain_sched({}, duration=scenarios.SCENARIO_4_DURATION),
    }
    runner = _RUNNERS.get(scenario_name)
    if runner is None:
        raise ValueError(f"Unknown LQR scenario: '{scenario_name}'")
    metrics = runner()

    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = dict(
        run_id=run_id,
        scenario=scenario_name,
        label=label,
        timestamp=ts,
        Q_PITCH=round(Q_pitch, 6),
        Q_PITCH_RATE=round(Q_pitch_rate, 6),
        Q_VEL=round(Q_vel, 6),
        R=round(R, 6),
    )
    row.update(metrics)

    # Log to CSV
    from run_log import log_run
    log_run(row, csv_path)

    fit_str = f"{row.get('fitness', 9999.0):.3f}"
    print(f"[{run_id:5d}] {label:<28}  {row.get('status', 'FAIL'):<5}  "
          f"fit={fit_str}  "
          f"Q=[{Q_pitch:.3f},{Q_pitch_rate:.3f},{Q_vel:.4f}] R={R:.2f}")

    return row


# ---------------------------------------------------------------------------
# (1+λ)-ES
# ---------------------------------------------------------------------------
def run_evo(hours: float = None, max_iters: int = None,
            seed: int = None, n_workers: int = None,
            csv_path: str = None, win: "ProgressWindow | None" = None,
            active_scenario: str = None, seed_override: dict = None,
            patience: int = _DEFAULT_PATIENCE, tol: float = _DEFAULT_TOL):

    if active_scenario is None:
        active_scenario = _DEFAULT_SCENARIO
    if hours is None and max_iters is None:
        hours = 1.0
    if n_workers is None:
        n_workers = min(LAMBDA, multiprocessing.cpu_count())
    if seed is not None:
        np.random.seed(seed)
    if csv_path is None:
        csv_path = get_scenario_csv_path(active_scenario)

    parent, parent_fit = _load_best_weights(active_scenario, seed_override)
    if parent_fit == float("inf"):
        print("No prior results found — evaluating seed weights first...")
        row = _eval_worker(
            (parent['Q_PITCH'], parent['Q_PITCH_RATE'], parent['Q_VEL'], parent['R'],
             "evo_seed", next_run_id(csv_path), csv_path, active_scenario)
        )
        parent_fit = float(row.get("fitness", float("inf")))
        print(f"Seed fitness: {parent_fit:.3f}\n")
    else:
        print(f"Seeding from best result: fitness={parent_fit:.3f}")
        print(f"  {_gains_to_str(parent)}\n")

    sigmas = {k: SIGMA_LOG_INIT for k in PARAM_RANGES}
    success_window = deque(maxlen=ADAPT_WINDOW)

    t_start = time.perf_counter()
    deadline = (t_start + hours * 3600.0) if hours else None
    gen = 0
    n_evals = 0
    best_fit = parent_fit
    best_gen = 0

    # Convergence tracking
    gens_without_improvement = 0
    prev_best_for_patience = best_fit

    print("=" * 80)
    print(f"LQR Cost Weight Optimizer (1+{LAMBDA})-ES")
    print(f"  Scenario: {active_scenario}")
    print(f"  workers={n_workers}  |  params={list(PARAM_RANGES)}")
    print(f"  early-stop: patience={patience}  tol={tol:.1e}")
    if hours:
        print(f"  Duration: {hours:.1f} h")
    else:
        print(f"  Max generations: {max_iters}")
    print("=" * 80)

    with multiprocessing.Pool(processes=n_workers) as pool:
        while True:
            if deadline and time.perf_counter() >= deadline:
                break
            if max_iters is not None and gen >= max_iters:
                break
            if gens_without_improvement >= patience:
                print(f"\nEarly stop: no improvement in {patience} generations.")
                break

            # Generate λ offspring
            children = []
            for _ in range(LAMBDA):
                child = {}
                for k, (lo, hi) in PARAM_RANGES.items():
                    log_val = math.log10(max(1e-9, parent[k])) if parent[k] > 0 else math.log10(lo)
                    log_val += np.random.normal(0.0, sigmas[k])
                    log_val = _clamp(log_val, math.log10(lo), math.log10(hi))
                    val = 10.0 ** log_val
                    # Allow near-zero for params that can legitimately be zero
                    if k in ("Q_PITCH_RATE", "Q_VEL") and val < ZERO_FLOOR:
                        val = 0.0
                    child[k] = val
                children.append(child)

            # Reserve run IDs
            ids = list(range(next_run_id(csv_path), next_run_id(csv_path) + LAMBDA))
            labels = [f"evo_g{gen:06d}_c{i}" for i in range(LAMBDA)]

            args = [
                (c['Q_PITCH'], c['Q_PITCH_RATE'], c['Q_VEL'], c['R'],
                 lbl, rid, csv_path, active_scenario)
                for c, lbl, rid in zip(children, labels, ids)
            ]

            rows = pool.map(_eval_worker, args)
            n_evals += LAMBDA

            # Find best offspring
            gen_best_fit = float("inf")
            gen_best_p = None
            for child_p, row in zip(children, rows):
                fit = float(row.get("fitness", float("inf")))
                if row.get("status") == "PASS" and fit < gen_best_fit:
                    gen_best_fit = fit
                    gen_best_p = child_p

            improved = gen_best_p is not None and gen_best_fit < parent_fit
            success_window.append(1 if improved else 0)

            if improved:
                parent, parent_fit = gen_best_p, gen_best_fit
                if gen_best_fit < best_fit:
                    best_fit = gen_best_fit
                    best_gen = gen

            # Convergence check
            rel_improvement = (prev_best_for_patience - best_fit) / (prev_best_for_patience + 1e-12)
            if rel_improvement > tol:
                gens_without_improvement = 0
                prev_best_for_patience = best_fit
            else:
                gens_without_improvement += 1

            # Adapt step sizes
            if len(success_window) >= ADAPT_WINDOW:
                sr = sum(success_window) / len(success_window)
                for k in sigmas:
                    if sr > SUCCESS_TARGET:
                        sigmas[k] = min(sigmas[k] * 1.22, SIGMA_LOG_MAX)
                    else:
                        sigmas[k] = max(sigmas[k] / 1.22, SIGMA_LOG_MIN)

            gen += 1

            # ── Progress update (every generation for window; print every 5) ──
            elapsed_s = time.perf_counter() - t_start
            if deadline:
                total_s  = hours * 3600.0
                pct      = min(100.0, elapsed_s / total_s * 100.0)
                remain_s = max(0.0, total_s - elapsed_s)
            else:
                pct      = min(100.0, gen / max_iters * 100.0)
                remain_s = 0.0

            sr = sum(success_window) / len(success_window) if success_window else 0.0
            gains_str = _gains_to_str(parent)

            if win is not None:
                win.update(
                    pct=pct, elapsed_s=elapsed_s, remaining_s=remain_s,
                    n_evals=n_evals, gen=gen,
                    best_fit=best_fit,
                    best_gains=gains_str,
                    success_rate=sr,
                    status="running",
                )

            if gen % 5 == 0:
                elapsed_min = elapsed_s / 60.0
                if deadline:
                    remain_str   = f"{int(remain_s // 60):02d}:{int(remain_s % 60):02d}"
                    progress_str = f"{pct:5.1f}%  remaining={remain_str}"
                else:
                    progress_str = f"{pct:5.1f}%  gen {gen}/{max_iters}"
                filled = int(pct / 100 * 40)
                bar = "[" + "#" * filled + "-" * (40 - filled) + "]"
                print(f"\n{bar} {progress_str}  evals={n_evals}  elapsed={elapsed_min:.1f}min"
                      f"  stagnant={gens_without_improvement}/{patience}")
                print(f"  best={best_fit:.4f}  parent={parent_fit:.4f}  best_at_gen={best_gen}")
                print(f"  gains: {gains_str}")

    elapsed_min = (time.perf_counter() - t_start) / 60.0
    print("\n" + "=" * 80)
    print(f"Optimization complete: {gen} generations, {n_evals} evals, {elapsed_min:.1f} min")
    print(f"All-time best: fitness={best_fit:.3f}  (found gen {best_gen})")
    print(f"Best weights: {_gains_to_str(parent)}")
    print(f"\nTo replay best result:")
    print(f"  python replay.py --top 1")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="LQR cost weight optimizer (Q/R search via evolutionary strategy).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_lqr.py --hours 2
  python optimize_lqr.py --iters 100 --workers 8
  python optimize_lqr.py --scenario 1_LQR_pitch_step --hours 1
  python optimize_lqr.py --seed-gains "Q_PITCH=0.063,Q_PITCH_RATE=0.00022,Q_VEL=1.1e-5,R=1.98"
""")
    ap.add_argument("--hours",   type=float, default=None,
                    help="Wall-clock hours to run (default: 1h if --iters not set)")
    ap.add_argument("--iters",   type=int,   default=None,
                    help="Maximum number of generations")
    ap.add_argument("--workers", type=int,   default=None,
                    help="Parallel worker processes (default: cpu_count)")
    ap.add_argument("--seed",    type=int,   default=None,
                    help="RNG seed for reproducibility")
    ap.add_argument("--scenario", type=str,  default=_DEFAULT_SCENARIO,
                    help=f"Scenario to optimize (default: {_DEFAULT_SCENARIO})")
    ap.add_argument("--seed-gains", type=str, default=None,
                    help='Explicit seed gains, e.g. "Q_PITCH=0.063,Q_PITCH_RATE=0.00022,Q_VEL=1.1e-5,R=1.98"')
    ap.add_argument("--patience", type=int,  default=_DEFAULT_PATIENCE,
                    help=f"Generations without improvement before early stop (default: {_DEFAULT_PATIENCE})")
    ap.add_argument("--tol",     type=float, default=_DEFAULT_TOL,
                    help=f"Relative improvement threshold for convergence (default: {_DEFAULT_TOL})")
    args = ap.parse_args()

    seed_override = _parse_seed_gains(args.seed_gains) if args.seed_gains else None

    win = ProgressWindow(f"LQR Optimizer — {args.scenario}")
    try:
        run_evo(hours=args.hours, max_iters=args.iters,
                seed=args.seed, n_workers=args.workers, win=win,
                active_scenario=args.scenario, seed_override=seed_override,
                patience=args.patience, tol=args.tol)
    finally:
        win.finish()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
