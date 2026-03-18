"""optimize_lqr.py — LQR + VelocityPI optimizer (Q/R/Kp_v/Ki_v search)

Searches over 6 parameters via (1+8)-ES:
  Q_PITCH, Q_PITCH_RATE, Q_VEL, R  — LQR cost weights
  KP_V, KI_V                        — VelocityPI lean-angle gains

Combined fitness: balance(0.10) + disturbance(0.35) + drive_slow(0.20)
                  + drive_medium(0.20) + obstacle(0.15)

Run:
    python optimize_lqr.py --iters 100 --workers 4
    python optimize_lqr.py --hours 1.5 --workers 8
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

from run_log import CSV_PATH, get_best_run, next_run_id, log_run
import scenarios
from lqr_design import compute_gain_table
from sim_config import ROBOT

# ── Search space ──────────────────────────────────────────────────────────────
# All parameters searched in log10 space.
# LQR Q/R: weight balance quality (existing Phase 1 tuning)
# KP_V/KI_V: VelocityPI lean-angle gains (new Phase 2 drive scenarios)
PARAM_RANGES = {
    "Q_PITCH":       (0.01,    1.0),     # pitch error weight
    "Q_PITCH_RATE":  (0.001,   1.0),     # pitch rate damping
    "Q_VEL":         (0.0001,  0.1),     # wheel velocity weight (may increase for drive)
    "R":             (0.1,     50.0),    # control effort
    "KP_V":          (0.01,    0.5),     # VelocityPI Kp [rad/(m/s)] — lean angle per vel error
    "KI_V":          (0.001,   0.2),     # VelocityPI Ki [rad/m]
}

# ── (1+λ)-ES hyper-parameters ────────────────────────────────────────────────
LAMBDA           = 8      # offspring per generation
SIGMA_LOG_INIT   = 0.30   # initial step size in log10 space
SIGMA_LOG_MIN    = 0.01
SIGMA_LOG_MAX    = 1.00
SUCCESS_TARGET   = 1.0 / 5.0
ADAPT_WINDOW     = 10


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


def _load_best_weights() -> tuple:
    """Seed from best CSV result, or use centre-of-range defaults.

    Falls back gracefully when CSV lacks new KP_V/KI_V columns
    (e.g., if seeding from a Phase 1 results file).
    """
    from sim_config import VELOCITY_PI_KP as _kp_default, VELOCITY_PI_KI as _ki_default
    row = get_best_run(scenario="lqr_combined")
    if row is None:
        return _default_weights(), float("inf")
    defaults = _default_weights()
    weights = {}
    for k in PARAM_RANGES:
        raw = row.get(k, "") or row.get(k.lower(), "")
        try:
            weights[k] = float(raw)
        except (ValueError, TypeError):
            weights[k] = defaults[k]
    # Use sim_config defaults for KP_V/KI_V if missing or zero
    if weights.get("KP_V", 0.0) <= 0.0:
        weights["KP_V"] = _kp_default
    if weights.get("KI_V", 0.0) <= 0.0:
        weights["KI_V"] = _ki_default
    fitness = float(row.get("fitness", float("inf")))
    return weights, fitness


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------
def _eval_worker(args):
    """Evaluate one Q/R/Kp_v/Ki_v candidate in a subprocess."""
    Q_pitch, Q_pitch_rate, Q_vel, R, Kp_v, Ki_v, label, run_id, csv_path = args

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Import here to get fresh module state
    from lqr_design import compute_gain_table
    from sim_config import ROBOT
    import scenarios

    # Temporarily switch to LQR controller + set velocity PI gains
    scenarios.USE_PD_CONTROLLER = False
    scenarios.VELOCITY_PI_KP = Kp_v
    scenarios.VELOCITY_PI_KI = Ki_v

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
            run_id=run_id, scenario="lqr_combined", label=label,
            timestamp="",
            Q_PITCH=Q_pitch, Q_PITCH_RATE=Q_pitch_rate, Q_VEL=Q_vel, R=R,
            KP_V=Kp_v, KI_V=Ki_v,
            rms_pitch_deg=999.0, max_pitch_deg=999.0, wheel_travel_m=999.0,
            settle_time_s=999.0, survived_s=0.0,
            fitness_balance=9999.0, fitness_disturbance=9999.0,
            fitness_drive_slow=9999.0, fitness_drive_med=9999.0, fitness_obstacle=9999.0,
            fitness=9999.0,
            status="FAIL", fail_reason=f"gain computation: {e}",
        )

    # Dummy gains dict (not used by LQR controller)
    gains = dict(KP=0, KD=0, KP_pos=0, KP_vel=0)

    # Run combined scenario (balance + disturbance + drive_slow + drive_med + obstacle)
    metrics = scenarios.run_combined_scenario(gains)

    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = dict(
        run_id=run_id,
        scenario="lqr_combined",
        label=label,
        timestamp=ts,
        Q_PITCH=round(Q_pitch, 4),
        Q_PITCH_RATE=round(Q_pitch_rate, 4),
        Q_VEL=round(Q_vel, 4),
        R=round(R, 4),
        KP_V=round(Kp_v, 4),
        KI_V=round(Ki_v, 4),
    )
    row.update(metrics)

    # Log to CSV
    from run_log import log_run
    log_run(row, csv_path)

    fit_str = f"{row.get('fitness', 9999.0):.3f}"
    f_slow = row.get('fitness_drive_slow', '?')
    f_med  = row.get('fitness_drive_med', '?')
    f_step = row.get('fitness_obstacle', '?')
    print(f"[{run_id:5d}] {label:<28}  {row.get('status', 'FAIL'):<5}  "
          f"fit={fit_str}  "
          f"Q=[{Q_pitch:.3f},{Q_pitch_rate:.3f},{Q_vel:.4f}] R={R:.2f}  "
          f"Kp_v={Kp_v:.3f} Ki_v={Ki_v:.3f}  "
          f"drv_s={f_slow} drv_m={f_med} obs={f_step}")

    return row


# ---------------------------------------------------------------------------
# (1+λ)-ES
# ---------------------------------------------------------------------------
def run_evo(hours: float = None, max_iters: int = None,
            seed: int = None, n_workers: int = None,
            csv_path: str = CSV_PATH):

    if hours is None and max_iters is None:
        hours = 1.0
    if n_workers is None:
        n_workers = min(LAMBDA, multiprocessing.cpu_count())
    if seed is not None:
        np.random.seed(seed)

    parent, parent_fit = _load_best_weights()
    if parent_fit == float("inf"):
        print("No prior results found — evaluating default weights first...")
        row = _eval_worker(
            (parent['Q_PITCH'], parent['Q_PITCH_RATE'], parent['Q_VEL'],
             parent['R'], parent['KP_V'], parent['KI_V'],
             "evo_seed", next_run_id(csv_path), csv_path)
        )
        parent_fit = float(row.get("fitness", float("inf")))
        print(f"Seed fitness: {parent_fit:.3f}\n")
    else:
        print(f"Seeding from best CSV run: fitness={parent_fit:.3f}")
        print(f"  {_gains_to_str(parent)}\n")

    sigmas = {k: SIGMA_LOG_INIT for k in PARAM_RANGES}
    success_window = deque(maxlen=ADAPT_WINDOW)

    t_start = time.perf_counter()
    deadline = (t_start + hours * 3600.0) if hours else None
    gen = 0
    n_evals = 0
    best_fit = parent_fit
    best_gen = 0

    print("=" * 80)
    print(f"LQR Cost Weight Optimizer (1+{LAMBDA})-ES")
    print(f"  workers={n_workers}  |  params={list(PARAM_RANGES)}")
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

            # Generate λ offspring
            children = []
            for _ in range(LAMBDA):
                child = {}
                for k, (lo, hi) in PARAM_RANGES.items():
                    log_val = math.log10(max(1e-9, parent[k]))
                    log_val += np.random.normal(0.0, sigmas[k])
                    log_val = _clamp(log_val, math.log10(lo), math.log10(hi))
                    child[k] = 10.0 ** log_val
                children.append(child)

            # Reserve run IDs
            ids = list(range(next_run_id(csv_path), next_run_id(csv_path) + LAMBDA))
            labels = [f"evo_g{gen:06d}_c{i}" for i in range(LAMBDA)]

            args = [
                (c['Q_PITCH'], c['Q_PITCH_RATE'], c['Q_VEL'], c['R'],
                 c['KP_V'], c['KI_V'],
                 lbl, rid, csv_path)
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

            # Adapt step sizes
            if len(success_window) >= ADAPT_WINDOW:
                sr = sum(success_window) / len(success_window)
                for k in sigmas:
                    if sr > SUCCESS_TARGET:
                        sigmas[k] = min(sigmas[k] * 1.22, SIGMA_LOG_MAX)
                    else:
                        sigmas[k] = max(sigmas[k] / 1.22, SIGMA_LOG_MIN)

            gen += 1

            if gen % 5 == 0:
                elapsed = (time.perf_counter() - t_start) / 60.0
                sigma_str = "  ".join(f"{k}={v:.2f}" for k, v in sigmas.items())
                print(f"\n[gen {gen:5d}]  best={best_fit:.3f}  parent={parent_fit:.3f}  "
                      f"evals={n_evals}  elapsed={elapsed:.1f}min")
                print(f"  parent: {_gains_to_str(parent)}")
                print(f"  sigma:  {sigma_str}")

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
  python optimize_lqr.py --iters 50 --seed 42
""")
    ap.add_argument("--hours", type=float, default=None,
                    help="Wall-clock hours to run (default: 1h if --iters not set)")
    ap.add_argument("--iters", type=int, default=None,
                    help="Maximum number of generations")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel worker processes (default: cpu_count)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for reproducibility")
    args = ap.parse_args()

    run_evo(hours=args.hours, max_iters=args.iters,
            seed=args.seed, n_workers=args.workers)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
