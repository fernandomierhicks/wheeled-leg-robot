"""optimize_vel_pi.py — VelocityPI Gain Optimizer (KP_V / KI_V search)

Searches over 2 VelocityPI parameters via (1+8)-ES:
  KP_V  [rad/(m/s)]  proportional gain
  KI_V  [rad/m]      integral gain

LQR Q/R weights are held fixed at the baselined values in sim_config.py.
Scenario: combined_PI — weighted sum of two sub-scenarios (50/50):
  3_VEL_PI_disturbance : position hold under ±1N impulse kicks (normalised [m/s])
  4_VEL_PI_staircase   : setpoint tracking 0→0.3→0.6→−0.3→0 m/s [m/s RMS]
Fitness: lower is better; fall adds large penalty to both sub-fitnesses.

Run:
    python optimize_vel_pi.py --hours 0.083   # 5 minutes
    python optimize_vel_pi.py --hours 1 --workers 8
"""
import argparse
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
from sim_config import (
    ROBOT, LQR_Q_PITCH, LQR_Q_PITCH_RATE, LQR_Q_VEL, LQR_R,
    VELOCITY_PI_KP, VELOCITY_PI_KI,
)

ACTIVE_SCENARIO = "5_VEL_PI_leg_cycling"

# ── Search space: VelocityPI gains ──────────────────────────────────────────
PARAM_RANGES = {
    "KP_V": (0.01, 5.0),    # [rad/(m/s)]
    "KI_V": (0.01, 5.0),    # [rad/m]
}

# ── Seed from current sim_config values ─────────────────────────────────────
SEED_WEIGHTS = {
    "KP_V": VELOCITY_PI_KP,
    "KI_V": VELOCITY_PI_KI,
}

# ── (1+λ)-ES hyper-parameters ────────────────────────────────────────────────
LAMBDA         = 8
SIGMA_LOG_INIT = 0.50    # moderate initial exploration in log space
SIGMA_LOG_MIN  = 0.01
SIGMA_LOG_MAX  = 1.00
SUCCESS_TARGET = 1.0 / 5.0
ADAPT_WINDOW   = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _gains_to_str(g: dict) -> str:
    return "  ".join(f"{k}={v:.4g}" for k, v in g.items())


def _load_best_weights(csv_path: str) -> tuple:
    row = get_best_run(scenario=ACTIVE_SCENARIO, csv_path=csv_path)
    if row is None:
        return dict(SEED_WEIGHTS), float("inf")
    weights = {}
    for k in PARAM_RANGES:
        try:
            weights[k] = float(row[k])
        except (ValueError, KeyError, TypeError):
            weights[k] = SEED_WEIGHTS[k]
    return weights, float(row.get("fitness", float("inf")))


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------
def _eval_worker(args):
    kp_v, ki_v, label, run_id, csv_path = args

    import os, sys, datetime
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from lqr_design import compute_gain_table
    from sim_config import ROBOT
    import scenarios

    # Fix LQR gains; search only over VelocityPI
    scenarios.USE_PD_CONTROLLER = False
    scenarios.USE_VELOCITY_PI   = True
    scenarios.VELOCITY_PI_KP    = kp_v
    scenarios.VELOCITY_PI_KI    = ki_v
    scenarios.LQR_K_TABLE = compute_gain_table(
        ROBOT,
        Q_diag=[LQR_Q_PITCH, LQR_Q_PITCH_RATE, LQR_Q_VEL],
        R_val=LQR_R,
    )

    # Call scenario runner directly (not evaluate()) to avoid double-logging
    metrics = scenarios.run_5_VEL_PI_leg_cycling({})

    row = dict(
        run_id    = run_id,
        scenario  = ACTIVE_SCENARIO,
        label     = label,
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        KP_V      = round(kp_v, 6),
        KI_V      = round(ki_v, 6),
    )
    row.update(metrics)

    from run_log import log_run
    log_run(row, csv_path)

    fit = row.get("fitness", 9999.0)
    print(f"[{run_id:5d}] {label:<28}  {row.get('status','FAIL'):<5}  "
          f"fit={fit:.4f}  "
          f"vel_rms={row.get('vel_track_rms_ms','?'):.3f}m/s  "
          f"pitch={row.get('rms_pitch_deg','?'):.2f}°  "
          f"KP_V={kp_v:.4g}  KI_V={ki_v:.4g}")
    return row


# ---------------------------------------------------------------------------
# (1+λ)-ES
# ---------------------------------------------------------------------------
def run_evo(hours: float = None, max_iters: int = None,
            seed: int = None, n_workers: int = None,
            csv_path: str = None, win: "ProgressWindow | None" = None):

    if hours is None and max_iters is None:
        hours = 1.0
    if n_workers is None:
        n_workers = min(LAMBDA, multiprocessing.cpu_count())
    if seed is not None:
        np.random.seed(seed)
    if csv_path is None:
        csv_path = get_scenario_csv_path(ACTIVE_SCENARIO)

    parent, parent_fit = _load_best_weights(csv_path)
    if parent_fit == float("inf"):
        print("No prior results — evaluating seed weights first...")
        row = _eval_worker((parent["KP_V"], parent["KI_V"],
                            "evo_seed", next_run_id(csv_path), csv_path))
        parent_fit = float(row.get("fitness", float("inf")))
        print(f"Seed fitness: {parent_fit:.4f}\n")
    else:
        print(f"Seeding from best CSV: fitness={parent_fit:.4f}")
        print(f"  {_gains_to_str(parent)}\n")

    sigmas         = {k: SIGMA_LOG_INIT for k in PARAM_RANGES}
    success_window = deque(maxlen=ADAPT_WINDOW)

    t_start  = time.perf_counter()
    deadline = (t_start + hours * 3600.0) if hours else None
    gen      = 0
    n_evals  = 0
    best_fit = parent_fit
    best_gen = 0

    print("=" * 70)
    print(f"VelocityPI Optimizer (1+{LAMBDA})-ES  |  Scenario: {ACTIVE_SCENARIO}")
    print(f"  workers={n_workers}  params={list(PARAM_RANGES)}")
    print(f"  Fixed LQR: Q=[{LQR_Q_PITCH:.4g},{LQR_Q_PITCH_RATE:.4g},{LQR_Q_VEL:.4g}]  R={LQR_R:.4g}")
    if hours:
        print(f"  Duration: {hours:.2f} h  ({hours*60:.1f} min)")
    else:
        print(f"  Max generations: {max_iters}")
    print("=" * 70)

    with multiprocessing.Pool(processes=n_workers) as pool:
        while True:
            if deadline and time.perf_counter() >= deadline:
                break
            if max_iters is not None and gen >= max_iters:
                break

            children = []
            for _ in range(LAMBDA):
                child = {}
                for k, (lo, hi) in PARAM_RANGES.items():
                    lv = math.log10(max(1e-9, parent[k]))
                    lv += np.random.normal(0.0, sigmas[k])
                    lv  = _clamp(lv, math.log10(lo), math.log10(hi))
                    child[k] = 10.0 ** lv
                children.append(child)

            base_id = next_run_id(csv_path)
            ids     = list(range(base_id, base_id + LAMBDA))
            labels  = [f"evo_g{gen:06d}_c{i}" for i in range(LAMBDA)]
            args    = [(c["KP_V"], c["KI_V"], lbl, rid, csv_path)
                       for c, lbl, rid in zip(children, labels, ids)]

            rows    = pool.map(_eval_worker, args)
            n_evals += LAMBDA

            gen_best_fit = float("inf")
            gen_best_p   = None
            for child_p, row in zip(children, rows):
                fit = float(row.get("fitness", float("inf")))
                if row.get("status") == "PASS" and fit < gen_best_fit:
                    gen_best_fit = fit
                    gen_best_p   = child_p

            improved = gen_best_p is not None and gen_best_fit < parent_fit
            success_window.append(1 if improved else 0)

            if improved:
                parent, parent_fit = gen_best_p, gen_best_fit
                if gen_best_fit < best_fit:
                    best_fit = gen_best_fit
                    best_gen = gen

            if len(success_window) >= ADAPT_WINDOW:
                sr = sum(success_window) / len(success_window)
                for k in sigmas:
                    if sr > SUCCESS_TARGET:
                        sigmas[k] = min(sigmas[k] * 1.22, SIGMA_LOG_MAX)
                    else:
                        sigmas[k] = max(sigmas[k] / 1.22, SIGMA_LOG_MIN)

            gen += 1

            elapsed_s = time.perf_counter() - t_start
            if deadline:
                total_s  = hours * 3600.0
                pct      = min(100.0, elapsed_s / total_s * 100.0)
                remain_s = max(0.0, total_s - elapsed_s)
            else:
                pct      = min(100.0, gen / max_iters * 100.0)
                remain_s = 0.0

            sr         = sum(success_window) / len(success_window) if success_window else 0.0
            gains_str  = _gains_to_str(parent)

            if win is not None:
                win.update(pct=pct, elapsed_s=elapsed_s, remaining_s=remain_s,
                           n_evals=n_evals, gen=gen,
                           best_fit=best_fit, best_gains=gains_str,
                           success_rate=sr, status="running")

            if gen % 5 == 0:
                elapsed_min = elapsed_s / 60.0
                if deadline:
                    remain_str   = f"{int(remain_s // 60):02d}:{int(remain_s % 60):02d}"
                    progress_str = f"{pct:5.1f}%  remaining={remain_str}"
                else:
                    progress_str = f"{pct:5.1f}%  gen {gen}/{max_iters}"
                filled = int(pct / 100 * 40)
                bar = "[" + "#" * filled + "-" * (40 - filled) + "]"
                print(f"\n{bar} {progress_str}  evals={n_evals}  elapsed={elapsed_min:.1f}min")
                print(f"  best={best_fit:.4f}  parent={parent_fit:.4f}  best_at_gen={best_gen}")
                print(f"  gains: {gains_str}")

    elapsed_min = (time.perf_counter() - t_start) / 60.0
    print("\n" + "=" * 70)
    print(f"Optimization complete: {gen} gens, {n_evals} evals, {elapsed_min:.1f} min")
    print(f"All-time best: fitness={best_fit:.4f}  (gen {best_gen})")
    print(f"Best VelocityPI: {_gains_to_str(parent)}")
    print(f"\nTo replay: python replay.py --top 1 --csv {get_scenario_csv_path(ACTIVE_SCENARIO)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="VelocityPI gain optimizer (KP_V / KI_V via evolutionary strategy).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_vel_pi.py --hours 0.083   # 5 minutes
  python optimize_vel_pi.py --hours 1
  python optimize_vel_pi.py --iters 100 --workers 8
""")
    ap.add_argument("--hours",   type=float, default=None)
    ap.add_argument("--iters",   type=int,   default=None)
    ap.add_argument("--workers", type=int,   default=None)
    ap.add_argument("--seed",    type=int,   default=None)
    args = ap.parse_args()

    win = ProgressWindow(f"VelocityPI Optimizer — {ACTIVE_SCENARIO}")
    try:
        run_evo(hours=args.hours, max_iters=args.iters,
                seed=args.seed, n_workers=args.workers, win=win)
    finally:
        win.finish()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
