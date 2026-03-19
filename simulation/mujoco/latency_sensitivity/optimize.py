"""optimize.py — Evolutionary gain optimizer for LQR_Control_optimization.

Uses (1+λ)-ES (Evolution Strategy) with adaptive step sizes to search over
the PD balance controller gain space.  Each candidate is evaluated by running
one or more simulation scenarios headlessly in parallel.

Search space: KP, KD, KP_pos, KP_vel  →  4-dimensional
Fitness: W_RMS * rms_pitch_deg + W_TRAVEL * wheel_travel_m  (+ fall penalty)

Run:
    python optimize.py --hours 1
    python optimize.py --iters 200 --workers 4 --seed 42
"""
import argparse
import copy
import multiprocessing
import os
import sys
import time
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from run_log import CSV_PATH, get_best_run, next_run_id, log_run
from scenarios import evaluate

# ── Search space ─────────────────────────────────────────────────────────────
# PD balance controller gains.  Optimiser works in log space (mutates log10).
PARAM_RANGES = {
    "KP":     (5.0,   500.0),   # pitch proportional gain [N·m/rad]
    "KD":     (0.1,    50.0),   # pitch derivative gain   [N·m·s/rad]
    "KP_pos": (0.01,   5.0),    # wheel position feedback [rad/m]
    "KP_vel": (0.01,   5.0),    # wheel velocity feedback [rad/(m/s)]
}

# ── (1+λ)-ES hyper-parameters ────────────────────────────────────────────────
LAMBDA           = 8      # offspring per generation
SIGMA_LOG_INIT   = 0.30   # initial step size in log10 space
SIGMA_LOG_MIN    = 0.01   # minimum step size
SIGMA_LOG_MAX    = 1.00   # maximum step size
SUCCESS_TARGET   = 1.0 / 5.0   # 1/5 success rule
ADAPT_WINDOW     = 10          # rolling window for success rate
ADAPT_UP         = 1.22
ADAPT_DOWN       = 1.0 / 1.22


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _gains_to_str(g: dict) -> str:
    return "  ".join(f"{k}={v:.3g}" for k, v in g.items())


def _log10_centre(lo, hi):
    return (math.log10(lo) + math.log10(hi)) / 2.0 if lo > 0 else 0.0


def _default_gains() -> dict:
    """Centre-of-range defaults in each parameter."""
    import math
    out = {}
    for k, (lo, hi) in PARAM_RANGES.items():
        if lo > 0:
            out[k] = 10 ** ((math.log10(lo) + math.log10(hi)) / 2.0)
        else:
            out[k] = (lo + hi) / 2.0
    return out


def _load_best_gains() -> tuple:
    """Seed from best CSV result, or use centre-of-range defaults."""
    row = get_best_run(scenario="balance")
    if row is None:
        return _default_gains(), float("inf")
    gains = {k: float(row.get(k, 0)) for k in PARAM_RANGES}
    fitness = float(row.get("fitness", float("inf")))
    return gains, fitness


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------
def _eval_worker(args):
    """Evaluate one gains candidate in a subprocess."""
    gains, label, run_id, csv_path = args
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scenarios import evaluate as _eval
    row = _eval(gains, scenario="balance", label=label,
                run_id=run_id, csv_path=csv_path)
    return row


def _next_ids(n: int, csv_path: str) -> list[int]:
    """Reserve n consecutive run IDs atomically (single-process safe)."""
    base = next_run_id(csv_path)
    return list(range(base, base + n))


# ---------------------------------------------------------------------------
# (1+λ)-ES
# ---------------------------------------------------------------------------
def run_evo(hours: float = None, max_iters: int = None,
            seed: int = None, n_workers: int = None,
            csv_path: str = CSV_PATH):
    import math

    if hours is None and max_iters is None:
        hours = 1.0
    if n_workers is None:
        n_workers = min(LAMBDA, multiprocessing.cpu_count())
    if seed is not None:
        np.random.seed(seed)

    parent, parent_fit = _load_best_gains()
    if parent_fit == float("inf"):
        print("No prior results found — evaluating default gains first...")
        row = _eval_worker((parent, "evo_seed", next_run_id(csv_path), csv_path))
        parent_fit = float(row.get("fitness", float("inf")))
        print(f"Seed fitness: {parent_fit:.3f}")
    else:
        print(f"Seeding from best CSV run: fitness={parent_fit:.3f}")
        print(f"  {_gains_to_str(parent)}")

    # Step sizes in log10 space for each param
    sigmas = {k: SIGMA_LOG_INIT for k in PARAM_RANGES}
    success_window = deque(maxlen=ADAPT_WINDOW)

    t_start   = time.perf_counter()
    deadline  = (t_start + hours * 3600.0) if hours else None
    gen       = 0
    n_evals   = 0
    best_fit  = parent_fit
    best_gen  = 0

    print("=" * 70)
    print(f"(1+{LAMBDA})-ES  |  workers={n_workers}  |  params={list(PARAM_RANGES)}")
    if hours:
        print(f"Duration: {hours:.1f} h")
    else:
        print(f"Max generations: {max_iters}")
    print("=" * 70)

    with multiprocessing.Pool(processes=n_workers) as pool:
        while True:
            if deadline and time.perf_counter() >= deadline:
                break
            if max_iters is not None and gen >= max_iters:
                break

            # Generate λ offspring by perturbing parent in log10 space
            children = []
            for _ in range(LAMBDA):
                child = {}
                for k, (lo, hi) in PARAM_RANGES.items():
                    log_val = math.log10(max(1e-9, parent[k]))
                    log_val = log_val + np.random.normal(0.0, sigmas[k])
                    log_val = _clamp(log_val,
                                     math.log10(lo), math.log10(hi))
                    child[k] = 10.0 ** log_val
                children.append(child)

            # Reserve run IDs before spawning workers
            ids    = _next_ids(LAMBDA, csv_path)
            labels = [f"evo_g{gen:06d}_c{i}" for i in range(LAMBDA)]
            args   = [(c, lbl, rid, csv_path)
                      for c, lbl, rid in zip(children, labels, ids)]

            rows = pool.map(_eval_worker, args)
            n_evals += LAMBDA

            # Find best offspring
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

            # Adapt step sizes (1/5 success rule)
            if len(success_window) >= ADAPT_WINDOW:
                sr = sum(success_window) / len(success_window)
                for k in sigmas:
                    if sr > SUCCESS_TARGET:
                        sigmas[k] = min(sigmas[k] * ADAPT_UP,   SIGMA_LOG_MAX)
                    else:
                        sigmas[k] = max(sigmas[k] * ADAPT_DOWN, SIGMA_LOG_MIN)

            gen += 1

            if gen % 5 == 0:
                elapsed = (time.perf_counter() - t_start) / 60.0
                sigma_str = "  ".join(f"{k}={v:.2f}" for k, v in sigmas.items())
                print(f"\n[gen {gen:5d}]  best={best_fit:.3f}  parent={parent_fit:.3f}  "
                      f"evals={n_evals}  elapsed={elapsed:.1f}min")
                print(f"  parent: {_gains_to_str(parent)}")
                print(f"  sigma:  {sigma_str}")

    elapsed_min = (time.perf_counter() - t_start) / 60.0
    print("\n" + "=" * 70)
    print(f"EVO complete: {gen} generations, {n_evals} evals, {elapsed_min:.1f} min")
    print(f"All-time best: fitness={best_fit:.3f}  (found gen {best_gen})")
    print(f"Best gains: {_gains_to_str(parent)}")
    print(f"\nTo replay best run:")
    print(f"  python replay.py --top 1")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evolutionary LQR gain optimizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize.py --hours 1
  python optimize.py --iters 100 --workers 4
  python optimize.py --iters 50 --seed 42
""")
    ap.add_argument("--hours",   type=float, default=None,
                    help="Wall-clock hours to run (default: 1h if --iters not set)")
    ap.add_argument("--iters",   type=int,   default=None,
                    help="Maximum number of generations")
    ap.add_argument("--workers", type=int,   default=None,
                    help="Parallel worker processes (default: cpu_count)")
    ap.add_argument("--seed",    type=int,   default=None,
                    help="RNG seed for reproducibility")
    args = ap.parse_args()
    run_evo(hours=args.hours, max_iters=args.iters,
            seed=args.seed, n_workers=args.workers)


if __name__ == "__main__":
    multiprocessing.freeze_support()   # required on Windows
    main()
