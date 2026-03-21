"""optimize_suspension.py — Suspension + Roll Leveling Optimizer

Searches over 4 leg impedance parameters via (1+8)-ES:
  LEG_K_S    [N·m/rad]     vertical spring stiffness
  LEG_B_S    [N·m·s/rad]  vertical damping (can evolve to zero)
  LEG_K_ROLL [rad/rad]     roll proportional gain (can evolve to zero)
  LEG_D_ROLL [rad·s/rad]  roll rate damping (can evolve to zero)

All other gains (LQR Q/R, VelocityPI, YawPI, HIP_IMPEDANCE_TORQUE_LIMIT)
are held fixed at their baselined values in sim_config.py.

Scenarios:
  suspension_combined  (default): 0.5×S5 + 0.5×S8
    S5 ensures vertical suspension doesn't degrade velocity tracking.
    S8 exercises the roll leveling controller specifically.
  8_terrain_compliance: S8 only (used as final pipeline step after all
    other gains are baselined).

Run:
    python optimize_suspension.py --hours 0.083   # 5 minutes
    python optimize_suspension.py --hours 1.0 --workers 8
    python optimize_suspension.py --scenario 8_terrain_compliance --hours 1
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
    YAW_PI_KP, YAW_PI_KI,
    LEG_K_S, LEG_B_S, LEG_K_ROLL, LEG_D_ROLL,
    SCENARIO_5_DURATION, SCENARIO_8_DURATION,
)

_DEFAULT_SCENARIO = "suspension_combined"

# ── Valid scenarios for this optimizer ───────────────────────────────────────
_VALID_SCENARIOS = {
    "suspension_combined",
    "8_terrain_compliance",
}

# ── Search space (wide — never clamp) ────────────────────────────────────────
PARAM_RANGES = {
    "LEG_K_S":    (0.1,  60.0),   # [N·m/rad]    vertical stiffness
    "LEG_B_S":    (0.01, 40.0),   # [N·m·s/rad]  vertical damping (can → 0)
    "LEG_K_ROLL": (1e-3, 20.0),   # [rad/rad]     roll P gain (can → 0)
    "LEG_D_ROLL": (1e-4, 10.0),   # [rad·s/rad]   roll D gain (can → 0)
}

# Gains that may evolve to zero
ZERO_FLOOR = 1e-6

# ── Seed from current sim_config values ──────────────────────────────────────
SEED_WEIGHTS = {
    "LEG_K_S":    LEG_K_S,
    "LEG_B_S":    LEG_B_S,
    "LEG_K_ROLL": LEG_K_ROLL,
    "LEG_D_ROLL": LEG_D_ROLL,
}

# ── (1+λ)-ES hyper-parameters ─────────────────────────────────────────────────
LAMBDA         = 8
SIGMA_LOG_INIT = 0.40
SIGMA_LOG_MIN  = 0.01
SIGMA_LOG_MAX  = 1.00
SUCCESS_TARGET = 1.0 / 5.0
ADAPT_WINDOW   = 10
_DEFAULT_PATIENCE = 30
_DEFAULT_TOL      = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _gains_to_str(g: dict) -> str:
    return "  ".join(f"{k}={v:.4g}" for k, v in g.items())


def _parse_seed_gains(seed_str: str) -> dict:
    result = {}
    for part in seed_str.split(","):
        k, _, v = part.strip().partition("=")
        result[k.strip()] = float(v.strip())
    return result


def _load_best_weights(active_scenario: str, csv_path: str,
                       seed_override: dict = None) -> tuple:
    if seed_override:
        return dict(seed_override), float("inf")
    row = get_best_run(scenario=active_scenario, csv_path=csv_path)
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
    k_s, b_s, k_roll, d_roll, label, run_id, csv_path, scenario_name = args

    import os, sys, datetime
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from lqr_design import compute_gain_table
    from sim_config import ROBOT, SCENARIO_5_DURATION, SCENARIO_8_DURATION
    import scenarios

    # Fix all other controllers; search only over suspension params
    scenarios.USE_PD_CONTROLLER = False
    scenarios.USE_VELOCITY_PI   = True
    scenarios.USE_YAW_PI        = True
    scenarios.VELOCITY_PI_KP    = VELOCITY_PI_KP
    scenarios.VELOCITY_PI_KI    = VELOCITY_PI_KI
    scenarios.YAW_PI_KP_GAIN    = YAW_PI_KP
    scenarios.YAW_PI_KI_GAIN    = YAW_PI_KI
    scenarios.LEG_K_ROLL_GAIN   = k_roll
    scenarios.LEG_D_ROLL_GAIN   = d_roll
    scenarios.LQR_K_TABLE = compute_gain_table(
        ROBOT,
        Q_diag=[LQR_Q_PITCH, LQR_Q_PITCH_RATE, LQR_Q_VEL],
        R_val=LQR_R,
    )

    # Patch LEG_K_S / LEG_B_S directly on the scenarios module (used in _run_sim_loop)
    import sim_config as _cfg
    _cfg.LEG_K_S = k_s
    _cfg.LEG_B_S = b_s
    scenarios.LEG_K_S = k_s
    scenarios.LEG_B_S = b_s

    if scenario_name == "8_terrain_compliance":
        # S8-only mode: roll leveling + terrain compliance
        s8 = scenarios.run_8_terrain_compliance({}, duration=SCENARIO_8_DURATION)
        fitness = s8.get("fitness", 9999.0)
        fell    = s8.get("fell", True)
        status  = "FAIL" if fell else "PASS"

        row = dict(
            run_id      = run_id,
            scenario    = scenario_name,
            label       = label,
            timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            LEG_K_S     = round(k_s,    6),
            LEG_B_S     = round(b_s,    6),
            LEG_K_ROLL  = round(k_roll, 6),
            LEG_D_ROLL  = round(d_roll, 6),
            fitness     = round(fitness, 4),
            status      = status,
            s8_fitness  = round(s8.get("fitness", 9999.0), 4),
            s8_roll_rms = round(s8.get("max_roll_deg", 0.0), 4),
            s8_pitch    = round(s8.get("rms_pitch_deg", 0.0), 4),
            fell        = fell,
        )
        print(f"[{run_id:5d}] {label:<28}  {status:<5}  fit={fitness:.4f}  "
              f"S8={s8.get('fitness',9999):.3f}(max_roll={row['s8_roll_rms']:.2f}°)  "
              f"K_s={k_s:.3g} B_s={b_s:.3g} K_roll={k_roll:.3g} D_roll={d_roll:.3g}")

    else:
        # Combined mode: 0.5×S5 + 0.5×S8
        s5 = scenarios.run_5_VEL_PI_leg_cycling({}, duration=SCENARIO_5_DURATION)
        s8 = scenarios.run_8_terrain_compliance({}, duration=SCENARIO_8_DURATION)

        fitness = 0.5 * s5.get("fitness", 9999.0) + 0.5 * s8.get("fitness", 9999.0)
        fell    = s5.get("fell", True) or s8.get("fell", True)
        status  = "FAIL" if fell else "PASS"

        row = dict(
            run_id      = run_id,
            scenario    = scenario_name,
            label       = label,
            timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            LEG_K_S     = round(k_s,    6),
            LEG_B_S     = round(b_s,    6),
            LEG_K_ROLL  = round(k_roll, 6),
            LEG_D_ROLL  = round(d_roll, 6),
            fitness     = round(fitness, 4),
            status      = status,
            s5_fitness  = round(s5.get("fitness", 9999.0), 4),
            s8_fitness  = round(s8.get("fitness", 9999.0), 4),
            s5_vel_rms  = round(s5.get("vel_track_rms_ms", 0.0), 4),
            s5_pitch    = round(s5.get("rms_pitch_deg",    0.0), 4),
            s8_roll_rms = round(s8.get("max_roll_deg",     0.0), 4),
            s8_pitch    = round(s8.get("rms_pitch_deg",    0.0), 4),
            fell        = fell,
        )
        print(f"[{run_id:5d}] {label:<28}  {status:<5}  fit={fitness:.4f}  "
              f"S5={s5.get('fitness',9999):.3f}(vel={row['s5_vel_rms']:.3f}m/s "
              f"pitch={row['s5_pitch']:.2f}°)  "
              f"S8={s8.get('fitness',9999):.3f}(max_roll={row['s8_roll_rms']:.2f}°)  "
              f"K_s={k_s:.3g} B_s={b_s:.3g} K_roll={k_roll:.3g} D_roll={d_roll:.3g}")

    from run_log import log_run
    log_run(row, csv_path)
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
    if active_scenario not in _VALID_SCENARIOS:
        raise ValueError(f"Scenario '{active_scenario}' not valid for suspension optimizer. "
                         f"Valid: {sorted(_VALID_SCENARIOS)}")
    if hours is None and max_iters is None:
        hours = 1.0
    if n_workers is None:
        n_workers = min(LAMBDA, multiprocessing.cpu_count())
    if seed is not None:
        np.random.seed(seed)
    if csv_path is None:
        csv_path = get_scenario_csv_path(active_scenario)

    parent, parent_fit = _load_best_weights(active_scenario, csv_path, seed_override)
    if parent_fit == float("inf"):
        print("No prior results — evaluating seed weights first...")
        row = _eval_worker((
            parent["LEG_K_S"], parent["LEG_B_S"],
            parent["LEG_K_ROLL"], parent["LEG_D_ROLL"],
            "evo_seed", next_run_id(csv_path), csv_path, active_scenario))
        parent_fit = float(row.get("fitness", float("inf")))
        print(f"Seed fitness: {parent_fit:.4f}\n")
    else:
        print(f"Seeding from best result: fitness={parent_fit:.4f}")
        print(f"  {_gains_to_str(parent)}\n")

    sigmas         = {k: SIGMA_LOG_INIT for k in PARAM_RANGES}
    success_window = deque(maxlen=ADAPT_WINDOW)

    t_start  = time.perf_counter()
    deadline = (t_start + hours * 3600.0) if hours else None
    gen      = 0
    n_evals  = 0
    best_fit = parent_fit
    best_gen = 0

    # Convergence tracking
    gens_without_improvement = 0
    prev_best_for_patience = best_fit

    print("=" * 70)
    print(f"Suspension Optimizer (1+{LAMBDA})-ES  |  Scenario: {active_scenario}")
    print(f"  workers={n_workers}  params={list(PARAM_RANGES)}")
    print(f"  Fixed LQR: Q=[{LQR_Q_PITCH:.4g},{LQR_Q_PITCH_RATE:.4g},{LQR_Q_VEL:.4g}]  R={LQR_R:.4g}")
    print(f"  Fixed VelocityPI: KP_V={VELOCITY_PI_KP:.4g}  KI_V={VELOCITY_PI_KI:.4g}")
    if active_scenario == "suspension_combined":
        print(f"  Fitness: 0.5×S5 + 0.5×S8")
    else:
        print(f"  Fitness: S8 only")
    print(f"  early-stop: patience={patience}  tol={tol:.1e}")
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
            if gens_without_improvement >= patience:
                print(f"\nEarly stop: no improvement in {patience} generations.")
                break

            children = []
            for _ in range(LAMBDA):
                child = {}
                for k, (lo, hi) in PARAM_RANGES.items():
                    lv = math.log10(max(1e-9, parent[k])) if parent[k] > 0 else math.log10(lo)
                    lv += np.random.normal(0.0, sigmas[k])
                    lv  = _clamp(lv, math.log10(lo), math.log10(hi))
                    val = 10.0 ** lv
                    # B_S, K_ROLL, D_ROLL can evolve to zero
                    if k in ("LEG_B_S", "LEG_K_ROLL", "LEG_D_ROLL") and val < ZERO_FLOOR:
                        val = 0.0
                    child[k] = val
                children.append(child)

            base_id = next_run_id(csv_path)
            ids     = list(range(base_id, base_id + LAMBDA))
            labels  = [f"evo_g{gen:06d}_c{i}" for i in range(LAMBDA)]
            args    = [(c["LEG_K_S"], c["LEG_B_S"], c["LEG_K_ROLL"], c["LEG_D_ROLL"],
                        lbl, rid, csv_path, active_scenario)
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

            # Convergence check
            rel_improvement = (prev_best_for_patience - best_fit) / (prev_best_for_patience + 1e-12)
            if rel_improvement > tol:
                gens_without_improvement = 0
                prev_best_for_patience = best_fit
            else:
                gens_without_improvement += 1

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

            sr        = sum(success_window) / len(success_window) if success_window else 0.0
            gains_str = _gains_to_str(parent)

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
                print(f"\n{bar} {progress_str}  evals={n_evals}  elapsed={elapsed_min:.1f}min"
                      f"  stagnant={gens_without_improvement}/{patience}")
                print(f"  best={best_fit:.4f}  parent={parent_fit:.4f}  best_at_gen={best_gen}")
                print(f"  gains: {gains_str}")

    elapsed_min = (time.perf_counter() - t_start) / 60.0
    print("\n" + "=" * 70)
    print(f"Optimization complete: {gen} gens, {n_evals} evals, {elapsed_min:.1f} min")
    print(f"All-time best: fitness={best_fit:.4f}  (gen {best_gen})")
    print(f"Best suspension: {_gains_to_str(parent)}")
    print(f"\nUpdate sim_config.py:")
    for k, v in parent.items():
        print(f"  {k} = {v:.6f}")
    print(f"\nTo replay S8: python replay.py --baseline --scenario 8_terrain_compliance")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Suspension + roll leveling optimizer (K_s, B_s, K_roll, D_roll via ES).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_suspension.py --hours 0.083   # 5 minutes (smoke test)
  python optimize_suspension.py --hours 1.0
  python optimize_suspension.py --iters 50 --workers 8
  python optimize_suspension.py --scenario 8_terrain_compliance --hours 1
""")
    ap.add_argument("--hours",   type=float, default=None)
    ap.add_argument("--iters",   type=int,   default=None)
    ap.add_argument("--workers", type=int,   default=None)
    ap.add_argument("--seed",    type=int,   default=None)
    ap.add_argument("--scenario", type=str,  default=_DEFAULT_SCENARIO,
                    help=f"Scenario to optimize (default: {_DEFAULT_SCENARIO}). "
                         f"Valid: {sorted(_VALID_SCENARIOS)}")
    ap.add_argument("--seed-gains", type=str, default=None,
                    help='Explicit seed gains, e.g. "LEG_K_S=2.04,LEG_B_S=4.18,LEG_K_ROLL=3.91,LEG_D_ROLL=0.92"')
    ap.add_argument("--patience", type=int,  default=_DEFAULT_PATIENCE)
    ap.add_argument("--tol",     type=float, default=_DEFAULT_TOL)
    args = ap.parse_args()

    seed_override = _parse_seed_gains(args.seed_gains) if args.seed_gains else None

    win = ProgressWindow(f"Suspension Optimizer — {args.scenario}")
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
