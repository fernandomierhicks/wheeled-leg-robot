"""optimize_self_balance.py — Evolutionary optimization of LQR gains for self-balance.

Uses scipy.optimize.differential_evolution (evolutionary algorithm) to search
the (Q[0,0], R) space and maximize fitness for self-balance scenario.

Usage:
    python optimize_self_balance.py --population 20 --generations 10

Output:
    optimization_log.csv — All evaluations
    best_solution.json — Optimal (Q[0,0], R, K, fitness)
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).parent))
from test_behavioral import run_self_balance_scenario
from lqr_design import compute_lqr_gain


# ─────────────────────────────────────────────────────────────────────────────
# Fitness function
# ─────────────────────────────────────────────────────────────────────────────

_eval_count = [0]  # Global counter for tracking evaluations
_log_file = None
_log_writer = None
_test_duration = 10.0  # Default, will be set by main()

def fitness_function(x):
    """
    Evaluate fitness for given Q[0,0] and R.

    Args:
        x: array([Q_pitch, R])

    Returns:
        Scalar fitness to MINIMIZE (lower = better)
        Negative because differential_evolution minimizes
    """
    Q_pitch, R = x
    _eval_count[0] += 1

    try:
        # Run test
        metrics = run_self_balance_scenario(
            duration_s=_test_duration,
            q_pitch=Q_pitch,
            r_val=R,
            blend=1.0
        )

        # Compute fitness: weighted combination of metrics (lower = better)
        # Weights chosen to balance smoothness, efficiency, and safety
        fitness = (
            0.40 * metrics.rms_pitch_deg +              # Smoothness (most important)
            0.25 * metrics.control_effort_integral +    # Efficiency
            0.20 * metrics.max_wheel_pos_drift_m +      # Position holding
            0.10 * metrics.peak_torque_nm +             # Safety margin
            0.05 * (metrics.pitch_oscillations_count / 100.0)  # Moderate oscillations
        )

        # Log to CSV
        _log_writer.writerow([
            _eval_count[0],
            Q_pitch,
            R,
            fitness,
            metrics.rms_pitch_deg,
            metrics.max_pitch_deviation_deg,
            metrics.pitch_settle_time_s,
            metrics.pitch_oscillations_count,
            metrics.max_wheel_pos_drift_m,
            metrics.control_effort_integral,
            metrics.peak_torque_nm,
            metrics.max_bearing_load_n,
            metrics.pass_fail,
        ])
        _log_file.flush()

        print(f"[{_eval_count[0]:3d}] Q={Q_pitch:6.1f}  R={R:6.3f}  "
              f"fitness={fitness:6.3f}  wobble={metrics.rms_pitch_deg:5.2f}°  "
              f"drift={metrics.max_wheel_pos_drift_m*1000:5.1f}mm  "
              f"effort={metrics.control_effort_integral:5.2f}")

        return fitness

    except Exception as e:
        print(f"[{_eval_count[0]:3d}] Q={Q_pitch:6.1f}  R={R:6.3f}  ERROR: {e}")
        return 1e6  # Penalize errors heavily


def main():
    global _log_file, _log_writer

    parser = argparse.ArgumentParser(description="Evolutionary LQR optimization for self-balance")
    parser.add_argument("--population", type=int, default=15,
                        help="Population size (default 15)")
    parser.add_argument("--generations", type=int, default=5,
                        help="Max generations (default 5)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Test duration in seconds (default 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for logs")
    args = parser.parse_args()

    global _test_duration
    _test_duration = args.duration

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "optimization_log.csv"
    best_path = output_dir / "best_solution.json"

    print(f"\n{'='*80}")
    print(f"EVOLUTIONARY OPTIMIZATION: Self-Balance LQR Tuning")
    print(f"{'='*80}")
    print(f"Population:    {args.population}")
    print(f"Generations:   {args.generations}")
    print(f"Max evals:     ~{args.population + args.generations * args.population}")
    print(f"Output dir:    {output_dir}")
    print(f"{'='*80}\n")

    # Open log file
    _log_file = open(log_path, 'w', newline='')
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow([
        'eval_id', 'Q_pitch', 'R', 'fitness',
        'rms_pitch_wobble_deg', 'max_pitch_deviation_deg',
        'pitch_settle_time_s', 'pitch_oscillations_count',
        'max_wheel_pos_drift_m', 'control_effort_integral',
        'peak_torque_nm', 'max_bearing_load_n', 'pass_fail'
    ])
    _log_file.flush()

    # Bounds for (Q[0,0], R)
    bounds = [
        (10.0, 200.0),    # Q[0,0]: pitch error weight
        (0.01, 2.0)       # R: control cost weight
    ]

    print("Starting evolutionary optimization...\n")
    start_time = datetime.now()

    # Run differential evolution
    result = differential_evolution(
        fitness_function,
        bounds,
        maxiter=args.generations,
        popsize=args.population,
        seed=args.seed,
        workers=1,  # Set to -1 for parallel evaluation if desired
        polish=True,  # Polish final solution with local optimizer
        atol=1e-4,
        tol=1e-4
    )

    elapsed = datetime.now() - start_time

    _log_file.close()

    # Extract best solution
    Q_best, R_best = result.x
    fitness_best = result.fun
    K_best = compute_lqr_gain(Q_pitch=Q_best, R=R_best)

    # Get final metrics (use same duration as optimization)
    metrics_best = run_self_balance_scenario(
        duration_s=_test_duration,
        q_pitch=Q_best,
        r_val=R_best,
        blend=1.0
    )

    # Save best solution
    best_solution = {
        "optimization": {
            "method": "Differential Evolution (scipy)",
            "population_size": args.population,
            "generations": args.generations,
            "total_evaluations": _eval_count[0],
            "elapsed_time_s": elapsed.total_seconds(),
            "seed": args.seed,
        },
        "best_solution": {
            "Q_pitch": float(Q_best),
            "R": float(R_best),
            "K": [float(k) for k in K_best],
            "fitness_score": float(fitness_best),
        },
        "best_metrics": {
            "rms_pitch_wobble_deg": float(metrics_best.rms_pitch_deg),
            "max_pitch_deviation_deg": float(metrics_best.max_pitch_deviation_deg),
            "pitch_settle_time_s": float(metrics_best.pitch_settle_time_s),
            "pitch_oscillations_count": int(metrics_best.pitch_oscillations_count),
            "max_wheel_pos_drift_m": float(metrics_best.max_wheel_pos_drift_m),
            "rms_wheel_tracking_error_m": float(metrics_best.rms_wheel_tracking_error_m),
            "control_effort_integral": float(metrics_best.control_effort_integral),
            "peak_torque_nm": float(metrics_best.peak_torque_nm),
            "avg_torque_nm": float(metrics_best.avg_torque_nm),
            "max_bearing_load_n": float(metrics_best.max_bearing_load_n),
            "max_femur_lateral_n": float(metrics_best.max_femur_lateral_n),
            "pass_fail": bool(metrics_best.pass_fail),
            "notes": metrics_best.notes,
        }
    }

    with open(best_path, 'w') as f:
        json.dump(best_solution, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total evaluations: {_eval_count[0]}")
    print(f"Elapsed time:      {elapsed.total_seconds():.1f} s ({elapsed.total_seconds()/60:.1f} min)")
    print(f"\nBEST SOLUTION:")
    print(f"  Q[0,0] = {Q_best:.2f}")
    print(f"  R      = {R_best:.4f}")
    print(f"  K      = [{K_best[0]:7.2f}, {K_best[1]:7.2f}, {K_best[2]:7.2f}, {K_best[3]:7.2f}]")
    print(f"  Fitness = {fitness_best:.4f}")
    print(f"\nBEST METRICS:")
    print(f"  RMS pitch wobble:     {metrics_best.rms_pitch_deg:6.3f}°")
    print(f"  Max pitch deviation:  {metrics_best.max_pitch_deviation_deg:6.3f}°")
    print(f"  Settle time:          {metrics_best.pitch_settle_time_s:6.3f} s")
    print(f"  Oscillations:         {metrics_best.pitch_oscillations_count:6.0f}")
    print(f"  Wheel drift:          {metrics_best.max_wheel_pos_drift_m*1000:6.1f} mm")
    print(f"  Control effort:       {metrics_best.control_effort_integral:6.2f} N·m·s")
    print(f"  Peak torque:          {metrics_best.peak_torque_nm:6.3f} N·m")
    print(f"  Pass/fail:            {'✓ PASS' if metrics_best.pass_fail else '✗ FAIL'}")
    print(f"\n{'='*80}")
    print(f"Log saved to: {log_path}")
    print(f"Best solution saved to: {best_path}")
    print(f"{'='*80}\n")

    return 0 if metrics_best.pass_fail else 1


if __name__ == "__main__":
    sys.exit(main())
