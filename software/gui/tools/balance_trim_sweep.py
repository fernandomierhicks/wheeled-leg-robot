#!/usr/bin/env python3
"""balance_trim_sweep.py — extract measured balance point vs leg height from a run.

The balance point is the pitch the robot must hold to stand still at a given
leg height. It is NOT lqr_pitch_trim_ret on its own: with a trim that is even
slightly wrong, the velocity PI integrator silently makes up the difference by
biasing theta_ref, and the robot stands there looking perfectly trimmed. The
integrator is therefore the instrument -- what it converges to IS the trim
error, so the portable quantity is their sum:

    balance_point(alpha) = pitch_trim_rad + theta_ref     (settled, stationary)

Feed the result straight back into lqr_pitch_trim_ret (alpha=0) and
lqr_pitch_trim_ext (alpha=1); if the sweep turns out non-linear in between,
that is the evidence for replacing the two-anchor interpolation with a table.

Usage:
    python software/gui/tools/balance_trim_sweep.py <run.jsonl|run.WLOG> [...]
    python software/gui/tools/balance_trim_sweep.py <run> --csv out.csv

Only RUNNING samples that are stationary (no commanded velocity or yaw) are
used. Each stationary stretch contributes only its TAIL, and the residual
drift across that tail is reported, because the integrator is slow enough that
averaging a whole stretch biases the answer toward wherever it started.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.wlog_metrics import decode_run  # noqa: E402

STATE_RUNNING = 3

V_STILL_MS      = 0.02    # |v_ref| below this counts as "not commanded to move"
OMEGA_STILL     = 0.02    # |omega_cmd_rds| likewise [rad/s]
ALPHA_BIN       = 0.05    # leg heights closer than this are pooled as one point
MIN_STRETCH_S   = 8.0     # a shorter stationary stretch says nothing about the trim
TAIL_FRAC       = 0.5     # average over the LAST half of each stretch only
DRIFT_WARN_DEG  = 0.25    # tail still moving by more than this => not converged

# Why the tail, and why the drift check: the velocity PI integrator is what
# measures the trim error, and it is slow -- on the 2026-08-09 v4 run it was
# still walking after 12.8 s of standing still (theta_ref -4.62 -> -3.54 deg,
# balance point -6.07 -> -6.54 deg). Averaging a whole stretch therefore biases
# the answer toward whatever the integrator started at. Taking the tail and
# reporting how much it is STILL moving is the difference between a measurement
# and a guess; if drift exceeds DRIFT_WARN_DEG, stand still longer rather than
# trusting the number.


def _stretches(ok, fs, min_s):
    """Contiguous index ranges where `ok` holds, at least min_s long."""
    out, start = [], None
    for i, good in enumerate(ok):
        if good and start is None:
            start = i
        elif not good and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(ok) - 1))
    return [(a, b) for a, b in out if (b - a + 1) / fs >= min_s]


def analyse(path):
    run = decode_run(path)
    f, fs = run.fields, run.sample_rate_hz
    m = f["robot_state"].astype(int) == STATE_RUNNING
    if m.sum() < MIN_STRETCH_S * fs:
        return run, []

    idx = np.where(m)[0]
    alpha = f["gain_sched_alpha"][idx]
    theta = f["theta_ref"][idx]
    trim  = f["pitch_trim_rad"][idx]
    pitch = f["pitch_rad"][idx]

    if np.allclose(theta, 0.0):
        print(f"  !! theta_ref is identically zero -- the velocity PI is OFF (vel_pi_en=0).\n"
              f"     Without it nothing measures the trim error and this run cannot be used.",
              file=sys.stderr)
        return run, []

    still = (np.abs(f["v_ref"][idx]) < V_STILL_MS) & \
            (np.abs(f["omega_cmd_rds"][idx]) < OMEGA_STILL)

    # Collect the converged tail of every long-enough stationary stretch, then
    # pool those tails by leg height.
    buckets = {}
    for a, b in _stretches(still, fs, MIN_STRETCH_S):
        tail = slice(b - int((b - a + 1) * TAIL_FRAC), b + 1)
        key = round(float(alpha[tail].mean()) / ALPHA_BIN)
        buckets.setdefault(key, []).append(tail)

    rows = []
    for key, tails in sorted(buckets.items()):
        bp    = np.concatenate([trim[t] + theta[t] for t in tails])
        al    = np.concatenate([alpha[t] for t in tails])
        tr_   = np.concatenate([trim[t] for t in tails])
        th_   = np.concatenate([theta[t] for t in tails])
        pi_   = np.concatenate([pitch[t] for t in tails])
        half  = len(bp) // 2
        drift = np.degrees(bp[half:].mean() - bp[:half].mean()) if half else float("nan")
        rows.append({
            "alpha":       float(al.mean()),
            "n_s":         len(bp) / fs,
            "balance_rad": float(bp.mean()),
            "balance_deg": float(np.degrees(bp.mean())),
            "sd_deg":      float(np.degrees(bp.std())),
            "drift_deg":   float(drift),
            "trim_deg":    float(np.degrees(tr_.mean())),
            "theta_deg":   float(np.degrees(th_.mean())),
            "pitch_deg":   float(np.degrees(pi_.mean())),
        })
    return run, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="host.jsonl or .WLOG bundles")
    ap.add_argument("--csv", help="append the rows to this CSV")
    args = ap.parse_args()

    all_rows = []
    for path in args.runs:
        print(f"\n=== {path} ===")
        try:
            run, rows = analyse(path)
        except Exception as exc:                     # noqa: BLE001 - report and continue
            print(f"  could not analyse: {exc}", file=sys.stderr)
            continue
        if not rows:
            print("  no settled stationary plateau long enough to measure")
            continue
        print(f"  {'alpha':>6} {'secs':>6} {'balance_pt':>11} {'sd':>7} {'drift':>7} "
              f"{'= trim':>8} {'+ theta_ref':>12}")
        for r in rows:
            flag = "  <-- STILL MOVING, stand still longer" \
                   if abs(r["drift_deg"]) > DRIFT_WARN_DEG else ""
            print(f"  {r['alpha']:>6.3f} {r['n_s']:>6.1f} "
                  f"{r['balance_deg']:>+10.2f}d {r['sd_deg']:>6.2f}d {r['drift_deg']:>+6.2f}d "
                  f"{r['trim_deg']:>+7.2f}d {r['theta_deg']:>+11.2f}d{flag}")
            r["run"] = os.path.basename(os.path.dirname(os.path.abspath(path)))
            all_rows.append(r)

    if not all_rows:
        return 1

    print("\n--- set these ---")
    lo = min(all_rows, key=lambda r: r["alpha"])
    hi = max(all_rows, key=lambda r: r["alpha"])
    print(f"  lqr_pitch_trim_ret = {lo['balance_rad']:+.4f}   (measured at alpha={lo['alpha']:.3f})")
    if hi["alpha"] - lo["alpha"] > 0.1:
        print(f"  lqr_pitch_trim_ext = {hi['balance_rad']:+.4f}   (measured at alpha={hi['alpha']:.3f})")
        if hi["alpha"] < 0.9:
            print(f"  NOTE: highest alpha measured is {hi['alpha']:.2f}, not 1.0 -- extrapolating "
                  f"to the _ext anchor. Measure nearer full extension before trusting it.")
        mid = [r for r in all_rows if lo["alpha"] + 0.1 < r["alpha"] < hi["alpha"] - 0.1]
        if mid:
            print("\n--- linearity check (is the 2-anchor interpolation enough?) ---")
            worst = 0.0
            for r in mid:
                t = (r["alpha"] - lo["alpha"]) / (hi["alpha"] - lo["alpha"])
                pred = lo["balance_deg"] + t * (hi["balance_deg"] - lo["balance_deg"])
                err = r["balance_deg"] - pred
                worst = max(worst, abs(err))
                print(f"  alpha={r['alpha']:.3f}: measured {r['balance_deg']:+.2f}d  "
                      f"linear {pred:+.2f}d  error {err:+.2f}d")
            print(f"  worst deviation from linear: {worst:.2f} deg -- "
                  + ("a lookup table would earn its keep." if worst > 1.0
                     else "two anchors are sufficient; a table would add nothing."))
    else:
        print("  (only one leg height sampled -- _ext anchor still unmeasured)")

    if args.csv:
        import csv
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["run", "alpha", "n_s", "balance_rad",
                                               "balance_deg", "sd_deg", "drift_deg", "trim_deg",
                                               "theta_deg", "pitch_deg"])
            if new:
                w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\nappended {len(all_rows)} row(s) to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
