#!/usr/bin/env python3
"""balance_trim_sweep.py — extract measured balance point vs leg height from a run.

The balance point is the pitch the robot must hold to stand still at a given
leg height -- i.e. the pitch that puts the CG over the contact patch. It is
measured directly as the mean pitch over a window where the robot is genuinely
in equilibrium, and it does NOT depend on what lqr_pitch_trim_* happened to be
set to during the run.

Equilibrium is the whole game, so it is tested on the physics rather than
assumed. A robot that is not accelerating needs no wheel torque, so:

    tau_sym ~ 0   AND   wheels not drifting   =>   pitch IS the balance point

That also falls out of the control law: tau_sym = -k_pitch*(pitch - theta_ref
- trim), so tau_sym = 0 means pitch = theta_ref + trim. The tool reports that
sum alongside the mean pitch as a free consistency check -- when the two
disagree, the window was not at equilibrium and the number should be thrown
away, not averaged in.

Note this implicitly requires the velocity PI to be enabled: with vel_pi_en=0
nothing stops the wheels running away, the drift gate never passes, and the
run yields no points.

Feed the result straight back into lqr_pitch_trim_ret (alpha=0) and
lqr_pitch_trim_ext (alpha=1); if the sweep turns out non-linear in between,
that is the evidence for replacing the two-anchor interpolation with a table.

Usage:
    python software/gui/tools/balance_trim_sweep.py <run.jsonl|run.WLOG> [...]
    python software/gui/tools/balance_trim_sweep.py <run> --csv out.csv

Only RUNNING samples that are stationary (no commanded velocity or yaw) are
used, and only the TAIL of each such stretch -- settling is slow, so the head
of a stretch is biased toward wherever the robot started. Tails that fail the
equilibrium gate are discarded outright rather than averaged in.
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.leg_height_sweep import alpha_plateaus  # noqa: E402
from analysis.wlog_metrics import decode_run  # noqa: E402

STATE_RUNNING = 3

V_STILL_MS      = 0.02    # |v_ref| below this counts as "not commanded to move"
OMEGA_STILL     = 0.02    # |omega_cmd_rds| likewise [rad/s]
ALPHA_BIN       = 0.05    # leg heights closer than this are pooled as one point
MIN_STRETCH_S   = 8.0     # a shorter stationary stretch says nothing about the trim
TAIL_FRAC       = 0.5     # use the LAST half of each stretch only
TAU_EQ_NM       = 0.005   # |mean tau_sym| above this => not in equilibrium
DRIFT_EQ_TPS    = 0.05    # |mean wheel velocity| above this => still creeping

# Why only the tail, and why the equilibrium gate: settling is slow. On the
# 2026-08-09 v4 run the robot was still creeping after 12.8 s of standing
# still, and windows that had NOT equilibrated disagreed with the tau_sym=0
# cross-check by up to 0.7 deg while the one that had agreed to 0.09 deg. So
# the head of each stretch is discarded and any window whose mean tau_sym or
# wheel drift is too large is thrown out entirely rather than averaged in --
# a biased sample is worse than a missing one here.


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


def _calib_from_sidecar(path):
    """(backoff_rad, range_rad) from a .PARAMS companion, or (None, None)."""
    try:
        from pathlib import Path
        from analysis.param_sidecar import load_matching_sidecar
        sc = load_matching_sidecar(Path(path))
        if sc is None:
            return None, None
        b = sc.initial_value("calib_backoff_rad")
        r = sc.initial_value("calib_range_l_rad")
        if b is None or r is None:
            return None, None
        return float(b), float(r)
    except Exception:                                # noqa: BLE001 - sidecar is optional
        return None, None


def analyse(path, backoff_rad=None, range_rad=None):
    if backoff_rad is None or range_rad is None:
        sb, sr = _calib_from_sidecar(path)
        backoff_rad = backoff_rad if backoff_rad is not None else sb
        range_rad   = range_rad   if range_rad   is not None else sr
    run = decode_run(path)
    f, fs = run.fields, run.sample_rate_hz
    m = f["robot_state"].astype(int) == STATE_RUNNING
    if m.sum() < MIN_STRETCH_S * fs:
        return run, [], []

    idx = np.where(m)[0]
    alpha = f["gain_sched_alpha"][idx]
    theta = f["theta_ref"][idx]
    trim  = f["pitch_trim_rad"][idx]
    pitch = f["pitch_rad"][idx]

    if np.allclose(theta, 0.0):
        print(f"  !! theta_ref is identically zero -- the velocity PI is OFF (vel_pi_en=0).\n"
              f"     Without it nothing measures the trim error and this run cannot be used.",
              file=sys.stderr)
        return run, [], []

    # alpha_force_ret_en=1 pins g_state.gain_sched_alpha to 0 in TELEMETRY too,
    # not just in the gain blend -- which is exactly the configuration you WANT
    # for a trim sweep (constant gains, only height varying), so refusing the run
    # would be unhelpful. Reconstruct the real leg height instead.
    #
    # The encoder is zeroed at the retract switch, so hip position is already
    # "distance from the switch" and alpha is a straight rescale of it:
    #     alpha = (|hip| - calib_backoff_rad) / (calib_range_rad - calib_backoff_rad)
    # matching define_limits()/hip_cmd_to_setpoints(). Neither calibration
    # constant is in telemetry, so they come from the .PARAMS sidecar (SD runs)
    # or from --backoff-deg/--range-deg.
    hip_meas = np.abs(f["hip_l_pos_rad"][idx])
    hip_cmd  = np.abs(f["hip_l_cmd_pos_rad"][idx])
    if np.ptp(alpha) < 1e-6 and np.ptp(hip_cmd) > np.radians(1.0):
        if backoff_rad is None or range_rad is None:
            print(f"  !! gain_sched_alpha is pinned at {alpha[0]:.3f} while the hip command "
                  f"moved {np.degrees(np.ptp(hip_cmd)):.1f} deg -- that is "
                  f"alpha_force_ret_en=1.\n"
                  f"     Leg height can be recovered from hip position, but this run has no "
                  f".PARAMS sidecar,\n"
                  f"     so pass the calibration constants: "
                  f"--backoff-deg 15 --range-deg 85", file=sys.stderr)
            return run, [], []
        span = range_rad - backoff_rad
        if span <= 0:
            print("  !! range must exceed backoff", file=sys.stderr)
            return run, [], []
        alpha = np.clip((hip_meas - backoff_rad) / span, 0.0, 1.0)
        print(f"  alpha reconstructed from hip position "
              f"(backoff {np.degrees(backoff_rad):.1f} deg, range {np.degrees(range_rad):.1f} deg): "
              f"{alpha.min():.3f} .. {alpha.max():.3f}")

    tau   = f["tau_sym"][idx]
    wheel = 0.5 * (f["wm_l_vel_turns_s"][idx] + f["wm_r_vel_turns_s"][idx])

    still = (np.abs(f["v_ref"][idx]) < V_STILL_MS) & \
            (np.abs(f["omega_cmd_rds"][idx]) < OMEGA_STILL)

    # "Stationary" is not the same as "at one leg height". A sweep that walks
    # through several heights without touching the sticks is stationary
    # throughout, so splitting on `still` alone yields one enormous stretch
    # spanning every height in the run; its tail straddles two of them, fails
    # the equilibrium gate, and the whole sweep reports nothing (which is
    # exactly what the 2026-08-09 four-height run did). Intersect with the
    # flat-alpha plateaus so each height is measured on its own.
    on_plateau = np.zeros(len(idx), dtype=bool)
    for a, b in alpha_plateaus(alpha, fs):
        on_plateau[a:b + 1] = True
    still &= on_plateau

    # Each stationary stretch contributes its tail; keep only the tails that
    # actually reached equilibrium, then pool the survivors by leg height.
    buckets, rejected = {}, []
    for a, b in _stretches(still, fs, MIN_STRETCH_S):
        t = slice(b - int((b - a + 1) * TAIL_FRAC), b + 1)
        mtau, mdrift = abs(tau[t].mean()), abs(wheel[t].mean())
        if mtau > TAU_EQ_NM or mdrift > DRIFT_EQ_TPS:
            rejected.append(((t.stop - t.start) / fs, mtau, mdrift))
            continue
        buckets.setdefault(round(float(alpha[t].mean()) / ALPHA_BIN), []).append(t)

    rows = []
    for _, tails in sorted(buckets.items()):
        pi_ = np.concatenate([pitch[t] for t in tails])
        al  = np.concatenate([alpha[t] for t in tails])
        xchk = np.concatenate([trim[t] + theta[t] for t in tails])
        # Spread across independent windows is the honest uncertainty; the
        # within-window sd just measures how hard the robot is working.
        per_window = [float(np.degrees(pitch[t].mean())) for t in tails]
        rows.append({
            "alpha":        float(al.mean()),
            "n_s":          len(pi_) / fs,
            "n_windows":    len(tails),
            "balance_rad":  float(pi_.mean()),
            "balance_deg":  float(np.degrees(pi_.mean())),
            "spread_deg":   float(max(per_window) - min(per_window)) if len(tails) > 1 else float("nan"),
            "xcheck_deg":   float(np.degrees(xchk.mean())),
            "tau_nm":       float(np.concatenate([tau[t] for t in tails]).mean()),
            "drift_tps":    float(np.concatenate([wheel[t] for t in tails]).mean()),
        })
    return run, rows, rejected


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="host.jsonl or .WLOG bundles")
    ap.add_argument("--csv", help="append the rows to this CSV")
    ap.add_argument("--backoff-deg", type=float, default=None,
                    help="calib_backoff_rad in DEGREES; only needed to recover leg "
                         "height when alpha_force_ret_en pinned the logged alpha "
                         "and there is no .PARAMS sidecar")
    ap.add_argument("--range-deg", type=float, default=None,
                    help="calib_range_*_rad in DEGREES; see --backoff-deg")
    args = ap.parse_args()

    all_rows = []
    for path in args.runs:
        print(f"\n=== {path} ===")
        try:
            run, rows, rejected = analyse(
                path,
                math.radians(args.backoff_deg) if args.backoff_deg is not None else None,
                math.radians(args.range_deg) if args.range_deg is not None else None)
        except Exception as exc:                     # noqa: BLE001 - report and continue
            print(f"  could not analyse: {exc}", file=sys.stderr)
            continue
        for secs, mtau, mdrift in rejected:
            print(f"  rejected a {secs:.1f}s window: not at equilibrium "
                  f"(mean tau {mtau:.4f} N.m, drift {mdrift:.3f} turns/s)")
        if not rows:
            print("  no window reached equilibrium -- stand still longer, "
                  "and check vel_pi_en=1")
            continue
        print(f"  {'alpha':>6} {'secs':>6} {'win':>4} {'balance_pt':>11} {'spread':>8} "
              f"{'xcheck':>9} {'mean tau':>10} {'drift':>9}")
        for r in rows:
            sp = "     n/a" if r["n_windows"] < 2 else f"{r['spread_deg']:>7.2f}d"
            flag = "  <-- xcheck disagrees, provisional" \
                   if abs(r["balance_deg"] - r["xcheck_deg"]) > 0.3 else ""
            print(f"  {r['alpha']:>6.3f} {r['n_s']:>6.1f} {r['n_windows']:>4d} "
                  f"{r['balance_deg']:>+10.2f}d {sp} {r['xcheck_deg']:>+8.2f}d "
                  f"{r['tau_nm']:>+10.4f} {r['drift_tps']:>+9.3f}{flag}")
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
            w = csv.DictWriter(fh, fieldnames=["run", "alpha", "n_s", "n_windows",
                                               "balance_rad", "balance_deg", "spread_deg",
                                               "xcheck_deg", "tau_nm", "drift_tps"])
            if new:
                w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\nappended {len(all_rows)} row(s) to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
