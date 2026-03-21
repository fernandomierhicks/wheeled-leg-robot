"""pipeline.py -- S1 to S8 Automated Tuning Pipeline

Runs the full 8-step gain chain sequentially and unattended:
  S1: LQR on 1_LQR_pitch_step         (seed: see --seed)
  S2: LQR on 2_leg_height_gain_sched  (seed: S1 best)  -- LQR BASELINED in sim_config.py
  S3: VelPI on 3_VEL_PI_disturbance   (seed: see --seed)
  S4: VelPI on 4_VEL_PI_staircase     (seed: S3 best)
  S5: VelPI on 5_VEL_PI_leg_cycling   (seed: S4 best)  -- VelocityPI BASELINED in sim_config.py
  S6: YawPI on 6_YAW_PI_turn          (seed: see --seed)
  S7: YawPI on 7_DRIVE_TURN           (seed: S6 best)  -- YawPI BASELINED in sim_config.py
  S8: Susp on 8_terrain_compliance    (seed: see --seed) -- Suspension BASELINED

After each baseline step the winning gains are written back to sim_config.py so that
subsequent subprocess imports pick up the updated values.

Seed modes (--seed):
  baseline  (default) — seed each chain from current sim_config.py values.
                        Use this when re-tuning from a known-good starting point.
  scratch             — each optimizer starts from its own hardcoded SEED_WEIGHTS
                        (original non-latency baseline). Use this when you want the
                        optimizer to explore freely without being biased by prior results.

Usage:
    python pipeline.py                          # all 8 steps, 1h each, baseline seed
    python pipeline.py --hours 2                # all 8 steps, 2h each
    python pipeline.py --seed scratch           # start from hardcoded defaults, not sim_config
    python pipeline.py --start S3 --end S5      # only S3 to S5 (reads S2 CSV for LQR seed)
    python pipeline.py --fresh                  # delete per-step CSVs, restart from zero
    python pipeline.py --workers 8 --hours 0.5
    python pipeline.py --hours 0.017            # smoke test (~1 min per step)
"""
import argparse
import os
import pathlib
import re
import subprocess
import sys
import time
import datetime

_HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from run_log import get_best_run, get_scenario_csv_path

# ---------------------------------------------------------------------------
# Pipeline step definitions
# ---------------------------------------------------------------------------
# Each entry: (step_id, optimizer_script, scenario_name, param_keys, seed_from_step, baseline_map)
#
#   step_id         — short identifier, used for --start/--end filtering
#   optimizer_script— filename in _HERE
#   scenario_name   — passed as --scenario to the optimizer
#   param_keys      — keys to extract from best CSV row as seed for next step
#   seed_from_step  — step_id whose best result seeds this step's --seed-gains (or None)
#   baseline_map    — {csv_param_key: sim_config_variable_name} — non-empty ⇒ write to sim_config.py
#
STEPS = [
    ("S1", "optimize_lqr.py",        "1_LQR_pitch_step",
     ["Q_PITCH", "Q_PITCH_RATE", "Q_VEL", "R"],
     None,
     {}),

    ("S2", "optimize_lqr.py",        "2_leg_height_gain_sched",
     ["Q_PITCH", "Q_PITCH_RATE", "Q_VEL", "R"],
     "S1",
     {"Q_PITCH":      "LQR_Q_PITCH",
      "Q_PITCH_RATE": "LQR_Q_PITCH_RATE",
      "Q_VEL":        "LQR_Q_VEL",
      "R":            "LQR_R"}),

    ("S3", "optimize_vel_pi.py",      "3_VEL_PI_disturbance",
     ["KP_V", "KI_V"],
     None,
     {}),

    ("S4", "optimize_vel_pi.py",      "4_VEL_PI_staircase",
     ["KP_V", "KI_V"],
     "S3",
     {}),

    ("S5", "optimize_vel_pi.py",      "5_VEL_PI_leg_cycling",
     ["KP_V", "KI_V"],
     "S4",
     {"KP_V": "VELOCITY_PI_KP",
      "KI_V": "VELOCITY_PI_KI"}),

    ("S6", "optimize_yaw_pi.py",      "6_YAW_PI_turn",
     ["KP_YAW", "KI_YAW"],
     None,
     {}),

    ("S7", "optimize_yaw_pi.py",      "7_DRIVE_TURN",
     ["KP_YAW", "KI_YAW"],
     "S6",
     {"KP_YAW": "YAW_PI_KP",
      "KI_YAW": "YAW_PI_KI"}),

    ("S8", "optimize_suspension.py",  "8_terrain_compliance",
     ["LEG_K_S", "LEG_B_S", "LEG_K_ROLL", "LEG_D_ROLL"],
     None,
     {"LEG_K_S":    "LEG_K_S",
      "LEG_B_S":    "LEG_B_S",
      "LEG_K_ROLL": "LEG_K_ROLL",
      "LEG_D_ROLL": "LEG_D_ROLL"}),
]

# ---------------------------------------------------------------------------
# sim_config.py variable names for seeding each chain's first step
# (only needed for steps with seed_from=None)
# ---------------------------------------------------------------------------
SIM_CONFIG_SEED_MAP = {
    "S1": {"Q_PITCH":      "LQR_Q_PITCH",
           "Q_PITCH_RATE": "LQR_Q_PITCH_RATE",
           "Q_VEL":        "LQR_Q_VEL",
           "R":            "LQR_R"},
    "S3": {"KP_V":         "VELOCITY_PI_KP",
           "KI_V":         "VELOCITY_PI_KI"},
    "S6": {"KP_YAW":       "YAW_PI_KP",
           "KI_YAW":       "YAW_PI_KI"},
    "S8": {"LEG_K_S":      "LEG_K_S",
           "LEG_B_S":      "LEG_B_S",
           "LEG_K_ROLL":   "LEG_K_ROLL",
           "LEG_D_ROLL":   "LEG_D_ROLL"},
}


# ---------------------------------------------------------------------------
# sim_config.py updater
# ---------------------------------------------------------------------------
_SIM_CONFIG = _HERE / "sim_config.py"


def update_sim_config(gains: dict, mapping: dict) -> None:
    """Regex-replace gain lines in sim_config.py.

    mapping: {csv_param_key: sim_config_variable_name}
    e.g.    {"Q_PITCH": "LQR_Q_PITCH", "R": "LQR_R"}
    """
    content = _SIM_CONFIG.read_text(encoding="utf-8")
    for param_key, config_var in mapping.items():
        val = gains[param_key]
        pattern = rf'^({re.escape(config_var)}\s*=\s*)[\d.e+\-]+'
        replacement = rf'\g<1>{val:.8g}'
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        if new_content == content:
            _log(f"  WARNING: could not find '{config_var}' in sim_config.py to replace")
        else:
            content = new_content
    _SIM_CONFIG.write_text(content, encoding="utf-8")


def _read_sim_config_gains(sim_config_map: dict) -> dict | None:
    """Read current gain values from sim_config.py by variable name."""
    content = _SIM_CONFIG.read_text(encoding="utf-8")
    gains = {}
    for param_key, config_var in sim_config_map.items():
        m = re.search(
            rf'^{re.escape(config_var)}\s*=\s*([\d.e+\-]+)',
            content, re.MULTILINE)
        if m is None:
            return None
        gains[param_key] = float(m.group(1))
    return gains


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
_LOG_FILE: pathlib.Path = None


def _log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if _LOG_FILE is not None:
        with _LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _banner(msg: str) -> None:
    bar = "=" * 72
    _log(bar)
    _log(f"  {msg}")
    _log(bar)


# ---------------------------------------------------------------------------
# Best gains extractor
# ---------------------------------------------------------------------------
def _read_best_gains(scenario_name: str, param_keys: list) -> dict | None:
    """Read best PASS row from the scenario's CSV and extract param_keys."""
    csv_path = get_scenario_csv_path(scenario_name)
    row = get_best_run(scenario=scenario_name, csv_path=csv_path)
    if row is None:
        return None
    gains = {}
    for k in param_keys:
        raw = row.get(k, "")
        try:
            gains[k] = float(raw)
        except (ValueError, TypeError):
            return None   # key missing or non-numeric — can't use as seed
    return gains


# ---------------------------------------------------------------------------
# Seed-gains formatter
# ---------------------------------------------------------------------------
def _format_seed_gains(gains: dict) -> str:
    """Format gains dict as 'K=v,K=v,...' string for --seed-gains CLI arg."""
    return ",".join(f"{k}={v:.8g}" for k, v in gains.items())


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------
def _run_step(step_id, optimizer_script, scenario_name, param_keys,
              seed_gains, hours, workers, patience, tol, fresh) -> dict | None:
    """Run one optimizer step. Returns best gains dict or None on failure."""

    csv_path = pathlib.Path(get_scenario_csv_path(scenario_name))

    # --fresh: delete existing CSV
    if fresh and csv_path.exists():
        csv_path.unlink()
        _log(f"  Deleted {csv_path.name} (--fresh)")

    # Skip if already has PASS results (resume behaviour)
    if not fresh:
        existing = _read_best_gains(scenario_name, param_keys)
        if existing is not None:
            _log(f"  Skipping {step_id}: existing PASS result found in {csv_path.name}")
            _log(f"  Best: {_format_seed_gains(existing)}")
            return existing

    # Build subprocess command
    cmd = [
        sys.executable,
        str(_HERE / optimizer_script),
        "--scenario", scenario_name,
        "--hours",    str(hours),
        "--patience", str(patience),
        "--tol",      str(tol),
    ]
    if workers is not None:
        cmd += ["--workers", str(workers)]
    if seed_gains is not None:
        cmd += ["--seed-gains", _format_seed_gains(seed_gains)]

    _log(f"  CMD: {' '.join(cmd)}")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(_HERE))
    elapsed = (time.perf_counter() - t0) / 60.0

    if result.returncode != 0:
        _log(f"  ERROR: optimizer exited with code {result.returncode}")
        return None

    best = _read_best_gains(scenario_name, param_keys)
    if best is None:
        _log(f"  ERROR: no PASS result found in {csv_path.name} after optimization")
        return None

    _log(f"  Step {step_id} done in {elapsed:.1f} min — best: {_format_seed_gains(best)}")
    return best


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="S1 to S8 automated tuning pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                          # all 8 steps, 1h each, seed from sim_config
  python pipeline.py --hours 2               # 2h per step
  python pipeline.py --seed scratch          # start from hardcoded defaults (free exploration)
  python pipeline.py --hours 0.017           # smoke test (~1 min per step)
  python pipeline.py --start S3 --end S5     # only S3 to S5 (S2 CSV must exist for LQR seed)
  python pipeline.py --fresh                 # delete step CSVs, start from scratch
  python pipeline.py --workers 8
""")
    ap.add_argument("--hours",    type=float, default=1.0,
                    help="Wall-clock hours per optimizer step (default: 1.0)")
    ap.add_argument("--workers",  type=int,   default=None,
                    help="Parallel worker processes per step (default: cpu_count)")
    ap.add_argument("--patience", type=int,   default=200,
                    help="Generations without improvement before early stop (default: 200)")
    ap.add_argument("--tol",      type=float, default=1e-4,
                    help="Relative improvement threshold for convergence (default: 1e-4)")
    ap.add_argument("--start",    type=str,   default=None,
                    help="First step to run, e.g. S3 (default: S1)")
    ap.add_argument("--end",      type=str,   default=None,
                    help="Last step to run inclusive, e.g. S5 (default: S8)")
    ap.add_argument("--fresh",    action="store_true",
                    help="Delete per-step CSVs before running (start from scratch)")
    ap.add_argument("--seed",     type=str, default="baseline",
                    choices=["baseline", "scratch"],
                    help="Seed source for each chain's first step: "
                         "'baseline' (default) reads current sim_config.py gains; "
                         "'scratch' uses each optimizer's hardcoded SEED_WEIGHTS")
    args = ap.parse_args()

    # Set up pipeline log file
    global _LOG_FILE
    logs_dir = _HERE / "logs"
    logs_dir.mkdir(exist_ok=True)
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE = logs_dir / f"pipeline_{ts_str}.log"

    # Filter steps by --start / --end
    step_ids = [s[0] for s in STEPS]
    start_idx = step_ids.index(args.start) if args.start else 0
    end_idx   = step_ids.index(args.end) + 1 if args.end else len(STEPS)
    active_steps = STEPS[start_idx:end_idx]

    _banner(f"S1-S8 Pipeline -- {len(active_steps)} steps  |  {args.hours:.2f}h/step  |  {ts_str}")
    _log(f"  Log: {_LOG_FILE}")
    _log(f"  Steps: {[s[0] for s in active_steps]}")
    _log(f"  seed={args.seed}  patience={args.patience}  tol={args.tol}  fresh={args.fresh}")

    # Track per-step best gains (for cross-step seeding)
    step_results: dict[str, dict] = {}

    # Pre-load existing results for steps before the start, needed as seeds
    for step_id, _, scenario_name, param_keys, seed_from, _ in STEPS[:start_idx]:
        existing = _read_best_gains(scenario_name, param_keys)
        if existing:
            step_results[step_id] = existing

    pipeline_t0 = time.perf_counter()
    failed_steps = []

    for i, (step_id, optimizer_script, scenario_name, param_keys, seed_from, baseline_map) in enumerate(active_steps):
        step_num = start_idx + i + 1
        total_steps = len(active_steps)

        _banner(f"[{step_num}/{total_steps}] {step_id} -- {optimizer_script} -- {scenario_name}")

        # Resolve seed gains
        seed_gains = None
        if seed_from is not None:
            # Mid-chain step: always seed from the previous step's best result
            seed_gains = step_results.get(seed_from)
            if seed_gains:
                _log(f"  Seeding from {seed_from}: {_format_seed_gains(seed_gains)}")
            else:
                _log(f"  WARNING: {seed_from} has no result yet — falling back to seed mode '{args.seed}'")

        if seed_gains is None and step_id in SIM_CONFIG_SEED_MAP:
            # Chain-starting step: apply --seed policy
            if args.seed == "baseline":
                seed_gains = _read_sim_config_gains(SIM_CONFIG_SEED_MAP[step_id])
                if seed_gains:
                    _log(f"  Seeding from sim_config.py (--seed baseline): {_format_seed_gains(seed_gains)}")
                else:
                    _log(f"  WARNING: could not read sim_config.py gains — falling back to optimizer defaults")
            else:
                _log(f"  Seeding from optimizer hardcoded SEED_WEIGHTS (--seed scratch)")

        best = _run_step(
            step_id=step_id,
            optimizer_script=optimizer_script,
            scenario_name=scenario_name,
            param_keys=param_keys,
            seed_gains=seed_gains,
            hours=args.hours,
            workers=args.workers,
            patience=args.patience,
            tol=args.tol,
            fresh=args.fresh,
        )

        if best is None:
            _log(f"  FAILED: {step_id} produced no usable result — continuing to next step")
            failed_steps.append(step_id)
            continue

        step_results[step_id] = best

        # Write baseline gains to sim_config.py
        if baseline_map:
            _log(f"  [BASELINE] Writing to sim_config.py:")
            for param_key, config_var in baseline_map.items():
                _log(f"    {config_var} = {best[param_key]:.8g}")
            update_sim_config(best, baseline_map)
            _log(f"  sim_config.py updated.")

    # Final summary
    total_min = (time.perf_counter() - pipeline_t0) / 60.0
    _banner(f"Pipeline complete — {total_min:.0f} min total")

    for step_id, _, scenario_name, param_keys, _, baseline_map in STEPS:
        if step_id not in step_results:
            continue
        gains = step_results[step_id]
        tag = " [BASELINED]" if baseline_map else ""
        _log(f"  {step_id}{tag}: {_format_seed_gains(gains)}")

    if failed_steps:
        _log(f"\n  FAILED steps: {failed_steps}")
        _log("  Re-run with --start <first_failed> to retry.")
    else:
        _log("\n  All steps completed successfully.")
        _log("  sim_config.py has been updated with all baselined gains.")
        _log("  Run 'python replay.py' to visualize any scenario.")


if __name__ == "__main__":
    main()
